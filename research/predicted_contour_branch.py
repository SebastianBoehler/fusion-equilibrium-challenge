"""Research-only derivatives through the current predicted marching-squares contour.

Discrete selections use only the predicted field and the DIII-D scoring geometry.
No reference contour, normal or true event enters branch selection. This is not
an inference decoder or a claim about the official scorer's derivatives.
"""

from dataclasses import dataclass

import numpy as np
import torch

from participant import fast_score
from participant.critical_level import stencil
from contour import grid_to_physical, resample_closed
from lcfs import _best_contour_at
from o_point import find_o_point


@dataclass
class Branch:
    edges: np.ndarray
    endpoints: np.ndarray
    bisection_indices: np.ndarray
    bisection_weights: np.ndarray
    bisection_bias: float
    event_indices: np.ndarray
    event_weights: np.ndarray
    event_bias: float
    event_kind: str
    actual: np.ndarray
    level: float
    decisions: tuple


def select(field):
    """Trace the unchanged scorer, retaining current grid edges and level branches."""
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (65, 65) or not np.isfinite(field).all():
        raise ValueError('Branch selection requires a finite 65x65 predicted field')
    r, z, coarse, mask = fast_score._geometry()
    sign = fast_score.official.AXIS_SIGN['DIII-D']
    phi = field * sign
    _, _, iz, ir, _ = find_o_point(phi, r, z, coarse)
    ids = np.arange(65 * 65).reshape(65, 65)
    border_ids = np.concatenate([ids[0], ids[-1], ids[:, 0], ids[:, -1]])
    low_id = int(border_ids[np.argmin(phi.ravel()[border_ids])])
    high_id = int(ids[iz, ir])
    lo, hi = float(phi.ravel()[low_id]), float(phi[iz, ir]) - 1e-9
    if hi <= lo:
        raise ValueError('Predicted field has no valid critical-level bracket')
    best = _best_contour_at(phi, hi, iz, ir, mask)
    if best is None:
        raise ValueError('Predicted field has no initial closed contour')
    low_weights, high_weights = np.array([1., 0.]), np.array([0., 1.])
    decisions = []
    for _ in range(fast_score.official.N_ITER):
        mid = .5 * (lo + hi)
        weights = .5 * (low_weights + high_weights)
        curve = _best_contour_at(phi, mid, iz, ir, mask)
        decisions.append(curve is not None)
        if curve is None:
            lo, low_weights = mid, weights
        else:
            hi, high_weights, best = mid, weights, curve
    physical = grid_to_physical(best, r, z)
    actual = resample_closed(physical, fast_score.official.N_POINTS)
    official, _ = fast_score.official.extract_lcfs_with_sign(
        field, r, z, 'DIII-D', coarse, mask,
        n_iter=fast_score.official.N_ITER, axis_sign=sign)
    if official is None or not np.array_equal(actual, official):
        raise ValueError('Current predicted branch does not reproduce the scorer')
    edges, endpoints = [], []
    for row, col in best:
        if row == round(row) and col != round(col):
            a, b = (int(row), int(np.floor(col))), (int(row), int(np.ceil(col)))
        elif col == round(col) and row != round(row):
            a, b = (int(np.floor(row)), int(col)), (int(np.ceil(row)), int(col))
        else:
            raise ValueError('Ambiguous grid-node contour vertex; no edge derivative selected')
        edges.append([ids[a], ids[b]])
        endpoints.append([[r[a[1]], z[a[0]]], [r[b[1]], z[b[0]]]])
    event = stencil(field)
    indices, weights = np.asarray(event['indices']), np.asarray(event['weights'])
    level = hi / sign
    return Branch(np.asarray(edges), np.asarray(endpoints),
                  np.array([low_id, high_id]), high_weights,
                  -1e-9 * float(high_weights[1]) / sign,
                  indices, weights, level - float(field.ravel()[indices] @ weights),
                  event['kind'], actual, level, tuple(decisions))


def vertices(field, branch, mode):
    """Differentiate edge interpolation with either declared level response."""
    flat = field.flatten().to(torch.float64)
    if mode == 'bisection':
        indices, weights, bias = (branch.bisection_indices, branch.bisection_weights,
                                  branch.bisection_bias)
    elif mode == 'event':
        indices, weights, bias = branch.event_indices, branch.event_weights, branch.event_bias
    else:
        raise ValueError('Unknown research derivative control')
    level = (flat[torch.as_tensor(indices)] * torch.as_tensor(weights)).sum() + bias
    edge_values = flat[torch.as_tensor(branch.edges)]
    denominator = edge_values[:, 1] - edge_values[:, 0]
    if torch.any(denominator == 0) or not torch.isfinite(denominator).all():
        raise FloatingPointError('Undefined contour edge interpolation')
    fraction = (level - edge_values[:, 0]) / denominator
    endpoints = torch.as_tensor(branch.endpoints)
    return endpoints[:, 0] + fraction[:, None] * (endpoints[:, 1] - endpoints[:, 0])


def resample(points):
    """Differentiate the scorer's 512 equally spaced arc-length samples."""
    length = (points[1:] - points[:-1]).norm(dim=1)
    if torch.any(length <= 0) or not torch.isfinite(length).all():
        raise FloatingPointError('Undefined contour arc-length segment')
    cumulative = torch.cat([length.new_zeros(1), length.cumsum(0)])
    targets = torch.arange(fast_score.official.N_POINTS, dtype=points.dtype) * (
        cumulative[-1] / fast_score.official.N_POINTS)
    index = torch.searchsorted(cumulative.detach(), targets.detach(), right=True) - 1
    index = index.clamp(0, len(length) - 1)
    fraction = (targets - cumulative[index]) / length[index]
    return points[index] + fraction[:, None] * (points[index + 1] - points[index])


def contour(field, branch, mode):
    return resample(vertices(field, branch, mode))


def kappa(points):
    width = points[:, 0].max() - points[:, 0].min()
    if not torch.isfinite(width) or width <= 0:
        raise FloatingPointError('Undefined predicted contour width')
    return (points[:, 1].max() - points[:, 1].min()) / width


def batch_kappa(fields, branches, mode):
    if len(fields) != len(branches):
        raise ValueError('Predicted branch count differs from current fields')
    return torch.stack([kappa(contour(field, branch, mode))
                        for field, branch in zip(fields, branches, strict=True)])
