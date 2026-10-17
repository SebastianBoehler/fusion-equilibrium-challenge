"""Identify the active discrete event defining the scorer's critical flux level."""

import numpy as np
import torch
from torch.nn import functional as F

from participant import fast_score
from participant.contour_normal_loss import elongation
from research.contour_level_audit import critical_trace
from research.monotone_contour_proxy import displacement

from lcfs import _best_contour_at
from o_point import find_o_point


def stencil(field):
    field = np.asarray(field, dtype=float)
    level, _ = critical_trace(field)
    r, z, coarse, mask_f = fast_score._geometry()
    sign = fast_score.official.AXIS_SIGN['DIII-D']
    phi = field * sign
    _, _, iz, ir, _ = find_o_point(phi, r, z, coarse)
    contour = _best_contour_at(phi, level * sign, iz, ir, mask_f)
    if contour is None:
        raise ValueError('Critical level does not reproduce its valid contour')
    border = np.concatenate([phi[0], phi[-1], phi[:, 0], phi[:, -1]])
    tolerance = 3 * (float(phi[iz, ir]) - float(border.min())) / (2 ** fast_score.official.N_ITER) + 1e-12
    candidates = []
    # Each marching-squares vertex is on a grid edge. Mask contact on that
    # edge has a fixed location; the critical flux is its two-pixel interpolation.
    for row, col in contour:
        edge = None
        if abs(row - round(row)) < 1e-8:
            a = (int(round(row)), int(np.floor(col)))
            b = (a[0], min(a[1] + 1, 64))
            edge = (a, b)
        elif abs(col - round(col)) < 1e-8:
            a = (int(np.floor(row)), int(round(col)))
            b = (min(a[0] + 1, 64), a[1])
            edge = (a, b)
        if edge is None:
            raise ValueError('Marching-squares vertex is not on a grid edge')
        a, b = edge
        ma, mb = mask_f[a], mask_f[b]
        if ma == mb:
            continue
        t = (.5 - ma) / (mb - ma)
        if 0 <= t <= 1:
            value = field[a] * (1 - t) + field[b] * t
            candidates.append((abs(value - level), 'envelope_contact',
                               [a[0] * 65 + a[1], b[0] * 65 + b[1]], [1 - t, t]))
    # Marching-squares topology also changes when the level crosses a node.
    # Inspect every near-level node. The globally nearest value can be an
    # unrelated coincidence outside the enclosing contour (observed on frame277).
    for node in np.argwhere(abs(field - level) <= tolerance):
        if np.linalg.norm(contour - node, axis=1).min() < 1.5:
            i = int(node[0] * 65 + node[1])
            value = field[tuple(node)]
            candidates.append((abs(value - level), 'topology_node', [i, i], [1., 0.]))
    if not candidates:
        raise ValueError('No supported active critical-level event')
    gap, kind, indices, weights = min(candidates)
    if gap > tolerance:
        raise ValueError(f'Critical-level active event unresolved: gap={gap}, tolerance={tolerance}')
    return {'indices': [int(i) for i in indices], 'weights': [float(w) for w in weights], 'kind': kind,
            'level': level, 'event_level_gap': gap, 'bisection_tolerance': tolerance}


def live_level(prediction, indices, weights):
    return (prediction.flatten(1).gather(1, indices) * weights).sum(1)


def projected(prediction, points, prepared, indices, weights, reference_level):
    values, gradient, contour, interior, _, centered, _, radius = prepared
    pred_mean = (prediction * interior).sum((1, 2), keepdim=True)
    pred_centered = prediction - pred_mean
    variance = (pred_centered.square() * interior).sum((1, 2), keepdim=True)
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise FloatingPointError('Undefined predicted flux alignment')
    slope = (pred_centered * centered * interior).sum((1, 2), keepdim=True) / variance
    sampled = F.grid_sample(prediction[:, None], points[:, :, None], align_corners=True)[:, 0, :, 0]
    residual = ((sampled - live_level(prediction, indices, weights)[:, None]) * slope.flatten()[:, None]
                - (values - reference_level[:, None]))
    return contour + displacement(residual, gradient, radius)


def loss(prediction, points, prepared, indices, weights, reference_level):
    return (elongation(projected(prediction, points, prepared, indices, weights, reference_level))
            - prepared[-2]).square().mean()
