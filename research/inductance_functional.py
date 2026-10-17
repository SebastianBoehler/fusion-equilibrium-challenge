"""Differentiable scored li on a fixed DIII-D training contour and interior mask."""

import torch
from torch.nn import functional as F
from participant.inductance_loss import _grid_coordinate


def inductance(psi, r, z, contour, inside):
    """Match published quadrature; contour/mask are fixed training targets.

    This does not differentiate contour extraction or predicted topology.
    Invalid zero-field denominators raise instead of changing normalization.
    """
    if psi.shape != (len(z), len(r)) or inside.shape != psi.shape:
        raise ValueError('Flux, coordinates and interior mask must match')
    if len(contour) < 8 or int(inside.sum()) < 4:
        raise ValueError('Insufficient training contour or interior support')
    if not torch.isfinite(psi).all() or not torch.isfinite(contour).all():
        raise FloatingPointError('Nonfinite flux or contour')
    dz, dr = torch.gradient(psi, spacing=(z, r), dim=(0, 1))
    bp = torch.linalg.vector_norm(torch.stack((dr, dz)), dim=0) / r[None, :]
    weights = inside * r[None, :]
    interior = (bp.square() * weights).sum() / weights.sum()
    points = torch.stack((_grid_coordinate(contour[:, 0], r), _grid_coordinate(contour[:, 1], z)), dim=1)
    boundary = F.grid_sample(bp[None, None], points[None, :, None],
                             mode='bilinear', padding_mode='border', align_corners=True)[0, 0, :, 0]
    lengths = torch.linalg.vector_norm(torch.roll(contour, -1, 0) - contour, dim=1)
    perimeter = lengths.sum()
    mean_boundary = ((boundary + torch.roll(boundary, -1)) * 0.5 * lengths).sum() / perimeter
    if not torch.isfinite(mean_boundary) or mean_boundary <= 0 or perimeter <= 0:
        raise FloatingPointError('Invalid boundary field or perimeter')
    result = interior / mean_boundary.square()
    if not torch.isfinite(result):
        raise FloatingPointError('Nonfinite inductance')
    return result
