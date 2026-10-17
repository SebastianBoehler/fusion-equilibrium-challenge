"""Batched fixed-contour inductance, with geometry prepared from training truth."""

import torch
from torch.nn import functional as F

def _grid_coordinate(query, coordinates):
    """Match np.interp fractional indices on rounded physical grids."""
    query = query.clamp(min=coordinates[0], max=coordinates[-1])
    lower = (torch.searchsorted(coordinates, query.contiguous()) - 1).clamp(0, len(coordinates) - 2)
    fraction = (query - coordinates[lower]) / (coordinates[lower + 1] - coordinates[lower])
    return 2 * (lower + fraction) / (len(coordinates) - 1) - 1


def geometry(contours, masks, r, z):
    """Pad exact contour vertices; vertex weights preserve trapezoidal quadrature."""
    if len(contours) != len(masks) or not len(contours):
        raise ValueError('Nonempty matching contours and masks required')
    if not torch.all(r > 0) or not torch.all(torch.diff(r) > 0) or not torch.all(torch.diff(z) > 0):
        raise ValueError('Invalid physical grid')
    points = torch.zeros(len(contours), max(map(len, contours)), 2, dtype=r.dtype, device=r.device)
    boundary_weights = torch.zeros(points.shape[:2], dtype=r.dtype, device=r.device)
    interior_weights = []
    for i, (contour, mask) in enumerate(zip(contours, masks, strict=True)):
        if len(contour) < 8 or not torch.isfinite(contour).all() or int(mask.sum()) < 4:
            raise ValueError('Invalid fixed training geometry')
        segments = torch.linalg.vector_norm(torch.roll(contour, -1, 0) - contour, dim=1)
        perimeter = segments.sum()
        if not torch.isfinite(perimeter) or perimeter <= 0:
            raise ValueError('Invalid training perimeter')
        count = len(contour)
        points[i, :count] = torch.stack((_grid_coordinate(contour[:, 0], r),
                                        _grid_coordinate(contour[:, 1], z)), dim=1)
        boundary_weights[i, :count] = (segments + torch.roll(segments, 1)) / (2 * perimeter)
        weights = mask * r[None, :]
        interior_weights.append(weights / weights.sum())
    return points, boundary_weights, torch.stack(interior_weights)


def inductance(psi, r, z, points, boundary_weights, interior_weights):
    """Same ratio as the scorer, without differentiating fixed contour movement."""
    if psi.ndim != 3 or psi.shape[1:] != (len(z), len(r)) or interior_weights.shape != psi.shape:
        raise ValueError('Flux and interior batch shapes must match')
    if points.shape[:2] != boundary_weights.shape or len(points) != len(psi):
        raise ValueError('Boundary batch shapes must match')
    if not torch.isfinite(psi).all():
        raise FloatingPointError('Nonfinite batch flux')
    dz, dr = torch.gradient(psi, spacing=(z, r), dim=(1, 2))
    bp = torch.linalg.vector_norm(torch.stack((dr, dz)), dim=0) / r[None, None, :]
    # Common rescaling cancels in the ratio and prevents weak-field squaring underflow.
    scale = bp.amax(dim=(1, 2), keepdim=True).detach()
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise FloatingPointError('Undefined zero-field inductance')
    bp = bp / scale
    interior = (bp.square() * interior_weights).sum((1, 2))
    sampled = F.grid_sample(bp[:, None], points[:, :, None], mode='bilinear',
                            padding_mode='border', align_corners=True)[:, 0, :, 0]
    boundary = (sampled * boundary_weights).sum(1)
    if not torch.isfinite(boundary).all() or torch.any(boundary <= 0):
        raise FloatingPointError('Undefined boundary-field inductance')
    result = interior / boundary.square()
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite batch inductance')
    return result
