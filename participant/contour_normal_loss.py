"""First-order contour displacement and elongation from a predicted flux field."""

import torch
from torch.nn import functional as F


def physical_points(points, r, z):
    """Invert fractional grid indices on the same rounded physical grid."""
    if points.shape[-1] != 2 or not torch.isfinite(points).all() or torch.any(points.abs() > 1):
        raise ValueError('Finite in-grid contour points required')
    coordinates = []
    for axis, grid in enumerate((r, z)):
        index = (points[..., axis] + 1) * ((len(grid) - 1) / 2)
        lower = index.floor().long().clamp(0, len(grid) - 2)
        coordinates.append(grid[lower] + (index - lower) * (grid[lower + 1] - grid[lower]))
    return torch.stack(coordinates, dim=-1)


def targets(truth, r, z, points, contours, interior):
    dz, dr = torch.gradient(truth, spacing=(z, r), dim=(1, 2))
    gradient = torch.stack((dr, dz), dim=1)
    sampled = F.grid_sample(gradient, points[:, :, None], align_corners=True)[:, :, :, 0].transpose(1, 2)
    norm_squared = sampled.square().sum(-1)
    if not torch.isfinite(norm_squared).all() or torch.any(norm_squared <= 0):
        raise FloatingPointError('Undefined true contour normal')
    values = F.grid_sample(truth[:, None], points[:, :, None], align_corners=True)[:, 0, :, 0]
    mean = (truth * interior).sum((1, 2), keepdim=True)
    centered = truth - mean
    contour = torch.as_tensor(contours, dtype=truth.dtype, device=truth.device)
    target_kappa = elongation(contour)
    radius = torch.minimum(torch.diff(r).median(), torch.diff(z).median())
    if not torch.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid physical grid spacing for local displacement')
    return values, sampled, contour, interior, mean, centered, target_kappa, radius


def elongation(contour):
    width = contour[:, :, 0].amax(1) - contour[:, :, 0].amin(1)
    height = contour[:, :, 1].amax(1) - contour[:, :, 1].amin(1)
    if not torch.isfinite(width).all() or torch.any(width <= 0):
        raise FloatingPointError('Undefined projected contour width')
    return height / width


def projected(prediction, points, prepared):
    values, gradient, contour, interior, mean, centered, _, radius = prepared
    pred_mean = (prediction * interior).sum((1, 2), keepdim=True)
    pred_centered = prediction - pred_mean
    variance = (pred_centered.square() * interior).sum((1, 2), keepdim=True)
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise FloatingPointError('Undefined amplitude alignment for predicted flux')
    slope = (pred_centered * centered * interior).sum((1, 2), keepdim=True) / variance
    aligned = pred_centered * slope + mean
    sampled = F.grid_sample(aligned[:, None], points[:, :, None], align_corners=True)[:, 0, :, 0]
    residual = sampled - values
    # Smooth damping bounds the first-order movement by one physical grid cell:
    # |e| g / (g² + (e / (2 radius))²) <= radius for every nonzero gradient.
    denominator = gradient.square().sum(-1) + (residual / (2 * radius)).square()
    displacement = -residual[:, :, None] * gradient / denominator[:, :, None]
    result = contour + displacement
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite projected contour')
    return result


def loss(prediction, points, prepared):
    estimated = elongation(projected(prediction, points, prepared))
    return (estimated - prepared[-2]).square().mean()
