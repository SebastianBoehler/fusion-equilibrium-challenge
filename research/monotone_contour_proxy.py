"""Training-only monotonic displacement prototype; leaves frozen024 unchanged."""

import math

import torch
from torch.nn import functional as F

from participant.contour_normal_loss import elongation


def displacement(residual, gradient, radius):
    norm = gradient.norm(dim=-1)
    if not torch.isfinite(norm).all() or torch.any(norm <= 0):
        raise FloatingPointError('Undefined true contour normal')
    # f(t)=2/pi*atan(pi*t/2): f'(t)>0, f'(0)=1, |f(t)|<1.
    # Native atan backward avoids cancellation when large residuals saturate.
    ratio = residual / (radius * norm)
    signed_magnitude = radius * (2 / math.pi) * torch.atan((math.pi / 2) * ratio)
    value = -signed_magnitude[:, :, None] * gradient / norm[:, :, None]
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite monotonic contour displacement')
    return value


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
    return contour + displacement(sampled - values, gradient, radius)


def loss(prediction, points, prepared):
    return (elongation(projected(prediction, points, prepared)) - prepared[-2]).square().mean()
