"""Label-anchored critical-level contour loss; never used by inference."""

import math

import torch
from torch.nn import functional as F

from .contour_normal_loss import elongation


def live_level(prediction, indices, weights):
    return (prediction.flatten(1).gather(1, indices) * weights).sum(1)


def projected(prediction, points, prepared, indices, weights, reference_level):
    values, gradient, contour, interior, _, centered, _, radius = prepared
    pred_mean = (prediction * interior).sum((1, 2), keepdim=True)
    pred_centered = prediction - pred_mean
    variance = (pred_centered.square() * interior).sum((1, 2), keepdim=True)
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise FloatingPointError('Undefined critical-level amplitude alignment')
    slope = (pred_centered * centered * interior).sum((1, 2), keepdim=True) / variance
    sampled = F.grid_sample(prediction[:, None], points[:, :, None], align_corners=True)[:, 0, :, 0]
    residual = ((sampled - live_level(prediction, indices, weights)[:, None]) * slope.flatten()[:, None]
                - (values - reference_level[:, None]))
    norm = gradient.norm(dim=-1)
    if not torch.isfinite(norm).all() or torch.any(norm <= 0):
        raise FloatingPointError('Undefined true critical contour normal')
    ratio = residual / (radius * norm)
    magnitude = radius * (2 / math.pi) * torch.atan((math.pi / 2) * ratio)
    displacement = -magnitude[:, :, None] * gradient / norm[:, :, None]
    result = contour + displacement
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite critical-level projected contour')
    return result


def loss(prediction, points, prepared, indices, weights, reference_level):
    estimated = elongation(projected(prediction, points, prepared, indices, weights, reference_level))
    return (estimated - prepared[-2]).square().mean()
