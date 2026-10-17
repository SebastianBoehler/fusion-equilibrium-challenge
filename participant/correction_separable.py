"""Separable bilinear energy and one global TRAIN correction projection."""

import math

import torch
from torch.nn import functional as F


def interpolation_factor(model):
    scale = model.correction_scale.detach().double().clone()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError('Invalid frozen correction scale')
    hats = F.interpolate(torch.eye(17, dtype=torch.float64, device=scale.device)
                         .reshape(17, 1, 17), size=65, mode='linear', align_corners=True)
    gram = hats[:, 0] @ hats[:, 0].T / 65
    positions = torch.arange(17, device=scale.device)
    distance = positions[:, None] - positions[None, :]
    if torch.any(gram[distance.abs() > 1] != 0):
        raise ValueError('Bilinear one-dimensional Gram is not tridiagonal')
    lower = torch.linalg.cholesky(gram)
    if torch.any(lower[(distance < 0) | (distance > 1)] != 0):
        raise ValueError('Bilinear Cholesky factor is not bidiagonal')
    return {'diagonal': lower.diagonal().clone(),
            'subdiagonal': lower.diagonal(-1).clone(), 'scale_square': scale.square(),
            'one_dimensional_gram': gram}


def _rms(latent, factor):
    if latent.ndim != 2 or not len(latent) or latent.shape[1] != 289:
        raise ValueError('Expected nonempty289-coordinate correction support')
    if not latent.is_floating_point() or latent.device != factor['scale_square'].device:
        raise ValueError('Correction coordinates and factor device/dtype differ')
    coordinates = latent.double().reshape(-1, 17, 17)
    diagonal, subdiagonal = factor['diagonal'], factor['subdiagonal']
    # G = L L.T; physical energy is ||L.T C L||_F squared times scale squared.
    vertical = coordinates * diagonal[None, :, None]
    vertical[:, :-1] += coordinates[:, 1:] * subdiagonal[None, :, None]
    transformed = vertical * diagonal[None, None, :]
    transformed[:, :, :-1] += vertical[:, :, 1:] * subdiagonal[None, None, :]
    return (transformed.square().sum((1, 2)) * factor['scale_square']).sqrt()


def correction_rms(latent, factor):
    value = _rms(latent, factor)
    if not torch.isfinite(value).all():
        raise FloatingPointError('Nonfinite separable correction energy')
    return value


@torch.no_grad()
def maximum_relative_rms(model, features, parent_std, factor):
    if (features.ndim != 2 or not len(features) or
            features.shape[1] != model.correction.in_features or
            parent_std.ndim != 1 or len(features) != len(parent_std) or
            features.device != parent_std.device or
            not parent_std.is_floating_point()):
        raise ValueError('Invalid global correction support')
    if not torch.isfinite(parent_std).all() or torch.any(parent_std <= 0):
        raise ValueError('Invalid parent physical field scale')
    maximum = torch.zeros((), dtype=torch.float64, device=features.device)
    for start in range(0, len(features), 4096):
        latent = model.correction(features[start:start + 4096])
        relative = _rms(latent, factor) / parent_std[start:start + 4096].double()
        maximum = torch.maximum(maximum, relative.max())
    # Keep chunk maxima on their device; collect the final maximum once.
    value = float(maximum)
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite global correction energy')
    return value


@torch.no_grad()
def project_correction(model, features, parent_std, factor, bound):
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError('Invalid global correction bound')
    before = maximum_relative_rms(model, features, parent_std, factor)
    multiplier = .99 * bound / before if before > bound else 1.
    if multiplier < 1:
        for parameter in model.correction.parameters():
            original = parameter.clone()
            parameter.mul_(multiplier)
            if not torch.equal(parameter, original * multiplier):
                raise ValueError('Shared global parameter factor changed')
    after = maximum_relative_rms(model, features, parent_std, factor)
    if after > bound:
        raise FloatingPointError('Separable global correction projection failed')
    return {'factor': multiplier, 'relative_rms_before': before,
            'relative_rms_after': after, 'support_frames': len(features),
            'global_parameters_only': True}
