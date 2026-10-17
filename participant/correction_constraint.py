"""Bilinear correction energy and a single global TRAIN parameter projection."""

import torch
from torch.nn import functional as F


def interpolation_gram(model):
    basis = torch.eye(289, dtype=torch.float64, device=model.correction_scale.device)
    decoded = F.interpolate(basis.reshape(-1, 1, 17, 17), size=(65, 65),
                            mode='bilinear', align_corners=True).flatten(1)
    return decoded @ decoded.T / 4225 * model.correction_scale.double().square()


def correction_rms(latent, gram):
    coordinates = latent.double()
    energy = ((coordinates @ gram) * coordinates).sum(1)
    if not torch.isfinite(energy).all() or torch.any(energy < 0):
        raise FloatingPointError('Invalid bilinear correction energy')
    return energy.sqrt()


@torch.no_grad()
def maximum_relative_rms(model, features, parent_std, gram):
    if len(features) != len(parent_std) or torch.any(parent_std <= 0):
        raise ValueError('Constraint support or parent scale invalid')
    maximum = 0.
    for start in range(0, len(features), 4096):
        latent = model.correction(features[start:start + 4096])
        value = correction_rms(latent, gram) / parent_std[start:start + 4096].double()
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite global correction support')
        maximum = max(maximum, float(value.max()))
    return maximum


@torch.no_grad()
def project_correction(model, features, parent_std, gram, bound):
    maximum = maximum_relative_rms(model, features, parent_std, gram)
    factor = .99 * bound / maximum if maximum > bound else 1.
    if factor < 1:
        for parameter in model.correction.parameters():
            original = parameter.clone()
            parameter.mul_(factor)
            if not torch.equal(parameter, original * factor):
                raise ValueError('Shared global correction factor changed')
    after = maximum_relative_rms(model, features, parent_std, gram)
    if after > bound:
        raise FloatingPointError('Global parameter projection failed the Gram bound')
    return {'factor': factor, 'relative_rms_before': maximum, 'relative_rms_after': after,
            'support_frames': len(features), 'global_parameters_only': True}
