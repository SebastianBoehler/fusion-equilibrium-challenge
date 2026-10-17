"""Actual seven-component finite loss with a global TRAIN parameter constraint."""

import numpy as np
import torch
from torch.nn import functional as F

from participant import fast_score
from research.finite_metric_loss import check_deadline, predict
from research.inductance_contour_audit import derive_predictions


def actual_derived(fields, deadline):
    arrays = fields.detach().float().numpy().astype(np.float64).reshape(-1, 65, 65)
    parts = []
    for start in range(0, len(arrays), 64):
        check_deadline(deadline)
        block = arrays[start:start + 64]
        values, _, failures = derive_predictions(
            block, [None] * len(block), fast_score._geometry())
        if failures or not np.isfinite(values).all():
            raise FloatingPointError('Undefined perturbed actual derived component')
        parts.append(values)
    return np.concatenate(parts)


def seven_adjoint(model, fields, parent_std, targets, variances, generator, scale, deadline):
    count = len(fields)
    signs = torch.randint(0, 2, (count, 289), generator=generator).float() * 2 - 1
    step = scale * parent_std / model.correction_scale
    perturbation = F.interpolate((signs * step[:, None]).reshape(-1, 1, 17, 17),
        size=(65, 65), mode='bilinear', align_corners=True).flatten(1) * model.correction_scale
    plus = actual_derived(fields.detach() + perturbation, deadline)
    minus = actual_derived(fields.detach() - perturbation, deadline)
    index = fast_score.official.CONS_SCALARS.index('kappa')
    weighted_plus = np.square(plus - targets) / variances * variances[index]
    weighted_minus = np.square(minus - targets) / variances * variances[index]
    component_differences = (weighted_plus - weighted_minus) / (2 * step.numpy()[:, None])
    difference = component_differences.sum(axis=1)
    adjoint = torch.from_numpy(difference).to(fields.dtype)[:, None] * signs
    if not torch.isfinite(adjoint).all():
        raise FloatingPointError('Nonfinite seven-component finite adjoint')
    return adjoint, {'perturbed_valid_frames': 2 * count, 'all_seven_derived_valid': True,
        'component_loss_directional_rms': {key: float(np.sqrt(np.mean(component_differences[:, j] ** 2)))
                                         for j, key in enumerate(fast_score.official.CONS_SCALARS)},
        'latent_adjoint_rms': float(adjoint.square().mean().sqrt()),
        'physical_perturbation_rms_relative_parent_std': float(
            (perturbation.square().mean(1).sqrt() / parent_std).mean())}


@torch.no_grad()
def project_parameters(model, features, base, parent_std, bound):
    """Apply one scalar to learned weights, then validate actual native fields."""
    _, before = predict(model, features, base)
    maximum = float(((before - base).square().mean(1).sqrt() / parent_std).max())
    factor = .99 * bound / maximum if maximum > bound else 1.
    if factor < 1:
        for parameter in model.correction.parameters():
            original = parameter.clone()
            parameter.mul_(factor)
            if not torch.equal(parameter, original * factor):
                raise ValueError('Global parameter projection differs between weights')
    _, fields = predict(model, features, base)
    actual_max = float(((fields - base).square().mean(1).sqrt() / parent_std).max())
    if not np.isfinite(actual_max) or actual_max > bound:
        raise FloatingPointError('Single global projection failed the declared physical bound')
    return fields, {'factor': factor, 'relative_rms_before': maximum,
                    'relative_rms_after': actual_max, 'global_parameters_only': True}
