"""Research-only finite measurements through the unchanged contour extractor."""

import time

import numpy as np
import torch
from torch.nn import functional as F

from participant import fast_score
from participant.flux_head import decode
from research.inductance_contour_audit import derive_predictions, error


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Predeclared600-second CPU probe limit reached')


def cache_predictor(model, inputs):
    """Cache frozen permitted-input features, retaining a real learned output head."""
    with torch.no_grad():
        features = model.parent.head[0](model.parent.blocks(model.parent.stem(inputs)))
        outputs = model.parent.head[1](features)
        params = {key: getattr(model, key) for key in ('psi_mean', 'coef_scale', 'basis')}
        base = decode(outputs, params, model.config)
        scalars = outputs[:, model.config['components']:]
        actual, actual_scalars = model(inputs)
    if not torch.equal(base, actual) or not torch.equal(scalars, actual_scalars):
        raise ValueError('New cached predictor does not reproduce the actual zero model')
    std = base.std(dim=1, unbiased=False)
    if not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError('Undefined parent physical field scale')
    return features, base, scalars, std


def predict(model, features, base):
    latent = model.correction(features)
    spatial = F.interpolate(latent.reshape(-1, 1, 17, 17), size=(65, 65),
                            mode='bilinear', align_corners=True).flatten(1)
    fields = base + spatial * model.correction_scale
    if not torch.isfinite(fields).all():
        raise FloatingPointError('Nonfinite updated predictor fields')
    return latent, fields


def actual_kappa(fields, deadline):
    """Score float32 stored fields, reselecting all contour branches on each call."""
    r, z, coarse, mask = fast_score._geometry()
    official = fast_score.official
    values = []
    arrays = fields.detach().to(torch.float32).numpy().astype(np.float64).reshape(-1, 65, 65)
    for i, field in enumerate(arrays):
        if i % 64 == 0:
            check_deadline(deadline)
        curve, _ = official.extract_lcfs_with_sign(
            field, r, z, 'DIII-D', coarse, mask,
            n_iter=official.N_ITER, axis_sign=official.AXIS_SIGN['DIII-D'])
        if curve is None or not np.isfinite(curve).all():
            raise FloatingPointError(f'Undefined perturbed contour at TRAIN frame{i}')
        width = float(np.ptp(curve[:, 0]))
        value = float(np.ptp(curve[:, 1])) / width if width > 0 else np.nan
        if not np.isfinite(value) or not .8 <= value <= 4:
            raise FloatingPointError(f'Invalid perturbed elongation at TRAIN frame{i}')
        values.append(value)
    return np.asarray(values)


def finite_adjoint(model, fields, parent_std, targets, generator, relative_scale, deadline):
    """Estimate the latent-coordinate loss adjoint with two actual measurements."""
    count = len(fields)
    signs = torch.randint(0, 2, (count, 17 * 17), generator=generator).float() * 2 - 1
    step = relative_scale * parent_std / model.correction_scale
    perturbation = F.interpolate((signs * step[:, None]).reshape(-1, 1, 17, 17),
        size=(65, 65), mode='bilinear', align_corners=True).flatten(1) * model.correction_scale
    plus = actual_kappa(fields.detach() + perturbation, deadline)
    minus = actual_kappa(fields.detach() - perturbation, deadline)
    difference = (np.square(plus - targets) - np.square(minus - targets)) / (2 * step.numpy())
    adjoint = torch.from_numpy(difference).to(fields.dtype)[:, None] * signs
    if not torch.isfinite(adjoint).all():
        raise FloatingPointError('Nonfinite finite-measurement adjoint')
    return adjoint, {'perturbed_valid_frames': 2 * count,
        'loss_directional_difference_rms': float(np.sqrt(np.mean(difference ** 2))),
        'latent_adjoint_rms': float(adjoint.square().mean().sqrt()),
        'physical_perturbation_rms_relative_parent_std': float(
            (perturbation.square().mean(1).sqrt() / parent_std).mean())}


def measured_metrics(fields, truth, base, parent_std, reference, deadline):
    check_deadline(deadline)
    arrays = fields.detach().float().numpy().astype(np.float64).reshape(-1, 65, 65)
    values, boundaries, failed = derive_predictions(arrays, reference[0], fast_score._geometry())
    check_deadline(deadline)
    if failed or len(boundaries) != len(fields) or not np.isfinite(values).all():
        raise FloatingPointError('Updated TRAIN extraction or derived value failed')
    targets = reference[1]
    if not np.isfinite(targets).all():
        raise FloatingPointError('Undefined pinned TRAIN reference scalar')
    variance = np.var(targets, axis=0)
    if np.any(variance <= 0):
        raise ValueError('Undefined TRAIN variance for component comparison')
    result = {'frames': len(fields), 'extraction_failures': failed,
        'derived_nonfinite': int(np.sum(~np.isfinite(values))),
        'boundary_mean': float(np.mean(boundaries)),
        'pixel_sse': float((fields.detach() - truth.flatten(1)).double().square().sum()),
        'relative_correction_rms_max': float(
            ((fields.detach() - base).square().mean(1).sqrt() / parent_std).max()),
        'derived': {key: error(values[:, j], targets[:, j])
                    for j, key in enumerate(fast_score.official.CONS_SCALARS)},
        'mean_training_standardized_derived_mse': float(np.mean(
            np.mean(np.square(values - targets), axis=0) / variance))}
    return result, values
