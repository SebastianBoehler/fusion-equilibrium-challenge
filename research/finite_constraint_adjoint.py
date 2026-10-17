"""Bounded seven-component CPU loss adjoint; numerical validation is pending."""

import time

import numpy as np
import torch
from torch.nn import functional as F

from participant.finite_metrics import measure_derived
from research.finite_device_bridge import MAX_BATCH_FRAMES


RELATIVE_STEP = 0.001


def _measure(fields, deadline):
    parts = []
    for start in range(0, len(fields), 32):
        if time.monotonic() >= deadline:
            raise TimeoutError('Seven-component CPU measurement deadline reached')
        parts.append(measure_derived(fields[start:start + 32].numpy()))
    if time.monotonic() >= deadline:
        raise TimeoutError('Seven-component CPU measurement deadline reached')
    return np.concatenate(parts)


def cpu_constraint_adjoint(fields, parent_std, correction_scale, targets,
                           full_train_variances, generator, deadline):
    """Targets are loss inputs only; return the mean-loss CPU latent adjoint."""
    if (not isinstance(fields, torch.Tensor) or fields.ndim != 2
            or not 0 < len(fields) <= MAX_BATCH_FRAMES or fields.shape[1] != 4225
            or fields.dtype != torch.float32 or fields.device.type != 'cpu'
            or fields.requires_grad):
        raise ValueError('Expected bounded detached float32 CPU fields')
    count = len(fields)
    if (not isinstance(parent_std, torch.Tensor) or parent_std.shape != (count,)
            or parent_std.dtype != torch.float32 or parent_std.device.type != 'cpu'
            or parent_std.requires_grad):
        raise ValueError('Expected matching frozen float32 CPU parent deviations')
    if (not isinstance(correction_scale, torch.Tensor)
            or correction_scale.numel() != 1 or correction_scale.dtype != torch.float32
            or correction_scale.device.type != 'cpu' or correction_scale.requires_grad):
        raise ValueError('Expected frozen float32 CPU correction scale')
    if (not isinstance(targets, np.ndarray) or targets.shape != (count, 7)
            or targets.dtype != np.float64 or not np.isfinite(targets).all()):
        raise ValueError('Expected finite float64 seven-component TRAIN loss targets')
    if (not isinstance(full_train_variances, np.ndarray)
            or full_train_variances.shape != (7,)
            or full_train_variances.dtype != np.float64
            or not np.isfinite(full_train_variances).all()
            or np.any(full_train_variances <= 0)):
        raise ValueError('Expected positive full TRAIN float64 population variances')
    if (not isinstance(generator, torch.Generator) or generator.device.type != 'cpu'
            or not np.isfinite(deadline) or time.monotonic() >= deadline):
        raise ValueError('Expected CPU generator and future finite measurement deadline')
    scale = correction_scale.reshape(())
    if (not torch.isfinite(fields).all() or not torch.isfinite(parent_std).all()
            or torch.any(parent_std <= 0) or not torch.isfinite(scale) or scale <= 0):
        raise FloatingPointError('Undefined physical field or perturbation scale')
    signs = torch.randint(0, 2, (count, 289), generator=generator).to(torch.float32) * 2 - 1
    step = RELATIVE_STEP * parent_std / scale
    if not torch.isfinite(step).all() or torch.any(step <= 0):
        raise FloatingPointError('Undefined latent perturbation step')
    perturbation = F.interpolate((signs * step[:, None]).reshape(-1, 1, 17, 17),
        size=(65, 65), mode='bilinear', align_corners=True).flatten(1) * scale
    plus = _measure(fields + perturbation, deadline)
    minus = _measure(fields - perturbation, deadline)
    plus_loss = (np.square(plus - targets) / full_train_variances).mean(axis=1)
    minus_loss = (np.square(minus - targets) / full_train_variances).mean(axis=1)
    direction = (plus_loss - minus_loss) / (2 * step.numpy() * count)
    adjoint = torch.from_numpy(direction).to(torch.float32)[:, None] * signs
    if not torch.isfinite(adjoint).all():
        raise FloatingPointError('Nonfinite seven-component CPU latent adjoint')
    return adjoint
