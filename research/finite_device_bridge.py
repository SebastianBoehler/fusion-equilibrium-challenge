"""Bounded research transfers; numerical and CUDA validation remain required."""

import torch
from torch.nn import functional as F


MAX_BATCH_FRAMES = 2048


def _check_latent(latent):
    if (latent.ndim != 2 or not 0 < len(latent) <= MAX_BATCH_FRAMES
            or latent.shape[1] != 289 or latent.dtype != torch.float32
            or latent.device.type not in ('cpu', 'cuda')):
        raise ValueError('Expected at most2048 float32 CPU/CUDA latent rows of width289')
    if not torch.isfinite(latent).all():
        raise FloatingPointError('Nonfinite correction coordinates')


def reconstruct_cpu_fields(latent, parent_fields, correction_scale):
    """Detach only a bounded latent batch; keep decoded parent fields on CPU."""
    _check_latent(latent)
    if (parent_fields.shape != (len(latent), 4225)
            or parent_fields.dtype != torch.float32
            or parent_fields.device.type != 'cpu' or parent_fields.requires_grad):
        raise ValueError('Expected matching detached float32 CPU parent fields')
    if (correction_scale.numel() != 1 or correction_scale.dtype != torch.float32
            or correction_scale.device != latent.device or correction_scale.requires_grad):
        raise ValueError('Expected frozen float32 scalar scale on the latent device')
    if not torch.isfinite(parent_fields).all():
        raise FloatingPointError('Nonfinite parent fields')
    scale = correction_scale.detach().to('cpu').reshape(())
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError('Correction scale must be positive and finite')
    with torch.no_grad():
        coordinates = latent.detach().to('cpu')
        spatial = F.interpolate(coordinates.reshape(-1, 1, 17, 17),
                                size=(65, 65), mode='bilinear', align_corners=True)
        fields = parent_fields + spatial.flatten(1) * scale
    if not torch.isfinite(fields).all():
        raise FloatingPointError('Nonfinite reconstructed CPU fields')
    return fields


def cpu_adjoint_to_device(latent, cpu_adjoint):
    """Return a detached adjoint; the caller retains the original latent graph."""
    _check_latent(latent)
    if not latent.requires_grad:
        raise ValueError('Latent coordinates must retain their training graph')
    if (cpu_adjoint.shape != latent.shape or cpu_adjoint.dtype != torch.float32
            or cpu_adjoint.device.type != 'cpu' or cpu_adjoint.requires_grad):
        raise ValueError('Expected matching detached float32 CPU latent adjoint')
    if not torch.isfinite(cpu_adjoint).all():
        raise FloatingPointError('Nonfinite CPU latent adjoint')
    return cpu_adjoint.detach().to(latent.device)
