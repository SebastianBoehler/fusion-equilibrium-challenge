"""Preallocated permitted-TRAIN parent caches; no truth targets or learned head."""

from dataclasses import dataclass

import numpy as np
import torch

from .finetune_common import prepare_inputs
from .flux_head import decode


CHUNK_FRAMES = 256


@dataclass(frozen=True)
class ParentCache:
    indices: np.ndarray
    features: torch.Tensor
    fields: torch.Tensor
    scalars: torch.Tensor
    parent_std: torch.Tensor

    def payload_bytes(self):
        return {key: value.numel() * value.element_size()
                for key, value in vars(self).items() if isinstance(value, torch.Tensor)}


@torch.no_grad()
def cache_parent(model, raw, valid):
    """Keep features on the predictor device and full decoded fields only on CPU."""
    if (raw.ndim != 2 or raw.dtype != np.float32 or valid.ndim != 1
            or valid.dtype != np.bool_ or len(raw) != len(valid)
            or raw.shape[1] != model.config['inputs']):
        raise ValueError('Invalid permitted TRAIN input or eligibility schema')
    if model.training or model.parent.training:
        raise ValueError('Frozen parent cache requires evaluation mode')
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError('No eligible TRAIN cache support')
    device = model.correction_scale.device
    count = len(indices)
    features = torch.empty((count, model.correction.in_features), dtype=torch.float32, device=device)
    fields = torch.empty((count, 4225), dtype=torch.float32, device='cpu')
    scalars = torch.empty((count, model.config['scalars']), dtype=torch.float32, device='cpu')
    parent_std = torch.empty(count, dtype=torch.float32, device=device)
    params = {key: getattr(model, key) for key in
              ('x_mean', 'x_std', 'psi_mean', 'coef_scale', 'basis')}
    for start in range(0, count, CHUNK_FRAMES):
        end = min(start + CHUNK_FRAMES, count)
        values = np.asarray(raw[indices[start:end]])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite eligible permitted TRAIN input')
        inputs = prepare_inputs(values, params, model.config, device)
        hidden = model.parent.head[0](model.parent.blocks(model.parent.stem(inputs)))
        output = model.parent.head[1](hidden)
        base = decode(output, params, model.config)
        scale = base.std(1, unbiased=False)
        if (not torch.isfinite(hidden).all() or not torch.isfinite(output).all()
                or not torch.isfinite(base).all() or not torch.isfinite(scale).all()
                or torch.any(scale <= 0)):
            raise FloatingPointError('Invalid frozen parent cache chunk')
        features[start:end].copy_(hidden)
        fields[start:end].copy_(base.to('cpu'))
        scalars[start:end].copy_(output[:, model.config['components']:].to('cpu'))
        parent_std[start:end].copy_(scale)
    return ParentCache(indices, features, fields, scalars, parent_std)
