"""Shared target validation and input/scalar handling for bounded fine-tuning runs."""

import json

import numpy as np
import torch

from .rich import canonicalize_currents


def geometry_targets(path, training_index, device):
    index = json.loads((path / 'index.json').read_text())
    if index['names'] != training_index['names'] or index['offsets'] != training_index['offsets']:
        raise ValueError('Geometry targets do not match the exact training frame order')
    return [torch.from_numpy(np.load(path / f'{key}.npy')).to(device)
            for key in ('points', 'values', 'span', 'valid')]


def prepare_inputs(values, params, config, device):
    transformed = canonicalize_currents(values) if config.get('canon_ip') else np.asarray(values)
    return ((torch.from_numpy(transformed).to(device) - params['x_mean'])
            / params['x_std']).clamp(-config['clip'], config['clip'])


def scalar_loss(output, target, weights):
    error = output - target
    finite = torch.isfinite(error)
    return (torch.where(finite, error, 0).square().sum(0)
            / finite.sum(0).clamp(min=1) * weights).sum()
