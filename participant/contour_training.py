"""Scorer-aligned training targets and the bounded contour correction objective."""

import json

import numpy as np
import torch

from .contour_normal_loss import loss as normal_loss, physical_points, targets as normal_targets
from .inductance_batch import TargetBatch
from .inductance_loss import inductance
from .scored_contour_targets import DESCRIPTION

ARMS = ('aligned-li', 'aligned-li-kappa')
KAPPA_WEIGHT = 0.04419808835


class ScoredContourBatch(TargetBatch):
    def __init__(self, path, train_index):
        super().__init__(path, train_index)
        index = json.loads((path / 'index.json').read_text())
        if index.get('target') != DESCRIPTION or index['boundary_storage_width'] != 512:
            raise ValueError('Scorer-extracted training contour targets required')
        valid = np.load(path / 'valid.npy', mmap_mode='r')
        if valid.shape != (train_index['frames'],) or valid.dtype != np.bool_:
            raise ValueError('Training contour validity mask shape/dtype mismatch')
        self.valid_indices = np.flatnonzero(valid)
        if len(self.valid_indices) != index['usable_frames'] or not len(self.valid_indices):
            raise ValueError('Training contour usable frame count mismatch')
        self.valid = valid
        self.description = index['target']
        expected = {'points': (train_index['frames'], 512, 2),
                    'weights': (train_index['frames'], 512),
                    'mask_bits': (train_index['frames'], 529),
                    'target_li': (train_index['frames'],)}
        if any(self.arrays[k].shape != shape for k, shape in expected.items()):
            raise ValueError('Scored contour target array shape mismatch')

    def get(self, indices, device):
        if not self.valid[indices].all():
            raise ValueError('Undefined training contour requested')
        return super().get(indices, device)


def contour_objective(model, x, truth, r, z, geometry, target_li, variance, kappa_weight):
    # Use one forward pass: both terms must update the same predicted field.
    psi, _ = model(x)
    psi = psi.reshape(-1, 65, 65)
    actual_truth = truth.reshape(-1, 65, 65)
    pixel = (psi - actual_truth).square().sum((1, 2)).mean() / variance
    li = (inductance(psi, r, z, *geometry) - target_li).square().mean()
    points, _, interior = geometry
    prepared = normal_targets(actual_truth, r, z, points, physical_points(points, r, z), interior)
    kappa = normal_loss(psi, points, prepared)
    value = pixel + 0.05 * li + kappa_weight * kappa
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite scorer-contour correction objective')
    return value, pixel, li, kappa
