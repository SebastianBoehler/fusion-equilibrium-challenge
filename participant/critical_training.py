"""Shared critical-level recipe and verified training-only target loader."""

import json

import numpy as np
import torch

from .contour_normal_loss import physical_points, targets
from .contour_training import KAPPA_WEIGHT, ScoredContourBatch
from .critical_loss import loss, live_level
from .inductance_loss import inductance

ARMS = ('critical-kappa', 'critical-li-kappa')
DESCRIPTION = 'true scorer active critical event; local label-anchored live level; monotone one-cell displacement'


class CriticalBatch(ScoredContourBatch):
    def __init__(self, contours, critical, train_index):
        super().__init__(contours, train_index)
        index = json.loads((critical / 'index.json').read_text())
        if (index['names'] != train_index['names'] or index['offsets'] != train_index['offsets']
                or index['frames'] != train_index['frames']
                or not index['complete'] or not index['training_only'] or index['target'] != DESCRIPTION
                or index['manifest_sha256'] != self.manifest_sha):
            raise ValueError('Critical target order, completion or manifest differs')
        self.critical = {key: np.load(critical / f'{key}.npy', mmap_mode='r')
                         for key in ('indices', 'weights', 'valid')}
        n = train_index['frames']
        if (self.critical['indices'].shape != (n, 2) or self.critical['indices'].dtype != np.int16
                or self.critical['weights'].shape != (n, 2) or self.critical['weights'].dtype != np.float32
                or self.critical['valid'].shape != (n,) or self.critical['valid'].dtype != np.bool_):
            raise ValueError('Critical target shape or dtype mismatch')
        if np.any(self.critical['valid'] & ~self.valid):
            raise ValueError('Critical target admits an invalid reference contour')
        self.valid_indices = np.flatnonzero(self.critical['valid'])
        if len(self.valid_indices) != index['usable_frames'] or not len(self.valid_indices):
            raise ValueError('Critical target usable count differs')
        self.critical_description = index['target']

    def critical_get(self, indices, device):
        if not self.critical['valid'][indices].all():
            raise ValueError('Unsupported critical target requested')
        return (torch.from_numpy(np.asarray(self.critical['indices'][indices], np.int64)).to(device),
                torch.from_numpy(np.asarray(self.critical['weights'][indices])).to(device))


def objective(model, x, truth, r, z, geometry, li_target, variance, indices, weights, arm):
    if arm not in ARMS:
        raise ValueError('Unknown critical-level arm')
    psi, _ = model(x)
    psi = psi.reshape(-1, 65, 65)
    actual_truth = truth.reshape(-1, 65, 65)
    pixel = (psi - actual_truth).square().sum((1, 2)).mean() / variance
    li = (inductance(psi, r, z, *geometry) - li_target).square().mean()
    points, _, interior = geometry
    prepared = targets(actual_truth, r, z, points, physical_points(points, r, z), interior)
    kappa = loss(psi, points, prepared, indices, weights, live_level(actual_truth, indices, weights))
    value = pixel + KAPPA_WEIGHT * kappa + (0.05 * li if arm == 'critical-li-kappa' else 0)
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite critical-level objective')
    return value, pixel, li, kappa
