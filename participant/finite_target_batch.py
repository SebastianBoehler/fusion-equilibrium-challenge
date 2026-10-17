"""Validated actual seven-component TRAIN loss targets; no inference inputs."""

import json

import numpy as np

from .artifacts import digest
from .fast_score import official


def usable_indices(valid, indices):
    picks = np.asarray(indices)
    if (picks.ndim != 1 or not len(picks) or picks.dtype.kind not in 'iu'
            or np.any(picks < 0) or np.any(picks >= len(valid))):
        raise ValueError('Invalid finite TRAIN target indices')
    if not valid[picks].all():
        raise ValueError('Excluded finite TRAIN target requested')
    return picks


def population_statistics(values, valid):
    count, mean, m2 = 0, np.zeros(7), np.zeros(7)
    for start in range(0, len(values), 8192):
        block = np.asarray(values[start:start + 8192][valid[start:start + 8192]])
        if not len(block):
            continue
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite usable seven-component TRAIN target')
        size, average = len(block), block.mean(0)
        delta = average - mean
        m2 += ((block - average) ** 2).sum(0) + delta ** 2 * count * size / (count + size)
        mean += delta * size / (count + size)
        count += size
    if not count:
        raise ValueError('No usable finite TRAIN targets')
    variance = m2 / count
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('Undefined seven-component TRAIN population variance')
    return {'frames': count, 'mean': mean.tolist(), 'variance': variance.tolist(),
            'population_denominator': count}


class FiniteTargetBatch:
    def __init__(self, path, train_index, manifest_sha256):
        validation = json.loads((path / 'validation.json').read_text())
        if not validation.get('complete') or not validation.get('training_only'):
            raise ValueError('Complete finite TRAIN target validation required')
        if validation['manifest_sha256'] != manifest_sha256:
            raise ValueError('Finite TRAIN target manifest changed')
        for name, expected in validation['file_sha256'].items():
            if digest(path / name) != expected:
                raise ValueError(f'Finite TRAIN target hash changed: {name}')
        index = json.loads((path / 'index.json').read_text())
        if (not index['training_only'] or not index['complete'] or
                index['names'] != train_index['names'] or index['offsets'] != train_index['offsets']
                or index['frames'] != train_index['frames'] or
                index['manifest_sha256'] != manifest_sha256 or
                index['components'] != list(official.CONS_SCALARS)):
            raise ValueError('Finite TRAIN target order/schema changed')
        self.values = np.load(path / 'values.npy', mmap_mode='r', allow_pickle=False)
        self.valid = np.load(path / 'valid.npy', mmap_mode='r', allow_pickle=False)
        n = train_index['frames']
        if (self.values.shape != (n, 7) or self.values.dtype != np.float64 or
                self.valid.shape != (n,) or self.valid.dtype != np.bool_ or
                int(self.valid.sum()) != validation['usable_frames']):
            raise ValueError('Finite TRAIN target shape/dtype/count changed')
        self.variances = np.asarray(validation['statistics']['variance'], np.float64)
        if self.variances.shape != (7,) or not np.isfinite(self.variances).all() or np.any(self.variances <= 0):
            raise ValueError('Invalid seven-component TRAIN normalization')
        self.valid_indices = np.flatnonzero(self.valid)

    def get(self, indices):
        return np.asarray(self.values[usable_indices(self.valid, indices)])
