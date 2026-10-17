"""Read-only pooled training target moments for scored li and contour elongation."""

from hashlib import file_digest
import json
from pathlib import Path
import sys
import time

import numpy as np


def update(acc, values):
    n = len(values)
    if n == 0 or not np.isfinite(values).all():
        raise ValueError('Nonempty finite training targets required')
    mean = float(values.mean())
    m2 = float(np.square(values - mean).sum())
    old_n = acc['count']
    delta = mean - acc['mean']
    total = old_n + n
    acc['mean'] += delta * n / total
    acc['M2'] += m2 + delta * delta * old_n * n / total
    acc['count'] = total
    acc['min'] = min(acc['min'], float(values.min()))
    acc['max'] = max(acc['max'], float(values.max()))


def coordinate(value, grid):
    index = (value.astype(float) + 1) * 32
    if np.any(index < 0) or np.any(index > 64):
        raise ValueError('Out-of-grid true contour')
    lower = np.minimum(np.floor(index).astype(int), 63)
    return grid[lower] + (index - lower) * (grid[lower + 1] - grid[lower])


def main(root, out):
    started = time.monotonic()
    if out.exists():
        raise FileExistsError('Preserve completed training-statistics receipts')
    validation_path = root / 'target-validation.json'
    validation = json.loads(validation_path.read_text())
    targets = root / 'train-contours'
    index = json.loads((targets / 'index.json').read_text())
    if not index['training_only'] or not validation['complete'] or not validation['all_usable_targets_checked']:
        raise ValueError('Validated training-only target pack required')
    with (targets / 'index.json').open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != validation['file_sha256']['index.json']:
            raise ValueError('Training target index changed')
    points = np.load(targets / 'points.npy', mmap_mode='r')
    li = np.load(targets / 'target_li.npy', mmap_mode='r')
    valid = np.load(targets / 'valid.npy', mmap_mode='r')
    r, z = (np.load(targets / f'{key}.npy').astype(float) for key in ('r', 'z'))
    if points.shape != (index['frames'], 512, 2) or valid.shape != (index['frames'],):
        raise ValueError('Training target shapes differ')
    moments = {k: {'count': 0, 'mean': 0., 'M2': 0., 'min': float('inf'), 'max': -float('inf')}
               for k in ('li', 'kappa')}
    for a in range(0, index['frames'], 4096):
        sl = slice(a, a + 4096)
        picks = valid[sl]
        if not picks.any():
            continue
        p = np.asarray(points[sl])[picks]
        width = coordinate(p[:, :, 0].max(1), r) - coordinate(p[:, :, 0].min(1), r)
        height = coordinate(p[:, :, 1].max(1), z) - coordinate(p[:, :, 1].min(1), z)
        if np.any(width <= 0):
            raise FloatingPointError('Undefined true contour width')
        update(moments['li'], np.asarray(li[sl])[picks].astype(float))
        update(moments['kappa'], height / width)
    for value in moments.values():
        if value['count'] != validation['usable_frames']:
            raise ValueError('Training target usable count mismatch')
        value['population_variance'] = value['M2'] / value['count']
        if value['population_variance'] <= 0:
            raise FloatingPointError('Undefined target variance')
    ratio = moments['li']['population_variance'] / moments['kappa']['population_variance']
    with validation_path.open('rb') as handle:
        validation_sha = file_digest(handle, 'sha256').hexdigest()
    result = {'complete': True, 'training_only': True, 'target_validation_sha256': validation_sha,
              'manifest_sha256': validation['manifest_sha256'], 'moments': moments,
              'equal_R2_weight_ratio_kappa_to_li': ratio,
              'current_loss_weight_ratio_kappa_to_li': 0.04419808835 / 0.05,
              'elapsed_CPU_seconds': time.monotonic() - started, 'GPU_used': False,
              'candidate_created': False,
              'limitations': 'Training-target population moments only. Kappa reconstructs physical extrema from float32 packed contour coordinates. An R2 error-weight ratio is algebra, not a gradient-balance choice, improvement, official-scorer equivalence or justification for an allocation.'}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
