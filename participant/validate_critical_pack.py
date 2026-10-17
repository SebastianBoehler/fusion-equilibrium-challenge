"""Validate every usable critical stencil against the frozen packed truth."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from .critical_training import CriticalBatch


def main():
    parser = argparse.ArgumentParser()
    for key in ('critical', 'contours', 'packed', 'split', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve completed critical validation')
    index = json.loads((args.packed / 'index.json').read_text())
    with args.split.open('rb') as handle:
        manifest_sha = file_digest(handle, 'sha256').hexdigest()
    batch = CriticalBatch(args.contours, args.critical, index)
    if batch.manifest_sha != manifest_sha:
        raise ValueError('Critical manifest differs from pinned split')
    arrays = {k: np.load(args.critical / f'{k}.npy', mmap_mode='r')
              for k in ('indices', 'weights', 'valid', 'kind', 'level', 'tolerance')}
    for key, dtype in (('kind', np.uint8), ('level', np.float32), ('tolerance', np.float32)):
        if arrays[key].shape != (index['frames'],) or arrays[key].dtype != dtype:
            raise ValueError('Critical event metadata shape/dtype differs')
    truth = np.load(args.packed / 'psi.npy', mmap_mode='r')
    if truth.shape != (index['frames'], 4225) or truth.dtype != np.float32:
        raise ValueError('Frozen packed truth shape/dtype differs')
    maximum = 0.
    for start in range(0, len(batch.valid_indices), 32768):
        picks = batch.valid_indices[start:start + 32768]
        indices, weights = arrays['indices'][picks], arrays['weights'][picks]
        if (np.any(indices < 0) or np.any(indices > 4224) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.all(weights.sum(1) == 1)):
            raise ValueError('Undefined usable critical stencil')
        levels, tolerance = arrays['level'][picks], arrays['tolerance'][picks]
        if (not np.isfinite(levels).all() or not np.isfinite(tolerance).all()
                or np.any(tolerance <= 0) or not np.isin(arrays['kind'][picks], [1, 2]).all()):
            raise ValueError('Undefined critical event type/level/tolerance')
        fields = np.asarray(truth[picks])
        live = (np.take_along_axis(fields, indices.astype(np.int64), 1) * weights).sum(1)
        residual = abs(live.astype(float) - levels.astype(float))
        rounding = 4 * np.finfo(np.float32).eps * np.maximum(abs(levels), abs(fields).max(1))
        if np.any(residual > tolerance + rounding):
            raise ValueError('Critical level fails actual frozen truth interpolation')
        maximum = max(maximum, float(residual.max()))
    valid = arrays['valid']
    if (np.any(arrays['kind'][~valid] != 0) or np.any(arrays['indices'][~valid] != -1)
            or not np.isnan(arrays['weights'][~valid]).all()
            or not np.isnan(arrays['level'][~valid]).all()
            or not np.isnan(arrays['tolerance'][~valid]).all()):
        raise ValueError('Excluded critical targets lack explicit invalid sentinels')
    excluded = np.flatnonzero(~valid)
    for start in range(0, len(excluded), 32768):
        try:
            batch.critical_get(excluded[start:start + 32768], 'cpu')
        except ValueError as exc:
            if str(exc) != 'Unsupported critical target requested':
                raise
        else:
            raise AssertionError('Excluded critical target request was accepted')
    pack = json.loads((args.critical / 'index.json').read_text())
    if (pack['original_reference_excluded_frames'] != int((~batch.valid).sum())
            or pack['unsupported_critical_events'] != int((batch.valid & ~valid).sum())
            or sum(pack['unsupported_reasons'].values()) != pack['unsupported_critical_events']):
        raise ValueError('Critical exclusion accounting differs')
    hashes = {}
    for path in sorted(args.critical.iterdir()):
        with path.open('rb') as handle:
            hashes[path.name] = file_digest(handle, 'sha256').hexdigest()
    receipt = {'complete': True, 'training_only': True, 'frames': index['frames'],
               'usable_frames': len(batch.valid_indices), 'all_usable_shapes_finiteness_weights_indices_levels_passed': True,
               'excluded_targets_rejected': True, 'excluded_request_count_checked': len(excluded),
               'original_reference_excluded_frames': pack['original_reference_excluded_frames'],
               'unsupported_critical_events': pack['unsupported_critical_events'],
               'unsupported_reasons': pack['unsupported_reasons'], 'maximum_float32_level_difference': maximum,
               'target_manifest_sha256': manifest_sha, 'file_sha256': hashes}
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print('critical target validation passed', receipt['usable_frames'], flush=True)


if __name__ == '__main__':
    main()
