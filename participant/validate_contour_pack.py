"""Validate every usable scorer-contour target and hash the completed pack."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from .contour_training import ScoredContourBatch


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def validate(targets, packed, split, out):
    if out.exists():
        raise FileExistsError('Preserve an existing target validation receipt')
    index = json.loads((packed / 'index.json').read_text())
    batch = ScoredContourBatch(targets, index)
    if batch.manifest_sha != digest(split):
        raise ValueError('Training target manifest hash mismatch')
    manifest = json.loads(split.read_text())
    if index['names'] != [Path(e['path']).stem for e in manifest['train']]:
        raise ValueError('Training shot order differs from manifest')
    if len(batch.r) != 65 or len(batch.z) != 65 or not np.all(np.diff(batch.r) > 0) or not np.all(np.diff(batch.z) > 0):
        raise ValueError('Invalid physical grids')
    checked = 0
    for start in range(0, index['frames'], 8192):
        sl = slice(start, start + 8192)
        valid = batch.valid[sl]
        p, w, masks, li = (np.asarray(batch.arrays[k][sl])[valid] for k in
                           ('points', 'weights', 'mask_bits', 'target_li'))
        if not valid.any():
            continue
        if not all(np.isfinite(v).all() for v in (p, w, li)) or np.any(abs(p) > 1):
            raise ValueError(f'Nonfinite/out-of-grid usable target near frame{start}')
        if np.any(w < 0) or not np.allclose(w.sum(1), 1, rtol=0, atol=2e-6) or np.any(li < 0):
            raise ValueError(f'Invalid quadrature near frame{start}')
        pixels = np.unpackbits(masks, axis=1, count=4225).sum(1)
        if np.any(pixels < 4):
            raise ValueError(f'Invalid training interior near frame{start}')
        checked += int(valid.sum())
    if checked != len(batch.valid_indices):
        raise ValueError('Usable target count mismatch')
    hashes = {p.name: digest(p) for p in sorted(targets.glob('*.npy'))}
    hashes['index.json'] = digest(targets / 'index.json')
    result = {'complete': True, 'training_only': True, 'frames': index['frames'],
              'usable_frames': checked, 'undefined_reference_frames': index['frames'] - checked,
              'manifest_sha256': batch.manifest_sha, 'target': batch.description,
              'all_usable_targets_checked': True, 'file_sha256': hashes}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'file_sha256'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for key in ('targets', 'packed', 'split', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    validate(a.targets, a.packed, a.split, a.out)


if __name__ == '__main__':
    main()
