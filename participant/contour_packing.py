"""Stream hash-verified training geometry into the shared target batch format."""

from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


def pack_targets(raw, packed, split_path, out, workers, width, extractor, description):
    if packed.name != 'train':
        raise ValueError('Contour targets require the training split')
    index = json.loads((packed / 'index.json').read_text())
    split = json.loads(split_path.read_text())
    entries = split['train']
    names = [Path(e['path']).stem for e in entries]
    if index['names'] != names:
        raise ValueError('Training names/order differ from the pinned manifest')
    out.mkdir(parents=True, exist_ok=False)
    n = index['frames']
    specs = {'points': (np.float32, (n, width, 2)), 'weights': (np.float32, (n, width)),
             'mask_bits': (np.uint8, (n, 529)), 'target_li': (np.float32, (n,)),
             'valid': (np.bool_, (n,))}
    arrays = {k: np.lib.format.open_memmap(out / f'{k}.npy', mode='w+', dtype=t, shape=s)
              for k, (t, s) in specs.items()}
    jobs = [(raw / f'{name}.parquet', e['sha256']) for name, e in zip(names, entries, strict=True)]
    r0 = z0 = None
    with ProcessPoolExecutor(workers) as pool:
        for i, (points, weights, masks, li, r, z, valid) in enumerate(pool.map(extractor, jobs, chunksize=2)):
            a, b = index['offsets'][i:i + 2]
            if len(points) != b - a or points.shape[1] > width or valid.shape != (b - a,):
                raise ValueError(f'Target frame/vertex order mismatch: {names[i]}')
            if not np.isfinite(li[valid]).all() or not np.isfinite(points[valid]).all():
                raise ValueError(f'Nonfinite usable targets: {names[i]}')
            if r0 is not None and (not np.array_equal(r, r0) or not np.array_equal(z, z0)):
                raise ValueError('Training physical grids differ')
            r0, z0 = r, z
            arrays['points'][a:b, :points.shape[1]] = points
            arrays['weights'][a:b, :points.shape[1]] = weights
            arrays['mask_bits'][a:b] = masks
            arrays['target_li'][a:b] = li
            arrays['valid'][a:b] = valid
            if i % 250 == 0:
                print(f'contour targets {i}/{len(names)} shots', flush=True)
    for array in arrays.values():
        array.flush()
    usable = int(arrays['valid'].sum())
    if usable == 0:
        raise ValueError('No usable training contour targets')
    np.save(out / 'r.npy', r0); np.save(out / 'z.npy', z0)
    with split_path.open('rb') as handle:
        manifest_sha = file_digest(handle, 'sha256').hexdigest()
    receipt = index | {'training_only': True, 'revision': split['revision'],
                       'manifest_sha256': manifest_sha, 'boundary_storage_width': width,
                       'interior_mask_pixels': 4225, 'target': description,
                       'usable_frames': usable, 'undefined_reference_frames': n - usable,
                       'source_hashes_verified': True, 'complete': True}
    (out / 'index.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'packed{n} frames, usable{usable}, vertices{width}', flush=True)
