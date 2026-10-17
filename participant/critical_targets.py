"""Pack active critical events from pinned DIII-D training fields only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .critical_level import stencil
from .critical_training import DESCRIPTION
from .data import array_value


def extract(job):
    path, sha, eligible = job
    torch.set_num_threads(1)
    with path.open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != sha:
            raise ValueError('Pinned training shot hash changed')
    table = pq.read_table(path, columns=['source', 'efit_psirz'])
    if array_value(table['source'][0]) != 'DIII-D':
        raise ValueError('Critical targets allow DIII-D training only')
    fields = np.asarray(array_value(table['efit_psirz'][0]), np.float32)
    if fields.shape != (len(eligible), 65, 65) or not np.isfinite(fields[eligible]).all():
        raise ValueError('Usable training field shape or finiteness changed')
    n = len(fields)
    values = {'indices': np.full((n, 2), -1, np.int16), 'weights': np.full((n, 2), np.nan, np.float32),
              'valid': np.zeros(n, np.bool_), 'kind': np.zeros(n, np.uint8),
              'level': np.full(n, np.nan, np.float32), 'tolerance': np.full(n, np.nan, np.float32)}
    errors = {}
    for i in np.flatnonzero(eligible):
        try:
            event = stencil(fields[i])
        except ValueError as exc:
            message = str(exc)
            errors[message] = errors.get(message, 0) + 1
            continue
        values['indices'][i], values['weights'][i] = event['indices'], event['weights']
        values['level'][i], values['tolerance'][i] = event['level'], event['bisection_tolerance']
        values['kind'][i], values['valid'][i] = event['kind'], True
    return values, errors


def main():
    parser = argparse.ArgumentParser()
    for key in ('raw', 'packed', 'contours', 'split', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.packed.name != 'train':
        raise ValueError('Critical targets require the training split')
    index = json.loads((args.packed / 'index.json').read_text())
    split = json.loads(args.split.read_text())
    reference = json.loads((args.contours / 'index.json').read_text())
    names = [Path(e['path']).stem for e in split['train']]
    with args.split.open('rb') as handle:
        manifest_sha = file_digest(handle, 'sha256').hexdigest()
    if (index['names'] != names or reference['names'] != names or reference['offsets'] != index['offsets']
            or not reference['complete'] or reference['manifest_sha256'] != manifest_sha):
        raise ValueError('Frozen training and contour order/manifest differ')
    eligible = np.load(args.contours / 'valid.npy', mmap_mode='r')
    n = index['frames']
    if eligible.shape != (n,) or eligible.dtype != np.bool_:
        raise ValueError('Invalid reference validity mask')
    args.out.mkdir(parents=True, exist_ok=False)
    specs = {'indices': (np.int16, (n, 2)), 'weights': (np.float32, (n, 2)),
             'valid': (np.bool_, (n,)), 'kind': (np.uint8, (n,)),
             'level': (np.float32, (n,)), 'tolerance': (np.float32, (n,))}
    arrays = {key: np.lib.format.open_memmap(args.out / f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
              for key, (dtype, shape) in specs.items()}
    jobs = [(args.raw / f'{name}.parquet', entry['sha256'], eligible[a:b].copy())
            for name, entry, a, b in zip(names, split['train'], index['offsets'][:-1], index['offsets'][1:], strict=True)]
    errors = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (values, counts) in enumerate(pool.map(extract, jobs, chunksize=2)):
            a, b = index['offsets'][i:i + 2]
            for key, value in values.items():
                if len(value) != b - a:
                    raise ValueError('Critical target frame order changed')
                arrays[key][a:b] = value
            for message, count in counts.items():
                errors[message] = errors.get(message, 0) + count
            if i % 250 == 0:
                print(f'critical targets{i}/{len(names)} shots', flush=True)
    for array in arrays.values():
        array.flush()
    usable = int(arrays['valid'].sum())
    if not usable:
        raise ValueError('No supported training critical events')
    receipt = index | {'complete': True, 'training_only': True, 'target': DESCRIPTION,
                       'manifest_sha256': manifest_sha, 'revision': split['revision'],
                       'usable_frames': usable, 'original_reference_usable_frames': int(eligible.sum()),
                       'original_reference_excluded_frames': int((~eligible).sum()),
                       'unsupported_critical_events': int(eligible.sum()) - usable,
                       'unsupported_reasons': errors, 'source_hashes_verified': True}
    (args.out / 'index.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'critical pack complete{n} frames,{usable} usable', flush=True)


if __name__ == '__main__':
    main()
