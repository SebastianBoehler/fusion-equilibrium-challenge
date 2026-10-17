"""Pack actual seven-component truth targets from pinned DIII-D TRAIN shots."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from .artifacts import digest, write_json
from .data import array_value
from .fast_score import _geometry, official
from .finite_metrics import measure_derived
from .finite_target_batch import population_statistics
from .reference_cache import scorer_identity


def extract(job):
    path, expected, eligible = job
    if digest(path) != expected:
        raise ValueError(f'Pinned TRAIN source hash changed: {path.name}')
    columns = ['source', 'efit_psirz', 'efit_grid_R', 'efit_grid_Z']
    table = pq.read_table(path, columns=columns)
    if len(table) != 1 or array_value(table['source'][0]) != 'DIII-D':
        raise ValueError('Finite loss targets require one DIII-D TRAIN shot')
    fields = np.asarray(array_value(table['efit_psirz'][0]), np.float32)
    if fields.shape != (len(eligible), 65, 65) or not np.isfinite(fields[eligible]).all():
        raise ValueError('Usable TRAIN truth shape/finiteness changed')
    r, z, _, _ = _geometry()
    for key, expected_grid in (('efit_grid_R', r), ('efit_grid_Z', z)):
        if not np.allclose(array_value(table[key][0]), expected_grid, rtol=0, atol=5e-7):
            raise ValueError('TRAIN grid differs from actual scorer geometry')
    values = np.full((len(fields), 7), np.nan, np.float64)
    if eligible.any():
        values[eligible] = measure_derived(fields[eligible])
    return values


def pack(raw, packed, contours, split_path, reference_validation, out, workers):
    if packed.name != 'train':
        raise ValueError('Finite loss targets require the TRAIN split')
    index = json.loads((packed / 'index.json').read_text())
    split = json.loads(split_path.read_text())
    original = json.loads((contours / 'index.json').read_text())
    validation = json.loads(reference_validation.read_text())
    manifest_sha = digest(split_path)
    names = [Path(e['path']).stem for e in split['train']]
    offsets = np.asarray(index['offsets'])
    n = index['frames']
    if (not validation['complete'] or not validation['training_only'] or
            not original['complete'] or not original['training_only'] or
            original['manifest_sha256'] != manifest_sha or validation['manifest_sha256'] != manifest_sha or
            index['names'] != names or original['names'] != names or
            original['offsets'] != index['offsets'] or len(offsets) != len(names) + 1 or
            offsets.dtype.kind not in 'iu' or original['frames'] != n or validation['frames'] != n or
            offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0) or
            original['revision'] != split['revision']):
        raise ValueError('Frozen TRAIN manifest/order/reference validation differs')
    for name in ('index.json', 'valid.npy', 'target_li.npy'):
        if digest(contours / name) != validation['file_sha256'][name]:
            raise ValueError(f'Original validated TRAIN reference changed: {name}')
    valid = np.load(contours / 'valid.npy', mmap_mode='r', allow_pickle=False)
    li = np.load(contours / 'target_li.npy', mmap_mode='r', allow_pickle=False)
    if (valid.shape != (n,) or valid.dtype != np.bool_ or li.shape != (n,) or li.dtype != np.float32 or
            int(valid.sum()) != original['usable_frames'] or
            int((~valid).sum()) != original['undefined_reference_frames']):
        raise ValueError('Original TRAIN eligibility count/schema changed')
    out.mkdir(parents=True, exist_ok=False)
    values = np.lib.format.open_memmap(out / 'values.npy', mode='w+', dtype=np.float64, shape=(n, 7))
    values[:] = np.nan
    jobs = [(raw / f'{name}.parquet', entry['sha256'], valid[a:b].copy())
            for name, entry, a, b in zip(names, split['train'], offsets[:-1], offsets[1:], strict=True)]
    with ProcessPoolExecutor(workers) as pool:
        for i, rows in enumerate(pool.map(extract, jobs, chunksize=2)):
            a, b = offsets[i:i + 2]
            if rows.shape != (b - a, 7) or not np.isfinite(rows[valid[a:b]]).all():
                raise ValueError('Actual seven-component TRAIN target shape/order/finiteness changed')
            values[a:b] = rows
            if i % 250 == 0:
                print(f'actual seven TRAIN targets {i}/{len(names)} shots', flush=True)
    values.flush()
    shutil.copyfile(contours / 'valid.npy', out / 'valid.npy')
    maximum_li = float(np.max(abs(values[valid, official.LI_IDX] - li[valid])))
    if maximum_li > 2e-5 or not np.isnan(values[~valid]).all():
        raise ValueError('Actual TRAIN li or original exclusion sentinels differ')
    receipt = index | {'complete': True, 'training_only': True,
        'manifest_sha256': manifest_sha, 'components': list(official.CONS_SCALARS),
        'usable_frames': int(valid.sum()), 'original_reference_excluded_frames': int((~valid).sum()),
        'additional_exclusions': 0, 'source_hashes_verified': True,
        'scorer_sha256': scorer_identity(), 'original_validation_sha256': digest(reference_validation)}
    write_json(out / 'index.json', receipt)
    stats = population_statistics(values, valid)
    result = {'complete': True, 'training_only': True, 'manifest_sha256': manifest_sha,
        'frames': n, 'usable_frames': stats['frames'], 'original_reference_excluded_frames': int((~valid).sum()),
        'additional_exclusions': 0, 'maximum_actual_li_quadrature_difference': maximum_li,
        'statistics': stats, 'file_sha256': {name: digest(out / name)
                                           for name in ('values.npy', 'valid.npy', 'index.json')}}
    write_json(out / 'validation.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ('raw', 'packed', 'contours', 'split', 'reference-validation', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.raw, args.packed, args.contours, args.split,
                          args.reference_validation, args.out, args.workers), indent=2))


if __name__ == '__main__':
    main()
