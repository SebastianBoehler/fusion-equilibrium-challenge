"""Extract rich features and targets per shot, then pack each split into memory maps."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .data import array_value
from .rich import INPUT_COLUMNS, VERSION, rich_features

TARGET_COLUMNS = ('efit_psirz', 'efit_q95', 'efit_beta_n', 'efit_li', 'efit_r_axis', 'efit_z_axis')
SCALAR_NAMES = ('q95', 'betaN', 'li', 'r_axis', 'z_axis')


def extract(raw: Path, out: Path) -> int:
    target = out / f'{raw.stem}.npz'
    if target.exists():
        with np.load(target) as values:
            return len(values['x'])
    table = pq.read_table(raw, columns=list(dict.fromkeys(INPUT_COLUMNS + TARGET_COLUMNS)))
    row = {name: array_value(table[name][0]) for name in table.column_names}
    inputs = {key: row[key] for key in INPUT_COLUMNS}
    x = rich_features(inputs)
    psi = np.asarray(row['efit_psirz'], dtype=np.float32).reshape(len(x), -1)
    if psi.shape[1] != 4225 or not np.isfinite(psi).all():
        raise ValueError(f'Invalid flux target: {raw.name}')
    scalars = np.column_stack([np.asarray(row[k], dtype=np.float32) for k in TARGET_COLUMNS[1:]])
    temporary = out / f'{raw.stem}.partial.npz'
    np.savez(temporary, x=x, psi=psi, scalars=scalars,
             times=np.asarray(row['efit_times'], dtype=np.float64))
    temporary.replace(target)
    return len(x)


def _extract(args):
    raw, out = args
    try:
        return raw.stem, extract(raw, out), None
    except Exception as exc:  # report every failing shot instead of stopping the pool
        return raw.stem, 0, repr(exc)


def pack(names: list[str], shots: Path, out: Path) -> None:
    counts = []
    for name in names:
        with np.load(shots / f'{name}.npz') as values:
            counts.append(len(values['x']))
    total = int(sum(counts))
    with np.load(shots / f'{names[0]}.npz') as values:
        width = values['x'].shape[1]
    out.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(out / 'x.npy', mode='w+', dtype=np.float32, shape=(total, width))
    psi = np.lib.format.open_memmap(out / 'psi.npy', mode='w+', dtype=np.float32, shape=(total, 4225))
    scalars = np.lib.format.open_memmap(out / 'scalars.npy', mode='w+', dtype=np.float32,
                                        shape=(total, len(SCALAR_NAMES)))
    start = 0
    for name, count in zip(names, counts):
        with np.load(shots / f'{name}.npz') as values:
            x[start:start + count] = values['x']
            psi[start:start + count] = values['psi']
            scalars[start:start + count] = values['scalars']
        start += count
    for array in (x, psi, scalars):
        array.flush()
    offsets = np.concatenate([[0], np.cumsum(counts)]).tolist()
    (out / 'index.json').write_text(json.dumps({'version': VERSION, 'names': names, 'offsets': offsets,
                                                'scalars': SCALAR_NAMES, 'frames': total}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    groups = {'train': split['train'], 'development': split['development'],
              'regression': sum(split['regression'].values(), [])}
    shots = args.out / 'shots'
    shots.mkdir(parents=True, exist_ok=True)
    jobs = [(args.raw / Path(e['path']).name, shots) for g in groups.values() for e in g]
    failures = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (name, frames, error) in enumerate(pool.map(_extract, jobs, chunksize=4)):
            if error:
                failures[name] = error
                print('FAILED', name, error, flush=True)
            if i % 250 == 0:
                print(f'extracted {i}/{len(jobs)}', flush=True)
    (args.out / 'failures.json').write_text(json.dumps(failures, indent=1))
    for group, entries in groups.items():
        names = [Path(e['path']).stem for e in entries if Path(e['path']).stem not in failures]
        pack(names, shots, args.out / group)
        print(group, len(names), 'shots packed', flush=True)


if __name__ == '__main__':
    main()
