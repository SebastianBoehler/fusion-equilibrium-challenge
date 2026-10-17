"""Process-parallel use of the unmodified organizer scorer on held-out packed shots.

Mirrors participant.score.composite exactly (same reference build, global means,
pooled sign choice and per-shot accumulation); only the per-shot loops run in a pool.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .score import official

_GEOMETRY = None


def _geometry():
    global _GEOMETRY
    if _GEOMETRY is None:
        mask = np.load(official.HERE / 'fusion_scoring/masks/d3d_envelope.npz')
        coarse = mask['mask_coarse'].astype(bool)
        _GEOMETRY = (mask['grid_R'], mask['grid_Z'], coarse, coarse.astype(float))
    return _GEOMETRY


def _reference(psi):
    r, z, coarse, mask_f = _geometry()
    return official.build_reference(np.asarray(psi, dtype=float), r, z, coarse, mask_f)


def _score(args):
    shot, ref, prediction, sign, means = args
    r, z, coarse, mask_f = _geometry()
    return official.score_shot(shot, ref, prediction, r, z, coarse, mask_f, sign, means)


def load_truth(packed: Path):
    index = json.loads((packed / 'index.json').read_text())
    psi = np.load(packed / 'psi.npy', mmap_mode='r')
    scalars = np.load(packed / 'scalars.npy', mmap_mode='r')
    offsets = index['offsets']
    shots = []
    for i in range(len(index['names'])):
        a, b = offsets[i], offsets[i + 1]
        shots.append({'psi': np.asarray(psi[a:b], dtype=np.float64).reshape(-1, 65, 65),
                      'q95': np.asarray(scalars[a:b, 0], dtype=np.float64),
                      'betaN': np.asarray(scalars[a:b, 1], dtype=np.float64)})
    return index, shots


def references(shots, cache: Path | None, workers: int):
    if cache is not None and cache.exists():
        import pickle
        return pickle.loads(cache.read_bytes())
    with ProcessPoolExecutor(workers) as pool:
        refs = list(pool.map(_reference, [s['psi'] for s in shots], chunksize=2))
    if cache is not None:
        import os
        import pickle
        temporary = cache.with_suffix(f'.{os.getpid()}.partial')
        temporary.write_bytes(pickle.dumps(refs))
        temporary.replace(cache)
    return refs


def composite(shots, refs, predictions, workers: int) -> dict:
    psi = np.concatenate([s['psi'].ravel() for s in shots])
    scal = np.column_stack([np.concatenate([s[k] for s in shots]) for k in ('q95', 'betaN')])
    cons = np.concatenate([ref[1] for ref in refs])
    finite_scal, finite_cons = np.isfinite(scal), np.isfinite(cons)
    stats = {'psi_sum': psi.sum(), 'psi_sumsq': np.square(psi).sum(), 'psi_n': psi.size,
             'scal_sum': np.nansum(scal, axis=0), 'scal_sumsq': np.nansum(np.square(scal), axis=0),
             'scal_n': finite_scal.sum(axis=0), 'cons_sum': np.nansum(cons, axis=0),
             'cons_sumsq': np.nansum(np.square(cons), axis=0), 'cons_n': finite_cons.sum(axis=0)}
    means = (psi.mean(),
             np.divide(stats['scal_sum'], stats['scal_n'], out=np.zeros(2), where=stats['scal_n'] > 0),
             np.divide(stats['cons_sum'], stats['cons_n'], out=np.zeros(7), where=stats['cons_n'] > 0))
    totals = {sign: 0.0 for sign in official.PSI_SIGNS}
    for shot, prediction in zip(shots, predictions, strict=True):
        residuals, _, _ = official.psi_residuals(shot['psi'], prediction['psirz'], means[0])
        for sign in totals:
            totals[sign] += residuals[sign]
    sign = min(totals, key=totals.get)
    acc = official.Accum()
    acc.psi_sign = sign
    jobs = [(s, r, p, sign, means) for s, r, p in zip(shots, refs, predictions, strict=True)]
    with ProcessPoolExecutor(workers) as pool:
        for part in pool.map(_score, jobs, chunksize=2):
            acc.add(part)
    return official.finalize_machine(acc, stats)


def load_predictions(path: Path, count: int):
    with np.load(path, allow_pickle=False) as values:
        return [{'psirz': values[f'shot_{i:04d}_psirz'].astype(np.float64),
                 'q95': values[f'shot_{i:04d}_q95'].astype(np.float64),
                 'betaN': values[f'shot_{i:04d}_betaN'].astype(np.float64)} for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--reference-cache', type=Path)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    index, shots = load_truth(args.packed)
    refs = references(shots, args.reference_cache, args.workers)
    for path in args.predictions:
        scores = composite(shots, refs, load_predictions(path, len(shots)), args.workers)
        out = path.with_suffix('.scores.json')
        out.write_text(json.dumps(scores, indent=1, default=float))
        keys = ('S', 'r2_psi', 'r2_qb', 'dlcfs', 'consistency', 'cons_fail_frac', 'lcfs_fail_frac')
        print(path.name, {k: scores[k] for k in keys}, flush=True)
        print('  scalars', scores['r2_each'], '\n  cons', scores['r2_cons_each'], flush=True)


if __name__ == '__main__':
    main()
