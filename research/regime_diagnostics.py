"""Locate full-development geometry errors by physical input regime at float32 precision."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle

import numpy as np

from participant import fast_score
from research.diagnostics import subsets


def measure(job):
    predicted, targets = job
    scorer = fast_score.official
    r, z, coarse, mask = fast_score._geometry()
    values = np.full(targets.shape, np.nan)
    for i, psi in enumerate(predicted):
        contour, _ = scorer.extract_lcfs_with_sign(
            psi.astype(float), r, z, scorer.MACHINE, coarse, mask,
            n_iter=scorer.N_ITER, axis_sign=scorer.AXIS_SIGN[scorer.MACHINE])
        derived = scorer.derive_frame(psi.astype(float), r, z, scorer.MACHINE,
                                     coarse, mask, contour=contour,
                                     axis_sign=scorer.AXIS_SIGN[scorer.MACHINE])
        values[i] = [derived[key] for key in scorer.CONS_SCALARS]
    return values


def main(args):
    if args.packed.name != 'development':
        raise ValueError('Use only the existing development split')
    index = json.loads((args.packed / 'index.json').read_text())
    report = json.loads((args.cache / 'report.json').read_text())
    if report['upload_dtype'] != 'float32' or report['score']['psi_sign'] != 1:
        raise ValueError('Expected the verified float32 ensemble in the reference sign')
    psi = np.load(args.cache / 'psi-pred.npy', mmap_mode='r')
    x = np.load(args.packed / 'x.npy', mmap_mode='r')
    refs = pickle.loads(args.refs.read_bytes())
    if len(refs) != len(index['names']):
        raise ValueError('Reference count differs from packed shot count')
    jobs, rows = [], []
    for i, (a, b) in enumerate(zip(index['offsets'][:-1], index['offsets'][1:], strict=True)):
        local = np.arange(b - a)
        rows.append(a + local)
        jobs.append((np.asarray(psi[a + local]), refs[i][1][local]))
    with ProcessPoolExecutor(12) as pool:
        predicted = np.concatenate(list(pool.map(measure, jobs)))
    truth = np.concatenate([target for _, target in jobs])
    full_truth = np.concatenate([ref[1] for ref in refs])
    means, stds = np.nanmean(full_truth, 0), np.nanstd(full_truth, 0)
    valid = np.isfinite(truth)
    failed = valid & ~np.isfinite(predicted)
    error = np.where(valid, (np.where(np.isfinite(predicted), predicted, means) - truth) / stds, 0)
    squared = error ** 2
    for j, key in enumerate(fast_score.official.CONS_SCALARS):
        r2 = 1 - squared[valid[:, j], j].mean()
        if abs(r2 - report['score']['r2_cons_each'][key]) > 1e-6:
            raise AssertionError(f'Regime residuals differ from published scoring: {key}')
    groups = subsets(x[np.concatenate(rows)])
    groups['all'] = np.ones(len(truth), dtype=bool)
    findings = {}
    for name, mask in groups.items():
        terms = {}
        for j, key in enumerate(fast_score.official.CONS_SCALARS):
            defined = mask & valid[:, j]
            terms[key] = {'valid_frames': int(defined.sum()),
                          'failure_frames': int(failed[mask, j].sum()),
                          'squared_error_share': float(squared[mask, j].sum() / squared[:, j].sum()),
                          'normalized_rmse': float(np.sqrt(squared[defined, j].mean())) if defined.any() else None,
                          'normalized_bias': float(error[defined, j].mean()) if defined.any() else None}
        findings[name] = {'frames': int(mask.sum()), 'frame_share': float(mask.mean()), 'terms': terms}
    result = {'frames': len(truth), 'stride_within_each_shot': 1, 'upload_dtype': 'float32',
              'normalization': 'full development reference standard deviation',
              'diagnostic_only': True, 'locked_shots_read': False, 'groups': findings}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--refs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
