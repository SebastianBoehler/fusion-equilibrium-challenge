"""Explain a frozen ensemble's errors on an existing development split."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np

from participant import fast_score
from participant.nn_submit import Predictor


def subsets(x):
    phase = np.where(x[:, 85] < 0.6, 0, np.where(x[:, 86] < 0.4, 2, 1))
    return {
        'ramp_up': phase == 0, 'flat_top': phase == 1, 'ramp_down': phase == 2,
        'low_current': np.abs(x[:, 20]) < 600, 'negative_current': x[:, 20] < 0,
        'core_coverage_under_half': x[:, 141:159].mean(1) < 0.5,
        'edge_coverage_under_half': x[:, 206:219].mean(1) < 0.5,
    }


def main(args):
    if args.packed.name != 'development':
        raise ValueError('Diagnostics must use the existing development split')
    args.out.mkdir(parents=True, exist_ok=False)
    index, shots = fast_score.load_truth(args.packed)
    x = np.load(args.packed / 'x.npy', mmap_mode='r')
    refs = fast_score.references(shots, args.refs, args.workers)
    model = Predictor(args.checkpoints)
    count = len(x)
    stored = np.lib.format.open_memmap(args.out / 'psi-pred.npy', mode='w+',
                                     dtype=np.float32, shape=(count, 65, 65))
    scalars = np.empty((count, 2), dtype=np.float32)
    rows, predictions = [], []
    frame_psi, frame_scalars = np.empty(count), np.empty((count, 2))
    spatial = np.zeros((65, 65), dtype=np.float64)
    oracle_sse, quantization_sse = 0.0, 0.0
    started = time.perf_counter()
    _, params, _ = model.members[0]
    basis, mean = params['basis'].numpy(), params['psi_mean'].numpy()
    for i, shot in enumerate(shots):
        a, b = index['offsets'][i:i + 2]
        t0 = time.perf_counter()
        pred = model.predict_inputs(np.asarray(x[a:b]))
        elapsed = time.perf_counter() - t0
        stored[a:b] = pred['psirz']
        scalars[a:b] = np.column_stack((pred['q95'], pred['betaN']))
        uploaded = {k: v.astype(np.float32).astype(float) for k, v in pred.items()}
        error = uploaded['psirz'] - shot['psi']
        frame_psi[a:b] = np.square(error).sum((1, 2))
        target_scalars = np.column_stack((shot['q95'], shot['betaN']))
        frame_scalars[a:b] = np.square(scalars[a:b] - target_scalars)
        spatial += np.square(error).sum(0)
        truth = shot['psi'].reshape(b - a, -1).astype(np.float32)
        oracle = mean + ((truth - mean) @ basis) @ basis.T
        oracle_sse += float(np.square(oracle.astype(float) - truth).sum())
        quantization_sse += float(np.square(pred['psirz'].astype(np.float16).astype(float)
                                            - pred['psirz']).sum())
        rows.append({'name': index['names'][i], 'frames': b - a, 'inference_seconds': elapsed,
                     'flux_sse': float(frame_psi[a:b].sum()),
                     'max_standardized_input': float(np.abs((x[a:b] - params['x_mean'].numpy())
                                                            / params['x_std'].numpy()).max()),
                     'clipped_value_fraction': float((np.abs((x[a:b] - params['x_mean'].numpy())
                                                             / params['x_std'].numpy()) > 10).mean())})
        predictions.append(uploaded)
        if (i + 1) % 64 == 0:
            print(f'Predicted {i + 1}/{len(shots)} shots', flush=True)
    stored.flush()
    np.save(args.out / 'scalar-pred.npy', scalars)
    inference_time = time.perf_counter() - started
    cons = np.concatenate([r[1] for r in refs])
    psi_sum = sum(float(s['psi'].sum()) for s in shots)
    psi_sumsq = sum(float(np.square(s['psi']).sum()) for s in shots)
    pixels = count * 4225
    scalar_truth = np.concatenate([np.column_stack((s['q95'], s['betaN'])) for s in shots])
    stats = {'psi_sum': psi_sum, 'psi_sumsq': psi_sumsq, 'psi_n': pixels,
             'scal_sum': np.nansum(scalar_truth, 0), 'scal_sumsq': np.nansum(scalar_truth ** 2, 0),
             'scal_n': np.isfinite(scalar_truth).sum(0), 'cons_sum': np.nansum(cons, 0),
             'cons_sumsq': np.nansum(cons ** 2, 0), 'cons_n': np.isfinite(cons).sum(0)}
    means = (psi_sum / pixels, np.nanmean(scalar_truth, 0), np.nanmean(cons, 0))
    jobs = [(s, r, p, 1, means) for s, r, p in zip(shots, refs, predictions, strict=True)]
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(fast_score._score, jobs, chunksize=2))
    acc = fast_score.official.Accum()
    acc.psi_sign = 1
    for row, part in zip(rows, parts, strict=True):
        acc.add(part)
        row['boundary_distance'] = part['shot_dlcfs']
        row.update({f'{key}_sse': float(value) for key, value in
                    zip(fast_score.official.CONS_SCALARS, part['ss_res_cons'], strict=True)})
    score = fast_score.official.finalize_machine(acc, stats)
    denominator = psi_sumsq - psi_sum ** 2 / pixels
    deficits = {'flux': 0.55 * (1 - score['r2_psi']),
                'scalars': 0.15 * (1 - score['r2_qb']), 'boundary': 0.1 * score['dlcfs']}
    deficits.update({k: 0.2 / 7 * (1 - value) for k, value in score['r2_cons_each'].items()})
    if abs(sum(deficits.values()) - (1 - score['S'])) > 2e-6:
        raise AssertionError('Diagnostic deficit does not reproduce composite score')
    groups = {}
    for name, mask in subsets(x).items():
        groups[name] = {'frames': int(mask.sum()), 'frame_share': float(mask.mean()),
                        'flux_sse_share': float(frame_psi[mask].sum() / frame_psi.sum()),
                        'scalar_sse_share': (np.nansum(frame_scalars[mask], 0)
                                             / np.nansum(frame_scalars, 0)).tolist()}
    ordered = sorted(rows, key=lambda row: row['flux_sse'], reverse=True)
    with (args.out / 'shots.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ordered)
    np.save(args.out / 'spatial-rmse.npy', np.sqrt(spatial / count))
    hashes = {}
    for path in args.checkpoints:
        with path.open('rb') as handle:
            hashes[str(path)] = file_digest(handle, 'sha256').hexdigest()
    report = {'score': score, 'weighted_deficit': deficits, 'groups': groups,
              'top_shots': {n: sum(r['flux_sse'] for r in ordered[:n]) / frame_psi.sum()
                            for n in (1, 5, 10, 26)},
              'pca_oracle_pixel_deficit': oracle_sse / denominator,
              'float16_rounding_pixel_deficit': quantization_sse / denominator,
              'inference_plus_diagnostics_seconds': inference_time,
              'inference_seconds': sum(r['inference_seconds'] for r in rows),
              'frames': count, 'shots': len(rows), 'upload_dtype': 'float32', 'checkpoint_sha256': hashes,
              'packed_index': str(args.packed / 'index.json'), 'locked_shots_read': False}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != 'checkpoint_sha256'},
                     indent=2, default=float), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--refs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=12)
    main(parser.parse_args())
