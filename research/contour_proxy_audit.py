"""Audit the024 displacement proxy on three whole training shots after recovery."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from participant import fast_score
from participant.contour_normal_loss import elongation, physical_points, projected, targets
from participant.contour_training import ScoredContourBatch
from participant.data import read_shot
from participant.inductance_loss import inductance
from participant.nn_submit import Predictor
from participant.rich import INPUT_COLUMNS, rich_features
from research.inductance_contour_audit import MODELS, derive_predictions, digest, error


ROOT = Path('run_data/analysis-027')


def distribution(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite proxy diagnostic')
    return dict(zip(('min', 'median', 'p90', 'p99', 'max'),
                    np.quantile(values, [0, .5, .9, .99, 1]).tolist(), strict=True))


def residual_ratio(prediction, points, prepared):
    values, gradient, _, interior, mean, centered, _, radius = prepared
    pred_mean = (prediction * interior).sum((1, 2), keepdim=True)
    centered_prediction = prediction - pred_mean
    variance = (centered_prediction.square() * interior).sum((1, 2), keepdim=True)
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise ValueError('Undefined diagnostic alignment')
    slope = (centered_prediction * centered * interior).sum((1, 2), keepdim=True) / variance
    aligned = centered_prediction * slope + mean
    sampled = F.grid_sample(aligned[:, None], points[:, :, None], align_corners=True)[:, 0, :, 0]
    return (sampled - values).abs() / (2 * radius * gradient.norm(dim=-1))


def measure(prediction, truth, reference, points, weights, interior, target_li, r, z, prepared):
    tensor = torch.from_numpy(prediction)
    ratio = residual_ratio(tensor, points, prepared).numpy()
    contour = prepared[2]
    estimated = projected(tensor, points, prepared)
    proxy_kappa = elongation(estimated).numpy()
    target_kappa = prepared[-2].numpy()
    bad = ratio > 1
    coords = contour.numpy()
    extrema = {'inner': coords[:, :, 0].argmin(1), 'outer': coords[:, :, 0].argmax(1),
               'bottom': coords[:, :, 1].argmin(1), 'top': coords[:, :, 1].argmax(1)}
    actual, boundary, failed = derive_predictions(prediction, reference[0], fast_score._geometry())
    fixed_li = inductance(tensor, r, z, points, weights, interior).numpy()
    return {
        'aligned_residual_over_twice_cell_gradient': distribution(ratio),
        'points_in_reversed_magnitude_gradient_region': int(bad.sum()),
        'total_boundary_points': int(bad.size),
        'point_fraction_in_reversed_magnitude_gradient_region': float(bad.mean()),
        'frames_with_any_reversed_magnitude_gradient_points': int(bad.any(1).sum()),
        'true_extrema_fraction_in_reversed_region': {
            name: float(bad[np.arange(len(bad)), indices].mean()) for name, indices in extrema.items()},
        'linear_first_order_normal_movement_m': distribution(ratio * (2 * float(prepared[-1]))),
        'bounded_proxy_movement_m': distribution((estimated - contour).norm(dim=-1).numpy()),
        'bounded_proxy_kappa': error(proxy_kappa, target_kappa),
        'fixed_true_contour_li': error(fixed_li, target_li.numpy()),
        'actual_predicted_contour_derived': {
            name: error(actual[:, j], reference[1][:, j])
            for j, name in enumerate(fast_score.official.CONS_SCALARS)},
        'actual_boundary_mean': float(np.mean(boundary)), 'extraction_failures': failed,
        'flux_sse': float(np.square(prediction.astype(float) - truth.astype(float)).sum())}


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    if ROOT.exists():
        raise FileExistsError('Preserve existing or active proxy audit')
    recovered = json.loads(Path('run_data/analysis-024/checkpoint-recovery.json').read_text())
    if not recovered['checkpoint_metadata_and_source_verified']:
        raise ValueError('Verified actual024 checkpoint recovery required')
    old = Path('run_data/analysis-020/packing-check')
    manifest_path = old / 'training-only-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((old / 'train/index.json').read_text())
    if index['names'] != [Path(e['path']).stem for e in manifest['train']]:
        raise ValueError('Training-only shot order mismatch')
    rows = [read_shot(e, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                      (*INPUT_COLUMNS, 'efit_psirz')) for e in manifest['train']]
    truth = np.concatenate([np.asarray(row['efit_psirz'], np.float32) for row in rows])
    if len(truth) != index['frames'] or len(truth) != 508:
        raise ValueError('Expected all508 frames of three whole training shots')
    target_root = Path('run_data/analysis-023/scored-targets')
    validation = json.loads(Path('run_data/analysis-023/target-validation.json').read_text())
    for name, sha in validation['file_sha256'].items():
        if digest(target_root / name) != sha:
            raise ValueError('Existing three-shot target artifact changed')
    batch = ScoredContourBatch(target_root, index)
    if len(batch.valid_indices) != 508:
        raise ValueError('Expected all508 references usable in this training diagnostic')
    p, w, interior, li = batch.get(batch.valid_indices, 'cpu')
    r, z = (torch.from_numpy(v) for v in (batch.r, batch.z))
    reference = fast_score.official.build_reference(truth.astype(float), *fast_score._geometry())
    normal = targets(torch.from_numpy(truth), r, z, p, physical_points(p, r, z), interior)
    models = {'parent': MODELS['parent']}
    for arm in ('aligned-li', 'aligned-li-kappa'):
        models[arm] = (Path('run_data/contour-024') / arm / 'model.pt',
                       recovered['checkpoint_sha256'][arm])
    measurements = {}
    for name, (path, sha) in models.items():
        if digest(path) != sha:
            raise ValueError('Actual diagnostic checkpoint hash mismatch')
        model = Predictor([path])
        prediction = np.concatenate([model.predict_inputs(rich_features(row))['psirz'].astype(np.float32)
                                     for row in rows])
        measurements[name] = measure(prediction, truth, reference, p, w, interior, li, r, z, normal)
        measurements[name]['checkpoint_sha256'] = sha
        print(name, 'proxy_kappa', measurements[name]['bounded_proxy_kappa']['mse'],
              'actual_kappa', measurements[name]['actual_predicted_contour_derived']['kappa']['mse'],
              'reversed_point_fraction', measurements[name]['point_fraction_in_reversed_magnitude_gradient_region'],
              flush=True)
    u = np.asarray([0, .5, 1, 2, 4, 16, 256], dtype=float)
    result = {'complete': True, 'training_only': True, 'frames': 508, 'train_shots': 3,
              'training_manifest_sha256': digest(manifest_path), 'checkpoint_measurements': measurements,
              'mathematical_damping': {
                  'u': 'absolute aligned boundary residual / (2 * cell * true gradient norm)',
                  'movement_magnitude_over_cell': '2u/(1+u^2)',
                  'magnitude_derivative_wrt_u': '2(1-u^2)/(1+u^2)^2',
                  'magnitude_decreases_for_u_greater_than_one': True,
                  'infinite_residual_movement_limit': 0,
                  'dimensionless_formula_values': dict(zip(u.tolist(), (2*u/(1+u*u)).tolist(), strict=True))},
              'cell_m': float(normal[-1]), 'elapsed_cpu_seconds': time.monotonic() - started,
              'locked_shots_read': False, 'candidate_created': False, 'new_gpu_scheduled': False,
              'limitations': 'Training-only descriptive measurements on three whole shots. Nonmonotonic damping is an algebraic defect, but prevalence does not prove it solely caused CUDA held-out regressions. True fields/contours appear only in this offline diagnostic, never inference inputs or submission repairs.'}
    ROOT.mkdir()
    with (ROOT / 'proxy-audit.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
