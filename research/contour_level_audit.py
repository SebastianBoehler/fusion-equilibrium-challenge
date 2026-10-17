"""Training-only audit of fixed-reference versus predicted critical flux levels."""

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
from participant.nn_submit import Predictor
from participant.rich import INPUT_COLUMNS, rich_features
from research.contour_proxy_audit import distribution
from research.inductance_contour_audit import MODELS, digest, error
from research.monotone_contour_proxy import displacement, projected as monotone_projected

# fast_score has loaded the organizer modules from its vendored source directory.
from contour import grid_to_physical, resample_closed
from lcfs import _best_contour_at
from o_point import find_o_point


ROOT = Path('run_data/analysis-029')


def critical_trace(psi):
    r, z, coarse, mask_f = fast_score._geometry()
    sign = fast_score.official.AXIS_SIGN['DIII-D']
    phi = np.asarray(psi, dtype=float) * sign
    _, _, iz, ir, _ = find_o_point(phi, r, z, coarse)
    border = np.concatenate([phi[0], phi[-1], phi[:, 0], phi[:, -1]])
    lo, hi = float(border.min()), float(phi[iz, ir]) - 1e-9
    if hi <= lo:
        raise ValueError('Undefined critical flux bracket in diagnostic')
    best = _best_contour_at(phi, hi, iz, ir, mask_f)
    if best is None:
        raise ValueError('Undefined closed surface in diagnostic')
    for _ in range(fast_score.official.N_ITER):
        mid = .5 * (lo + hi)
        contour = _best_contour_at(phi, mid, iz, ir, mask_f)
        if contour is None:
            lo = mid
        else:
            hi, best = mid, contour
    traced = resample_closed(grid_to_physical(best, r, z), 512)
    actual, _ = fast_score.official.extract_lcfs_with_sign(
        psi, r, z, 'DIII-D', coarse, mask_f,
        n_iter=fast_score.official.N_ITER, axis_sign=sign)
    if actual is None or not np.array_equal(traced, actual):
        raise ValueError('Diagnostic critical-level trace differs from unchanged scorer')
    return hi / sign, actual


def alignment(prediction, points, prepared):
    _, _, _, interior, mean, centered, _, _ = prepared
    pred_mean = (prediction * interior).sum((1, 2), keepdim=True)
    pred_centered = prediction - pred_mean
    variance = (pred_centered.square() * interior).sum((1, 2), keepdim=True)
    if not torch.isfinite(variance).all() or torch.any(variance <= 0):
        raise FloatingPointError('Undefined diagnostic field alignment')
    slope = (pred_centered * centered * interior).sum((1, 2), keepdim=True) / variance
    offset = mean - pred_mean * slope
    aligned = prediction * slope + offset
    values = F.grid_sample(aligned[:, None], points[:, :, None], align_corners=True)[:, 0, :, 0]
    return values, slope.flatten(), offset.flatten()


def proxy_measure(contour, target_kappa, actual_kappa):
    kappa = elongation(contour).numpy()
    proxy_error = kappa - target_kappa
    actual_error = actual_kappa - target_kappa
    return {'kappa_against_true_contour': error(kappa, target_kappa),
            'kappa_approximation_error_against_actual_predicted_contour': error(kappa, actual_kappa),
            'sign_agreement_with_actual_kappa_error': float((proxy_error * actual_error >= 0).mean()),
            'error_correlation_with_actual': float(np.corrcoef(proxy_error, actual_error)[0, 1])}


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    if ROOT.exists():
        raise FileExistsError('Preserve existing or active critical-level audit')
    old = Path('run_data/analysis-020/packing-check')
    manifest_path = old / 'training-only-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((old / 'train/index.json').read_text())
    assert index['names'] == [Path(e['path']).stem for e in manifest['train']]
    rows = [read_shot(e, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                      (*INPUT_COLUMNS, 'efit_psirz')) for e in manifest['train']]
    fields = np.concatenate([np.asarray(row['efit_psirz'], np.float32) for row in rows])
    assert len(fields) == index['frames'] == 508
    target_root = Path('run_data/analysis-023/scored-targets')
    validation = json.loads(Path('run_data/analysis-023/target-validation.json').read_text())
    for name, sha in validation['file_sha256'].items():
        assert digest(target_root / name) == sha
    batch = ScoredContourBatch(target_root, index)
    assert len(batch.valid_indices) == 508
    points, _, interior, _ = batch.get(batch.valid_indices, 'cpu')
    r, z = (torch.from_numpy(v) for v in (batch.r, batch.z))
    prepared = targets(torch.from_numpy(fields), r, z, points, physical_points(points, r, z), interior)
    truth_trace = [critical_trace(field) for field in fields]
    true_level = np.asarray([v[0] for v in truth_trace])
    target_kappa = np.asarray([(c[:, 1].max() - c[:, 1].min()) /
                              (c[:, 0].max() - c[:, 0].min()) for _, c in truth_trace])
    std = fields.astype(float).std(axis=(1, 2))
    if np.any(std <= 0):
        raise ValueError('Undefined frame field scale')
    recovered = json.loads(Path('run_data/analysis-024/checkpoint-recovery.json').read_text())
    assert recovered['checkpoint_metadata_and_source_verified']
    models = {'parent': MODELS['parent']} | {
        arm: (Path('run_data/contour-024') / arm / 'model.pt', recovered['checkpoint_sha256'][arm])
        for arm in ('aligned-li', 'aligned-li-kappa')}
    result = {'complete': False, 'training_only': True, 'frames': 508, 'shots': 3,
              'training_manifest_sha256': digest(manifest_path), 'models': {},
              'locked_shots_read': False, 'new_gpu_scheduled': False,
              'learned_candidate_created': False, 'projected_field_files_saved': False}
    ROOT.mkdir()
    output = ROOT / 'level-audit.json'
    for name, (path, sha) in models.items():
        assert digest(path) == sha
        model = Predictor([path])
        predicted = np.concatenate([model.predict_inputs(rich_features(row))['psirz'].astype(np.float32)
                                    for row in rows])
        traces = [critical_trace(field) for field in predicted]
        level = torch.tensor([v[0] for v in traces], dtype=torch.float32)
        actual_kappa = np.asarray([(c[:, 1].max() - c[:, 1].min()) /
                                  (c[:, 0].max() - c[:, 0].min()) for _, c in traces])
        psi = torch.from_numpy(predicted)
        values, slope, offset = alignment(psi, points, prepared)
        aligned_level = slope * level + offset
        residual = values - aligned_level[:, None]
        adjusted = prepared[2] + displacement(residual, prepared[1], prepared[-1])
        shift = aligned_level.numpy().astype(float) - true_level
        result['models'][name] = {
            'checkpoint_sha256': sha, 'critical_trace_matches_actual_scorer_all_frames': True,
            'absolute_aligned_predicted_critical_level_shift_over_true_field_std': distribution(abs(shift) / std),
            'signed_aligned_critical_level_shift_mean': float(shift.mean()),
            'old_fixed_reference': proxy_measure(projected(psi, points, prepared), target_kappa, actual_kappa),
            'monotone_fixed_reference': proxy_measure(monotone_projected(psi, points, prepared), target_kappa, actual_kappa),
            'monotone_predicted_critical_level': proxy_measure(adjusted, target_kappa, actual_kappa)}
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(name, result['models'][name], flush=True)
    result['complete'] = True
    result['elapsed_cpu_seconds'] = time.monotonic() - started
    result['source_sha256'] = digest(Path('research/contour_level_audit.py'))
    result['limitations'] = 'Offline diagnostics on three whole training shots, no learned candidate or achievable score. Critical levels are extracted from each prediction; this does not implement their training derivative, resolve saturation/closure, establish causality or demonstrate generalization. True normals/targets stay in diagnostics only.'
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
