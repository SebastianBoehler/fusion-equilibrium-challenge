"""Inspect contour extrema and closure levels on the existing training sample."""

import json
from pathlib import Path

import numpy as np
import torch

from participant import fast_score
from participant.data import read_shot
from participant.nn_submit import Predictor
from participant.rich import INPUT_COLUMNS, rich_features
from research.inductance_contour_audit import MODELS, digest
from vendor.fusion_starter.fusion_scoring.derive import _bilinear, _points_in_polygon


def selected_predictions(paths, probe, packed):
    model = Predictor(paths)
    selected = []
    offset = 0
    for entry in probe['training_shots']:
        row = read_shot(entry, probe['revision'],
                        Path('run_data/analysis-019/training-raw'), INPUT_COLUMNS)
        x = rich_features({k: row[k] for k in INPUT_COLUMNS})
        indices = entry['frames']
        assert np.array_equal(x[indices], packed['x'][offset:offset + len(indices)])
        # The incumbent includes context2: infer complete shots before selecting frames.
        selected.append(model.predict_inputs(x)['psirz'][indices])
        offset += len(indices)
    assert offset == len(packed['psi'])
    return np.concatenate(selected)


def extents(contour):
    return np.array([contour[:, 0].min(), contour[:, 0].max(),
                     contour[:, 1].min(), contour[:, 1].max()])


def measure(prediction, truth, contours):
    r, z, coarse, mask_f = fast_score._geometry()
    official = fast_score.official
    rr, zz = np.meshgrid(r, z)
    sign = min(official.PSI_SIGNS, key=lambda s: np.square(truth - s * prediction).sum())
    axis_sign = official.AXIS_SIGN['DIII-D'] * sign
    rows, failures = [], 0
    for field, target, reference in zip(prediction, truth, contours, strict=True):
        contour, _ = official.extract_lcfs_with_sign(
            field, r, z, 'DIII-D', coarse, mask_f, axis_sign=axis_sign)
        if contour is None or reference is None:
            failures += 1
            continue
        e, t = extents(contour), extents(reference)
        width, height = e[1] - e[0], e[3] - e[2]
        true_width, true_height = t[1] - t[0], t[3] - t[2]
        height_term = (height - true_height) / true_width
        width_term = -height * (width - true_width) / (width * true_width)
        kappa_error = height / width - true_height / true_width
        assert abs(kappa_error - height_term - width_term) < 1e-12
        mask = _points_in_polygon(rr.ravel(), zz.ravel(), reference).reshape(65, 65)
        inside = field[mask].astype(float)
        design = np.column_stack((inside, np.ones(len(inside))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, target[mask], rcond=None)
        if rank != 2:
            raise ValueError('Undefined amplitude/gauge alignment')
        aligned = coefficients[0] * field + coefficients[1]
        true_boundary = _bilinear(target, r, z, reference[:, 0], reference[:, 1])
        fixed_boundary = _bilinear(aligned, r, z, reference[:, 0], reference[:, 1])
        own_boundary = _bilinear(aligned, r, z, contour[:, 0], contour[:, 1])
        span = abs(target[mask].max() - target[mask].min())
        if span <= 0:
            raise ValueError('Undefined true interior flux span')
        rows.append({'extent_error_m': (e - t).tolist(),
                     'height_error_m': float(height - true_height),
                     'width_error_m': float(width - true_width),
                     'kappa_error': float(kappa_error),
                     'height_kappa_term': float(height_term), 'width_kappa_term': float(width_term),
                     'aligned_closure_level_error_fraction':
                         float((np.mean(own_boundary) - np.mean(true_boundary)) / span),
                     'fixed_contour_flux_error_fraction':
                         float(np.mean((fixed_boundary - true_boundary) ** 2) / span ** 2)})
    if not rows:
        raise ValueError('No valid contour pairs in training diagnostic')
    values = {key: np.array([row[key] for row in rows]) for key in rows[0]}
    h, w = values['height_kappa_term'], values['width_kappa_term']
    mse = {key: float(np.mean(v ** 2)) for key, v in values.items() if v.ndim == 1}
    mse.pop('fixed_contour_flux_error_fraction')
    summary = {'valid_frames': len(rows), 'excluded_frames': failures, 'flux_sign': sign,
               'extent_rmse_m': dict(zip(('R_min', 'R_max', 'Z_min', 'Z_max'),
                                         np.sqrt(np.mean(values['extent_error_m'] ** 2, axis=0)).tolist())),
               'mse': mse, 'kappa_height_width_cross_term': float(2 * np.mean(h * w)),
               'fixed_contour_flux_mse_fraction': float(values['fixed_contour_flux_error_fraction'].mean())}
    assert abs(mse['kappa_error'] - mse['height_kappa_term'] - mse['width_kappa_term']
               - summary['kappa_height_width_cross_term']) < 1e-12
    return summary, rows


def main():
    torch.set_num_threads(2)
    root = Path('run_data/analysis-020')
    out = root / 'contour-extrema-audit.json'
    if out.exists():
        raise FileExistsError('Preserve the existing extrema audit')
    probe = json.loads((root / 'batch-and-correction-probe.json').read_text())
    pack_path = root / 'training-target-pack.npz'
    if digest(pack_path) != probe['target_pack_sha256']:
        raise ValueError('Existing diagnostic pack changed')
    packed = np.load(pack_path)
    truth = packed['psi'].astype(float)
    r, z, coarse, mask_f = fast_score._geometry()
    contours = fast_score.official.build_reference(truth, r, z, coarse, mask_f)[0]
    incumbent = json.loads(Path('run_data/submission-010/build-receipt.json').read_text())['checkpoints']
    hashes = {str(path): sha for path, sha in MODELS.values()} | incumbent
    for path, sha in hashes.items():
        if digest(Path(path)) != sha:
            raise ValueError(f'Checkpoint changed: {path}')
    groups = {name: [path] for name, (path, _) in MODELS.items()}
    groups['eleven'] = list(map(Path, incumbent))
    result = {'training_only': True, 'frames': len(truth), 'target_pack_sha256': digest(pack_path),
              'checkpoint_sha256': hashes, 'whole_shot_temporal_context_preserved': True,
              'models': {}, 'per_frame': {}, 'locked_shots_read': False,
              'limitations': '96 selected training frames only; affine alignment is offline analysis, never a model input or prediction repair.'}
    for name, paths in groups.items():
        prediction = selected_predictions(paths, probe, packed)
        summary, rows = measure(prediction, truth, contours)
        result['models'][name], result['per_frame'][name] = summary, rows
        print(name, json.dumps(summary), flush=True)
    result['complete'] = True
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
