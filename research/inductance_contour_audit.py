"""Audit fixed EFIT versus scorer-extracted contours on existing training frames."""

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from participant import fast_score
from participant.data import read_shot
from participant.inductance_loss import inductance
from participant.nn_submit import Predictor
from vendor.fusion_starter.fusion_scoring.derive import internal_inductance


ROOT = Path('run_data/analysis-020')
MODELS = {
    'parent': (Path('run_data/nn-008/mlp-shape/model.pt'),
               'fda8f02587063b305787728b78376c979b0e21e7d7fd9a25471671a9d3c17a95'),
    'pixel': (Path('run_data/inductance-020/pixel/model.pt'),
              '1c3ab36766bd3e83d3943ee82767a14c96c47f1b30ece3e5d9b7413b08155c3b'),
    'pixel-li': (Path('run_data/inductance-020/pixel-li/model.pt'),
                 '86c9311b911bb27a776498900d30f3e642e84d0ec1e70bf82ef63dd00ab49d62'),
}


def digest(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def error(prediction, truth):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    residual = prediction[valid] - truth[valid]
    return {'valid_frames': int(valid.sum()), 'excluded_frames': int((~valid).sum()),
            'mse': float(np.mean(residual ** 2)) if valid.any() else None,
            'mean_error': float(np.mean(residual)) if valid.any() else None,
            'max_absolute_error': float(np.max(abs(residual))) if valid.any() else None}


def raw_contours(probe, packed):
    contours, offset = [], 0
    columns = ('source', 'efit_psirz', 'efit_grid_R', 'efit_grid_Z', 'efit_lcfs_n',
               'efit_lcfs_r', 'efit_lcfs_z')
    for entry in probe['training_shots']:
        row = read_shot(entry, probe['revision'],
                        Path('run_data/analysis-019/training-raw'), columns)
        indices = entry['frames']
        assert np.array_equal(np.asarray(row['efit_psirz'], np.float32)[indices],
                              packed['psi'][offset:offset + len(indices)])
        for i in indices:
            count = int(row['efit_lcfs_n'][i])
            contours.append(np.column_stack((row['efit_lcfs_r'][i][:count],
                                             row['efit_lcfs_z'][i][:count])).astype(float))
        offset += len(indices)
    assert offset == len(packed['psi'])
    return contours


def derive_predictions(prediction, contours, geometry, sign=1):
    r, z, coarse, mask_f = geometry
    official = fast_score.official
    axis_sign = official.AXIS_SIGN['DIII-D'] * sign
    values, boundaries, failed = [], [], 0
    for p, c in zip(prediction, contours, strict=True):
        predicted, _ = official.extract_lcfs_with_sign(
            p, r, z, 'DIII-D', coarse, mask_f, n_iter=official.N_ITER, axis_sign=axis_sign)
        failed += int(predicted is None)
        derived = official.derive_frame(p, r, z, 'DIII-D', coarse, mask_f,
                                        contour=predicted, axis_sign=axis_sign)
        values.append([derived[k] for k in official.CONS_SCALARS])
        if predicted is not None and c is not None:
            boundaries.append(official.symmetric_hausdorff(predicted, c) / official.major_radius(c))
    return np.asarray(values), boundaries, failed


def main():
    torch.set_num_threads(2)
    output = ROOT / 'contour-audit.json'
    if output.exists():
        raise FileExistsError('Preserve the existing contour audit')
    probe = json.loads((ROOT / 'batch-and-correction-probe.json').read_text())
    pack_path = ROOT / 'training-target-pack.npz'
    if digest(pack_path) != probe['target_pack_sha256']:
        raise ValueError('Existing training diagnostic pack changed')
    packed = np.load(pack_path, allow_pickle=False)
    truth = packed['psi'].astype(float)
    raw = raw_contours(probe, packed)
    r, z, coarse, mask_f = fast_score._geometry()
    assert np.allclose(r, packed['r'], rtol=0, atol=5e-7)
    assert np.allclose(z, packed['z'], rtol=0, atol=5e-7)
    official = fast_score.official
    auto, reference, _, _ = official.build_reference(truth, r, z, coarse, mask_f)
    li_index = official.CONS_SCALARS.index('li')
    target = packed['target_li'].astype(float)
    raw_numpy = np.array([internal_inductance(p, packed['r'], packed['z'], 'DIII-D', c)
                          for p, c in zip(truth, raw, strict=True)])
    assert np.max(abs(raw_numpy - target)) < 2e-5
    distances = [official.symmetric_hausdorff(a, b) / official.major_radius(b)
                 for a, b in zip(raw, auto, strict=True) if b is not None]
    result = {'training_only': True, 'locked_shots_read': False,
              'frames': len(truth), 'training_shots': probe['training_shots'],
              'target_pack_sha256': digest(pack_path),
              'target_comparison': error(target, reference[:, li_index]),
              'raw_functional_numerical_check': error(raw_numpy, target),
              'raw_vs_extracted_contour_mean_normalized_hausdorff': float(np.mean(distances)),
              'raw_vs_extracted_contour_valid_frames': len(distances),
              'models': {}, 'complete': False,
              'limitations': 'Selected 96 training frames only; not held-out model selection or official scoring.'}
    geom = [torch.from_numpy(packed[k]) for k in
            ('points', 'boundary_weights', 'interior_weights')]
    for name, (path, sha) in MODELS.items():
        if digest(path) != sha:
            raise ValueError(f'Checkpoint hash mismatch: {path}')
        model = Predictor([path])
        if any(cfg.get('context', 0) for _, _, cfg in model.members):
            raise ValueError('Selected training frames do not support temporal context')
        prediction = model.predict_inputs(packed['x'])['psirz']
        fixed_raw = inductance(torch.from_numpy(prediction), torch.from_numpy(packed['r']),
                               torch.from_numpy(packed['z']), *geom).numpy()
        fixed_auto = np.array([internal_inductance(p, r, z, 'DIII-D', c)
                               for p, c in zip(prediction, auto, strict=True)])
        sign = min(official.PSI_SIGNS, key=lambda s: np.square(truth - s * prediction).sum())
        values, boundaries, failed = derive_predictions(prediction, auto, (r, z, coarse, mask_f), sign)
        result['models'][name] = {
            'checkpoint_sha256': sha, 'flux_sign': sign,
            'fixed_raw_efit_contour_li': error(fixed_raw, target),
            'fixed_extracted_truth_contour_li': error(fixed_auto, reference[:, li_index]),
            'predicted_contour_li': error(values[:, li_index], reference[:, li_index]),
            'derived': {key: error(values[:, j], reference[:, j])
                        for j, key in enumerate(official.CONS_SCALARS)},
            'normalized_boundary_mean': float(np.mean(boundaries)) if boundaries else None,
            'boundary_valid_frames': len(boundaries), 'extraction_failures': failed}
        print(name, json.dumps(result['models'][name]), flush=True)
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('target_comparison', result['target_comparison'], flush=True)


if __name__ == '__main__':
    main()
