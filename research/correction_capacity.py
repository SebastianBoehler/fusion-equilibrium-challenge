"""Training-only spatial approximation diagnostic, never an inference correction."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from participant import fast_score
from participant.data import read_shot
from participant.nn_submit import Predictor
from participant.rich import INPUT_COLUMNS, rich_features
from research.inductance_contour_audit import MODELS, derive_predictions, digest, error

ROOT = Path('run_data/analysis-025')


def approximate(residual, grid):
    basis = F.interpolate(torch.eye(grid * grid, dtype=torch.float64).reshape(-1, 1, grid, grid),
                          size=(65, 65), mode='bilinear', align_corners=True).flatten(1).T
    fit = torch.linalg.lstsq(basis, residual.T.contiguous(), driver='gels').solution
    projected = (basis @ fit).T
    normal_residual = basis.T @ (residual - projected).T
    if not torch.isfinite(projected).all() or float(normal_residual.abs().max()) > 1e-8:
        raise FloatingPointError('Spatial least-squares solution failed its normal equation')
    return projected, float(normal_residual.abs().max())


def metrics(prediction, truth, reference):
    actual, boundary, failed = derive_predictions(prediction, reference[0], fast_score._geometry())
    return {'flux_sse': float(np.square(prediction.astype(float) - truth).sum()),
            'boundary_mean': float(np.mean(boundary)), 'extraction_failures': failed,
            'derived_errors': {name: error(actual[:, j], reference[1][:, j])
                               for j, name in enumerate(fast_score.official.CONS_SCALARS)}}


def main():
    ROOT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    manifest_path = Path('run_data/analysis-020/packing-check/training-only-manifest.json')
    manifest = json.loads(manifest_path.read_text())
    index = json.loads(Path('run_data/analysis-020/packing-check/train/index.json').read_text())
    if index['names'] != [Path(e['path']).stem for e in manifest['train']]:
        raise ValueError('Training shot order mismatch')
    rows = [read_shot(entry, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                      (*INPUT_COLUMNS, 'efit_psirz')) for entry in manifest['train']]
    truth = np.concatenate([np.asarray(row['efit_psirz'], np.float32) for row in rows])
    if len(truth) != index['frames'] or len(truth) != 508:
        raise ValueError('Expected the three whole hash-verified training shots')
    parent_path, parent_sha = MODELS['parent']
    if digest(parent_path) != parent_sha:
        raise ValueError('Frozen parent changed')
    model = Predictor([parent_path])
    prediction = np.concatenate([model.predict_inputs(rich_features(row))['psirz'].astype(np.float32)
                                 for row in rows])
    reference = fast_score.official.build_reference(truth.astype(float), *fast_score._geometry())
    if any(c is None for c in reference[0]):
        raise ValueError('Diagnostic requires valid training reference contours')
    residual = torch.from_numpy((truth.astype(float) - prediction).reshape(-1, 4225))
    baseline = metrics(prediction, truth, reference)
    arms = {}
    for grid in (17, 33):
        projected, normal_error = approximate(residual, grid)
        corrected = (prediction.astype(float) + projected.numpy().reshape(-1, 65, 65)).astype(np.float32)
        measured = metrics(corrected, truth, reference)
        measured['normal_equation_max_error'] = normal_error
        measured['residual_pixel_sse_ratio'] = measured['flux_sse'] / baseline['flux_sse']
        measured['correction_gradient_roughness'] = float(np.mean(np.diff(projected.numpy().reshape(-1, 65, 65), axis=1) ** 2)
                                                        + np.mean(np.diff(projected.numpy().reshape(-1, 65, 65), axis=2) ** 2))
        arms[str(grid)] = measured
    result = {'complete': True, 'training_only': True, 'frames': len(truth), 'train_shots': 3,
              'training_manifest_sha256': digest(manifest_path), 'parent_sha256': parent_sha,
              'baseline': baseline, 'spatial_projections': arms, 'elapsed_cpu_seconds': time.monotonic() - started,
              'labels_used_as_inference_inputs': False, 'candidate_created': False,
              'limitations': 'Per-frame truth-residual least squares on three training shots. Measures spatial pixel approximation only; ignores predictability from allowed inputs, finite model capacity, optimization and generalization. These fields cannot be submitted.'}
    (ROOT / 'capacity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
