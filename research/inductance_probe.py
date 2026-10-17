"""Verify the differentiable fixed-contour functional on frozen training shots."""

from pathlib import Path
import json

import numpy as np
import torch

from participant import score  # Initialize the unchanged scorer's absolute imports.
from participant.data import read_shot
from participant.geometry_targets import COLUMNS
from participant.rich import INPUT_COLUMNS, rich_features
from research.inductance_functional import inductance
from vendor.fusion_starter.fusion_scoring.derive import _points_in_polygon, internal_inductance


def main():
    torch.set_num_threads(2)
    output = Path('run_data/analysis-019/inductance-functional-probe.json')
    assert not output.exists(), 'Do not repeat a completed frozen diagnostic'
    split = json.loads(Path('run_data/full-split.json').read_text())
    rng = np.random.default_rng(20261020)
    checks = []
    for entry in split['train'][:3]:
        columns = tuple(dict.fromkeys(INPUT_COLUMNS + COLUMNS))
        row = read_shot(entry, split['revision'], Path('run_data/analysis-019/training-raw'), columns)
        r, z = (np.asarray(row[key], float) for key in ('efit_grid_R', 'efit_grid_Z'))
        x = rich_features({key: row[key] for key in INPUT_COLUMNS})
        # The first 21 rich features are the raw interpolated actuator/Ip inputs.
        indices = np.argsort(np.abs(x[:, 20]))[:16]
        rr, zz = np.meshgrid(r, z)
        for index in indices:
            count = int(row['efit_lcfs_n'][index])
            contour = np.column_stack((row['efit_lcfs_r'][index][:count], row['efit_lcfs_z'][index][:count])).astype(float)
            field = np.asarray(row['efit_psirz'][index], float)
            expected = internal_inductance(field, r, z, 'DIII-D', contour)
            if not np.isfinite(expected):
                raise ValueError('Training reference inductance is invalid')
            inside = _points_in_polygon(rr.ravel(), zz.ravel(), contour).reshape(field.shape)
            rt, zt, ct = (torch.tensor(v, dtype=torch.float64) for v in (r, z, contour))
            mask = torch.tensor(inside)
            psi = torch.tensor(field, dtype=torch.float64)
            result = inductance(psi, rt, zt, ct, mask)
            assert abs(float(result) - expected) < 1e-10, (float(result), expected)
            transforms = {'amplitude_0.1': psi * 0.1, 'amplitude_1.1': psi * 1.1,
                          'amplitude_1e-6': psi * 1e-6, 'sign': -psi, 'gauge': psi + 1.0}
            errors = {name: abs(float(inductance(v, rt, zt, ct, mask)) - expected)
                      for name, v in transforms.items()}
            assert max(errors.values()) < 1e-9
            perturbation = torch.tensor(rng.normal(size=field.shape), dtype=torch.float64)
            prediction = (psi + psi.std() * 1e-3 * perturbation).requires_grad_()
            loss = (inductance(prediction, rt, zt, ct, mask) - result).square()
            gradient, = torch.autograd.grad(loss, prediction)
            assert torch.isfinite(gradient).all() and gradient.norm() > 0
            checks.append({'shot': entry['path'], 'frame': int(index), 'abs_ip_kA': float(abs(x[index, 20])),
                           'li': expected, 'match_error': abs(float(result) - expected),
                           'invariance_errors': errors, 'gradient_norm': float(gradient.norm())})
    # Zero field has undefined normalization and must fail explicitly.
    rejected = False
    try:
        inductance(torch.zeros_like(psi), rt, zt, ct, mask)
    except FloatingPointError:
        rejected = True
    assert rejected
    receipt = {'training_only': True, 'locked_shots_read': False, 'frames': len(checks),
               'fixed_true_training_contours': True, 'target': 'same functional of training psi; not raw EFIT LI',
               'published_scorer': 'vendored3.2.0', 'gpu_job_scheduled': False,
               'max_match_error': max(v['match_error'] for v in checks),
               'max_invariance_error': max(max(v['invariance_errors'].values()) for v in checks),
               'low_current_frames_abs_ip_lt_600kA': sum(v['abs_ip_kA'] < 600 for v in checks),
               'zero_field_denominator_rejected': rejected,
               'limitations': 'Float64 CPU prototype; fixed contours; no batched float32 CUDA training or ensemble improvement established',
               'checks': checks}
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print({k: v for k, v in receipt.items() if k != 'checks'})


if __name__ == '__main__':
    main()
