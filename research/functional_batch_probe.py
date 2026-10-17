"""Training-only float32 batch checks and matched smooth-correction CPU controls."""

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from participant import score  # Initialize vendored absolute imports.
from participant.data import read_shot
from participant.finetune_common import prepare_inputs
from participant.flux_head import decode, parameters
from participant.geometry_targets import COLUMNS
from participant.rich import INPUT_COLUMNS, rich_features
from participant.inductance_loss import geometry, inductance
from participant.smooth_correction import SmoothCorrection
from vendor.fusion_starter.fusion_scoring.derive import _points_in_polygon, internal_inductance

PARENT_SHA = 'fda8f02587063b305787728b78376c979b0e21e7d7fd9a25471671a9d3c17a95'


def prepare():
    split = json.loads(Path('run_data/full-split.json').read_text())
    rng = np.random.default_rng(20261021)
    xs, fields, contours, masks, expected, receipts, currents = [], [], [], [], [], [], []
    r0 = z0 = None
    for entry in split['train'][:3]:
        row = read_shot(entry, split['revision'], Path('run_data/analysis-019/training-raw'),
                        tuple(dict.fromkeys(INPUT_COLUMNS + COLUMNS)))
        r, z = (np.asarray(row[key], float) for key in ('efit_grid_R', 'efit_grid_Z'))
        if r0 is not None:
            assert np.array_equal(r, r0) and np.array_equal(z, z0)
        r0, z0 = r, z
        x = rich_features({key: row[key] for key in INPUT_COLUMNS})
        low = np.argsort(abs(x[:, 20]))[:16]
        others = np.setdiff1d(np.arange(len(x)), low)
        indices = np.r_[low, rng.choice(others, 16, replace=False)]
        rr, zz = np.meshgrid(r, z)
        for index in indices:
            count = int(row['efit_lcfs_n'][index])
            contour = np.column_stack((row['efit_lcfs_r'][index][:count],
                                       row['efit_lcfs_z'][index][:count])).astype(float)
            psi = np.asarray(row['efit_psirz'][index], np.float32)
            li = internal_inductance(psi, r, z, 'DIII-D', contour)
            assert np.isfinite(li)
            xs.append(x[index]); fields.append(psi); expected.append(li)
            currents.append(float(abs(x[index, 20])))
            contours.append(torch.tensor(contour, dtype=torch.float64))
            masks.append(torch.tensor(_points_in_polygon(rr.ravel(), zz.ravel(), contour).reshape(65, 65)))
        receipts.append({'path': entry['path'], 'sha256': entry['sha256'], 'frames': indices.tolist()})
    rt, zt = (torch.tensor(v, dtype=torch.float64) for v in (r0, z0))
    geom = tuple(v.float() for v in geometry(contours, masks, rt, zt))
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(fields)), rt.float(), zt.float(), geom, \
        torch.tensor(expected, dtype=torch.float32), receipts, currents, split['revision']


def check_batch(psi, r, z, geom, expected):
    measured = inductance(psi, r, z, *geom)
    error = float((measured - expected).abs().max())
    assert error < 2e-5, error
    errors = {}
    for name, transformed in {'weak_1e-6': psi * 1e-6, 'weak_1e-12': psi * 1e-12,
                              'amplitude_1.1': psi * 1.1, 'sign': -psi, 'gauge': psi + 1}.items():
        delta = float((inductance(transformed, r, z, *geom) - measured).abs().max())
        errors[name] = delta
        assert delta < 1e-4, (name, delta)
    noisy = (psi + psi.std((1, 2), keepdim=True) * 1e-3 * torch.randn_like(psi)).requires_grad_()
    gradient, = torch.autograd.grad((inductance(noisy, r, z, *geom) - measured).square().mean(), noisy)
    assert torch.isfinite(gradient).all() and torch.all(gradient.flatten(1).norm(dim=1) > 0)
    rejected = False
    try:
        inductance(torch.zeros_like(psi), r, z, *geom)
    except FloatingPointError:
        rejected = True
    assert rejected
    return {'max_scorer_absolute_error': error, 'max_invariance_errors': errors,
            'finite_nonzero_gradient_frames': len(psi), 'zero_field_rejected': rejected}


def control(parent, x, truth, r, z, geom, target_li, add_li, lr=2e-5, steps=3):
    model = SmoothCorrection(parent)
    model.train()
    params = parameters(parent, 'cpu')
    x = prepare_inputs(x.numpy(), params, parent['config'], 'cpu')
    with torch.no_grad():
        expected_out = model.parent(x)
        expected_psi = decode(expected_out, params, parent['config'])
        initial, scalars = model(x)
        assert torch.equal(initial, expected_psi)
        assert torch.equal(scalars, expected_out[:, parent['config']['components']:])
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=lr, weight_decay=0)
    variance = parent['coef_scale'].square().sum()
    records = []
    for step in range(steps):
        prediction, scalars = model(x)
        pixel = (prediction.reshape(-1, 65, 65) - truth).square().sum((1, 2)).mean() / variance
        li = (inductance(prediction.reshape(-1, 65, 65), r, z, *geom) - target_li).square().mean()
        norms = {}
        for name, term in (('pixel', pixel), ('li', li)):
            gradients = torch.autograd.grad(term, tuple(model.correction.parameters()), retain_graph=True)
            norms[name] = float(torch.cat([v.flatten() for v in gradients]).norm())
        objective = pixel + (0.05 * li if add_li else 0)
        assert torch.isfinite(objective)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            updated, scalar_after = model(x)
            assert torch.equal(scalar_after, scalars)
            change = (updated - prediction).reshape(-1, 65, 65)
            rms = change.square().mean().sqrt()
            roughness = change.diff(n=2, dim=1).square().mean() + change.diff(n=2, dim=2).square().mean()
            records.append({'step': step + 1, 'pixel_before': float(pixel), 'li_mse_before': float(li),
                            'term_gradient_norms': norms, 'total_norm_before_clip': float(gradient),
                            'flux_update_rms': float(rms), 'normalized_update_roughness': float(roughness / rms.square()),
                            'pixel_after': float((updated.reshape(-1, 65, 65) - truth).square().sum((1, 2)).mean() / variance),
                            'li_mse_after': float((inductance(updated.reshape(-1, 65, 65), r, z, *geom) - target_li).square().mean())})
    assert all(v.grad is None for v in model.parent.parameters())
    return {'zero_correction_exact': True, 'scalars_unchanged': True, 'frozen_parent_gradients': True,
            'trainable_parameters': sum(v.numel() for v in model.correction.parameters()), 'steps': records}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    out = Path('run_data/analysis-020')
    out.mkdir(parents=True, exist_ok=False)
    parent_path = Path('run_data/nn-008/mlp-shape/model.pt')
    with parent_path.open('rb') as handle:
        assert file_digest(handle, 'sha256').hexdigest() == PARENT_SHA
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    x, psi, r, z, geom, target, receipts, currents, revision = prepare()
    np.savez(out / 'training-target-pack.npz', x=x.numpy(), psi=psi.numpy(), r=r.numpy(), z=z.numpy(),
             points=geom[0].numpy(), boundary_weights=geom[1].numpy(), interior_weights=geom[2].numpy(),
             target_li=target.numpy())
    result = {'training_only': True, 'locked_shots_read': False, 'revision': revision,
              'training_shots': receipts, 'frames': len(psi), 'parent_sha256': PARENT_SHA,
              'current_range_kA': [min(currents), max(currents)], 'float32_batch': check_batch(psi, r, z, geom, target),
              'pixel_control': control(parent, x, psi, r, z, geom, target, False),
              'pixel_plus_li': control(parent, x, psi, r, z, geom, target, True),
              'gpu_job_scheduled': False,
              'limitations': 'Three fixed-batch CPU steps; fixed true contours; no full pack, CUDA or score improvement established'}
    with (out / 'training-target-pack.npz').open('rb') as handle:
        result['target_pack_sha256'] = file_digest(handle, 'sha256').hexdigest()
    (out / 'batch-and-correction-probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('training_shots', 'pixel_control', 'pixel_plus_li')}, indent=2))
    for key in ('pixel_control', 'pixel_plus_li'):
        print(key, json.dumps(result[key]))


if __name__ == '__main__':
    main()
