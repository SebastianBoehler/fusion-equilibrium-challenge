"""Validate a displacement-based elongation objective on existing training frames."""

import json
from pathlib import Path

import numpy as np
import torch

from participant import fast_score
from participant.contour_normal_loss import loss as normal_loss, projected, targets
from participant.finetune_common import prepare_inputs
from participant.flux_head import parameters
from participant.smooth_correction import SmoothCorrection
from research.contour_anchor_probe import prepare, terms
from research.inductance_contour_audit import MODELS, derive_predictions, digest, error


def gradient(loss, model):
    values = torch.autograd.grad(loss, tuple(model.correction.parameters()), retain_graph=True)
    return torch.cat([v.flatten() for v in values])


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    root = Path('run_data/analysis-023/joint')
    root.mkdir(exist_ok=True)
    if (root / 'normal-contour-probe.json').exists():
        raise FileExistsError('Preserve the completed probe receipt')
    pack_path = Path('run_data/analysis-020/training-target-pack.npz')
    probe = json.loads(Path('run_data/analysis-020/batch-and-correction-probe.json').read_text())
    assert digest(pack_path) == probe['target_pack_sha256']
    packed = np.load(pack_path)
    parent_path, sha = MODELS['parent']
    assert digest(parent_path) == sha
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    x = prepare_inputs(packed['x'], parameters(parent, 'cpu'), parent['config'], 'cpu')
    truth = torch.from_numpy(packed['psi'])
    fixed, reference, check = prepare(packed)
    r, z, functional, _, _ = fixed
    points = functional[0]
    normal = targets(truth, r, z, points, np.asarray(reference[0]), functional[2])
    perfect = float(normal_loss(truth, points, normal))
    assert perfect < 1e-10
    model = SmoothCorrection(parent).train()
    initial, initial_scalar = model(x)
    initial = initial.reshape(-1, 65, 65)
    base_loss = normal_loss(initial, points, normal)
    invariance = {}
    for name, prediction in [('weak', initial * 1e-12), ('sign', -initial),
                              ('amplitude', initial * 1.1), ('gauge', initial + 1)]:
        delta = float(abs(normal_loss(prediction, points, normal) - base_loss))
        invariance[name] = delta
        assert delta < 2e-5, (name, delta)
    variance = parent['coef_scale'].square().sum()
    first = terms(initial, truth, variance, fixed)
    li_gradient, shape_gradient = gradient(first['li'], model), gradient(base_loss, model)
    assert torch.isfinite(shape_gradient).all() and shape_gradient.norm() > 0
    weight = float(0.05 * li_gradient.norm() / shape_gradient.norm())
    assert np.isfinite(weight) and weight > 0
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
    records = []
    for step in range(6):
        prediction, scalars = model(x)
        prediction = prediction.reshape(-1, 65, 65)
        losses = terms(prediction, truth, variance, fixed)
        kappa = normal_loss(prediction, points, normal)
        objective = (losses['pixel'] + 0.05 * losses['li'] + weight * kappa
                     + 0.1 * (losses['boundary'] + losses['axis']))
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1)
        assert torch.isfinite(norm)
        optimizer.step()
        assert torch.equal(scalars, initial_scalar)
        records.append({'step': step + 1, **{k: float(v) for k, v in losses.items()},
                        'projected_kappa_mse': float(kappa), 'gradient_norm': float(norm)})
    with torch.no_grad():
        updated, final_scalar = model(x)
        final = updated.reshape(-1, 65, 65)
        final_losses = terms(final, truth, variance, fixed)
        estimated = projected(final, points, normal)
    actual, boundary, failed = derive_predictions(final.numpy(), reference[0], fast_score._geometry())
    parent_audit = json.loads(Path('run_data/analysis-020/contour-audit.json').read_text())['models']['parent']
    derived = {name: error(actual[:, j], reference[1][:, j])
               for j, name in enumerate(fast_score.official.CONS_SCALARS)}
    maximum_movement = float((estimated - normal[2]).norm(dim=2).max())
    numerical_valid = maximum_movement <= float(normal[-1]) + 1e-6
    assert numerical_valid
    success = (numerical_valid and failed == 0 and derived['li']['mse'] < parent_audit['derived']['li']['mse']
               and derived['kappa']['mse'] < parent_audit['derived']['kappa']['mse']
               and np.mean(boundary) < parent_audit['normalized_boundary_mean']
               and float(final_losses['pixel']) < float(first['pixel']))
    assert torch.equal(final_scalar, initial_scalar)
    assert all(p.grad is None for p in model.parent.parameters())
    result = {'training_only': True, 'frames': len(truth), 'target_pack_sha256': digest(pack_path),
              'parent_sha256': sha, 'target_checks': check, 'perfect_projection_loss': perfect,
              'invariance_absolute_errors': invariance, 'weight_from_initial_training_gradients': weight,
              'initial_gradient_norms': {'li': float(li_gradient.norm()), 'projected_kappa': float(shape_gradient.norm())},
              'boundary_and_axis_weights': 0.1,
              'updates': records, 'final_terms': {k: float(v) for k, v in final_losses.items()},
              'actual_derived_errors': derived, 'normalized_boundary_mean': float(np.mean(boundary)),
              'extraction_failures': failed, 'scalar_outputs_exact': True, 'parent_gradients_absent': True,
              'max_projected_displacement_m': maximum_movement,
              'physical_grid_trust_radius_m': float(normal[-1]), 'numerical_validity_check': numerical_valid,
              'mechanism_success_check': bool(success), 'gpu_scheduled': False,
              'limitations': 'First-order fixed-reference contour model; six CPU updates on96 selected training frames; no full-corpus, CUDA or held-out gain established.'}
    (root / 'normal-contour-probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
