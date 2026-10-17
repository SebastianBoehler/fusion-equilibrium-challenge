"""Training-only mechanism probe for scorer contours and boundary/axis anchors."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from participant import fast_score
from participant.finetune_common import prepare_inputs
from participant.flux_head import parameters
from participant.geometry_loss import losses as shape_losses
from participant.geometry_targets import targets as shape_targets
from participant.inductance_loss import geometry, inductance
from participant.smooth_correction import SmoothCorrection
from research.inductance_contour_audit import MODELS, derive_predictions, digest, error
from vendor.fusion_starter.fusion_scoring.derive import _points_in_polygon


def prepare(packed):
    r, z, coarse, mask_f = fast_score._geometry()
    started = time.monotonic()
    contours, reference, _, _ = fast_score.official.build_reference(
        packed['psi'].astype(float), r, z, coarse, mask_f)
    elapsed = time.monotonic() - started
    if any(c is None for c in contours) or not np.isfinite(reference).all():
        raise ValueError('Mechanism probe requires valid existing training references')
    rr, zz = np.meshgrid(r, z)
    masks = [torch.tensor(_points_in_polygon(rr.ravel(), zz.ravel(), c).reshape(65, 65))
             for c in contours]
    rt, zt = torch.tensor(r), torch.tensor(z)
    fixed = tuple(v.float() for v in geometry([torch.tensor(c) for c in contours], masks, rt, zt))
    row = {'efit_psirz': packed['psi'], 'efit_grid_R': r, 'efit_grid_Z': z,
           'efit_lcfs_n': np.array([len(c) for c in contours]),
           'efit_lcfs_r': np.array(contours)[:, :, 0], 'efit_lcfs_z': np.array(contours)[:, :, 1],
           'efit_r_axis': reference[:, 0], 'efit_z_axis': reference[:, 1]}
    anchors = tuple(torch.from_numpy(v) for v in shape_targets(row))
    if not torch.all(anchors[-1]):
        raise ValueError('Invalid scorer contour/axis targets')
    truth = torch.from_numpy(packed['psi'])
    target = inductance(truth, rt.float(), zt.float(), *fixed).detach()
    li_index = fast_score.official.CONS_SCALARS.index('li')
    maximum = float(np.max(abs(target.numpy() - reference[:, li_index])))
    if maximum >= 2e-5:
        raise ValueError(f'Scorer contour quadrature mismatch: {maximum}')
    return (rt.float(), zt.float(), fixed, target, anchors), (contours, reference), {
        'scorer_target_max_difference': maximum,
        'reference_preparation_seconds': elapsed, 'reference_frames': len(truth)}


def terms(prediction, truth, variance, prepared):
    r, z, fixed, target, anchors = prepared
    field = prediction.reshape(-1, 65, 65)
    boundary, axis = shape_losses(field, *anchors)
    return {'pixel': (field - truth).square().sum((1, 2)).mean() / variance,
            'li': (inductance(field, r, z, *fixed) - target).square().mean(),
            'boundary': boundary, 'axis': axis}


def gradients(model, x, truth, variance, prepared):
    prediction, _ = model(x)
    losses = terms(prediction, truth, variance, prepared)
    vectors = {name: torch.cat([g.flatten() for g in torch.autograd.grad(
        loss, tuple(model.correction.parameters()), retain_graph=True)])
               for name, loss in losses.items()}
    norms = {name: float(v.norm()) for name, v in vectors.items()}
    cosines = {f'{a}:{b}': float(torch.dot(vectors[a], vectors[b]) /
                              (vectors[a].norm() * vectors[b].norm()))
               for a, b in [('pixel', 'li'), ('li', 'boundary'), ('li', 'axis')]}
    if not all(np.isfinite(v) for v in [*norms.values(), *cosines.values()]):
        raise FloatingPointError('Nonfinite correction gradient diagnostic')
    return {'norms': norms, 'cosines': cosines}


def control(parent, x, truth, prepared, reference, anchor_weight):
    model = SmoothCorrection(parent).train()
    variance = parent['coef_scale'].square().sum()
    initial, initial_scalars = model(x)
    gradient = gradients(model, x, truth, variance, prepared)
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
    records = []
    for step in range(6):
        prediction, scalars = model(x)
        loss = terms(prediction, truth, variance, prepared)
        objective = loss['pixel'] + 0.05 * loss['li'] + anchor_weight * (loss['boundary'] + loss['axis'])
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite prototype update')
        optimizer.step()
        records.append({'step': step + 1, **{k: float(v) for k, v in loss.items()}})
        assert torch.equal(scalars, initial_scalars)
    with torch.no_grad():
        updated, scalars = model(x)
        final = {k: float(v) for k, v in terms(updated, truth, variance, prepared).items()}
    assert torch.equal(scalars, initial_scalars)
    assert all(p.grad is None for p in model.parent.parameters())
    values, boundary, failed = derive_predictions(updated.numpy().reshape(-1, 65, 65),
                                                  reference[0], fast_score._geometry())
    return {'anchor_weight': anchor_weight, 'learning_rate': 4e-6, 'updates': records,
            'initial_gradient': gradient, 'final_terms': final,
            'scalar_outputs_exact': True, 'parent_gradients_absent': True,
            'derived_errors': {name: error(values[:, j], reference[1][:, j])
                               for j, name in enumerate(fast_score.official.CONS_SCALARS)},
            'normalized_boundary_mean': float(np.mean(boundary)), 'extraction_failures': failed,
            'physical_flux_change_rms': float((updated - initial).square().mean().sqrt())}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    root = Path('run_data/analysis-020')
    out = root / 'contour-anchor-probe.json'
    if out.exists():
        raise FileExistsError('Preserve the existing prototype receipt')
    probe = json.loads((root / 'batch-and-correction-probe.json').read_text())
    pack_path = root / 'training-target-pack.npz'
    if digest(pack_path) != probe['target_pack_sha256']:
        raise ValueError('Existing training pack changed')
    packed = np.load(pack_path)
    parent_path, sha = MODELS['parent']
    if digest(parent_path) != sha:
        raise ValueError('Frozen parent changed')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    x = prepare_inputs(packed['x'], parameters(parent, 'cpu'), parent['config'], 'cpu')
    prepared, reference, target_check = prepare(packed)
    truth = torch.from_numpy(packed['psi'])
    result = {'training_only': True, 'frames': len(truth), 'parent_sha256': sha,
              'target_pack_sha256': digest(pack_path), 'target_checks': target_check,
              'controls': {name: control(parent, x, truth, prepared, reference, weight)
                           for name, weight in [('aligned_li', 0), ('aligned_li_geometry', 0.1)]},
              'locked_shots_read': False, 'gpu_scheduled': False,
              'limitations': 'Six fixed-batch CPU updates on96 selected training frames; no CUDA, full-corpus preparation or held-out gain established.'}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
