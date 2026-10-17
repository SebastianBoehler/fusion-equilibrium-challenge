"""Check monotonic geometry and actual contours on three complete training shots."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from participant import fast_score
from participant.contour_normal_loss import loss as old_loss, physical_points, targets
from participant.contour_training import KAPPA_WEIGHT, ScoredContourBatch
from participant.data import read_shot
from participant.finetune_common import prepare_inputs
from participant.flux_head import decode, parameters
from participant.inductance_loss import inductance
from participant.nn_train import Network
from participant.rich import INPUT_COLUMNS, rich_features
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.inductance_contour_audit import MODELS, derive_predictions, digest, error
from research.monotone_contour_proxy import displacement, loss, projected


ROOT = Path('run_data/analysis-028')


def mathematical_check():
    results = {}
    for dtype in (torch.float32, torch.float64):
        residual = torch.cat((-torch.logspace(8, -8, 1001, dtype=dtype),
                              torch.zeros(1, dtype=dtype),
                              torch.logspace(-8, 8, 1001, dtype=dtype)))[None].requires_grad_()
        gradient = torch.zeros(1, residual.shape[1], 2, dtype=dtype)
        gradient[..., 0] = 1
        movement = displacement(residual, gradient, torch.tensor(1, dtype=dtype))
        signed = -movement[..., 0]
        derivative = torch.autograd.grad(signed.sum(), residual)[0]
        if not (torch.isfinite(derivative).all() and torch.all(derivative > 0)
                and torch.all(signed.diff() >= 0) and torch.all(signed.abs() <= 1)
                and torch.all(residual.detach() * signed >= 0)):
            raise AssertionError('Monotonicity, direction, bound or finite positive derivative failed')
        results[str(dtype)] = {'range': [-1e8, 1e8], 'samples': residual.numel(),
                               'minimum_signed_derivative': float(derivative.min()),
                               'maximum_magnitude': float(signed.abs().max()),
                               'direction_monotonicity_positive_gradients_passed': True}
    return results


def measured(psi, reference):
    actual, boundary, failed = derive_predictions(psi.detach().numpy(), reference[0], fast_score._geometry())
    return {'extraction_failures': failed, 'boundary_mean': float(np.mean(boundary)),
            'derived_errors': {name: error(actual[:, j], reference[1][:, j])
                               for j, name in enumerate(fast_score.official.CONS_SCALARS)}}


def update_arm(parent, inputs, truth, normal, geometry, li, r, z, reference, name):
    model = SmoothCorrection(parent).train()
    with torch.no_grad():
        initial, scalar = model(inputs)
        assert torch.equal(initial, decode(model.parent(inputs), parameters(parent, 'cpu'), parent['config']))
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
    variance = parent['coef_scale'].square().sum()
    history = []
    for _ in range(6):
        psi, _ = model(inputs)
        psi = psi.reshape(-1, 65, 65)
        pixel = (psi - truth).square().sum((1, 2)).mean() / variance
        functional_li = (inductance(psi, r, z, *geometry) - li).square().mean()
        kappa = (loss if name == 'monotone-kappa' else old_loss)(psi, geometry[0], normal)
        weight = 0 if name == 'aligned-li' else KAPPA_WEIGHT
        value = pixel + .05 * functional_li + weight * kappa
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1)
        if not torch.isfinite(value) or not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite real training update')
        optimizer.step()
        history.append({'pixel': float(pixel.detach()), 'li': float(functional_li.detach()),
                        'kappa_proxy': float(kappa.detach()), 'gradient_norm': float(norm)})
    with torch.no_grad():
        final, final_scalar = model(inputs)
        assert torch.equal(final_scalar, scalar)
        folded = fold_checkpoint(parent, model)
        cfg = folded['config']
        exported = Network(cfg['inputs'], cfg['components'] + cfg['scalars'],
                           cfg['width'], cfg['depth'], cfg['dropout']).eval()
        exported.load_state_dict(folded['state'])
        out = exported(inputs)
        maximum = float((final - decode(out, parameters(folded, 'cpu'), cfg)).abs().max())
        assert maximum < 3e-6 and torch.equal(out[:, 4225:], scalar)
        assert all(p.grad is None for p in model.parent.parameters())
        assert torch.equal(folded['state']['head.1.weight'][4225:],
                           parent['state']['head.1.weight'][parent['config']['components']:])
    return {'six_updates': history, 'export_max_difference': maximum,
            'frozen_parent_scalars_zero_initial_equivalence_passed': True,
            'actual_extracted_contours': measured(final.reshape(-1, 65, 65), reference)}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    started = time.monotonic()
    if ROOT.exists():
        raise FileExistsError('Preserve existing or active monotonic repair diagnostic')
    paired = json.loads(Path('run_data/analysis-024/paired-decision.json').read_text())
    if not paired['complete'] or paired['candidate_promoted']:
        raise ValueError('Complete024 paired rejection required before successor preparation')
    checks = mathematical_check()
    old = Path('run_data/analysis-020/packing-check')
    manifest_path = old / 'training-only-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((old / 'train/index.json').read_text())
    assert index['names'] == [Path(e['path']).stem for e in manifest['train']]
    rows = [read_shot(e, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                      (*INPUT_COLUMNS, 'efit_psirz')) for e in manifest['train']]
    raw_inputs = np.concatenate([rich_features(row) for row in rows])
    fields = np.concatenate([np.asarray(row['efit_psirz'], np.float32) for row in rows])
    assert len(fields) == len(raw_inputs) == index['frames'] == 508
    target_root = Path('run_data/analysis-023/scored-targets')
    validation = json.loads(Path('run_data/analysis-023/target-validation.json').read_text())
    for name, sha in validation['file_sha256'].items():
        assert digest(target_root / name) == sha
    batch = ScoredContourBatch(target_root, index)
    assert len(batch.valid_indices) == 508
    p, w, interior, li = batch.get(batch.valid_indices, 'cpu')
    r, z = (torch.from_numpy(v) for v in (batch.r, batch.z))
    truth = torch.from_numpy(fields)
    reference = fast_score.official.build_reference(fields.astype(float), *fast_score._geometry())
    normal = targets(truth, r, z, p, physical_points(p, r, z), interior)
    path, sha = MODELS['parent']
    assert digest(path) == sha
    parent = torch.load(path, map_location='cpu', weights_only=False)
    inputs = prepare_inputs(raw_inputs, parameters(parent, 'cpu'), parent['config'], 'cpu')
    with torch.no_grad():
        base = SmoothCorrection(parent)(inputs)[0].reshape(-1, 65, 65)
        base_loss = loss(base, p, normal)
        invariance = {name: float(abs(loss(value, p, normal) - base_loss)) for name, value in
                      [('weak', base * 1e-12), ('sign', -base), ('gauge', base + 1), ('amplitude', base * 1.1)]}
        assert max(invariance.values()) < 2e-5 and float(loss(truth, p, normal)) < 1e-10
        maximum = float((projected(base, p, normal) - normal[2]).norm(dim=-1).max())
        assert maximum <= float(normal[-1]) + 1e-6
        try:
            loss(torch.zeros_like(base), p, normal)
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Undefined zero field was accepted')
    result = {'complete': False, 'training_only': True, 'frames': 508, 'shots': 3,
              'parent_sha256': sha, 'training_manifest_sha256': digest(manifest_path),
              'target_manifest_sha256': batch.manifest_sha, 'mathematical_checks': checks,
              'invariance_errors': invariance, 'max_real_movement_m': maximum,
              'bound_m': float(normal[-1]), 'zero_field_rejected': True,
              'parent_actual': measured(base, reference), 'arms': {},
              'new_gpu_scheduled': False, 'learned_artifact_saved': False, 'locked_shots_read': False}
    ROOT.mkdir()
    output = ROOT / 'monotone-check.json'
    for name in ('aligned-li', 'old-kappa', 'monotone-kappa'):
        result['arms'][name] = update_arm(parent, inputs, truth, normal, (p, w, interior),
                                         li, r, z, reference, name)
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(name, result['arms'][name]['actual_extracted_contours'], flush=True)
    result['complete'] = True
    result['elapsed_cpu_seconds'] = time.monotonic() - started
    result['source_sha256'] = {name: digest(Path(name)) for name in
                               ('research/monotone_contour_proxy.py', 'research/monotone_contour_check.py')}
    result['limitations'] = 'Three whole training shots only. Actual contours measure mechanism tradeoffs, not CUDA, generalization or official gains. No inference labels, field files or learned artifact.'
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
