"""Validate the actual031 shared objective and loader on all508 training frames."""

from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import torch

from participant.critical_level import stencil
from participant.critical_artifact import save_correction, restore_correction
from participant.critical_loss import loss, live_level, projected
from participant.critical_training import ARMS, CriticalBatch, DESCRIPTION, objective
from participant.flux_head import decode, parameters
from participant.nn_train import Network
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.critical_level_stencil import projected as prototype_projected
from research.inductance_contour_audit import digest
from research.monotone_contour_check import measured
from research.training_contour_sample import load


ROOT = Path('run_data/analysis-031')


def check_arm(sample, batch, indices, weights, arm):
    parent, inputs = sample['parent'], sample['inputs']
    model = SmoothCorrection(parent).train()
    optimizer = torch.optim.AdamW(model.correction.parameters(), lr=4e-6, weight_decay=0)
    variance = parent['coef_scale'].square().sum()
    initial, scalar = model(inputs)
    assert torch.equal(initial, decode(model.parent(inputs), parameters(parent, 'cpu'), parent['config']))
    history = []
    geometry = batch.get(batch.valid_indices, 'cpu')
    for _ in range(6):
        value, pixel, li, kappa = objective(model, inputs, sample['truth'], sample['r'], sample['z'],
                                          geometry[:3], geometry[3], variance, indices, weights, arm)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), 1)
        assert torch.isfinite(value) and torch.isfinite(norm)
        optimizer.step()
        history.append({'pixel': float(pixel.detach()), 'li': float(li.detach()),
                        'kappa': float(kappa.detach()), 'gradient_norm': float(norm)})
    with torch.no_grad():
        final, final_scalar = model(inputs)
        assert torch.equal(final_scalar, scalar)
        folded = fold_checkpoint(parent, model)
        cfg = folded['config']
        exported = Network(cfg['inputs'], cfg['components'] + cfg['scalars'],
                           cfg['width'], cfg['depth'], cfg['dropout']).eval()
        exported.load_state_dict(folded['state'])
        out = exported(inputs)
        folded_field = decode(out, parameters(folded, 'cpu'), cfg)
        maximum = float((final - folded_field).abs().max())
        assert maximum < 3e-6 and torch.equal(out[:, 4225:], scalar)
        assert all(p.grad is None for p in model.parent.parameters())
        assert torch.equal(folded['state']['head.1.weight'][4225:],
                           parent['state']['head.1.weight'][parent['config']['components']:])
        buffer = BytesIO()
        save_correction(model, {'arm': arm}, buffer)
        buffer.seek(0)
        restored = restore_correction(parent, torch.load(buffer, weights_only=False))
        restored_field, restored_scalar = restored(inputs)
        assert torch.equal(restored_field, final) and torch.equal(restored_scalar, scalar)
    return {'six_updates': history, 'export_max_difference': maximum,
            'zero_initial_frozen_parent_scalar_outputs_weights_exact': True,
            'native_correction_serialization_restores_exact_fields_and_scalars': True,
            'folded_actual_extracted_contours': measured(folded_field.reshape(-1, 65, 65), sample['reference']),
            'actual_extracted_contours': measured(final.reshape(-1, 65, 65), sample['reference'])}


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    if ROOT.exists():
        raise FileExistsError('Preserve existing or active031 recipe check')
    sample = load()
    old = json.loads(Path('run_data/analysis-030/critical-check-repaired.json').read_text())
    assert old['complete'] and old['training_manifest_sha256'] == sample['manifest_sha256']
    verified = [stencil(field.numpy()) for field in sample['truth']]
    for a, b in zip(verified, old['stencils'], strict=True):
        assert a['indices'] == b['indices'] and a['weights'] == b['weights']
        assert a['level'] == b['level']
    ROOT.mkdir()
    target = ROOT / 'probe-targets'
    target.mkdir()
    arrays = {'indices': np.asarray([v['indices'] for v in verified], np.int16),
              'weights': np.asarray([v['weights'] for v in verified], np.float32),
              'valid': np.ones(508, np.bool_)}
    for name, value in arrays.items():
        np.save(target / f'{name}.npy', value)
    train_index = json.loads(Path('run_data/analysis-020/packing-check/train/index.json').read_text())
    metadata = train_index | {'complete': True, 'training_only': True, 'target': DESCRIPTION,
                              'manifest_sha256': sample['target_manifest_sha256'], 'usable_frames': 508}
    (target / 'index.json').write_text(json.dumps(metadata, indent=2) + '\n')
    batch = CriticalBatch(Path('run_data/analysis-023/scored-targets'), target, train_index)
    indices, weights = batch.critical_get(batch.valid_indices, 'cpu')
    ref_level = live_level(sample['truth'], indices, weights)
    with torch.no_grad():
        base = SmoothCorrection(sample['parent'])(sample['inputs'])[0].reshape(-1, 65, 65)
        p, _, _ = sample['geometry']
        normal = sample['normal']
        migrated = projected(base, p, normal, indices, weights, ref_level)
        previous = prototype_projected(base, p, normal, indices, weights, ref_level)
        maximum = float((migrated - previous).abs().max())
        assert maximum < 2e-7
        own_loss = loss(base, p, normal, indices, weights, ref_level)
        invariance = {name: float(abs(loss(value, p, normal, indices, weights, ref_level) - own_loss))
                      for name, value in [('weak', base * 1e-12), ('sign', -base),
                                          ('gauge', base + 1), ('amplitude', base * 1.1)]}
        assert max(invariance.values()) < 2e-5
        assert float(loss(sample['truth'], p, normal, indices, weights, ref_level)) < 1e-10
        movement = float((migrated - normal[2]).norm(dim=-1).max())
        assert movement <= float(normal[-1]) + 1e-6
        try:
            loss(torch.zeros_like(base), p, normal, indices, weights, ref_level)
        except FloatingPointError:
            pass
        else:
            raise AssertionError('Undefined zero field was accepted')
    result = {'complete': False, 'training_only': True, 'frames': 508, 'shots': 3,
              'training_manifest_sha256': sample['manifest_sha256'],
              'target_manifest_sha256': sample['target_manifest_sha256'],
              'active_stencil_migration_exact': True, 'loss_migration_max_difference': maximum,
              'invariance_errors': invariance, 'movement_bound_m': float(normal[-1]),
              'max_real_movement_m': movement, 'zero_field_rejected': True,
              'parent_actual': measured(base, sample['reference']), 'arms': {},
              'new_gpu_scheduled': False, 'learned_artifact_saved': False, 'locked_shots_read': False}
    output = ROOT / 'recipe-check.json'
    for arm in ARMS:
        result['arms'][arm] = check_arm(sample, batch, indices, weights, arm)
        output.write_text(json.dumps(result, indent=2) + '\n')
        a = result['arms'][arm]['actual_extracted_contours']
        print(arm, 'kappa', a['derived_errors']['kappa']['mse'],
              'li', a['derived_errors']['li']['mse'], 'boundary', a['boundary_mean'], flush=True)
    result['complete'] = True
    result['elapsed_cpu_seconds'] = time.monotonic() - started
    result['source_sha256'] = {name: digest(Path(name)) for name in
                               ('participant/critical_level.py', 'participant/critical_loss.py',
                                'participant/critical_training.py', 'research/critical_recipe_check.py')}
    result['limitations'] = 'Three training shots only; no CUDA, held-out or official gain. The live level uses the true active event as a local surrogate. Branch switches cause non-exact scorer derivatives;030 preserves those failures.'
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
