"""Validate active critical-level derivatives and a six-update geometry repair."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from participant.contour_training import KAPPA_WEIGHT
from participant.flux_head import decode, parameters
from participant.inductance_loss import inductance
from participant.nn_train import Network
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.contour_level_audit import critical_trace
from research.critical_level_stencil import live_level, loss, projected, stencil
from research.inductance_contour_audit import digest
from research.monotone_contour_check import measured, update_arm
from research.training_contour_sample import load


ROOT = Path('run_data/analysis-030')


def gradient_check(fields, stencils):
    responses = []
    yy, xx = np.meshgrid(np.linspace(-1, 1, 65), np.linspace(-1, 1, 65), indexing='ij')
    smooth = .5 * (xx + yy)
    random = torch.Generator().manual_seed(20261030)
    small = torch.randn((3, 1, 17, 17), generator=random, dtype=torch.float64)
    heads = torch.nn.functional.interpolate(small, (65, 65), mode='bilinear', align_corners=True)[:, 0].numpy()
    heads /= abs(heads).max(axis=(1, 2), keepdims=True)
    for field, s in zip(fields, stencils, strict=True):
        eps = float(np.ptp(field)) / 2 ** 16
        poke = np.zeros(4225)
        poke[s['indices']] = 1
        frame = []
        for direction in (poke.reshape(65, 65), smooth, *heads):
            plus, _ = critical_trace(field.astype(float) + eps * direction)
            minus, _ = critical_trace(field.astype(float) - eps * direction)
            numerical = float((plus - minus) / (2 * eps))
            predicted = float(sum(direction.flat[i] * w for i, w in zip(s['indices'], s['weights'], strict=True)))
            frame.append({'numerical': numerical, 'stencil': predicted,
                          'absolute_difference': abs(numerical - predicted)})
        responses.append(frame)
    maximum = max(v['absolute_difference'] for frame in responses for v in frame)
    smooth_max = max(v['absolute_difference'] for frame in responses for v in frame[1:])
    return {'responses': responses, 'maximum_absolute_difference': maximum,
            'smooth_head_maximum_difference': smooth_max, 'tolerance': .04,
            'all_smooth_directions_passed': smooth_max <= .04,
            'all_isolated_node_perturbations_passed': all(frame[0]['absolute_difference'] <= .04 for frame in responses)}


def train(sample, indices, weights, reference_level):
    parent, inputs, truth = (sample[k] for k in ('parent', 'inputs', 'truth'))
    normal, geometry = sample['normal'], sample['geometry']
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
        li = (inductance(psi, sample['r'], sample['z'], *geometry) - sample['li']).square().mean()
        kappa = loss(psi, geometry[0], normal, indices, weights, reference_level)
        value = pixel + .05 * li + KAPPA_WEIGHT * kappa
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
        maximum = float((final - decode(out, parameters(folded, 'cpu'), cfg)).abs().max())
        assert maximum < 3e-6 and torch.equal(out[:, 4225:], scalar)
        assert all(p.grad is None for p in model.parent.parameters())
        assert torch.equal(folded['state']['head.1.weight'][4225:],
                           parent['state']['head.1.weight'][parent['config']['components']:])
    return {'six_updates': history, 'export_max_difference': maximum,
            'zero_initial_frozen_parent_scalars_export_passed': True,
            'actual_extracted_contours': measured(final.reshape(-1, 65, 65), sample['reference'])}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    started = time.monotonic()
    output = ROOT / 'critical-check-repaired.json'
    if output.exists():
        raise FileExistsError('Preserve existing or active critical repair check')
    sample = load()
    stencils = [stencil(field.numpy()) for field in sample['truth']]
    indices = torch.tensor([s['indices'] for s in stencils], dtype=torch.long)
    weights = torch.tensor([s['weights'] for s in stencils], dtype=torch.float32)
    target_level = live_level(sample['truth'], indices, weights)
    counts = {kind: sum(s['kind'] == kind for s in stencils)
              for kind in ('topology_node', 'envelope_contact')}
    responses = gradient_check(sample['truth'].numpy(), stencils)
    result = {'complete': False, 'training_only': True, 'frames': 508, 'shots': 3,
              'supported_counts': counts, 'stencils': stencils, 'critical_level_derivative': responses,
              'training_manifest_sha256': sample['manifest_sha256'],
              'target_manifest_sha256': sample['target_manifest_sha256'],
              'new_gpu_scheduled': False, 'learned_artifact_saved': False, 'locked_shots_read': False}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('critical derivative max difference', responses['maximum_absolute_difference'], flush=True)
    print('smooth correction direction max difference', responses['smooth_head_maximum_difference'], flush=True)
    with torch.no_grad():
        base = SmoothCorrection(sample['parent'])(sample['inputs'])[0].reshape(-1, 65, 65)
        p, _, _ = sample['geometry']
        normal = sample['normal']
        base_loss = loss(base, p, normal, indices, weights, target_level)
        invariance = {name: float(abs(loss(value, p, normal, indices, weights, target_level) - base_loss))
                      for name, value in [('weak', base * 1e-12), ('sign', -base),
                                          ('gauge', base + 1), ('amplitude', base * 1.1)]}
        assert max(invariance.values()) < 2e-5
        assert float(loss(sample['truth'], p, normal, indices, weights, target_level)) < 1e-10
        maximum = float((projected(base, p, normal, indices, weights, target_level) - normal[2]).norm(dim=-1).max())
        assert maximum <= float(normal[-1]) + 1e-6
    # Verify this differentiable surrogate's complete backward independently
    # with a central finite difference. This does not claim an exact scorer derivative.
    probe = base.double().requires_grad_()
    prepared = tuple(value.double() for value in sample['normal'])
    direction = torch.sin(torch.arange(probe.numel(), dtype=torch.float64).reshape(probe.shape) / 431)
    value = loss(probe, p.double(), prepared, indices, weights.double(), target_level.double())
    analytic = float((torch.autograd.grad(value, probe)[0] * direction).sum())
    eps = 1e-6
    numerical = float((loss(probe.detach() + eps * direction, p.double(), prepared, indices, weights.double(), target_level.double())
                       - loss(probe.detach() - eps * direction, p.double(), prepared, indices, weights.double(), target_level.double())) / (2 * eps))
    assert abs(analytic - numerical) <= 1e-5 * max(1, abs(analytic))
    result['surrogate_backward_finite_difference'] = {'analytic': analytic, 'numerical': numerical,
                                                      'absolute_difference': abs(analytic - numerical), 'passed': True}
    result.update({'invariance_errors': invariance, 'movement_bound_m': float(normal[-1]),
                   'max_real_movement_m': maximum, 'parent_actual': measured(base, sample['reference'])})
    control = update_arm(sample['parent'], sample['inputs'], sample['truth'], sample['normal'],
                         sample['geometry'], sample['li'], sample['r'], sample['z'],
                         sample['reference'], 'aligned-li')
    result['matched_li_control'] = control
    result['critical_kappa'] = train(sample, indices, weights, target_level)
    result['complete'] = True
    result['elapsed_cpu_seconds'] = time.monotonic() - started
    result['source_sha256'] = {name: digest(Path(name)) for name in
                               ('research/critical_level_stencil.py', 'research/critical_contour_check.py',
                                'research/training_contour_sample.py')}
    result['limitations'] = 'Three whole training shots only. True critical events define a local surrogate, not exact extraction at every future prediction. No CUDA/generalization or official gain, no inference labels or saved learned artifact.'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for name in ('parent_actual', 'matched_li_control', 'critical_kappa'):
        actual = result[name] if name == 'parent_actual' else result[name]['actual_extracted_contours']
        print(name, 'kappa', actual['derived_errors']['kappa']['mse'],
              'li', actual['derived_errors']['li']['mse'], 'boundary', actual['boundary_mean'], flush=True)


if __name__ == '__main__':
    main()
