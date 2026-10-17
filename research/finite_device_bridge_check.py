"""Fixed eight-frame CPU boundary check; no learning or candidate selection."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from participant.artifacts import digest, write_json
from participant.critical_artifact import restore_correction
from participant.data import array_value, read_shot
from participant.finite_parent_cache import cache_parent
from participant.finetune_common import prepare_inputs
from participant.flux_head import decode, parameters
from participant.nn_train import Network
from participant.rich import INPUT_COLUMNS, rich_features
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.finite_device_bridge import cpu_adjoint_to_device, reconstruct_cpu_fields
from research.finite_metric_loss import predict


ROOT = Path('run_data/analysis-050')


def reject(action):
    try:
        action()
    except (ValueError, FloatingPointError):
        return
    raise AssertionError('Invalid device-boundary request accepted')


def check_control(model, parent, raw, truth, inputs, limit):
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cache = cache_parent(model, raw, np.ones(len(raw), dtype=np.bool_))
    latent, native = predict(model, cache.features, cache.fields)
    bridge_latent = model.correction(cache.features)
    bridge = reconstruct_cpu_fields(bridge_latent, cache.fields, model.correction_scale)
    if not torch.equal(bridge, native.detach()) or bridge.requires_grad:
        raise ValueError('CPU reconstruction differs or retains a field graph')
    variance = parent['coef_scale'].square().sum()
    pixel = (native - truth).square().sum(1).mean() / variance
    adjoint = torch.autograd.grad(pixel, latent, retain_graph=True)[0].detach()
    parameters_to_check = tuple(model.correction.parameters())
    expected = torch.autograd.grad(pixel, parameters_to_check)
    bridge_latent.backward(cpu_adjoint_to_device(bridge_latent, adjoint))
    if not all(torch.equal(p.grad, g) and torch.isfinite(g).all()
               for p, g in zip(parameters_to_check, expected, strict=True)):
        raise ValueError('Returned adjoint changes the actual native parameter gradient')
    if not any(torch.any(g != 0) for g in expected):
        raise ValueError('Actual pixel gradient is zero; boundary check is inconclusive')
    with torch.no_grad():
        actual, scalars = model(inputs)
        if not torch.equal(actual, native) or not torch.equal(scalars, cache.scalars):
            raise ValueError('Native cached route differs from actual permitted-input inference')
        folded = fold_checkpoint(parent, model)
        cfg = folded['config']
        exported = Network(cfg['inputs'], cfg['components'] + cfg['scalars'],
                           cfg['width'], cfg['depth'], cfg['dropout']).eval()
        exported.load_state_dict(folded['state'])
        output = exported(inputs)
        decoded = decode(output, parameters(folded, 'cpu'), cfg)
        maximum = float((decoded - bridge).abs().max())
        if maximum > limit or not torch.equal(output[:, 4225:], scalars):
            raise ValueError('Original folded field/scalar gate failed')
    if (any(p.grad is not None for p in model.parent.parameters())
            or not all(torch.equal(model.state_dict()[k], v) for k, v in original.items())):
        raise ValueError('Backward changed frozen parameters/statistics')
    nan_latent, nan_parent, nan_adjoint = (v.detach().clone() for v in
                                         (latent, cache.fields, adjoint))
    nan_latent[0, 0] = nan_parent[0, 0] = nan_adjoint[0, 0] = float('nan')
    reconstruct = lambda v, p=cache.fields, s=model.correction_scale: reconstruct_cpu_fields(v, p, s)
    controls = [lambda: reconstruct(latent.double()), lambda: reconstruct(latent[:, :-1]),
                lambda: reconstruct(latent[:1].repeat(2049, 1)),
                lambda: reconstruct(nan_latent), lambda: reconstruct(latent, cache.fields.double()),
                lambda: reconstruct(latent, cache.fields[:, :-1]),
                lambda: reconstruct(latent, nan_parent),
                lambda: reconstruct(latent, cache.fields.detach().requires_grad_()),
                lambda: reconstruct(latent, s=model.correction_scale * 0),
                lambda: reconstruct(latent, s=model.correction_scale.double()),
                lambda: cpu_adjoint_to_device(latent.detach(), adjoint),
                lambda: cpu_adjoint_to_device(latent, adjoint.double()),
                lambda: cpu_adjoint_to_device(latent, adjoint[:, :-1]),
                lambda: cpu_adjoint_to_device(latent, nan_adjoint),
                lambda: cpu_adjoint_to_device(latent, adjoint.clone().requires_grad_())]
    for action in controls:
        reject(action)
    return {'reconstruction_bit_equal': True, 'parameter_gradients_bit_equal': True,
            'nonzero_finite_gradient': True, 'native_fields_scalars_exact': True,
            'folded_maximum_field_difference': maximum, 'folded_scalars_exact': True,
            'parameters_statistics_exact': True, 'rejected_controls': len(controls)}


def main():
    plan = json.loads((ROOT / 'predeclared-plan.json').read_text())
    (ROOT / 'running.lock').mkdir()
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': digest(ROOT / 'predeclared-plan.json'),
              'optimizer_steps': 0, 'GPU_count': 0, 'job_submissions': 0,
              'prediction_or_learned_artifacts_saved': False,
              'limitations': plan['limitations']}
    output = ROOT / 'boundary-check.json'
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        torch.set_num_threads(plan['CPU_threads'])
        for key, name in plan['paths'].items():
            if digest(Path(name)) != plan['input_sha256'][key]:
                raise ValueError('Pinned boundary-check input changed')
        for name, expected in plan['source_sha256'].items():
            if digest(Path(name)) != expected:
                raise ValueError('Pinned boundary source changed')
        manifest = json.loads(Path(plan['paths']['manifest']).read_text())
        index = json.loads(Path(plan['paths']['index']).read_text())
        if index['names'] != [Path(e['path']).stem for e in manifest['train']]:
            raise ValueError('TRAIN shot order differs')
        indices = np.asarray(plan['indices'])
        valid = np.load(plan['paths']['eligibility'], allow_pickle=False)
        if not np.array_equal(indices, np.arange(8)) or not valid[indices].all():
            raise ValueError('Original fixed frame eligibility differs')
        entry = manifest['train'][0]
        if Path(entry['path']).name != Path(plan['paths']['raw']).name:
            raise ValueError('Fixed first TRAIN shot differs')
        row = read_shot(entry, manifest['revision'], Path(plan['paths']['raw']).parent,
                        INPUT_COLUMNS)
        raw = rich_features(row)[indices]
        # Parquet loads this shot's label column; decode only the eight pinned frames.
        column = pq.read_table(plan['paths']['raw'], columns=['efit_psirz'])['efit_psirz'][0].values
        truth = torch.from_numpy(np.stack([array_value(column[int(i)]) for i in indices])
                                 .astype(np.float32)).flatten(1)
        if truth.shape != (8, 4225) or not torch.isfinite(truth).all():
            raise ValueError('Invalid actual TRAIN loss targets')
        parent = torch.load(plan['paths']['parent'], map_location='cpu', weights_only=False)
        inputs = prepare_inputs(raw, parameters(parent, 'cpu'), parent['config'], 'cpu')
        fixture = torch.load(plan['paths']['fixture'], map_location='cpu', weights_only=False)
        models = [SmoothCorrection(parent).eval(), restore_correction(parent, fixture)]
        result['controls'] = [check_control(m, parent, raw, truth, inputs,
                            plan['checks']['actual_folded_float32_fields_max_abs']) for m in models]
        for key, name in plan['paths'].items():
            if digest(Path(name)) != plan['input_sha256'][key]:
                raise ValueError('Pinned input changed during boundary check')
        for name, expected in plan['source_sha256'].items():
            if digest(Path(name)) != expected:
                raise ValueError('Frozen source changed during boundary check')
        elapsed = time.monotonic() - started
        if elapsed > plan['hard_local_wall_seconds']:
            raise TimeoutError('Predeclared boundary-check wall limit exceeded')
        result.update(complete=True, checks_passed=True, actual_body_seconds=elapsed,
                      frames=8, indices=plan['indices'], TRAIN_label_frames_decoded=8,
                      label_column_shot_frames=len(column), frozen_inputs_sources_exact=True,
                      CUDA_verified=False, finite_contour_oracle_verified=False)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
                      actual_body_seconds=time.monotonic() - started)
        raise
    finally:
        write_json(output, result)


if __name__ == '__main__':
    main()
