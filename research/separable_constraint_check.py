"""Fixed040 numerical comparison of the separate bilinear constraint route."""

from datetime import datetime, timezone
from hashlib import file_digest
import json
from pathlib import Path
import time

import numpy as np
import torch

from participant import correction_constraint as dense
from participant import correction_separable as separable
from participant.data import read_shot
from participant.finetune_common import prepare_inputs
from participant.flux_head import parameters
from participant.rich import INPUT_COLUMNS, rich_features
from participant.smooth_correction import SmoothCorrection
from research.finite_metric_loss import cache_predictor, predict
from research.finite_probe_gates import frozen_state


ROOT = Path('run_data/analysis-040')


def sha(path):
    with Path(path).open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def save(path, result):
    path.write_text(json.dumps(result, indent=2) + '\n')


def normalized_difference(first, second):
    return float(((first - second).abs() / torch.maximum(
        torch.ones_like(first), torch.maximum(first.abs(), second.abs()))).max())


def require_rejection(call):
    try:
        call()
    except (ValueError, FloatingPointError):
        return
    raise ValueError('Invalid correction request was accepted')


def timings(call, plan):
    for _ in range(plan['cost']['energy_warmup_calls_per_method']):
        call()
    values = []
    for _ in range(plan['cost']['energy_timed_calls_per_method']):
        tick = time.perf_counter()
        call()
        values.append(time.perf_counter() - tick)
    return {'seconds': values, 'median_seconds': float(np.median(values)),
            'local_CPU_only': True}


@torch.no_grad()
def main():
    plan_path = ROOT / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    (ROOT / 'running.lock').mkdir()
    output = ROOT / 'constraint-check.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(plan_path), 'optimizer_steps': 0, 'GPU_allocations': 0,
        'job_submissions': 0, 'scorer_extractions': 0, 'labels_or_true_fields_read': False,
        'fields_or_learned_artifacts_saved': False, 'numerical_fixture_only': True,
        'source_sha256': {name: sha(name) for name in (
            'participant/correction_separable.py', 'research/separable_constraint_check.py')},
        'limitations': plan['limitations']}
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    try:
        torch.set_num_threads(plan['torch_cpu_threads'])
        for key in ('parent', 'fixture', 'TRAIN_manifest', 'TRAIN_index', 'frozen037_source_manifest'):
            if sha(plan[key]) != plan[key + '_sha256']:
                raise ValueError(f'Pinned input changed: {key}')
        for name, expected in plan['immutable036_source_sha256'].items():
            if sha(name) != expected:
                raise ValueError('Immutable036 source changed')
        for line in Path(plan['frozen037_source_manifest']).read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            if sha(name) != expected:
                raise ValueError('Frozen037 source changed')
        manifest = json.loads(Path(plan['TRAIN_manifest']).read_text())
        index = json.loads(Path(plan['TRAIN_index']).read_text())
        if index['names'] != [Path(e['path']).stem for e in manifest['train']]:
            raise ValueError('TRAIN shot order changed')
        rows = []
        for entry in manifest['train']:
            if not (Path(plan['raw_TRAIN_cache']) / Path(entry['path']).name).exists():
                raise FileNotFoundError('Pinned raw TRAIN input is absent')
            rows.append(read_shot(entry, manifest['revision'], Path(plan['raw_TRAIN_cache']), INPUT_COLUMNS))
        counts = [len(row['efit_times']) for row in rows]
        if np.cumsum([0] + counts).tolist() != index['offsets'] or sum(counts) != plan['frames']:
            raise ValueError('TRAIN input offsets/count changed')
        parent = torch.load(plan['parent'], map_location='cpu', weights_only=False)
        model = SmoothCorrection(parent).eval()
        raw = np.concatenate([rich_features(row) for row in rows])
        inputs = prepare_inputs(raw, parameters(parent, 'cpu'), parent['config'], 'cpu')
        features, base, scalars, std = cache_predictor(model, inputs)
        fixture = torch.load(plan['fixture'], map_location='cpu', weights_only=False)
        model.correction.load_state_dict(fixture['state'])
        gram, factor = dense.interpolation_gram(model), separable.interpolation_factor(model)
        gates = plan['fixed_gates']
        kron = torch.kron(factor['one_dimensional_gram'], factor['one_dimensional_gram']) * factor['scale_square']
        gram_error = normalized_difference(gram, kron)
        if gram_error > gates['full_dense_vs_Kronecker_max_normalized_error']:
            raise ValueError('Full Gram differs from separable Kronecker identity')
        generator = torch.Generator().manual_seed(plan['seed'])
        checker = (torch.arange(17)[:, None] + torch.arange(17)[None, :]) % 2 * 2 - 1
        controls = torch.stack((torch.zeros(289), torch.ones(289), checker.flatten().float(),
                                torch.randn(289, generator=generator))).double()
        latent = model.correction(features)
        errors = [normalized_difference(dense.correction_rms(latent, gram),
                    separable.correction_rms(latent, factor))]
        for scale in (1e-5, 1., 1e5):
            errors.append(normalized_difference(dense.correction_rms(controls * scale, gram),
                          separable.correction_rms(controls * scale, factor)))
        if max(errors) > gates['RMS_vs_dense_max_normalized_error']:
            raise ValueError('Separable RMS differs from dense energy')
        result['energy'] = {'Gram_max_normalized_difference': gram_error,
            'RMS_max_normalized_difference': max(errors), 'actual_TRAIN_frames': len(features),
            'mathematical_control_rows': 12, 'all_values_finite': True}
        result['energy_timings'] = {'dense': timings(lambda: dense.correction_rms(latent, gram), plan),
            'separable': timings(lambda: separable.correction_rms(latent, factor), plan)}
        projections = {}
        final_fields = None
        for name, module, operator in (('dense', dense, gram), ('separable', separable, factor)):
            model.correction.load_state_dict(fixture['state'])
            tick = time.perf_counter()
            projection = module.project_correction(model, features, std, operator, plan['global_bound'])
            projection['local_CPU_seconds'] = time.perf_counter() - tick
            _, fields = predict(model, features, base)
            actual = float(((fields - base).square().mean(1).sqrt() / std).max())
            if actual > gates['projected_native_relative_RMS_max']:
                raise ValueError('Actual native correction exceeds original bound')
            projection['actual_native_relative_RMS_max'] = actual
            frozen_state(model, parent, inputs, fields, scalars)
            projections[name] = projection
            final_fields = fields
        factor_difference = abs(projections['dense']['factor'] - projections['separable']['factor'])
        maximum_difference = abs(projections['dense']['relative_rms_before'] -
                                 projections['separable']['relative_rms_before'])
        if (factor_difference > gates['global_factor_absolute_difference'] or
                maximum_difference > gates['global_maximum_absolute_difference']):
            raise ValueError('Separable global maximum or shared factor changed')
        invalid_calls = [lambda: separable.correction_rms(latent[:0], factor),
            lambda: separable.correction_rms(torch.full_like(latent[:1], float('nan')), factor),
            lambda: separable.maximum_relative_rms(model, features[:0], std[:0], factor),
            lambda: separable.maximum_relative_rms(model, features, torch.zeros_like(std), factor),
            lambda: separable.maximum_relative_rms(model, features, -std, factor),
            lambda: separable.maximum_relative_rms(model, features, torch.full_like(std, float('nan')), factor),
            lambda: separable.maximum_relative_rms(model, torch.full_like(features, float('nan')), std, factor),
            lambda: separable.project_correction(model, features, std, factor, 0.)]
        for call in invalid_calls:
            require_rejection(call)
        frozen_state(model, parent, inputs, final_fields, scalars)
        elapsed = time.monotonic() - started
        if elapsed > plan['cost']['hard_local_CPU_wall_seconds']:
            raise TimeoutError('Declared CPU wall limit exceeded')
        result.update(complete=True, all_fixed_gates_passed=True, projections=projections,
            shared_factor_absolute_difference=factor_difference, maximum_absolute_difference=maximum_difference,
            invalid_requests_rejected=len(invalid_calls), all79_frozen037_source_hashes_exact=True,
            frozen_parent_scalars_statistics_exact=True, actual_local_CPU_seconds=elapsed,
            completed_at=datetime.now(timezone.utc).isoformat(), torch_version=torch.__version__,
            numpy_version=np.__version__)
        save(output, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
            actual_local_CPU_seconds=time.monotonic() - started,
            failed_at=datetime.now(timezone.utc).isoformat())
        save(output, result)
        raise


if __name__ == '__main__':
    main()
