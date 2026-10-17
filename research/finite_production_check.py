"""Fixed036 TRAIN-only parallel-scoring, projection and export route check."""

import concurrent.futures
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from participant import fast_score
from participant.correction_constraint import correction_rms, interpolation_gram, project_correction
from participant.data import read_shot
from participant.finite_metrics import measure_derived
from participant.flux_head import decode, parameters
from participant.nn_train import Network
from participant.rich import INPUT_COLUMNS
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.finite_metric_loss import cache_predictor, check_deadline, predict
from research.finite_probe_gates import frozen_state
from research.inductance_contour_audit import digest, derive_predictions
from research.training_contour_sample import load


ROOT = Path('run_data/analysis-036')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def full_scores(sample, native, folded, scalar):
    manifest = json.loads(Path('run_data/analysis-020/packing-check/training-only-manifest.json').read_text())
    rows = [read_shot(entry, manifest['revision'], Path('run_data/analysis-019/training-raw'),
        (*INPUT_COLUMNS, 'efit_psirz', 'efit_q95', 'efit_beta_n')) for entry in manifest['train']]
    shots = [{'psi': np.asarray(row['efit_psirz'], np.float32).astype(float),
              'q95': np.asarray(row['efit_q95'], np.float32).astype(float),
              'betaN': np.asarray(row['efit_beta_n'], np.float32).astype(float)} for row in rows]
    if not np.array_equal(np.concatenate([s['psi'] for s in shots]), sample['truth'].numpy()):
        raise ValueError('Full scorer truth differs from pinned TRAIN input order')
    offsets = np.cumsum([0] + [len(s['psi']) for s in shots])
    refs = fast_score.references(shots, None, 12)
    scores = {}
    for key, fields in (('native', native), ('folded', folded)):
        # Reproduce float32 submission storage in memory only.
        memory = io.BytesIO()
        np.savez(memory, psirz=fields.numpy().reshape(-1, 65, 65), scalar=scalar.numpy())
        memory.seek(0)
        with np.load(memory, allow_pickle=False) as stored:
            if stored['psirz'].dtype != np.float32 or stored['scalar'].dtype != np.float32:
                raise ValueError('Fixture export did not retain float32 arrays')
            values = [{'psirz': stored['psirz'][a:b].astype(float),
                'q95': stored['scalar'][a:b, 0].astype(float),
                'betaN': stored['scalar'][a:b, 1].astype(float)}
                for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
        scores[key] = fast_score.composite(shots, refs, values, 12)
    return scores


def main():
    plan_path = ROOT / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    for key, path in (('previous035_plan_sha256', 'run_data/analysis-035/predeclared-plan.json'),
                      ('previous035_result_sha256', 'run_data/analysis-035/predictor-check.json'),
                      ('parent_sha256', 'run_data/nn-008/mlp-shape/model.pt'),
                      ('fixture_sha256', 'run_data/critical-031/critical-kappa/correction-state.pt')):
        if digest(Path(path)) != plan[key]:
            raise ValueError(f'Immutable evidence changed: {path}')
    previous = json.loads(Path('run_data/analysis-035/predictor-check.json').read_text())
    for path, sha in previous['source_sha256'].items():
        if digest(Path(path)) != sha:
            raise ValueError('Immutable035 source changed')
    (ROOT / 'running.lock').mkdir()
    output = ROOT / 'production-check.json'
    sources = [Path(__file__), Path('participant/finite_metrics.py'),
               Path('participant/correction_constraint.py')]
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest(plan_path), 'source_sha256': {str(p): digest(p) for p in sources},
        'training_only': True, 'optimizer_steps': 0, 'gpu_allocations': 0,
        'prediction_or_learned_artifacts_saved': False, 'locked_shots_read': False,
        'numerical_fixture_only': True, 'limitations': plan['limitations']}
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    deadline = started + plan['hard_local_cpu_wall_seconds']
    try:
        torch.set_num_threads(2)
        sample = load()
        result['training_manifest_sha256'] = sample['manifest_sha256']
        if len(sample['inputs']) != plan['frames']:
            raise ValueError('Pinned TRAIN sample count changed')
        model = SmoothCorrection(sample['parent']).eval()
        features, base, scalars, std = cache_predictor(model, sample['inputs'])
        payload = torch.load('run_data/critical-031/critical-kappa/correction-state.pt',
                             map_location='cpu', weights_only=False)
        model.correction.load_state_dict(payload['state'], strict=True)
        gram = interpolation_gram(model)
        latent, before = predict(model, features, base)
        physical = F.interpolate(latent.detach().reshape(-1, 1, 17, 17), size=(65, 65),
            mode='bilinear', align_corners=True).flatten(1) * model.correction_scale
        analytic = correction_rms(latent.detach(), gram)
        decoded = physical.double().square().mean(1).sqrt()
        native_rms = (before.detach() - base).double().square().mean(1).sqrt()
        discrepancy = float(torch.max(torch.stack(((analytic - decoded).abs() / std,
                                                  (analytic - native_rms).abs() / std))))
        if discrepancy > 1e-6:
            raise ValueError('Gram/native RMS discrepancy exceeds predeclared1e-6')
        projection_started = time.monotonic()
        projection = project_correction(model, features, std, gram, plan['global_projection_bound'])
        projection['actual_cpu_seconds'] = time.monotonic() - projection_started
        _, native = predict(model, features, base)
        native = native.detach()
        native_max = float(((native - base).square().mean(1).sqrt() / std).max())
        if native_max > plan['global_projection_bound']:
            raise ValueError('Gram projection failed actual native bound')
        result['constraint'] = projection | {'maximum_normalized_RMS_discrepancy': discrepancy,
                                             'actual_native_relative_rms_max': native_max}
        frozen_state(model, sample['parent'], sample['inputs'], native, scalars)
        memory = io.BytesIO()
        torch.save({'state': model.correction.state_dict()}, memory)
        memory.seek(0)
        restored = SmoothCorrection(sample['parent']).eval()
        restored.correction.load_state_dict(torch.load(memory, weights_only=True)['state'])
        with torch.no_grad():
            restored_fields, restored_scalars = restored(sample['inputs'])
        if not torch.equal(native, restored_fields) or not torch.equal(scalars, restored_scalars):
            raise ValueError('Native fixture restoration is not exact')
        generator = torch.Generator().manual_seed(plan['seed'])
        signs = torch.randint(0, 2, (508, 289), generator=generator).float() * 2 - 1
        step = plan['finite_scale_relative_parent_std'] * std / model.correction_scale
        perturbation = F.interpolate((signs * step[:, None]).reshape(-1, 1, 17, 17),
            size=(65, 65), mode='bilinear', align_corners=True).flatten(1) * model.correction_scale
        fixtures = {'native': native.numpy(), 'plus': (native + perturbation).numpy(),
                    'minus': (native - perturbation).numpy()}
        serial = {}
        result['timings'] = {}
        for key, fields in fixtures.items():
            check_deadline(deadline)
            tick = time.monotonic()
            serial[key] = measure_derived(fields)
            result['timings'][key] = {'serial_seconds': time.monotonic() - tick}
        pool_started = time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(max_workers=plan['cpu_workers']) as pool:
            for key, fields in fixtures.items():
                check_deadline(deadline)
                tick = time.monotonic()
                actual = measure_derived(fields, pool)
                elapsed = time.monotonic() - tick
                if not np.array_equal(actual, serial[key]):
                    raise ValueError('Parallel actual derived values differ from serial/order')
                result['timings'][key].update(parallel_seconds=elapsed, bit_equal=True)
            result['persistent_pool_including_startup_seconds'] = time.monotonic() - pool_started
        original, _, failures = derive_predictions(native.numpy().reshape(-1, 65, 65).astype(float),
                                                   [None] * 508, fast_score._geometry())
        if failures or not np.array_equal(original, serial['native']):
            raise ValueError('Production measurement differs from035 unchanged scorer route')
        ck = fold_checkpoint(sample['parent'], model)
        cfg = ck['config']
        folded_model = Network(cfg['inputs'], cfg['components'] + cfg['scalars'],
                               cfg['width'], cfg['depth'], cfg['dropout']).eval()
        folded_model.load_state_dict(ck['state'])
        with torch.no_grad():
            outputs = folded_model(sample['inputs'])
            folded = decode(outputs, parameters(ck, 'cpu'), cfg)
        if not torch.equal(scalars, outputs[:, cfg['components']:]):
            raise ValueError('Folded fixture scalar outputs changed')
        field_difference = float((folded - native).abs().max())
        if field_difference > 3e-6:
            raise ValueError('Folded fixture exceeds predeclared3e-6 field difference')
        folded_values = measure_derived(folded.numpy())
        scalar = scalars[:, :2] * model.s_std[:2] + model.s_mean[:2]
        scores = full_scores(sample, native, folded, scalar)
        check_deadline(deadline)
        result['exports'] = {'native_restore_exact': True, 'float32_memory_storage': True,
            'scalar_exact': True, 'max_field_difference': field_difference,
            'derived_max_difference': dict(zip(fast_score.official.CONS_SCALARS,
                np.max(abs(folded_values - serial['native']), axis=0).tolist(), strict=True)),
            'all_derived_valid': True, 'full_scorer': scores}
        optimistic_rate = max(508 / result['timings'][k]['parallel_seconds'] for k in ('plus', 'minus'))
        result['local_cost_only'] = {'warm_actual_derived_frames_per_second_optimistic': optimistic_rate,
            'one_full_usable_TRAIN_finite_pass_seconds_at_local_rate': 2 * 1469328 / optimistic_rate,
            'projection_CPU_sample_seconds': projection['actual_cpu_seconds'],
            'production_projection_support_frames': 1469328,
            'full_global_projections_per2048_batch_pass': 718,
            'setup_export_full_scoring_reserve_seconds': 900,
            'Galvani_feasibility_established': False, 'full_TRAIN_constraint_verified': False}
        result.update(complete=True, all_numerical_checks_passed=True,
            actual_cpu_elapsed_seconds=time.monotonic() - started,
            completed_at=datetime.now(timezone.utc).isoformat())
        save(output, result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
            actual_cpu_elapsed_seconds=time.monotonic() - started,
            failed_at=datetime.now(timezone.utc).isoformat())
        save(output, result)
        raise


if __name__ == '__main__':
    main()
