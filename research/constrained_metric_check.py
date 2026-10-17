"""One fixed TRAIN predictor probe: seven actual metrics and radial projection."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from participant import fast_score
from participant.smooth_correction import SmoothCorrection
from research.constrained_metric_loss import project_parameters, seven_adjoint
from research.finite_metric_loss import cache_predictor, check_deadline, measured_metrics, predict
from research.finite_probe_gates import acceptance, frozen_state
from research.inductance_contour_audit import digest
from research.training_contour_sample import load


ROOT = Path('run_data/analysis-035')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    plan_path = ROOT / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    previous_path = Path('run_data/analysis-034/predictor-check.json')
    if digest(previous_path) != plan['previous034_result_sha256']:
        raise ValueError('Immutable034 evidence changed')
    old_plan_path = Path('run_data/analysis-034/predeclared-plan.json')
    if (digest(old_plan_path) != plan['previous034_plan_sha256'] or
            json.loads(old_plan_path.read_text())['acceptance_checks'] != plan['acceptance_checks']):
        raise ValueError('Original034 acceptance gates changed')
    previous = json.loads(previous_path.read_text())
    for path, sha in previous['source_sha256'].items():
        if digest(Path(path)) != sha:
            raise ValueError('Immutable034 source changed')
    (ROOT / 'running.lock').mkdir()
    output = ROOT / 'predictor-check.json'
    sources = [Path(__file__), Path('research/constrained_metric_loss.py'),
               Path('research/finite_probe_gates.py'), Path('research/finite_metric_loss.py')]
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest(plan_path), 'training_only': True, 'frames': plan['frames'],
        'optimizer_steps_per_arm': plan['steps_per_arm'], 'arms': {},
        'gpu_allocations': 0, 'locked_shots_read': False,
        'fields_or_learned_artifacts_saved': False,
        'source_sha256': {str(path): digest(path) for path in sources}}
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    deadline = started + plan['hard_local_cpu_wall_seconds']
    try:
        torch.set_num_threads(2)
        torch.manual_seed(plan['seed'])
        original = json.loads(Path('run_data/analysis-032/derivative-check.json').read_text())
        for path, sha in original['scorer_and_research_sha256'].items():
            if digest(Path(path)) != sha:
                raise ValueError('Immutable scorer or032 source changed')
        result['scorer_and_shared_source_sha256'] = original['scorer_and_research_sha256']
        sample = load()
        parent_sha = digest(Path('run_data/nn-008/mlp-shape/model.pt'))
        if (len(sample['inputs']) != plan['frames'] or parent_sha != previous['parent_sha256'] or
                sample['manifest_sha256'] != previous['training_manifest_sha256']):
            raise ValueError('Pinned whole TRAIN sample or parent changed')
        result.update(training_manifest_sha256=sample['manifest_sha256'], parent_sha256=parent_sha,
                      target_manifest_sha256=sample['target_manifest_sha256'])
        reference = sample['reference']
        targets = reference[1]
        variances = np.var(targets, axis=0)
        if not np.isfinite(targets).all() or np.any(variances <= 0):
            raise FloatingPointError('Undefined actual TRAIN reference normalization')
        result['actual_TRAIN_reference_variance'] = dict(zip(fast_score.official.CONS_SCALARS,
                                                            variances.tolist(), strict=True))
        pixel_variance = sample['parent']['coef_scale'].square().sum()
        bound = plan['acceptance_checks']['relative_correction_rms_to_parent_std_max']
        baseline = None
        for arm in plan['arms']:
            model = SmoothCorrection(sample['parent']).eval()
            features, base, scalars, parent_std = cache_predictor(model, sample['inputs'])
            if baseline is None:
                baseline, _ = measured_metrics(base, sample['truth'], base, parent_std, reference, deadline)
                result['parent'] = baseline
            generator = torch.Generator().manual_seed(plan['seed'])
            optimizer = torch.optim.AdamW(model.correction.parameters(), lr=plan['lr'],
                                           weight_decay=plan['weight_decay'])
            history = []
            result['arms'][arm] = {'complete': False, 'history': history}
            for step in range(plan['steps_per_arm']):
                check_deadline(deadline)
                optimizer.zero_grad(set_to_none=True)
                latent, fields = predict(model, features, base)
                pixel = (fields - sample['truth'].flatten(1)).square().sum(1).mean() / pixel_variance
                adjoint = torch.autograd.grad(pixel, latent, retain_graph=True)[0]
                measurement = {}
                if arm == 'bounded-actual-seven':
                    metric_adjoint, measurement = seven_adjoint(model, fields, parent_std, targets,
                        variances, generator, plan['finite_scale_relative_parent_std'], deadline)
                    adjoint = adjoint + plan['functional_weight'] * metric_adjoint / len(fields)
                if not torch.isfinite(adjoint).all() or not torch.isfinite(pixel):
                    raise FloatingPointError('Nonfinite constrained predictor adjoint')
                latent.backward(adjoint)
                norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), plan['clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite constrained predictor gradient')
                optimizer.step()
                final, projection = project_parameters(model, features, base, parent_std, bound)
                metrics, _ = measured_metrics(final, sample['truth'], base, parent_std, reference, deadline)
                history.append({'step': step + 1, 'pixel_objective_before': float(pixel.detach()),
                    'gradient_norm_before_clip': float(norm), 'measurement': measurement,
                    'projection': projection, 'actual_metrics_after': metrics})
                save(output, result)
                print(arm, step + 1, {'kappa_mse': metrics['derived']['kappa']['mse'],
                    'Z_axis_mse': metrics['derived']['Z_axis']['mse'],
                    'relative_rms': projection['relative_rms_after']}, flush=True)
            frozen_state(model, sample['parent'], sample['inputs'], final, scalars)
            result['arms'][arm].update(complete=True, final=metrics,
                frozen_parent_scalars_statistics_exact=True, cached_inference_exact=True)
            save(output, result)
        evaluation = acceptance(baseline, result['arms']['bounded-pixel-control']['final'],
                                result['arms']['bounded-actual-seven']['final'], plan['acceptance_checks'])
        result.update(complete=True, **evaluation,
            actual_cpu_elapsed_seconds=time.monotonic() - started,
            completed_at=datetime.now(timezone.utc).isoformat(), limitations=plan['limitations'])
        save(output, result)
        print({key: value for key, value in result.items() if key not in ('arms', 'parent')}, flush=True)
    except Exception as exc:
        result.update(failed=True, failure=f'{type(exc).__name__}: {exc}',
                      actual_cpu_elapsed_seconds=time.monotonic() - started,
                      failed_at=datetime.now(timezone.utc).isoformat())
        save(output, result)
        raise


if __name__ == '__main__':
    main()
