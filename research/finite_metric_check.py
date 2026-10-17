"""One predeclared CPU predictor-update probe using actual extracted elongation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from participant import fast_score
from participant.smooth_correction import SmoothCorrection
from research.finite_metric_loss import (
    cache_predictor, check_deadline, finite_adjoint, measured_metrics, predict)
from research.inductance_contour_audit import digest
from research.training_contour_sample import load


ROOT = Path('run_data/analysis-034')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    plan_path = ROOT / 'predeclared-plan.json'
    plan = json.loads(plan_path.read_text())
    if digest(Path('run_data/analysis-033/sampling-audit.json')) != plan['previous033_receipt_sha256']:
        raise ValueError('Immutable033 evidence changed')
    (ROOT / 'running.lock').mkdir()
    output = ROOT / 'predictor-check.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': digest(plan_path), 'training_only': True, 'frames': plan['frames'],
        'optimizer_steps_per_arm': plan['steps_per_arm'], 'arms': {},
        'gpu_allocations': 0, 'locked_shots_read': False,
        'fields_or_learned_artifacts_saved': False,
        'source_sha256': {str(path): digest(path) for path in
                          (Path(__file__), Path('research/finite_metric_loss.py'))}}
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    deadline = started + plan['hard_local_cpu_wall_seconds']
    try:
        torch.set_num_threads(2)
        torch.manual_seed(plan['seed'])
        sample = load()
        if len(sample['inputs']) != plan['frames']:
            raise ValueError('Predeclared whole TRAIN sample count changed')
        previous = json.loads(Path('run_data/analysis-033/sampling-audit.json').read_text())
        parent_sha = digest(Path('run_data/nn-008/mlp-shape/model.pt'))
        if (sample['manifest_sha256'] != previous['training_manifest_sha256'] or
                parent_sha != previous['parent_sha256']):
            raise ValueError('Pinned TRAIN sample or parent changed')
        result.update(training_manifest_sha256=sample['manifest_sha256'],
            parent_sha256=parent_sha,
            target_manifest_sha256=sample['target_manifest_sha256'])
        reference = sample['reference']
        targets = reference[1][:, fast_score.official.CONS_SCALARS.index('kappa')]
        variance = sample['parent']['coef_scale'].square().sum()
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
                pixel = (fields - sample['truth'].flatten(1)).square().sum(1).mean() / variance
                adjoint = torch.autograd.grad(pixel, latent, retain_graph=True)[0]
                measurement = {}
                if arm == 'actual-kappa-finite':
                    metric_adjoint, measurement = finite_adjoint(model, fields, parent_std, targets,
                        generator, plan['finite_scale_relative_parent_std'], deadline)
                    adjoint = adjoint + plan['kappa_weight'] * metric_adjoint / len(fields)
                if not torch.isfinite(adjoint).all() or not torch.isfinite(pixel):
                    raise FloatingPointError('Nonfinite predictor update adjoint')
                latent.backward(adjoint)
                norm = torch.nn.utils.clip_grad_norm_(model.correction.parameters(), plan['clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite predictor parameter gradient')
                optimizer.step()
                with torch.no_grad():
                    _, final = predict(model, features, base)
                    metrics, _ = measured_metrics(final, sample['truth'], base, parent_std, reference, deadline)
                history.append({'step': step + 1, 'pixel_objective_before': float(pixel.detach()),
                    'gradient_norm_before_clip': float(norm), 'measurement': measurement,
                    'actual_metrics_after': metrics})
                save(output, result)
                print(arm, step + 1, {'kappa_mse': metrics['derived']['kappa']['mse'],
                    'boundary_mean': metrics['boundary_mean'], 'gradient_norm': float(norm)}, flush=True)
            with torch.no_grad():
                actual, actual_scalars = model(sample['inputs'])
            if not torch.equal(final, actual) or not torch.equal(scalars, actual_scalars):
                raise ValueError('Cached updated predictor differs from actual permitted-input inference')
            if any(p.grad is not None for p in model.parent.parameters()):
                raise ValueError('Frozen parent received a gradient')
            for key, value in sample['parent']['state'].items():
                if not torch.equal(model.parent.state_dict()[key], value):
                    raise ValueError('Frozen parent parameter changed')
            for key in ('x_mean', 'x_std', 'psi_mean', 'coef_scale', 'basis', 's_mean', 's_std'):
                if not torch.equal(getattr(model, key).cpu(), sample['parent'][key]):
                    raise ValueError('Frozen decoding statistics changed')
            result['arms'][arm].update(complete=True, final=metrics,
                frozen_parent_scalars_statistics_exact=True, cached_inference_exact=True)
            save(output, result)
        candidate = result['arms']['actual-kappa-finite']['final']
        control = result['arms']['pixel-control']['final']
        limits = plan['acceptance_checks']
        ratios = {key: candidate[key] / baseline[key] for key in
                  ('pixel_sse', 'boundary_mean', 'mean_training_standardized_derived_mse')}
        component_ratios = {key: value['mse'] / baseline['derived'][key]['mse']
                            for key, value in candidate['derived'].items()}
        control_ratio = candidate['derived']['kappa']['mse'] / control['derived']['kappa']['mse']
        checks = {'kappa_vs_parent': component_ratios['kappa'] <= limits['actual_kappa_mse_ratio_to_parent_max'],
            'kappa_vs_pixel_control': control_ratio <= limits['actual_kappa_mse_ratio_to_pixel_control_max'],
            'pixel': ratios['pixel_sse'] <= limits['pixel_sse_ratio_to_parent_max'],
            'boundary': ratios['boundary_mean'] <= limits['boundary_mean_ratio_to_parent_max'],
            'all_seven_components': max(component_ratios.values()) <= limits['each_of_7_derived_mse_ratio_to_parent_max'],
            'mean_standardized_derived': ratios['mean_training_standardized_derived_mse'] <=
                limits['mean_training_standardized_derived_mse_ratio_to_parent_max'],
            'relative_correction_bound': candidate['relative_correction_rms_max'] <=
                limits['relative_correction_rms_to_parent_std_max']}
        result.update(complete=True, checks=checks, acceptance_passed=all(checks.values()),
            ratios_to_parent=ratios, derived_mse_ratios_to_parent=component_ratios,
            kappa_mse_ratio_to_pixel_control=control_ratio,
            actual_cpu_elapsed_seconds=time.monotonic() - started,
            completed_at=datetime.now(timezone.utc).isoformat(),
            limitations=plan['limitations'])
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
