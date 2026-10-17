"""Read-only follow-up explaining the new032 derivative discrepancies."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from participant import fast_score
from participant.smooth_correction import SmoothCorrection
from research.inductance_contour_audit import digest
from research.predicted_contour_branch import contour, kappa, select, vertices
from research.predicted_contour_check import normalized_error, summary
from research.training_contour_sample import load
from contour import grid_to_physical, resample_closed
from lcfs import _best_contour_at
from o_point import find_o_point


ROOT = Path('run_data/analysis-032')


def event_contour(field, base):
    """Diagnostic only: retain the base event and offset, re-extract grid edges."""
    r, z, coarse, mask = fast_score._geometry()
    sign = fast_score.official.AXIS_SIGN['DIII-D']
    _, _, iz, ir, _ = find_o_point(field * sign, r, z, coarse)
    level = field.ravel()[base.event_indices] @ base.event_weights + base.event_bias
    points = _best_contour_at(field * sign, level * sign, iz, ir, mask)
    if points is None:
        return None
    return resample_closed(grid_to_physical(points, r, z), fast_score.official.N_POINTS)


def extremes(points):
    return [int(points[:, j].argmin()) for j in (0, 1)] + [
        int(points[:, j].argmax()) for j in (0, 1)]


def main():
    old = json.loads((ROOT / 'derivative-check.json').read_text())
    if not old['complete']:
        raise ValueError('Original032 check must finish before diagnosing its discrepancies')
    for path, sha in old['scorer_and_research_sha256'].items():
        if digest(Path(path)) != sha:
            raise ValueError('Preserve the original032 code used in its receipt')
    (ROOT / 'tail-check.lock').mkdir()
    output = ROOT / 'tail-check.json'
    with output.open('x') as handle:
        json.dump({'complete': False, 'training_only': True}, handle)
    started = time.monotonic()
    torch.set_num_threads(2)
    sample = load()
    if sample['manifest_sha256'] != old['training_manifest_sha256']:
        raise ValueError('Training sample changed')
    with torch.no_grad():
        fields, _ = SmoothCorrection(sample['parent']).eval()(sample['inputs'])
    fields = fields.reshape(-1, 65, 65).double()
    torch.manual_seed(20261032)
    from torch.nn import functional as F
    pattern = F.interpolate(torch.randn(1, 1, 17, 17, dtype=torch.float64),
                            size=(65, 65), mode='bilinear', align_corners=True)[0, 0]
    pattern -= pattern.mean()
    pattern /= pattern.square().mean().sqrt()
    direction = pattern[None] * fields.std(dim=(1, 2), unbiased=False)[:, None, None]
    report = old['modes']['event']['actual_finite_differences']['0.0001']
    ad, actual_fd = (np.array(report[key]) for key in (
        'autodiff_directional_derivatives', 'actual_directional_derivatives'))
    errors = normalized_error(ad, actual_fd)
    selected = np.flatnonzero(errors > .1)
    details, own = [], {}
    for frame in range(508):
        base = select(fields[frame].numpy())
        field, delta = fields[frame], direction[frame]
        for mode in ('bisection', 'event'):
            derivative = old['modes'][mode]['actual_finite_differences']['0.0001'][
                'autodiff_directional_derivatives'][frame]
            values = []
            for step in (1e-7, 1e-8, 1e-9):
                plus, minus = (contour(field + sign * step * delta, base, mode)
                               for sign in (1, -1))
                fd = float((kappa(plus) - kappa(minus)) / (2 * step))
                values.append({'step': step, 'normalized_error': float(normalized_error(
                    np.asarray(derivative), np.asarray(fd))),
                    'resampled_extrema_changed': extremes(plus) != extremes(minus)})
            if values[0]['normalized_error'] >= 1e-4:
                own[f'{mode}/{frame}'] = values
        if frame not in selected:
            continue
        step = 1e-4
        plus, minus = ((field + sign * step * delta).numpy() for sign in (1, -1))
        bp, bm = select(plus), select(minus)
        same_event = all(np.array_equal(b.event_indices, base.event_indices) and
                         np.array_equal(b.event_weights, base.event_weights) for b in (bp, bm))
        cp, cm = event_contour(plus, base), event_contour(minus, base)
        fixed_event_fd = (float((kappa(torch.from_numpy(cp)) - kappa(torch.from_numpy(cm))) /
                                (2 * step)) if cp is not None and cm is not None else None)
        own_fd = float((kappa(contour(torch.from_numpy(plus), base, 'event')) -
                        kappa(contour(torch.from_numpy(minus), base, 'event'))) / (2 * step))
        event_level_ad = float(delta.flatten()[torch.as_tensor(base.event_indices)] @
                               torch.as_tensor(base.event_weights))
        entry = {'frame': int(frame), 'actual_normalized_derivative_error': float(errors[frame]),
            'selected_event_unchanged': same_event,
            'grid_edges_unchanged': all(np.array_equal(b.edges, base.edges) for b in (bp, bm)),
            'bisection_decisions_unchanged': all(b.decisions == base.decisions for b in (bp, bm)),
            'actual_level_directional_fd': float((bp.level - bm.level) / (2 * step)),
            'event_level_directional_derivative': event_level_ad,
            'actual_kappa_directional_fd': float(actual_fd[frame]),
            'event_kappa_directional_derivative': float(ad[frame]),
            'fixed_grid_and_event_directional_fd': own_fd,
            'reextracted_grid_fixed_event_directional_fd': fixed_event_fd,
            'fixed_event_counterfactual_extraction_failures': int(cp is None) + int(cm is None)}
        details.append(entry)
    same = [v for v in details if v['selected_event_unchanged'] and
            v['reextracted_grid_fixed_event_directional_fd'] is not None]
    counterfactual_errors = [float(normalized_error(np.asarray(v['event_kappa_directional_derivative']),
                            np.asarray(v['reextracted_grid_fixed_event_directional_fd']))) for v in same]
    result = {'complete': True, 'training_only': True, 'frames_inspected': 508,
        'tail_frames_at_original_step': len(details), 'details': details,
        'tail_selected_event_changed_frames': sum(not v['selected_event_unchanged'] for v in details),
        'tail_grid_edges_changed_frames': sum(not v['grid_edges_unchanged'] for v in details),
        'tail_bisection_decisions_changed_frames': sum(not v['bisection_decisions_unchanged'] for v in details),
        'own_finite_difference_offenders': own,
        'same_event_tail_reextracted_fixed_event_fd_error': summary(counterfactual_errors) if same else None,
        'same_event_tail_counterfactual_valid_frames': len(same),
        'original_receipt_sha256': digest(ROOT / 'derivative-check.json'),
        'source_sha256': digest(Path(__file__)), 'training_manifest_sha256': sample['manifest_sha256'],
        'optimizer_steps': 0, 'gpu_allocations': 0, 'fields_or_learned_artifacts_saved': False,
        'actual_cpu_elapsed_seconds': time.monotonic() - started,
        'limitations': 'Counterfactual retained event/offset is a diagnostic, not the actual scorer '
        'or an inference repair. Original predeclared gates stay failed; no training run justified.'}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print({key: value for key, value in result.items() if key not in ('details',)}, flush=True)


if __name__ == '__main__':
    main()
