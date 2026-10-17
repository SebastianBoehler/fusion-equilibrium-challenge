"""Separate critical-level response from the scorer's finite contour sampling phase."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from participant import fast_score
from participant.smooth_correction import SmoothCorrection
from research.inductance_contour_audit import digest
from research.predicted_contour_branch import kappa, select, vertices
from research.predicted_contour_check import normalized_error, summary
from research.training_contour_sample import load
from contour import resample_closed


ROOT = Path('run_data/analysis-033')
OLD = Path('run_data/analysis-032')


def numpy_kappa(points):
    return float((points[:, 1].max() - points[:, 1].min()) /
                 (points[:, 0].max() - points[:, 0].min()))


def rotate(points, count):
    if not np.array_equal(points[0], points[-1]):
        raise ValueError('Expected an explicitly closed scorer polygon')
    unique = points[:-1]
    shifted = np.roll(unique, -count, axis=0)
    return np.concatenate([shifted, shifted[:1]])


def fd_summary(ad, fd):
    return {'normalized_error': summary(normalized_error(ad, fd)),
            'sign_agreement': float(np.mean(np.sign(ad) == np.sign(fd)))}


def main():
    original_path = OLD / 'derivative-check.json'
    original = json.loads(original_path.read_text())
    previous = json.loads((OLD / 'tail-check.json').read_text())
    if not original['complete'] or not previous['complete']:
        raise ValueError('Complete032 diagnostics required')
    for path, sha in original['scorer_and_research_sha256'].items():
        if digest(Path(path)) != sha:
            raise ValueError('Original032 source changed')
    ROOT.mkdir()
    (ROOT / 'sampling-audit.lock').mkdir()
    out = ROOT / 'sampling-audit.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
        'working_assumption': 'A changed first polygon vertex shifts the512 arc-length '
        'samples even when the physical polygon is identical; this may explain some '
        'finite elongation responses separately from critical-level quantization.',
        'success_check': 'Measure unchanged-polygon raw extrema and resampled extrema '
        'under cyclic vertex reordering, then compare raw-polygon directional derivatives '
        'with the original512-sample discrepancies on every508 TRAINframe.',
        'training_only': True, 'shots': 3, 'frames': 508, 'optimizer_steps': 0,
        'gpu_allocations': 0, 'locked_shots_read': False, 'new_gpu_dispatched': False,
        'fields_or_learned_artifacts_saved': False,
        'original032_receipt_sha256': digest(original_path),
        'limitations': 'Cyclic vertex reorderings are offline diagnostic counterfactuals. '
        'The official extractor is unchanged; no inference correction, metric repair, '
        'gradient eligibility or model improvement follows automatically.'}
    with out.open('x') as handle:
        json.dump(result, handle)
    def save():
        out.write_text(json.dumps(result, indent=2) + '\n')
    started = time.monotonic()
    torch.set_num_threads(2)
    sample = load()
    if sample['manifest_sha256'] != original['training_manifest_sha256']:
        raise ValueError('Pinned TRAIN sample changed')
    result['training_manifest_sha256'] = sample['manifest_sha256']
    result['parent_sha256'] = digest(Path('run_data/nn-008/mlp-shape/model.pt'))
    if result['parent_sha256'] != original['parent_sha256']:
        raise ValueError('Parent checkpoint changed')
    with torch.no_grad():
        fields, _ = SmoothCorrection(sample['parent']).eval()(sample['inputs'])
    fields = fields.reshape(-1, 65, 65).double()
    torch.manual_seed(20261032)
    pattern = F.interpolate(torch.randn(1, 1, 17, 17, dtype=torch.float64),
                            size=(65, 65), mode='bilinear', align_corners=True)[0, 0]
    pattern -= pattern.mean()
    pattern /= pattern.square().mean().sqrt()
    directions = pattern[None] * fields.std(dim=(1, 2), unbiased=False)[:, None, None]
    epsilon = 1e-4
    old = original['modes']['event']['actual_finite_differences']['0.0001']
    old_ad = np.array(old['autodiff_directional_derivatives'])
    old_fd = np.array(old['actual_directional_derivatives'])
    rows, raw_ad, raw_fd, resampled_fd = [], [], [], []
    for frame, (field, direction) in enumerate(zip(fields, directions, strict=True)):
        base = select(field.numpy())
        leaf = field.clone().requires_grad_()
        raw = vertices(leaf, base, 'event')
        value = kappa(raw)
        value.backward()
        derivative = float((leaf.grad * direction).sum())
        polygon = raw.detach().numpy()
        reorder_values, raw_values = [], []
        for count in (0, 1, 2, (len(polygon) - 1) // 2):
            reordered = rotate(polygon, count)
            raw_values.append(numpy_kappa(reordered))
            reorder_values.append(numpy_kappa(resample_closed(
                reordered, fast_score.official.N_POINTS)))
        if np.ptp(raw_values) != 0:
            raise ValueError('Cyclic reordering changed raw geometric extrema')
        if abs(reorder_values[0] - numpy_kappa(base.actual)) > 1e-10:
            raise ValueError('Base arc-length samples differ from actual032 route')
        shifted_fields = [field + sign * epsilon * direction for sign in (1, -1)]
        shifted = [select(values.numpy()) for values in shifted_fields]
        raw_plus, raw_minus = [numpy_kappa(vertices(values, branch, 'event').numpy())
                              for values, branch in zip(shifted_fields, shifted, strict=True)]
        sampled_plus, sampled_minus = [numpy_kappa(branch.actual) for branch in shifted]
        fd = (raw_plus - raw_minus) / (2 * epsilon)
        sfd = (sampled_plus - sampled_minus) / (2 * epsilon)
        if abs(sfd - old_fd[frame]) > 1e-7:
            raise ValueError('Expanded033 diagnostic changed the original032 finite response')
        stable = all(np.array_equal(branch.event_indices, base.event_indices) and
                     np.array_equal(branch.event_weights, base.event_weights) for branch in shifted)
        row = {'frame': frame, 'raw_kappa': float(value.detach()),
            'actual512_kappa': numpy_kappa(base.actual),
            'same_polygon_resampling_phase_span': float(np.ptp(reorder_values)),
            'same_polygon_raw_kappa_span': float(np.ptp(raw_values)),
            'cyclic_reordering512_kappa': reorder_values,
            'first_grid_edge_changed': any(not np.array_equal(branch.edges[0], base.edges[0])
                                          for branch in shifted),
            'selected_event_unchanged': stable,
            'raw_event_directional_derivative': derivative, 'raw_actual_directional_fd': fd,
            'sampled_event_directional_derivative': float(old_ad[frame]),
            'sampled_actual_directional_fd': sfd,
            'raw_normalized_fd_error': float(normalized_error(np.asarray(derivative), np.asarray(fd))),
            'sampled_normalized_fd_error': float(normalized_error(np.asarray(old_ad[frame]), np.asarray(sfd)))}
        rows.append(row)
        raw_ad.append(derivative); raw_fd.append(fd); resampled_fd.append(sfd)
        if (frame + 1) % 100 == 0:
            result['current_frames_complete'] = frame + 1
            save()
            print('frames', frame + 1, flush=True)
    raw_ad, raw_fd = np.array(raw_ad), np.array(raw_fd)
    tail = np.array([row['sampled_normalized_fd_error'] > .1 for row in rows])
    first_changed = np.array([row['first_grid_edge_changed'] for row in rows])
    spans = np.array([row['same_polygon_resampling_phase_span'] for row in rows])
    metric_delta = np.array([row['raw_kappa'] - row['actual512_kappa'] for row in rows])
    result.update(complete=True, per_frame=rows, actual_fd_step=epsilon,
        raw_polygon_derivative=fd_summary(raw_ad, raw_fd),
        original512_derivative=fd_summary(old_ad, np.array(resampled_fd)),
        same_polygon_resampling_phase_span=summary(spans),
        same_polygon_raw_extrema_identical_all508=True,
        raw_vs512_kappa_absolute_difference=summary(abs(metric_delta)),
        original_tail_frames=int(tail.sum()), first_grid_edge_changed_frames=int(first_changed.sum()),
        original_tail_first_edge_changed_frames=int(np.sum(tail & first_changed)),
        original_tail_raw_fd_error=summary(normalized_error(raw_ad[tail], raw_fd[tail])),
        source_sha256=digest(Path(__file__)), actual_cpu_elapsed_seconds=time.monotonic() - started,
        completed_at=datetime.now(timezone.utc).isoformat())
    save()
    print({key: value for key, value in result.items() if key != 'per_frame'}, flush=True)


if __name__ == '__main__':
    main()
