"""Pinned TRAIN-only forward and derivative checks for predicted contour branches."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import skimage
import torch
from torch.nn import functional as F

from participant import fast_score
from participant.smooth_correction import SmoothCorrection
from research.inductance_contour_audit import digest
from research.predicted_contour_branch import batch_kappa, contour, kappa, select
from research.training_contour_sample import load


ROOT = Path('run_data/analysis-032')
MODES = ('bisection', 'event')
STEPS = (1e-3, 1e-4, 1e-5)


def summary(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite derivative diagnostic')
    return {'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'maximum': float(values.max())}


def normalized_error(a, b):
    return abs(a - b) / np.maximum(1., np.maximum(abs(a), abs(b)))


def actual_kappa(field):
    r, z, coarse, mask = fast_score._geometry()
    points, _ = fast_score.official.extract_lcfs_with_sign(
        field, r, z, 'DIII-D', coarse, mask,
        n_iter=fast_score.official.N_ITER,
        axis_sign=fast_score.official.AXIS_SIGN['DIII-D'])
    if points is None:
        raise ValueError('Actual perturbed training contour failed extraction')
    return float((points[:, 1].max() - points[:, 1].min()) /
                 (points[:, 0].max() - points[:, 0].min()))


def main():
    ROOT.mkdir()
    (ROOT / 'derivative-check.lock').mkdir()
    output = ROOT / 'derivative-check.json'
    result = {'complete': False, 'started_at': datetime.now(timezone.utc).isoformat(),
              'training_only': True, 'frames': 508, 'shots': 3,
              'optimizer_steps': 0, 'gpu_allocations': 0, 'locked_shots_read': False,
              'fields_or_learned_artifacts_saved': False,
              'working_assumption': 'Current predicted marching-squares edges and arc-length '
              'samples can replace the true-normal projection in a training loss.',
              'predeclared_checks': {'forward_max_absolute_metres': 1e-8,
              'own_branch_max_normalized_fd_error': 1e-4,
              'actual_fd_at_step': 1e-4, 'actual_fd_median_normalized_error': .05,
              'actual_fd_p95_normalized_error': .10, 'actual_fd_sign_agreement': .95},
              'limitations': 'One smooth direction per frame on previously reused TRAIN shots; '
              'branch changes remain discrete. Not an exact derivative or official equivalence.',
              'inference_inputs': 'Permitted features only; true fields and reference contours '
              'are training scoring targets. Predicted field alone selects the branch.',
              'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                           'skimage': skimage.__version__}, 'modes': {}}
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    with output.open('x') as handle:
        json.dump(result, handle)
    started = time.monotonic()
    torch.set_num_threads(2)
    sample = load()
    result['training_manifest_sha256'] = sample['manifest_sha256']
    result['target_manifest_sha256'] = sample['target_manifest_sha256']
    parent = SmoothCorrection(sample['parent']).eval()
    with torch.no_grad():
        fields, _ = parent(sample['inputs'])
    fields = fields.reshape(-1, 65, 65).double()
    branches = [select(field.numpy()) for field in fields]
    result['event_counts'] = {kind: sum(b.event_kind == kind for b in branches)
                              for kind in sorted({b.event_kind for b in branches})}
    result['scorer_and_research_sha256'] = {
        str(path): digest(path) for path in [
            Path('vendor/fusion_starter/local_score.py'),
            *[Path('vendor/fusion_starter/fusion_scoring') / f'{name}.py'
              for name in ('lcfs', 'contour', 'derive', 'o_point')],
            Path('research/predicted_contour_branch.py'), Path(__file__)]}
    result['parent_sha256'] = digest(Path('run_data/nn-008/mlp-shape/model.pt'))
    result['current_status'] = 'Selected all508 actual predicted branches'
    save()
    torch.manual_seed(20261032)
    direction = F.interpolate(torch.randn(1, 1, 17, 17, dtype=torch.float64),
                              size=(65, 65), mode='bilinear', align_corners=True)[0, 0]
    direction -= direction.mean()
    direction /= direction.square().mean().sqrt()
    direction = direction[None] * fields.std(dim=(1, 2), unbiased=False)[:, None, None]
    actual = np.asarray([float(kappa(torch.from_numpy(b.actual))) for b in branches])
    autodiff = {}
    for mode in MODES:
        leaf = fields.clone().requires_grad_()
        values = batch_kappa(leaf, branches, mode)
        values.sum().backward()
        if not torch.isfinite(leaf.grad).all():
            raise FloatingPointError('Nonfinite current-branch gradient')
        ad = (leaf.grad * direction).sum((1, 2)).numpy()
        autodiff[mode] = ad
        coordinates = max(float((contour(field, b, mode) - torch.from_numpy(b.actual)).abs().max())
                          for field, b in zip(fields, branches, strict=True))
        with torch.no_grad():
            epsilon = 1e-7
            own_fd = ((batch_kappa(fields + epsilon * direction, branches, mode) -
                       batch_kappa(fields - epsilon * direction, branches, mode)) /
                      (2 * epsilon)).numpy()
        own = summary(normalized_error(ad, own_fd))
        result['modes'][mode] = {'forward_max_absolute_metres': coordinates,
            'kappa_max_absolute_difference': float(np.max(abs(values.detach().numpy() - actual))),
            'own_branch_fd_normalized_error': own, 'own_branch_fd_step': epsilon,
            'forward_pass': coordinates < 1e-8, 'own_derivative_pass': own['maximum'] < 1e-4,
            'actual_finite_differences': {}}
        print(mode, result['modes'][mode], flush=True)
        save()
    for epsilon in STEPS:
        finite = np.asarray([(actual_kappa((field + epsilon * delta).numpy()) -
                              actual_kappa((field - epsilon * delta).numpy())) / (2 * epsilon)
                             for field, delta in zip(fields, direction, strict=True)])
        for mode in MODES:
            ad = autodiff[mode]
            errors = normalized_error(ad, finite)
            agreement = float(np.mean(np.sign(ad) == np.sign(finite)))
            result['modes'][mode]['actual_finite_differences'][str(epsilon)] = {
                'normalized_error': summary(errors), 'sign_agreement': agreement,
                'autodiff_directional_derivatives': ad.tolist(),
                'actual_directional_derivatives': finite.tolist(), 'extraction_failures': 0}
        result['current_status'] = f'Completed actual scorer finite differences at{epsilon}'
        print(result['current_status'], flush=True)
        save()
    for mode, report in result['modes'].items():
        fd = report['actual_finite_differences']['0.0001']
        report['actual_derivative_check_pass'] = (
            fd['normalized_error']['median'] <= .05 and
            fd['normalized_error']['p95'] <= .10 and fd['sign_agreement'] >= .95)
        report['eligible_for_next_training_only_update_probe'] = (
            report['forward_pass'] and report['own_derivative_pass'] and
            report['actual_derivative_check_pass'])
    result.update(complete=True, actual_cpu_elapsed_seconds=time.monotonic() - started,
                  completed_at=datetime.now(timezone.utc).isoformat(),
                  current_status='Complete; no optimizer or GPU run')
    save()


if __name__ == '__main__':
    main()
