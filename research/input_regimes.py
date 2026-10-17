"""Summarize flux and submitted-scalar errors by measured input regime."""

import numpy as np

from research.diagnostics import subsets


def summarize(shots, predictions, x):
    flux_sse, truth_sum, truth_sumsq, scalar_sse = [], [], [], []
    for shot, prediction in zip(shots, predictions, strict=True):
        truth, predicted = shot['psi'], prediction['psirz']
        flux_sse.append(np.square(predicted - truth).sum((1, 2)))
        truth_sum.append(truth.sum((1, 2)))
        truth_sumsq.append(np.square(truth).sum((1, 2)))
        scalar_sse.append(np.column_stack([
            np.square(prediction[key] - shot[key]) for key in ('q95', 'betaN')]))
    sse = np.concatenate(flux_sse)
    total, square = np.concatenate(truth_sum), np.concatenate(truth_sumsq)
    scalar = np.concatenate(scalar_sse)
    if len(sse) != len(x):
        raise ValueError('Regime inputs must match complete prediction order')
    groups = {'all': np.ones(len(x), dtype=bool), **subsets(x)}
    result = {}
    for name, mask in groups.items():
        frames = int(mask.sum())
        if not frames:
            raise ValueError(f'Empty development input regime: {name}')
        pixels = frames * 65 * 65
        denominator = square[mask].sum() - total[mask].sum() ** 2 / pixels
        result[name] = {'frames': frames,
                        'flux_squared_error': float(sse[mask].sum()),
                        'flux_error_share': float(sse[mask].sum() / sse.sum()),
                        'flux_R2_within_regime': float(1 - sse[mask].sum() / denominator),
                        'flux_rmse': float(np.sqrt(sse[mask].sum() / pixels)),
                        'scalar_rmse': dict(zip(('q95', 'betaN'),
                            np.sqrt(np.nanmean(scalar[mask], 0)).tolist(), strict=True))}
    return result
