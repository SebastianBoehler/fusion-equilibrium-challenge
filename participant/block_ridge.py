"""Fit the incumbent quadratic basis using bounded float64 sufficient statistics."""

import numpy as np

from .cache import blocks, input_blocks
from .kinetic import KineticModel


def moments(chunks):
    count, mean, squared = 0, None, None
    for x in chunks:
        n, average = len(x), x.mean(axis=0)
        within = np.sum((x - average) ** 2, axis=0)
        if mean is None:
            count, mean, squared = n, average, within
        else:
            delta = average - mean
            squared += within + delta ** 2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
    if not count:
        raise ValueError('No training frames')
    scale = np.sqrt(squared / count)
    scale[scale == 0] = 1.0
    return count, mean, scale


def fit_statistics(records: list[dict]) -> tuple[KineticModel, dict]:
    model = KineticModel()
    n, model.input_mean, model.input_scale = moments(input_blocks(records))
    _, model.mean, model.scale = moments(model._basis(x) for x in input_blocks(records))
    width = len(model.mean) + 1
    gram = np.zeros((3, width, width))
    cross = np.zeros((width, 4227))
    labels = np.zeros(3, dtype=np.int64)
    for raw, y in blocks(records):
        basis = (model._basis(raw) - model.mean) / model.scale
        x = np.column_stack([basis, np.ones(len(basis))])
        masks = [np.isfinite(y[:, :-2]).all(axis=1),
                 np.isfinite(y[:, -2]), np.isfinite(y[:, -1])]
        for index, mask in enumerate(masks):
            subset = x[mask]
            gram[index] += subset.T @ subset
            labels[index] += int(mask.sum())
            if index == 0:
                cross[:, :-2] += subset.T @ y[mask, :-2]
            else:
                cross[:, index - 3] += subset.T @ y[mask, index - 3]
    if np.any(labels < width):
        raise ValueError('Insufficient defined training labels')
    model.training_frames = n
    return model, {'gram': gram, 'cross': cross, 'label_counts': labels}


def solve(model: KineticModel, stats: dict, alpha: float) -> KineticModel:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Positive finite regularization required')
    result = KineticModel()
    for name in ('input_mean', 'input_scale', 'mean', 'scale', 'training_frames'):
        setattr(result, name, getattr(model, name))
    result.regularization = alpha
    result.weights = np.empty_like(stats['cross'])
    penalty = np.eye(result.weights.shape[0]) * alpha
    penalty[-1, -1] = 0
    result.weights[:, :-2] = np.linalg.solve(stats['gram'][0] + penalty,
                                            stats['cross'][:, :-2])
    for index in (1, 2):
        result.weights[:, index - 3] = np.linalg.solve(stats['gram'][index] + penalty,
                                                       stats['cross'][:, index - 3])
    return result
