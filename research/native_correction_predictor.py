"""Predict native correction fields directly, without folding or field requantization."""

import numpy as np
import torch

from participant.critical_artifact import restore_correction
from participant.finetune_common import prepare_inputs
from participant.flux_head import parameters


class NativeCorrectionPredictor:
    def __init__(self, parent, payload):
        self.model = restore_correction(parent, payload)
        self.params = parameters(parent, 'cpu')
        self.config = parent['config']

    @torch.no_grad()
    def predict_inputs(self, values):
        raw = np.asarray(values, dtype=np.float32)
        inputs = prepare_inputs(raw, self.params, self.config, 'cpu')
        psi, scalars = self.model(inputs)
        scalars = scalars[:, :2] * self.params['s_std'][:2] + self.params['s_mean'][:2]
        result = {'psirz': psi.numpy().reshape(-1, 65, 65),
                  'q95': scalars[:, 0].numpy(), 'betaN': scalars[:, 1].numpy()}
        return checked_prediction(result, len(raw))


def checked_prediction(result, frames):
    shapes = {'psirz': (frames, 65, 65), 'q95': (frames,), 'betaN': (frames,)}
    if set(result) != set(shapes):
        raise ValueError('Exact submitted prediction keys required')
    for key, shape in shapes.items():
        value = result[key]
        if value.dtype != np.float32 or value.shape != shape:
            raise ValueError(f'Incorrect float32 prediction shape or dtype: {key}')
        if not np.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite prediction: {key}')
    return result


def paired_predictions(values, offsets, native, folded):
    predictions = {'native': [], 'folded': []}
    difference = {key: {'max_absolute': 0.0, 'changed_values': 0, 'values': 0}
                  for key in ('psirz', 'q95', 'betaN')}
    if offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError('Offsets must cover every input frame in order')
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        if b <= a:
            raise ValueError('Shot offsets must be strictly increasing')
        inputs = np.asarray(values[a:b], dtype=np.float32)
        direct = native.predict_inputs(inputs)
        exported = checked_prediction(folded.predict_inputs(inputs), b - a)
        predictions['native'].append(direct)
        predictions['folded'].append(exported)
        for key, stats in difference.items():
            delta = exported[key].astype(np.float64) - direct[key].astype(np.float64)
            stats['max_absolute'] = max(stats['max_absolute'], float(np.abs(delta).max()))
            stats['changed_values'] += int(np.count_nonzero(delta))
            stats['values'] += int(delta.size)
    return predictions, difference
