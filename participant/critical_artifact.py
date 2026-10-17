"""Preserve the native smooth correction for scorer-level export comparisons."""

import torch

from .smooth_correction import SmoothCorrection


def save_correction(model, tune, destination):
    state = {name: value.detach().cpu() for name, value in model.correction.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError('Nonfinite native correction artifact')
    torch.save({'state': state, 'critical_tune': tune}, destination)


def restore_correction(parent, payload):
    model = SmoothCorrection(parent).eval()
    if not all(torch.isfinite(value).all() for value in payload['state'].values()):
        raise FloatingPointError('Nonfinite recovered native correction')
    model.correction.load_state_dict(payload['state'], strict=True)
    return model
