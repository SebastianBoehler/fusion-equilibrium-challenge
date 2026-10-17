"""Acceptance and frozen-state checks for successor finite-metric TRAIN probes."""

import torch


def frozen_state(model, checkpoint, inputs, cached_fields, cached_scalars):
    with torch.no_grad():
        fields, scalars = model(inputs)
    if not torch.equal(fields, cached_fields) or not torch.equal(scalars, cached_scalars):
        raise ValueError('Cached predictor differs from actual permitted-input inference')
    if any(p.grad is not None for p in model.parent.parameters()):
        raise ValueError('Frozen parent received a gradient')
    for key, value in checkpoint['state'].items():
        if not torch.equal(model.parent.state_dict()[key], value):
            raise ValueError('Frozen parent parameter changed')
    for key in ('x_mean', 'x_std', 'psi_mean', 'coef_scale', 'basis', 's_mean', 's_std'):
        if not torch.equal(getattr(model, key).cpu(), checkpoint[key]):
            raise ValueError('Frozen normalization or decoding statistic changed')


def acceptance(parent, control, candidate, limits):
    ratios = {key: candidate[key] / parent[key] for key in
              ('pixel_sse', 'boundary_mean', 'mean_training_standardized_derived_mse')}
    components = {key: value['mse'] / parent['derived'][key]['mse']
                  for key, value in candidate['derived'].items()}
    control_ratio = candidate['derived']['kappa']['mse'] / control['derived']['kappa']['mse']
    checks = {'kappa_vs_parent': components['kappa'] <= limits['actual_kappa_mse_ratio_to_parent_max'],
        'kappa_vs_pixel_control': control_ratio <= limits['actual_kappa_mse_ratio_to_pixel_control_max'],
        'pixel': ratios['pixel_sse'] <= limits['pixel_sse_ratio_to_parent_max'],
        'boundary': ratios['boundary_mean'] <= limits['boundary_mean_ratio_to_parent_max'],
        'all_seven_components': max(components.values()) <= limits['each_of_7_derived_mse_ratio_to_parent_max'],
        'mean_standardized_derived': ratios['mean_training_standardized_derived_mse'] <=
            limits['mean_training_standardized_derived_mse_ratio_to_parent_max'],
        'relative_correction_bound': candidate['relative_correction_rms_max'] <=
            limits['relative_correction_rms_to_parent_std_max']}
    return {'checks': checks, 'acceptance_passed': all(checks.values()),
        'ratios_to_parent': ratios, 'derived_mse_ratios_to_parent': components,
        'kappa_mse_ratio_to_pixel_control': control_ratio}
