"""Independent actual seven-component reference for the fixed053 experiment."""

import numpy as np
import torch
from torch.nn import functional as F

from participant.artifacts import write_json
from participant.finite_metrics import measure_derived
from research import finite_constraint_adjoint as helper
from research.finite_device_bridge import cpu_adjoint_to_device, reconstruct_cpu_fields
from research.finite_metric_loss import predict
from participant.smooth_correction import fold_checkpoint
from participant.nn_train import Network
from participant.flux_head import decode, parameters


def require(value, message):
    if not value:
        raise ValueError(message)


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(seed)


def reference(fields, std, scale, targets, variances, seed, diagnostic_path):
    count = len(fields)
    signs = torch.randint(0, 2, (count, 289), generator=generator(seed)).to(torch.float32)*2-1
    step = .001 * std / scale
    delta = F.interpolate((signs*step[:, None]).reshape(count,1,17,17),
                          size=(65,65), mode='bilinear', align_corners=True).flatten(1)*scale
    plus_fields, minus_fields = fields+delta, fields-delta
    plus, minus = measure_derived(plus_fields.numpy()), measure_derived(minus_fields.numpy())
    components_plus = np.square(plus-targets)/variances
    components_minus = np.square(minus-targets)/variances
    losses_plus, losses_minus = components_plus.mean(1), components_minus.mean(1)
    direction = (losses_plus-losses_minus)/(2*step.numpy()*count)
    central_denominator = (2*step.numpy()*count).astype(np.float64)
    by_component = (components_plus-components_minus)/(central_denominator[:,None]*7)
    write_json(diagnostic_path, {
        "plus_measurements":plus.tolist(), "minus_measurements":minus.tolist(),
        "plus_losses":losses_plus.tolist(), "minus_losses":losses_minus.tolist(),
        "central_direction":direction.tolist(), "component_contributions":by_component.tolist(),
        "maximum_component_accounting_absolute_difference":float(np.max(np.abs(by_component.sum(1)-direction))),
        "relative_tolerance":1e-12,"absolute_tolerance":1e-14})
    require(np.allclose(by_component.sum(1), direction, rtol=1e-12, atol=1e-14),
            'Independent component contribution accounting differs')
    adjoint = torch.from_numpy(direction).to(torch.float32)[:,None]*signs
    require(torch.isfinite(adjoint).all(), 'Nonfinite independent actual reference')
    return adjoint, (plus_fields,minus_fields), (plus,minus), (losses_plus,losses_minus), by_component


def check_model(model, parent, cache, inputs, targets, variances, seed, deadline, diagnostic_path):
    initial = {key:value.detach().clone() for key,value in model.state_dict().items()}
    latent, native = predict(model, cache.features, cache.fields)
    bridge_latent = model.correction(cache.features)
    fields = reconstruct_cpu_fields(bridge_latent, cache.fields, model.correction_scale)
    require(torch.equal(fields, native.detach()) and not fields.requires_grad, 'CPU fields differ')
    collected = []
    original_measure = helper._measure
    def capture(actual_fields, actual_deadline):
        values = original_measure(actual_fields, actual_deadline)
        collected.append((actual_fields.detach().clone(),values.copy()))
        return values
    helper._measure = capture
    try:
        actual = helper.cpu_constraint_adjoint(fields, cache.parent_std, model.correction_scale,
                    targets, variances, generator(seed), deadline)
    finally:
        helper._measure = original_measure
    expected, ref_fields, measurements, losses, contributions = reference(
        fields, cache.parent_std, model.correction_scale, targets, variances, seed, diagnostic_path)
    require(len(collected)==2 and all(torch.equal(got[0], ref)
            for got,ref in zip(collected,ref_fields,strict=True)), 'Actual plus/minus fields differ')
    require(all(np.array_equal(got[1],ref) for got,ref in zip(collected,measurements,strict=True)),
            'Actual measured seven components differ')
    actual_losses = [(np.square(got[1]-targets)/variances).mean(1) for got in collected]
    require(all(np.array_equal(got,ref) for got,ref in zip(actual_losses,losses,strict=True))
            and torch.equal(actual,expected), 'Seven loss/adjoint identity failed')
    weights = tuple(model.correction.parameters())
    gradients = torch.autograd.grad(latent, weights, grad_outputs=expected)
    bridge_latent.backward(cpu_adjoint_to_device(bridge_latent, actual))
    require(all(torch.equal(p.grad,g) and torch.isfinite(g).all()
                for p,g in zip(weights,gradients,strict=True))
            and any(torch.any(g!=0) for g in gradients), 'Parameter gradient identity/nonzero gate failed')
    with torch.no_grad():
        native_fields,scalars = model(inputs)
        require(torch.equal(native_fields,native) and torch.equal(scalars,cache.scalars),
                'Actual native fields/scalars differ')
        folded = fold_checkpoint(parent,model)
        cfg = folded['config']
        exported = Network(cfg['inputs'],cfg['components']+cfg['scalars'],
                           cfg['width'],cfg['depth'],cfg['dropout']).eval()
        exported.load_state_dict(folded['state'])
        output = exported(inputs)
        decoded = decode(output,parameters(folded,'cpu'),cfg)
        maximum = float((decoded-fields).abs().max())
        require(maximum<=3e-6 and torch.equal(output[:,4225:],scalars), 'Folded numerical gates failed')
    require(all(torch.equal(model.state_dict()[key],value) for key,value in initial.items())
            and all(p.grad is None for p in model.parent.parameters()), 'Frozen parameters/statistics changed')
    return {'actual_plus_minus_fields_bit_equal':True,'actual_measurements_bit_equal':True,
        'float64_loss_and_float32_adjoint_bit_equal':True,'parameter_gradients_bit_equal':True,
        'finite_nonzero_gradient':True,'native_fields_scalars_exact':True,
        'folded_maximum_field_difference':maximum,'folded_scalars_exact':True,
        'parameters_statistics_exact':True,'plus_measurements':measurements[0].tolist(),
        'minus_measurements':measurements[1].tolist(),'plus_frame_losses':losses[0].tolist(),
        'minus_frame_losses':losses[1].tolist(),'component_direction_contributions':contributions.tolist(),
        'latent_adjoint_rms':float(actual.square().mean().sqrt())}
