"""Predeclared053 invalid controls; reject before actual contour work."""
import time
import numpy as np
import torch
from research import finite_constraint_adjoint as helper


def reject(action):
    try:
        action()
    except (ValueError,FloatingPointError):
        return
    raise AssertionError('Invalid reference request accepted')


def helper_rejections(fields,std,scale,targets,variances,seed,deadline):
    fields_bad = fields.clone();fields_bad[0,0] = float('nan')
    std_bad = std.clone();std_bad[0] = float('nan')
    target_bad = targets.copy();target_bad[0,0] = np.nan
    variance_bad = variances.copy();variance_bad[0] = np.nan
    base = dict(fields=fields,parent_std=std,correction_scale=scale,targets=targets,
                full_train_variances=variances,generator=torch.Generator().manual_seed(seed),deadline=deadline)
    invalid = [dict(fields=fields.double()),dict(fields=fields[:,:-1]),
        dict(fields=fields[:1].repeat(2049,1)),dict(fields=fields_bad),
        dict(fields=fields.clone().requires_grad_()),dict(fields=torch.empty(fields.shape,device='meta')),
        dict(parent_std=std.double()),dict(parent_std=std[:-1]),dict(parent_std=std*0),
        dict(parent_std=std_bad),dict(parent_std=std.clone().requires_grad_()),
        dict(correction_scale=scale.double()),dict(correction_scale=scale*0),
        dict(correction_scale=scale.clone().requires_grad_()),dict(targets=targets.astype(np.float32)),
        dict(targets=targets[:,:-1]),dict(targets=target_bad),
        dict(full_train_variances=variances.astype(np.float32)),dict(full_train_variances=variances[:-1]),
        dict(full_train_variances=variances*0),dict(full_train_variances=variance_bad),
        dict(generator=None),dict(deadline=time.monotonic()-1),dict(deadline=float('nan'))]
    original = helper._measure
    calls = []
    def prohibited_measure(*args):
        calls.append(True)
        raise AssertionError('Invalid control reached numerical oracle')
    helper._measure = prohibited_measure
    try:
        for overrides in invalid:
            reject(lambda overrides=overrides:helper.cpu_constraint_adjoint(**(base|overrides)))
    finally:
        helper._measure = original
    if calls:
        raise AssertionError('Invalid control reached numerical oracle')
    return {'invalid_helper_controls_rejected':len(invalid),'oracle_calls':0,
            'nonCPU_control_device':'meta; no GPU allocation'}
