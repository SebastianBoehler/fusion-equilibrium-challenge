"""Decode PCA/direct-grid heads and algebraically fold a frozen PCA basis into its head."""

import numpy as np
import torch


def parameters(checkpoint, device):
    keys = ['x_mean', 'x_std', 'psi_mean', 'coef_scale', 's_mean', 's_std']
    if checkpoint['config'].get('representation', 'pca') == 'pca':
        keys.append('basis')
    return {key: checkpoint[key].to(device) for key in keys}


def flux_delta(out, params, config):
    flux = out[:, :config['components']] * params['coef_scale']
    representation = config.get('representation', 'pca')
    if representation == 'pca':
        flux = flux @ params['basis'].T
    elif representation != 'grid':
        raise ValueError(f'Unknown flux representation: {representation}')
    return flux


def decode(out, params, config):
    return params['psi_mean'] + flux_delta(out, params, config)


def direct_grid(checkpoint):
    if checkpoint['config'].get('representation', 'pca') != 'pca':
        raise ValueError('Direct-grid conversion requires a PCA checkpoint')
    ck = dict(checkpoint)
    cfg = ck['config']
    k = cfg['components']
    basis = ck.pop('basis')
    scale = (basis.double().square() @ ck['coef_scale'].double().square()).sqrt().clamp(min=1e-5).float()
    transform = basis.double() * ck['coef_scale'].double()[None, :] / scale.double()[:, None]
    state = dict(ck['state'])
    weights, bias = state['head.1.weight'], state['head.1.bias']
    state['head.1.weight'] = torch.cat(((transform @ weights[:k].double()).float(), weights[k:]))
    state['head.1.bias'] = torch.cat(((transform @ bias[:k].double()).float(), bias[k:]))
    ck['state'], ck['coef_scale'] = state, scale
    ck['config'] = cfg | {'representation': 'grid', 'components': 4225,
                         'parent_pca_components': k}
    return ck


def export(out, params, config, index, path):
    arrays = {}
    k = config['components']
    with torch.no_grad():
        for i, (a, b) in enumerate(zip(index['offsets'][:-1], index['offsets'][1:], strict=True)):
            psi = decode(out[a:b], params, config).reshape(-1, 65, 65).cpu().numpy()
            scalar = (out[a:b, k:k + 2] * params['s_std'][:2] + params['s_mean'][:2]).cpu().numpy()
            if not np.isfinite(psi).all() or not np.isfinite(scalar).all():
                raise FloatingPointError('Nonfinite direct-grid prediction')
            arrays[f'shot_{i:04d}_psirz'] = psi.astype(np.float32)
            arrays[f'shot_{i:04d}_q95'] = scalar[:, 0].astype(np.float32)
            arrays[f'shot_{i:04d}_betaN'] = scalar[:, 1].astype(np.float32)
    np.savez(path, **arrays)
