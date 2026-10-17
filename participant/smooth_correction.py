"""Frozen PCA predictor with a zero-initialized, physically scaled smooth correction."""

import torch
from torch import nn
from torch.nn import functional as F

from .flux_head import decode, direct_grid, parameters
from .nn_train import Network


class SmoothCorrection(nn.Module):
    """Prototype: fixed 17x17 correction, bilinearly decoded onto the 65x65 grid."""

    def __init__(self, checkpoint):
        super().__init__()
        cfg = checkpoint['config']
        if cfg.get('representation', 'pca') != 'pca' or cfg.get('context', 0) != 0:
            raise ValueError('Prototype requires a PCA parent without temporal context')
        self.config = cfg
        self.parent = Network(cfg.get('network_inputs', cfg['inputs']), cfg['components'] + cfg['scalars'],
                              cfg['width'], cfg['depth'], cfg['dropout'])
        self.parent.load_state_dict(checkpoint['state'])
        self.parent.requires_grad_(False)
        self.parent.eval()
        for name, value in parameters(checkpoint, 'cpu').items():
            self.register_buffer(name, value.detach().clone())
        # Freeze training statistics; correction coordinates all have one physical scale.
        self.register_buffer('correction_scale', checkpoint['coef_scale'].square().sum().sqrt() / 65)
        self.correction = nn.Linear(cfg['width'], 17 * 17)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            features = self.parent.head[0](self.parent.blocks(self.parent.stem(x)))
            out = self.parent.head[1](features)
            params = {name: getattr(self, name) for name in ('psi_mean', 'coef_scale', 'basis')}
            base = decode(out, params, self.config)
        correction = self.correction(features).reshape(-1, 1, 17, 17)
        correction = F.interpolate(correction, size=(65, 65), mode='bilinear', align_corners=True)
        return base + correction.flatten(1) * self.correction_scale, out[:, self.config['components']:]


def fold_checkpoint(parent, model):
    """Fold a trained linear spatial correction into the existing grid checkpoint API."""
    checkpoint = direct_grid(parent)
    interpolation = F.interpolate(torch.eye(17 * 17).reshape(-1, 1, 17, 17),
                                  size=(65, 65), mode='bilinear', align_corners=True).flatten(1).T
    transform = interpolation * model.correction_scale.detach().cpu() / checkpoint['coef_scale'][:, None]
    weights = checkpoint['state']['head.1.weight'].clone()
    bias = checkpoint['state']['head.1.bias'].clone()
    weights[:4225] += transform @ model.correction.weight.detach().cpu()
    bias[:4225] += transform @ model.correction.bias.detach().cpu()
    checkpoint['state'] = checkpoint['state'] | {'head.1.weight': weights, 'head.1.bias': bias}
    checkpoint['config'] = checkpoint['config'] | {'smooth_correction': {
        'grid': 17, 'scale': float(model.correction_scale), 'parent_frozen': True, 'scalars_frozen': True}}
    return checkpoint
