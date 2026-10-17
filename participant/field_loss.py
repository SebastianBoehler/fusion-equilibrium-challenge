"""Relative poloidal-field gradient error on physical training grid coordinates."""

import torch


def field_vectors(psi, r, z):
    field = psi.reshape(-1, len(z), len(r))
    dz, dr = torch.gradient(field, spacing=(z, r), dim=(1, 2))
    return torch.stack((dr / r[None, None, :], dz / r[None, None, :]), dim=1)


def loss(predicted, target, r, z):
    expected = field_vectors(target, r, z)
    energy = expected.square().mean((1, 2, 3))
    valid = energy > 1e-12
    if not valid.any():
        raise ValueError('Training batch has no valid target field energy')
    residual = field_vectors(predicted, r, z) - expected
    return (residual.square().mean((1, 2, 3))[valid] / energy[valid]).mean()
