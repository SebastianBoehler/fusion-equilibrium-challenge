"""Flux losses at true training contours and magnetic-axis neighborhoods."""

import torch
from torch.nn import functional as F

from .geometry_targets import BOUNDARY_POINTS


def losses(psi, points, values, span, valid):
    samples = F.grid_sample(psi.reshape(-1, 1, 65, 65), points[:, :, None],
                            mode='bilinear', padding_mode='border', align_corners=True)
    error = (samples[:, 0, :, 0] - values) / span.clamp(min=1e-5)[:, None]
    boundary = (error[:, :BOUNDARY_POINTS] - error[:, BOUNDARY_POINTS:BOUNDARY_POINTS + 1])
    axis = torch.stack((error[:, -4] - error[:, -3], error[:, -2] - error[:, -1]), dim=1)
    mask = valid.to(error.dtype)
    count = mask.sum().clamp(min=1)
    return ((boundary.square().mean(1) * mask).sum() / count,
            (axis.square().mean(1) * mask).sum() / count)
