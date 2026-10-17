"""Positive triangle quadrature of pressure inside an explicit predicted contour."""

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .mast_green import MU0


def contour_beta_n(r, z, pressure, contour, axis, fvac, minor_radius, ip):
    """A supplied star-shaped contour defines integration, never equilibrium labels."""
    r, z, pressure, contour, axis = map(np.asarray, (r, z, pressure, contour, axis))
    if (r.ndim != 1 or z.ndim != 1 or min(len(r), len(z)) < 4
            or pressure.shape != (len(z), len(r)) or contour.ndim != 2
            or contour.shape[1] != 2 or len(contour) < 4 or axis.shape != (2,)
            or not all(np.isfinite(a).all() for a in (r, z, pressure, contour, axis))
            or np.any(r <= 0) or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)
            or not np.array_equal(contour[0], contour[-1])):
        raise ValueError('Invalid pressure grid, axis or closed contour')
    normalization = np.asarray([fvac, minor_radius, ip, axis[0]], dtype=float)
    if not np.isfinite(normalization).all() or np.any(normalization <= 0):
        raise ValueError('Invalid contour-pressure normalization')
    if (np.any(contour[:, 0] < r[0]) or np.any(contour[:, 0] > r[-1])
            or np.any(contour[:, 1] < z[0]) or np.any(contour[:, 1] > z[-1])):
        raise ValueError('Pressure contour lies outside the input grid')
    vertices = contour[:-1] - axis
    if np.any(np.linalg.norm(vertices, axis=1) <= 0):
        raise ValueError('Pressure axis lies on the contour')
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]
    angles = np.arctan2(cross, (vertices * following).sum(1))
    if (np.any(cross <= 0) or not np.isclose(angles.sum(), 2 * np.pi,
                                          rtol=0, atol=1e-10)):
        raise ValueError('Pressure contour must be simple, counterclockwise and star-shaped')
    nodes, weights = np.polynomial.legendre.leggauss(3)
    nodes, weights = (nodes + 1) / 2, weights / 2
    u, v = np.meshgrid(nodes, nodes, indexing='ij')
    quadrature = np.outer(weights, weights) * (1 - u)
    points = (axis + u[None, :, :, None] * vertices[:, None, None, :]
              + ((1 - u) * v)[None, :, :, None] * following[:, None, None, :])
    rm, zm = points[..., 0], points[..., 1]
    spline = RectBivariateSpline(z, r, pressure)
    sampled = spline.ev(zm.ravel(), rm.ravel()).reshape(rm.shape)
    if not np.isfinite(sampled).all() or np.any(sampled < 0):
        raise ValueError('Invalid reconstructed pressure inside the core')
    volume_weights = cross[:, None, None] * quadrature[None] * rm
    pbar = float(np.sum(volume_weights * sampled) / volume_weights.sum())
    b0 = fvac / axis[0]
    beta_n = 100 * (2 * MU0 * pbar / b0 ** 2) * minor_radius * b0 / (ip / 1e6)
    if not np.isfinite(beta_n):
        raise FloatingPointError('Nonfinite contour-pressure betaN')
    return float(beta_n), pbar
