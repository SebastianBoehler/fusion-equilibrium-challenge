"""External vertical-force diagnostic; no equilibrium solver or predictions."""

import numpy as np
from scipy.special import ellipe, ellipk

from participant.mast_green import MU0, greens, pressure_profiles
from participant.mast_inputs import SIGNALS, aligned_currents


def flux_dz(rc, zc, r, z):
    """Analytic derivative of the frozen nonsingular circular-ring Green flux."""
    rc, zc, r, z = np.broadcast_arrays(rc, zc, r, z)
    if not all(np.isfinite(a).all() for a in (rc, zc, r, z)):
        raise ValueError('Nonfinite ring coordinates')
    if np.any(rc <= 0) or np.any(r <= 0):
        raise ValueError('Nonpositive ring radius')
    dz = z - zc
    distance = (r + rc) ** 2 + dz ** 2
    m = 4 * r * rc / distance
    if np.any(m <= 0) or np.any(m >= 1):
        raise ValueError('Coincident or degenerate ring points')
    result = MU0 / (4 * np.pi) * dz / np.sqrt(distance) * (
        2 * ellipk(m) - (2 - m) / (1 - m) * ellipe(m))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite flux derivative')
    return result


def central_dz(rc, zc, r, z, h):
    return (greens(rc, zc, r, z + h) - greens(rc, zc, r, z - h)) / (2 * h)


def baseline_current(row):
    """Reproduce only the frozen current weights, without constructing psi."""
    currents = aligned_currents(row)
    radius, pressure = pressure_profiles(row)
    order = np.argsort(radius)
    radius, pressure = radius[order], pressure[:, order]
    spacing = np.gradient(radius)
    weighted = np.where(np.isfinite(pressure), pressure, 0) * spacing
    norm = weighted.sum(axis=1)
    r0 = (weighted * radius).sum(axis=1) / norm
    variance = (weighted * (radius - r0[:, None]) ** 2).sum(axis=1) / norm
    if np.any(norm <= 0) or np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError('Invalid frozen baseline pressure moments')
    r, z = np.asarray(row['efit_grid_R']), np.asarray(row['efit_grid_Z'])
    if (r.shape != (65,) or z.shape != (65,) or not np.isfinite(r).all()
            or not np.isfinite(z).all() or np.any(r <= 0)
            or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)):
        raise ValueError('Invalid frozen baseline grid')
    sr, sz = np.meshgrid((r[:-1] + r[1:]) / 2, (z[:-1] + z[1:]) / 2)
    sr, sz = sr.ravel(), sz.ravel()
    weights = np.exp(-((sr[None] - r0[:, None]) ** 2 + sz[None] ** 2)
                     / (2 * variance[:, None]))
    ip = currents[:, SIGNALS.index('plasma_current')] * 1000
    if np.any(ip <= 0):
        raise ValueError('Nonpositive frozen baseline plasma current')
    weights *= (ip / weights.sum(axis=1))[:, None]
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite frozen current distribution')
    return currents, weights, sr, sz, r0, variance


def coil_derivatives(row, sr, sz, h, deadline_check):
    """Preserve the original four-point geometry and feed-current convention."""
    names = np.asarray(row['coil_input_column']).astype(str)
    columns = []
    for name in np.unique(names):
        selected = names == name
        rc, zc = row['coil_R'][selected], row['coil_Z'][selected]
        width, height = row['coil_width'][selected], row['coil_height'][selected]
        response = np.zeros((3, len(sr)))
        for a in (-.25, .25):
            for b in (-.25, .25):
                args = ((rc + a * width)[None], (zc + b * height)[None],
                        sr[:, None], sz[:, None])
                response[0] += flux_dz(*args).sum(axis=1) / 4
                response[1] += central_dz(*args, h).sum(axis=1) / 4
                response[2] += central_dz(*args, h / 2).sum(axis=1) / 4
                deadline_check()
        if name != 'magnetics_sol_current':
            response /= selected.sum()
        signal = name.removeprefix('magnetics_')
        if signal not in SIGNALS:
            raise ValueError(f'Unsupported coil input: {name}')
        columns.append((SIGNALS.index(signal), response))
    if not columns:
        raise ValueError('No external coil geometry')
    return columns


def vertical_force(weights, derivative):
    weights, derivative = np.asarray(weights), np.asarray(derivative)
    if (weights.ndim != 2 or weights.shape != derivative.shape
            or not np.isfinite(weights).all() or not np.isfinite(derivative).all()
            or np.any(weights < 0) or np.any(weights.sum(axis=1) <= 0)):
        raise ValueError('Invalid current weights or derivative')
    contribution = weights * derivative
    force = 2 * np.pi * contribution.sum(axis=1)
    scale = 2 * np.pi * np.abs(contribution).sum(axis=1)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('Undefined force normalization')
    return force, scale, force / scale
