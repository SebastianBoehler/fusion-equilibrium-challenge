"""First MAST candidate: measured coils plus a Thomson-weighted plasma current.

The current-density surrogate is circular and proportional to electron pressure.
It solves the linear magnetic-field problem, not nonlinear force balance. Ion
pressure equals electron pressure in the beta estimate. No MAST labels, masks,
learned constants, or leaderboard-derived statistics enter this model.
"""

from functools import lru_cache

import numpy as np
from scipy.special import ellipk, ellipe

from .mast_inputs import SIGNALS, aligned_currents

MU0 = 4e-7 * np.pi
E_CHARGE = 1.602176634e-19
# UKAEA MAST-U research plan: the retained TF system gives 0.585 T at
# R=0.7 m for 85 kA, equivalent to 24 turns by Ampere's law.
TF_TURNS = 24


def greens(rc, zc, r, z):
    """Poloidal flux per radian from one ampere in a circular filament."""
    m = 4 * r * rc / ((r + rc) ** 2 + (z - zc) ** 2)
    if np.any(m >= 1) or np.any(m <= 0):
        raise ValueError("Green's function requires distinct positive-radius points")
    return MU0 / (2 * np.pi) * np.sqrt(r * rc / m) * (
        (2 - m) * ellipk(m) - 2 * ellipe(m))


@lru_cache(maxsize=4)
def plasma_kernel(r_coordinates: tuple, z_coordinates: tuple):
    r, z = np.asarray(r_coordinates), np.asarray(z_coordinates)
    rr, zz = np.meshgrid(r, z)
    # Cell centres avoid the filament singularity at the output-grid nodes.
    sr = (r[:-1] + r[1:]) / 2
    sz = (z[:-1] + z[1:]) / 2
    source_r, source_z = np.meshgrid(sr, sz)
    kernel = greens(source_r.ravel()[None], source_z.ravel()[None],
                    rr.ravel()[:, None], zz.ravel()[:, None])
    return kernel, source_r.ravel(), source_z.ravel()


def coil_kernel(row):
    r, z = np.meshgrid(row['efit_grid_R'], row['efit_grid_Z'])
    names = np.asarray(row['coil_input_column']).astype(str)
    columns = []
    for name in np.unique(names):
        selected = names == name
        rc, zc = row['coil_R'][selected], row['coil_Z'][selected]
        width, height = row['coil_width'][selected], row['coil_height'][selected]
        response = np.zeros(r.shape)
        for a in (-.25, .25):
            for b in (-.25, .25):
                response += greens((rc + a * width)[None, None, :],
                                   (zc + b * height)[None, None, :],
                                   r[:, :, None], z[:, :, None]).sum(axis=2) / 4
        # PF currents already contain ampere-turns; solenoid feed is amperes.
        if name != 'magnetics_sol_current':
            response /= selected.sum()
        signal = name.removeprefix('magnetics_')
        if signal not in SIGNALS:
            raise ValueError(f'Unsupported poloidal coil input: {name}')
        columns.append((SIGNALS.index(signal), response.ravel()))
    return columns


def pressure_profiles(row):
    radius = np.asarray(row['thomson_core_R'], dtype=float)
    te = np.asarray(row['thomson_core_Te'], dtype=float)
    ne = np.asarray(row['thomson_core_ne'], dtype=float)
    if te.shape != ne.shape or te.shape[1] != len(radius) + 1:
        raise ValueError('Expected the documented MAST ghost Thomson channel')
    te, ne = te[:, 1:], ne[:, 1:]
    pressure = te * ne * E_CHARGE
    pressure[(te <= 0) | (ne <= 0) | ~np.isfinite(pressure)] = np.nan
    times = np.asarray(row['thomson_core_times'], dtype=float)
    centers = np.asarray(row['efit_times'], dtype=float)
    out = np.full((len(centers), len(radius)), np.nan)
    for j in range(len(radius)):
        valid = np.isfinite(times) & np.isfinite(pressure[:, j])
        if valid.sum() >= 2:
            if np.any(np.diff(times[valid]) <= 0):
                raise ValueError('Invalid Thomson time order')
            out[:, j] = np.interp(centers, times[valid], pressure[valid, j])
    if np.any(np.isfinite(out).sum(axis=1) < 8):
        raise ValueError('Insufficient measured pressure for a MAST frame')
    return radius, out


def predict(row):
    currents = aligned_currents(row)
    radius, pressure = pressure_profiles(row)
    # Quadrature weights account for the nonuniform Thomson channel spacing.
    order = np.argsort(radius)
    radius, pressure = radius[order], pressure[:, order]
    spacing = np.gradient(radius)
    weighted = np.where(np.isfinite(pressure), pressure, 0) * spacing
    norm = weighted.sum(axis=1)
    r0 = (weighted * radius).sum(axis=1) / norm
    variance = (weighted * (radius - r0[:, None]) ** 2).sum(axis=1) / norm
    if np.any(norm <= 0) or np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError('Invalid Thomson pressure moments')
    kernel, sr, sz = plasma_kernel(tuple(row['efit_grid_R']), tuple(row['efit_grid_Z']))
    weights = np.exp(-((sr[None] - r0[:, None]) ** 2 + sz[None] ** 2)
                     / (2 * variance[:, None]))
    ip = currents[:, SIGNALS.index('plasma_current')] * 1000
    if np.any(ip <= 0):
        raise ValueError('The current-density candidate requires positive plasma current')
    weights *= (ip / weights.sum(axis=1))[:, None]
    psi = weights @ kernel.T
    for column, response in coil_kernel(row):
        psi += currents[:, column, None] * 1000 * response
    fvac = abs(currents[:, SIGNALS.index('tf_current')]) * 1000 * MU0 * TF_TURNS / (2 * np.pi)
    b0 = fvac / r0
    if np.any(b0 <= 0):
        raise ValueError('Nonpositive measured toroidal field')
    # The circular Gaussian contains 95% of its current within this radius.
    a95 = np.sqrt(-2 * variance * np.log(.05))
    q95 = 2 * np.pi * a95 ** 2 * b0 / (MU0 * .95 * ip * r0)
    paxis = np.nanmax(pressure, axis=1) * 2  # equal ion/electron temperatures
    paverage = paxis * .95 / -np.log(.05)
    beta = 2 * MU0 * paverage / b0 ** 2
    beta_n = 100 * beta * a95 * b0 / (ip / 1e6)
    result = {'psirz': psi.reshape(-1, 65, 65).astype(np.float32),
              'q95': q95.astype(np.float32), 'betaN': beta_n.astype(np.float32)}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite MAST prediction')
    return result
