"""Thermal volume-pressure beta functional; pressure reconstruction is separate."""

import numpy as np

from .mast_green import MU0


def volume_beta_n(r, z, pressure, core, fvac, axis_r, minor_radius, ip):
    """Integrate total pressure in Pa over an explicitly provided predicted core."""
    r, z, pressure, core = map(np.asarray, (r, z, pressure, core))
    if (r.ndim != 1 or z.ndim != 1 or min(len(r), len(z)) < 3
            or pressure.shape != (len(z), len(r)) or core.shape != pressure.shape
            or core.dtype != np.bool_ or not core.any()
            or not np.isfinite(r).all() or not np.isfinite(z).all()
            or np.any(r <= 0) or np.any(np.diff(r) <= 0)
            or np.any(np.diff(z) <= 0) or not np.isfinite(pressure).all()
            or np.any(pressure < 0)):
        raise ValueError('Invalid volume-pressure grid or core support')
    geometry = np.asarray([fvac, axis_r, minor_radius, ip], dtype=float)
    if not np.isfinite(geometry).all() or np.any(geometry <= 0):
        raise ValueError('Invalid volume-pressure normalization')
    weights = core * r[None, :]
    volume = np.trapezoid(np.trapezoid(weights, r, axis=1), z)
    pbar = np.trapezoid(np.trapezoid(weights * pressure, r, axis=1), z) / volume
    b0 = fvac / axis_r
    beta_n = 100 * (2 * MU0 * pbar / b0 ** 2) * minor_radius * b0 / (ip / 1e6)
    if not np.isfinite(beta_n):
        raise FloatingPointError('Nonfinite volume-pressure betaN')
    return float(beta_n), float(pbar)
