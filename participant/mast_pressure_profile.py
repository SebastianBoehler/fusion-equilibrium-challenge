"""Electron pressure on supplied flux coordinates, without extrapolation."""

import numpy as np
from scipy.interpolate import PchipInterpolator


def electron_pressure_on_flux(sample_flux, pressure_pa, query_flux):
    """Geometry and measurement alignment must establish these coordinates first."""
    arrays = [np.asarray(a) for a in (sample_flux, pressure_pa, query_flux)]
    if any(a.ndim != 1 or a.dtype.kind not in 'fiu' for a in arrays):
        raise ValueError('Pressure interpolation requires real one-dimensional arrays')
    flux, pressure, query = [a.astype(np.float64) for a in arrays]
    if (len(flux) < 3 or pressure.shape != flux.shape or not len(query)
            or not all(np.isfinite(a).all() for a in (flux, pressure, query))
            or np.any(np.diff(flux) <= 0) or flux[0] < 0 or flux[-1] > 1
            or np.any(pressure < 0)):
        raise ValueError('Invalid measured electron-pressure support')
    if np.any(query < flux[0]) or np.any(query > flux[-1]):
        raise ValueError('Electron-pressure query is outside measured support')
    values = PchipInterpolator(flux, pressure, extrapolate=False)(query)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise FloatingPointError('Invalid interpolated electron pressure')
    return values
