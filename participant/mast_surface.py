"""Safety factor on a predicted flux surface, without MAST equilibrium priors."""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import maximum_filter
from skimage.measure import find_contours, points_in_poly

from .mast_green import MU0, TF_TURNS, pressure_profiles
from .mast_inputs import SIGNALS, aligned_currents


def enclosing_contour(psi, level, axis, conductors):
    for contour in find_contours(psi, level):
        if (len(contour) >= 4 and np.allclose(contour[0], contour[-1])
                and points_in_poly(np.asarray([axis]), contour)[0]
                and not points_in_poly(conductors, contour).any()):
            return contour
    return None


def flux_surface(psi, r, z, center, conductors, fraction=.95):
    """Trace the surface between the plasma O-point and its last closed contour."""
    candidates = (psi == maximum_filter(psi, size=3))
    candidates[[0, -1], :] = False
    candidates[:, [0, -1]] = False
    iz, ir = np.where(candidates)
    if not len(ir):
        raise ValueError('Predicted flux has no interior plasma O-point')
    nearest = np.argmin((r[ir] - center) ** 2 + z[iz] ** 2)
    axis = (int(iz[nearest]), int(ir[nearest]))
    axis_flux = float(psi[axis])
    border = np.concatenate((psi[0], psi[-1], psi[:, 0], psi[:, -1]))
    low, high = float(border.min()), axis_flux - 1e-9
    if high <= low or enclosing_contour(psi, high, axis, conductors) is None:
        raise ValueError('Predicted plasma O-point has no enclosing flux surface')
    for _ in range(20):
        middle = (low + high) / 2
        if enclosing_contour(psi, middle, axis, conductors) is None:
            low = middle
        else:
            high = middle
    level = axis_flux + fraction * (high - axis_flux)
    contour = enclosing_contour(psi, level, axis, conductors)
    if contour is None:
        raise ValueError('Requested normalized flux surface is not closed')
    coordinates = np.column_stack((
        np.interp(contour[:, 1], np.arange(len(r)), r),
        np.interp(contour[:, 0], np.arange(len(z)), z)))
    return coordinates


def safety_factor(psi, r, z, contour, fpol):
    """q = F/(2*pi) integral dl/(R*|grad psi|), with psi in Wb/radian."""
    spline = RectBivariateSpline(z, r, psi)
    midpoints = (contour[1:] + contour[:-1]) / 2
    rm, zm = midpoints.T
    gradient = np.hypot(spline.ev(zm, rm, dx=1), spline.ev(zm, rm, dy=1))
    if np.any(gradient <= 0) or not np.isfinite(gradient).all() or fpol <= 0:
        raise ValueError('Cannot integrate q over a degenerate magnetic surface')
    length = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    return float(fpol / (2 * np.pi) * np.sum(length / (rm * gradient)))


def predict_q95(row, psi):
    radius, pressure = pressure_profiles(row)
    order = np.argsort(radius)
    radius, pressure = radius[order], pressure[:, order]
    weighted = np.where(np.isfinite(pressure), pressure, 0) * np.gradient(radius)
    center = (weighted * radius).sum(axis=1) / weighted.sum(axis=1)
    currents = aligned_currents(row)
    fvac = abs(currents[:, SIGNALS.index('tf_current')]) * 1000 * MU0 * TF_TURNS / (2 * np.pi)
    r, z = row['efit_grid_R'], row['efit_grid_Z']
    # A plasma surface cannot enclose the PF or solenoid conductor filaments.
    # These positions are measured machine inputs, not an equilibrium envelope.
    conductors = np.column_stack((
        np.interp(row['coil_Z'], z, np.arange(len(z))),
        np.interp(row['coil_R'], r, np.arange(len(r)))))
    values = []
    for frame, target, fpol in zip(psi, center, fvac, strict=True):
        contour = flux_surface(frame, r, z, target, conductors)
        values.append(safety_factor(frame, r, z, contour, fpol))
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite surface-based MAST q95')
    return result
