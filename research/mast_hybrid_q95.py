"""Bounded q diagnostic with direct external and spline plasma derivatives."""
import numpy as np
from scipy.interpolate import RectBivariateSpline

from participant.mast_inputs import SIGNALS
from research.mast_unit_current import response


def gradient(psi, r, z, points):
    psi, r, z, points = map(np.asarray, (psi, r, z, points))
    if (psi.shape != (len(z), len(r)) or points.ndim != 2
            or points.shape[1] != 2 or not all(np.isfinite(a).all()
                for a in (psi, r, z, points)) or np.any(r <= 0)
            or np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0)
            or np.any(points[:, 0] < r[0]) or np.any(points[:, 0] > r[-1])
            or np.any(points[:, 1] < z[0]) or np.any(points[:, 1] > z[-1])):
        raise ValueError('Invalid bounded spline field/query')
    spline = RectBivariateSpline(z, r, psi)
    return np.stack([spline.ev(points[:, 1], points[:, 0], dy=1),
                     spline.ev(points[:, 1], points[:, 0], dx=1)])


def integrate(contour, derivatives, fpol):
    contour, derivatives = np.asarray(contour), np.asarray(derivatives)
    if (contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 4
            or derivatives.shape != (2, len(contour)-1)
            or not np.isfinite(contour).all() or not np.isfinite(derivatives).all()
            or np.any(contour[:, 0] <= 0) or not np.isfinite(fpol) or fpol <= 0
            or not np.array_equal(contour[0], contour[-1])):
        raise ValueError('Invalid q contour/derivatives/field normalization')
    norm = np.hypot(*derivatives)
    length = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    if np.any(norm <= 0) or np.any(length <= 0):
        raise ValueError('Degenerate q gradient/segment')
    mid = (contour[1:]+contour[:-1])/2
    value = float(fpol/(2*np.pi)*np.sum(length/(mid[:, 0]*norm)))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Nonpositive/nonfinite q')
    return value


def external_grid(row, currents, kernels):
    field = np.zeros((65, 65))
    for column, unit in kernels:
        field += currents[column]*1000*unit.reshape(65, 65)
    if not np.isfinite(field).all():
        raise ValueError('Nonfinite original external grid field')
    return field


def external_gradient(row, currents, points, h, deadline):
    r, z = points.T
    names = np.asarray(row['coil_input_column']).astype(str)
    fields = np.zeros((3, len(r)))
    for name in np.unique(names):
        selected = names == name
        unit = response(*(row[key][selected] for key in
            ('coil_R', 'coil_Z', 'coil_width', 'coil_height')), r, z, h, deadline)
        normal = 1 if name == 'magnetics_sol_current' else selected.sum()
        fields += unit/normal*currents[SIGNALS.index(name.removeprefix('magnetics_'))]*1000
    if not np.isfinite(fields).all():
        raise ValueError('Nonfinite direct external field')
    # Both radial steps share the analytic vertical flux derivative.
    return np.stack([np.stack([r*fields[1], -r*fields[0]]),
                     np.stack([r*fields[2], -r*fields[0]])])
