"""Free-boundary Grad-Shafranov equilibrium with fixed external flux (Picard iteration).

phi = phi_ext + K @ J, with the plasma current a function of normalized flux inside the
last closed surface. K is the toroidal-filament Green kernel, so no boundary condition is
needed. Convention: positive Ip gives a magnetic axis at the maximum of phi.

The LCFS is the lowest flux level whose axis-connected region of {phi >= level} still avoids
the edge of the allowed region (an X-point or limiter contact). Vertical position is set each
iteration by rigidly shifting J to zero the external vertical force, which stabilizes the
otherwise vertically unstable fixed-coil iteration.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from participant.mast_green import plasma_kernel


@dataclass(frozen=True)
class Profile:
    alpha_m: float = 1.0   # J ~ (1 - psiN^alpha_m)^alpha_n
    alpha_n: float = 2.0
    beta0: float = 0.4     # pressure-driven (R/R0) vs poloidal-current (R0/R) share
    iterations: int = 40
    relax: float = 0.5
    te_power: float = 1.5  # kinetic profile J ~ Te^te_power (1.5 = Spitzer)
    beta0_range: tuple = (-1.0, 4.0)
    vertical: str = "force"  # "force": zero external vertical force; "zero": centroid at Z=0
    efit_c: float | None = None  # if set: g = (1 - psiN)(1 + c psiN), the Lao n=1 family


def to_cells(field):
    return (field[..., 1:, 1:] + field[..., :-1, 1:] + field[..., 1:, :-1]
            + field[..., :-1, :-1]) / 4


def to_nodes(cells):
    """Average cell values back onto nodes (edges use the adjacent cells)."""
    padded = np.pad(cells, 1, mode="edge")
    return to_cells(padded)


def lcfs_level(phi, allowed, axis):
    """Lowest level whose axis component of {phi >= level} & allowed stays off the edge."""
    edge = allowed & ~ndimage.binary_erosion(allowed)
    lo, hi = phi[allowed].min(), phi[axis]
    region = None
    for _ in range(30):
        mid = (lo + hi) / 2
        labels, _ = ndimage.label((phi >= mid) & allowed)
        comp = labels == labels[axis]
        if (comp & edge).any():
            lo = mid
        else:
            hi, region = mid, comp
    if region is None:
        raise ValueError("No closed flux surface around the axis")
    return hi, region


def find_axis(phi, allowed, guess):
    """Local flux maximum in the allowed region nearest the previous axis."""
    peaks = (phi == ndimage.maximum_filter(phi, size=3)) & allowed
    pr, pc = np.nonzero(peaks)
    if not len(pr):
        raise ValueError("No magnetic axis candidate")
    k = np.argmin((pr - guess[0]) ** 2 + (pc - guess[1]) ** 2)
    return int(pr[k]), int(pc[k])


def vertical_shift(cells, sr, sz, br_cells, dz):
    """Rigid vertical shift (in cells) that zeroes the external vertical force on J."""
    shifts = np.linspace(-12, 12, 49)
    force = []
    for s in shifts:
        moved = ndimage.shift(cells, (s, 0), order=1, mode="constant")
        force.append((moved * sr * br_cells).sum())
    force = np.array(force)
    sign = np.nonzero(np.diff(np.sign(force)))[0]
    if not len(sign):
        return float(shifts[np.argmin(np.abs(force))])
    k = sign[np.argmin(np.abs(shifts[sign]))]
    f0, f1 = force[k], force[k + 1]
    return float(shifts[k] - f0 * (shifts[k + 1] - shifts[k]) / (f1 - f0))


def midplane_edges(region, r, z, z_mid=0.0):
    row = region[int(np.argmin(np.abs(z - z_mid)))]
    cols = np.nonzero(row)[0]
    return (r[cols[0]], r[cols[-1]]) if len(cols) else (np.nan, np.nan)


def kinetic_shape(phi, axis_value, level, r, z, points, region, psin, power=1.5):
    """Current shape g(psiN) ~ Te^1.5 from Thomson points mapped onto the present flux."""
    pr, pz, te = points
    rows = (pz - z[0]) / (z[1] - z[0])
    cols = (pr - r[0]) / (r[1] - r[0])
    at = ndimage.map_coordinates(phi, [rows, cols], order=1, mode="nearest")
    x = (axis_value - at) / (axis_value - level)
    ok = (x >= 0) & (x < 1)
    if ok.sum() < 5:
        return None
    edges = np.linspace(0, 1, 9)
    which = np.digitize(x[ok], edges) - 1
    w = te[ok] ** power
    xs, vs = [], []
    for b in range(8):
        if (which == b).any():
            xs.append(x[ok][which == b].mean())
            vs.append(w[which == b].mean())
    xs, vs = np.array(xs + [1.0]), np.array(vs + [0.0])
    return np.where(region, np.interp(psin, xs, vs), 0.0)


def solve(phi_ext, ip, r, z, allowed, profile=Profile(), init=None, edges=None, axis_r=None,
          points=None, info=None, z_target=None, edge_out=None, bz_sign=1.0, bz_fixed=0.0,
          bz_gain=0.05):
    """Equilibrium flux (65x65, axis maximum) and the plasma current cells for one frame.

    edges: optional measured (R_in, R_out) of the LCFS on the Z=0 midplane. The pressure
    share beta0 is then adapted each iteration so the LCFS midplane centre matches them.
    axis_r: optional measured magnetic-axis radius (e.g. the Thomson Te peak); beta0 is then
    adapted so the sub-grid axis radius matches it.
    points: optional Thomson (R, Z, Te) arrays; the current shape becomes Te^1.5 as a function
    of normalized flux (Spitzer, Ohmic approximation) instead of the parametric profile.
    info: optional dict filled with the final pressure share beta0 and reference radius r0.
    z_target: optional current-centroid height (m) to pin, overriding profile.vertical.
    edge_out: optional measured outboard midplane LCFS radius (m). A uniform vertical-field
    correction psi_c = bz R^2 / 2 (stand-in for unmeasured passive currents) is adapted so the
    LCFS crosses the midplane there; bz_sign orients the update (calibrated once).
    """
    kernel, sr, sz = plasma_kernel(tuple(r), tuple(z))
    shape = (len(z) - 1, len(r) - 1)
    sr, sz = sr.reshape(shape), sz.reshape(shape)
    dA = (r[1] - r[0]) * (z[1] - z[0])
    br = -np.gradient(phi_ext, z, axis=0) / r[None, :]
    br_cells = to_cells(br)
    if init is None:
        r0, z0, w = r.mean(), 0.0, (r[-1] - r[0]) / 8
        init = np.exp(-((sr - r0) ** 2 + (sz - z0) ** 2) / (2 * w * w)) * to_cells(allowed)
    cells = init * ip / (init.sum() * dA)
    r0 = (cells * sr).sum() / cells.sum()
    z0 = (cells * sz).sum() / cells.sum()
    axis = (int(np.argmin(np.abs(z - z0))), int(np.argmin(np.abs(r - r0))))
    beta0 = profile.beta0
    bz = float(bz_fixed)
    rr2 = (r[None, :] ** 2) / 2
    for _ in range(profile.iterations):
        phi = phi_ext + bz * rr2 + (kernel @ (cells.ravel() * dA)).reshape(phi_ext.shape)
        axis = find_axis(phi, allowed, axis)
        level, region = lcfs_level(phi, allowed, axis)
        if edges is not None:
            r_in, r_out = midplane_edges(region, r, z)
            if np.isfinite(r_in):
                miss = (edges[0] + edges[1] - r_in - r_out) / (edges[1] - edges[0])
                beta0 = float(np.clip(beta0 + 2.0 * miss, *profile.beta0_range))
        if edge_out is not None:
            row = phi[int(np.argmin(np.abs(z)))]
            r_out = np.nan
            for j in range(axis[1], len(r) - 1):
                if row[j] >= level > row[j + 1]:
                    r_out = r[j] + (row[j] - level) / (row[j] - row[j + 1]) * (r[1] - r[0])
                    break
            if np.isfinite(r_out):
                bz = float(np.clip(bz + bz_sign * bz_gain * (edge_out - r_out), -0.05, 0.05))
        if axis_r is not None:
            iz, ir = axis
            if 0 < ir < len(r) - 1:
                a, b, c = phi[iz, ir - 1], phi[iz, ir], phi[iz, ir + 1]
                frac = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0.0
                r_axis = r[ir] + frac * (r[1] - r[0])
                beta0 = float(np.clip(beta0 + 4.0 * (axis_r - r_axis), *profile.beta0_range))
        psin = np.clip((phi[axis] - phi) / (phi[axis] - level), 0, 1)
        shape_fn = None
        if points is not None:
            shape_fn = kinetic_shape(phi, phi[axis], level, r, z, points, region, psin,
                                     profile.te_power)
        if shape_fn is None and profile.efit_c is not None:
            shape_fn = np.where(region, np.clip((1 - psin) * (1 + profile.efit_c * psin), 0, None), 0)
        if shape_fn is None:
            shape_fn = np.where(region, (1 - psin ** profile.alpha_m) ** profile.alpha_n, 0)
        rr = r[None, :]
        new = to_cells(shape_fn * (beta0 * rr / r0 + (1 - beta0) * r0 / rr))
        if new.sum() <= 0:
            raise ValueError("Empty plasma region")
        new *= ip / (new.sum() * dA)
        if z_target is not None:
            shift = (z_target - (new * sz).sum() / new.sum()) / (z[1] - z[0])
        elif profile.vertical == "zero":
            shift = -((new * sz).sum() / new.sum()) / (z[1] - z[0])
        else:
            shift = vertical_shift(new, sr, sz, br_cells, z[1] - z[0])
        new = ndimage.shift(new, (shift, 0), order=1, mode="constant")
        new *= ip / (new.sum() * dA)
        cells = (1 - profile.relax) * cells + profile.relax * new
        r0 = (cells * sr).sum() / cells.sum()
    phi = phi_ext + bz * rr2 + (kernel @ (cells.ravel() * dA)).reshape(phi_ext.shape)
    if info is not None:
        info.update(beta0=beta0, r0=r0, bz=bz)
    return phi, cells
