"""Normalized beta from Thomson electron pressure mapped onto an equilibrium (both machines).

betaN = 100 * beta_t * a * B0 / Ip[MA], beta_t = 2 mu0 <p>_V / B0^2, with B0 the vacuum
field at the LCFS geometric centre and <p> the volume average of p(psiN). p(psiN) is the
binned mean of measured n_e T_e at Thomson points inside the LCFS (electrons only; the
caller supplies any total/electron pressure factor).
"""

import numpy as np
from scipy import ndimage

from participant.mast_green import E_CHARGE, MU0
from research.gs_picard import find_axis, lcfs_level


def beta_n_electron(phi, r, z, allowed, points, fpol, ip):
    pr, pz, pe = points  # pe in Pa
    axis = find_axis(phi, allowed, (len(z) // 2, len(r) // 2))
    level, region = lcfs_level(phi, allowed, axis)
    psin = (phi[axis] - phi) / (phi[axis] - level)
    at = ndimage.map_coordinates(phi, [(pz - z[0]) / (z[1] - z[0]), (pr - r[0]) / (r[1] - r[0])],
                                 order=1, mode="nearest")
    x = (phi[axis] - at) / (phi[axis] - level)
    ok = (x >= 0) & (x < 1) & np.isfinite(pe)
    if ok.sum() < 5:
        raise ValueError("Too few pressure points inside the LCFS")
    bins = np.linspace(0, 1, 9)
    which = np.digitize(x[ok], bins) - 1
    xs = [x[ok][which == b].mean() for b in range(8) if (which == b).any()]
    vs = [pe[ok][which == b].mean() for b in range(8) if (which == b).any()]
    prof = np.interp(np.clip(psin, 0, 1), np.r_[xs, 1.0], np.r_[vs, vs[-1] * 0.5])
    rr = np.broadcast_to(r[None, :], phi.shape)
    weight = region * rr
    p_avg = (prof * weight).sum() / weight.sum()
    cols = np.nonzero(region.any(axis=0))[0]
    r_in, r_out = r[cols[0]], r[cols[-1]]
    r0, a = (r_in + r_out) / 2, (r_out - r_in) / 2
    b0 = fpol / r0
    return 100 * (2 * MU0 * p_avg / b0 ** 2) * a * b0 / (ip / 1e6)


def electron_pressure(te_points, ne_points):
    """Combine aligned Te (eV) and ne (m^-3) point sets into p_e (Pa)."""
    (pr, pz, te), (_, _, ne) = te_points, ne_points
    return pr, pz, te * ne * E_CHARGE


def beta_n_equilibrium(phi, cells, r, z, allowed, beta0, r0, fpol, ip):
    """betaN from the solved pressure-driven current J_p = R p'(psi) (includes all species).

    J = lambda [beta0 R/r0 + (1-beta0) r0/R] g, so J_p = J * beta0 (R/r0) / [...]. p'(psi) is
    binned in psiN and integrated from the LCFS inward; <p> is the volume average.
    """
    from research.gs_picard import to_nodes
    axis = find_axis(phi, allowed, (len(z) // 2, len(r) // 2))
    level, region = lcfs_level(phi, allowed, axis)
    span = phi[axis] - level
    psin = (phi[axis] - phi) / span
    j = to_nodes(cells)
    rr = np.broadcast_to(r[None, :], phi.shape)
    denom = beta0 * rr / r0 + (1 - beta0) * r0 / rr
    pprime = np.where(region & (np.abs(denom) > 1e-9), j * (beta0 / r0) / denom, 0.0)
    bins = np.linspace(0, 1, 17)
    which = np.clip(np.digitize(psin, bins) - 1, 0, 15)
    mean_pp = np.array([pprime[region & (which == k)].mean() if (region & (which == k)).any()
                        else 0.0 for k in range(16)])
    # p(psiN) = span * integral_{psiN}^{1} p'(x) dx, on bin edges then interpolated.
    width = np.diff(bins)
    p_edges = span * np.r_[np.cumsum((mean_pp * width)[::-1])[::-1], 0.0]
    p = np.where(region, np.interp(np.clip(psin, 0, 1), bins, p_edges), 0.0)
    weight = region * rr
    p_avg = (p * weight).sum() / weight.sum()
    cols = np.nonzero(region.any(axis=0))[0]
    r_in, r_out = r[cols[0]], r[cols[-1]]
    b0 = fpol / ((r_in + r_out) / 2)
    return 100 * (2 * MU0 * p_avg / b0 ** 2) * ((r_out - r_in) / 2) * b0 / (ip / 1e6)


def beta_n_from_flux(phi, r, z, allowed, fpol, ip):
    """betaN from a stored flux map alone: J = -Delta* phi/(mu0 R), then per psiN bin a
    least-squares split J = R p'(psi) + FF'/(mu0 R); p' integrated from the LCFS inward."""
    from research.mast_proxy_d3d import delta_star
    axis = find_axis(phi, allowed, (len(z) // 2, len(r) // 2))
    level, region = lcfs_level(phi, allowed, axis)
    span = phi[axis] - level
    psin = (phi[axis] - phi) / span
    j = -delta_star(phi[None], r, z)[0] / (MU0 * r[None, :])
    inner = region & ndimage.binary_erosion(region)
    rr = np.broadcast_to(r[None, :], phi.shape)
    bins = np.linspace(0, 1, 9)
    which = np.clip(np.digitize(psin, bins) - 1, 0, 7)
    pp = np.zeros(8)
    for k in range(8):
        m = inner & (which == k)
        if m.sum() >= 4 and np.ptp(rr[m]) > 0:
            a = np.column_stack((rr[m], 1 / rr[m]))
            pp[k] = np.linalg.lstsq(a, j[m], rcond=None)[0][0]
    p_edges = span * np.r_[np.cumsum((pp * np.diff(bins))[::-1])[::-1], 0.0]
    p = np.where(region, np.interp(np.clip(psin, 0, 1), bins, p_edges), 0.0)
    weight = region * rr
    p_avg = (p * weight).sum() / weight.sum()
    cols = np.nonzero(region.any(axis=0))[0]
    r_in, r_out = r[cols[0]], r[cols[-1]]
    b0 = fpol / ((r_in + r_out) / 2)
    return 100 * (2 * MU0 * p_avg / b0 ** 2) * ((r_out - r_in) / 2) * b0 / (ip / 1e6)
