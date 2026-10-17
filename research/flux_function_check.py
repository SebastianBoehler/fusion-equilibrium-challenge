"""Label-free MAST check: Te must be a flux function (inboard/outboard symmetry in psiN).

For a candidate flux map, map each midplane Thomson channel to psiN; split channels at the
magnetic axis into inboard and outboard branches; interpolate the inboard Te(psiN) onto the
outboard psiN values inside the LCFS; report the normalized RMS mismatch. A flux map whose
interior geometry is right gives matching branches. No labels, demos or envelope.
"""

import numpy as np
from scipy import ndimage

from research.gs_picard import find_axis, lcfs_level


def mismatch(phi, r, z, allowed, pr, te):
    ok = np.isfinite(te)
    if ok.sum() < 20:
        return np.nan
    axis = find_axis(phi, allowed, (len(z) // 2, int(np.argmin(np.abs(r - 0.92)))))
    level, _ = lcfs_level(phi, allowed, axis)
    rows = (0.0 - z[0]) / (z[1] - z[0]) * np.ones(ok.sum())
    cols = (pr[ok] - r[0]) / (r[1] - r[0])
    at = ndimage.map_coordinates(phi, [rows, cols], order=1, mode="nearest")
    x = (phi[axis] - at) / (phi[axis] - level)
    y = ndimage.median_filter(te[ok][np.argsort(pr[ok])], size=5, mode="nearest")
    x, rr = x[np.argsort(pr[ok])], np.sort(pr[ok])
    r_axis = r[axis[1]]
    inner = (rr < r_axis) & (x >= 0) & (x < 1)
    outer = (rr > r_axis) & (x >= 0) & (x < 1)
    if inner.sum() < 5 or outer.sum() < 5:
        return np.nan
    order = np.argsort(x[inner])
    predicted = np.interp(x[outer], x[inner][order], y[inner][order])
    return float(np.sqrt(np.mean((predicted - y[outer]) ** 2)) / max(y.max(), 1.0))


def te_edge_radius(pr, te, level=30.0):
    """Outermost midplane radius where median-filtered Te falls through `level` eV."""
    ok = np.isfinite(te)
    if ok.sum() < 20:
        return np.nan
    order = np.argsort(pr[ok])
    x, y = pr[ok][order], ndimage.median_filter(te[ok][order], size=5, mode="nearest")
    peak = int(np.argmax(y))
    outer = np.nonzero(y[peak:] < level)[0]
    if not len(outer):
        return np.nan
    k = peak + outer[0]
    return float(np.interp(level, [y[k], y[k - 1]], [x[k], x[k - 1]]))


def lcfs_outboard_radius(phi, r, z, allowed):
    """Outboard midplane crossing of the axis-enclosing LCFS level (sub-grid)."""
    axis = find_axis(phi, allowed, (len(z) // 2, int(np.argmin(np.abs(r - 0.92)))))
    level, _ = lcfs_level(phi, allowed, axis)
    row = phi[int(np.argmin(np.abs(z)))]
    for j in range(axis[1], len(r) - 1):
        if row[j] >= level > row[j + 1]:
            return float(r[j] + (row[j] - level) / (row[j] - row[j + 1]) * (r[1] - r[0]))
    return np.nan
