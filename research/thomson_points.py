"""Thomson electron temperature as (R, Z, Te) points per EFIT frame, for both machines.

Positions are fixed chord coordinates; each channel is linearly interpolated in time to the
EFIT times between valid samples (no extrapolation beyond the measured window).
"""

import numpy as np


def _interp_channels(times, values, centers):
    out = np.full((len(centers), values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        ok = np.isfinite(times) & np.isfinite(values[:, j]) & (values[:, j] > 0)
        if ok.sum() >= 2:
            t, v = times[ok], values[ok, j]
            inside = (centers >= t[0]) & (centers <= t[-1])
            out[inside, j] = np.interp(centers[inside], t, v)
    return out


def d3d_points(row, quantity="Te"):
    """Core (vertical chord) and tangential (midplane) DIII-D channels."""
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    cr, cz = np.asarray(row["thomson_chord_R"], float), np.asarray(row["thomson_chord_Z"], float)
    centers = np.asarray(row["efit_times"], float)
    rs, zs, tes = [], [], []
    for system, tag in (("core", "TS_core"), ("edge", "TS_tangential")):
        sel = np.char.startswith(names, tag)
        te = np.asarray(row[f"thomson_{system}_{quantity}"], float)
        if te.ndim != 2 or te.shape[1] != sel.sum():
            continue
        tes.append(_interp_channels(np.asarray(row[f"thomson_{system}_times"], float), te, centers))
        rs.append(cr[sel])
        zs.append(cz[sel])
    if not tes:
        return np.zeros(0), np.zeros(0), np.zeros((len(centers), 0))
    return np.concatenate(rs), np.concatenate(zs), np.concatenate(tes, axis=1)


def mast_points(row, quantity="Te"):
    """Midplane core channels; data channel 0 has no coordinate (documented ghost)."""
    radius = np.asarray(row["thomson_core_R"], float)
    te = np.asarray(row[f"thomson_core_{quantity}"], float)[:, 1:]
    centers = np.asarray(row["efit_times"], float)
    return radius, np.zeros_like(radius), _interp_channels(
        np.asarray(row["thomson_core_times"], float), te, centers)
