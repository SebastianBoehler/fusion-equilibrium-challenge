"""MAST passive-current flux from measured current histories (research/passive.py).

Coil elements use the confirmed conventions: PF columns carry total ampere-turns spread over
their elements; each solenoid element carries half the feed current (mast-021). The plasma
current is one filament at (0.92 m, 0). Passive conductors come from the public EFIT
geometry (research/geometry/), with literature resistivities; nothing is fitted.
"""

import json
from pathlib import Path

import numpy as np

from research.mast_gs import SOL_SCALE
from research.passive import flux_response, induced_currents, inductance

GEOMETRY = Path(__file__).parent / "geometry/mast_pf_passive.json"
INCONEL, STEEL = 1.03e-6, 7.4e-7  # ohm m (Berkery 2021; FreeGSNKE MAST-U files)
GROUPS = {  # predeclared ablations (runs/search-space-review-20261018.md)
    "V": ("vertw", "uhorw", "lhorw"),
    "VC": ("vertw", "uhorw", "lhorw", "incon", "ring", "topcol", "botcol", "endcrown_l",
           "endcrown_u", "rodgr"),
    "ALL": ("vertw", "uhorw", "lhorw", "incon", "ring", "topcol", "botcol", "endcrown_l",
            "endcrown_u", "rodgr", "mid", "p2larm", "p2uarm", "p2ldivpl", "p2udivpl", "coil_cases"),
    # rodgr upper/lower blocks treated as graphite armour (excluded); declared before results
    "ALLg": ("vertw", "uhorw", "lhorw", "incon", "ring", "topcol", "botcol", "endcrown_l",
             "endcrown_u", "mid", "p2larm", "p2uarm", "p2ldivpl", "p2udivpl", "coil_cases"),
}


def geometry_filaments(groups, max_len=0.08):
    """Toroidal filaments from the public EFIT passive geometry. Each rectangle is split
    along its long side into pieces <= max_len (independent loops, area-weighted
    resistance rho 2 pi R / area). The mid-column graphite tile layer (rodgr, |Z| < 1.08 m)
    is toroidally segmented and excluded; the Inconel tube uses Inconel resistivity."""
    geo = json.loads(GEOMETRY.read_text())
    out = []
    for name in GROUPS[groups]:
        g = geo[name]
        for r, z, w, h in zip(g["r"], g["z"], g["width"], g["height"]):
            w, h = abs(w), abs(h)
            if name == "rodgr" and abs(z) < 1.08:
                continue
            rho = INCONEL if name == "incon" else STEEL
            n = max(1, int(np.ceil(max(w, h) / max_len)))
            for k in range(n):
                f = (k + 0.5) / n - 0.5
                fr, fz = (r + f * w, z) if w >= h else (r, z + f * h)
                area = w * h / n
                out.append((fr, fz, rho * 2 * np.pi * fr / area, max(min(w, h), max(w, h) / n) / 4))
    r, z, res, rad = (np.array(v, float) for v in zip(*out))
    return r, z, res, rad


def signal_on_grid(row, name, grid_ms):
    t = np.asarray(row["magnetics_time"], float)
    v = np.asarray(row[f"magnetics_{name}"], float)
    ok = np.isfinite(t) & np.isfinite(v)
    return np.interp(grid_ms, t[ok], v[ok])


def passive_flux(row, groups, step_ms=1.0, lead_ms=300.0):
    """Induced passive flux on the grid at the EFIT times: (T, 65*65) in Wb/rad, and the
    filament currents (T, K) in A."""
    fr, fz, res, rad = geometry_filaments(groups)
    efit = np.asarray(row["efit_times"], float)
    grid_ms = np.arange(efit.min() - lead_ms, efit.max() + step_ms, step_ms)
    names = np.asarray(row["coil_input_column"]).astype(str)
    drive = np.zeros((len(grid_ms), len(fr)))
    for column in np.unique(names):
        sel = names == column
        signal = column.removeprefix("magnetics_")
        current = signal_on_grid(row, signal, grid_ms) * 1000
        per_element = (SOL_SCALE if signal == "sol_current" else 1 / sel.sum())
        response = flux_response(row["coil_R"][sel], row["coil_Z"][sel], fr, fz).sum(axis=1)
        drive += np.outer(current * per_element, response)
    ip = signal_on_grid(row, "plasma_current", grid_ms) * 1000
    drive += np.outer(ip, flux_response(np.array([0.92]), np.array([0.0]), fr, fz)[:, 0])
    currents = induced_currents(grid_ms / 1000, drive, inductance(fr, fz, rad), res,
                                efit / 1000)
    rr, zz = np.meshgrid(row["efit_grid_R"], row["efit_grid_Z"])
    to_grid = flux_response(fr, fz, rr.ravel(), zz.ravel())
    return currents @ to_grid.T, currents

