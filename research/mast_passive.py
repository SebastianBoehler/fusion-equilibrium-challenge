"""MAST passive-current flux from measured current histories (research/passive.py).

Coil elements use the confirmed conventions: PF columns carry total ampere-turns spread over
their elements; each solenoid element carries half the feed current (mast-021). The plasma
current is one filament at (0.92 m, 0). Vessel parameters are explicit arguments because
they come from machine literature, not from data.
"""

import numpy as np

from participant.mast_inputs import SIGNALS
from research.passive import flux_response, induced_currents, inductance, wall_filaments

SOL_SCALE = 0.5


def signal_on_grid(row, name, grid_ms):
    t = np.asarray(row["magnetics_time"], float)
    v = np.asarray(row[f"magnetics_{name}"], float)
    ok = np.isfinite(t) & np.isfinite(v)
    return np.interp(grid_ms, t[ok], v[ok])


def passive_flux(row, vessel, step_ms=1.0, lead_ms=300.0):
    """Induced passive flux on the grid at the EFIT times: (T, 65*65) in Wb/rad."""
    fr, fz, res, rad = wall_filaments(**vessel)
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


SIGNAL_CHECK = SIGNALS  # imported for callers that validate signal availability
