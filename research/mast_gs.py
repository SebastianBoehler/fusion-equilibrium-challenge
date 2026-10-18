"""MAST machine constants and the base allowed plasma region (input geometry only).

Solenoid convention: each solenoid element carries half the released feed current
(SOL_SCALE, mast-021; matches FAIR-MAST level 2 "AMC_SOL CURRENT" scale 0.5). Allowed
region: grid minus a 2-cell margin, minus R < r_min (centre column), minus cells within
6 cm of any PF conductor and within sol_clearance of solenoid elements.
"""

from pathlib import Path

import numpy as np

ROOT = Path("run_data")
SOL_SCALE = 0.5


def allowed_region(row, r_min=0.19, sol_clearance=0.06):
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    rr, zz = np.meshgrid(r, z)
    ok = np.zeros(rr.shape, bool)
    ok[2:-2, 2:-2] = True
    ok &= rr >= r_min
    cr, cz = np.asarray(row["coil_R"]), np.asarray(row["coil_Z"])
    sol = np.asarray(row["coil_input_column"]).astype(str) == "magnetics_sol_current"
    for k in range(len(cr)):
        clearance = sol_clearance if sol[k] else 0.06
        ok &= (rr - cr[k]) ** 2 + (zz - cz[k]) ** 2 > clearance ** 2
    return ok
