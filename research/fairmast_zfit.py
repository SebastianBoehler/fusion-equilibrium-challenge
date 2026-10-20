"""Magnetic vertical-position estimate per frame on non-release FAIR-MAST shots (step A).

For a solved run, the plasma part of the flux (solved psi minus the run's own external flux)
is shifted rigidly by dz in [-12, 12] cm; the predicted centre-column Bz probes (CCBV,
vacuum-calibrated per probe as in research.fairmast_probe_model) and flux loops are
recomputed, and the dz minimizing the normalized residual is kept (parabolic refinement).
Output per frame: dz, the run's current-centroid height, and coil-current features for the
regression (step B). Magnetics are validation/calibration targets only; nothing here reads
equilibrium reconstructions.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

from participant.mast_inputs import SIGNALS, aligned_currents
from research.fairmast_harness import external
from research.fairmast_probe_model import coil_fields, passive_fields, probe_geometry

SHIFTS = np.arange(-0.12, 0.1201, 0.02)


def features(row):
    c = aligned_currents(row)
    g = lambda n: c[:, SIGNALS.index(n)]
    ip = np.abs(g("plasma_current"))
    asym = lambda u, l: (g(u) - g(l)) / (np.abs(g(u)) + np.abs(g(l)) + 1e-6)
    return np.column_stack([asym("p2u_current", "p2l_current"), asym("p3u_current", "p3l_current"),
                            asym("p4u_current", "p4l_current"), asym("p5u_current", "p5l_current"),
                            (g("p6u_current") - g("p6l_current")) / 2 / np.clip(ip, 1, None) * 1e3, ip / 1000])


def shot_fit(run, data, probes, i, e):
    stored = np.load(run / "shots" / f"{i:04d}.npz")
    cfg = json.loads(str(stored["meta"]))["config"]
    name = Path(e["path"]).stem
    row = dict(np.load(data / "inputs" / f"{name}.npz", allow_pickle=False))
    frames = np.load(data / "magnetics" / f"{name}.npz")["frames_ms"].astype(float)
    pr = np.load(probes / f"{e['shot']}.npz")
    r, z = row["efit_grid_R"], row["efit_grid_Z"]
    psi = stored["psirz"].astype(float)
    plasma = psi - external(row, cfg, r, z)
    R, Z = probe_geometry(pr, "ccbv")
    sol, groups = cfg.get("sol_scale", 0.5), cfg.get("passive")
    ext_bz = coil_fields(row, sol, R, Z, frames)[1] + passive_fields(row, groups, sol, R, Z, frames)[1]
    tp = pr["time"] * 1000
    ip_t = np.interp(tp, row["magnetics_time"], row["magnetics_plasma_current"])
    vac = tp[(tp > -80) & (tp < -5) & (np.abs(ip_t) < 5)]
    if len(vac) < 20:
        return None
    vac_bz = coil_fields(row, sol, R, Z, vac)[1] + passive_fields(row, groups, sol, R, Z, vac)[1]
    mv = np.array([np.interp(vac, tp, ch) for ch in pr["ccbv_field"]]).T
    meas = np.array([np.interp(frames, tp, ch) for ch in pr["ccbv_field"]]).T
    for j in range(len(R)):  # vacuum calibration per probe
        ok = np.isfinite(mv[:, j])
        if ok.sum() < 20 or np.std(vac_bz[ok, j]) < 1e-3:
            meas[:, j] = np.nan; continue
        gain, off = np.polyfit(vac_bz[ok, j], mv[ok, j], 1)
        meas[:, j] = (meas[:, j] - off) / gain if abs(gain) > 0.2 else np.nan
    rows_, cols_ = (Z - z[0]) / (z[1] - z[0]), (R - r[0]) / (r[1] - r[0])
    sz = ((z[1:] + z[:-1]) / 2)
    out = []
    feats = features(row)
    for t in range(len(frames)):
        good = np.isfinite(meas[t])
        if good.sum() < 15:
            continue
        cost = []
        for dz in SHIFTS:
            sh = ndimage.shift(plasma[t], (dz / (z[1] - z[0]), 0), order=1, mode="nearest")
            bz = np.gradient(sh, r, axis=1)
            pred = ext_bz[t] + map_coordinates(bz, [rows_, cols_], order=1) / R
            cost.append(np.mean((pred[good] - meas[t, good]) ** 2))
        cost = np.array(cost); k = int(np.argmin(cost))
        dz = SHIFTS[k]
        if 0 < k < len(SHIFTS) - 1:  # parabolic refinement
            a, b, c = cost[k - 1], cost[k], cost[k + 1]
            den = a - 2 * b + c
            if den > 0:
                dz += 0.5 * (a - c) / den * (SHIFTS[1] - SHIFTS[0])
        cells = np.clip(psi[t] - psi[t].min(), 0, None)
        zc = float((cells * z[:, None]).sum() / cells.sum())  # crude height of the flux maximum region
        iz = np.unravel_index(np.argmax(psi[t][:, (r > 0.6) & (r < 1.3)]), psi[t][:, (r > 0.6) & (r < 1.3)].shape)[0]
        out.append([i, t, dz, z[iz], *feats[t], k in (0, len(SHIFTS) - 1)])
    return out


def main():
    run, data, idx_file, out_file = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    probes = Path("run_data/fairmast/probes")
    shots = json.loads((data / "data.json").read_text())["shots"]
    rows = []
    for i in json.loads(idx_file.read_text()):
        e = shots[i]
        if not ((run / "shots" / f"{i:04d}.npz").exists() and (probes / f"{e['shot']}.npz").exists()):
            continue
        res = shot_fit(run, data, probes, i, e)
        if res:
            rows += res
    a = np.array(rows, float)
    np.save(out_file, a)
    print(f"frames {len(a)} | dz p10/50/90 cm {np.round(np.percentile(a[:, 2], [10, 50, 90]) * 100, 1)} | at scan edge {a[:, -1].mean():.2f}")


if __name__ == "__main__":
    main()
