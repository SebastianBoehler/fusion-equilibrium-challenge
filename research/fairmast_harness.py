"""Magnetic validation of MAST solver variants on non-release FAIR-MAST shots.

For each run directory (research.mast_gs_robust with MAST_DATA_DIR=run_data/fairmast/data),
the predicted flux 2*pi*psi (psi in Wb/rad) is interpolated at the measured flux-loop
positions (centre column CC03-CC09 at R = 0.178 m; loops on the P3-P6 coils) and compared
with the measured loop flux (Wb) at the frame times. Magnetics are validation targets only.
Channels that are flat or missing in a shot (e.g. CC04) are excluded for that shot.

Metrics (fixed physical scale and sign; nothing is fitted):
- total: pooled R2 of measured vs predicted loop flux, mean over loops;
- plasma: the run's own external flux (coils + passive, same config) is subtracted from both
  measurement and prediction, so only the plasma contribution is compared. This is the part
  the plasma model controls (current position, profile, vertical class).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from participant.mast_green import coil_kernel
from participant.mast_inputs import SIGNALS, aligned_currents


def external(row, cfg, r, z):
    currents = aligned_currents(row)
    ext = np.zeros((len(currents), r.size * z.size))
    for column, response in coil_kernel(row):
        scale = cfg.get("sol_scale", 0.5) if SIGNALS[column] == "sol_current" else 1.0
        ext += currents[:, column, None] * 1000 * scale * response[None]
    if cfg.get("passive"):
        from research.mast_passive import passive_flux
        ext += passive_flux(row, cfg["passive"], sol_scale=cfg.get("sol_scale", 0.5))[0]
    return ext.reshape(-1, len(z), len(r))


def loop_values(run, data):
    shots = json.loads((data / "data.json").read_text())["shots"]
    out, names = [], None
    for i, e in enumerate(shots):
        f = run / "shots" / f"{i:04d}.npz"
        if not f.exists():
            continue
        stored = np.load(f)
        cfg = json.loads(str(stored["meta"]))["config"]
        name = Path(e["path"]).stem
        row = dict(np.load(data / "inputs" / f"{name}.npz", allow_pickle=False))
        mag = np.load(data / "magnetics" / f"{name}.npz")
        r, z = row["efit_grid_R"], row["efit_grid_Z"]
        geo = list(mag["flux_loop_geometry_channel"])
        idx = [geo.index(c) for c in mag["flux_loop_channel"]]
        rows = (mag["flux_loop_z"][idx] - z[0]) / (z[1] - z[0])
        cols = (mag["flux_loop_r"][idx] - r[0]) / (r[1] - r[0])
        at = lambda maps: np.array([map_coordinates(m, [rows, cols], order=1) for m in maps]) * 2 * np.pi
        pred, ext = at(stored["psirz"].astype(float)), at(external(row, cfg, r, z))
        t = mag["frames_ms"] / 1000
        meas = np.array([np.interp(t, mag["mag_time"], ch) for ch in mag["flux_loop_flux"]]).T
        bad = ~np.isfinite(meas).all(axis=0) | (np.nanstd(meas, axis=0) < 1e-4)
        meas[:, bad] = np.nan
        out.append((pred, ext, meas))
        names = list(mag["flux_loop_channel"])
    pred, ext, meas = (np.concatenate(x) for x in zip(*out))
    return pred, ext, meas, names


def r2(p, m):
    ok = np.isfinite(p) & np.isfinite(m)
    return float(1 - ((p[ok] - m[ok]) ** 2).sum() / ((m[ok] - m[ok].mean()) ** 2).sum())


def main():
    a = argparse.ArgumentParser()
    a.add_argument("runs", nargs="+", type=Path)
    a.add_argument("--data", type=Path, default=Path("run_data/fairmast/data"))
    args = a.parse_args()
    for run in args.runs:
        pred, ext, meas, names = loop_values(run, args.data)
        total = {n: r2(pred[:, j], meas[:, j]) for j, n in enumerate(names)}
        plasma = {n: r2(pred[:, j] - ext[:, j], meas[:, j] - ext[:, j]) for j, n in enumerate(names)}
        broken = {k for k, v in total.items() if v < -5}  # e.g. CC04: wrong scale in some shots
        mean = lambda d, pre: np.nanmean([v for k, v in d.items() if k.startswith(pre) and k not in broken])
        print(f"{run.name:16s} frames {len(pred):5d} | total CC {mean(total, 'CC'):.3f} coils {mean(total, 'P'):.3f}"
              f" | plasma CC {mean(plasma, 'CC'):.3f} coils {mean(plasma, 'P'):.3f}")


if __name__ == "__main__":
    main()
