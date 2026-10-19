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
    a.add_argument("--probes", action="store_true")
    args = a.parse_args()
    for run in args.runs:
        pred, ext, meas, names = loop_values(run, args.data)
        total = {n: r2(pred[:, j], meas[:, j]) for j, n in enumerate(names)}
        plasma = {n: r2(pred[:, j] - ext[:, j], meas[:, j] - ext[:, j]) for j, n in enumerate(names)}
        broken = {k for k, v in total.items() if v < -5}  # e.g. CC04: wrong scale in some shots
        mean = lambda d, pre: np.nanmean([v for k, v in d.items() if k.startswith(pre) and k not in broken])
        print(f"{run.name:16s} frames {len(pred):5d} | total CC {mean(total, 'CC'):.3f} coils {mean(total, 'P'):.3f}"
              f" | plasma CC {mean(plasma, 'CC'):.3f} coils {mean(plasma, 'P'):.3f}")
        if args.probes:
            pr = probe_report(run, args.data)
            print("    probes " + " | ".join(f"{k}: R2 {v[0]:.3f} corr {v[1]:+.3f} (n={v[2]})" for k, v in pr.items()))



def probe_values(run, data, probes=Path("run_data/fairmast/probes")):
    """Predicted vs measured pickup-coil fields: CCBV (Bz, centre column), OBV (Bz) and OBR (BR),
    outboard. B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR (psi in Wb/rad)."""
    shots = json.loads((data / "data.json").read_text())["shots"]
    out = {"ccbv": [], "obv": [], "obr": []}
    for i, e in enumerate(shots):
        f, pf = run / "shots" / f"{i:04d}.npz", probes / f"{e['shot']}.npz"
        if not (f.exists() and pf.exists()):
            continue
        name = Path(e["path"]).stem
        row = np.load(data / "inputs" / f"{name}.npz")
        frames = np.load(data / "magnetics" / f"{name}.npz")["frames_ms"] / 1000
        pr = np.load(pf)
        r, z = row["efit_grid_R"], row["efit_grid_Z"]
        psi = np.load(f)["psirz"].astype(float)
        dpsi_dz, dpsi_dr = np.gradient(psi, z, r, axis=(1, 2))
        for arr, comp in (("ccbv", "z"), ("obv", "z"), ("obr", "r")):
            geo = [g.lower() for g in pr[f"{arr}_geometry_channel"]]
            idx = [geo.index(c.lower()) for c in pr[f"{arr}_channel"]]
            pr_r, pr_z = pr[f"{arr}_r"][idx], pr[f"{arr}_z"][idx]
            rows, cols = (pr_z - z[0]) / (z[1] - z[0]), (pr_r - r[0]) / (r[1] - r[0])
            grad = dpsi_dr if comp == "z" else dpsi_dz
            sign = 1.0 if comp == "z" else -1.0
            pred = np.array([sign * map_coordinates(g, [rows, cols], order=1) / pr_r for g in grad])
            meas = np.array([np.interp(frames, pr["time"], ch) for ch in pr[f"{arr}_field"]]).T
            bad = ~np.isfinite(meas).all(axis=0) | (np.nanstd(meas, axis=0) < 1e-4)
            meas[:, bad] = np.nan
            out[arr].append((pred, meas))
    return {k: tuple(np.concatenate(x) for x in zip(*v)) for k, v in out.items() if v}


def probe_report(run, data):
    res = {}
    for arr, (pred, meas) in probe_values(run, data).items():
        per = [r2(pred[:, j], meas[:, j]) for j in range(pred.shape[1])]
        good = [v for v in per if v > -5]
        corr = np.nanmedian([np.corrcoef(pred[np.isfinite(meas[:, j]), j], meas[np.isfinite(meas[:, j]), j])[0, 1]
                             for j in range(pred.shape[1]) if np.isfinite(meas[:, j]).sum() > 10])
        res[arr] = (np.nanmean(good) if good else np.nan, corr, len(good))
    return res


if __name__ == "__main__":
    main()
