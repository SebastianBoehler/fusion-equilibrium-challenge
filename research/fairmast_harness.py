"""Magnetic validation of MAST solver variants on non-release FAIR-MAST shots.

For each run directory (research.mast_gs_robust with MAST_DATA_DIR=run_data/fairmast/data),
the predicted flux 2*pi*psi (psi in Wb/rad) is interpolated at the 15 measured flux-loop
positions (centre column CC03-CC09 at R = 0.178 m; loops on the P3-P6 coils) and compared
with the measured loop flux (Wb) at the frame times. Magnetics are validation targets only.

Reported per run: pooled R2 per loop group (fixed physical scale and sign: no fitting), the
mean over loops, and R2 of inter-loop differences (CC loops minus their mean, which removes
common offsets and is most sensitive to the inboard plasma shape and position).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def loop_predictions(run, data):
    shots = json.loads((data / "data.json").read_text())["shots"]
    pred, meas, names = [], [], None
    for i, e in enumerate(shots):
        f = run / "shots" / f"{i:04d}.npz"
        if not f.exists():
            continue
        name = Path(e["path"]).stem
        row = np.load(data / "inputs" / f"{name}.npz")
        mag = np.load(data / "magnetics" / f"{name}.npz")
        r, z = row["efit_grid_R"], row["efit_grid_Z"]
        geo = list(mag["flux_loop_geometry_channel"])
        idx = [geo.index(c) for c in mag["flux_loop_channel"]]
        lr, lz = mag["flux_loop_r"][idx], mag["flux_loop_z"][idx]
        rows, cols = (lz - z[0]) / (z[1] - z[0]), (lr - r[0]) / (r[1] - r[0])
        psi = np.load(f)["psirz"].astype(float)
        p = np.array([map_coordinates(fr, [rows, cols], order=1) for fr in psi]) * 2 * np.pi
        t = mag["frames_ms"] / 1000
        m = np.array([np.interp(t, mag["mag_time"], ch) for ch in mag["flux_loop_flux"]]).T
        pred.append(p), meas.append(m)
        names = list(mag["flux_loop_channel"])
    return np.concatenate(pred), np.concatenate(meas), names


def r2(p, m):
    ok = np.isfinite(p) & np.isfinite(m)
    return 1 - ((p[ok] - m[ok]) ** 2).sum() / ((m[ok] - m[ok].mean()) ** 2).sum()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("runs", nargs="+", type=Path)
    a.add_argument("--data", type=Path, default=Path("run_data/fairmast/data"))
    args = a.parse_args()
    report = {}
    for run in args.runs:
        p, m, names = loop_predictions(run, args.data)
        per = {n: round(float(r2(p[:, j], m[:, j])), 3) for j, n in enumerate(names)}
        cc = [j for j, n in enumerate(names) if n.startswith("CC")]
        dp = p[:, cc] - p[:, cc].mean(axis=1, keepdims=True)
        dm = m[:, cc] - m[:, cc].mean(axis=1, keepdims=True)
        corr = float(np.corrcoef(p.ravel(), m.ravel())[0, 1])
        report[str(run)] = {"frames": len(p), "mean_r2": round(float(np.mean(list(per.values()))), 3),
                            "cc_shape_r2": round(float(r2(dp, dm)), 3), "pooled_corr": round(corr, 3),
                            "scale_meas_over_pred": round(float(np.median(m / p)), 3), "per_loop": per}
        print(run.name, json.dumps({k: v for k, v in report[str(run)].items() if k != "per_loop"}))
        print("   ", per)
    return report


if __name__ == "__main__":
    main()
