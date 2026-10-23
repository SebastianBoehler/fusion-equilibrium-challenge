"""V1: independent identity check of FAIR-MAST matches via total flux-loop agreement.

The measured total loop flux of the matched FAIR-MAST shot is dominated by the coil currents.
If the match is right, it must agree with our solver's flux (built from the RELEASE coil currents)
at the loop positions. Per shot: mean over loops of the per-loop R2 across frames of total flux.
Predeclared acceptance (23 Oct): R2 >= 1st percentile of the R2 among matches with dist < 0.002.
Usage: python -m research.match_verify DATA MAG MATCH.json SOLVE_DIR OUT.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def one(job):
    data, mag_dir, solve, m, e = job
    name = Path(e["path"]).stem
    f = mag_dir / "magnetics" / f"{name}.npz"
    if not f.exists():
        return m["i"], np.nan
    row = np.load(data / "inputs" / f"{name}.npz")
    mag = np.load(f)
    psi = np.load(solve / "shots" / f"{m['i']:04d}.npz")["psirz"].astype(float)
    r, z = row["efit_grid_R"], row["efit_grid_Z"]
    geo = list(mag["flux_loop_geometry_channel"])
    idx = [geo.index(c) for c in mag["flux_loop_channel"]]
    rows = (mag["flux_loop_z"][idx] - z[0]) / (z[1] - z[0])
    cols = (mag["flux_loop_r"][idx] - r[0]) / (r[1] - r[0])
    pred = np.array([map_coordinates(p, [rows, cols], order=1) for p in psi]) * 2 * np.pi
    t = row["efit_times"].astype(float) / 1000
    meas = np.array([np.interp(t, mag["mag_time"], ch) for ch in mag["flux_loop_flux"]]).T
    vals = []
    for j, n in enumerate(mag["flux_loop_channel"]):
        mj = meas[:, j]
        if str(n).upper().startswith("CC04") or not np.isfinite(mj).all() or np.std(mj) < 1e-4:
            continue
        vals.append(1 - np.sum((pred[:, j] - mj) ** 2) / np.sum((mj - mj.mean()) ** 2))
    return m["i"], float(np.median(vals)) if vals else np.nan


def main():
    data, mag_dir, match, solve, out = (Path(x) for x in sys.argv[1:6])
    shots = json.loads((data / "data.json").read_text())["shots"]
    ms = json.loads(match.read_text())
    with ProcessPoolExecutor(12) as ex:
        res = dict(ex.map(one, [(data, mag_dir, solve, m, shots[m["i"]]) for m in ms]))
    for m in ms:
        m["loop_r2"] = res[m["i"]]
    ref = np.array([m["loop_r2"] for m in ms if m["dist"] < 0.002 and np.isfinite(m["loop_r2"])])
    thr = float(np.percentile(ref, 1))
    for m in ms:
        m["verified"] = bool(np.isfinite(m["loop_r2"]) and m["loop_r2"] >= thr)
    out.write_text(json.dumps(ms, indent=0))
    d = np.array([m["dist"] for m in ms]); v = np.array([m["verified"] for m in ms]); lr = np.array([m["loop_r2"] for m in ms])
    print(f"threshold (p1 of {len(ref)} near-certain) {thr:.3f} | ref p50 {np.median(ref):.3f}")
    for lo, hi in ((0, 0.002), (0.002, 0.003), (0.003, 0.005), (0.005, 0.0075), (0.0075, 0.01)):
        s = (d >= lo) & (d < hi)
        print(f"dist [{lo}, {hi}): n {s.sum():4d} verified {v[s].sum():4d} loop R2 p10/50 {np.round(np.nanpercentile(lr[s], [10, 50]), 3) if s.any() else '-'}")


if __name__ == "__main__":
    main()
