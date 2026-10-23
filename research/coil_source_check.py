"""Which coil currents do the measured flux loops support: release or FAIR-MAST level-2 pf_active?

For matched test shots: total predicted loop flux = (base psi - ext(release currents)) + ext(X),
X in {release, level-2}; the plasma part is held fixed. Metric: median over loops of the per-loop
R2 across frames (as research.match_verify). Inputs: PF currents from level-2 pf_active (fetched
separately to PF_DIR/<shot>.npz with coil_time, coil_current, coil_channel). No equilibrium data.
Usage: python -m research.coil_source_check DATA MAG PF_DIR MATCH.json SOLVE_DIR
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from research.fairmast_harness import external

PF = {"sol": ("SOL",), "p2u": ("P2IU FEED", "P2OU FEED"), "p2l": ("P2IL FEED", "P2OL FEED"),
      "p3u": ("P3U FEED",), "p3l": ("P3L FEED",), "p4u": ("P4U FEED",), "p4l": ("P4L FEED",),
      "p5u": ("P5U FEED",), "p5l": ("P5L FEED",), "p6u": ("P6U",), "p6l": ("P6L",)}


def level2_row(row, pf):
    """Release row with its PF/solenoid currents replaced by level-2 pf_active (kA, release time base)."""
    t = np.asarray(row["magnetics_time"], float)
    ch = {str(c): pf["coil_current"][k] / 1000 for k, c in enumerate(pf["coil_channel"])}
    tt = pf["coil_time"] * 1000
    new = dict(row)
    for key, names in PF.items():
        new[f"magnetics_{key}_current"] = np.interp(t, tt, sum(ch[n] for n in names)).astype(np.float32)
    return new


def loop_r2(psi, row, mag):
    r, z = row["efit_grid_R"], row["efit_grid_Z"]
    geo = list(mag["flux_loop_geometry_channel"])
    idx = [geo.index(c) for c in mag["flux_loop_channel"]]
    rows = (mag["flux_loop_z"][idx] - z[0]) / (z[1] - z[0]); cols = (mag["flux_loop_r"][idx] - r[0]) / (r[1] - r[0])
    pred = np.array([map_coordinates(p, [rows, cols], order=1) for p in psi]) * 2 * np.pi
    t = np.asarray(row["efit_times"], float) / 1000
    meas = np.array([np.interp(t, mag["mag_time"], c) for c in mag["flux_loop_flux"]]).T
    out = {}
    for j, n in enumerate(mag["flux_loop_channel"]):
        m = meas[:, j]
        if str(n).upper().startswith("CC04") or not np.isfinite(m).all() or np.std(m) < 1e-4:
            continue
        out[str(n)] = 1 - np.sum((pred[:, j] - m) ** 2) / np.sum((m - m.mean()) ** 2)
    return out


def one(job):
    data, mag_dir, pf_dir, solve, m, e = job
    name = Path(e["path"]).stem
    pf_f = pf_dir / f"{m['shot']}.npz"
    if not pf_f.exists():
        return None
    row = dict(np.load(data / "inputs" / f"{name}.npz", allow_pickle=False))
    mag = np.load(mag_dir / "magnetics" / f"{name}.npz")
    stored = np.load(solve / "shots" / f"{m['i']:04d}.npz")
    cfg = json.loads(str(stored["meta"]))["config"]
    psi = stored["psirz"].astype(float)
    r, z = row["efit_grid_R"], row["efit_grid_Z"]
    ext_rel = external(row, cfg, r, z)
    ext_l2 = external(level2_row(row, np.load(pf_f)), cfg, r, z)
    a, b = loop_r2(psi, row, mag), loop_r2(psi - ext_rel + ext_l2, row, mag)
    return m["shot"], np.median(list(a.values())), np.median(list(b.values()))


def main():
    data, mag_dir, pf_dir, match, solve = (Path(x) for x in sys.argv[1:6])
    shots = json.loads((data / "data.json").read_text())["shots"]
    ms = json.loads(match.read_text())
    with ProcessPoolExecutor(12) as ex:
        res = [x for x in ex.map(one, [(data, mag_dir, pf_dir, solve, m, shots[m["i"]]) for m in ms]) if x]
    a = np.array([x[1] for x in res]); b = np.array([x[2] for x in res])
    print(f"shots {len(res)} | loop R2 median release {np.median(a):.3f}  level-2 {np.median(b):.3f} | "
          f"level-2 better on {np.mean(b > a):.0%} of shots")
    for s, x, y in res[:12]:
        print(s, round(x, 3), round(y, 3))


if __name__ == "__main__":
    main()
