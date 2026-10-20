"""Per-frame current-peaking selection from Thomson edges (label-free test-time adaptation).

Inputs: solver runs that differ only in the Te exponent (same coils, passive model and
vertical scheme). Per frame, the candidate whose outboard midplane LCFS radius is closest to
the Thomson Te = 30 eV edge is chosen. The choice index is median-filtered over 5 frames.
Frames without a usable edge (outside 1.2-1.55 m) keep the default candidate. Output: a run
directory in mast_gs_robust format (shots/NNNN.npz with psirz, meta incl. the chosen
exponents), usable by research.fairmast_harness and research.mast_final.
--rule dip instead picks by the Ip ramp rate (current penetration): runs = [ramp-up, flat-top,
ramp-down] for dIp/dt > +0.2 %/ms, in between, < -0.2 %/ms (thresholds fixed a priori).
Usage: python -m research.peaking_select --runs DIR_0.5 DIR_0.75 DIR_1.0 --default 1 --out DIR
(MAST_DATA_DIR selects the split, as for the solver.)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from research.flux_function_check import lcfs_outboard_radius, te_edge_radius
from research.mast_gs_robust import DATA, region
from research.thomson_points import mast_points


def select_shot(task):
    index, entry, runs, default, out, rule = task
    target = out / "shots" / f"{index:04d}.npz"
    if target.exists():
        return index
    row = dict(np.load(DATA / "inputs" / (Path(entry["path"]).stem + ".npz"), allow_pickle=False))
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    allowed = region(row, "inner")
    pr, _, te = mast_points(row)
    stored = [np.load(Path(d) / "shots" / f"{index:04d}.npz") for d in runs]
    psis = [s["psirz"] for s in stored]
    choice = np.full(len(psis[0]), default)
    if rule == "dip":  # current-penetration rule: runs = [ramp-up, flat-top, ramp-down]
        from participant.mast_inputs import SIGNALS, aligned_currents
        t_ms = np.asarray(row["efit_times"], float)
        ip = aligned_currents(row)[:, SIGNALS.index("plasma_current")]
        rate = np.gradient(ip, t_ms) / max(np.abs(ip).max(), 1) * 100  # % of Ip max per ms
        choice = np.where(rate > 0.2, 0, np.where(rate < -0.2, 2, 1))
    for t in range(len(choice) if rule == "edge" else 0):
        edge = te_edge_radius(pr, te[t])
        if not (np.isfinite(edge) and 1.2 <= edge <= 1.55):
            continue
        miss = []
        for p in psis:
            try:
                miss.append(abs(lcfs_outboard_radius(p[t].astype(float), r, z, allowed) - edge))
            except ValueError:
                miss.append(np.inf)
        if np.isfinite(min(miss)):
            choice[t] = int(np.argmin(miss))
    if len(choice) >= 5 and rule == "edge":
        choice = ndimage.median_filter(choice, size=5, mode="nearest")
    psi = np.array([psis[c][t] for t, c in enumerate(choice)])
    meta = json.loads(str(stored[default]["meta"]))
    meta["peaking_choice"] = choice.tolist()
    meta["config"] = dict(meta["config"], peaking_select=[str(d) for d in runs])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target.with_suffix(".partial.npz"), psirz=psi, meta=json.dumps(meta),
             beta0=stored[default]["beta0"], betan_share=stored[default]["betan_share"])
    target.with_suffix(".partial.npz").replace(target)
    return index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--default", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shots", default="all")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rule", choices=("edge", "dip"), default="edge")
    a = p.parse_args()
    shots = json.loads((DATA / "data.json").read_text())["shots"]
    idx = list(range(len(shots))) if a.shots == "all" else json.loads(Path(a.shots).read_text())
    with ProcessPoolExecutor(a.workers) as pool:
        done = list(pool.map(select_shot, [(i, shots[i], a.runs, a.default, a.out, a.rule) for i in idx]))
    counts = np.bincount(np.concatenate([json.loads(str(np.load(a.out / "shots" / f"{i:04d}.npz")["meta"]))["peaking_choice"]
                                         for i in done]), minlength=len(a.runs))
    print(json.dumps({"shots": len(done), "choice_counts": counts.tolist()}))


if __name__ == "__main__":
    main()
