"""Split-sensor validation of research.mag_tta on FAIR-MAST non-test shots (predeclared 22 Oct).

For each fit/eval split, the TTA is fitted on the FIT sensor groups only and scored on the EVAL
groups (never seen by the fit), against the base run (variant 0, dz = 0). Metric per group: mean
over channels of the pooled per-channel R2 of the plasma contribution (harness convention).
Gate: >= +0.02 on the eval groups on tuning AND holdout, no group worse by > 0.01.
Usage: python -m research.mag_tta_eval DATA IDX.json PROBES OUT.json RUN0 RUN1 ... [--workers N]
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from research.mag_tta import blend, fit_frame, load_maps, predict, sensors, cl_at, cl_fit

SPLITS = {"loops->probes": (("CC", "P"), ("ccbv", "obv", "obr")),
          "CC+probes->P": (("CC", "ccbv", "obv", "obr"), ("P",)),
          "P+probes->CC": (("P", "ccbv", "obv", "obr"), ("CC",)),
          "all->all": (("CC", "P", "ccbv", "obv", "obr"), ("CC", "P", "ccbv", "obv", "obr"))}


def shot(args):
    data, probes, runs, i, e = args
    name = Path(e["path"]).stem
    try:
        maps, cfg = load_maps(runs, i)
    except FileNotFoundError:
        return None
    row = dict(np.load(data / "inputs" / f"{name}.npz", allow_pickle=False))
    mag = np.load(data / "magnetics" / f"{name}.npz")
    pf = probes / f"{e['shot']}.npz"
    pr = np.load(pf) if pf.exists() else None
    frames = mag["frames_ms"].astype(float)
    r, z = row["efit_grid_R"], row["efit_grid_Z"]
    groups, ext = sensors(row, mag, pr, cfg, frames)
    plasma = maps - ext[None]                                     # (K, T, 65, 65)
    scale = {g: np.nanstd(s["meas"], axis=0) + 1e-9 for g, s in groups.items()}
    out = defaultdict(lambda: defaultdict(list))                  # split -> (group,name) -> [(pred, meas)]
    fits = defaultdict(list)
    clc = {}
    for t in range(plasma.shape[1]):
        for g in groups.values():
            g["meas_t"] = g["meas"][t]
        base_pred = {g: predict(plasma[0, t], r, z, s) for g, s in groups.items()}
        for split, (fit, ev) in SPLITS.items():
            w, dz, _ = fit_frame(plasma[:, t], r, z, groups, fit, scale)
            m = blend(plasma[:, t], w, dz, z)
            fits[split].append([*w, dz])
            resid = {g: groups[g]["meas_t"] - predict(m, r, z, groups[g]) for g in fit if g in groups}
            c = cl_fit(resid, groups, fit, scale, r, z, clc)
            for g in ev:
                if g not in groups:
                    continue
                p = predict(m, r, z, groups[g])
                if g not in clc:
                    clc[g] = cl_at(groups[g], r, z)
                pv = p + clc[g] @ c
                for j, n in enumerate(groups[g]["names"]):
                    out[split][(g, n)].append((p[j], base_pred[g][j], groups[g]["meas_t"][j], pv[j]))
    return {s: dict(v) for s, v in out.items()}, {s: np.array(v) for s, v in fits.items()}


def r2(p, m):
    ok = np.isfinite(p) & np.isfinite(m)
    return 1 - ((p[ok] - m[ok]) ** 2).sum() / ((m[ok] - m[ok].mean()) ** 2).sum()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("data", type=Path); a.add_argument("idx", type=Path); a.add_argument("probes", type=Path)
    a.add_argument("out", type=Path); a.add_argument("runs", nargs="+"); a.add_argument("--workers", type=int, default=4)
    args = a.parse_args()
    shots = json.loads((args.data / "data.json").read_text())["shots"]
    idx = json.loads(args.idx.read_text())
    jobs = [(args.data, args.probes, args.runs, i, shots[i]) for i in idx]
    pooled = defaultdict(lambda: defaultdict(list)); fits = defaultdict(list)
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(shot, jobs):
            if res is None:
                continue
            for s, d in res[0].items():
                for k, v in d.items():
                    pooled[s][k] += v
            for s, f in res[1].items():
                fits[s].append(f)
    report = {}
    for s, d in pooled.items():
        per = defaultdict(lambda: [[], [], []])
        for (g, n), v in d.items():
            v = np.array(v)
            if len(v) < 50 or r2(v[:, 1], v[:, 2]) < -5:          # broken / sparse channel
                continue
            per[g][0].append(r2(v[:, 1], v[:, 2])); per[g][1].append(r2(v[:, 0], v[:, 2])); per[g][2].append(r2(v[:, 3], v[:, 2]))
        f = np.concatenate(fits[s])
        report[s] = {g: {"base": round(float(np.mean(b)), 4), "tta": round(float(np.mean(t)), 4),
                         "tta+cl": round(float(np.mean(tv)), 4), "n": len(b)} for g, (b, t, tv) in per.items()}
        report[s]["_fit"] = {"w_base_mean": round(float(f[:, 0].mean()), 3),
                             "dz_p10_50_90_cm": np.round(np.percentile(f[:, -1], [10, 50, 90]) * 100, 1).tolist(),
                             "frames": len(f)}
        print(s, json.dumps(report[s]), flush=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
