"""Apply magnetics TTA (E2: profile-variant blend + vertical shift, all sensor groups) to a split.

Writes a solve-dir compatible with research.mast_final (shots/NNNN.npz with psirz, meta,
betan_share). Shots without an accepted FAIR-MAST match, magnetics or any variant keep the base
run's file unchanged. Human decision 22 Oct 2026 (re-identification + public magnetics); disclose.
Usage (MAST_DATA_DIR=DATA): python -m research.mag_tta_apply DATA MAG MATCH.json OUT BASE VAR1 ... [--max-dist]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np

from research.mag_tta import blend, fit_frame, load_maps, sensors

GROUPS = ("CC", "P", "ccbv", "obv", "obr")


def one(job):
    data, mag_dir, out, runs, i, e, m, fallback = job
    target = out / "shots" / f"{i:04d}.npz"
    base = Path(runs[0]) / "shots" / f"{i:04d}.npz"
    if not base.exists() and fallback:          # shots not re-solved (e.g. unmatched in candidate L2)
        shutil.copyfile(Path(fallback) / "shots" / f"{i:04d}.npz", target)
        return i, "fallback", None
    name = Path(e["path"]).stem
    mf = mag_dir / "magnetics" / f"{name}.npz"
    if m is None or not mf.exists() or not all((Path(r) / "shots" / f"{i:04d}.npz").exists() for r in runs):
        shutil.copyfile(base, target)
        return i, "base", None
    maps, cfg = load_maps(runs, i)
    row = dict(np.load(data / "inputs" / f"{name}.npz", allow_pickle=False))
    mag = dict(np.load(mf))
    pf = mag_dir / "probes" / f"{m['shot']}.npz"
    pr = np.load(pf) if pf.exists() else None
    frames = np.asarray(row["efit_times"], float)
    mag["frames_ms"] = frames
    r, z = row["efit_grid_R"], row["efit_grid_Z"]
    groups, ext = sensors(row, mag, pr, cfg, frames)
    if not groups:
        shutil.copyfile(base, target)
        return i, "no_sensors", None
    plasma = maps - ext[None]
    scale = {g: np.nanstd(s["meas"], axis=0) + 1e-9 for g, s in groups.items()}
    psi, fits = np.empty_like(maps[0]), []
    for t in range(maps.shape[1]):
        for g in groups.values():
            g["meas_t"] = g["meas"][t]
        w, dz, _ = fit_frame(plasma[:, t], r, z, groups, GROUPS, scale)
        if not np.isfinite(w).all():
            w, dz = np.eye(len(runs))[0], 0.0
        psi[t] = ext[t] + blend(plasma[:, t], w, dz, z)
        fits.append([*w, dz])
    stored = dict(np.load(base))
    meta = json.loads(str(stored["meta"]))
    meta["mag_tta"] = {"fairmast_shot": int(m["shot"]), "match_dist": m["dist"], "runs": [str(x) for x in runs],
                       "groups": sorted(groups)}
    stored.update(psirz=psi.astype(np.float32), meta=json.dumps(meta), tta_fit=np.array(fits, np.float32))
    np.savez(target, **stored)
    return i, "tta", np.array(fits)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("data", type=Path); a.add_argument("mag", type=Path); a.add_argument("match", type=Path)
    a.add_argument("out", type=Path); a.add_argument("runs", nargs="+")
    a.add_argument("--max-dist", type=float, default=0.005); a.add_argument("--workers", type=int, default=12)
    a.add_argument("--only", type=Path, help="JSON list of shot indices (smoke tests)")
    a.add_argument("--fallback", help="solve dir for shots missing from the base run")
    args = a.parse_args()
    (args.out / "shots").mkdir(parents=True, exist_ok=False)
    shots = json.loads((args.data / "data.json").read_text())["shots"]
    match = {m["i"]: m for m in json.loads(args.match.read_text()) if m["dist"] <= args.max_dist}
    only = set(json.loads(args.only.read_text())) if args.only else None
    jobs = [(args.data, args.mag, args.out, args.runs, i, e, match.get(i), args.fallback) for i, e in enumerate(shots)
            if only is None or i in only]
    status, fits = {}, []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, s, f in ex.map(one, jobs):
            status[i] = s
            if f is not None:
                fits.append(f)
    f = np.concatenate(fits) if fits else np.zeros((0, len(args.runs) + 1))
    summary = {"counts": {k: list(status.values()).count(k) for k in set(status.values())},
               "w_base_mean": float(f[:, 0].mean()) if len(f) else None,
               "dz_p10_50_90_cm": np.round(np.percentile(f[:, -1], [10, 50, 90]) * 100, 1).tolist() if len(f) else None,
               "max_dist": args.max_dist, "runs": args.runs}
    (args.out / "tta-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
