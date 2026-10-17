"""MAST 022: does a self-consistent plasma (gs_picard) fix the shape terms on DIII-D?

Uses the exact external flux from mast_proxy_d3d (true psi minus the Green integral of the
true plasma current) so only the plasma model is under test, plus measured Ip. Labels are
diagnostic targets only. Region: grid minus a 2-cell margin and minus R < 1.0 m (the DIII-D
inner wall), never the scorer envelope, to mimic the MAST setting.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from participant.data import SIGNALS, features
from participant.score import composite, official
from research.gs_picard import Profile, find_axis, lcfs_level, midplane_edges, solve
from research.mast_proxy_d3d import load, shot_parts
from research.thomson_points import d3d_points


def region(r, z, r_min):
    allowed = np.zeros((len(z), len(r)), bool)
    allowed[2:-2, 2:-2] = True
    allowed &= (r >= r_min)[None, :]
    return allowed


def profile_kind(path):
    """Thomson points per frame for the kinetic profile (None if the shot lacks them)."""
    import pyarrow.parquet as pq
    from participant.data import array_value
    cols = ["efit_times", "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z",
            *(f"thomson_{s}_{k}" for s in ("core", "edge") for k in ("times", "Te"))]
    table = pq.read_table(path, columns=cols)
    return d3d_points({n: array_value(table[n][0]) for n in cols})


def run_shot(args):
    path, profile, stride, constraint, use_kinetic = args
    row = load(path)
    kinetic = profile_kind(path)
    envelope = np.load(official.HERE / "fusion_scoring/masks/d3d_envelope.npz")["mask_coarse"]
    p = shot_parts(row, envelope.astype(float))
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    ip = features(row)[:, SIGNALS.index("plasma_current")] * 1000
    allowed = region(r, z, 1.0)
    q = np.sign(np.median(ip))  # solve with positive current; flip reversed-Ip shots back
    out, fails, init, solved = [], 0, None, []
    frames = range(0, len(p["phi"]), stride)
    for t in frames:
        edges = axis_r = None
        if constraint != "none":  # oracle stand-ins for MAST Thomson information
            true = q * p["phi"][t]
            axis = find_axis(true, allowed, (len(z) // 2, len(r) // 2))
            if constraint == "edges":
                edges = midplane_edges(lcfs_level(true, allowed, axis)[1], r, z)
            else:
                axis_r = r[axis[1]]
        try:
            pts = None
            if use_kinetic:
                pr, pz, te = kinetic
                ok = np.isfinite(te[t])
                pts = (pr[ok], pz[ok], te[t][ok])
            info = {}
            phi, cells = solve(q * p["ext"][t], q * ip[t], r, z, allowed, profile, None, edges,
                               axis_r, pts, info)
            solved.append((phi, cells, info))
            phi = q * phi
        except ValueError:
            phi, fails, init = p["ext"][t] + p["circ"][t], fails + 1, None
            solved.append(None)
        out.append(p["sign"] * phi)
    idx = list(frames)
    return {"row": {k: (np.asarray(row[k])[idx] if k in ("efit_psirz", "efit_q95",
                                                          "efit_beta_n") else row[k])
                    for k in ("efit_psirz", "efit_q95", "efit_beta_n")},
            "pred": np.array(out), "fails": fails, "name": path.name, "solved": solved,
            "gauss": p["sign"] * (p["ext"] + p["circ"])[idx]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=Path, default=Path("run_data/full/raw/data/diii_d_train"))
    parser.add_argument("--out", type=Path, default=Path("run_data/mast-022"))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--alpha-m", type=float, default=1.0)
    parser.add_argument("--alpha-n", type=float, default=2.0)
    parser.add_argument("--beta0", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--te-power", type=float, default=1.5)
    parser.add_argument("--efit-c", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--constraint", choices=("none", "edges", "axis"), default="edges")
    parser.add_argument("--kinetic", action="store_true", help="Te^1.5 current profile")
    args = parser.parse_args()
    profile = Profile(args.alpha_m, args.alpha_n, args.beta0, args.iterations,
                      te_power=args.te_power, efit_c=args.efit_c)
    started = time.monotonic()
    paths = sorted(args.shots.glob("*.parquet"))[:args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(run_shot, [(p, profile, args.stride, args.constraint, args.kinetic) for p in paths]))
    rows = [res["row"] for res in results]
    report = {"profile": vars(args) | {"shots": str(args.shots), "out": str(args.out)},
              "frames": int(sum(len(res["pred"]) for res in results)),
              "fails": int(sum(res["fails"] for res in results))}
    for name, key in (("gs_picard", "pred"), ("gauss_circular", "gauss")):
        scored = composite(rows, [{"psirz": res[key], "q95": np.asarray(row["efit_q95"], float),
                                   "betaN": np.asarray(row["efit_beta_n"], float)}
                                  for res, row in zip(results, rows)])
        report[name] = {k: scored[k] for k in ("S", "r2_psi", "dlcfs", "consistency",
                                                "r2_cons_each", "lcfs_fail_frac")}
        print(name, json.dumps(report[name]), flush=True)
    for res in results:
        true = res["row"]["efit_psirz"]
        tot = np.square(true - true.mean()).sum()
        print(res["name"], "fails", res["fails"], "r2 gs %.3f gauss %.3f" % (
            1 - np.square(true - res["pred"]).sum() / tot,
            1 - np.square(true - res["gauss"]).sum() / tot))
    report["seconds"] = time.monotonic() - started
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"m{args.alpha_m}-n{args.alpha_n}-b{args.beta0}-{args.constraint}-k{int(args.kinetic)}"
    (args.out / f"report-{tag}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("frames", "fails", "seconds")}))


if __name__ == "__main__":
    main()
