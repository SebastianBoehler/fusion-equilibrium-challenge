"""MAST 023: kinetic free-boundary equilibria for every MAST frame (input-only).

External flux: measured PF coils + solenoid at half the feed current per element (mast-021).
Plasma: research.gs_picard with J ~ Te^1.0 mapped onto flux (calibrated on DIII-D, mast-022),
axis radius anchored at the Thomson Te peak, vertical position from external force balance.
Allowed region: grid minus a 2-cell margin, minus R < 0.19 m (centre column), minus cells
within 6 cm of any PF/solenoid conductor. No MAST labels, demo files or envelope.

Per-shot outputs are written to <out>/shots/ so an interrupted run can continue; frames
whose solve fails keep the candidate-004 flux and are counted.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy import ndimage

from participant.mast_green import coil_kernel
from participant.mast_inputs import SIGNALS, aligned_currents
from research.gs_picard import Profile, find_axis, lcfs_level, midplane_edges, solve
from research.thomson_points import mast_points

ROOT = Path("run_data")
SOL_SCALE = 0.5
PROFILE = Profile(beta0=0.8, iterations=60, te_power=1.0)


def allowed_region(row, r_min=0.19, sol_clearance=0.06):
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    rr, zz = np.meshgrid(r, z)
    ok = np.zeros(rr.shape, bool)
    ok[2:-2, 2:-2] = True
    ok &= rr >= r_min
    cr, cz = np.asarray(row["coil_R"]), np.asarray(row["coil_Z"])
    sol = np.asarray(row["coil_input_column"]).astype(str) == "magnetics_sol_current"
    for k in range(len(cr)):
        clearance = sol_clearance if sol[k] else 0.06
        ok &= (rr - cr[k]) ** 2 + (zz - cz[k]) ** 2 > clearance ** 2
    return ok


def te_peak_radius(pr, te):
    """Axis radius estimate: parabola through the smoothed Te maximum (None if unusable)."""
    ok = np.isfinite(te)
    if ok.sum() < 10:
        return None
    order = np.argsort(pr[ok])
    x, y = pr[ok][order], ndimage.uniform_filter1d(te[ok][order], 7)
    k = int(np.argmax(y))
    if k < 3 or k > len(x) - 4:
        return None
    c = np.polyfit(x[k - 3:k + 4], y[k - 3:k + 4], 2)
    return float(-c[1] / (2 * c[0])) if c[0] < 0 else float(x[k])


def run_shot(task):
    index, entry, out = task
    target = out / "shots" / f"{index:04d}.npz"
    if target.exists():
        return json.loads(str(np.load(target)["meta"]))
    row = dict(np.load(ROOT / "mast-001/inputs" / (Path(entry["path"]).stem + ".npz"),
                       allow_pickle=False))
    if str(row["source_hash"]) != entry["sha256"]:
        raise ValueError(f"Input cache differs from pinned source at {index}")
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    currents = aligned_currents(row)
    ext = np.zeros((len(currents), r.size * z.size))
    for column, response in coil_kernel(row):
        scale = SOL_SCALE if SIGNALS[column] == "sol_current" else 1.0
        ext += currents[:, column, None] * 1000 * scale * response[None]
    ext = ext.reshape(-1, len(z), len(r))
    ip = currents[:, SIGNALS.index("plasma_current")] * 1000
    pr, pz, te = mast_points(row)
    allowed = allowed_region(row)
    with zipfile.ZipFile(ROOT / "submission-mast-004/fusion-mast-solenoid-004.zip") as zf, \
            zf.open("mast_public_test.npz") as inner, zipfile.ZipFile(inner) as old:
        fallback = np.lib.format.read_array(old.open(f"shot_{index:04d}_psirz.npy"))
    psi, fails, axis_rs, kappas = [], 0, [], []
    for t in range(len(ip)):
        ok = np.isfinite(te[t])
        axis_r = te_peak_radius(pr, te[t])
        try:
            if ip[t] <= 0:
                raise ValueError("Nonpositive plasma current")
            phi, _ = solve(ext[t], ip[t], r, z, allowed, PROFILE, None, None, axis_r,
                           (pr[ok], pz[ok], te[t][ok]) if ok.sum() >= 5 else None)
            axis = find_axis(phi, allowed, (len(z) // 2, len(r) // 2))
            region = lcfs_level(phi, allowed, axis)[1]
            rows = np.nonzero(region.any(axis=1))[0]
            r_in, r_out = midplane_edges(region, r, z)
            kappas.append(float((z[rows[-1]] - z[rows[0]]) / (r_out - r_in)))
            axis_rs.append(float(r[axis[1]]))
            psi.append(phi.astype(np.float32))
        except ValueError:
            fails += 1
            psi.append(fallback[t])
    psi = np.array(psi)
    if psi.shape != fallback.shape or not np.isfinite(psi).all():
        raise ValueError(f"Invalid equilibrium output at {index}")
    meta = {"index": index, "frames": len(ip), "fails": fails,
            "axis_r_median": float(np.median(axis_rs)) if axis_rs else None,
            "kappa_grid_median": float(np.median(kappas)) if kappas else None,
            "r2_vs_004": float(1 - np.square(psi - fallback).sum()
                               / np.square(fallback - fallback.mean()).sum())}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial.npz")
    np.savez(tmp, psirz=psi, meta=json.dumps(meta))
    tmp.replace(target)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "mast-023")
    parser.add_argument("--shots", type=str, default="pilot", help="'pilot' or 'all'")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    shots = json.loads((ROOT / "mast-001/data.json").read_text())["shots"]
    indices = list(range(0, len(shots), 50)) if args.shots == "pilot" else list(range(len(shots)))
    started = time.monotonic()
    metas = []
    with ProcessPoolExecutor(args.workers) as pool:
        for meta in pool.map(run_shot, [(i, shots[i], args.out) for i in indices]):
            metas.append(meta)
            if len(metas) % 25 == 0 or len(metas) == len(indices):
                print(f"{len(metas)}/{len(indices)} shots {time.monotonic() - started:.0f}s",
                      flush=True)
    frames = sum(m["frames"] for m in metas)
    summary = {"shots": len(metas), "frames": frames,
               "fails": sum(m["fails"] for m in metas),
               "r2_vs_004_median": float(np.median([m["r2_vs_004"] for m in metas])),
               "axis_r_median": float(np.nanmedian([m["axis_r_median"] or np.nan for m in metas])),
               "kappa_grid_median": float(np.nanmedian([m["kappa_grid_median"] or np.nan
                                                        for m in metas])),
               "seconds": time.monotonic() - started}
    (args.out / f"summary-{args.shots}.json").write_text(json.dumps(
        {"summary": summary, "shots": metas}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
