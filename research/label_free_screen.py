"""Label-free screen of MAST solver runs (screening only; does not rank working candidates).

Per sampled frame: Te flux-function mismatch, outboard LCFS minus the Thomson 30 eV edge,
whether the LCFS is X-point-defined (contact point |Z| > 1 m), and the scorer's own
psi-derived scalars (kappa, volume, li) computed on our prediction without the envelope mask.
Usage: python -m research.label_free_screen RUN_DIR [RUN_DIR ...] [--stride 10]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

from research.flux_function_check import lcfs_outboard_radius, mismatch, te_edge_radius
from research.gs_picard import find_axis, lcfs_level
from research.mast_gs_robust import DATA, region
from research.thomson_points import mast_points

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/fusion_starter/fusion_scoring"))
from derive import derive_frame  # noqa: E402  (unmodified organizer derivation)


def screen(run, stride=10, frame_stride=3):
    shots = json.loads((DATA / "data.json").read_text())["shots"]
    out = {k: [] for k in ("mismatch", "edge", "xpt", "kappa", "volume", "li")}
    for i in range(0, len(shots), stride):
        f = run / "shots" / f"{i:04d}.npz"
        if not f.exists():
            continue
        row = dict(np.load(DATA / "inputs" / (Path(shots[i]["path"]).stem + ".npz"), allow_pickle=False))
        r, z = row["efit_grid_R"], row["efit_grid_Z"]
        al = region(row, "inner")
        edge_mask = al & ~ndimage.binary_erosion(al)
        pr, _, te = mast_points(row)
        psi = np.load(f)["psirz"].astype(float)
        for t in range(0, len(psi), frame_stride):
            phi = psi[t]
            try:
                out["mismatch"].append(mismatch(phi, r, z, al, pr, te[t]))
                rt, ro = te_edge_radius(pr, te[t]), lcfs_outboard_radius(phi, r, z, al)
                if np.isfinite(rt) and np.isfinite(ro):
                    out["edge"].append(ro - rt)
                ax = find_axis(phi, al, (len(z) // 2, int(np.argmin(np.abs(r - 0.92)))))
                lev, _ = lcfs_level(phi, al, ax)
                lab, _ = ndimage.label((phi >= lev - 1e-4 * (phi[ax] - lev)) & al)
                touch = np.argwhere((lab == lab[ax]) & edge_mask)
                if len(touch):
                    k = np.argmax(phi[touch[:, 0], touch[:, 1]])
                    out["xpt"].append(abs(z[touch[k, 0]]) > 1.0)
            except ValueError:
                pass
            d = derive_frame(phi, r, z, "MAST")
            for k in ("kappa", "volume", "li"):
                out[k].append(d[k])
    e = np.array(out["edge"])
    q = lambda k: np.round(np.nanpercentile(out[k], [10, 50, 90]), 2).tolist()
    return {"te_mismatch_p50": round(float(np.nanmedian(out["mismatch"])), 4),
            "edge_bias_cm": round(float(np.median(e)) * 100, 1), "edge_absdev_cm": round(float(np.mean(np.abs(e))) * 100, 1),
            "xpoint_fraction": round(float(np.mean(out["xpt"])), 2), "kappa": q("kappa"), "volume": q("volume"), "li": q("li")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--stride", type=int, default=10)
    a = p.parse_args()
    for run in a.runs:
        print(run.name, json.dumps(screen(run, a.stride)))
