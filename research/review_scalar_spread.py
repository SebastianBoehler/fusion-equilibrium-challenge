"""Same as sens.py, but with OUR vessel region (mast_gs_robust.region 'inner') as mask_coarse/mask_f.
Compares sub3 (057L) vs 030 (044) and records X-point imbalance diagnostics."""
import json, sys, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vendor/fusion_starter/fusion_scoring"))
from derive import derive_frame, SCALAR_NAMES
from lcfs import extract_lcfs
from research.mast_gs_robust import region
DATA = ROOT / "run_data/mast-001"
shots = json.loads((DATA / "data.json").read_text())["shots"]
A, B = ROOT / "run_data/mast-057L/shots", ROOT / "run_data/mast-044/shots"

def one(i):
    fa, fb = A / f"{i:04d}.npz", B / f"{i:04d}.npz"
    if not (fa.exists() and fb.exists()): return []
    row = dict(np.load(DATA / "inputs" / (Path(shots[i]["path"]).stem + ".npz"), allow_pickle=False))
    r, z = row["efit_grid_R"].astype(float), row["efit_grid_Z"].astype(float)
    m = region(row, "inner"); mf = m.astype(float)
    pa = np.load(fa)["psirz"].astype(float); pb = np.load(fb)["psirz"].astype(float)
    out = []
    for t in range(0, min(len(pa), len(pb)), 4):
        rec = [i, t]
        for v in (pa[t], pb[t]):
            C = extract_lcfs(v, r, z, "MAST", m, mf)
            d = derive_frame(v, r, z, "MAST", m, mf, contour=C)
            rec += [d[k] for k in SCALAR_NAMES]
            rec += [np.nan]*2 if C is None else [C[:,1].max(), C[:,1].min()]
        out.append(rec)
    return out

if __name__ == "__main__":
    idx = list(range(0, len(shots), 8))
    with Pool(8) as p:
        rows = np.array([r for o in p.map(one, idx) for r in o], float)
    np.save(ROOT / "run_data/review-20261021-sens2.npy", rows); print(rows.shape)
