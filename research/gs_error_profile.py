"""Per-scalar error profile (bias, spread, true spread) of a GS proxy configuration.

Diagnostic for mast-022: runs the same frames as gs_proxy_d3d and derives the seven scorer
scalars on the prediction and the truth with the vendored derive_frame (DIII-D envelope).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from participant.score import official
from research.gs_proxy_d3d import Profile, run_shot  # noqa: F401
from derive import SCALAR_NAMES, derive_frame  # vendored scorer, on sys.path via official


def derived(args):
    res = args
    mask = np.load(official.HERE / "fusion_scoring/masks/d3d_envelope.npz")
    r, z, coarse = mask["grid_R"], mask["grid_Z"], mask["mask_coarse"].astype(bool)
    pairs = []
    for t in range(len(res["pred"])):
        true = derive_frame(res["row"]["efit_psirz"][t], r, z, "DIII-D", coarse, coarse.astype(float))
        pred = derive_frame(res["pred"][t], r, z, "DIII-D", coarse, coarse.astype(float))
        pairs.append([[true[k], pred[k]] for k in SCALAR_NAMES])
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--alpha-m", type=float, default=1.0)
    parser.add_argument("--alpha-n", type=float, default=2.0)
    parser.add_argument("--beta0", type=float, default=0.8)
    parser.add_argument("--te-power", type=float, default=1.5)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--constraint", default="edges")
    parser.add_argument("--kinetic", action="store_true")
    args = parser.parse_args()
    profile = Profile(args.alpha_m, args.alpha_n, args.beta0, args.iterations,
                      te_power=args.te_power)
    paths = sorted(Path("run_data/full/raw/data/diii_d_train").glob("*.parquet"))
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(run_shot, [(p, profile, args.stride, args.constraint,
                                            args.kinetic) for p in paths]))
        pairs = np.array([x for chunk in pool.map(derived, results) for x in chunk])
    out = {}
    for j, name in enumerate(SCALAR_NAMES):
        t, p = pairs[:, j, 0], pairs[:, j, 1]
        ok = np.isfinite(t) & np.isfinite(p)
        e = p[ok] - t[ok]
        out[name] = {"bias": float(e.mean()), "err_std": float(e.std()),
                     "true_std": float(t[ok].std()), "corr": float(np.corrcoef(t[ok], p[ok])[0, 1]),
                     "valid": int(ok.sum())}
        print(name, json.dumps({k: round(v, 4) for k, v in out[name].items()}))


if __name__ == "__main__":
    main()
