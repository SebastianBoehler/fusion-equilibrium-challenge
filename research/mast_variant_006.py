"""MAST variant 006 = candidate 005 with betaN from the kinetic-GS flux, smoothed in time.

betaN per frame from research.betan.beta_n_from_flux (DIII-D proxy R2 0.969 with the exact
external field), kept only inside (0, 10), then a 5-frame running median over the valid
frames of each shot and nearest-valid fill. Shots with fewer than 3 valid frames keep 002.
Everything else is byte-identical to 005's MAST arrays. Output: float16 MAST-only archive.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.ndimage import median_filter

from participant.mast_submit import digest
from research.mast_candidate_005 import ROOT, shot_scalars

SOURCE = ROOT / "submission-mast-005/fusion-mast-kinetic-005.zip"
OUT = ROOT / "submission-mast-006"


def smooth(b):
    ok = np.isfinite(b) & (b > 0) & (b < 10)
    if ok.sum() < 3:
        return None
    good = np.nonzero(ok)[0]
    filtered = median_filter(b[good], size=5, mode="nearest")
    nearest = np.abs(np.arange(len(b))[:, None] - good[None]).argmin(axis=1)
    return filtered[nearest]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=SOURCE)
    parser.add_argument("--psi-dir", default="run_data/mast-023")
    parser.add_argument("--robust-region", action="store_true")
    parser.add_argument("--inner", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--zip-name", default="fusion-mast-kinetic-betan-006-mastonly.zip")
    args = parser.parse_args()
    source, out = args.base, args.out
    out.mkdir(exist_ok=False)
    shots = json.loads((ROOT / "mast-001/data.json").read_text())["shots"]
    with ProcessPoolExecutor(12) as pool:
        betas = {i: b for i, _, b in pool.map(shot_scalars, [
            (i, e, args.psi_dir, "inner" if args.inner else args.robust_region)
            for i, e in enumerate(shots)])}
    work, zipped = out / "mast_public_test.npz", out / args.zip_name
    kept, values = 0, []
    with zipfile.ZipFile(source) as src, src.open("mast_public_test.npz") as inner, \
            zipfile.ZipFile(inner) as old, \
            zipfile.ZipFile(work, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as new:
        for i in range(len(shots)):
            arrays = {k: np.lib.format.read_array(old.open(f"shot_{i:04d}_{k}.npy"))
                      for k in ("psirz", "q95", "betaN")}
            b = smooth(betas[i])
            if b is None:
                kept += 1
            else:
                arrays["betaN"] = b.astype(np.float32)
            arrays["psirz"] = arrays["psirz"].astype(np.float16)
            values.append(arrays["betaN"])
            for k, v in arrays.items():
                if not np.isfinite(v).all():
                    raise ValueError(f"Nonfinite {k} at shot {i}")
                with new.open(f"shot_{i:04d}_{k}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, v, allow_pickle=False)
    with zipfile.ZipFile(zipped, "x", zipfile.ZIP_STORED) as archive:
        archive.write(work, "mast_public_test.npz")
    work.unlink()
    allb = np.concatenate(values)
    receipt = {"base": str(source), "base_sha256": digest(source), "shots_on_002_betaN": kept,
               "betaN_p5_50_95": np.percentile(allb, [5, 50, 95]).round(3).tolist(),
               "zip_sha256": digest(zipped), "zip_bytes": zipped.stat().st_size}
    (out / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
