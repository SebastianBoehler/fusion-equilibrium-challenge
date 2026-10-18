"""Within-shot temporal smoothing of predicted channels (label-free post-processing).

psirz: [1/4, 1/2, 1/4] kernel along time inside each shot (end frames renormalized), which
removes frame-to-frame solver noise from Thomson inputs while keeping slow evolution.
q95: 5-frame running median within each shot. betaN: copied unchanged.
Kernel widths are fixed a priori, not fitted to scores. Output: MAST-only float16 archive.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.ndimage import convolve1d, median_filter


def inner(path):
    return zipfile.ZipFile(zipfile.ZipFile(path).open("mast_public_test.npz"))


def smooth_psi(psi):
    if len(psi) < 3:
        return psi
    kernel = np.array([0.25, 0.5, 0.25])
    num = convolve1d(psi, kernel, axis=0, mode="constant")
    den = convolve1d(np.ones(len(psi)), kernel, mode="constant")
    return num / den[:, None, None]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--psi", type=Path, required=True)
    p.add_argument("--q95", type=Path, required=True)
    p.add_argument("--betan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    src = {k: inner(getattr(a, k)) for k in ("psi", "q95", "betan")}
    work = a.out.with_suffix(".work.npz")
    with zipfile.ZipFile(work, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for i in range(1206):
            psi = np.lib.format.read_array(src["psi"].open(f"shot_{i:04d}_psirz.npy")).astype(np.float64)
            q = np.lib.format.read_array(src["q95"].open(f"shot_{i:04d}_q95.npy")).astype(np.float64)
            b = np.lib.format.read_array(src["betan"].open(f"shot_{i:04d}_betaN.npy"))
            arr = {"psirz": smooth_psi(psi).astype(np.float16),
                   "q95": (median_filter(q, size=5, mode="nearest") if len(q) >= 5 else q).astype(np.float32),
                   "betaN": b.astype(np.float32)}
            if not all(np.isfinite(v).all() for v in arr.values()):
                raise ValueError(f"Nonfinite output at shot {i}")
            for k, v in arr.items():
                with new.open(f"shot_{i:04d}_{k}.npy", "w", force_zip64=True) as m:
                    np.lib.format.write_array(m, v, allow_pickle=False)
    with zipfile.ZipFile(a.out, "x", zipfile.ZIP_STORED) as arc:
        arc.write(work, "mast_public_test.npz")
    work.unlink()
    receipt = {"psi": str(a.psi), "q95": str(a.q95), "betan": str(a.betan),
               "psi_kernel": [0.25, 0.5, 0.25], "q95_median": 5,
               "zip_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
               "zip_bytes": a.out.stat().st_size}
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
