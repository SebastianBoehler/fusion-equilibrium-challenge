"""Channel-wise combination of two built MAST members (no averaging, no fitted weights).

Each scored channel (psirz, q95, betaN) is copied unchanged from the member named for it. The
scorer separates the channels, so this is a pure selection. Output: float32 npz in the same key
layout (shot_XXXX_<channel>), ready for participant.final_archive.
Usage: python -m research.combine_members --psirz A.npz --q95 A.npz --betan B.npz --shots 1205 --out OUT.npz
"""

import argparse
import zipfile
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    for k in ("psirz", "q95", "betan"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--shots", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    src = {"psirz": np.load(a.psirz), "q95": np.load(a.q95), "betaN": np.load(a.betan)}
    out = {}
    for i in range(a.shots):
        T = None
        for ch, z in src.items():
            arr = np.asarray(z[f"shot_{i:04d}_{ch}"], np.float32)
            T = len(arr) if T is None else T
            if len(arr) != T or not np.isfinite(arr).all():
                raise ValueError(f"shot {i} {ch}: frames {len(arr)} vs {T} or non-finite")
            out[f"shot_{i:04d}_{ch}"] = arr
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.out, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as arc:   # as research.mast_final
        for key, arr in out.items():
            with arc.open(f"{key}.npy", "w", force_zip64=True) as m:
                np.lib.format.write_array(m, arr)
    print("wrote", a.out, len(out), "arrays")


if __name__ == "__main__":
    main()
