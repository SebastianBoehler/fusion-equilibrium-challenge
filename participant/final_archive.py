"""Submission ZIP from per-config prediction files (public or Final names).

Each member is a zip-of-.npy file named <config>.npz with keys shot_NNNN_{psirz,q95,betaN}.
Checks per member: expected shot count, contiguous indices, float32, finite, matching
frame counts, psirz (T, 65, 65). Members are stored uncompressed at the archive root.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

NAMES = {"diii_d_public_test.npz", "mast_public_test.npz",
         "diii_d_private_test.npz", "mast_private_test.npz"}


def check(path: Path, expected: int) -> int:
    frames = 0
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if len(names) != 3 * expected:
            raise ValueError(f"{path.name}: {len(names)} arrays, expected {3 * expected}")
        for i in range(expected):
            arr = {k: np.lib.format.read_array(z.open(f"shot_{i:04d}_{k}.npy")) for k in ("psirz", "q95", "betaN")}
            n = len(arr["q95"])
            if arr["psirz"].shape != (n, 65, 65) or arr["betaN"].shape != (n,):
                raise ValueError(f"{path.name} shot {i}: shape mismatch")
            if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in arr.values()):
                raise ValueError(f"{path.name} shot {i}: dtype or non-finite")
            frames += n
    return frames


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--member", nargs=2, action="append", metavar=("NPZ", "SHOTS"), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    receipt = {"members": {}}
    for path, shots in a.member:
        path = Path(path)
        if path.name not in NAMES:
            raise ValueError(f"Unexpected member name {path.name}")
        frames = check(path, int(shots))
        receipt["members"][path.name] = {"shots": int(shots), "frames": frames,
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with zipfile.ZipFile(a.out, "x", zipfile.ZIP_STORED, allowZip64=True) as arc:
        for path, _ in a.member:
            arc.write(path, Path(path).name)
    receipt.update(zip_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(), zip_bytes=a.out.stat().st_size)
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
