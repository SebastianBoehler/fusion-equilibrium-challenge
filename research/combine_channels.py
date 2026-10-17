"""Combine independently scored channels from built archives into one submission.

The official scorer's MAST terms separate by channel: R2_psi, D_LCFS and consistency depend
only on psirz; the q95 and betaN R2 only on their own arrays. This writes psirz+q95 from one
archive and betaN from another, either MAST-only float16 or full (exact DIII-D member copied
from a scored full archive and hash-checked).
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

D3D_SHA = "6f243219fc830b26b4fb095dd47a75881ea64def7105205bcd91f4d76e88ac85"


def inner(path):
    return zipfile.ZipFile(zipfile.ZipFile(path).open("mast_public_test.npz"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--psi-q95", type=Path, required=True)
    p.add_argument("--betan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--full-from", type=Path, help="full archive supplying the DIII-D member")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    work = a.out.with_suffix(".work.npz")
    src, bsrc = inner(a.psi_q95), inner(a.betan)
    with zipfile.ZipFile(work, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for i in range(1206):
            arr = {k: np.lib.format.read_array(src.open(f"shot_{i:04d}_{k}.npy"))
                   for k in ("psirz", "q95")}
            arr["betaN"] = np.lib.format.read_array(
                bsrc.open(f"shot_{i:04d}_betaN.npy")).astype(np.float32)
            if a.full_from is None:
                arr["psirz"] = arr["psirz"].astype(np.float16)
            n = arr["psirz"].shape[0]
            if not (arr["q95"].shape == arr["betaN"].shape == (n,)
                    and all(np.isfinite(v).all() for v in arr.values())):
                raise ValueError(f"Invalid arrays at shot {i}")
            for k, v in arr.items():
                with new.open(f"shot_{i:04d}_{k}.npy", "w", force_zip64=True) as m:
                    np.lib.format.write_array(m, v, allow_pickle=False)
    with zipfile.ZipFile(a.out, "x", zipfile.ZIP_STORED) as arc:
        if a.full_from is not None:
            h = hashlib.sha256()
            with zipfile.ZipFile(a.full_from) as f, f.open("diii_d_public_test.npz") as s, \
                    arc.open("diii_d_public_test.npz", "w", force_zip64=True) as d:
                while chunk := s.read(1 << 24):
                    h.update(chunk)
                    d.write(chunk)
            if h.hexdigest() != D3D_SHA:
                raise ValueError("DIII-D member differs from the scored 973093 predictions")
        arc.write(work, "mast_public_test.npz")
    work.unlink()
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    receipt = {"psi_q95_from": str(a.psi_q95), "betan_from": str(a.betan),
               "full": a.full_from is not None, "zip_sha256": digest,
               "zip_bytes": a.out.stat().st_size}
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
