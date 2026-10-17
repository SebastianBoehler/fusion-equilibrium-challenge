"""Channel-wise ensemble of built MAST archives (equal-weight means, no fitted weights).

psirz, q95 and betaN are each averaged over their own list of source archives. The
scorer separates these channels (R2_psi, D_LCFS and consistency from psirz; q95 and betaN
R2 from their arrays), so each list can differ. Output is MAST-only float16, or full with
the scored DIII-D member copied and hash-checked.
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


def mean_of(sources, key, i):
    arrays = [np.lib.format.read_array(s.open(f"shot_{i:04d}_{key}.npy")).astype(np.float64)
              for s in sources]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError(f"Shape mismatch for {key} at shot {i}")
    return np.mean(arrays, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--psi", type=Path, nargs="+", required=True)
    p.add_argument("--q95", type=Path, nargs="+", required=True)
    p.add_argument("--betan", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--full-from", type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    srcs = {k: [inner(x) for x in getattr(a, k)] for k in ("psi", "q95", "betan")}
    work = a.out.with_suffix(".work.npz")
    with zipfile.ZipFile(work, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for i in range(1206):
            arr = {"psirz": mean_of(srcs["psi"], "psirz", i),
                   "q95": mean_of(srcs["q95"], "q95", i),
                   "betaN": mean_of(srcs["betan"], "betaN", i)}
            arr["psirz"] = arr["psirz"].astype(np.float32 if a.full_from else np.float16)
            arr["q95"], arr["betaN"] = (arr["q95"].astype(np.float32),
                                        arr["betaN"].astype(np.float32))
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
    receipt = {k: [str(x) for x in getattr(a, k)] for k in ("psi", "q95", "betan")}
    receipt.update(full=a.full_from is not None,
                   zip_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),
                   zip_bytes=a.out.stat().st_size)
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
