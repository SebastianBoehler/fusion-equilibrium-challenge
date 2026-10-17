"""MAST variant 007 = candidate 005 with per-frame runaway guards (after official 985081).

985081 showed solver runaways (axis parked at R ~1.9 m on the worst shots). A frame keeps
the kinetic-GS flux only if its own axis lies in R 0.7-1.15 m and |Z| < 0.3 m and a closed
LCFS exists in the allowed region; otherwise it takes the candidate-004 flux. q95 of a
guarded frame is the nearest unguarded frame's q95 in the same shot (004/002 q95 if none).
betaN stays 002. Output: float16 MAST-only archive.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import zipfile

import numpy as np

from participant.mast_submit import digest
from research.gs_picard import find_axis, lcfs_level
from research.mast_gs import ROOT, allowed_region

BASE = ROOT / "submission-mast-005/fusion-mast-kinetic-005.zip"
FALLBACK = ROOT / "submission-mast-004/fusion-mast-solenoid-004.zip"
OUT = ROOT / "submission-mast-007"


def keep_mask(task):
    index, entry = task
    row = dict(np.load(ROOT / "mast-001/inputs" / (Path(entry["path"]).stem + ".npz"),
                       allow_pickle=False))
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    allowed = allowed_region(row)
    psi = np.load(ROOT / f"mast-023/shots/{index:04d}.npz")["psirz"].astype(np.float64)
    keep = np.zeros(len(psi), bool)
    for t, phi in enumerate(psi):
        try:
            iz, ir = find_axis(phi, allowed, (len(z) // 2, int(np.argmin(np.abs(r - 0.9)))))
            lcfs_level(phi, allowed, (iz, ir))
            keep[t] = 0.7 <= r[ir] <= 1.15 and abs(z[iz]) < 0.3
        except ValueError:
            pass
    return index, keep


def main() -> None:
    OUT.mkdir(exist_ok=False)
    shots = json.loads((ROOT / "mast-001/data.json").read_text())["shots"]
    with ProcessPoolExecutor(12) as pool:
        masks = dict(pool.map(keep_mask, list(enumerate(shots))))
    work, zipped = OUT / "mast_public_test.npz", OUT / "fusion-mast-guarded-007-mastonly.zip"
    guarded = frames = 0
    with zipfile.ZipFile(BASE) as zb, zb.open("mast_public_test.npz") as fb, \
            zipfile.ZipFile(fb) as base, zipfile.ZipFile(FALLBACK) as zf, \
            zf.open("mast_public_test.npz") as ff, zipfile.ZipFile(ff) as fall, \
            zipfile.ZipFile(work, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as new:
        for i in range(len(shots)):
            arr = {k: np.lib.format.read_array(base.open(f"shot_{i:04d}_{k}.npy"))
                   for k in ("psirz", "q95", "betaN")}
            old = {k: np.lib.format.read_array(fall.open(f"shot_{i:04d}_{k}.npy"))
                   for k in ("psirz", "q95")}
            keep = masks[i]
            guarded += int((~keep).sum())
            frames += len(keep)
            arr["psirz"] = np.where(keep[:, None, None], arr["psirz"], old["psirz"])
            if keep.any():
                good = np.nonzero(keep)[0]
                near = good[np.abs(np.arange(len(keep))[:, None] - good[None]).argmin(axis=1)]
                arr["q95"] = arr["q95"][near]
            else:
                arr["q95"] = old["q95"]
            arr["psirz"] = arr["psirz"].astype(np.float16)
            for k, v in arr.items():
                if not np.isfinite(v).all():
                    raise ValueError(f"Nonfinite {k} at shot {i}")
                with new.open(f"shot_{i:04d}_{k}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, v, allow_pickle=False)
    with zipfile.ZipFile(zipped, "x", zipfile.ZIP_STORED) as archive:
        archive.write(work, "mast_public_test.npz")
    work.unlink()
    receipt = {"guarded_frames": guarded, "frames": frames,
               "base_sha256": digest(BASE), "fallback_sha256": digest(FALLBACK),
               "zip_sha256": digest(zipped), "zip_bytes": zipped.stat().st_size}
    (OUT / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
