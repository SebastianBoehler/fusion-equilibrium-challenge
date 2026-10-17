"""Small MAST-only feedback archive: float16 flux, deflated npz, no DIII-D file.

Codabench reports mast_S and every MAST component for a submission without DIII-D
predictions (G_ratio then shows 0). This cuts a 4 GB upload to about 0.5 GB for slow links.
The source archive's MAST arrays are converted exactly once; q95/betaN stay float32.
"""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from participant.mast_submit import digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="full candidate zip")
    parser.add_argument("--out", type=Path, required=True, help="new MAST-only zip")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    work = args.out.with_suffix(".npz")
    worst = 0.0
    with zipfile.ZipFile(args.source) as src, src.open("mast_public_test.npz") as inner, \
            zipfile.ZipFile(inner) as old, \
            zipfile.ZipFile(work, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as new:
        names = sorted(n for n in old.namelist() if n.endswith(".npy"))
        if len(names) != 3 * 1206:
            raise ValueError("Expected 1,206 MAST shots")
        for name in names:
            values = np.lib.format.read_array(old.open(name))
            if name.endswith("_psirz.npy"):
                half = values.astype(np.float16)
                if not np.isfinite(half).all():
                    raise ValueError(f"float16 overflow in {name}")
                worst = max(worst, float(np.abs(half - values).max() / np.abs(values).max()))
                values = half
            with new.open(name, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, values, allow_pickle=False)
    with zipfile.ZipFile(args.out, "x", zipfile.ZIP_STORED) as archive:
        archive.write(work, "mast_public_test.npz")
    receipt = {"source": str(args.source), "source_sha256": digest(args.source),
               "mast_only": True, "psirz_dtype": "float16",
               "max_relative_rounding": worst, "zip_sha256": digest(args.out),
               "zip_bytes": args.out.stat().st_size}
    work.unlink()
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
