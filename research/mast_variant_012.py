"""MAST variant 012 = candidate 004 + induced vessel-current flux (isolation test).

Vessel per literature review 1: stainless cylinder 4 m diameter x 4.4 m high, 20 mm plate
(Darke et al., FED 1999), modelled as a rectangular toroidal shell R 0.19-2.02 m, Z +-2.2 m
(research/passive.wall_filaments). Only the flux changes; q95/betaN as 004. MAST-only float16.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import zipfile

import numpy as np

from participant.mast_submit import digest
from research.mast_passive import passive_flux

ROOT = Path("run_data")
BASE = ROOT / "submission-mast-004/fusion-mast-solenoid-004.zip"
OUT = ROOT / "submission-mast-012"
VESSEL = dict(r_in=0.19, r_out=2.02, z_half=2.2, thickness=0.02, resistivity=7.2e-7,
              spacing=0.1)


def shot_passive(task):
    index, entry = task
    row = dict(np.load(ROOT / "mast-001/inputs" / (Path(entry["path"]).stem + ".npz"),
                       allow_pickle=False))
    flux, currents = passive_flux(row, VESSEL)
    return index, flux.reshape(-1, 65, 65), float(np.abs(currents.sum(axis=1)).max())


def main() -> None:
    OUT.mkdir(exist_ok=False)
    shots = json.loads((ROOT / "mast-001/data.json").read_text())["shots"]
    with ProcessPoolExecutor(4) as pool:
        results = {i: (f, c) for i, f, c in pool.map(shot_passive, list(enumerate(shots)))}
    work, zipped = OUT / "mast_public_test.npz", OUT / "fusion-mast-passive-012-mastonly.zip"
    ss_pass = ss_pred = 0.0
    with zipfile.ZipFile(BASE) as zb, zb.open("mast_public_test.npz") as fb, \
            zipfile.ZipFile(fb) as base, \
            zipfile.ZipFile(work, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as new:
        for i in range(len(shots)):
            arr = {k: np.lib.format.read_array(base.open(f"shot_{i:04d}_{k}.npy"))
                   for k in ("psirz", "q95", "betaN")}
            flux = results[i][0]
            ss_pass += float(np.square(flux).sum())
            ss_pred += float(np.square(arr["psirz"].astype(np.float64)).sum())
            arr["psirz"] = (arr["psirz"].astype(np.float64) + flux).astype(np.float16)
            for k, v in arr.items():
                if not np.isfinite(v).all():
                    raise ValueError(f"Nonfinite {k} at shot {i}")
                with new.open(f"shot_{i:04d}_{k}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, v, allow_pickle=False)
    with zipfile.ZipFile(zipped, "x", zipfile.ZIP_STORED) as archive:
        archive.write(work, "mast_public_test.npz")
    work.unlink()
    peaks = [c for _, c in results.values()]
    receipt = {"vessel": VESSEL, "base_sha256": digest(BASE),
               "passive_ss_over_prediction_ss": ss_pass / ss_pred,
               "peak_total_vessel_kA_p50_p95": np.percentile(peaks, [50, 95]).round(-2).tolist(),
               "zip_sha256": digest(zipped), "zip_bytes": zipped.stat().st_size}
    (OUT / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
