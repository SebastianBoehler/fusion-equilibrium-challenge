"""Build MAST candidate 004: official 002 flux with each solenoid element at half the feed current.

Only psirz changes: psi_004 = psi_002 - 0.5 * psi_sol, where psi_sol is the solenoid flux of
the exact 002 Green model (656 elements, each at the full feed current). q95/betaN and the
DIII-D predictions are byte-identical to 976708. Evidence: runs/mast-021-solenoid.md.
"""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from participant.mast_green import coil_kernel
from participant.mast_inputs import SIGNALS, aligned_currents
from participant.mast_submit import digest, validate

ROOT = Path("run_data")
SOURCE_ZIP = ROOT / "submission-mast-002/fusion-mast-surface-002.zip"
SOURCE_ZIP_SHA = "abb3d704c49dd4c137815d9c8e688cab1b120217abbdfdd41af0ea5830d6aa0e"
D3D_SHA = "6f243219fc830b26b4fb095dd47a75881ea64def7105205bcd91f4d76e88ac85"
OUT = ROOT / "submission-mast-004"
SCALE = 0.5


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    if digest(SOURCE_ZIP) != SOURCE_ZIP_SHA:
        raise ValueError("Official 002 archive changed")
    manifest = json.loads((ROOT / "mast-001/data.json").read_text())
    shots = manifest["shots"]
    target, zipped = OUT / "mast_public_test.npz", OUT / "fusion-mast-solenoid-004.zip"
    sol_index, response, geometry, frames = SIGNALS.index("sol_current"), None, None, 0
    with zipfile.ZipFile(SOURCE_ZIP) as source, source.open("mast_public_test.npz") as inner, \
            zipfile.ZipFile(inner) as old, \
            zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for i, entry in enumerate(shots):
            with np.load(ROOT / "mast-001/inputs" / (Path(entry["path"]).stem + ".npz"),
                         allow_pickle=False) as data:
                if str(data["source_hash"]) != entry["sha256"]:
                    raise ValueError(f"Input cache differs from pinned source at {i}")
                row = {k: data[k] for k in data.files}
            key = tuple(np.asarray(row[k]).tobytes() for k in
                        ("coil_input_column", "coil_R", "coil_Z", "coil_width",
                         "coil_height", "efit_grid_R", "efit_grid_Z"))
            if geometry is None:
                geometry, response = key, dict(coil_kernel(row))[sol_index]
            elif key != geometry:
                raise ValueError(f"Coil/grid geometry differs at shot {i}")
            pred = {k: np.lib.format.read_array(old.open(f"shot_{i:04d}_{k}.npy"))
                    for k in ("psirz", "q95", "betaN")}
            n = len(row["efit_times"])
            validate(pred, n)
            sol = aligned_currents(row)[:, sol_index, None] * 1000 * response[None]
            pred["psirz"] = (pred["psirz"].astype(np.float64)
                             - (1 - SCALE) * sol.reshape(n, 65, 65)).astype(np.float32)
            validate(pred, n)
            frames += n
            for k, values in pred.items():
                with new.open(f"shot_{i:04d}_{k}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, values, allow_pickle=False)
            if (i + 1) % 200 == 0:
                print(f"shot {i + 1}/{len(shots)} {time.monotonic() - started:.0f}s", flush=True)
    with np.load(target, allow_pickle=False) as saved:
        if len(saved.files) != 3 * len(shots):
            raise ValueError("Saved MAST archive has missing or extra keys")
    receipt = {
        "method": "002 Green-model flux with solenoid elements at half the feed current "
                  "(two-in-hand winding hypothesis); q95/betaN and DIII-D unchanged from 976708",
        "solenoid_current_per_element": "0.5 x magnetics_sol_current",
        "evidence": "runs/mast-021-solenoid.md; run_data/mast-021/solenoid-hypothesis.json",
        "MAST_equilibrium_data_used": False, "MAST_envelope_used": False,
        "MAST_leaderboard_feedback_used": True,
        "feedback_use": "Official 976708 pooled and worst-20 per-shot R2_psi were compared with "
                        "predictions of discrete solenoid-convention hypotheses; no continuous fit.",
        "AI_assistance": "Claude implemented the hypothesis test and this candidate",
        "source_zip_sha256": SOURCE_ZIP_SHA, "mast_shots": len(shots), "mast_frames": frames,
        "dataset_revision": manifest["revision"], "DIII_D_source_submission": 973093,
        "DIII_D_predictions_sha256": D3D_SHA,
        "mast_predictions_sha256": digest(target),
        "source_sha256": {p: digest(Path(p)) for p in (
            "research/mast_candidate_004.py", "participant/mast_green.py",
            "participant/mast_inputs.py", "participant/mast_submit.py")},
    }
    (OUT / "manifest.json").write_text(json.dumps(receipt, indent=2))
    with zipfile.ZipFile(SOURCE_ZIP) as source, \
            zipfile.ZipFile(zipped, "x", zipfile.ZIP_STORED) as archive:
        h = sha256()
        with source.open("diii_d_public_test.npz") as src, \
                archive.open("diii_d_public_test.npz", "w", force_zip64=True) as dst:
            while chunk := src.read(1 << 24):
                h.update(chunk)
                dst.write(chunk)
        if h.hexdigest() != D3D_SHA:
            raise ValueError("DIII-D predictions differ from the scored 973093 file")
        archive.write(target, target.name)
        archive.write(OUT / "manifest.json", "manifest.json")
    receipt.update(zip_sha256=digest(zipped), zip_bytes=zipped.stat().st_size,
                   seconds=time.monotonic() - started)
    (OUT / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ("zip_sha256", "zip_bytes", "mast_frames", "seconds")}))


if __name__ == "__main__":
    main()
