"""Build MAST candidate 005 from the mast-023 kinetic equilibria.

psirz: kinetic free-boundary flux (mast-023; failed frames already hold the 004 flux).
q95:   field-line integral on psiN = 0.95 with the vacuum TF function (mast-024: DIII-D proxy
       R2 0.858, unbiased).
betaN: kept from official 002. The flux-only split (mast-024: DIII-D proxy R2 0.969 with the
       exact external field) is too noisy on MAST pilot shots; its values are only recorded.
q95 frames that fail or leave wide physical bounds take the nearest valid q95 of the same
shot (official 002 values only if a shot has none).
DIII-D predictions are byte-identical to 973093. Writes a full archive and a float16
MAST-only feedback archive.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import zipfile

import numpy as np

from participant.mast_green import MU0, TF_TURNS
from participant.mast_inputs import SIGNALS, aligned_currents
from participant.mast_submit import digest, validate
from research.betan import beta_n_from_flux
from research.mast_gs import allowed_region
from research.mast_gs_robust import region as robust_region
from research.q95_proxy import q_at

ROOT = Path("run_data")
SOURCE_ZIP = ROOT / "submission-mast-004/fusion-mast-solenoid-004.zip"
OUT = ROOT / "submission-mast-005"
PSI_DIR = ROOT / "mast-023"
ZIP_NAME = "fusion-mast-kinetic-005.zip"
REGION = allowed_region
INNER = [False]
Q_BOUNDS, B_BOUNDS = (1.5, 30.0), (0.0, 10.0)


def shot_scalars(task):
    index, entry, psi_dir, robust = (*task, PSI_DIR, False)[:4] if len(task) == 2 else task
    row = dict(np.load(ROOT / "mast-001/inputs" / (Path(entry["path"]).stem + ".npz"),
                       allow_pickle=False))
    psi = np.load(Path(psi_dir) / f"shots/{index:04d}.npz")["psirz"].astype(np.float64)
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    currents = aligned_currents(row)
    fvac = np.abs(currents[:, SIGNALS.index("tf_current")]) * 1000 * MU0 * TF_TURNS / (2 * np.pi)
    ip = currents[:, SIGNALS.index("plasma_current")] * 1000
    if robust == "inner":
        allowed = robust_region(row, "inner")
    else:
        allowed = (robust_region if robust else allowed_region)(row)
    q95, beta = np.full(len(psi), np.nan), np.full(len(psi), np.nan)
    for t, phi in enumerate(psi):
        try:
            q95[t] = abs(q_at(phi, r, z, allowed, fvac[t]))
        except ValueError:
            pass
        try:
            beta[t] = beta_n_from_flux(phi, r, z, allowed, fvac[t], ip[t])
        except ValueError:
            pass
    return index, q95, beta


def main() -> None:
    global OUT, PSI_DIR, ZIP_NAME, REGION
    parser = argparse.ArgumentParser()
    parser.add_argument("--psi-dir", type=Path, default=PSI_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--zip-name", default=ZIP_NAME)
    parser.add_argument("--robust-region", action="store_true")
    parser.add_argument("--inner", action="store_true", help="solver used the inner wall")
    args = parser.parse_args()
    OUT, PSI_DIR, ZIP_NAME = args.out, args.psi_dir, args.zip_name
    REGION = robust_region if args.robust_region else allowed_region
    INNER[0] = args.inner
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    shots = json.loads((ROOT / "mast-001/data.json").read_text())["shots"]
    with ProcessPoolExecutor(12) as pool:
        scalars = {i: (q, b) for i, q, b in pool.map(shot_scalars, [
            (i, e, str(PSI_DIR), "inner" if INNER[0] else REGION is robust_region)
            for i, e in enumerate(shots)])}
    target, zipped = OUT / "mast_public_test.npz", OUT / ZIP_NAME
    stats = {"q95_fallback": 0, "betaN_flux_out_of_bounds": 0, "q95_shots_on_002": 0,
             "frames": 0}
    q_all, b_all = [], []
    with zipfile.ZipFile(SOURCE_ZIP) as source, source.open("mast_public_test.npz") as inner, \
            zipfile.ZipFile(inner) as old, \
            zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        for i in range(len(shots)):
            pred = {k: np.lib.format.read_array(old.open(f"shot_{i:04d}_{k}.npy"))
                    for k in ("q95", "betaN")}
            pred["psirz"] = np.load(PSI_DIR / f"shots/{i:04d}.npz")["psirz"]
            q, b = scalars[i]
            q_ok = np.isfinite(q) & (q > Q_BOUNDS[0]) & (q < Q_BOUNDS[1])
            b_ok = np.isfinite(b) & (b > B_BOUNDS[0]) & (b < B_BOUNDS[1])
            stats["q95_fallback"] += int((~q_ok).sum())
            stats["betaN_flux_out_of_bounds"] += int((~b_ok).sum())
            stats["frames"] += len(q)
            if q_ok.any():
                good = np.nonzero(q_ok)[0]
                nearest = good[np.abs(np.arange(len(q))[:, None] - good[None]).argmin(axis=1)]
                pred["q95"] = q[nearest].astype(np.float32)
            else:
                stats["q95_shots_on_002"] += 1
            q_all.append(pred["q95"])
            b_all.append(pred["betaN"])
            validate(pred, len(q))
            for k, values in pred.items():
                with new.open(f"shot_{i:04d}_{k}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, values, allow_pickle=False)
    q_all, b_all = np.concatenate(q_all), np.concatenate(b_all)
    stats.update({f"q95_p{p}": float(np.percentile(q_all, p)) for p in (5, 50, 95)})
    stats.update({f"betaN_p{p}": float(np.percentile(b_all, p)) for p in (5, 50, 95)})
    receipt = {
        "method": "Kinetic free-boundary equilibria (mast-023) with q95 from the same flux "
                  "(mast-024); betaN unchanged from 002; solenoid at half feed current (mast-021)",
        "MAST_equilibrium_data_used": False, "MAST_envelope_used": False,
        "MAST_leaderboard_feedback_used": True,
        "feedback_use": "Official 976708 components motivated the solenoid fix (mast-021).",
        "AI_assistance": "Claude implemented and validated the solver on DIII-D TRAIN shots",
        "stats": stats, "mast_predictions_sha256": digest(target),
        "DIII_D_predictions_sha256":
            "6f243219fc830b26b4fb095dd47a75881ea64def7105205bcd91f4d76e88ac85",
        "source_sha256": {p: digest(Path(p)) for p in (
            "research/mast_candidate_005.py", "research/mast_gs.py", "research/gs_picard.py",
            "research/betan.py", "research/q95_proxy.py", "research/thomson_points.py")},
    }
    (OUT / "manifest.json").write_text(json.dumps(receipt, indent=2))
    with zipfile.ZipFile(SOURCE_ZIP) as source, \
            zipfile.ZipFile(zipped, "x", zipfile.ZIP_STORED) as archive:
        with source.open("diii_d_public_test.npz") as src, \
                archive.open("diii_d_public_test.npz", "w", force_zip64=True) as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
        archive.write(target, target.name)
        archive.write(OUT / "manifest.json", "manifest.json")
    receipt.update(zip_sha256=digest(zipped), zip_bytes=zipped.stat().st_size,
                   seconds=time.monotonic() - started)
    (OUT / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: receipt[k] for k in ("stats", "zip_bytes", "seconds")}))


if __name__ == "__main__":
    main()
