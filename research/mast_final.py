"""Final-phase MAST predictions from one solver run (single-run pipeline).

Input: a research.mast_gs_robust output directory (run with --vertical config-union --inner
inner --smooth-anchor; MAST_DATA_DIR pointing at the split's data.json + inputs/).
Per shot:
- psirz: the solver flux, smoothed in time with [1/4, 1/2, 1/4] (ends renormalized), as in 026/030.
- q95: field-line integral at psiN 0.95 on the same flux (bounds 1.5-30, nearest valid
  fill), then a 5-frame running median.
- betaN: flux pressure split on the same flux (bounds 0-10), 5-frame median over valid
  frames, nearest fill (variant 006). With --betan share: the solver's pressure-share
  betaN stored by mast_gs_robust (DIII-D proxy R2 0.928), same bounds/median/fill.
  With --betan flux-kin: equal-weight mean of the processed flux betaN and the kinetic
  Thomson-pressure betaN of the 002 Green model (participant.mast_green.predict), whose
  errors are partly independent (corr 0.64); weights fixed a priori, not fitted.
- Fallbacks if a shot has no valid frame: the 002 Green-model scalar, else q95 6.0 /
  betaN 1.0 (fixed a priori, not fitted).
No labels, demos or envelopes are read. Output: <out>/<config>.npz (zip of .npy, float32)
plus receipt.json.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.ndimage import median_filter

from participant.mast_green import MU0, TF_TURNS, predict
from participant.mast_inputs import SIGNALS, aligned_currents
from participant.mast_submit import validate
from research.betan import beta_n_from_flux
from research.mast_gs_robust import DATA, region
from research.q95_proxy import q_at
from research.temporal_smooth import smooth_psi

Q_BOUNDS, B_BOUNDS = (1.5, 30.0), (0.0, 10.0)


def fill(values, ok, median=None):
    """Nearest-valid fill; with median, a running median over the valid frames first."""
    good = np.nonzero(ok)[0]
    kept = values[good] if median is None else median_filter(values[good], size=median, mode="nearest")
    return kept[np.abs(np.arange(len(values))[:, None] - good[None]).argmin(axis=1)]


def shot(task):
    index, entry, solve_dir, betan_source = task
    row = dict(np.load(DATA / "inputs" / (Path(entry["path"]).stem + ".npz"), allow_pickle=False))
    stored = np.load(Path(solve_dir) / f"shots/{index:04d}.npz")
    psi = stored["psirz"].astype(np.float64)
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    currents = aligned_currents(row)
    fvac = np.abs(currents[:, SIGNALS.index("tf_current")]) * 1000 * MU0 * TF_TURNS / (2 * np.pi)
    ip = currents[:, SIGNALS.index("plasma_current")] * 1000
    allowed = region(row, "inner")
    q, b = np.full(len(psi), np.nan), np.full(len(psi), np.nan)
    for t, phi in enumerate(psi):
        try:
            q[t] = abs(q_at(phi, r, z, allowed, fvac[t]))
        except ValueError:
            pass
        if betan_source in ("flux", "flux-kin"):
            try:
                b[t] = beta_n_from_flux(phi, r, z, allowed, fvac[t], ip[t])
            except ValueError:
                pass
    if betan_source == "share":
        b = stored["betan_share"].astype(np.float64)
    q_ok = np.isfinite(q) & (q > Q_BOUNDS[0]) & (q < Q_BOUNDS[1])
    b_ok = np.isfinite(b) & (b > B_BOUNDS[0]) & (b < B_BOUNDS[1])
    flags = []
    if q_ok.any() and b_ok.sum() >= 3:
        q, b = fill(q, q_ok), fill(b, b_ok, median=5)
    else:
        try:
            base = predict(row)
        except ValueError:
            base = {"q95": np.full(len(psi), 6.0), "betaN": np.full(len(psi), 1.0)}
            flags.append("constant_scalars")
        q = fill(q, q_ok) if q_ok.any() else np.asarray(base["q95"], float)
        b = fill(b, b_ok, median=5) if b_ok.sum() >= 3 else np.asarray(base["betaN"], float)
        flags.append("scalar_fallback")
    q = median_filter(q, size=5, mode="nearest") if len(q) >= 5 else q
    if betan_source == "flux-kin":
        try:
            b = 0.5 * (b + np.asarray(predict(row)["betaN"], float))
        except ValueError:
            flags.append("no_kinetic_betan")
    pred = {"psirz": smooth_psi(psi).astype(np.float32), "q95": q.astype(np.float32),
            "betaN": b.astype(np.float32)}
    validate(pred, len(psi))
    return index, pred, {"q95_invalid": int((~q_ok).sum()), "betaN_invalid": int((~b_ok).sum()), "flags": flags}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--solve-dir", type=Path, required=True)
    p.add_argument("--config", choices=("mast_public_test", "mast_private_test"), required=True)
    p.add_argument("--expected", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--betan", choices=("flux", "share", "flux-kin"), default="flux")
    a = p.parse_args()
    shots = json.loads((DATA / "data.json").read_text())["shots"]
    if len(shots) != a.expected:
        raise ValueError(f"Expected {a.expected} shots, found {len(shots)}")
    a.out.mkdir(parents=True, exist_ok=False)
    target, stats = a.out / f"{a.config}.npz", {"frames": 0, "q95_invalid": 0, "betaN_invalid": 0, "flagged": {}}
    with ProcessPoolExecutor(a.workers) as pool, \
            zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as arc:
        for index, pred, info in pool.map(shot, [(i, e, a.solve_dir, a.betan) for i, e in enumerate(shots)]):
            for k, v in pred.items():
                with arc.open(f"shot_{index:04d}_{k}.npy", "w", force_zip64=True) as m:
                    np.lib.format.write_array(m, v, allow_pickle=False)
            stats["frames"] += len(pred["q95"])
            stats["q95_invalid"] += info["q95_invalid"]
            stats["betaN_invalid"] += info["betaN_invalid"]
            if info["flags"]:
                stats["flagged"][index] = info["flags"]
    receipt = {"config": a.config, "data_dir": str(DATA), "solve_dir": str(a.solve_dir), "betan": a.betan,
               "solver_config": json.loads(str(np.load(a.solve_dir / "shots/0000.npz")["meta"])).get("config"),
               "MAST_equilibrium_data_used": False, "MAST_envelope_used": False, "stats": stats,
               "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size}
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
