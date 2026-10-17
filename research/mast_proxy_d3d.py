"""MAST 020: score the MAST physics-pipeline assumptions on DIII-D TRAIN shots.

The true flux splits exactly into a plasma part (Green integral of the plasma current
from the Grad-Shafranov operator, restricted to the scorer envelope) and an external
remainder (coils plus vessel/eddy currents). Each variant replaces one part with a
modelled version, so its flux-R2 loss isolates one assumption. Labels are diagnostic
targets only; nothing here produces a submission.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from participant.data import array_value, features, SIGNALS
from participant.mast_green import MU0, greens, plasma_kernel
from participant.score import composite, official

COLUMNS = ("source", "efit_times", "efit_psirz", "efit_q95", "efit_beta_n",
           "efit_grid_R", "efit_grid_Z", "magnetics_time", "magnetics_plasma_current",
           "magnetics_plasma_current_times",
           *(f"magnetics_{s}" for s in SIGNALS if s != "plasma_current"),
           *(f"coil_{s}" for s in ("input_column", "R", "Z", "width", "height",
                                   "angle1", "angle2")))
SUB = 5  # sub-filaments per coil side


def load(path: Path) -> dict:
    table = pq.read_table(path, columns=list(COLUMNS))
    row = {n: array_value(table[n][0]) for n in table.column_names}
    if row["source"] != "DIII-D":
        raise ValueError("Expected a DIII-D shot")
    return row


def coil_responses(row) -> tuple[list[int], np.ndarray]:
    """Flux per kA(-turn) on the grid for each shipped coil rectangle/parallelogram."""
    r, z = np.meshgrid(row["efit_grid_R"], row["efit_grid_Z"])
    u = (np.arange(SUB) + .5) / SUB - .5
    uu, vv = (a.ravel() for a in np.meshgrid(u, u))
    index, columns = [], []
    for k, name in enumerate(np.asarray(row["coil_input_column"]).astype(str)):
        w, h = row["coil_width"][k], row["coil_height"][k]
        a1, a2 = np.radians(row["coil_angle1"][k]), np.radians(row["coil_angle2"][k])
        # EFIT AF/AF2 shear: AF tilts the horizontal sides, AF2 the vertical sides.
        rc = row["coil_R"][k] + uu * w + (vv * h / np.tan(a2) if a2 else 0)
        zc = row["coil_Z"][k] + vv * h + (uu * w * np.tan(a1) if a1 else 0)
        g = greens(rc[None, None], zc[None, None], r[..., None], z[..., None]).mean(axis=2)
        index.append(SIGNALS.index(name.removeprefix("magnetics_")))
        columns.append(1000 * g.ravel())
    return index, np.array(columns)


def delta_star(psi, r, z):
    """Second-order Grad-Shafranov operator at interior nodes (zeros on the edge)."""
    dr, dz = r[1] - r[0], z[1] - z[0]
    out = np.zeros_like(psi)
    rr = r[None, None, 1:-1]
    out[:, 1:-1, 1:-1] = ((psi[:, 1:-1, 2:] - 2 * psi[:, 1:-1, 1:-1] + psi[:, 1:-1, :-2]) / dr**2
                          - (psi[:, 1:-1, 2:] - psi[:, 1:-1, :-2]) / (2 * dr * rr)
                          + (psi[:, 2:, 1:-1] - 2 * psi[:, 1:-1, 1:-1] + psi[:, :-2, 1:-1]) / dz**2)
    return out


def gaussian(sr, sz, r0, z0, var_r, var_z, ip):
    w = np.exp(-(sr[None] - r0[:, None])**2 / (2 * var_r[:, None])
               - (sz[None] - z0[:, None])**2 / (2 * var_z[:, None]))
    return w * (ip / w.sum(axis=1))[:, None]


def shot_parts(row, envelope):
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    psi = np.asarray(row["efit_psirz"], dtype=float)
    currents = features(row)  # kA or kA-turn, aligned to efit_times
    ip_meas = currents[:, SIGNALS.index("plasma_current")] * 1000
    dA = (r[1] - r[0]) * (z[1] - z[0])
    # Orientation: work in phi = s * psi so that the plasma current carries the measured sign.
    j = -delta_star(psi, r, z) / (MU0 * r[None, None, :]) * envelope
    s = np.sign(np.median(j.sum(axis=(1, 2)) * dA * ip_meas))
    phi, j = s * psi, s * j
    jc = (j[:, 1:, 1:] + j[:, :-1, 1:] + j[:, 1:, :-1] + j[:, :-1, :-1]) / 4  # cell centres
    kernel, sr, sz = plasma_kernel(tuple(r), tuple(z))
    cell = jc.reshape(len(psi), -1) * dA
    plasma = (cell @ kernel.T).reshape(psi.shape)
    ext = phi - plasma
    ip_j = cell.sum(axis=1)
    index, responses = coil_responses(row)
    drive = currents[:, index]  # (T, coils), multiplies responses (coils, pixels)
    # Source residual of the external part inside the envelope: should be near zero.
    j_ext = -delta_star(ext, r, z) / (MU0 * r[None, None, :]) * envelope
    mom = np.clip(cell, 0, None)
    norm = mom.sum(axis=1)
    r0, z0 = (mom * sr).sum(1) / norm, (mom * sz).sum(1) / norm
    var_r = (mom * (sr - r0[:, None])**2).sum(1) / norm
    var_z = (mom * (sz - z0[:, None])**2).sum(1) / norm
    circ = gaussian(sr, sz, r0, z0, (var_r + var_z) / 2, (var_r + var_z) / 2, ip_j) @ kernel.T
    ell = gaussian(sr, sz, r0, z0, var_r, var_z, ip_j) @ kernel.T
    return dict(sign=s, phi=phi, plasma=plasma, ext=ext, drive=drive, responses=responses,
                circ=circ.reshape(psi.shape), ell=ell.reshape(psi.shape),
                ip_ratio=ip_j / ip_meas, src_ratio=np.abs(j_ext).sum() / np.abs(j).sum(),
                kappa_g=np.sqrt(var_z / var_r))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=Path, default=Path("run_data/full/raw/data/diii_d_train"))
    parser.add_argument("--out", type=Path, default=Path("run_data/mast-020"))
    parser.add_argument("--official", action="store_true", help="also run the official scorer")
    args = parser.parse_args()
    started = time.monotonic()
    mask = np.load(official.HERE / "fusion_scoring/masks/d3d_envelope.npz")
    envelope = mask["mask_coarse"].astype(float)
    paths = sorted(args.shots.glob("*.parquet"))
    rows, parts = [], []
    for path in paths:
        row = load(path)
        rows.append(row)
        parts.append(shot_parts(row, envelope))
    # Per-coil least-squares gains on the external flux (diagnostic only).
    n = parts[0]["drive"].shape[1]
    ata, aty = np.zeros((n, n)), np.zeros(n)
    for p in parts:
        gram = p["responses"] @ p["responses"].T
        ata += (p["drive"].T @ p["drive"]) * gram
        aty += (p["drive"] * (p["ext"].reshape(len(p["ext"]), -1) @ p["responses"].T)).sum(0)
    gains = np.linalg.solve(ata, aty)
    phi_all = np.concatenate([p["phi"].ravel() for p in parts])
    ss_tot = np.square(phi_all - phi_all.mean()).sum()

    def coil(p, g):
        return ((p["drive"] * g) @ p["responses"]).reshape(p["phi"].shape)

    variants = {
        "coil_nominal+plasma_true": lambda p: coil(p, np.ones(n)) + p["plasma"],
        "coil_fit+plasma_true": lambda p: coil(p, gains) + p["plasma"],
        "ext_true+gauss_circular": lambda p: p["ext"] + p["circ"],
        "ext_true+gauss_elliptic": lambda p: p["ext"] + p["ell"],
        "coil_nominal+gauss_circular": lambda p: coil(p, np.ones(n)) + p["circ"],
        "coil_fit+gauss_elliptic": lambda p: coil(p, gains) + p["ell"],
        "coil_nominal_only": lambda p: coil(p, np.ones(n)),
    }
    report = {"shots": [p.name for p in paths], "frames": int(sum(len(p["phi"]) for p in parts)),
              "gate_ip_ratio_median_abs_err": float(np.median(np.abs(
                  np.concatenate([p["ip_ratio"] for p in parts]) - 1))),
              "gate_external_source_fraction_max": float(max(p["src_ratio"] for p in parts)),
              "signs": [float(p["sign"]) for p in parts],
              "coil_gains": {SIGNALS[i]: float(g) for i, g in zip(
                  coil_responses(rows[0])[0], gains)},
              "variance_share_plasma": float(np.square(np.concatenate(
                  [p["plasma"].ravel() for p in parts]) - phi_all.mean()).sum() / ss_tot),
              "gaussian_elongation_median": float(np.median(np.concatenate(
                  [p["kappa_g"] for p in parts]))),
              "variants": {}}
    for name, build in variants.items():
        preds = [build(p) for p in parts]
        res = sum(np.square(p["phi"] - q).sum() for p, q in zip(parts, preds))
        entry = {"r2_psi": float(1 - res / ss_tot)}
        if args.official:
            scored = composite(rows, [{"psirz": p["sign"] * q,
                                       "q95": np.asarray(r["efit_q95"], float),
                                       "betaN": np.asarray(r["efit_beta_n"], float)}
                                      for r, p, q in zip(rows, parts, preds)])
            entry.update({k: scored[k] for k in ("S", "r2_psi", "dlcfs", "consistency",
                                                  "r2_cons_each", "lcfs_fail_frac")})
        report["variants"][name] = entry
        print(name, json.dumps(entry), flush=True)
    report["seconds"] = time.monotonic() - started
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "proxy-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}, indent=1))


if __name__ == "__main__":
    main()
