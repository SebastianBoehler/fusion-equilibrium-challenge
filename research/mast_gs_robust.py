"""MAST free-boundary kinetic equilibria from released inputs (label-free).

Per frame: external flux of the measured PF coils (totals spread over their elements; each
solenoid element at half the feed current), optionally plus passive-structure currents
induced by the measured current histories (research/mast_passive.py). The plasma is a
Picard free-boundary solve (research/gs_picard.py) with J ~ Te(psiN)^p from Thomson
(p = 1.0, calibrated on DIII-D) and the pressure share adapted so the magnetic axis sits at
the Te anchor: the centroid of Te above 70% of its peak (accepted within 0.75-1.10 m, else
the shot median, else 0.92 m). Allowed region: grid minus coil/solenoid clearances, inner
wall, R <= 1.6 m, |Z| <= 1.7 m (machine geometry).

Vertical position (--vertical):
- zero: current centroid at Z = 0.
- config-union: shots in the shifted class (P2 asymmetry < -0.02 or |P6 radial field|/Ip
  above the input noise band) get the smoothed force-balance height forced downward; all
  other shots Z = 0. Reproduces scored candidate 030 (official 996142) bit for bit.

Options: --no-smooth-anchor (default: 5-frame median of anchors), --anchor-prior DIR (Te flux-function
symmetry correction of the anchor from a previous run), --edge/--edge-nobz (outboard LCFS at
the Te = 30 eV crossing via a uniform B_Z correction during the solve, optionally left out of
the output), --passive {V,VC,ALL,ALLg}, --te-power. Frames failing the plausibility check
(axis R 0.7-1.15 m, |Z| < 0.45 m, closed LCFS) keep the candidate-004 flux.
Input directory: $MAST_DATA_DIR (data.json + inputs/), default run_data/mast-001.
Output: <out>/shots/NNNN.npz (psirz, beta0, betan_share, meta incl. the run config);
resuming into a directory made with a different config is refused.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy import ndimage

from participant.mast_green import MU0, TF_TURNS, coil_kernel
from participant.mast_inputs import SIGNALS, aligned_currents
from research.betan import beta_n_equilibrium
from research.gs_picard import Profile, find_axis, lcfs_level, solve
from research.mast_gs import ROOT, SOL_SCALE, allowed_region
from research.thomson_points import mast_points

# Env var so spawned workers and research.mast_final see the same input directory.
DATA = Path(os.environ.get("MAST_DATA_DIR", ROOT / "mast-001"))
PUBLIC = DATA.resolve() == (ROOT / "mast-001").resolve()

FORCE = Profile(beta0=0.8, iterations=60, te_power=1.0, beta0_range=(-0.5, 2.0))
ZERO = replace(FORCE, vertical="zero")
AXIS_RANGE, AXIS_DEFAULT = (0.75, 1.10), 0.92
R_MIN = {"default": 0.19, "inner": 0.175}
SHIFTS_CM = np.arange(-3.0, 5.01, 1.0)  # anchor-correction scan (label-free p10 -1, p90 +3 cm)


@dataclass(frozen=True)
class RunConfig:
    vertical: str = "config-union"
    inner: str = "inner"
    smooth_anchor: bool = True
    edge: bool = False
    edge_nobz: bool = False
    te_power: float = 1.0
    passive: str | None = None
    anchor_prior: str | None = None
    sol_scale: float = SOL_SCALE  # per-element share of sol_current (0.5: official feedback + flux loops)


def te_axis(pr, te):
    ok = np.isfinite(te)
    if ok.sum() < 15:
        return None
    order = np.argsort(pr[ok])
    x, y = pr[ok][order], ndimage.median_filter(te[ok][order], size=9, mode="nearest")
    top = y >= 0.7 * y.max()
    centre = float((x[top] * y[top]).sum() / y[top].sum())
    return centre if AXIS_RANGE[0] <= centre <= AXIS_RANGE[1] else None


def region(row, inner="default"):
    """Allowed plasma region; "inner" moves the centre-column limit to just outside the
    solenoid conductors (0.175 m) so diverted plasmas are not wall-limited artificially."""
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    return (allowed_region(row, R_MIN[inner], 0.06 if inner == "default" else 0.02)
            & (r[None, :] <= 1.6) & (np.abs(z)[:, None] <= 1.7))


def fill(values, ok, median):
    """Running median over the valid entries, then nearest-valid fill."""
    good = np.nonzero(ok)[0]
    smooth = ndimage.median_filter(np.asarray(values, float)[good], size=median, mode="nearest")
    return smooth[np.abs(np.arange(len(values))[:, None] - good[None]).argmin(axis=1)]


def fallback_psi(row, index, sol_scale=SOL_SCALE):
    """Candidate-004-style flux (002 Green model with the solenoid at sol_scale): from the
    public 004 archive when available and sol_scale is 0.5, else recomputed (bit-identical
    for 0.5); coils-only flux if 002 cannot run."""
    archive = ROOT / "submission-mast-004/fusion-mast-solenoid-004.zip"
    if PUBLIC and sol_scale == SOL_SCALE and archive.exists():
        with zipfile.ZipFile(archive) as zf, zf.open("mast_public_test.npz") as inner, \
                zipfile.ZipFile(inner) as old:
            return np.lib.format.read_array(old.open(f"shot_{index:04d}_psirz.npy"))
    from participant.mast_green import predict
    currents, kernel = aligned_currents(row), dict(coil_kernel(row))
    sol = SIGNALS.index("sol_current")
    try:
        psi = predict(row)["psirz"].astype(np.float64).reshape(-1, 65 * 65)
        psi -= (1 - sol_scale) * currents[:, sol, None] * 1000 * kernel[sol][None]
    except ValueError:  # e.g. Ip <= 0 or unusable Thomson in some frame
        psi = sum(currents[:, c, None] * 1000 * (sol_scale if c == sol else 1.0) * k[None]
                  for c, k in kernel.items())
    return psi.reshape(-1, 65, 65).astype(np.float32)


def plausible(phi, r, z, allowed):
    try:
        iz, ir = find_axis(phi, allowed, (len(z) // 2, int(np.argmin(np.abs(r - AXIS_DEFAULT)))))
        lcfs_level(phi, allowed, (iz, ir))
    except ValueError:
        return False
    return 0.7 <= r[ir] <= 1.15 and abs(z[iz]) < 0.45


def shifted_class(currents, ip):
    """Downward-shifted (single-null) class from coil currents only: lower divertor coil
    stronger, median (P2U-P2L)/(|P2U|+|P2L|) < -0.02 (run 032), or a P6 radial-field current
    |median (P6U-P6L)/2 / Ip[kA]| x 1e3 > 4, outside the input noise band (run 042)."""
    p2u, p2l = (currents[:, SIGNALS.index(f"p2{s}_current")] for s in ("u", "l"))
    p2 = float(np.median((p2u - p2l) / (np.abs(p2u) + np.abs(p2l) + 1e-6))) < -0.02
    p6 = (currents[:, SIGNALS.index("p6u_current")] - currents[:, SIGNALS.index("p6l_current")]) / 2
    p6 = abs(float(np.median(p6 / np.clip(np.abs(ip / 1000), 1, None)))) * 1e3 > 4.0
    return p2 or p6


def symmetry_corrected(anchors, shot_anchor, prior, index, row, allowed, pr, te):
    """Per (second) frame, the radial shift of the prior solved flux that minimizes the
    midplane Te inboard/outboard mismatch (Te is a flux function); 5-frame median of valid
    shifts, nearest fill; added to the anchor. Label-free test-time adaptation (run 054)."""
    from research.flux_function_check import mismatch
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    old = np.load(Path(prior) / "shots" / f"{index:04d}.npz")["psirz"].astype(np.float64)
    deltas = np.full(len(anchors), np.nan)
    for t in range(0, len(anchors), 2):
        m = []
        for s in SHIFTS_CM:
            try:
                phi = ndimage.shift(old[t], (0, s / 100 / (r[1] - r[0])), order=1, mode="nearest")
                m.append(mismatch(phi, r, z, allowed, pr, te[t]))
            except ValueError:
                m.append(np.nan)
        m = np.array(m)
        if np.isfinite(m).sum() >= 6:
            deltas[t] = SHIFTS_CM[np.nanargmin(m)] / 100
    if np.isfinite(deltas).sum() < 3:
        return anchors
    corr = fill(deltas, np.isfinite(deltas), 5)
    return [(a if a is not None else shot_anchor) + c for a, c in zip(anchors, corr)]


def run_shot(task):
    index, entry, out, cfg = task
    config = asdict(cfg)
    target = out / "shots" / f"{index:04d}.npz"
    if target.exists():
        meta = json.loads(str(np.load(target)["meta"]))
        if meta.get("config") != config:
            raise ValueError(f"{target} was made with {meta.get('config')}, not {config}")
        return meta
    row = dict(np.load(DATA / "inputs" / (Path(entry["path"]).stem + ".npz"), allow_pickle=False))
    r, z = np.asarray(row["efit_grid_R"]), np.asarray(row["efit_grid_Z"])
    currents = aligned_currents(row)
    ext = np.zeros((len(currents), r.size * z.size))
    for column, response in coil_kernel(row):
        scale = cfg.sol_scale if SIGNALS[column] == "sol_current" else 1.0
        ext += currents[:, column, None] * 1000 * scale * response[None]
    if cfg.passive is not None:
        from research.mast_passive import passive_flux
        ext += passive_flux(row, cfg.passive, sol_scale=cfg.sol_scale)[0]
    ext = ext.reshape(-1, len(z), len(r))
    ip = currents[:, SIGNALS.index("plasma_current")] * 1000
    pr, pz, te = mast_points(row)
    allowed = region(row, cfg.inner)
    anchors = [te_axis(pr, te[t]) for t in range(len(ip))]
    valid = [a for a in anchors if a is not None]
    shot_anchor = float(np.median(valid)) if valid else AXIS_DEFAULT
    if cfg.smooth_anchor and valid:
        anchors = list(fill(np.array([a if a is not None else np.nan for a in anchors]),
                            np.array([a is not None for a in anchors]), 5))
    if cfg.anchor_prior:
        anchors = symmetry_corrected(anchors, shot_anchor, cfg.anchor_prior, index, row, allowed, pr, te)
    anchors = [a if a is not None else shot_anchor for a in anchors]
    points = [(pr[ok], pz[ok], te[t][ok]) if ok.sum() >= 5 else None
              for t, ok in enumerate(np.isfinite(te))]
    fallback = fallback_psi(row, index, cfg.sol_scale)
    shifted = cfg.vertical == "config-union" and shifted_class(currents, ip)
    z_targets = [0.0 if cfg.vertical == "config-union" else None] * len(ip)
    if shifted:  # pass 1: force balance; smoothed centroid heights, forced downward
        heights = np.full(len(ip), np.nan)
        for t in range(len(ip)):
            if ip[t] <= 0:
                continue
            try:
                _, cells = solve(ext[t], ip[t], r, z, allowed, FORCE, axis_r=anchors[t], points=points[t])
                heights[t] = float((cells * ((z[1:] + z[:-1]) / 2)[:, None]).sum() / cells.sum())
            except ValueError:
                pass
        good = np.isfinite(heights) & (np.abs(heights) <= 0.4)
        if good.any():
            z_targets = [-abs(v) for v in fill(heights, good, 7)]
    edges = [None] * len(ip)
    if cfg.edge:  # outboard LCFS at the Te = 30 eV crossing, 5-frame median
        from research.flux_function_check import te_edge_radius
        raw = np.array([te_edge_radius(pr, te[t]) for t in range(len(ip))])
        good = np.isfinite(raw) & (raw >= 1.2) & (raw <= 1.55)
        if good.sum() >= 3:
            edges = list(fill(raw, good, 5))
    profile = ZERO if cfg.te_power == 1.0 else replace(ZERO, te_power=cfg.te_power)
    fvac = np.abs(currents[:, SIGNALS.index("tf_current")]) * 1000 * MU0 * TF_TURNS / (2 * np.pi)
    psi, kept = [], []
    beta0s, betan_share = np.full(len(ip), np.nan), np.full(len(ip), np.nan)
    for t in range(len(ip)):
        info, good = {}, False
        try:
            if ip[t] <= 0:
                raise ValueError("Nonpositive plasma current")
            phi, cells = solve(ext[t], ip[t], r, z, allowed, profile, axis_r=anchors[t],
                               points=points[t], info=info, z_target=z_targets[t],
                               edge_out=edges[t], bz_gain=0.1, bz_in_output=not cfg.edge_nobz)
            good = plausible(phi, r, z, allowed)
            if good:  # pressure-share betaN (DIII-D proxy R2 0.928)
                beta0s[t] = info["beta0"]
                try:
                    betan_share[t] = beta_n_equilibrium(phi, cells, r, z, allowed, info["beta0"],
                                                        info["r0"], fvac[t], ip[t])
                except ValueError:
                    pass
        except ValueError:
            pass
        kept.append(bool(good))
        psi.append(phi.astype(np.float32) if good else fallback[t])
    meta = {"index": index, "config": config, "frames": len(ip), "kept": int(sum(kept)),
            "lsn_class": bool(shifted), "anchors_valid": len(valid), "kept_mask": kept}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".partial.npz")
    np.savez(tmp, psirz=np.array(psi), meta=json.dumps(meta), beta0=beta0s, betan_share=betan_share)
    tmp.replace(target)
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shots", default="all", help="'all' or a JSON file with shot indices")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--vertical", choices=("zero", "config-union"), default="config-union")
    p.add_argument("--inner", choices=("default", "inner"), default="inner")
    p.add_argument("--no-smooth-anchor", action="store_true")
    p.add_argument("--anchor-prior", help="previous run directory for the symmetry correction")
    p.add_argument("--edge", action="store_true")
    p.add_argument("--edge-nobz", action="store_true")
    p.add_argument("--te-power", type=float, default=1.0)
    p.add_argument("--passive", choices=("V", "VC", "ALL", "ALLg"))
    p.add_argument("--sol-scale", type=float, default=SOL_SCALE)
    a = p.parse_args()
    cfg = RunConfig(a.vertical, a.inner, not a.no_smooth_anchor, a.edge, a.edge_nobz,
                    a.te_power, a.passive, a.anchor_prior, a.sol_scale)
    shots = json.loads((DATA / "data.json").read_text())["shots"]
    indices = list(range(len(shots))) if a.shots == "all" else json.loads(Path(a.shots).read_text())
    started, metas = time.monotonic(), []
    with ProcessPoolExecutor(a.workers) as pool:
        for meta in pool.map(run_shot, [(i, shots[i], a.out, cfg) for i in indices]):
            metas.append(meta)
            if len(metas) % 50 == 0:
                print(f"{len(metas)}/{len(indices)} {time.monotonic() - started:.0f}s", flush=True)
    frames, kept = sum(m["frames"] for m in metas), sum(m["kept"] for m in metas)
    summary = {"config": asdict(cfg), "shots": len(metas), "frames": frames, "kept": kept,
               "kept_fraction": kept / frames, "seconds": time.monotonic() - started}
    (a.out / f"summary-{Path(a.shots).stem}.json").write_text(json.dumps({"summary": summary, "shots": metas}))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
