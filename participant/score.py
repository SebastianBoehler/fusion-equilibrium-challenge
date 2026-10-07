"""Adapt real held-out rows to the unmodified organizer composite scorer."""

import numpy as np

from vendor.fusion_starter import local_score as official


def composite(rows: list[dict], predictions: list[dict]) -> dict:
    mask = np.load(official.HERE / "fusion_scoring/masks/d3d_envelope.npz")
    r, z = mask["grid_R"], mask["grid_Z"]
    coarse = mask["mask_coarse"].astype(bool)
    mask_f = coarse.astype(float)
    shots = [{"psi": np.asarray(row["efit_psirz"], dtype=float),
              "q95": np.asarray(row["efit_q95"], dtype=float),
              "betaN": np.asarray(row["efit_beta_n"], dtype=float)} for row in rows]
    refs = [official.build_reference(s["psi"], r, z, coarse, mask_f) for s in shots]
    psi = np.concatenate([s["psi"].ravel() for s in shots])
    scal = np.column_stack([np.concatenate([s[k] for s in shots])
                            for k in ("q95", "betaN")])
    cons = np.concatenate([ref[1] for ref in refs])
    finite_scal, finite_cons = np.isfinite(scal), np.isfinite(cons)
    if not np.isfinite(psi).all():
        raise ValueError("Nonfinite held-out flux targets")
    stats = {"psi_sum": psi.sum(), "psi_sumsq": np.square(psi).sum(),
             "psi_n": psi.size, "scal_sum": np.nansum(scal, axis=0),
             "scal_sumsq": np.nansum(np.square(scal), axis=0),
             "scal_n": finite_scal.sum(axis=0), "cons_sum": np.nansum(cons, axis=0),
             "cons_sumsq": np.nansum(np.square(cons), axis=0),
             "cons_n": finite_cons.sum(axis=0)}
    means = (psi.mean(), np.divide(stats["scal_sum"], stats["scal_n"],
                                  out=np.zeros(2), where=stats["scal_n"] > 0),
             np.divide(stats["cons_sum"], stats["cons_n"],
                       out=np.zeros(7), where=stats["cons_n"] > 0))
    totals = {sign: 0.0 for sign in official.PSI_SIGNS}
    for shot, prediction in zip(shots, predictions, strict=True):
        residuals, _, _ = official.psi_residuals(shot["psi"], prediction["psirz"], means[0])
        for sign in totals:
            totals[sign] += residuals[sign]
    sign = min(totals, key=totals.get)
    acc = official.Accum()
    acc.psi_sign = sign
    for shot, reference, prediction in zip(shots, refs, predictions, strict=True):
        acc.add(official.score_shot(shot, reference, prediction, r, z, coarse,
                                   mask_f, sign, means))
    return official.finalize_machine(acc, stats)
