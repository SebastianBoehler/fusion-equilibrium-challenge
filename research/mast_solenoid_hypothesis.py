"""MAST 021: does a 2x solenoid-current error explain official MAST R2_psi = 0.7646?

Hypothesis H_k: the true external flux uses k times our solenoid ampere-turns. Under H_k the
truth differs from our submitted flux by (k - 1) * psi_sol, so the official pooled R2 and the
official worst-shot ranking are predicted without any MAST label. Inputs: the exact submitted
976708 flux file and pinned MAST input caches (currents and coil geometry only).
"""

import json
from pathlib import Path

import numpy as np

from participant.mast_green import coil_kernel
from participant.mast_inputs import SIGNALS, aligned_currents

ROOT = Path("run_data")
OFFICIAL_WORST = {  # official 976708 report: lowest per-shot R2_psi
    1070: .1703, 1170: .1891, 788: .1975, 197: .2078, 233: .2085, 654: .2676, 678: .2827,
    1194: .3115, 117: .3135, 481: .321, 60: .3288, 424: .3353, 28: .3363, 730: .3381,
    351: .3383, 219: .3412, 194: .3423, 317: .3592, 756: .3593, 827: .3623}
OFFICIAL_R2 = .764565
SCALES = (0.0, 0.25, 0.5, 0.6, 0.75, 1.25, 1.5, 2.0)


def main() -> None:
    shots = json.loads((ROOT / "mast-001/data.json").read_text())["shots"]
    submitted = np.load(ROOT / "submission-mast-002/mast_public_test.npz")
    sol_index = SIGNALS.index("sol_current")
    geometry, response = None, None
    psum = psq = n = 0.0
    parts = []
    for i, entry in enumerate(shots):
        row = dict(np.load(ROOT / "mast-001/inputs" / (Path(entry["path"]).stem + ".npz"),
                           allow_pickle=False))
        key = tuple(np.asarray(row[k]).tobytes() for k in
                    ("coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
                     "efit_grid_R", "efit_grid_Z"))
        if geometry is None:
            geometry = key
            response = dict(coil_kernel(row))[sol_index]
        elif key != geometry:
            raise ValueError(f"Coil/grid geometry differs at shot {i}")
        pred = submitted[f"shot_{i:04d}_psirz"].astype(np.float64).reshape(-1, 65 * 65)
        sol = aligned_currents(row)[:, sol_index, None] * 1000 * response[None]
        if sol.shape != pred.shape:
            raise ValueError(f"Frame mismatch at shot {i}")
        parts.append([pred.size, pred.sum(), np.square(pred).sum(), sol.sum(),
                      np.square(sol).sum(), (pred * sol).sum()])
        if i % 200 == 0:
            print("shot", i, flush=True)
    n, sp, spp, ss, sss, sps = np.array(parts).T
    stats = {}
    for k in SCALES:
        d = k - 1
        st, stt = sp + d * ss, spp + 2 * d * sps + d * d * sss  # sums of t and t^2
        res = d * d * sss
        tot = stt.sum() - st.sum() ** 2 / n.sum()
        per_shot = 1 - res / (stt - st ** 2 / n)
        order = np.argsort(per_shot)
        ranks = [int(np.where(order == i)[0][0]) for i in OFFICIAL_WORST]
        stats[k] = {"pooled_r2_if_true": float(1 - res.sum() / tot),
                    "worst20_overlap": len(set(order[:20].tolist()) & set(OFFICIAL_WORST)),
                    "official_worst_median_rank": float(np.median(ranks)),
                    "official_worst_sim_r2_median": float(np.median(
                        [per_shot[i] for i in OFFICIAL_WORST]))}
        print(k, json.dumps(stats[k]), flush=True)
    sol_var = sss.sum() - ss.sum() ** 2 / n.sum()
    pred_var = spp.sum() - sp.sum() ** 2 / n.sum()
    out = {"official_r2_psi": OFFICIAL_R2, "scales": stats,
           "solenoid_variance_share_of_prediction": float(sol_var / pred_var)}
    (ROOT / "mast-021").mkdir(exist_ok=True)
    (ROOT / "mast-021/solenoid-hypothesis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "scales"}))


if __name__ == "__main__":
    main()
