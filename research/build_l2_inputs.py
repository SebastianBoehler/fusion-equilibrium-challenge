"""Inputs for matched test shots with FAIR-MAST level-2 PF/solenoid currents (candidate L2).

Per matched shot with pf_active: the release row with its PF + solenoid currents replaced by
level-2 pf_active (research.coil_source_check.level2_row); Ip, TF, Thomson, grid and geometry
stay release. The vertical class is frozen to the release-input decision via `shifted_override`
(the class rule was tuned on release currents). Unmatched shots are symlinked unchanged.
Motivation: measured flux loops agree better with level-2 currents on 75% of 61 sampled shots.
Usage: python -m research.build_l2_inputs DATA MATCH.json PF_DIR OUT
"""

import json
import sys
from pathlib import Path

import numpy as np

from participant.mast_inputs import SIGNALS, aligned_currents
from research.coil_source_check import level2_row
from research.mast_gs_robust import shifted_class


def main():
    data, match, pf_dir, out = (Path(x) for x in sys.argv[1:5])
    (out / "inputs").mkdir(parents=True, exist_ok=False)
    (out / "data.json").write_text((data / "data.json").read_text())
    shots = json.loads((data / "data.json").read_text())["shots"]
    matched = {m["i"]: m["shot"] for m in json.loads(match.read_text())}
    done, changed = [], 0
    for i, e in enumerate(shots):
        name = Path(e["path"]).stem
        src = (data / "inputs" / f"{name}.npz").resolve()
        pf = pf_dir / f"{matched.get(i)}.npz"
        if i not in matched or not pf.exists():
            (out / "inputs" / f"{name}.npz").symlink_to(src)
            continue
        row = dict(np.load(src, allow_pickle=False))
        cur = aligned_currents(row)
        cls = shifted_class(cur, cur[:, SIGNALS.index("plasma_current")] * 1000)
        new = level2_row(row, np.load(pf))
        new["shifted_override"] = np.array(cls)
        np.savez(out / "inputs" / f"{name}.npz", **new)
        done.append(i)
        changed += 1
    (out / "l2-shots.json").write_text(json.dumps(done))
    print("level-2 inputs for", changed, "shots; others unchanged")


if __name__ == "__main__":
    main()
