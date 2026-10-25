"""Equal-weight ensemble of members' development predictions, scored with the official functions.

Averages runs/<name>/development.npz (psirz, q95, betaN) over the given members in float32,
saves float16 like the members, then runs participant.fast_score on it.
Usage (cluster, from the repo copy): python -m participant.ensemble_dev ROOT OUT_NAME MEMBER [MEMBER ...]
"""

import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    root, out_name, members = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    zs = [np.load(root / "runs" / m / "development.npz") for m in members]
    keys = zs[0].files
    out = {}
    for k in keys:
        out[k] = np.mean([z[k].astype(np.float32) for z in zs], axis=0).astype(np.float16)
    d = root / "runs" / out_name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "development.npz", **out)
    (d / "members.txt").write_text("\n".join(members))
    subprocess.run([sys.executable, "-u", "-m", "participant.fast_score", "--packed", str(root / "packed/development"),
                    "--predictions", str(d / "development.npz"),
                    "--reference-cache", str(root / "packed/development-refs.pkl"), "--workers", "16"], check=True)


if __name__ == "__main__":
    main()
