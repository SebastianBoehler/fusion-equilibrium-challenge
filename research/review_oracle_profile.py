"""Oracle ceiling: per-frame in-sample selection among existing ALLgf profile variants by plasma-loop residual."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from research.fairmast_harness import loop_values, r2
D = Path("run_data/fairmast"); data = D / "data2"
runs = ["d2-v4-ALLgf", "d2-gf-te05", "d2-gf-te10", "d2-gf-te15", "d2-gf-e10", "d2-gf-e20"]
P = {}
for n in runs:
    pred, ext, meas, names = loop_values(D / n, data)
    P[n] = (pred - ext, meas - ext, names)
    print(n, pred.shape)
base = runs[0]; mp, names = P[base][1], P[base][2]
ok = np.isfinite(mp)
broken = [j for j, nm in enumerate(names) if r2(P[base][0][:, j] + 0, mp[:, j]) < -5]
use = [j for j in range(len(names)) if j not in broken]
scale = np.nanstd(mp, axis=0)
def score(sel):
    cc = np.nanmean([r2(sel[:, j], mp[:, j]) for j in use if names[j].startswith("CC")])
    co = np.nanmean([r2(sel[:, j], mp[:, j]) for j in use if names[j].startswith("P")])
    return round(cc, 3), round(co, 3)
for n in runs: print(f"{n:14s} plasma CC/coils", score(P[n][0]))
res = np.stack([np.nansum(((P[n][0] - mp)[:, use] / scale[use]) ** 2, axis=1) for n in runs])
pick = res.argmin(0)
sel = np.stack([P[n][0] for n in runs])[pick, np.arange(len(pick))]
print("oracle pick counts", dict(zip(runs, np.bincount(pick, minlength=len(runs)))))
print("ORACLE plasma CC/coils", score(sel))
