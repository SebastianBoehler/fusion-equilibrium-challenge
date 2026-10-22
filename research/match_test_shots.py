"""Match released (anonymised) MAST shots to FAIR-MAST shot numbers by their Ip(t) trace.

Human decision 22 Oct 2026 (disclosed in provenance): re-identification is allowed so that the
public FAIR-MAST magnetics of the test shots can be used for label-free test-time adaptation.
Only Ip traces are compared (research/fairmast_ip_index.py); no labels or EFIT data are involved.
Distance: RMS of the Ip difference over the common time window where either |Ip| > 30 kA,
relative to the released Ip maximum; both signs of the FAIR-MAST trace are tried.
Usage: python -m research.match_test_shots DATA_DIR INDEX_DIR OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def load_index(index_dir):
    parts = [np.load(f) for f in sorted(Path(index_dir).glob("chunk*.npz"))]
    shots = np.concatenate([p["shots"] for p in parts])
    fp = np.concatenate([p["fp"] for p in parts]).astype(float)
    shots, first = np.unique(shots, return_index=True)        # overlapping index runs
    return shots, fp[first], parts[0]["grid"]


def release_fp(row, grid):
    return np.interp(grid * 1000, np.asarray(row["magnetics_time"], float),
                     np.asarray(row["magnetics_plasma_current"], float), left=np.nan, right=np.nan)


def match(fp_rel, shots, fp):
    best = []
    for sgn in (1.0, -1.0):
        f = sgn * fp
        act = (np.abs(f) > 30) | (np.abs(fp_rel) > 30)[None]
        ok = act & np.isfinite(f) & np.isfinite(fp_rel)[None]
        n = ok.sum(1)
        d = np.where(ok, (f - fp_rel[None]) ** 2, 0).sum(1) / np.maximum(n, 1)
        cover = n / np.maximum(act.sum(1), 1)
        d = np.sqrt(d) / max(np.nanmax(np.abs(fp_rel)), 1)
        d[cover < 0.9] = np.inf
        best.append(d)
    d = np.minimum(*best)
    order = np.argsort(d)
    return int(shots[order[0]]), float(d[order[0]]), float(d[order[1]])


def main():
    data, index_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    shots, fp, grid = load_index(index_dir)
    entries = json.loads((data / "data.json").read_text())["shots"]
    res = []
    for i, e in enumerate(entries):
        row = np.load(data / "inputs" / f"{Path(e['path']).stem}.npz")
        s, d1, d2 = match(release_fp(row, grid), shots, fp)
        res.append({"i": i, "shot": s, "dist": round(d1, 5), "second": round(d2, 5), "truth": e.get("shot")})
    out.write_text(json.dumps(res, indent=0))
    d = np.array([r["dist"] for r in res]); m = np.array([r["second"] / max(r["dist"], 1e-9) for r in res])
    print(f"{len(res)} shots | dist p50/p90/max {np.round(np.percentile(d, [50, 90, 100]), 4)} | margin p10 {np.percentile(m, 10):.1f}")
    if all(r["truth"] is not None for r in res):
        print("self-check accuracy", np.mean([r["shot"] == r["truth"] for r in res]))
    print("duplicate matches:", len(res) - len({r["shot"] for r in res}))


if __name__ == "__main__":
    main()
