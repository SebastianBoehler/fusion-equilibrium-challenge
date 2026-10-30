# Code map (after the 18 October cleanup)

Run everything with `PYTHONPATH=. .venv/bin/python -B -m <module>`. Removed legacy code is
at tag `pre-cleanup-20261018`.

## DIII-D (Award 1, and the Award 2 denominator)

| Module | Role |
|---|---|
| `participant/data.py` | Pinned DIII-D shot reader (cache first, hash-verified), base features |
| `participant/rich.py` | Actuator-history and Thomson features; applies the organizer Ip timing fix |
| `participant/nn_prepare.py` | Packs TRAIN splits into memory maps (training input) |
| `participant/nn_train.py` | Residual MLP → PCA flux basis + q95/βN heads (one ensemble member) |
| `participant/flux_head.py` | PCA/direct flux decoding |
| `participant/nn_submit.py` | Ensemble inference → `diii_d_<split>.npz` + archive (manifest + stream order) |
| `participant/score.py`, `participant/fast_score.py` | Unmodified organizer scorer on held-out TRAIN shots (serial and parallel) |

## MAST (Award 2 numerator; label-free)

| Module | Role |
|---|---|
| `participant/mast_inputs.py` | Allowlisted input reader (rejects any `efit_*` label column), cache, current alignment |
| `participant/mast_green.py` | Coil Green kernels and the 002 Green-model baseline (fallback flux; kinetic βN) |
| `participant/mast_surface.py` | Surface q integral (used by checks) |
| `participant/mast_submit.py` | Archive validation helpers (`validate`, `digest`) |
| `research/mast_gs.py` | Machine constants (`SOL_SCALE`), base allowed region |
| `research/gs_picard.py` | Free-boundary Picard GS solver (LCFS bisection, vertical control, edge B_Z option) |
| `research/mast_gs_robust.py` | **Solver runner**: anchors, vertical class (`config-union`), passive, edge, anchor correction; per-shot outputs with config |
| `research/mast_passive.py`, `research/passive.py` | Passive-structure filament circuits from the public EFIT geometry (`research/geometry/`) |
| `research/thomson_points.py` | Thomson (R, Z, Te) per frame, both machines |
| `research/q95_proxy.py` | `q_at`: field-line q at ψN 0.95; DIII-D validation main |
| `research/betan.py` | βN from the flux pressure split / the solver pressure share / electron pressure |
| `research/flux_function_check.py` | Te flux-function mismatch, Te = 30 eV edge (label-free checks; anchor correction) |
| `research/mast_final.py` | **Final builder**: one solver run → smoothed flux, q95, βN (`flux`, `share`, `flux-kin`) |
| `research/temporal_smooth.py` | Within-shot smoothing helper (`smooth_psi`) and the 026-style builder |
| `research/ensemble_channels.py` | Channel-wise combination of built archives; full archives with the hash-checked DIII-D member |
| `research/gs_proxy_d3d.py`, `research/mast_proxy_d3d.py` | DIII-D proxy harness for MAST physics choices (labels as targets only) |

## Final tools and ops

| Path | Role |
|---|---|
| `participant/final_manifest.py` | Manifest + stream order for any config; `--check-public` reproduces both public manifests |
| `participant/final_archive.py` | Validated submission ZIP from per-config members (public or private names) |
| `ops/build-mast-candidate.sh` | Solver run → MAST-only candidate (public split) |
| `ops/mast-gs-array.sbatch` | 4-shard Slurm CPU array for `mast_gs_robust` |
| `ops/cpu-requirements.txt` | Pinned cluster CPU environment |
| `ops/final/build-l2.sh`, `ops/final/build-l2u.sh` | Final submission 3 MAST rebuild (level-2 currents for re-identified shots) |
| `ops/nn-train.sbatch` | One DIII-D ensemble member per A100 array task (`repro/nn_members.json`) |
| `repro/` | Reproduction package: provenance, DIII-D split, member configs, rebuild steps |

## Regression check for runner refactors

Run union (shots 26, 1 and 0, one per class), `--passive ALLg`, `--anchor-prior
run_data/mast-044` and `--edge --edge-nobz` on shot 1. Each output must equal the stored run
bit for bit:

| Variant | Stored run |
|---|---|
| union | `run_data/mast-044` |
| `--passive ALLg` | `run_data/mast-053L` |
| `--anchor-prior` | `run_data/mast-054L` |
| `--edge --edge-nobz` | `run_data/mast-045` |

All four passed on 18 October after the cleanup.
