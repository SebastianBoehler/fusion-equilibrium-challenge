# Fusion Equilibrium Challenge — participant code

Our entry for the [NeurIPS 2026 Fusion Equilibrium Challenge](https://www.codabench.org/competitions/17739/):
predict poloidal flux maps, q95 and βN from diagnostics, for DIII-D (Award 1) and
zero-shot for MAST (Award 2, G = S_MAST / S_DIII-D).

**DIII-D: an ensemble of 11 residual MLPs.**
- Trained on DIII-D TRAIN. Inputs are actuator histories and Thomson features; outputs are
  a PCA flux basis plus scalar heads (`participant/nn_*`, `participant/rich.py`).
- Public-test S 0.995.

**MAST: a label-free physics pipeline.**
- **Inputs:** measured coil currents (solenoid at half the feed current per element);
  optional passive-structure currents from the public EFIT geometry.
- **Plasma:** a free-boundary Grad–Shafranov solve with J ∝ Te(ψN) from Thomson scattering
  and a Te-based axis anchor.
- **Vertical position:** a coil-current class for downward-shifted plasmas.
- **Post-processing:** within-shot temporal smoothing. q95 comes from a field-line integral;
  βN from the equilibrium pressure split, optionally averaged with the kinetic estimate.
- **No MAST equilibrium labels, demos or envelope are used** (`research/mast_gs_robust.py`,
  `research/mast_final.py`).

## Layout

| Path | Contents |
|---|---|
| `participant/` | DIII-D models, data readers, scorer adapters, MAST inputs, Final manifest/archive tools |
| `research/` | MAST solver, passive model, scalar estimators, archive builders, DIII-D proxy harness |
| `tests/` | Targeted checks (see `tests/README.md`; not a blanket test suite) |
| `ops/` | Cluster array launcher, candidate builder, pinned CPU requirements |
| `vendor/fusion_starter/` | Unmodified organizer scorer and data fixes (`VENDORED.md`) |
| `docs/` | Forum rules brief, handoff, code map |
| `runs/` | Dated run notes; `runs/final-runbook.md` has the Final-phase commands |

Setup: `uv sync`, then run modules with `PYTHONPATH=. .venv/bin/python -B -m <module>`.
Run evidence (`run_data/`, `.journal/`) is local and git-ignored.

**Licence:** MIT (`LICENSE`); the organizer starter keeps its own MIT licence.
