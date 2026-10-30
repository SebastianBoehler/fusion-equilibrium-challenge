# Reproduction package

This covers the Final requirements (forum, 23 September and October): source with a
pinned environment, exact weights, a training entrypoint, an inference entrypoint,
provenance, and declared hardware and wall time.

Environment: `uv sync` (`uv.lock`; Python 3.11, torch 2.14.1). Run modules as
`PYTHONPATH=. .venv/bin/python -B -m <module>`.

## Provenance

| Machine | Data used | External or pretrained models |
|---|---|---|
| DIII-D | `Sophelio/fusion-equilibrium-challenge` `diii_d_train` at revision `1e280905…47cb`; split in `diii_d_split.json` (6,641 train / 256 development / 16 regression; the 128 locked shots are never read). Final submission 3: 22 members (`nn_members.json` + `nn_members_d1.json`) | None |
| MAST | Released `mast_*_test` inputs: coil currents, Ip, TF, Thomson Te, grid. MAST machine geometry: coils from the release; passive structure from the public FAIR-MAST `pf_passive` group (`research/geometry/`); literature resistivities. **Public FAIR-MAST level-2 diagnostics** (no equilibria): `magnetics` (ip, flux loops, pickup coils) and `pf_active` (coil currents), for validation on 301 non-test shots and, in Final submission 3, for re-identified test shots (see below) | None. No MAST equilibrium labels or EFIT/CPF products, no demos, no envelope, no training on MAST |

**Development-feedback use, disclosed:**
- MAST variants and channels (flux, q95, βN source) were selected by Development scores.
- Official per-shot worst lists informed the vertical-class design. The final rule uses
  coil currents only: P2 asymmetry or P6 radial field.

**Final submission 3: re-identification and public magnetics, disclosed:**
- On 22 Oct 2026 the entrant decided, without consulting the organizers first, to re-identify
  anonymised MAST private-test shots. Released Ip(t) was matched to public FAIR-MAST level-2
  `magnetics/ip` (`research/fairmast_ip_index.py`, `research/match_test_shots.py`).
- **Matching rule (predeclared):** RMS Ip distance below 0.003 of max |Ip|, and the next-best
  shot at least 5× further away. A total-flux-loop check removes matches with inconsistent
  loop data (`research/match_verify.py`). 845 of 1,205 shots matched.
- **What the matched data changed:** the release PF and solenoid currents differ from the
  FAIR-MAST level-2 `pf_active` currents of the same shot (P2 3–5%, P5 up to 5–7%). For the
  664 matched shots of the centred vertical class, the solver is re-run with the level-2
  currents; the vertical class stays the release-input decision (`research/build_l2_inputs.py`).
- **Why:** flux loops never enter the solver, yet the measured loops agree better with the
  level-2 solution on 78% of those shots. Shifted-class shots were worse (41%) and keep the
  release-current solution. This class split was chosen with the same loop data, which makes
  it a data-informed selection.
- **Not used:** a per-frame magnetics fit (`research/mag_tta*.py`) was built and validated on
  non-test shots but not submitted.
- **Never read:** FAIR-MAST `equilibrium` and any EFIT, EFIT++ or CPF-derived quantity.

**DIII-D Final submission 3:**
- The ensemble grows to 22 members: 11 new seeds of the same configurations, trained on the
  same split.
- Development S: 0.996037 → 0.996130. This misses the predeclared +0.0001 gate by 7e-6, and the
  entrant adopted it anyway.

**AI assistance:** Claude (Anthropic) implemented, ran and validated the pipelines under
the entrant's direction.

## DIII-D: rebuild from scratch

1. **Download TRAIN.** Hash-verified, about 92 GB. All files are fetched, but only split shots are read:
   ```bash
   python -m participant.final_manifest --repo-id Sophelio/fusion-equilibrium-challenge \
     --config diii_d_train --revision 1e280905b85f2a6fdde7e06fca8cf3a1edf447cb \
     --out data/train --download-to data/train/raw
   ```
2. **Prepare features.** CPU, about 3 min on 32 cores:
   ```bash
   python -m participant.nn_prepare --split repro/diii_d_split.json --raw data/train/raw \
     --out data/packed --workers 32
   ```
3. **Train the 11 members.** Each is one A100 task; configurations and seeds are in
   `nn_members.json`:
   ```bash
   sbatch --array=0-10 --export=ALL,PACKED=data/packed,RUNS=data/runs ops/nn-train.sbatch
   ```
   Measured training time is **2.84 A100-hours** for the 11 members in `nn_members.json` (12–37 min each) plus **3.23 A100-job-hours** for the 11 in `nn_members_d1.json` (14–39 min each, including development scoring); about 6 A100-hours in total on an
   A100-PCIE-40GB, well under the 24-hour ceiling. GPU training may not be bit-deterministic (unverified).
   The shipped weights are the exact ones used (SHA256 values in `nn_members.json`).
4. **Inference.** CPU, about 4 min. See `runs/final-runbook.md` step 2.
   - Rebuilding the public predictions with today's environment matched the submitted
     file to float32 rounding (max abs 1.9e-6).

## MAST: label-free, CPU only (no training)

- **Steps:** see `runs/final-runbook.md` step 3: `research.mast_gs_robust`, then
  `research.mast_final`.
- **Hardware:** about 60–100 min on 12–14 CPU cores (Apple M-series or Slurm CPU nodes).
  There is no GPU work.
- **Determinism:** deterministic given the inputs. Scored public-test runs reproduce bit
  for bit (`docs/code-map.md`).

## Weights

The 22 `model.pt` files (about 2.7 GB, each over 100 MB) are to be hosted with SHA256 hashes. The host is a pending human decision. Their hashes are in `nn_members.json` and `nn_members_d1.json`.

**MAST Final submission 3 (L2-u) rebuild:** `ops/final/build-l2.sh` (base re-solve with level-2 currents; its TTA tail is not used) and then `ops/final/build-l2u.sh`. Both are recorded in `runs/mag-tta-20261022.md`, and they need the FAIR-MAST `pf_active` and `magnetics` downloads (`research/fairmast_pf.py`, `research/fairmast_test_magnetics.py`).
