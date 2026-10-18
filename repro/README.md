# Reproduction package

This covers the Final requirements (forum, 23 September and October): source with a
pinned environment, exact weights, a training entrypoint, an inference entrypoint,
provenance, and declared hardware and wall time.

Environment: `uv sync` (`uv.lock`; Python 3.11, torch 2.14.1). Run modules as
`PYTHONPATH=. .venv/bin/python -B -m <module>`.

## Provenance

| Machine | Data used | External or pretrained models |
|---|---|---|
| DIII-D | `Sophelio/fusion-equilibrium-challenge` `diii_d_train` at revision `1e280905…47cb`; split in `diii_d_split.json` (6,641 train / 256 development / 16 regression; the 128 locked shots are never read) | None |
| MAST | Released `mast_*_test` **inputs** only: coil currents, Ip, TF, Thomson Te, grid; MAST machine geometry (coils from the release; passive-structure geometry from the public FAIR-MAST `pf_passive` group, `research/geometry/`); literature resistivities | None. No MAST equilibrium labels, demos or envelope; no training on MAST |

**Development-feedback use, disclosed:**
- MAST variants and channels (flux, q95, βN source) were selected by Development scores.
- Official per-shot worst lists informed the vertical-class design. The final rule uses
  coil currents only: P2 asymmetry or P6 radial field.

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
   Measured training time is **2.84 A100-hours** in total (12–37 min per member) on an
   A100-PCIE-40GB, well under the 24-hour ceiling. GPU training may not be bit-deterministic (unverified).
   The shipped weights are the exact ones used (SHA256 values in `nn_members.json`).
4. **Inference.** CPU, about 4 min. See `runs/final-runbook.md` step 2.
   - Rebuilding the public predictions with today's environment matched the submitted
     file to float32 rounding (max abs 1.9e-6).

## MAST: label-free, CPU only (no training)

- **Steps:** see `runs/final-runbook.md` step 3: `research.mast_gs_robust`, then
  `research.mast_final`.
- **Hardware:** about 60–100 min on 12–14 CPU cores (Apple M-series or Galvani CPU nodes).
  There is no GPU work.
- **Determinism:** deterministic given the inputs. Scored public-test runs reproduce bit
  for bit (`docs/handoff/CODE_MAP.md`).

## Weights

The 11 `model.pt` files (1.34 GB in total, each over 100 MB) are to be hosted with SHA256
hashes; the host is a pending human decision. Their hashes are in `nn_members.json`.
