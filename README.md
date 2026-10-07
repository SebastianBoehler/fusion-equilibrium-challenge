# Fusion Equilibrium Challenge — participant workspace

Goal: a reproducible entry, with a NeurIPS presentation or joint-paper contribution.
The official starter remains at remote `upstream`; our private remote is `origin`.
Organizer scoring modules remain unchanged.

Install the pinned CPU environment with `uv sync`. The official dependency is
under `vendor/fusion_starter/`; its revision is recorded in `VENDORED.md`.

## First experiment

- Eight released DIII-D training shots; two disjoint held-out training shots.
- Standardized actuator currents, with the official plasma-current timestamp fix.
- Direct ridge regression for flux, q95, and betaN; alpha fixed at 1.
- No public-test fitting or calibration. No MAST labels, including demo labels.
- Perfect-target scorer self-check; checkpoint reload and target-redaction checks.
- Report the official composite and components. This is a pipeline smoke test,
  not a reliable population performance estimate or leaderboard prediction.
- Galvani CPU job: four CPUs, 16 GiB, no GPU, hard wall-time 20 minutes.

Code: `participant/`. Frozen data/source/job manifests and results: ignored
`run_data/`. This first model deliberately supports DIII-D only. Cross-machine
geometry features are the next experiment; no placeholder MAST predictions.

Completed local checks are in `RUNS.md`. The selected first-submission model
uses quadratic current interactions and Thomson profile summaries. On eight
held-out shots it reached S = 0.784501 at upload precision, trained on 64
separate shots. A fresh eight-shot check scored 0.514215. Codabench accepted
submission **967617** on 7 October 2026 at 23:15 Berlin; scoring remains pending.

Run a frozen manifest:

```sh
python -m participant.smoke --manifest run_data/smoke-001/data.json \
  --out run_data/smoke-001/result
```

The broader frozen split is `protocols/iteration-002-data.json`.
`participant.kinetic_compare` checks raw target redaction, saved-model reload,
and the exact upload precision. `participant.public_inputs` prepares the public
inputs; `participant.submit_build` verifies their hashes, official stream order,
all 874 shots, every array shape, and finite outputs before writing the upload.
Public inputs never enter fitting, normalization, or candidate selection.

The next frozen run trains the same model on 256 shots and reports the two
validation groups separately:

```sh
python -m participant.scale --manifest protocols/scale-256-data.json \
  --cache run_data/scale-256/data --out run_data/scale-256/result
```

Development closes 18 October 2026 at 00:00 UTC. Final closes 26 October at
00:00 UTC. Challenge 2 requires DIII-D composite S >= 0.85 and predictions for
both machines. Podium authorship/talks are decided by final verification.
