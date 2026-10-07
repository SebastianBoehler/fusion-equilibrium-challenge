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
`run_data/`. This first model deliberately supports DIII-D only. Challenge 1 remains the main campaign. MAST is a separate future branch.

Completed local checks are in `RUNS.md`. The selected first-submission model
uses quadratic current interactions and Thomson profile summaries. On eight
held-out shots it reached S = 0.784501 at upload precision, trained on 64
separate shots. A fresh eight-shot check scored 0.514215. Codabench submission
**967617** scored **0.758805**, rank **88 of 96** at 23:32 Berlin on 7 October.
The 256-shot run improved the two validation groups to 0.902553 and 0.864606.
Its stronger checkpoint is submitted as **967643**, accepted at 23:50 Berlin.
The second official score is **0.916864**, rank **74 of 96** at 00:22 Berlin
on 8 October. The displayed top-five cutoff is **0.9963**.

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

The completed 256-shot run trains the same model and reports the two
validation groups separately:

```sh
python -m participant.scale --manifest protocols/scale-256-data.json \
  --cache run_data/scale-256/data --out run_data/scale-256/result
```

Development closes 18 October 2026 at 00:00 UTC. Final closes 26 October at
00:00 UTC. Challenge 2 requires DIII-D composite S >= 0.85 and predictions for
both machines. Podium authorship/talks are decided by final verification.

The broader campaign is frozen in `protocols/breadth-004.json`. The generator
uses the complete verified training catalog to reserve 256 development shots
and 128 locked confirmation shots. The old 16 shots are regression checks.
The first one-hour CPU pilot compares 256 and 1,024 shots with fixed alpha and
fixed alpha/N. Float32 target storage is checked lossless against the release;
features stay float64. Fitting and scoring stream one shot at a time.

Verify the new numerical and scoring path on real released shots:

```sh
OPENBLAS_NUM_THREADS=4 python -m tests.verify_pipeline
```
