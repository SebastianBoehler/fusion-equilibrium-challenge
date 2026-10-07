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

Run a frozen manifest:

```sh
python -m participant.smoke --manifest run_data/smoke-001/data.json \
  --out run_data/smoke-001/result
```

Development closes 18 October 2026 at 00:00 UTC. Final closes 26 October at
00:00 UTC. Challenge 2 requires DIII-D composite S >= 0.85 and predictions for
both machines. Podium authorship/talks are decided by final verification.
