# Fusion Equilibrium Challenge: DIII-D ensemble and label-free MAST Grad–Shafranov

Our entry for the [NeurIPS 2026 Fusion Equilibrium Challenge](https://fusion-equilibrium-challenge.sophelio.io/)
(FusionEC-26, Codabench competition 17739). The task is to predict poloidal flux maps ψ(R, Z),
q95 and βN from tokamak diagnostics:
- **Challenge 1:** DIII-D, where models may learn from labels;
- **Challenge 2:** zero-shot transfer to MAST, without any MAST equilibrium labels.

| | Approach | Development board (18 Oct) |
|---|---|---|
| **DIII-D** | Ensemble of residual MLPs → PCA flux basis + q95/βN heads | S 0.9951 |
| **MAST** | Free-boundary Grad–Shafranov solve, J ∝ Te(ψN)^0.75 from Thomson, measured coils and passive structure; no learning, no labels | S 0.775, G 0.779 (11th of 50) |

**Final standings (provisional, 30 Oct 2026):** 19th of 58 in Challenge 1 and 16th of 46 in
Challenge 2. The organizers have not published scores yet, and places may change after
their reproduction review.

The method, results and lessons are in **[docs/approach.md](docs/approach.md)**.

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # pinned environment from uv.lock
uv sync --group torch    # adds PyTorch, needed for DIII-D training and inference
```

Run modules from the repository root:

```bash
PYTHONPATH=. .venv/bin/python -B -m <module> --help
```

The challenge data is public on Hugging Face (`Sophelio/fusion-equilibrium-challenge` and
`Sophelio/fusion-equilibrium-challenge-final`). The readers download it at pinned revisions
and check every file's hash.

- **DIII-D training and inference:** [repro/README.md](repro/README.md).
- **MAST solve and Final archive:** [runs/final-runbook.md](runs/final-runbook.md), steps 1–4.
  The MAST path runs on CPU only, in about 60–100 minutes on 12 cores for the 1,205
  private-test shots.

## Repository layout

| Path | Contents |
|---|---|
| `participant/` | DIII-D models, data readers, scorer adapters, MAST input reader, submission manifest and archive tools |
| `research/` | MAST Grad–Shafranov solver, passive-structure model, q95/βN estimators, FAIR-MAST magnetics harness, DIII-D proxy harness |
| `repro/` | Reproduction package: provenance, DIII-D split, ensemble member configs and weight hashes |
| `ops/` | Slurm launchers (DIII-D training, sharded MAST solve) and the Final submission 3 build scripts |
| `tests/` | Targeted regression and diagnostic checks; [tests/README.md](tests/README.md) explains the scope of each |
| `docs/` | [Approach](docs/approach.md) and [code map](docs/code-map.md) |
| `runs/` | Dated lab notes from the competition ([index](runs/README.md)) |
| `vendor/fusion_starter/` | Organizer scorer and data fixes, copied unchanged ([VENDORED.md](VENDORED.md)) |

Data, run outputs and model weights are not in the repository (see `.gitignore`).

## Compliance and disclosure

The MAST predictions never use any of these:
- MAST equilibrium reconstructions (EFIT, EFIT++, CPF or the FAIR-MAST `equilibrium` group);
- the MAST demo shots;
- the organizer's `mast_envelope.npz` mask.

`participant/mast_inputs.py` rejects label columns when reading. We used three kinds of data
that need disclosure. Each is stated in [repro/README.md](repro/README.md):
- Development-leaderboard feedback, to choose between submitted variants;
- public FAIR-MAST magnetics, on 301 non-test shots, for validation;
- in Final submission 3, re-identification of anonymised test shots, to use their public
  level-2 coil currents.

Most of the code was written with Claude (Anthropic), working as a coding agent under the
author's direction.

## Submission snapshots

These tags pin the exact code for each Final submission. Keep them:
- `final-submissions-1-2`;
- `final-submission-3`;
- `final-submission-3-v5`.

`pre-cleanup-20261018` holds earlier code: the DIII-D ridge, contour and critical-point
campaigns. Those campaigns' run notes are in the history before this cleanup.

## Citation

See [CITATION.cff](CITATION.cff). If you use the MAST solver or the magnetics harness,
please cite this repository.

## License

MIT ([LICENSE](LICENSE)). The vendored organizer starter keeps its own MIT license.
