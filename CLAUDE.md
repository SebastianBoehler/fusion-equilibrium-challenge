# Repository instructions for Claude

Entry for the NeurIPS 2026 Fusion Equilibrium Challenge (closed 28 October 2026). Start
with [README](README.md), [approach](docs/approach.md) and [code map](docs/code-map.md).

## Rules that still bind this code

- **MAST path, never use:**
  - MAST or MAST-U equilibrium reconstructions from any source (EFIT, EFIT++, CPF,
    FAIR-MAST `equilibrium`);
  - the MAST demo shots;
  - `vendor/fusion_starter/fusion_scoring/masks/mast_envelope.npz`;
  - anything fitted to these.
- **MAST path, allowed:** released inputs, machine geometry, public FAIR-MAST diagnostics
  (magnetics, `pf_active`, `pf_passive` geometry), DIII-D labels, own synthetic equilibria.
- **DIII-D:** the 128 locked held-out shots are never read.
- **Do not move or delete** the submission tags (`final-submissions-1-2`,
  `final-submission-3`, `final-submission-3-v5`). Never rewrite published history.

## Working rules

- Use `PYTHONPATH=. .venv/bin/python -B -m <module>`. Keep modules below about 300 lines
  and changes surgical.
- **Regression before replacing a scored path.** A refactor of `research/mast_gs_robust.py`
  must reproduce the stored runs bit for bit; see the code map.
- Label-free proxies (Te flux-function mismatch, X-point fraction) screen out broken
  candidates; they do not rank working ones.
- Do not overstate results. Official scores are authoritative; state unknowns.
- Never commit data, weights, `run_data/`, or any `manifest.json` or ZIP from
  `push_predictions.py`, which can contain a Hugging Face token.
