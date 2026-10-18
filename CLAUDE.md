# Repository instructions for Claude

NeurIPS 2026 Fusion Equilibrium Challenge (Codabench 17739). Award 1: DIII-D composite S.
Award 2: G = S_MAST / best S_DIII-D (needs DIII-D S ≥ 0.85). Read first:
[current handoff](docs/handoff/CLAUDE_HANDOFF.md), [code map](docs/handoff/CODE_MAP.md),
[Final runbook](runs/final-runbook.md), [forum brief](docs/forum-brief-20261017.md).

## Challenge rules (organizer bounds, forum brief of 17 October)

**MAST path: forbidden**
- MAST or MAST-U equilibrium reconstructions from any source, including FAIR-MAST
  `equilibrium.*`.
- Anything trained, calibrated, validated or selected against those reconstructions.
- Aggregate label statistics, label-derived bases or priors.
- The three MAST demo shots, beyond format inspection.
- `mast_envelope.npz` as an input, feature, mask or prior.
- Probe submissions designed to infer label statistics, and constants fitted to such
  inferences.

**MAST path: allowed**
- Released MAST inputs and machine geometry: coils, wall, passive-structure geometry, grids.
- Public MAST diagnostics outside the release (FAIR-MAST magnetics, per-winding currents).
  The human approved this on 18 October.
  - Re-identifying anonymised test shots needs the human's explicit decision first.
- Label-free test-time adaptation (physics-residual fitting).
- DIII-D data and other machines' equilibria.
- Own synthetic equilibria.

**Tuning on Development feedback:** comparing and selecting *submitted* variants by their
Development scores is allowed, including several settings of one parameter. Disclose it.

**DIII-D:** labels may be used freely. The locked128 held-out shots stay unread (our own
evaluation discipline).

**Phase dates (Codabench phase API, checked 18 Oct):** Development closed on 18 Oct at
00:00 UTC. The Final runs 19 Oct 00:00 UTC → 26 Oct 00:00 UTC. Verify the dates via the API before
planning uploads.

**Final:**
- At most three submissions.
- Each needs a reproduction-package reference in `Code URL`: a public repo pinned to a
  commit SHA, or an archive with SHA256.
- The rebuild ceiling is 24 A100 GPU-hours; CPU work is uncapped but must be declared.
- MAST predictions must be our own work. AI assistance is allowed but must be disclosed.

A MAST rule breach loses Award 2. A false provenance statement can disqualify the team from
all awards.

## Working rules

- Use `.venv/bin/python -B`. Keep modules below about 300 lines and changes surgical.
- **Predeclare numerical experiments:** inputs, gates and resource caps. Report
  preparation, execution, validation and official scoring separately.
- **Regression before replacing a scored path.** A refactor of `research/mast_gs_robust.py`
  must reproduce the stored runs bit for bit; see the code map for the check.
- **Label-free proxies** (Te flux-function mismatch, X-point fraction) screen out broken
  candidates; they do not rank working ones (027 efps lesson).
- **Do not overstate results.** Don't turn sampling estimates or repeated checks into
  performance claims. State unknowns.
- Evidence (`run_data/`, `.journal/`) is git-ignored. Never delete or rerun completed
  receipts. Code removed in the 18 October cleanup is at tag `pre-cleanup-20261018`.

## Operations

- **Cluster:** Galvani (`<cluster>`, account `<account>`).
  - CPU arrays may run in parallel.
  - Leave unrelated jobs under the account untouched.
  - The CPU queue can stall for many hours. Fall back to local runs, with one thread per
    worker (`OMP_NUM_THREADS=1`).
- **Uploads:** the human uploads, from the uni connection.
  - Before each upload: a fresh quota check and confirmation of the human's workflow.
  - The last upload of a Development day must be a full archive, because it becomes the
    selected row.
- **Never** make purchases, send operator messages, create/share/reuse tokens, or change
  credentials, network, browser or admin settings.
- **Public repo:** make it public only with the human's explicit confirmation, at Final
  submission.
- The heartbeat stays paused unless the human asks for it.
