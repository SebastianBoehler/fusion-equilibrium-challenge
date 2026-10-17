# Repository instructions for Claude

For campaign continuation, first read `docs/handoff/CLAUDE_HANDOFF.md`.
For implementation or cleanup, also read `docs/handoff/CODE_MAP.md` and
`docs/handoff/CLEANUP.md`. Current receipts supersede historical notes.

## Working rules

- Continue this local checkout; substantial current code is untracked and evidence is ignored.
- Preserve unrelated edits, immutable plans, source freezes, failed results and accepted jobs.
- Use the existing `.venv/bin/python -B`; verify the experiment's exact runtime and module origins.
- Keep changes surgical and modules below 300 lines where practical.
- Predeclare numerical experiments: input order, identities, success/failure gates and finite resource caps.
- Verify the smallest meaningful behavior. Read `tests/README.md` before running tests.
- Check source pins before editing or moving existing experiment code. Use a separate frozen revision when needed.
- Separate preparation, execution, numerical validation, official scoring and promotion in reports.
- A completed run is not permission to rerun its immutable launcher or resume its finished client.

## Hard campaign boundaries

- Keep locked128 and MAST equilibrium labels, demos and envelopes unread.
- MAST uses allowlisted measured currents, Thomson data and permitted geometry only.
- TRAIN equilibrium labels, contours and derived quantities are loss/diagnostic targets, never predictor inputs or boundary priors.
- Public test data is inference/support validation only; no fitting, calibration or submission tuning.
- Official-feedback use remains disclosed. Local scorer 3.2.0 / official 3.3.1 equivalence is unverified.
- Preserve the incumbent unless the full promotion controls in the handoff pass.
- Monitor accepted DIII-D job **2961003** before any new Fusion allocation; never duplicate or requeue it.
- Keep unrelated cluster jobs untouched. Use Galvani serially, at most one Fusion A100 at a time.
- Production budgets may be two hours or longer when separately justified; frozen diagnostic caps remain unchanged.
- Keep the existing heartbeat **paused** unless the human explicitly requests resumption.
- No purchases, operator messages, token creation/sharing/reuse, or credential/network/browser/admin changes.
- A new upload requires a fresh acceptance/quota check and confirmation of the human's intended upload workflow.

The human expects evidence-backed progress, concise updates, periodic reassessment,
and explicit unknowns. Do not convert repeatedly passed checks or sampling estimates
into claimed performance improvements.
