# MAST 021 — solenoid current convention explains official R²ψ 0.7646

13 October 2026. Follows from `mast-020-d3d-proxy.md`. No MAST labels, demo files or
envelope were used.

## Physical motivation

- **Shipped geometry:** the MAST solenoid is 656 conductor elements in 4 layers × 164,
  at R = 0.119–0.157 m, each 11.5 × 17 mm. All share `magnetics_sol_current` (kA).
- **Current model:** the Green model (`participant/mast_green.py`) puts the full feed
  current in every element.
- **MAST-U precedent:** the MAST-U solenoid is documented as 324 turns of a 12 × 18 mm
  conductor, "wound two-in-hand in four layers". Two-in-hand means two parallel strands
  per turn, so each element carries half the feed current.
- **Literature flux swing:** documented values (0.9 V·s at ±40 kA originally; about 1.23 V·s
  after the 2003 replacement) fall between the 1× and 0.5× predictions. The literature
  alone does not decide.

## Test using existing official feedback

`.venv/bin/python -B -m research.mast_solenoid_hypothesis` (12 s; output in
`run_data/mast-021/solenoid-hypothesis.json`) works as follows:

- It takes the exact submitted 976708 flux.
- Under hypothesis H_k (each element at k × the feed current), the truth differs from our
  flux by (k − 1)·ψ_sol.
- That predicts the official pooled R² and per-shot R² before any other model error.

| k | Predicted pooled R² | Official worst-20 in simulated worst-20 | Simulated R² of official worst-20 (median) |
|---:|---:|---:|---:|
| 0.5 | **0.769** | 9 | **0.34** |
| 0.6 | 0.857 | 9 | 0.59 |
| 0.75 | 0.948 | 9 | 0.86 |
| 1.25 | 0.961 | 10 | 0.90 |
| 1.5 | 0.867 | 7 | 0.67 |
| 2.0 | 0.614 | 6 | 0.08 |

What the official report shows:
- Pooled R² = 0.7646.
- Worst-20 per-shot R² ranges 0.17–0.36, median ≈ 0.32.
- A random draw would put about 0.3 of the official worst 20 into the simulated worst 20.

Only k ≈ 0.5 matches both the pooled value and the worst-shot level. Other model errors
can only lower R². The small residual (0.769 vs 0.7646) implies that the remaining
flux error is about 0.4–2% of the variance.

- **Selection:** this is a single discrete physical hypothesis checked against disclosed
  feedback. It is not a fitted continuous parameter.
- **Solenoid share:** the solenoid holds 32% of our predicted MAST flux variance.

## Candidate 004

Built by `research/mast_candidate_004.py`:
- **MAST flux:** 002 flux − 0.5 ψ_sol.
- **Unchanged:** q95/betaN and the DIII-D file are byte-identical to 976708.
- **Expected change:** MAST R²ψ from 0.76 to about 0.98–0.995. That is about +0.12
  MAST S, if the hypothesis holds.
- **Unknown effect:** the change to D_LCFS and consistency is not estimated, because the
  solenoid's stray-field gradient near the plasma is small but nonzero.

## Official confirmation: 985132 (004 MAST-only), 13 Oct

- MAST S rose from 0.4923 to **0.5849**.
- R²ψ rose from 0.7646 to **0.9283**; the per-shot mean rose from 0.750 to 0.922.
- D_LCFS improved from 0.282 to 0.256.
- q95, βN and consistency are unchanged (all seven consistency terms are still negative;
  li is −3.41 and κ −1.85).

The solenoid convention is confirmed. The remaining flux error (about 7% of variance) is
larger than the residual budget estimated above, because that estimate assumed the other
errors were uncorrelated with the solenoid term.

A two-point quadratic fit in the solenoid scale k gives an optimum of k ≈ 0.57, worth only
about +0.005 R². It is not pursued, since it would be leaderboard fitting. The remaining
error comes from the plasma model (004's Gaussian has l_i ≈ 1.65, too peaked) and from
unmodelled vessel currents.
