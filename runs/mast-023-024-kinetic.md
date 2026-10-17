# MAST 023/024 — kinetic equilibria on all MAST frames; q95/βN from the flux

13 October 2026.

## 023: production equilibria (`research/mast_gs.py`)

**Inputs per frame**, all from input columns only:
- External flux from the measured PF coils, plus the solenoid at 0.5 × the feed current
  per element (mast-021).
- Measured Ip.
- Thomson Te points, with J ∝ Te^1.0 on ψ_N.
- Axis radius from a parabola through the smoothed Te peak.
- Vertical position from external force balance.

**Allowed region:** the grid minus a 2-cell margin, minus R < 0.19 m (centre column), minus
cells within 6 cm of any conductor. No envelope. Frames whose solve fails keep the 004 flux.

**Pilot (25 shots / 1,412 frames, 94 s):**
- 4 frames failed.
- Median axis R is 0.94 m; median grid elongation is about 1.65.
- The outboard midplane LCFS edge (≈1.33–1.39 m) agrees with where the Thomson Te
  signal ends. The 004 Gaussian puts it at 1.49–1.55 m. This is a label-free check.

The full run covers 1,206 shots with resumable per-shot outputs in
`run_data/mast-023/shots/`.

## 024: q95 and βN (`research/q95_proxy.py`, `research/betan.py`)

DIII-D proxy, 402 frames, kinetic GS flux with the exact external field:

| Estimator | R² | corr | median ratio |
|---|---:|---:|---:|
| q95: field-line integral at ψ_N 0.95, vacuum F (TF 144 turns) | 0.858 | 0.939 | 0.984 |
| βN, Thomson electron pressure only | −0.684 | 0.493 | 0.463 |
| βN from solver pressure share (β0) | 0.928 | 0.976 | 0.918 |
| βN from flux only (per-bin J = R p′ + FF′/μ0R) | 0.969 | 0.984 | 0.990 |

**MAST spot check, without labels:**
- New q95 is about 4–5.5 through the flat-top. Official 002 has early-frame outliers
  (23.8, 7.06) that likely hurt its pooled R².
- Flux-only βN on MAST is noisy, with negative values and jumps. The coil/vessel error
  probably enters the pressure split.
- Because the q95/βN term averages the two R² values before clipping, candidate 005 keeps
  002's βN (official R² 0.119) and replaces only q95.

## Full run and candidate 005

**023 full run:**
- 1,206 shots / 69,075 frames in 61 min on 12 workers.
- 52 frames (0.08%) failed and keep the 004 flux.
- Median axis R is 0.909 m; median grid elongation is 1.66.
- Median per-shot R² between the 005 and 004 flux is 0.74.

**Label-free physics check** on pilot frames, using the scorer's own `internal_inductance`
and contour shape on our own LCFS (no envelope):

| Flux | l_i p10/50/90 | κ p10/50/90 |
|---|---|---|
| 005 kinetic GS | 0.66 / 0.95 / 1.11 | 1.44 / 1.66 / 1.88 |
| 004 Gaussian | 1.34 / 1.65 / 1.94 | 1.29 / 1.40 / 1.60 |

Typical MAST values are l_i ≈ 0.9–1.3 and κ ≈ 1.7–2.0. The 004 Gaussian is over-peaked
and too round, so 005 should improve consistency and D_LCFS. 005 and 004 disagree on flux
(R² ≈ 0.74), so at least one of them has a materially lower R²ψ. Only official scoring can
decide which.

**Candidate 005** (`research/mast_candidate_005.py`):
- **Flux:** the 023 flux.
- **q95:** new values. 6,261 frames (9%) take the nearest valid frame of the same shot.
  Distribution p5/50/95 = 3.08 / 5.07 / 8.59.
- **βN and DIII-D:** unchanged from 002 / 973093.
- **Full archive:** 4,032,339,592 bytes, SHA256 `b54901d3…2fec5`; the DIII-D member is
  byte-identical.
- **MAST-only float16 archive:** 508,319,958 bytes, SHA256 `73b053fa…6f99`.

## Official results (13 Oct, scorer 3.3.1; MAST-only uploads, so G shows 0)

Saved in `run_data/official-985081-985085/results.json`.

| Submission | MAST S | R²ψ | q95 | βN | D_LCFS | Consistency | LCFS fail |
|---|---:|---:|---:|---:|---:|---:|---:|
| 976708 (002, previous best) | 0.4923 | 0.7646 | −0.540 | 0.119 | 0.282 | 0 | 0 |
| 985081 (005) | 0.3808 | 0.5720 | −0.438 | 0.119 | 0.338 | 0 | 0.038 |
| 985085 (006 = 005 + flux βN) | 0.3808 | 0.5720 | −0.438 | 0.124 | 0.338 | 0 | 0.038 |

**005 consistency R²:**

| R_axis | Z_axis | κ | tri_top | tri_bot | volume | li |
|---:|---:|---:|---:|---:|---:|---:|
| −29.9 | −3.15 | −0.83 | −0.88 | −3.50 | −2.44 | −0.16 |

The worst shots score R²ψ between −4.7 and −1.9.

**Diagnosis:**
- **Solver runaway.** On the worst shots the solver parked the axis at R ≈ 1.88–1.91 m,
  at the outer wall. A per-frame check finds 11,680 of 69,075 frames (17%) with the axis
  outside R 0.7–1.15 m or |Z| ≥ 0.3 m, or without a closed LCFS.
- **Not proven by the proxy.** The kinetic solver's DIII-D proxy success (exact external
  field, oracle axis) did not carry over to MAST. The likely causes are the coil-only
  external field (no vessel currents), the Te-peak axis anchor, and the allowed region.
- **βN:** the flux-derived βN is about equal to 002's (0.124 vs 0.119); no gain.
- **Pending:** whether the half-current solenoid itself helps is untested until 004 is scored.

**Variant 007** (`research/mast_variant_007.py`): 005 with per-frame guards falling back to
004 flux. 506,779,180 bytes, SHA256 `e8d77fec…c713`. Not uploaded.
