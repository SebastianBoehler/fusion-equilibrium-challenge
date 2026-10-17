# MAST official results — 14 October 2026 (scorer 3.3.1)

Source: `run_data/official-20261014/` (unmodified scores.json, reports and scoring zips).

| Candidate | ID | MAST S | R²ψ | q95 | βN | D_LCFS | Consistency |
|---|---:|---:|---:|---:|---:|---:|---:|
| 016 (Te¹, inner wall, smoothed Z, inner-wall q95) | 986730 | **0.7139** | 0.9779 | **0.5528** | 0.1187 | **0.1923** | **0.2244** |
| 015b (linear basis, inner, Z = 0, flux βN) | 986738 | 0.7022 | 0.9779 | 0.3226 | **0.3178** | 0.1981 | 0.1804 |
| 014q (Te¹, inner, Z = 0, old-wall q95) | 986734 | 0.6612 | **0.9796** | −0.1704 | 0.1187 | 0.1942 | 0.2094 |
| 004 (Gaussian, solenoid fix) | 985199 | 0.5849 | 0.9283 | −0.5397 | 0.1187 | 0.2567 | 0 |
| 012 (004 + 20 mm vessel) | 986742 | 0.5629 | 0.8936 | −0.5397 | 0.1187 | 0.2860 | 0 |

**Consistency R²:**

| Candidate | R_axis | Z_axis | κ | tri_top | tri_bot | volume | li |
|---|---:|---:|---:|---:|---:|---:|---:|
| 016 | 0.477 | −0.734 | 0.062 | 0.443 | −0.402 | 0.510 | 0.079 |
| 014q | 0.470 | −0.084 | 0.192 | −0.096 | 0.302 | 0.502 | −0.015 |
| 015b | 0.414 | −0.084 | 0.176 | −0.124 | 0.302 | 0.371 | −0.038 |

## What we learned

1. **The kinetic free-boundary solver works on MAST.**
   - R²ψ rose from 0.928 to 0.978–0.980.
   - D_LCFS improved from 0.257 to 0.19.
   - Consistency went from 0 to 0.18–0.22.
   - The label-free checks (Te consistency, edge bias, κ/l_i plausibility) predicted the
     direction correctly.
2. **Inner-wall q95** (X-point-defined ψ_N) beats old-wall q95: 0.553 (016) or 0.323 (015b)
   versus −0.170 (014q). The median shift to about 6.5–7 was a correction, not a bias.
3. **Flux-derived βN** (linear-basis pressure split) reaches 0.318 against 0.119 for the
   Thomson-based 002 estimate.
4. **Vessel-current model, as implemented, hurts:** R²ψ 0.894 vs 0.928. Drop it (or revisit
   with segmented passive structures).
5. **Smoothed vertical position (016) vs Z = 0 (014q).**
   - R²ψ: −0.0017.
   - Z_axis consistency: −0.73 vs −0.08; both clip to 0, so it costs nothing.
   - tri_top: +0.54. tri_bot: −0.70. κ: −0.13.
   - Net flux terms: 0.6635 vs 0.6612, slightly in 016's favour.
   - Up-down asymmetry is still poorly predicted.
6. **Linear basis vs Te¹** at the same wall and Z (015b vs 014q): R²ψ −0.0016, consistency
   −0.03. Te¹ is slightly better for flux; the linear basis gives the better βN.

## Combination 017 (the last slot on 14 Oct)

The scorer's channels are independent: R²ψ, D_LCFS and consistency depend only on ψ; the
q95 and βN R² values depend only on their own arrays. So 016's ψ and q95 plus 015b's βN
give a predicted MAST S of 0.6635 + 0.15 · mean(0.5528, 0.3178) = **0.7288**, and G ≈
0.7324 with DIII-D 0.995079.

Full archive `run_data/submission-mast-017/fusion-mast-combo-017.zip`: 4,031,460,867 bytes,
SHA `1ee2ef60…6560`; the DIII-D member is byte-identical.

## Next levers (by points available, against the leader at consistency 0.87, qb 0.80, D_LCFS 0.05)

- **Consistency (0.20 weight, now 0.22).** Z_axis and tri_bot are negative or weak, κ and
  l_i are weak, and up-down asymmetry is the common thread. Ideas: a better vertical
  model (P6 anti-series response), an LSN/DND classifier from coil asymmetry, X-point
  location.
- **q95/βN (0.15 weight, now ~0.44).** βN from the Te¹ flux of 016, combined estimators.
- **D_LCFS (0.10 weight, now 0.19).**
- **R²ψ (0.55 weight, 0.978).** The remaining 2.2% may come from vessel currents
  (segmented model), plasma profile and vertical position.

## 017 official (full archive, the last slot on 14 Oct)

- MAST S **0.728555** (predicted 0.7288; the difference is float32 vs float16 flux).
- G **0.732158**; DIII-D S 0.995079.
- Ranks over all 132 rows: **G 23/132** (was 40); DIII-D 23/132.
- To reach G #10 needs MAST S ≈ 0.81; #5 needs about 0.90.

## Profiling the bottleneck (14 Oct afternoon)

**The worst official shots** (0678, 0157, 0972, 0438, 0497, 1070, 0422, 0537: R²ψ
0.57–0.70, D_LCFS 0.65–0.97) are all frames where 016's smoothed force balance put the
axis **up** at Z ≈ +0.31 to +0.36 m. In those frames the plasma shrank (R_in 0.33–0.50,
LCFS Z range [−0.3, +1.25]).

**What the official statistics imply about the true Z_axis.** For the constant Z = 0
prediction (014q), Z_axis R² = −0.08, so mean²/var of the true Z_axis ≈ 0.08. This is
consistent with mostly-centred plasmas plus a minority shifted down; for example, about 7%
of frames at −0.3 m.

**Coil asymmetries** (one shot in three, shot-median values; `scratchpad/zasym.py`):
- P6U (anti-series radial-field coil) correlates −0.58 with our predicted Z. P6 sets the
  direction force balance chooses.
- P2 asymmetry (U−L)/(|U|+|L|):
  - Below −0.02 (lower divertor coil stronger) in 21% of shots. That group holds most
    shifts: 016 puts 31% of it down and 7% up.
  - Outside the group: 1.9% up, 3.2% down.
  - The up-shifted and down-shifted shots share the same P2 signature (−0.05/−0.06 against
    −0.002 for centred shots), and the same P3 signature (0.35–0.41 vs 0.13).
- **Interpretation:** these are lower-single-null shots. The literature says LSN was common
  and USN rare. Force balance gets the sign wrong for about a quarter of them, which
  produces the worst shots.

**Run 032, vertical mode "config"** (`mast_gs_robust --vertical config`):
- **LSN class:** median P2 asymmetry < −0.02. Smoothed force-balance magnitude, sign forced
  downward, |Z| ≤ 0.4 m.
- **All other shots:** Z = 0.
- Builds candidate 018 (q95 from its own flux; βN to be taken from 015b in the final
  combination).

## Expert review — physics-ML transfer (14 Oct)

**Metric facts** (challenge paper, Eqs. 2–6): D_LCFS is the symmetric Hausdorff distance
divided by the time-averaged true LCFS major radius; failed derivations are mean-substituted.

**Gap decomposition against the leader:**

| Component | Points |
|---|---:|
| Consistency | 0.130 (62%) |
| q95/βN | 0.054 |
| D_LCFS | 0.014 |
| R²ψ | 0.010 |

**Explaining the leaders.** The reviewer's best explanation for a leader with consistency
0.87 and R²ψ 0.996 is supervised training on MAST labels (FAIR-MAST via Rule 2, or the
demo files).
- Rule 2 permits external public datasets with disclosure. The organizers may disqualify
  ground-truth leakage or test-set memorization.
- Test and private-fold shots are very likely in FAIR-MAST, and the private shot IDs are
  unknown.
- Our campaign boundary forbids MAST labels. **Not pursued**; flagged to the human.

**Realistic label-free target:** MAST S ≈ 0.76–0.79.

**Ranked label-free plan:**
1. Shot-level temporal regularization of solver parameters.
2. Topology and extraction audit (up/down, X-point, LCFS leaks); started with 032.
3. DIII-D-trained dimensionless profile-parameter regressors (βp, li, β_tot/β_e) → βN, li.
4. Per-frame inverse solve with a Te-isotherm and Thomson-edge objective.
5. Segmented passive currents, low priority.

Each change goes through the DIII-D proxy gate first.

## Expert review — MAST operations (14 Oct)

- **CDN was the default.** |Z_mag| ≤ 16 mm; κ ≈ 1.93, δ ≈ 0.37 (Meyer, EPS 2002 P1.056).
- **LSN was made in two ways.**
  - 2003–04: unbalanced P2 currents (Meyer, EPS 2004 P4.191). δ_lower ≈ 0.46,
    δ_upper ≈ 0.25–0.32, Ip < 0.7 MA.
  - 2009–13: a downward shift, because the divertor coils are symmetric
    (Kirk, arXiv:1305.3061). LSN example: Ip 600 kA, Bt 0.55 T, q95 2.8, βN 3.4–4.0.
- **Shift size.** Down by about 15–30 cm. USN plasmas existed, with Z_mag up to about
  +10 cm in a published scan (Tournianski, EPS 2006).
- **P6 is anti-series.** It provides the radial field for vertical control, so P6
  normalized by Ip is the main vertical-position signal; P4/P5 are series-symmetric.
- **Other numbers.**
  - li(2) ≈ 0.87, typical for H-mode (Cunningham 2013).
  - P2 polarity reversal mid-shot is possible (Pangione, arXiv:1310.8450).
  - Open-loop vertical growth rates 34–403 s⁻¹.
- **Thomson.**
  - The chord sits at Z = +14.5 mm and misses the core in LSN (innermost point ψ_N ≈ 0.15–0.3).
  - A 30 cm shift is detectable only weakly, through peak-Te reduction.
- **Implication.** Model Z_axis as two populations, centred and shifted down, gated by a
  classifier; tri_bot depends on the configuration.

**Test design.** Run 032/018 uses the P2 classifier plus a forced downward shift. A
P6-based classifier is the alternative if 018 underperforms.

## Expert review — equilibrium physicist (14 Oct)

- **EFIT++ magnetics-only basis** (MAST-U, Pentland 2025): Lao n_p = n_F = 1, i.e.
  p′ = α0 + α1ψ̃ − (α0+α1)ψ̃² = (1−ψ̃)(α0 + (α0+α1)ψ̃), with FF′ of the same form, Dirichlet
  edges and Ip normalization. Original-MAST orders are unknown; Berkery's MAST
  magnetics-only run used p′ ∝ ψ(1−ψ) and a 4th-order FF′ with a finite edge. Our
  "linear" (015) case is the special case α1 = −α0.
- **Passive currents** (Berkery 2021, VALEN for shot 23822): 100–150 kA early, 25–50 kA
  at 0.2–0.3 s, driven by the P1 swing. EFIT uses 18 groups including the P6 cases.
  Geometric resistances can be off by 2×, which fits 012 hurting.
- **Vertical position.** The only measured up-down asymmetry is P6 (anti-series). Rough
  sensitivity: δZ ≈ 7 mm per mT of radial-field error, so force balance amplifies noise at
  the scale of MAST's real Z spread. FreeGSNKE solves with Newton–Krylov (no pin).
  Multiple solution branches exist (diverted vs outboard-limited).
- **Shape.** For near-DN, κ ≈ ΔZ_X/2a, set by the X-point heights and the minor radius. δ
  asymmetry is limited by Z accuracy.
- **Ranked upgrades:**
  1. The EFIT profile family, (1−ψ̃)(1 + cψ̃) per term, with parameters set by Ip, the
     Te-peak R_axis and the Thomson edges.
  2. Thomson-edge isoflux constraints plus a 1–2 parameter external correction
     (uniform B_Z, quadrupole).
  3. Robust Z from the P6 linear response, smoothed and shrunk toward 0 by a predeclared
     factor.
  4. X-point and divertor-plate passive group.
  5. Newton–Krylov, refined grid, spline critical points.
