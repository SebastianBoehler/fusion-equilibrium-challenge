# Search-space review — 17 October 2026

## 1. What the 13 official MAST results show (weighted points per channel)

| Candidate | S | 0.55·R²ψ | 0.15·qb | 0.10·(1−D) | 0.20·cons |
|---|---:|---:|---:|---:|---:|
| 002 Green + Gaussian, solenoid ×1 | 0.492 | 0.4205 | 0 | 0.0718 | 0 |
| 004 + solenoid ×0.5 | 0.585 | 0.5106 | 0 | 0.0744 | 0 |
| 012 004 + uniform vessel | 0.563 | 0.4915 | 0 | 0.0714 | 0 |
| 005 GS Picard (runaway) | 0.381 | 0.3146 | 0 | 0.0662 | 0 |
| 014q GS, inner wall, Z = 0 | 0.661 | 0.5388 | 0 | 0.0806 | 0.0418 |
| 015b GS, linear basis, flux βN | 0.702 | 0.5378 | 0.0481 | 0.0802 | 0.0360 |
| 016 GS Te¹, smoothed force Z | 0.714 | 0.5378 | 0.0504 | 0.0808 | 0.0448 |
| 020 config-Z, linear basis | 0.727 | 0.5389 | 0.0504 | 0.0838 | 0.0536 |
| 019 config-Z, Te¹ | 0.749 | 0.5399 | 0.0644 | 0.0841 | 0.0602 |
| 022 + smoothed anchor + edge B_Z | 0.709 | 0.5088 | 0.0667 | 0.0826 | 0.0508 |
| 021b + smoothed anchor | 0.756 | 0.5403 | 0.0653 | 0.0842 | 0.0664 |
| **026 + temporal smoothing** | **0.762** | 0.5405 | 0.0677 | 0.0843 | 0.0692 |
| Leader (yszhang, for scale) | 0.951 | 0.5473 | 0.121 | 0.098 | 0.185 |

## 2. Patterns

1. **Flux R² has plateaued since the first self-consistent solve.** Seven variants sit
   within R²ψ 0.978–0.983. Only *external-field* changes ever moved it: the solenoid fix
   (+0.16), the uniform vessel (−0.035) and the uniform B_Z correction (−0.057). Ensembling
   our solvers would gain +0.0004, because their errors are common-mode. **The flux ceiling
   is the coil/passive model, not the plasma model.** efps (027/040) is the first new
   external-field candidate since the solenoid.
2. **Consistency rises with topology and stability fixes:**
   - Inner wall: X-point boundaries 31% → 76%.
   - Vertical classification: Z_axis −0.73 → +0.30, tri_bot −0.40 → +0.29.
   - Temporal smoothing.
   - Remaining weak terms: l_i ≈ 0.0 always; κ 0.25; tri 0.28–0.35; Z_axis 0.31;
     R_axis and volume about 0.61.
3. **q95 follows the boundary definition:** −0.54 (Gaussian) → −0.17 (wall-limited) →
   0.55 (X-point) → 0.585 (temporally smoothed).
4. **βN is stuck at 0.31–0.32** for every flux-derived estimator. The Thomson-pressure
   estimator gives 0.12. This is the weakest scalar channel (leaders about 0.8 on qb).
5. **What failed, and why:**
   - Runaway equilibria: instability.
   - Crude passive models: a wrong field shape.
   - Global B_Z: corrupts the far field.
   - Linear EFIT basis: q95 and R_axis worse.
   - An MAST-statistics-calibrated profile: dropped for compliance.

## 3. Allowed solution space (forum, 23 Sep / Oct) vs coverage

| Allowed family | Coverage | Gap / opportunity |
|---|---|---|
| Physics forward model from released inputs + machine geometry | **Deep** | Coil-circuit completeness: efps (running), passive structures as *segmented* circuits from geometry, Newton–Krylov static solve |
| Label-free test-time adaptation (physics residual fitting) on MAST inputs | Partial (Te anchor, edge) | **Per-frame inverse fit**: few parameters (profile pair, Z, one external correction) fitted to Thomson consistency (Te isotherms, edge) without global flux penalty |
| DIII-D labels (any use; MAST path must stay label-free) | Calibration only (Te^p, proxy) | **Dimensionless regressors trained on DIII-D** for βN (and βp, l_i) from Thomson pressure, Ip, B_T and geometry from our own equilibria. βN is our weakest channel. Validate on DIII-D held-out shots |
| Own synthetic equilibria with MAST geometry | Not used | An emulator only re-learns our solver; low value now |
| Public MAST diagnostics outside the release | Not used | Test shots carry no shot ID, so per-shot joins would need deliberate re-identification (not done). Non-test public shots' *input* diagnostics (P2 per-winding, efps) could settle the efps convention |
| Self-distillation / temporal models | Smoothing only | Shot-level regularization of solver parameters (β0, anchor, Z) |
| Variant selection with disclosure | Used (channel combination) | — |
| Ensembles | Flux: no gain (correlated) | Scalar ensembles (q95, βN) across estimators: untested |

## 4. Priorities for the remaining Development slots and the Final

1. **efps (027/040).** External-field completeness is the only lever that has moved flux R².
2. **βN learned on DIII-D** (dimensionless), applied to our MAST equilibria: the weakest
   scalar channel, so the largest qb upside.
3. **Per-frame inverse fit** on Thomson consistency (no global field term): R_axis,
   volume, κ, boundary.
4. **Segmented passive circuits** from geometry: a second external-field candidate.
