# MAST 022 — self-consistent free-boundary plasma on the DIII-D proxy

13 October 2026. This follows from mast-020: the Gaussian plasma, not the coils, limits
consistency and D_LCFS.

## Method

`research/gs_picard.py` solves φ = φ_ext + K·J by Picard iteration.
- **Current profile:** J = λ[β0·R/R0 + (1−β0)·R0/R]·(1−ψ_N^α_m)^α_n inside the LCFS.
- **LCFS:** found by bisection, as the lowest flux level whose axis-connected region stays
  off the edge of the allowed region.
- **Vertical position:** a rigid shift that zeroes the external vertical force.
- **Radial position:** β0 is adapted until the LCFS midplane centre matches measured
  inboard/outboard edges. On MAST those edges come from the midplane Thomson; this proxy
  uses the true edges.
- **Allowed region:** the grid minus a 2-cell margin and minus R < 1.0 m. The scorer
  envelope is not used.

Harness: `research/gs_proxy_d3d.py`. It uses the exact external flux from mast-020 and the
measured Ip, on 16 development shots at stride 10 (402 frames). q95/βN are set to the
truth, so S is not comparable to submissions.

## Findings so far

- **Shot orientation:** reversed-Ip shots must be solved with |Ip| and the flux flipped back.
- **Warm start is broken:** reusing the previous frame's current makes the flux amplitude
  grow without bound. All runs use cold starts until this is fixed.
- **Radial sensitivity:** with a fixed pressure share, the radial position is very
  sensitive (±20 cm with the exact external field). The edge constraint fixes the centre
  but not the width.

| Profile (α_m, α_n), β0 start 0.8 | S | R²ψ | D_LCFS | Consistency | R_axis | Z_axis | κ | volume | li |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle circular Gaussian (reference) | 0.857 | 0.985 | 0.067 | 0.357 | −0.28 | 0.58 | −6.5 | −2.4 | 0.39 |
| (1, 2) | 0.842 | 0.966 | 0.069 | 0.337 | 0.70 | −1.5 | −5.1 | 0.29 | −9.3 |
| (1, 1) | 0.827 | 0.933 | 0.067 | 0.352 | 0.50 | −18.9 | −0.55 | 0.18 | −0.14 |
| (2, 2) | 0.844 | 0.963 | 0.058 | 0.352 | 0.70 | −9.5 | −1.9 | 0.08 | −0.92 |
| (1, 3) | 0.849 | 0.948 | 0.069 | 0.420 | 0.63 | 0.13 | −6.3 | 0.69 | −35 |

Profile shape trades κ and li against Z_axis and volume.

## Open problems, in order

1. **Vertical position (Z_axis).** The force-zero rigid shift does not settle at the true
   height. Next to try: Newton-type vertical feedback, or symmetric constraints for
   double-null MAST.
2. **Plasma width and κ.** Add the measured edge width as a second constraint, e.g. on the
   current-peaking exponent.
3. **Thomson-derived edges.** Calibrate the density-fraction threshold for the LCFS edge
   on DIII-D's tangential midplane system, then apply it to the MAST midplane Thomson.
4. **Warm-start bug.**

No MAST run or candidate had been built from this solver at this point.

## Resolution (same day)

- **Vertical position:** the force model was right; the true current has zero vertical
  force to within 0.16 cells. Errors came from runaway iterations, because the zero-force
  search covered only ±3 cells and otherwise applied no correction. Fix: search ±12 cells
  and fall back to the minimum-|F| shift. Z_axis correlation went from −0.68 to 0.97.
- **Profile:** the current shape is now measured rather than parametric. Thomson Te
  points are mapped onto the current flux, and J ∝ Te^p on ψ_N (`kinetic_shape`). On
  DIII-D, p = 1.5 (Spitzer) still biased κ by −0.10 and l_i by +0.18. p = 1.0 is unbiased
  (κ −0.02, l_i −0.03). This one dimensionless constant is calibrated on TRAIN and
  transferred.
- **Radial anchor:** the oracle axis radius (`--constraint axis`) beats the edge
  constraint. On MAST, the axis radius comes from the Thomson Te peak.

Official-scorer proxy: 402 frames, exact external field, oracle axis R, real DIII-D
Thomson Te, p = 1.0.

| Variant | S* | R²ψ | D_LCFS | Consistency | R_axis | Z_axis | κ | tri_top | tri_bot | volume | li |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kinetic GS | 0.951 | 0.9966 | 0.024 | 0.777 | 0.93 | 0.96 | 0.55 | 0.97 | 0.97 | 0.74 | 0.33 |
| oracle Gaussian | 0.857 | 0.985 | 0.067 | 0.357 | −0.28 | 0.58 | −6.5 | 0.73 | 0.81 | −2.4 | 0.39 |

\*q95/βN are set to the truth.

The warm-start bug remains, so all runs use cold starts.
