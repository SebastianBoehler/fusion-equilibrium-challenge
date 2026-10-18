# Literature review 2 and data checks — 17 October 2026

A five-persona review: MAST operations, equilibrium reconstruction, kinetic profiles,
ML transfer and strategy. Its notes are in the session scratchpad (`litreview2/`).
The claims below were re-checked on the release **inputs** only (no labels).

## Verified findings

1. **efps is P2's per-turn feed current; the P2 totals already include it.**
   - efps / (P2/20) has median 0.984 (upper) and 0.991 (lower), with p10–p90 0.95–1.01.
   - FAIR-MAST level-2 mapping: P2 has 12 inner + 8 outer turns.
   - MAST-U documentation: the former P2 power supply now feeds Pc.
   - **Candidate 027 counted P2 twice**, which explains the over-shaping:
     - MAST S 0.703 vs 0.762;
     - q95 R² 0.113 vs 0.585;
     - tri_bot −0.10 vs 0.28.
   - efps is now a consistency check only.
2. **P2 up/down asymmetry is calibration noise.**
   - In 935 shots with large positive P2, the asymmetry spans −0.019 to +0.010 (p1–p99).
   - The 223-shot "LSN" class came from weak or negative P2 shots.
3. **The P6 radial-field coil carries the up-down signal.**
   - Measure: m6 = median((P6U − P6L)/2 / |Ip|) × 10³.
   - Noise band: about −2.4 to +2.7 (p5 −2.43).
   - 157 shots have |m6| > 4 (141 positive, 16 negative). The old class held only 111 of
     them, plus 112 weak-P6 shots.
   - Literature: single-null MAST plasmas were shifted downward (Harrison et al. 2014).
   - Unknown: whether the P6 sign gives the direction.
4. **Our label-free proxies are biased.**
   - X-point fraction rewards strong divertor fields.
   - Midplane Te mismatch cannot see κ, triangularity or Z.
   - New gate: an input-accounting audit of every added current against the totals and the
     turn table.

## Physics leads not yet tested

- **Unrelaxed current profile:**
  - MAST frames span 185–405 ms.
  - Spitzer τ_R is 3–9 s (a = 0.5 m, Z_eff = 2).
  - 42% of frames come before the Ip maximum.
  - So J ∝ Te^p cannot follow the ramp history. This is consistent with l_i R² ≈ 0.
  - Magnetics-only MAST EFIT allows a finite edge current (Berkery 2021); our Te¹ profile
    is zero at the edge.
- **Passive currents:** these sit mainly in the centre-column case and are driven by the
  solenoid swing (Berkery 2021). A uniform B_Z correction is the wrong pattern.
- **βN:** fast ions carry 10–50% of the pressure and beam power is not released, so a
  Thomson-only βN has an unknown scale. Through the Shafranov coupling, any l_i bias also
  leaks into βp.

## Actions

| # | Action | Status |
|---|---|---|
| 1 | Drop efps from the field model | Done (027 not promoted; 040 cancelled) |
| 2 | Re-decide the shifted class by \|m6\| > 4 (`--vertical config-p6`), threshold predeclared from the input noise band | Run 042/043 → candidate 029 |
| 3 | Edge-current / bootstrap profile component chosen per frame by a physics residual; DIII-D gate first | Next |
| 4 | Centre-column passive circuit from geometry and the measured dI/dt | Later |
| 5 | Synthetic-truth validation harness from our own solver | Later |

**Disclosure:** six of the eight worst official shots have strong P6. That feedback is
consistent with the class, but it did not set the threshold.

## Result: current-diffusion l_i check on DIII-D (predeclared gate failed)

`research/current_diffusion.py` simulates a 1D cylindrical Spitzer + neoclassical diffusion
from a flat start at 5% of Ip, driven by measured Ip(t) and Thomson Te. It ran on the 16
local DIII-D TRAIN shots (3,941 frames; EFIT l_i used only as the target).

| Profile | Pooled r | Median within-shot r | l_i p10/50/90 |
|---|---:|---:|---|
| EFIT (truth) | — | — | 0.82 / 1.00 / 1.25 |
| Diffusion | 0.00 | 0.46 | 0.50 / 0.50 / 1.42 |
| Relaxed Te¹ (current model) | 0.30 | 0.29 | 0.76 / 0.87 / 1.00 |

**Gate: fail.** The within-shot gain is +0.18 against the required +0.20, and the pooled r
is lower.

**Reading:** classical penetration is far too slow. l_i stays at the flat-profile value 0.5,
while real current penetrates faster (MHD and sawtooth redistribution). Within-shot l_i
variation is 77% of the total on DIII-D, so time evolution matters, but this model does not
capture its level. Not pursued without a new hypothesis; the initial condition is not tuned.

## Result: candidate 029 (P6-magnitude class), official 995442

| Candidate | MAST S | R²ψ | D_LCFS | Consistency | Z_axis | tri_top |
|---|---:|---:|---:|---:|---:|---:|
| 029 | 0.7576 | 0.98241 | 0.1612 | 0.328 | 0.255 | 0.296 |
| 026 | 0.7618 | 0.98272 | 0.1567 | 0.346 | 0.314 | 0.346 |

**Per-shot official feedback (disclosed):**
- 15 of the 20 worst shots are weak-P6 shots that 029 returned to Z = 0, most with
  D_LCFS ≈ 0.7. **The P2-ratio class does capture real downward shifts.** Finding 2 above
  ("calibration noise") is wrong for these shots.
- The 46 newly added strong-P6 shots improved: 0678 now has R²ψ 0.87 and D_LCFS 0.11;
  1055 has D_LCFS 0.17.

**Next variant, 030:** the union of the two rules. The shifted class is the P2-ratio class
**or** |m6| > 4. It equals 026 except for those 46 shots (`run_data/mast-044/sources.json`).
