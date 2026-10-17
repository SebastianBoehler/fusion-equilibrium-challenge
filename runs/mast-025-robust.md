# MAST 025 — robust kinetic equilibria (after official 985081)

13 October 2026.

## Diagnosis of the 005 runaway

From frames of the worst official shots (93, 175, 24, 652, 180):
- **No radial anchor.** When the Te-peak estimate failed (`None`), the plasma drifted to
  the allowed-region wall at R = 1.91 m. The coils alone do not hold it at the default
  pressure share.
- **Unreliable Te-peak estimate.** It sometimes landed at 1.40–1.45 m or 0.25 m. β0 then
  pinned at −1 chasing an impossible target.
- **Vertical drift.** Occasional (e.g. Z = 0.94 m).

## Changes (`research/mast_gs_robust.py`)

- **Axis anchor:** Te median-filtered over 9 channels; the centroid of Te above 70% of
  its peak; accepted only within 0.75–1.10 m. Otherwise the shot median of valid anchors,
  otherwise 0.92 m (MAST machine geometry).
- **Allowed region:** additionally R ≤ 1.6 m and |Z| ≤ 1.7 m.
- **β0 range:** [−0.5, 2.0].
- **Per-frame plausibility:** axis R 0.7–1.15 m, |Z| < 0.3 m, and a closed LCFS.
  Failing frames take the 004 flux.

## Pilot

35 shots (all ten official-worst shots plus every 50th), 1,806 frames:
- **98.1% of frames are plausible**, versus 83% for 023.
- **Agreement with 004** (official R²ψ 0.928): pooled R² 0.928; per-shot p10/50/90 =
  0.86 / 0.92 / 0.95; the worst shot is 0.80. No catastrophic shots.
- **Plasma shape** (scorer `internal_inductance` and contour shape on our own LCFS):
  l_i p10/50/90 = 0.79 / 0.95 / 1.19 and κ = 1.50 / 1.67 / 1.85. For 004: l_i 1.20 / 1.53 /
  1.84 and κ 1.25 / 1.37 / 1.59. Typical MAST is l_i ≈ 0.9–1.3 and κ ≈ 1.7–2.0.

**Te flux-function mismatch** (`research/flux_function_check.py`): the normalized RMS
difference between inboard Te(ψ_N) and outboard Te(ψ_N) on 579 pilot frames.

| Flux map | Median mismatch | Mean mismatch | Official R²ψ |
|---|---:|---:|---:|
| 002 | 0.261 | 0.254 | 0.765 |
| 004 | 0.226 | 0.220 | 0.928 |
| 025 | **0.069** | **0.089** | pending |

025 beats 004 on 88.8% of frames. The metric ranks 002 below 004, matching the official
order. Caveat: 025 uses Te for both its current profile and its axis anchor, so the test
is not independent of the method.

Full run: 1,206 shots, written to `run_data/mast-025/shots/`.

## Full run and candidates (13 Oct evening)

**Full run:** 1,206 shots / 69,075 frames in 58 min. 66,211 frames (95.9%) keep the
robust-solver flux; the rest take the 004 flux.

| Candidate | Contents | Bytes | SHA256 |
|---|---|---:|---|
| **008 full** | 025 flux; field-line q95 on the robust region (131 frames nearest-filled; p5/50/95 = 3.56 / 5.26 / 8.41); βN from 002; DIII-D byte-identical to 973093 | 4,031,595,073 | `ad91b39e…9eae` |
| 008 MAST-only | float16 copy of 008's MAST arrays | 506,116,760 | `b8d56b70…c61b` |
| **009 MAST-only** | 008 plus smoothed flux βN (2 shots keep 002; p5/50/95 = 0.15 / 1.35 / 3.13) | 506,026,351 | `71ee9687…1cf4` |

**Estimator smoothness on 10 pilot shots** (median relative jump between consecutive frames):

| Flux | q95 | βN | βN out of bounds |
|---|---:|---:|---:|
| 023 | 4.4% | 10.2% | 15% of frames |
| 025 | 2.5% | 6.8% | 4% of frames |

**Upload plan for 14 Oct** (5 fresh slots):
1. 008 full.
2. 009 MAST-only.
3. Hold the remaining slots for follow-ups based on 008's per-scalar report.

The 007 archives are superseded by 008.

## 026: symmetric vertical variant → candidate 010 (13 Oct, 21:52)

Same as 025, but `--vertical zero` holds the current centroid at Z = 0 every iteration
instead of using the external-force balance.

**Motivation:** with 025, the predicted Z_axis (scorer `magnetic_axis` with our geometry
mask) has a 7.8 cm spread, and a median within-shot std of 4.9 cm. 004 has Z ≈ 0. The
consistency term clips each scalar at 0 separately, so a bad Z_axis cannot drag the term
below 004's. Vertical jitter can still cost R²ψ.

**Pilot (35 shots):**
- 99.7% of frames plausible (025: 98.1%).
- Te mismatch median 0.0675 (025: 0.0694; 004: 0.226).
- R² vs 004: 0.931 (025: 0.925).

**Full run:** 1,206 shots / 69,075 frames in 51 min. **99.87%** of frames keep the solver
flux; 88 take the 004 flux.

**Candidate 010:** 026 flux, field-line q95 (73 frames nearest-filled; p5/50/95 = 4.11 /
5.40 / 8.61), βN from 002, DIII-D byte-identical.

| Archive | Bytes | SHA256 |
|---|---:|---|
| full | 4,031,175,519 | `e31fa469…f0ec2` |
| MAST-only | 495,096,050 | `a6784459…78dd` |

## Upload design for 14 Oct

1. 010 MAST-only.
2. 008 MAST-only.
3. 009 MAST-only.

Read each per-scalar report as it finishes. Then:

4. Full archive of the best (010 and 008 full archives are already built; for 009 the
   fallback is 008 full or a new build).
5. Spare.

MAST-only rows take the leaderboard selection, so the full archive must be the last upload
of the day.
