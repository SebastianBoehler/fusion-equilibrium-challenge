# Expert synthesis and next experiments — 14 October 2026

Inputs:
- Four expert reviews: equilibrium physicist, MAST operations, physics-ML transfer, scorer
  analyst.
- Official results 017/016/015b/014q/012; profiling of the worst shots.
- Details are in `runs/mast-official-20261014.md`.

## Where MAST S is lost (017: 0.7286)

| Term | Now | Leader | Points available | Main cause (evidence) |
|---|---:|---:|---:|---|
| Consistency (0.20) | 0.22 | 0.87 | 0.130 | Up-down placement (worst shots at Z ≈ +0.33 m; tri_bot/Z_axis flip with Z handling); current-profile width (l_i, κ); radial position (R_axis 0.48) |
| q95/βN (0.15) | 0.44 | 0.80 | 0.054 | βN (0.32): pressure split depends on profile and anchor; q95 (0.55) depends on the boundary |
| D_LCFS (0.10) | 0.19 | 0.05 | 0.014 | The worst shots (≈0.9) dominate; it is a per-shot mean |
| R²ψ (0.55) | 0.978 | 0.996 | 0.010 | Vertical misplacement on the worst shots; profile; passive currents |

**Realistic label-free target:** MAST S 0.76–0.79, G about 0.77–0.80, top-15 territory.
The leaders' 0.94 probably relies on MAST EFIT labels (FAIR-MAST), which is ruled out by
our boundary and by the leakage risk.

## Consensus findings

1. **Up-down: default symmetric (DN); offsets only for an identified LSN tail.**
   - Force balance amplifies noise (about 7 mm per mT of radial-field error).
   - Only P6 (anti-series) carries a measured asymmetry. P2 imbalance marks the early-era
     LSN.
   - USN is rare.
   - Run 032 (P2 class, forced downward) tests this.
2. **Current profile.** EFIT++ magnetics-only basis per term: (1−ψ̃)(1 + cψ̃). The profile
   width drives l_i and κ more than anything else (scorer experiments).
   - DIII-D proxy, axis-anchored: c = 0 gives κ −0.38 / l_i −0.02; c = 1 gives κ −0.00 /
     l_i −0.30; Te¹ gives κ 0.55 / l_i 0.33.
   - On MAST, linear vs Te¹ was about even (l_i −0.04 vs −0.02).
3. **Passive currents.** A crude wall hurts. A 1–2 parameter external correction (uniform
   B_Z, quadrupole) fitted to Thomson edges is the suggested stand-in.
4. **Thomson as constraints.** Outboard and inboard LCFS at the Te ≈ 30–40 eV crossing; an
   inboard/outboard Te-isotherm match for the axis and Shafranov shift. An EFIT-like
   per-frame inverse solve.
5. **Temporal regularization** of per-frame parameters within a shot (edge-preserving).
6. **Scorer facts.** A per-frame flux offset affects only R²ψ. Failures cost at most a
   mean-prediction equivalent. Pooled R² is dominated by tail frames. float16 is harmless.

## Experiment queue (all label-free; DIII-D proxy gate where applicable)

| # | Experiment | Target terms | Status |
|---|---|---|---|
| E1 | 032 → 018: P2-class LSN shifted downward, otherwise Z = 0 | Z_axis, tri_bot, R²ψ and D_LCFS on the worst shots | Running; MAST-only for 15 Oct |
| E2 | 018 + "no-up" variant: P2-class upward frames → 0 instead of mirrored | Safety comparison for E1 | To build (cheap) |
| E3 | Thomson outboard-edge isoflux constraint via a uniform B_Z correction (radial position from the edge instead of only the Te-peak anchor) | R_axis, volume, κ, D_LCFS | Implement and gate on DIII-D proxy |
| E4 | P6-based vertical signal (P6/Ip), shrinkage factor predeclared at 0.5 | Z_axis tail | Design |
| E5 | DIII-D-trained dimensionless regressor for βp/l_i → profile constraint and βN | βN, l_i | Design |
| E6 | Temporal smoothing of β0 and the axis anchor per shot | κ, l_i, q95 jitter | Cheap; implement |

**Upload plan for 15 Oct:**
1. 018 MAST-only, with 015b βN.
2. E2 or E3/E6 variants as ready.
3. Final full archive of the best component mix, built by the channel-combination method.

## 12:57 — E1 done: 032 → 018 → 019

- **032:** the 223 P2-class (LSN) shots re-solved with a forced-downward smoothed force
  balance; 99.9% plausible. The other 930 shots are identical to 029 (Z = 0).
- **Worst official shots, median Z_axis 016 → 018:**
  - LSN class: 0157 +0.04→−0.23; 0972 +0.36→−0.37; 0438 +0.31→−0.32; 0497 −0.02→−0.13;
    1070 +0.01→−0.27; 0422 +0.29→−0.29.
  - Non-class: 0678 +0.31→0.00; 0537 +0.27→0.00.
- **018 full:** q95 p5/50/95 = 4.73 / 6.99 / 10.20; βN from 002; 4,031,208,729 bytes.
- **019 MAST-only** = 018 ψ/q95 + 015b βN (`research/combine_channels.py`): 496,252,963
  bytes, SHA `1a901d9c…9eb7`. **First upload on 15 Oct.**
- **Comparison set already scored** (flux terms): 016 (smoothed force balance) 0.6635;
  014q (all Z = 0) 0.6612. 019 tests "P2-class down, others zero" against both.

**Also implemented:** E6 anchor smoothing (`--smooth-anchor`, untested); EFIT-family profile
(`Profile.efit_c`; DIII-D proxy: c = 1 fixes κ but hurts l_i, Te¹ remains best there).

## 16 Oct evening — analysis and new candidates

**Leaderboard** (public API, 19:20): G 0.7322, rank 24/143; DIII-D 26/143. The jygao row
is gone. Leaders: yszhang 0.978 (MAST consistency 0.92, D_LCFS 0.02), hornedyak 0.948,
dmitryzhevn 0.941.

**Flux error is common-mode.** 029 (Te¹) and 030 (linear) differ by only 0.47% of
variance. By the identity ‖(a+b)/2 − y‖² = ½‖a−y‖² + ½‖b−y‖² − ¼‖a−b‖² with the official
R² values, their ensemble would score R²ψ 0.97993 against 0.97956 for the best member. The
remaining 2% error is shared (external-field model), not profile.

**Worst-shot attribution** (official worst-20 lists):

| Run | Early campaign (114 shots) | P2-LSN class (223 shots) | Per-shot R² |
|---|---:|---:|---|
| 016 (smoothed force balance) | 13/20 | 12/20 | — |
| 014q / 015b (Z = 0) | 2/20 | 11–13/20 | 0.70–0.87 |

The early shots are hurt by upward force-balance placement. Under Z = 0, the LSN-class
shots are the worst. Both support 019 (LSN class down, others Z = 0).

**Bound for 019 against 029:** ‖d‖²/T = 0.0033, so R²ψ ∈ [0.976, 0.983].

**New candidates (MAST-only, βN from 015b unless noted):**
- **020:** config + linear EFIT++ basis (033). SHA `6ae087c5…2e2a`.
- **021:** config + Te¹ + anchor smoothing (034), building.
- **022:** config + Te¹ + anchor smoothing + **E3 edge constraint** (035), queued.
  E3 adds a uniform B_Z correction (stand-in for passive currents), adapted so the LCFS
  crosses the midplane at the Te = 30 eV edge (smoothed over 5 frames, clamped to
  1.2–1.55 m). Measured response: +0.01 T moves R_out by 3–9 cm.
  - **Pilot, 035 vs 032:** Te mismatch 0.0704 → **0.0551**; edge mean absolute deviation
    10.9 → 8.3 cm (partly circular); κ and l_i unchanged; volume median 7.6 → 7.9 m³.

**Profile-bias hypothesis (23:35).** Pooled R² punishes a constant bias. MAST literature
gives typical l_i ≈ 0.85 (Cunningham 2013, H-mode) and CDN κ ≈ 1.93 ± 0.05 (Meyer 2002).

| Pilot (with edge constraint) | l_i p10/50/90 | κ p10/50/90 | Te mismatch |
|---|---|---|---:|
| 035 (Te¹) | 0.83 / 1.04 / 1.28 | 1.62 / 1.81 / 1.97 | 0.055 |
| 036 (Te^0.5) | 0.61 / 0.71 / 0.83 | 1.81 / 2.08 / 2.32 | 0.056 |

The literature brackets **p ≈ 0.75**. Physically, bootstrap and edge current flatten J
relative to Spitzer. This is a physics prior, not fitted to labels; the literature values
are partly from H-mode subsets, so the leaderboard tests it.

Queued: **037 → 023** = 035 settings with `--te-power 0.75`.

**Upload ladder for 17 Oct:**

| Slot | Candidate | Change |
|---|---|---|
| 1 | 019 | vertical fix |
| 2 | 021 | + anchor smoothing |
| 3 | 022 | + edge constraint |
| 4 | 023 | + Te^0.75 |
| 5 | full archive of the best | — |

βN channel experiments ride along in 021/022. 020 (linear) is dropped from the ladder but
kept on disk.

## 17 Oct, 00:00–00:30 — official 019/020; cluster enabled

**Official, MAST-only:**

| Submission | MAST S | R²ψ | q95 | βN | D_LCFS | Consistency | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 994550 (019: config, Te¹) | **0.748477** | 0.981633 | 0.5407 | 0.3178 | 0.1591 | **0.3005** | Z_axis **+0.30** (was −0.73), tri_bot **+0.26** (was −0.40), κ 0.20, tri_top 0.32, volume 0.56, R_axis 0.47, li −0.00 |
| 994567 (020: config, linear basis) | 0.726686 | 0.979805 | 0.3538 | 0.3178 | 0.1618 | 0.2680 | — |

- The vertical fix is confirmed. G with DIII-D 0.995079 ≈ **0.7522** (was 0.7322).
- The Te¹ basis beats linear on every channel. Linear is dropped.
- **019 full** (DIII-D + 019 MAST, float32): `run_data/submission-mast-019/fusion-mast-config-betan-019-full.zip`,
  SHA `ff6f7d97…a0a7`. The private fold forwards the best entry per team, so a full entry
  must exist.
- **021b** = 034 flux/q95 (config + anchor smoothing) + βN from its own Te¹ flux: SHA
  `733ef909…6521`.

**Cluster** (Slurm CPU partition; cluster runs authorized on 17 Oct). Root:
`<work dir>/mast-gs-20261017`.
- MAST inputs were downloaded there from HF at the pinned revision with SHA checks (482 s).
- `research.mast_gs_robust.fallback_psi` recomputes the 004 flux when the archive is absent;
  it is bit-identical to the archive.
- `ops/mast-gs-array.sbatch` shards a run over 4 × 32-core CPU nodes.
- Submitted: 2973839 (037: Te^0.75 + edge + smooth anchor + config) and 2973844 (035: edge +
  smooth anchor + config). The local 035 run continues as a backup.

## 17 Oct — 022 result, compliance pruning, channel-selected full 024

**994704 (022: config + smooth anchor + edge B_Z constraint; uploaded by the human's
other session):**
- MAST S **0.708768**: R²ψ **0.9250** (was 0.9816), q95 **0.5714** (best so far), βN
  0.3178, D_LCFS 0.1744, consistency 0.2538.
- Consistency per scalar: R_axis 0.646 (was 0.471), volume 0.668 (was 0.556); κ −0.075,
  tri_top 0.168, tri_bot −0.149, li −0.141.
- **Interpretation:** the uniform-field stand-in (ψ_c = B_z R²/2) alters the flux across
  the whole grid and costs R²ψ. The same constraint improves radial placement and size.
- **Next:** 038 = the same solve with `--edge-nobz` (the correction positions J; output =
  coil + plasma flux only). Cluster job 2973907.

**Compliance pruning:**
- The forum clarification prohibits priors or constants derived from MAST equilibrium
  statistics.
- **Dropped:** 023 (Te^0.75, chosen to match published MAST l_i/κ). Cluster job 2973839
  was cancelled, and its local run blocked by `run_data/mast-037/shots` being a file.
- **Kept, with disclosure:** solenoid factor (FreeGSNKE 0.5 multiplier + submitted-variant
  selection); geometry-based inner wall; input-derived P2 LSN class; DIII-D-calibrated Te¹
  profile; channel selection across submitted variants.

**024 full** = 018/019 ψ (float32) + 022 q95 + 015b βN + 973093 DIII-D
(`research/ensemble_channels.py`): `run_data/submission-mast-024/fusion-mast-best-024-full.zip`,
SHA `06993a0e…1094`. Predicted from the separable channels:
0.684094 + 0.15 · (0.571435 + 0.31776)/2 = **0.750784**; G ≈ 0.7545.

**Repository:** the current code was committed and pushed to `origin/main` (031a014).
The repo remains private until the final submission.

## 17 Oct, 04:00 — 021b official; leaderboard; efps finding

- **994648 (021b):** MAST S **0.756107**. R²ψ 0.98226, D_LCFS 0.1583, consistency
  0.3321 (R_axis 0.593, Z_axis 0.301, κ 0.217, tri_top 0.333, tri_bot 0.289, volume 0.591,
  li 0.001), q95 0.5606, βN 0.3096 (own Te flux; 015b's 0.3178 stays best).
- **Full 025** = 021b ψ + 022 q95 + 015b βN. Predicted MAST S **0.7575**, G ≈ 0.7613;
  SHA `f5e2dd59…f1ac`.
- **026** = 021b ψ smoothed [¼ ½ ¼] in time + median-5 of 022 q95 + 015b βN.
  ‖d‖²/T = 0.00032, so R²ψ moves by at most ±0.0003.
- **Leaderboard at 03:30:** our selected row is the MAST-only 022 (G 0), rank 143/145.
  It needs a full archive as the last upload. Top G: yszhang 0.978, catskills 0.965
  (DIII-D 0.917), hornedyak 0.948 (DIII-D 0.860), dmitryzhevn 0.941.
  - **DIII-D-denominator lever:** the gate allows DIII-D 0.85, up to 1.16× G, at the cost
    of Challenge 1. Flagged to the human; not pursued.

**Forum facts:**
- MAST `sol_current` is exactly 2× the FAIR-MAST level-2 solenoid current. This
  independently confirms the 0.5 factor.
- **`efps_current` is a P2 link-board feed current.** Our model ignored it.
  **Correction (17 Oct):** ignoring it was right. efps is the P2 per-turn current and is
  already inside the P2 totals; the efps runs below double-counted P2 (027: MAST S 0.703).

**efps statistics:** median 5.4 kA (p95 11.4 kA); non-zero in 85% of frames; correlation
0.81 with P2U. P2's per-turn current has median 4.9 kA.

**Pilot** (035 settings minus edge: config + smooth anchor; 35 shots):

| efps flows through | X-point-defined frames | Te mismatch | κ median | l_i median |
|---|---:|---:|---:|---:|
| none (034) | 0.77 | 0.0686 | 1.81 | 1.06 |
| **all P2 turns** | **0.92** | **0.0664** | 1.88 | 1.01 |
| inner section | 0.86 | 0.0665 | 1.85 | 1.03 |
| outer section | 0.84 | 0.0669 | 1.84 | 1.04 |

The corpus is diverted-only (published dataset description), so the X-point fraction is a
label-free structural check. "All" is the most physical.

**Full run 039** (config + smooth anchor + efps all): running locally and as cluster array
2973911.
