# Code audit — 17 October 2026 (MAST pipeline vs intent)

*Refers to the code before the 18 October cleanup (tag `pre-cleanup-20261018`); the findings
were addressed or the code was removed (`docs/handoff/CLEANUP.md`).*

Read-only review of the MAST zero-shot path (`gs_picard`, `mast_gs_robust`, `mast_gs`,
builders, combine/ensemble/temporal, `participant/mast_*`). Notes and checks are in the
session scratchpad. No repository code was changed by the audit.

## Does the code do what we meant to test?

| Idea | Status | Evidence |
|---|---|---|
| Coil currents spread over elements; solenoid 0.5× feed per element (656) | Verified | `mast_green.py:60-61`, `mast_gs_robust.py:123` |
| efps through all P2U+P2L elements, positive sign; neg-all control | Verified | `mast_gs_robust.py:62-84` |
| Picard solve, Te¹ kinetic shape, β0 adapted to the Te axis anchor | Verified | `gs_picard.py:153-201` |
| LCFS bisection, inner wall 0.175 m, solenoid clearance 0.02 m | Verified | `gs_picard.py:45-60`, `mast_gs_robust.py:53-59` |
| Fallback to the 004 flux | Verified bit-identical (shots 0, 93) | reviewer check |
| q95 contour integral, βN from the flux pressure split | Verified | `q95_proxy.q_at`, `betan.py` |
| Temporal smoothing [¼,½,¼] + q95 median-5 | Verified | `temporal_smooth.py` |
| **LSN class "P2L stronger"** | **Deviation** | see below |
| config-linear / `--te-power` / `--edge` in vertical pass 1 | Deviation: pass 1 always uses kinetic Te¹, force mode, no edge | `mast_gs_robust.py:164-166` |
| Edge target "clamped" to 1.2–1.55 m | Deviation: out-of-range frames are dropped; <3 valid frames → no edge constraint for the shot | `mast_gs_robust.py:185-190` |

### LSN classifier (the only deviation that touches scored candidates)

The intent was "lower divertor coil stronger". The code uses the *signed* ratio
`(U−L)/(|U|+|L|)`, so its meaning flips when the P2 currents are negative or of mixed sign.
Of the 223 shots it flags:
- 26 have large P2 currents, both positive, with P2L stronger. These match the intent.
- 63 have negative or mixed-sign P2. For the both-negative ones the flag means P2U is
  stronger.
- 134 have a P2 total below 5 kA·turn. There the class is decided by near-zero currents.

In 034 (the flux behind 021b and 026), 20 of the 34 both-negative shots and 98 of the 133
positive small-P2 shots are pushed down by more than 0.1 m.

The 019 official result still showed that the config mode helps overall (Z_axis
consistency −0.08 → +0.30). That gain may come from setting everything else to Z = 0 rather
than from the downward pushes.

**Label-free A/B (run 041):** the same 223 shots were re-solved with Z = 0 and compared
paired, shot by shot, against 034.

| Group | Shots | Z = 0 lower Te mismatch on | Median change |
|---|---:|---:|---:|
| Negative or mixed-sign P2 | 63 | 28 | 0 |
| Small P2 | 134 | 88 | −0.0002 |
| Intended LSN | 26 | 20 | −0.008 |

The outboard edge is unchanged in all groups.

The check prefers Z = 0 even for intended-LSN shots, where the official 019 result shows
the downward shift helps. **The midplane Te mismatch does not measure vertical position.**
The classifier stays unchanged until a channel-separable official test is justified.

**Proxy reliability.** Candidate 027 (efps) improved both label-free proxies (X-point
fraction 0.77 → 0.92, Te mismatch 0.0686 → 0.0664) but lost 0.059 MAST S officially.
These proxies screen out broken candidates. They do not rank working ones.

## Bugs and fragility (none changes a scored candidate retroactively)

1. **Resume mixes configurations silently.** `run_shot` reuses any existing shot file
   without checking its flags. The meta and summary record no flags (034's flags had to be
   recovered from the session log).
2. **Builders don't check that the solve and scalar regions match.** `--inner` is passed
   independently to 005/006. A missing flag changes the q95/βN LCFS without an error.
3. **005 copies the DIII-D member without a hash check, and its receipt provenance is
   wrong.** It always says "mast-023" and hashes `mast_gs.py`, even for robust runs. The
   021base receipt shows this.
4. **`combine_channels --full-from` keeps float16 psirz** when the source is MAST-only.
   (The 026 full archive was built with `ensemble_channels`, which casts correctly.)
5. **Robust runs have no input pin check and no final finite/shape check.** `mast_gs.py`
   had both.
6. **The recompute path crashes the whole pool** if the 004 archive is absent and a frame
   has Ip ≤ 0. This matters for the private test set.
7. `te_edge_radius` can index −1 when the Te peak is below 30 eV (masked by the range
   filter).

## Dead code, duplication, length

- **Unused on the MAST path:**
  - `solve(init, edges, info, bz_fixed, bz_sign)`, `Profile.efit_c`
    and `vertical_shift(sz, dz)`.
  - The `vessel` option and the `force`/`smooth`/`config-linear` modes.
  - `mast_gs.te_peak_radius`/`PROFILE`.
  - `beta_n_electron`/`beta_n_equilibrium` (DIII-D validation only).
  - No warm-start path exists any more.
- **Worker plumbing.** Workers receive an 11-field positional task tuple padded with
  defaults; `solve` has 16 parameters, 13 of them passed positionally; 005 relies on
  mutated globals that macOS spawn workers never see.
- **Duplication:**
  - The archive helpers (`inner()`, DIII-D copy, receipt) are repeated in four builders.
  - The "valid median, then nearest fill" pattern is repeated five times.
  - External-flux assembly is repeated four times.
  - The midplane LCFS crossing is repeated twice.
- `mast_gs_robust.run_shot` is 110 lines. Stale docstrings: "Te^1.5", "|Z| < 0.3".

## Final-phase blockers

- `mast_inputs`, the builders and `temporal_smooth` hard-code `mast_public_test` and 1206
  shots.
- The fallback flux depends on a public-test archive (004 zip).
- Per-shot official worst-shot indices are hard-coded in the pilot list. This affects only
  which shots are piloted, not predictions. Remove it from the final package.

## Compliance

- **No MAST labels anywhere.** The MAST path reads no equilibrium labels, demos or
  envelope. The input allowlist rejects any `efit_*` column except times/grid, and rejects
  `magnetics_dsep`.
- **DIII-D labels are used only in DIII-D calibration and proxy scripts.**
- **To disclose:**
  - Official per-shot worst-shot lists guided the pilot selection and the
    vertical-mode design.
  - Variant and channel selection used official scores.

## Recommended cleanup

Leave the scored experiment modules unchanged; their hashes are cited in receipts. Build
one consolidated final-phase pipeline next to them:
- a `RunConfig` dataclass stored in every shot's meta, refusing mismatched resumes;
- the dataset config as a parameter;
- a fallback that doesn't need public-test archives;
- one `archive_io` helper module;
- only the live options.

Gate it with a regression check: bit-identical flux against 034/039 on a few shots.
