# Search-space review 2 — night of 17→18 October 2026

**Best result:** 030, full archive, submission 996142.

| MAST S | G | Rank (G) | R²ψ | q95 R² | βN R² | D_LCFS | Consistency |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.7751 | 0.7790 | 10 / 50 | 0.9837 | 0.585 | 0.318 | 0.143 | 0.404 |

Leaderboard for reference: #1 0.941, #3 0.857, #8 0.786.

## 1. Where the points are (weighted gap to a top-3 MAST S of about 0.854, i.e. +0.079)

| Channel | Ours | Plausible target | Weighted gain |
|---|---:|---:|---:|
| Consistency (0.20) | 0.404 | 0.65 | **+0.049** |
| q95/βN mean (0.15) | 0.452 | 0.70 | **+0.037** |
| D_LCFS (0.10) | 0.143 | 0.08 | +0.006 |
| R²ψ (0.55) | 0.9837 | 0.990 | +0.003 |

Consistency and the scalars carry about 85% of the gap. Within consistency, the weak
terms are l_i (0.01), κ (0.26), tri_bot (0.29), Z_axis (0.50), tri_top (0.52),
R_axis (0.62) and volume (0.63).

## 2. What moved the score (all 17 official MAST results)

- **External field / topology:** the solenoid ×0.5, the inner wall and the vertical
  class are the large steps (+0.09, +0.08, +0.035, +0.013).
- **Shot-level vertical classification** from coil currents: 019 (+0.035) and 030
  (+0.013).
- **Temporal smoothing:** +0.006. Tonight's label-free noise analysis says the current
  kernel is already at or past the optimum; widening it is not justified.
- **Failed:**
  - efps (P2 counted twice);
  - uniform vessel;
  - uniform B_Z in the output;
  - linear basis;
  - P6-only class;
  - current diffusion (DIII-D gate).

## 3. The allowed method space (forum, 17 October brief) vs our coverage

| Allowed family | Coverage | Opportunity |
|---|---|---|
| Physics forward model from released inputs and geometry | Deep | Passive structure: only crude, failed versions so far |
| Label-free test-time adaptation (physics residual fitting) | Partial | Our residuals (Te symmetry, Te edge) are blind to Z, κ and δ |
| **Public MAST diagnostics outside the release** | **Not used** | See section 4 |
| Tune/select submitted variants with disclosure (several settings of one parameter allowed) | Used for selection | Allows a small parameter scan by upload (e.g. β0 range, anchor, class threshold) |
| DIII-D data, other machines' equilibria, OMFIT tables | DIII-D calibration only | A dimensionless scalar model trained on DIII-D; transfer is uncertain |
| Own synthetic equilibria with MAST geometry | Not used | A validation harness; an emulator only re-learns our solver |
| Self-distillation from own predictions | Smoothing only | — |

## 4. The largest untapped lever: public MAST magnetics (needs the human's decision)

- **The rules allow it.** The forum permits "Public MAST diagnostic inputs outside the
  release, such as per-winding P2 currents". FAIR-MAST publishes MAST magnetics (flux
  loops, pickup coils) as inputs, not labels.
- **Our own rule blocks it.** CLAUDE.md is stricter: "MAST uses allowlisted measured
  currents, Thomson data and permitted geometry only".
- **Two possible uses:**
  - **A. Clearly clean.** Use vacuum or coil-only shots from FAIR-MAST, which are not
    test shots. Flux loops measure the coil and passive-structure field exactly, so they
    can calibrate a passive-structure circuit model (the EFIT++/VALEN practice), which is
    then applied to the test shots unchanged. This targets the external field, the only
    lever that has ever moved R²ψ and shape at scale.
  - **B. Grey; do not do this without asking the organizers.** Re-identify the anonymised
    test shots by matching current traces, then use their own magnetics. That would be
    very powerful, but it sits next to the forbidden FAIR-MAST `equilibrium.*` data. The
    organizers anonymised the shot IDs on purpose.
- **Cost of A:** a FAIR-MAST download (public S3/zarr), a vacuum-shot selection, and a
  circuit fit. About 1–2 days, and it needs a network data download.

## 5. Tonight's actions

1. **Pressure-share βN.** Its estimator reached DIII-D R² 0.928. It is wired into the
   runner, and the flux is bit-identical. The full union run is cluster job 2975184. It
   becomes a βN-channel candidate.
2. **Single-run Final pipeline candidates** 031 and 032 are ready. They measure the scalar
   cost of dropping the separate q95/βN runs.
3. **Literature:** MAST passive geometry and βN/q95 estimators (litreview3).
