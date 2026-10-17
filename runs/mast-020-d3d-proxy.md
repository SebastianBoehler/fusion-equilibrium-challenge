# MAST 020 — DIII-D TRAIN proxy for the MAST physics pipeline

13 October 2026. The human set MAST as the main focus and authorized this proxy.

## Question

Official MAST flux R² is 0.76, while most top-20 Challenge-2 entries reach about 0.98.
MAST labels, demo files and the envelope are forbidden. The only measurable stand-in is
the same physics pipeline run on DIII-D TRAIN shots, where labels exist as diagnostic
targets. Which assumption costs the flux R²: the coil model, the missing vessel and eddy
currents, or the circular Gaussian plasma?

## Predeclared design

- **Inputs:** the 16 locally cached development shots
  (`run_data/full/raw/data/diii_d_train`), in sorted filename order. They are all in the
  256-shot development set. None of the 128 locked shots is read.
- **Exact split:** ψ_true = ψ_plasma + ψ_ext. The plasma current is
  J = −Δ*ψ_true / (μ0 R), restricted to the DIII-D scorer envelope. ψ_plasma is the Green
  integral of J. ψ_ext is the remainder, i.e. all coils plus vessel and eddy currents.
- **Variants:** pooled flux R² for each, with the official scorer components on top:
  - V_sanity: ψ_ext + Green(J)
  - V_coil_nominal: coils at the shipped currents + ψ_plasma (true)
  - V_coil_fit: per-coil least-squares gains + ψ_plasma (true). Diagnostic only, fitted on these TRAIN shots.
  - V_gauss_oracle: ψ_ext (true) + circular Gaussian matched to J's centroid, width and Ip
  - V_mast_like: coils at the shipped currents + oracle Gaussian
- **Gates:**
  - Median |Ip(J)/Ip_measured − 1| < 0.05.
  - The remaining source in ψ_ext inside the envelope is under 5% of |J|.
  - If either fails, the split is invalid and the variant numbers are not interpreted.
  - The originally listed V_sanity gate was dropped before execution: ψ_ext is defined as
    the remainder, so that variant equals ψ_true by construction.
- **Resource cap:** local CPU only, at most 30 minutes. No GPU, upload or MAST arrays.
- **Interpretation limit:** these are DIII-D numbers. They rank the error sources; they
  do not predict a MAST score.

## Results

`.venv/bin/python -B -m research.mast_proxy_d3d` ran on 16 shots / 3,941 frames in 3 s.
Output: `run_data/mast-020/proxy-report.json`.

**Gates pass:**
- Median |Ip(J)/Ip − 1| = 0.0046.
- Maximum external-source fraction = 0.042.

| Variant | Pooled flux R² |
|---|---:|
| coils at shipped currents + true plasma | 0.225 |
| per-coil fitted gains + true plasma | 0.977 |
| true external + circular Gaussian (oracle moments) | 0.984 |
| true external + elliptic Gaussian (oracle moments) | 0.972 |
| fitted coils + elliptic Gaussian | 0.946 |
| coils at shipped currents + circular Gaussian | 0.223 |
| coils at shipped currents only | −1.74 |

- **The external model dominates.** A crude Gaussian plasma costs only about 0.02–0.04 R²
  when the external flux is right.
- **The solenoid is the biggest single error.** The fitted ECOILA gain is about 140,
  versus 1 nominal. It is the kA-unit solenoid with an ambiguous turn convention.
- **F1/F2 gains of 0.53–0.67** probably absorb the unshipped, co-located ECOILB group and
  vessel currents. The individual gains are not identifiable here.
- **Transfer to MAST:** the solenoid turn convention is the first suspect for MAST's 0.76.
  It is tested in `mast-021-solenoid.md`.

### Official-scorer components (`--official`; output in `run_data/mast-020-official/`)

| Variant | S | R²ψ | D_LCFS | Consistency | κ R² | volume R² |
|---|---:|---:|---:|---:|---:|---:|
| shipped coils + true plasma | 0.568 | 0.230 | 0.015 | 0.963 | 0.966 | 0.849 |
| fitted coils + true plasma | 0.838 | 0.977 | 0.085 | 0.296 | −3.52 | −2.03 |
| true external + circular Gaussian | 0.858 | 0.985 | 0.067 | 0.365 | −6.68 | −2.64 |
| true external + elliptic Gaussian | 0.822 | 0.972 | 0.053 | 0.212 | −15.0 | −5.23 |
| shipped coils + circular Gaussian | 0.441 | 0.229 | 0.073 | 0.360 | −7.02 | −3.02 |

q95/betaN are set to the truth in these runs, so S includes a perfect qb term.

- **Flux R² and shape are nearly decoupled.** The solenoid error is an almost uniform
  offset: it destroys R²ψ but leaves consistency at 0.96.
- **The Gaussian plasma, not the coils, limits consistency and D_LCFS.** Even with the
  exact external field it reaches only 0.36 / 0.067.
- **Least-squares coil gains improve R²ψ but distort the shaping gradients.** Consistency
  falls to 0.30 with them.
- **Next step:** a self-consistent plasma current, J as a function of normalized flux
  inside the separatrix, solved together with the external field.
