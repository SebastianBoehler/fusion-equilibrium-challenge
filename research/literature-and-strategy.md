# Literature review and next-step strategy (8 October 2026)

This is a handoff brief: recent results, what the literature offers, and a
ranked experiment list. It covers Challenge 1 (DIII-D S) and Challenge 2
(MAST G_ratio). Code and run records: `participant/nn_*.py`, `participant/rich.py`,
[runs/nn-008.md](../runs/nn-008.md).

## 1. Where we stand (data)

| | Dev S (256 shots, official scorer) | Flux R² | q95/βN R² | D_LCFS | Consistency |
|---|---:|---:|---:|---:|---:|
| Old quadratic ridge + GBR scalars (official 0.9521) | 0.9575 | 0.98732 | 0.9594 | 0.0238 | 0.8648 |
| Five-member rich-MLP average (`submission-006`) | **0.99558** | 0.99987 | 0.99801 | 0.01007 | 0.98477 |
| Public top entries (test fold, not comparable) | 0.9963–0.9964 | 0.9999 | 0.998 | 0.0075–0.0084 | 0.987 |

Weakest consistency terms (five-member average): κ 0.972, li 0.974,
R_axis 0.986, Z_axis 0.989.

The shape-weight loss (uniform weight on whitened PCA coefficients) was the
largest single gain after the model-class change: consistency 0.9755 → 0.9813.
Weight 0.05 beat weight 0.2. Averaging gives about +0.0006 S.

Round 2 was queued on 8 October and was still running when this was written.
Results will land in `nn-008/runs/<name>/development.scores.json` on Galvani:

| Run | Job |
|---|---|
| s05-seed6 | 2949437 |
| s05-seed7 | 2949438 |
| wide-s05 | 2949439 |
| ctx2-s05 (`--context 2`) | 2949440 |
| s10-seed10 | 2949441 |
| s05-e80 | 2949442 |

### Error anatomy of the five-member average (`tests/nn_error_breakdown.py`)

- **Shot-level heavy tail.** 5 of 256 development shots (2%) hold 33% of flux
  SSE; 26 shots (10%) hold 57%.
- **The worst shots have unusual current or field direction.**
  - Ip < 0 with Bt > 0 (`7a2b87121e`)
  - Ip > 0 with Bt < 0 (`bddf1641b0`)
  - Ip < 0 with Bt < 0 (`6a6c6281a9`)

  Only 7.9% of development frames have Ip < 0.
- **The stored flux is sign-normalized.** The magnetic axis is the flux
  minimum in 100% of development shots for *both* Ip signs. ECOILA also flips
  sign with Ip (median −11 kA versus +12 kA). So stored ψ ≈ sign(Ip)·ψ_physical.
  The network must learn this sign flip from about 8% of the data.
  → Experiment A.
- **Ramp-down:** 2.6% of frames hold 7.1% of flux SSE and 16% of q95 SSE.
- **Ramp-up:** 9.1% of frames hold 28% of q95 SSE.
- **q95 is extremely tailed:** the top 1% of frames hold 55% of q95 SSE.
  These are early high-q95 frames, and pooled R² is dominated by them.
  → Experiment E.

## 2. Ranked experiments for Challenge 1

Expected effects are judgement calls, not measurements. Each experiment is
small in our code and runs as one ~20-minute A100 job (`variant.sbatch`).

**A. Exact current-sign canonicalization (high value, about 1 hour).**
Multiply every toroidal-current feature by s = sign(Ip): ECOILA, F1A…F9B, Ip,
and their slopes and lags (columns 0, 2–20 of each 21-block in `rich.py`).
Replace bcoil by |bcoil|; Bt sign does not change ψ, because the Grad–Shafranov
source contains FF′, which is invariant under F → −F. Keep the target as
stored. This makes reversed-Ip shots use the same mapping as normal shots,
effectively adding about 8% more data to the rare regime. Before training,
verify on training shots that stored ψ is invariant under the flip (plot the
mean ψ of matched shapes). Expected effect: lower error on the heavy-tailed
shots that dominate SSE.

**B. Geometry losses from training-only EFIT labels (high value).** Training
rows contain `efit_lcfs_r/z`, `efit_r_axis` and `efit_z_axis`. Currently
`nn_prepare.py` packs only the axis values and q95/βN/li. Add:

- **Isoflux loss:** bilinearly interpolate predicted ψ at the true LCFS points.
  Penalize their variance along the contour, and their mean offset from the
  true boundary flux. This is EFIT's own isoflux constraint, and it targets
  D_LCFS, κ and δ directly.
- **Axis loss:** penalize |∇ψ_pred| at the true axis (it must be a stationary
  point), and the error of ψ_pred(axis) against ψ_true(axis). This targets
  R_axis and Z_axis.

Both are cheap on the 65×65 grid with the decoder `psi_mean + c @ basis.T`.

**C. Sobolev (gradient) loss for li (medium-high).** li in the scorer uses
B_p = |∇ψ|/R integrated inside the LCFS. Pixel MSE barely constrains
gradients. Add a finite-difference loss on ∂ψ/∂R and ∂ψ/∂Z, optionally masked
to the inside of the true LCFS. Gradient-matching ("Sobolev training",
Czarnecki et al., NeurIPS 2017) is the standard tool. It complements the
shape weight, which mainly helped li and κ.

**D. Vacuum Grad–Shafranov residual (medium).** Outside the plasma and away
from coils, Δ*ψ = R ∂_R(R⁻¹ ∂_R ψ) + ∂²_Z ψ = 0 exactly. Penalize Δ*ψ_pred on
grid points outside the true LCFS (with a 2-cell margin, excluding coil
cells). arXiv:2511.19114 shows purely data-driven GS surrogates have large
physics residuals, and physics losses cut them by about 10⁴. Our pixel R² is
already 0.99987; the gain would be in derived quantities.

**E. q95 parametrisation (medium; targets 0.15 × scalar term).** Predict q95
as q_cyl × (1 + head), or predict 1/q95, while keeping the loss in raw q95
units (pooled R² is the metric). Here q_cyl ∝ |Bt|·a²κ/(R₀·|Ip|), built from
|bcoil| and |Ip|. This tames the early-ramp tail, where 1% of frames hold 55%
of q95 SSE. Cheap alternative: add |bcoil|/|Ip| as an explicit feature.

**F. Temporal context** (`--context K`, implemented; ctx2-s05 is running). If
it helps, try K = 4 and a small 1-D convolution or transformer over the full
shot sequence. Ramp-up and ramp-down errors and vessel eddy currents are the
motivation. Eqnet (NSTX-U) reports its biggest wins when eddy currents are
significant.

**G. Ensembling and finishing.** Use more seeds of the best recipe, plus an
EMA or SWA of weights (free gain). For the final model, train on train +
development once the recipe is frozen. Score the locked 128 confirmation
shots once, before that final refit.

**H. Up-down mirror augmentation (low, with a caveat).** ψ(R, −Z) with the A
and B coils swapped is an exact symmetry for currents. The core Thomson chord
only covers Z > 0, so mirrored samples carry inconsistent Thomson features.
Only usable with Thomson spatial features masked or dropped for mirrored
samples.

## 3. Challenge 2: MAST zero-shot (currently G = 0)

On the public board the best G_ratio is 0.941 (MAST R²ψ 0.9957, consistency
0.873). Most entries sit at 0.5–0.8, and several have negative MAST scalar R².
With our DIII-D S ≈ 0.995, G ≈ S_MAST, and any reasonable MAST submission
scores above 0. **Rules:** no MAST equilibrium or EFIT data, including the
three demo labels, for any purpose. Allowed: MAST inputs, MAST geometry, other
machines, self-supervised or test-time adaptation on MAST inputs, and synthetic
equilibria from our *own* free-boundary solver calibrated on MAST inputs.
Development-phase leaderboard feedback may be used if disclosed. Probe
submissions are not allowed.

Three routes, most promising first:

1. **Synthetic MAST training set from an open free-boundary solver.**
   - **Solvers:** FreeGS (it ships MAST and MAST-U style machine examples),
     FreeGSNKE (a Newton–Krylov forward solver validated against EFIT++ on
     MAST-U, reproducing "flux quantities and shape targets … to a very high
     degree of accuracy"), or TokaMaker (OpenFUSIONToolkit, FEM).
   - **Sampling:** coil currents from the shipped MAST test inputs (P2–P6,
     solenoid; the 2× solenoid ambiguity must be resolved by physics, not
     labels). Ip from inputs. Pressure and current profile parameters from
     Thomson (pressure ≈ 2·n_e·T_e) and a scanned li or βp.
   - **Training:** the same rich-feature MLP or neural operator, trained on
     the synthetic set.
   - **Domain gaps:** passive vessel currents (MAST has significant eddy
     currents; model them as fitted filaments, or randomize them during
     generation), and the solver's profile assumptions.

   The NN-emulator literature on FreeGSNKE (Agnello et al.) used 5–10-layer
   MLPs from coil currents, Ip, li and βp to shape targets, which is exactly
   the input space we have.
2. **Direct physics inference per frame:** a forward free-boundary solve with
   measured coil currents and Ip, and profiles from Thomson. No ML. Slow
   (about 69k frames), but embarrassingly parallel on CPU, and CPU data
   generation is uncapped by the rules. This is a useful upper bound and a
   source of pseudo-labels.
3. **Machine-agnostic hybrid:** ψ = ψ_vacuum + ψ_plasma.
   - **ψ_vacuum:** Green's functions from the shipped coil rectangles (MAST
     has 812 elements). The starter README cites R² ≈ 0.94 from a
     Green's-function vacuum fit on DIII-D.
   - **ψ_plasma:** learned on DIII-D in normalised coordinates
     ((R − R₀)/a, Z/a; flux in units of μ₀·Ip·R₀).

   The plasma-term model must transfer across aspect ratio (A ≈ 1.3–1.5 for
   MAST versus about 2.7 for DIII-D). This is the hardest route. The
   cross-device neural-operator study (arXiv:2606.15512) found that
   single-geometry pretraining transfers poorly, while multi-geometry
   pretraining adapts well. That argues for generating synthetic multi-machine
   data instead.

Remember the MAST flux sign convention (axis = maximum; the scorer
normalizes one global sign per machine). MAST flux amplitude is about one
third of DIII-D's, and amplitude is *not* normalized.

## 4. Literature map

| Work | What it is | Relevance |
|---|---|---|
| EFIT-AI / EFIT-MORNN (Akcay, Amara et al., GA; APS-DPP [2022](https://meetings-archive.aps.org/dpp/2022/go05/11), [2023](https://meetings-archive.aps.org/dpp/2023/ti01/6/), [2024](https://meetings-archive.aps.org/dpp/2024/go06/9/)) | FC + CNN surrogate for magnetic, MSE and kinetic EFIT on DIII-D. Jointly learns ψ and J_φ on the EFIT grid ("enforces toroidal force balance"). Uses an ensemble for uncertainty, and a database of more than 6M equilibria | Joint ψ + J_φ output is a physics-consistency trick; we have no J_φ labels, but Δ*ψ ∝ R·J_φ can be self-supervised (experiment D) |
| Kinetic-EFIT diagnostic study ([APS 2024](https://meetings-archive.aps.org/dpp/2024/co05/12/)) | Single-output networks gained accuracy as diagnostics were added | Supports separate heads or models for q95/βN |
| Next Step Fusion, *Reconstructing the plasma boundary with a reduced set of diagnostics* ([arXiv:2505.10709](https://arxiv.org/abs/2505.10709)) | DIII-D LCFS from coil currents only (MND 4 cm) versus + Ip + Vloop (3 cm). MLP with 150-80 units, LCFS as 90 polar points, about 25k shots. Coil-only models learn control-system current patterns | Our feature set is far richer. An auxiliary LCFS-points head is a cheap extra supervision signal (experiment B) |
| Joung et al., *Deep neural network GS solver constrained with measured magnetic signals* ([arXiv:1911.02882](https://arxiv.org/abs/1911.02882)) | KSTAR; network outputs ψ(R,Z) with a GS-residual loss; under 1 ms | GS-residual regularisation precedent (experiment D) |
| Jang et al., *GS equilibria via data-free PINNs* ([arXiv:2311.13491](https://arxiv.org/abs/2311.13491)) | Parameterised PINN; inputs pressure, aspect ratio, κ, δ | Physics losses; parameterisation idea for a cross-machine plasma term |
| Physics-informed neural operator for nonlinear GS ([arXiv:2511.19114](https://arxiv.org/abs/2511.19114)) | Five operator architectures. Data-driven models have large GS residuals; physics loss cuts them about 10⁴; semi-supervised is best | Motivates experiments C and D |
| Cross-device GS neural operator ([arXiv:2606.15512](https://arxiv.org/abs/2606.15512)) | Solov'ev testbed with eight geometries. Single-geometry pretraining transfers poorly; multi-geometry plus about 100 target samples gives under 4% relative L2 | Challenge 2: build synthetic multi-machine pretraining data |
| TCV liuqe-nn ([arXiv:2606.09487](https://arxiv.org/abs/2606.09487), [code](https://github.com/matgrand/liuqe-nn)) | DeepONet-style branch/trunk; autodiff physics regularisation; about 10k discharges; under 100 µs | Coordinate (trunk) decoder for grid-agnostic output, useful when MAST and DIII-D grids differ |
| Eqnet / Pertnet, NSTX-U (Wai, Boyer, Kolemen; [arXiv:2202.13915](https://arxiv.org/abs/2202.13915), [code](https://github.com/PlasmaControl/nstxu-nns)) | Spherical-tokamak NN reconstruction and forward GS; best gains when vessel eddy currents matter | The closest machine class to MAST; eddy-current features (our slope and lag features) |
| FreeGSNKE (Amorisco et al.) and MAST-U validation (Pentland et al., [arXiv:2407.12432](https://arxiv.org/abs/2407.12432), Phys. Scr. 2025) | Open Python free-boundary solver (static and dynamic, Newton–Krylov) with a MAST-U machine description | Challenge 2 synthetic data (route 1) |
| FreeGSNKE NN emulators (Agnello et al.; [IAEA FEC](https://conferences.iaea.org/event/392/contributions/36351/attachments/20324/34381/Agnello-TEC-CTL.pdf); MAST-U PCS [arXiv:2608.26216](https://arxiv.org/abs/2608.26216), [2608.28468](https://arxiv.org/abs/2608.28468)) | MLPs (5–10 layers, 100–200 units) from coil currents, Ip, li and βp to shape targets, trained on a synthetic library and deployed on MAST-U | Shows synthetic-to-real works on MAST-U for shape targets |
| TokaMaker / OpenFUSIONToolkit ([arXiv:2311.07719](https://arxiv.org/abs/2311.07719), [code](https://github.com/hansec/OpenFUSIONToolkit)) | FEM free-boundary GS, Python interface, cross-checked against FreeGS and EFIT | Alternative solver for route 1 |
| Serino et al., adaptive Newton free-boundary solver ([arXiv:2407.03499](https://arxiv.org/abs/2407.03499)) | Shape-calculus Jacobians for the free-boundary problem | If we need a differentiable solve |
| FusionMAE ([arXiv:2509.12945](https://arxiv.org/abs/2509.12945)) | Masked-autoencoder pretraining on fusion diagnostics | Self-supervised pretraining on MAST *inputs* is explicitly allowed |
| Challenge paper ([arXiv:2609.01750](https://arxiv.org/abs/2609.01750)) | Dataset and metric description; baselines only on a 3-shot demo; 87.9% of DIII-D and 66.4% of MAST frames kept as diverted | Confirms there is no published strong baseline; diverted-only filtering |

## 5. Pitfalls to keep

- Never read `efit_*` columns (except `efit_times`) or `magnetics_dsep` as
  inputs. The redaction checks in `rich.py` and `nn_prepare.py` must stay.
- Do not use MAST EFIT data, the MAST demo labels, or the MAST envelope as a
  model input or prior.
- Do not round ψ; float16 is fine, fixed-decimal rounding is not.
- Keep the 128 confirmation shots locked until a final recipe is chosen.
- Galvani SSH needs 2FA every few hours. Shrink job requests (16 CPU, 120 GB,
  ≤ 3 h) to get backfilled; fair-share is low because of other A100 jobs.
- Upload: a 1.4 GB direct upload takes about an hour. The HF pointer route
  needs a user-created fine-grained read token.

## 6. Update — rank estimate, gap breakdown, second literature round

### Estimated rank (public leaderboard read 8 Oct; top 50 of about 100 entries returned)

| If official DIII-D S equals | Rank |
|---|---:|
| 0.9967 (current #1, argonclouds) | 1 |
| 0.9960 | 11 |
| **0.9956 (our development score)** | **14** |
| 0.9950 (development score minus the earlier dev→official gap) | 17 |

Last time development scored 0.9575 and the official score was 0.9521. The
deficit grew by a factor of about 1.13 on the test fold. Applying that to 0.9956
gives about 0.9950, so expect **rank about 14–17 out of about 100** (from 69).
Challenge 2: the top G_ratio is now 1.007 (MAST beating DIII-D), then 0.941 and
0.906. Ours is 0.

### Where the remaining points are (development five-member average vs #1)

| Term (weight) | Ours | #1 | Weighted gap | Share |
|---|---:|---:|---:|---:|
| Consistency (0.20) | 0.98477 | 0.9881 | 0.00067 | **about 60%** |
| D_LCFS (0.10) | 0.01007 | 0.0068 | 0.00033 | **about 30%** |
| q95/βN (0.15) | 0.99801 | 0.9984 | 0.00006 | about 5% |
| Flux R² (0.55) | 0.99987 | 0.9999 | 0.00002 | about 1.5% |

Flux R² is saturated. **Nearly all remaining potential is geometry.** That means
κ (0.972), li (0.974), R_axis and Z_axis (about 0.986–0.989), and the boundary
distance. Experiments B (isoflux and axis losses) and C (gradient loss) target
exactly this. More ensemble members and more width mostly add flux accuracy,
which is already saturated; their benefit has flattened (+0.0006 from five
members).

### Round-2 status

- s05-seed6 (job 2949437): S 0.994587. This matches seeds 2 and 5 (0.99466 and
  0.99449), so the recipe is stable.
- Five round-2 jobs crashed on a V100 node, because the torch cu130 build has
  no sm_70 kernels. They were resubmitted restricted to `a100-galvani,2080-galvani`
  as jobs 2950124–2950128. **Never route these jobs to V100.**
- New jobs: `canon-s05` and `canon-s05-b` (2950129–2950130), experiment A,
  implemented as `--canon-ip` (`rich.canonicalize_currents`).
- Compare canonicalized runs against s05 seeds 2, 5 and 6 on S, on the worst
  five shots, and on frames with Ip < 0.

### Second literature round: additions that map onto our gaps

- **q95 physics prior (experiment E, sharpened).** The ITER Physics Design
  Guidelines q95 fit (Uckan 1989, implemented in
  [bluemira rules_of_thumb](https://bluemira.readthedocs.io/en/latest/autoapi/bluemira/plasma_physics/rules_of_thumb/index.html)):
  q95 ≈ (5a²B_t/(R₀I_p[MA])) · (1+κ²(1+2δ²−1.2δ³))/2 · (1.17−0.65ε)/(1−ε²)².
  - **Recipe:** add auxiliary heads for a, R₀, κ and δ. Their targets come from
    the training-only `efit_lcfs_r/z`.
  - **Output:** q95 = Uckan(predicted geometry, |bcoil|, |Ip|) × (1 + residual).
  - **Why:** this tames the early high-q95 tail, where 1% of frames hold 55% of
    q95 SSE. The same module has Wesson's li fit, a possible prior for li.
- **X-points are where flux surrogates fail.** The double-null neural-operator
  study ([arXiv:2608.05555](https://arxiv.org/pdf/2608.05555)) localizes its
  remaining discrepancies at the X-points. κ and δ_bot depend on X-point
  geometry in diverted plasmas. Weight the isoflux loss (B) more heavily near
  the X-point. The true X-point is approximately the lowest (or highest) LCFS
  point in training labels.
- **Startup eddy currents.**
  - DIII-D work ([APS 2022](https://archive.aps.org/dpp/2022/up11/63)) finds
    that EFIT fits degrade during plasma formation, tracking vessel eddy-current
    response. It proposes AC Green's functions.
  - SPARC studies ([APS 2023](https://meetings-archive.aps.org/dpp/2023/jp11/128))
    show eddy currents are largest at start-up and in strike-point sweeps.

  Our ramp-up and ramp-down frames are over-represented in error (ramp-down:
  2.6% of frames, 7.1% of flux SSE). Vessel currents obey v = 0 = R·I + M·dI/dt
  (see the circuit form in the GSPD paper,
  [arXiv:2306.13163](https://arxiv.org/pdf/2306.13163)). The physically right
  feature is therefore an exponentially decayed history of dI/dt: a few
  time constants of about 5–50 ms, per coil and for Ip. Our step-window slopes
  approximate this. Adding exp-filtered dI/dt features
  (∫ e^{−(t−t′)/τ} dI/dt′) is cheap in `rich.current_features`.
- **The EAST NN flux reconstruction** ([arXiv:2403.10114](https://arxiv.org/abs/2403.10114))
  scores the LCFS with Fréchet distance and reports 98% X-point and LCFS
  similarity. It confirms that geometry has to be checked separately from
  pixel error. Our official-scorer loop already does this.
- **A Bayesian NN for KSTAR equilibria** ([arXiv:2301.11555](https://arxiv.org/pdf/2301.11555))
  gives uncertainty estimates. These are useful for flagging out-of-distribution
  shots, such as reversed helicity or rare shapes, and for targeted
  oversampling.
- **Challenge 2: synthetic MAST-U libraries are proven at scale.** The FreeGSNKE
  emulator papers ([arXiv:2605.14939](https://arxiv.org/pdf/2605.14939),
  [arXiv:2509.01789](https://arxiv.org/pdf/2509.01789)) generated about 1.7–1.8M
  MAST-U equilibria. They used an MCMC walk over coil currents and profile
  parameters, seeded from about 4,000 real equilibria.
  - **For us:** seeding from MAST EFIT is barred. Seed from the MAST test-set
    coil currents, Ip and Thomson-derived pressure instead, and sample the
    profile shape parameters (βp, li) broadly.
  - **Sim-to-real tools:** domain-adversarial training plus input noise and
    calibration-error augmentation, as used on MAST-U in
    [APS 2025](https://archive.aps.org/dpp/2025/pp13/79).

### Recommended order for the next 48 hours

1. Read the round-2 and canonicalization results. Rebuild the average if it
   improves; candidates are canon, ctx2, wide-s05 and s05-e80.
2. Experiment B: isoflux and axis losses (needs `efit_lcfs_r/z` packed; one
   extra prepare pass). This is the biggest expected gain, and it targets 90%
   of the remaining gap.
3. Experiment C: gradient loss inside the LCFS, for li.
4. Experiment E with the Uckan prior.
5. In parallel on CPU: a FreeGS or FreeGSNKE MAST synthetic-data pilot for
   Challenge 2 (about 1,000 equilibria) to test feasibility.
