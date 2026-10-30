# Approach, results and lessons

This is a summary of our entry. The dated notes in [`runs/`](../runs/README.md) hold the
detailed evidence. The rebuild instructions are in [`repro/README.md`](../repro/README.md).

## The task

Given magnetic and kinetic diagnostics for a tokamak shot, predict per time frame:
- the poloidal flux map ψ(R, Z);
- the safety factor q95;
- the normalised beta βN.

The challenge had two awards:
- **Challenge 1** scores DIII-D predictions with a composite S. Models may train on DIII-D
  equilibrium labels.
- **Challenge 2** scores zero-shot transfer to MAST, as G = S_MAST / S_DIII-D. MAST
  equilibrium reconstructions are off limits for training, calibration and selection.

The two machines differ strongly. DIII-D is a conventional tokamak. MAST is a spherical
tokamak with a large solenoid and a close-fitting vessel. A model trained on DIII-D does
not transfer to MAST, so our MAST path is physics, not learning.

## DIII-D: residual MLP ensemble

- **Inputs:** actuator histories (coil currents, Ip, toroidal field) and Thomson-scattering
  features (`participant/rich.py`). The organizer Ip timing correction is applied.
- **Model:** a residual MLP that predicts coefficients of a PCA flux basis, plus scalar
  heads for q95 and βN (`participant/nn_train.py`, `participant/flux_head.py`).
- **Ensemble:** 11 members across a few configurations and seeds. The last Final submission
  used 22 members (11 extra seeds).
- **Training cost:** about 6 A100-hours for all 22 members.
- **Split:** 6,641 train / 256 development / 16 regression shots (`repro/diii_d_split.json`).
  We set aside 128 more shots as a locked held-out set and never read them.

DIII-D was close to saturated. On the Development leaderboard our S was 0.9951. Our ensemble
went from 0.996037 to 0.996130 on our own development split when it grew to 22 members.

## MAST: label-free free-boundary Grad–Shafranov

Pipeline: `research/mast_gs_robust.py` (solver runner) → `research/mast_final.py` (scalars,
smoothing) → `participant/final_archive.py`.

1. **External field.**
   - Measured PF coil currents are spread over their filament elements.
   - The solenoid carries **0.5 × the released feed current** per element (656 elements).
     Three independent lines of evidence support this factor: Development feedback, a
     vacuum flux-loop fit, and a gain of 0.98 on the CCBV pickup coils.
   - Optional **passive structure** comes from the public FAIR-MAST `pf_passive` geometry
     (`research/geometry/`, `research/mast_passive.py`). Its eddy currents are driven by
     the measured dI/dt with literature resistivities.
2. **Plasma.**
   - A Picard free-boundary GS solver (`research/gs_picard.py`) uses a current density
     J ∝ Te(ψN)^0.75, taken from midplane Thomson scattering.
   - The pressure share β0 is adapted so that the magnetic axis sits at the Te centroid.
3. **Vertical position.** Shots are assigned to a vertical class using coil currents only
   (`config-union`). Shots with P2 up/down asymmetry below −0.02, or a P6 radial-field
   signal |m6| > 4, get a smoothed force-balance height forced downward. All other shots
   sit at Z = 0.
4. **Scalars.**
   - q95 comes from a field-line integral at ψN = 0.95.
   - βN is the mean of the flux pressure split and the kinetic Thomson estimate.
5. **Post-processing.** Within each shot, ψ is smoothed in time with weights [¼, ½, ¼], and
   q95 with a median-5 filter.

### Validation without labels

We could not score MAST candidates against equilibria, so we built a magnetics harness on
**301 public FAIR-MAST shots outside the challenge** (`research/fairmast_*.py`). It compares
predicted and measured flux loops and pickup coils, on disjoint tuning and holdout subsets.
It ranks physics choices without touching any reconstruction. For example, the passive
structure plus Te^0.75 raised the plasma flux-loop correlation from 0.53 to 0.76 on the holdout.

Cheaper label-free proxies screened out broken candidates, such as the Te flux-function
mismatch and the fraction of X-point frames. They could not rank working ones. Magnetics on
non-test shots could.

### Final submission 3: re-identified test shots (disclosed)

On 22 October we matched anonymised MAST test shots to public FAIR-MAST shots by their Ip
waveforms, under a predeclared rule. 845 of 1,205 shots matched. For the matched shots in
the centred class, we re-solved with the FAIR-MAST level-2 coil currents, which differ from
the released ones by a few percent. No equilibrium product was read. The full disclosure is
in `repro/README.md`.

## Results

| Stage | DIII-D S | MAST S | G | Place |
|---|---:|---:|---:|---|
| Development leaderboard (18 Oct, entry 030) | 0.9951 | 0.775 | 0.779 | 11 / 50 (G) |
| Final standings (29 Oct) | n/a | n/a | n/a | Challenge 1: 19 / 58, Challenge 2: 16 / 46 |

The organizers published the Final standings as places only, without scores.

**Where the MAST gap was.** The flux R² was close to the leaders (0.984 vs about 0.995).
The flux-map consistency score was far below them (0.40 vs about 0.9). So were the shape
scalars: κ, triangularity and l_i. The bottleneck was the plasma shape and current profile,
not the flux itself.

**Development steps that mattered** (gains in MAST S on the official board):

| Change | Gain |
|---|---:|
| Solenoid × 0.5 | +0.09 |
| Self-consistent solver with inner wall | +0.08 |
| Vertical class | +0.035, +0.013 |
| Temporal smoothing | +0.006 |

**Changes that failed:**
- Adding `efps` currents: they are already included in the P2 totals, so this double-counts.
- A uniform vessel current.
- A uniform B_Z in the output.
- A linear flux basis.
- A P6-only vertical class.

## Lessons

- **Verify facts against data, not against forum summaries.** Two "known" facts were wrong:
  the solenoid scaling and whether `efps` is separate from P2.
- **Proxies screen; they do not rank.** Use physical measurements on held-out shots to
  choose between working candidates.
- **Pooled R² on shape scalars is dominated by bias and outliers.** A few mis-placed shots
  cost more than a better average.
- **We searched too narrowly.** After 13 October nearly all effort went into one family of
  forward-physics models, tuned against magnetics.

Directions we did not explore:
- per-frame inverse solves constrained by magnetics;
- synthetic MAST equilibria (for example FreeGSNKE) with domain randomisation, to train a
  learned or hybrid model;
- Newton–Krylov solvers and explicit X-point handling;
- shape priors derived from physics rather than labels.

## Use of AI assistance

Claude (Anthropic), working as a coding agent, implemented, ran and validated most of the
pipelines under the author's direction. It also wrote most of the dated notes in `runs/`.
In those notes, "the human" means the author. The challenge rules required this disclosure.
