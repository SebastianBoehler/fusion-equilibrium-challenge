# Magnetic validation harness — 18–19 October 2026

> **Correction (19 Oct, code review).** The converter doubled the solenoid current. The
> release `sol_current` equals FAIR-MAST level 2: the medians agree within 1%, against 1.01
> for P4. Every "solenoid s" below is therefore **physically 2s**:
> - the "0.25" runs are the correct 0.5 × release;
> - the "0.5" runs are 1.0.
>
> The vacuum fit (0.49–0.50 × level 2 per element) confirms the original
> `SOL_SCALE = 0.5`. It does not support 0.25. Final submission 2 (`--sol-scale 0.25` on
> release data) used half the correct solenoid. The variant ranking holds at the correct
> scale: passive ALLg helps, Te exponent 0.75 helps, the vertical class helps.

**Data:** FAIR-MAST level 2 for 301 non-release MAST shots (seed 20261018, shots 27000–30470).
Only coil currents, Ip, TF, flux loops and Thomson were read; never `equilibrium` or EFIT
products. The human approved the download on 18 October.

**Pipeline:** `research/fairmast_download.py` → `research/fairmast_inputs.py` →
`research/mast_gs_robust.py` (`MAST_DATA_DIR=run_data/fairmast/data`) →
`research/fairmast_harness.py`. The flux loops are validation targets only, never solver
inputs.

## 1. The solenoid is half as strong as modelled

**Vacuum test.** Before breakdown the solenoid is precharged and there is no plasma. On
three shots (27229, 30411, 30420), regressing the measured centre-column loop flux on our
computed flux gives a solenoid coefficient of **0.49–0.50** for every loop (CC03–CC09). The
reference is our model of 0.5 × `sol_current` per element. The PF coefficient is about 1.0–1.5,
so the units and the PF scale are consistent.

**Plasma frames, 20 shots.** Flux-loop R² for each solenoid scale:

| Loops | 0.5 | 0.25 |
|---|---:|---:|
| Centre column | −0.44 | 0.94 |
| Coil loops | 0.16–0.84 | 0.78–0.96 |

## 2. Variant ranking (80 shots, 7,365 frames; CC04 excluded as broken)

The "plasma" columns subtract the run's own coil-plus-passive flux from both measurement and
prediction, so only the plasma contribution is compared.

| Variant | Total coil-loop R² | Plasma R², CC | Plasma R², coils |
|---|---:|---:|---:|
| Solenoid 0.5 (Development 030 recipe) | 0.45 | −0.45 | 0.24 |
| Solenoid 0.25 | 0.90 | 0.40 | 0.56 |
| 0.25 + passive V | 0.94 | 0.44 | 0.73 |
| 0.25 + passive ALLg | 0.94 | 0.77 | 0.76 |
| + Te exponent 0.75 | 0.94 | 0.78 | 0.77 |
| + Te exponent 1.5 | 0.94 | 0.73 | 0.75 |
| + Z = 0 for all shots | 0.93 | 0.73 | 0.72 |
| + Te-symmetry anchor correction | 0.95 | 0.76 | 0.78 |

The ALLg plasma-CC value is 0.551 in the first, buggier pass; 0.766 after the CC04 exclusion
fix. Rows from "+ Te exponent 0.75" down are each added on top of 0.25 + ALLg.

## 3. Reading

- The external-field correction (solenoid 0.25, passive structure from public geometry)
  is confirmed by measured magnetics.
- The coil-current vertical class is confirmed: Z = 0 for all shots is worse.
- A broader current profile (lower Te exponent) is preferred.

**Not measured by flux loops:**
- internal shape (κ, δ, l_i) and the LCFS;
- the diverted fraction, which drops at 0.25 on the public set (0.81 → 0.34).

**Leaderboard status:** none of this was scored in Development, which closed on 18 Oct at
00:00 UTC.

## 4. Te exponent and anchor scan (80 shots; mean of the plasma CC and coil metrics)

| Configuration (all solenoid 0.25 + ALLg) | Mean plasma R² |
|---|---:|
| Exponent 1.0 | 0.761 |
| Exponent 0.75 | 0.779 |
| Exponent 0.5 | 0.780 |
| Exponent 0.75 + anchor correction | 0.781 |

The exponent plateaus below 0.75.

## Final submissions (blind; the best MAST S and DIII-D S across scored submissions count)

DIII-D is the same in all three: the incumbent ensemble, `run_data/final/d3d-pred`.

| # | MAST configuration | Basis |
|---|---|---|
| 1 | Solenoid 0.5, union class, exponent 1.0, βN `flux-kin` | Development 030 recipe |
| 2 | Solenoid 0.25, passive ALLg, exponent 0.75, βN `flux-kin` | Best measured-magnetics match |
| 3 | Submission 2 + Te-symmetry anchor correction | Best combined harness score |

## 5. Corrected data + calibrated probes (19 Oct evening; 80 shots, 7,365 frames)

**Setup:**
- Inputs re-converted with the solenoid fix (`run_data/fairmast/data2`); every run uses
  the correct solenoid of 0.5 × release.
- Probes were calibrated per probe on the vacuum precharge, with exact external fields at
  each probe (`research/fairmast_probe_model.py`).
- Probe columns give R² as mean / median.

| Variant | Plasma loops CC | Plasma loops coils | CCBV | OBV | OBR |
|---|---:|---:|---|---|---|
| Submission 1 config (no passive, Te¹) | 0.58 | 0.56 | 0.78 / 0.81 | 0.26 / 0.46 | 0.80 / 0.86 |
| **Passive ALLg, Te^0.75** | **0.78** | **0.77** | 0.71 / 0.81 | **0.79 / 0.76** | 0.79 / **0.95** |
| + inner wall 0.19 m | 0.78 | 0.77 | 0.70 / 0.81 | 0.79 / 0.76 | 0.79 / 0.95 |
| + Te^0.5 | 0.78 | 0.78 | 0.70 / 0.81 | 0.78 / 0.78 | 0.82 / 0.94 |

**Vacuum gains** at the correct solenoid: CCBV 0.98, OBV 0.85–0.89, OBR 0.83–0.94. With a
doubled solenoid, the CCBV gain is 0.44.

**Decision: the last Final slot (submission 3) is `--passive ALLg --te-power 0.75`** at the
default solenoid 0.5. The inner wall and Te^0.5 do not change anything measurably.

## 6. Round 2 on the same 80 shots (passive ALLg, Te^0.75 base)

Probe columns give R² as mean / median.

| Variant | Plasma loops CC | Coils | CCBV | OBV | OBR |
|---|---:|---:|---|---|---|
| **Base (submission 3)** | 0.784 | 0.773 | **0.71 / 0.81** | 0.79 / 0.76 | 0.79 / 0.95 |
| + anchor correction | 0.772 | 0.790 | 0.70 / 0.79 | 0.82 / 0.78 | 0.77 / 0.96 |
| Passive ALL (rodgr blocks as steel) | 0.812 | 0.771 | **−0.08 / 0.46** | 0.78 / 0.76 | 0.80 / 0.94 |
| Passive VC (incl. rodgr) | 0.805 | 0.764 | **−0.04 / 0.57** | 0.74 / 0.70 | 0.85 / 0.89 |
| + edge (nobz) | 0.785 | 0.766 | 0.71 / 0.81 | 0.78 / 0.76 | 0.83 / 0.96 |

**Reading:**
- The centre-column probes reject steel rodgr blocks. Graphite armour (ALLg) is
  consistent with them.
- The anchor correction and the edge option are mixed or neutral.
- The base stays. Next: confirmation on a disjoint 80-shot holdout
  (`run_data/fairmast/holdout80.json`).

## 7. Holdout confirmation and the fine passive model (20 Oct)

Disjoint 80-shot holdout (6,996 frames). Probe columns give R² as mean / median.

| Variant | Plasma loops CC | Coils | CCBV | OBV | OBR |
|---|---:|---:|---|---|---|
| Submission 1 config | 0.53 | 0.53 | 0.66 / 0.72 | 0.08 / 0.20 | 0.67 / 0.76 |
| Passive ALLg, Te^0.75 | 0.75 | 0.81 | 0.62 / 0.71 | 0.73 / 0.71 | 0.79 / 0.85 |
| **Passive ALLgf (pieces ≤ 1 cm), Te^0.75** | **0.76** | **0.81** | **0.75 / 0.78** | 0.73 / 0.71 | 0.79 / 0.85 |

**Ablation OUT** (no centre-column structure, tuning set): CCBV 0.88 / 0.91, but plasma
loops CC fall to 0.69. The Inconel tube's current is real; the earlier CCBV deficit was a
discretization artefact. The tube sits 6.5 mm from the probes, with 8 cm filament pieces.

**ALLgf on the tuning set:** CCBV 0.86 / 0.88; all other metrics are unchanged against
ALLg.

**Final candidate:** `run_data/final/submission-3-v5/fusion-final-3.zip` (ALLgf, Te^0.75,
solenoid 0.5, βN `flux-kin`). It is validated and tagged `final-submission-3-v5`. The
earlier ALLg build in `run_data/final/submission-3/` is the fallback.
