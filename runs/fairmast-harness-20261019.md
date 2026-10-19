# Magnetic validation harness — 18–19 October 2026

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
