# Experiment checks and regression checks

These files are mostly command-line integration/diagnostic scripts, not an automatically
discovered data-free pytest suite. Read the script and its run receipt before execution.
Several read released TRAIN data, build predictions, fit old models or score real shots.
Importing an analytic helper does not mean the script's `main()` is safe to run generically.

| Files | Purpose / execution scope |
|---|---|
| `mast_green_checks.py`, `mast_surface_checks.py` | Physical identities plus real input-only MAST inference and target isolation; completed baseline experiments, not needed again unchanged |
| `field_checks.py`, `geometry_checks.py` | Real TRAIN gradient/numerical checks for their loss modules; run only for a relevant separately declared change |
| `fast_score_checks.py` | Parallel-versus-serial scoring agreement; actual TRAIN references and scorer version matter |
| `scalar_file_checks.py` | Real-record file-descriptor stress regression; called from the old pipeline verifier |
| `scoring_resume_checks.py` | Cached-score identity/interruption/import rejection regressions; called by `verify_pipeline.py` |
| `verify_pipeline.py` | Legacy ridge/packing/scalar/scoring integration, including fitting and real data; not a current MAST/MLP smoke test |
| `verify_constant_profile.py` | Specific released constant-profile/timestamp edge case and target redaction |
| `nn_ensemble_check.py`, `nn_regression_check.py` | Saved checkpoint/scorer comparisons on declared subsets; no population or official-equivalence claim |
| `nn_error_breakdown.py` | Diagnostic analysis despite its directory/name; not a unit test |

No test was deleted in the handoff cleanup: no redundant invariant was established.
Mock patches in scoring-resume checks guard against unwanted recomputation; they do
not replace scientific inputs with fabricated training data.

For documentation-only cleanup, syntax parsing and link/source-identity checks suffice:

```sh
.venv/bin/python -B - <<'PY'
import ast
from pathlib import Path
for directory in ('participant', 'research', 'tests'):
    for path in Path(directory).rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
print('Python syntax parsed; no modules imported or data loaded')
PY
```

This is not behavioral verification. For implementation changes, select the smallest
real regression seam and required input scope. Preserve locked128 and all MAST label/demo/
envelope boundaries; avoid blanket test execution, benchmark loops and old job dispatches.
Completed numerical proofs reside under `run_data/` and `runs/`; retain them without replay.
