# Experiment checks and regression checks

These files are mostly command-line integration/diagnostic scripts, not an automatically
discovered data-free pytest suite. Read the script and its run receipt before execution.
Several read released TRAIN data, build predictions, fit old models or score real shots.
Importing an analytic helper does not mean the script's `main()` is safe to run generically.

| Files | Purpose / execution scope |
|---|---|
| `mast_green_checks.py`, `mast_surface_checks.py` | Physical identities and input-only MAST inference checks |
| `fast_score_checks.py` | Parallel-versus-serial scoring agreement on real TRAIN references |
| `nn_ensemble_check.py`, `nn_regression_check.py` | Saved-checkpoint scorer comparisons on declared DIII-D subsets |
| `nn_error_breakdown.py` | Diagnostic analysis, not a unit test |

The MAST runner's bit-identity regression is described in `docs/code-map.md`.

The 18 October cleanup removed six checks of the retired ridge pipeline (tag
`pre-cleanup-20261018`).

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
