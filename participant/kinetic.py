"""Current interactions augmented with measured electron pressure profiles."""

import numpy as np

from .data import features, targets
from .quadratic import QuadraticModel


def kinetic_features(row: dict) -> np.ndarray:
    columns = [features(row)]
    for system in ("core", "edge"):
        times = np.asarray(row[f"thomson_{system}_times"], dtype=float)
        te = np.asarray(row[f"thomson_{system}_Te"], dtype=float)
        ne = np.asarray(row[f"thomson_{system}_ne"], dtype=float)
        if te.shape != ne.shape or te.shape[0] != len(times) or len(times) == 0:
            raise ValueError(f"Invalid Thomson {system} profiles")
        summaries = []
        for values in (te, ne, te * ne):
            physical = np.where(np.isfinite(values), np.maximum(values, 0), np.nan)
            summaries.extend(np.nanquantile(physical, [0.5, 0.9], axis=1))
        matrix = np.column_stack(summaries)
        aligned = []
        for values in matrix.T:
            valid = np.isfinite(times) & np.isfinite(values)
            if valid.sum() == 0:
                raise ValueError(f"Undefined Thomson {system} summary")
            selected = values[valid]
            if np.all(selected == selected[0]):
                # A measured constant has no time variation, including zero diagnostics.
                aligned.append(np.full(len(row["efit_times"]), selected[0]))
            else:
                if np.any(np.diff(times[valid]) <= 0):
                    raise ValueError(f"Invalid Thomson {system} timestamps")
                aligned.append(np.interp(row["efit_times"], times[valid], selected))
        columns.append(np.column_stack(aligned))
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite kinetic model inputs")
    return result


class KineticModel(QuadraticModel):
    def fit(self, rows: list[dict]) -> "KineticModel":
        return self.fit_arrays(np.concatenate([kinetic_features(row) for row in rows]),
                               np.concatenate([targets(row) for row in rows]))

    def predict(self, row: dict) -> dict:
        if row["source"] != "DIII-D":
            raise ValueError("The kinetic model supports DIII-D only")
        return self.predict_inputs(kinetic_features(row))
