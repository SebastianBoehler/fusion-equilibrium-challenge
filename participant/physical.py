"""Dimensionless shaping-current ratios relative to measured plasma current."""

import numpy as np

from .kinetic import KineticModel


class PhysicalModel(KineticModel):
    @staticmethod
    def _ratios(x: np.ndarray) -> np.ndarray:
        ip = x[:, 20]
        if np.any(ip == 0):
            raise ValueError("Current-ratio model requires nonzero plasma current")
        result = np.column_stack([x, x[:, :20] / ip[:, None]])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite current ratios")
        return result

    def fit_arrays(self, x: np.ndarray, y: np.ndarray) -> "PhysicalModel":
        super().fit_arrays(self._ratios(x), y)
        return self

    def predict_inputs(self, x: np.ndarray) -> dict:
        return super().predict_inputs(self._ratios(x))


class PhysicalRegularizedModel(PhysicalModel):
    regularization = 1000.0
