"""Keep the flux model and learn nonlinear q95/betaN from kinetic diagnostics."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from .kinetic import kinetic_features
from .quadratic import QuadraticModel


class ScalarTreeModel:
    def fit_arrays(self, x: np.ndarray, y: np.ndarray) -> "ScalarTreeModel":
        self.flux = QuadraticModel().fit_arrays(x[:, :21], y)
        valid = np.isfinite(y[:, -2:]).all(axis=1)
        self.scalars = ExtraTreesRegressor(
            n_estimators=64, min_samples_leaf=8, max_features=0.8,
            n_jobs=1, random_state=20261007).fit(x[valid], y[valid, -2:])
        return self

    def predict_inputs(self, x: np.ndarray) -> dict:
        prediction = self.flux.predict_inputs(x[:, :21])
        scalars = self.scalars.predict(x)
        if not np.isfinite(scalars).all():
            raise ValueError("Nonfinite scalar predictions")
        prediction.update(q95=scalars[:, 0], betaN=scalars[:, 1])
        return prediction

    def predict(self, row: dict) -> dict:
        if row["source"] != "DIII-D":
            raise ValueError("The scalar-tree model supports DIII-D only")
        return self.predict_inputs(kinetic_features(row))

    def save(self, path: Path) -> None:
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "ScalarTreeModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError("Wrong scalar-tree checkpoint type")
        return model
