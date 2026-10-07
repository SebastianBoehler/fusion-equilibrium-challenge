"""Local smooth interpolation of released DIII-D training equilibria."""

from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

from .data import features, targets


class InterpolationModel:
    def fit(self, rows: list[dict]) -> "InterpolationModel":
        x = np.concatenate([features(row) for row in rows])
        y = np.concatenate([targets(row) for row in rows])
        return self.fit_arrays(x, y)

    def fit_arrays(self, x: np.ndarray, y: np.ndarray) -> "InterpolationModel":
        valid = np.isfinite(y).all(axis=1)
        x, self.y = x[valid], y[valid]
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.x = (x - self.mean) / self.scale
        self._build()
        return self

    def _build(self) -> None:
        self.interpolator = RBFInterpolator(
            self.x, self.y, neighbors=64, smoothing=0.1,
            kernel="thin_plate_spline", degree=1)

    def predict(self, row: dict) -> dict:
        if row["source"] != "DIII-D":
            raise ValueError("The interpolation model supports DIII-D only")
        return self.predict_inputs(features(row))

    def predict_inputs(self, inputs: np.ndarray) -> dict:
        x = (inputs - self.mean) / self.scale
        y = self.interpolator(x)
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite interpolated predictions")
        return {"psirz": y[:, :-2].reshape(-1, 65, 65),
                "q95": y[:, -2], "betaN": y[:, -1]}

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, scale=self.scale,
                            x=self.x, y=self.y)

    @classmethod
    def load(cls, path: Path) -> "InterpolationModel":
        model = cls()
        with np.load(path, allow_pickle=False) as values:
            model.mean, model.scale = values["mean"], values["scale"]
            model.x, model.y = values["x"], values["y"]
        model._build()
        return model
