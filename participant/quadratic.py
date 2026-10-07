"""Regularized current interactions for the nonlinear equilibrium response."""

from pathlib import Path

import numpy as np

from .ridge import RidgeModel


class QuadraticModel(RidgeModel):
    regularization = 100.0

    def _basis(self, inputs: np.ndarray) -> np.ndarray:
        x = (inputs - self.input_mean) / self.input_scale
        i, j = np.triu_indices(x.shape[1])
        return np.column_stack([x, x[:, i] * x[:, j]])

    def fit_arrays(self, x: np.ndarray, y: np.ndarray) -> "QuadraticModel":
        self.input_mean, self.input_scale = x.mean(axis=0), x.std(axis=0)
        self.input_scale[self.input_scale == 0] = 1.0
        super().fit_arrays(self._basis(x), y)
        return self

    def predict_inputs(self, inputs: np.ndarray) -> dict:
        return super().predict_inputs(self._basis(inputs))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, input_mean=self.input_mean,
                            input_scale=self.input_scale, mean=self.mean,
                            scale=self.scale, weights=self.weights,
                            training_frames=self.training_frames)

    @classmethod
    def load(cls, path: Path) -> "QuadraticModel":
        model = cls()
        with np.load(path, allow_pickle=False) as values:
            for key in ("input_mean", "input_scale", "mean", "scale", "weights"):
                setattr(model, key, values[key])
            model.training_frames = int(values["training_frames"])
        return model
