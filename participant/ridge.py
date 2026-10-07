"""A frozen, standardized linear baseline for DIII-D reconstruction."""

from pathlib import Path

import numpy as np

from .data import features, targets


class RidgeModel:
    def fit(self, rows: list[dict]) -> "RidgeModel":
        x = np.concatenate([features(row) for row in rows])
        y = np.concatenate([targets(row) for row in rows])
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        x = (x - self.mean) / self.scale
        self.weights = np.empty((x.shape[1] + 1, y.shape[1]))
        x = np.column_stack([x, np.ones(len(x))])
        penalty = np.eye(x.shape[1])
        penalty[-1, -1] = 0.0
        # Flux labels share one finite mask; scalar labels are masked independently.
        valid = np.isfinite(y[:, :-2]).all(axis=1)
        self.weights[:, :-2] = np.linalg.solve(
            x[valid].T @ x[valid] + penalty, x[valid].T @ y[valid, :-2])
        for index in (-2, -1):
            mask = np.isfinite(y[:, index])
            if mask.sum() < x.shape[1]:
                raise ValueError("Insufficient defined scalar training labels")
            self.weights[:, index] = np.linalg.solve(
                x[mask].T @ x[mask] + penalty, x[mask].T @ y[mask, index])
        self.training_frames = len(x)
        return self

    def predict(self, row: dict) -> dict:
        if row["source"] != "DIII-D":
            raise ValueError("This first baseline supports DIII-D only")
        x = (features(row) - self.mean) / self.scale
        y = np.column_stack([x, np.ones(len(x))]) @ self.weights
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite predictions")
        return {"psirz": y[:, :-2].reshape(-1, 65, 65),
                "q95": y[:, -2], "betaN": y[:, -1]}

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, scale=self.scale,
                            weights=self.weights, training_frames=self.training_frames)

    @classmethod
    def load(cls, path: Path) -> "RidgeModel":
        model = cls()
        with np.load(path, allow_pickle=False) as values:
            model.mean = values["mean"]
            model.scale = values["scale"]
            model.weights = values["weights"]
            model.training_frames = int(values["training_frames"])
        return model
