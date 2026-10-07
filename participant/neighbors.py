"""Five-neighbor baseline fitted only on released DIII-D training frames."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .data import features, targets


class NeighborModel:
    def fit(self, rows: list[dict]) -> "NeighborModel":
        x = np.concatenate([features(row) for row in rows])
        y = np.concatenate([targets(row) for row in rows])
        valid = np.isfinite(y).all(axis=1)
        x, self.y = x[valid], y[valid]
        if len(x) < 5:
            raise ValueError("Five-neighbor model needs five fully defined training frames")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.x = (x - self.mean) / self.scale
        self.tree = cKDTree(self.x)
        return self

    def predict(self, row: dict) -> dict:
        if row["source"] != "DIII-D":
            raise ValueError("The neighbor baseline supports DIII-D only")
        x = (features(row) - self.mean) / self.scale
        distances, indices = self.tree.query(x, k=5, workers=2)
        weights = 1 / np.maximum(distances, 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        y = np.einsum("nk,nkd->nd", weights, self.y[indices])
        if not np.isfinite(y).all():
            raise ValueError("Nonfinite neighbor predictions")
        return {"psirz": y[:, :-2].reshape(-1, 65, 65),
                "q95": y[:, -2], "betaN": y[:, -1]}

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, scale=self.scale, x=self.x, y=self.y)

    @classmethod
    def load(cls, path: Path) -> "NeighborModel":
        model = cls()
        with np.load(path, allow_pickle=False) as values:
            model.mean, model.scale = values["mean"], values["scale"]
            model.x, model.y = values["x"], values["y"]
        model.tree = cKDTree(model.x)
        return model
