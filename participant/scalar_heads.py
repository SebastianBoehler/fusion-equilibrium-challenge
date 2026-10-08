"""Independent scalar predictors attached to one unchanged kinetic flux model."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from .cache import arrays
from .kinetic import KineticModel


def training_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError('Nonempty scalar training records required')
    frames = sum(record['frames'] for record in records)
    x = np.empty((frames, 33), dtype=np.float64)
    y = np.empty((frames, 2), dtype=np.float64)
    start = 0
    for record in records:
        inputs, targets = arrays(record)
        try:
            stop = start + record['frames']
            x[start:stop] = inputs
            y[start:stop] = targets[:, -2:]
            start = stop
        finally:
            inputs._mmap.close()
            targets._mmap.close()
    return x, y


class ScalarHeads:
    def __init__(self, flux: KineticModel, method: str):
        self.flux, self.method = flux, method
        self.heads = []

    def fit(self, records: list[dict]) -> 'ScalarHeads':
        if self.method not in ('extra-trees', 'gradient-boosting'):
            raise ValueError('Unknown scalar method')
        x, y = training_arrays(records)
        for index in range(2):
            valid = np.isfinite(y[:, index])
            if valid.sum() < 100:
                raise ValueError('Insufficient finite scalar training labels')
            if self.method == 'extra-trees':
                model = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=8,
                                            max_features=0.8, n_jobs=1, random_state=20261010)
            else:
                model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=31,
                                                       learning_rate=0.1, early_stopping=False,
                                                       random_state=20261010)
            self.heads.append(model.fit(x[valid], y[valid, index]))
        self.label_counts = np.isfinite(y).sum(axis=0).tolist()
        self.training_frames = len(x)
        return self

    def predict_scalars(self, x: np.ndarray) -> dict:
        if self.method == 'ridge':
            prediction = self.flux.predict_inputs(x)
            return {key: prediction[key] for key in ('q95', 'betaN')}
        predictions = {key: head.predict(x) for key, head in zip(('q95', 'betaN'), self.heads, strict=True)}
        if not all(np.isfinite(v).all() for v in predictions.values()):
            raise ValueError('Nonfinite scalar prediction')
        return predictions

    def predict_inputs(self, x: np.ndarray) -> dict:
        prediction = self.flux.predict_inputs(x)
        prediction.update(self.predict_scalars(x))
        return prediction

    def save(self, path: Path) -> None:
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> 'ScalarHeads':
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ValueError('Wrong scalar-head checkpoint')
        return model
