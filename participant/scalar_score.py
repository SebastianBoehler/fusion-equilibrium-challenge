"""Reuse frozen flux statistics; replace only independent scalar residuals."""

import json
from pathlib import Path

import numpy as np

from .artifacts import write_json
from .cache import arrays
from .score import official
from .scoring_state import decode


def compare(records: list[dict], heads: dict, frozen_flux: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    accumulators = {name: official.Accum() for name in heads}
    stats = {}
    for index, record in enumerate(records):
        receipt = json.loads((frozen_flux / f'{index:04d}-statistics.json').read_text())
        if receipt['source_identity'] != record['identity']:
            raise ValueError('Scalar comparison and frozen flux validation order differ')
        for key, value in receipt['reference_statistics'].items():
            stats[key] = stats.get(key, 0) + np.asarray(value)
        x, y = arrays(record)
        for name, head in heads.items():
            prediction = {k: v.astype(np.float32) for k, v in head.predict_scalars(x).items()}
            np.savez_compressed(out / f'{name}-{index:04d}.npz', **prediction)
            with np.load(out / f'{name}-{index:04d}.npz', allow_pickle=False) as saved:
                part = decode(receipt['prediction_statistics'])
                residuals = []
                for column, key in zip((-2, -1), ('q95', 'betaN')):
                    valid = np.isfinite(y[:, column])
                    delta = np.asarray(y[valid, column], dtype=float) - saved[key][valid].astype(float)
                    residuals.append(float(delta @ delta))
                part['ss_res_scal'] = np.asarray(residuals)
            accumulators[name].add(part)
    signs = {1.0: accumulators[next(iter(heads))].ss_res_psi[1.0],
             -1.0: accumulators[next(iter(heads))].ss_res_psi[-1.0]}
    sign = min(signs, key=signs.get)
    for acc in accumulators.values():
        acc.psi_sign = sign
    scores = {name: official.finalize_machine(acc, stats) for name, acc in accumulators.items()}
    write_json(out / 'scores.json', {'scores': scores, 'frozen_flux': str(frozen_flux),
                                   'scope': 'Only q95/betaN residuals changed; exact frozen flux metrics retained'})
    return scores
