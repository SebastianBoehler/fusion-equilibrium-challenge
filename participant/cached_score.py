"""Stream saved predictions through the unmodified organizer metric functions."""

from pathlib import Path

import numpy as np

from .artifacts import write_json
from .cache import arrays
from .reference_cache import geometry, reference, scorer_identity, shot
from .score import official


def score_models(records: list[dict], models: dict, cache: Path, out: Path) -> dict:
    if not records or not models:
        raise ValueError('Nonempty validation and model mapping required')
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for index, record in enumerate(records):
        _, part = reference(record, cache)
        for name, value in part.items():
            stats[name] = stats.get(name, 0) + value
        x, _ = arrays(record)
        for name, model in models.items():
            prediction = model.predict_inputs(x)
            prediction = {k: v.astype(np.float16 if k == 'psirz' else np.float32)
                          for k, v in prediction.items()}
            if not all(np.isfinite(v).all() for v in prediction.values()):
                raise ValueError(f'Nonfinite exported prediction: {name}')
            directory = out / name
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory / f'{index:04d}.npz', **prediction)
        if (index + 1) % 16 == 0:
            print(f'References and saved predictions: {index + 1}/{len(records)}', flush=True)
    means = (stats['psi_sum'] / stats['psi_n'],
             np.divide(stats['scal_sum'], stats['scal_n'], out=np.zeros(2), where=stats['scal_n'] > 0),
             np.divide(stats['cons_sum'], stats['cons_n'], out=np.zeros(7), where=stats['cons_n'] > 0))
    totals = {name: {s: 0.0 for s in official.PSI_SIGNS} for name in models}
    for index, record in enumerate(records):
        gt = shot(record)
        for name in models:
            with np.load(out / name / f'{index:04d}.npz', allow_pickle=False) as pred:
                residuals, _, _ = official.psi_residuals(gt['psi'], pred['psirz'], means[0])
            for sign, value in residuals.items():
                totals[name][sign] += value
    accumulators = {name: official.Accum() for name in models}
    for name, acc in accumulators.items():
        acc.psi_sign = min(totals[name], key=totals[name].get)
    r, z, coarse, mask_f = geometry()
    for index, record in enumerate(records):
        gt, (ref, _) = shot(record), reference(record, cache)
        for name, acc in accumulators.items():
            with np.load(out / name / f'{index:04d}.npz', allow_pickle=False) as data:
                prediction = {k: data[k] for k in data.files}
            part = official.score_shot(gt, ref, prediction, r, z, coarse, mask_f, acc.psi_sign, means)
            acc.add(part)
            write_json(out / name / f'{index:04d}-statistics.json',
                       {'source_identity': record['identity'],
                        'reference_statistics': {k: np.asarray(v).tolist() for k, v in
                                                 reference(record, cache)[1].items()},
                        'prediction_statistics': {k: v.tolist() if isinstance(v, np.ndarray) else v
                                                  for k, v in part.items()}})
        if (index + 1) % 16 == 0:
            print(f'Official scoring: {index + 1}/{len(records)} shots', flush=True)
    scores = {name: official.finalize_machine(acc, stats) for name, acc in accumulators.items()}
    write_json(out / 'scores.json', {'scorer_sha256': scorer_identity(), 'scores': scores,
                                   'precision': {'psirz': 'float16', 'scalars': 'float32'}})
    return scores
