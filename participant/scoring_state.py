"""Seal scoring inputs and restore exact per-shot organizer statistics."""

import json
from pathlib import Path

import numpy as np

from .artifacts import digest, identity, write_json


def decode(part: dict) -> dict:
    value = dict(part)
    for key in ('ss_res_psi', 'shot_r2_psi'):
        value[key] = {float(k): v for k, v in value[key].items()}
    for key in ('ss_res_scal', 'ss_res_cons', 'cons_frames', 'cons_fails'):
        value[key] = np.asarray(value[key])
    return value


def seal(out: Path, records: list[dict], models: dict, means: tuple,
         signs: dict, scorer: str) -> str:
    context = {'scorer_sha256': scorer,
               'sources': [r['identity'] for r in records],
               'targets': [r['files']['targets'] for r in records],
               'means': [np.asarray(v).tolist() for v in means], 'signs': signs,
               'predictions': {name: [digest(out / name / f'{i:04d}.npz')
                                      for i in range(len(records))] for name in models}}
    path = out / 'scoring-context.json'
    if path.exists() and json.loads(path.read_text()) != context:
        raise ValueError('Scoring context changed: refuse to reuse statistics')
    if not path.exists():
        write_json(path, context)
    return identity(context)


def restore(path: Path, record: dict, reference_stats: dict, context: str,
            prediction: Path, imported: dict) -> dict | None:
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if 'statistics_sha256' in receipt:
        payload = {k: v for k, v in receipt.items() if k != 'statistics_sha256'}
        if receipt['statistics_sha256'] != identity(payload):
            raise ValueError('Scoring statistics content changed')
    elif receipt.get('context_sha256') is not None:
        raise ValueError('Sealed scoring statistics have no content identity')
    if receipt['source_identity'] != record['identity']:
        raise ValueError('Scoring statistics source/order mismatch')
    expected = {k: np.asarray(v).tolist() for k, v in reference_stats.items()}
    if receipt['reference_statistics'] != expected:
        raise ValueError('Scoring statistics reference mismatch')
    prediction_hash = digest(prediction)
    if receipt.get('context_sha256') != context:
        key = f'{path.parent.name}/{path.name}'
        if receipt.get('context_sha256') is not None or imported.get(key) != digest(path):
            raise ValueError('Unverified scoring statistics import')
        receipt.update(context_sha256=context, prediction_sha256=prediction_hash)
        receipt['statistics_sha256'] = identity(receipt)
        write_json(path, receipt)
    if receipt['prediction_sha256'] != prediction_hash:
        raise ValueError('Scored prediction changed')
    return decode(receipt['prediction_statistics'])


def save(path: Path, record: dict, reference_stats: dict, part: dict,
         context: str, prediction: Path) -> None:
    receipt = {'source_identity': record['identity'],
                     'context_sha256': context, 'prediction_sha256': digest(prediction),
                     'reference_statistics': {k: np.asarray(v).tolist()
                                              for k, v in reference_stats.items()},
                     'prediction_statistics': {k: v.tolist() if isinstance(v, np.ndarray) else v
                                               for k, v in part.items()}}
    receipt['statistics_sha256'] = identity(receipt)
    write_json(path, receipt)
