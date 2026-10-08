"""Relocate verified legacy scoring artifacts without changing frozen jobs."""

import json
from pathlib import Path
import shutil
import tarfile
from hashlib import sha256

from .artifacts import code_digest, digest, identity, write_json
from .reference_cache import scorer_identity
from .score import official


def import_state(previous: Path, records: list[dict], models: dict,
                 cache: Path, out: Path, manifest: Path) -> dict:
    if out.exists():
        raise ValueError('Legacy scoring import requires a fresh destination')
    spec = json.loads((previous / 'spec.json').read_text())
    outcome = json.loads((previous / 'outcome.json').read_text())
    if outcome['status'] != 'failed' or 'TIMEOUT' not in outcome['error']:
        raise ValueError('A frozen timed-out development run is required')
    for key in ('source_sha256', 'script_sha256', 'submission_nonce', 'task_id'):
        if outcome[key] != spec[key]:
            raise ValueError('Previous controller/spec identity mismatch')
    if digest(previous / 'source.tar.gz') != spec['source_sha256']:
        raise ValueError('Previous source archive changed')
    if digest(previous / 'job.sbatch') != spec['script_sha256']:
        raise ValueError('Previous batch changed')
    batch = (previous / 'job.sbatch').read_text()
    if manifest.name != 'data.json' or f'pilot={manifest.parent}\n' not in batch or \
            '--manifest "$pilot/data.json" --pilot "$pilot/result"' not in batch:
        raise ValueError('Previous scoring manifest/pilot arguments differ')
    expected_predictions = {f'{i:04d}.npz' for i in range(len(records))}
    for name in models:
        saved = {p.name for p in (previous / 'result/breadth' / name).glob('*.npz')}
        if saved != expected_predictions:
            raise ValueError('Previous full scoring population differs')
    vendor = Path('vendor/fusion_starter')
    scorer_files = [vendor / 'local_score.py', *sorted((vendor / 'fusion_scoring').glob('*.py')),
                    vendor / 'fusion_scoring/masks/d3d_envelope.npz']
    unchanged = [*scorer_files, Path('participant/reference_cache.py'),
                 *(Path('participant') / f for f in ('data.py', 'kinetic.py', 'quadratic.py',
                                                    'ridge.py', 'cache.py', 'artifacts.py', 'score.py'))]
    with tarfile.open(previous / 'source.tar.gz') as archive:
        members = {m.name.removeprefix('./'): m for m in archive.getmembers() if m.isfile()}
        for path in [*unchanged, Path('participant/cached_score.py'),
                     Path('participant/development_compare.py')]:
            archived = archive.extractfile(members[str(path)]).read()
            if sha256(archived).hexdigest() != digest(previous / 'repo' / path):
                raise ValueError(f'Frozen extracted source changed: {path}')
        for path in unchanged:
            if digest(previous / 'repo' / path) != digest(path):
                raise ValueError(f'Prediction/reference implementation changed: {path}')
    original_adapter = previous / 'repo/participant/reference_cache.py'
    old_scorer = code_digest([*scorer_files, original_adapter])
    new_scorer = scorer_identity()
    out.mkdir(parents=True, exist_ok=True)
    imported, reference_count, prediction_count = {}, 0, 0
    for index, record in enumerate(records):
        provenance = {'source': record['identity'], 'targets': record['files']['targets'],
                      'scorer_sha256': old_scorer, 'scoring_version': official.SCORING_VERSION}
        source = previous / 'cache/references' / identity(provenance)
        receipt = json.loads((source / 'receipt.json').read_text())
        if receipt['identity'] != identity(provenance) or receipt['provenance'] != provenance:
            raise ValueError('Previous reference provenance mismatch')
        if digest(source / 'reference.npz') != receipt['file_sha256']:
            raise ValueError('Previous reference content changed')
        relocated = {**provenance, 'scorer_sha256': new_scorer}
        destination = cache / identity(relocated)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / 'reference.npz', destination / 'reference.npz')
        if digest(destination / 'reference.npz') != receipt['file_sha256']:
            raise ValueError('Reference copy changed content')
        write_json(destination / 'receipt.json', {'identity': identity(relocated),
                   'provenance': relocated, 'file_sha256': receipt['file_sha256']})
        reference_count += 1
        for name in models:
            target = out / name
            target.mkdir(exist_ok=True)
            prediction = previous / 'result/breadth' / name / f'{index:04d}.npz'
            shutil.copyfile(prediction, target / prediction.name)
            if digest(prediction) != digest(target / prediction.name):
                raise ValueError('Prediction copy changed content')
            prediction_count += 1
            statistic = prediction.with_name(f'{index:04d}-statistics.json')
            if statistic.exists():
                value = json.loads(statistic.read_text())
                if value['source_identity'] != record['identity'] or 'context_sha256' in value:
                    raise ValueError('Wrong legacy scoring statistic identity/format')
                shutil.copyfile(statistic, target / statistic.name)
                imported[f'{name}/{statistic.name}'] = digest(statistic)
    receipt = {'previous_job_id': json.loads((previous / 'identity.json').read_text())['job_id'],
               'previous_source_sha256': spec['source_sha256'], 'manifest_sha256': digest(manifest),
               'old_scorer_sha256': old_scorer, 'new_scorer_sha256': new_scorer,
               'references': reference_count, 'predictions': prediction_count,
               'statistics': imported,
               'guard': 'Unchanged prediction/reference code; every saved prediction is rechecked against its model'}
    write_json(out / 'imported-state.json', receipt)
    print(f'Imported {reference_count} references, {prediction_count} predictions, '
          f'{len(imported)} verified legacy statistics', flush=True)
    return receipt
