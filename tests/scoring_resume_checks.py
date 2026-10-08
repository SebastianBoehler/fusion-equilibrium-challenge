"""Exercise interruption, sealed state, and legacy relocation with real released shots."""

import json
from pathlib import Path
import shutil
import tarfile
from unittest.mock import patch

from participant.artifacts import code_digest, digest, identity, write_json
from participant.cached_score import score_models
from participant.reference_cache import scorer_identity
from participant.scoring_import import import_state


def rejects(operation):
    try:
        operation()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Invalid scoring state was accepted')


def verify(records, model, root, original):
    models = {'blocked': model}
    cache, out = root / 'references', root / 'scores'
    last = out / 'blocked' / f'{len(records) - 1:04d}-statistics.json'
    last.unlink()
    from participant.score import official
    scorer = official.score_shot
    with patch('participant.reference_cache.official.build_reference',
               side_effect=AssertionError('Unexpected reference rebuild')):
        with patch('participant.cached_score.official.score_shot', wraps=scorer) as calls:
            assert score_models(records, models, cache, out) == original
            assert calls.call_count == 1
        with patch('participant.cached_score.official.score_shot',
                   side_effect=AssertionError('Completed scoring repeated')):
            assert score_models(records, models, cache, out) == original
    receipt = json.loads((out / 'scores.json').read_text())
    assert receipt['reused_statistics'] == len(records) and receipt['computed_statistics'] == 0
    rejects(lambda: score_models(list(reversed(records)), models, cache, out))
    clean = last.read_bytes()
    changed = json.loads(clean)
    changed['prediction_statistics']['lcfs_frames'] += 1
    write_json(last, changed)
    rejects(lambda: score_models(records, models, cache, out))
    last.write_bytes(clean)
    verify_import(records, models, cache, out, root, original)


def verify_import(records, models, cache, out, root, original):
    previous = root / 'frozen'
    manifest = root / 'pilot/data.json'
    manifest.parent.mkdir()
    shutil.copyfile('protocols/smoke-001-data.json', manifest)
    (previous / 'repo').mkdir(parents=True)
    vendor = Path('vendor/fusion_starter')
    scorer_files = [vendor / 'local_score.py', *sorted((vendor / 'fusion_scoring').glob('*.py')),
                    vendor / 'fusion_scoring/masks/d3d_envelope.npz']
    files = [*scorer_files, *(Path('participant') / f for f in
             ('reference_cache.py', 'data.py', 'kinetic.py', 'quadratic.py', 'ridge.py',
              'cache.py', 'artifacts.py', 'score.py', 'cached_score.py', 'development_compare.py'))]
    with tarfile.open(previous / 'source.tar.gz', 'w:gz') as archive:
        for path in files:
            destination = previous / 'repo' / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            archive.add(path, arcname=str(path))
    # Metadata represents a terminated fixture; all numerical artifacts are real.
    (previous / 'job.sbatch').write_text(
        f'pilot={manifest.parent}\n--manifest "$pilot/data.json" --pilot "$pilot/result"\n')
    spec = {'source_sha256': digest(previous / 'source.tar.gz'),
            'script_sha256': digest(previous / 'job.sbatch'),
            'task_id': 'real-shot-resume-test', 'submission_nonce': 'test'}
    write_json(previous / 'spec.json', spec)
    write_json(previous / 'outcome.json', {**spec, 'status': 'failed', 'error': 'TIMEOUT'})
    write_json(previous / 'identity.json', {'job_id': 'test'})
    old_scorer = code_digest([*scorer_files, previous / 'repo/participant/reference_cache.py'])
    from participant.score import official
    for record in records:
        provenance = {'source': record['identity'], 'targets': record['files']['targets'],
                      'scorer_sha256': scorer_identity(), 'scoring_version': official.SCORING_VERSION}
        source = cache / identity(provenance)
        provenance['scorer_sha256'] = old_scorer
        target = previous / 'cache/references' / identity(provenance)
        target.mkdir(parents=True)
        shutil.copyfile(source / 'reference.npz', target / 'reference.npz')
        write_json(target / 'receipt.json', {'identity': identity(provenance),
                   'provenance': provenance, 'file_sha256': digest(target / 'reference.npz')})
    shutil.copytree(out / 'blocked', previous / 'result/breadth/blocked')
    for path in (previous / 'result/breadth/blocked').glob('*-statistics.json'):
        value = json.loads(path.read_text())
        for key in ('context_sha256', 'prediction_sha256', 'statistics_sha256'):
            value.pop(key)
        write_json(path, value)
    new_cache, new_out = root / 'relocated', root / 'imported'
    imported = import_state(previous, records, models, new_cache, new_out, manifest)
    assert len(imported['statistics']) == len(records)
    with patch('participant.reference_cache.official.build_reference',
               side_effect=AssertionError('Relocated reference rebuilt')):
        with patch('participant.cached_score.official.score_shot',
                   side_effect=AssertionError('Imported scoring repeated')):
            assert score_models(records, models, new_cache, new_out) == original
    context = json.loads((out / 'scoring-context.json').read_text())
    context['scorer_sha256'] = old_scorer
    write_json(previous / 'result/breadth/scoring-context.json', context)
    write_json(previous / 'result/breadth/scores.json',
               {'context_sha256': identity(context), 'scorer_sha256': old_scorer, 'scores': original})
    write_json(previous / 'outcome.json', {**spec, 'status': 'failed', 'error': 'FAILED exit=1:0'})
    for path in (out / 'blocked').glob('*-statistics.json'):
        payload = {k: v for k, v in json.loads(path.read_text()).items() if k != 'statistics_sha256'}
        payload['context_sha256'] = identity(context)
        write_json(previous / 'result/breadth/blocked' / path.name,
                   {**payload, 'statistics_sha256': identity(payload)})
    sealed_out = root / 'sealed-import'
    imported = import_state(previous, records, models, new_cache, sealed_out, manifest)
    assert imported['complete_breadth'] and len(imported['statistics']) == len(records)
    with patch('participant.cached_score.official.score_shot',
               side_effect=AssertionError('Complete scoring repeated')):
        assert score_models(records, models, new_cache, sealed_out) == original
    bad_statistic = previous / 'result/breadth/blocked/0000-statistics.json'
    clean = bad_statistic.read_bytes()
    value = json.loads(clean)
    value['prediction_statistics']['lcfs_frames'] += 1
    write_json(bad_statistic, value)
    rejects(lambda: import_state(previous, records, models, new_cache, root / 'bad-sealed', manifest))
    bad_statistic.write_bytes(clean)
    rejects(lambda: import_state(previous, records[:1], models, new_cache, root / 'wrong-count', manifest))
    frozen = previous / 'repo/participant/reference_cache.py'
    frozen.write_text(frozen.read_text() + '\n# damaged fixture\n')
    rejects(lambda: import_state(previous, records, models, new_cache, root / 'wrong-source', manifest))
