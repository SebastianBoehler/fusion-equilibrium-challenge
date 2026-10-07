"""A frozen 256-to-1024-shot learning curve with two regularization controls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .artifacts import digest, write_json
from .block_ridge import fit_statistics, solve
from .cache import arrays, feature_identity, prepare_shot
from .cached_score import score_models
from .kinetic import KineticModel
from .reference_cache import scorer_identity


def prepare(entries, revision, raw, cache):
    def one(entry):
        return prepare_shot(entry, revision, raw, cache)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(one, entries):
            records.append(record)
            if len(records) % 32 == 0 or len(records) == len(entries):
                print(f'Feature cache: {len(records)}/{len(entries)} shots', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.manifest.read_text())
    train_entries = protocol['train'][:1024]
    development = protocol['development'][:protocol['screening_count']]
    regression = sum(protocol['regression'].values(), [])
    all_paths = [e['path'] for e in train_entries + protocol['development'] +
                 protocol['locked_confirmation'] + regression]
    if len(all_paths) != len(set(all_paths)) or len(train_entries) != 1024:
        raise ValueError('Disjoint frozen training/validation shots required')
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = {'job_id': os.environ.get('SLURM_JOB_ID'), 'node': platform.node(),
               'manifest_sha256': digest(args.manifest), 'feature_sha256': feature_identity(),
               'scorer_sha256': scorer_identity(), 'incumbent_sha256': digest(args.incumbent),
               'hypothesis': 'More shot coverage improves the fixed kinetic basis independently of alpha/N',
               'selection_rule': 'Screen only; no promotion before full 256-shot development and locked confirmation',
               'shot_counts': {'train': 1024, 'development_screen': len(development),
                               'regression': len(regression), 'locked_confirmation_read': 0}}
    write_json(args.out / 'runtime.json', runtime)
    records = prepare(train_entries, protocol['revision'], args.raw, args.cache / 'features')
    regression_records = prepare(regression, protocol['revision'], args.raw, args.cache / 'features')
    incumbent = KineticModel.load(args.incumbent)
    normalizer, sufficient = fit_statistics(records[:256])
    reproduced = solve(normalizer, sufficient, 100)
    if reproduced.training_frames != 56264:
        raise ValueError('Incumbent training frame count differs')
    for record in regression_records:
        x, _ = arrays(record)
        a, b = incumbent.predict_inputs(x), reproduced.predict_inputs(x)
        for name in a:
            np.testing.assert_allclose(b[name], a[name], rtol=1e-7, atol=1e-9)
    regression_scores = score_models(regression_records, {'incumbent': incumbent, 'reproduced': reproduced},
                                    args.cache / 'references', args.out / 'regression')
    if regression_scores['incumbent'] != regression_scores['reproduced']:
        raise ValueError('Exported incumbent regression scores differ after blocked fitting')
    normalizer, sufficient = fit_statistics(records)
    models = {'incumbent-256': incumbent,
              'breadth-1024-fixed': solve(normalizer, sufficient, 100),
              'breadth-1024-normalized': solve(normalizer, sufficient,
                                               protocol['normalized_lambda'] * normalizer.training_frames)}
    checkpoints = {}
    for name, model in models.items():
        path = args.out / f'{name}.npz'
        model.save(path)
        checkpoints[name] = {'sha256': digest(path), 'training_frames': model.training_frames,
                             'alpha': float(model.regularization) if name != 'incumbent-256' else 100,
                             'lambda': (float(model.regularization) if name != 'incumbent-256' else 100)
                                       / model.training_frames}
        reloaded = KineticModel.load(path)
        x, _ = arrays(records[0])
        for key, value in model.predict_inputs(x).items():
            np.testing.assert_array_equal(reloaded.predict_inputs(x)[key], value)
        models[name] = reloaded
    heldout = prepare(development, protocol['revision'], args.raw, args.cache / 'features')
    scores = score_models(heldout, models, args.cache / 'references', args.out / 'development-screen')
    result = {**runtime, 'status': 'completed', 'regression_scores': regression_scores,
              'scores': scores, 'checkpoints': checkpoints, 'checkpoint_reload': True,
              'raw_target_redaction': True, 'lossless_target_storage': True,
              'peak_rss_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'peak_rss_unit': 'KiB' if platform.system() == 'Linux' else 'bytes',
              'elapsed_seconds': time.monotonic() - started,
              'scope': 'Training-release screening only; no public-test fitting; confirmation remains locked'}
    write_json(args.out / 'scientific-result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
