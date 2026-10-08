"""Compare breadth finalists on full development, then isolate scalar heads."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .artifacts import code_digest, digest, write_json
from .breadth_pilot import prepare
from .cache import arrays, feature_identity
from .cached_score import score_models
from .kinetic import KineticModel
from .reference_cache import scorer_identity
from .scalar_heads import ScalarHeads
from .scalar_score import compare
from .scoring_import import import_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.manifest.read_text())
    pilot = json.loads((args.pilot / 'scientific-result.json').read_text())
    if pilot['status'] != 'completed' or pilot['manifest_sha256'] != digest(args.manifest):
        raise ValueError('Completed matching breadth pilot required')
    # The frozen pilot scorer identity includes the adapter's absolute source path.
    # Verify its bytes and reconstruct that exact identity across checkout paths.
    original_adapter = args.pilot.parent / 'repo/participant/reference_cache.py'
    if digest(original_adapter) != digest(Path('participant/reference_cache.py')):
        raise ValueError('Reference adapter changed since frozen breadth pilot')
    vendor = Path('vendor/fusion_starter')
    original_scorer = code_digest([vendor / 'local_score.py',
                                  *sorted((vendor / 'fusion_scoring').glob('*.py')),
                                  vendor / 'fusion_scoring/masks/d3d_envelope.npz', original_adapter])
    if pilot['feature_sha256'] != feature_identity() or pilot['scorer_sha256'] != original_scorer:
        raise ValueError('Feature/scorer code changed since frozen breadth pilot')
    args.out.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, receipt in pilot['checkpoints'].items():
        path = args.pilot / f'{name}.npz'
        if digest(path) != receipt['sha256']:
            raise ValueError(f'Pilot checkpoint hash mismatch: {name}')
        models[name] = KineticModel.load(path)
    records = prepare(protocol['development'], protocol['revision'], args.raw, args.cache / 'features')
    imported = None
    if args.resume_from is not None:
        imported = import_state(args.resume_from, records, models, args.cache / 'references',
                                args.out / 'breadth', args.manifest)
    breadth_scores = score_models(records, models, args.cache / 'references', args.out / 'breadth')
    flux_name = max(breadth_scores, key=lambda name: breadth_scores[name]['S'])
    heads = {'ridge': ScalarHeads(models[flux_name], 'ridge')}
    training = prepare(protocol['train'][:1024], protocol['revision'], args.raw, args.cache / 'features')
    for method in ('extra-trees', 'gradient-boosting'):
        heads[method] = ScalarHeads(models[flux_name], method).fit(training)
    checkpoint_receipts = {}
    for method, head in heads.items():
        path = args.out / f'{method}.joblib'
        head.save(path)
        reloaded = ScalarHeads.load(path)
        x, _ = arrays(records[0])
        for key, value in head.predict_inputs(x).items():
            np.testing.assert_array_equal(reloaded.predict_inputs(x)[key], value)
        heads[method] = reloaded
        checkpoint_receipts[method] = {'sha256': digest(path), 'flux_checkpoint': flux_name,
                                       'scalar_training_frames': getattr(head, 'training_frames',
                                                                         head.flux.training_frames),
                                       'label_counts': getattr(head, 'label_counts', None)}
    scalar_scores = compare(records, heads, args.out / 'breadth' / flux_name, args.out / 'scalars')
    if scalar_scores['ridge'] != breadth_scores[flux_name]:
        raise ValueError('Frozen-flux scalar baseline failed exact score reproduction')
    best = max(scalar_scores, key=lambda name: scalar_scores[name]['S'])
    regression_scores = {}
    for group, entries in protocol['regression'].items():
        regression = prepare(entries, protocol['revision'], args.raw, args.cache / 'features')
        regression_scores[group] = score_models(regression,
                                                {'incumbent': models['incumbent-256'], 'candidate': heads[best]},
                                                args.cache / 'references', args.out / 'regression' / group)
    difference = scalar_scores[best]['S'] - breadth_scores['incumbent-256']['S']
    improves_regression = all(s['candidate']['S'] > s['incumbent']['S']
                             for s in regression_scores.values())
    # A larger untouched confirmation group is reserved for the final method decision.
    eligible = difference >= 0.002 and improves_regression
    result = {'status': 'completed', 'job_id': os.environ.get('SLURM_JOB_ID'), 'node': platform.node(),
              'manifest_sha256': digest(args.manifest), 'pilot_result_sha256': digest(args.pilot / 'scientific-result.json'),
              'feature_sha256': feature_identity(), 'scorer_sha256': scorer_identity(),
              'breadth_scores': breadth_scores, 'scalar_scores': scalar_scores,
              'flux_choice': flux_name, 'scalar_choice': best, 'checkpoints': checkpoint_receipts,
              'regression_scores': regression_scores, 'development_improvement': difference,
              'eligible_for_development_submission': eligible,
              'eligibility_rule': 'Full development S gain >= 0.002 and both old regression groups improve',
              'locked_confirmation_read': 0, 'checkpoint_reload': True,
              'resumed_from_job_id': imported['previous_job_id'] if imported else None,
              'scoring_reuse': json.loads((args.out / 'breadth/scores.json').read_text()),
              'elapsed_seconds': time.monotonic() - started,
              'peak_rss_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'peak_rss_unit': 'KiB' if platform.system() == 'Linux' else 'bytes',
              'scope': 'DIII-D training-release only; full 256-shot development; fixed-flux scalar comparison'}
    write_json(args.out / 'scientific-result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
