"""Complete float32 controls after independent recovery of an actual 031 arm."""

import argparse
from argparse import Namespace
import json
from pathlib import Path

import torch

from participant.contour_training import KAPPA_WEIGHT
from participant.critical_artifact import restore_correction
from participant.critical_training import ARMS, DESCRIPTION
from research.checkpoint_comparison import evaluate
from research.inductance_comparison import PARENT, controls, sha256, validate_frozen_weights

ROOT = Path('run_data/analysis-031')
RECOVERED = Path('run_data/critical-031')


def validate(candidate, expected_sha, arm, incumbent):
    proposal = json.loads((ROOT / 'proposal-v2/proposal-receipt.json').read_text())
    remote = json.loads((ROOT / 'remote-artifact-sha256.json').read_text())
    validation = json.loads((RECOVERED / 'critical-validation.json').read_text())
    target = json.loads((RECOVERED / 'train-critical/index.json').read_text())
    if sha256(candidate) != expected_sha or remote[f'{arm}/model.pt'] != expected_sha:
        raise ValueError('Checkpoint differs from its recovered remote hash')
    if (remote['source.tar.gz'] != proposal['source_archive_sha256']
            or sha256(RECOVERED / 'source.tar.gz') != proposal['source_archive_sha256']):
        raise ValueError('Recovered source archive differs from frozen 031')
    for name, digest in proposal['source_sha256'].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Current evaluation dependency differs from submitted source: {name}')
    if (not validation['complete'] or not validation['training_only']
            or not validation['all_usable_shapes_finiteness_weights_indices_levels_passed']
            or not validation['excluded_targets_rejected']
            or validation['target_manifest_sha256'] != proposal['split_sha256']):
        raise ValueError('Full critical target validation and exclusion checks required')
    if sha256(RECOVERED / 'critical-validation.json') != remote['critical-validation.json']:
        raise ValueError('Critical validation receipt differs from its remote hash')
    for name, digest in validation['file_sha256'].items():
        if sha256(RECOVERED / 'train-critical' / name) != digest:
            raise ValueError(f'Recovered critical target changed: {name}')
    if (target['target'] != DESCRIPTION or target['manifest_sha256'] != proposal['split_sha256']
            or target['frames'] != 1469502 or not target['complete'] or not target['training_only']):
        raise ValueError('Critical target provenance differs')
    ck = torch.load(candidate, map_location='cpu', weights_only=False)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    config, tune = ck['config'], ck['config']['critical_tune']
    required = {'parent_sha256': incumbent['checkpoints'][str(PARENT)],
                'target_manifest_sha256': proposal['split_sha256'], 'arm': arm,
                'critical_validation_sha256': remote['critical-validation.json'],
                'critical_index_sha256': validation['file_sha256']['index.json'],
                'seed': 20261031, 'epochs': 10, 'lr': 4e-6, 'batch': 2048,
                'li_weight': .05 if arm == 'critical-li-kappa' else 0,
                'kappa_weight': KAPPA_WEIGHT, 'scalar_head_frozen': True, 'parent_frozen': True,
                'target_contour': DESCRIPTION, 'training_frames': target['frames'],
                'usable_frames': validation['usable_frames'],
                'original_reference_excluded_frames': validation['original_reference_excluded_frames'],
                'unsupported_critical_events': validation['unsupported_critical_events'],
                'local_scorer_version': '3.2.0', 'movement_bound_grid_cells': 1,
                'critical_event_training_only': True,
                'level_derivative': 'live interpolation at true active event; local surrogate, not exact future closure derivative'}
    if any(tune.get(key) != value for key, value in required.items()):
        raise ValueError('Actual checkpoint differs from the frozen 031 recipe')
    smooth = config['smooth_correction']
    if (config['representation'] != 'grid' or smooth['grid'] != 17
            or not smooth['parent_frozen'] or not smooth['scalars_frozen']):
        raise ValueError('Expected the unchanged folded inference representation')
    validate_frozen_weights(parent, ck)
    native_path = RECOVERED / arm / 'correction-state.pt'
    if sha256(native_path) != remote[f'{arm}/correction-state.pt']:
        raise ValueError('Native correction differs from its recovered remote hash')
    native = torch.load(native_path, map_location='cpu', weights_only=False)
    if native['critical_tune'] != tune:
        raise ValueError('Native and folded checkpoint metadata differ')
    restore_correction(parent, native)


def main(args):
    candidate = RECOVERED / args.arm / 'model.pt'
    out = ROOT / f'{args.arm}-comparison.json'
    if out.exists():
        raise FileExistsError('Preserve existing or active scoring; do not duplicate it')
    members, incumbent = controls(candidate)
    validate(candidate, args.sha256, args.arm, incumbent)
    launch = json.loads((ROOT / 'proposal-v2/launch-receipt.json').read_text())
    scoring = Namespace(development=Path('run_data/nn-008/development'),
                        regression=Path('run_data/nn-008/regression'),
                        refs=Path('run_data/nn-008/development-refs.pkl'),
                        split=Path('run_data/full-split.json'), baseline_cache=None, out=out)
    result = evaluate(scoring, members)
    result['experiment'] = {'arm': args.arm, 'gpu_job_id': launch['gpu_job_id'],
                            'cpu_job_id': launch['cpu_job_id'], 'parent_replaced': str(PARENT),
                            'remote_local_hash_match': True,
                            'frozen_parent_and_scalar_weights_exact': True,
                            'control_members': {name: [str(path) for path in paths]
                                                for name, paths in members.items()}}
    result['complete'] = True
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--sha256', required=True)
    main(parser.parse_args())
