"""Compare a recovered024 arm against all frozen ensemble controls at float32."""

import argparse
from argparse import Namespace
import json
from pathlib import Path

import torch

from participant.contour_training import ARMS, KAPPA_WEIGHT
from participant.scored_contour_targets import DESCRIPTION
from research.checkpoint_comparison import evaluate
from research.inductance_comparison import PARENT, controls, sha256, validate_frozen_weights

ROOT = Path('run_data/analysis-024')
RECOVERED = Path('run_data/contour-024')


def validate(candidate, expected_sha, arm, incumbent):
    if sha256(candidate) != expected_sha:
        raise ValueError('Recovered checkpoint differs from the remote SHA256')
    proposal = json.loads((ROOT / 'proposal/proposal-receipt.json').read_text())
    target = json.loads((RECOVERED / 'target-validation.json').read_text())
    if not target['complete'] or not target['training_only'] or not target['all_usable_targets_checked']:
        raise ValueError('Completed full-corpus target validation required')
    if target['manifest_sha256'] != proposal['split_sha256'] or target['target'] != DESCRIPTION:
        raise ValueError('Scorer target provenance mismatch')
    ck = torch.load(candidate, map_location='cpu', weights_only=False)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    config = ck['config']
    tune = config['inductance_tune']
    required = {'parent_sha256': incumbent['checkpoints'][str(PARENT)],
                'target_manifest_sha256': proposal['split_sha256'], 'arm': arm,
                'seed': 20261024, 'epochs': 10, 'lr': 4e-6, 'li_weight': 0.05,
                'kappa_weight': KAPPA_WEIGHT if arm == 'aligned-li-kappa' else 0.0,
                'batch': 2048, 'scalar_head_frozen': True, 'target_contour': DESCRIPTION,
                'training_frames': target['frames'], 'usable_frames': target['usable_frames'],
                'local_scorer_version': '3.2.0', 'normal_movement_bound_grid_cells': 1}
    if any(tune.get(k) != value for k, value in required.items()):
        raise ValueError('Candidate does not match the frozen024 recipe')
    if config['representation'] != 'grid' or not config['smooth_correction']['parent_frozen']:
        raise ValueError('Expected frozen-parent folded grid export')
    validate_frozen_weights(parent, ck)


def main(args):
    candidate = RECOVERED / args.arm / 'model.pt'
    out = ROOT / f'{args.arm}-comparison.json'
    if out.exists():
        raise FileExistsError('Preserve existing or active scoring; do not duplicate it')
    members, incumbent = controls(candidate)
    validate(candidate, args.sha256, args.arm, incumbent)
    scoring = Namespace(development=Path('run_data/nn-008/development'),
                        regression=Path('run_data/nn-008/regression'),
                        refs=Path('run_data/nn-008/development-refs.pkl'),
                        split=Path('run_data/full-split.json'), baseline_cache=None, out=out)
    result = evaluate(scoring, members)
    result['experiment'] = {'arm': args.arm, 'gpu_job_id': 2950821, 'cpu_job_id': 2950820,
                            'parent_replaced': str(PARENT), 'remote_local_hash_match': True,
                            'frozen_parent_and_scalar_weights_exact': True,
                            'control_members': {name: [str(p) for p in paths]
                                                for name, paths in members.items()}}
    result['complete'] = True
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--sha256', required=True)
    main(parser.parse_args())
