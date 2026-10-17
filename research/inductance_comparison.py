"""Compare one completed smooth-correction arm with frozen ensemble controls."""

import argparse
from argparse import Namespace
from hashlib import file_digest
import json
from pathlib import Path

import torch

from research.checkpoint_comparison import evaluate


PARENT = Path('run_data/nn-008/mlp-shape/model.pt')
RECEIPT = Path('run_data/submission-010/build-receipt.json')


def sha256(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def controls(candidate):
    receipt = json.loads(RECEIPT.read_text())
    eleven = [Path(p) for p in receipt['checkpoints']]
    if len(eleven) != 11 or eleven.count(PARENT) != 1:
        raise ValueError('Expected the exact frozen eleven-member ensemble')
    for path in eleven:
        if sha256(path) != receipt['checkpoints'][str(path)]:
            raise ValueError(f'Frozen ensemble checkpoint changed: {path}')
    ten = json.loads(Path('run_data/submission-009/build-receipt.json').read_text())
    if [str(p) for p in eleven[:10]] != list(ten['checkpoints']):
        raise ValueError('Frozen ten-member order differs from archive009')
    groups = {'candidate': [candidate]}
    for name, members in [('eight', eleven[:8]), ('ten', eleven[:10]), ('eleven', eleven)]:
        if members.count(PARENT) != 1:
            raise ValueError(f'{name} must contain the parent exactly once')
        groups[name + '_parent_replaced'] = [p for p in members if p != PARENT] + [candidate]
        groups[name + '_plus_candidate'] = [*members, candidate]
    return groups, receipt


def validate_candidate(candidate, digest, arm, receipt):
    if sha256(candidate) != digest:
        raise ValueError('Candidate differs from its recovered remote SHA256')
    checkpoint = torch.load(candidate, map_location='cpu', weights_only=False)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    tune = config['inductance_tune']
    if (tune['parent_sha256'] != receipt['checkpoints'][str(PARENT)]
            or tune['arm'] != arm or tune['epochs'] != 10 or tune['seed'] != 20261021):
        raise ValueError('Candidate does not match the approved completed arm')
    if config['representation'] != 'grid' or not tune['scalar_head_frozen']:
        raise ValueError('Expected the folded grid export with a frozen scalar head')
    validate_frozen_weights(parent, checkpoint)


def validate_frozen_weights(parent, checkpoint):
    config = checkpoint['config']
    for key in ('x_mean', 'x_std', 'psi_mean', 's_mean', 's_std'):
        if not torch.equal(parent[key], checkpoint[key]):
            raise ValueError(f'Frozen input/flux/scalar statistics changed: {key}')
    for key, value in parent['state'].items():
        exported = checkpoint['state'][key]
        if key in ('head.1.weight', 'head.1.bias'):
            value, exported = value[-parent['config']['scalars']:], exported[-config['scalars']:]
        if not torch.equal(value, exported):
            raise ValueError(f'Frozen parent or scalar weights changed: {key}')


def main(args):
    candidate = Path('run_data/inductance-020') / args.arm / 'model.pt'
    out = Path('run_data/analysis-020') / f'{args.arm}-comparison.json'
    if out.exists():
        raise FileExistsError('Preserve an existing comparison; do not duplicate scoring')
    groups, receipt = controls(candidate)
    validate_candidate(candidate, args.sha256, args.arm, receipt)
    scoring = Namespace(development=Path('run_data/nn-008/development'),
                        regression=Path('run_data/nn-008/regression'),
                        refs=Path('run_data/nn-008/development-refs.pkl'),
                        split=Path('run_data/full-split.json'), baseline_cache=None, out=out)
    result = evaluate(scoring, groups)
    result['experiment'] = {'arm': args.arm, 'job_id': 2950799,
                            'parent_replaced': str(PARENT), 'remote_local_hash_match': True,
                            'frozen_parent_and_scalar_weights_exact': True,
                            'control_members': {name: [str(p) for p in paths]
                                                for name, paths in groups.items()}}
    result['complete'] = True
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('pixel', 'pixel-li'), required=True)
    parser.add_argument('--sha256', required=True)
    main(parser.parse_args())
