"""Freeze shot-disjoint splits and a nested learning curve from the complete release."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

from .artifacts import write_json


def create(tree: Path, incumbent: Path, out: Path) -> dict:
    catalog = json.loads(tree.read_text())
    previous = json.loads(incumbent.read_text())
    if len(catalog) != 7041:
        raise ValueError('Expected the verified full 7041-shot DIII-D training tree')
    lookup = {e['path']: e for e in catalog}
    used = previous['train'] + sum(previous['validation'].values(), [])
    for entry in used:
        if lookup[entry['path']] != entry:
            raise ValueError('Incumbent and full-release manifests differ')
    used_paths = {e['path'] for e in used}
    remaining = [e for e in catalog if e['path'] not in used_paths]
    random.Random(20261010).shuffle(remaining)
    development, confirmation = remaining[:256], remaining[256:384]
    training = previous['train'] + remaining[384:]
    protocol = {'revision': previous['revision'], 'selection_seed': 20261010,
                'tree_sha256': sha256(tree.read_bytes()).hexdigest(),
                'incumbent_manifest_sha256': sha256(incumbent.read_bytes()).hexdigest(),
                'train': training, 'development': development, 'locked_confirmation': confirmation,
                'regression': previous['validation'], 'screening_count': 64,
                'learning_curve': [256, 1024, 4096, len(training)],
                'normalized_lambda': 100 / 56264,
                'confirmation_policy': 'Only score the selected finalist, once, after full development comparison'}
    flat = training + development + confirmation + sum(previous['validation'].values(), [])
    if len(flat) != 7041 or len({e['path'] for e in flat}) != 7041:
        raise ValueError('Training and all held-out groups must partition the release')
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = create(args.tree, args.incumbent, args.out)
    print(json.dumps({'protocol_sha256': sha256(args.out.read_bytes()).hexdigest(),
                      'train': len(protocol['train']), 'development': len(protocol['development']),
                      'locked_confirmation': len(protocol['locked_confirmation']),
                      'learning_curve': protocol['learning_curve']}, indent=2))


if __name__ == '__main__':
    main()
