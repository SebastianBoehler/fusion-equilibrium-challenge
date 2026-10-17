"""Measure PCA and upload precision ceilings using real frozen regression labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from participant import fast_score


def main(args):
    if args.packed.name != 'regression':
        raise ValueError('Ceiling checks use only the two existing regression groups')
    index, shots = fast_score.load_truth(args.packed)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    basis, mean = ck['basis'].numpy(), ck['psi_mean'].numpy()
    refs = fast_score.references(shots, args.refs, 8)
    results = {}
    for variant in ('perfect_float16', 'pca_float32', 'pca_float16'):
        predictions = []
        for shot in shots:
            psi = shot['psi']
            if variant.startswith('pca'):
                flat = psi.reshape(len(psi), -1).astype(np.float32)
                psi = (mean + ((flat - mean) @ basis) @ basis.T).reshape(psi.shape)
            if variant.endswith('float16'):
                psi = psi.astype(np.float16).astype(float)
            predictions.append({'psirz': psi, 'q95': shot['q95'], 'betaN': shot['betaN']})
        results[variant] = fast_score.composite(shots, refs, predictions, 8)
        print(variant, results[variant], flush=True)
    args.out.write_text(json.dumps({'shots': len(index['names']), 'scalars': 'oracle labels',
                                    'diagnostic_only': True, 'results': results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--refs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
