"""Score identical frozen ensemble predictions before and after float16 rounding."""

import argparse
import json
from pathlib import Path

import numpy as np

from participant import fast_score
from participant.nn_submit import Predictor


def main(args):
    checkpoints = [Path(p) for p in json.loads(args.receipt.read_text())['checkpoints']]
    report = {}
    for name in ('development', 'regression'):
        packed = args.root / name
        index, shots = fast_score.load_truth(packed)
        refs_path = args.root / 'development-refs.pkl' if name == 'development' else args.out.parent / 'regression-refs.pkl'
        refs = fast_score.references(shots, refs_path, 12)
        if name == 'development':
            psi = np.load(args.cache / 'psi-pred.npy', mmap_mode='r')
            scalar = np.load(args.cache / 'scalar-pred.npy', mmap_mode='r')
        else:
            model = Predictor(checkpoints)
            x = np.load(packed / 'x.npy', mmap_mode='r')
            outputs = [model.predict_inputs(np.asarray(x[a:b])) for a, b in
                       zip(index['offsets'][:-1], index['offsets'][1:], strict=True)]
            psi = np.concatenate([p['psirz'] for p in outputs])
            scalar = np.concatenate([np.column_stack((p['q95'], p['betaN'])) for p in outputs])
        groups = {'all': list(range(len(shots)))}
        if name == 'regression':
            entries = json.loads(args.split.read_text())['regression']
            groups = {k: [index['names'].index(Path(e['path']).stem) for e in values]
                      for k, values in entries.items()}
        for group, positions in groups.items():
            predictions = []
            for i in positions:
                a, b = index['offsets'][i:i + 2]
                predictions.append({'psirz': np.asarray(psi[a:b], dtype=float),
                                    'q95': np.asarray(scalar[a:b, 0], dtype=float),
                                    'betaN': np.asarray(scalar[a:b, 1], dtype=float)})
            for dtype in ('float16', 'float32'):
                uploaded = [p | {'psirz': p['psirz'].astype(dtype).astype(float)} for p in predictions]
                score = fast_score.composite([shots[i] for i in positions], [refs[i] for i in positions],
                                             uploaded, 12)
                report[f'{name}/{group}/{dtype}'] = score
                print(name, group, dtype, {k: score[k] for k in
                      ('S', 'r2_psi', 'r2_qb', 'dlcfs', 'consistency')}, flush=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
