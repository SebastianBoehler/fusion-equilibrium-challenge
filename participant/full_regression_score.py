"""Save all components and failures for the two frozen regression groups."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from . import fast_score
from .nn_submit import Predictor


def run(packed, split, checkpoint, out, workers):
    if out.exists():
        raise FileExistsError('Preserve completed regression receipts')
    index, shots = fast_score.load_truth(packed)
    x = np.load(packed / 'x.npy', mmap_mode='r')
    model = Predictor([checkpoint])
    predictions = []
    for i in range(len(shots)):
        a, b = index['offsets'][i:i + 2]
        prediction = model.predict_inputs(np.asarray(x[a:b]))
        predictions.append({k: v.astype(np.float32).astype(np.float64) for k, v in prediction.items()})
    scores = {}
    for name, entries in json.loads(split.read_text())['regression'].items():
        wanted = [Path(e['path']).stem for e in entries]
        picks = [index['names'].index(w) for w in wanted]
        truth = [shots[i] for i in picks]
        refs = fast_score.references(truth, None, workers)
        scores[name] = fast_score.composite(truth, refs, [predictions[i] for i in picks], workers)
        print(name, scores[name], flush=True)
    with checkpoint.open('rb') as handle:
        sha = file_digest(handle, 'sha256').hexdigest()
    out.write_text(json.dumps({'complete': True, 'prediction_dtype': 'float32',
                               'checkpoint_sha256': sha, 'scores': scores}, indent=2, default=float) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for key in ('packed', 'split', 'checkpoint', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    a = parser.parse_args()
    run(a.packed, a.split, a.checkpoint, a.out, a.workers)


if __name__ == '__main__':
    main()
