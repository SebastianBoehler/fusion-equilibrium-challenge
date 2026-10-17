"""Save complete float32 scores for a candidate and its ensemble contribution."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from participant import fast_score
from research.input_regimes import summarize
from tests.nn_ensemble_check import predictions


def evaluate(args, members):
    if args.development.name != 'development' or args.regression.name != 'regression':
        raise ValueError('Use only the frozen development and old regression directories')
    hashes = {}
    for path in dict.fromkeys(p for group in members.values() for p in group):
        with path.open('rb') as handle:
            hashes[str(path)] = file_digest(handle, 'sha256').hexdigest()
    result = {'upload_dtype': 'float32', 'locked_shots_read': False,
              'checkpoint_sha256': hashes, 'scores': {}, 'development_regimes': {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dev_index, dev_shots = fast_score.load_truth(args.development)
    dev_refs = fast_score.references(dev_shots, args.refs, 12)
    x = np.load(args.development / 'x.npy', mmap_mode='r')
    if args.baseline_cache is not None:
        report = json.loads((args.baseline_cache / 'report.json').read_text())
        expected = {str(p): hashes[str(p)] for p in args.ensemble}
        if report['upload_dtype'] != 'float32' or report['checkpoint_sha256'] != expected:
            raise ValueError('Baseline cache differs from supplied ensemble checkpoints')
        psi = np.load(args.baseline_cache / 'psi-pred.npy', mmap_mode='r')
        scal = np.load(args.baseline_cache / 'scalar-pred.npy', mmap_mode='r')
        cached = [{'psirz': psi[a:b], 'q95': scal[a:b, 0], 'betaN': scal[a:b, 1]}
                  for a, b in zip(dev_index['offsets'][:-1], dev_index['offsets'][1:], strict=True)]
        result['development_regimes']['baseline_ensemble'] = summarize(dev_shots, cached, x)
    reg_index, reg_shots = fast_score.load_truth(args.regression)
    groups = json.loads(args.split.read_text())['regression']
    picks = {name: [reg_index['names'].index(Path(e['path']).stem) for e in entries]
             for name, entries in groups.items()}
    reg_refs = {name: fast_score.references([reg_shots[i] for i in pick], None, 8)
                for name, pick in picks.items()}
    for name, checkpoints in members.items():
        scores = result['scores'][name] = {}
        dev_predictions = predictions(args.development, dev_index, checkpoints)
        scores['development'] = fast_score.composite(
            dev_shots, dev_refs, dev_predictions, 12)
        result['development_regimes'][name] = summarize(dev_shots, dev_predictions, x)
        del dev_predictions
        print(name, 'development', scores['development'], flush=True)
        args.out.write_text(json.dumps(result, indent=2, default=float))
        reg_predictions = predictions(args.regression, reg_index, checkpoints)
        for group, pick in picks.items():
            scores[group] = fast_score.composite([reg_shots[i] for i in pick],
                reg_refs[group], [reg_predictions[i] for i in pick], 8)
            print(name, group, scores[group], flush=True)
            args.out.write_text(json.dumps(result, indent=2, default=float))
    return result


def main(args):
    members = {'candidate': [args.candidate], 'ensemble_plus_candidate':
               [*args.ensemble, args.candidate]}
    if args.retained is not None:
        members['ensemble_plus_retained_plus_candidate'] = [
            *args.ensemble, args.retained, args.candidate]
    if args.comparison_ensemble is not None:
        members['comparison_ensemble_plus_candidate'] = [
            *args.comparison_ensemble, args.candidate]
    evaluate(args, members)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('development', 'regression', 'refs', 'split', 'candidate', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--ensemble', type=Path, nargs='+', required=True)
    parser.add_argument('--retained', type=Path)
    parser.add_argument('--comparison-ensemble', type=Path, nargs='+')
    parser.add_argument('--baseline-cache', type=Path)
    main(parser.parse_args())
