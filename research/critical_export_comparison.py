"""Score recovered 031 native and folded fields on the complete allowed groups."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from participant import fast_score
from participant.critical_training import ARMS
from participant.nn_submit import Predictor
from research.critical_comparison import RECOVERED, ROOT, validate
from research.inductance_comparison import PARENT, controls, sha256
from research.input_regimes import summarize
from research.native_correction_predictor import NativeCorrectionPredictor, paired_predictions


def score_deltas(native, folded):
    result = {}
    for key, before in native.items():
        after = folded[key]
        if isinstance(before, dict):
            result[key] = score_deltas(before, after)
        elif (isinstance(before, (int, float)) and isinstance(after, (int, float))
              and not isinstance(before, bool)):
            result[key] = after - before
    return result


def packed_groups(name, split):
    packed = Path('run_data/nn-008') / name
    index, shots = fast_score.load_truth(packed)
    expected = split[name] if name == 'development' else sum(split[name].values(), [])
    if (index['names'] != [Path(entry['path']).stem for entry in expected]
            or index['frames'] != index['offsets'][-1]
            or len(index['names']) + 1 != len(index['offsets'])):
        raise ValueError(f'Packed {name} shot order differs from the frozen split')
    if name == 'development' and (len(shots) != 256 or index['frames'] != 57315):
        raise ValueError('Complete frozen development group required')
    if name == 'regression' and (len(shots) != 16 or any(len(v) != 8 for v in split[name].values())):
        raise ValueError('Both complete old regression groups required')
    x = np.load(packed / 'x.npy', mmap_mode='r')
    if len(x) != index['frames'] or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Finite float32 inputs must cover the exact packed frames')
    return packed, index, shots, x


def evaluate(native, folded, split, result, out):
    for name in ('development', 'regression'):
        packed, index, shots, x = packed_groups(name, split)
        predictions, difference = paired_predictions(x, index['offsets'], native, folded)
        result['field_differences'][name] = difference
        result['packed_index_sha256'][name] = sha256(packed / 'index.json')
        groups = {'development': list(range(len(shots)))} if name == 'development' else {
            group: [index['names'].index(Path(e['path']).stem) for e in entries]
            for group, entries in split[name].items()}
        for group, picks in groups.items():
            truth = [shots[i] for i in picks]
            cache = Path('run_data/nn-008/development-refs.pkl') if name == 'development' else None
            refs = fast_score.references(truth, cache, 12)
            scores = result['scores'][group] = {}
            for variant in ('native', 'folded'):
                # The organizer scorer promotes these exact float32 values to float64.
                values = [predictions[variant][i] for i in picks]
                scores[variant] = fast_score.composite(truth, refs, values, 12)
                if name == 'development':
                    result['development_regimes'][variant] = summarize(truth, values, x)
                print(group, variant, scores[variant], flush=True)
                out.write_text(json.dumps(result, indent=2, default=float) + '\n')
            result['folded_minus_native'][group] = score_deltas(scores['native'], scores['folded'])
            out.write_text(json.dumps(result, indent=2, default=float) + '\n')
        del predictions, shots, x
    return result


def main(args):
    out = ROOT / f'{args.arm}-export-comparison.json'
    # Reserve the result before any lengthy work to prevent duplicate active scoring.
    with out.open('x') as handle:
        handle.write(json.dumps({'complete': False, 'arm': args.arm, 'status': 'validating'}) + '\n')
    candidate = RECOVERED / args.arm / 'model.pt'
    _, incumbent = controls(candidate)
    validate(candidate, args.sha256, args.arm, incumbent)
    proposal = json.loads((ROOT / 'proposal-v2/proposal-receipt.json').read_text())
    split_path = Path('run_data/full-split.json')
    if sha256(split_path) != proposal['split_sha256']:
        raise ValueError('Evaluation split differs from the submitted full split')
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    native_path = RECOVERED / args.arm / 'correction-state.pt'
    payload = torch.load(native_path, map_location='cpu', weights_only=False)
    native, folded = NativeCorrectionPredictor(parent, payload), Predictor([candidate])
    result = {'complete': False, 'arm': args.arm, 'prediction_dtype': 'float32',
              'local_scorer_version': '3.2.0', 'locked_shots_read': False,
              'native_metadata_matches_folded': True,
              'checkpoint_sha256': {'parent': sha256(PARENT), 'folded': args.sha256,
                                     'native': sha256(native_path)},
              'source_archive_sha256': proposal['source_archive_sha256'],
              'split_sha256': proposal['split_sha256'], 'packed_index_sha256': {},
              'field_differences': {}, 'scores': {}, 'development_regimes': {},
              'folded_minus_native': {},
              'scope': 'single checkpoint export comparison; ensemble promotion requires separate full controls'}
    evaluate(native, folded, json.loads(split_path.read_text()), result, out)
    result['complete'] = True
    out.write_text(json.dumps(result, indent=2, default=float) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--sha256', required=True)
    main(parser.parse_args())
