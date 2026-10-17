"""Check the new native prediction/scoring route on three real training shots."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import torch

from participant import fast_score
from participant.critical_artifact import save_correction
from participant.data import read_shot
from participant.nn_submit import Predictor
from participant.rich import INPUT_COLUMNS, rich_features
from participant.smooth_correction import SmoothCorrection, fold_checkpoint
from research.critical_export_comparison import packed_groups, score_deltas
from research.inductance_comparison import PARENT, sha256
from research.native_correction_predictor import NativeCorrectionPredictor, paired_predictions


def main():
    output = Path('run_data/analysis-031/native-prediction-check.json')
    if output.exists():
        raise FileExistsError('Preserve the existing route check')
    started = time.monotonic()
    torch.set_num_threads(2)
    proposal = json.loads(Path('run_data/analysis-031/proposal-v2/proposal-receipt.json').read_text())
    if sha256(PARENT) != proposal['parent_sha256']:
        raise ValueError('Frozen parent changed')
    root = Path('run_data/analysis-020/packing-check')
    manifest_path = root / 'training-only-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((root / 'train/index.json').read_text())
    rows = [read_shot(entry, manifest['revision'], Path('run_data/analysis-019/training-raw'),
                      (*INPUT_COLUMNS, 'efit_psirz', 'efit_q95', 'efit_beta_n'))
            for entry in manifest['train']]
    if index['names'] != [Path(entry['path']).stem for entry in manifest['train']]:
        raise ValueError('Training shot order differs')
    shots = [{'psi': np.asarray(row['efit_psirz'], np.float32).astype(np.float64),
              'q95': np.asarray(row['efit_q95'], np.float32).astype(np.float64),
              'betaN': np.asarray(row['efit_beta_n'], np.float32).astype(np.float64)} for row in rows]
    raw = np.concatenate([rich_features(row) for row in rows])
    assert len(shots) == 3 and len(raw) == 508
    assert index['offsets'] == np.r_[0, np.cumsum([len(s['psi']) for s in shots])].tolist()
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    model = SmoothCorrection(parent).eval()
    stream = BytesIO()
    save_correction(model, {'route_check': 'zero correction; no optimization'}, stream)
    stream.seek(0)
    native = NativeCorrectionPredictor(parent, torch.load(stream, weights_only=False))
    parent_predictor = Predictor([PARENT])
    for a, b in zip(index['offsets'][:-1], index['offsets'][1:], strict=True):
        before, after = parent_predictor.predict_inputs(raw[a:b]), native.predict_inputs(raw[a:b])
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    with TemporaryDirectory(prefix='fusion-native-route-') as temporary:
        folded_path = Path(temporary) / 'model.pt'
        torch.save(fold_checkpoint(parent, model), folded_path)
        predictions, differences = paired_predictions(raw, index['offsets'], native, Predictor([folded_path]))
    refs = fast_score.references(shots, None, 2)
    scores = {}
    for variant, values in predictions.items():
        scores[variant] = fast_score.composite(shots, refs, values, 2)
        promoted = [{key: array.astype(np.float64) for key, array in prediction.items()}
                    for prediction in values]
        assert scores[variant] == fast_score.composite(shots, refs, promoted, 2)
    split_path = Path('run_data/full-split.json')
    assert sha256(split_path) == proposal['split_sha256']
    split = json.loads(split_path.read_text())
    packed_counts = {}
    for name in ('development', 'regression'):
        _, packed_index, packed_shots, _ = packed_groups(name, split)
        packed_counts[name] = {'shots': len(packed_shots), 'frames': packed_index['frames']}
        del packed_shots
    source_files = ('research/native_correction_predictor.py', 'research/critical_export_comparison.py',
                    'research/native_prediction_check.py')
    result = {'complete': True, 'cpu_only': True, 'training_only_inference_and_scoring': True,
              'training_manifest_sha256': sha256(manifest_path), 'parent_sha256': sha256(PARENT),
              'training_shots': 3, 'training_frames': 508, 'zero_correction_parent_prediction_exact': True,
              'float32_storage_vs_float64_promotion_scores_exact': True,
              'field_differences': differences, 'scores': scores,
              'folded_minus_native': score_deltas(scores['native'], scores['folded']),
              'allowed_packed_order_counts_verified': packed_counts,
              'held_out_predictions_scored': False, 'actual_031_artifacts_checked': False,
              'locked_shots_read': False, 'learned_checkpoint_or_fields_saved': False,
              'source_sha256': {name: sha256(Path(name)) for name in source_files},
              'actual_cpu_seconds': time.monotonic() - started,
              'limitations': 'Zero-correction route check; actual trained 031 artifacts and full native/folded scores remain pending.'}
    output.write_text(json.dumps(result, indent=2, default=float) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
