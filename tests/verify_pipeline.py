"""Verify blocked fitting and cached scoring against the incumbent on released shots."""

import json
from pathlib import Path
import tempfile

import numpy as np

from participant.block_ridge import fit_statistics, solve
from participant.cache import arrays, prepare_shot
from participant.cached_score import score_models
from participant.kinetic import KineticModel
from participant.prepared import kinetic_inputs, load_entries
from participant.score import composite


def main():
    manifest = json.loads(Path('protocols/smoke-001-data.json').read_text())
    raw = Path('run_data/local-eightshot/data')
    entries = manifest['train'] + manifest['heldout']
    with tempfile.TemporaryDirectory(prefix='fusion-verify-', dir='run_data') as temporary:
        root = Path(temporary)
        records = [prepare_shot(e, manifest['revision'], raw, root / 'features') for e in entries]
        train = records[:len(manifest['train'])]
        x, y = zip(*(arrays(r) for r in train))
        dense = KineticModel().fit_arrays(np.concatenate(x), np.concatenate(y).astype(float))
        normalizer, statistics = fit_statistics(train)
        blocked = solve(normalizer, statistics, 100)
        differences = []
        for record in records:
            x, _ = arrays(record)
            for name, values in dense.predict_inputs(x).items():
                other = blocked.predict_inputs(x)[name]
                np.testing.assert_allclose(other, values, rtol=1e-9, atol=1e-9)
                differences.append(float(np.max(abs(other - values))))
        checkpoint = root / 'blocked.npz'
        blocked.save(checkpoint)
        reloaded = KineticModel.load(checkpoint)
        validation = records[len(train):]
        cached = score_models(validation, {'blocked': reloaded}, root / 'references', root / 'scores')
        rows = load_entries(manifest['heldout'], manifest['revision'], raw)
        predictions = []
        for x in kinetic_inputs(manifest['heldout'], manifest['revision'], raw):
            prediction = dense.predict_inputs(x)
            predictions.append({k: v.astype(np.float16 if k == 'psirz' else np.float32)
                                for k, v in prediction.items()})
        original = composite(rows, predictions)
        assert cached['blocked'] == original, (cached['blocked'], original)
        # A warm reference cache must give the same score without rebuilding contours.
        from unittest.mock import patch
        with patch('participant.reference_cache.official.build_reference',
                   side_effect=AssertionError('Unexpected reference rebuild')):
            warm = score_models(validation, {'blocked': reloaded}, root / 'references', root / 'warm')
        assert warm == cached
        print(json.dumps({'status': 'passed', 'released_train_shots': len(train),
                          'released_validation_shots': len(validation),
                          'max_prediction_difference': max(differences),
                          'scores': cached, 'lossless_labels': True, 'warm_cache': True}, indent=2))


if __name__ == '__main__':
    main()
