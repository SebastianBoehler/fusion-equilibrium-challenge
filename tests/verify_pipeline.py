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
from participant.scalar_heads import ScalarHeads
from participant.scalar_score import compare


def main():
    manifest = json.loads(Path('protocols/smoke-001-data.json').read_text())
    raw = Path('run_data/local-eightshot/data')
    entries = manifest['train'] + manifest['heldout']
    with tempfile.TemporaryDirectory(prefix='fusion-verify-', dir='run_data') as temporary:
        root = Path(temporary)
        records = [prepare_shot(e, manifest['revision'], raw, root / 'features') for e in entries]
        train = records[:len(manifest['train'])]
        from tests.scalar_file_checks import verify as verify_scalar_files
        verify_scalar_files(train)
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
        from tests.scoring_resume_checks import verify
        verify(validation, reloaded, root, cached)
        # A warm reference cache must give the same score without rebuilding contours.
        from unittest.mock import patch
        with patch('participant.reference_cache.official.build_reference',
                   side_effect=AssertionError('Unexpected reference rebuild')):
            warm = score_models(validation, {'blocked': reloaded}, root / 'references', root / 'warm')
        assert warm == cached
        heads = {'ridge': ScalarHeads(reloaded, 'ridge')}
        for method in ('extra-trees', 'gradient-boosting'):
            head = ScalarHeads(reloaded, method).fit(train)
            path = root / (method + '.joblib')
            head.save(path)
            heads[method] = ScalarHeads.load(path)
            for record in validation:
                values, _ = arrays(record)
                np.testing.assert_array_equal(heads[method].predict_inputs(values)['psirz'],
                                              reloaded.predict_inputs(values)['psirz'])
                for key, value in head.predict_scalars(values).items():
                    np.testing.assert_array_equal(heads[method].predict_scalars(values)[key], value)
        scalar_scores = compare(validation, heads, root / 'scores/blocked', root / 'scalar-scores')
        assert scalar_scores['ridge'] == original
        direct_predictions = []
        for record in validation:
            values, _ = arrays(record)
            direct_predictions.append({k: v.astype(np.float16 if k == 'psirz' else np.float32)
                                       for k, v in heads['extra-trees'].predict_inputs(values).items()})
        assert scalar_scores['extra-trees'] == composite(rows, direct_predictions)
        print(json.dumps({'status': 'passed', 'released_train_shots': len(train),
                          'released_validation_shots': len(validation),
                          'max_prediction_difference': max(differences),
                          'scores': cached, 'lossless_labels': True, 'warm_cache': True,
                          'isolated_scalar_scores': scalar_scores}, indent=2))


if __name__ == '__main__':
    main()
