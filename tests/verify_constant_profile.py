"""Regression check for the real all-zero edge diagnostic with one reversed timestamp."""

import argparse
import json
from pathlib import Path

import numpy as np

from participant.data import read_shot, targets
from participant.cache import PREPARE_COLUMNS
from participant.kinetic import kinetic_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    args = parser.parse_args()
    entry = {'path': 'data/diii_d_train/d3d_shot_c905bd75ff.parquet',
             'sha256': '44e4721ec3570c0e612655fa604f44e5bba51d17bb797687ebfd8c96537893c6'}
    row = read_shot(entry, '1e280905b85f2a6fdde7e06fca8cf3a1edf447cb', args.raw)
    assert np.count_nonzero(np.diff(row['thomson_edge_times']) <= 0) == 1
    assert np.all(row['thomson_edge_Te'] == 0) and np.all(row['thomson_edge_ne'] == 0)
    projected = read_shot(entry, '1e280905b85f2a6fdde7e06fca8cf3a1edf447cb',
                          args.raw, columns=PREPARE_COLUMNS)
    np.testing.assert_array_equal(targets(projected), targets(row))
    np.testing.assert_array_equal(kinetic_features(projected), kinetic_features(row))
    values = kinetic_features(row)
    assert np.isfinite(values).all() and np.all(values[:, -6:] == 0)
    redacted = {k: v for k, v in row.items() if not k.startswith('efit_') or k == 'efit_times'}
    redacted.pop('magnetics_dsep', None)
    np.testing.assert_array_equal(values, kinetic_features(redacted))
    print(json.dumps({'status': 'passed', 'source_sha256': entry['sha256'],
                      'frames': len(values), 'edge_features': 'observed constant zero',
                      'raw_target_redaction': True}, indent=2))


if __name__ == '__main__':
    main()
