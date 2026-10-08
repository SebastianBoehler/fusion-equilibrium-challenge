"""Bound scalar-ingestion file handles without changing real training values."""

import resource
import errno

import numpy as np

from participant.cache import arrays
from participant.scalar_heads import training_arrays


def verify(records):
    pairs = [arrays(record) for record in records]
    expected_x = np.concatenate([x for x, _ in pairs])
    expected_y = np.concatenate([y[:, -2:] for _, y in pairs]).astype(np.float64)
    for x, y in pairs:
        x._mmap.close()
        y._mmap.close()
    actual_x, actual_y = training_arrays(records)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
    try:
        opened = []
        try:
            for record in records * 64:
                opened.append(arrays(record))
        except OSError as error:
            assert error.errno == errno.EMFILE
        else:
            raise AssertionError('Original ingestion did not exhaust the reduced file limit')
        finally:
            for x, y in opened:
                x._mmap.close()
                y._mmap.close()
        # Repeat released records only to stress ingestion, not model training.
        stressed_x, stressed_y = training_arrays(records * 64)
        width = len(expected_x)
        for index in range(64):
            sl = slice(index * width, (index + 1) * width)
            np.testing.assert_array_equal(stressed_x[sl], expected_x)
            np.testing.assert_array_equal(stressed_y[sl], expected_y)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
