"""Check the process-parallel scorer reproduces the serial official adapter exactly."""
import sys
import time
from pathlib import Path

from participant import fast_score
from participant.score import composite


def main(packed: Path, predictions: Path):
    _, shots = fast_score.load_truth(packed)
    preds = fast_score.load_predictions(predictions, len(shots))
    started = time.time()
    refs = fast_score.references(shots, None, 12)
    parallel = fast_score.composite(shots, refs, preds, 12)
    print('parallel', round(time.time() - started, 1), 's')
    rows = [{'efit_psirz': s['psi'], 'efit_q95': s['q95'], 'efit_beta_n': s['betaN']} for s in shots]
    started = time.time()
    serial = composite(rows, preds)
    print('serial', round(time.time() - started, 1), 's')
    assert parallel == serial, (parallel, serial)
    perfect = fast_score.composite(shots, refs, [{'psirz': s['psi'], 'q95': s['q95'], 'betaN': s['betaN']}
                                                 for s in shots], 12)
    assert perfect['S'] == 1.0, perfect['S']
    print('identical to serial adapter; perfect S = 1.0')
    print({k: parallel[k] for k in ('S', 'r2_psi', 'r2_qb', 'dlcfs', 'consistency')}, parallel['r2_cons_each'])


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
