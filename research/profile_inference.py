"""Measure the frozen ensemble on real held-out shots without spending GPU time."""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from participant.nn_submit import Predictor


def main(args):
    index = json.loads((args.packed / 'index.json').read_text())
    x = np.load(args.packed / 'x.npy', mmap_mode='r')
    batches = [np.array(x[a:b], copy=True)
               for a, b in zip(index['offsets'][:8], index['offsets'][1:9])]
    model = Predictor(args.checkpoints)
    measurements = {}
    for threads in (1, 2, 4):
        torch.set_num_threads(threads)
        model.predict_inputs(batches[0])
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            for batch in batches:
                model.predict_inputs(batch)
            timings.append(time.perf_counter() - start)
        measurements[threads] = {'seconds': timings, 'median_seconds': float(np.median(timings)),
                                 'frames_per_second': sum(map(len, batches)) / np.median(timings)}
        print(threads, measurements[threads], flush=True)
    best = min(measurements, key=lambda n: measurements[n]['median_seconds'])
    torch.set_num_threads(best)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        model.predict_inputs(batches[0])
    operators = sorted(profiler.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    total = sum(e.self_cpu_time_total for e in operators)
    report = {'machine': platform.platform(), 'torch': torch.__version__, 'device': 'cpu',
              'members': len(model.members), 'shots': len(batches), 'frames': sum(map(len, batches)),
              'thread_measurements': measurements, 'fastest_threads': best,
              'operators': [{'name': e.key, 'self_ms': e.self_cpu_time_total / 1000,
                             'self_fraction': e.self_cpu_time_total / total, 'calls': e.count}
                            for e in operators[:12]]}
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['operators'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
