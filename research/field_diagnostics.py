"""Separate field-gradient error from contour error on sampled development frames."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle

import numpy as np

from participant import fast_score
from vendor.fusion_starter.fusion_scoring import derive


def measure(job):
    predicted, ref, frame_numbers = job
    r, z, coarse, mask = fast_score._geometry()
    out = []
    for frame, psi in zip(frame_numbers, predicted, strict=True):
        contour, target = ref[0][frame], ref[1][frame, -1]
        if contour is None or not np.isfinite(target):
            continue
        actual = derive.derive_frame(psi, r, z, 'DIII-D', coarse, mask)['li']
        fixed_contour = derive.internal_inductance(psi, r, z, 'DIII-D', contour)
        out.append((target, actual, fixed_contour))
    return out


def main(args):
    if args.packed.name != 'development':
        raise ValueError('Use the existing development split')
    index = json.loads((args.packed / 'index.json').read_text())
    psi = np.load(args.cache / 'psi-pred.npy', mmap_mode='r')
    refs = pickle.loads(args.refs.read_bytes())
    jobs = []
    for i, (a, b) in enumerate(zip(index['offsets'][:-1], index['offsets'][1:], strict=True)):
        frame_numbers = np.arange(0, b - a, 16)
        jobs.append((np.asarray(psi[a + frame_numbers], dtype=float), refs[i], frame_numbers))
    with ProcessPoolExecutor(12) as pool:
        values = np.concatenate([v for v in pool.map(measure, jobs) if len(v)])
    finite = np.isfinite(values).all(1)
    kept = values[finite]
    variance = np.square(kept[:, 0] - kept[:, 0].mean()).sum()
    report = {'frames': len(values), 'joint_valid_frames': len(kept), 'stride_within_each_shot': 16,
              'li_r2_actual_contour': float(1 - np.square(kept[:, 0] - kept[:, 1]).sum() / variance),
              'li_r2_reference_contour': float(1 - np.square(kept[:, 0] - kept[:, 2]).sum() / variance),
              'reference_contour_is_diagnostic_only': True}
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--refs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
