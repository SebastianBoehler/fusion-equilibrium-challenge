"""Validate immutable MAST016 frame evidence and reconstruct its exact q export."""
import json
from pathlib import Path
import numpy as np
from participant.artifacts import digest
from research.mast_full_hybrid_runtime import require, array_hash


def prefix_shots(plan, deadline):
    status = json.loads(Path(plan['prefix_status_path']).read_text())
    require(digest(Path(plan['prefix_status_path'])) == plan['prefix_status_sha256'], '017prefix status changed')
    require(len(status['shots']) == plan['suffix_start_shot'], '017prefix completed shot count changed')
    total = 0
    gate_names = {'native_bit_equal', 'spline_reconstruction', 'paired_q95', 'paired_current', 'current_closure'}
    for index, pin in enumerate(plan['prefix_record_files']):
        deadline(); path = Path(pin['path'])
        require(path.name == f'frames-{index:04d}.jsonl' and digest(path) == pin['sha256']
                and path.stat().st_size == pin['bytes'], '017prefix record identity changed')
        values, times = [], []
        with path.open() as stream:
            for frame, line in enumerate(stream):
                v = json.loads(line)
                require(v['index'] == frame and set(v['gates']) == gate_names
                    and all(value is True for value in v['gates'].values()), '017prefix gate/order failed')
                pair = np.asarray(v['hybrid_h_h2_q95'], dtype=np.float64)
                require(pair.shape == (2,) and np.isfinite(pair).all() and np.all(pair>0)
                    and abs(pair[0]/pair[1]-1) <= plan['gates']['direct_h_h2_q_relative'], '017prefix paired q failed')
                q = np.float32(v['export_float32_q95'])
                require(np.isfinite(q) and q>0 and q.tobytes() == np.float32(pair[1]).tobytes()
                    and np.float32(v['native_q95']).tobytes() == np.float32(v['stored_q95']).tobytes(),
                    '017prefix exported/native float32 identity failed')
                delta = abs(v['plasma_current_orders4_8_A'][0]-v['plasma_current_orders4_8_A'][1])
                scale = v['scale_current_A']
                require(scale>0 and np.isfinite(scale) and (delta/scale <= 1e-4 or delta <= .001)
                    and abs(v['current_normalized_error']) <= .01, '017prefix current gates failed')
                values.append(q); times.append(v['time_ms'])
        require(len(values) == pin['complete_JSONL_records'], '017prefix record count changed')
        q = np.asarray(values, dtype=np.float32); total += len(q)
        if index < plan['suffix_start_shot']:
            old = status['shots'][index]
            require(old['completed'] and old['index'] == index and old['shot'] == plan['entries'][index]['path']
                and old['frames'] == len(q) and old['q_export_sha256'] == array_hash(q), '017completed prefix export/binding changed')
            summary = {'frames': len(q), 'q_sha256': array_hash(q),
                       'frozen_psi_betaN_sha256': old['frozen_psi_betaN_sha256']}
        else:
            require(index == plan['suffix_start_shot'] and len(q) == plan['suffix_start_frame'], '017partial prefix coverage changed')
            summary = {'prefix_frames': len(q), 'prefix_times_ms': times}
        yield index, q, summary
    require(total == plan['prefix_frames'], '017total prefix coverage changed')
