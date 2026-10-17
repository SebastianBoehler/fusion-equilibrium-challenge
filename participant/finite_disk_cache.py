"""Disk-backed frozen parent fields, with unchanged046 chunk inference."""

import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import digest
from .finite_parent_cache import CHUNK_FRAMES, ParentCache, cache_parent


def cache_parent_disk(model, raw, valid, destination, provenance):
    """Use a bounded CPU inference buffer; disk pages still require RSS profiling."""
    if (raw.ndim != 2 or raw.dtype != np.float32 or valid.ndim != 1
            or valid.dtype != np.bool_ or len(raw) != len(valid)
            or raw.shape[1] != model.config['inputs']):
        raise ValueError('Invalid permitted TRAIN disk-cache schema')
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError('No eligible TRAIN disk-cache support')
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    path = destination / 'parent-fields.npy'
    count = len(indices)
    fields = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                      shape=(count, 4225))
    device = model.correction_scale.device
    features = torch.empty((count, model.correction.in_features), dtype=torch.float32, device=device)
    scalars = torch.empty((count, model.config['scalars']), dtype=torch.float32)
    std = torch.empty(count, dtype=torch.float32, device=device)
    chunks = []
    for start in range(0, count, CHUNK_FRAMES):
        end = min(start + CHUNK_FRAMES, count)
        values = np.asarray(raw[indices[start:end]])
        chunk = cache_parent(model, values, np.ones(len(values), dtype=np.bool_))
        features[start:end].copy_(chunk.features)
        fields[start:end] = chunk.fields.numpy()
        scalars[start:end].copy_(chunk.scalars)
        std[start:end].copy_(chunk.parent_std)
        chunks.append(end - start)
    fields.flush()
    np.save(destination / 'indices.npy', indices, allow_pickle=False)
    receipt = {'complete': True, 'provenance': provenance, 'frames': count,
               'chunks': chunks, 'fields_sha256': digest(path),
               'indices_sha256': digest(destination / 'indices.npy'),
               'fields_payload_bytes': fields.nbytes,
               'maximum_CPU_chunk_fields_payload_bytes': CHUNK_FRAMES * 4225 * 4,
               'peak_resident_memory_measured': False}
    with (destination / 'receipt.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    return ParentCache(indices, features, torch.from_numpy(fields), scalars, std)
