"""Transfer only the current training contour batch from CPU storage."""

import json

import numpy as np
import torch


class TargetBatch:
    def __init__(self, path, train_index):
        index = json.loads((path / 'index.json').read_text())
        if not index.get('training_only') or not index.get('complete'):
            raise ValueError('Complete training-only inductance targets required')
        if index['names'] != train_index['names'] or index['offsets'] != train_index['offsets']:
            raise ValueError('Target frame order differs from training data')
        self.arrays = {key: np.load(path / f'{key}.npy', mmap_mode='r')
                       for key in ('points', 'weights', 'mask_bits', 'target_li')}
        self.r, self.z = (np.load(path / f'{key}.npy') for key in ('r', 'z'))
        self.manifest_sha = index['manifest_sha256']

    def get(self, indices, device):
        arrays = {key: np.asarray(value[indices]).copy() for key, value in self.arrays.items()}
        masks = np.unpackbits(arrays['mask_bits'], axis=1, count=4225).reshape(-1, 65, 65)
        interior = masks * self.r[None, None, :]
        interior /= interior.sum((1, 2), keepdims=True)
        return tuple(torch.from_numpy(value).to(device) for value in
                     (arrays['points'], arrays['weights'], interior.astype(np.float32), arrays['target_li']))
