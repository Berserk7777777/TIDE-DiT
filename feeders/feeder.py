from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class FeatureFeeder(Dataset):
    def __init__(self, path, split='train'):
        root = Path(path)
        split = str(split).lower()
        split_map = {
            'train': ('train.npy', 'train_label.npy'),
            'val': ('ztest.npy', 'z_label.npy'),
            'test': ('ztest.npy', 'z_label.npy'),
            'ztest': ('ztest.npy', 'z_label.npy'),
            'gtest': ('gtest.npy', 'g_label.npy'),
        }
        if split not in split_map:
            raise ValueError(f"Unsupported feature split: {split}")

        x_name, y_name = split_map[split]
        self.x = np.load(root / x_name)
        self.y = np.load(root / y_name)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], int(self.y[index])
