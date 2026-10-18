"""Kvasir-v2 dataset that returns items in the MedMNIST format used by the official code.

Each item is (image float32 CHW in [0, 1], label int64 of shape (1,)), so the official
mapper, MedSAM wrapper, cov_descr, CachedDataset and fit_spdnet work unchanged.
No resizing or normalisation here: the official MedSAM transform does both (as for DermaMNIST).
"""
import csv
import os
from typing import NamedTuple

import numpy as np
from PIL import Image

SPLITS = ('train', 'val', 'test')


class KvasirInfo(NamedTuple):
    """The three fields fit_spdnet reads from a MedMNIST dataset descriptor."""
    dataset_name: str
    labels: tuple
    n_channels: int


def read_split(split_csv):
    with open(split_csv, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def kvasir_info(split_csv):
    names = {int(r['label']): r['label_name'] for r in read_split(split_csv)}
    assert sorted(names) == list(range(len(names))), f'labels must be 0..K-1, got {sorted(names)}'
    return KvasirInfo('kvasir_v2', tuple(names[i] for i in range(len(names))), 3)


def train_class_counts(split_csv):
    """Train images per class; used for the class weights of the loss."""
    rows = read_split(split_csv)
    k = len({r['label'] for r in rows})
    return np.bincount([int(r['label']) for r in rows if r['split'] == 'train'], minlength=k)


class KvasirDataset:
    """One split. Order follows the CSV, which keeps the feature cache aligned with the labels."""

    def __init__(self, root, split_csv, split):
        assert split in SPLITS, split
        self.root = root
        self.items = [(r['relpath'], int(r['label'])) for r in read_split(split_csv) if r['split'] == split]
        self.labels = np.array([[y] for _, y in self.items], dtype=np.int64)   # (N, 1) like MedMNIST

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        relpath, label = self.items[i]
        with Image.open(os.path.join(self.root, relpath)) as img:
            a = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
        return np.ascontiguousarray(a.transpose(2, 0, 1)), np.array([label], dtype=np.int64)


if __name__ == '__main__':
    # Self-test: python src/kvasir_dataset.py [kvasir root]
    import sys
    import repo
    root = sys.argv[1] if len(sys.argv) > 1 else repo.KVASIR_ROOT
    print(kvasir_info(repo.SPLIT_CSV))
    for s in SPLITS:
        ds = KvasirDataset(root, repo.SPLIT_CSV, s)
        print(f'{s:5s}: {len(ds)} images, per class {np.bincount(ds.labels.ravel()).tolist()}')
    x, y = KvasirDataset(root, repo.SPLIT_CSV, 'test')[0]
    print('test[0]:', x.shape, x.dtype, f'[{x.min():.3f}, {x.max():.3f}]', y)
