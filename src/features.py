"""Feature cache shared by extract_features.py and train.py.

Cache layout: <cache_dir>/<name>/<split>/{covs,gaps,labels}.nd, written by the official CachedDataset.
  covs   (256, 256)  covariance descriptor of the MedSAM feature map  -> SPDNet input
  gaps   (256,)      global average pooling of the same feature map    -> GAP + Linear input
  labels ()          class index
"""
import hashlib
import os

import numpy as np
import torch

import repo  # noqa: F401  (adds the official code to sys.path)
from torchutil import torch_dtype_from_int

COLUMNS = ('covs', 'gaps', 'labels')
SPLITS = ('train', 'val', 'test')
CACHE_NAMES = {'kvasir': os.path.join('kvasir_v2', 'medsam'),
               'derma': os.path.join('dermamnist', 'medsam_gap')}


def cache_dir(cfg, dataset):
    return os.path.join(cfg.cache_dir, CACHE_NAMES[dataset])


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def make_mapper(features):
    """One MedSAM forward pass -> (cov, gap, label). Both descriptors come from the same feature map."""
    from common import cov_descr
    from data import default_transform

    def mapper(datapoint):
        image, label = default_transform(datapoint)   # numpy -> tensor (official)
        fmap = features(image)                        # MedSAM transform + frozen encoder -> (256, 64, 64)
        gap = fmap.mean(dim=(-2, -1))                 # the mean that cov_descr subtracts
        return cov_descr(fmap), gap, label.squeeze(-1)

    return mapper


class NdColumn:
    """Read-only memmap reader for one .nd file.

    Used instead of CachedDataset because CachedDataset re-creates (and so erases) a cache it
    considers invalid. File format: uint64 dtype id, N, ndim, shape[ndim], then raw data.
    """

    def __init__(self, path):
        with open(path, 'rb') as f:
            dtype_id, n, ndim = np.frombuffer(f.read(24), dtype='<u8')
            self.shape = tuple(int(v) for v in np.frombuffer(f.read(8 * int(ndim)), dtype='<u8'))
            offset = f.tell()
        np_dtype = torch.empty((), dtype=torch_dtype_from_int(int(dtype_id)), device='cpu').numpy().dtype
        self.n = int(n)
        self.data = np.memmap(path, dtype=np_dtype, mode='r', offset=offset, shape=(self.n, *(self.shape or (1,))))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.data[i])).reshape(self.shape)
        return x.to(device=torch.empty(()).device)


class FeatureDataset(torch.utils.data.Dataset):
    """(input, label) pairs for one split; input is 'covs' or 'gaps'."""

    def __init__(self, split_dir, column):
        self.x = NdColumn(os.path.join(split_dir, column + '.nd'))
        self.y = NdColumn(os.path.join(split_dir, 'labels.nd'))
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
