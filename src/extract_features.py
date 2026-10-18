"""Step 1 of the pipeline: run frozen MedSAM once per image and cache covs + gaps + labels.

    python src/extract_features.py --dataset kvasir [--root <kvasir-dataset-v2>]
    python src/extract_features.py --dataset derma

Same pipeline as the official main.handle_spdnet_gve; the only change is that the mapper also
keeps the GAP vector, so both methods see features from the identical forward pass.
An existing complete cache is reused and only re-checked.
"""
import argparse
import datetime
import json
import os
import time

import repo

import numpy as np
import torch

import config
from common.gve import PretrainedFeatures
from data import CachedDataset
from features import COLUMNS, SPLITS, NdColumn, cache_dir, make_mapper, sha256_of


def load_splits(dataset, cfg, root):
    """Returns (train, val, test) datasets that yield MedMNIST-style (image, label) pairs."""
    if dataset == 'kvasir':
        from kvasir_dataset import KvasirDataset
        return [KvasirDataset(root, repo.SPLIT_CSV, s) for s in SPLITS]
    from experiments import datasets, prepare_datasets   # official DermaMNIST split (64x64)
    derma = next(d for d in datasets if d.dataset_name == 'dermamnist')
    return prepare_datasets(os.path.join(cfg.dataset_dir, 'medmnist'), derma, requires_validation_set=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('kvasir', 'derma'), required=True)
    ap.add_argument('--root', default=repo.KVASIR_ROOT, help='Kvasir-v2 image folder')
    args = ap.parse_args()

    cfg = config.load_user_config()
    device, gpu = repo.init_torch()
    out_dir = cache_dir(cfg, args.dataset)
    print(f'device: {gpu} | torch {torch.__version__} | cache -> {out_dir}')

    checkpoint = os.path.join(cfg.model_dir, 'medsam_vit_b.pth')
    mapper = make_mapper(PretrainedFeatures(cfg.model_dir, 'medsam'))
    record = {'started': datetime.datetime.now().isoformat(timespec='seconds'), 'dataset': args.dataset,
              'checkpoint_sha256': sha256_of(checkpoint), 'device': gpu, 'torch': torch.__version__,
              'preprocessing': 'official transform_demo: resize 1024^2 bicubic + per-channel min-max',
              'feature_layer': 'MedSAM image_encoder output (256, 64, 64)', 'splits': {}}
    if args.dataset == 'kvasir':
        record['split_csv_sha256'] = sha256_of(repo.SPLIT_CSV)

    for split, ds in zip(SPLITS, load_splits(args.dataset, cfg, args.root)):
        files = [os.path.join(out_dir, split, c + '.nd') for c in COLUMNS]
        reused = all(os.path.exists(f) for f in files)
        print(f'\n[{split}] {len(ds)} images, ' + ('checking existing cache' if reused else 'extracting'))
        t0 = time.time()
        CachedDataset(tuple(files), ds, device, mapper, use_tqdm=True)   # official writer
        seconds = time.time() - t0

        # Check: shapes, labels in the same order as the split, covariance symmetric.
        covs, gaps, labels = (NdColumn(f) for f in files)
        assert covs.shape == (256, 256) and gaps.shape == (256,)
        y = np.array([int(labels[i]) for i in range(len(labels))])
        assert (y == np.asarray(ds.labels).ravel()).all(), 'cached labels do not match the split'
        c0 = covs[0]
        print(f'  labels OK, per class {np.bincount(y).tolist()} | '
              f'symmetry error {(c0 - c0.T).abs().max():.1e} | {seconds / 60:.1f} min')

        # Derma only: our covariances must equal the official cache used for the paper reproduction.
        official = os.path.join(cfg.cache_dir, 'dermamnist', 'medsam', split, 'images.nd')
        if args.dataset == 'derma' and os.path.exists(official):
            off = NdColumn(official)
            idx = np.linspace(0, len(off) - 1, 50).astype(int)
            rel = max(float((covs[i] - off[i]).norm() / off[i].norm()) for i in idx)
            print(f'  max relative difference to the official covariance cache (50 images): {rel:.1e}')

        record['splits'][split] = {'n': len(ds), 'seconds': round(seconds, 1), 'reused_cache': reused}

    record['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
    # Do not overwrite the timing of the real extraction with a re-check.
    reused_all = all(s['reused_cache'] for s in record['splits'].values())
    path = os.path.join(out_dir, 'extraction_check.json' if reused_all else 'extraction_info.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)
    print(f'\nsaved {path}')


if __name__ == '__main__':
    main()
