"""Step 2: train a classifier head on the cached MedSAM features (5 seeded repeats).

    python src/train.py --dataset kvasir --method spdnet                    # covariance -> SPDNet
    python src/train.py --dataset kvasir --method gap --standardize zscore  # GAP -> z-score -> Linear (main baseline)
    python src/train.py --dataset kvasir --method gap                       # GAP -> Linear (ablation)
    add --dataset derma for DermaMNIST, --debug for a 1-epoch smoke test

Both methods call the same official fit_spdnet (loss with class weights, Adam, batch size,
early stopping on val balanced accuracy, 5 repeats). Only the head differs:
  spdnet: covs (256x256) -> BasicSPDNet 256->128->64->32 -> ReEig -> LogEig -> Linear
  gap:    gaps (256,) -> optional z-score (train statistics) -> Linear
For gap, BasicSPDNet is swapped for GapLinear while fit_spdnet runs, so the protocol is shared.
"""
import argparse
import contextlib
import json
import os
import shutil
import time

import repo

import numpy as np
import torch

import config
import experiments.spdnet as spdnet_exp
import eval_report
from features import FeatureDataset, cache_dir

SHORT_NAMES = {
    'kvasir': ['dyed-lifted', 'dyed-resect', 'esophagitis', 'cecum', 'pylorus', 'z-line', 'polyps', 'UC'],
    'derma': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
}
TITLES = {'kvasir': 'Kvasir-v2', 'derma': 'DermaMNIST'}


class GapLinear(torch.nn.Module):
    """Linear head on GAP features. mean/std are fixed buffers (zeros/ones = no standardisation)."""

    def __init__(self, in_features, out_features, mean=None, std=None, device=None):
        super().__init__()
        mean = torch.zeros(in_features) if mean is None else mean
        std = torch.ones(in_features) if std is None else std
        self.register_buffer('mean', mean.to(device=device, dtype=torch.float32))
        self.register_buffer('std', std.to(device=device, dtype=torch.float32))
        self.linear = torch.nn.Linear(in_features, out_features, device=device)

    def forward(self, x):
        return self.linear((x - self.mean) / self.std)


@contextlib.contextmanager
def backbone(factory):
    """Temporarily replace the SPDNet backbone used inside fit_spdnet."""
    original = spdnet_exp.BasicSPDNet
    spdnet_exp.BasicSPDNet = factory
    try:
        yield
    finally:
        spdnet_exp.BasicSPDNet = original


def dataset_descriptor(dataset, cfg):
    """(descriptor for fit_spdnet, train class counts)."""
    if dataset == 'kvasir':
        from kvasir_dataset import kvasir_info, train_class_counts
        return kvasir_info(repo.SPLIT_CSV), torch.tensor(train_class_counts(repo.SPLIT_CSV))
    from experiments import datasets, training_set_label_proportions
    info = next(d for d in datasets if d.dataset_name == 'dermamnist')
    return info, training_set_label_proportions(os.path.join(cfg.dataset_dir, 'medmnist'), info)


def zscore_stats(train):
    """Per-dimension mean/std of the train GAP vectors only (no val/test leakage)."""
    g = torch.from_numpy(np.array(train.x.data, dtype=np.float32))
    std = g.std(0)
    print(f'z-score: per-dim std ranges {std.min():.2e} .. {std.max():.2e} (clamped at 1e-6)')
    return g.mean(0), std.clamp_min(1e-6)


def export_runs(run_dir, out_dir):
    """Copy the small per-repeat outputs (predictions, targets, metrics) into the repo."""
    for r in sorted(d for d in os.listdir(run_dir) if d.isdigit()):
        dst = os.path.join(out_dir, 'runs', r)
        os.makedirs(dst, exist_ok=True)
        for f in ('prediction.npy', 'target.npy', 'np_metrics.json', 'torch_metrics.json'):
            shutil.copy2(os.path.join(run_dir, r, f), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('kvasir', 'derma'), default='kvasir')
    ap.add_argument('--method', choices=('spdnet', 'gap'), required=True)
    ap.add_argument('--standardize', choices=('none', 'zscore'), default='none', help='gap only')
    ap.add_argument('--debug', action='store_true', help='1 epoch, 1 repeat, written to <model_dir>/debug')
    args = ap.parse_args()
    if args.method == 'spdnet' and args.standardize != 'none':
        ap.error('--standardize applies to --method gap only')

    cfg = config.load_user_config()
    device, gpu = repo.init_torch()
    info, counts = dataset_descriptor(args.dataset, cfg)
    k = len(info.labels)

    column = 'covs' if args.method == 'spdnet' else 'gaps'
    train, val, test = (FeatureDataset(os.path.join(cache_dir(cfg, args.dataset), s), column)
                        for s in ('train', 'val', 'test'))
    assert len(train) == int(counts.sum()), 'cache size does not match the train split'

    # 1. Head
    model_type = 'SPDNet' if args.method == 'spdnet' else 'GAP-Linear'
    features = 'medsam' if args.standardize == 'none' else 'medsam-zscore'
    if args.method == 'spdnet':
        factory = spdnet_exp.BasicSPDNet
    else:
        mean, std = zscore_stats(train) if args.standardize == 'zscore' else (None, None)

        def factory(feature_count, out, device=None, **_):
            return GapLinear(feature_count, out, mean, std, device=device)
    n_params = sum(p.numel() for p in factory(256, k, device=device).parameters())

    # 2. Train with the official protocol
    model_root = os.path.join(cfg.model_dir, 'debug' if args.debug else '', model_type)
    run_dir = os.path.join(model_root, info.dataset_name, features)
    reused = not args.debug and all(os.path.exists(os.path.join(run_dir, str(i), 'np_metrics.json'))
                                    for i in range(1, 6))
    if reused:
        print(f'NOTE: all 5 repeats in {run_dir} are finished; they are loaded, not retrained.')
    print(f'{args.dataset} | {args.method} | standardize={args.standardize} | {gpu} | '
          f'train/val/test {len(train)}/{len(val)}/{len(test)} | {n_params:,} trainable parameters')

    kwargs = dict(class_proportions=counts, feature_count=256, class_count=k)
    if args.debug:
        kwargs.update(max_epochs=1, repeats=1)
    t0 = time.time()
    with backbone(factory):
        metrics, _ = spdnet_exp.fit_spdnet(model_root, os.path.join(cfg.result_dir, model_type, 'logging'),
                                           'prescreening', info, features, train, val, test, **kwargs)
    minutes = (time.time() - t0) / 60
    print(f'\ndone in {minutes:.1f} min')
    print(json.dumps({m: round(v, 4) for m, v in metrics.items()}, indent=1))
    if args.debug:
        return

    # 3. Save summary + per-run outputs, then the full evaluation report
    name = f'{args.dataset}_{args.method}' + ('_zscore' if args.standardize == 'zscore' else '')
    out_dir = os.path.join(repo.RESULTS, name)
    export_runs(run_dir, out_dir)
    if not reused:   # keep the real training time when the runs were only reloaded
        with open(os.path.join(out_dir, 'fit_summary.json'), 'w', encoding='utf-8') as f:
            json.dump({'method': args.method, 'standardize': args.standardize, 'minutes': round(minutes, 1),
                       'trainable_params': n_params, 'mean_metrics_sklearn': metrics, 'device': gpu}, f, indent=2)
    head = 'covariance + SPDNet' if args.method == 'spdnet' else 'GAP + Linear'
    title = f'{TITLES[args.dataset]} - MedSAM + {head}' + (' (z-score)' if args.standardize == 'zscore' else '')
    eval_report.report(os.path.join(out_dir, 'runs'), out_dir, SHORT_NAMES[args.dataset], title)
    print(f'\nresults: {out_dir}')


if __name__ == '__main__':
    main()
