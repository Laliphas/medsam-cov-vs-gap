"""Paired comparison of two experiments on the same test set (numpy only).

    python src/compare.py kvasir_spdnet kvasir_gap_zscore

Metric = mean over the 5 repeats (the paper's protocol). 95% CI of the difference A - B from a
paired bootstrap over test images (the same resampled images for both methods).
Exploratory: ensemble of the 5 repeats (averaged probabilities) with an exact McNemar test.
"""
import argparse
import os
from math import comb

import numpy as np

import eval_report

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def load(name):
    """(probabilities (R, N, K), targets (N,)) of results/<name>/runs."""
    runs = eval_report.load_runs(os.path.join(RESULTS, name, 'runs'))
    t = runs[0][2]
    assert all((r[2] == t).all() for r in runs), 'targets differ between repeats'
    return np.stack([r[1] for r in runs]), t


def balanced_accuracy(prob, t, w):
    """Mean over repeats of balanced accuracy, for each row of image weights w (B, N)."""
    onehot = np.eye(prob.shape[-1], dtype=np.float32)[t]
    total = w @ onehot                                                   # (B, K) images per class
    hits = [w @ (onehot * (p.argmax(1) == t)[:, None]) for p in prob]   # (B, K) per repeat
    return np.mean([(h / total).mean(1) for h in hits], axis=0)


def macro_auroc(prob, t, w):
    """Mean over repeats of macro AUROC with image weights (ties ignored; scores are continuous)."""
    k = prob.shape[-1]
    out = np.zeros(len(w))
    for p in prob:
        for c in range(k):
            order = np.argsort(p[:, c], kind='mergesort')
            pos = (t[order] == c)
            wo = w[:, order]
            neg_below = np.cumsum(wo * ~pos, axis=1)                    # negatives ranked below each image
            num = (wo[:, pos] * neg_below[:, pos]).sum(1)
            out += num / (wo[:, pos].sum(1) * wo[:, ~pos].sum(1))
    return out / (len(prob) * k)


def mcnemar_exact(a_right, b_right):
    """Two-sided exact McNemar p-value from the discordant pairs."""
    b, c = int((a_right & ~b_right).sum()), int((~a_right & b_right).sum())
    n, m = b + c, min(b, c)
    return b, c, min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2 ** n)


def compare(name_a, name_b, n_boot=10_000, seed=0):
    pa, t = load(name_a)
    pb, tb = load(name_b)
    assert (t == tb).all(), 'the two experiments use different test sets'
    n = len(t)
    rng = np.random.default_rng(seed)
    w = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=n_boot)]).astype(np.float32)

    rows = {}
    for metric, fn in (('Balanced accuracy', balanced_accuracy), ('Macro AUROC', macro_auroc)):
        a, b = fn(pa, t, w), fn(pb, t, w)
        d = 100 * (a - b)
        lo, hi = np.percentile(d[1:], [2.5, 97.5])
        rows[metric] = dict(a=100 * a[0], b=100 * b[0], diff=d[0], lo=lo, hi=hi, significant=lo > 0 or hi < 0)

    ens_a, ens_b = pa.mean(0).argmax(1) == t, pb.mean(0).argmax(1) == t
    b_only, c_only, p = mcnemar_exact(ens_a, ens_b)
    rows['ensemble'] = dict(acc_a=100 * ens_a.mean(), acc_b=100 * ens_b.mean(), a_only=b_only, b_only=c_only, p=p)
    return rows


def print_comparison(name_a, name_b, rows):
    print(f'\n{name_a}  vs  {name_b}')
    for metric in ('Balanced accuracy', 'Macro AUROC'):
        r = rows[metric]
        print(f'  {metric:18s} {r["a"]:.1f} vs {r["b"]:.1f}  diff {r["diff"]:+.2f} '
              f'[{r["lo"]:+.2f}, {r["hi"]:+.2f}]' + ('  significant' if r['significant'] else ''))
    e = rows['ensemble']
    print(f'  ensemble accuracy  {e["acc_a"]:.1f} vs {e["acc_b"]:.1f}  '
          f'(only A right {e["a_only"]}, only B right {e["b_only"]}, McNemar p = {e["p"]:.3f}) [exploratory]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--n-boot', type=int, default=10_000)
    args = ap.parse_args()
    print_comparison(args.a, args.b, compare(args.a, args.b, args.n_boot))


if __name__ == '__main__':
    main()
