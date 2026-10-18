"""Evaluation report for one experiment (numpy only).

    python src/eval_report.py --runs results/kvasir_spdnet/runs --out results/kvasir_spdnet --classes ...

Input: one folder per repeat with prediction.npy (softmax), target.npy and the official
np_metrics.json / torch_metrics.json. Output in --out:
  summary.md, per_run.csv          metrics per repeat and mean +- std (full test set)
  per_class.csv                    precision / recall / F1 / AUROC per class
  confusion_matrix_sum.csv/.png    confusion matrix summed over repeats
Sanity checks in summary.md: our metrics equal np_metrics.json, and torch_metrics.json
equals the accuracy of the first 256 test images only (a bug in the official logging).
"""
import argparse
import json
import os

import numpy as np

METRICS = ['Accuracy', 'Balanced accuracy', 'Macro F1', 'Macro Precision', 'Macro AUROC', 'Micro AUROC']


def load_runs(runs_dir):
    """[(run id, probabilities (N, K), targets (N,))] for every numbered subfolder."""
    out = []
    for r in sorted((d for d in os.listdir(runs_dir) if d.isdigit()), key=int):
        prob = np.load(os.path.join(runs_dir, r, 'prediction.npy')).astype(np.float64)
        t = np.load(os.path.join(runs_dir, r, 'target.npy')).astype(int).ravel()
        out.append((r, prob, t))
    return out


def confusion(t, p, k):
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (t, p), 1)
    return m                                                     # rows = true, cols = predicted


def auroc(y, s):
    """Binary AUROC via the Mann-Whitney U statistic with average ranks for ties (= sklearn)."""
    order = np.argsort(s, kind='mergesort')
    s_sorted = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = y.sum()
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos))


def metrics(t, prob, k):
    cm = confusion(t, prob.argmax(1), k)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=prec + rec > 0)
    onehot = np.eye(k, dtype=int)[t]
    auc = np.array([auroc(onehot[:, c], prob[:, c]) for c in range(k)])
    return {'cm': cm, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'support': cm.sum(1),
            'Accuracy': tp.sum() / len(t), 'Balanced accuracy': rec.mean(), 'Macro F1': f1.mean(),
            'Macro Precision': prec.mean(), 'Macro AUROC': auc.mean(),
            'Micro AUROC': auroc(onehot.ravel(), prob.ravel())}


def pm(x, scale=100):
    """'mean ± std' over repeats (sample std)."""
    return f'{scale * np.mean(x):.1f} ± {scale * np.std(x, ddof=1):.1f}'


def save_confusion_png(cmn, classes, title, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    k = len(classes)
    fig, ax = plt.subplots(figsize=(k + 2, k + 1.5))
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f'{100 * cmn[i, j]:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if cmn[i, j] > 0.5 else 'black')
    ax.set_xticks(range(k), classes, rotation=45, ha='right')
    ax.set_yticks(range(k), classes)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(runs_dir, out_dir, classes, title=''):
    os.makedirs(out_dir, exist_ok=True)
    k = len(classes)
    runs = load_runs(runs_dir)
    res = [metrics(t, prob, k) for _, prob, t in runs]
    ids = [r for r, _, _ in runs]

    # 1. Per run and mean ± std
    rows = ['| Run | ' + ' | '.join(METRICS) + ' |', '|' + '---|' * (len(METRICS) + 1)]
    rows += [f'| {r} | ' + ' | '.join(f'{100 * m[n]:.1f}' for n in METRICS) + ' |' for r, m in zip(ids, res)]
    rows.append('| **mean ± std** | ' + ' | '.join(f'**{pm([m[n] for m in res])}**' for n in METRICS) + ' |')
    with open(os.path.join(out_dir, 'per_run.csv'), 'w') as f:
        f.write('run;' + ';'.join(METRICS) + '\n')
        f.writelines(f'{r};' + ';'.join(f'{m[n]:.6f}' for n in METRICS) + '\n' for r, m in zip(ids, res))

    # 2. Per class (mean ± std over runs)
    stat = {s: np.array([m[s] for m in res]) for s in ('prec', 'rec', 'f1', 'auc')}
    support = res[0]['support']
    pc = ['| Class | Support (test) | Precision | Recall | F1 | AUROC |', '|---|---|---|---|---|---|']
    with open(os.path.join(out_dir, 'per_class.csv'), 'w') as f:
        f.write('class;support;' + ';'.join(f'{s}_mean;{s}_std' for s in ('precision', 'recall', 'f1', 'auroc')) + '\n')
        for c, name in enumerate(classes):
            vals = [stat[s][:, c] for s in ('prec', 'rec', 'f1', 'auc')]
            f.write(f'{name};{support[c]};' + ';'.join(f'{v.mean():.4f};{v.std(ddof=1):.4f}' for v in vals) + '\n')
            pc.append(f'| {name} | {support[c]} | ' + ' | '.join(pm(v) for v in vals) + ' |')

    # 3. Confusion matrix summed over runs
    cm = sum(m['cm'] for m in res)
    cmn = cm / cm.sum(1, keepdims=True)
    np.savetxt(os.path.join(out_dir, 'confusion_matrix_sum.csv'), cm, fmt='%d', delimiter=';',
               header=';'.join(classes) + '  (rows=true, cols=pred, summed over runs)')
    cmt = ['| true \\ pred | ' + ' | '.join(classes) + ' |', '|' + '---|' * (k + 1)]
    cmt += [f'| **{name}** | ' + ' | '.join(f'**{100 * cmn[i, j]:.0f}**' if i == j else f'{100 * cmn[i, j]:.0f}'
                                            for j in range(k)) + ' |' for i, name in enumerate(classes)]
    save_confusion_png(cmn, classes, title or 'Row-normalised confusion (%)',
                       os.path.join(out_dir, 'confusion_matrix.png'))

    # 4. Sanity checks against the official metric files
    ck = ['| Run | max abs diff to np_metrics.json | torch test/Accuracy | first-256 acc | full-test acc |',
          '|---|---|---|---|---|']
    for (r, prob, t), m in zip(runs, res):
        with open(os.path.join(runs_dir, r, 'np_metrics.json')) as f:
            off = json.load(f)
        with open(os.path.join(runs_dir, r, 'torch_metrics.json')) as f:
            tacc = json.load(f).get('test/Accuracy')
        diff = max(abs(m[n] - off[n]) for n in ('Accuracy', 'Balanced accuracy', 'Macro F1', 'Macro AUROC'))
        first = (prob[:256].argmax(1) == t[:256]).mean()
        ck.append(f'| {r} | {diff:.1e} | {tacc:.5f} | {first:.5f} | {m["Accuracy"]:.5f} |')

    md = [f'# {title or "Evaluation report"}', '',
          f'{len(runs)} runs. All numbers in %, computed on the full test set.', '',
          '## Per run and mean ± std (sample std)', '', *rows, '',
          '## Per class (mean ± std over runs)', '', *pc, '',
          '## Confusion matrix, row-normalised (%), summed over runs', '', *cmt, '',
          '## Sanity checks', '', *ck, '',
          'Column 2 ≈ 0: our numbers equal the official sklearn `np_metrics.json`. '
          'Column 3 = column 4: the official torch test metrics only cover the first 256 test images.']
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='folder with 1/ 2/ ... subfolders')
    ap.add_argument('--out', required=True)
    ap.add_argument('--classes', nargs='+', required=True)
    ap.add_argument('--title', default='')
    a = ap.parse_args()
    report(a.runs, a.out, a.classes, a.title)
    with open(os.path.join(a.out, 'summary.md'), encoding='utf-8') as f:
        print(f.read())


if __name__ == '__main__':
    main()
