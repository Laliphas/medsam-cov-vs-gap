"""Data step 3: fixed, leakage-aware train/val/test split -> splits/kvasir_v2_split.csv

    python data_prep/make_split.py

Decisions (fixed before any training; reasons in notebooks/report.ipynb, section 2):
  * 70/10/20 per class (as DermaMNIST 7:1:2), one split, seed 42
  * images with dHash distance <= 20 form one group and always stay in the same split;
    a group whose images carry different labels is dropped
Algorithm: union-find groups -> per class, shuffle groups with the seed -> fill test to 20%,
then val to 10%, rest = train -> assert that no group crosses splits.
The SHA-256 of the CSV is stored in kvasir_v2_split_info.json to prove the split is unchanged.
"""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INV = os.path.join(REPO, 'splits', 'kvasir_v2_inventory.csv')
PAIRS = os.path.join(REPO, 'splits', 'kvasir_v2_near_dup_pairs.csv')
OUT = os.path.join(REPO, 'splits', 'kvasir_v2_split.csv')
INFO = os.path.join(REPO, 'splits', 'kvasir_v2_split_info.json')

SEED = 42
RATIOS = {'test': 0.20, 'val': 0.10}        # train = the rest (0.70)
DUP_THRESHOLD = 20                           # dHash-256 Hamming distance


def sha256_of(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)    # smallest path is the root -> deterministic ids


def main():
    with open(INV, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['ok'] == 'True']
    label_of = {r['relpath']: r['label_name'] for r in rows}
    idx_of = {r['relpath']: int(r['label']) for r in rows}

    # 1. Group near-duplicates
    uf = UnionFind(label_of)
    with open(PAIRS, encoding='utf-8') as f:
        for p in csv.DictReader(f):
            if int(p['distance']) <= DUP_THRESHOLD:
                uf.union(p['relpath_a'], p['relpath_b'])
    groups = defaultdict(list)
    for rp in sorted(label_of):
        groups[uf.find(rp)].append(rp)

    # 2. Drop groups with conflicting labels
    dropped = []
    for gid in [g for g, m in groups.items() if len({label_of[x] for x in m}) > 1]:
        dropped += groups.pop(gid)
    print(f'groups: {len(groups)} (size > 1: {sum(len(m) > 1 for m in groups.values())}), dropped: {dropped}')

    # 3. Split per class, with the group as the unit
    rng = np.random.default_rng(SEED)
    by_class = defaultdict(list)
    for gid in sorted(groups):
        by_class[label_of[gid]].append(gid)
    split_of = {}
    for cls in sorted(by_class):
        gids = by_class[cls]
        n_total = sum(len(groups[g]) for g in gids)
        target = {s: round(r * n_total) for s, r in RATIOS.items()}
        filled = Counter()
        for k in rng.permutation(len(gids)):
            g = gids[k]
            size = len(groups[g])
            dest = next((s for s in ('test', 'val') if filled[s] + size <= target[s]), 'train')
            filled[dest] += size
            for m in groups[g]:
                split_of[m] = (dest, g)

    out_rows = [{'relpath': rp, 'label': idx_of[rp], 'label_name': label_of[rp],
                 'split': split_of[rp][0], 'group_id': split_of[rp][1]} for rp in sorted(split_of)]
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['relpath', 'label', 'label_name', 'split', 'group_id'])
        w.writeheader()
        w.writerows(out_rows)

    # 4. Checks
    g2s = defaultdict(set)
    for r in out_rows:
        g2s[r['group_id']].add(r['split'])
    assert all(len(s) == 1 for s in g2s.values()), 'a duplicate group crosses splits'
    assert len(out_rows) == len(rows) - len(dropped)

    table = Counter((r['label_name'], r['split']) for r in out_rows)
    classes = sorted({r['label_name'] for r in out_rows})
    tot = Counter(r['split'] for r in out_rows)
    print(f'{"class":24s} {"train":>6s} {"val":>5s} {"test":>5s}')
    for c in classes + ['total']:
        n = [tot[s] if c == 'total' else table[(c, s)] for s in ('train', 'val', 'test')]
        print(f'{c:24s} {n[0]:6d} {n[1]:5d} {n[2]:5d}')

    info = {
        'seed': SEED, 'ratios': {'train': 0.70, **RATIOS}, 'dup_threshold_dhash256': DUP_THRESHOLD,
        'n_images': len(out_rows), 'dropped_label_conflict': dropped,
        'counts': {s: tot[s] for s in ('train', 'val', 'test')},
        'per_class': {c: {s: table[(c, s)] for s in ('train', 'val', 'test')} for c in classes},
        'class_to_idx': {c: idx_of[next(r['relpath'] for r in out_rows if r['label_name'] == c)] for c in classes},
        'sha256_inventory': sha256_of(INV), 'sha256_near_dup_pairs': sha256_of(PAIRS),
        'sha256_split_csv': sha256_of(OUT),
    }
    with open(INFO, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f'split CSV sha256: {info["sha256_split_csv"]} (must not change between runs)')


if __name__ == '__main__':
    main()
