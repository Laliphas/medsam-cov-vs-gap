"""Data step 1: inventory of every Kvasir-v2 image -> splits/kvasir_v2_inventory.csv

    python data_prep/audit_kvasir.py [--root <kvasir-dataset-v2>]

Per image: label (sorted folder names = torchvision ImageFolder order), readable?, size, mode, SHA-1.
Prints class counts, broken files, image sizes per class (possible shortcut) and exact duplicates.
"""
import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'splits', 'kvasir_v2_inventory.csv')
EXTS = {'.jpg', '.jpeg', '.png'}
COLUMNS = ['relpath', 'label', 'label_name', 'ok', 'width', 'height', 'mode', 'n_bytes', 'sha1', 'error']


def audit_one(path):
    """Never raises: a broken file becomes a row with ok=False."""
    try:
        with open(path, 'rb') as f:
            sha1 = hashlib.sha1(f.read()).hexdigest()
        with Image.open(path) as img:
            img.verify()                           # integrity check; the object is unusable afterwards
        with Image.open(path) as img:
            width, height, mode = img.width, img.height, img.mode
        return {'ok': True, 'width': width, 'height': height, 'mode': mode,
                'n_bytes': os.path.getsize(path), 'sha1': sha1, 'error': ''}
    except Exception as e:
        return {'ok': False, 'width': '', 'height': '', 'mode': '', 'n_bytes': '', 'sha1': '',
                'error': f'{type(e).__name__}: {e}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('KVASIR_ROOT', os.path.join(REPO, 'data', 'kvasir-dataset-v2')))
    root = ap.parse_args().root

    # 1. Classes = sorted sub-folders, so label indices are the same on every machine
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print('classes:', class_to_idx)

    # 2. Audit every image
    rows = []
    for cls in classes:
        for fname in sorted(os.listdir(os.path.join(root, cls))):
            if os.path.splitext(fname)[1].lower() in EXTS:
                row = audit_one(os.path.join(root, cls, fname))
                row.update(relpath=f'{cls}/{fname}', label_name=cls, label=class_to_idx[cls])
                rows.append(row)

    # 3. Write CSV, sorted by path so the file is byte-identical on every run
    rows.sort(key=lambda r: r['relpath'])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {OUT}')

    # 4. Summary
    good = [r for r in rows if r['ok']]
    print('\nimages per class:', dict(sorted(Counter(r['label_name'] for r in rows).items())))
    print('unreadable:', [(r['relpath'], r['error']) for r in rows if not r['ok']])
    print('modes:', dict(Counter(r['mode'] for r in good)))
    print('image size per class (if a class has its own resolution, size is a shortcut):')
    for cls in classes:
        sizes = Counter((r['width'], r['height']) for r in good if r['label_name'] == cls)
        print(f'  {cls:24s}', ', '.join(f'{w}x{h}:{n}' for (w, h), n in sizes.most_common()))
    groups = defaultdict(list)
    for r in good:
        groups[r['sha1']].append(r)
    dups = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dups if len({r['label_name'] for r in g}) > 1]
    print(f'exact duplicate groups: {len(dups)} ({sum(map(len, dups))} files), across classes: {len(cross)}')
    for g in dups[:20]:
        print('  ', [r['relpath'] for r in g])


if __name__ == '__main__':
    main()
