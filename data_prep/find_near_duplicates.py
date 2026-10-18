"""Data step 2: find near-duplicate images (e.g. consecutive video frames) with a 256-bit dHash.

    python data_prep/find_near_duplicates.py [--root <kvasir-dataset-v2>]

Kvasir has no patient IDs, so near-duplicates are the best available proxy for leakage.
1. Crop the image centre (x 25-85%, y 15-75%) to skip the black border, text overlay and green box;
   hashing the whole image measured the screen layout, not the tissue.
2. Grayscale 17x16 -> compare each pixel with its right neighbour -> 256 bits.
3. Hamming distance between all pairs (0 = same image, ~128 = unrelated).
Outputs in splits/: kvasir_v2_dhash.csv, kvasir_v2_near_dup_pairs.csv (distance <= 40),
near_dup_examples.png (pairs at several distances, to choose the threshold by eye).
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(REPO, 'splits')
MAX_REPORT = 40      # keep all pairs up to this distance; the grouping threshold is chosen in make_split.py


def dhash(path, size=16):
    """size x size-bit difference hash of the image centre, as 4 uint64."""
    with Image.open(path) as img:
        img.draft('L', (256, 256))                 # fast reduced JPEG decoding
        g = img.convert('L')
        w, h = g.size
        g = g.crop((int(.25 * w), int(.15 * h), int(.85 * w), int(.75 * h)))
        g = g.resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).view('>u8').astype(np.uint64)


def popcount(x):
    return np.unpackbits(np.ascontiguousarray(x).view(np.uint8), axis=-1).sum(-1)


def save_examples(pairs, rows, root, path, th=160):
    show = []
    for t in (0, 5, 10, 15, 20, 25, 30, 40):
        show += [p for p in pairs if t <= p[0] < t + 5][:2]
    if not show:
        return
    sheet = Image.new('RGB', (th * 2 + 10, (th + 22) * len(show)), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (d, i, j) in enumerate(show):
        y = n * (th + 22)
        for col, idx in enumerate((i, j)):
            with Image.open(os.path.join(root, rows[idx]['relpath'])) as im:
                sheet.paste(im.convert('RGB').resize((th, int(th * 0.8))), (col * (th + 10), y + 20))
        draw.text((2, y + 2), f'd={d}  {rows[i]["label_name"]} | {rows[j]["label_name"]}', fill='black')
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('KVASIR_ROOT', os.path.join(REPO, 'data', 'kvasir-dataset-v2')))
    root = ap.parse_args().root
    with open(os.path.join(SPLITS_DIR, 'kvasir_v2_inventory.csv'), encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['ok'] == 'True']

    # 1. Hash every image
    hashes = np.stack([dhash(os.path.join(root, r['relpath'])) for r in rows])     # (N, 4)
    with open(os.path.join(SPLITS_DIR, 'kvasir_v2_dhash.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['relpath', 'label_name', 'dhash'])
        w.writerows([r['relpath'], r['label_name'], ''.join(f'{int(v):016x}' for v in h)] for r, h in zip(rows, hashes))

    # 2. All pairs within MAX_REPORT
    pairs = []
    for i in range(len(rows)):
        d = popcount(hashes[i] ^ hashes[i + 1:])
        pairs += [(int(d[k]), i, i + 1 + int(k)) for k in np.nonzero(d <= MAX_REPORT)[0]]
    pairs.sort()
    with open(os.path.join(SPLITS_DIR, 'kvasir_v2_near_dup_pairs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['distance', 'relpath_a', 'relpath_b', 'class_a', 'class_b', 'same_class'])
        for d, i, j in pairs:
            a, b = rows[i], rows[j]
            w.writerow([d, a['relpath'], b['relpath'], a['label_name'], b['label_name'],
                        a['label_name'] == b['label_name']])

    # 3. Summary per threshold + example sheet
    print('threshold | pairs | cross-class pairs | images with >= 1 pair')
    for t in (0, 5, 10, 20, 30, 40):
        sel = [(i, j) for d, i, j in pairs if d <= t]
        cross = sum(rows[i]['label_name'] != rows[j]['label_name'] for i, j in sel)
        print(f'   <={t:2d}   | {len(sel):5d} | {cross:5d} | {len({x for p in sel for x in p}):5d}')
    same = Counter(rows[i]['label_name'] for d, i, j in pairs
                   if d <= 20 and rows[i]['label_name'] == rows[j]['label_name'])
    print('same-class pairs at distance <= 20:', dict(sorted(same.items())))
    save_examples(pairs, rows, root, os.path.join(SPLITS_DIR, 'near_dup_examples.png'))


if __name__ == '__main__':
    main()
