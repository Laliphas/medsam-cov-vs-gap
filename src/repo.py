"""Repo paths and environment setup. Import this first in every script in src/."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_SRC = os.path.join(REPO, 'third_party', '2026-isbi-mayr', 'src')
SPLIT_CSV = os.path.join(REPO, 'splits', 'kvasir_v2_split.csv')
RESULTS = os.path.join(REPO, 'results')
KVASIR_ROOT = os.environ.get('KVASIR_ROOT', os.path.join(REPO, 'data', 'kvasir-dataset-v2'))

os.environ.setdefault('GEOMSTATS_BACKEND', 'pytorch')   # official code needs the torch backend
if OFFICIAL_SRC not in sys.path:
    sys.path.insert(0, OFFICIAL_SRC)                      # make the patched official modules importable

for _stream in (sys.stdout, sys.stderr):                  # Windows consoles: print '?' instead of crashing
    try:
        _stream.reconfigure(errors='replace')
    except AttributeError:
        pass


def init_torch():
    """Same default device as the official main.py. Returns (device, device name)."""
    import torch
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        return torch.empty(()).device, torch.cuda.get_device_name(0)
    return torch.device('cpu'), 'cpu'
