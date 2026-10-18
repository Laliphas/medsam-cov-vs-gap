"""Encoder check: load frozen MedSAM ViT-B via the official code path and trace it.

    python checks/phase0_encoder_trace.py --model-dir <models> --derma <dermamnist_64.npz> --kvasir <img.jpg>

Prints parameter counts and frozen state, tensor shapes from patch_embed to the neck output
(256, 64, 64), and the forward time per image (estimate of the extraction cost).
Output from our run: checks/outputs/phase0_encoder_trace.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import repo  # noqa: F401,E402  (official code on sys.path)

import numpy as np
import torch


def load_derma_image(npz_path: str, index: int = 0) -> torch.Tensor:
    """Same conversion as medmnist_dataset.initialize_mnist._transform_image:
    uint8 HWC -> /255 -> CHW float."""
    d = np.load(npz_path)
    img = d['train_images'][index]                     # (64, 64, 3) uint8
    lbl = int(d['train_labels'][index][0])
    x = torch.tensor(np.moveaxis(img / 255.0, -1, 0), dtype=torch.float32)
    return x, lbl


def load_rgb(path: str) -> torch.Tensor:
    from PIL import Image
    img = np.asarray(Image.open(path).convert('RGB'))   # HWC uint8
    return torch.tensor(np.moveaxis(img / 255.0, -1, 0), dtype=torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True, help='folder containing medsam_vit_b.pth')
    ap.add_argument('--derma', default=None)
    ap.add_argument('--kvasir', nargs='*', default=[])
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.set_default_device('cuda')                 # main.py does the same
    print(f'torch {torch.__version__} | device {device}'
          + (f' | {torch.cuda.get_device_name(0)}' if device == 'cuda' else ''))

    from common.gve import PretrainedFeatures
    feats = PretrainedFeatures(args.model_dir, 'medsam')   # official loader
    invoker = feats._model                                  # SamInvoker
    sam = invoker.model
    enc = sam.image_encoder

    # ---------------- 0.6 parameter counts / frozen check -----------------
    n_all = sum(p.numel() for p in sam.parameters())
    n_enc = sum(p.numel() for p in enc.parameters())
    n_pe = sum(p.numel() for p in sam.prompt_encoder.parameters())
    n_md = sum(p.numel() for p in sam.mask_decoder.parameters())
    print('\n[0.6] parameters')
    print(f'  whole SAM      : {n_all/1e6:8.2f} M')
    print(f'  image_encoder  : {n_enc/1e6:8.2f} M  ({100*n_enc/n_all:.1f} %)')
    print(f'  prompt_encoder : {n_pe/1e6:8.2f} M')
    print(f'  mask_decoder   : {n_md/1e6:8.2f} M')
    print(f'  sam.training   : {sam.training}  (eval mode set in pretrained/medsam.py)')
    print(f'  requires_grad on encoder params: {any(p.requires_grad for p in enc.parameters())}'
          '  <- True, but PretrainedFeatures wraps the call in torch.no_grad()')
    print(f'  pixel_mean/std stored on SAM (NOT used by this pipeline): '
          f'{sam.pixel_mean.flatten().tolist()} / {sam.pixel_std.flatten().tolist()}')

    # ---------------- 0.5 hooks -------------------------------------------
    trace = []

    def hook(name):
        def f(_m, _inp, out):
            trace.append((name, tuple(out.shape), out.dtype,
                          float(out.float().mean()), float(out.float().std())))
        return f

    targets = {
        'patch_embed (Conv16x16,s16)': enc.patch_embed,
        'blocks[0]  (window attn)': enc.blocks[0],
        'blocks[2]  (GLOBAL attn)': enc.blocks[2],
        'blocks[11] (GLOBAL attn, last)': enc.blocks[11],
        'neck[0]    (Conv1x1 768->256)': enc.neck[0],
        'neck[1]    (LayerNorm2d)': enc.neck[1],
        'neck[2]    (Conv3x3 256->256)': enc.neck[2],
        'neck[3]    (LayerNorm2d) = OUTPUT': enc.neck[3],
    }
    handles = [m.register_forward_hook(hook(n)) for n, m in targets.items()]
    print('\n  window_size per block:', [b.window_size for b in enc.blocks])
    print('  pos_embed shape       :', tuple(enc.pos_embed.shape))

    def run(name, x):
        trace.clear()
        # also record the tensor right after the official preprocessing
        pre = invoker._transform(x.unsqueeze(0).to(device))
        t_times = []
        out = None
        for _ in range(args.repeats):
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = feats(x.to(device))                    # official call path
            if device == 'cuda':
                torch.cuda.synchronize()
            t_times.append(time.perf_counter() - t0)
        print(f'\n=== {name} ===')
        print(f'  input (C,H,W)            : {tuple(x.shape)}  range [{x.min():.3f}, {x.max():.3f}]')
        print(f'  after transform_demo     : {tuple(pre.shape)}  per-channel min {pre.amin((0,2,3)).tolist()}'
              f' max {pre.amax((0,2,3)).tolist()}')
        seen = set()
        for n, s, dt, mu, sd in trace:                    # first forward only
            if n in seen:
                continue
            seen.add(n)
            print(f'  {n:34s} -> {str(s):22s} mean {mu:+.3f} std {sd:.3f}')
        print(f'  PretrainedFeatures output: {tuple(out.shape)}')
        print(f'  forward time             : {1000*min(t_times):.0f} ms (best of {args.repeats})')
        if device == 'cuda':
            print(f'  peak VRAM                : {torch.cuda.max_memory_allocated()/1024**3:.2f} GB')
        return out, min(t_times)

    torch.manual_seed(0)
    run('random input (3,64,64)', torch.rand(3, 64, 64))
    t_ref = None
    if args.derma:
        x, lbl = load_derma_image(args.derma)
        _, t_ref = run(f'DermaMNIST train[0] (label {lbl})', x)
    for p in args.kvasir:
        run(f'Kvasir {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)[:8]}', load_rgb(p))

    for h in handles:
        h.remove()

    if t_ref is not None:
        print('\n[0.4] extraction-time estimate at this speed (batch size 1, as in the official code):')
        print(f'  DermaMNIST 10,015 imgs : {10015*t_ref/3600:.1f} h')
        print(f'  Kvasir-v2   8,000 imgs : {8000*t_ref/3600:.1f} h')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
