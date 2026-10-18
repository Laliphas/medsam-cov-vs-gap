"""Read-only checks of what the official code does (nothing is modified).

    python checks/phase1_checks.py --model-dir <models> --derma <dermamnist_64.npz> --kvasir <img.jpg>

1.2 preprocessing: official per-channel min-max vs MedSAM's global min-max
1.3 cov_descr divides by N+1 (paper: N-1)
1.6 MedSAM covariances have rank 255: the null vector is parallel to 1/gamma of the final LayerNorm2d
1.4 BasicSPDNet shapes, parameters, MixOptimizer groups (Stiefel retraction)
1.5 class weights of the loss
Output from our run: checks/outputs/phase1_checks.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import repo  # noqa: F401,E402  (official code on sys.path)

import numpy as np
import torch

torch.set_printoptions(precision=4, sci_mode=True)


def section(t):
    print('\n' + '=' * 72 + f'\n{t}\n' + '=' * 72)


def load_derma(npz, idx=0):
    d = np.load(npz)
    img = d['train_images'][idx]
    return torch.tensor(np.moveaxis(img / 255.0, -1, 0), dtype=torch.float32), d


def load_rgb(p):
    from PIL import Image
    a = np.asarray(Image.open(p).convert('RGB'))
    return torch.tensor(np.moveaxis(a / 255.0, -1, 0), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--derma', required=True)
    ap.add_argument('--kvasir', required=True)
    args = ap.parse_args()

    from common import cov_descr
    from common.gve import PretrainedFeatures
    from pretrained.medsam import transform_demo

    feats = PretrainedFeatures(args.model_dir, 'medsam')
    sam = feats._model.model
    enc = sam.image_encoder

    x_derma, dnpz = load_derma(args.derma)
    x_kv = load_rgb(args.kvasir)

    # ------------------------------------------------------------------ 1.2
    section('1.2  Preprocessing: official per-channel min-max vs MedSAM_Inference global min-max')
    tf = transform_demo()                          # ToDtype -> Resize bicubic 1024 -> NormalizeMinMax(start_dim=2)
    for name, x in (('Derma', x_derma), ('Kvasir', x_kv)):
        per_ch = tf(x.unsqueeze(0))                # what the official pipeline feeds MedSAM
        resized = tf.transforms[1](x.unsqueeze(0)) # resize only
        glob = (resized - resized.min()) / (resized.max() - resized.min()).clamp_min(1e-8)
        print(f'{name}: per-channel mean RGB = {per_ch.mean((0,2,3)).tolist()}')
        print(f'{name}: global      mean RGB = {glob.mean((0,2,3)).tolist()}')
        with torch.no_grad():
            f_pc = enc(per_ch)
            f_gl = enc(glob)
        cos = torch.nn.functional.cosine_similarity(f_pc.flatten(), f_gl.flatten(), dim=0)
        c1, c2 = cov_descr(f_pc[0]), cov_descr(f_gl[0])
        print(f'{name}: cosine(features) = {cos:.4f} | rel. diff of covariance = '
              f'{(c1 - c2).norm() / c1.norm():.3f}')

    # ------------------------------------------------------------------ 1.3
    section('1.3  Covariance: official cov_descr vs torch.cov')
    torch.manual_seed(0)
    fr = torch.randn(256, 64, 64, dtype=torch.float64)
    c_off = cov_descr(fr)
    c_ref = torch.cov(fr.flatten(1))               # unbiased: / (N-1)
    ratio = (c_off / c_ref).mean().item()
    print(f'cov_descr shape: {tuple(c_off.shape)} | mean ratio cov_descr/torch.cov = {ratio:.8f}')
    print(f'4095/4097 = {4095/4097:.8f}  -> code divides by N+1, paper says N-1')
    print(f'symmetric: max|C-C^T| = {(c_off - c_off.T).abs().max():.2e}')

    # ------------------------------------------------------------------ 1.6
    section('1.6  Is the MedSAM covariance SPD or only PSD?  (LayerNorm2d hypothesis)')
    gamma = enc.neck[3].weight.detach().double()       # LayerNorm2d affine scale (256,)
    print(f'neck[3].weight (gamma): min {gamma.min():.4f}  max {gamma.max():.4f}  '
          f'any zero: {(gamma.abs() < 1e-8).any().item()}')
    v = (1.0 / gamma)
    v = v / v.norm()
    for name, x in (('Derma train[0]', x_derma), ('Kvasir', x_kv)):
        f = feats(x).double()                          # (256,64,64), official path
        C = cov_descr(f)
        ev, U = torch.linalg.eigh(C)
        print(f'\n{name}:')
        print(f'  smallest 5 eigenvalues : {ev[:5].tolist()}')
        print(f'  largest  3 eigenvalues : {ev[-3:].tolist()}')
        print(f'  ratio  lambda_1 / lambda_2 : {(ev[0] / ev[1]).item():.2e}   (≈0 -> one exact null direction)')
        print(f'  numerical rank (tol 1e-10*max): {(ev > 1e-10 * ev[-1]).sum().item()} / 256')
        print(f'  |cos(eigvec_min, 1/gamma)|    : {(U[:, 0] @ v).abs().item():.6f}  (1.0 = parallel)')
        print(f'  ||C v|| / ||C||               : {((C @ v).norm() / C.norm()).item():.2e}')
        print(f'  eigenvalues < 1e-4 (ReEig threshold): {(ev < 1e-4).sum().item()} / 256')
        # also in float32 (what is actually cached) -> can produce tiny negative eigenvalues
        ev32 = torch.linalg.eigvalsh(cov_descr(f.float()))
        print(f'  float32 smallest eigenvalue   : {ev32[0].item():.3e}')

    # ------------------------------------------------------------------ 1.4
    section('1.4  SPDNet (official BasicSPDNet) shapes, parameters and optimizer groups')
    from nn_lit.spdnets import BasicSPDNet
    from spdnet.optimizers import MixOptimizer
    net = BasicSPDNet(256, 7).double()
    seq = net._spdnet.spd_transform
    shapes = []
    hs = [m.register_forward_hook(lambda m, i, o, n=n: shapes.append((n, tuple(o.shape))))
          for n, m in seq.named_children()]
    X = C.unsqueeze(0).repeat(4, 1, 1)                 # batch of 4 real covariances
    logits = net(X)
    for h in hs:
        h.remove()
    print(f'input batch: {tuple(X.shape)}')
    for n, s in shapes:
        print(f'  {n:8s} -> {s}')
    print(f'  linear   -> {tuple(logits.shape)}  (Linear in={net._spdnet.linear.in_features}, out={net._spdnet.linear.out_features})')
    tot = 0
    for n, p in net.named_parameters():
        print(f'  param {n:40s} {str(tuple(p.shape)):22s} {type(p).__name__:18s} {p.numel():7d}')
        tot += p.numel()
    print(f'  total trainable numbers: {tot}')
    opt = MixOptimizer(net.parameters(), optimizer=torch.optim.Adam, lr=1e-3)
    print(f'  MixOptimizer -> Stiefel (Riemannian SGD): {len(opt.stiefel_parameters)} tensors | '
          f'Adam: {len(opt.other_parameters)} tensors')
    W = net._spdnet.spd_transform.bimap1._W
    print(f'  ||W^T W - I|| before step: {(W[0,0].T @ W[0,0] - torch.eye(128, dtype=W.dtype)).norm():.2e}')
    loss = torch.nn.functional.cross_entropy(net(X), torch.tensor([0, 1, 2, 3]))
    loss.backward(); opt.step()
    print(f'  ||W^T W - I|| after  step: {(W[0,0].T @ W[0,0] - torch.eye(128, dtype=W.dtype)).norm():.2e}'
          '  (QR retraction keeps W on the Stiefel manifold)')

    # ------------------------------------------------------------------ 1.5
    section('1.5  Class weights used by the loss (DermaMNIST train split)')
    from layers.losses import get_classifier_loss
    lbl = dnpz['train_labels'].flatten()
    counts = np.bincount(lbl)
    print(f'split sizes train/val/test: {len(lbl)}/{len(dnpz["val_labels"])}/{len(dnpz["test_labels"])}')
    print(f'train counts per class    : {counts.tolist()}')
    w = (1 / counts) / (1 / counts).sum()
    print(f'loss weight per class     : {np.round(w, 4).tolist()}')
    print(f'majority-class baseline   : acc {counts.max()/counts.sum():.3f}, balanced acc {1/len(counts):.3f}')
    mode, lf = get_classifier_loss('auto_ce_logits', 7, class_proportions=torch.tensor(counts))
    print(f'get_classifier_loss mode  : {mode}')


if __name__ == '__main__':
    main()
