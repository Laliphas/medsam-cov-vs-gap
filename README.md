# MedSAM covariance + SPDNet vs GAP: DermaMNIST and Kvasir-v2

Pre-screening task. A frozen **MedSAM** encoder turns each image into a 256 × 64 × 64 feature map, which is summarised in two ways and classified:

* **Covariance + SPDNet**: 256 × 256 covariance of the 4,096 patch vectors, then a Riemannian network (the method of Mayr et al., ISBI 2026, [arXiv 2511.04190](https://arxiv.org/abs/2511.04190))
* **GAP + Linear**: the 256-d mean of the same patch vectors, then a linear layer (baseline)

Tasks:

1. Reproduce MedSAM + covariance + SPDNet on DermaMNIST with the official split.
2. Adapt the method to Kvasir-v2.
3. Compare with a GAP + Linear baseline on the same splits.

## Results

Mean ± sample std over 5 seeded runs, full test set. Training time covers all 5 runs, using cached features.

| Dataset | Head | Balanced acc. | Macro AUROC | Train time | Params |
|---|---|---|---|---|---|
| DermaMNIST | Covariance + SPDNet | 54.4 ± 1.8 | **88.6 ± 0.3** | ~176 min | 50,183 |
| | GAP + z-score + Linear | **56.0 ± 0.8** | 87.3 ± 0.3 | 5 min | 1,799 |
| | GAP + Linear (ablation) | 45.6 ± 0.4 | 81.8 ± 0.0 | 13 min | 1,799 |
| Kvasir-v2 | Covariance + SPDNet | **85.3 ± 0.8** | **98.8 ± 0.1** | 131 min | 51,208 |
| | GAP + z-score + Linear | 84.8 ± 0.4 | 98.7 ± 0.0 | 11 min | 2,056 |
| | GAP + Linear (ablation) | 75.0 ± 3.1 | 97.0 ± 0.7 | 26 min | 2,056 |

Paper, MedSAM + SPDNet on DermaMNIST: 56.3 balanced accuracy, 88.5 AUROC.

**SPDNet − GAP z-score**, paired bootstrap over test images (10,000 resamples, 95% CI), from `python src/compare.py`:

| Dataset | Balanced accuracy | Macro AUROC |
|---|---|---|
| DermaMNIST | −1.5 [−5.1, +2.0] | **+1.3 [+0.2, +2.5]** |
| Kvasir-v2 | +0.5 [−0.9, +1.9] | +0.05 [−0.1, +0.2] |

**Findings**

* Balanced accuracy ties on both datasets.
* The one significant gain for SPDNet (AUROC on DermaMNIST) disappears on Kvasir-v2.
* Scaling the GAP features (z-score) is worth about +10 points. It matters far more than which head is used.
* SPDNet costs 12–35× more training time, 25× more parameters and 256× more storage per image.

The full analysis, with the design decisions and their reasons, figures and the answers to the four questions, is in [`notebooks/report.ipynb`](notebooks/report.ipynb).

## Repository layout

```
data_prep/      Kvasir-v2: audit -> near-duplicate search -> fixed split   (numpy + PIL)
src/            feature extraction, training, evaluation, comparison
  repo.py             paths; puts the patched official code on sys.path
  kvasir_dataset.py   Kvasir-v2 in the MedMNIST item format of the official code
  features.py         MedSAM -> (covariance, GAP, label) mapper + read-only cache reader
  extract_features.py step 1: run MedSAM once, cache both descriptors
  train.py            step 2: SPDNet or GAP head with the official training loop
  reproduce_derma.py  task 1: official pipeline on DermaMNIST
  eval_report.py      per-run / per-class metrics, confusion matrix, sanity checks
  compare.py          paired bootstrap CI + McNemar
checks/         read-only checks of the encoder and the official code (+ outputs)
splits/         the Kvasir-v2 split and its inputs (CSV + SHA-256 in kvasir_v2_split_info.json)
results/        one folder per experiment: summary.md, per_run/per_class CSV, confusion matrix,
                fit_summary.json, runs/<seed>/ predictions + targets
patches/        our changes to the official code (5 small edits, see below)
notebooks/      report.ipynb: reads results/ only, no GPU needed
```

## Setup

```bash
pip install -r requirements.txt          # install torch for your CUDA version first
python setup_third_party.py              # official code @ 6457686 + patches/official_changes.patch
```

You also need:

* the MedSAM checkpoint `medsam_vit_b.pth`, placed in the model folder
* DermaMNIST: downloaded by `medmnist`, 64 × 64 version
* [Kvasir-v2](https://datasets.simula.no/kvasir/): put it in `data/kvasir-dataset-v2/`, or set `KVASIR_ROOT`

On the first run, the official code asks for the cache, dataset, result and model folders and stores them in its `config.json`.

## How to run

All commands are run from the repository root.

```bash
# Task 1: DermaMNIST reproduction (official pipeline, MedSAM only)
python src/reproduce_derma.py

# Task 2: Kvasir-v2 split (already committed in splits/; rerunning gives the same SHA-256)
python data_prep/audit_kvasir.py
python data_prep/find_near_duplicates.py
python data_prep/make_split.py

# Features: one MedSAM pass per image, caching both covariance and GAP
python src/extract_features.py --dataset kvasir
python src/extract_features.py --dataset derma       # adds GAP for DermaMNIST

# Heads (5 seeded runs each; add --debug for a 1-epoch smoke test)
python src/train.py --dataset kvasir --method spdnet
python src/train.py --dataset kvasir --method gap --standardize zscore
python src/train.py --dataset kvasir --method gap
python src/train.py --dataset derma  --method gap --standardize zscore
python src/train.py --dataset derma  --method gap

# Statistics and report
python src/compare.py kvasir_spdnet kvasir_gap_zscore
jupyter notebook notebooks/report.ipynb
```

## Experimental settings

| | |
|---|---|
| Encoder | MedSAM ViT-B image encoder, frozen (`torch.no_grad`); output of the neck, 256 × 64 × 64 |
| Preprocessing | official `transform_demo`: resize to 1024² (bicubic) and per-channel min-max. Kvasir images are resized directly, ignoring aspect ratio, exactly as DermaMNIST |
| Covariance | official `cov_descr`: (Z − mean)(Z − mean)ᵀ / (N + 1) over N = 4,096 patches |
| GAP | mean over the same 4,096 patches, i.e. the mean the covariance subtracts |
| SPDNet | BiMap 256→128→64→32, ReEig (threshold 1e-4), LogEig, Linear |
| GAP head | optional z-score using train-set mean/std (std clamped at 1e-6), then Linear |
| Training | official `fit_spdnet`, the same for both heads. Class-weighted cross-entropy (w ∝ 1/n). Adam (lr 1e-3) for Euclidean weights; Riemannian SGD with QR retraction on the Stiefel manifold for BiMap. Batch 32, max 250 epochs. Early stopping on validation balanced accuracy (patience 15); checkpoint at minimum validation loss. 5 repeats, seeds 1–5 |
| Metrics | balanced accuracy (main), macro F1, macro precision, macro and micro AUROC; mean ± sample std over runs |
| Statistics | paired bootstrap over test images; the exact McNemar test on the 5-run ensemble is exploratory |

The GAP head reuses the official training loop. `train.py` temporarily swaps `BasicSPDNet` for `GapLinear`, so loss, optimiser, early stopping and evaluation are the same code for both heads.

## Data splits

* **DermaMNIST**: official MedMNIST split, 7,007 / 1,003 / 2,005 images, 7 classes, 64 × 64.
* **Kvasir-v2**: 8,000 images in 8 classes; no official split and no patient IDs.
  * **Audit**: SHA-1 per file. One file appears in two classes; that group is dropped.
  * **Near-duplicates**: 256-bit dHash computed on the image centre, so the screen overlay is ignored. Images at distance ≤ 20 are grouped with union-find.
  * **Split**: stratified by class and grouped, so a near-duplicate group never crosses splits. 70 / 10 / 20, seed 42, fixed before training: **5,598 / 800 / 1,600**.
  * **Integrity**: the split file's SHA-256 is `4b477c45…79ad1f`. `extract_features.py` records this hash with the cache.

## Changes to the official code

[`patches/official_changes.patch`](patches/official_changes.patch), applied to commit `6457686`:

| File | Change | Why |
|---|---|---|
| `experiments/spdnet.py` | `pl.seed_everything(i)` per repeat | runs were unseeded |
| `main.py` | `handle_spdnet_gve(..., gves=...)` | run MedSAM only (task 1) |
| `nn_lit/__init__.py` | optional `wandb` import | runs without a wandb account |
| `medmnist_dataset.py` | create the dataset folder | first download failed |
| `setup.py` | `sys.executable -m pip` | package install on Windows |

## Issues found in the official code

* Saved `torch_metrics.json` test values cover **only the first 256 test images**, verified for every run (see the sanity table in each `results/*/summary.md`). All numbers here are recomputed on the full test set, and they match the official sklearn `np_metrics.json`.
* `cov_descr` divides by N + 1; the paper says N − 1. This is a constant scale and does not affect the conclusions.
* MedSAM covariances have **rank 255**. The final LayerNorm2d forces one exact null direction (∥ 1/γ), and about 60 eigenvalues fall below the ReEig threshold, so ReEig is essential ([`checks/outputs/phase1_checks.txt`](checks/outputs/phase1_checks.txt)).
* A finished run is silently reused if its `*_metrics.json` exists, and the feature cache is validated only by its length.
* The official results CSV labels the DermaMNIST row "BreastMNIST" ([`results/official_results_derma.csv`](results/official_results_derma.csv)).

## Limitations

* Only one encoder was tested. The paper reports much stronger SPDNet results with DINOv2.
* One fixed Kvasir split. There are no patient IDs, so frames from one procedure may still cross splits.
* Image size and on-screen overlays correlate with class; either head could use them as shortcuts.
* Both heads use the official default hyperparameters, with no tuning.
