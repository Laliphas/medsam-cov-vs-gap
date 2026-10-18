# DermaMNIST - MedSAM + covariance + SPDNet (official code)

5 runs. All numbers in %, computed on the full test set.

## Per run and mean ± std (sample std)

| Run | Accuracy | Balanced accuracy | Macro F1 | Macro Precision | Macro AUROC | Micro AUROC |
|---|---|---|---|---|---|---|
| 1 | 60.4 | 54.5 | 39.9 | 38.0 | 88.5 | 90.3 |
| 2 | 59.2 | 56.1 | 40.6 | 38.3 | 89.0 | 90.0 |
| 3 | 60.0 | 54.7 | 39.7 | 37.8 | 88.3 | 90.0 |
| 4 | 58.4 | 51.4 | 37.0 | 40.2 | 88.7 | 90.3 |
| 5 | 56.5 | 55.5 | 40.2 | 37.6 | 88.5 | 88.5 |
| **mean ± std** | **58.9 ± 1.6** | **54.4 ± 1.8** | **39.5 ± 1.4** | **38.4 ± 1.0** | **88.6 ± 0.3** | **89.8 ± 0.8** |

## Per class (mean ± std over runs)

| Class | Support (test) | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| akiec | 66 | 25.0 ± 4.2 | 51.2 ± 15.5 | 32.3 ± 1.3 | 91.3 ± 0.8 |
| bcc | 103 | 45.5 ± 6.7 | 48.3 ± 11.8 | 45.6 ± 5.2 | 93.4 ± 0.5 |
| bkl | 220 | 47.7 ± 6.2 | 41.7 ± 10.4 | 43.2 ± 5.3 | 84.8 ± 1.2 |
| df | 23 | 8.3 ± 1.6 | 46.1 ± 5.0 | 14.0 ± 2.2 | 88.1 ± 2.1 |
| mel | 223 | 33.1 ± 4.3 | 49.5 ± 2.9 | 39.4 ± 2.4 | 78.9 ± 0.6 |
| nv | 1341 | 93.6 ± 1.4 | 64.2 ± 3.3 | 76.1 ± 1.9 | 88.3 ± 0.8 |
| vasc | 29 | 15.6 ± 3.9 | 80.0 ± 7.9 | 25.8 ± 4.8 | 95.6 ± 0.9 |

## Confusion matrix, row-normalised (%), summed over runs

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| **akiec** | **51** | 18 | 10 | 7 | 8 | 1 | 4 |
| **bcc** | 15 | **48** | 3 | 17 | 6 | 3 | 8 |
| **bkl** | 20 | 6 | **42** | 5 | 15 | 10 | 3 |
| **df** | 17 | 8 | 7 | **46** | 12 | 10 | 1 |
| **mel** | 9 | 2 | 10 | 6 | **50** | 13 | 10 |
| **nv** | 2 | 2 | 5 | 5 | 14 | **64** | 7 |
| **vasc** | 1 | 2 | 0 | 8 | 1 | 8 | **80** |

## Sanity checks

| Run | max abs diff to np_metrics.json | torch test/Accuracy | first-256 acc | full-test acc |
|---|---|---|---|---|
| 1 | 2.2e-16 | 0.56641 | 0.56641 | 0.60449 |
| 2 | 1.1e-16 | 0.58203 | 0.58203 | 0.59152 |
| 3 | 1.1e-16 | 0.58594 | 0.58594 | 0.60000 |
| 4 | 0.0e+00 | 0.55859 | 0.55859 | 0.58354 |
| 5 | 5.6e-17 | 0.54297 | 0.54297 | 0.56509 |

Column 2 ≈ 0: our numbers equal the official sklearn `np_metrics.json`. Column 3 = column 4: the official torch test metrics only cover the first 256 test images.
