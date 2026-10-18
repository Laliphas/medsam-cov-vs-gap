# DermaMNIST - MedSAM + GAP + Linear

5 runs. All numbers in %, computed on the full test set.

## Per run and mean ± std (sample std)

| Run | Accuracy | Balanced accuracy | Macro F1 | Macro Precision | Macro AUROC | Micro AUROC |
|---|---|---|---|---|---|---|
| 1 | 46.6 | 45.6 | 29.8 | 32.5 | 81.8 | 80.5 |
| 2 | 46.7 | 45.4 | 29.8 | 32.5 | 81.8 | 80.5 |
| 3 | 46.8 | 45.6 | 29.9 | 32.5 | 81.8 | 80.5 |
| 4 | 46.8 | 45.0 | 29.7 | 32.4 | 81.8 | 80.5 |
| 5 | 46.7 | 46.2 | 30.0 | 32.6 | 81.8 | 80.5 |
| **mean ± std** | **46.7 ± 0.1** | **45.6 ± 0.4** | **29.8 ± 0.1** | **32.5 ± 0.1** | **81.8 ± 0.0** | **80.5 ± 0.0** |

## Per class (mean ± std over runs)

| Class | Support (test) | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| akiec | 66 | 21.7 ± 0.2 | 45.5 ± 0.0 | 29.4 ± 0.2 | 89.5 ± 0.0 |
| bcc | 103 | 26.3 ± 0.2 | 36.7 ± 0.4 | 30.6 ± 0.2 | 87.1 ± 0.0 |
| bkl | 220 | 46.5 ± 0.4 | 18.3 ± 0.2 | 26.2 ± 0.2 | 78.8 ± 0.0 |
| df | 23 | 4.2 ± 0.2 | 47.8 ± 3.1 | 7.7 ± 0.4 | 80.9 ± 0.1 |
| mel | 223 | 31.9 ± 0.1 | 38.6 ± 0.0 | 34.9 ± 0.1 | 67.0 ± 0.0 |
| nv | 1341 | 89.5 ± 0.1 | 52.9 ± 0.2 | 66.5 ± 0.1 | 79.2 ± 0.0 |
| vasc | 29 | 7.4 ± 0.0 | 79.3 ± 0.0 | 13.5 ± 0.0 | 90.1 ± 0.0 |

## Confusion matrix, row-normalised (%), summed over runs

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| **akiec** | **45** | 20 | 0 | 20 | 12 | 0 | 3 |
| **bcc** | 12 | **37** | 2 | 20 | 6 | 14 | 10 |
| **bkl** | 22 | 12 | **18** | 14 | 13 | 11 | 10 |
| **df** | 18 | 21 | 0 | **48** | 13 | 0 | 0 |
| **mel** | 8 | 5 | 4 | 8 | **39** | 20 | 17 |
| **nv** | 2 | 4 | 3 | 13 | 10 | **53** | 16 |
| **vasc** | 3 | 3 | 3 | 3 | 3 | 3 | **79** |

## Sanity checks

| Run | max abs diff to np_metrics.json | torch test/Accuracy | first-256 acc | full-test acc |
|---|---|---|---|---|
| 1 | 1.1e-16 | 0.46875 | 0.46875 | 0.46633 |
| 2 | 2.2e-16 | 0.46875 | 0.46875 | 0.46683 |
| 3 | 2.2e-16 | 0.46875 | 0.46875 | 0.46783 |
| 4 | 0.0e+00 | 0.46094 | 0.46094 | 0.46833 |
| 5 | 5.6e-17 | 0.46875 | 0.46875 | 0.46683 |

Column 2 ≈ 0: our numbers equal the official sklearn `np_metrics.json`. Column 3 = column 4: the official torch test metrics only cover the first 256 test images.
