# DermaMNIST - MedSAM + GAP + Linear (z-score)

5 runs. All numbers in %, computed on the full test set.

## Per run and mean ± std (sample std)

| Run | Accuracy | Balanced accuracy | Macro F1 | Macro Precision | Macro AUROC | Micro AUROC |
|---|---|---|---|---|---|---|
| 1 | 54.7 | 56.3 | 39.0 | 36.9 | 87.1 | 86.9 |
| 2 | 57.1 | 56.5 | 39.9 | 38.0 | 87.3 | 87.4 |
| 3 | 56.2 | 56.6 | 39.5 | 37.7 | 87.8 | 87.3 |
| 4 | 56.9 | 54.6 | 39.2 | 37.4 | 87.3 | 87.4 |
| 5 | 56.8 | 55.9 | 39.5 | 37.7 | 87.2 | 87.4 |
| **mean ± std** | **56.3 ± 1.0** | **56.0 ± 0.8** | **39.4 ± 0.4** | **37.5 ± 0.4** | **87.3 ± 0.3** | **87.3 ± 0.2** |

## Per class (mean ± std over runs)

| Class | Support (test) | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| akiec | 66 | 33.0 ± 2.7 | 49.4 ± 7.3 | 39.2 ± 1.4 | 90.8 ± 0.6 |
| bcc | 103 | 36.0 ± 1.3 | 60.6 ± 7.4 | 45.0 ± 2.3 | 92.7 ± 0.5 |
| bkl | 220 | 48.6 ± 1.6 | 45.2 ± 1.6 | 46.8 ± 1.5 | 83.0 ± 0.6 |
| df | 23 | 6.5 ± 0.9 | 44.3 ± 11.3 | 11.3 ± 1.5 | 84.5 ± 1.3 |
| mel | 223 | 31.8 ± 1.2 | 46.9 ± 0.8 | 37.9 ± 1.1 | 76.2 ± 0.9 |
| nv | 1341 | 93.6 ± 0.2 | 59.3 ± 1.3 | 72.6 ± 1.0 | 87.3 ± 0.1 |
| vasc | 29 | 13.3 ± 1.3 | 86.2 ± 4.9 | 22.9 ± 1.8 | 96.7 ± 0.1 |

## Confusion matrix, row-normalised (%), summed over runs

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| **akiec** | **49** | 17 | 10 | 9 | 10 | 0 | 5 |
| **bcc** | 9 | **61** | 2 | 11 | 6 | 3 | 8 |
| **bkl** | 13 | 9 | **45** | 4 | 16 | 9 | 4 |
| **df** | 10 | 24 | 5 | **44** | 12 | 1 | 3 |
| **mel** | 6 | 3 | 12 | 8 | **47** | 13 | 10 |
| **nv** | 1 | 5 | 5 | 8 | 13 | **59** | 9 |
| **vasc** | 0 | 8 | 1 | 1 | 1 | 3 | **86** |

## Sanity checks

| Run | max abs diff to np_metrics.json | torch test/Accuracy | first-256 acc | full-test acc |
|---|---|---|---|---|
| 1 | 5.6e-17 | 0.51953 | 0.51953 | 0.54713 |
| 2 | 0.0e+00 | 0.53125 | 0.53125 | 0.57057 |
| 3 | 5.6e-17 | 0.51953 | 0.51953 | 0.56209 |
| 4 | 0.0e+00 | 0.51562 | 0.51562 | 0.56908 |
| 5 | 0.0e+00 | 0.52344 | 0.52344 | 0.56758 |

Column 2 ≈ 0: our numbers equal the official sklearn `np_metrics.json`. Column 3 = column 4: the official torch test metrics only cover the first 256 test images.
