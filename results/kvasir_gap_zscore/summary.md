# Kvasir-v2 - MedSAM + GAP + Linear (z-score)

5 runs. All numbers in %, computed on the full test set.

## Per run and mean ± std (sample std)

| Run | Accuracy | Balanced accuracy | Macro F1 | Macro Precision | Macro AUROC | Micro AUROC |
|---|---|---|---|---|---|---|
| 1 | 84.4 | 84.4 | 84.4 | 84.5 | 98.7 | 98.9 |
| 2 | 84.9 | 84.9 | 84.8 | 84.8 | 98.7 | 99.0 |
| 3 | 84.5 | 84.5 | 84.4 | 84.5 | 98.7 | 98.9 |
| 4 | 85.4 | 85.4 | 85.4 | 85.4 | 98.7 | 99.0 |
| 5 | 84.8 | 84.8 | 84.7 | 84.7 | 98.7 | 99.0 |
| **mean ± std** | **84.8 ± 0.4** | **84.8 ± 0.4** | **84.8 ± 0.4** | **84.8 ± 0.4** | **98.7 ± 0.0** | **99.0 ± 0.0** |

## Per class (mean ± std over runs)

| Class | Support (test) | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| dyed-lifted | 200 | 75.4 ± 1.8 | 76.7 ± 1.2 | 76.0 ± 0.8 | 97.6 ± 0.0 |
| dyed-resect | 200 | 78.7 ± 1.0 | 73.6 ± 2.5 | 76.0 ± 1.1 | 97.8 ± 0.0 |
| esophagitis | 200 | 77.1 ± 1.4 | 79.5 ± 1.9 | 78.2 ± 0.3 | 97.9 ± 0.0 |
| cecum | 200 | 89.9 ± 1.0 | 94.9 ± 1.5 | 92.3 ± 0.2 | 99.7 ± 0.0 |
| pylorus | 200 | 97.7 ± 0.5 | 97.2 ± 0.7 | 97.4 ± 0.2 | 99.9 ± 0.0 |
| z-line | 200 | 78.3 ± 1.2 | 76.3 ± 2.4 | 77.3 ± 0.8 | 97.9 ± 0.0 |
| polyps | 200 | 89.1 ± 2.0 | 87.5 ± 0.5 | 88.3 ± 0.8 | 99.1 ± 0.0 |
| UC | 200 | 92.2 ± 0.6 | 92.7 ± 1.2 | 92.5 ± 0.4 | 99.6 ± 0.0 |

## Confusion matrix, row-normalised (%), summed over runs

| true \ pred | dyed-lifted | dyed-resect | esophagitis | cecum | pylorus | z-line | polyps | UC |
|---|---|---|---|---|---|---|---|---|
| **dyed-lifted** | **77** | 20 | 0 | 0 | 0 | 0 | 3 | 1 |
| **dyed-resect** | 24 | **74** | 0 | 2 | 0 | 0 | 0 | 0 |
| **esophagitis** | 0 | 0 | **80** | 0 | 0 | 20 | 0 | 0 |
| **cecum** | 0 | 0 | 0 | **95** | 0 | 0 | 4 | 0 |
| **pylorus** | 0 | 0 | 1 | 0 | **97** | 1 | 1 | 0 |
| **z-line** | 0 | 0 | 23 | 0 | 0 | **76** | 0 | 0 |
| **polyps** | 0 | 0 | 0 | 6 | 1 | 0 | **88** | 5 |
| **UC** | 0 | 0 | 0 | 3 | 1 | 0 | 3 | **93** |

## Sanity checks

| Run | max abs diff to np_metrics.json | torch test/Accuracy | first-256 acc | full-test acc |
|---|---|---|---|---|
| 1 | 0.0e+00 | 0.78125 | 0.78125 | 0.84437 |
| 2 | 1.1e-16 | 0.76562 | 0.76562 | 0.84875 |
| 3 | 0.0e+00 | 0.77344 | 0.77344 | 0.84500 |
| 4 | 0.0e+00 | 0.78516 | 0.78516 | 0.85437 |
| 5 | 1.1e-16 | 0.76562 | 0.76562 | 0.84750 |

Column 2 ≈ 0: our numbers equal the official sklearn `np_metrics.json`. Column 3 = column 4: the official torch test metrics only cover the first 256 test images.
