# Kvasir-v2 - MedSAM + covariance + SPDNet

5 runs. All numbers in %, computed on the full test set.

## Per run and mean ± std (sample std)

| Run | Accuracy | Balanced accuracy | Macro F1 | Macro Precision | Macro AUROC | Micro AUROC |
|---|---|---|---|---|---|---|
| 1 | 85.9 | 85.9 | 85.8 | 86.1 | 98.9 | 99.0 |
| 2 | 84.4 | 84.4 | 84.4 | 84.6 | 98.6 | 98.8 |
| 3 | 85.1 | 85.1 | 85.1 | 85.7 | 98.8 | 98.9 |
| 4 | 86.4 | 86.4 | 86.3 | 86.5 | 98.8 | 99.0 |
| 5 | 84.7 | 84.7 | 84.7 | 84.7 | 98.7 | 98.9 |
| **mean ± std** | **85.3 ± 0.8** | **85.3 ± 0.8** | **85.2 ± 0.8** | **85.5 ± 0.8** | **98.8 ± 0.1** | **98.9 ± 0.1** |

## Per class (mean ± std over runs)

| Class | Support (test) | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| dyed-lifted | 200 | 77.1 ± 4.1 | 76.5 ± 6.9 | 76.5 ± 2.3 | 97.8 ± 0.2 |
| dyed-resect | 200 | 79.9 ± 4.5 | 76.8 ± 6.2 | 78.0 ± 1.6 | 98.1 ± 0.2 |
| esophagitis | 200 | 81.0 ± 4.2 | 74.8 ± 4.1 | 77.6 ± 1.2 | 97.9 ± 0.1 |
| cecum | 200 | 91.6 ± 1.6 | 94.5 ± 1.7 | 93.0 ± 0.5 | 99.7 ± 0.1 |
| pylorus | 200 | 96.8 ± 1.2 | 96.3 ± 0.7 | 96.5 ± 0.5 | 99.9 ± 0.0 |
| z-line | 200 | 75.9 ± 2.1 | 82.5 ± 4.5 | 79.0 ± 1.8 | 97.9 ± 0.2 |
| polyps | 200 | 91.0 ± 2.1 | 88.5 ± 2.3 | 89.7 ± 0.9 | 99.1 ± 0.1 |
| UC | 200 | 90.7 ± 2.9 | 92.5 ± 2.6 | 91.5 ± 0.7 | 99.6 ± 0.1 |

## Confusion matrix, row-normalised (%), summed over runs

| true \ pred | dyed-lifted | dyed-resect | esophagitis | cecum | pylorus | z-line | polyps | UC |
|---|---|---|---|---|---|---|---|---|
| **dyed-lifted** | **76** | 19 | 0 | 0 | 0 | 0 | 3 | 1 |
| **dyed-resect** | 22 | **77** | 0 | 1 | 0 | 0 | 0 | 1 |
| **esophagitis** | 0 | 0 | **75** | 0 | 0 | 25 | 0 | 0 |
| **cecum** | 0 | 0 | 0 | **94** | 0 | 0 | 2 | 3 |
| **pylorus** | 0 | 0 | 2 | 0 | **96** | 2 | 0 | 0 |
| **z-line** | 0 | 0 | 16 | 0 | 1 | **82** | 1 | 0 |
| **polyps** | 1 | 0 | 0 | 4 | 1 | 0 | **88** | 4 |
| **UC** | 0 | 1 | 0 | 3 | 1 | 0 | 3 | **92** |

## Sanity checks

| Run | max abs diff to np_metrics.json | torch test/Accuracy | first-256 acc | full-test acc |
|---|---|---|---|---|
| 1 | 1.1e-16 | 0.80859 | 0.80859 | 0.85875 |
| 2 | 0.0e+00 | 0.71875 | 0.71875 | 0.84437 |
| 3 | 1.1e-16 | 0.82812 | 0.82812 | 0.85125 |
| 4 | 0.0e+00 | 0.76562 | 0.76562 | 0.86375 |
| 5 | 1.1e-16 | 0.73828 | 0.73828 | 0.84688 |

Column 2 ≈ 0: our numbers equal the official sklearn `np_metrics.json`. Column 3 = column 4: the official torch test metrics only cover the first 256 test images.
