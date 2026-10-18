# Kvasir-v2 - MedSAM + GAP + Linear

5 runs. All numbers in %, computed on the full test set.

## Per run and mean ± std (sample std)

| Run | Accuracy | Balanced accuracy | Macro F1 | Macro Precision | Macro AUROC | Micro AUROC |
|---|---|---|---|---|---|---|
| 1 | 69.8 | 69.8 | 69.0 | 69.5 | 95.9 | 96.0 |
| 2 | 77.2 | 77.2 | 77.0 | 77.1 | 97.5 | 97.7 |
| 3 | 75.4 | 75.4 | 75.3 | 75.6 | 97.1 | 97.4 |
| 4 | 77.3 | 77.3 | 77.2 | 77.2 | 97.5 | 97.7 |
| 5 | 75.4 | 75.4 | 75.3 | 75.6 | 97.1 | 97.4 |
| **mean ± std** | **75.0 ± 3.1** | **75.0 ± 3.1** | **74.8 ± 3.3** | **75.0 ± 3.2** | **97.0 ± 0.7** | **97.2 ± 0.7** |

## Per class (mean ± std over runs)

| Class | Support (test) | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|---|
| dyed-lifted | 200 | 67.2 ± 1.9 | 58.6 ± 11.9 | 62.1 ± 8.0 | 94.9 ± 1.0 |
| dyed-resect | 200 | 69.3 ± 4.8 | 66.2 ± 3.5 | 67.5 ± 1.6 | 95.6 ± 0.6 |
| esophagitis | 200 | 73.3 ± 4.1 | 66.9 ± 0.2 | 69.9 ± 2.1 | 96.8 ± 0.6 |
| cecum | 200 | 75.7 ± 2.1 | 88.0 ± 1.8 | 81.4 ± 1.5 | 98.2 ± 0.4 |
| pylorus | 200 | 91.0 ± 4.4 | 91.7 ± 3.0 | 91.4 ± 3.6 | 99.4 ± 0.5 |
| z-line | 200 | 66.1 ± 2.5 | 70.5 ± 7.4 | 68.1 ± 4.9 | 96.0 ± 0.8 |
| polyps | 200 | 77.9 ± 3.4 | 78.2 ± 2.0 | 78.0 ± 1.9 | 97.2 ± 0.7 |
| UC | 200 | 79.4 ± 4.5 | 80.2 ± 3.3 | 79.8 ± 3.8 | 98.1 ± 0.8 |

## Confusion matrix, row-normalised (%), summed over runs

| true \ pred | dyed-lifted | dyed-resect | esophagitis | cecum | pylorus | z-line | polyps | UC |
|---|---|---|---|---|---|---|---|---|
| **dyed-lifted** | **59** | 26 | 0 | 4 | 0 | 1 | 6 | 4 |
| **dyed-resect** | 25 | **66** | 0 | 3 | 0 | 0 | 2 | 4 |
| **esophagitis** | 0 | 0 | **67** | 0 | 0 | 32 | 0 | 0 |
| **cecum** | 0 | 1 | 0 | **88** | 0 | 0 | 6 | 4 |
| **pylorus** | 0 | 0 | 2 | 0 | **92** | 3 | 1 | 2 |
| **z-line** | 0 | 2 | 22 | 0 | 5 | **70** | 1 | 0 |
| **polyps** | 3 | 0 | 0 | 10 | 2 | 0 | **78** | 6 |
| **UC** | 0 | 1 | 0 | 11 | 2 | 1 | 6 | **80** |

## Sanity checks

| Run | max abs diff to np_metrics.json | torch test/Accuracy | first-256 acc | full-test acc |
|---|---|---|---|---|
| 1 | 0.0e+00 | 0.44531 | 0.44531 | 0.69812 |
| 2 | 1.1e-16 | 0.64062 | 0.64062 | 0.77187 |
| 3 | 0.0e+00 | 0.65234 | 0.65234 | 0.75438 |
| 4 | 0.0e+00 | 0.64453 | 0.64453 | 0.77312 |
| 5 | 1.1e-16 | 0.65234 | 0.65234 | 0.75438 |

Column 2 ≈ 0: our numbers equal the official sklearn `np_metrics.json`. Column 3 = column 4: the official torch test metrics only cover the first 256 test images.
