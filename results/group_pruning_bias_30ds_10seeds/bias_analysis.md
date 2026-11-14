# Stratified vs uniform bias sampling

Folder `results\group_pruning_bias_30ds_10seeds`. Rows analysed: 900 (joint implementation only). Seeds: 10 (0..9). Datasets: 30.
Comparison that counts: stratified vs uniform. Original vs uniform isolates the RNG-policy change. Accuracy differences are in percentage points (pp). Wilcoxon signed-rank on per-dataset mean differences; the test is skipped (n/a) with fewer than 5 non-zero differences. Subgroups of about 10 datasets give very little power: read them as descriptive.

### scorer=none, keep_fraction=1.0

**stratified - uniform** (MAIN: stratification effect)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 81.49 | 81.53 | -0.032 | +0.000 | 8/8/14 | 0.353 (22) | 0.727 | 0.652 | 9/7/14 | 0.200 (23) |
| imbalanced | 10 | 79.35 | 79.45 | -0.097 | -0.087 | 1/1/8 | 0.012 (9) | 0.619 | 0.541 | 3/1/6 | 0.250 (9) |
| small | 10 | 85.86 | 85.90 | -0.039 | +0.000 | 2/5/3 | 0.812 (5) | 0.826 | 0.787 | 3/4/3 | 1.000 (6) |
| rest | 10 | 79.27 | 79.23 | +0.041 | +0.048 | 5/2/3 | 0.461 (8) | 0.736 | 0.627 | 3/2/5 | 0.547 (8) |

**original - uniform** (control: RNG policy only)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 81.59 | 81.53 | +0.061 | +0.000 | 13/6/11 | 0.290 (24) | 0.640 | 0.652 | 12/6/12 | 0.900 (24) |
| imbalanced | 10 | 79.51 | 79.45 | +0.059 | +0.013 | 5/1/4 | 0.426 (9) | 0.555 | 0.541 | 5/1/4 | 0.820 (9) |
| small | 10 | 85.99 | 85.90 | +0.094 | +0.000 | 3/4/3 | 0.562 (6) | 0.787 | 0.787 | 3/4/3 | 1.000 (6) |
| rest | 10 | 79.26 | 79.23 | +0.031 | +0.030 | 5/1/4 | 0.910 (9) | 0.577 | 0.627 | 4/1/5 | 1.000 (9) |

### Per dataset: mean +- std of accuracy (%) across seeds, [min, max]

| dataset | group | seeds | original | stratified | uniform |
|---|---|---:|---|---|---|
| ACSF1 | imbalanced | 10 | 91.70 +- 0.95 [91.0, 93.0] | 91.50 +- 0.71 [91.0, 93.0] | 91.60 +- 0.70 [91.0, 93.0] |
| ArrowHead | small | 10 | 85.71 +- 1.08 [84.0, 87.4] | 86.11 +- 0.85 [85.1, 87.4] | 85.94 +- 0.82 [84.6, 86.9] |
| BeetleFly | small | 10 | 87.50 +- 2.64 [85.0, 90.0] | 88.00 +- 2.58 [85.0, 90.0] | 87.50 +- 2.64 [85.0, 90.0] |
| Chinatown | small | 10 | 98.25 +- 0.00 [98.3, 98.3] | 98.25 +- 0.00 [98.3, 98.3] | 98.25 +- 0.00 [98.3, 98.3] |
| Computers | rest | 10 | 71.96 +- 1.23 [70.8, 74.4] | 72.12 +- 1.99 [67.2, 74.0] | 72.32 +- 1.01 [70.4, 74.0] |
| CricketY | imbalanced | 10 | 82.97 +- 0.64 [82.1, 83.8] | 82.97 +- 0.58 [81.8, 83.6] | 83.08 +- 0.74 [82.3, 84.4] |
| ECG5000 | imbalanced | 10 | 94.49 +- 0.05 [94.4, 94.6] | 94.51 +- 0.05 [94.4, 94.6] | 94.51 +- 0.06 [94.4, 94.6] |
| ECGFiveDays | small | 10 | 100.00 +- 0.00 [100.0, 100.0] | 100.00 +- 0.00 [100.0, 100.0] | 100.00 +- 0.00 [100.0, 100.0] |
| EthanolLevel | rest | 10 | 59.92 +- 1.06 [57.8, 61.2] | 59.42 +- 0.92 [57.8, 60.6] | 59.26 +- 1.02 [57.8, 61.2] |
| FacesUCR | imbalanced | 10 | 95.77 +- 0.17 [95.5, 96.0] | 95.73 +- 0.12 [95.5, 95.9] | 95.79 +- 0.11 [95.7, 96.0] |
| FreezerRegularTrain | rest | 10 | 99.95 +- 0.02 [99.9, 100.0] | 99.96 +- 0.01 [99.9, 100.0] | 99.96 +- 0.01 [99.9, 100.0] |
| Ham | rest | 10 | 70.57 +- 0.83 [68.6, 71.4] | 70.67 +- 1.41 [67.6, 72.4] | 70.38 +- 1.58 [67.6, 72.4] |
| Herring | small | 10 | 65.78 +- 0.49 [65.6, 67.2] | 65.62 +- 0.00 [65.6, 65.6] | 65.62 +- 0.00 [65.6, 65.6] |
| InsectEPGRegularTrain | imbalanced | 10 | 100.00 +- 0.00 [100.0, 100.0] | 100.00 +- 0.00 [100.0, 100.0] | 100.00 +- 0.00 [100.0, 100.0] |
| Lightning2 | small | 10 | 73.77 +- 0.00 [73.8, 73.8] | 73.77 +- 0.00 [73.8, 73.8] | 73.77 +- 0.00 [73.8, 73.8] |
| MiddlePhalanxOutlineAgeGroup | imbalanced | 10 | 60.58 +- 0.62 [59.7, 61.7] | 59.94 +- 0.53 [59.1, 61.0] | 60.19 +- 0.44 [59.7, 61.0] |
| MiddlePhalanxOutlineCorrect | rest | 10 | 84.98 +- 0.79 [84.2, 86.3] | 84.81 +- 0.64 [83.8, 85.6] | 84.71 +- 0.54 [83.8, 85.9] |
| PhalangesOutlinesCorrect | rest | 10 | 83.37 +- 0.48 [82.8, 84.4] | 83.52 +- 0.56 [82.6, 84.6] | 83.26 +- 0.39 [82.5, 83.8] |
| Phoneme | imbalanced | 10 | 28.87 +- 0.49 [27.6, 29.3] | 28.92 +- 0.56 [27.4, 29.4] | 29.00 +- 0.09 [28.9, 29.1] |
| RefrigerationDevices | rest | 10 | 48.27 +- 0.56 [47.2, 49.3] | 48.29 +- 0.84 [47.5, 49.9] | 48.51 +- 0.68 [47.2, 49.9] |
| Rock | small | 10 | 81.40 +- 1.35 [80.0, 84.0] | 80.80 +- 1.69 [78.0, 84.0] | 80.80 +- 1.03 [80.0, 82.0] |
| SemgHandMovementCh2 | imbalanced | 10 | 68.29 +- 1.27 [65.8, 70.4] | 67.78 +- 1.52 [65.6, 70.2] | 68.07 +- 1.27 [66.2, 70.4] |
| SmallKitchenAppliances | rest | 10 | 82.85 +- 0.47 [81.9, 83.2] | 83.15 +- 0.81 [81.9, 84.8] | 83.23 +- 0.51 [82.4, 84.0] |
| SonyAIBORobotSurface1 | small | 10 | 89.15 +- 0.39 [88.7, 89.9] | 89.10 +- 0.29 [88.5, 89.5] | 89.22 +- 0.56 [88.4, 90.0] |
| ToeSegmentation2 | small | 10 | 91.85 +- 0.40 [91.5, 92.3] | 91.54 +- 0.81 [90.0, 92.3] | 91.92 +- 0.83 [90.8, 93.1] |
| Trace | rest | 10 | 100.00 +- 0.00 [100.0, 100.0] | 100.00 +- 0.00 [100.0, 100.0] | 100.00 +- 0.00 [100.0, 100.0] |
| UWaveGestureLibraryAll | imbalanced | 10 | 97.22 +- 0.06 [97.1, 97.3] | 97.23 +- 0.09 [97.1, 97.4] | 97.19 +- 0.09 [97.0, 97.3] |
| Wine | small | 10 | 86.48 +- 1.52 [85.2, 88.9] | 85.37 +- 2.04 [83.3, 88.9] | 85.93 +- 1.99 [83.3, 88.9] |
| Worms | imbalanced | 10 | 75.19 +- 1.29 [72.7, 76.6] | 74.94 +- 2.04 [70.1, 76.6] | 75.06 +- 1.92 [71.4, 76.6] |
| Yoga | rest | 10 | 90.77 +- 0.32 [90.2, 91.1] | 90.81 +- 0.19 [90.4, 91.1] | 90.71 +- 0.51 [90.1, 91.8] |

