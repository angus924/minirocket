# Stratified vs uniform bias sampling

Folder `results\group_pruning_combined_30ds_5seeds`. Rows analysed: 1800 (joint implementation only). Seeds: 5 (0..4). Datasets: 30.
Comparison that counts: stratified vs uniform. Original vs uniform isolates the RNG-policy change. Accuracy differences are in percentage points (pp). Wilcoxon signed-rank on per-dataset mean differences; the test is skipped (n/a) with fewer than 5 non-zero differences. Subgroups of about 10 datasets give very little power: read them as descriptive.

### scorer=none, keep_fraction=1.0

**stratified - uniform** (MAIN: stratification effect)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 81.56 | 81.58 | -0.019 | +0.000 | 11/8/11 | 0.770 (22) | 0.641 | 0.650 | 14/6/10 | 0.663 (24) |
| imbalanced | 10 | 79.50 | 79.45 | +0.044 | +0.000 | 4/2/4 | 0.945 (8) | 0.457 | 0.530 | 6/1/3 | 0.301 (9) |
| small | 10 | 86.07 | 86.00 | +0.065 | +0.000 | 2/5/3 | 0.812 (5) | 0.729 | 0.824 | 4/4/2 | 0.562 (6) |
| rest | 10 | 79.12 | 79.29 | -0.165 | +0.004 | 5/1/4 | 0.734 (9) | 0.736 | 0.596 | 4/1/5 | 0.426 (9) |

### scorer=ridge, keep_fraction=0.75

**stratified - uniform** (MAIN: stratification effect)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 81.60 | 81.54 | +0.057 | +0.000 | 14/6/10 | 0.310 (24) | 0.641 | 0.615 | 13/6/11 | 0.812 (24) |
| imbalanced | 10 | 78.94 | 78.93 | +0.009 | +0.014 | 5/1/4 | 0.734 (9) | 0.598 | 0.636 | 5/1/4 | 0.820 (9) |
| small | 10 | 86.71 | 86.67 | +0.046 | +0.000 | 3/4/3 | 0.438 (6) | 0.561 | 0.493 | 3/4/3 | 0.688 (6) |
| rest | 10 | 79.14 | 79.02 | +0.116 | +0.141 | 6/1/3 | 0.496 (9) | 0.765 | 0.716 | 5/1/4 | 0.820 (9) |

### scorer=ridge, keep_fraction=0.5

**stratified - uniform** (MAIN: stratification effect)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 81.45 | 81.45 | -0.001 | +0.000 | 9/8/13 | 0.975 (22) | 0.910 | 0.837 | 12/5/13 | 0.711 (25) |
| imbalanced | 10 | 78.54 | 78.50 | +0.041 | -0.007 | 3/2/5 | 1.000 (8) | 0.892 | 0.633 | 3/1/6 | 0.250 (9) |
| small | 10 | 86.92 | 87.02 | -0.106 | +0.000 | 2/4/4 | 0.312 (6) | 0.870 | 0.950 | 5/2/3 | 0.547 (8) |
| rest | 10 | 78.88 | 78.82 | +0.063 | +0.000 | 4/2/4 | 0.844 (8) | 0.968 | 0.928 | 4/2/4 | 1.000 (8) |

### scorer=ridge, keep_fraction=0.3

**stratified - uniform** (MAIN: stratification effect)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 81.13 | 80.96 | +0.171 | +0.000 | 14/3/13 | 0.441 (27) | 0.972 | 1.133 | 16/3/11 | 0.400 (27) |
| imbalanced | 10 | 78.01 | 77.73 | +0.272 | -0.040 | 4/1/5 | 0.652 (9) | 1.033 | 0.807 | 4/1/5 | 0.250 (9) |
| small | 10 | 86.94 | 86.65 | +0.284 | +0.106 | 6/1/3 | 0.426 (9) | 1.128 | 1.361 | 5/1/4 | 0.570 (9) |
| rest | 10 | 78.45 | 78.50 | -0.044 | -0.035 | 4/1/5 | 0.734 (9) | 0.754 | 1.229 | 7/1/2 | 0.074 (9) |

### scorer=ridge, keep_fraction=0.2

**stratified - uniform** (MAIN: stratification effect)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 80.60 | 80.45 | +0.149 | +0.011 | 15/6/9 | 0.406 (24) | 1.056 | 1.229 | 12/6/12 | 0.169 (24) |
| imbalanced | 10 | 77.01 | 77.24 | -0.231 | +0.000 | 3/3/4 | 0.469 (7) | 1.039 | 0.963 | 1/1/8 | 0.129 (9) |
| small | 10 | 86.96 | 86.36 | +0.598 | +0.416 | 6/2/2 | 0.148 (8) | 1.367 | 1.651 | 5/3/2 | 0.156 (7) |
| rest | 10 | 77.83 | 77.75 | +0.079 | +0.171 | 6/1/3 | 0.734 (9) | 0.761 | 1.074 | 6/2/2 | 0.109 (8) |

### scorer=ridge, keep_fraction=0.1

**stratified - uniform** (MAIN: stratification effect)

| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses | Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|
| all datasets | 30 | 79.76 | 79.27 | +0.491 | +0.280 | 21/3/6 | 0.046 (27) | 1.363 | 1.593 | 15/3/12 | 0.313 (27) |
| imbalanced | 10 | 75.64 | 75.51 | +0.136 | +0.080 | 6/1/3 | 0.652 (9) | 1.162 | 1.225 | 4/1/5 | 1.000 (9) |
| small | 10 | 86.78 | 85.43 | +1.354 | +0.900 | 9/1/0 | 0.004 (9) | 1.709 | 2.064 | 5/1/4 | 0.734 (9) |
| rest | 10 | 76.85 | 76.86 | -0.016 | +0.061 | 6/1/3 | 0.910 (9) | 1.219 | 1.491 | 6/1/3 | 0.250 (9) |

