# Group pruning results

Rows: 300. Pruned features differing from `full[:, kept]`: **0** (expected 0).

Ratios compare dataset-mean TEST transform time with the unpruned model. "vs full original" is the combined effect of pruning and joint PPV. Break-even is the median number of series after which the extra training cost (selection) is repaid by cheaper inference.

## Bias strategy: original

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.50 | joint | 756 | 5682 | 89.33 | -0.68 | 0.536 [0.516, 0.670] | 0.508 | 0.96 | 69 |
| filter | 0.50 | original | 756 | 5682 | 89.33 | -0.68 | 0.536 [0.517, 0.661] | 0.536 | 0.96 | 53 |
| filter | 0.20 | joint | 302 | 2440 | 88.38 | -1.62 | 0.224 [0.203, 0.366] | 0.212 | 0.87 | 0 |
| filter | 0.20 | original | 302 | 2440 | 88.38 | -1.62 | 0.225 [0.206, 0.349] | 0.225 | 0.88 | 0 |
| none | 1.00 | joint | 1512 | 9996 | 90.00 | +0.00 | 1.000 [1.000, 1.000] | 0.948 | 1.00 | inf |
| none | 1.00 | original | 1512 | 9996 | 90.00 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.50 | joint | 756 | 5117 | 89.10 | -0.91 | 0.560 [0.477, 0.571] | 0.531 | 2.83 | 812 |
| ridge | 0.50 | original | 756 | 5117 | 89.10 | -0.91 | 0.545 [0.474, 0.559] | 0.545 | 2.76 | 612 |
| ridge | 0.20 | joint | 302 | 1828 | 88.59 | -1.42 | 0.231 [0.222, 0.231] | 0.219 | 4.21 | 825 |
| ridge | 0.20 | original | 302 | 1828 | 88.59 | -1.42 | 0.224 [0.219, 0.230] | 0.224 | 4.09 | 615 |

## Bias strategy: stratified

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.50 | joint | 756 | 5766 | 89.07 | -1.04 | 0.543 [0.523, 0.675] | 0.508 | 0.99 | 48 |
| filter | 0.50 | original | 756 | 5766 | 89.07 | -1.04 | 0.555 [0.535, 0.677] | 0.555 | 0.99 | 44 |
| filter | 0.20 | joint | 302 | 2491 | 89.20 | -0.91 | 0.226 [0.203, 0.380] | 0.212 | 0.91 | 29 |
| filter | 0.20 | original | 302 | 2491 | 89.20 | -0.91 | 0.225 [0.204, 0.368] | 0.225 | 0.91 | 26 |
| none | 1.00 | joint | 1512 | 9996 | 90.11 | +0.00 | 1.000 [1.000, 1.000] | 0.936 | 1.00 | inf |
| none | 1.00 | original | 1512 | 9996 | 90.11 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.50 | joint | 756 | 5154 | 88.70 | -1.41 | 0.543 [0.472, 0.553] | 0.508 | 2.90 | 767 |
| ridge | 0.50 | original | 756 | 5154 | 88.70 | -1.41 | 0.534 [0.497, 0.557] | 0.534 | 2.83 | 636 |
| ridge | 0.20 | joint | 302 | 1882 | 88.48 | -1.62 | 0.239 [0.224, 0.244] | 0.224 | 4.31 | 818 |
| ridge | 0.20 | original | 302 | 1882 | 88.48 | -1.62 | 0.233 [0.225, 0.243] | 0.233 | 4.19 | 648 |

## Bias strategy: uniform

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.50 | joint | 756 | 5667 | 88.99 | -0.95 | 0.525 [0.504, 0.681] | 0.500 | 0.97 | 58 |
| filter | 0.50 | original | 756 | 5667 | 88.99 | -0.95 | 0.528 [0.506, 0.679] | 0.528 | 0.97 | 47 |
| filter | 0.20 | joint | 302 | 2476 | 89.37 | -0.57 | 0.224 [0.203, 0.375] | 0.214 | 0.84 | 0 |
| filter | 0.20 | original | 302 | 2476 | 89.37 | -0.57 | 0.226 [0.206, 0.360] | 0.226 | 0.85 | 0 |
| none | 1.00 | joint | 1512 | 9996 | 89.94 | +0.00 | 1.000 [1.000, 1.000] | 0.954 | 1.00 | inf |
| none | 1.00 | original | 1512 | 9996 | 89.94 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.50 | joint | 756 | 5123 | 88.30 | -1.64 | 0.528 [0.499, 0.587] | 0.504 | 2.98 | 876 |
| ridge | 0.50 | original | 756 | 5123 | 88.30 | -1.64 | 0.525 [0.497, 0.562] | 0.525 | 2.90 | 673 |
| ridge | 0.20 | joint | 302 | 1840 | 87.93 | -2.01 | 0.213 [0.210, 0.244] | 0.204 | 3.93 | 840 |
| ridge | 0.20 | original | 302 | 1840 | 87.93 | -2.01 | 0.216 [0.213, 0.244] | 0.216 | 3.81 | 637 |

