# Group pruning results

Rows: 3600. Pruned features differing from `full[:, kept]`: **0** (expected 0).

Ratios compare dataset-mean TEST transform time with the unpruned model. "vs full original" is the combined effect of pruning and joint PPV. Break-even is the median number of series after which the extra training cost (selection) is repaid by cheaper inference.

## Bias strategy: stratified

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| none | 1.00 | joint | 1663 | 9996 | 81.56 | +0.00 | 1.000 [1.000, 1.000] | 0.804 | 1.00 | inf |
| none | 1.00 | original | 1663 | 9996 | 81.56 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.75 | joint | 1247 | 7247 | 81.60 | +0.03 | 0.694 [0.627, 0.783] | 0.559 | 1.41 | 488 |
| ridge | 0.75 | original | 1247 | 7247 | 81.60 | +0.03 | 0.711 [0.663, 0.782] | 0.711 | 1.37 | 449 |
| ridge | 0.50 | joint | 832 | 4585 | 81.45 | -0.12 | 0.475 [0.424, 0.551] | 0.382 | 1.73 | 375 |
| ridge | 0.50 | original | 832 | 4585 | 81.45 | -0.12 | 0.475 [0.433, 0.541] | 0.475 | 1.66 | 356 |
| ridge | 0.30 | joint | 499 | 2635 | 81.13 | -0.43 | 0.307 [0.277, 0.356] | 0.247 | 2.30 | 390 |
| ridge | 0.30 | original | 499 | 2635 | 81.13 | -0.43 | 0.304 [0.274, 0.348] | 0.304 | 2.18 | 386 |
| ridge | 0.20 | joint | 333 | 1683 | 80.60 | -0.97 | 0.222 [0.201, 0.255] | 0.178 | 2.60 | 371 |
| ridge | 0.20 | original | 333 | 1683 | 80.60 | -0.97 | 0.215 [0.197, 0.247] | 0.215 | 2.45 | 360 |
| ridge | 0.10 | joint | 166 | 791 | 79.76 | -1.81 | 0.141 [0.128, 0.158] | 0.114 | 2.95 | 398 |
| ridge | 0.10 | original | 166 | 791 | 79.76 | -1.81 | 0.133 [0.122, 0.149] | 0.133 | 2.76 | 388 |

## Bias strategy: uniform

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| none | 1.00 | joint | 1663 | 9996 | 81.58 | +0.00 | 1.000 [1.000, 1.000] | 0.802 | 1.00 | inf |
| none | 1.00 | original | 1663 | 9996 | 81.58 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.75 | joint | 1247 | 7230 | 81.54 | -0.04 | 0.689 [0.623, 0.774] | 0.553 | 1.41 | 477 |
| ridge | 0.75 | original | 1247 | 7230 | 81.54 | -0.04 | 0.705 [0.660, 0.769] | 0.705 | 1.37 | 460 |
| ridge | 0.50 | joint | 832 | 4602 | 81.45 | -0.14 | 0.475 [0.426, 0.548] | 0.381 | 1.72 | 408 |
| ridge | 0.50 | original | 832 | 4602 | 81.45 | -0.14 | 0.473 [0.435, 0.532] | 0.473 | 1.65 | 343 |
| ridge | 0.30 | joint | 499 | 2640 | 80.96 | -0.62 | 0.306 [0.275, 0.352] | 0.245 | 2.30 | 403 |
| ridge | 0.30 | original | 499 | 2640 | 80.96 | -0.62 | 0.300 [0.276, 0.340] | 0.300 | 2.17 | 382 |
| ridge | 0.20 | joint | 333 | 1682 | 80.45 | -1.14 | 0.222 [0.204, 0.253] | 0.178 | 2.58 | 388 |
| ridge | 0.20 | original | 333 | 1682 | 80.45 | -1.14 | 0.213 [0.197, 0.242] | 0.213 | 2.43 | 371 |
| ridge | 0.10 | joint | 166 | 787 | 79.27 | -2.32 | 0.140 [0.127, 0.156] | 0.112 | 2.93 | 414 |
| ridge | 0.10 | original | 166 | 787 | 79.27 | -2.32 | 0.130 [0.121, 0.145] | 0.130 | 2.74 | 380 |

