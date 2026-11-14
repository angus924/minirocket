# Group pruning results

Rows: 7458. Pruned features differing from `full[:, kept]`: **0** (expected 0).

Ratios compare dataset-mean TEST transform time with the unpruned model. "vs full original" is the combined effect of pruning and joint PPV. Break-even is the median number of series after which the extra training cost (selection) is repaid by cheaper inference.

## Bias strategy: original

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.75 | joint | 1199 | 7946 | 86.07 | -0.01 | 0.811 [0.789, 0.830] | 0.657 | 2.10 | 233 |
| filter | 0.75 | original | 1199 | 7946 | 86.07 | -0.01 | 0.804 [0.788, 0.820] | 0.804 | 2.08 | 197 |
| filter | 0.50 | joint | 799 | 5555 | 86.03 | -0.05 | 0.570 [0.533, 0.613] | 0.463 | 1.09 | 25 |
| filter | 0.50 | original | 799 | 5555 | 86.03 | -0.05 | 0.562 [0.534, 0.596] | 0.562 | 1.09 | 18 |
| filter | 0.30 | joint | 479 | 3538 | 85.68 | -0.40 | 0.384 [0.348, 0.426] | 0.311 | 0.50 | 0 |
| filter | 0.30 | original | 479 | 3538 | 85.68 | -0.40 | 0.370 [0.342, 0.403] | 0.370 | 0.51 | 0 |
| filter | 0.20 | joint | 320 | 2455 | 85.29 | -0.79 | 0.285 [0.256, 0.320] | 0.232 | 0.37 | 0 |
| filter | 0.20 | original | 320 | 2455 | 85.29 | -0.79 | 0.270 [0.247, 0.298] | 0.270 | 0.39 | 0 |
| filter | 0.10 | joint | 160 | 1268 | 84.65 | -1.43 | 0.175 [0.157, 0.196] | 0.142 | 0.25 | 0 |
| filter | 0.10 | original | 160 | 1268 | 84.65 | -1.43 | 0.162 [0.148, 0.179] | 0.162 | 0.27 | 0 |
| none | 1.00 | joint | 1598 | 9996 | 86.08 | +0.00 | 1.000 [1.000, 1.000] | 0.811 | 1.00 | inf |
| none | 1.00 | original | 1598 | 9996 | 86.08 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.75 | joint | 1199 | 7133 | 85.97 | -0.11 | 0.672 [0.645, 0.704] | 0.545 | 4.87 | 1347 |
| ridge | 0.75 | original | 1199 | 7133 | 85.97 | -0.11 | 0.695 [0.677, 0.719] | 0.695 | 4.78 | 1112 |
| ridge | 0.50 | joint | 799 | 4527 | 85.81 | -0.27 | 0.447 [0.427, 0.473] | 0.362 | 4.24 | 1019 |
| ridge | 0.50 | original | 799 | 4527 | 85.81 | -0.27 | 0.463 [0.449, 0.481] | 0.463 | 4.17 | 873 |
| ridge | 0.30 | joint | 479 | 2582 | 85.29 | -0.79 | 0.288 [0.275, 0.303] | 0.233 | 4.66 | 1017 |
| ridge | 0.30 | original | 479 | 2582 | 85.29 | -0.79 | 0.292 [0.281, 0.304] | 0.292 | 4.57 | 825 |
| ridge | 0.20 | joint | 320 | 1654 | 84.78 | -1.30 | 0.212 [0.201, 0.222] | 0.172 | 4.96 | 993 |
| ridge | 0.20 | original | 320 | 1654 | 84.78 | -1.30 | 0.211 [0.202, 0.219] | 0.211 | 4.87 | 889 |
| ridge | 0.10 | joint | 160 | 784 | 83.85 | -2.23 | 0.138 [0.129, 0.145] | 0.112 | 4.99 | 1068 |
| ridge | 0.10 | original | 160 | 784 | 83.85 | -2.23 | 0.132 [0.126, 0.137] | 0.132 | 4.90 | 958 |

