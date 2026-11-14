# Group pruning results

Rows: 2486. Pruned features differing from `full[:, kept]`: **0** (expected 0).

Ratios compare dataset-mean TEST transform time with the unpruned model. "vs full original" is the combined effect of pruning and joint PPV. Break-even is the median number of series after which the extra training cost (selection) is repaid by cheaper inference.

## Bias strategy: original

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.75 | joint | 1199 | 7952 | 86.10 | +0.03 | 0.811 [0.789, 0.829] | 0.654 | 2.07 | 237 |
| filter | 0.75 | original | 1199 | 7952 | 86.10 | +0.03 | 0.804 [0.788, 0.819] | 0.804 | 2.04 | 198 |
| filter | 0.50 | joint | 799 | 5558 | 85.94 | -0.14 | 0.571 [0.529, 0.619] | 0.460 | 1.13 | 23 |
| filter | 0.50 | original | 799 | 5558 | 85.94 | -0.14 | 0.563 [0.534, 0.597] | 0.563 | 1.13 | 16 |
| filter | 0.30 | joint | 479 | 3536 | 85.51 | -0.57 | 0.379 [0.341, 0.423] | 0.306 | 0.53 | 0 |
| filter | 0.30 | original | 479 | 3536 | 85.51 | -0.57 | 0.367 [0.338, 0.401] | 0.367 | 0.54 | 0 |
| filter | 0.20 | joint | 320 | 2457 | 85.20 | -0.88 | 0.285 [0.256, 0.319] | 0.230 | 0.42 | 0 |
| filter | 0.20 | original | 320 | 2457 | 85.20 | -0.88 | 0.270 [0.248, 0.297] | 0.270 | 0.43 | 0 |
| filter | 0.10 | joint | 160 | 1271 | 84.51 | -1.57 | 0.174 [0.156, 0.196] | 0.140 | 0.28 | 0 |
| filter | 0.10 | original | 160 | 1271 | 84.51 | -1.57 | 0.160 [0.146, 0.176] | 0.160 | 0.30 | 0 |
| none | 1.00 | joint | 1598 | 9996 | 86.07 | +0.00 | 1.000 [1.000, 1.000] | 0.806 | 1.00 | inf |
| none | 1.00 | original | 1598 | 9996 | 86.07 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.75 | joint | 1199 | 7060 | 85.92 | -0.15 | 0.657 [0.629, 0.692] | 0.530 | 2.53 | 656 |
| ridge | 0.75 | original | 1199 | 7060 | 85.92 | -0.15 | 0.677 [0.657, 0.704] | 0.677 | 2.49 | 610 |
| ridge | 0.50 | joint | 799 | 4508 | 85.70 | -0.38 | 0.445 [0.424, 0.472] | 0.359 | 2.72 | 530 |
| ridge | 0.50 | original | 799 | 4508 | 85.70 | -0.38 | 0.457 [0.441, 0.478] | 0.457 | 2.67 | 472 |
| ridge | 0.30 | joint | 479 | 2575 | 85.14 | -0.93 | 0.285 [0.272, 0.301] | 0.230 | 3.28 | 594 |
| ridge | 0.30 | original | 479 | 2575 | 85.14 | -0.93 | 0.290 [0.280, 0.302] | 0.290 | 3.22 | 533 |
| ridge | 0.20 | joint | 320 | 1654 | 84.53 | -1.55 | 0.210 [0.200, 0.221] | 0.170 | 3.33 | 578 |
| ridge | 0.20 | original | 320 | 1654 | 84.53 | -1.55 | 0.209 [0.201, 0.218] | 0.209 | 3.27 | 543 |
| ridge | 0.10 | joint | 160 | 776 | 83.72 | -2.35 | 0.138 [0.130, 0.146] | 0.112 | 3.45 | 642 |
| ridge | 0.10 | original | 160 | 776 | 83.72 | -2.35 | 0.132 [0.125, 0.137] | 0.132 | 3.39 | 581 |

