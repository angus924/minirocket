# Group pruning results

Rows: 100. Pruned features differing from `full[:, kept]`: **0** (expected 0).

Ratios compare dataset-mean TEST transform time with the unpruned model. "vs full original" is the combined effect of pruning and joint PPV. Break-even is the median number of series after which the extra training cost (selection) is repaid by cheaper inference.

## Bias strategy: original

| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] | ratio (vs full original) | train cost ratio | break-even series |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| filter | 0.50 | joint | 790 | 14729 | 88.59 | -0.31 | 0.600 [0.515, 0.667] | 0.604 | 0.93 | 34 |
| filter | 0.50 | original | 790 | 14729 | 88.59 | -0.31 | 0.604 [0.513, 0.676] | 0.604 | 0.93 | 34 |
| filter | 0.20 | joint | 316 | 7017 | 88.57 | -0.33 | 0.304 [0.224, 0.378] | 0.305 | 0.83 | 0 |
| filter | 0.20 | original | 316 | 7017 | 88.57 | -0.33 | 0.304 [0.223, 0.381] | 0.304 | 0.83 | 0 |
| none | 1.00 | joint | 1579 | 24948 | 88.90 | +0.00 | 1.000 [1.000, 1.000] | 1.006 | 1.00 | inf |
| none | 1.00 | original | 1579 | 24948 | 88.90 | +0.00 | 1.000 [1.000, 1.000] | 1.000 | 1.00 | inf |
| ridge | 0.50 | joint | 790 | 11559 | 89.04 | +0.14 | 0.457 [0.412, 0.497] | 0.460 | 1.48 | 398 |
| ridge | 0.50 | original | 790 | 11559 | 89.04 | +0.14 | 0.458 [0.414, 0.497] | 0.458 | 1.48 | 403 |
| ridge | 0.20 | joint | 316 | 4314 | 89.02 | +0.13 | 0.183 [0.160, 0.205] | 0.184 | 1.95 | 475 |
| ridge | 0.20 | original | 316 | 4314 | 89.02 | +0.13 | 0.187 [0.161, 0.211] | 0.187 | 1.95 | 494 |

