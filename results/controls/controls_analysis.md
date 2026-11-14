# Controls: does choosing the groups matter?

Datasets: 113 (seeds 0-2 averaged within each). Full-model accuracy 86.08%; the controls run reproduced the primary full-model accuracy on 113/113 datasets (same parameters).
Time = TEST transform (joint PPV) relative to the unpruned model timed in the same session.
`random` keeps the same number of groups as the pruned models; `small_ridge` / `small_filter` are fresh, smaller MiniRocket models with the feature count of the ridge / filter model (rounded down to a multiple of 84).

## Accuracy and time per method

| keep | method | features | accuracy % | delta vs full pp | time ratio |
|---:|---|---:|---:|---:|---:|
| 0.75 | ridge | 7,133 | 85.97 | -0.11 | 0.672 |
| 0.75 | filter | 7,946 | 86.07 | -0.01 | 0.811 |
| 0.75 | random | 7,509 | 85.97 | -0.11 | 0.772 |
| 0.75 | small_ridge | 7,091 | 85.98 | -0.10 | 0.789 |
| 0.75 | small_filter | 7,905 | 86.00 | -0.08 | 0.861 |
| 0.5 | ridge | 4,527 | 85.81 | -0.27 | 0.447 |
| 0.5 | filter | 5,555 | 86.03 | -0.05 | 0.570 |
| 0.5 | random | 5,017 | 85.89 | -0.19 | 0.541 |
| 0.5 | small_ridge | 4,485 | 85.82 | -0.26 | 0.603 |
| 0.5 | small_filter | 5,514 | 86.02 | -0.06 | 0.683 |
| 0.3 | ridge | 2,582 | 85.29 | -0.79 | 0.288 |
| 0.3 | filter | 3,538 | 85.68 | -0.40 | 0.384 |
| 0.3 | random | 3,032 | 85.51 | -0.57 | 0.363 |
| 0.3 | small_ridge | 2,540 | 85.64 | -0.44 | 0.426 |
| 0.3 | small_filter | 3,494 | 85.68 | -0.40 | 0.515 |
| 0.2 | ridge | 1,654 | 84.78 | -1.30 | 0.212 |
| 0.2 | filter | 2,455 | 85.29 | -0.79 | 0.285 |
| 0.2 | random | 2,024 | 85.20 | -0.88 | 0.267 |
| 0.2 | small_ridge | 1,612 | 85.20 | -0.88 | 0.294 |
| 0.2 | small_filter | 2,415 | 85.43 | -0.65 | 0.393 |
| 0.1 | ridge | 784 | 83.85 | -2.23 | 0.138 |
| 0.1 | filter | 1,268 | 84.65 | -1.43 | 0.175 |
| 0.1 | random | 1,015 | 84.58 | -1.50 | 0.175 |
| 0.1 | small_ridge | 741 | 84.47 | -1.61 | 0.148 |
| 0.1 | small_filter | 1,223 | 84.86 | -1.22 | 0.216 |

## Paired tests (accuracy, method A minus method B)

| keep | A - B | mean diff pp | wins/ties/losses | Wilcoxon p |
|---:|---|---:|---|---:|
| 0.75 | ridge - random | +0.001 | 51/26/36 | 0.2730 |
| 0.75 | filter - random | +0.093 | 53/27/33 | 0.1617 |
| 0.75 | ridge - small_ridge | -0.011 | 47/24/42 | 0.6395 |
| 0.75 | filter - small_filter | +0.064 | 52/28/33 | 0.1107 |
| 0.5 | ridge - random | -0.085 | 46/23/44 | 0.7460 |
| 0.5 | filter - random | +0.142 | 57/17/39 | 0.2175 |
| 0.5 | ridge - small_ridge | -0.018 | 48/19/46 | 0.6977 |
| 0.5 | filter - small_filter | +0.011 | 46/18/49 | 0.8571 |
| 0.3 | ridge - random | -0.219 | 43/23/47 | 0.3232 |
| 0.3 | filter - random | +0.168 | 57/12/44 | 0.2439 |
| 0.3 | ridge - small_ridge | -0.353 | 46/16/51 | 0.1231 |
| 0.3 | filter - small_filter | -0.005 | 47/18/48 | 0.6790 |
| 0.2 | ridge - random | -0.415 | 45/16/52 | 0.0410 |
| 0.2 | filter - random | +0.093 | 52/13/48 | 0.4504 |
| 0.2 | ridge - small_ridge | -0.412 | 38/16/59 | 0.0296 |
| 0.2 | filter - small_filter | -0.135 | 47/12/54 | 0.5013 |
| 0.1 | ridge - random | -0.738 | 46/8/59 | 0.0350 |
| 0.1 | filter - random | +0.064 | 56/11/46 | 0.6060 |
| 0.1 | ridge - small_ridge | -0.628 | 52/9/52 | 0.1445 |
| 0.1 | filter - small_filter | -0.212 | 50/9/54 | 0.3605 |

20 tests, unadjusted. With 20 tests about 1 p < 0.05 is expected by chance; read patterns across keep levels.

