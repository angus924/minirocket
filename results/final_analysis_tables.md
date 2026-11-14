# Final paired tests (computed from committed measurements.csv files)

## 1. Primary run `group_pruning_full` (113 datasets, seeds averaged within dataset)

| keep | scorer | datasets | mean diff pp vs full | wins/ties/losses | Wilcoxon p |
|---:|---|---:|---:|---|---:|
| 0.75 | ridge | 113 | -0.106 | 35/34/44 | 0.2293 |
| 0.75 | filter | 113 | -0.013 | 45/26/42 | 0.9057 |
| 0.5 | ridge | 113 | -0.273 | 25/29/59 | 0.0056 |
| 0.5 | filter | 113 | -0.047 | 39/21/53 | 0.5030 |
| 0.3 | ridge | 113 | -0.790 | 29/24/60 | 0.0000 |
| 0.3 | filter | 113 | -0.403 | 44/13/56 | 0.0482 |
| 0.2 | ridge | 113 | -1.295 | 21/17/75 | 0.0000 |
| 0.2 | filter | 113 | -0.788 | 40/12/61 | 0.0017 |
| 0.1 | ridge | 113 | -2.234 | 16/17/80 | 0.0000 |
| 0.1 | filter | 113 | -1.432 | 34/10/69 | 0.0000 |

**filter - ridge** (same keep fraction; note filter keeps more features at the same group count)

| keep | datasets | mean diff pp | wins/ties/losses | Wilcoxon p |
|---:|---:|---:|---|---:|
| 0.75 | 113 | +0.093 | 49/23/41 | 0.8848 |
| 0.5 | 113 | +0.227 | 52/19/42 | 0.3269 |
| 0.3 | 113 | +0.387 | 56/11/46 | 0.1502 |
| 0.2 | 113 | +0.507 | 56/14/43 | 0.0774 |
| 0.1 | 113 | +0.802 | 59/11/43 | 0.0382 |

## 2. Combined run: stratified - uniform, 24 testable slices, Holm-corrected over all of them

| scorer | keep | subgroup | datasets | mean diff pp | wins/ties/losses | raw p | Holm p |
|---|---:|---|---:|---:|---|---:|---:|
| none | 1.0 | all | 30 | -0.019 | 11/8/11 | 0.7701 | 1.000 |
| ridge | 0.1 | all | 30 | +0.491 | 21/3/6 | 0.0461 | 1.000 |
| ridge | 0.1 | small | 10 | +1.354 | 9/1/0 | 0.0039 | 0.094 |
| ridge | 0.2 | all | 30 | +0.149 | 15/6/9 | 0.4061 | 1.000 |
| ridge | 0.3 | all | 30 | +0.171 | 14/3/13 | 0.4410 | 1.000 |
| ridge | 0.5 | all | 30 | -0.001 | 9/8/13 | 0.9746 | 1.000 |
| ridge | 0.75 | all | 30 | +0.057 | 14/6/10 | 0.3104 | 1.000 |

Raw p < 0.05: 2 of 24; Holm p < 0.05: 0 of 24. Under no effect about 1.2 raw p-values below 0.05 are expected by chance. (The slices are correlated, so Holm is conservative; the point is that isolated nominal hits are not evidence.)

