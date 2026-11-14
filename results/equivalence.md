# Bootstrap intervals for accuracy changes (from the committed measurements.csv files)

Seeds averaged per dataset, datasets resampled 10,000 times (generator seed 0).

| Comparison | Datasets | Mean change pp | 90% interval | 95% interval |
|---|---:|---:|---|---|
| filter, keep 75% vs full | 113 | -0.013 | -0.16 to +0.13 | -0.19 to +0.16 |
| filter, keep 50% vs full | 113 | -0.047 | -0.29 to +0.19 | -0.33 to +0.24 |
| filter, keep 30% vs full | 113 | -0.403 | -0.67 to -0.15 | -0.72 to -0.10 |
| filter, keep 20% vs full | 113 | -0.788 | -1.15 to -0.44 | -1.23 to -0.38 |
| filter, keep 10% vs full | 113 | -1.432 | -2.03 to -0.88 | -2.15 to -0.79 |
| ridge, keep 75% vs full | 113 | -0.106 | -0.23 to +0.02 | -0.26 to +0.04 |
| ridge, keep 50% vs full | 113 | -0.273 | -0.47 to -0.08 | -0.51 to -0.05 |
| ridge, keep 30% vs full | 113 | -0.790 | -1.09 to -0.50 | -1.15 to -0.45 |
| ridge, keep 20% vs full | 113 | -1.295 | -1.67 to -0.94 | -1.76 to -0.88 |
| ridge, keep 10% vs full | 113 | -2.234 | -2.83 to -1.69 | -2.97 to -1.61 |
| stratified vs uniform (20 seeds) | 113 | -0.002 | -0.037 to +0.035 | -0.044 to +0.043 |
