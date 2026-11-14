# MiniRocket benchmark results

Original MiniRocket versus joint PPV on **113 datasets**. Features, predictions and accuracy matched in all **1,017 paired comparisons**.

## Reading the timing tables

**Ratio = joint PPV mean time / original mean time.** A ratio of 0.70 means 30% less time; 1.10 means 10% more time. Time reduction is 100 x (1 - ratio), so negative values mean slower.

**95% CI is the confidence interval for the overall time ratio beside it.** We resample 113 dataset pairs with replacement, calculate the ratio of their mean times, and repeat 10,000 times. The middle 95% of those ratios forms the interval. This describes sensitivity to the dataset mix, not the range of individual dataset ratios or a guarantee for future runs.

Times are seconds for a whole split. We average repetitions and fitting seeds within each dataset, then give datasets equal weight when calculating the means. One dataset cannot provide a dataset-bootstrap interval; it is shown as unavailable.

## TRAIN transform time

| Features | Original (s) | Joint PPV (s) | Time ratio [95% CI] | Time reduction |
|---:|---:|---:|---|---:|
| 2,436 | 0.1783 | 0.1272 | 0.713 [0.643, 0.787] | 28.7% |
| 4,956 | 0.2673 | 0.1811 | 0.678 [0.626, 0.745] | 32.2% |
| 9,996 | 0.4543 | 0.2331 | 0.513 [0.484, 0.540] | 48.7% |

## TEST transform time

| Features | Original (s) | Joint PPV (s) | Time ratio [95% CI] | Time reduction |
|---:|---:|---:|---|---:|
| 2,436 | 0.3732 | 0.2839 | 0.761 [0.740, 0.781] | 23.9% |
| 4,956 | 0.5799 | 0.3762 | 0.649 [0.634, 0.670] | 35.1% |
| 9,996 | 0.9679 | 0.5139 | 0.531 [0.520, 0.544] | 46.9% |

## Accuracy and other costs

| Features | Accuracy, both methods | Full pipeline time reduction | Mean worker peak memory, original / joint (MiB) | Peak memory change |
|---:|---:|---:|---:|---:|
| 2,436 | 85.50% | 14.6% | 231.01 / 233.61 | +1.1% |
| 4,956 | 85.94% | 18.7% | 295.70 / 299.63 | +1.3% |
| 9,996 | 86.06% | 31.8% | 409.47 / 416.90 | +1.8% |

Accuracy is the mean dataset TEST accuracy. Full pipeline time includes prepared-input loading, extractor fitting, TRAIN/TEST transforms, classifier fitting and prediction. Memory is the whole worker process peak, including warmup and classification; it is not transform-only allocation. Positive memory change means higher memory use.

## Dataset exceptions

Using a 2% band around equal time:

- **2,436 features:** TEST transforms were slower on 3 of 113 datasets.
- **4,956 features:** TEST transforms were slower on 1 of 113 datasets.
- **9,996 features:** TEST transforms were slower on 0 of 113 datasets.

These are descriptive counts. See [per_dataset.csv](per_dataset.csv) for all dataset results, including TRAIN regressions.

## Scope and details

Both methods use the same supplied TRAIN/TEST splits and classifier pipeline. Learning uses TRAIN only. This previously explored archive on one machine provides hardware-dependent evidence, not a universal speed guarantee or an accuracy improvement.

- [summary.csv](summary.csv): all aggregate values and intervals, including pipeline/memory intervals and sensitivity checks.
- [measurements.csv](measurements.csv): individual cases and timing repetitions.
- [metadata.json](metadata.json): data source, hardware, software and settings.
- [verification.json](verification.json): correctness and file-integrity checks.
- [Benchmark guide](../../benchmarks/README.md): full methodology and instructions.
