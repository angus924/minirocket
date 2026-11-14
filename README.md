[`ROCKET`](https://github.com/angus924/rocket) &middot; [**`MINIROCKET`**](https://github.com/angus924/minirocket) &middot; [`HYDRA`](https://github.com/angus924/hydra)

# MINIROCKET

***MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification***

[KDD 2021](https://dl.acm.org/doi/abs/10.1145/3447548.3467231) / [arXiv:2012.08791](https://arxiv.org/abs/2012.08791) (preprint)

> <div align="justify">Until recently, the most accurate methods for time series classification were limited by high computational complexity.  ROCKET achieves state-of-the-art accuracy with a fraction of the computational expense of most existing methods by transforming input time series using random convolutional kernels, and using the transformed features to train a linear classifier.  We reformulate ROCKET into a new method, MINIROCKET, making it up to 75 times faster on larger datasets, and making it almost deterministic (and optionally, with additional computational expense, fully deterministic), while maintaining essentially the same accuracy.  Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes.  MINIROCKET is significantly faster than any other method of comparable accuracy (including ROCKET), and significantly more accurate than any other method of even roughly-similar computational expense.  As such, we suggest that MINIROCKET should now be considered and used as the default variant of ROCKET.</div>

Please cite as:

```bibtex
@inproceedings{dempster_etal_2021,
  author    = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title     = {{MiniRocket}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  publisher = {ACM},
  address   = {New York},
  year      = {2021},
  pages     = {248--257}
}
```

## Repository overview

MiniRocket turns time series into features by applying convolutional kernels and
measuring the proportion of responses above fitted thresholds (PPV). A linear
classifier uses those features to predict the class. This repository retains the
original implementation and adds the following extensions for fixed-length
univariate data:

| Part | What it does | Detailed guide |
|---|---|---|
| **Joint PPV** | Counts responses against several thresholds in one pass. It preserves the original features and predictions exactly while reducing pooling work. | [Joint PPV benchmark and setup](./benchmarks/README.md) |
| **Group pruning** | Selects whole kernel/dilation groups using TRAIN data, then skips the convolution and pooling work for removed groups. Retained features stay exact, but using fewer groups can change accuracy. | [Group pruning](./README_GROUP_PRUNING.md#6-module-1-group-pruning) |
| **Bias strategies** | Compares the original threshold fitting with uniform and class-stratified sampling of training examples. This tests whether balancing the classes used to fit thresholds improves accuracy or stability. | [Class-stratified biases](./README_GROUP_PRUNING.md#8-module-3-class-stratified-biases) |
| **Controls** | Compares pruning with random group selection and smaller fresh models to establish where the speed and accuracy differences come from. | [Control experiments](./README_GROUP_PRUNING.md#7-module-2-controls) |

Joint PPV reduces the cost of pooling within each group; pruning reduces the
number of groups computed. They can be used together, with the classifier
retrained on the retained features. Bias strategies change how thresholds are
fitted and can be evaluated with either pooling implementation and with pruning.

Start with the [results](#results) and [combined benchmark commands](#run-the-combined-benchmark)
below. The [joint PPV guide](./benchmarks/README.md) covers its benchmark protocol,
setup and outputs; the [group pruning and bias guide](./README_GROUP_PRUNING.md)
covers algorithms, Python usage, controls and detailed experiments. The original
multivariate and variable-length implementations are listed under [Code](#code).

## Podcast

Hear more about MINIROCKET and time series classification on the [Data Skeptic](https://dataskeptic.com/blog/episodes/2021/minirocket) podcast!

## GPU Implementation \*NEW\*

A GPU implementation of MINIROCKET, developed by Malcolm McLean and Ignacio Oguiza, is available through [`tsai`](https://github.com/timeseriesAI/tsai).  See the [examples](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb).  Many thanks to Malcolm and Ignacio for their work in developing the GPU implementation and making it part of `tsai`.

## `sktime`\* / Multivariate

MINIROCKET (including a basic multivariate implementation) is also available through [`sktime`](https://github.com/alan-turing-institute/sktime).  See the [examples](https://github.com/alan-turing-institute/sktime/blob/master/examples/minirocket.ipynb).

\* *for larger datasets (10,000+ training examples), the `sktime` methods should be integrated with SGD or similar as per [`softmax.py`](./code/softmax.py) (replace calls to `fit(...)` and `transform(...)` from `minirocket.py` with calls to the relevant `sktime` methods as appropriate)*

## Results

### Joint PPV (Mac)

Measured on **macOS, Apple ARM64, 16 GB RAM**, with one Numba thread.

The published results below compare original MiniRocket with optimized joint PPV
using pooling helpers 1 through 8, at matching feature counts. To benchmark joint
PPV together with group pruning and bias strategies, use the
[combined benchmark](#run-the-combined-benchmark).

* [Full report: accuracy, transform time and RAM](./results/joint_ppv_2026-10-26/report.md)
* [Summary CSV](./results/joint_ppv_2026-10-26/summary.csv)
* [Per-dataset comparisons](./results/joint_ppv_2026-10-26/per_dataset.csv)
* [Raw measurements](./results/joint_ppv_2026-10-26/measurements.csv)
* [Verification](./results/joint_ppv_2026-10-26/verification.json)
* [Run the combined benchmark](#run-the-combined-benchmark)
* [Group pruning and bias strategy guide](./README_GROUP_PRUNING.md)

The study covers 113 compatible UCR datasets, three fitting seeds on each supplied
TRAIN/TEST split, and 2,436 / 4,956 / 9,996 actual features. Features and predictions
match exactly between implementations. These measurements use a different
protocol from the paper's 109-dataset, 30-resample and large-dataset scalability
studies.

Across this fresh run, joint PPV reduced average transform time while preserving
accuracy exactly:

| Actual features | TRAIN time reduction | TEST time reduction | Full pipeline time reduction | Worker peak memory increase |
|---:|---:|---:|---:|---:|
| 2,436 | 28.7% | 23.9% | 14.6% | 1.1% |
| 4,956 | 32.2% | 35.1% | 18.7% | 1.3% |
| 9,996 | 48.7% | 46.9% | 31.8% | 1.8% |

These are ratios of arithmetic means after averaging fitting seeds within each
dataset and weighting datasets equally. Peak memory covers the whole worker
process, not transform-only allocation. Some datasets were slower; confidence
intervals and regressions are included in the full report. Results apply to the
measured machine and compiler environment and are not a universal speed guarantee
or an accuracy improvement.

### Group pruning and bias strategies (Windows)

Measured on **Windows 11, AMD Ryzen 9 8940HX**, with one Numba thread. The
[pruning benchmark](./results/group_pruning_full/report.md) used 113 datasets,
three seeds and original biases, starting from **9,996 features (10,000 requested)**.
All reductions below compare against unpruned original MiniRocket on that machine.

| Method | TEST transform time reduction | Mean accuracy |
|---|---:|---:|
| Original MiniRocket | Baseline | 86.08% |
| Joint PPV, unpruned | 18.9% | 86.08% |
| Joint PPV + filter pruning, keeping half the groups | 53.7% | 86.03% |
| Joint PPV + ridge pruning, keeping half the groups | 63.8% | 85.81% |

Pruning adds selection work during training and can reduce accuracy; filter
scoring is the cheaper option.

The [controls](./results/controls/controls_analysis.md) found similar accuracy
with random group selection. The
[bias study](./results/group_pruning_bias_all_20seeds/bias_analysis.md) found no
consistent benefit from class-stratified thresholds over uniform sampling.

The Mac and Windows studies use different hardware and timing protocols, so their
speed reductions should not be compared directly. See the
[group pruning and bias guide](./README_GROUP_PRUNING.md) for details.

### Run the combined benchmark

The combined runner evaluates **original pooling and optimized joint PPV** at
every group-retention level, with original, uniform and class-stratified bias
strategies. Groups are selected on TRAIN data, and the classifier is refitted on
the retained features. Both pooling implementations must reproduce the full
model's retained feature columns exactly; pruning and different bias strategies
can change classification accuracy.

Follow the [setup and dataset download instructions](./benchmarks/README.md#setup-and-dataset-download),
then run the commands below from the repository root. Choose a **new output
directory** for each experiment.

#### Run the full benchmark

Run all compatible datasets at requested feature counts `2500 5000 10000`
(actual `2436 4956 9996`), with fitting seeds `0 1 2`, all three bias strategies,
and all six default group-retention levels. The combined runner accepts one
feature budget per invocation, so the loop runs the three budgets sequentially
and writes each to its own folder. Analysis follows each successful run.

The commands use the extension guide's recommended `filter` scorer. Iterative
ridge scoring is substantially more expensive; use `--scorers ridge filter`
to compare both scoring methods. A full sweep can take many hours.

```bash
# Limit native threads for comparable timings.
export NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

for features in 2500 5000 10000; do
  run_dir="./results/combined-benchmark-run/features-${features}/"

  .venv/bin/python -B -m experiments.group_pruning \
    --data ./data/Univariate_ts/ \
    --output-dir "$run_dir" \
    --feature-count "$features" \
    --scorers filter || break

  .venv/bin/python -B -m experiments.summarise "$run_dir" || break
  .venv/bin/python -B -m experiments.analyse_bias "$run_dir" \
    --by-keep-fraction \
    --subsets ./results/bias_study_groups_all113.csv || break
done
```

The runner keeps `100% / 75% / 50% / 30% / 20% / 10%` of groups by default; the
100% case is the unpruned baseline. It writes `measurements.csv` and
`metadata.json`. Start with `report.md` in each feature folder, generated by
`experiments.summarise` along with `summary.csv`. The bias analysis writes
`combined_analysis.md`. The loop stops if a benchmark or analysis command fails.

#### Optional: change experiment settings

For a smaller run, select fewer datasets, a smaller feature budget or fewer fitting
seeds. This quick check uses GunPoint, 2,500 requested features and seed `0`:

```bash
.venv/bin/python -B -m experiments.group_pruning \
  --data ./data/Univariate_ts/ \
  --output-dir ./results/gunpoint-combined-run/ \
  --scorers filter \
  --datasets GunPoint \
  --feature-count 2500 \
  --seeds 0

.venv/bin/python -B -m experiments.summarise ./results/gunpoint-combined-run/
.venv/bin/python -B -m experiments.analyse_bias ./results/gunpoint-combined-run/ \
  --by-keep-fraction \
  --subsets ./results/bias_study_groups_all113.csv
```

Omit `--datasets` to use all compatible datasets, and omit `--seeds` to use the
default three seeds (`0 1 2`). A single invocation defaults to 10,000 requested
features; use `--feature-count` to select another budget. Both recipes use all
three bias strategies by default. Use `--bias-strategies original` for pruning
alone, or `--keep-fractions 1.0` to compare bias strategies without pruning.
Include `1.0` when changing keep fractions so the summary has an unpruned
baseline. The all-dataset version of the 2,500-feature, seed-0, filter-scored
configuration was checked on all 113 compatible datasets.

For all options, run `.venv/bin/python -B -m experiments.group_pruning --help`
or see the [extension guide's options](./README_GROUP_PRUNING.md#10-all-options-command-line-and-python).

The joint PPV table above uses the original unpruned benchmark's batching and
timing protocol. The combined runner uses a different protocol and does not
measure worker peak memory. To reproduce that table, follow the
[original benchmark guide](./benchmarks/README.md#run-the-full-benchmark) and its
[recorded settings and environment](./results/joint_ppv_2026-10-26/metadata.json).
Timings depend on hardware and software.

## Requirements\*

* Python, NumPy, pandas
* A Numba version compatible with your Python and NumPy versions
* scikit-learn or similar
* PyTorch or similar (for larger datasets)

\* *PyTorch is needed only for the softmax example. The archive benchmark also uses SciPy, aeon and threadpoolctl; see the [benchmark setup](./benchmarks/README.md#setup-and-dataset-download) and [recorded environment](./results/joint_ppv_2026-10-26/metadata.json).*

## Code

### [`minirocket_pruned.py`](./code/minirocket_pruned.py)

Group pruning for both original pooling and optimized joint PPV. A keep mask
switches off whole kernel/dilation groups so their convolution and pooling work
is skipped. Retained features match the corresponding full-model columns exactly.
Groups can be ranked using filter or iterative ridge scoring on TRAIN data.

### [`minirocket_stratified.py`](./code/minirocket_stratified.py)

Uniform and class-stratified sampling of training examples for bias fitting.
Uniform sampling provides the control for measuring the effect of stratification.
Fitted parameters retain the original layout and work with both pooling
implementations and group pruning. See the [extension guide](./README_GROUP_PRUNING.md)
for Python examples, controls and experiment results.

### [`minirocket_joint_ppv_simd.py`](./code/minirocket_joint_ppv_simd.py)

Optimized univariate transform. It keeps the original convolution arithmetic,
fitted parameters and feature order, and compares each response value against
several thresholds in one pass. Static helpers handle one through eight
thresholds, taking groups of eight followed by the remainder. Numba/LLVM compiles
the pooling loops into native code; SIMD vectorization depends on the target
CPU and compiler.

### [`_joint_ppv_simd_kernels.py`](./code/_joint_ppv_simd_kernels.py)

Generated static pooling helpers and dispatcher. Regenerate with
`python tools/generate_joint_ppv.py`; use `--check` to verify that the generated
file matches the generator. See [`generate_joint_ppv.py`](./tools/generate_joint_ppv.py).

### [`minirocket_baseline.py`](./code/minirocket_baseline.py)

Input validation and reproducible seeded fitting around the unchanged original
univariate implementation. The unpruned original/joint PPV comparison uses this
fitting adapter; the optimized module exposes it as `fit`.

### [`minirocket.py`](./code/minirocket.py)

Original univariate MiniRocket implementation and numerical reference.

### [`minirocket_dv.py`](./code/minirocket_dv.py) (MINIROCKET<sub>DV</sub>)
### [`softmax.py`](./code/softmax.py) (PyTorch / 10,000+ Training Examples)
### [`minirocket_multivariate.py`](./code/minirocket_multivariate.py) (equivalent to [sktime/MiniRocketMultivariate](https://github.com/alan-turing-institute/sktime/blob/master/sktime/transformations/panel/rocket/_minirocket_multivariate.py))
### [`minirocket_variable.py`](./code/minirocket_variable.py) (variable-length input; *experimental*)

## Important Notes

### Compilation

The functions in [`minirocket.py`](./code/minirocket.py) and [`minirocket_dv.py`](./code/minirocket_dv.py) are compiled by Numba on import, which may take some time.  Compiled functions are cached for reuse. The optimized transform uses lazy compilation, so its first call for an input signature may take longer. Cache invalidation or a different environment can require recompilation; warm up before timing steady-state execution.

### Input Data Type

Input data should be of type `np.float32`.  Alternatively, you can change the Numba signatures to accept, e.g., `np.float64`.

### Normalisation

Unlike ROCKET, MINIROCKET does **not** require the input time series to be normalised.  (However, whether or not it makes sense to normalise the input time series may depend on your particular application.)

## Examples

The modules live in `code/`. From the repository root, make them importable in
your current terminal before running a Python script or starting Python:

```bash
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
```

Use a Python environment with the dependencies above. For the univariate
examples, load `X_training` and `X_test` as finite two-dimensional `np.float32`
arrays shaped `(examples, time_points)`, with the same series length of at least
nine points. Labels are one-dimensional arrays. The snippets assume these data
variables have already been loaded.

The examples below use `StandardScaler` followed by `RidgeClassifierCV`, matching
the benchmark's classifier settings. Scaling is fitted on TRAIN features only;
TEST features are transformed using that fitted scaler. This replaces the old
`normalize=True` argument and is an explicit change from the historical examples.
The internal Ridge cross-validation uses features and scaling fitted on the full
outer TRAIN split; it is not fully nested validation.

**Optimized joint PPV (univariate)**

```python
import numpy as np
from minirocket_joint_ppv_simd import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parameters = fit(X_training, num_features=10_000, random_state=0)
X_training_transform = transform(X_training, parameters)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)
predictions = classifier.predict(X_test_transform)
```

The requested 10,000 features round down to 9,996. This optimization supports
fixed-length univariate input. The original multivariate and variable-length
implementations below remain separate.

**MINIROCKET**

```python
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data variables described above.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters = fit(X_training)

X_training_transform = transform(X_training, parameters)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training_transform.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)

predictions = classifier.predict(X_test_transform)
```

**MINIROCKET<sub>DV</sub>**

```python
from minirocket_dv import fit_transform
from minirocket import transform
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data variables described above.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters, X_training_transform = fit_transform(X_training)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training_transform.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)

predictions = classifier.predict(X_test_transform)
```

**PyTorch / 10,000+ Training Examples**

```python
from softmax import train, predict

model_etc = train("InsectSound_TRAIN_shuffled.csv", num_classes = 10, training_size = 22952)
# note: 22,952 = 25,000 - 2,048 (validation)

predictions, accuracy = predict("InsectSound_TEST.csv", *model_etc)
```

**Variable-Length Input** (*Experimental*)

```python
from minirocket_variable import fit, transform, filter_by_length
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data variables described above.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

# special instructions for variable-length input:
# * concatenate variable-length input time series into a single 1d numpy array
# * provide another 1d array with the lengths of each of the input time series
# * input data should be np.float32 (as above); lengths should be np.int32

# optionally, use a different reference length when setting dilation (default is
# the length of the longest time series), and use fit(...) with time series of
# at least this length, e.g.:
# >>> reference_length = X_training_lengths.mean()
# >>> X_training_1d_filtered, X_training_lengths_filtered = \
# >>> filter_by_length(X_training_1d, X_training_lengths, reference_length)
# >>> parameters = fit(X_training_1d_filtered, X_training_lengths_filtered, reference_length)

parameters = fit(X_training_1d, X_training_lengths)

X_training_transform = transform(X_training_1d, X_training_lengths, parameters)

classifier = make_pipeline(
    StandardScaler(),
    RidgeClassifierCV(alphas=X_training_transform.shape[0] * np.logspace(-3, 3, 10)),
)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test_1d, X_test_lengths, parameters)

predictions = classifier.predict(X_test_transform)
```

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing mean ranks were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:<sub>:rocket:</sub><sub><sub>:rocket:</sub></sub></div>
