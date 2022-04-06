[`ROCKET`](https://github.com/angus924/rocket) &middot; [**`MINIROCKET`**](https://github.com/angus924/minirocket) &middot; [`HYDRA`](https://github.com/angus924/hydra)

# MINIROCKET

***MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification***

[arXiv:2012.08791](https://arxiv.org/abs/2012.08791) (preprint)

> <div align="justify">Until recently, the most accurate methods for time series classification were limited by high computational complexity.  ROCKET achieves state-of-the-art accuracy with a fraction of the computational expense of most existing methods by transforming input time series using random convolutional kernels, and using the transformed features to train a linear classifier.  We reformulate ROCKET into a new method, MINIROCKET, making it up to 75 times faster on larger datasets, and making it almost deterministic (and optionally, with additional computational expense, fully deterministic), while maintaining essentially the same accuracy.  Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes.  MINIROCKET is significantly faster than any other method of comparable accuracy (including ROCKET), and significantly more accurate than any other method of even roughly-similar computational expense.  As such, we suggest that MINIROCKET should now be considered and used as the default variant of ROCKET.</div>

Please cite as:

```bibtex
@article{dempster_etal_2020,
  author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title   = {{MINIROCKET}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  year    = {2020},
  journal = {arXiv:2012.08791}
}
```

## Podcast

Hear more about MINIROCKET and time series classification on the [Data Skeptic](https://dataskeptic.com/blog/episodes/2021/minirocket) podcast!

## GPU Implementation \*NEW\*

A GPU implementation of MINIROCKET, developed by Malcolm McLean and Ignacio Oguiza, is available through [`tsai`](https://github.com/timeseriesAI/tsai).  See the [examples](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/10_Time_Series_Classification_and_Regression_with_MiniRocket.ipynb).  Many thanks to Malcolm and Ignacio for their work in developing the GPU implementation and making it part of `tsai`.

## `sktime`\* / Multivariate

MINIROCKET (including a basic multivariate implementation) is also available through [`sktime`](https://github.com/alan-turing-institute/sktime).  See the [examples](https://github.com/alan-turing-institute/sktime/blob/master/examples/minirocket.ipynb).

\* *for larger datasets (10,000+ training examples), the `sktime` methods should be integrated with SGD or similar as per [`softmax.py`](./code/softmax.py) (replace calls to `fit(...)` and `transform(...)` from `minirocket.py` with calls to the relevant `sktime` methods as appropriate)*

## Results

* UCR Archive (109 Datasets, 30 Resamples)
  * [Mean Accuracy + Training/Test Times](./results/results_ucr109_mean.csv)
  * [Accuracy Per Resample](./results/accuracy_ucr109_resamples.csv)
* Scalability / Training Set Size\*
  * [MosquitoSound](./results/time_training_MosquitoSound.csv) (139,780 &times; 3,750)
  * [InsectSound](./results/time_training_InsectSound.csv) (25,000 &times; 600)
  * [FruitFlies](./results/time_training_FruitFlies.csv) (17,259 &times; 5,000)
* Scalability / Time Series Length
  * [DucksAndGeese](./results/time_training_DucksAndGeese.csv) (50 &times; 236,784)

\* *`num_training_examples` does* ***not*** *include the validation set of 2,048 training examples, but the transform time for the validation set* ***is*** *included in `time_training_seconds`*

## Requirements\*

* Python, NumPy, pandas
* Numba (0.50+)
* scikit-learn or similar
* PyTorch or similar (for larger datasets)

\* *all pre-packaged with or otherwise available through Anaconda*

## Code

### [`minirocket.py`](./code/minirocket.py)
### [`minirocket_dv.py`](./code/minirocket_dv.py) (MINIROCKET<sub>DV</sub>)
### [`softmax.py`](./code/softmax.py) (PyTorch / 10,000+ Training Examples)
### [`minirocket_multivariate.py`](./code/minirocket_multivariate.py) (equivalent to [sktime/MiniRocketMultivariate](https://github.com/alan-turing-institute/sktime/blob/master/sktime/transformations/panel/rocket/_minirocket_multivariate.py))
### [`minirocket_variable.py`](./code/minirocket_variable.py) (variable-length input; *experimental*)

## Important Notes

### Compilation

The functions in [`minirocket.py`](./code/minirocket.py) and [`minirocket_dv.py`](./code/minirocket_dv.py) are compiled by Numba on import, which may take some time.  By default, the compiled functions are now cached, so this should only happen once (i.e., on the first import).

### Input Data Type

Input data should be of type `np.float32`.  Alternatively, you can change the Numba signatures to accept, e.g., `np.float64`.

### Normalisation

Unlike ROCKET, MINIROCKET does **not** require the input time series to be normalised.  (However, whether or not it makes sense to normalise the input time series may depend on your particular application.)

## Examples

**MINIROCKET**

```python
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV

[...] # load data, etc.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters = fit(X_training)

X_training_transform = transform(X_training, parameters)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test, parameters)

predictions = classifier.predict(X_test_transform)
```

**MINIROCKET<sub>DV</sub>**

```python
from minirocket_dv import fit_transform
from minirocket import transform
from sklearn.linear_model import RidgeClassifierCV

[...] # load data, etc.

# note:
# * input time series do *not* need to be normalised
# * input data should be np.float32

parameters, X_training_transform = fit_transform(X_training)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
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

[...] # load data, etc.

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

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)

X_test_transform = transform(X_test_1d, X_test_lengths, parameters)

predictions = classifier.predict(X_test_transform)
```

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing mean ranks were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:<sub>:rocket:</sub><sub><sub>:rocket:</sub></sub></div>
