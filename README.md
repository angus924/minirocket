# MINIROCKET

***MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification***

[arXiv:???](https://arxiv.org/) (TBA)

> <div align="justify">Until recently, the most accurate methods for time series classification were limited by high computational complexity.  ROCKET achieves state-of-the-art accuracy with a fraction of the computational expense of most existing methods by transforming input time series using random convolutional kernels, and using the transformed features to train a linear classifier.  We reformulate ROCKET into a new method, MINIROCKET, making it up to 75 times faster on larger datasets, and making it almost deterministic (and optionally, with additional computational expense, fully deterministic), while maintaining essentially the same accuracy.  Using this method, it is possible to train and test a classifier on all of 109 datasets from the UCR archive to state-of-the-art accuracy in less than 10 minutes.  MINIROCKET is significantly faster than any other method of comparable accuracy (including ROCKET), and significantly more accurate than any other method of even roughly-similar computational expense.  As such, we suggest that MINIROCKET should now be considered and used as the default variant of ROCKET.</div>

Please cite as:

```bibtex
@article{dempster_etal_2020,
  author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title   = {{MINIROCKET}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
  year    = {2020},
  journal = {arXiv:???}
}
```

## `sktime`

(TBA)

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

## Examples

**MINIROCKET**

```python
from minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV

[...] # load data, etc.

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

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing mean ranks were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:rocket:<sub>:rocket:</sub><sub><sub>:rocket:</sub></sub></div>
