"""MiniRocket fitting with class-stratified (or explicitly uniform) bias sampling.

Original MiniRocket draws, for every (dilation, kernel) group, ONE random training example
and sets that group's thresholds to quantiles of the example's kernel response
(code/minirocket.py::_fit_biases). Here the example(s) used per group come from an explicit
index table built in Python:

  strategy='uniform'    : indices drawn uniformly from the training set (control).
  strategy='stratified' : classes are cycled through in random order, so every class supplies
                          (almost) the same number of groups; the example is drawn uniformly
                          inside the class.

(examples_per_group=m) pools the responses of m examples before taking quantiles (fit cost
grows about m-fold). Only the biases differ from the original fit: dilations, features per
dilation and the quantile levels come from the original ((_fit_dilations) / (_quantiles)),
so the returned tuple is a drop-in (parameters) for every transform in this repo.

Why 'uniform' exists: the original uses Numba's RNG inside the jitted function, this module uses
a NumPy Generator seeded by (random_state). Comparing 'stratified' against 'uniform' (same
code path, same seed policy) isolates the effect of stratification from the RNG change.
"""
from itertools import combinations

import numpy as np
from numba import njit

from minirocket import _fit_dilations, _quantiles
from minirocket_baseline import validate_input

NUM_KERNELS = 84
_INDICES = np.array(list(combinations(range(9), 3)), dtype=np.int32)


def draw_example_table(y, num_groups, examples_per_group=1, strategy='stratified', random_state=0):
    """Return int32 [num_groups, examples_per_group] of training-example indices.
    Row g = dilation_index * 84 + kernel_index is the group's example(s).
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError('y must be a nonempty 1D array')
    if examples_per_group < 1:
        raise ValueError('examples_per_group must be positive')
    rng = np.random.default_rng(random_state)
    needed = num_groups * examples_per_group
    if strategy == 'uniform':
        return rng.integers(0, len(y), size=(num_groups, examples_per_group)).astype(np.int32)
    if strategy != 'stratified':
        raise ValueError("strategy must be 'uniform' or 'stratified'")
    classes, inverse = np.unique(y, return_inverse=True)
    counts = np.bincount(inverse)
    order = np.argsort(inverse, kind='stable')          # example indices grouped by class
    class_start = np.cumsum(counts) - counts
    repeats = -(-needed // len(classes))
    stream = np.concatenate([rng.permutation(len(classes)) for _ in range(repeats)])[:needed]
    within = (rng.random(needed) * counts[stream]).astype(np.int64)
    within = np.minimum(within, counts[stream] - 1)
    return order[class_start[stream] + within].astype(np.int32).reshape(num_groups, examples_per_group)


@njit(cache=True, fastmath=True)
def _fit_biases_from_table(X, dilations, num_features_per_dilation, quantiles, example_table):
    num_examples, input_length = X.shape
    indices = _INDICES
    num_kernels = len(indices)
    num_dilations = len(dilations)
    per_group = example_table.shape[1]

    num_features = num_kernels * np.sum(num_features_per_dilation)
    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0
    group = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation
            pooled = np.zeros(per_group * input_length, dtype=np.float32)

            for slot in range(per_group):

                _X = X[example_table[group, slot]]

                A = -_X
                G = _X + _X + _X

                C_alpha = np.zeros(input_length, dtype=np.float32)
                C_alpha[:] = A

                C_gamma = np.zeros((9, input_length), dtype=np.float32)
                C_gamma[9 // 2] = G

                start = dilation
                end = input_length - padding

                for gamma_index in range(9 // 2):
                    C_alpha[-end:] = C_alpha[-end:] + A[:end]
                    C_gamma[gamma_index, -end:] = G[:end]
                    end += dilation

                for gamma_index in range(9 // 2 + 1, 9):
                    C_alpha[:-start] = C_alpha[:-start] + A[start:]
                    C_gamma[gamma_index, :-start] = G[start:]
                    start += dilation

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                pooled[slot * input_length:(slot + 1) * input_length] = C

            biases[feature_index_start:feature_index_end] = \
                np.quantile(pooled, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end
            group += 1

    return biases


def fit(X, y, num_features=10_000, max_dilations_per_kernel=32, examples_per_group=1,
        strategy='stratified', random_state=0):
    """Return ``(dilations, num_features_per_dilation, biases)`` like ``minirocket.fit``."""
    X = validate_input(X, 2)
    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError('X and y must have the same number of examples')
    for name, value, lower in (('num_features', num_features, 84),
                               ('max_dilations_per_kernel', max_dilations_per_kernel, 1)):
        if isinstance(value, bool) or int(value) != value or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    dilations, per_dilation = _fit_dilations(X.shape[1], int(num_features), int(max_dilations_per_kernel))
    quantiles = _quantiles(NUM_KERNELS * int(np.sum(per_dilation)))
    table = draw_example_table(y, len(dilations) * NUM_KERNELS, examples_per_group, strategy, random_state)
    biases = _fit_biases_from_table(X, dilations, per_dilation, quantiles, table)
    return dilations, per_dilation, biases
