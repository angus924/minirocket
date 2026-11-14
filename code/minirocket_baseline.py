"""Validated, reproducibly seeded adapter for unchanged univariate MiniRocket.

The adapter seeds both Python NumPy and Numba random streams.
Python's global NumPy state is restored afterwards;
Numba's thread-local RNG is reset, so callers should seed their own Numba work.
"""
from numbers import Integral

import numpy as np
from numba import njit


@njit(cache=True)
def _seed_numba(seed):
    np.random.seed(seed)


def validate_input(X, ndim, allow_empty=False):
    X = np.asarray(X)
    if X.ndim != ndim or not np.issubdtype(X.dtype, np.number) or np.iscomplexobj(X):
        raise ValueError(f'Expected a real numeric {ndim}D array')
    if (not allow_empty and X.shape[0] == 0) or any(s == 0 for s in X.shape[1:]):
        raise ValueError('Input axes must be nonempty')
    if X.shape[-1] < 9:
        raise ValueError('Series must contain at least 9 samples')
    if not np.isfinite(X).all():
        raise ValueError('Input must contain only finite values')
    with np.errstate(over='ignore'):
        result = np.ascontiguousarray(X, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Input values must be representable as finite float32')
    return result


def _validate_options(num_features, max_dilations_per_kernel, random_state):
    for name, value, lower in [('num_features', num_features, 84),
                               ('max_dilations_per_kernel', max_dilations_per_kernel, 1),
                               ('random_state', random_state, 0)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    if random_state >= 2 ** 32:
        raise ValueError('random_state must be below 2**32')


def _fit(X, module, num_features, max_dilations_per_kernel, random_state):
    _validate_options(num_features, max_dilations_per_kernel, random_state)
    state = np.random.get_state()
    try:
        np.random.seed(int(random_state))
        _seed_numba(int(random_state))
        return module.fit(X, int(num_features), int(max_dilations_per_kernel))
    finally:
        np.random.set_state(state)


def fit_univariate(X, num_features=10_000, max_dilations_per_kernel=32, random_state=0):
    import minirocket
    return _fit(validate_input(X, 2), minirocket, num_features, max_dilations_per_kernel, random_state)


def transform_univariate(X, parameters):
    import minirocket
    X = validate_input(X, 2, allow_empty=True)
    if X.shape[0] == 0:
        return np.empty((0, len(parameters[-1])), dtype=np.float32)
    return minirocket.transform(X, parameters)


