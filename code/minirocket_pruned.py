#Group-level pruning for MiniRocket.

from itertools import combinations

import numpy as np
from numba import njit, prange

from _joint_ppv_simd_kernels import _pool_joint
from minirocket import _PPV
from minirocket_baseline import validate_input

NUM_KERNELS = 84
# Positions of the three beta (=2) weights in each kernel; same order as the original code.
_INDICES = np.array(list(combinations(range(9), 3)), dtype=np.int32)


# -- group bookkeeping (plain NumPy) -------------------------------------------------

def group_layout(parameters):
    """Return (sizes, starts), each ``[num_dilations, 84]``, in the full feature layout."""
    per_dilation = np.asarray(parameters[1], dtype=np.int64)
    sizes = np.repeat(per_dilation[:, None], NUM_KERNELS, axis=1)
    flat = sizes.ravel()
    starts = (np.cumsum(flat) - flat).reshape(sizes.shape)
    return sizes, starts


def validate_keep(parameters, keep):
    keep = np.asarray(keep, dtype=np.bool_)
    expected = (len(parameters[0]), NUM_KERNELS)
    if keep.shape != expected:
        raise ValueError(f'keep must have shape {expected}, got {keep.shape}')
    if not keep.any():
        raise ValueError('keep at least one group')
    return np.ascontiguousarray(keep)


def keep_all(parameters):
    return np.ones((len(parameters[0]), NUM_KERNELS), dtype=np.bool_)


def kept_columns(parameters, keep):
    """Indices into the full feature vector of the kept groups, in output order."""
    keep = validate_keep(parameters, keep)
    sizes, starts = group_layout(parameters)
    return np.concatenate([np.arange(s, s + n) for s, n in zip(starts[keep], sizes[keep])]).astype(np.int64)


def pruned_width(parameters, keep):
    keep = validate_keep(parameters, keep)
    return int(group_layout(parameters)[0][keep].sum())


# transforms

@njit(cache=True, fastmath=True, parallel=True)
def _transform_joint(X, parameters, keep, width):
    num_examples, input_length = X.shape
    dilations, num_features_per_dilation, biases = parameters
    indices = _INDICES
    num_kernels = len(indices)
    num_dilations = len(dilations)

    features = np.zeros((num_examples, width), dtype=np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]
        A = -_X
        G = _X + _X + _X

        feature_index_start = 0   # position in the FULL layout (indexes biases)
        output_start = 0          # position in the compact output

        for dilation_index in range(num_dilations):

            n = np.int64(num_features_per_dilation[dilation_index])

            any_kept = False
            for kernel_index in range(num_kernels):
                if keep[dilation_index, kernel_index]:
                    any_kept = True

            if not any_kept:
                # No kernel of this dilation survives: skip C_alpha / C_gamma too.
                feature_index_start += num_kernels * n
                continue

            _padding0 = dilation_index % 2
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

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

            for kernel_index in range(num_kernels):

                if not keep[dilation_index, kernel_index]:
                    feature_index_start += n
                    continue

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 != 0:
                    C = C[padding:-padding]

                # Sliced views, so the generated pooling helpers need no change:
                # thresholds read from biases[full slice], counts written to the compact slice.
                _pool_joint(C, biases[feature_index_start:feature_index_start + n], 0, n,
                            features[example_index, output_start:output_start + n])

                output_start += n
                feature_index_start += n

    return features


@njit(cache=True, fastmath=True, parallel=True)
def _transform_original_style(X, parameters, keep, width):
    """Original pooling (one pass per threshold) with the same group mask."""
    num_examples, input_length = X.shape
    dilations, num_features_per_dilation, biases = parameters
    indices = _INDICES
    num_kernels = len(indices)
    num_dilations = len(dilations)

    features = np.zeros((num_examples, width), dtype=np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]
        A = -_X
        G = _X + _X + _X

        feature_index_start = 0
        output_start = 0

        for dilation_index in range(num_dilations):

            n = np.int64(num_features_per_dilation[dilation_index])

            any_kept = False
            for kernel_index in range(num_kernels):
                if keep[dilation_index, kernel_index]:
                    any_kept = True

            if not any_kept:
                feature_index_start += num_kernels * n
                continue

            _padding0 = dilation_index % 2
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

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

            for kernel_index in range(num_kernels):

                if not keep[dilation_index, kernel_index]:
                    feature_index_start += n
                    continue

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(n):
                        features[example_index, output_start + feature_count] = \
                            _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(n):
                        features[example_index, output_start + feature_count] = \
                            _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()

                output_start += n
                feature_index_start += n

    return features


def transform(X, parameters, keep, implementation='joint'):
    # Transform with only the kept groups. 
    if implementation not in ('joint', 'original'):
        raise ValueError("implementation must be 'joint' or 'original'")
    X = validate_input(X, 2, allow_empty=True)
    keep = validate_keep(parameters, keep)
    width = pruned_width(parameters, keep)
    if X.shape[0] == 0:
        return np.empty((0, width), dtype=np.float32)
    # Beyond 2**24 samples the joint int32 counters stop being exact; use original pooling.
    if implementation == 'joint' and X.shape[-1] <= 2 ** 24:
        return _transform_joint(X, parameters, keep, width)
    return _transform_original_style(X, parameters, keep, width)


# ranking and pruning 

def _default_classifier(n_train):
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # Same configuration as benchmarks.common.make_classifier.
    return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=n_train * np.logspace(-3, 3, 10)))


def group_scores(coef, parameters, keep):
    keep = validate_keep(parameters, keep)
    coef = np.atleast_2d(np.asarray(coef, dtype=np.float64))
    sizes, _ = group_layout(parameters)
    if coef.shape[1] != int(sizes[keep].sum()):
        raise ValueError('coef width does not match the kept columns')
    scores = np.full(keep.shape, np.nan)
    offset = 0
    for j, k in zip(*np.nonzero(keep)):          # row-major, same order as kept_columns
        n = int(sizes[j, k])
        block = coef[:, offset:offset + n]
        scores[j, k] = np.sqrt(np.sum(block ** 2) / n)
        offset += n
    return scores


def filter_group_scores(features, y, parameters, keep):
    """Cheap, classifier-free score: mean ANOVA F over the group's features (NaN if pruned)."""
    from sklearn.feature_selection import f_classif
    keep = validate_keep(parameters, keep)
    with np.errstate(all='ignore'):
        f, _ = f_classif(features[:, kept_columns(parameters, keep)], y)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    sizes, _ = group_layout(parameters)
    scores = np.full(keep.shape, np.nan)
    offset = 0
    for j, k in zip(*np.nonzero(keep)):
        n = int(sizes[j, k])
        scores[j, k] = f[offset:offset + n].mean()
        offset += n
    return scores


def prune_groups(features, y, parameters, keep_fraction, scorer='ridge',
                 drop_fraction=0.2, classifier_factory=None, keep=None):
    """Select groups from FULL training features (transform(X_train, p, keep_all(p))).

    scorer='ridge'  : iterative backward elimination. Each round refits the classifier on
                      the surviving columns and drops the lowest drop_fraction of groups.
    scorer='filter' : one-shot ANOVA-F ranking (no classifier fits).

    Uses TRAIN data only. Returns (keep, history); history -> has one dict per round.
    """
    features = np.asarray(features)
    keep = keep_all(parameters) if keep is None else validate_keep(parameters, keep).copy()
    if not 0 < keep_fraction <= 1:
        raise ValueError('keep_fraction must be in (0, 1]')
    if scorer not in ('ridge', 'filter'):
        raise ValueError("scorer must be 'ridge' or 'filter'")
    if not 0 < drop_fraction < 1:
        raise ValueError('drop_fraction must be in (0, 1)')
    factory = classifier_factory or _default_classifier
    target = max(1, int(round(keep_fraction * keep.sum())))
    history = []
    while keep.sum() > target:
        remaining = int(keep.sum())
        columns = kept_columns(parameters, keep)
        if scorer == 'ridge':
            classifier = factory(len(features))
            classifier.fit(features[:, columns], y)
            scores = group_scores(classifier.steps[-1][1].coef_, parameters, keep)
            n_drop = min(remaining - target, max(1, int(np.ceil(drop_fraction * remaining))))
        else:
            scores = filter_group_scores(features, y, parameters, keep)
            n_drop = remaining - target
        flat_keep = keep.reshape(-1)
        candidates = np.flatnonzero(flat_keep)
        order = np.argsort(scores.reshape(-1)[candidates], kind='stable')
        flat_keep[candidates[order[:n_drop]]] = False
        history.append(dict(groups_before=remaining, groups_after=int(keep.sum()),
                            features_after=pruned_width(parameters, keep)))
    return keep, history
