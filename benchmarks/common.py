"""Offline dataset, classifier and measurement helpers for the joint-PPV study."""
import hashlib
import importlib.metadata
import platform
import sys
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))


class UnsupportedDatasetError(ValueError):
    """Valid input needing preprocessing before equal-length comparisons."""


def _shorten(value, width):
    value = ' '.join(str(value).split())
    return value if len(value) <= width else value[:width - 3] + '...'


@dataclass(frozen=True)
class ModelSpec:
    name: str
    fit: object
    transform: object
    input_dim: int = 3


@dataclass(frozen=True)
class Dataset:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    sources: tuple
    fingerprint: str


def _labels(y):
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0:
        raise ValueError('Labels must be a nonempty 1D array')
    if y.dtype.kind in 'fc' and (np.iscomplexobj(y) or not np.isfinite(y).all()):
        raise ValueError('Labels must be finite and real')
    values = []
    for value in y:
        if value is None:
            raise ValueError('Missing labels are unsupported')
        label = str(value).strip()
        if not label or label.lower() in ('nan', 'none', '?'):
            raise ValueError('Missing labels are unsupported')
        # Numeric archive labels may be written as 1 or 1.0 or 1e0.
        try:
            number = Decimal(label)
        except InvalidOperation:
            number = None  # Non-numeric class names are preserved.
        if number is not None:
            if not number.is_finite() or not np.isfinite(float(number)):
                raise ValueError('Numeric labels must be finite')
            sign, digits, exponent = number.as_tuple()
            digits = list(digits)
            while digits and digits[-1] == 0:
                digits.pop()
                exponent += 1
            if not digits:
                label = '0'
            else:
                coefficient = ''.join(map(str, digits))
                prefix = '-' if sign else ''
                label = prefix + coefficient + ('0' * exponent if exponent >= 0 else f'e{exponent}')
        values.append(label)
    return np.asarray(values)


def _series(X):
    from minirocket_baseline import validate_input
    try:
        X = np.asarray(X)
    except ValueError as error:
        raise UnsupportedDatasetError('Variable-length series require an explicit preprocessing policy') from error
    if X.ndim == 2:
        X = X[:, None, :]
    if X.ndim == 3 and X.dtype.kind in 'fiu':
        if X.shape[-1] < 9:
            raise UnsupportedDatasetError('Series must contain at least 9 samples')
        if not np.isfinite(X).all():
            raise UnsupportedDatasetError('Input must contain only finite values; missing values require preprocessing')
    return validate_input(X, 3)


def _dataset(name, X_train, y_train, X_test, y_test, sources):
    X_train, X_test = _series(X_train), _series(X_test)
    y_train, y_test = _labels(y_train), _labels(y_test)
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f'{name}: train/test channel count must match')
    if X_train.shape[-1] != X_test.shape[-1]:
        raise UnsupportedDatasetError(f'{name}: train/test series lengths must match')
    if len(y_train) != len(X_train) or len(y_test) != len(X_test):
        raise ValueError(f'{name}: label count must match example count')
    if len(np.unique(y_train)) < 2:
        raise ValueError(f'{name}: training split requires at least two classes')
    digest = hashlib.sha256()
    for value in (X_train, y_train, X_test, y_test):
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
    return Dataset(name, X_train, y_train, X_test, y_test, tuple(map(str, sources)), digest.hexdigest())


def _load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            required = ('X_train', 'y_train', 'X_test', 'y_test')
            missing = set(required) - set(data.files)
            if missing:
                raise ValueError(f'{path}: missing keys {sorted(missing)}')
            return _dataset(path.stem, *(data[k] for k in required), (path,))
    except UnsupportedDatasetError:
        raise
    except ValueError as error:
        raise ValueError(f'{path}: {error}. NPZ labels must use numeric or string arrays, not pickled objects') from error


def _load_pair(name, train, test):
    if train.suffix.lower() == '.ts':
        for path in (train, test):
            with path.open(encoding='utf-8-sig') as handle:
                for line in handle:
                    tokens = line.strip().lower().split()
                    if tokens[:1] == ['@data']:
                        break
                    if tokens[:2] == ['@timestamps', 'true']:
                        raise UnsupportedDatasetError('timestamped series require explicit preprocessing')
                    if tokens[:2] == ['@equallength', 'false']:
                        raise UnsupportedDatasetError('Variable-length series require explicit preprocessing')
        from aeon.datasets import load_from_ts_file
        X_train, y_train = load_from_ts_file(str(train))
        X_test, y_test = load_from_ts_file(str(test))
    else:
        def read(path):
            table = np.loadtxt(path, dtype=str, delimiter='\t', ndmin=2)
            if table.shape[1] < 10:
                raise ValueError(f'{path}: expected label plus at least nine samples')
            return table[:, 1:].astype(np.float32), table[:, 0]
        X_train, y_train = read(train)
        X_test, y_test = read(test)
    return _dataset(name, X_train, y_train, X_test, y_test, (train, test))


def load_benchmark_dataset(args, name, train, test, skipped):
    """Skip known incompatibilities only when discovering a whole archive."""
    try:
        return _load_npz(train) if test is None else _load_pair(name, train, test)
    except UnsupportedDatasetError as error:
        if getattr(args, 'datasets', None) or Path(args.data).is_file():
            raise UnsupportedDatasetError(f'{name}: {error}') from error
        skipped.append({'name': name, 'sources': [str(train)] + ([str(test)] if test else []),
                        'reason': str(error)})
        print(f'\nSkipping {_shorten(name, 64)}\n  {_shorten(error, 76)}', flush=True)
        return None


def _dataset_sources(args):
    path = Path(args.data).expanduser().resolve()
    names = getattr(args, 'datasets', None)
    if path.is_file():
        if path.suffix.lower() != '.npz' or names:
            raise ValueError('A file input must be a split .npz without --datasets')
        return [(path.stem, path, None)]
    if not path.is_dir():
        raise FileNotFoundError(f'Local dataset path does not exist: {path}')
    found = {}
    # Prefer .ts when both archive encodings are present.
    for extension in ('.ts', '.tsv'):
        for train in sorted(path.rglob(f'*_TRAIN{extension}')):
            name = train.name.removesuffix(f'_TRAIN{extension}')
            if names and name not in names:
                continue
            test = train.with_name(f'{name}_TEST{extension}')
            if not test.is_file():
                raise FileNotFoundError(f'Missing local test split: {test}')
            if name in found:
                if found[name][1].suffix == extension:
                    raise ValueError(f'Ambiguous local dataset name: {name}')
                continue
            found[name] = (name, train, test)
    for archive in sorted(path.rglob('*.npz')):
        if not names or archive.stem in names:
            if archive.stem in found:
                if found[archive.stem][2] is None:
                    raise ValueError(f'Ambiguous local dataset name: {archive.stem}')
                continue  # Prefer an archive TRAIN/TEST pair over an NPZ.
            found[archive.stem] = (archive.stem, archive, None)
    if names:
        missing = set(names) - set(found)
        if missing:
            raise FileNotFoundError(f'No complete local splits for: {sorted(missing)}')
        return [found[name] for name in names]
    if not found:
        raise FileNotFoundError(f'No local TRAIN/TEST .ts/.tsv pairs or split NPZ files under {path}')
    return [found[name] for name in sorted(found)]



def make_classifier(n_train, classifier, seed):
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if classifier == 'ridge':
        estimator = RidgeClassifierCV(alphas=n_train * np.logspace(-3, 3, 10))
    else:
        raise ValueError('Unknown classifier')
    return make_pipeline(StandardScaler(), estimator)


def parameter_nbytes(parameters):
    """Bytes of stored NumPy arrays only, deduplicated by array identity."""
    seen = set()
    def count(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return value.nbytes
        if is_dataclass(value) and not isinstance(value, type):
            return sum(count(getattr(value, field.name)) for field in fields(value))
        if isinstance(value, dict):
            return sum(count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(count(item) for item in value)
        return 0
    return count(parameters)


def transform_batches(model, X, parameters, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = None
    for start in range(0, len(X), batch_size):
        block = np.asarray(model.transform(X[start:start + batch_size], parameters))
        if block.ndim != 2 or len(block) != min(batch_size, len(X) - start) or not np.isfinite(block).all():
            raise ValueError(f'{model.name}: transform must return finite [examples,features]')
        if output is None:
            output = np.empty((len(X), block.shape[1]), dtype=block.dtype)
        if block.shape[1] != output.shape[1]:
            raise ValueError(f'{model.name}: inconsistent feature width across batches')
        output[start:start + len(block)] = block
    if output is None:
        return np.asarray(model.transform(X, parameters))
    return output


def verify_feature_values(features, expected_digest=None):
    """Hash the complete matrix, including dtype/order; outside transform timers."""
    digest = hashlib.sha256()
    digest.update(str(features.shape).encode())
    digest.update(features.dtype.str.encode())
    digest.update(memoryview(np.ascontiguousarray(features)).cast('B'))
    value = digest.hexdigest()
    if expected_digest is not None and value != expected_digest:
        raise ValueError('Feature equality check failed: matrices differ from the configured reference')
    return value


def _environment(args):
    versions = {}
    for package in ('numpy', 'numba', 'scikit-learn', 'scipy', 'aeon'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(), 'versions': versions,
            'configuration': {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
            'split_policy': 'Supplied train/test splits; seeds repeat fits, not resampled splits',
            'timing_policy': 'Exclude dataset I/O, fingerprinting, imports and train-only JIT warmup',
            'memory_policy': 'NumPy parameter bytes and dense feature bytes; not peak RSS'}


