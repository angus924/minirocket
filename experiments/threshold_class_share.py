"""How much of a threshold's variation across training series is explained by the series' class?

Why: class-stratified thresholds only help if the class of the series that sets a group's
thresholds matters. Every group takes its thresholds from one training series, so each
threshold is a random variable over "which series was drawn". For every threshold we compute
it from EVERY training series (the same code as the real fit, with all groups pointed at one
series), then measure eta^2 = between-class variance / total variance, once with the true
labels and once with shuffled labels (the chance level, which is above 0 for few series).

Median eta^2 near the shuffled value: class does not matter, stratifying cannot help.
Median eta^2 well above it but far below 1: class matters a little, but most variation is
between series of the SAME class, which stratifying does not control.

  NUMBA_NUM_THREADS=1 python -B -m experiments.threshold_class_share --data ./data/Univariate_ts/ \
      --datasets GunPoint ItalyPowerDemand ArrowHead
  NUMBA_NUM_THREADS=1 python -B -m experiments.threshold_class_share --data ./data/Univariate_ts/ \
      --output results/threshold_class_share.csv        # all datasets, resumable

At most --max-series training series per dataset are used (a random subset, seed 0, that keeps
every class with at least two series), because the cost grows with the number of series.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import csv
from argparse import Namespace
from pathlib import Path

import numpy as np

from benchmarks.common import _dataset_sources, load_benchmark_dataset
from minirocket import _fit_dilations, _quantiles
import minirocket_stratified as stratified

FIELDS = ['dataset', 'group', 'n_train_used', 'classes_used', 'groups', 'features',
          'median_eta2_true', 'median_eta2_shuffled', 'q25_true', 'q75_true']


def thresholds_of_every_series(X, num_features=10_000):
    """float32 [series, features]: thresholds if that series were drawn for EVERY group."""
    dilations, per_dilation = _fit_dilations(X.shape[1], num_features, 32)
    quantiles = _quantiles(stratified.NUM_KERNELS * int(np.sum(per_dilation)))
    groups = len(dilations) * stratified.NUM_KERNELS
    rows = []
    for series in range(len(X)):
        table = np.full((groups, 1), series, dtype=np.int32)
        rows.append(stratified._fit_biases_from_table(X, dilations, per_dilation, quantiles, table))
    return np.stack(rows), groups


def eta_squared(values, labels):
    """Between-class share of the variance, per column; columns with no variance are dropped."""
    mean = values.mean(axis=0)
    total = ((values - mean) ** 2).sum(axis=0)
    between = np.zeros_like(total)
    for label in np.unique(labels):
        member = values[labels == label]
        between += len(member) * (member.mean(axis=0) - mean) ** 2
    keep = total > 0
    return between[keep] / total[keep]


def pick_series(y, max_series, rng):
    """Random subset of at most max_series series; classes with fewer than 2 series are dropped."""
    labels, counts = np.unique(y, return_counts=True)
    usable = np.isin(y, labels[counts >= 2])
    index = np.flatnonzero(usable)
    if len(index) > max_series:
        index = np.sort(rng.choice(index, max_series, replace=False))
        labels, counts = np.unique(y[index], return_counts=True)
        index = index[np.isin(y[index], labels[counts >= 2])]
    return index


def analyse(X_train, y_train, max_series, num_features=10_000):
    rng = np.random.default_rng(0)
    index = pick_series(np.asarray(y_train), max_series, rng)
    labels = np.asarray(y_train)[index]
    if len(np.unique(labels)) < 2:
        return None                          # e.g. every class has a single training series
    X =np.ascontiguousarray(X_train[index], dtype=np.float32)
    values, groups = thresholds_of_every_series(X, num_features)
    true = eta_squared(values, labels)
    # chance level: median eta^2 under shuffled labels, averaged over 20 shuffles (one shuffle is noisy)
    shuffled = np.mean([np.median(eta_squared(values, rng.permutation(labels))) for _ in range(20)])
    return dict(n_train_used=len(index), classes_used=len(np.unique(labels)), groups=groups,
                features=values.shape[1], median_eta2_true=float(np.median(true)),
                median_eta2_shuffled=float(shuffled),
                q25_true=float(np.percentile(true, 25)), q75_true=float(np.percentile(true, 75)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--output', type=Path, help='CSV, appended to and resumable')
    p.add_argument('--groups-file', type=Path, default=Path('results/bias_study_groups_all113.csv'))
    p.add_argument('--max-series', type=int, default=100)
    args = p.parse_args(argv)
    groups = {}
    if args.groups_file.exists():
        with args.groups_file.open(newline='') as handle:
            groups = {r['dataset']: r['group'] for r in csv.DictReader(handle)}
    done = set()
    if args.output and args.output.exists():
        with args.output.open(newline='') as handle:
            done = {r['dataset'] for r in csv.DictReader(handle)}
    handle = writer = None
    if args.output:
        new = not args.output.exists()
        handle = args.output.open('a', newline='')
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
    print(f"{'dataset':<32}{'group':<12}{'series':>7}{'classes':>8}{'eta2 true':>11}{'shuffled':>10}", flush=True)
    for name, train, test in _dataset_sources(Namespace(data=args.data, datasets=args.datasets)):
        if name in done:
            continue
        dataset = load_benchmark_dataset(Namespace(data=args.data, datasets=args.datasets), name, train, test, [])
        if dataset is None or dataset.X_train.shape[1] != 1:
            continue
        result = analyse(dataset.X_train[:, 0], dataset.y_train, args.max_series)
        if result is None:
            print(f'{name:<32}skipped: fewer than two classes have two or more training series', flush=True)
            continue
        row = dict(dataset=name, group=groups.get(name, ''), **result)
        print(f"{name:<32}{row['group']:<12}{row['n_train_used']:>7}{row['classes_used']:>8}"
              f"{row['median_eta2_true']:>11.3f}{row['median_eta2_shuffled']:>10.3f}", flush=True)
        if writer:
            writer.writerow(row)
            handle.flush()


if __name__ == '__main__':
    main()
