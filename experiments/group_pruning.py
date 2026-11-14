"""Group pruning x bias strategy experiment grid.

Deliberately separate from ``benchmarks.run``: that pipeline hard-codes an original/joint_ppv
pair (see README_GROUP_PRUNING.md, conflict C1). This runner reuses its dataset loaders and
classifier factory only, and writes its own results folder.

Per (dataset, seed, bias strategy) it
  1. fits the FULL layout once (never refits for a smaller layout),
  2. transforms TRAIN once with all groups and ranks groups on TRAIN only,
  3. for every keep fraction: prunes, refits the classifier on the kept TRAIN columns,
     times the pruned TEST transform for both the original and the joint-PPV implementation,
     and checks that the pruned features equal ``reference_full_test[:, kept_columns]``.

Run from the repository root:
  NUMBA_NUM_THREADS=1 python -B -m experiments.group_pruning --data ./data/Univariate_ts/ \
      --output-dir ./results/group_pruning_pilot/ --datasets GunPoint ECG200 --seeds 0 1
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.common import _dataset_sources, _environment, load_benchmark_dataset, make_classifier
import minirocket_pruned as pruned
import minirocket_stratified as stratified
from minirocket_baseline import fit_univariate, transform_univariate

FIELDS = ['dataset', 'seed', 'bias_strategy', 'scorer', 'keep_fraction', 'implementation',
          'groups_total', 'groups_kept', 'features_total', 'features_kept',
          'n_train', 'n_test', 'length', 'fit_seconds', 'full_train_transform_seconds',
          'selection_seconds', 'selection_rounds', 'classifier_fit_seconds',
          'train_transform_seconds', 'test_transform_seconds', 'test_transform_median_seconds',
          'predict_seconds', 'accuracy', 'equal_to_full_columns', 'test_features_sha256',
          'split_sha256']


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='Local archive; never downloads')
    p.add_argument('--output-dir', required=True, type=Path, help='New folder for this run')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--bias-strategies', nargs='+', default=['original', 'uniform', 'stratified'],
                   choices=['original', 'uniform', 'stratified'])
    p.add_argument('--keep-fractions', nargs='+', type=float, default=[1.0, 0.75, 0.5, 0.3, 0.2, 0.1])
    p.add_argument('--scorers', nargs='+', default=['ridge'], choices=['ridge', 'filter'])
    p.add_argument('--feature-count', type=int, default=10000)
    p.add_argument('--examples-per-group', type=int, default=1)
    p.add_argument('--drop-fraction', type=float, default=0.2)
    p.add_argument('--max-dilations-per-kernel', type=int, default=32)
    p.add_argument('--transform-repetitions', type=int, default=3)
    p.add_argument('--resume', action='store_true', help='Skip (dataset, seed, strategy) cases already in the CSV')
    return p


def timed(function):
    start = perf_counter()
    value = function()
    return value, perf_counter() - start


def fit_parameters(strategy, X, y, args, seed):
    if strategy == 'original':
        return fit_univariate(X, num_features=args.feature_count,
                              max_dilations_per_kernel=args.max_dilations_per_kernel, random_state=seed)
    return stratified.fit(X, y, num_features=args.feature_count,
                          max_dilations_per_kernel=args.max_dilations_per_kernel,
                          examples_per_group=args.examples_per_group, strategy=strategy, random_state=seed)


def warm_up(args):
    """Compile every jitted function on synthetic data, outside all timers."""
    X = np.random.default_rng(0).normal(size=(8, 64)).astype(np.float32)
    y = np.arange(8) % 2
    for strategy in args.bias_strategies:
        parameters = fit_parameters(strategy, X, y, args, 0)
        keep = pruned.keep_all(parameters)
        keep[0, 0] = False
        for implementation in ('original', 'joint'):
            pruned.transform(X[:2], parameters, keep, implementation)
    transform_univariate(X[:2], parameters)


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_case(dataset, seed, strategy, args):
    X_train = np.ascontiguousarray(dataset.X_train[:, 0])
    X_test = np.ascontiguousarray(dataset.X_test[:, 0])
    y_train, y_test = dataset.y_train, dataset.y_test
    parameters, fit_seconds = timed(lambda: fit_parameters(strategy, X_train, y_train, args, seed))
    ones = pruned.keep_all(parameters)
    groups_total, features_total = int(ones.sum()), len(parameters[2])

    # Full-model TRAIN transform: needed for ranking and as the training-cost baseline, per implementation.
    full_train_seconds = {}
    train_full, full_train_seconds['joint'] = timed(lambda: pruned.transform(X_train, parameters, ones, 'joint'))
    _, full_train_seconds['original'] = timed(lambda: pruned.transform(X_train, parameters, ones, 'original'))
    reference_test = transform_univariate(X_test, parameters)        # untimed exactness reference

    configs = []
    for fraction in args.keep_fractions:
        if fraction >= 1.0:
            configs.append(('none', 1.0))
        else:
            configs.extend((scorer, fraction) for scorer in args.scorers)

    rows = []
    for scorer, fraction in configs:
        if fraction >= 1.0:
            keep, selection_seconds, rounds = ones, 0.0, 0
        else:
            (keep, history), selection_seconds = timed(lambda: pruned.prune_groups(
                train_full, y_train, parameters, fraction, scorer=scorer, drop_fraction=args.drop_fraction,
                classifier_factory=lambda n: make_classifier(n, 'ridge', seed)))
            rounds = len(history)
        columns = pruned.kept_columns(parameters, keep)
        classifier = make_classifier(len(X_train), 'ridge', seed)
        _, classifier_seconds = timed(lambda: classifier.fit(train_full[:, columns], y_train))
        for implementation in ('original', 'joint'):
            _, train_seconds = timed(lambda: pruned.transform(X_train, parameters, keep, implementation))
            times, features = [], None
            for _ in range(args.transform_repetitions):
                features, seconds = timed(lambda: pruned.transform(X_test, parameters, keep, implementation))
                times.append(seconds)
            predictions, predict_seconds = timed(lambda: classifier.predict(features))
            equal = bool(np.array_equal(np.ascontiguousarray(features).view(np.uint32),
                                        np.ascontiguousarray(reference_test[:, columns]).view(np.uint32)))
            if not equal:
                print(f'WARNING: pruned features differ from full[:, kept] '
                      f'({dataset.name}/{strategy}/{scorer}/{fraction}/{implementation})', flush=True)
            rows.append(dict(
                dataset=dataset.name, seed=seed, bias_strategy=strategy, scorer=scorer, keep_fraction=fraction,
                implementation=implementation, groups_total=groups_total, groups_kept=int(keep.sum()),
                features_total=features_total, features_kept=len(columns), n_train=len(X_train),
                n_test=len(X_test), length=X_train.shape[1], fit_seconds=fit_seconds,
                full_train_transform_seconds=full_train_seconds[implementation],
                selection_seconds=selection_seconds, selection_rounds=rounds,
                classifier_fit_seconds=classifier_seconds, train_transform_seconds=train_seconds,
                test_transform_seconds=float(np.mean(times)), test_transform_median_seconds=float(np.median(times)),
                predict_seconds=predict_seconds, accuracy=float(np.mean(predictions == y_test)),
                equal_to_full_columns=equal, test_features_sha256=sha(features), split_sha256=dataset.fingerprint))
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and not args.resume:
        raise SystemExit(f'{args.output_dir} exists; choose a new folder or pass --resume')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = args.output_dir / 'measurements.csv', args.output_dir / 'metadata.json'
    done = set()
    if csv_path.exists():
        with csv_path.open(newline='') as handle:
            done = {(r['dataset'], int(r['seed']), r['bias_strategy']) for r in csv.DictReader(handle)}
    warm_up(args)
    skipped, datasets_seen = [], []
    new_file = not csv_path.exists()
    with csv_path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for name, train, test in _dataset_sources(args):
            dataset = load_benchmark_dataset(args, name, train, test, skipped)
            if dataset is None:
                continue
            if dataset.X_train.shape[1] != 1:
                skipped.append(dict(name=name, reason='multivariate')); continue
            datasets_seen.append(dict(name=name, fingerprint=dataset.fingerprint))
            for seed in args.seeds:
                for strategy in args.bias_strategies:
                    if (name, seed, strategy) in done:
                        continue
                    for row in run_case(dataset, seed, strategy, args):
                        writer.writerow(row)
                    handle.flush()
                    print(f'{name} seed={seed} {strategy} done', flush=True)
    metadata = _environment(args)
    metadata.update(datasets=datasets_seen, skipped_datasets=skipped,
                    protocol='full layout fitted once; ranking on TRAIN only; classifier refit on kept TRAIN columns',
                    csv=csv_path.name)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
