"""Controls for group pruning: does CHOOSING the groups matter, or would any shrink do?

For every (dataset, seed) of the primary run (results/group_pruning_full/, bias 'original') this fits the
same full MiniRocket (fit_univariate, same seed) and, for every keep fraction, measures:

  random        same number of groups as the pruned models, chosen uniformly at random (no TRAIN ranking).
                Accuracy and exactness use the same full-layout contract as the pruned models.
  small_ridge   a fresh, smaller MiniRocket whose feature count equals the ridge-pruned model's feature count
                for that (dataset, seed, keep fraction); no selection at all.
  small_filter  the same, matched to the filter-pruned model's feature count.
  full          the unpruned model, timed in THIS session so time ratios are not compared across sessions.

Joint-PPV transform only (pooling does not change accuracy). TEST transform timed as in experiments.group_pruning
(mean of --transform-repetitions). The ridge/filter rows themselves are read from the primary run, not re-run.

  NUMBA_NUM_THREADS=1 python -B -m experiments.controls --data ./data/Univariate_ts/ --output-dir results/controls
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.common import _dataset_sources, _environment, load_benchmark_dataset, make_classifier
import minirocket_pruned as pruned
from minirocket_baseline import fit_univariate

FIELDS = ['dataset', 'seed', 'control', 'keep_fraction', 'groups_total', 'groups_kept', 'features_kept',
          'matched_features', 'n_train', 'n_test', 'fit_seconds', 'train_transform_seconds',
          'classifier_fit_seconds', 'test_transform_seconds', 'predict_seconds', 'accuracy',
          'equal_to_full_columns']
PRIMARY = Path('results/group_pruning_full/measurements.csv')


def timed(function):
    start = perf_counter()
    value = function()
    return value, perf_counter() - start


def primary_feature_counts():
    """{(dataset, seed, scorer, keep): features_kept} from the primary run."""
    out = {}
    with PRIMARY.open(newline='') as handle:
        for r in csv.DictReader(handle):
            if r['implementation'] == 'joint' and r['bias_strategy'] == 'original' and r['scorer'] != 'none':
                out[(r['dataset'], int(r['seed']), r['scorer'], float(r['keep_fraction']))] = int(r['features_kept'])
    return out


def evaluate(parameters, keep, X_train, y_train, X_test, y_test, seed, repetitions, train_features=None):
    """Fit the classifier on TRAIN features of `keep` and time the TEST transform. Returns a dict and TEST features."""
    if train_features is None:
        train_features, train_seconds = timed(lambda: pruned.transform(X_train, parameters, keep, 'joint'))
    else:
        train_seconds = float('nan')
    classifier = make_classifier(len(X_train), 'ridge', seed)
    _, classifier_seconds = timed(lambda: classifier.fit(train_features, y_train))
    times = []
    for _ in range(repetitions):
        test_features, seconds = timed(lambda: pruned.transform(X_test, parameters, keep, 'joint'))
        times.append(seconds)
    predictions, predict_seconds = timed(lambda: classifier.predict(test_features))
    return dict(train_transform_seconds=train_seconds, classifier_fit_seconds=classifier_seconds,
                test_transform_seconds=float(np.mean(times)), predict_seconds=predict_seconds,
                accuracy=float(np.mean(predictions == y_test))), test_features


def run_case(dataset, seed, counts, args):
    X_train = np.ascontiguousarray(dataset.X_train[:, 0]); X_test = np.ascontiguousarray(dataset.X_test[:, 0])
    y_train, y_test = dataset.y_train, dataset.y_test
    base = dict(dataset=dataset.name, seed=seed, n_train=len(X_train), n_test=len(X_test))
    parameters, fit_seconds = timed(lambda: fit_univariate(X_train, num_features=10000, random_state=seed))
    ones = pruned.keep_all(parameters)
    groups_total = int(ones.sum())
    full_train, full_train_seconds = timed(lambda: pruned.transform(X_train, parameters, ones, 'joint'))
    result, full_test = evaluate(parameters, ones, X_train, y_train, X_test, y_test, seed, args.transform_repetitions,
                                 train_features=full_train)
    result['train_transform_seconds'] = full_train_seconds
    rows = [dict(base, control='full', keep_fraction=1.0, groups_total=groups_total, groups_kept=groups_total,
                 features_kept=full_test.shape[1], matched_features='', fit_seconds=fit_seconds,
                 equal_to_full_columns=True, **result)]
    for fraction in args.keep_fractions:
        # random mask: same group count as prune_groups would keep
        target = max(1, int(round(fraction * groups_total)))
        rng = np.random.default_rng(10_000 * seed + int(round(1000 * fraction)))
        keep = np.zeros(groups_total, dtype=np.bool_)
        keep[rng.choice(groups_total, size=target, replace=False)] = True
        keep = keep.reshape(ones.shape)
        columns = pruned.kept_columns(parameters, keep)
        result, test_features = evaluate(parameters, keep, X_train, y_train, X_test, y_test, seed,
                                         args.transform_repetitions, train_features=full_train[:, columns])
        _, result['train_transform_seconds'] = timed(lambda: pruned.transform(X_train, parameters, keep, 'joint'))
        equal = bool(np.array_equal(np.ascontiguousarray(test_features).view(np.uint32),
                                    np.ascontiguousarray(full_test[:, columns]).view(np.uint32)))
        rows.append(dict(base, control='random', keep_fraction=fraction, groups_total=groups_total, groups_kept=target,
                         features_kept=len(columns), matched_features='', fit_seconds=fit_seconds,
                         equal_to_full_columns=equal, **result))
        # smaller fresh models matched to the pruned models' feature counts
        for scorer in ('ridge', 'filter'):
            matched = counts.get((dataset.name, seed, scorer, fraction))
            if matched is None:
                continue
            small, small_fit = timed(lambda: fit_univariate(X_train, num_features=max(84, matched), random_state=seed))
            small_ones = pruned.keep_all(small)
            result, test_features = evaluate(small, small_ones, X_train, y_train, X_test, y_test, seed,
                                             args.transform_repetitions)
            rows.append(dict(base, control=f'small_{scorer}', keep_fraction=fraction, groups_total=int(small_ones.sum()),
                             groups_kept=int(small_ones.sum()), features_kept=test_features.shape[1],
                             matched_features=matched, fit_seconds=small_fit, equal_to_full_columns='', **result))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--keep-fractions', nargs='+', type=float, default=[0.75, 0.5, 0.3, 0.2, 0.1])
    p.add_argument('--transform-repetitions', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if args.output_dir.exists() and not args.resume:
        raise SystemExit(f'{args.output_dir} exists; choose a new folder or pass --resume')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = primary_feature_counts()
    csv_path = args.output_dir / 'measurements.csv'
    done = set()
    if csv_path.exists():
        with csv_path.open(newline='') as handle:
            done = {(r['dataset'], int(r['seed'])) for r in csv.DictReader(handle)}
    # compile outside the timers
    Xw = np.random.default_rng(0).normal(size=(8, 64)).astype(np.float32)
    pw = fit_univariate(Xw, num_features=200, random_state=0)
    kw = pruned.keep_all(pw); kw[0, 0] = False
    pruned.transform(Xw[:2], pw, kw, 'joint')
    skipped, seen = [], []
    new_file = not csv_path.exists()
    with csv_path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for name, train, test in _dataset_sources(args):
            dataset = load_benchmark_dataset(args, name, train, test, skipped)
            if dataset is None or dataset.X_train.shape[1] != 1:
                continue
            seen.append(name)
            for seed in args.seeds:
                if (name, seed) in done:
                    continue
                writer.writerows(run_case(dataset, seed, counts, args)); handle.flush()
                print(f'{name} seed={seed} done', flush=True)
    meta = _environment(args)
    meta.update(datasets=seen, skipped_datasets=skipped, primary=str(PRIMARY),
                protocol='random group masks and smaller fresh models matched to the primary run; joint PPV; TRAIN-only fitting')
    (args.output_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
