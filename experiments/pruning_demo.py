"""Group pruning on one dataset, in memory: a quick look without a full run.

Adapted from ``pruning_service.py``, which Shreyas Shinde wrote to connect this code to the group
project's UI (github.com/er-shreyas-shinde/minirocket-group-project, folder "Arya's Additions").
The steps are the same as ``experiments/group_pruning.py::run_case`` and reuse its functions:

  1. fit the FULL layout once ('original', 'uniform' or 'stratified' biases),
  2. transform TRAIN once with every group and rank groups on TRAIN only,
  3. for each keep fraction: prune, refit the classifier on the kept TRAIN columns, time the
     pruned TEST transform (joint PPV) and check the features equal the full model's kept columns.

Differences from the runner: one seed, one scorer, one bias strategy, joint PPV only, nothing
written to disk. It also returns the keep masks (which groups survive) and a model-size figure.
The size is DERIVED, not measured: ridge weights plus the scaler's mean and scale, 8 bytes each,
for the kept features. The time ratio is against the unpruned joint-PPV model in the same session
(mean of ``--repetitions`` runs), not against the original transform.

Run from the repository root (reads the local archive, never downloads):
  NUMBA_NUM_THREADS=1 python -B -m experiments.pruning_demo --dataset GunPoint
  NUMBA_NUM_THREADS=1 python -B -m experiments.pruning_demo --dataset ECG200 --scorer filter --bias-strategy stratified
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import csv
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.common import _dataset_sources, load_benchmark_dataset, make_classifier
import minirocket_pruned as pruned
import minirocket_stratified as stratified
from minirocket_baseline import transform_univariate
from experiments.group_pruning import fit_parameters as _runner_fit, sha

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DEFAULT_KEEP_FRACTIONS = (1.0, 0.75, 0.5, 0.3, 0.2, 0.1)
BIAS_STRATEGIES = ('original', 'uniform', 'stratified')
SCORERS = ('ridge', 'filter')


def load_dataset(name, data='./data/Univariate_ts/'):
    """X_train, y_train, X_test, y_test (2-D float32) for one univariate dataset from the local archive."""
    args = Namespace(data=data, datasets=[name])
    (_, train, test), = _dataset_sources(args)
    dataset = load_benchmark_dataset(args, name, train, test, [])
    if dataset.X_train.shape[1] != 1:
        raise ValueError(f'{name} is multivariate; group pruning is univariate')
    return (np.ascontiguousarray(dataset.X_train[:, 0]), dataset.y_train,
            np.ascontiguousarray(dataset.X_test[:, 0]), dataset.y_test)


@dataclass
class PruningResult:
    rows: list                         # one dict per keep fraction, full model first
    masks: dict                        # keep fraction -> bool array [num_dilations, 84]
    dilations: list                    # dilation of each mask row
    settings: dict = field(default_factory=dict)


def fit_parameters(X_train, y_train, bias_strategy='original', num_features=10_000, seed=0):
    """Same call as the runner (32 dilations per kernel at most, one series per group)."""
    if bias_strategy not in BIAS_STRATEGIES:
        raise ValueError(f'bias_strategy must be one of {BIAS_STRATEGIES}')
    args = Namespace(feature_count=num_features, max_dilations_per_kernel=32, examples_per_group=1)
    return _runner_fit(bias_strategy, X_train, y_train, args, seed)


def _timed(function, repetitions=1):
    """(value of the last call, mean seconds over the repetitions), like the runner."""
    times, value = [], None
    for _ in range(repetitions):
        start = perf_counter()
        value = function()
        times.append(perf_counter() - start)
    return value, float(np.mean(times))


def _model_kilobytes(classifier, n_features):
    """Derived size: ridge weight rows + scaler mean and scale, float64, for the kept features."""
    rows = np.atleast_2d(classifier.steps[-1][1].coef_).shape[0]   # binary coef_ is one row
    return (rows + 2) * n_features * 8 / 1024


def run_pruning(X_train, y_train, X_test, y_test, keep_fractions=DEFAULT_KEEP_FRACTIONS,
                scorer='ridge', bias_strategy='original', seed=0, num_features=10_000,
                drop_fraction=0.2, repetitions=3):
    """Prune at every keep fraction (1.0 is always added) and measure each on TEST.

    Row keys: keep_fraction, groups_kept, groups_total, features_kept, accuracy (0-1),
    accuracy_change_pp and time_ratio (both vs keep 1.0), test_seconds, selection_seconds,
    model_kb, exact, features_sha256 (same fingerprint the runner writes).
    """
    if scorer not in SCORERS:
        raise ValueError(f'scorer must be one of {SCORERS}')
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    fractions = sorted({float(f) for f in keep_fractions} | {1.0}, reverse=True)
    if not all(0 < f <= 1 for f in fractions):
        raise ValueError('keep fractions must be in (0, 1]')

    parameters = fit_parameters(X_train, y_train, bias_strategy, num_features, seed)
    everything = pruned.keep_all(parameters)
    warm = everything.copy()
    warm[0, 0] = False
    for keep in (everything, warm):                                  # compile outside the timers
        pruned.transform(X_test[:2], parameters, keep, 'joint')
    train_full = pruned.transform(X_train, parameters, everything, 'joint')
    reference_test = transform_univariate(X_test, parameters)        # untimed exactness reference

    rows, masks = [], {}
    for fraction in fractions:
        if fraction >= 1.0:
            keep, selection_seconds = everything, 0.0
        else:
            (keep, _), selection_seconds = _timed(lambda: pruned.prune_groups(
                train_full, y_train, parameters, fraction, scorer=scorer, drop_fraction=drop_fraction,
                classifier_factory=lambda n: make_classifier(n, 'ridge', seed)))
        columns = pruned.kept_columns(parameters, keep)
        classifier = make_classifier(len(X_train), 'ridge', seed).fit(train_full[:, columns], y_train)
        features, test_seconds = _timed(lambda: pruned.transform(X_test, parameters, keep, 'joint'), repetitions)
        exact = bool(np.array_equal(np.ascontiguousarray(features).view(np.uint32),
                                    np.ascontiguousarray(reference_test[:, columns]).view(np.uint32)))
        masks[fraction] = keep.copy()
        rows.append(dict(keep_fraction=fraction, groups_kept=int(keep.sum()), groups_total=int(everything.sum()),
                         features_kept=int(len(columns)),
                         accuracy=float(np.mean(classifier.predict(features) == y_test)),
                         test_seconds=test_seconds, selection_seconds=selection_seconds,
                         model_kb=_model_kilobytes(classifier, len(columns)), exact=exact,
                         features_sha256=sha(features)))

    full = rows[0]
    for row in rows:
        row['accuracy_change_pp'] = 100 * (row['accuracy'] - full['accuracy'])
        row['time_ratio'] = row['test_seconds'] / full['test_seconds'] if full['test_seconds'] > 0 else float('nan')
    settings = dict(scorer=scorer, bias_strategy=bias_strategy, seed=seed, num_features=num_features,
                    drop_fraction=drop_fraction, repetitions=repetitions, n_train=int(len(X_train)),
                    n_test=int(len(X_test)), length=int(X_train.shape[1]))
    return PruningResult(rows, masks, [int(d) for d in parameters[0]], settings)


def threshold_sources(y_train, num_groups, seed=0):
    """How many groups take their thresholds from each class: {strategy: {class: groups}}."""
    y_train = np.asarray(y_train)
    out = {}
    for strategy in ('uniform', 'stratified'):
        table = stratified.draw_example_table(y_train, num_groups, 1, strategy, seed)[:, 0]
        labels, counts = np.unique(y_train[table], return_counts=True)
        out[strategy] = {str(label): int(count) for label, count in zip(labels, counts)}
    return out


def format_table(result):
    s = result.settings
    lines = [f"Group pruning: scorer={s['scorer']}, biases={s['bias_strategy']}, seed={s['seed']}, "
             f"{s['n_train']} train / {s['n_test']} test series of length {s['length']}",
             f"{'keep':>5} {'groups':>11} {'features':>8} {'accuracy':>9} {'change':>8} "
             f"{'time':>6} {'size KB':>8} exact"]
    for r in result.rows:
        lines.append(f"{r['keep_fraction']:>5.2f} {r['groups_kept']:>5}/{r['groups_total']:<5} "
                     f"{r['features_kept']:>8} {100 * r['accuracy']:>8.2f}% {r['accuracy_change_pp']:>+7.2f}pp "
                     f"{r['time_ratio']:>5.2f}x {r['model_kb']:>8.0f} {'yes' if r['exact'] else 'NO'}")
    lines.append(f"time = mean TEST transform time over {s['repetitions']} runs, relative to the unpruned "
                 f"joint-PPV model in this session; size KB is derived from the feature count, not measured")
    return '\n'.join(lines)


def format_survivors(result):
    """Groups kept per dilation at the smallest keep fraction."""
    fraction = min(result.masks)
    keep = result.masks[fraction]
    lines = [f'Groups kept per dilation at keep {fraction:.2f} (of 84 kernels each):']
    lines += [f'  dilation {d:>4}: {int(n):>2}' for d, n in zip(result.dilations, keep.sum(axis=1))]
    return '\n'.join(lines)


def published_pruning_summary():
    """Joint-PPV, original-bias rows of results/group_pruning_full/summary.csv (113 datasets x 3 seeds)."""
    path = RESULTS / 'group_pruning_full' / 'summary.csv'
    if not path.exists():
        return []
    rows = []
    with path.open(newline='') as handle:
        for r in csv.DictReader(handle):
            if r['implementation'] != 'joint' or r['bias_strategy'] != 'original':
                continue
            rows.append(dict(scorer=r['scorer'], keep_fraction=float(r['keep_fraction']),
                             datasets=int(r['datasets']), features_kept=float(r['features_kept']),
                             accuracy_percent=float(r['accuracy_percent']),
                             accuracy_change_pp=float(r['accuracy_delta_pp_vs_full']),
                             time_ratio=float(r['test_transform_ratio_vs_full_same_impl'])))
    return sorted(rows, key=lambda r: (r['scorer'] != 'none', r['scorer'], -r['keep_fraction']))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--data', default='./data/Univariate_ts/', help='Local archive; never downloads')
    p.add_argument('--scorer', default='ridge', choices=SCORERS)
    p.add_argument('--bias-strategy', default='original', choices=BIAS_STRATEGIES)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--keep-fractions', nargs='+', type=float, default=list(DEFAULT_KEEP_FRACTIONS))
    p.add_argument('--feature-count', type=int, default=10000)
    p.add_argument('--repetitions', type=int, default=3)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    X_train, y_train, X_test, y_test = load_dataset(args.dataset, args.data)
    result = run_pruning(X_train, y_train, X_test, y_test, args.keep_fractions, args.scorer,
                         args.bias_strategy, args.seed, args.feature_count, repetitions=args.repetitions)
    print(format_table(result))
    print()
    print(format_survivors(result))
    if args.bias_strategy != 'original':
        groups = result.rows[0]['groups_total']
        print()
        print(f'Which class sets the thresholds of the {groups} groups (seed {args.seed}):')
        for strategy, counts in threshold_sources(y_train, groups, args.seed).items():
            print(f'  {strategy:<10} ' + ', '.join(f'class {c}: {n}' for c, n in counts.items()))


if __name__ == '__main__':
    main()
