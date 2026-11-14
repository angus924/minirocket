"""Summarise experiments/group_pruning.py output: Pareto table, time ratios, break-even.

  python -B -m experiments.summarise ./results/group_pruning_pilot/

Datasets get equal weight (seeds are averaged inside each dataset first). Ratios are ratios of
dataset-mean times; 95% intervals resample datasets with replacement (10,000 draws, fixed seed),
so they describe sensitivity to the dataset mix, not run-to-run noise.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

NUMERIC = ['groups_total', 'groups_kept', 'features_total', 'features_kept', 'n_test', 'fit_seconds',
           'full_train_transform_seconds', 'selection_seconds', 'classifier_fit_seconds',
           'test_transform_seconds', 'predict_seconds', 'accuracy']


def bootstrap(numerator, denominator, draws=10_000, seed=0):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    if len(numerator) < 2:
        return float('nan'), float('nan')
    index = np.random.default_rng(seed).integers(0, len(numerator), size=(draws, len(numerator)))
    ratios = numerator[index].mean(axis=1) / denominator[index].mean(axis=1)
    return tuple(np.percentile(ratios, [2.5, 97.5]))


def load(folder):
    with (folder / 'measurements.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r['keep_fraction'] = float(r['keep_fraction'])
        r['equal_to_full_columns'] = r['equal_to_full_columns'] == 'True'
        for key in NUMERIC:
            r[key] = float(r[key])
    return rows


def per_dataset_means(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['bias_strategy'], r['scorer'], r['keep_fraction'], r['implementation'], r['dataset'])].append(r)
    return {key: {k: float(np.mean([r[k] for r in items])) for k in NUMERIC} for key, items in groups.items()}


def summarise(rows):
    means = per_dataset_means(rows)
    configs = sorted({k[:4] for k in means}, key=lambda c: (c[0], c[1], -c[2], c[3]))
    datasets = sorted({k[4] for k in means})
    out = []
    for strategy, scorer, fraction, implementation in configs:
        base_key = lambda impl: (strategy, 'none', 1.0, impl)
        have = [d for d in datasets if (strategy, scorer, fraction, implementation, d) in means
                and (*base_key(implementation), d) in means and (*base_key('original'), d) in means]
        if not have:
            continue
        cfg = [means[(strategy, scorer, fraction, implementation, d)] for d in have]
        full_same = [means[(*base_key(implementation), d)] for d in have]
        full_orig = [means[(*base_key('original'), d)] for d in have]
        t = np.array([c['test_transform_seconds'] for c in cfg])
        t_same = np.array([c['test_transform_seconds'] for c in full_same])
        t_orig = np.array([c['test_transform_seconds'] for c in full_orig])
        acc = np.array([c['accuracy'] for c in cfg]); acc_full = np.array([c['accuracy'] for c in full_same])
        # total cost of ownership: training cost now, inference cost per series later
        train_cost = np.array([c['fit_seconds'] + c['full_train_transform_seconds'] + c['selection_seconds']
                               + c['classifier_fit_seconds'] for c in cfg])
        train_full = np.array([c['fit_seconds'] + c['full_train_transform_seconds'] + c['classifier_fit_seconds']
                               for c in full_same])
        infer = np.array([(c['test_transform_seconds'] + c['predict_seconds']) / c['n_test'] for c in cfg])
        infer_full = np.array([(c['test_transform_seconds'] + c['predict_seconds']) / c['n_test'] for c in full_same])
        saving = infer_full - infer
        with np.errstate(divide='ignore', invalid='ignore'):
            breakeven = np.where(saving > 0, np.maximum(train_cost - train_full, 0) / saving, np.inf)
        ratio_same = bootstrap(t, t_same); ratio_orig = bootstrap(t, t_orig)
        out.append(dict(
            bias_strategy=strategy, scorer=scorer, keep_fraction=fraction, implementation=implementation,
            datasets=len(have),
            groups_kept=np.mean([c['groups_kept'] for c in cfg]), features_kept=np.mean([c['features_kept'] for c in cfg]),
            accuracy_percent=100 * acc.mean(), accuracy_delta_pp_vs_full=100 * (acc - acc_full).mean(),
            test_transform_ratio_vs_full_same_impl=t.mean() / t_same.mean(),
            ratio_same_ci_low=ratio_same[0], ratio_same_ci_high=ratio_same[1],
            test_transform_ratio_vs_full_original=t.mean() / t_orig.mean(),
            ratio_orig_ci_low=ratio_orig[0], ratio_orig_ci_high=ratio_orig[1],
            selection_seconds_mean=np.mean([c['selection_seconds'] for c in cfg]),
            train_cost_ratio_vs_full=train_cost.mean() / train_full.mean(),
            median_breakeven_series=float(np.median(breakeven)),
            datasets_never_breaking_even=int(np.isinf(breakeven).sum())))
    return out


def write(folder, summary, rows):
    fields = list(summary[0])
    with (folder / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    mismatches = sum(not r['equal_to_full_columns'] for r in rows)
    lines = ['# Group pruning results', '',
             f'Rows: {len(rows)}. Pruned features differing from `full[:, kept]`: **{mismatches}** (expected 0).', '',
             'Ratios compare dataset-mean TEST transform time with the unpruned model. "vs full original" is the '
             'combined effect of pruning and joint PPV. Break-even is the median number of series after which '
             'the extra training cost (selection) is repaid by cheaper inference.', '']
    for strategy in dict.fromkeys(s['bias_strategy'] for s in summary):
        lines += [f'## Bias strategy: {strategy}', '',
                  '| scorer | keep | impl | groups | features | accuracy % | delta pp | ratio (same impl) [95% CI] '
                  '| ratio (vs full original) | train cost ratio | break-even series |',
                  '|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|']
        for s in (x for x in summary if x['bias_strategy'] == strategy):
            lines.append(f"| {s['scorer']} | {s['keep_fraction']:.2f} | {s['implementation']} | {s['groups_kept']:.0f} | "
                         f"{s['features_kept']:.0f} | {s['accuracy_percent']:.2f} | {s['accuracy_delta_pp_vs_full']:+.2f} | "
                         f"{s['test_transform_ratio_vs_full_same_impl']:.3f} [{s['ratio_same_ci_low']:.3f}, {s['ratio_same_ci_high']:.3f}] | "
                         f"{s['test_transform_ratio_vs_full_original']:.3f} | {s['train_cost_ratio_vs_full']:.2f} | "
                         f"{s['median_breakeven_series']:.0f} |")
        lines.append('')
    (folder / 'report.md').write_text('\n'.join(lines) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    folder = parser.parse_args(argv).folder
    rows = load(folder)
    summary = summarise(rows)
    if not summary:
        raise SystemExit('No complete configurations to summarise (need keep_fraction 1.0 rows).')
    write(folder, summary, rows)
    print(f'Wrote {folder / "summary.csv"} and {folder / "report.md"}')


if __name__ == '__main__':
    main()
