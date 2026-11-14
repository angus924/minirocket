"""Analyse the stratified-bias study (and the keep-fraction slices of the combined run).

  python -B -m experiments.analyse_bias results/group_pruning_bias_30ds_10seeds/ \
      --subsets results/bias_study_subset.csv

Input: measurements.csv written by experiments.group_pruning. Rows with scorer 'none' and
keep_fraction 1.0 are the full model; with --by-keep-fraction every (scorer, keep_fraction) slice is
analysed separately (use that for the combined run).

The comparison that counts is STRATIFIED vs UNIFORM (same code path, same RNG policy). ORIGINAL vs
UNIFORM only shows the effect of the random-number policy. Accuracy does not depend on the pooling
implementation, so only the 'joint' rows are used.

Per dataset: mean and standard deviation (ddof=1) of accuracy across seeds. Across datasets: paired
difference of means (stratified - uniform), Wilcoxon signed-rank test, wins/ties/losses, and whether
the seed-to-seed standard deviation is lower for stratified. Subgroups come from the subset file
(group column: imbalanced = smallest class share < 20% of TRAIN, small = under 100 training series
and not imbalanced, rest). Datasets absent from the subset file are labelled 'unlabelled'.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(folder):
    with (folder / 'measurements.csv').open(newline='') as handle:
        rows = [r for r in csv.DictReader(handle) if r['implementation'] == 'joint']
    for r in rows:
        r['accuracy'] = float(r['accuracy'])
        r['keep_fraction'] = float(r['keep_fraction'])
        r['seed'] = int(r['seed'])
    return rows


def wilcoxon(differences):
    """Return (n_nonzero, p) or (n_nonzero, nan) when the test is not defined."""
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0]
    if len(nonzero) < 5:
        return len(nonzero), float('nan')
    from scipy.stats import wilcoxon as test
    return len(nonzero), float(test(nonzero).pvalue)


def per_dataset(rows):
    """{(strategy, dataset): accuracies ordered by seed}"""
    cells = defaultdict(dict)
    for r in rows:
        cells[(r['bias_strategy'], r['dataset'])][r['seed']] = r['accuracy']
    return {k: np.array([v[s] for s in sorted(v)]) for k, v in cells.items()}


def compare(cells, datasets, a, b):
    """Paired comparison of strategy a minus strategy b over datasets that have both."""
    have = [d for d in datasets if (a, d) in cells and (b, d) in cells and len(cells[(a, d)]) == len(cells[(b, d)])]
    if not have:
        return None
    mean_a = np.array([cells[(a, d)].mean() for d in have]); mean_b = np.array([cells[(b, d)].mean() for d in have])
    std_a = np.array([cells[(a, d)].std(ddof=1) if len(cells[(a, d)]) > 1 else 0.0 for d in have])
    std_b = np.array([cells[(b, d)].std(ddof=1) if len(cells[(b, d)]) > 1 else 0.0 for d in have])
    diff = mean_a - mean_b
    n_acc, p_acc = wilcoxon(diff)
    n_std, p_std = wilcoxon(std_a - std_b)
    return dict(n=len(have), mean_a=mean_a.mean(), mean_b=mean_b.mean(), diff_pp=100 * diff.mean(),
                median_diff_pp=100 * float(np.median(diff)), wins=int((diff > 0).sum()), ties=int((diff == 0).sum()),
                losses=int((diff < 0).sum()), p_acc=p_acc, n_acc=n_acc, std_a_pp=100 * std_a.mean(),
                std_b_pp=100 * std_b.mean(), std_lower=int((std_a < std_b).sum()), std_higher=int((std_a > std_b).sum()),
                std_equal=int((std_a == std_b).sum()), p_std=p_std, n_std=n_std)


def fmt_p(p):
    return 'n/a' if np.isnan(p) else f'{p:.3f}'


def table(title, cells, subsets, strategies_present):
    datasets = sorted({d for (_, d) in cells})
    labels = {d: subsets.get(d, 'unlabelled') for d in datasets}
    splits = [('all datasets', datasets)] + [(g, [d for d in datasets if labels[d] == g])
                                             for g in ('imbalanced', 'small', 'rest') if any(labels[d] == g for d in datasets)]
    lines = [f'### {title}', '']
    pairs = [(a, b, role) for a, b, role in (('stratified', 'uniform', 'MAIN: stratification effect'),
                                              ('original', 'uniform', 'control: RNG policy only'))
             if a in strategies_present and b in strategies_present]
    for a, b, role in pairs:
        lines += [f'**{a} - {b}** ({role})', '',
                  '| subgroup | datasets | mean acc % (a) | mean acc % (b) | mean diff pp | median diff pp | wins/ties/losses '
                  '| Wilcoxon p (n) | seed std pp (a) | seed std pp (b) | std lower/equal/higher for a | Wilcoxon p on std (n) |',
                  '|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|']
        for name, members in splits:
            c = compare(cells, members, a, b)
            if c is None:
                continue
            lines.append(f"| {name} | {c['n']} | {100 * c['mean_a']:.2f} | {100 * c['mean_b']:.2f} | {c['diff_pp']:+.3f} | "
                         f"{c['median_diff_pp']:+.3f} | {c['wins']}/{c['ties']}/{c['losses']} | {fmt_p(c['p_acc'])} ({c['n_acc']}) | "
                         f"{c['std_a_pp']:.3f} | {c['std_b_pp']:.3f} | {c['std_lower']}/{c['std_equal']}/{c['std_higher']} | "
                         f"{fmt_p(c['p_std'])} ({c['n_std']}) |")
        lines.append('')
    return lines


def per_dataset_table(cells, subsets):
    strategies = sorted({s for (s, _) in cells})
    lines = ['### Per dataset: mean +- std of accuracy (%) across seeds, [min, max]', '',
             '| dataset | group | seeds | ' + ' | '.join(strategies) + ' |', '|---|---|---:|' + '---|' * len(strategies)]
    for d in sorted({d for (_, d) in cells}):
        seeds = max(len(cells[(s, d)]) for s in strategies if (s, d) in cells)
        parts = []
        for s in strategies:
            a = cells.get((s, d))
            parts.append('-' if a is None else f'{100 * a.mean():.2f} +- {100 * a.std(ddof=1) if len(a) > 1 else 0:.2f} '
                                               f'[{100 * a.min():.1f}, {100 * a.max():.1f}]')
        lines.append(f"| {d} | {subsets.get(d, 'unlabelled')} | {seeds} | " + ' | '.join(parts) + ' |')
    return lines + ['']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path)
    p.add_argument('--subsets', type=Path, default=Path('results/bias_study_subset.csv'))
    p.add_argument('--by-keep-fraction', action='store_true', help='analyse each (scorer, keep_fraction) slice')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    rows = load(args.folder)
    subsets = {}
    if args.subsets.exists():
        with args.subsets.open(newline='') as handle:
            subsets = {r['dataset']: r['group'] for r in csv.DictReader(handle)}
    output = args.output or args.folder / ('combined_analysis.md' if args.by_keep_fraction else 'bias_analysis.md')

    seeds = sorted({r['seed'] for r in rows})
    lines = ['# Stratified vs uniform bias sampling', '',
             f'Folder `{args.folder}`. Rows analysed: {len(rows)} (joint implementation only). '
             f'Seeds: {len(seeds)} ({seeds[0]}..{seeds[-1]}). Datasets: {len({r["dataset"] for r in rows})}.',
             'Comparison that counts: stratified vs uniform. Original vs uniform isolates the RNG-policy change. '
             'Accuracy differences are in percentage points (pp). Wilcoxon signed-rank on per-dataset mean differences; '
             'the test is skipped (n/a) with fewer than 5 non-zero differences. Subgroups of about 10 datasets give '
             'very little power: read them as descriptive.', '']
    if args.by_keep_fraction:
        slices = sorted({(r['scorer'], r['keep_fraction']) for r in rows}, key=lambda x: (x[0], -x[1]))
    else:
        slices = [('none', 1.0)]
    for scorer, fraction in slices:
        subset = [r for r in rows if r['scorer'] == scorer and r['keep_fraction'] == fraction]
        if not subset:
            continue
        cells = per_dataset(subset)
        present = {s for (s, _) in cells}
        lines += table(f'scorer={scorer}, keep_fraction={fraction}', cells, subsets, present)
        if not args.by_keep_fraction:
            lines += per_dataset_table(cells, subsets)
    output.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
