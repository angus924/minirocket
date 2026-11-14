"""Dataset-bootstrap intervals for the accuracy change of pruning and of stratified thresholds.

  python -B -m experiments.equivalence            # prints a table, writes results/equivalence.md

Seeds are averaged within each dataset first; the dataset is the unit that is resampled
(10,000 resamples, generator seed 0). A 90% interval lying inside +-m says the change is
smaller than m (the TOST decision at alpha = 0.05), but the margin m must be fixed BEFORE
looking at the interval (Lakens 2017), so this script reports the interval, not a margin.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'results'
RESAMPLES = 10_000


def per_dataset(folder, **flt):
    cell = defaultdict(list)
    with (R / folder / 'measurements.csv').open(newline='') as handle:
        for r in csv.DictReader(handle):
            if r['implementation'] == 'joint' and all(r[k] == v for k, v in flt.items()):
                cell[r['dataset']].append(float(r['accuracy']))
    return {d: float(np.mean(v)) for d, v in cell.items()}


def interval(a, b, level=90):
    """Mean of a - b over datasets (pp) and its dataset-bootstrap interval."""
    datasets = sorted(set(a) & set(b))
    diff = 100 * np.array([a[d] - b[d] for d in datasets])
    rng = np.random.default_rng(0)
    means = diff[rng.integers(0, len(diff), (RESAMPLES, len(diff)))].mean(axis=1)
    low, high = np.percentile(means, [(100 - level) / 2, 100 - (100 - level) / 2])
    return len(datasets), float(diff.mean()), float(low), float(high)


def main():
    lines = ['| Comparison | Datasets | Mean change pp | 90% interval | 95% interval |', '|---|---:|---:|---|---|']
    full = per_dataset('group_pruning_full', scorer='none', bias_strategy='original')
    for scorer in ('filter', 'ridge'):
        for keep in ('0.75', '0.5', '0.3', '0.2', '0.1'):
            pruned = per_dataset('group_pruning_full', scorer=scorer, keep_fraction=keep, bias_strategy='original')
            n, mean, lo90, hi90 = interval(pruned, full, 90)
            _, _, lo95, hi95 = interval(pruned, full, 95)
            lines.append(f'| {scorer}, keep {float(keep):.0%} vs full | {n} | {mean:+.3f} | '
                         f'{lo90:+.2f} to {hi90:+.2f} | {lo95:+.2f} to {hi95:+.2f} |')
    strat = per_dataset('group_pruning_bias_all_20seeds', bias_strategy='stratified')
    unif = per_dataset('group_pruning_bias_all_20seeds', bias_strategy='uniform')
    n, mean, lo90, hi90 = interval(strat, unif, 90)
    _, _, lo95, hi95 = interval(strat, unif, 95)
    lines.append(f'| stratified vs uniform (20 seeds) | {n} | {mean:+.3f} | '
                 f'{lo90:+.3f} to {hi90:+.3f} | {lo95:+.3f} to {hi95:+.3f} |')
    text = '\n'.join(lines)
    print(text)
    (R / 'equivalence.md').write_text(
        '# Bootstrap intervals for accuracy changes (from the committed measurements.csv files)\n\n'
        'Seeds averaged per dataset, datasets resampled 10,000 times (generator seed 0).\n\n' + text + '\n')


if __name__ == '__main__':
    main()
