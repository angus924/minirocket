"""Does choosing the groups matter? Compare pruned models with random masks and smaller fresh models.

  python -B -m experiments.analyse_controls        # writes results/controls/controls_analysis.md and pareto.png

Unit = dataset (seeds averaged first). Wilcoxon signed-rank on per-dataset accuracy differences.
Time ratio = sum over datasets of mean TEST transform time / same for the unpruned model, where each set of rows
is divided by the unpruned model timed IN ITS OWN SESSION (primary run for ridge/filter, controls run for
random/small), so no cross-session time is compared directly.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

R = Path(__file__).resolve().parent.parent / 'results'
KEEPS = (0.75, 0.5, 0.3, 0.2, 0.1)


def load():
    """{(method, keep): {dataset: (mean accuracy, mean test time, mean features)}} plus full references."""
    cells = defaultdict(lambda: defaultdict(list))
    with (R / 'group_pruning_full' / 'measurements.csv').open(newline='') as h:
        for r in csv.DictReader(h):
            if r['implementation'] != 'joint' or r['bias_strategy'] != 'original':
                continue
            method = 'full_primary' if r['scorer'] == 'none' else r['scorer']
            cells[(method, float(r['keep_fraction']))][r['dataset']].append(
                (float(r['accuracy']), float(r['test_transform_seconds']), int(r['features_kept'])))
    with (R / 'controls' / 'measurements.csv').open(newline='') as h:
        for r in csv.DictReader(h):
            method = 'full_controls' if r['control'] == 'full' else r['control']
            cells[(method, float(r['keep_fraction']))][r['dataset']].append(
                (float(r['accuracy']), float(r['test_transform_seconds']), int(r['features_kept'])))
    return {k: {d: tuple(np.mean(v, axis=0)) for d, v in by.items()} for k, by in cells.items()}


def paired(a, b, datasets):
    d = np.array([a[x][0] - b[x][0] for x in datasets])
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else float('nan')
    return 100 * d.mean(), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), p


def ratio(cell, full, datasets):
    return sum(cell[d][1] for d in datasets) / sum(full[d][1] for d in datasets)


def main():
    c = load()
    full_p, full_c = c[('full_primary', 1.0)], c[('full_controls', 1.0)]
    datasets = sorted(set(full_p) & set(full_c))
    for k in KEEPS:
        for m in ('ridge', 'filter', 'random', 'small_ridge', 'small_filter'):
            datasets = [d for d in datasets if d in c.get((m, k), {})]
    acc_full = 100 * np.mean([full_p[d][0] for d in datasets])
    same = sum(abs(full_p[d][0] - full_c[d][0]) < 1e-12 for d in datasets)
    ref = {'ridge': full_p, 'filter': full_p, 'random': full_c, 'small_ridge': full_c, 'small_filter': full_c}
    L = ['# Controls: does choosing the groups matter?', '',
         f'Datasets: {len(datasets)} (seeds 0-2 averaged within each). Full-model accuracy {acc_full:.2f}%; the controls run '
         f'reproduced the primary full-model accuracy on {same}/{len(datasets)} datasets (same parameters).',
         'Time = TEST transform (joint PPV) relative to the unpruned model timed in the same session.',
         '`random` keeps the same number of groups as the pruned models; `small_ridge` / `small_filter` are fresh, smaller '
         'MiniRocket models with the feature count of the ridge / filter model (rounded down to a multiple of 84).', '',
         '## Accuracy and time per method', '',
         '| keep | method | features | accuracy % | delta vs full pp | time ratio |', '|---:|---|---:|---:|---:|---:|']
    for k in KEEPS:
        for m in ('ridge', 'filter', 'random', 'small_ridge', 'small_filter'):
            cell = c[(m, k)]
            acc = 100 * np.mean([cell[d][0] for d in datasets])
            feats = np.mean([cell[d][2] for d in datasets])
            L.append(f'| {k} | {m} | {feats:,.0f} | {acc:.2f} | {acc - acc_full:+.2f} | {ratio(cell, ref[m], datasets):.3f} |')
    L += ['', '## Paired tests (accuracy, method A minus method B)', '',
          '| keep | A - B | mean diff pp | wins/ties/losses | Wilcoxon p |', '|---:|---|---:|---|---:|']
    for k in KEEPS:
        for a, b in (('ridge', 'random'), ('filter', 'random'), ('ridge', 'small_ridge'), ('filter', 'small_filter')):
            m, w, t, l, p = paired(c[(a, k)], c[(b, k)], datasets)
            L.append(f'| {k} | {a} - {b} | {m:+.3f} | {w}/{t}/{l} | {p:.4f} |')
    L += ['', '20 tests, unadjusted. With 20 tests about 1 p < 0.05 is expected by chance; read patterns across keep levels.', '']
    (R / 'controls' / 'controls_analysis.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    plot(c, datasets, full_p, full_c, acc_full)


def plot(c, datasets, full_p, full_c, acc_full):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Default validated categorical slots 1-4 (fixed order), plus marker and line-style redundancy.
    series = [('ridge', 'ridge-pruned', '#2a78d6', 'o', '-', full_p),
              ('filter', 'filter-pruned', '#eb6834', 's', '--', full_p),
              ('random', 'random groups', '#1baf7a', '^', ':', full_c),
              ('small_ridge', 'smaller MiniRocket', '#eda100', 'D', '-.', full_c)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for key, label, colour, marker, style, full in series:
        xs = [1.0] + [ratio(c[(key, k)], full, datasets) for k in KEEPS]
        ys = [acc_full] + [100 * np.mean([c[(key, k)][d][0] for d in datasets]) for k in KEEPS]
        ax.plot(xs, ys, linestyle=style, linewidth=2, color=colour, marker=marker, markersize=8,
                markeredgecolor='white', markeredgewidth=1.5, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points', fontsize=8,
                    color='#3b3b38', va='center')
    ax.set_xlabel('TEST transform time, relative to the unpruned model (joint PPV)')
    ax.set_ylabel('mean accuracy over datasets (%)')
    ax.set_title(f'Accuracy vs test time, {len(datasets)} UCR datasets, 3 seeds', fontsize=11, loc='left')
    ax.grid(color='#e6e5e0', linewidth=0.8); ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#b5b4ab')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_xlim(0, 1.12)
    fig.tight_layout()
    fig.savefig(R / 'controls' / 'pareto.png')
    print('Wrote', R / 'controls' / 'pareto.png')


if __name__ == '__main__':
    main()
