"""Final paired tests computed from the committed measurements (no new timing).

  python -B -m experiments.final_analysis            # writes results/final_analysis_tables.md

1. Primary pruning run: pruned vs full accuracy, and filter vs ridge accuracy, per keep fraction.
   Unit = dataset (seeds averaged first); Wilcoxon signed-rank on per-dataset differences.
2. Combined run (30 datasets x 5 seeds): stratified - uniform for every (scorer, keep) slice and subgroup,
   with Holm correction over ALL of those tests, to show which nominal p-values survive multiplicity.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'results'


def rows(folder):
    with (R / folder / 'measurements.csv').open(newline='') as h:
        out = [r for r in csv.DictReader(h) if r['implementation'] == 'joint']
    for r in out:
        r['accuracy'] = float(r['accuracy']); r['keep_fraction'] = float(r['keep_fraction'])
    return out


def per_dataset(rs, **flt):
    cell = defaultdict(list)
    for r in rs:
        if all(r[k] == v for k, v in flt.items()):
            cell[r['dataset']].append(r['accuracy'])
    return {d: float(np.mean(v)) for d, v in cell.items()}


def paired(a, b):
    ds = sorted(set(a) & set(b))
    d = np.array([a[x] - b[x] for x in ds])
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else float('nan')
    return len(ds), 100 * d.mean(), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), p


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i]); adj[i] = min(1.0, running)
    return adj


def main():
    L = ['# Final paired tests (computed from committed measurements.csv files)', '']
    rs = rows('group_pruning_full')
    full = per_dataset(rs, scorer='none', bias_strategy='original')
    L += [f'## 1. Primary run `group_pruning_full` ({len(full)} datasets, seeds averaged within dataset)', '',
          '| keep | scorer | datasets | mean diff pp vs full | wins/ties/losses | Wilcoxon p |', '|---:|---|---:|---:|---|---:|']
    for keep in (0.75, 0.5, 0.3, 0.2, 0.1):
        for scorer in ('ridge', 'filter'):
            n, m, w, t, l, p = paired(per_dataset(rs, scorer=scorer, keep_fraction=keep, bias_strategy='original'), full)
            L.append(f'| {keep} | {scorer} | {n} | {m:+.3f} | {w}/{t}/{l} | {p:.4f} |')
    L += ['', '**filter - ridge** (same keep fraction; note filter keeps more features at the same group count)', '',
          '| keep | datasets | mean diff pp | wins/ties/losses | Wilcoxon p |', '|---:|---:|---:|---|---:|']
    for keep in (0.75, 0.5, 0.3, 0.2, 0.1):
        n, m, w, t, l, p = paired(per_dataset(rs, scorer='filter', keep_fraction=keep, bias_strategy='original'),
                                  per_dataset(rs, scorer='ridge', keep_fraction=keep, bias_strategy='original'))
        L.append(f'| {keep} | {n} | {m:+.3f} | {w}/{t}/{l} | {p:.4f} |')

    rc = rows('group_pruning_combined_30ds_5seeds')
    with (R / 'bias_study_subset.csv').open(newline='') as h:
        grp = {r['dataset']: r['group'] for r in csv.DictReader(h)}
    tests = []
    for scorer, keep in sorted({(r['scorer'], r['keep_fraction']) for r in rc}):
        a = per_dataset(rc, scorer=scorer, keep_fraction=keep, bias_strategy='stratified')
        b = per_dataset(rc, scorer=scorer, keep_fraction=keep, bias_strategy='uniform')
        for name in ('all', 'imbalanced', 'small', 'rest'):
            ks = [d for d in a if name == 'all' or grp.get(d) == name]
            n, m, w, t, l, p = paired({d: a[d] for d in ks}, {d: b[d] for d in ks})
            tests.append((scorer, keep, name, n, m, w, t, l, p))
    valid = [i for i, t in enumerate(tests) if not np.isnan(t[-1])]
    adj = holm(np.array([tests[i][-1] for i in valid]))
    adjusted = {i: a for i, a in zip(valid, adj)}
    L += ['', f'## 2. Combined run: stratified - uniform, {len(valid)} testable slices, Holm-corrected over all of them', '',
          '| scorer | keep | subgroup | datasets | mean diff pp | wins/ties/losses | raw p | Holm p |', '|---|---:|---|---:|---:|---|---:|---:|']
    for i, (s, k, g, n, m, w, t, l, p) in enumerate(tests):
        if not np.isnan(p) and (p < 0.1 or g == 'all'):
            L.append(f'| {s} | {k} | {g} | {n} | {m:+.3f} | {w}/{t}/{l} | {p:.4f} | {adjusted[i]:.3f} |')
    L += ['', f'Raw p < 0.05: {sum(tests[i][-1] < 0.05 for i in valid)} of {len(valid)}; '
              f'Holm p < 0.05: {int((adj < 0.05).sum())} of {len(valid)}. '
              f'Under no effect about {0.05 * len(valid):.1f} raw p-values below 0.05 are expected by chance. '
              '(The slices are correlated, so Holm is conservative; the point is that isolated nominal hits are not evidence.)', '']
    (R / 'final_analysis_tables.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
