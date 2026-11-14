"""Paper-vs-ours accuracy table (reproduction evidence for the report and README).

Reference = the authors' published per-dataset numbers from angus924/minirocket
(results_ucr109_mean.csv and accuracy_ucr109_resamples.csv; restore them from commit 0b1c245,
see README_GROUP_PRUNING.md step 2). Ours = ORIGINAL MiniRocket from a benchmark run.

Like-for-like column: the authors' resample 0 (the paper says the 30 resamples include the default
split) against our supplied-split accuracy. The 30-resample mean is NOT comparable to a single split.

  python -B -m experiments.compare_to_paper \
      --ours results/joint_ppv_2026-10-26/per_dataset.csv --features 9996 \
      --reference-dir results/reference --output results/reference/paper_vs_ours.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def read(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ours', type=Path, default=Path('results/joint_ppv_2026-10-26/per_dataset.csv'))
    p.add_argument('--features', type=int, default=9996, help='actual feature count to compare')
    p.add_argument('--reference-dir', type=Path, default=Path('results/reference'))
    p.add_argument('--output', type=Path, default=Path('results/reference/paper_vs_ours.csv'))
    args = p.parse_args(argv)

    ours = {r['dataset']: float(r['original_accuracy_percent']) / 100
            for r in read(args.ours) if int(r['features']) == args.features}
    resamples = {r['dataset']: np.array([float(v) for k, v in r.items() if k != 'dataset']) for r in
                 read(args.reference_dir / 'accuracy_ucr109_resamples.csv')}
    means = {r['dataset']: float(r['accuracy']) for r in read(args.reference_dir / 'results_ucr109_mean.csv')}

    common = sorted(set(ours) & set(resamples))
    extra = sorted(set(ours) - set(resamples))
    rows = []
    for d in common:
        reference0, std = float(resamples[d][0]), float(resamples[d].std())
        diff = ours[d] - reference0
        rows.append(dict(dataset=d, paper_resample0=round(reference0, 6), paper_30_resample_mean=means[d],
                         paper_std_across_resamples=round(std, 6), ours=round(ours[d], 6),
                         diff_pp=round(100 * diff, 3),
                         diff_in_paper_stds=round(diff / std, 2) if std > 0 else ''))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    diffs = np.array([r['diff_pp'] for r in rows])
    zs = np.array([r['diff_in_paper_stds'] for r in rows if r['diff_in_paper_stds'] != ''], dtype=float)
    print(f'Compared {len(rows)} datasets (ours also has {len(extra)} extra: {", ".join(extra) or "-"})')
    print(f'Mean accuracy: paper resample 0 = {100 * np.mean([r["paper_resample0"] for r in rows]):.2f}% | '
          f'ours = {100 * np.mean([r["ours"] for r in rows]):.2f}% | '
          f'paper 30-resample mean = {100 * np.mean([r["paper_30_resample_mean"] for r in rows]):.2f}% (different protocol)')
    print(f'Difference (ours - paper resample 0), pp: mean {diffs.mean():+.3f}, median {np.median(diffs):+.3f}, '
          f'|diff| > 2 pp on {(np.abs(diffs) > 2).sum()} datasets, > 2 paper-stds on {(np.abs(zs) > 2).sum()}')
    try:
        from scipy.stats import wilcoxon
        nonzero = diffs[diffs != 0]
        if len(nonzero) >= 10:
            print(f'Wilcoxon signed-rank on paired differences: p = {wilcoxon(nonzero).pvalue:.3f}')
    except ImportError:
        print('scipy not installed: skipped Wilcoxon test')
    worst = sorted(rows, key=lambda r: -abs(r['diff_pp']))[:5]
    print('Largest differences:', ', '.join(f"{r['dataset']} {r['diff_pp']:+.1f}pp" for r in worst))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
