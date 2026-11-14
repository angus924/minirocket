"""Compare the macOS joint-PPV study with the Windows main run, fit by fit (no new runs).

  python -B -m experiments.cross_platform         # prints the summary, writes results/cross_platform.md
  python -B -m experiments.cross_platform --recompute ./data/Univariate_ts/
      # also refits all 339 fits here (about 20 minutes) and compares every stage with the macOS hashes;
      # writes results/cross_platform_stages.csv. Without --recompute an existing CSV is summarised again.

Both runs fit the ORIGINAL MiniRocket with 9,996 features on the supplied split, seeds 0-2:
  macOS    results/joint_ppv_2026-10-26/measurements.csv   (implementation 'original')
  Windows  results/group_pruning_full/measurements.csv     (keep 1.0, 'original' thresholds, original transform)
The script checks that both used the same split, then counts the fits whose accuracy differs.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / 'results'


def read(path, keep):
    with path.open(newline='') as handle:
        return {(r['dataset'], int(r['seed'])): r for r in csv.DictReader(handle) if keep(r)}


def main():
    mac = read(R / 'joint_ppv_2026-10-26' / 'measurements.csv',
               lambda r: r['implementation'] == 'original' and r['requested_features'] == '10000')
    win = read(R / 'group_pruning_full' / 'measurements.csv',
               lambda r: r['implementation'] == 'original' and r['bias_strategy'] == 'original'
               and float(r['keep_fraction']) == 1.0)
    keys = sorted(set(mac) & set(win))
    same_split = sum(mac[k]['split_sha256'] == win[k]['split_sha256'] for k in keys)
    differences = []                       # (dataset, seed, test series, accuracy difference in series)
    for dataset, seed in keys:
        n_test = int(mac[(dataset, seed)]['n_test'])
        series = round((float(mac[(dataset, seed)]['accuracy']) - float(win[(dataset, seed)]['accuracy'])) * n_test)
        if series:
            differences.append((dataset, seed, n_test, series))
    datasets = sorted({d for d, *_ in differences})
    per_dataset = {d: np.mean([float(mac[k]['accuracy']) - float(win[k]['accuracy']) for k in keys if k[0] == d])
                   for d in sorted({k[0] for k in keys})}
    mean_pp = 100 * float(np.mean(list(per_dataset.values())))
    sizes = [abs(s) for *_, s in differences]
    largest = sorted(differences, key=lambda x: -abs(x[3]))[:5]
    # yardstick: how often does a DIFFERENT seed on the SAME machine (Windows) keep the accuracy?
    seed_pairs = [(d, a, b) for d in sorted({k[0] for k in keys}) for a, b in ((0, 1), (0, 2), (1, 2))
                  if (d, a) in win and (d, b) in win]
    seed_same = sum(win[(d, a)]['accuracy'] == win[(d, b)]['accuracy'] for d, a, b in seed_pairs)
    lines = [
        f'fits compared (dataset x seed): {len(keys)}',
        f'same split hash: {same_split} of {len(keys)}',
        f'identical accuracy: {len(keys) - len(differences)}; different: {len(differences)} '
        f'on {len(datasets)} datasets',
        f'size of a difference: {min(sizes)} to {max(sizes)} test series '
        f'(more accurate on macOS in {sum(s > 0 for *_, s in differences)}, on Windows in '
        f'{sum(s < 0 for *_, s in differences)})',
        f'mean difference over datasets (macOS - Windows): {mean_pp:+.3f} pp',
        'largest: ' + '; '.join(f'{d} seed {s}: {n:+d} of {t} series' for d, s, t, n in largest),
        f'yardstick, different seed on Windows: identical accuracy in {seed_same} of {len(seed_pairs)} seed pairs '
        f'(same seed on the two machines: {len(keys) - len(differences)} of {len(keys)})',
    ]
    text = '\n'.join(lines)
    print(text)
    (R / 'cross_platform.md').write_text('# macOS vs Windows, original MiniRocket, fit by fit\n\n```\n' + text + '\n```\n')
    if STAGES_CSV.exists():                # keep the stage summary without redoing the 20-minute refit
        stage_summary()
    return mac, keys


STAGES = ('fit_input_sha256', 'extractor_parameters_sha256', 'train_feature_sha256', 'test_feature_sha256',
          'scaler_parameters_sha256', 'classifier_parameters_sha256', 'prediction_sha256')
STAGES_CSV = R / 'cross_platform_stages.csv'


def stage_summary():
    """Summarise results/cross_platform_stages.csv and append it to results/cross_platform.md."""
    with STAGES_CSV.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    columns = [s.replace('_sha256', '_equal') for s in STAGES]
    lines = [f'Fits refitted on Windows: {len(rows)}. Stage identical to the macOS hash:']
    for column in columns:
        lines.append(f'  {column:<32} {sum(r[column] == "True" for r in rows):>4} of {len(rows)}')
    lines.append('First stage that differs (fit input excluded):')
    first = {}
    for r in rows:
        stage = next((c for c in columns[1:] if r[c] != 'True'), 'none')
        first.setdefault(stage, []).append(r['accuracy_differs'] == 'True')
    for stage in columns[1:] + ['none']:
        if stage in first:
            lines.append(f'  {stage:<32} {len(first[stage]):>4} fits, of which {sum(first[stage])} have a '
                         f'different accuracy on this machine than on macOS')
    text = '\n'.join(lines)
    print(text)
    with (R / 'cross_platform.md').open('a') as handle:
        handle.write('\n## Refit on Windows, compared stage by stage with the macOS hashes\n\n```\n' + text + '\n```\n')


def recompute(mac, keys, data):
    """Refit every macOS fit on THIS machine and compare the macOS harness's stage hashes.

    Uses the same calls and the same hash function as benchmarks/worker.py, so a matching hash
    means the arrays are identical bit for bit. The first stage that differs says where the
    platforms part ways: thresholds, features, scaler, classifier or predictions.
    """
    from argparse import Namespace
    from benchmarks.common import _dataset_sources, load_benchmark_dataset, make_classifier
    from benchmarks.worker import array_hash
    from minirocket_baseline import fit_univariate, transform_univariate
    from benchmarks.common import verify_feature_values
    sources = {name: (train, test) for name, train, test in _dataset_sources(Namespace(data=data, datasets=None))}
    loaded = {}
    out = STAGES_CSV.open('w', newline='')
    writer = csv.writer(out)
    writer.writerow(['dataset', 'seed', 'accuracy_differs'] + [s.replace('_sha256', '_equal') for s in STAGES])
    for dataset_name, seed in keys:
        if dataset_name not in loaded:
            loaded.clear()
            loaded[dataset_name] = load_benchmark_dataset(Namespace(data=data, datasets=[dataset_name]),
                                                          dataset_name, *sources[dataset_name], [])
        dataset = loaded[dataset_name]
        X_train, X_test = np.ascontiguousarray(dataset.X_train[:, 0]), np.ascontiguousarray(dataset.X_test[:, 0])
        parameters = fit_univariate(X_train, num_features=10000, max_dilations_per_kernel=32, random_state=seed)
        train, test = transform_univariate(X_train, parameters), transform_univariate(X_test, parameters)
        classifier = make_classifier(len(X_train), 'ridge', seed).fit(train, dataset.y_train)
        scaler, estimator = classifier.steps[0][1], classifier.steps[-1][1]
        predictions = classifier.predict(test)
        ours = dict(fit_input_sha256=array_hash(X_train, dataset.y_train),
                    extractor_parameters_sha256=array_hash(*parameters),
                    train_feature_sha256=verify_feature_values(train), test_feature_sha256=verify_feature_values(test),
                    scaler_parameters_sha256=array_hash(scaler.mean_, scaler.var_, scaler.scale_, scaler.n_samples_seen_),
                    classifier_parameters_sha256=array_hash(estimator.coef_, estimator.intercept_, estimator.classes_),
                    prediction_sha256=array_hash(predictions))
        row = dict(mac[(dataset_name, seed)])
        for split in ('train', 'test'):        # the CSV keeps one hash per timing repetition
            row[f'{split}_feature_sha256'] = json.loads(row[f'{split}_feature_repetition_sha256'])[-1]
        equal = [ours[s] == row[s] for s in STAGES]
        differs = float(row['accuracy']) != float(np.mean(predictions == dataset.y_test))
        writer.writerow([dataset_name, seed, differs] + equal)
        out.flush()
    out.close()


if __name__ == '__main__':
    import sys
    if '--recompute' in sys.argv:          # refit first, so main() summarises the fresh CSV
        STAGES_CSV.unlink(missing_ok=True)
    mac, keys = main()
    if '--recompute' in sys.argv:
        recompute(mac, keys, sys.argv[sys.argv.index('--recompute') + 1] if len(sys.argv) > sys.argv.index('--recompute') + 1
                  else './data/Univariate_ts/')
        stage_summary()
