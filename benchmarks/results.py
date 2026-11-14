"""Canonical measurements and paired original/joint benchmark reports."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean, median

import numpy as np

CORE_FILES = ('report.md', 'metadata.json', 'measurements.csv', 'per_dataset.csv', 'summary.csv', 'verification.json')
RAW_FIELDS = ('dataset', 'implementation', 'seed', 'requested_features', 'features',
    'n_train', 'n_test', 'length', 'accuracy', 'fit_seconds', 'classifier_fit_seconds',
    'predict_seconds', 'load_seconds', 'train_transform_seconds', 'test_transform_seconds',
    'train_transform_repetition_seconds', 'test_transform_repetition_seconds',
    'train_feature_repetition_sha256', 'test_feature_repetition_sha256', 'split_sha256',
    'fit_input_sha256', 'extractor_parameters_sha256', 'scaler_parameters_sha256',
    'classifier_parameters_sha256', 'prediction_sha256', 'classifier_alpha',
    'extractor_parameter_array_bytes', 'scaler_parameter_array_bytes',
    'classifier_parameter_array_bytes', 'train_feature_bytes', 'test_feature_bytes',
    'peak_memory_bytes', 'peak_memory_metric', 'peak_rss_bytes', 'case_order_position')
INT_FIELDS = ('seed', 'requested_features', 'features', 'n_train', 'n_test', 'length',
    'extractor_parameter_array_bytes', 'scaler_parameter_array_bytes',
    'classifier_parameter_array_bytes', 'train_feature_bytes', 'test_feature_bytes',
    'peak_memory_bytes', 'peak_rss_bytes', 'case_order_position')
EXACT_FIELDS = ('features', 'n_train', 'n_test', 'length', 'accuracy', 'split_sha256',
    'fit_input_sha256', 'extractor_parameters_sha256', 'scaler_parameters_sha256',
    'classifier_parameters_sha256', 'prediction_sha256', 'classifier_alpha',
    'extractor_parameter_array_bytes', 'scaler_parameter_array_bytes',
    'classifier_parameter_array_bytes', 'train_feature_bytes', 'test_feature_bytes',
    'train_feature_repetition_sha256', 'test_feature_repetition_sha256')
STAGES = ('fit_seconds', 'classifier_fit_seconds', 'predict_seconds', 'load_seconds',
          'train_transform_seconds', 'test_transform_seconds')
PAIRED_METRICS = dict(train='train_transform_seconds', test='test_transform_seconds',
    pipeline='pipeline_seconds', model_cost='model_cost_seconds', memory='peak_memory_bytes', rss='peak_rss_bytes')


def _ratio_summary(candidate, baseline, samples=10000):
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if (a.ndim != 1 or a.shape != b.shape or not len(a) or samples < 1
            or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0)):
        raise ValueError('Require matching finite positive dataset means')
    rng = np.random.default_rng(0)
    ratios = np.empty(samples)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indices = rng.integers(len(a), size=(stop - start, len(a)))
        ratios[start:stop] = a[indices].mean(axis=1) / b[indices].mean(axis=1)
    low, high = np.quantile(ratios, [.025, .975]) if len(a) > 1 else (None, None)
    return dict(ratio_of_arithmetic_means=float(a.mean() / b.mean()),
                mean_per_dataset_ratio=float(np.mean(a / b)),
                ratio_of_means_ci95_low=None if low is None else float(low),
                ratio_of_means_ci95_high=None if high is None else float(high))



def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty result table')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_measurements(path, rows):
    write_csv(path, [{k: r[k] for k in RAW_FIELDS} for r in rows])


def project_worker(row, threads):
    if (int(row['numba_threads']) != threads or int(row['transform_cache_hits']) < 1
            or int(row['transform_cache_misses']) or int(row['fit_cache_misses'])):
        raise ValueError('Worker thread or compilation-cache checks failed')
    if any(p['num_threads'] != threads for p in json.loads(row['threadpools_json'])):
        raise ValueError('Worker exposed BLAS thread count differs')
    return {k: row[k] for k in RAW_FIELDS}


def _parse(row, repetitions):
    if set(RAW_FIELDS)-set(row):
        raise ValueError('Measurement columns are missing')
    r = {k: row[k] for k in RAW_FIELDS}
    for k in INT_FIELDS:
        value = float(r[k]); r[k] = int(value)
        if r[k] != value: raise ValueError(f'Noninteger {k}')
    for k in (*STAGES, 'accuracy', 'classifier_alpha'):
        r[k] = float(r[k])
        if not math.isfinite(r[k]): raise ValueError(f'Nonfinite {k}')
    if not 0 <= r['accuracy'] <= 1 or r['classifier_alpha'] <= 0:
        raise ValueError('Invalid accuracy or classifier regularization')
    if any(r[k] <= 0 for k in STAGES) or any(r[k] <= 0 for k in INT_FIELDS if k not in ('seed', 'case_order_position')):
        raise ValueError('Nonpositive timing, dimension or memory')
    if r['seed'] < 0 or r['case_order_position'] < 0 or r['length'] < 9:
        raise ValueError('Invalid case identity or input length')
    if r['features'] != r['requested_features']//84*84:
        raise ValueError('Feature budget rounding differs')
    for k in ('split_sha256', 'fit_input_sha256', 'extractor_parameters_sha256', 'scaler_parameters_sha256', 'classifier_parameters_sha256', 'prediction_sha256'):
        if len(r[k]) != 64 or any(c not in '0123456789abcdef' for c in r[k]): raise ValueError('Malformed fingerprint')
    for split in ('train', 'test'):
        key = f'{split}_transform_repetition_seconds'; samples = json.loads(r[key])
        hashes = json.loads(r[f'{split}_feature_repetition_sha256'])
        if (len(samples) != repetitions or len(hashes) != repetitions or len(set(hashes)) != 1
                or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes)
                or any(not math.isfinite(t) or t <= 0 for t in samples)
                or not math.isclose(mean(samples), r[f'{split}_transform_seconds'], rel_tol=1e-12)):
            raise ValueError('Invalid, unstable or inconsistent transform repetitions')
        if r[f'{split}_feature_bytes'] != r[f'n_{split}']*r['features']*4:
            raise ValueError('Output byte count differs from float32 shape')
        r[f'{split}_median_seconds'] = median(samples)
    r['model_cost_seconds'] = sum(r[k] for k in STAGES if k != 'load_seconds')
    r['pipeline_seconds'] = r['model_cost_seconds'] + r['load_seconds']
    r['fitted_array_bytes'] = sum(r[k] for k in ('extractor_parameter_array_bytes', 'scaler_parameter_array_bytes', 'classifier_parameter_array_bytes'))
    return r


def analyze(rows, config):
    datasets, seeds, budgets = config['datasets'], config['seeds'], config['budgets']
    if not datasets or not seeds or not budgets or any(len(x) != len(set(x)) for x in (datasets, seeds, budgets)):
        raise ValueError('Empty or duplicate experiment dimensions')
    if config['bootstrap_seed'] != 0 or config['bootstrap_samples'] < 1 or not 0 <= config['similarity_tolerance'] < 1:
        raise ValueError('Invalid bootstrap or practical-similarity settings')
    rows = [_parse(r, config['repetitions']) for r in rows]
    index = {(r['dataset'], r['implementation'], r['seed'], r['requested_features']): r for r in rows}
    expected = set(product(datasets, ('original', 'joint_ppv'), seeds, budgets))
    if len(index) != len(rows) or set(index) != expected:
        raise ValueError('Missing, duplicate or unexpected cases')
    if len({r['peak_memory_metric'] for r in rows}) != 1:
        raise ValueError('Cannot compare different memory definitions')
    for d in datasets:
        identities = {(r['split_sha256'], r['fit_input_sha256'], r['n_train'], r['n_test'], r['length']) for r in rows if r['dataset'] == d}
        if len(identities) != 1:
            raise ValueError(f'Dataset input changed across seeds or budgets: {d}')
    for d, seed, budget in product(datasets, seeds, budgets):
        a, b = (index[d, m, seed, budget] for m in ('original', 'joint_ppv'))
        for k in EXACT_FIELDS:
            if a[k] != b[k]: raise ValueError(f'Exact original/joint mismatch: {d}/{seed}/{budget}/{k}')
        if a['case_order_position'] == b['case_order_position']:
            raise ValueError('Methods have the same execution-order position')
    numeric = (*STAGES, 'accuracy', 'pipeline_seconds', 'model_cost_seconds', 'peak_memory_bytes',
        'peak_rss_bytes', 'train_median_seconds', 'test_median_seconds', 'fitted_array_bytes',
        'extractor_parameter_array_bytes', 'scaler_parameter_array_bytes', 'classifier_parameter_array_bytes',
        'train_feature_bytes', 'test_feature_bytes')
    groups = defaultdict(list)
    for r in rows: groups[r['dataset'], r['implementation'], r['requested_features']].append(r)
    means = {key: {k: mean(r[k] for r in values) for k in numeric} for key, values in groups.items()}
    per_dataset, summaries = [], []
    for budget in budgets:
        dsrows = []
        for d in sorted(datasets):
            a, b = (means[d, m, budget] for m in ('original', 'joint_ppv'))
            case = groups[d, 'original', budget][0]
            p = dict(dataset=d, requested_features=budget, features=case['features'], seeds=len(seeds),
                n_train=case['n_train'], n_test=case['n_test'], length=case['length'],
                original_accuracy_percent=100*a['accuracy'], joint_accuracy_percent=100*b['accuracy'],
                accuracy_delta_pp=100*(b['accuracy']-a['accuracy']), exact_equal=True)
            for k in (*STAGES, 'pipeline_seconds', 'model_cost_seconds', 'peak_memory_bytes', 'peak_rss_bytes'):
                p['original_'+k], p['joint_'+k] = a[k], b[k]
            for label, metric in PAIRED_METRICS.items(): p[label+'_ratio'] = b[metric]/a[metric]
            for split in ('train', 'test'):
                p[f'{split}_median_repetition_ratio'] = b[f'{split}_median_seconds']/a[f'{split}_median_seconds']
            for k in ('fitted_array_bytes', 'extractor_parameter_array_bytes', 'scaler_parameter_array_bytes', 'classifier_parameter_array_bytes', 'train_feature_bytes', 'test_feature_bytes'):
                p[k] = a[k]  # Exactly equal in each matched case.
            dsrows.append(p)
        per_dataset.extend(dsrows)
        s = dict(requested_features=budget, features=budget//84*84, datasets=len(datasets), seeds=len(seeds),
            matched_cases=len(datasets)*len(seeds), original_accuracy_percent=mean(p['original_accuracy_percent'] for p in dsrows),
            joint_accuracy_percent=mean(p['joint_accuracy_percent'] for p in dsrows),
            accuracy_delta_pp=mean(p['accuracy_delta_pp'] for p in dsrows), exact_equal=True)
        for label, metric in PAIRED_METRICS.items():
            a = [means[d, 'original', budget][metric] for d in sorted(datasets)]
            b = [means[d, 'joint_ppv', budget][metric] for d in sorted(datasets)]
            stats = _ratio_summary(b, a, samples=config['bootstrap_samples'])
            s['original_'+metric], s['joint_'+metric] = mean(a), mean(b)
            s[label+'_ratio'] = stats['ratio_of_arithmetic_means']
            s[label+'_ratio_ci95_low'] = stats['ratio_of_means_ci95_low']
            s[label+'_ratio_ci95_high'] = stats['ratio_of_means_ci95_high']
            if label in ('train', 'test'):
                s[label+'_time_reduction_percent'] = 100*(1-s[label+'_ratio'])
                s['median_dataset_'+label+'_ratio'] = median(p[label+'_ratio'] for p in dsrows)
                s[label+'_median_repetition_ratio'] = mean(means[d,'joint_ppv',budget][label+'_median_seconds'] for d in datasets)/mean(means[d,'original',budget][label+'_median_seconds'] for d in datasets)
                tol = config['similarity_tolerance']
                s[label+'_faster_datasets'] = sum(p[label+'_ratio'] < 1-tol for p in dsrows)
                s[label+'_slower_datasets'] = sum(p[label+'_ratio'] > 1+tol for p in dsrows)
                s[label+'_similar_datasets'] = len(dsrows)-s[label+'_faster_datasets']-s[label+'_slower_datasets']
        for method, tag in (('original', 'original'), ('joint_ppv', 'joint')):
            subset = [r for r in rows if r['implementation'] == method and r['requested_features'] == budget]
            worst = max(subset, key=lambda r: r['peak_memory_bytes'])
            s[tag+'_worst_peak_bytes'], s[tag+'_worst_peak_dataset'], s[tag+'_worst_peak_seed'] = worst['peak_memory_bytes'], worst['dataset'], worst['seed']
        for k in ('fitted_array_bytes','train_feature_bytes','test_feature_bytes'):
            s['mean_'+k] = mean(p[k] for p in dsrows)
        for label in ('original_first', 'joint_first'):
            values = []
            for d, seed in product(datasets, seeds):
                a, b = (index[d,m,seed,budget] for m in ('original','joint_ppv'))
                first = 'original_first' if a['case_order_position'] < b['case_order_position'] else 'joint_first'
                if first == label: values.append(b['test_transform_seconds']/a['test_transform_seconds'])
            s[label+'_pairs'] = len(values)
            s[label+'_median_test_ratio'] = median(values) if values else None
        summaries.append(s)
    verification = dict(measured_cases=len(rows), matched_cases=len(rows)//2,
        datasets=len(datasets), feature_budgets=len(budgets),
        stable_feature_hashes=len(rows)*config['repetitions']*2,
        features_predictions_and_state_equal=True, fitted_and_output_array_sizes_equal=True,
        full_coverage=True, finite_timings_and_coherent_means=True,
        dataset_fingerprints={d: index[d,'original',seeds[0],budgets[0]]['split_sha256'] for d in sorted(datasets)})
    return per_dataset, summaries, verification


def _config(metadata):
    return dict(metadata['configuration'], datasets=metadata['included_datasets'])


def _read_csv(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def _same_table(path, expected):
    saved = _read_csv(path)
    if len(saved) != len(expected): raise ValueError('Saved table coverage differs')
    for a,b in zip(saved,expected):
        if set(a) != set(b): raise ValueError('Saved table columns differ')
        for k,v in b.items():
            if v is None: equal = a[k] == ''
            elif isinstance(v,bool): equal = a[k] == str(v)
            elif isinstance(v,(int,float)): equal = math.isclose(float(a[k]),v,rel_tol=1e-11,abs_tol=1e-12)
            else: equal = a[k] == str(v)
            if not equal: raise ValueError(f'Saved value differs: {path.name}/{k}')


def _report(meta, summary, checks, folder=None):
    config = meta['configuration']
    datasets = checks['datasets']
    lines = ['# MiniRocket benchmark results', '',
        f"Original MiniRocket versus joint PPV on **{datasets} datasets**. "
        f"Features, predictions and accuracy matched in all **{checks['matched_cases']:,} paired comparisons**.", '',
        '## Reading the timing tables', '',
        '**Ratio = joint PPV mean time / original mean time.** '
        'A ratio of 0.70 means 30% less time; 1.10 means 10% more time. '
        'Time reduction is 100 x (1 - ratio), so negative values mean slower.', '',
        '**95% CI is the confidence interval for the overall time ratio beside it.** '
        f"We resample {datasets} dataset pairs with replacement, calculate the ratio of their mean times, "
        f"and repeat {config['bootstrap_samples']:,} times. The middle 95% of those ratios forms the interval. "
        'This describes sensitivity to the dataset mix, not the range of individual dataset ratios '
        'or a guarantee for future runs.', '',
        'Times are seconds for a whole split. We average repetitions and fitting seeds within each '
        'dataset, then give datasets equal weight when calculating the means. '
        'One dataset cannot provide a dataset-bootstrap interval; it is shown as unavailable.']

    def interval(row, label):
        lo, hi = row[label+'_ratio_ci95_low'], row[label+'_ratio_ci95_high']
        value = f"{row[label+'_ratio']:.3f}"
        return value+' [unavailable]' if lo is None else f'{value} [{lo:.3f}, {hi:.3f}]'

    for split, title in [('train', 'TRAIN transform time'), ('test', 'TEST transform time')]:
        lines += ['', f'## {title}', '',
            '| Features | Original (s) | Joint PPV (s) | Time ratio [95% CI] | Time reduction |',
            '|---:|---:|---:|---|---:|']
        for row in summary:
            lines.append(f"| {row['features']:,} | {row['original_'+split+'_transform_seconds']:.4g} | "
                         f"{row['joint_'+split+'_transform_seconds']:.4g} | {interval(row, split)} | "
                         f"{100*(1-row[split+'_ratio']):.1f}% |")

    lines += ['', '## Accuracy and other costs', '',
        '| Features | Accuracy, both methods | Full pipeline time reduction | Mean worker peak memory, original / joint (MiB) | Peak memory change |',
        '|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['features']:,} | {row['original_accuracy_percent']:.2f}% | "
                     f"{100*(1-row['pipeline_ratio']):.1f}% | "
                     f"{row['original_peak_memory_bytes']/2**20:.2f} / {row['joint_peak_memory_bytes']/2**20:.2f} | "
                     f"{100*(row['memory_ratio']-1):+.1f}% |")
    lines += ['', 'Accuracy is the mean dataset TEST accuracy. Full pipeline time includes prepared-input '
        'loading, extractor fitting, TRAIN/TEST transforms, classifier fitting and prediction. '
        'Memory is the whole worker process peak, including warmup and classification; it is not '
        'transform-only allocation. Positive memory change means higher memory use.', '',
        '## Dataset exceptions', '',
        f"Using a {100*config['similarity_tolerance']:g}% band around equal time:", '']
    for row in summary:
        lines.append(f"- **{row['features']:,} features:** TEST transforms were slower on "
                     f"{row['test_slower_datasets']} of {datasets} datasets.")
    lines += ['', 'These are descriptive counts. See [per_dataset.csv](per_dataset.csv) for all '
        'dataset results, including TRAIN regressions.', '',
        '## Scope and details', '',
        'Both methods use the same supplied TRAIN/TEST splits and classifier pipeline. '
        'Learning uses TRAIN only. This previously explored archive on one machine provides '
        'hardware-dependent evidence, not a universal speed guarantee or an accuracy improvement.', '',
        '- [summary.csv](summary.csv): all aggregate values and intervals, including pipeline/memory '
        'intervals and sensitivity checks.',
        '- [measurements.csv](measurements.csv): individual cases and timing repetitions.',
        '- [metadata.json](metadata.json): data source, hardware, software and settings.',
        '- [verification.json](verification.json): correctness and file-integrity checks.']
    root = Path(__file__).resolve().parents[1]
    if folder is not None and Path(folder).resolve().is_relative_to(root):
        depth = len(Path(folder).resolve().relative_to(root).parts)
        guide = '/'.join(['..']*depth + ['benchmarks', 'README.md'])
        lines.append(f'- [Benchmark guide]({guide}): full methodology and instructions.')
    return '\n'.join(lines)+'\n'


def finalize(folder):
    folder=Path(folder)
    if (folder/'verification.json').exists():
        verify_package(folder)
        return _read_csv(folder/'summary.csv')
    meta=json.loads((folder/'metadata.json').read_text());rows=_read_csv(folder/'measurements.csv')
    per,summary,checks=analyze(rows,_config(meta))
    partials=[folder/name for name in ('per_dataset.csv','summary.csv','report.md') if (folder/name).exists()]
    if partials:
        # Preserve interrupted derived outputs outside the six-file package.
        handle,archive=tempfile.mkstemp(prefix='.'+folder.name+'-interrupted-',suffix='.tar.gz',dir=folder.parent)
        __import__('os').close(handle)
        with tarfile.open(archive,'w:gz') as backup:
            for p in partials:backup.add(p,arcname=p.name)
    staging=Path(tempfile.mkdtemp(prefix='.'+folder.name+'-finalize-',dir=folder.parent))
    try:
        write_csv(staging/'per_dataset.csv',per);write_csv(staging/'summary.csv',summary)
        meta.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
        (staging/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        (staging/'report.md').write_text(_report(meta,summary,checks,folder=folder))
        checks['runtime_validation']=meta.get('runtime_validation',{})
        checks['artifacts_sha256']={name:hashlib.sha256((folder/name if name=='measurements.csv' else staging/name).read_bytes()).hexdigest() for name in CORE_FILES if name!='verification.json'}
        (staging/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
        # Verification is published last. A partial publication can be rebuilt.
        for name in ('per_dataset.csv','summary.csv','metadata.json','report.md','verification.json'):
            (staging/name).replace(folder/name)
    finally:
        shutil.rmtree(staging)
    verify_package(folder)
    return summary


def verify_package(folder):
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text());checks=json.loads((folder/'verification.json').read_text())
    if set(checks['artifacts_sha256']) != set(CORE_FILES)-{'verification.json'} or meta['status'] != 'complete':
        raise ValueError('Incomplete package integrity coverage or status')
    for name,digest in checks['artifacts_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Artifact integrity check failed: {name}')
    per,summary,actual=analyze(_read_csv(folder/'measurements.csv'),_config(meta))
    if any(checks[k]!=v for k,v in actual.items()):raise ValueError('Verification summary differs from measurements')
    _same_table(folder/'per_dataset.csv',per);_same_table(folder/'summary.csv',summary)
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_folder',type=Path);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();result=verify_package(args.run_folder) if args.verify else finalize(args.run_folder)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
