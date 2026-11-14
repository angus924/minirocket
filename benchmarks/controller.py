"""Focused original versus exact SIMD joint-PPV archive comparison."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from datetime import datetime, timezone

import numpy as np

try:
    from .common import ROOT, _dataset_sources, _environment, _shorten, load_benchmark_dataset
except ImportError:
    if __package__:
        raise
    from common import ROOT, _dataset_sources, _environment, _shorten, load_benchmark_dataset


EXECUTION_ORDER_SEED = 20261026


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,help='Local archive; never downloads')
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--max-dilations-per-kernel',type=int,default=32)
    parser.add_argument('--classifier',choices=['ridge'],default='ridge')
    parser.add_argument('--verbose',action='store_true')
    parser.set_defaults(output=Path('results/local/joint_ppv_full_archive.csv'),
                        order_seed=EXECUTION_ORDER_SEED)
    parser.add_argument('--feature-counts', dest='budgets', nargs='+', type=int, default=[2500,5000,10000],
                        metavar='COUNT', help='Requested feature counts, rounded down to multiples of 84')
    parser.add_argument('--transform-repetitions', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--inventory',type=Path,help='Frozen preflight compatibility/fingerprints')
    return parser


def build_variants(args):
    return [dict(name=f'{family}_{budget}',implementation=family,num_features=budget)
            for budget in args.budgets for family in ('original','joint_ppv')]


def case_order(dataset,seed,variants,order_seed):
    identity=int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:8],'little')
    budgets=list(dict.fromkeys(v['num_features'] for v in variants))
    rng=np.random.default_rng(np.random.SeedSequence([order_seed,identity,seed]))
    ordered=[]
    for i in rng.permutation(len(budgets)):
        pair=[v for v in variants if v['num_features']==budgets[i]]
        ordered.extend(pair[j] for j in rng.permutation(len(pair)))
    return ordered


def source_hashes():
    files=['code/minirocket.py','code/minirocket_baseline.py',
           'code/minirocket_joint_ppv_simd.py','code/_joint_ppv_simd_kernels.py',
           'benchmarks/common.py','benchmarks/memory.py',
           'benchmarks/controller.py','benchmarks/worker.py',
           'benchmarks/inventory.py',
           'tools/generate_joint_ppv.py']
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}


def worker_environment(threads):
    return dict(os.environ,NUMBA_NUM_THREADS=str(threads),NUMBA_NRT_STATS='0',
                OPENBLAS_NUM_THREADS=str(threads),OMP_NUM_THREADS=str(threads),
                MKL_NUM_THREADS=str(threads),VECLIB_MAXIMUM_THREADS=str(threads),
                BLIS_NUM_THREADS=str(threads))


def runtime_signature(args):
    from llvmlite.binding import get_host_cpu_name,get_host_cpu_features
    from importlib.metadata import version
    return dict(llvmlite=version('llvmlite'),threadpoolctl=version('threadpoolctl'),
                cpu_name=get_host_cpu_name(),cpu_features=dict(get_host_cpu_features()),
                numpy_build=np.__config__.CONFIG,
                numba_environment={k:v for k,v in os.environ.items() if k.startswith('NUMBA_')},
                worker_thread_environment={k:v for k,v in worker_environment(args.threads).items()
                    if k in ('NUMBA_NUM_THREADS','NUMBA_NRT_STATS','OPENBLAS_NUM_THREADS',
                             'OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')})


def execute_worker(job, directory, threads):
    path = Path(directory) / 'job.json'
    path.write_text(json.dumps(job))
    env = worker_environment(threads)
    start = perf_counter()
    module = 'benchmarks.worker'
    completed = subprocess.run([sys.executable, '-B', '-m', module, '--job', str(path)],
                               cwd=ROOT, env=env, text=True, capture_output=True)
    wall = perf_counter() - start
    if completed.returncode:
        raise RuntimeError(f"Worker failed: {job['dataset']}/{job['variant']['name']}/seed{job['seed']}\n"
                           f'{completed.stderr}\n{completed.stdout}')
    try:
        row = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f'Worker returned invalid JSON: {completed.stdout}') from error
    row['worker_wall_seconds'] = wall
    return row


def print_dataset(rows, verbose=False):
    if not rows:
        return
    names = list(dict.fromkeys(row['model'] for row in rows))
    widths = (20, 6, 10, 11, 11, 12)
    def line(cells):
        return '| ' + ' | '.join(f'{_shorten(cell, width):{"<" if i == 0 else ">"}{width}}'
                                  for i, (cell, width) in enumerate(zip(cells, widths))) + ' |'
    print(f'\n{_shorten(rows[0]["dataset"], 64)} | {"per-seed results" if verbose else "averages across seeds"}', flush=True)
    print(line(('Model', 'Seed' if verbose else 'Seeds', 'Accuracy', 'Train (s)', 'Test (s)', 'Peak (MiB)')), flush=True)
    print('|' + '|'.join('-' * (width + 2) for width in widths) + '|', flush=True)
    for name in names:
        group = [row for row in rows if row['model'] == name]
        for values in ([[row] for row in group] if verbose else [group]):
            mean = lambda key: float(np.mean([float(row[key]) for row in values]))
            print(line((name, values[0]['seed'] if verbose else len(values), f'{mean("accuracy"):.2%}',
                        f'{mean("training_seconds"):.4f}', f'{mean("test_seconds"):.4f}',
                        f'{mean("peak_memory_bytes") / 2**20:.1f}')), flush=True)


def _config(args):
    return {key: str(value) if isinstance(value,Path) else value
            for key, value in vars(args).items() if key not in ('resume', 'overwrite', 'verbose', 'output')}


def _read_existing(output):
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row['dataset'], row['model'], int(row['seed'])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Resume CSV contains duplicate runs')
    return rows, set(keys)


EXACT_KEYS = ('split_sha256','requested_features','features','fit_input_sha256',
              'train_feature_sha256','test_feature_sha256','extractor_parameters_sha256',
              'parameter_array_bytes','extractor_parameter_array_bytes','total_parameter_array_bytes',
              'scaler_parameter_array_bytes','classifier_parameter_array_bytes',
              'scaler_parameters_sha256','classifier_parameters_sha256','classifier_alpha',
              'prediction_sha256','accuracy')


def validate_pairs(rows):
    by_key={(r['dataset'],r['model'],int(r['seed'])):r for r in rows}
    for r in rows:
        if not r['model'].startswith('joint_ppv_'):continue
        budget=r['model'].rsplit('_',1)[1]
        control=by_key.get((r['dataset'],f'original_{budget}',int(r['seed'])))
        if control is None:continue
        for key in EXACT_KEYS:
            equal=(float(r[key])==float(control[key]) if key in ('accuracy','classifier_alpha')
                   else str(r[key])==str(control[key]))
            if not equal:raise ValueError(f'Exact original/joint mismatch: {r["dataset"]}/{budget}/{key}')


def validate_inventory(saved, current):
    def normalized(sources):
        return sorted((str(name), str(train), str(test) if test is not None else None)
                      for name, train, test in sources)
    if normalized(saved) != normalized(current):
        raise ValueError('Resume source inventory changed: missing, added or relocated dataset splits')


def run_pipeline(args, variants=None):
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the inode after closing: deleting lock files can allow two holders.
    with output.with_suffix(output.suffix + '.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Benchmark already running: output lock held for {output}') from error
        return _run_pipeline_locked(args, variants)


def _run_pipeline_locked(args, variants=None):
    if (args.threads<1 or args.batch_size<1 or args.max_dilations_per_kernel<1 or
        not args.budgets or len(set(args.budgets))!=len(args.budgets) or min(args.budgets)<84 or
        args.transform_repetitions<1 or args.order_seed<0 or not args.seeds or
        len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0 or max(args.seeds)>=2**32):
        raise ValueError('Invalid configuration; previous results are never overwritten')
    output = Path(args.output).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args) if variants is None else variants
    frozen=json.loads(args.inventory.read_text()) if args.inventory else None
    frozen_splits={r['name']:r for r in frozen['datasets']} if frozen else {}
    metadata = _environment(args)
    metadata.update(configuration_signature=_config(args),source_sha256=source_hashes(),
                    runtime_signature=runtime_signature(args),
                    archive_inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest() if frozen else None,
                    status='running',models=variants,datasets=[],skipped_datasets=[],completed_runs=0,
                    executable=str(Path(sys.executable).resolve()),
                    memory_policy='Fresh-process lifetime physical footprint preferred; RSS also recorded. Includes imports, training-only warmup, input, fit, repeated outputs, scaling/classifier and earlier hashing. Controller archive parsing excluded. No subtracting or summing lifetime peaks as allocations.',
                    allocation_policy='Identical legacy batch-copy delivery for both implementations.',
                    timing_policy='Three transforms per split, means reported; nested model-call timing during same calls. Hashing and warmup excluded. Pipeline sums prepared-input load, fit, mean transforms, classifier fit and prediction. Worker wall includes startup/warmup/all repeats/hashes.',
                    ordering_policy='Dataset and seed order randomized; same-budget pairs adjacent with randomized method order, derived from fixed order seed and dataset name only, independent of test values/labels/results.',
                    leakage_policy='Extractor fit, JIT warmup, feature scaler and RidgeClassifierCV fit use TRAIN only. TEST is transformed/scored after fitting. Scaling and extractor fit are conditional on full outer TRAIN before internal RidgeCV; internal CV scores are not claimed unbiased or fully nested.')
    rows, done = [], set()
    if args.resume:
        if not output.exists() or not sidecar.exists():
            raise ValueError('Resume requires both CSV and metadata')
        previous = json.loads(sidecar.read_text())
        for key in ('configuration_signature', 'source_sha256', 'versions', 'models',
                    'python', 'platform', 'executable','runtime_signature','archive_inventory_sha256'):
            if previous.get(key) != metadata[key]:
                raise ValueError(f'Resume configuration/provenance mismatch: {key}')
        metadata = previous
        rows, done = _read_existing(output)
        saved_splits = {item['name']: item['fingerprint'] for item in metadata['datasets']}
        expected_models = {variant['name'] for variant in variants}
        for row in rows:
            if (row['dataset'] not in saved_splits or row['model'] not in expected_models
                    or int(row['seed']) not in args.seeds
                    or row.get('split_sha256') != saved_splits[row['dataset']]):
                raise ValueError('Resume row identity or split differs from saved provenance')
    elif output.exists() or sidecar.exists():
        raise FileExistsError(f'Refusing to replace {output}')
    def save():
        temporary = sidecar.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(metadata, indent=2, default=str) + '\n')
        temporary.replace(sidecar)
    metadata['status'] = 'running'
    metadata.pop('error', None)
    save()
    try:
        prewarm=subprocess.run([sys.executable,'-B','-m','benchmarks.worker','--prewarm'],
                              cwd=ROOT,env=worker_environment(args.threads),text=True,capture_output=True)
        if prewarm.returncode:raise RuntimeError(f'JIT prewarm failed: {prewarm.stderr}')
        metadata['jit_cache_policy']=json.loads(prewarm.stdout)
        save()
        sources = _dataset_sources(args)
        if getattr(args, 'order_seed', None) is not None:
            source_rng = np.random.default_rng(np.random.SeedSequence([args.order_seed, 1]))
            sources = [sources[i] for i in source_rng.permutation(len(sources))]
        inventory = [(name, str(train), str(test) if test is not None else None)
                     for name, train, test in sources]
        if frozen:validate_inventory(frozen['source_inventory'],inventory)
        if args.resume:
            validate_inventory(metadata.get('source_inventory', []), inventory)
        metadata['source_inventory'] = inventory
        metadata['discovered_datasets'] = len(sources)
        known = {item['name']: item for item in metadata['datasets']}
        metadata['skipped_datasets'] = []
        with tempfile.TemporaryDirectory(prefix='minirocket-pipeline-') as directory:
            path = Path(directory) / 'split.npz'
            with output.open('a' if args.resume else 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])) if rows else None
                for name, train, test in sources:
                    dataset = load_benchmark_dataset(args, name, train, test, metadata['skipped_datasets'])
                    if dataset is None:
                        if frozen and name in frozen_splits:
                            raise ValueError(f'Frozen inventory split became incompatible: {name}')
                        if args.resume and name in known:
                            raise ValueError(f'Resume split became incompatible: {name}')
                        save()
                        continue
                    if dataset.X_train.shape[1] != 1:
                        if args.datasets or Path(args.data).is_file():
                            raise ValueError('This benchmark requires univariate datasets')
                        metadata['skipped_datasets'].append(dict(name=name,sources=dataset.sources,
                                                                 reason='Requires univariate data'))
                        save()
                        continue
                    if frozen and (name not in frozen_splits or
                                   dataset.fingerprint!=frozen_splits[name]['fingerprint']):
                        raise ValueError(f'Frozen inventory fingerprint differs: {name}')
                    if name in known and known[name]['fingerprint'] != dataset.fingerprint:
                        raise ValueError(f'Resume split fingerprint changed: {name}')
                    if name not in known:
                        info = dict(name=name, sources=dataset.sources, fingerprint=dataset.fingerprint,
                                    n_train=len(dataset.X_train), n_test=len(dataset.X_test), length=dataset.X_train.shape[-1])
                        metadata['datasets'].append(info)
                        known[name] = info
                    fingerprint = dataset.fingerprint
                    np.savez(path, X_train=dataset.X_train, y_train=dataset.y_train,
                             X_test=dataset.X_test, y_test=dataset.y_test)
                    del dataset
                    save()
                    dataset_rows = [row for row in rows if row['dataset'] == name]
                    validate_pairs(dataset_rows)
                    seed_order = args.seeds
                    if getattr(args, 'order_seed', None) is not None:
                        seed_rng = np.random.default_rng(np.random.SeedSequence(
                            [args.order_seed, int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little'), 2]))
                        seed_order = [args.seeds[i] for i in seed_rng.permutation(len(args.seeds))]
                    for seed in seed_order:
                        order=case_order(name,seed,variants,args.order_seed)
                        for order_position, variant in enumerate(order):
                            key = (name, variant['name'], seed)
                            if key in done:
                                continue
                            print(f'Running {name:24.24} {variant["name"]:16} seed={seed}', flush=True)
                            job = dict(input=str(path), dataset=name, fingerprint=fingerprint,
                                       seed=seed, variant=variant, configuration=_config(args),
                                       order_position=order_position)
                            row = execute_worker(job, directory, args.threads)
                            if (row['dataset'], row['model'], row['seed'], row['split_sha256']) != (*key, fingerprint):
                                raise ValueError('Worker returned a different dataset/model/seed/split')
                            dataset_rows.append(row)
                            validate_pairs(dataset_rows)
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            handle.flush()
                            rows.append(row)
                            done.add(key)
                            metadata['completed_runs'] = len(rows)
                            metadata['last_run'] = dict(dataset=name, model=variant['name'], seed=seed)
                            save()
                    print_dataset(dataset_rows, args.verbose)
        if not rows:
            raise ValueError('No compatible datasets found')
        expected = len(metadata['datasets']) * len(variants) * len(args.seeds)
        if len(rows) != expected:
            raise ValueError('Run count differs from complete dataset/model/seed coverage')
        metadata.update(status='complete', completed_runs=len(rows), expected_runs=expected)
        save()
        print(f'\nComplete: {output}\nConfiguration: {sidecar}', flush=True)
        return output
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), completed_runs=len(rows))
        save()
        raise


def main(argv=None):
    return run_pipeline(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
