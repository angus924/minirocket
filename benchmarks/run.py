"""Run original and pool8 joint PPV and produce one verified result package."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from . import controller, results
from .inventory import inspect_archive
from .common import ROOT, make_classifier


BOOTSTRAP_RESAMPLES = 10_000


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = p.add_argument_group('Required inputs')
    required.add_argument('--data', required=True, help='Local archive directory or split NPZ file')
    required.add_argument('--output-dir', required=True, type=Path, help='New directory for the six result files')
    experiment = p.add_argument_group('Optional experiment settings')
    experiment.add_argument('--datasets', nargs='+', help='Dataset names to select; omit for all compatible datasets')
    experiment.add_argument('--feature-counts', dest='budgets', nargs='+', type=int,
                            default=[2500, 5000, 10000], metavar='COUNT',
                            help='Requested feature counts; each rounds down to a multiple of 84')
    experiment.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2], help='Model-fitting seeds')
    experiment.add_argument('--threads', type=int, default=1, help='Requested native threads per worker')
    experiment.add_argument('--batch-size', type=int, default=256,
                            help='Time series per transform call; the full feature matrix is still retained')
    experiment.add_argument('--max-dilations-per-kernel', type=int, default=32, help='Dilation cap')
    experiment.add_argument('--transform-repetitions', type=int, default=3, help='Measured transforms per split')
    experiment.add_argument('--similarity-tolerance', type=float, default=.02,
                            help='Fractional band around timing ratio 1 for descriptive faster/similar/slower counts')
    metadata = p.add_argument_group('Optional dataset metadata')
    metadata.add_argument('--dataset-label', help='Descriptive label for the input dataset collection')
    metadata.add_argument('--dataset-source-url', help='Record where the data came from; does not download data')
    recovery = p.add_argument_group('Recovery')
    recovery.add_argument('--resume', action='store_true', help='Resume a recorded failed run after all workers have exited')
    p.epilog = ('Execution-order randomization and 10,000 paired bootstrap resamples '
                'are fixed protocol settings recorded in metadata.json.')
    return p


def _json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def _hardware():
    def sysctl(key):
        try:return subprocess.check_output(['sysctl','-n',key],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):return None
    cpu=sysctl('machdep.cpu.brand_string') if sys.platform=='darwin' else platform.processor()
    ram=sysctl('hw.memsize') if sys.platform=='darwin' else None
    if ram is None:
        try:ram=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
        except (OSError,ValueError):ram=None
    return dict(cpu=cpu or platform.machine(),architecture=platform.machine(),ram_bytes=int(ram) if ram else None,
                os=platform.system(),os_release=platform.release(),
                macos_version=platform.mac_ver()[0] or None,logical_cpus=os.cpu_count())


def _software():
    import numpy as np
    from llvmlite.binding import llvm_version_info
    versions={}
    for name in ('numba','llvmlite','numpy','scipy','scikit-learn','aeon','threadpoolctl'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    backend=np.__config__.CONFIG.get('Build Dependencies',{}).get('blas',{})
    return dict(python=platform.python_version(),packages=versions,llvm=list(llvm_version_info),
                blas=dict(backend=backend.get('name','unknown'),version=backend.get('version','unknown')))


def run(args):
    if args.threads<1 or args.batch_size<1 or args.max_dilations_per_kernel<1 or args.transform_repetitions<1 or not 0<=args.similarity_tolerance<1:
        raise ValueError('Invalid experiment settings')
    for values in (args.seeds,args.budgets):
        if not values or len(values)!=len(set(values)):raise ValueError('Empty or duplicate experiment dimensions')
    if min(args.seeds)<0 or max(args.seeds)>=2**32 or min(args.budgets)<84:
        raise ValueError('Invalid seed or feature budget')
    folder=args.output_dir.expanduser().resolve();folder.parent.mkdir(parents=True,exist_ok=True)
    # A persistent sibling lock does not add a seventh public package file.
    with (folder.parent/('.'+folder.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run_locked(args,folder)


def _run_locked(args,folder):
    if not args.resume and folder.exists():raise FileExistsError('Choose a new output directory')
    if args.resume and (folder/'verification.json').exists():
        results.verify_package(folder)
        raise FileExistsError('This package is complete; use results --verify to check it')
    internal=folder/'.internal';internal.mkdir(parents=True,exist_ok=args.resume)
    os.environ['NUMBA_CACHE_DIR']=str(internal/'numba-cache')
    for k,v in controller.worker_environment(args.threads).items():
        if k in ('NUMBA_NUM_THREADS','NUMBA_NRT_STATS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]=v
    pipeline=make_classifier(8,'ridge',0)
    scaler,ridge=(model.get_params() for _,model in pipeline.steps)
    configuration=dict(budgets=args.budgets,actual_features=[b//84*84 for b in args.budgets],seeds=args.seeds,
        repetitions=args.transform_repetitions,batch_size=args.batch_size,dilation_cap=args.max_dilations_per_kernel,
        threads=args.threads,bootstrap_samples=BOOTSTRAP_RESAMPLES,bootstrap_seed=0,
        similarity_tolerance=args.similarity_tolerance,order_seed=controller.EXECUTION_ORDER_SEED,
        scaler=dict(name='StandardScaler',with_mean=scaler['with_mean'],with_std=scaler['with_std']),
        classifier=dict(name='RidgeClassifierCV',alphas='n_train * logspace(-3,3,10)',
            cv=ridge['cv'],fit_intercept=ridge['fit_intercept'],class_weight=ridge['class_weight']))
    metadata_path=folder/'metadata.json';guard_path=internal/'guard.json'
    previous_hashes,previous_execute=controller.source_hashes,controller.execute_worker
    def hashes():
        values=previous_hashes()
        for n in ('benchmarks/run.py','benchmarks/results.py','benchmarks/README.md'):
            values[n]=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
        return values
    frozen=hashes()
    inventory_path=internal/'inventory.json';raw_path=internal/'raw.csv'
    controller_args=controller.build_parser().parse_args(['--data',str(Path(args.data).expanduser().resolve()),
        '--output',str(raw_path),'--threads',str(args.threads),'--batch-size',str(args.batch_size),
        '--max-dilations-per-kernel',str(args.max_dilations_per_kernel),'--transform-repetitions',str(args.transform_repetitions),
        '--seeds',*map(str,args.seeds),'--feature-counts',*map(str,args.budgets),
        '--inventory',str(inventory_path),*(['--datasets',*args.datasets] if args.datasets else []),*(['--resume'] if args.resume else [])])
    if args.resume:
        guard=json.loads(guard_path.read_text());meta=json.loads(metadata_path.read_text())
        if guard['status']!='failed':raise ValueError('Resume requires a recorded failure; audit killed controllers and orphan workers first')
        if guard['source_sha256']!=frozen or meta['configuration']!=configuration or guard['runtime']!=controller.runtime_signature(controller_args):
            raise ValueError('Resume source/configuration/runtime changed')
        guard.setdefault('attempts',[]).append(dict(utc=datetime.now(timezone.utc).isoformat(),previous_error=guard.get('error')))
    else:
        inv=inspect_archive(args.data,args.datasets);_json(inventory_path,inv)
        if not inv['datasets']:raise ValueError('No compatible datasets')
        guard=dict(status='running',source_sha256=frozen,runtime=controller.runtime_signature(controller_args),attempts=[])
        meta=dict(schema_version=1,status='running',started_utc=datetime.now(timezone.utc).isoformat(),
            hardware=_hardware(),software=_software(),archive=dict(name=args.dataset_label or 'Local dataset collection',download_url=args.dataset_source_url),
            split_policy='Supplied official TRAIN/TEST splits; seeds repeat fits, not resampled splits',
            included_datasets=[d['name'] for d in inv['datasets']],
            excluded_datasets=[dict(dataset=d['name'],reason=d['reason']) for d in inv['skipped_datasets']],
            discovered_datasets=inv['discovered_datasets'],configuration=configuration,methodology_version=1)
    meta['status']='running';guard.update(status='running',pid=os.getpid());guard.pop('error',None)
    _json(metadata_path,meta);_json(guard_path,guard)
    observed_threads=set();observed_pools=set()
    def execute(job,directory,threads):
        if hashes()!=frozen:raise ValueError('Measured sources changed during run')
        r=previous_execute(job,directory,threads);results.project_worker(r,threads)
        observed_threads.add(int(r['numba_threads']))
        for pool in json.loads(r['threadpools_json']):observed_pools.add((pool.get('internal_api','unknown'),pool.get('version'),pool['num_threads']))
        return r
    start=perf_counter()
    try:
        controller.source_hashes=hashes;controller.execute_worker=execute
        controller.run_pipeline(controller_args)
        if hashes()!=frozen:raise ValueError('Measured sources changed during run')
        with raw_path.open() as f:full=list(csv.DictReader(f))
        sidecar=json.loads(raw_path.with_suffix('.csv.json').read_text());inv=json.loads(inventory_path.read_text())
        if sidecar['status']!='complete' or {d['name']:d['fingerprint'] for d in sidecar['datasets']}!={d['name']:d['fingerprint'] for d in inv['datasets']}:
            raise ValueError('Completed inventory coverage differs')
        canonical=[]
        for r in full:
            canonical.append(results.project_worker(r,args.threads));observed_threads.add(int(r['numba_threads']))
            for pool in json.loads(r['threadpools_json']):observed_pools.add((pool.get('internal_api','unknown'),pool.get('version'),pool['num_threads']))
        results.write_measurements(folder/'measurements.csv',canonical)
        meta.update(status='measured',measurement_seconds=perf_counter()-start,
            runtime_validation=dict(checked_cases=len(full),effective_numba_threads=sorted(observed_threads),
                exposed_libraries=[dict(backend=x[0],version=x[1],threads=x[2]) for x in sorted(observed_pools,key=str)],
                requested_threads=args.threads,all_cache_hits_and_zero_compile_misses=True,
                blas_thread_note='Accelerate actual thread counts may be opaque; environment limit requested before worker startup'))
        _json(metadata_path,meta)
        summary=results.finalize(folder)
    except BaseException as error:
        guard.update(status='failed',error=repr(error));_json(guard_path,guard)
        meta['status']='failed';_json(metadata_path,meta)
        raise
    finally:
        controller.source_hashes,controller.execute_worker=previous_hashes,previous_execute
    try:
        guard['status']='complete';_json(guard_path,guard)
        shutil.rmtree(internal)
    except OSError as error:
        print('Verified package retained; private execution scratch could not be removed:',error,file=sys.stderr)
    print('\nVerified package:',folder)
    print('Features | Original accuracy | Joint accuracy | TRAIN ratio [95% CI] | TEST ratio [95% CI]')
    for s in summary:print(s['features'],s['original_accuracy_percent'],s['joint_accuracy_percent'],
        (s['train_ratio'],s['train_ratio_ci95_low'],s['train_ratio_ci95_high']),
        (s['test_ratio'],s['test_ratio_ci95_low'],s['test_ratio_ci95_high']),sep=' | ')
    return folder


def main():run(build_parser().parse_args())
if __name__=='__main__':main()
