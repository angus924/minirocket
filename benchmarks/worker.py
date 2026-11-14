"""One original or exact joint-PPV case, with training-only fitted state."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .common import ModelSpec, _load_npz, make_classifier, parameter_nbytes, transform_batches, verify_feature_values
from .memory import snapshot


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return digest.hexdigest()


def transform_repetitions(transform, repetitions):
    if isinstance(repetitions, bool) or not isinstance(repetitions,int) or repetitions < 1:
        raise ValueError('Repetitions must be a positive integer')
    features = None
    times, hashes = [], []
    for _ in range(repetitions):
        del features
        start = perf_counter(); features = transform(); times.append(perf_counter()-start)
        snapshot()
        digest = verify_feature_values(features)
        if hashes and digest != hashes[0]:
            raise ValueError('Feature values changed between repetitions')
        hashes.append(digest)
    return features,times,hashes


def run(job):
    if os.environ.get('NUMBA_NRT_STATS','0') != '0':
        raise ValueError('Allocation instrumentation must be disabled')
    from numba import get_num_threads, set_num_threads
    from threadpoolctl import threadpool_info, threadpool_limits
    from minirocket_baseline import fit_univariate, transform_univariate
    import minirocket
    variant,config,seed = job['variant'],job['configuration'],job['seed']
    if variant['implementation'] == 'joint_ppv':
        from minirocket_joint_ppv_simd import transform as transform_function
        from minirocket_joint_ppv_simd import _transform as compiled_transform
    elif variant['implementation'] == 'original':
        transform_function = transform_univariate
        compiled_transform = minirocket.transform
    else:
        raise ValueError('Unknown implementation')
    set_num_threads(config['threads'])
    with threadpool_limits(limits=config['threads']):
        start = perf_counter(); dataset = _load_npz(Path(job['input']))
        if dataset.X_train.shape[1] != 1 or dataset.fingerprint != job['fingerprint']:
            raise ValueError('Prepared input split/channels differ from requested source')
        X_train = np.ascontiguousarray(dataset.X_train[:,0])
        X_test = np.ascontiguousarray(dataset.X_test[:,0])
        load_seconds = perf_counter()-start
        options = dict(num_features=variant['num_features'],
                       max_dilations_per_kernel=config['max_dilations_per_kernel'],random_state=seed)
        calls = []
        def measured_transform(X,parameters):
            start = perf_counter(); result = transform_function(X,parameters)
            calls.append(perf_counter()-start)
            return result
        model = ModelSpec(variant['name'],None,measured_transform,input_dim=2)
        def transform(X,parameters):
            return transform_batches(model,X,parameters,config['batch_size'])
        start = perf_counter()
        # Compilation/warmup is discarded and can inspect TRAIN only.
        warm_parameters = fit_univariate(X_train[:min(8,len(X_train))],**options)
        warm_features = transform(X_train[:1],warm_parameters)
        classifier = make_classifier(len(X_train),config['classifier'],seed)
        del warm_parameters,warm_features; calls.clear(); gc.collect()
        warmup_seconds = perf_counter()-start
        cache = dict(transform_cache_hits=sum(compiled_transform.stats.cache_hits.values()),
                     transform_cache_misses=sum(compiled_transform.stats.cache_misses.values()),
                     fit_cache_misses=sum(minirocket._fit_biases.stats.cache_misses.values()))
        if not cache['transform_cache_hits'] or cache['transform_cache_misses'] or cache['fit_cache_misses']:
            raise ValueError('Measured worker encountered cold JIT cache; prepopulate first')
        before = snapshot()
        start = perf_counter(); parameters = fit_univariate(X_train,**options)
        fit_seconds = perf_counter()-start; after_fit = snapshot()
        train_features,train_times,train_hashes = transform_repetitions(
            lambda:transform(X_train,parameters),config['transform_repetitions'])
        start = perf_counter(); classifier.fit(train_features,dataset.y_train)
        classifier_fit_seconds = perf_counter()-start; after_train = snapshot()
        test_features,test_times,test_hashes = transform_repetitions(
            lambda:transform(X_test,parameters),config['transform_repetitions'])
        start = perf_counter(); predictions = classifier.predict(test_features)
        predict_seconds = perf_counter()-start; peaks = snapshot()
        width = variant['num_features']//84*84
        for X,features in ((X_train,train_features),(X_test,test_features)):
            if features.shape != (len(X),width) or features.dtype != np.float32 or not np.isfinite(features).all():
                raise ValueError('Invalid feature output')
        means = dict(train=float(np.mean(train_times)),test=float(np.mean(test_times)))
        offset=0; inner={}
        for split,X in (('train',X_train),('test',X_test)):
            batches=(len(X)+config['batch_size']-1)//config['batch_size'];samples=[]
            for _ in range(config['transform_repetitions']):
                samples.append(sum(calls[offset:offset+batches]));offset+=batches
            inner[f'{split}_model_call_repetition_seconds']=json.dumps(samples)
            inner[f'{split}_model_call_seconds']=float(np.mean(samples))
            inner[f'{split}_delivery_overhead_seconds']=means[split]-float(np.mean(samples))
        if offset != len(calls): raise AssertionError('Unexpected model call count')
        scaler,estimator=classifier.steps[0][1],classifier.steps[-1][1]
        raw_bytes=parameter_nbytes(parameters)
        scaler_bytes=parameter_nbytes((scaler.mean_,scaler.var_,scaler.scale_))
        classifier_bytes=parameter_nbytes((estimator.coef_,estimator.intercept_,estimator.classes_))
        return dict(dataset=job['dataset'],model=variant['name'],seed=seed,
            implementation=variant['implementation'],split_sha256=dataset.fingerprint,
            classifier=config['classifier'],n_train=len(X_train),n_test=len(X_test),
            channels=1,length=X_train.shape[1],requested_features=variant['num_features'],features=width,
            max_dilations_per_kernel=config['max_dilations_per_kernel'],numba_threads=get_num_threads(),
            threadpools_json=json.dumps(threadpool_info()),transform_repetitions=config['transform_repetitions'],
            allocation_policy='common_legacy_batch_copy',accuracy=float(np.mean(predictions==dataset.y_test)),
            load_seconds=load_seconds,warmup_seconds=warmup_seconds,fit_seconds=fit_seconds,
            train_transform_seconds=means['train'],test_transform_seconds=means['test'],
            classifier_fit_seconds=classifier_fit_seconds,predict_seconds=predict_seconds,
            train_transform_repetition_seconds=json.dumps(train_times),test_transform_repetition_seconds=json.dumps(test_times),
            train_feature_repetition_sha256=json.dumps(train_hashes),test_feature_repetition_sha256=json.dumps(test_hashes),
            training_seconds=fit_seconds+means['train']+classifier_fit_seconds,
            test_seconds=means['test']+predict_seconds,
            pipeline_seconds=load_seconds+fit_seconds+means['train']+classifier_fit_seconds+means['test']+predict_seconds,
            train_feature_bytes=train_features.nbytes,test_feature_bytes=test_features.nbytes,
            parameter_array_bytes=raw_bytes,extractor_parameter_array_bytes=raw_bytes,
            scaler_parameter_array_bytes=scaler_bytes,classifier_parameter_array_bytes=classifier_bytes,
            total_parameter_array_bytes=raw_bytes+scaler_bytes+classifier_bytes,
            pooling_prepare_seconds=0.,pooling_metadata_bytes=0,
            **peaks,pre_pipeline_peak_memory_bytes=before['peak_memory_bytes'],
            after_fit_peak_memory_bytes=after_fit['peak_memory_bytes'],after_train_peak_memory_bytes=after_train['peak_memory_bytes'],
            train_feature_sha256=train_hashes[-1],test_feature_sha256=test_hashes[-1],
            fit_input_sha256=array_hash(X_train,dataset.y_train),extractor_parameters_sha256=array_hash(*parameters),
            scaler_parameters_sha256=array_hash(scaler.mean_,scaler.var_,scaler.scale_,scaler.n_samples_seen_),
            classifier_parameters_sha256=array_hash(estimator.coef_,estimator.intercept_,estimator.classes_),
            classifier_alpha=float(estimator.alpha_),prediction_sha256=array_hash(predictions),
            case_order_position=job['order_position'],worker_pid=os.getpid(),**inner,**cache)


def prewarm():
    """Populate fixed dtype signatures in a disposable process, using synthetic data."""
    from numba import set_num_threads
    from minirocket_baseline import fit_univariate,transform_univariate
    from minirocket_joint_ppv_simd import transform
    set_num_threads(int(os.environ.get('NUMBA_NUM_THREADS','1')))
    X=np.random.default_rng(0).normal(size=(8,33)).astype(np.float32)
    parameters=fit_univariate(X,num_features=2500,random_state=0)
    a,b=transform_univariate(X,parameters),transform(X,parameters)
    if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
        raise AssertionError('Prewarm exactness failed')
    return dict(policy='Synthetic-data disposable process before measurement; worker cache hits required',
                exact=True,cache_directory=os.environ.get('NUMBA_CACHE_DIR'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--job',type=Path)
    mode.add_argument('--prewarm',action='store_true')
    args=parser.parse_args()
    print(json.dumps(prewarm() if args.prewarm else run(json.loads(args.job.read_text()))),flush=True)


if __name__=='__main__':main()
