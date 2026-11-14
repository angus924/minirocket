"""Keep-all pruned joint transform vs existing joint transform: identical output, similar speed."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import argparse
from argparse import Namespace
from time import perf_counter

import numpy as np

from benchmarks.common import _dataset_sources, load_benchmark_dataset
import minirocket_joint_ppv_simd as joint
import minirocket_pruned as pruned
from minirocket_baseline import fit_univariate


def best(function, repetitions=3):
    times = []
    for _ in range(repetitions):
        start = perf_counter(); value = function(); times.append(perf_counter() - start)
    return value, min(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='./data/Univariate_ts/')
    p.add_argument('--datasets', nargs='+', default=['GunPoint', 'Wafer'])
    args = p.parse_args()
    args.data_ = args.data
    for name, train, test in _dataset_sources(Namespace(data=args.data, datasets=args.datasets)):
        ds = load_benchmark_dataset(Namespace(data=args.data, datasets=args.datasets), name, train, test, [])
        X_train = np.ascontiguousarray(ds.X_train[:, 0]); X_test = np.ascontiguousarray(ds.X_test[:, 0])
        parameters = fit_univariate(X_train, num_features=10000, random_state=0)
        keep = pruned.keep_all(parameters)
        joint.transform(X_test[:2], parameters); pruned.transform(X_test[:2], parameters, keep, 'joint')  # warm-up
        a, ta = best(lambda: joint.transform(X_test, parameters))
        b, tb = best(lambda: pruned.transform(X_test, parameters, keep, 'joint'))
        print(f'{name}: n_test={len(X_test)} existing joint {ta:.4f}s, pruned keep-all {tb:.4f}s, '
              f'ratio {tb / ta:.3f}, identical={np.array_equal(a, b)}', flush=True)


if __name__ == '__main__':
    main()
