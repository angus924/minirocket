"""CPU energy of the TEST transform: full model vs pruned models (README_GROUP_PRUNING.md, step 8).

For each dataset: fit original MiniRocket (seed 0), transform TRAIN once, prune groups with the ridge scorer
on TRAIN only (same selection as experiments.group_pruning), then for the full model and each pruned model:
  * loop the joint-PPV TEST transform until the measurement lasts at least --min-seconds,
  * measure CPU energy with CodeCarbon (measure_power_secs=1),
  * measure an IDLE period of the same wall time (sleep) the same way and subtract it,
  * record process CPU time as a second, independent number.
Reported: net CPU energy per series and CPU seconds per series.

Only the measured CPU energy (CodeCarbon's CPU reading; on Windows via the Energy Meter Interface, 'windows_emi')
is used. CodeCarbon's RAM figure is a constant-power estimate and its GPU figure is not related to this workload,
so both are recorded in the CSV but excluded from the headline numbers.

  NUMBA_NUM_THREADS=1 python -B -m experiments.energy --data ./data/Univariate_ts/ \
      --datasets GunPoint Computers ECG5000 Wafer Crop --output-dir results/energy
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import argparse
import csv
import json
import platform
from argparse import Namespace
from importlib.metadata import version
from pathlib import Path
from time import perf_counter, process_time, sleep

import numpy as np

from benchmarks.common import _dataset_sources, load_benchmark_dataset, make_classifier
import minirocket_pruned as pruned
from minirocket_baseline import fit_univariate

FIELDS = ['dataset', 'keep_fraction', 'features_kept', 'groups_kept', 'n_test', 'repetition', 'loops',
          'duration_seconds', 'cpu_energy_kwh', 'idle_cpu_energy_kwh', 'net_cpu_energy_kwh',
          'cpu_seconds', 'net_energy_joules_per_series', 'cpu_seconds_per_series',
          'ram_energy_kwh_not_used', 'gpu_energy_kwh_not_used']


def measure(function):
    """Run function() under a CodeCarbon tracker; return (seconds, cpu_seconds, cpu_kwh, ram_kwh, gpu_kwh)."""
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level='error')
    cpu_start, start = process_time(), perf_counter()
    tracker.start()
    function()
    tracker.stop()
    seconds, cpu_seconds = perf_counter() - start, process_time() - cpu_start
    d = tracker.final_emissions_data
    return seconds, cpu_seconds, float(d.cpu_energy), float(d.ram_energy), float(d.gpu_energy)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True)
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--keep-fractions', nargs='+', type=float, default=[0.5, 0.2])
    p.add_argument('--min-seconds', type=float, default=30.0)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--drop-fraction', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f'{args.output_dir} exists; choose a new folder')
    args.output_dir.mkdir(parents=True)

    rows = []
    loader_args = Namespace(data=args.data, datasets=args.datasets)
    for name, train, test in _dataset_sources(loader_args):
        dataset = load_benchmark_dataset(loader_args, name, train, test, [])
        X_train = np.ascontiguousarray(dataset.X_train[:, 0]); X_test = np.ascontiguousarray(dataset.X_test[:, 0])
        parameters = fit_univariate(X_train, num_features=10000, random_state=args.seed)
        ones = pruned.keep_all(parameters)
        train_full = pruned.transform(X_train, parameters, ones, 'joint')
        masks = [(1.0, ones)]
        for fraction in args.keep_fractions:
            keep, _ = pruned.prune_groups(train_full, dataset.y_train, parameters, fraction, scorer='ridge',
                                          drop_fraction=args.drop_fraction,
                                          classifier_factory=lambda n: make_classifier(n, 'ridge', args.seed))
            masks.append((fraction, keep))
        for fraction, keep in masks:
            pruned.transform(X_test[:2], parameters, keep, 'joint')                 # warm-up outside the meter
            once_start = perf_counter(); pruned.transform(X_test, parameters, keep, 'joint')
            once = perf_counter() - once_start
            loops = max(1, int(np.ceil(args.min_seconds / max(once, 1e-6))))
            for repetition in range(args.repetitions):
                def work():
                    for _ in range(loops):
                        pruned.transform(X_test, parameters, keep, 'joint')
                seconds, cpu_seconds, cpu_kwh, ram_kwh, gpu_kwh = measure(work)
                _, _, idle_kwh, _, _ = measure(lambda: sleep(seconds))             # same wall time, idle
                net_kwh = cpu_kwh - idle_kwh
                series = loops * len(X_test)
                rows.append(dict(dataset=name, keep_fraction=fraction, features_kept=pruned.pruned_width(parameters, keep),
                                 groups_kept=int(keep.sum()), n_test=len(X_test), repetition=repetition, loops=loops,
                                 duration_seconds=seconds, cpu_energy_kwh=cpu_kwh, idle_cpu_energy_kwh=idle_kwh,
                                 net_cpu_energy_kwh=net_kwh, cpu_seconds=cpu_seconds,
                                 net_energy_joules_per_series=net_kwh * 3.6e6 / series,
                                 cpu_seconds_per_series=cpu_seconds / series,
                                 ram_energy_kwh_not_used=ram_kwh, gpu_energy_kwh_not_used=gpu_kwh))
                print(f'{name} keep={fraction} rep={repetition} {seconds:.0f}s net {1000 * net_kwh * 3600:.1f} J '
                      f'({net_kwh * 3.6e6 / series * 1e6:.1f} uJ/series)', flush=True)
        with (args.output_dir / 'energy.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)

    # summary: per dataset mean over repetitions, ratio to the full model
    lines = ['# Energy of the TEST transform (CPU, idle-subtracted)', '',
             f'CodeCarbon {version("codecarbon")}, measure_power_secs=1, CPU reading only; machine: {platform.processor()}, '
             f'{platform.platform()}. Each measurement loops the joint-PPV transform for >= {args.min_seconds:.0f} s; '
             f'idle of equal duration subtracted; {args.repetitions} repetitions. NUMBA_NUM_THREADS=1.', '',
             '| dataset | keep | features | uJ/series (mean +- std) | energy ratio vs full | CPU us/series | CPU-time ratio vs full |',
             '|---|---:|---:|---|---:|---:|---:|']
    for name in dict.fromkeys(r['dataset'] for r in rows):
        mine = [r for r in rows if r['dataset'] == name]
        full = [r for r in mine if r['keep_fraction'] == 1.0]
        e_full = np.mean([r['net_energy_joules_per_series'] for r in full]); c_full = np.mean([r['cpu_seconds_per_series'] for r in full])
        for fraction in dict.fromkeys(r['keep_fraction'] for r in mine):
            sel = [r for r in mine if r['keep_fraction'] == fraction]
            e = np.array([r['net_energy_joules_per_series'] for r in sel]); c = np.array([r['cpu_seconds_per_series'] for r in sel])
            lines.append(f"| {name} | {fraction:.2f} | {sel[0]['features_kept']} | {1e6 * e.mean():.1f} +- {1e6 * e.std(ddof=1) if len(e) > 1 else 0:.1f} | "
                         f"{e.mean() / e_full:.3f} | {1e6 * c.mean():.1f} | {c.mean() / c_full:.3f} |")
    (args.output_dir / 'energy.md').write_text('\n'.join(lines) + '\n')
    (args.output_dir / 'metadata.json').write_text(json.dumps(dict(
        codecarbon=version('codecarbon'), platform=platform.platform(), processor=platform.processor(),
        min_seconds=args.min_seconds, repetitions=args.repetitions, drop_fraction=args.drop_fraction,
        datasets=args.datasets, keep_fractions=args.keep_fractions, method='CodeCarbon CPU energy (windows_emi) minus idle of equal duration'),
        indent=2) + '\n')
    print('Wrote', args.output_dir / 'energy.csv', 'and energy.md')


if __name__ == '__main__':
    main()
