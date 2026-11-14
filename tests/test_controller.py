"""Fair execution, strict provenance and training-only learning contracts."""
import fcntl
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class ControllerTests(unittest.TestCase):
    def controller(self):
        self.assertIsNotNone(importlib.util.find_spec('benchmarks.controller'),
                             'The focused original/joint archive runner is missing')
        from benchmarks import controller
        return controller

    def test_feature_counts_option_selects_matched_variants(self):
        from benchmarks.run import build_parser
        for parser, extra in ((build_parser(), ['--output-dir', 'unused-results']),
                              (self.controller().build_parser(), [])):
            argv = ['--data', 'unused', '--feature-counts', '2500', '5000', '10000'] + extra
            args = parser.parse_args(argv)
            variants = self.controller().build_variants(args)
            self.assertEqual([v['num_features'] for v in variants],
                             [2500, 2500, 5000, 5000, 10000, 10000])

    def test_public_runner_needs_only_data_and_output_and_accepts_dataset_metadata(self):
        from benchmarks.run import build_parser
        parser = build_parser()
        args = parser.parse_args(['--data', 'unused', '--output-dir', 'new-results'])
        self.assertEqual(args.budgets, [2500, 5000, 10000])
        self.assertEqual(args.seeds, [0, 1, 2])
        self.assertEqual(args.threads, 1)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.transform_repetitions, 3)
        self.assertIsNone(args.dataset_label)
        self.assertIsNone(args.dataset_source_url)
        args = parser.parse_args(['--data', 'unused', '--output-dir', 'new-results',
                                  '--dataset-label', 'Fixture collection',
                                  '--dataset-source-url', 'https://example.org/fixture.zip'])
        self.assertEqual(args.dataset_label, 'Fixture collection')
        self.assertEqual(args.dataset_source_url, 'https://example.org/fixture.zip')

    def test_public_runner_rejects_retired_and_internal_options(self):
        import io
        from contextlib import redirect_stderr
        from benchmarks.run import build_parser
        for option, value in (('--archive-name', 'Fixture'), ('--download-url', 'https://example.org'),
                              ('--order-seed', '42'), ('--bootstrap-samples', '100')):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    build_parser().parse_args(['--data', 'unused', '--output-dir', 'new-results', option, value])
                self.assertEqual(error.exception.code, 2)

    def test_source_guard_works_with_only_current_benchmark_files(self):
        runner = self.controller()
        files = (
            'code/minirocket.py', 'code/minirocket_baseline.py',
            'code/minirocket_joint_ppv_simd.py', 'code/_joint_ppv_simd_kernels.py',
            'benchmarks/common.py', 'benchmarks/memory.py',
            'benchmarks/controller.py', 'benchmarks/worker.py',
            'benchmarks/inventory.py',
            'tools/generate_joint_ppv.py',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in files:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((runner.ROOT / name).read_bytes())
            with patch.object(runner, 'ROOT', root):
                before = runner.source_hashes()
                target = root / 'benchmarks/worker.py'
                target.write_bytes(target.read_bytes() + b'\n# Changed worker\n')
                self.assertNotEqual(before[target.relative_to(root).as_posix()],
                                    runner.source_hashes()[target.relative_to(root).as_posix()])


    def test_pair_guard_rejects_any_divergent_learned_state_or_features(self):
        runner = self.controller()
        for budget in (2500,5000,10000):
            control = dict.fromkeys(runner.EXACT_KEYS, 'same')
            control.update(dataset='D',model=f'original_{budget}',seed=0,accuracy=.5,classifier_alpha=1.0)
            candidate = dict(control,model=f'joint_ppv_{budget}')
            runner.validate_pairs([control,candidate])
            for key in runner.EXACT_KEYS:
                changed = dict(candidate,**{key: .25 if key=='accuracy' else 'changed'})
                with self.assertRaises(ValueError,msg=key):
                    runner.validate_pairs([control,changed])

    def test_lock_prevents_concurrent_writes(self):
        runner = self.controller()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out.csv'
            with Path(str(output)+'.lock').open('a') as f:
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,'lock|running'):
                    runner.main(['--data','unused','--output',str(output)])
            self.assertFalse(output.exists())

    def test_resume_rejects_changed_settings_or_sources_before_workers_start(self):
        runner = self.controller()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'out.csv'
            argv = ['--data', 'unused', '--output', str(output), '--resume']
            args = runner.build_parser().parse_args(argv)
            for field in ('configuration_signature', 'source_sha256'):
                previous = dict(configuration_signature=runner._config(args),
                                source_sha256=runner.source_hashes())
                previous[field] = 'changed'
                output.write_text('preserved measurements')
                output.with_suffix('.csv.json').write_text(json.dumps(previous))
                with self.subTest(field=field), \
                     patch.object(runner, 'execute_worker') as launch, \
                     self.assertRaisesRegex(ValueError, 'Resume configuration/provenance'):
                    runner.main(argv)
                launch.assert_not_called()
                self.assertEqual(output.read_text(), 'preserved measurements')

    def test_order_keeps_budget_pairs_adjacent_and_ignores_test_contents(self):
        runner = self.controller()
        variants = runner.build_variants(runner.build_parser().parse_args(['--data','unused']))
        self.assertEqual(len(variants),6)
        for seed in range(3):
            order = runner.case_order('Dataset',seed,variants,20261026)
            self.assertEqual({v['name'] for v in order},{v['name'] for v in variants})
            for i in range(0,6,2):
                self.assertEqual(order[i]['num_features'],order[i+1]['num_features'])
                self.assertNotEqual(order[i]['implementation'],order[i+1]['implementation'])

    def test_frozen_inventory_rejects_changed_split_before_measuring(self):
        runner=self.controller()
        from benchmarks.inventory import inspect_archive
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'Tiny.npz';inventory=root/'inventory.json'
            np.savez(source,X_train=np.zeros((4,9),np.float32),y_train=['a','b']*2,
                     X_test=np.ones((2,9),np.float32),y_test=['a','b'])
            frozen=inspect_archive(source);frozen['datasets'][0]['fingerprint']='changed'
            inventory.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError,'inventory|fingerprint'):
                runner.main(['--data',str(source),'--inventory',str(inventory),
                             '--feature-counts','2500','--seeds','0','--output',str(root/'out.csv')])


if __name__ == '__main__':
    unittest.main()
