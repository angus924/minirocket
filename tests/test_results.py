"""User-visible measurement, paired-analysis and result-package contracts."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path


def row(dataset, implementation, seed, seconds, accuracy=.75):
    return dict(dataset=dataset, implementation=implementation, seed=seed,
        requested_features=84, features=84, n_train=8, n_test=4, length=33,
        accuracy=accuracy, fit_seconds=.1, classifier_fit_seconds=.2, predict_seconds=.01,
        load_seconds=.02, train_transform_seconds=seconds, test_transform_seconds=seconds,
        train_transform_repetition_seconds=json.dumps([seconds]*3),
        test_transform_repetition_seconds=json.dumps([seconds]*3),
        train_feature_repetition_sha256=json.dumps(['a'*64]*3),
        test_feature_repetition_sha256=json.dumps(['b'*64]*3),
        split_sha256='c'*64, fit_input_sha256='d'*64,
        extractor_parameters_sha256='e'*64, scaler_parameters_sha256='f'*64,
        classifier_parameters_sha256='1'*64, prediction_sha256='2'*64,
        classifier_alpha=1., extractor_parameter_array_bytes=400,
        scaler_parameter_array_bytes=2016, classifier_parameter_array_bytes=344,
        train_feature_bytes=2688, test_feature_bytes=1344,
        peak_memory_bytes=1000, peak_memory_metric='peak_rss', peak_rss_bytes=1000,
        case_order_position=0 if implementation=='original' else 1)


class ResultPackageTests(unittest.TestCase):
    def module(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec("benchmarks.results"))
        from benchmarks import results
        return results

    def config(self):
        return dict(datasets=['A','B'],seeds=[0,1],budgets=[84],repetitions=3,
                    bootstrap_samples=10000,bootstrap_seed=0,similarity_tolerance=.02)

    def rows(self):
        return [row(d,m,s,t) for d,o,j in [('A',1.,.5),('B',10.,9.)]
                for s in [0,1] for m,t in [('original',o),('joint_ppv',j)]]

    def test_mean_of_ratios_differs_from_ratio_of_means(self):
        r=self.module()._ratio_summary([1.,9.],[2.,3.],samples=100)
        self.assertEqual(r['ratio_of_arithmetic_means'],2.)
        self.assertEqual(r['mean_per_dataset_ratio'],1.75)
        self.assertLessEqual(r['ratio_of_means_ci95_low'],2.)
        self.assertGreaterEqual(r['ratio_of_means_ci95_high'],2.)
        r=self.module()._ratio_summary([1.],[2.],samples=10)
        self.assertIsNone(r['ratio_of_means_ci95_low'])
        self.assertIsNone(r['ratio_of_means_ci95_high'])

        with self.assertRaises(ValueError):
            self.module()._ratio_summary([0.], [1.])

    def test_paired_aggregation_and_intervals_use_datasets(self):
        p=self.module();per,summary,checks=p.analyze(self.rows(),self.config())
        self.assertEqual(len(per),2);self.assertEqual(len(summary),1)
        s=summary[0]
        self.assertAlmostEqual(s['original_test_transform_seconds'],5.5)
        self.assertAlmostEqual(s['joint_test_transform_seconds'],4.75)
        self.assertAlmostEqual(s['test_ratio'],9.5/11)
        self.assertAlmostEqual(s['median_dataset_test_ratio'],.7)
        self.assertAlmostEqual(s['test_ratio_ci95_low'],.5)
        self.assertAlmostEqual(s['test_ratio_ci95_high'],.9)
        self.assertEqual(s['original_accuracy_percent'],75.)
        self.assertEqual(s['accuracy_delta_pp'],0.)
        self.assertEqual((s['test_faster_datasets'],s['test_similar_datasets'],s['test_slower_datasets']),(2,0,0))
        self.assertEqual(checks['matched_cases'],4)
        self.assertTrue(checks['features_predictions_and_state_equal'])
        self.assertAlmostEqual(per[0]['original_pipeline_seconds'],1.+1.+.33)

    def test_feature_budgets_have_separate_original_controls(self):
        rows=self.rows();extra=copy.deepcopy(rows)
        for r in extra:
            r.update(requested_features=168,features=168,train_feature_bytes=5376,test_feature_bytes=2688)
            if r['implementation']=='original':
                for split in ('train','test'):
                    r[split+'_transform_seconds']*=2
                    r[split+'_transform_repetition_seconds']=json.dumps([r[split+'_transform_seconds']]*3)
        cfg=self.config();cfg['budgets']=[84,168]
        per,summary,_=self.module().analyze(rows+extra,cfg)
        self.assertEqual(len(per),4)
        self.assertAlmostEqual(summary[0]['test_ratio'],9.5/11)
        self.assertAlmostEqual(summary[1]['test_ratio'],9.5/22)

    def test_memory_and_accuracy_average_seeds_before_datasets(self):
        rows = self.rows()
        for r in rows:
            if r['dataset'] == 'A':
                peak = [100, 300][r['seed']]
                r['accuracy'] = [.5, .7][r['seed']]
                if r['implementation'] == 'joint_ppv':
                    peak //= 2
            else:
                r['accuracy'] = .8
                peak = 1000 if r['implementation'] == 'original' else 1200
            r['peak_memory_bytes'] = r['peak_rss_bytes'] = peak
        per, summary, _ = self.module().analyze(rows, self.config())
        self.assertEqual(per[0]['original_peak_memory_bytes'], 200)
        self.assertEqual(per[0]['joint_peak_memory_bytes'], 100)
        s = summary[0]
        self.assertEqual(s['original_peak_memory_bytes'], 600)
        self.assertEqual(s['joint_peak_memory_bytes'], 650)
        self.assertAlmostEqual(s['memory_ratio'], 650 / 600)
        self.assertEqual(s['joint_worst_peak_bytes'], 1200)
        self.assertEqual(s['joint_worst_peak_dataset'], 'B')
        self.assertAlmostEqual(s['original_accuracy_percent'], 70)
        self.assertAlmostEqual(s['joint_accuracy_percent'], 70)

    def test_invalid_or_mismatched_measurements_are_rejected(self):
        p=self.module();rows=self.rows();cfg=self.config()
        for broken in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):p.analyze(broken,cfg)
        for field,value in [('accuracy',.5),('prediction_sha256','0'*64),('test_feature_repetition_sha256',json.dumps(['0'*64]*3)),('test_transform_seconds',-1),('test_feature_bytes',7)]:
            broken=copy.deepcopy(rows);broken[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):p.analyze(broken,cfg)
        broken=copy.deepcopy(rows);broken[1]['test_transform_repetition_seconds']=json.dumps([.4,.5,.7])
        with self.assertRaises(ValueError):p.analyze(broken,cfg)

    def test_one_dataset_has_no_confidence_interval(self):
        cfg=self.config();cfg['datasets']=['A']
        _,summary,_=self.module().analyze([r for r in self.rows() if r['dataset']=='A'],cfg)
        self.assertIsNone(summary[0]['test_ratio_ci95_low'])
        self.assertIsNone(summary[0]['test_ratio_ci95_high'])

    def test_report_explains_overall_ratios_and_intervals_with_readable_results(self):
        p = self.module()
        _, summary, checks = p.analyze(self.rows(), self.config())
        meta = dict(configuration=self.config())
        report = p._report(meta, summary, checks)
        self.assertIn('Ratio = joint PPV mean time / original mean time', report)
        self.assertIn('0.70 means 30% less time', report)
        self.assertIn('resample 2 dataset pairs with replacement', report)
        self.assertIn('10,000', report)
        self.assertIn('middle 95%', report)
        self.assertIn('overall time ratio', report)
        self.assertIn('0.864 [0.500, 0.900]', report)
        self.assertIn('13.6%', report)
        self.assertIn('[summary.csv](summary.csv)', report)
        self.assertIn('[per_dataset.csv](per_dataset.csv)', report)
        self.assertNotIn('Order sensitivity at', report)
        self.assertNotIn('Median-repetition TEST ratio', report)
        self.assertLess(max(line.count('|') for line in report.splitlines()), 9)

    def test_report_keeps_slowdowns_visible_and_handles_unavailable_intervals(self):
        p = self.module()
        cfg = self.config(); cfg['datasets'] = ['A']
        rows = [r for r in self.rows() if r['dataset'] == 'A']
        for r in rows:
            if r['implementation'] == 'joint_ppv':
                for split in ('train', 'test'):
                    r[split+'_transform_seconds'] = 2
                    r[split+'_transform_repetition_seconds'] = json.dumps([2]*3)
        _, summary, checks = p.analyze(rows, cfg)
        report = p._report(dict(configuration=cfg), summary, checks)
        self.assertIn('2.000 [unavailable]', report)
        self.assertIn('-100.0%', report)
        self.assertIn('slower on 1 of 1 datasets', report)
        self.assertIn('One dataset cannot provide a dataset-bootstrap interval', report)

    def test_package_has_six_files_and_detects_corrupted_summary(self):
        p=self.module()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);meta=dict(status='measured',configuration=self.config(),archive=dict(name='Fixture',download_url=None),included_datasets=['A','B'],excluded_datasets=[])
            p.write_measurements(folder/'measurements.csv',self.rows())
            (folder/'metadata.json').write_text(json.dumps(meta))
            p.finalize(folder)
            self.assertEqual({f.name for f in folder.iterdir()},set(p.CORE_FILES))
            p.verify_package(folder)
            with (folder/'summary.csv').open() as f:rows=list(csv.DictReader(f))
            rows[0]['test_ratio']='.01'
            p.write_csv(folder/'summary.csv',rows)
            with self.assertRaises(ValueError):p.verify_package(folder)

    def test_runtime_projection_rejects_compilation_and_thread_mismatches(self):
        p=self.module();r=dict(self.rows()[0],transform_cache_hits=1,transform_cache_misses=0,fit_cache_misses=0,numba_threads=1,threadpools_json='[]')
        for key,value in [('transform_cache_misses',1),('transform_cache_hits',0),('numba_threads',2),('threadpools_json','[{"num_threads":2}]')]:
            with self.subTest(key=key),self.assertRaises(ValueError):p.project_worker(dict(r,**{key:value}),1)

    def test_split_consistency_across_seeds_and_budgets_is_required(self):
        rows=copy.deepcopy(self.rows())
        for r in rows:
            if r['dataset']=='A' and r['seed']==1:r['split_sha256']='0'*64
        with self.assertRaises(ValueError):self.module().analyze(rows,self.config())

    def test_incomplete_integrity_manifest_is_rejected(self):
        p=self.module()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);meta=dict(status='measured',configuration=self.config(),archive={},included_datasets=['A','B'],excluded_datasets=[])
            p.write_measurements(folder/'measurements.csv',self.rows());(folder/'metadata.json').write_text(json.dumps(meta));p.finalize(folder)
            checks=json.loads((folder/'verification.json').read_text());checks['artifacts_sha256']={}
            (folder/'verification.json').write_text(json.dumps(checks))
            with self.assertRaises(ValueError):p.verify_package(folder)

    def test_partial_report_can_be_recovered_without_new_measurements(self):
        p=self.module()
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);meta=dict(status='measured',configuration=self.config(),archive={},included_datasets=['A','B'],excluded_datasets=[])
            p.write_measurements(folder/'measurements.csv',self.rows());(folder/'metadata.json').write_text(json.dumps(meta))
            (folder/'report.md').write_text('Interrupted report')
            p.finalize(folder);p.verify_package(folder)
            self.assertEqual({f.name for f in folder.iterdir()},set(p.CORE_FILES))

    def test_normal_workflow_produces_compact_package_and_checks_test_isolation(self):
        import importlib.util
        import subprocess
        import sys
        self.assertIsNotNone(importlib.util.find_spec('benchmarks.run'))
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'Tiny.npz';rng=np.random.default_rng(401)
            train=rng.normal(size=(8,33)).astype(np.float32);test=rng.normal(size=(4,33)).astype(np.float32)
            def save(x,y):np.savez(data,X_train=train,y_train=['a','b']*4,X_test=x,y_test=y)
            save(test,['a','b']*2)
            def run(output):
                # The CLI sets process-wide Numba/thread settings. Keep those
                # isolated from the kernel tests in this interpreter.
                result = subprocess.run(
                    [sys.executable, '-B', '-m', 'benchmarks.run',
                     '--data', str(data), '--seeds', '0', '--feature-counts', '2500',
                     '--batch-size', '3',
                     '--dataset-label', 'Fixture collection',
                     '--dataset-source-url', 'https://example.org/fixture.zip',
                     '--output-dir', str(output)],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run(root/'first')
            self.assertEqual({p.name for p in (root/'first').iterdir()},set(self.module().CORE_FILES))
            meta=json.loads((root/'first/metadata.json').read_text());encoded=json.dumps(meta)
            self.assertNotIn(str(root),encoded)
            self.assertNotIn('source_sha256',encoded);self.assertNotIn('git_revision',encoded)
            self.assertEqual(meta['archive']['name'], 'Fixture collection')
            self.assertEqual(meta['archive']['download_url'], 'https://example.org/fixture.zip')
            self.assertEqual(meta['configuration']['bootstrap_samples'], 10000)
            self.assertEqual(meta['configuration']['bootstrap_seed'], 0)
            self.assertEqual(meta['configuration']['order_seed'], 20261026)
            self.assertTrue(meta['runtime_validation']['all_cache_hits_and_zero_compile_misses'])
            self.assertEqual(meta['runtime_validation']['effective_numba_threads'], [1])
            with (root/'first/measurements.csv').open() as f:first={r['implementation']:r for r in csv.DictReader(f)}
            for field in self.module().EXACT_FIELDS:
                self.assertEqual(first['original'][field], first['joint_ppv'][field], field)
            for r in first.values():
                self.assertEqual(r['features'], '2436')
                for split in ('train', 'test'):
                    times = json.loads(r[split+'_transform_repetition_seconds'])
                    hashes = json.loads(r[split+'_feature_repetition_sha256'])
                    self.assertEqual(len(times), 3)
                    self.assertGreater(min(times), 0)
                    self.assertEqual(len(set(hashes)), 1)
                    self.assertEqual(int(r[split+'_feature_bytes']),
                                     int(r['n_'+split])*int(r['features'])*4)
            self.module().verify_package(root/'first')
            save(test*100+10,['b','a']*2)
            run(root/'second')
            with (root/'second/measurements.csv').open() as f:second={r['implementation']:r for r in csv.DictReader(f)}
            for method in first:
                for field in ('fit_input_sha256','extractor_parameters_sha256',
                              'train_feature_repetition_sha256','scaler_parameters_sha256',
                              'classifier_parameters_sha256','classifier_alpha'):
                    self.assertEqual(first[method][field],second[method][field])
                self.assertNotEqual(first[method]['test_feature_repetition_sha256'],second[method]['test_feature_repetition_sha256'])


if __name__=='__main__':unittest.main()
