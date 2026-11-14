"""Offline fixture and utility checks, never comparative benchmark runs."""
import argparse
import io
import sys
import tempfile
import unittest
from dataclasses import dataclass
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import common


class LocalLoaderTests(unittest.TestCase):
    def load(self, args):
        return [common._load_npz(train) if test is None
                else common._load_pair(name, train, test)
                for name, train, test in common._dataset_sources(args)]

    def args(self, data, datasets=None):
        return argparse.Namespace(data=str(data), datasets=datasets)

    def test_npz_preserves_channels_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.npz'
            X = np.arange(108).reshape(4, 3, 9)
            np.savez(path, X_train=X, y_train=['a', 'b', 'a', 'b'],
                     X_test=X[:2], y_test=['b', 'a'])
            dataset, = self.load(self.args(path))
            self.assertEqual(dataset.X_train.shape, (4, 3, 9))
            self.assertEqual(dataset.X_train.dtype, np.float32)
            np.testing.assert_array_equal(dataset.y_test, ['b', 'a'])
            self.assertTrue(dataset.fingerprint)

    def test_tsv_label_first_and_original_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'Tiny'
            folder.mkdir()
            np.savetxt(folder / 'Tiny_TRAIN.tsv', np.array([[1] + list(range(9)), [2] + list(range(9, 18))]), delimiter='\t')
            np.savetxt(folder / 'Tiny_TEST.tsv', np.array([[2] + list(range(18, 27))]), delimiter='\t')
            dataset, = self.load(self.args(directory, ['Tiny']))
            self.assertEqual(dataset.X_train.shape, (2, 1, 9))
            np.testing.assert_array_equal(dataset.X_test[0, 0], list(range(18, 27)))
            np.testing.assert_array_equal(dataset.y_train, ['1', '2'])

    def test_loader_rejects_nonfinite_or_mismatched_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.npz'
            for test in [np.zeros((2, 10)), np.full((2, 9), np.nan)]:
                np.savez(path, X_train=np.zeros((2, 9)), y_train=[0, 1],
                         X_test=test, y_test=[0, 1])
                with self.assertRaises(ValueError):
                    self.load(self.args(path))

    def test_no_local_pair_does_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self.load(self.args(directory, ['Missing']))

    def test_ts_reader_preserves_univariate_values(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            header = '@problemName Tiny\n@timestamps false\n@univariate true\n@equalLength true\n@seriesLength 9\n@classLabel true a b\n@data\n'
            rows = '0,1,2,3,4,5,6,7,8:a\n1,2,3,4,5,6,7,8,9:b\n'
            for split in ['TRAIN', 'TEST']:
                (base / f'Tiny_{split}.ts').write_text(header + rows)
            dataset, = self.load(self.args(base, ['Tiny']))
            self.assertEqual(dataset.X_train.shape, (2, 1, 9))
            np.testing.assert_array_equal(dataset.X_train[0, 0], list(range(9)))

    def test_timestamped_ts_requires_explicit_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            header = '@problemName Tiny\n@timestamps true\n@univariate true\n@classLabel true a b\n@data\n'
            row = ','.join(f'({i*i},{i})' for i in range(9))
            for split in ['TRAIN', 'TEST']:
                (base / f'Tiny_{split}.ts').write_text(header + row + ':a\n' + row + ':b\n')
            with self.assertRaisesRegex(ValueError, 'timestamp'):
                self.load(self.args(base, ['Tiny']))

    def test_label_identity_and_nonfinite_values(self):
        np.testing.assert_array_equal(common._labels(np.array([9007199254740992, 9007199254740993], dtype=np.int64)),
                                      ['9007199254740992', '9007199254740993'])
        for value in ['1e999', '+Infinity', '-inf', 'NaN']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                common._labels([value])

    def test_duplicate_npz_names_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ['a', 'b']:
                folder = Path(directory) / name
                folder.mkdir()
                np.savez(folder / 'Tiny.npz', X_train=np.zeros((2,9)), y_train=[0,1],
                         X_test=np.zeros((2,9)), y_test=[0,1])
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                self.load(self.args(directory))


class ArchiveCompatibilityTests(unittest.TestCase):
    def test_archive_skips_ragged_ts_and_continues_to_valid_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            header = '@problemName Tiny\n@timestamps false\n@univariate true\n@equalLength false\n@classLabel true a b\n@data\n'
            rows = '0,1,2,3,4,5,6,7,8:a\n0,1,2,3,4,5,6,7,8,9:b\n'
            for split in ['TRAIN', 'TEST']:
                (folder / f'Ragged_{split}.ts').write_text(header + rows)
            good = folder / 'Valid.npz'
            np.savez(good, X_train=np.zeros((2,9)), y_train=[0,1],
                     X_test=np.zeros((2,9)), y_test=[0,1])
            args = argparse.Namespace(data=directory, datasets=None)
            skipped = []
            output = io.StringIO()
            with redirect_stdout(output):
                result = common.load_benchmark_dataset(args, 'Ragged', folder / 'Ragged_TRAIN.ts', folder / 'Ragged_TEST.ts', skipped)
                valid = common.load_benchmark_dataset(args, 'Valid', good, None, skipped)
            self.assertIsNone(result)
            self.assertEqual(valid.name, 'Valid')
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0]['name'], 'Ragged')
            self.assertIn('Variable-length', skipped[0]['reason'])
            self.assertIn('Skipping Ragged', output.getvalue())

    def test_explicit_selection_still_rejects_incompatible_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Missing.npz'
            np.savez(path, X_train=np.full((2,9), np.nan), y_train=[0,1],
                     X_test=np.zeros((2,9)), y_test=[0,1])
            for args in [argparse.Namespace(data=directory, datasets=['Missing']),
                         argparse.Namespace(data=str(path), datasets=None)]:
                with self.subTest(args=args), self.assertRaisesRegex(common.UnsupportedDatasetError, 'Missing.*finite'):
                    common.load_benchmark_dataset(args, 'Missing', path, None, [])

    def test_archive_skips_missing_short_and_mismatched_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Unsupported.npz'
            args = argparse.Namespace(data=directory, datasets=None)
            for train, test in [(np.full((2,9), np.nan), np.zeros((2,9))),
                                (np.zeros((2,8)), np.zeros((2,8))),
                                (np.zeros((2,9)), np.zeros((2,10)))]:
                np.savez(path, X_train=train, y_train=[0,1], X_test=test, y_test=[0,1])
                skipped = []
                with redirect_stdout(io.StringIO()):
                    result = common.load_benchmark_dataset(args, 'Unsupported', path, None, skipped)
                self.assertIsNone(result)
                self.assertEqual(len(skipped), 1)

    def test_unrelated_file_errors_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Broken.npz'
            np.savez(path, X_train=np.zeros((2,9)))
            args = argparse.Namespace(data=directory, datasets=None)
            skipped = []
            with self.assertRaisesRegex(ValueError, 'missing keys'):
                common.load_benchmark_dataset(args, 'Broken', path, None, skipped)
            self.assertEqual(skipped, [])


class BenchmarkUtilityTests(unittest.TestCase):
    def test_feature_verification_detects_one_changed_value(self):
        features = np.array([[0., .5], [1., .25]], dtype=np.float32)
        digest = common.verify_feature_values(features)
        self.assertEqual(common.verify_feature_values(features.copy(), digest), digest)
        changed = features.copy()
        changed[1, 1] = np.nextafter(changed[1, 1], np.float32(1.))
        with self.assertRaisesRegex(ValueError, 'Feature equality'):
            common.verify_feature_values(changed, digest)

    def test_modern_ridge_matches_independent_l2_normalization(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(12, 6))
        X[:, -1] = 1
        test = rng.normal(size=(5, 6))
        test[:, -1] = 1
        y = np.array([0, 1] * 6)
        means = X.mean(axis=0)
        norms = np.sqrt(((X - means) ** 2).sum(axis=0))
        norms[norms == 0] = 1
        reference = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit((X - means) / norms, y)
        candidate = common.make_classifier(len(y), 'ridge', 0).fit(X, y)
        np.testing.assert_allclose(candidate.decision_function(test), reference.decision_function((test - means) / norms), atol=1e-10)

    def test_parameter_bytes_count_shared_arrays_once(self):
        @dataclass
        class Parameters:
            weights: object
            rest: object
        weights = np.zeros((3, 9), dtype=np.float32)
        biases = np.zeros(4, dtype=np.float32)
        p = Parameters(weights, (weights, biases))
        self.assertEqual(common.parameter_nbytes(p), 124)

    def test_batch_transform_preserves_row_order(self):
        X = np.arange(35, dtype=np.float32).reshape(5, 7)
        model = common.ModelSpec('fixture', lambda X, seed: None,
                                 lambda X, params: X[:, :2], input_dim=2)
        result = common.transform_batches(model, X, None, 2)
        np.testing.assert_array_equal(result, X[:, :2])

    def test_verbose_is_available_and_defaults_off(self):
        from controller import build_parser as focused_parser
        parser = focused_parser()
        self.assertFalse(parser.parse_args(['--data', 'unused.npz']).verbose)
        self.assertTrue(parser.parse_args(['--data', 'unused.npz', '--verbose']).verbose)
if __name__ == '__main__':
    unittest.main()
