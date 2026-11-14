"""Synthetic correctness checks for original-model adapters."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import minirocket_baseline as baseline


class SeededBaselineTests(unittest.TestCase):
    def test_univariate_seed_repeatability_and_original_output(self):
        import minirocket
        X = np.random.default_rng(8).normal(size=(4, 33)).astype(np.float32)
        before = np.random.get_state()
        a = baseline.fit_univariate(X, num_features=168, random_state=9)
        after = np.random.get_state()
        for left, right in zip(before, after):
            np.testing.assert_equal(left, right)
        b = baseline.fit_univariate(X, num_features=168, random_state=9)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(baseline.transform_univariate(X, a), minirocket.transform(X, a))

    def test_invalid_input_rejected_before_numba(self):
        for X in [np.zeros((2, 8)), np.zeros((0, 20)), np.full((2, 20), np.nan)]:
            with self.assertRaises(ValueError):
                baseline.fit_univariate(X)
        with self.assertRaises(ValueError):
            baseline.fit_univariate(np.zeros((2, 0)))
        with self.assertRaises(ValueError):
            baseline.fit_univariate(np.zeros((2, 20)), num_features=83)
        with self.assertRaises(ValueError):
            baseline.fit_univariate(np.zeros((2, 20)), random_state=-1)


if __name__ == '__main__':
    unittest.main()
