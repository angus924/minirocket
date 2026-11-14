"""Small checks for the analysis helpers in experiments/ (eta squared and the bootstrap interval)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT))
from experiments import equivalence, threshold_class_share as share  # noqa: E402


class EtaSquaredTests(unittest.TestCase):
    def test_perfect_separation_is_one_and_equal_means_is_zero(self):
        labels = np.array([0, 0, 1, 1])
        separated = np.array([[1.0], [1.0 + 1e-9], [5.0], [5.0 + 1e-9]])
        self.assertAlmostEqual(float(share.eta_squared(separated, labels)[0]), 1.0, places=6)
        same_means = np.array([[1.0], [3.0], [1.0], [3.0]])
        self.assertAlmostEqual(float(share.eta_squared(same_means, labels)[0]), 0.0, places=9)

    def test_constant_columns_are_dropped(self):
        values = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0], [1.0, 8.0]])
        self.assertEqual(share.eta_squared(values, np.array([0, 0, 1, 1])).shape, (1,))

    def test_series_selection_keeps_only_classes_with_two_series(self):
        y = np.array(['a', 'a', 'b', 'c', 'c', 'c'])
        index = share.pick_series(y, 100, np.random.default_rng(0))
        self.assertEqual(sorted(y[index]), ['a', 'a', 'c', 'c', 'c'])


class IntervalTests(unittest.TestCase):
    def test_interval_brackets_the_mean_and_is_reproducible(self):
        a = {f'd{i}': 0.5 + 0.01 * (i % 3) for i in range(30)}
        b = {f'd{i}': 0.5 for i in range(30)}
        n, mean, low, high = equivalence.interval(a, b)
        self.assertEqual(n, 30)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)
        self.assertEqual((n, mean, low, high), equivalence.interval(a, b))


if __name__ == '__main__':
    unittest.main()
