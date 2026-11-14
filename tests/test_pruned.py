"""Group pruning and stratified biases must preserve the original MiniRocket numbers."""
import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import minirocket as original
import minirocket_joint_ppv_simd as joint
import minirocket_pruned as pruned
import minirocket_stratified as stratified
from minirocket_baseline import fit_univariate, transform_univariate

LENGTH, BUDGET = 41, 840          # small: 4 dilations x 84 kernels, 10 features per kernel


def bits(array):
    return np.ascontiguousarray(array).view(np.uint32)


def setup_data(seed=0):
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(10, LENGTH)).astype(np.float32)
    test = rng.normal(size=(4, LENGTH)).astype(np.float32)
    return train, test, fit_univariate(train, num_features=BUDGET, random_state=3)


def python_version(function):
    """Pure-Python body of a jitted function (Numba dispatchers expose ``py_func``)."""
    return getattr(function, 'py_func', function)


class GroupMaskTests(unittest.TestCase):
    def test_layout_is_contiguous_dilation_major_and_complete(self):
        _, _, parameters = setup_data()
        sizes, starts = pruned.group_layout(parameters)
        self.assertEqual(sizes.shape, (len(parameters[0]), 84))
        flat_starts, flat_sizes = starts.ravel(), sizes.ravel()
        self.assertEqual(flat_starts[0], 0)
        np.testing.assert_array_equal(flat_starts[1:], np.cumsum(flat_sizes)[:-1])
        self.assertEqual(flat_sizes.sum(), len(parameters[2]))
        for j, n in enumerate(parameters[1]):
            self.assertTrue((sizes[j] == n).all())

    def test_keep_all_matches_original_and_joint_bit_for_bit(self):
        _, test, parameters = setup_data()
        expected = transform_univariate(test, parameters)
        np.testing.assert_array_equal(bits(joint.transform(test, parameters)), bits(expected))
        for implementation in ('joint', 'original'):
            got = pruned.transform(test, parameters, pruned.keep_all(parameters), implementation)
            np.testing.assert_array_equal(bits(got), bits(expected), implementation)

    def test_pruned_output_is_exact_column_slice_of_full_transform(self):
        _, test, parameters = setup_data()
        full = transform_univariate(test, parameters)
        shape = pruned.keep_all(parameters).shape
        masks = []
        for seed, rate in enumerate([0.05, 0.3, 0.5, 0.8]):
            mask = np.random.default_rng(seed).random(shape) < rate
            mask[0, 0] = True
            masks.append(mask)
        whole_dilation = np.ones(shape, bool); whole_dilation[1, :] = False
        last_only = np.zeros(shape, bool); last_only[-1, -1] = True
        first_only = np.zeros(shape, bool); first_only[0, 0] = True
        odd_kernels = np.zeros(shape, bool); odd_kernels[:, 1::2] = True      # padding parity pattern
        masks += [whole_dilation, last_only, first_only, odd_kernels]
        for mask in masks:
            columns = pruned.kept_columns(parameters, mask)
            self.assertEqual(len(columns), pruned.pruned_width(parameters, mask))
            for implementation in ('joint', 'original'):
                got = pruned.transform(test, parameters, mask, implementation)
                self.assertEqual(got.shape, (len(test), len(columns)))
                np.testing.assert_array_equal(bits(got), bits(full[:, columns]), implementation)

    def test_inputs_are_validated_and_parameters_untouched(self):
        _, test, parameters = setup_data()
        copies = [value.copy() for value in parameters]
        mask = pruned.keep_all(parameters); mask[::2] = False
        pruned.transform(test, parameters, mask)
        for a, b in zip(parameters, copies):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(pruned.transform(test[:0], parameters, mask).shape, (0, pruned.pruned_width(parameters, mask)))
        with self.assertRaises(ValueError):
            pruned.transform(test, parameters, np.zeros_like(mask))             # nothing kept
        with self.assertRaises(ValueError):
            pruned.transform(test, parameters, mask[:-1])                        # wrong shape
        with self.assertRaises(ValueError):
            pruned.transform(test, parameters, mask, implementation='other')

    def test_direct_convolution_oracle_matches_layout_and_padding_rule(self):
        # Independent of the C_alpha / C_gamma trick: the paper's definition of the convolution.
        _, test, parameters = setup_data()
        dilations, per_dilation, biases = parameters
        full = transform_univariate(test, parameters)
        _, starts = pruned.group_layout(parameters)
        triples = list(combinations(range(9), 3))
        for example in range(len(test)):
            x = test[example]
            for j, d in enumerate(map(int, dilations)):
                for k in range(84):
                    weights = -np.ones(9); weights[list(triples[k])] = 2
                    response = np.zeros(LENGTH)
                    for tap in range(9):
                        shift = (tap - 4) * d
                        lo, hi = max(0, -shift), min(LENGTH, LENGTH - shift)
                        response[lo:hi] += weights[tap] * x[lo + shift:hi + shift].astype(np.float64)
                    if (j % 2 + k) % 2:
                        response = response[4 * d:-4 * d]
                    s, n = starts[j, k], per_dilation[j]
                    expected = np.array([(response > b).mean() for b in biases[s:s + n]])
                    np.testing.assert_allclose(full[example, s:s + n], expected, atol=1e-6)

    def test_renumbering_dilations_would_change_padding_so_it_is_not_done(self):
        _, test, parameters = setup_data()
        dilations, per_dilation, biases = parameters
        full = transform_univariate(test, parameters)
        keep = np.ones((len(dilations), 84), bool); keep[0] = False
        compacted = (dilations[1:], per_dilation[1:], biases[int(per_dilation[0]) * 84:])
        naive = transform_univariate(test, compacted)
        self.assertFalse(np.array_equal(naive, full[:, pruned.kept_columns(parameters, keep)]))
        np.testing.assert_array_equal(bits(pruned.transform(test, parameters, keep)),
                                      bits(full[:, pruned.kept_columns(parameters, keep)]))


class RankingTests(unittest.TestCase):
    def planted(self, n=60, seed=0):
        _, _, parameters = setup_data()
        rng = np.random.default_rng(seed)
        y = np.arange(n) % 3
        features = rng.normal(size=(n, len(parameters[2]))).astype(np.float32)
        _, starts = pruned.group_layout(parameters)
        s = starts[2, 7]
        features[:, s:s + 3] += 4.0 * (y[:, None] == np.arange(3)[None, :])      # informative group (2, 7)
        return parameters, features, y

    def test_group_scores_shape_nan_and_normalisation(self):
        _, _, parameters = setup_data()
        keep = pruned.keep_all(parameters); keep[1, :] = False
        width = pruned.pruned_width(parameters, keep)
        coef = np.ones((3, width))
        scores = pruned.group_scores(coef, parameters, keep)
        self.assertEqual(scores.shape, keep.shape)
        self.assertTrue(np.isnan(scores[1]).all() and not np.isnan(scores[0]).any())
        np.testing.assert_allclose(scores[~np.isnan(scores)], np.sqrt(3))        # ||W_g||_F / sqrt(p_g)
        with self.assertRaises(ValueError):
            pruned.group_scores(coef[:, :-1], parameters, keep)
        self.assertEqual(pruned.group_scores(np.ones(width), parameters, keep).shape, keep.shape)  # binary: 1 row

    def test_both_scorers_keep_the_planted_group(self):
        parameters, features, y = self.planted()
        for scorer in ('ridge', 'filter'):
            keep, history = pruned.prune_groups(features, y, parameters, 0.05, scorer=scorer)
            self.assertTrue(keep[2, 7], scorer)
            self.assertEqual(int(keep.sum()), max(1, int(round(0.05 * keep.size))), scorer)
            self.assertEqual(history[-1]['groups_after'], int(keep.sum()))

    def test_pruning_is_deterministic_and_respects_a_starting_mask(self):
        parameters, features, y = self.planted(seed=1)
        a, _ = pruned.prune_groups(features, y, parameters, 0.25)
        b, _ = pruned.prune_groups(features, y, parameters, 0.25)
        np.testing.assert_array_equal(a, b)
        start = pruned.keep_all(parameters); start[:, :10] = False
        c, _ = pruned.prune_groups(features, y, parameters, 0.5, keep=start)
        self.assertFalse(c[:, :10].any())
        self.assertEqual(int(c.sum()), int(round(0.5 * start.sum())))
        with self.assertRaises(ValueError):
            pruned.prune_groups(features, y, parameters, 0.0)


class StratifiedTests(unittest.TestCase):
    def test_table_is_class_balanced_reproducible_and_seeded(self):
        y = np.array(['a'] * 12 + ['b'] * 3 + ['c'] * 5)
        groups = 336
        table = stratified.draw_example_table(y, groups, 1, 'stratified', 0)
        self.assertEqual(table.shape, (groups, 1))
        self.assertEqual(table.dtype, np.int32)
        counts = [int((y[table[:, 0]] == c).sum()) for c in 'abc']
        self.assertLessEqual(max(counts) - min(counts), 1)                       # equal class share
        uniform = stratified.draw_example_table(y, groups, 1, 'uniform', 0)
        self.assertGreater(int((y[uniform[:, 0]] == 'a').sum()), counts[0])      # majority over-represented
        np.testing.assert_array_equal(table, stratified.draw_example_table(y, groups, 1, 'stratified', 0))
        self.assertFalse(np.array_equal(table, stratified.draw_example_table(y, groups, 1, 'stratified', 1)))
        pooled = stratified.draw_example_table(y, groups, 3, 'stratified', 0)
        self.assertEqual(pooled.shape, (groups, 3))
        self.assertTrue(all(len(set(y[row])) == 3 for row in pooled))            # one per class when m == #classes
        self.assertTrue(((0 <= pooled) & (pooled < len(y))).all())
        with self.assertRaises(ValueError):
            stratified.draw_example_table(y, groups, 1, 'other', 0)

    def test_fit_returns_original_compatible_parameters(self):
        train, test, reference = setup_data()
        y = np.arange(len(train)) % 2
        parameters = stratified.fit(train, y, num_features=BUDGET, strategy='stratified', random_state=0)
        np.testing.assert_array_equal(parameters[0], reference[0])
        np.testing.assert_array_equal(parameters[1], reference[1])
        self.assertEqual([a.dtype for a in parameters], [a.dtype for a in reference])
        self.assertEqual(parameters[2].shape, reference[2].shape)
        features = joint.transform(test, parameters)                             # usable by every transform
        np.testing.assert_array_equal(bits(features), bits(transform_univariate(test, parameters)))
        again = stratified.fit(train, y, num_features=BUDGET, strategy='stratified', random_state=0)
        np.testing.assert_array_equal(again[2], parameters[2])
        with self.assertRaises(ValueError):
            stratified.fit(train, y[:-1], num_features=BUDGET)

    def test_single_example_loop_reproduces_original_fit_given_same_draws(self):
        # Runs the pure-Python bodies so both fits can share NumPy's RNG stream.
        train, _, (dilations, per_dilation, _) = setup_data()
        quantiles = original._quantiles(84 * int(per_dilation.sum()))
        groups = len(dilations) * 84
        np.random.seed(7)
        table = np.array([[np.random.randint(len(train))] for _ in range(groups)], dtype=np.int32)
        np.random.seed(7)
        expected = python_version(original._fit_biases)(train, dilations, per_dilation, quantiles)
        got = python_version(stratified._fit_biases_from_table)(train, dilations, per_dilation, quantiles, table)
        np.testing.assert_array_equal(got, expected)

    def test_pooled_quantiles_match_direct_convolution_oracle(self):
        train, _, (dilations, per_dilation, _) = setup_data()
        y = np.arange(len(train)) % 2
        parameters = stratified.fit(train, y, num_features=BUDGET, examples_per_group=2,
                                    strategy='stratified', random_state=5)
        table = stratified.draw_example_table(y, len(dilations) * 84, 2, 'stratified', 5)
        quantiles = original._quantiles(84 * int(per_dilation.sum()))
        _, starts = pruned.group_layout(parameters)
        triples = list(combinations(range(9), 3))
        group = 0
        for j, d in enumerate(map(int, dilations)):
            for k in range(84):
                weights = -np.ones(9); weights[list(triples[k])] = 2
                pooled = []
                for index in table[group]:
                    response = np.zeros(LENGTH)
                    for tap in range(9):
                        shift = (tap - 4) * d
                        lo, hi = max(0, -shift), min(LENGTH, LENGTH - shift)
                        response[lo:hi] += weights[tap] * train[index][lo + shift:hi + shift].astype(np.float64)
                    pooled.append(response)
                s, n = starts[j, k], per_dilation[j]
                expected = np.quantile(np.concatenate(pooled), quantiles[s:s + n].astype(np.float64))
                np.testing.assert_allclose(parameters[2][s:s + n], expected, atol=1e-4)
                group += 1


if __name__ == '__main__':
    unittest.main()
