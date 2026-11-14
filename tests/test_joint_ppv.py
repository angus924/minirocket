"""Joint pooling must preserve the original transform bit for bit."""
import sys
import importlib.util
import platform
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from minirocket_baseline import fit_univariate, transform_univariate
import minirocket_joint_ppv_simd as joint


class JointPoolingTests(unittest.TestCase):
    def test_template_produces_working_arbitrary_threshold_pooling(self):
        from tools.generate_joint_ppv import generate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generated_pooling_fixture.py'
            path.write_text(generate())
            spec = importlib.util.spec_from_file_location('generated_pooling_fixture', path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            try:
                spec.loader.exec_module(module)
                responses = np.array([-2., 0., 1., 1., 3.], dtype=np.float64)
                biases = np.resize(np.array([1., -2., 1., 4., 0.]), 35)
                expected = np.resize(np.array([.2, .8, .2, 0., .6], dtype=np.float32), 35)
                np.testing.assert_array_equal(module.ppv_joint(responses, biases), expected)
            finally:
                sys.modules.pop(spec.name, None)

    def test_ties_duplicate_thresholds_and_original_threshold_order(self):
        responses = np.array([-2., 0., 1., 1., 3.], dtype=np.float32)
        biases = np.array([1., -2., 1., 4., 0.], dtype=np.float32)
        expected = np.array([.2, .8, .2, 0., .6], dtype=np.float32)
        for count in range(1, 25):
            np.testing.assert_array_equal(joint.ppv_joint(responses, np.resize(biases, count)),
                                          np.resize(expected, count))

    def test_joint_transform_matches_all_features_for_lengths_budgets_and_seeds(self):
        rng = np.random.default_rng(12)
        for length, budget, seed in [(9, 84, 0), (17, 168, 1), (31, 9996, 0),
                                     (100, 9996, 1), (301, 9996, 2), (1000, 9996, 0), (150, 5000, 1), (512, 10000, 1),
                                     (1639, 2500, 1)]:
            with self.subTest(length=length, budget=budget, seed=seed):
                train = rng.normal(size=(4, length)).astype(np.float32)
                test = rng.normal(size=(3, length)).astype(np.float32)
                parameters = fit_univariate(train, num_features=budget, random_state=seed)
                np.testing.assert_array_equal(joint.transform(test, parameters).view(np.uint32),
                                              transform_univariate(test, parameters).view(np.uint32))

    def test_constant_inputs_cropped_responses_and_many_biases(self):
        X = np.vstack([np.zeros(25), np.ones(25), -np.ones(25)]).astype(np.float32)
        for counts in ([1, 1], [3, 5], [8, 17], [119, 119]):
            parameters = (np.array([1, 3], dtype=np.int32), np.array(counts, dtype=np.int32),
                          np.resize(np.array([0., 1., -1., 3., 3.], dtype=np.float32), 84 * sum(counts)))
            np.testing.assert_array_equal(joint.transform(X, parameters),
                                          transform_univariate(X, parameters))

    def test_fit_preserves_original_parameters_and_batch_order(self):
        X = np.random.default_rng(9).normal(size=(4, 75)).astype(np.float32)
        expected = fit_univariate(X, random_state=2)
        parameters = joint.fit(X, random_state=2)
        for a, b in zip(parameters, expected):
            np.testing.assert_array_equal(a, b)
        copies = [value.copy() for value in parameters]
        whole = joint.transform(X, parameters)
        np.testing.assert_array_equal(whole, np.concatenate([joint.transform(X[:2], parameters),
                                                           joint.transform(X[2:], parameters)]))
        np.testing.assert_array_equal(joint.transform(X[::-1], parameters), whole[::-1])
        self.assertEqual(joint.transform(X[:0], parameters).shape, (0, 9996))
        for a, b in zip(parameters, copies):
            np.testing.assert_array_equal(a, b)


    def test_pooling_ties_subnormals_strides_and_block_remainders(self):
        candidate = joint
        tiny = np.nextafter(np.float32(0), np.float32(1))
        responses = np.array([-3, -tiny, -0., 0., tiny, 1., 1., 3.], np.float32)
        biases = np.array([1, 0, tiny, -tiny, 3, -3, 1], np.float32)
        for count in range(1, 34):
            b = np.resize(biases, count)
            for x in (responses, responses[::-1], responses[::2]):
                a = candidate.ppv_joint(x, b)
                expected = np.array([(x > bias).astype(np.float32).mean() for bias in b], np.float32)
                np.testing.assert_array_equal(a.view(np.uint32), expected.view(np.uint32))


    def test_constant_extreme_inputs_and_custom_threshold_counts(self):
        candidate = joint
        from minirocket_baseline import transform_univariate
        tiny = np.nextafter(np.float32(0), np.float32(1))
        X = np.vstack([np.zeros(33), np.ones(33), np.full(33, tiny),
                       np.resize([1e38, -1e38, 0.], 33)]).astype(np.float32)
        for counts in ([1, 4], [2, 3], [8, 17]):
            p = (np.array([1, 3], np.int32), np.array(counts, np.int32),
                 np.resize(np.array([0., 1., -1., tiny], np.float32), 84 * sum(counts)))
            np.testing.assert_array_equal(candidate.transform(X, p).view(np.uint32),
                                          transform_univariate(X, p).view(np.uint32))


    def test_long_input_dispatch_preserves_original_fallback(self):
        candidate = joint
        from unittest.mock import patch
        class LongInput:
            shape = (1, 2 ** 24 + 1)
        x, parameters = LongInput(), object()
        sentinel = object()
        with patch.object(candidate, 'validate_input', return_value=x), \
             patch.object(candidate, 'transform_univariate', return_value=sentinel) as fallback:
            self.assertIs(candidate.transform(x, parameters), sentinel)
            fallback.assert_called_once_with(x, parameters)


    def test_small_width_helpers_vectorize_on_audited_runtime(self):
        import numba
        from numba import njit
        import _joint_ppv_simd_kernels as kernels
        if platform.machine() != 'arm64' or numba.__version__ != '0.63.1':
            self.skipTest('Compiler regression check is specific to the audited ARM64/Numba runtime')
        for width in range(1, 5):
            # Fresh compilation avoids the empty inspection result of disk-cached overloads.
            function = njit(fastmath=True)(getattr(kernels, f'_pool_{width}').py_func)
            function(np.arange(512, dtype=np.float32), np.zeros(width, np.float32),
                     0, np.empty(width, np.float32))
            llvm = function.inspect_llvm(function.signatures[0])
            self.assertRegex(llvm, r'fcmp[^\n]*<\d+ x float>', f'width {width} lost SIMD')


if __name__ == '__main__':
    unittest.main()
