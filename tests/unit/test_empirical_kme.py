from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.rkme.empirical import (  # noqa: E402
    blockwise_weighted_kernel_sum,
    blockwise_weighted_kernel_sum_jax,
    blockwise_weighted_self_kernel_sum_jax,
    build_empirical_kme,
    dense_weighted_kernel_sum,
    empirical_mmd2,
    episode_balanced_weights,
)
from policy_learnware_v0.evaluation.retrieval_accel import (  # noqa: E402
    nested_prefix_self_kernel_sums_jax,
)
from policy_learnware_v0.rkme.gaussian import GaussianKernel  # noqa: E402


class EmpiricalKMETest(unittest.TestCase):
    def test_each_episode_has_equal_total_mass(self) -> None:
        offsets = np.array([0, 2, 6, 7], dtype=np.int64)
        weights = episode_balanced_weights(offsets)
        for index in range(3):
            self.assertAlmostEqual(weights[offsets[index] : offsets[index + 1]].sum(), 1 / 3)
        self.assertAlmostEqual(weights.sum(), 1.0)

    def test_blockwise_matches_dense(self) -> None:
        rng = np.random.default_rng(4)
        left = rng.normal(size=(11, 3))
        right = rng.normal(size=(7, 3))
        left_weights = rng.uniform(size=11)
        right_weights = rng.normal(size=7)  # beta may be negative.
        kernel = GaussianKernel(0.8)
        dense = dense_weighted_kernel_sum(
            left, left_weights, right, right_weights, kernel
        )
        blocked = blockwise_weighted_kernel_sum(
            left, left_weights, right, right_weights, kernel, block_size=3
        )
        self.assertAlmostEqual(dense, blocked, places=12)

    def test_identical_empirical_kme_has_zero_mmd(self) -> None:
        events = SimpleNamespace(
            points=np.array([[0.0], [1.0], [2.0], [3.0]]),
            episode_offsets=np.array([0, 1, 4]),
        )
        empirical = build_empirical_kme(events, GaussianKernel(1.0), protocol_id="p")
        self.assertAlmostEqual(empirical_mmd2(empirical, empirical, block_size=2), 0.0)

    @unittest.skipUnless(importlib.util.find_spec("jax") is not None, "JAX unavailable")
    def test_jax_blockwise_sums_match_dense(self) -> None:
        rng = np.random.default_rng(4)
        left = rng.normal(size=(11, 3))
        right = rng.normal(size=(7, 3))
        left_weights = rng.uniform(size=11)
        left_weights /= left_weights.sum()
        right_weights = rng.uniform(size=7)
        right_weights /= right_weights.sum()
        kernel = GaussianKernel(0.9)
        expected_cross = dense_weighted_kernel_sum(
            left, left_weights, right, right_weights, kernel
        )
        expected_self = dense_weighted_kernel_sum(
            left, left_weights, left, left_weights, kernel
        )
        self.assertAlmostEqual(
            blockwise_weighted_kernel_sum_jax(
                left,
                left_weights,
                right,
                right_weights,
                kernel,
                block_size=4,
            ),
            expected_cross,
            places=10,
        )
        self.assertAlmostEqual(
            blockwise_weighted_self_kernel_sum_jax(
                left, left_weights, kernel, block_size=4
            ),
            expected_self,
            places=10,
        )

    @unittest.skipUnless(importlib.util.find_spec("jax") is not None, "JAX unavailable")
    def test_nested_prefix_self_norms_match_independent_dense_formulas(self) -> None:
        rng = np.random.default_rng(17)
        points = rng.normal(size=(14, 4))
        offsets = np.array([0, 2, 7, 10, 14], dtype=np.int64)
        prefixes = (1, 2, 4)
        kernel = GaussianKernel(1.3)
        actual = nested_prefix_self_kernel_sums_jax(
            points,
            offsets,
            prefixes,
            kernel,
            block_size=3,
        )
        for count in prefixes:
            stop = int(offsets[count])
            weights = episode_balanced_weights(offsets[: count + 1])
            expected = dense_weighted_kernel_sum(
                points[:stop],
                weights,
                points[:stop],
                weights,
                kernel,
            )
            self.assertAlmostEqual(actual[count], expected, places=10)


if __name__ == "__main__":
    unittest.main()
