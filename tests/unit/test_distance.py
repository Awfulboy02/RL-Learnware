from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.rkme.distance import empirical_to_reduced_distance  # noqa: E402
from policy_learnware_v0.rkme.empirical import build_empirical_kme  # noqa: E402
from policy_learnware_v0.rkme.gaussian import GaussianKernel  # noqa: E402
from policy_learnware_v0.rkme.reducer import (  # noqa: E402
    ReducedRKME,
    ReducerConfig,
    reduce_kme,
)


class DistanceTest(unittest.TestCase):
    def test_exact_unregularized_representation_has_zero_distance(self) -> None:
        target = build_empirical_kme(
            SimpleNamespace(
                points=np.array([[0.0], [1.0], [2.0]]),
                episode_offsets=np.array([0, 3]),
            ),
            GaussianKernel(0.9),
            protocol_id="p",
        )
        source = reduce_kme(
            target,
            ReducerConfig(
                support_budget=3,
                support_steps=0,
                kmeans_steps=0,
                ridge=0.0,
                pinv_rcond=1e-12,
            ),
        )
        result = empirical_to_reduced_distance(target, source, block_size=2)
        self.assertLess(result.distance, 1e-7)

    def test_small_negative_roundoff_is_clamped_and_recorded(self) -> None:
        target = build_empirical_kme(
            SimpleNamespace(
                points=np.array([[0.0]]), episode_offsets=np.array([0, 1])
            ),
            GaussianKernel(1.0),
            protocol_id="p",
        )
        source = ReducedRKME(
            supports=np.array([[0.0]]),
            beta=np.array([1.0]),
            bandwidth=1.0,
            rkme_norm2=1.0 - 1e-12,
            empirical_norm2=1.0,
            reduction_error=0.0,
            protocol_id="p",
        )
        result = empirical_to_reduced_distance(target, source)
        self.assertTrue(result.clamped)
        self.assertEqual(result.distance, 0.0)
        self.assertLess(result.raw_squared_distance, 0.0)

    def test_protocol_mismatch_fails(self) -> None:
        target = build_empirical_kme(
            SimpleNamespace(
                points=np.array([[0.0]]), episode_offsets=np.array([0, 1])
            ),
            GaussianKernel(1.0),
            protocol_id="target",
        )
        source = ReducedRKME(
            supports=np.array([[0.0]]),
            beta=np.array([1.0]),
            bandwidth=1.0,
            rkme_norm2=1.0,
            empirical_norm2=1.0,
            reduction_error=0.0,
            protocol_id="source",
        )
        with self.assertRaisesRegex(ValueError, "different protocols"):
            empirical_to_reduced_distance(target, source)


if __name__ == "__main__":
    unittest.main()
