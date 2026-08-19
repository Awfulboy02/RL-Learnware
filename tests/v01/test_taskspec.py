from __future__ import annotations

import unittest

import numpy as np

from policy_learnware_v0.rkme.distance import empirical_to_reduced_distance
from policy_learnware_v0.rkme.empirical import build_empirical_kme, empirical_mmd2
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.rkme.reducer import ReducedRKME
from policy_learnware_v0.v01.taskspec import (
    WeightedSemanticSample,
    compute_taskspec_matrix,
    direct_routing_scores,
    empirical_mmd_with_raw,
    exact_self_norm,
)


class TaskSpecV01Test(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = GaussianKernel(1.3)
        self.left = WeightedSemanticSample.from_points(
            np.asarray([[0.0, 0.2], [0.4, 0.5], [1.0, 0.7], [0.8, 0.1]]),
            np.asarray([0, 2, 4]),
        )
        self.right = WeightedSemanticSample.from_points(
            np.asarray([[0.2, 0.1], [0.3, 0.8], [0.9, 0.6], [0.7, 0.0]]),
            np.asarray([0, 2, 4]),
        )

    def test_raw_wrapper_matches_v0_empirical_mmd(self) -> None:
        left = build_empirical_kme(
            self.left,
            self.kernel,
            episode_offsets=self.left.episode_offsets,
        )
        right = build_empirical_kme(
            self.right,
            self.kernel,
            episode_offsets=self.right.episode_offsets,
        )
        result = empirical_mmd_with_raw(left, right, self.kernel, block_size=2)
        self.assertAlmostEqual(
            result.mmd2, empirical_mmd2(left, right, block_size=2), places=12
        )
        self.assertAlmostEqual(result.d_phi**2, result.mmd2, places=12)

    def test_prefix_rebalances_episode_mass(self) -> None:
        prefix = self.left.prefix(1)
        np.testing.assert_allclose(prefix.weights, [0.5, 0.5])

    def test_direct_score_has_full_distance_ranking(self) -> None:
        sources = {}
        for name, supports, beta in (
            ("a", [[0.0, 0.0], [0.5, 0.5]], [0.5, 0.5]),
            ("b", [[3.0, 3.0], [4.0, 4.0]], [0.5, 0.5]),
        ):
            support_array = np.asarray(supports, dtype=np.float64)
            beta_array = np.asarray(beta, dtype=np.float64)
            norm = float(beta_array @ self.kernel.gram(support_array) @ beta_array)
            sources[name] = ReducedRKME(
                supports=support_array,
                beta=beta_array,
                bandwidth=self.kernel.bandwidth,
                rkme_norm2=norm,
                empirical_norm2=norm,
                reduction_error=0.0,
            )
        scores = direct_routing_scores(self.left, sources, self.kernel, block_size=2)
        empirical = build_empirical_kme(
            self.left,
            self.kernel,
            episode_offsets=self.left.episode_offsets,
            block_size=2,
        )
        distances = {
            name: empirical_to_reduced_distance(empirical, source, block_size=2).distance
            for name, source in sources.items()
        }
        self.assertEqual(min(scores, key=scores.get), min(distances, key=distances.get))

    def test_self_norm_can_be_cached(self) -> None:
        value = exact_self_norm(self.left, self.kernel, block_size=2)
        self.assertTrue(np.isfinite(value))
        result = empirical_mmd_with_raw(
            self.left,
            self.left,
            self.kernel,
            left_norm2=value,
            right_norm2=value,
            block_size=2,
        )
        self.assertLessEqual(abs(result.raw_mmd2), 1.0e-12)

    def test_sparse_matrix_consumes_frozen_plan(self) -> None:
        supports = np.asarray([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
        beta = np.asarray([0.5, 0.5], dtype=np.float64)
        source = ReducedRKME(
            supports=supports,
            beta=beta,
            bandwidth=self.kernel.bandwidth,
            rkme_norm2=float(beta @ self.kernel.gram(supports) @ beta),
            empirical_norm2=1.0,
            reduction_error=0.0,
        )
        from policy_learnware_v0.v01.plans import build_pair_plan

        plan = build_pair_plan(
            [
                {"task": "private", "factor": 1.0, "variant_id": "v0"},
                {"task": "private", "factor": 2.0, "variant_id": "v1"},
            ],
            banks=2,
            gate_prefix=1,
            routing_prefix=2,
            within_bank_pairs=((0, 1),),
        )
        samples = {
            ("v0", 0): self.left,
            ("v0", 1): self.right,
            ("v1", 0): self.right,
            ("v1", 1): self.left,
        }
        result = compute_taskspec_matrix(
            samples, plan, kernel=self.kernel, sources={"source": source}, block_size=2
        )
        self.assertEqual(len(result.pair_rows), 4)
        self.assertEqual(len(result.routing_rows), 4)
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value)) if value else set()
            return set()

        self.assertTrue({"task", "source_task", "factor"}.isdisjoint(keys(result.to_dict())))


if __name__ == "__main__":
    unittest.main()
