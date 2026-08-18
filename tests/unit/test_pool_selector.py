from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.pool.learnware import (  # noqa: E402
    LearnwarePool,
    PoolValidationError,
    SelectorEntry,
    SelectorTaskSpec,
    load_public_pool,
    save_public_pool,
)
import policy_learnware_v0.reuse.selector as selector_module  # noqa: E402
from policy_learnware_v0.reuse.selector import (  # noqa: E402
    NearestSpecSelector,
    SelectorError,
    TargetSpecView,
    target_source_cross_terms,
)
from policy_learnware_v0.rkme.empirical import (  # noqa: E402
    EmpiricalKME,
    build_empirical_kme,
)
from policy_learnware_v0.rkme.gaussian import GaussianKernel  # noqa: E402
from policy_learnware_v0.rkme.reducer import ReducedRKME  # noqa: E402


LW_A = "lw-" + "a" * 20
LW_B = "lw-" + "b" * 20


def _reduced(location: float, *, protocol: str = "protocol", bandwidth: float = 1.0):
    return ReducedRKME(
        supports=np.array([[location]], dtype=np.float64),
        beta=np.array([1.0]),
        bandwidth=bandwidth,
        rkme_norm2=1.0,
        empirical_norm2=1.0,
        reduction_error=0.0,
        protocol_id=protocol,
    )


def _pool(*, tied: bool = False) -> LearnwarePool:
    left = SelectorTaskSpec.from_rkme(
        _reduced(0.0), protocol_id="protocol", kernel_bandwidth=1.0
    )
    right = SelectorTaskSpec.from_rkme(
        _reduced(0.0 if tied else 5.0),
        protocol_id="protocol",
        kernel_bandwidth=1.0,
    )
    return LearnwarePool(
        pool_id="pool",
        protocol_id="protocol",
        kernel_bandwidth=1.0,
        entries=(
            SelectorEntry(LW_B, "protocol", right),
            SelectorEntry(LW_A, "protocol", left),
        ),
    )


def _target(*, protocol: str = "protocol", bandwidth: float = 1.0) -> EmpiricalKME:
    return EmpiricalKME(
        points=np.array([[0.0]]),
        weights=np.array([1.0]),
        episode_offsets=np.array([0, 1]),
        bandwidth=bandwidth,
        norm2=1.0,
        protocol_id=protocol,
        dataset_digest="query",
    )


class PoolSelectorTest(unittest.TestCase):
    def test_nearest_rkme_and_lexical_tie_break(self) -> None:
        result = NearestSpecSelector(_pool()).select(
            _target(),
            target_dataset_digest="query",
            probe_episode_count=1,
            probe_steps=1,
        )
        self.assertEqual(result.selected_opaque_id, LW_A)
        self.assertAlmostEqual(result.sorted_distances[0].distance, 0.0)
        self.assertEqual(type(result).from_dict(result.to_dict()), result)

        tampered = result.to_dict()
        tampered["selected_opaque_id"] = LW_B
        with self.assertRaises(SelectorError):
            type(result).from_dict(tampered)

        tie = NearestSpecSelector(_pool(tied=True)).select(
            _target(),
            target_dataset_digest="query",
            probe_episode_count=1,
            probe_steps=1,
        )
        self.assertEqual(tie.selected_opaque_id, LW_A)

    def test_precomputed_exact_terms_match_regular_selection(self) -> None:
        pool = _pool()
        target = build_empirical_kme(
            SimpleNamespace(
                points=np.array([[0.0], [0.5], [1.0]], dtype=np.float64),
                episode_offsets=np.array([0, 1, 3], dtype=np.int64),
            ),
            GaussianKernel(1.0),
            protocol_id="protocol",
            dataset_digest="query",
        )
        selector = NearestSpecSelector(pool)
        expected = selector.select(target)
        cross = target_source_cross_terms(target.points, target.weights, pool)
        actual = selector.select_from_precomputed_terms(
            target_empirical_norm2=target.norm2,
            target_source_cross=cross,
            target_dataset_digest=target.dataset_digest,
            probe_episode_count=target.episode_count,
            probe_steps=target.transition_count,
        )
        self.assertEqual(actual.selected_opaque_id, expected.selected_opaque_id)
        self.assertEqual(
            tuple(item.opaque_id for item in actual.sorted_distances),
            tuple(item.opaque_id for item in expected.sorted_distances),
        )
        np.testing.assert_allclose(
            [item.distance_squared for item in actual.sorted_distances],
            [item.distance_squared for item in expected.sorted_distances],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_selector_public_view_has_no_private_metadata(self) -> None:
        payload = _pool().public_manifest()
        encoded = json.dumps(payload).lower()
        for forbidden in (
            "task_name",
            "algorithm",
            "training_seed",
            "outer_iteration",
            "return",
            "policy_bundle",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_protocol_and_bandwidth_mismatch_fail_closed(self) -> None:
        selector = NearestSpecSelector(_pool())
        with self.assertRaisesRegex(SelectorError, "protocol"):
            selector.select(
                _target(protocol="other"),
                target_dataset_digest="query",
                probe_episode_count=1,
                probe_steps=1,
            )
        with self.assertRaisesRegex(SelectorError, "bandwidth"):
            selector.select(
                _target(bandwidth=2.0),
                target_dataset_digest="query",
                probe_episode_count=1,
                probe_steps=1,
            )
        with self.assertRaisesRegex(PoolValidationError, "bandwidth"):
            SelectorTaskSpec.from_rkme(
                _reduced(0.0, bandwidth=2.0),
                protocol_id="protocol",
                kernel_bandwidth=1.0,
            )

    def test_trusted_empirical_norm_is_reused_with_explicit_audit_available(self) -> None:
        selector = NearestSpecSelector(_pool())
        target = build_empirical_kme(
            SimpleNamespace(
                points=np.array([[0.0]], dtype=np.float64),
                episode_offsets=np.array([0, 1], dtype=np.int64),
            ),
            GaussianKernel(1.0),
            protocol_id="protocol",
            dataset_digest="query",
        )
        with patch(
            "policy_learnware_v0.reuse.selector."
            "blockwise_weighted_self_kernel_sum_auto"
        ) as recompute:
            result = selector.select(target)
        recompute.assert_not_called()
        self.assertEqual(result.selected_opaque_id, LW_A)
        with patch(
            "policy_learnware_v0.reuse.selector."
            "blockwise_weighted_self_kernel_sum_auto",
            wraps=selector_module.blockwise_weighted_self_kernel_sum_auto,
        ) as forced_audit:
            selector.select(target, verify_target_norm2=True)
        forced_audit.assert_called_once()

        inconsistent = EmpiricalKME(
            points=np.array([[0.0]]),
            weights=np.array([1.0]),
            episode_offsets=np.array([0, 1]),
            bandwidth=1.0,
            norm2=0.5,
            protocol_id="protocol",
            dataset_digest="query",
        )
        with self.assertRaisesRegex(SelectorError, "norm disagrees"):
            selector.select(inconsistent)

        inconsistent_view = TargetSpecView(
            semantic_events=np.array([[0.0]]),
            weights=np.array([1.0]),
            episode_offsets=np.array([0, 1]),
            empirical_norm2=0.5,
            protocol_id="protocol",
            kernel_bandwidth=1.0,
            dataset_digest="query",
        )
        with self.assertRaisesRegex(SelectorError, "norm disagrees"):
            selector.select(inconsistent_view)

    def test_loaded_empirical_is_untrusted_and_reaudited(self) -> None:
        selector = NearestSpecSelector(_pool())
        built = build_empirical_kme(
            SimpleNamespace(
                points=np.array([[0.0]], dtype=np.float64),
                episode_offsets=np.array([0, 1], dtype=np.int64),
            ),
            GaussianKernel(1.0),
            protocol_id="protocol",
            dataset_digest="query",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empirical.npz"
            built.save_npz(path)
            loaded = EmpiricalKME.load_npz(path)
            with patch(
                "policy_learnware_v0.reuse.selector."
                "blockwise_weighted_self_kernel_sum_auto",
                wraps=selector_module.blockwise_weighted_self_kernel_sum_auto,
            ) as recompute:
                result = selector.select(loaded)
            recompute.assert_called_once()
            self.assertEqual(result.selected_opaque_id, LW_A)

            inconsistent_path = Path(directory) / "inconsistent.npz"
            inconsistent = EmpiricalKME(
                points=np.array([[0.0]]),
                weights=np.array([1.0]),
                episode_offsets=np.array([0, 1]),
                bandwidth=1.0,
                norm2=0.5,
                protocol_id="protocol",
                dataset_digest="query",
            )
            inconsistent.save_npz(inconsistent_path)
            loaded_inconsistent = EmpiricalKME.load_npz(inconsistent_path)
            with self.assertRaisesRegex(SelectorError, "norm disagrees"):
                selector.select(loaded_inconsistent)

    def test_untyped_cached_norm_remains_fail_closed(self) -> None:
        payload = {
            "points": np.array([[0.0]]),
            "weights": np.array([1.0]),
            "episode_offsets": np.array([0, 1]),
            "bandwidth": 1.0,
            "norm2": 0.5,
            "protocol_id": "protocol",
            "dataset_digest": "query",
        }
        with self.assertRaisesRegex(SelectorError, "norm disagrees"):
            NearestSpecSelector(_pool()).select(payload)

    def test_public_pool_round_trip_contains_actual_rkme_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "published"
            save_public_pool(_pool(), target)
            loaded = load_public_pool(target)
            self.assertEqual(loaded.public_manifest(), _pool().public_manifest())
            for original, restored in zip(_pool().entries, loaded.entries):
                np.testing.assert_array_equal(
                    original.task_spec.supports, restored.task_spec.supports
                )
                np.testing.assert_array_equal(original.task_spec.beta, restored.task_spec.beta)


if __name__ == "__main__":
    unittest.main()
