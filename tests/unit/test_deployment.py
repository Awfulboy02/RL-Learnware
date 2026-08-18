from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.evaluation.deployment import (  # noqa: E402
    DeploymentResult,
    deploy_selected,
)
from policy_learnware_v0.pool.registry import (  # noqa: E402
    DeploymentRegistry,
    RegistryRecord,
    load_private_registry,
    save_private_registry,
)
from policy_learnware_v0.reuse.selector import (  # noqa: E402
    DistanceRecord,
    SelectionResult,
)


LW_A = "lw-" + "a" * 20
LW_B = "lw-" + "b" * 20
POOL_DIGEST = "c" * 64


def _selection(selected: str = LW_A) -> SelectionResult:
    distances = (
        DistanceRecord(selected, 0.0, 0.0, False),
        DistanceRecord(LW_B if selected == LW_A else LW_A, 1.0, 1.0, False),
    )
    return SelectionResult(
        selection_id="selection",
        protocol_id="protocol",
        target_dataset_digest="query",
        selected_opaque_id=selected,
        sorted_distances=distances,
        probe_episode_count=1,
        probe_steps=10,
        selector_runtime_seconds=0.0,
        clamp_count=0,
        pool_id="pool",
        pool_digest=POOL_DIGEST,
    )


def _registry() -> DeploymentRegistry:
    return DeploymentRegistry(
        (
            RegistryRecord(
                opaque_id=LW_A,
                protocol_id="protocol",
                policy_bundle=Path("/private/a"),
                policy_bundle_digest="a" * 64,
                native_observation_dim=3,
                native_action_dim=2,
                source_task="source-a",
                provenance={"algorithm": "fpo", "return": 123.0},
            ),
            RegistryRecord(
                opaque_id=LW_B,
                protocol_id="protocol",
                policy_bundle=Path("/private/b"),
                policy_bundle_digest="b" * 64,
                native_observation_dim=9,
                native_action_dim=4,
                source_task="source-b",
            ),
        ),
        pool_id="pool",
        pool_digest=POOL_DIGEST,
    )


class _Policy:
    observation_dim = 3
    action_dim = 2


class DeploymentTest(unittest.TestCase):
    def test_incompatible_selected_policy_fails_without_load_or_fallback(self) -> None:
        loaded: list[str] = []

        def loader(record):
            loaded.append(record.opaque_id)
            return _Policy()

        result = deploy_selected(
            _selection(LW_A),
            _registry(),
            SimpleNamespace(observation_dim=9, action_dim=4),
            policy_loader=loader,
            evaluator=lambda policy: [1.0],
        )
        self.assertEqual(result.deployment_failure, "incompatible_native_schema")
        self.assertEqual(loaded, [])
        self.assertFalse(result.deployable)

    def test_compatible_path_loads_exactly_selected_policy_once(self) -> None:
        loaded: list[str] = []

        def loader(record):
            loaded.append(record.opaque_id)
            return _Policy()

        result = deploy_selected(
            _selection(LW_A),
            _registry(),
            {"observation_dim": 3, "action_dim": 2},
            policy_loader=loader,
            evaluator=lambda policy: [1.0, 3.0],
        )
        self.assertTrue(result.deployable)
        self.assertEqual(result.mean_return, 2.0)
        self.assertEqual(loaded, [LW_A])
        restored = DeploymentResult.from_dict(result.to_dict())
        self.assertEqual(restored, result)

        malformed = result.to_dict()
        malformed["mean_return"] = 9.0
        with self.assertRaises(ValueError):
            DeploymentResult.from_dict(malformed)

    def test_private_registry_round_trip_remains_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            save_private_registry(_registry(), path)
            restored = load_private_registry(path)
            self.assertEqual(len(restored), 2)
            self.assertEqual(restored.get(LW_A).source_task, "source-a")
            self.assertEqual(restored.get(LW_A).provenance["return"], 123.0)


if __name__ == "__main__":
    unittest.main()
