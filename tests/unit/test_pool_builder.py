from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from policy_learnware_v0.policy.bundle import PolicyBundleMetadata  # noqa: E402
from policy_learnware_v0.pool.builder import build_pool  # noqa: E402
from policy_learnware_v0.pool.learnware import PoolValidationError  # noqa: E402
from policy_learnware_v0.rkme.reducer import ReducedRKME  # noqa: E402
from policy_learnware_v0.schemas import EnvSchema, FrozenProtocol  # noqa: E402


def _schema(task: str) -> EnvSchema:
    return EnvSchema(
        backend="synthetic.test-only",
        task=task,
        observation_dim=3,
        action_dim=2,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        horizon=1000,
        action_repeat=1,
        control_dt=1.0,
        flatten_fingerprint="flat",
        implementation_digest="implementation",
    )


def _protocol() -> FrozenProtocol:
    schemas = {task: _schema(task) for task in ("A", "B")}
    components = {
        name: chr(ord("a") + index) * 64
        for index, name in enumerate(
            (
                "environment_manifest",
                "probe_implementation",
                "normalization",
                "encoder",
                "kernel",
                "source_dataset_manifests",
            )
        )
    }
    return FrozenProtocol.create(
        config={
            "pool": {
                "pool_id": "pool",
                "checkpoint_outer": 6,
                "actual_environment_steps": 5_898_240,
            },
            "reducer": {"reconstruction_tolerance": 0.1},
        },
        env_schemas=schemas,
        packed_layout={
            "width": 109,
            "max_observation_dim": 24,
            "max_action_dim": 6,
            "latent_dim": 2,
            "support_budget": 1,
            "kernel_bandwidth": 1.0,
            "layout_version": "pack109-v0",
        },
        component_digests=components,
        runtime_versions={"runtime": "test"},
    )


def _spec(task: str, location: float, protocol_id: str) -> ReducedRKME:
    return ReducedRKME(
        supports=np.asarray([[location, 0.0]], dtype=np.float64),
        beta=np.asarray([1.0]),
        bandwidth=1.0,
        rkme_norm2=1.0,
        empirical_norm2=1.0,
        reduction_error=0.0,
        protocol_id=protocol_id,
        source_dataset_digest="d" * 64,
        source_task=task,
        source_dataset_manifest_digest="e" * 64,
    )


def _policy(task: str, digest: str) -> PolicyBundleMetadata:
    return PolicyBundleMetadata(
        bundle_dir=Path("/immutable") / task,
        bundle_digest=digest,
        task=task,
        algorithm="ppo",
        training_seed=0,
        outer_iteration=6,
        environment_steps=5_898_240,
        observation_dim=3,
        action_dim=2,
        manifest={},
        policy_spec={},
        provenance={},
    )


class PoolBuilderTest(unittest.TestCase):
    def test_pool_is_bound_to_frozen_protocol_and_private_registry(self) -> None:
        protocol = _protocol()
        built = build_pool(
            {
                "A": _spec("A", 0.0, protocol.protocol_id),
                "B": _spec("B", 2.0, protocol.protocol_id),
            },
            {"A": _policy("A", "a" * 64), "B": _policy("B", "b" * 64)},
            protocol=protocol,
        )
        self.assertEqual(len(built.public_pool.entries), 2)
        built.private_registry.validate_against(built.public_pool)
        self.assertNotIn("algorithm", str(built.public_pool.public_manifest()))

    def test_swapped_taskspec_fails_closed(self) -> None:
        protocol = _protocol()
        with self.assertRaisesRegex(PoolValidationError, "not bound"):
            build_pool(
                {
                    "A": _spec("B", 0.0, protocol.protocol_id),
                    "B": _spec("A", 2.0, protocol.protocol_id),
                },
                {"A": _policy("A", "a" * 64), "B": _policy("B", "b" * 64)},
                protocol=protocol,
            )


if __name__ == "__main__":
    unittest.main()
