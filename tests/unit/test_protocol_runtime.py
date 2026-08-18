from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from policy_learnware_v0.cli import (  # noqa: E402
    CommandFailure,
    _load_frozen_protocol,
    _runtime_package_versions,
    _verify_frozen_protocol_runtime,
)
from policy_learnware_v0.envs.mujoco_playground import (  # noqa: E402
    MujocoPlaygroundEnvAdapter,
    mujoco_playground_package_version,
)
from policy_learnware_v0.hashing import sha256_json  # noqa: E402
from policy_learnware_v0.schemas import EnvSchema, FrozenProtocol  # noqa: E402


def _protocol(
    *,
    source_digest: str | None = "f" * 64,
    runtime_versions: dict[str, str] | None = None,
) -> FrozenProtocol:
    schema = EnvSchema(
        backend="synthetic.test-only",
        task="TaskA",
        observation_dim=3,
        action_dim=2,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        horizon=3,
        action_repeat=1,
        control_dt=1.0,
        flatten_fingerprint="flatten",
        implementation_digest="synthetic",
    )
    component_names = (
        "environment_manifest",
        "probe_implementation",
        "normalization",
        "encoder",
        "kernel",
        "source_dataset_manifests",
    )
    return FrozenProtocol.create(
        config={"draft": "synthetic"},
        env_schemas={"TaskA": schema},
        packed_layout={
            "width": 109,
            "max_observation_dim": 24,
            "max_action_dim": 6,
            "latent_dim": 32,
            "support_budget": 100,
            "kernel_bandwidth": 1.0,
            "layout_version": "pack109-padding-mask-v0",
        },
        component_digests={
            **{
                name: f"{index:x}" * 64
                for index, name in enumerate(component_names, start=1)
            },
            **(
                {"implementation_source": source_digest}
                if source_digest is not None
                else {}
            ),
        },
        runtime_versions=runtime_versions
        or {
            "python": sys.version,
            "jax": "0.7.2",
            "playground": "0.0.5",
        },
    )


class PlaygroundPackageVersionTest(unittest.TestCase):
    def test_canonical_playground_distribution_is_detected(self) -> None:
        def version(distribution: str) -> str:
            if distribution == "playground":
                return "0.0.5"
            raise importlib_metadata.PackageNotFoundError(distribution)

        with patch(
            "policy_learnware_v0.envs.mujoco_playground.importlib.metadata.version",
            side_effect=version,
        ):
            self.assertEqual(mujoco_playground_package_version(), "0.0.5")
            self.assertEqual(MujocoPlaygroundEnvAdapter.package_version(), "0.0.5")

    def test_aliased_distribution_is_a_supported_fallback(self) -> None:
        def version(distribution: str) -> str:
            if distribution == "mujoco-playground":
                return "9.9.9"
            raise importlib_metadata.PackageNotFoundError(distribution)

        with patch(
            "policy_learnware_v0.envs.mujoco_playground.importlib.metadata.version",
            side_effect=version,
        ):
            self.assertEqual(mujoco_playground_package_version(), "9.9.9")

    def test_runtime_manifest_uses_canonical_playground_key(self) -> None:
        with patch(
            "policy_learnware_v0.cli.importlib_metadata.version",
            side_effect=lambda distribution: f"version-of-{distribution}",
        ), patch(
            "policy_learnware_v0.cli.mujoco_playground_package_version",
            return_value="0.0.5",
        ):
            versions = _runtime_package_versions()
        self.assertEqual(versions["playground"], "0.0.5")
        self.assertNotIn("mujoco-playground", versions)


class FrozenProtocolRuntimeTest(unittest.TestCase):
    def test_matching_source_and_runtime_are_accepted(self) -> None:
        protocol = _protocol()
        with patch(
            "policy_learnware_v0.cli._implementation_source_digest",
            return_value="f" * 64,
        ), patch(
            "policy_learnware_v0.cli._runtime_package_versions",
            return_value={"jax": "0.7.2", "playground": "0.0.5"},
        ):
            _verify_frozen_protocol_runtime(protocol)

    def test_runtime_accepts_legacy_playground_key_alias(self) -> None:
        protocol = _protocol(
            runtime_versions={
                "python": sys.version,
                "jax": "0.7.2",
                "mujoco-playground": "0.0.5",
            }
        )
        with patch(
            "policy_learnware_v0.cli._implementation_source_digest",
            return_value="f" * 64,
        ), patch(
            "policy_learnware_v0.cli._runtime_package_versions",
            return_value={"jax": "0.7.2", "playground": "0.0.5"},
        ):
            _verify_frozen_protocol_runtime(protocol)

    def test_implementation_source_drift_fails_closed(self) -> None:
        protocol = _protocol()
        with patch(
            "policy_learnware_v0.cli._implementation_source_digest",
            return_value="e" * 64,
        ):
            with self.assertRaisesRegex(CommandFailure, "implementation_source"):
                _verify_frozen_protocol_runtime(protocol)

    def test_missing_implementation_source_binding_fails_closed(self) -> None:
        protocol = _protocol(source_digest=None)
        with self.assertRaisesRegex(CommandFailure, "implementation_source"):
            _verify_frozen_protocol_runtime(protocol)

    def test_missing_python_binding_fails_closed(self) -> None:
        protocol = _protocol(
            runtime_versions={"jax": "0.7.2", "playground": "0.0.5"}
        )
        with patch(
            "policy_learnware_v0.cli._implementation_source_digest",
            return_value="f" * 64,
        ):
            with self.assertRaisesRegex(CommandFailure, "python runtime binding"):
                _verify_frozen_protocol_runtime(protocol)

    def test_python_runtime_drift_fails_closed(self) -> None:
        protocol = _protocol(
            runtime_versions={
                "python": "3.0.0 synthetic",
                "jax": "0.7.2",
                "playground": "0.0.5",
            }
        )
        with patch(
            "policy_learnware_v0.cli._implementation_source_digest",
            return_value="f" * 64,
        ):
            with self.assertRaisesRegex(CommandFailure, "python runtime mismatch"):
                _verify_frozen_protocol_runtime(protocol)

    def test_runtime_package_drift_fails_closed(self) -> None:
        protocol = _protocol()
        with patch(
            "policy_learnware_v0.cli._implementation_source_digest",
            return_value="f" * 64,
        ), patch(
            "policy_learnware_v0.cli._runtime_package_versions",
            return_value={"jax": "0.8.0", "playground": "0.0.5"},
        ):
            with self.assertRaisesRegex(CommandFailure, "runtime package mismatch.*jax"):
                _verify_frozen_protocol_runtime(protocol)

    def test_missing_frozen_package_binding_fails_closed(self) -> None:
        protocol = _protocol(
            runtime_versions={"python": sys.version, "jax": "0.7.2"}
        )
        with patch(
            "policy_learnware_v0.cli._implementation_source_digest",
            return_value="f" * 64,
        ), patch(
            "policy_learnware_v0.cli._runtime_package_versions",
            return_value={"jax": "0.7.2", "playground": "unavailable"},
        ):
            with self.assertRaisesRegex(CommandFailure, "lacks runtime package"):
                _verify_frozen_protocol_runtime(protocol)

    def test_persisted_protocol_loader_invokes_runtime_verification(self) -> None:
        protocol = _protocol()
        layout = MagicMock()
        layout.protocol_manifest = Path("protocol_manifest.json")
        layout.frozen_protocol = Path("frozen_protocol.json")
        layout.kernel = Path("kernel.json")
        layout.verify_manifest_files.return_value = {
            "protocol_id": protocol.protocol_id
        }
        config = SimpleNamespace(draft_hash=sha256_json(protocol.config))
        with patch(
            "policy_learnware_v0.cli.FrozenProtocol.load", return_value=protocol
        ), patch(
            "policy_learnware_v0.cli._verify_frozen_protocol_runtime"
        ) as verify_runtime, patch(
            "policy_learnware_v0.cli.GaussianKernel.load_json",
            return_value=SimpleNamespace(bandwidth=1.0),
        ):
            restored = _load_frozen_protocol(layout, config)
        self.assertIs(restored, protocol)
        verify_runtime.assert_called_once_with(protocol)


if __name__ == "__main__":
    unittest.main()
