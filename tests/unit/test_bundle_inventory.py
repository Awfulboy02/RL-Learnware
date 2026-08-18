from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.policy.bundle import (  # noqa: E402
    BundleValidationError,
    validate_bundle,
)
from policy_learnware_v0.policy.inventory import (  # noqa: E402
    resolve_successful_attempt,
    scan_policy_inventory,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(
    path: Path,
    *,
    task: str = "WalkerWalk",
    algorithm: str = "ppo",
    seed: int = 3,
    outer: int = 6,
    steps: int = 5_898_240,
    observation_dim: int = 3,
    action_dim: int = 2,
) -> None:
    path.mkdir(parents=True)
    np.savez_compressed(
        path / "actor.npz",
        layer_00_kernel=np.zeros((observation_dim, 4)),
        layer_00_bias=np.zeros(4),
        layer_01_kernel=np.zeros((4, 2 * action_dim)),
        layer_01_bias=np.zeros(2 * action_dim),
    )
    np.savez_compressed(
        path / "obs_stats.npz",
        count=np.asarray(10.0),
        mean=np.zeros(observation_dim),
        var_sum=np.ones(observation_dim),
        std=np.ones(observation_dim),
    )
    raw = np.zeros((8, action_dim), dtype=np.float32)
    np.savez_compressed(
        path / "golden_io.npz",
        observation=np.zeros((8, observation_dim), dtype=np.float32),
        prng_key_data=np.asarray([1, 2], dtype=np.uint32),
        raw_action=raw,
        environment_action=np.tanh(raw),
    )
    common = {
        "schema": "policy-learnware.policy-bundle.v0",
        "algorithm": algorithm,
        "task": task,
    }
    _write_json(
        path / "policy_spec.json",
        {
            **common,
            "observation_size": observation_dim,
            "action_size": action_dim,
            "actor_layer_sizes": [observation_dim, 4, 2 * action_dim],
            "actor_weights_file": "actor.npz",
            "golden_parity_file": "golden_io.npz",
            "environment_action_transform": "tanh(raw_action)",
            "observation_preprocessing": {
                "statistics_file": "obs_stats.npz",
                "normalize": True,
            },
            "training_config": {
                "episode_length": 1000,
                "normalize_observations": True,
            },
        },
    )
    _write_json(
        path / "provenance.json",
        {
            **common,
            "training_seed": seed,
            "outer_iteration": outer,
            "environment_steps": steps,
            "fpo_commit": "a" * 40,
            "expected_fpo_commit": "a" * 40,
            "fpo_commit_matches_expected": True,
            "fpo_tracked_dirty": False,
            "fpo_tracked_changes": [],
        },
    )
    payload_names = [
        "actor.npz",
        "obs_stats.npz",
        "golden_io.npz",
        "policy_spec.json",
        "provenance.json",
    ]
    _write_json(
        path / "bundle_manifest.json",
        {
            **common,
            "complete": True,
            "seed": seed,
            "outer_iteration": outer,
            "environment_steps": steps,
            "files": {
                name: {"bytes": (path / name).stat().st_size, "sha256": _sha(path / name)}
                for name in payload_names
            },
        },
    )


class BundleInventoryTest(unittest.TestCase):
    def test_checksum_and_structural_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "outer_000006"
            _make_bundle(bundle)
            metadata = validate_bundle(bundle, expected_outer=6)
            self.assertEqual(metadata.observation_dim, 3)
            self.assertEqual(metadata.action_dim, 2)
            self.assertEqual(metadata.algorithm, "ppo")

            with (bundle / "actor.npz").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(BundleValidationError, "size mismatch"):
                validate_bundle(bundle)

    def test_inventory_uses_successful_attempt_not_attempt_01(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory) / "runs" / "full" / "job-a"
            attempt1 = job_dir / "attempt_01"
            attempt2 = job_dir / "attempt_02"
            attempt1.mkdir(parents=True)
            attempt2.mkdir()
            _write_json(
                attempt1 / "queue_result.json",
                {"state": "failed", "attempt": 1, "returncode": 1},
            )
            success = {"state": "succeeded", "attempt": 2, "returncode": 0}
            _write_json(attempt2 / "queue_result.json", success)
            _write_json(job_dir / "queue_result.json", success)
            _write_json(
                attempt2 / "queue_job.json",
                {
                    "job": {
                        "phase": "full",
                        "task": "WalkerWalk",
                        "algorithm": "ppo",
                        "seed": 3,
                    }
                },
            )
            _make_bundle(attempt2 / "checkpoints" / "outer_000006")

            attempt, path, _ = resolve_successful_attempt(job_dir)
            self.assertEqual(attempt, 2)
            self.assertEqual(path.name, "attempt_02")
            report = scan_policy_inventory(
                Path(directory) / "runs",
                checkpoint_outer=6,
                expected_environment_steps=5_898_240,
            )
            self.assertFalse(report.rejected)
            self.assertEqual(len(report.items), 1)
            self.assertEqual(report.items[0].attempt, 2)

    def test_require_parity_needs_hook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "full").mkdir()
            with self.assertRaisesRegex(ValueError, "parity_verifier"):
                scan_policy_inventory(
                    directory,
                    checkpoint_outer=6,
                    require_parity=True,
                )


if __name__ == "__main__":
    unittest.main()
