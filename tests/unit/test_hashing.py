from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from policy_learnware_v0.hashing import (
    CanonicalizationError,
    canonical_json,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.io import (
    ArtifactExistsError,
    atomic_write_npz,
    deterministic_npz_bytes,
    read_npz,
)
from policy_learnware_v0.schemas import EnvSchema, FrozenProtocol


class HashingTest(unittest.TestCase):
    def test_canonical_json_is_order_independent(self) -> None:
        left = {"b": np.asarray([2, 3], dtype=np.int32), "a": -0.0}
        right = {"a": 0.0, "b": [2, 3]}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(sha256_json(left), sha256_json(right))
        with self.assertRaises(CanonicalizationError):
            canonical_json({"bad": float("nan")})

    def test_named_array_hash_binds_dtype_shape_and_content(self) -> None:
        base = {"x": np.asarray([[1, 2]], dtype=np.int32)}
        self.assertEqual(sha256_ndarrays(base), sha256_ndarrays(dict(base)))
        self.assertNotEqual(
            sha256_ndarrays(base),
            sha256_ndarrays({"x": np.asarray([[1, 2]], dtype=np.int64)}),
        )

    def test_npz_encoding_is_deterministic_and_immutable_by_default(self) -> None:
        arrays = {"b": np.asarray([2.0]), "a": np.asarray([1], dtype=np.int32)}
        self.assertEqual(deterministic_npz_bytes(arrays), deterministic_npz_bytes(arrays))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.npz"
            first_digest = atomic_write_npz(path, arrays)
            self.assertEqual(set(read_npz(path)), {"a", "b"})
            with self.assertRaises(ArtifactExistsError):
                atomic_write_npz(path, arrays)
            second_digest = atomic_write_npz(path, arrays, overwrite=True)
            self.assertEqual(first_digest, second_digest)

    def test_frozen_protocol_id_round_trip(self) -> None:
        schema = EnvSchema(
            backend="synthetic.test-only",
            task="A",
            observation_dim=3,
            action_dim=2,
            action_low=np.asarray([-1, -1], dtype=np.float32),
            action_high=np.asarray([1, 1], dtype=np.float32),
            horizon=5,
            action_repeat=1,
            control_dt=1.0,
            flatten_fingerprint="flatten",
            implementation_digest="implementation",
        )
        protocol = FrozenProtocol.create(
            config={"draft": "x"},
            env_schemas={"A": schema},
            packed_layout={
                "width": 109,
                "max_observation_dim": 24,
                "max_action_dim": 6,
                "latent_dim": 32,
                "support_budget": 100,
                "kernel_bandwidth": 1.0,
                "layout_version": "pack109-v0",
            },
            component_digests={
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
            },
            runtime_versions={"numpy": np.__version__},
        )
        restored = FrozenProtocol.from_dict(protocol.to_dict())
        self.assertEqual(restored.protocol_id, protocol.protocol_id)
        self.assertEqual(restored.env_schemas["A"].digest, schema.digest)
        with self.assertRaises(TypeError):
            restored.config["draft"] = "mutated"


if __name__ == "__main__":
    unittest.main()
