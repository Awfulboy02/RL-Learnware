from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.representation.canonicalizer import (  # noqa: E402
    TransitionCanonicalizer,
)
from policy_learnware_v0.representation.normalization import (  # noqa: E402
    NormalizationStats,
)


def make_dataset() -> SimpleNamespace:
    return SimpleNamespace(
        observation=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        action=np.array([[0.25, -0.5], [0.5, -1.0]], dtype=np.float32),
        reward=np.array([2.0, 4.0], dtype=np.float32),
        next_observation=np.array(
            [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]], dtype=np.float32
        ),
        terminated=np.array([False, True]),
        truncated=np.array([False, False]),
        episode_offsets=np.array([0, 2], dtype=np.int64),
        reset_seeds=np.array([11], dtype=np.int64),
        probe_seeds=np.array([22], dtype=np.int64),
    )


class CanonicalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = NormalizationStats(
            observation_mean=np.zeros(24),
            observation_std=np.ones(24),
            reward_mean=1.0,
            reward_std=1.0,
            observation_task_count=np.ones(24),
            source_task_count=6,
        )
        self.schema = SimpleNamespace(
            task="tiny", observation_dim=3, action_dim=2, flatten_fingerprint="fp"
        )

    def test_exact_109_layout_and_zero_padding(self) -> None:
        packed = TransitionCanonicalizer(self.stats, max_action_dim=6).pack(
            make_dataset(), self.schema
        )
        self.assertEqual(packed.packed.shape, (2, 109))
        row = packed.packed[0]
        np.testing.assert_allclose(row[0:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(row[3:24], 0.0)
        np.testing.assert_allclose(row[24:48], [1.0, 1.0, 1.0] + [0.0] * 21)
        np.testing.assert_allclose(row[48:54], [0.25, -0.5, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(row[54:60], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(row[60], 1.0)
        np.testing.assert_allclose(row[61:64], [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(row[64:85], 0.0)
        np.testing.assert_array_equal(row[85:109], row[24:48])

    def test_done_flags_are_not_encoder_inputs(self) -> None:
        first = make_dataset()
        second = make_dataset()
        second.terminated[:] = False
        second.truncated[:] = True
        canonicalizer = TransitionCanonicalizer(self.stats)
        np.testing.assert_array_equal(
            canonicalizer.pack(first, self.schema).packed,
            canonicalizer.pack(second, self.schema).packed,
        )

    def test_native_shape_mismatch_fails(self) -> None:
        dataset = make_dataset()
        dataset.observation = dataset.observation[:, :2]
        with self.assertRaisesRegex(ValueError, "native observation"):
            TransitionCanonicalizer(self.stats).pack(dataset, self.schema)


if __name__ == "__main__":
    unittest.main()
