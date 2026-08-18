from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.representation.normalization import (  # noqa: E402
    NormalizationStats,
    fit_normalizer,
)


def dataset(observation: np.ndarray, reward: np.ndarray) -> SimpleNamespace:
    observation = np.asarray(observation, dtype=np.float32)
    return SimpleNamespace(
        observation=observation,
        next_observation=observation.copy(),
        reward=np.asarray(reward, dtype=np.float32),
    )


class NormalizationTest(unittest.TestCase):
    def test_task_balanced_valid_slot_moments_ignore_sample_count(self) -> None:
        # Task A contributes one transition [0]; task B contributes three [10,20].
        # A transition-pooled slot-0 mean would be 7.5, while task balance gives 5.
        datasets = {
            "a": dataset([[0.0]], [0.0]),
            "b": dataset([[10.0, 20.0]] * 3, [10.0] * 3),
        }
        schemas = {
            "a": SimpleNamespace(task="a", observation_dim=1),
            "b": SimpleNamespace(task="b", observation_dim=2),
        }
        stats = fit_normalizer(datasets, schemas, max_observation_dim=4)
        np.testing.assert_allclose(stats.observation_mean, [5.0, 20.0, 0.0, 0.0])
        np.testing.assert_allclose(stats.observation_std, [5.0, 1.0e-6, 1.0, 1.0])
        np.testing.assert_allclose(stats.observation_task_count, [2.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(stats.reward_mean, 5.0)
        self.assertAlmostEqual(stats.reward_std, 5.0)

    def test_target_cannot_fit(self) -> None:
        datasets = {"a": dataset([[1.0]], [1.0])}
        schemas = {"a": SimpleNamespace(task="a", observation_dim=1)}
        with self.assertRaisesRegex(ValueError, "source-only"):
            fit_normalizer(datasets, schemas, role="target")

    def test_npz_round_trip(self) -> None:
        stats = NormalizationStats(
            observation_mean=np.array([1.0, 2.0]),
            observation_std=np.array([3.0, 4.0]),
            reward_mean=5.0,
            reward_std=6.0,
            observation_task_count=np.array([2.0, 1.0]),
            source_task_count=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalization.npz"
            stats.save_npz(path)
            loaded = NormalizationStats.load_npz(path)
        np.testing.assert_array_equal(loaded.observation_mean, stats.observation_mean)
        np.testing.assert_array_equal(loaded.observation_std, stats.observation_std)
        self.assertEqual(loaded.reward_mean, stats.reward_mean)


if __name__ == "__main__":
    unittest.main()
