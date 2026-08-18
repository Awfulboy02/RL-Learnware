from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.representation.contrastive import (  # noqa: E402
    TaskBalancedBatchSampler,
    positive_pair_mask,
    supervised_contrastive_loss,
)
from policy_learnware_v0.representation.encoder import (  # noqa: E402
    EncoderConfig,
    EncoderDependencyError,
    TransitionSemanticEncoder,
    jax_encoder_available,
)


class ContrastiveTest(unittest.TestCase):
    def test_positives_are_same_task_different_episode(self) -> None:
        tasks = np.array([0, 0, 0, 1, 1])
        episodes = np.array([0, 0, 1, 0, 1])
        expected = np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(positive_pair_mask(tasks, episodes), expected)

    def test_supcon_prefers_task_aligned_embeddings(self) -> None:
        tasks = np.array([0, 0, 1, 1])
        episodes = np.array([0, 1, 0, 1])
        aligned = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        mixed = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
        self.assertLess(
            supervised_contrastive_loss(aligned, tasks, episodes),
            supervised_contrastive_loss(mixed, tasks, episodes),
        )

    def test_balanced_sampler_and_positive_coverage(self) -> None:
        datasets = {}
        for task_index in range(3):
            datasets[f"task{task_index}"] = SimpleNamespace(
                packed=np.full((8, 5), task_index, dtype=np.float32),
                episode_offsets=np.array([0, 2, 4, 6, 8], dtype=np.int64),
            )
        batch = TaskBalancedBatchSampler(datasets, batch_size=13, seed=7).sample()
        # Nominal 13 is rounded down to a strictly balanced 12.
        self.assertEqual(batch.transitions.shape, (12, 5))
        np.testing.assert_array_equal(np.bincount(batch.task_labels), [4, 4, 4])
        self.assertTrue(np.all(positive_pair_mask(batch.task_labels, batch.episode_ids).any(1)))

    def test_missing_jax_has_friendly_runtime_error(self) -> None:
        if jax_encoder_available():
            self.skipTest("JAX/Flax is installed in this interpreter")
        with self.assertRaisesRegex(EncoderDependencyError, "JAX and Flax"):
            TransitionSemanticEncoder.initialize(EncoderConfig(train_steps=0))


if __name__ == "__main__":
    unittest.main()
