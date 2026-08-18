from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from policy_learnware_v0.config import ProbeConfig, load_protocol_draft
from policy_learnware_v0.envs.base import SyntheticEnvAdapter
from policy_learnware_v0.probe.collector import collect_probe_episodes
from policy_learnware_v0.probe.dataset import (
    DatasetManifest,
    EpisodeDataset,
    assert_dataset_splits_disjoint,
    load_dataset_artifact,
    save_dataset_artifact,
)
from policy_learnware_v0.probe.seed_plan import SeedPlan


def make_variable_length_dataset() -> EpisodeDataset:
    observation = np.arange(15, dtype=np.float32).reshape(5, 3)
    return EpisodeDataset(
        observation=observation,
        action=np.zeros((5, 2), dtype=np.float32),
        reward=np.arange(5, dtype=np.float32),
        next_observation=observation + 1,
        terminated=np.asarray([False, True, False, False, False]),
        truncated=np.asarray([False, False, False, False, True]),
        episode_offsets=np.asarray([0, 2, 5]),
        reset_seeds=np.asarray([10, 20]),
        probe_seeds=np.asarray([11, 21]),
    )


class EpisodeDatasetTest(unittest.TestCase):
    def test_variable_length_offsets_and_prefix(self) -> None:
        dataset = make_variable_length_dataset()
        self.assertEqual(dataset.episode_count, 2)
        self.assertEqual(dataset.transition_count, 5)
        self.assertEqual(dataset.episode_slice(1), slice(2, 5))
        prefix = dataset.prefix(1)
        self.assertEqual(prefix.observation.shape, (2, 3))
        self.assertEqual(prefix.episode_offsets.tolist(), [0, 2])

    def test_invalid_transition_alignment_is_rejected(self) -> None:
        dataset = make_variable_length_dataset()
        with self.assertRaises(ValueError):
            EpisodeDataset(
                **{
                    **dataset.to_arrays(),
                    "episode_offsets": np.asarray([0, 3, 5]),
                }
            )

    def test_npz_and_manifest_round_trip(self) -> None:
        dataset = make_variable_length_dataset()
        with tempfile.TemporaryDirectory() as directory:
            npz_path = Path(directory) / "data.npz"
            json_path = Path(directory) / "data.json"
            manifest = save_dataset_artifact(
                dataset,
                npz_path=npz_path,
                manifest_path=json_path,
                split="source_taskspec",
                task="SyntheticTask",
                protocol_draft_hash="a" * 64,
            )
            restored, restored_manifest = load_dataset_artifact(npz_path, json_path)
            self.assertEqual(restored.digest, dataset.digest)
            self.assertEqual(restored_manifest, manifest)
            self.assertIsInstance(restored_manifest, DatasetManifest)

    def test_probe_collection_is_reproducible_clipped_and_aligned(self) -> None:
        environment = SyntheticEnvAdapter(horizon=4)
        config = ProbeConfig(
            type="clipped_gaussian", sigma=1.0, action_low=-1.0, action_high=1.0
        )
        kwargs = dict(
            env=environment,
            split="target_query",
            episode_ids=[0, 1],
            config=config,
            seed_plan=SeedPlan(123),
            task_index=0,
        )
        first = collect_probe_episodes(**kwargs)
        second = collect_probe_episodes(**kwargs)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.episode_offsets.tolist(), [0, 4, 8])
        np.testing.assert_array_equal(first.action, second.action)
        self.assertTrue(np.all(first.action >= -1.0))
        self.assertTrue(np.all(first.action <= 1.0))
        self.assertTrue(np.all(first.truncated[[3, 7]]))
        # If transitions were shifted, replaying the first recorded action would
        # not reproduce the first stored next observation.
        state, reset_observation = environment.reset(int(first.reset_seeds[0]))
        _, result = environment.step(state, first.action[0])
        np.testing.assert_array_equal(first.observation[0], reset_observation)
        np.testing.assert_array_equal(first.next_observation[0], result.observation)

    def test_production_protocol_cannot_fall_back_to_numpy_probe_rng(self) -> None:
        protocol = load_protocol_draft(
            PROJECT / "configs" / "dmc6_outer006_v0.yaml"
        )
        environment = SyntheticEnvAdapter(
            task="FingerSpin", observation_dim=9, action_dim=2, horizon=4
        )
        with self.assertRaisesRegex(RuntimeError, "vectorized JAX collector"):
            collect_probe_episodes(
                environment,
                "encoder_train",
                [0],
                protocol,
                prefer_vectorized=False,
            )

    def test_split_overlap_is_rejected(self) -> None:
        dataset = make_variable_length_dataset()
        with self.assertRaises(ValueError):
            assert_dataset_splits_disjoint(
                {"encoder_train": {"A": dataset}, "target_query": {"A": dataset}}
            )


if __name__ == "__main__":
    unittest.main()
