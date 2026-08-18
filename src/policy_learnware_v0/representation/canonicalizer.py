"""Padding-and-mask canonicalizer for v0 transition events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .normalization import NormalizationStats


def _readonly_array(value: Any, dtype: np.dtype[Any], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PackedEpisodeDataset:
    """Canonical fixed-width transitions plus their episode block structure."""

    packed: np.ndarray
    episode_offsets: np.ndarray
    reset_seeds: np.ndarray
    probe_seeds: np.ndarray
    task: str = ""
    schema_fingerprint: str = ""

    def __post_init__(self) -> None:
        packed = _readonly_array(self.packed, np.float32, name="packed")
        offsets = np.asarray(self.episode_offsets, dtype=np.int64)
        reset = np.asarray(self.reset_seeds, dtype=np.int64)
        probe = np.asarray(self.probe_seeds, dtype=np.int64)
        if packed.ndim != 2:
            raise ValueError(f"packed must have shape [T,D], got {packed.shape}")
        if offsets.ndim != 1 or offsets.size < 2:
            raise ValueError("episode_offsets must have shape [N+1]")
        if offsets[0] != 0 or offsets[-1] != packed.shape[0]:
            raise ValueError("episode_offsets must start at 0 and end at T")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("empty or non-monotonic episodes are not allowed")
        episode_count = offsets.size - 1
        if reset.shape != (episode_count,) or probe.shape != (episode_count,):
            raise ValueError("reset_seeds and probe_seeds must both have shape [N]")
        offsets = np.array(offsets, copy=True)
        reset = np.array(reset, copy=True)
        probe = np.array(probe, copy=True)
        offsets.setflags(write=False)
        reset.setflags(write=False)
        probe.setflags(write=False)
        object.__setattr__(self, "packed", packed)
        object.__setattr__(self, "episode_offsets", offsets)
        object.__setattr__(self, "reset_seeds", reset)
        object.__setattr__(self, "probe_seeds", probe)

    @property
    def transition_count(self) -> int:
        return int(self.packed.shape[0])

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    @property
    def packed_dim(self) -> int:
        return int(self.packed.shape[1])

    def episode_slice(self, episode_index: int) -> slice:
        if episode_index < 0 or episode_index >= self.episode_count:
            raise IndexError(episode_index)
        return slice(
            int(self.episode_offsets[episode_index]),
            int(self.episode_offsets[episode_index + 1]),
        )

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        from ..io import atomic_write_npz

        return atomic_write_npz(
            path,
            {
                "packed": self.packed,
                "episode_offsets": self.episode_offsets,
                "reset_seeds": self.reset_seeds,
                "probe_seeds": self.probe_seeds,
                "task": np.asarray(self.task),
                "schema_fingerprint": np.asarray(self.schema_fingerprint),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "PackedEpisodeDataset":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                packed=data["packed"],
                episode_offsets=data["episode_offsets"],
                reset_seeds=data["reset_seeds"],
                probe_seeds=data["probe_seeds"],
                task=str(data["task"]),
                schema_fingerprint=str(data["schema_fingerprint"]),
            )


@dataclass(frozen=True)
class TransitionCanonicalizer:
    stats: NormalizationStats
    max_action_dim: int = 6

    def __post_init__(self) -> None:
        if int(self.max_action_dim) <= 0:
            raise ValueError("max_action_dim must be positive")
        object.__setattr__(self, "max_action_dim", int(self.max_action_dim))

    @property
    def max_observation_dim(self) -> int:
        return self.stats.max_observation_dim

    @property
    def packed_dim(self) -> int:
        return 4 * self.max_observation_dim + 2 * self.max_action_dim + 1

    def pack(self, dataset: Any, schema: Any) -> PackedEpisodeDataset:
        observation = np.asarray(dataset.observation)
        action = np.asarray(dataset.action)
        reward = np.asarray(dataset.reward)
        next_observation = np.asarray(dataset.next_observation)
        observation_dim = int(schema.observation_dim)
        action_dim = int(schema.action_dim)

        if observation.ndim != 2 or observation.shape[1] != observation_dim:
            raise ValueError(
                f"{schema.task}: native observation shape {observation.shape} does "
                f"not match schema dimension {observation_dim}"
            )
        transition_count = int(observation.shape[0])
        if next_observation.shape != observation.shape:
            raise ValueError(f"{schema.task}: next_observation shape mismatch")
        if action.shape != (transition_count, action_dim):
            raise ValueError(
                f"{schema.task}: native action shape {action.shape} does not match "
                f"({transition_count}, {action_dim})"
            )
        if reward.shape != (transition_count,):
            raise ValueError(f"{schema.task}: reward must have shape [T]")
        if observation_dim > self.max_observation_dim:
            raise ValueError("native observation exceeds protocol maximum")
        if action_dim > self.max_action_dim:
            raise ValueError("native action exceeds protocol maximum")
        for name, value in (
            ("observation", observation),
            ("action", action),
            ("reward", reward),
            ("next_observation", next_observation),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{schema.task}: {name} contains non-finite values")

        normalized_observation = self.stats.normalize_observation(observation)
        normalized_next_observation = self.stats.normalize_observation(next_observation)
        normalized_reward = self.stats.normalize_reward(reward)
        obs_mask = np.zeros(self.max_observation_dim, dtype=np.float32)
        obs_mask[:observation_dim] = 1.0
        action_mask = np.zeros(self.max_action_dim, dtype=np.float32)
        action_mask[:action_dim] = 1.0

        packed = np.zeros((transition_count, self.packed_dim), dtype=np.float32)
        cursor = 0
        packed[:, cursor : cursor + observation_dim] = normalized_observation
        cursor += self.max_observation_dim
        packed[:, cursor : cursor + self.max_observation_dim] = obs_mask
        cursor += self.max_observation_dim
        # The protocol's action normalization is identity after env normalization.
        packed[:, cursor : cursor + action_dim] = action.astype(np.float32, copy=False)
        cursor += self.max_action_dim
        packed[:, cursor : cursor + self.max_action_dim] = action_mask
        cursor += self.max_action_dim
        packed[:, cursor] = normalized_reward
        cursor += 1
        packed[:, cursor : cursor + observation_dim] = normalized_next_observation
        cursor += self.max_observation_dim
        packed[:, cursor : cursor + self.max_observation_dim] = obs_mask

        # terminated/truncated intentionally do not enter ``packed``.  Offsets are
        # the sole episode-boundary representation supplied to the encoder/KME.
        return PackedEpisodeDataset(
            packed=packed,
            episode_offsets=np.asarray(dataset.episode_offsets),
            reset_seeds=np.asarray(dataset.reset_seeds),
            probe_seeds=np.asarray(dataset.probe_seeds),
            task=str(schema.task),
            schema_fingerprint=str(getattr(schema, "flatten_fingerprint", "")),
        )


def pack_transitions(
    dataset: Any,
    schema: Any,
    stats: NormalizationStats,
    *,
    max_action_dim: int = 6,
) -> PackedEpisodeDataset:
    """Functional wrapper matching the coding-plan API."""

    return TransitionCanonicalizer(stats, max_action_dim=max_action_dim).pack(
        dataset, schema
    )
