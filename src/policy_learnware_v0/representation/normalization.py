"""Source-only, task-balanced normalization for transition events.

The normalizer deliberately does *not* use policy-bundle observation statistics.
Each source task contributes one equally weighted distribution, regardless of its
number of transitions.  For an observation slot, only tasks whose registered
schema contains that slot contribute to its moments.  Consequently padding zeros
can never affect fitted statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _readonly_float_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class NormalizationStats:
    """Frozen canonicalizer statistics.

    Invalid observation slots have mean zero and standard deviation one.  The
    ``observation_task_count`` vector records how many source tasks contributed
    to each slot, making the valid-slot rule auditable.
    """

    observation_mean: np.ndarray
    observation_std: np.ndarray
    reward_mean: float
    reward_std: float
    observation_task_count: np.ndarray
    source_task_count: int
    std_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        mean = _readonly_float_vector(self.observation_mean, name="observation_mean")
        std = _readonly_float_vector(self.observation_std, name="observation_std")
        counts = _readonly_float_vector(
            self.observation_task_count, name="observation_task_count"
        )
        if mean.shape != std.shape or mean.shape != counts.shape:
            raise ValueError("observation statistics must have identical shapes")
        if np.any(std <= 0):
            raise ValueError("observation_std must be strictly positive")
        if np.any(counts < 0):
            raise ValueError("observation_task_count cannot be negative")
        if int(self.source_task_count) <= 0:
            raise ValueError("source_task_count must be positive")
        if not np.isfinite(self.reward_mean) or not np.isfinite(self.reward_std):
            raise ValueError("reward statistics must be finite")
        if float(self.reward_std) <= 0:
            raise ValueError("reward_std must be strictly positive")
        if float(self.std_floor) <= 0:
            raise ValueError("std_floor must be strictly positive")
        object.__setattr__(self, "observation_mean", mean)
        object.__setattr__(self, "observation_std", std)
        object.__setattr__(self, "observation_task_count", counts)
        object.__setattr__(self, "reward_mean", float(self.reward_mean))
        object.__setattr__(self, "reward_std", float(self.reward_std))
        object.__setattr__(self, "source_task_count", int(self.source_task_count))
        object.__setattr__(self, "std_floor", float(self.std_floor))

    @property
    def max_observation_dim(self) -> int:
        return int(self.observation_mean.shape[0])

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Normalize a native-width observation array without padding it."""

        value = np.asarray(observation)
        if value.ndim not in (1, 2):
            raise ValueError(f"observation must have rank 1 or 2, got {value.shape}")
        width = int(value.shape[-1])
        if width > self.max_observation_dim:
            raise ValueError(
                f"observation width {width} exceeds protocol maximum "
                f"{self.max_observation_dim}"
            )
        result = (
            value.astype(np.float64, copy=False) - self.observation_mean[:width]
        ) / self.observation_std[:width]
        return result.astype(np.float32)

    def normalize_reward(self, reward: np.ndarray | float) -> np.ndarray:
        value = np.asarray(reward, dtype=np.float64)
        return ((value - self.reward_mean) / self.reward_std).astype(np.float32)

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        from ..io import atomic_write_npz

        return atomic_write_npz(
            path,
            {
                "observation_mean": self.observation_mean,
                "observation_std": self.observation_std,
                "reward_mean": np.asarray(self.reward_mean, dtype=np.float64),
                "reward_std": np.asarray(self.reward_std, dtype=np.float64),
                "observation_task_count": self.observation_task_count,
                "source_task_count": np.asarray(self.source_task_count, dtype=np.int64),
                "std_floor": np.asarray(self.std_floor, dtype=np.float64),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "NormalizationStats":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                observation_mean=data["observation_mean"],
                observation_std=data["observation_std"],
                reward_mean=float(data["reward_mean"]),
                reward_std=float(data["reward_std"]),
                observation_task_count=data["observation_task_count"],
                source_task_count=int(data["source_task_count"]),
                std_floor=float(data["std_floor"]),
            )


def _native_transition_arrays(dataset: Any, schema: Any) -> tuple[np.ndarray, ...]:
    observation = np.asarray(dataset.observation)
    next_observation = np.asarray(dataset.next_observation)
    reward = np.asarray(dataset.reward)
    observation_dim = int(schema.observation_dim)

    if observation.ndim != 2 or observation.shape[1] != observation_dim:
        raise ValueError(
            f"{schema.task}: observation shape {observation.shape} does not match "
            f"registered native dimension {observation_dim}"
        )
    if next_observation.shape != observation.shape:
        raise ValueError(
            f"{schema.task}: next_observation shape {next_observation.shape} does "
            f"not match observation shape {observation.shape}"
        )
    if reward.ndim != 1 or reward.shape[0] != observation.shape[0]:
        raise ValueError(f"{schema.task}: reward must have shape [T]")
    if observation.shape[0] == 0:
        raise ValueError(f"{schema.task}: cannot fit from an empty dataset")
    for name, value in (
        ("observation", observation),
        ("next_observation", next_observation),
        ("reward", reward),
    ):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{schema.task}: {name} contains non-finite values")
    return observation, next_observation, reward


def fit_normalizer(
    datasets: Mapping[str, Any],
    schemas: Mapping[str, Any],
    *,
    max_observation_dim: int | None = None,
    std_floor: float = 1.0e-6,
    role: str = "source",
    include_next_observation: bool = True,
) -> NormalizationStats:
    """Fit the frozen v0 normalizer from source-side datasets only.

    ``role`` is intentionally explicit so a target-side call fails loudly.  The
    default preserves the two-argument interface in the coding plan.
    Observation moments include both sides of each transition by default because
    the same transform is applied to :math:`o_h` and :math:`o_{h+1}`.
    """

    if role.lower() != "source":
        raise ValueError("fit_normalizer is source-only; target data must use frozen stats")
    if not datasets:
        raise ValueError("at least one source dataset is required")
    if set(datasets) != set(schemas):
        missing_data = sorted(set(schemas) - set(datasets))
        missing_schema = sorted(set(datasets) - set(schemas))
        raise ValueError(
            "dataset/schema task keys differ: "
            f"missing_data={missing_data}, missing_schema={missing_schema}"
        )
    if std_floor <= 0:
        raise ValueError("std_floor must be strictly positive")

    observed_max = max(int(schemas[task].observation_dim) for task in datasets)
    width = observed_max if max_observation_dim is None else int(max_observation_dim)
    if width < observed_max:
        raise ValueError(
            f"max_observation_dim={width} is smaller than native maximum {observed_max}"
        )

    # Per-task first and second moments are accumulated before averaging tasks.
    # This is the key difference from transition-balanced pooled statistics.
    task_mean_sum = np.zeros(width, dtype=np.float64)
    task_second_sum = np.zeros(width, dtype=np.float64)
    task_count = np.zeros(width, dtype=np.float64)
    reward_mean_sum = 0.0
    reward_second_sum = 0.0

    for task in sorted(datasets):
        observation, next_observation, reward = _native_transition_arrays(
            datasets[task], schemas[task]
        )
        values = (
            np.concatenate((observation, next_observation), axis=0)
            if include_next_observation
            else observation
        ).astype(np.float64, copy=False)
        native_width = int(values.shape[1])
        task_mean_sum[:native_width] += np.mean(values, axis=0)
        task_second_sum[:native_width] += np.mean(np.square(values), axis=0)
        task_count[:native_width] += 1.0

        reward64 = reward.astype(np.float64, copy=False)
        reward_mean_sum += float(np.mean(reward64))
        reward_second_sum += float(np.mean(np.square(reward64)))

    valid = task_count > 0
    observation_mean = np.zeros(width, dtype=np.float64)
    observation_second = np.ones(width, dtype=np.float64)
    observation_mean[valid] = task_mean_sum[valid] / task_count[valid]
    observation_second[valid] = task_second_sum[valid] / task_count[valid]
    observation_variance = np.maximum(
        observation_second - np.square(observation_mean), 0.0
    )
    observation_std = np.ones(width, dtype=np.float64)
    observation_std[valid] = np.maximum(
        np.sqrt(observation_variance[valid]), float(std_floor)
    )

    number_of_tasks = len(datasets)
    reward_mean = reward_mean_sum / number_of_tasks
    reward_variance = max(
        reward_second_sum / number_of_tasks - reward_mean * reward_mean, 0.0
    )
    reward_std = max(float(np.sqrt(reward_variance)), float(std_floor))

    return NormalizationStats(
        observation_mean=observation_mean,
        observation_std=observation_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
        observation_task_count=task_count,
        source_task_count=number_of_tasks,
        std_floor=std_floor,
    )
