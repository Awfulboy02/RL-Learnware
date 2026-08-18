"""Supervised-contrastive utilities with episode-aware positive pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def positive_pair_mask(
    task_labels: np.ndarray, episode_ids: np.ndarray
) -> np.ndarray:
    """Return ``P(i)``: same task, different episode, never the anchor itself."""

    tasks = np.asarray(task_labels)
    episodes = np.asarray(episode_ids)
    if tasks.ndim != 1 or episodes.ndim != 1 or tasks.shape != episodes.shape:
        raise ValueError("task_labels and episode_ids must be aligned 1-D arrays")
    same_task = tasks[:, None] == tasks[None, :]
    different_episode = episodes[:, None] != episodes[None, :]
    mask = same_task & different_episode
    np.fill_diagonal(mask, False)
    return mask


def supervised_contrastive_loss(
    embeddings: np.ndarray,
    task_labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    temperature: float = 0.1,
    require_positive_for_every_anchor: bool = True,
) -> float:
    """Compute the v0 source-side SupCon objective with NumPy.

    This NumPy implementation is useful for unit tests and offline diagnostics.
    The differentiable JAX counterpart is ``supervised_contrastive_loss_jax``.
    """

    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("embeddings must have shape [B,Q] with B >= 2")
    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")
    positives = positive_pair_mask(task_labels, episode_ids)
    positive_count = positives.sum(axis=1)
    valid = positive_count > 0
    if require_positive_for_every_anchor and not np.all(valid):
        bad = np.flatnonzero(~valid).tolist()
        raise ValueError(f"anchors without a different-episode positive: {bad}")
    if not np.any(valid):
        raise ValueError("batch has no valid positive pairs")

    logits = (z @ z.T) / float(temperature)
    non_self = ~np.eye(z.shape[0], dtype=bool)
    row_max = np.max(np.where(non_self, logits, -np.inf), axis=1, keepdims=True)
    shifted = logits - row_max
    exp_logits = np.where(non_self, np.exp(shifted), 0.0)
    log_denominator = np.log(np.sum(exp_logits, axis=1)) + row_max[:, 0]
    log_probability = logits - log_denominator[:, None]
    per_anchor = -np.sum(np.where(positives, log_probability, 0.0), axis=1) / np.maximum(
        positive_count, 1
    )
    return float(np.mean(per_anchor[valid]))


def supervised_contrastive_loss_jax(
    embeddings: Any,
    task_labels: Any,
    episode_ids: Any,
    *,
    temperature: float = 0.1,
) -> Any:
    """Differentiable JAX implementation; dependencies are imported lazily."""

    try:
        import jax.numpy as jnp
        from jax.scipy.special import logsumexp
    except ImportError as exc:  # pragma: no cover - exercised without JAX via encoder
        raise ImportError(
            "JAX is required for differentiable SupCon training; install jax, flax "
            "and optax in the training environment"
        ) from exc

    if temperature <= 0:
        raise ValueError("temperature must be strictly positive")
    z = jnp.asarray(embeddings)
    tasks = jnp.asarray(task_labels)
    episodes = jnp.asarray(episode_ids)
    batch_size = z.shape[0]
    non_self = ~jnp.eye(batch_size, dtype=bool)
    positives = (
        (tasks[:, None] == tasks[None, :])
        & (episodes[:, None] != episodes[None, :])
        & non_self
    )
    logits = (z @ z.T) / temperature
    masked_logits = jnp.where(non_self, logits, -jnp.inf)
    log_denominator = logsumexp(masked_logits, axis=1)
    positive_count = jnp.sum(positives, axis=1)
    per_anchor = -jnp.sum(
        jnp.where(positives, logits - log_denominator[:, None], 0.0), axis=1
    ) / jnp.maximum(positive_count, 1)
    valid = positive_count > 0
    # The sampler guarantees valid anchors.  Keeping the masked mean here makes
    # validation robust while avoiding Python branching inside jit.
    return jnp.sum(jnp.where(valid, per_anchor, 0.0)) / jnp.maximum(jnp.sum(valid), 1)


@dataclass(frozen=True)
class ContrastiveBatch:
    transitions: np.ndarray
    task_labels: np.ndarray
    episode_ids: np.ndarray
    task_names: tuple[str, ...]
    source_indices: np.ndarray

    def __post_init__(self) -> None:
        transitions = np.asarray(self.transitions, dtype=np.float32)
        task_labels = np.asarray(self.task_labels, dtype=np.int32)
        episode_ids = np.asarray(self.episode_ids, dtype=np.int64)
        source_indices = np.asarray(self.source_indices, dtype=np.int64)
        size = transitions.shape[0]
        if transitions.ndim != 2:
            raise ValueError("transitions must have shape [B,D]")
        if any(value.shape != (size,) for value in (task_labels, episode_ids)):
            raise ValueError("batch labels must have shape [B]")
        if source_indices.shape != (size, 2):
            raise ValueError("source_indices must have shape [B,2]")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "task_labels", task_labels)
        object.__setattr__(self, "episode_ids", episode_ids)
        object.__setattr__(self, "source_indices", source_indices)


class TaskBalancedBatchSampler:
    """Sample tasks equally and transitions through uniformly chosen episodes.

    To preserve the SupCon definition, each task must have at least two episodes
    and each batch contributes at least two samples per task.  If ``batch_size``
    is not divisible by the number of tasks, the effective size is rounded down;
    this makes the nominal 1024 configuration a truly balanced 1020-example batch
    for six tasks rather than silently over-weighting four tasks.
    """

    def __init__(
        self,
        datasets: Mapping[str, Any],
        *,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if len(datasets) < 2:
            raise ValueError("contrastive training requires at least two tasks")
        self._tasks = tuple(sorted(datasets))
        self._datasets = {task: datasets[task] for task in self._tasks}
        self._samples_per_task = int(batch_size) // len(self._tasks)
        if self._samples_per_task < 2:
            raise ValueError("batch_size must provide at least two samples per task")
        self._effective_batch_size = self._samples_per_task * len(self._tasks)
        self._rng = np.random.default_rng(seed)
        for task, dataset in self._datasets.items():
            offsets = np.asarray(dataset.episode_offsets, dtype=np.int64)
            if offsets.ndim != 1 or offsets.size < 3 or np.any(np.diff(offsets) <= 0):
                raise ValueError(f"{task}: at least two non-empty episodes are required")

    @property
    def task_names(self) -> tuple[str, ...]:
        return self._tasks

    @property
    def effective_batch_size(self) -> int:
        return self._effective_batch_size

    def _episode_schedule(self, episode_count: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        remaining = self._samples_per_task
        while remaining:
            permutation = self._rng.permutation(episode_count)
            take = min(remaining, episode_count)
            chunks.append(permutation[:take])
            remaining -= take
        return np.concatenate(chunks)

    def sample(self) -> ContrastiveBatch:
        transitions: list[np.ndarray] = []
        labels: list[int] = []
        episode_ids: list[int] = []
        source_indices: list[tuple[int, int]] = []
        for task_index, task in enumerate(self._tasks):
            dataset = self._datasets[task]
            offsets = np.asarray(dataset.episode_offsets, dtype=np.int64)
            schedule = self._episode_schedule(offsets.size - 1)
            for episode_index in schedule:
                start = int(offsets[episode_index])
                stop = int(offsets[episode_index + 1])
                transition_index = int(self._rng.integers(start, stop))
                transitions.append(np.asarray(dataset.packed[transition_index]))
                labels.append(task_index)
                episode_ids.append(int(episode_index))
                source_indices.append((task_index, transition_index))

        order = self._rng.permutation(len(transitions))
        return ContrastiveBatch(
            transitions=np.stack(transitions, axis=0)[order],
            task_labels=np.asarray(labels, dtype=np.int32)[order],
            episode_ids=np.asarray(episode_ids, dtype=np.int64)[order],
            task_names=self._tasks,
            source_indices=np.asarray(source_indices, dtype=np.int64)[order],
        )
