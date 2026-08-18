"""Flat-array episode dataset schema; no pickle and no object arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..hashing import sha256_ndarrays
from ..io import atomic_write_json, atomic_write_npz, read_json, read_npz
from .seed_plan import SPLIT_NAMESPACES


EPISODE_DATASET_SCHEMA = "policy-learnware.episode-dataset.v0"
DATASET_MANIFEST_SCHEMA = "policy-learnware.dataset-manifest.v0"


def _readonly_array(
    value: np.ndarray,
    *,
    dtype: np.dtype[Any],
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    if array.dtype.hasobject:
        raise TypeError(f"{name} cannot be an object array")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EpisodeDataset:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_offsets: np.ndarray
    reset_seeds: np.ndarray
    probe_seeds: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "observation": _readonly_array(
                self.observation, dtype=np.dtype(np.float32), name="observation", ndim=2
            ),
            "action": _readonly_array(
                self.action, dtype=np.dtype(np.float32), name="action", ndim=2
            ),
            "reward": _readonly_array(
                self.reward, dtype=np.dtype(np.float32), name="reward", ndim=1
            ),
            "next_observation": _readonly_array(
                self.next_observation,
                dtype=np.dtype(np.float32),
                name="next_observation",
                ndim=2,
            ),
            "terminated": _readonly_array(
                self.terminated,
                dtype=np.dtype(np.bool_),
                name="terminated",
                ndim=1,
            ),
            "truncated": _readonly_array(
                self.truncated,
                dtype=np.dtype(np.bool_),
                name="truncated",
                ndim=1,
            ),
            "episode_offsets": _readonly_array(
                self.episode_offsets,
                dtype=np.dtype(np.int64),
                name="episode_offsets",
                ndim=1,
            ),
            "reset_seeds": _readonly_array(
                self.reset_seeds,
                dtype=np.dtype(np.int64),
                name="reset_seeds",
                ndim=1,
            ),
            "probe_seeds": _readonly_array(
                self.probe_seeds,
                dtype=np.dtype(np.int64),
                name="probe_seeds",
                ndim=1,
            ),
        }
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        self._validate()

    def _validate(self) -> None:
        transition_count = self.observation.shape[0]
        if transition_count <= 0:
            raise ValueError("EpisodeDataset cannot be empty")
        if self.observation.shape != self.next_observation.shape:
            raise ValueError("observation and next_observation shapes differ")
        if self.action.shape[0] != transition_count:
            raise ValueError("action transition count differs")
        for name in ("reward", "terminated", "truncated"):
            if getattr(self, name).shape != (transition_count,):
                raise ValueError(f"{name} transition count differs")
        if not np.all(np.isfinite(self.observation)):
            raise ValueError("observation contains non-finite values")
        if not np.all(np.isfinite(self.next_observation)):
            raise ValueError("next_observation contains non-finite values")
        if not np.all(np.isfinite(self.action)):
            raise ValueError("action contains non-finite values")
        if not np.all(np.isfinite(self.reward)):
            raise ValueError("reward contains non-finite values")
        offsets = self.episode_offsets
        if offsets.size < 2 or offsets[0] != 0 or offsets[-1] != transition_count:
            raise ValueError("episode_offsets must start at 0 and end at T")
        if np.any(np.diff(offsets) <= 0):
            raise ValueError("episodes must be non-empty and offsets strictly increasing")
        episode_count = offsets.size - 1
        if self.reset_seeds.shape != (episode_count,) or self.probe_seeds.shape != (
            episode_count,
        ):
            raise ValueError("seed arrays must contain one value per episode")
        if np.any(self.reset_seeds < 0) or np.any(self.probe_seeds < 0):
            raise ValueError("seeds must be non-negative")
        done = np.logical_or(self.terminated, self.truncated)
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            if np.any(done[int(start) : int(stop) - 1]):
                raise ValueError("done flag appears before an episode's final transition")
            if not done[int(stop) - 1]:
                raise ValueError("every stored episode must end with terminated or truncated")

    @property
    def transition_count(self) -> int:
        return int(self.observation.shape[0])

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    @property
    def observation_dim(self) -> int:
        return int(self.observation.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    def episode_slice(self, episode_index: int) -> slice:
        if not 0 <= episode_index < self.episode_count:
            raise IndexError(episode_index)
        return slice(
            int(self.episode_offsets[episode_index]),
            int(self.episode_offsets[episode_index + 1]),
        )

    def prefix(self, episode_count: int) -> "EpisodeDataset":
        if not 1 <= episode_count <= self.episode_count:
            raise ValueError("episode_count prefix lies outside the dataset")
        stop = int(self.episode_offsets[episode_count])
        return EpisodeDataset(
            observation=self.observation[:stop],
            action=self.action[:stop],
            reward=self.reward[:stop],
            next_observation=self.next_observation[:stop],
            terminated=self.terminated[:stop],
            truncated=self.truncated[:stop],
            episode_offsets=self.episode_offsets[: episode_count + 1],
            reset_seeds=self.reset_seeds[:episode_count],
            probe_seeds=self.probe_seeds[:episode_count],
        )

    def to_arrays(self, *, copy: bool = True) -> dict[str, np.ndarray]:
        result = {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_offsets": self.episode_offsets,
            "reset_seeds": self.reset_seeds,
            "probe_seeds": self.probe_seeds,
        }
        return (
            {name: np.array(array, copy=True) for name, array in result.items()}
            if copy
            else result
        )

    @property
    def digest(self) -> str:
        return sha256_ndarrays(self.to_arrays(copy=False))

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        return atomic_write_npz(path, self.to_arrays(copy=False), overwrite=overwrite)

    @classmethod
    def load_npz(cls, path: str | Path) -> "EpisodeDataset":
        return cls(**read_npz(path))


@dataclass(frozen=True)
class DatasetManifest:
    split: str
    task: str
    protocol_draft_hash: str
    episode_count: int
    transition_count: int
    reset_seeds: tuple[int, ...]
    probe_seeds: tuple[int, ...]
    dataset_sha256: str
    schema: str = DATASET_MANIFEST_SCHEMA

    @classmethod
    def from_dataset(
        cls,
        dataset: EpisodeDataset,
        *,
        split: str,
        task: str,
        protocol_draft_hash: str,
    ) -> "DatasetManifest":
        return cls(
            split=split,
            task=task,
            protocol_draft_hash=protocol_draft_hash,
            episode_count=dataset.episode_count,
            transition_count=dataset.transition_count,
            reset_seeds=tuple(int(seed) for seed in dataset.reset_seeds),
            probe_seeds=tuple(int(seed) for seed in dataset.probe_seeds),
            dataset_sha256=dataset.digest,
        )

    def __post_init__(self) -> None:
        if self.schema != DATASET_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported dataset manifest schema: {self.schema!r}")
        if not self.split or not self.task:
            raise ValueError("split and task must be non-empty")
        if self.split not in SPLIT_NAMESPACES:
            raise ValueError(f"unknown dataset split: {self.split!r}")
        for name, digest in (
            ("protocol_draft_hash", self.protocol_draft_hash),
            ("dataset_sha256", self.dataset_sha256),
        ):
            if len(digest) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
        if self.episode_count != len(self.reset_seeds) or self.episode_count != len(
            self.probe_seeds
        ):
            raise ValueError("manifest seed counts disagree with episode_count")
        if self.episode_count <= 0 or self.transition_count < self.episode_count:
            raise ValueError("invalid manifest counts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "split": self.split,
            "task": self.task,
            "protocol_draft_hash": self.protocol_draft_hash,
            "episode_count": self.episode_count,
            "transition_count": self.transition_count,
            "reset_seeds": list(self.reset_seeds),
            "probe_seeds": list(self.probe_seeds),
            "dataset_sha256": self.dataset_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        expected = {
            "schema",
            "split",
            "task",
            "protocol_draft_hash",
            "episode_count",
            "transition_count",
            "reset_seeds",
            "probe_seeds",
            "dataset_sha256",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "invalid DatasetManifest keys; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            schema=str(value["schema"]),
            split=str(value["split"]),
            task=str(value["task"]),
            protocol_draft_hash=str(value["protocol_draft_hash"]),
            episode_count=int(value["episode_count"]),
            transition_count=int(value["transition_count"]),
            reset_seeds=tuple(int(item) for item in value["reset_seeds"]),
            probe_seeds=tuple(int(item) for item in value["probe_seeds"]),
            dataset_sha256=str(value["dataset_sha256"]),
        )


def save_dataset_artifact(
    dataset: EpisodeDataset,
    *,
    npz_path: str | Path,
    manifest_path: str | Path,
    split: str,
    task: str,
    protocol_draft_hash: str,
    overwrite: bool = False,
) -> DatasetManifest:
    manifest = DatasetManifest.from_dataset(
        dataset,
        split=split,
        task=task,
        protocol_draft_hash=protocol_draft_hash,
    )
    dataset.save_npz(npz_path, overwrite=overwrite)
    atomic_write_json(manifest_path, manifest.to_dict(), overwrite=overwrite)
    return manifest


def load_dataset_artifact(
    npz_path: str | Path, manifest_path: str | Path
) -> tuple[EpisodeDataset, DatasetManifest]:
    dataset = EpisodeDataset.load_npz(npz_path)
    manifest = DatasetManifest.from_dict(read_json(manifest_path))
    if dataset.digest != manifest.dataset_sha256:
        raise ValueError("dataset content does not match manifest dataset_sha256")
    if dataset.episode_count != manifest.episode_count or (
        dataset.transition_count != manifest.transition_count
    ):
        raise ValueError("dataset counts do not match manifest")
    return dataset, manifest


def assert_dataset_splits_disjoint(
    datasets_by_split: Mapping[str, Mapping[str, EpisodeDataset]],
) -> None:
    owner: dict[tuple[str, int, int], str] = {}
    for split, datasets in datasets_by_split.items():
        for task, dataset in datasets.items():
            for reset_seed, probe_seed in zip(
                dataset.reset_seeds, dataset.probe_seeds, strict=True
            ):
                key = (task, int(reset_seed), int(probe_seed))
                previous = owner.get(key)
                if previous is not None and previous != split:
                    raise ValueError(
                        f"dataset seed overlap between {previous!r} and {split!r}: {key}"
                    )
                owner[key] = split
