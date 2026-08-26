"""Source-fitted heterogeneous transition canonicalization for v0.3.

``CanonicalTransitionBatch`` is intentionally only a value/shape contract.  It
cannot prove that native task arrays were padded with the one globally frozen
protocol.  This module supplies that missing provenance layer:

* native banks retain their registered task ABI and logical data role;
* a task-balanced normalizer can be fitted only from source representation
  roles;
* one global registry freezes observation/action widths and native schemas;
* every transformed batch is wrapped by a digest-bound receipt.

Formal cross-task Raw KME code is expected to require
``CanonicalizedBankReceipt`` rather than accepting a bare transition batch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from .windowing import CanonicalTransitionBatch


NATIVE_BANK_SCHEMA = "policy-learnware.v03-native-transition-bank.v0"
NATIVE_SHAPE_RECORD_SCHEMA = "policy-learnware.v03-native-shape-record.v0"
NATIVE_SHAPE_REGISTRY_SCHEMA = "policy-learnware.v03-native-shape-registry.v0"
GLOBAL_NORMALIZER_SCHEMA = "policy-learnware.v03-global-normalizer.v0"
GLOBAL_CANONICALIZER_SCHEMA = "policy-learnware.v03-global-canonicalizer.v0"
CANONICALIZED_BANK_RECEIPT_SCHEMA = (
    "policy-learnware.v03-canonicalized-bank-receipt.v0"
)

SourceFitRole = Literal[
    "source_representation_train", "source_representation_validation"
]
NativeBankRole = Literal[
    "source_representation_train",
    "source_representation_validation",
    "source_reference_spec",
    "development_query",
    "confirmatory_query",
]

SOURCE_FIT_ROLES = frozenset(
    {"source_representation_train", "source_representation_validation"}
)
NATIVE_BANK_ROLES = frozenset(
    {
        *SOURCE_FIT_ROLES,
        "source_reference_spec",
        "development_query",
        "confirmatory_query",
    }
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


class CanonicalizationError(ValueError):
    """Native shape, source fit, transform, or receipt is invalid."""


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise CanonicalizationError(f"{where} must be a canonical safe ID")
    return value


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise CanonicalizationError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise CanonicalizationError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CanonicalizationError(f"{where} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise CanonicalizationError(f"{where} must be a positive integer")
    return result


def _readonly_float(value: Any, *, where: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.shape[0] <= 0 or not np.all(np.isfinite(array)):
        raise CanonicalizationError(
            f"{where} must be a non-empty finite {ndim}D numeric array"
        )
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _readonly_bool(value: Any, *, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.ndim != 1 or raw.shape[0] <= 0:
        raise CanonicalizationError(f"{where} must be a non-empty boolean vector")
    result = np.ascontiguousarray(raw, dtype=np.bool_).copy()
    result.setflags(write=False)
    return result


def _readonly_int(value: Any, *, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.shape[0] <= 0:
        raise CanonicalizationError(f"{where} must be a non-empty integer vector")
    result = np.ascontiguousarray(raw, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NativeTransitionBank:
    """Immutable native-width transition bank before global padding."""

    bank_id: str
    task_private_id: str
    data_role: NativeBankRole
    native_schema_digest: str
    raw_dataset_digest: str
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_id: np.ndarray
    timestep: np.ndarray
    schema: str = NATIVE_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NATIVE_BANK_SCHEMA:
            raise CanonicalizationError("unsupported NativeTransitionBank schema")
        object.__setattr__(self, "bank_id", _safe_id(self.bank_id, "bank_id"))
        object.__setattr__(
            self,
            "task_private_id",
            _safe_id(self.task_private_id, "task_private_id"),
        )
        if self.data_role not in NATIVE_BANK_ROLES:
            raise CanonicalizationError(f"unknown native bank role: {self.data_role!r}")
        object.__setattr__(
            self,
            "native_schema_digest",
            _digest(self.native_schema_digest, "native_schema_digest"),
        )
        object.__setattr__(
            self,
            "raw_dataset_digest",
            _digest(self.raw_dataset_digest, "raw_dataset_digest"),
        )

        observation = _readonly_float(self.observation, where="observation", ndim=2)
        action = _readonly_float(self.action, where="action", ndim=2)
        reward = _readonly_float(self.reward, where="reward", ndim=1)
        next_observation = _readonly_float(
            self.next_observation, where="next_observation", ndim=2
        )
        terminated = _readonly_bool(self.terminated, where="terminated")
        truncated = _readonly_bool(self.truncated, where="truncated")
        episode_id = _readonly_int(self.episode_id, where="episode_id")
        timestep = _readonly_int(self.timestep, where="timestep")
        rows = observation.shape[0]
        if action.shape[0] != rows or next_observation.shape != observation.shape:
            raise CanonicalizationError("native observation/action row shapes disagree")
        for name, array in (
            ("reward", reward),
            ("terminated", terminated),
            ("truncated", truncated),
            ("episode_id", episode_id),
            ("timestep", timestep),
        ):
            if array.shape != (rows,):
                raise CanonicalizationError(f"{name} must have shape [N]")

        # Reuse the canonical transition validator for episode and done rules,
        # but do not confuse this native-width validation with canonicalization.
        CanonicalTransitionBatch(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            observation_mask=None,
            action_mask=None,
            episode_id=episode_id,
            timestep=timestep,
        )
        for name, array in (
            ("observation", observation),
            ("action", action),
            ("reward", reward),
            ("next_observation", next_observation),
            ("terminated", terminated),
            ("truncated", truncated),
            ("episode_id", episode_id),
            ("timestep", timestep),
        ):
            object.__setattr__(self, name, array)

    @property
    def observation_dim(self) -> int:
        return int(self.observation.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    @property
    def native_bank_digest(self) -> str:
        return sha256_json(
            {
                "schema": self.schema,
                "bank_id": self.bank_id,
                "task_private_id": self.task_private_id,
                "data_role": self.data_role,
                "native_schema_digest": self.native_schema_digest,
                "raw_dataset_digest": self.raw_dataset_digest,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "arrays_digest": sha256_ndarrays(
                    {
                        "observation": self.observation,
                        "action": self.action,
                        "reward": self.reward,
                        "next_observation": self.next_observation,
                        "terminated": self.terminated,
                        "truncated": self.truncated,
                        "episode_id": self.episode_id,
                        "timestep": self.timestep,
                    }
                ),
            }
        )


@dataclass(frozen=True)
class NativeShapeRecord:
    task_private_id: str
    observation_dim: int
    action_dim: int
    native_schema_digest: str
    schema: str = NATIVE_SHAPE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NATIVE_SHAPE_RECORD_SCHEMA:
            raise CanonicalizationError("unsupported NativeShapeRecord schema")
        object.__setattr__(
            self,
            "task_private_id",
            _safe_id(self.task_private_id, "task_private_id"),
        )
        object.__setattr__(
            self,
            "observation_dim",
            _positive_int(self.observation_dim, "observation_dim"),
        )
        object.__setattr__(
            self, "action_dim", _positive_int(self.action_dim, "action_dim")
        )
        object.__setattr__(
            self,
            "native_schema_digest",
            _digest(self.native_schema_digest, "native_schema_digest"),
        )

    @property
    def record_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_private_id": self.task_private_id,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "native_schema_digest": self.native_schema_digest,
        }


@dataclass(frozen=True)
class NativeShapeRegistry:
    records: tuple[NativeShapeRecord, ...]
    max_observation_dim: int
    max_action_dim: int
    registry_digest: str | None = None
    schema: str = NATIVE_SHAPE_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NATIVE_SHAPE_REGISTRY_SCHEMA:
            raise CanonicalizationError("unsupported NativeShapeRegistry schema")
        records = tuple(self.records)
        if not records or not all(isinstance(item, NativeShapeRecord) for item in records):
            raise CanonicalizationError("native shape registry requires typed records")
        if len({item.task_private_id for item in records}) != len(records):
            raise CanonicalizationError("native shape registry contains duplicate tasks")
        records = tuple(sorted(records, key=lambda item: item.task_private_id))
        max_observation_dim = _positive_int(
            self.max_observation_dim, "max_observation_dim"
        )
        max_action_dim = _positive_int(self.max_action_dim, "max_action_dim")
        if max_observation_dim < max(item.observation_dim for item in records):
            raise CanonicalizationError("global observation width is smaller than native ABI")
        if max_action_dim < max(item.action_dim for item in records):
            raise CanonicalizationError("global action width is smaller than native ABI")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "max_observation_dim", max_observation_dim)
        object.__setattr__(self, "max_action_dim", max_action_dim)
        expected = sha256_json(self._payload_without_digest())
        if self.registry_digest is None:
            object.__setattr__(self, "registry_digest", expected)
        elif _digest(self.registry_digest, "registry_digest") != expected:
            raise CanonicalizationError("registry_digest does not match native shapes")

    @classmethod
    def from_source_banks(
        cls,
        banks: Sequence[NativeTransitionBank],
        *,
        max_observation_dim: int | None = None,
        max_action_dim: int | None = None,
    ) -> "NativeShapeRegistry":
        values = tuple(banks)
        if not values or not all(isinstance(item, NativeTransitionBank) for item in values):
            raise CanonicalizationError("registry construction requires native banks")
        if any(item.data_role not in SOURCE_FIT_ROLES for item in values):
            raise CanonicalizationError("native shape registry must be frozen from source roles")
        grouped: dict[str, NativeShapeRecord] = {}
        for bank in values:
            record = NativeShapeRecord(
                task_private_id=bank.task_private_id,
                observation_dim=bank.observation_dim,
                action_dim=bank.action_dim,
                native_schema_digest=bank.native_schema_digest,
            )
            previous = grouped.get(bank.task_private_id)
            if previous is not None and previous != record:
                raise CanonicalizationError(
                    f"source banks disagree on native shape for {bank.task_private_id!r}"
                )
            grouped[bank.task_private_id] = record
        observed_observation = max(item.observation_dim for item in grouped.values())
        observed_action = max(item.action_dim for item in grouped.values())
        return cls(
            records=tuple(grouped.values()),
            max_observation_dim=(
                observed_observation
                if max_observation_dim is None
                else max_observation_dim
            ),
            max_action_dim=(
                observed_action if max_action_dim is None else max_action_dim
            ),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "records": [item.to_dict() for item in self.records],
            "max_observation_dim": self.max_observation_dim,
            "max_action_dim": self.max_action_dim,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "registry_digest": self.registry_digest}

    def record_for(self, task_private_id: str) -> NativeShapeRecord:
        matches = [item for item in self.records if item.task_private_id == task_private_id]
        if len(matches) != 1:
            raise CanonicalizationError(
                f"task is absent from frozen native shape registry: {task_private_id!r}"
            )
        return matches[0]

    def validate_bank(self, bank: NativeTransitionBank) -> NativeShapeRecord:
        if not isinstance(bank, NativeTransitionBank):
            raise CanonicalizationError("canonicalizer requires NativeTransitionBank")
        record = self.record_for(bank.task_private_id)
        observed = (
            bank.observation_dim,
            bank.action_dim,
            bank.native_schema_digest,
        )
        expected = (
            record.observation_dim,
            record.action_dim,
            record.native_schema_digest,
        )
        if observed != expected:
            raise CanonicalizationError("native bank shape/schema differs from registry")
        return record


def _readonly_vector(value: Any, *, width: int, where: str) -> np.ndarray:
    result = _readonly_float(value, where=where, ndim=1)
    if result.shape != (width,):
        raise CanonicalizationError(f"{where} must have shape [{width}]")
    return result


@dataclass(frozen=True)
class GlobalNormalizer:
    registry_digest: str
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    reward_mean: float
    reward_std: float
    observation_task_count: np.ndarray
    action_task_count: np.ndarray
    source_bank_digests: tuple[str, ...]
    source_fit_roles: tuple[SourceFitRole, ...]
    std_floor: float = 1.0e-6
    normalizer_digest: str | None = None
    schema: str = GLOBAL_NORMALIZER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GLOBAL_NORMALIZER_SCHEMA:
            raise CanonicalizationError("unsupported GlobalNormalizer schema")
        object.__setattr__(
            self, "registry_digest", _digest(self.registry_digest, "registry_digest")
        )
        observation_mean = _readonly_float(
            self.observation_mean, where="observation_mean", ndim=1
        )
        observation_std = _readonly_vector(
            self.observation_std,
            width=observation_mean.size,
            where="observation_std",
        )
        action_mean = _readonly_float(self.action_mean, where="action_mean", ndim=1)
        action_std = _readonly_vector(
            self.action_std, width=action_mean.size, where="action_std"
        )
        observation_task_count = _readonly_vector(
            self.observation_task_count,
            width=observation_mean.size,
            where="observation_task_count",
        )
        action_task_count = _readonly_vector(
            self.action_task_count,
            width=action_mean.size,
            where="action_task_count",
        )
        if np.any(observation_std <= 0) or np.any(action_std <= 0):
            raise CanonicalizationError("normalizer standard deviations must be positive")
        if np.any(observation_task_count < 0) or np.any(action_task_count < 0):
            raise CanonicalizationError("normalizer task counts cannot be negative")
        for name in ("reward_mean", "reward_std", "std_floor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise CanonicalizationError(f"{name} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise CanonicalizationError(f"{name} must be finite")
            if name != "reward_mean" and number <= 0:
                raise CanonicalizationError(f"{name} must be positive")
            object.__setattr__(self, name, number)
        source_digests = tuple(sorted(_digest(item, "source_bank_digest") for item in self.source_bank_digests))
        if not source_digests or len(set(source_digests)) != len(source_digests):
            raise CanonicalizationError("source_bank_digests must be non-empty and unique")
        roles = tuple(sorted(set(self.source_fit_roles)))
        if not roles or any(role not in SOURCE_FIT_ROLES for role in roles):
            raise CanonicalizationError("normalizer fit roles must be source-only")
        for name, array in (
            ("observation_mean", observation_mean),
            ("observation_std", observation_std),
            ("action_mean", action_mean),
            ("action_std", action_std),
            ("observation_task_count", observation_task_count),
            ("action_task_count", action_task_count),
        ):
            object.__setattr__(self, name, array)
        object.__setattr__(self, "source_bank_digests", source_digests)
        object.__setattr__(self, "source_fit_roles", roles)
        expected = sha256_json(self._payload_without_digest())
        if self.normalizer_digest is None:
            object.__setattr__(self, "normalizer_digest", expected)
        elif _digest(self.normalizer_digest, "normalizer_digest") != expected:
            raise CanonicalizationError("normalizer_digest does not match fitted state")

    @property
    def max_observation_dim(self) -> int:
        return int(self.observation_mean.size)

    @property
    def max_action_dim(self) -> int:
        return int(self.action_mean.size)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "registry_digest": self.registry_digest,
            "statistics_digest": sha256_ndarrays(
                {
                    "observation_mean": self.observation_mean,
                    "observation_std": self.observation_std,
                    "action_mean": self.action_mean,
                    "action_std": self.action_std,
                    "observation_task_count": self.observation_task_count,
                    "action_task_count": self.action_task_count,
                    "reward": np.asarray(
                        [self.reward_mean, self.reward_std], dtype=np.float64
                    ),
                }
            ),
            "source_bank_digests": list(self.source_bank_digests),
            "source_fit_roles": list(self.source_fit_roles),
            "std_floor": self.std_floor,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "normalizer_digest": self.normalizer_digest,
            "max_observation_dim": self.max_observation_dim,
            "max_action_dim": self.max_action_dim,
        }


def _task_balanced_stats(
    grouped: Mapping[str, tuple[np.ndarray, ...]], width: int, std_floor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_sum = np.zeros(width, dtype=np.float64)
    second_sum = np.zeros(width, dtype=np.float64)
    task_count = np.zeros(width, dtype=np.float64)
    for arrays in grouped.values():
        values = np.concatenate(arrays, axis=0).astype(np.float64, copy=False)
        native_width = values.shape[1]
        mean_sum[:native_width] += np.mean(values, axis=0)
        second_sum[:native_width] += np.mean(np.square(values), axis=0)
        task_count[:native_width] += 1.0
    valid = task_count > 0
    mean = np.zeros(width, dtype=np.float64)
    second = np.ones(width, dtype=np.float64)
    mean[valid] = mean_sum[valid] / task_count[valid]
    second[valid] = second_sum[valid] / task_count[valid]
    variance = np.maximum(second - np.square(mean), 0.0)
    std = np.ones(width, dtype=np.float64)
    std[valid] = np.maximum(np.sqrt(variance[valid]), std_floor)
    return mean, std, task_count


def fit_global_normalizer(
    banks: Sequence[NativeTransitionBank],
    registry: NativeShapeRegistry,
    *,
    std_floor: float = 1.0e-6,
) -> GlobalNormalizer:
    """Fit task-balanced statistics from source representation banks only."""

    values = tuple(banks)
    if not values or not all(isinstance(item, NativeTransitionBank) for item in values):
        raise CanonicalizationError("normalizer fit requires native transition banks")
    if not isinstance(registry, NativeShapeRegistry):
        raise CanonicalizationError("normalizer fit requires NativeShapeRegistry")
    if not math.isfinite(float(std_floor)) or float(std_floor) <= 0:
        raise CanonicalizationError("std_floor must be finite and positive")
    if any(item.data_role not in SOURCE_FIT_ROLES for item in values):
        raise CanonicalizationError(
            "global normalizer fit is source-only; query/target banks are forbidden"
        )
    if len({item.native_bank_digest for item in values}) != len(values):
        raise CanonicalizationError("normalizer fit contains duplicate physical banks")
    for bank in values:
        registry.validate_bank(bank)

    observations: dict[str, list[np.ndarray]] = {}
    actions: dict[str, list[np.ndarray]] = {}
    rewards: dict[str, list[np.ndarray]] = {}
    for bank in values:
        observations.setdefault(bank.task_private_id, []).extend(
            [bank.observation, bank.next_observation]
        )
        actions.setdefault(bank.task_private_id, []).append(bank.action)
        rewards.setdefault(bank.task_private_id, []).append(bank.reward[:, None])
    observation_stats = _task_balanced_stats(
        {key: tuple(item) for key, item in observations.items()},
        registry.max_observation_dim,
        float(std_floor),
    )
    action_stats = _task_balanced_stats(
        {key: tuple(item) for key, item in actions.items()},
        registry.max_action_dim,
        float(std_floor),
    )
    reward_task_means = []
    reward_task_seconds = []
    for task in sorted(rewards):
        task_values = np.concatenate(rewards[task], axis=0)[:, 0]
        reward_task_means.append(float(np.mean(task_values)))
        reward_task_seconds.append(float(np.mean(np.square(task_values))))
    reward_mean = float(np.mean(reward_task_means))
    reward_variance = max(float(np.mean(reward_task_seconds)) - reward_mean**2, 0.0)
    reward_std = max(math.sqrt(reward_variance), float(std_floor))
    return GlobalNormalizer(
        registry_digest=str(registry.registry_digest),
        observation_mean=observation_stats[0],
        observation_std=observation_stats[1],
        action_mean=action_stats[0],
        action_std=action_stats[1],
        reward_mean=reward_mean,
        reward_std=reward_std,
        observation_task_count=observation_stats[2],
        action_task_count=action_stats[2],
        source_bank_digests=tuple(item.native_bank_digest for item in values),
        source_fit_roles=tuple(item.data_role for item in values),
        std_floor=float(std_floor),
    )


@dataclass(frozen=True)
class CanonicalizedBankReceipt:
    batch: CanonicalTransitionBatch
    bank_id: str
    task_private_id: str
    data_role: NativeBankRole
    raw_dataset_digest: str
    native_bank_digest: str
    native_shape_record_digest: str
    native_shape_registry_digest: str
    normalizer_digest: str
    canonicalizer_digest: str
    canonical_transition_digest: str
    canonical_observation_dim: int
    canonical_action_dim: int
    receipt_digest: str | None = None
    schema: str = CANONICALIZED_BANK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CANONICALIZED_BANK_RECEIPT_SCHEMA:
            raise CanonicalizationError("unsupported CanonicalizedBankReceipt schema")
        if not isinstance(self.batch, CanonicalTransitionBatch):
            raise CanonicalizationError("receipt requires CanonicalTransitionBatch")
        object.__setattr__(self, "bank_id", _safe_id(self.bank_id, "bank_id"))
        object.__setattr__(
            self,
            "task_private_id",
            _safe_id(self.task_private_id, "task_private_id"),
        )
        if self.data_role not in NATIVE_BANK_ROLES:
            raise CanonicalizationError("receipt has an unknown data role")
        for name in (
            "raw_dataset_digest",
            "native_bank_digest",
            "native_shape_record_digest",
            "native_shape_registry_digest",
            "normalizer_digest",
            "canonicalizer_digest",
            "canonical_transition_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        observation_dim = _positive_int(
            self.canonical_observation_dim, "canonical_observation_dim"
        )
        action_dim = _positive_int(self.canonical_action_dim, "canonical_action_dim")
        if self.batch.observation.shape[1] != observation_dim:
            raise CanonicalizationError("receipt observation width differs from batch")
        if self.batch.action.shape[1] != action_dim:
            raise CanonicalizationError("receipt action width differs from batch")
        if self.batch.transition_digest != self.canonical_transition_digest:
            raise CanonicalizationError(
                "canonical_transition_digest differs from receipt batch"
            )
        object.__setattr__(self, "canonical_observation_dim", observation_dim)
        object.__setattr__(self, "canonical_action_dim", action_dim)
        expected = sha256_json(self._payload_without_digest())
        if self.receipt_digest is None:
            object.__setattr__(self, "receipt_digest", expected)
        elif _digest(self.receipt_digest, "receipt_digest") != expected:
            raise CanonicalizationError("receipt_digest does not match canonicalization")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bank_id": self.bank_id,
            "task_private_id": self.task_private_id,
            "data_role": self.data_role,
            "raw_dataset_digest": self.raw_dataset_digest,
            "native_bank_digest": self.native_bank_digest,
            "native_shape_record_digest": self.native_shape_record_digest,
            "native_shape_registry_digest": self.native_shape_registry_digest,
            "normalizer_digest": self.normalizer_digest,
            "canonicalizer_digest": self.canonicalizer_digest,
            "canonical_transition_digest": self.canonical_transition_digest,
            "canonical_observation_dim": self.canonical_observation_dim,
            "canonical_action_dim": self.canonical_action_dim,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "receipt_digest": self.receipt_digest}


@dataclass(frozen=True)
class GlobalCanonicalizerSpec:
    registry: NativeShapeRegistry
    normalizer: GlobalNormalizer
    padding_value: float = 0.0
    mask_rule: str = "left_prefix_native_true"
    done_rule: str = "BOUNDARY_ONLY"
    mathematical_dtype: str = "float64"
    canonicalizer_digest: str | None = None
    schema: str = GLOBAL_CANONICALIZER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GLOBAL_CANONICALIZER_SCHEMA:
            raise CanonicalizationError("unsupported GlobalCanonicalizerSpec schema")
        if not isinstance(self.registry, NativeShapeRegistry):
            raise CanonicalizationError("canonicalizer requires NativeShapeRegistry")
        if not isinstance(self.normalizer, GlobalNormalizer):
            raise CanonicalizationError("canonicalizer requires GlobalNormalizer")
        if self.normalizer.registry_digest != self.registry.registry_digest:
            raise CanonicalizationError("normalizer belongs to another shape registry")
        if (
            self.normalizer.max_observation_dim != self.registry.max_observation_dim
            or self.normalizer.max_action_dim != self.registry.max_action_dim
        ):
            raise CanonicalizationError("normalizer widths differ from shape registry")
        if not math.isfinite(float(self.padding_value)):
            raise CanonicalizationError("padding_value must be finite")
        object.__setattr__(self, "padding_value", float(self.padding_value))
        if self.mask_rule != "left_prefix_native_true":
            raise CanonicalizationError("unsupported canonical mask rule")
        if self.done_rule != "BOUNDARY_ONLY":
            raise CanonicalizationError("unsupported done rule")
        if self.mathematical_dtype != "float64":
            raise CanonicalizationError("v0.3 canonical mathematical dtype is float64")
        expected = sha256_json(self._payload_without_digest())
        if self.canonicalizer_digest is None:
            object.__setattr__(self, "canonicalizer_digest", expected)
        elif _digest(self.canonicalizer_digest, "canonicalizer_digest") != expected:
            raise CanonicalizationError("canonicalizer_digest does not match protocol")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "native_shape_registry_digest": self.registry.registry_digest,
            "normalizer_digest": self.normalizer.normalizer_digest,
            "max_observation_dim": self.registry.max_observation_dim,
            "max_action_dim": self.registry.max_action_dim,
            "padding_value": self.padding_value,
            "mask_rule": self.mask_rule,
            "done_rule": self.done_rule,
            "mathematical_dtype": self.mathematical_dtype,
            "normalization": "task-balanced-source-only-observation-action-reward",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "canonicalizer_digest": self.canonicalizer_digest,
        }

    def transform(self, bank: NativeTransitionBank) -> CanonicalizedBankReceipt:
        """Transform any registered source/query bank with frozen source stats."""

        record = self.registry.validate_bank(bank)
        observation_dim = record.observation_dim
        action_dim = record.action_dim
        rows = bank.observation.shape[0]
        observation = np.full(
            (rows, self.registry.max_observation_dim),
            self.padding_value,
            dtype=np.float64,
        )
        next_observation = np.full_like(observation, self.padding_value)
        action = np.full(
            (rows, self.registry.max_action_dim),
            self.padding_value,
            dtype=np.float64,
        )
        observation[:, :observation_dim] = (
            bank.observation - self.normalizer.observation_mean[:observation_dim]
        ) / self.normalizer.observation_std[:observation_dim]
        next_observation[:, :observation_dim] = (
            bank.next_observation
            - self.normalizer.observation_mean[:observation_dim]
        ) / self.normalizer.observation_std[:observation_dim]
        action[:, :action_dim] = (
            bank.action - self.normalizer.action_mean[:action_dim]
        ) / self.normalizer.action_std[:action_dim]
        reward = (bank.reward - self.normalizer.reward_mean) / self.normalizer.reward_std
        observation_mask = np.zeros(
            self.registry.max_observation_dim, dtype=np.bool_
        )
        observation_mask[:observation_dim] = True
        action_mask = np.zeros(self.registry.max_action_dim, dtype=np.bool_)
        action_mask[:action_dim] = True
        batch = CanonicalTransitionBatch(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=bank.terminated,
            truncated=bank.truncated,
            observation_mask=observation_mask,
            action_mask=action_mask,
            episode_id=bank.episode_id,
            timestep=bank.timestep,
        )
        return CanonicalizedBankReceipt(
            batch=batch,
            bank_id=bank.bank_id,
            task_private_id=bank.task_private_id,
            data_role=bank.data_role,
            raw_dataset_digest=bank.raw_dataset_digest,
            native_bank_digest=bank.native_bank_digest,
            native_shape_record_digest=record.record_digest,
            native_shape_registry_digest=str(self.registry.registry_digest),
            normalizer_digest=str(self.normalizer.normalizer_digest),
            canonicalizer_digest=str(self.canonicalizer_digest),
            canonical_transition_digest=batch.transition_digest,
            canonical_observation_dim=self.registry.max_observation_dim,
            canonical_action_dim=self.registry.max_action_dim,
        )


def require_formal_cross_task_raw_receipts(
    receipts: Sequence[CanonicalizedBankReceipt],
) -> tuple[CanonicalizedBankReceipt, ...]:
    """Fail closed before a cross-task Raw KME/source-index operation.

    A bare ``CanonicalTransitionBatch`` may be internally well formed while
    having been padded under an incompatible width or normalization protocol;
    it is therefore deliberately rejected here.
    """

    values = tuple(receipts)
    if len(values) < 2 or not all(
        isinstance(item, CanonicalizedBankReceipt) for item in values
    ):
        raise CanonicalizationError(
            "formal cross-task Raw computation requires at least two typed "
            "CanonicalizedBankReceipt values"
        )
    if len({item.task_private_id for item in values}) < 2:
        raise CanonicalizationError("cross-task Raw computation requires distinct tasks")
    bindings = {
        (
            item.native_shape_registry_digest,
            item.normalizer_digest,
            item.canonicalizer_digest,
            item.canonical_observation_dim,
            item.canonical_action_dim,
        )
        for item in values
    }
    if len(bindings) != 1:
        raise CanonicalizationError(
            "cross-task Raw receipts do not share one canonical coordinate system"
        )
    return values


__all__ = [
    "CANONICALIZED_BANK_RECEIPT_SCHEMA",
    "CanonicalizationError",
    "CanonicalizedBankReceipt",
    "GLOBAL_CANONICALIZER_SCHEMA",
    "GLOBAL_NORMALIZER_SCHEMA",
    "GlobalCanonicalizerSpec",
    "GlobalNormalizer",
    "NATIVE_BANK_ROLES",
    "NATIVE_BANK_SCHEMA",
    "NATIVE_SHAPE_RECORD_SCHEMA",
    "NATIVE_SHAPE_REGISTRY_SCHEMA",
    "NativeShapeRecord",
    "NativeShapeRegistry",
    "NativeTransitionBank",
    "SOURCE_FIT_ROLES",
    "fit_global_normalizer",
    "require_formal_cross_task_raw_receipts",
]
