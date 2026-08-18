"""Versioned, NumPy-native core schemas shared across the v0 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .hashing import canonicalize, sha256_json


ENV_SCHEMA_VERSION = "policy-learnware.env-schema.v0"
FROZEN_PROTOCOL_SCHEMA = "policy-learnware.protocol.v0"
REQUIRED_PROTOCOL_COMPONENTS = frozenset(
    {
        "environment_manifest",
        "probe_implementation",
        "normalization",
        "encoder",
        "kernel",
        "source_dataset_manifests",
    }
)
REQUIRED_PACKED_LAYOUT_FIELDS = frozenset(
    {
        "width",
        "max_observation_dim",
        "max_action_dim",
        "latent_dim",
        "support_budget",
        "kernel_bandwidth",
        "layout_version",
    }
)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _readonly_vector(
    value: np.ndarray | list[float],
    *,
    size: int,
    name: str,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EnvSchema:
    backend: str
    task: str
    observation_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    horizon: int
    action_repeat: int
    control_dt: float
    flatten_fingerprint: str
    implementation_digest: str
    observation_dtype: str = "float32"
    action_dtype: str = "float32"
    schema: str = ENV_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("backend", "task", "flatten_fingerprint", "implementation_digest"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"EnvSchema.{name} must be a non-empty string")
        if self.schema != ENV_SCHEMA_VERSION:
            raise ValueError(f"unsupported EnvSchema version: {self.schema!r}")
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        if self.horizon <= 0 or self.action_repeat <= 0:
            raise ValueError("horizon and action_repeat must be positive")
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        low = _readonly_vector(
            self.action_low, size=self.action_dim, name="action_low"
        )
        high = _readonly_vector(
            self.action_high, size=self.action_dim, name="action_high"
        )
        if np.any(low >= high):
            raise ValueError("every action_low component must be below action_high")
        object.__setattr__(self, "action_low", low)
        object.__setattr__(self, "action_high", high)
        # Validate that declared dtype strings are understood by NumPy.
        np.dtype(self.observation_dtype)
        np.dtype(self.action_dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "backend": self.backend,
            "task": self.task,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "horizon": self.horizon,
            "action_repeat": self.action_repeat,
            "control_dt": self.control_dt,
            "flatten_fingerprint": self.flatten_fingerprint,
            "implementation_digest": self.implementation_digest,
            "observation_dtype": self.observation_dtype,
            "action_dtype": self.action_dtype,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvSchema":
        expected = {
            "schema",
            "backend",
            "task",
            "observation_dim",
            "action_dim",
            "action_low",
            "action_high",
            "horizon",
            "action_repeat",
            "control_dt",
            "flatten_fingerprint",
            "implementation_digest",
            "observation_dtype",
            "action_dtype",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                f"invalid EnvSchema keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            schema=str(value["schema"]),
            backend=str(value["backend"]),
            task=str(value["task"]),
            observation_dim=int(value["observation_dim"]),
            action_dim=int(value["action_dim"]),
            action_low=np.asarray(value["action_low"], dtype=np.float32),
            action_high=np.asarray(value["action_high"], dtype=np.float32),
            horizon=int(value["horizon"]),
            action_repeat=int(value["action_repeat"]),
            control_dt=float(value["control_dt"]),
            flatten_fingerprint=str(value["flatten_fingerprint"]),
            implementation_digest=str(value["implementation_digest"]),
            observation_dtype=str(value["observation_dtype"]),
            action_dtype=str(value["action_dtype"]),
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class StepResult:
    """One native environment transition, before canonicalization."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float32)
        if observation.ndim != 1:
            raise ValueError("StepResult.observation must be a flat vector")
        if not np.all(np.isfinite(observation)):
            raise ValueError("StepResult.observation contains non-finite values")
        if not np.isfinite(self.reward):
            raise ValueError("StepResult.reward must be finite")
        object.__setattr__(self, "observation", np.array(observation, copy=True))
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "info", dict(self.info))


@dataclass(frozen=True)
class FrozenProtocol:
    """Hash-addressed manifest that binds every TaskSpec-visible component."""

    schema: str
    protocol_id: str
    config: Mapping[str, Any]
    env_schemas: Mapping[str, EnvSchema]
    packed_layout: Mapping[str, Any]
    component_digests: Mapping[str, str]
    runtime_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.schema != FROZEN_PROTOCOL_SCHEMA:
            raise ValueError(f"unsupported FrozenProtocol version: {self.schema!r}")
        if set(self.env_schemas) != {
            schema.task for schema in self.env_schemas.values()
        }:
            raise ValueError("env_schemas keys must exactly match EnvSchema.task")
        missing_components = REQUIRED_PROTOCOL_COMPONENTS - set(self.component_digests)
        if missing_components:
            raise ValueError(
                f"FrozenProtocol misses required component digests: {sorted(missing_components)}"
            )
        missing_layout = REQUIRED_PACKED_LAYOUT_FIELDS - set(self.packed_layout)
        if missing_layout:
            raise ValueError(
                f"FrozenProtocol misses packed layout fields: {sorted(missing_layout)}"
            )
        if int(self.packed_layout["width"]) != 109:
            raise ValueError("v0 packed layout width must be 109")
        for name, digest in self.component_digests.items():
            if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("component_digests must map names to SHA-256 hex strings")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ValueError(f"invalid SHA-256 digest for component {name!r}") from exc
        expected_id = sha256_json(self._payload_without_id())
        if self.protocol_id != expected_id:
            raise ValueError(
                f"protocol_id mismatch: expected {expected_id}, got {self.protocol_id}"
            )
        object.__setattr__(self, "config", _deep_freeze(canonicalize(self.config)))
        object.__setattr__(
            self, "env_schemas", MappingProxyType(dict(self.env_schemas))
        )
        object.__setattr__(
            self, "packed_layout", _deep_freeze(canonicalize(self.packed_layout))
        )
        object.__setattr__(
            self, "component_digests", MappingProxyType(dict(self.component_digests))
        )
        object.__setattr__(
            self, "runtime_versions", MappingProxyType(dict(self.runtime_versions))
        )

    @classmethod
    def create(
        cls,
        *,
        config: Mapping[str, Any],
        env_schemas: Mapping[str, EnvSchema],
        packed_layout: Mapping[str, Any],
        component_digests: Mapping[str, str],
        runtime_versions: Mapping[str, str],
    ) -> "FrozenProtocol":
        payload = {
            "schema": FROZEN_PROTOCOL_SCHEMA,
            "config": canonicalize(config),
            "env_schemas": {
                task: schema.to_dict() for task, schema in sorted(env_schemas.items())
            },
            "packed_layout": canonicalize(packed_layout),
            "component_digests": dict(component_digests),
            "runtime_versions": dict(runtime_versions),
        }
        return cls(
            schema=FROZEN_PROTOCOL_SCHEMA,
            protocol_id=sha256_json(payload),
            config=canonicalize(config),
            env_schemas=dict(env_schemas),
            packed_layout=canonicalize(packed_layout),
            component_digests=dict(component_digests),
            runtime_versions=dict(runtime_versions),
        )

    def _payload_without_id(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "config": canonicalize(self.config),
            "env_schemas": {
                task: schema.to_dict()
                for task, schema in sorted(self.env_schemas.items())
            },
            "packed_layout": canonicalize(self.packed_layout),
            "component_digests": dict(self.component_digests),
            "runtime_versions": dict(self.runtime_versions),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"protocol_id": self.protocol_id, **self._payload_without_id()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenProtocol":
        expected = {
            "schema",
            "protocol_id",
            "config",
            "env_schemas",
            "packed_layout",
            "component_digests",
            "runtime_versions",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "invalid FrozenProtocol keys; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        raw_schemas = value["env_schemas"]
        if not isinstance(raw_schemas, Mapping):
            raise ValueError("env_schemas must be a mapping")
        return cls(
            schema=str(value["schema"]),
            protocol_id=str(value["protocol_id"]),
            config=dict(value["config"]),
            env_schemas={
                str(task): EnvSchema.from_dict(schema)
                for task, schema in raw_schemas.items()
            },
            packed_layout=dict(value["packed_layout"]),
            component_digests={
                str(name): str(digest)
                for name, digest in value["component_digests"].items()
            },
            runtime_versions={
                str(name): str(version)
                for name, version in value["runtime_versions"].items()
            },
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> str:
        from .io import atomic_write_json

        return atomic_write_json(path, self.to_dict(), overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenProtocol":
        from .io import read_json

        return cls.from_dict(read_json(path))
