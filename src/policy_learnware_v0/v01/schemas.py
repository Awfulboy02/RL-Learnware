"""Versioned sidecar schemas and opaque identity primitives for v0.1."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import secrets
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json
from ..schemas import EnvSchema


PRIVATE_CONTEXT_SCHEMA = "policy-learnware.v01-private-context.v0"
SHIFT_MANIFEST_SCHEMA = "policy-learnware.v01-shift-manifest.v0"
MEASUREMENT_SCHEMA_VIEW_SCHEMA = "policy-learnware.v01-measurement-schema-view.v0"
ENVIRONMENT_INSTANCE_SCHEMA = "policy-learnware.v01-environment-instance.v0"
VARIANT_DATASET_MANIFEST_SCHEMA = "policy-learnware.v01-variant-dataset-manifest.v0"
ORACLE_EPISODE_SCHEMA = "policy-learnware.v01-oracle-episode.v0"
ORACLE_AGGREGATE_SCHEMA = "policy-learnware.v01-oracle-aggregate.v0"
PROTOCOL_IDENTIFIERS_SCHEMA = "policy-learnware.v01-protocol-identifiers.v0"


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise ValueError(f"{where} must be a SHA-256 hex digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a SHA-256 hex digest") from exc
    return result


def _variant_id(value: Any) -> str:
    result = _nonempty(value, "variant_id")
    if not result.startswith("v01v-") or len(result) != 25:
        raise ValueError("variant_id must be v01v- followed by 20 lowercase hex characters")
    try:
        int(result[5:], 16)
    except ValueError as exc:
        raise ValueError("variant_id suffix must be hexadecimal") from exc
    if result != result.lower():
        raise ValueError("variant_id must be lowercase")
    return result


def _deep_freeze(value: Any) -> Any:
    canonical = canonicalize(value)
    if isinstance(canonical, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in canonical.items()})
    if isinstance(canonical, list):
        return tuple(_deep_freeze(item) for item in canonical)
    return canonical


def _readonly_vector(value: Any, size: int, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{where} must be a finite vector of shape ({size},)")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PrivateContextRecord:
    """Private factor mapping; the nonce never enters measurement artifacts."""

    private_context_id: str
    private_nonce: str
    task: str
    shift_id: str
    factor: float
    d_theta: float
    schema: str = PRIVATE_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRIVATE_CONTEXT_SCHEMA:
            raise ValueError(f"unsupported private context schema: {self.schema!r}")
        context_id = _nonempty(self.private_context_id, "private_context_id")
        if not context_id.startswith("v01c-") or len(context_id) != 37:
            raise ValueError("private_context_id must contain an independent 128-bit token")
        try:
            int(context_id[5:], 16)
        except ValueError as exc:
            raise ValueError("private_context_id suffix must be hexadecimal") from exc
        nonce = _nonempty(self.private_nonce, "private_nonce").lower()
        if len(nonce) != 64:
            raise ValueError("private_nonce must encode 256 random bits")
        try:
            int(nonce, 16)
        except ValueError as exc:
            raise ValueError("private_nonce must be hexadecimal") from exc
        factor = _finite(self.factor, "factor")
        if factor <= 0.0:
            raise ValueError("factor must be positive")
        d_theta = _finite(self.d_theta, "d_theta")
        if d_theta < 0.0:
            raise ValueError("d_theta must be non-negative")
        if not math.isclose(d_theta, abs(math.log(factor)), abs_tol=1e-15):
            raise ValueError("d_theta must equal abs(log(factor)) relative to nominal 1.0")
        object.__setattr__(self, "private_context_id", context_id.lower())
        object.__setattr__(self, "private_nonce", nonce)
        object.__setattr__(self, "task", _nonempty(self.task, "task"))
        object.__setattr__(self, "shift_id", _nonempty(self.shift_id, "shift_id"))
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "d_theta", d_theta)

    @classmethod
    def new(
        cls,
        *,
        task: str,
        shift_id: str,
        factor: float,
        context_token: bytes | None = None,
        nonce_token: bytes | None = None,
    ) -> "PrivateContextRecord":
        """Generate IDs with an OS CSPRNG; token injection exists only for tests."""

        context = secrets.token_bytes(16) if context_token is None else bytes(context_token)
        nonce = secrets.token_bytes(32) if nonce_token is None else bytes(nonce_token)
        if len(context) != 16 or len(nonce) != 32:
            raise ValueError("context_token and nonce_token must contain 16 and 32 bytes")
        factor_value = _finite(factor, "factor")
        return cls(
            private_context_id=f"v01c-{context.hex()}",
            private_nonce=nonce.hex(),
            task=task,
            shift_id=shift_id,
            factor=factor_value,
            d_theta=abs(math.log(factor_value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "private_context_id": self.private_context_id,
            "private_nonce": self.private_nonce,
            "task": self.task,
            "shift_id": self.shift_id,
            "factor": self.factor,
            "d_theta": self.d_theta,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrivateContextRecord":
        fields = {"schema", "private_context_id", "private_nonce", "task", "shift_id", "factor", "d_theta"}
        _strict(value, fields, "PrivateContextRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class ShiftManifest:
    shift_id: str
    factor: float
    registry_digest: str
    base_protocol_id: str
    task: str
    private_context_id: str
    schema: str = SHIFT_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHIFT_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported ShiftManifest schema: {self.schema!r}")
        object.__setattr__(self, "shift_id", _nonempty(self.shift_id, "shift_id"))
        factor = _finite(self.factor, "factor")
        if factor <= 0.0:
            raise ValueError("factor must be positive")
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "registry_digest", _digest(self.registry_digest, "registry_digest"))
        object.__setattr__(self, "base_protocol_id", _digest(self.base_protocol_id, "base_protocol_id"))
        object.__setattr__(self, "task", _nonempty(self.task, "task"))
        # Reuse the strict parser without exposing a nonce.
        context_id = _nonempty(self.private_context_id, "private_context_id").lower()
        if not context_id.startswith("v01c-") or len(context_id) != 37:
            raise ValueError("invalid private_context_id")
        try:
            int(context_id[5:], 16)
        except ValueError as exc:
            raise ValueError("invalid private_context_id") from exc
        object.__setattr__(self, "private_context_id", context_id)

    @classmethod
    def create(
        cls,
        *,
        shift_id: str,
        factor: float,
        registry_digest: str,
        base_protocol_id: str,
        task: str,
        private_context_id: str,
    ) -> "ShiftManifest":
        return cls(
            shift_id=shift_id,
            factor=factor,
            registry_digest=registry_digest,
            base_protocol_id=base_protocol_id,
            task=task,
            private_context_id=private_context_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "shift_id": self.shift_id,
            "factor": self.factor,
            "registry_digest": self.registry_digest,
            "base_protocol_id": self.base_protocol_id,
            "task": self.task,
            "private_context_id": self.private_context_id,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShiftManifest":
        fields = {"schema", "shift_id", "factor", "registry_digest", "base_protocol_id", "task", "private_context_id"}
        _strict(value, fields, "ShiftManifest")
        return cls(**{key: value[key] for key in fields})


def derive_variant_id(
    *, measurement_protocol_id: str, private_nonce: str, shift_manifest_digest: str
) -> str:
    protocol = _digest(measurement_protocol_id, "measurement_protocol_id")
    manifest = _digest(shift_manifest_digest, "shift_manifest_digest")
    nonce = _nonempty(private_nonce, "private_nonce").lower()
    if len(nonce) != 64:
        raise ValueError("private_nonce must encode 256 random bits")
    try:
        int(nonce, 16)
    except ValueError as exc:
        raise ValueError("private_nonce must be hexadecimal") from exc
    suffix = sha256_json(
        {
            "schema": "policy-learnware.v01-variant-id.v0",
            "measurement_protocol_id": protocol,
            "private_nonce": nonce,
            "shift_manifest_digest": manifest,
        }
    )[:20]
    return f"v01v-{suffix}"


@dataclass(frozen=True)
class MeasurementSchemaView:
    observation_dim: int
    action_dim: int
    observation_dtype: str
    action_dtype: str
    action_low: np.ndarray
    action_high: np.ndarray
    horizon: int
    action_repeat: int
    control_dt: float
    flatten_fingerprint_without_task: str
    schema: str = MEASUREMENT_SCHEMA_VIEW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MEASUREMENT_SCHEMA_VIEW_SCHEMA:
            raise ValueError(f"unsupported MeasurementSchemaView schema: {self.schema!r}")
        observation_dim = _nonnegative_int(self.observation_dim, "observation_dim")
        action_dim = _nonnegative_int(self.action_dim, "action_dim")
        if observation_dim == 0 or action_dim == 0:
            raise ValueError("observation_dim and action_dim must be positive")
        horizon = _nonnegative_int(self.horizon, "horizon")
        repeat = _nonnegative_int(self.action_repeat, "action_repeat")
        if horizon == 0 or repeat == 0:
            raise ValueError("horizon and action_repeat must be positive")
        control_dt = _finite(self.control_dt, "control_dt")
        if control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        low = _readonly_vector(self.action_low, action_dim, "action_low")
        high = _readonly_vector(self.action_high, action_dim, "action_high")
        if np.any(low >= high):
            raise ValueError("every action_low component must be less than action_high")
        np.dtype(_nonempty(self.observation_dtype, "observation_dtype"))
        np.dtype(_nonempty(self.action_dtype, "action_dtype"))
        object.__setattr__(self, "action_low", low)
        object.__setattr__(self, "action_high", high)
        object.__setattr__(self, "control_dt", control_dt)
        object.__setattr__(self, "flatten_fingerprint_without_task", _nonempty(
            self.flatten_fingerprint_without_task, "flatten_fingerprint_without_task"
        ))

    @classmethod
    def from_env_schema(cls, env_schema: EnvSchema) -> "MeasurementSchemaView":
        return cls(
            observation_dim=env_schema.observation_dim,
            action_dim=env_schema.action_dim,
            observation_dtype=env_schema.observation_dtype,
            action_dtype=env_schema.action_dtype,
            action_low=env_schema.action_low,
            action_high=env_schema.action_high,
            horizon=env_schema.horizon,
            action_repeat=env_schema.action_repeat,
            control_dt=env_schema.control_dt,
            flatten_fingerprint_without_task=env_schema.flatten_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "observation_dtype": self.observation_dtype,
            "action_dtype": self.action_dtype,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "horizon": self.horizon,
            "action_repeat": self.action_repeat,
            "control_dt": self.control_dt,
            "flatten_fingerprint_without_task": self.flatten_fingerprint_without_task,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def schema_view_id(self) -> str:
        return f"v01s-{self.digest[:20]}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurementSchemaView":
        fields = {
            "schema", "observation_dim", "action_dim", "observation_dtype", "action_dtype",
            "action_low", "action_high", "horizon", "action_repeat", "control_dt",
            "flatten_fingerprint_without_task",
        }
        _strict(value, fields, "MeasurementSchemaView")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class EnvironmentInstanceRecord:
    variant_id: str
    env_schema_digest: str
    base_model_digest: str
    shifted_model_digest: str
    changed_leaf: str
    changed_index_count: int
    before_leaf_digest: str
    after_leaf_digest: str
    operator_digest: str
    runtime_versions: Mapping[str, str]
    finite_termination_audit_summary: Mapping[str, Any] = field(default_factory=dict)
    measurement_schema_view_digest: str | None = None
    shift_manifest_digest: str | None = None
    schema: str = ENVIRONMENT_INSTANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_INSTANCE_SCHEMA:
            raise ValueError(f"unsupported EnvironmentInstanceRecord schema: {self.schema!r}")
        object.__setattr__(self, "variant_id", _variant_id(self.variant_id))
        for name in (
            "env_schema_digest", "base_model_digest", "shifted_model_digest",
            "before_leaf_digest", "after_leaf_digest", "operator_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("measurement_schema_view_digest", "shift_manifest_digest"):
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"{name} is required for a v0.1 environment instance")
            object.__setattr__(self, name, _digest(value, name))
        object.__setattr__(self, "changed_leaf", _nonempty(self.changed_leaf, "changed_leaf"))
        object.__setattr__(self, "changed_index_count", _nonnegative_int(self.changed_index_count, "changed_index_count"))
        if not isinstance(self.runtime_versions, Mapping) or not self.runtime_versions:
            raise ValueError("runtime_versions must be a non-empty mapping")
        if not all(isinstance(key, str) and isinstance(value, str) and value for key, value in self.runtime_versions.items()):
            raise ValueError("runtime_versions must map strings to non-empty strings")
        if not isinstance(self.finite_termination_audit_summary, Mapping):
            raise ValueError("finite_termination_audit_summary must be a mapping")
        object.__setattr__(self, "runtime_versions", MappingProxyType(dict(sorted(self.runtime_versions.items()))))
        object.__setattr__(self, "finite_termination_audit_summary", _deep_freeze(self.finite_termination_audit_summary))

    @classmethod
    def create(cls, **values: Any) -> "EnvironmentInstanceRecord":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema": self.schema,
                "variant_id": self.variant_id,
                "env_schema_digest": self.env_schema_digest,
                "measurement_schema_view_digest": self.measurement_schema_view_digest,
                "shift_manifest_digest": self.shift_manifest_digest,
                "base_model_digest": self.base_model_digest,
                "shifted_model_digest": self.shifted_model_digest,
                "changed_leaf": self.changed_leaf,
                "changed_index_count": self.changed_index_count,
                "before_leaf_digest": self.before_leaf_digest,
                "after_leaf_digest": self.after_leaf_digest,
                "operator_digest": self.operator_digest,
                "runtime_versions": dict(self.runtime_versions),
                "finite_termination_audit_summary": self.finite_termination_audit_summary,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentInstanceRecord":
        fields = {
            "schema", "variant_id", "env_schema_digest", "measurement_schema_view_digest",
            "shift_manifest_digest", "base_model_digest", "shifted_model_digest", "changed_leaf",
            "changed_index_count", "before_leaf_digest", "after_leaf_digest", "operator_digest",
            "runtime_versions", "finite_termination_audit_summary",
        }
        _strict(value, fields, "EnvironmentInstanceRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class VariantDatasetManifest:
    variant_id: str
    bank: int
    episode_count: int
    transition_count: int
    reset_seeds: tuple[int, ...]
    probe_seeds: tuple[int, ...]
    dataset_digest: str
    base_protocol_id: str
    measurement_contract_digest: str
    measurement_schema_view_digest: str
    schema: str = VARIANT_DATASET_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VARIANT_DATASET_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported VariantDatasetManifest schema: {self.schema!r}")
        object.__setattr__(self, "variant_id", _variant_id(self.variant_id))
        for name in ("bank", "episode_count", "transition_count"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if self.episode_count == 0 or self.transition_count == 0:
            raise ValueError("dataset counts must be positive")
        reset = tuple(_nonnegative_int(item, "reset_seeds[]") for item in self.reset_seeds)
        probe = tuple(_nonnegative_int(item, "probe_seeds[]") for item in self.probe_seeds)
        if len(reset) != self.episode_count or len(probe) != self.episode_count:
            raise ValueError("seed arrays must contain one seed per episode")
        if len(set(zip(reset, probe, strict=True))) != self.episode_count:
            raise ValueError("dataset contains duplicate reset/probe seed pairs")
        object.__setattr__(self, "reset_seeds", reset)
        object.__setattr__(self, "probe_seeds", probe)
        for name in ("dataset_digest", "base_protocol_id", "measurement_contract_digest", "measurement_schema_view_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({name: getattr(self, name) for name in self.__dataclass_fields__})

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VariantDatasetManifest":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "VariantDatasetManifest")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class OracleEpisodeRecord:
    task_private: str
    variant_id: str
    candidate_id: str
    episode_index: int
    reset_seed: int
    policy_seed: int
    raw_episodic_sum: float
    mean_step_return: float
    instance_digest: str
    bundle_digest: str
    evaluator_contract_digest: str
    schema: str = ORACLE_EPISODE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_EPISODE_SCHEMA:
            raise ValueError(f"unsupported OracleEpisodeRecord schema: {self.schema!r}")
        object.__setattr__(self, "task_private", _nonempty(self.task_private, "task_private"))
        object.__setattr__(self, "variant_id", _variant_id(self.variant_id))
        object.__setattr__(self, "candidate_id", _nonempty(self.candidate_id, "candidate_id"))
        for name in ("episode_index", "reset_seed", "policy_seed"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        raw = _finite(self.raw_episodic_sum, "raw_episodic_sum")
        mean = _finite(self.mean_step_return, "mean_step_return")
        if not math.isclose(mean, raw / 1000.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("mean_step_return must equal raw_episodic_sum / 1000")
        object.__setattr__(self, "raw_episodic_sum", raw)
        object.__setattr__(self, "mean_step_return", mean)
        for name in ("instance_digest", "bundle_digest", "evaluator_contract_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleEpisodeRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "OracleEpisodeRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class OracleAggregateRecord:
    task_private: str
    variant_id: str
    nominal_variant_id: str
    candidate_id: str
    episode_count: int
    mean_step_return: float
    mean_return_ci_low: float
    mean_return_ci_high: float
    delta_return: float
    delta_ci_low: float
    delta_ci_high: float
    abs_transfer_gap: float
    abs_gap_ci_low: float
    abs_gap_ci_high: float
    schema: str = ORACLE_AGGREGATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_AGGREGATE_SCHEMA:
            raise ValueError(f"unsupported OracleAggregateRecord schema: {self.schema!r}")
        object.__setattr__(self, "task_private", _nonempty(self.task_private, "task_private"))
        object.__setattr__(self, "variant_id", _variant_id(self.variant_id))
        object.__setattr__(self, "nominal_variant_id", _variant_id(self.nominal_variant_id))
        object.__setattr__(self, "candidate_id", _nonempty(self.candidate_id, "candidate_id"))
        count = _nonnegative_int(self.episode_count, "episode_count")
        if count == 0:
            raise ValueError("episode_count must be positive")
        for name in (
            "mean_step_return", "mean_return_ci_low", "mean_return_ci_high", "delta_return",
            "delta_ci_low", "delta_ci_high", "abs_transfer_gap", "abs_gap_ci_low", "abs_gap_ci_high",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.mean_return_ci_low > self.mean_return_ci_high or self.delta_ci_low > self.delta_ci_high or self.abs_gap_ci_low > self.abs_gap_ci_high:
            raise ValueError("aggregate confidence interval bounds are reversed")
        if not math.isclose(self.abs_transfer_gap, abs(self.delta_return), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("abs_transfer_gap must equal abs(delta_return)")
        if self.abs_gap_ci_low < 0.0:
            raise ValueError("abs-gap confidence interval cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleAggregateRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "OracleAggregateRecord")
        return cls(**{key: value[key] for key in fields})


def derive_measurement_protocol_id(
    *, config_projection: Mapping[str, Any], registry_digest: str,
    component_digests: Mapping[str, str],
) -> str:
    components = {name: _digest(value, f"component_digests[{name!r}]") for name, value in component_digests.items()}
    if not components:
        raise ValueError("measurement component_digests cannot be empty")
    return sha256_json({
        "schema": "policy-learnware.v01-measurement-protocol.v0",
        "config": config_projection,
        "registry_digest": _digest(registry_digest, "registry_digest"),
        "component_digests": components,
    })


def derive_oracle_protocol_id(
    *, config_projection: Mapping[str, Any], registry_digest: str,
    component_digests: Mapping[str, str],
) -> str:
    components = {name: _digest(value, f"component_digests[{name!r}]") for name, value in component_digests.items()}
    if not components:
        raise ValueError("oracle component_digests cannot be empty")
    return sha256_json({
        "schema": "policy-learnware.v01-oracle-protocol.v0",
        "config": config_projection,
        "registry_digest": _digest(registry_digest, "registry_digest"),
        "component_digests": components,
    })


def derive_measurement_run_id(
    *, measurement_protocol_id: str, variant_schema_view_digests: Mapping[str, str],
    pair_plan_digest: str,
) -> str:
    variants = {
        _variant_id(name): _digest(value, f"variant_schema_view_digests[{name!r}]")
        for name, value in variant_schema_view_digests.items()
    }
    if not variants:
        raise ValueError("variant_schema_view_digests cannot be empty")
    return sha256_json({
        "schema": "policy-learnware.v01-measurement-run.v0",
        "measurement_protocol_id": _digest(measurement_protocol_id, "measurement_protocol_id"),
        "variant_schema_view_digests": variants,
        "pair_plan_digest": _digest(pair_plan_digest, "pair_plan_digest"),
    })


def derive_experiment_protocol_id(
    *, measurement_run_id: str, oracle_protocol_id: str,
    analysis_projection: Mapping[str, Any], component_digests: Mapping[str, str],
) -> str:
    components = {name: _digest(value, f"component_digests[{name!r}]") for name, value in component_digests.items()}
    if not components:
        raise ValueError("experiment component_digests cannot be empty")
    return sha256_json({
        "schema": "policy-learnware.v01-experiment-protocol.v0",
        "measurement_run_id": _digest(measurement_run_id, "measurement_run_id"),
        "oracle_protocol_id": _digest(oracle_protocol_id, "oracle_protocol_id"),
        "analysis": analysis_projection,
        "component_digests": components,
    })


@dataclass(frozen=True)
class ProtocolIdentifiers:
    measurement_protocol_id: str
    oracle_protocol_id: str
    measurement_run_id: str
    experiment_protocol_id: str
    schema: str = PROTOCOL_IDENTIFIERS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROTOCOL_IDENTIFIERS_SCHEMA:
            raise ValueError(f"unsupported ProtocolIdentifiers schema: {self.schema!r}")
        for name in (
            "measurement_protocol_id", "oracle_protocol_id", "measurement_run_id",
            "experiment_protocol_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtocolIdentifiers":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "ProtocolIdentifiers")
        return cls(**{key: value[key] for key in fields})


__all__ = [
    "ENVIRONMENT_INSTANCE_SCHEMA", "EnvironmentInstanceRecord", "MEASUREMENT_SCHEMA_VIEW_SCHEMA",
    "MeasurementSchemaView", "ORACLE_AGGREGATE_SCHEMA", "ORACLE_EPISODE_SCHEMA",
    "OracleAggregateRecord", "OracleEpisodeRecord", "PRIVATE_CONTEXT_SCHEMA",
    "PROTOCOL_IDENTIFIERS_SCHEMA", "PrivateContextRecord", "ProtocolIdentifiers",
    "SHIFT_MANIFEST_SCHEMA", "ShiftManifest", "VARIANT_DATASET_MANIFEST_SCHEMA",
    "VariantDatasetManifest", "derive_experiment_protocol_id", "derive_measurement_protocol_id",
    "derive_measurement_run_id", "derive_oracle_protocol_id", "derive_variant_id",
]
