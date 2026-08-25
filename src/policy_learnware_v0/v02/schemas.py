"""Immutable v0.2 records and independently scoped protocol identities.

The v0.2 sidecar deliberately does not alter v0/v0.1 schemas.  Records in this
module are hash addressed, copy NumPy inputs, and fail closed when a persisted
digest no longer matches its canonical payload.

The draft ``v02-environment-instance.v0`` material is intentionally retired:
it omitted the concrete registry/model/operator fields needed by the training
server.  ``v1`` below is the sole package/server projection and is guarded by
cross-tree golden vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json


ENVIRONMENT_INSTANCE_SCHEMA = "policy-learnware.v02-environment-instance.v1"
MODEL_DIFF_SCHEMA = "policy-learnware.v02-live-model-diff.v0"
MODEL_DIFF_SUPPORTED_LEAVES = frozenset(
    {
        "_mjx_model.body_mass",
        "_mjx_model.body_inertia",
        "_mjx_model.dof_damping",
        "_mjx_model.actuator_gainprm",
        "_mjx_model.actuator_gear",
        "_mjx_model.geom_friction",
    }
)
AXIS_ANCHOR_BINDING_SCHEMA = "policy-learnware.v02-axis-anchor-binding.v0"
SOURCE_ANCHOR_SCHEMA = "policy-learnware.v02-source-anchor.v0"
TARGET_CONTEXT_SCHEMA = "policy-learnware.v02-target-context.v0"
RUNTIME_CONTRACT_SCHEMA = "policy-learnware.v02-runtime-contract.v0"
EXECUTION_ABI_SCHEMA = "policy-learnware.v02-execution-abi.v0"
SOURCE_COMPETENCE_SCHEMA = "policy-learnware.v02-source-competence.v0"
ENVIRONMENT_SPEC_SCHEMA = "policy-learnware.v02-environment-spec.v0"
PUBLIC_MARKET_ENTRY_SCHEMA = "policy-learnware.v02-public-market-entry.v0"
PROTOCOL_IDENTIFIERS_SCHEMA = "policy-learnware.v02-protocol-identifiers.v0"

PROTOCOL_KINDS = frozenset(
    {
        "benchmark",
        "training",
        "policy_market",
        "probe",
        "representation",
        "representation_index",
        "selector",
        "evaluation",
    }
)
TARGET_REGIMES = frozenset(
    {
        "safety_exact",
        "heldout_interpolation",
        "heldout_extrapolation",
        "market_ood_boundary",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_CANONICAL_ENVIRONMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OPAQUE_TARGET_ID = re.compile(r"^v02q-[0-9a-f]{32}$")


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _nonempty(value: Any, where: str, *, safe: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    if safe and not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{where} is not a safe identifier")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise ValueError(f"{where} must be a SHA-256 hex digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a SHA-256 hex digest") from exc
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _deep_freeze(value: Any) -> Any:
    canonical = canonicalize(value)
    if isinstance(canonical, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in canonical.items()})
    if isinstance(canonical, list):
        return tuple(_deep_freeze(item) for item in canonical)
    return canonical


def _readonly_array(value: Any, *, ndim: int, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or any(size <= 0 for size in array.shape):
        raise ValueError(f"{where} must be a non-empty {ndim}-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{where} must be finite")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _checked_components(value: Mapping[str, str], where: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{where} must be a non-empty mapping")
    result: dict[str, str] = {}
    for name, digest in value.items():
        key = _nonempty(name, f"{where} key", safe=True)
        result[key] = _digest(digest, f"{where}[{key!r}]")
    return dict(sorted(result.items()))


def _host_model_array(value: Any) -> np.ndarray:
    try:
        import jax
    except ImportError:
        host = value
    else:
        host = jax.device_get(value)
    return np.ascontiguousarray(np.asarray(host))


def _changed_flat_indices(left: np.ndarray, right: np.ndarray) -> list[int]:
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError("model leaf shape/dtype changed")
    left_bytes = left.reshape(-1).view(np.uint8).reshape(left.size, left.dtype.itemsize)
    right_bytes = right.reshape(-1).view(np.uint8).reshape(right.size, right.dtype.itemsize)
    return np.flatnonzero(np.any(left_bytes != right_bytes, axis=1)).tolist()


def canonical_array_digest(value: Any) -> str:
    """Digest one numerical model leaf using the server's byte projection."""

    array = _host_model_array(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise ValueError("model leaves must have a numerical dtype")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError("model leaves must be finite")
    return sha256_json(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


def _model_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _path_token(key: Any) -> str:
    for attribute in ("name", "key", "idx"):
        if hasattr(key, attribute):
            return str(getattr(key, attribute))
    return str(key)


def _generic_model_leaves(
    value: Any, prefix: tuple[str, ...] = ()
) -> list[tuple[str, Any]]:
    if isinstance(value, (np.ndarray, np.generic, bool, int, float)):
        return [(".".join(prefix) or "<root>", value)]
    if is_dataclass(value) and not isinstance(value, type):
        result: list[tuple[str, Any]] = []
        for item in fields(value):
            result.extend(
                _generic_model_leaves(getattr(value, item.name), prefix + (item.name,))
            )
        return result
    if isinstance(value, Mapping):
        result = []
        for key in sorted(value, key=str):
            result.extend(_generic_model_leaves(value[key], prefix + (str(key),)))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for index, item in enumerate(value):
            result.extend(_generic_model_leaves(item, prefix + (str(index),)))
        return result
    if hasattr(value, "__dict__"):
        result = []
        for name in sorted(vars(value)):
            if not name.startswith("_"):
                result.extend(
                    _generic_model_leaves(getattr(value, name), prefix + (name,))
                )
        if result:
            return result
    raise TypeError(f"cannot flatten model value of type {_model_type(value)}")


def _live_model_leaves(model: Any) -> list[tuple[str, Any]]:
    if type(model).__module__.startswith("mujoco.mjx"):
        try:
            import jax
        except ImportError as exc:  # pragma: no cover - production dependency gate
            raise ValueError("MJX model auditing requires JAX") from exc
        rows, _ = jax.tree_util.tree_flatten_with_path(model)
        return [
            (".".join(_path_token(item) for item in path), value)
            for path, value in rows
        ]
    return _generic_model_leaves(model)


def canonical_model_snapshot(model: Any) -> dict[str, Any]:
    """Return the package/server shared complete model snapshot projection."""

    records: list[dict[str, Any]] = []
    for path, value in _live_model_leaves(model):
        array = _host_model_array(value)
        records.append(
            {
                "path": path,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "digest": canonical_array_digest(array),
            }
        )
    records.sort(key=lambda item: item["path"])
    if not records or len({item["path"] for item in records}) != len(records):
        raise ValueError("model snapshot paths must be non-empty and unique")
    material = {"model_type": _model_type(model), "leaves": records}
    return {**material, "digest": sha256_json(material)}


def canonical_model_diff_projection(
    *,
    nominal_model_digest: str,
    bound_model_digest: str,
    changes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize the sole live model-diff material hashed by package/server."""

    nominal = _digest(nominal_model_digest, "nominal_model_digest")
    bound = _digest(bound_model_digest, "bound_model_digest")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(changes):
        _strict(
            raw,
            {"leaf", "before_digest", "after_digest", "changed_flat_indices"},
            f"model diff change[{index}]",
        )
        leaf = _nonempty(raw["leaf"], f"model diff change[{index}].leaf", safe=True)
        if leaf not in MODEL_DIFF_SUPPORTED_LEAVES:
            raise ValueError(f"model diff contains an unallowlisted leaf: {leaf!r}")
        before = _digest(
            raw["before_digest"], f"model diff change[{index}].before_digest"
        )
        after = _digest(
            raw["after_digest"], f"model diff change[{index}].after_digest"
        )
        indices = raw["changed_flat_indices"]
        if (
            not isinstance(indices, (list, tuple))
            or not indices
            or any(isinstance(item, bool) or not isinstance(item, int) for item in indices)
            or list(indices) != sorted(set(indices))
            or indices[0] < 0
        ):
            raise ValueError("changed_flat_indices must be sorted unique nonnegative integers")
        if before == after:
            raise ValueError("a model diff change must alter its leaf digest")
        normalized.append(
            {
                "leaf": leaf,
                "before_digest": before,
                "after_digest": after,
                "changed_flat_indices": list(indices),
            }
        )
    if [item["leaf"] for item in normalized] != sorted(
        {item["leaf"] for item in normalized}
    ):
        raise ValueError("model diff changes must have unique leaves in sorted order")
    if bool(normalized) != (nominal != bound):
        raise ValueError("model digest equality and model diff changes disagree")
    return {
        "schema": MODEL_DIFF_SCHEMA,
        "nominal_model_digest": nominal,
        "bound_model_digest": bound,
        "changes": normalized,
    }


def derive_live_model_diff(model: Any, bound_model: Any) -> tuple[dict[str, Any], str]:
    """Compute the canonical projection/digest from two live model objects."""

    nominal = canonical_model_snapshot(model)
    bound = canonical_model_snapshot(bound_model)
    if nominal["model_type"] != bound["model_type"]:
        raise ValueError("model type changed during dynamics binding")
    nominal_rows = {item["path"]: item for item in nominal["leaves"]}
    bound_rows = {item["path"]: item for item in bound["leaves"]}
    if set(nominal_rows) != set(bound_rows):
        raise ValueError("model pytree structure changed during dynamics binding")
    nominal_values = dict(_live_model_leaves(model))
    bound_values = dict(_live_model_leaves(bound_model))
    changes: list[dict[str, Any]] = []
    for path in sorted(nominal_rows):
        left = _host_model_array(nominal_values[path])
        right = _host_model_array(bound_values[path])
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"model leaf shape/dtype changed: {path}")
        indices = _changed_flat_indices(left, right)
        if indices:
            changes.append(
                {
                    "leaf": f"_mjx_model.{path}",
                    "before_digest": nominal_rows[path]["digest"],
                    "after_digest": bound_rows[path]["digest"],
                    "changed_flat_indices": indices,
                }
            )
    projection = canonical_model_diff_projection(
        nominal_model_digest=nominal["digest"],
        bound_model_digest=bound["digest"],
        changes=changes,
    )
    return projection, sha256_json(projection)


_ENVIRONMENT_INSTANCE_FIELDS = {
    "task",
    "backend",
    "nominal",
    "factor",
    "environment_class",
    "registry_config_digest",
    "runtime_digest",
    "expected_nominal_model_digest",
    "expected_bound_model_digest",
    "operator_digest",
    "axis_binding_digest",
    "model_diff_digest",
}


def canonical_environment_instance_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sole package/server environment-instance identity material."""

    _strict(value, _ENVIRONMENT_INSTANCE_FIELDS, "environment instance material")
    task = _nonempty(value["task"], "environment instance.task")
    backend = _nonempty(value["backend"], "environment instance.backend")
    if not _CANONICAL_ENVIRONMENT_ID.fullmatch(task):
        raise ValueError("environment instance.task is not a canonical safe identifier")
    if not _CANONICAL_ENVIRONMENT_ID.fullmatch(backend):
        raise ValueError("environment instance.backend is not a canonical safe identifier")
    if not isinstance(value["nominal"], bool):
        raise ValueError("environment instance.nominal must be boolean")
    factor = _finite(value["factor"], "environment instance.factor")
    if factor <= 0.0:
        raise ValueError("environment instance.factor must be positive")
    environment_class = _nonempty(
        value["environment_class"], "environment instance.environment_class"
    )
    if "." not in environment_class:
        raise ValueError("environment instance.environment_class must be fully qualified")
    operator = value["operator_digest"]
    binding = value["axis_binding_digest"]
    operator = None if operator is None else _digest(operator, "operator_digest")
    binding = None if binding is None else _digest(binding, "axis_binding_digest")
    nominal_model = _digest(
        value["expected_nominal_model_digest"], "expected_nominal_model_digest"
    )
    bound_model = _digest(
        value["expected_bound_model_digest"], "expected_bound_model_digest"
    )
    model_diff = _digest(value["model_diff_digest"], "model_diff_digest")
    if value["nominal"]:
        if factor != 1.0 or operator is not None or binding is not None:
            raise ValueError("nominal environment instance must be factor-one and unbound")
        if nominal_model != bound_model:
            raise ValueError("nominal environment instance model digests differ")
        expected_model_diff = sha256_json(
            canonical_model_diff_projection(
                nominal_model_digest=nominal_model,
                bound_model_digest=bound_model,
                changes=(),
            )
        )
        if model_diff != expected_model_diff:
            raise ValueError("nominal environment instance model_diff_digest mismatch")
    else:
        if factor == 1.0 or operator is None or binding is None:
            raise ValueError("shifted environment instance must bind operator and axis")
        if nominal_model == bound_model:
            raise ValueError("shifted environment instance did not change the model")
    return {
        "schema": ENVIRONMENT_INSTANCE_SCHEMA,
        "task": task,
        "backend": backend,
        "nominal": value["nominal"],
        "factor": factor,
        "environment_class": environment_class,
        "registry_config_digest": _digest(
            value["registry_config_digest"], "registry_config_digest"
        ),
        "runtime_digest": _digest(value["runtime_digest"], "runtime_digest"),
        "expected_nominal_model_digest": nominal_model,
        "expected_bound_model_digest": bound_model,
        "operator_digest": operator,
        "axis_binding_digest": binding,
        "model_diff_digest": model_diff,
    }


@dataclass(frozen=True)
class EnvironmentInstanceRecord:
    environment_instance_digest: str
    task: str
    backend: str
    nominal: bool
    factor: float
    environment_class: str
    registry_config_digest: str
    runtime_digest: str
    expected_nominal_model_digest: str
    expected_bound_model_digest: str
    operator_digest: str | None
    axis_binding_digest: str | None
    model_diff_digest: str
    schema: str = ENVIRONMENT_INSTANCE_SCHEMA

    @staticmethod
    def _payload(**values: Any) -> dict[str, Any]:
        return canonical_environment_instance_projection(values)

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_INSTANCE_SCHEMA:
            raise ValueError(f"unsupported EnvironmentInstanceRecord schema: {self.schema!r}")
        material = {name: getattr(self, name) for name in _ENVIRONMENT_INSTANCE_FIELDS}
        payload = self._payload(**material)
        expected = sha256_json(payload)
        actual = _digest(self.environment_instance_digest, "environment_instance_digest")
        if actual != expected:
            raise ValueError("environment_instance_digest does not match canonical payload")
        for name in _ENVIRONMENT_INSTANCE_FIELDS:
            object.__setattr__(self, name, payload[name])
        object.__setattr__(self, "environment_instance_digest", actual)

    @classmethod
    def create(cls, **values: Any) -> "EnvironmentInstanceRecord":
        payload = cls._payload(**values)
        return cls(
            environment_instance_digest=sha256_json(payload),
            **{name: payload[name] for name in _ENVIRONMENT_INSTANCE_FIELDS},
        )

    def to_dict(self) -> dict[str, Any]:
        material = {name: getattr(self, name) for name in _ENVIRONMENT_INSTANCE_FIELDS}
        return {
            "environment_instance_digest": self.environment_instance_digest,
            **self._payload(**material),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentInstanceRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "EnvironmentInstanceRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class AxisAnchorBinding:
    axis_binding_digest: str
    axis_id: str
    factor_id: str
    operator_digest: str
    model_diff_digest: str
    schema: str = AXIS_ANCHOR_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AXIS_ANCHOR_BINDING_SCHEMA:
            raise ValueError(f"unsupported AxisAnchorBinding schema: {self.schema!r}")
        axis_id = _nonempty(self.axis_id, "axis_id", safe=True)
        factor_id = _nonempty(self.factor_id, "factor_id", safe=True)
        operator = _digest(self.operator_digest, "operator_digest")
        model_diff = _digest(self.model_diff_digest, "model_diff_digest")
        payload = {
            "schema": self.schema,
            "axis_id": axis_id,
            "factor_id": factor_id,
            "operator_digest": operator,
            "model_diff_digest": model_diff,
        }
        actual = _digest(self.axis_binding_digest, "axis_binding_digest")
        if actual != sha256_json(payload):
            raise ValueError("axis_binding_digest does not match canonical payload")
        object.__setattr__(self, "axis_id", axis_id)
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "operator_digest", operator)
        object.__setattr__(self, "model_diff_digest", model_diff)
        object.__setattr__(self, "axis_binding_digest", actual)

    @classmethod
    def create(
        cls, *, axis_id: str, factor_id: str, operator_digest: str,
        model_diff_digest: str,
    ) -> "AxisAnchorBinding":
        payload = {
            "schema": AXIS_ANCHOR_BINDING_SCHEMA,
            "axis_id": _nonempty(axis_id, "axis_id", safe=True),
            "factor_id": _nonempty(factor_id, "factor_id", safe=True),
            "operator_digest": _digest(operator_digest, "operator_digest"),
            "model_diff_digest": _digest(model_diff_digest, "model_diff_digest"),
        }
        return cls(axis_binding_digest=sha256_json(payload), **{
            key: payload[key]
            for key in ("axis_id", "factor_id", "operator_digest", "model_diff_digest")
        })

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AxisAnchorBinding":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "AxisAnchorBinding")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class SourceAnchorRecord:
    anchor_id: str
    environment_instance_digest: str
    axis_binding_digest: str | None
    split_role: str = "source"
    schema: str = SOURCE_ANCHOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_ANCHOR_SCHEMA:
            raise ValueError(f"unsupported SourceAnchorRecord schema: {self.schema!r}")
        if self.split_role != "source":
            raise ValueError("SourceAnchorRecord.split_role must be 'source'")
        environment = _digest(self.environment_instance_digest, "environment_instance_digest")
        binding = None
        if self.axis_binding_digest is not None:
            binding = _digest(self.axis_binding_digest, "axis_binding_digest")
        payload = {
            "schema": self.schema,
            "environment_instance_digest": environment,
            "axis_binding_digest": binding,
            "split_role": "source",
        }
        anchor_id = _digest(self.anchor_id, "anchor_id")
        if anchor_id != sha256_json(payload):
            raise ValueError("anchor_id does not match canonical source-anchor payload")
        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "environment_instance_digest", environment)
        object.__setattr__(self, "axis_binding_digest", binding)

    @classmethod
    def create(
        cls, *, environment_instance_digest: str, axis_binding_digest: str | None,
    ) -> "SourceAnchorRecord":
        environment = _digest(environment_instance_digest, "environment_instance_digest")
        binding = (
            None if axis_binding_digest is None
            else _digest(axis_binding_digest, "axis_binding_digest")
        )
        payload = {
            "schema": SOURCE_ANCHOR_SCHEMA,
            "environment_instance_digest": environment,
            "axis_binding_digest": binding,
            "split_role": "source",
        }
        return cls(
            anchor_id=sha256_json(payload),
            environment_instance_digest=environment,
            axis_binding_digest=binding,
        )

    @property
    def is_nominal(self) -> bool:
        return self.axis_binding_digest is None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceAnchorRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceAnchorRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class TargetContextRecord:
    opaque_target_id: str
    task_contract_id: str
    regime: str
    source_anchor_ref: str | None
    private_environment_instance_digest: str
    split_manifest_digest: str
    schema: str = TARGET_CONTEXT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TARGET_CONTEXT_SCHEMA:
            raise ValueError(f"unsupported TargetContextRecord schema: {self.schema!r}")
        target_id = _nonempty(self.opaque_target_id, "opaque_target_id")
        if not _OPAQUE_TARGET_ID.fullmatch(target_id):
            raise ValueError("opaque_target_id must be v02q- followed by 128-bit lowercase hex")
        if self.regime not in TARGET_REGIMES:
            raise ValueError(f"unsupported target regime: {self.regime!r}")
        source_ref = self.source_anchor_ref
        if self.regime == "safety_exact":
            if source_ref is None:
                raise ValueError("safety_exact target contexts require source_anchor_ref")
            source_ref = _digest(source_ref, "source_anchor_ref")
        elif source_ref is not None:
            raise ValueError("held-out target contexts cannot reference a source anchor")
        object.__setattr__(self, "opaque_target_id", target_id)
        object.__setattr__(self, "task_contract_id", _digest(
            self.task_contract_id, "task_contract_id"
        ))
        object.__setattr__(self, "source_anchor_ref", source_ref)
        object.__setattr__(self, "private_environment_instance_digest", _digest(
            self.private_environment_instance_digest,
            "private_environment_instance_digest",
        ))
        object.__setattr__(self, "split_manifest_digest", _digest(
            self.split_manifest_digest, "split_manifest_digest"
        ))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetContextRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "TargetContextRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class RuntimeContract:
    protocol_family_id: str
    task_contract_digest: str
    observation_schema_digest: str
    action_schema_digest: str
    observation_dim: int
    action_dim: int
    action_transform_id: str
    policy_runtime_id: str
    state_schema_id: str
    schema: str = RUNTIME_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_CONTRACT_SCHEMA:
            raise ValueError(f"unsupported RuntimeContract schema: {self.schema!r}")
        object.__setattr__(self, "protocol_family_id", _nonempty(
            self.protocol_family_id, "protocol_family_id", safe=True
        ))
        for name in (
            "task_contract_digest", "observation_schema_digest", "action_schema_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "observation_dim", _positive_int(
            self.observation_dim, "observation_dim"
        ))
        object.__setattr__(self, "action_dim", _positive_int(self.action_dim, "action_dim"))
        for name in ("action_transform_id", "policy_runtime_id", "state_schema_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name, safe=True))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def compatibility_digest(self) -> str:
        """Digest the private, task-anonymous minimum execution ABI.

        ``task_contract_digest`` remains part of the full provenance record but
        is deliberately absent here.  It must never become a public selector
        filter or shrink the private full-pool oracle by source task identity.
        """

        return self.execution_abi.digest

    @property
    def execution_abi(self) -> "ExecutionABIRecord":
        return ExecutionABIRecord(
            protocol_family_id=self.protocol_family_id,
            observation_tensor_abi_digest=self.observation_schema_digest,
            action_tensor_abi_digest=self.action_schema_digest,
            action_transform_id=self.action_transform_id,
            policy_runtime_id=self.policy_runtime_id,
            state_abi_id=self.state_schema_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeContract":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "RuntimeContract")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class ExecutionABIRecord:
    """Private, task-anonymous policy calling convention.

    This record deliberately excludes task/reward/schema identity.  It may be
    consulted only after an immutable public selection has been published.
    """

    protocol_family_id: str
    observation_tensor_abi_digest: str
    action_tensor_abi_digest: str
    action_transform_id: str
    policy_runtime_id: str
    state_abi_id: str
    schema: str = EXECUTION_ABI_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_ABI_SCHEMA:
            raise ValueError(f"unsupported ExecutionABIRecord schema: {self.schema!r}")
        object.__setattr__(
            self,
            "protocol_family_id",
            _nonempty(self.protocol_family_id, "protocol_family_id", safe=True),
        )
        for name in ("observation_tensor_abi_digest", "action_tensor_abi_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("action_transform_id", "policy_runtime_id", "state_abi_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name, safe=True))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionABIRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "ExecutionABIRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class SourceCompetenceRecord:
    learnware_id: str
    opaque_source_anchor_id: str
    return_contract_id: str
    validation_seed_digest: str
    episode_count: int
    mean: float
    std: float
    lcb: float | None
    normalized_competence: float
    competence_floor: float
    passed: bool
    championization_digest: str
    private_attestation_digest: str
    schema: str = SOURCE_COMPETENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_COMPETENCE_SCHEMA:
            raise ValueError(f"unsupported SourceCompetenceRecord schema: {self.schema!r}")
        object.__setattr__(self, "learnware_id", _nonempty(
            self.learnware_id, "learnware_id", safe=True
        ))
        object.__setattr__(self, "opaque_source_anchor_id", _digest(
            self.opaque_source_anchor_id, "opaque_source_anchor_id"
        ))
        for name in (
            "return_contract_id", "validation_seed_digest", "championization_digest",
            "private_attestation_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "episode_count", _positive_int(
            self.episode_count, "episode_count"
        ))
        for name in ("mean", "std", "normalized_competence", "competence_floor"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.std < 0.0:
            raise ValueError("std cannot be negative")
        if self.lcb is not None:
            object.__setattr__(self, "lcb", _finite(self.lcb, "lcb"))
        if not 0.0 <= self.normalized_competence <= 1.0:
            raise ValueError("normalized_competence must lie in [0, 1]")
        if not 0.0 <= self.competence_floor <= 1.0:
            raise ValueError("competence_floor must lie in [0, 1]")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be boolean")
        if self.passed != (self.normalized_competence >= self.competence_floor):
            raise ValueError("passed must agree with the absolute competence floor")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceCompetenceRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceCompetenceRecord")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class EnvironmentSpec:
    supports: np.ndarray
    beta: np.ndarray
    empirical_norm2: float
    rkme_norm2: float
    reconstruction_error: float
    reducer_digest: str
    support_budget: int
    latent_dim: int
    representation_protocol_id: str
    measurement_protocol_id: str
    canonical_view_digest: str
    kernel_bandwidth: float
    probe_dataset_digest: str
    environment_spec_digest: str | None = None
    schema: str = ENVIRONMENT_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_SPEC_SCHEMA:
            raise ValueError(f"unsupported EnvironmentSpec schema: {self.schema!r}")
        supports = _readonly_array(self.supports, ndim=2, where="supports")
        beta = _readonly_array(self.beta, ndim=1, where="beta")
        if beta.shape != (supports.shape[0],):
            raise ValueError("beta must have one weight per support")
        if np.any(beta < 0.0) or not np.isclose(np.sum(beta), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("beta must be a probability simplex")
        support_budget = _positive_int(self.support_budget, "support_budget")
        latent_dim = _positive_int(self.latent_dim, "latent_dim")
        if supports.shape != (support_budget, latent_dim):
            raise ValueError("supports shape must equal (support_budget, latent_dim)")
        for name in ("empirical_norm2", "rkme_norm2", "reconstruction_error"):
            number = _finite(getattr(self, name), name)
            if number < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)
        bandwidth = _finite(self.kernel_bandwidth, "kernel_bandwidth")
        if bandwidth <= 0.0:
            raise ValueError("kernel_bandwidth must be positive")
        for name in (
            "reducer_digest", "representation_protocol_id", "measurement_protocol_id",
            "canonical_view_digest", "probe_dataset_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "supports", supports)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "support_budget", support_budget)
        object.__setattr__(self, "latent_dim", latent_dim)
        object.__setattr__(self, "kernel_bandwidth", bandwidth)
        expected = sha256_json(self._payload_without_digest())
        if self.environment_spec_digest is None:
            object.__setattr__(self, "environment_spec_digest", expected)
        else:
            actual = _digest(self.environment_spec_digest, "environment_spec_digest")
            if actual != expected:
                raise ValueError("environment_spec_digest does not match canonical payload")
            object.__setattr__(self, "environment_spec_digest", actual)

    def _payload_without_digest(self) -> dict[str, Any]:
        return canonicalize({
            "schema": self.schema,
            "supports": self.supports,
            "beta": self.beta,
            "empirical_norm2": self.empirical_norm2,
            "rkme_norm2": self.rkme_norm2,
            "reconstruction_error": self.reconstruction_error,
            "reducer_digest": self.reducer_digest,
            "support_budget": self.support_budget,
            "latent_dim": self.latent_dim,
            "representation_protocol_id": self.representation_protocol_id,
            "measurement_protocol_id": self.measurement_protocol_id,
            "canonical_view_digest": self.canonical_view_digest,
            "kernel_bandwidth": self.kernel_bandwidth,
            "probe_dataset_digest": self.probe_dataset_digest,
        })

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "environment_spec_digest": self.environment_spec_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentSpec":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "EnvironmentSpec")
        return cls(**{key: value[key] for key in fields})


@dataclass(frozen=True)
class PublicMarketEntry:
    opaque_learnware_id: str
    normalized_source_competence: float
    tie_break_token: str
    schema: str = PUBLIC_MARKET_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_MARKET_ENTRY_SCHEMA:
            raise ValueError(f"unsupported PublicMarketEntry schema: {self.schema!r}")
        object.__setattr__(
            self,
            "opaque_learnware_id",
            _nonempty(self.opaque_learnware_id, "opaque_learnware_id", safe=True),
        )
        competence = _finite(self.normalized_source_competence, "normalized_source_competence")
        if not 0.0 <= competence <= 1.0:
            raise ValueError("normalized_source_competence must lie in [0, 1]")
        object.__setattr__(self, "tie_break_token", _digest(self.tie_break_token, "tie_break_token"))
        object.__setattr__(self, "normalized_source_competence", competence)

    @property
    def opaque_id(self) -> str:
        """Internal compatibility alias; never serialized."""

        return self.opaque_learnware_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "opaque_learnware_id": self.opaque_learnware_id,
            "normalized_source_competence": self.normalized_source_competence,
            "tie_break_token": self.tie_break_token,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicMarketEntry":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "PublicMarketEntry")
        return cls(**{key: value[key] for key in fields})


def derive_protocol_id(
    kind: str,
    *,
    config_projection: Mapping[str, Any],
    dependency_digests: Mapping[str, str],
) -> str:
    """Derive a domain-separated protocol identity.

    Every protocol kind has a distinct schema tag.  A caller therefore cannot
    accidentally substitute a selector digest for a probe digest even when the
    two canonical payloads happen to be structurally identical.
    """

    if kind not in PROTOCOL_KINDS:
        raise ValueError(f"unsupported v0.2 protocol kind: {kind!r}")
    if not isinstance(config_projection, Mapping):
        raise ValueError("config_projection must be a mapping")
    dependencies = _checked_components(dependency_digests, "dependency_digests")
    return sha256_json({
        "schema": f"policy-learnware.v02-{kind.replace('_', '-')}-protocol.v0",
        "config_projection": canonicalize(config_projection),
        "dependency_digests": dependencies,
    })


def derive_benchmark_protocol_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "benchmark",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_training_protocol_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "training",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_policy_market_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "policy_market",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_probe_protocol_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "probe",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_representation_protocol_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "representation",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_representation_index_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "representation_index",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_selector_protocol_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "selector",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_evaluation_protocol_id(
    *, config_projection: Mapping[str, Any], dependency_digests: Mapping[str, str],
) -> str:
    return derive_protocol_id(
        "evaluation",
        config_projection=config_projection,
        dependency_digests=dependency_digests,
    )


def derive_experiment_protocol_id(
    *,
    benchmark_protocol_id: str,
    training_protocol_id: str,
    policy_market_id: str,
    probe_protocol_id: str,
    representation_protocol_ids: Mapping[str, str],
    representation_index_ids: Mapping[str, str],
    selector_protocol_ids: Mapping[str, str],
    evaluation_protocol_id: str,
    statistics_protocol_id: str,
    cost_contract_digest: str,
) -> str:
    payload = {
        "schema": "policy-learnware.v02-experiment-protocol.v0",
        "benchmark_protocol_id": _digest(benchmark_protocol_id, "benchmark_protocol_id"),
        "training_protocol_id": _digest(training_protocol_id, "training_protocol_id"),
        "policy_market_id": _digest(policy_market_id, "policy_market_id"),
        "probe_protocol_id": _digest(probe_protocol_id, "probe_protocol_id"),
        "representation_protocol_ids": _checked_components(
            representation_protocol_ids, "representation_protocol_ids"
        ),
        "representation_index_ids": _checked_components(
            representation_index_ids, "representation_index_ids"
        ),
        "selector_protocol_ids": _checked_components(
            selector_protocol_ids, "selector_protocol_ids"
        ),
        "evaluation_protocol_id": _digest(evaluation_protocol_id, "evaluation_protocol_id"),
        "statistics_protocol_id": _digest(statistics_protocol_id, "statistics_protocol_id"),
        "cost_contract_digest": _digest(cost_contract_digest, "cost_contract_digest"),
    }
    return sha256_json(payload)


@dataclass(frozen=True)
class ProtocolIdentifiers:
    benchmark_protocol_id: str
    training_protocol_id: str
    policy_market_id: str
    probe_protocol_id: str
    representation_protocol_ids: Mapping[str, str]
    representation_index_ids: Mapping[str, str]
    selector_protocol_ids: Mapping[str, str]
    evaluation_protocol_id: str
    statistics_protocol_id: str
    cost_contract_digest: str
    experiment_protocol_id: str
    schema: str = PROTOCOL_IDENTIFIERS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PROTOCOL_IDENTIFIERS_SCHEMA:
            raise ValueError(f"unsupported ProtocolIdentifiers schema: {self.schema!r}")
        for name in (
            "benchmark_protocol_id", "training_protocol_id", "policy_market_id",
            "probe_protocol_id", "evaluation_protocol_id", "statistics_protocol_id",
            "cost_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in (
            "representation_protocol_ids", "representation_index_ids", "selector_protocol_ids",
        ):
            checked = _checked_components(getattr(self, name), name)
            object.__setattr__(self, name, MappingProxyType(checked))
        if set(self.representation_protocol_ids) != set(self.representation_index_ids):
            raise ValueError(
                "representation protocol and representation index IDs must have identical keys"
            )
        expected = derive_experiment_protocol_id(
            benchmark_protocol_id=self.benchmark_protocol_id,
            training_protocol_id=self.training_protocol_id,
            policy_market_id=self.policy_market_id,
            probe_protocol_id=self.probe_protocol_id,
            representation_protocol_ids=self.representation_protocol_ids,
            representation_index_ids=self.representation_index_ids,
            selector_protocol_ids=self.selector_protocol_ids,
            evaluation_protocol_id=self.evaluation_protocol_id,
            statistics_protocol_id=self.statistics_protocol_id,
            cost_contract_digest=self.cost_contract_digest,
        )
        actual = _digest(self.experiment_protocol_id, "experiment_protocol_id")
        if actual != expected:
            raise ValueError("experiment_protocol_id does not match component protocol IDs")
        object.__setattr__(self, "experiment_protocol_id", actual)

    @classmethod
    def create(cls, **components: Any) -> "ProtocolIdentifiers":
        experiment = derive_experiment_protocol_id(**components)
        return cls(experiment_protocol_id=experiment, **components)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize({name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProtocolIdentifiers":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "ProtocolIdentifiers")
        return cls(**{key: value[key] for key in fields})


__all__ = [
    "AXIS_ANCHOR_BINDING_SCHEMA",
    "AxisAnchorBinding",
    "ENVIRONMENT_INSTANCE_SCHEMA",
    "MODEL_DIFF_SCHEMA",
    "MODEL_DIFF_SUPPORTED_LEAVES",
    "EXECUTION_ABI_SCHEMA",
    "ENVIRONMENT_SPEC_SCHEMA",
    "EnvironmentInstanceRecord",
    "EnvironmentSpec",
    "ExecutionABIRecord",
    "PROTOCOL_IDENTIFIERS_SCHEMA",
    "PROTOCOL_KINDS",
    "PUBLIC_MARKET_ENTRY_SCHEMA",
    "ProtocolIdentifiers",
    "PublicMarketEntry",
    "RUNTIME_CONTRACT_SCHEMA",
    "RuntimeContract",
    "SOURCE_ANCHOR_SCHEMA",
    "SOURCE_COMPETENCE_SCHEMA",
    "SourceAnchorRecord",
    "SourceCompetenceRecord",
    "TARGET_CONTEXT_SCHEMA",
    "TARGET_REGIMES",
    "TargetContextRecord",
    "canonical_array_digest",
    "canonical_environment_instance_projection",
    "canonical_model_diff_projection",
    "canonical_model_snapshot",
    "derive_live_model_diff",
    "derive_benchmark_protocol_id",
    "derive_evaluation_protocol_id",
    "derive_experiment_protocol_id",
    "derive_policy_market_id",
    "derive_probe_protocol_id",
    "derive_protocol_id",
    "derive_representation_index_id",
    "derive_representation_protocol_id",
    "derive_selector_protocol_id",
    "derive_training_protocol_id",
]
