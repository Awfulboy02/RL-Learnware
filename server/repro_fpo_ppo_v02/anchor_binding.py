#!/usr/bin/env python3
"""Digest-bound source-anchor mutation before JIT/state construction.

The reviewed anchor manifest supplies every scientific literal: task, factor,
axis/operator identity, exact flat indices, and before/after digests.  This
module only validates and executes that frozen plan.  It deliberately has no
built-in task/axis/factor table and therefore cannot silently fill a
``[REVIEW REQUIRED]`` choice.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .provenance import (
        ContractError,
        NumericalIntegrityError,
        json_ready,
        load_strict_json,
        require_digest,
        require_exact_keys,
        require_git_commit,
        require_safe_id,
        sha256_json,
        validate_self_digest,
    )
except ImportError:  # pragma: no cover - exercised by executable entry points
    from provenance import (
        ContractError,
        NumericalIntegrityError,
        json_ready,
        load_strict_json,
        require_digest,
        require_exact_keys,
        require_git_commit,
        require_safe_id,
        sha256_json,
        validate_self_digest,
    )


ANCHOR_MANIFEST_SCHEMA = "policy-learnware.v02-anchor-manifest.v0"
ANCHOR_OPERATOR_SCHEMA = "policy-learnware.v02-anchor-operator.v0"
ENVIRONMENT_INSTANCE_SCHEMA = "policy-learnware.v02-environment-instance.v1"
MODEL_DIFF_SCHEMA = "policy-learnware.v02-live-model-diff.v0"
SOURCE_ANCHOR_SCHEMA = "policy-learnware.v02-source-anchor.v0"

# Engineering capability allowlist only.  Which leaves/indices form each
# task's two scientific axes remains entirely manifest-driven and reviewed.
SUPPORTED_MODEL_LEAVES = frozenset(
    {
        "_mjx_model.body_mass",
        "_mjx_model.body_inertia",
        "_mjx_model.dof_damping",
        "_mjx_model.actuator_gainprm",
        "_mjx_model.actuator_gear",
        "_mjx_model.geom_friction",
    }
)


class AnchorBindingError(RuntimeError):
    """The live native environment cannot satisfy its frozen anchor manifest."""


@dataclass(frozen=True)
class ModelLeafRecord:
    path: str
    dtype: str
    shape: tuple[int, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ModelSnapshot:
    model_type: str
    leaves: tuple[ModelLeafRecord, ...]
    digest: str

    def __post_init__(self) -> None:
        paths = tuple(item.path for item in self.leaves)
        if not self.model_type or not paths or paths != tuple(sorted(set(paths))):
            raise ContractError("model snapshot paths must be non-empty, unique, and sorted")
        expected = sha256_json(
            {
                "model_type": self.model_type,
                "leaves": [item.to_dict() for item in self.leaves],
            }
        )
        if self.digest != expected:
            raise ContractError("model snapshot digest mismatch")

    @property
    def by_path(self) -> Mapping[str, ModelLeafRecord]:
        return MappingProxyType({item.path: item for item in self.leaves})


@dataclass(frozen=True)
class MutationSpec:
    leaf: str
    flat_indices: tuple[int, ...]
    multiplier: float
    expected_before_digest: str
    expected_after_digest: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MutationSpec":
        require_exact_keys(
            value,
            {
                "leaf",
                "flat_indices",
                "multiplier",
                "expected_before_digest",
                "expected_after_digest",
            },
            "anchor mutation",
        )
        leaf = value["leaf"]
        if leaf not in SUPPORTED_MODEL_LEAVES:
            raise ContractError(f"unsupported/unallowlisted model leaf: {leaf!r}")
        raw_indices = value["flat_indices"]
        if (
            not isinstance(raw_indices, list)
            or not raw_indices
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_indices)
            or raw_indices != sorted(set(raw_indices))
            or raw_indices[0] < 0
        ):
            raise ContractError("mutation flat_indices must be sorted unique nonnegative integers")
        multiplier = value["multiplier"]
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise ContractError("mutation multiplier must be numeric")
        multiplier = float(multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0.0 or multiplier == 1.0:
            raise ContractError("shifted mutation multiplier must be finite, positive, and non-unit")
        before = require_digest(value["expected_before_digest"], "expected_before_digest")
        after = require_digest(value["expected_after_digest"], "expected_after_digest")
        if before == after:
            raise ContractError("a shifted mutation must change its leaf digest")
        return cls(
            leaf=leaf,
            flat_indices=tuple(raw_indices),
            multiplier=multiplier,
            expected_before_digest=before,
            expected_after_digest=after,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf": self.leaf,
            "flat_indices": list(self.flat_indices),
            "multiplier": self.multiplier,
            "expected_before_digest": self.expected_before_digest,
            "expected_after_digest": self.expected_after_digest,
        }


@dataclass(frozen=True)
class AnchorOperator:
    operator_id: str
    axis_id: str
    axis_registry_digest: str
    factor: float
    mutations: tuple[MutationSpec, ...]
    schema: str = ANCHOR_OPERATOR_SCHEMA

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnchorOperator":
        require_exact_keys(
            value,
            {"schema", "operator_id", "axis_id", "axis_registry_digest", "factor", "mutations"},
            "anchor operator",
        )
        if value["schema"] != ANCHOR_OPERATOR_SCHEMA:
            raise ContractError(f"unsupported anchor operator schema: {value['schema']!r}")
        factor = value["factor"]
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise ContractError("operator factor must be numeric")
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0 or factor == 1.0:
            raise ContractError("shifted operator factor must be finite, positive, and non-unit")
        raw_mutations = value["mutations"]
        if not isinstance(raw_mutations, list) or not raw_mutations:
            raise ContractError("shifted operator requires at least one mutation")
        mutations = tuple(MutationSpec.from_dict(item) for item in raw_mutations)
        leaves = tuple(item.leaf for item in mutations)
        if leaves != tuple(sorted(set(leaves))):
            raise ContractError("operator mutations must have unique leaves in sorted order")
        if any(item.multiplier != factor for item in mutations):
            raise ContractError("every coupled mutation must use the frozen anchor factor")
        return cls(
            schema=value["schema"],
            operator_id=require_safe_id(value["operator_id"], "operator_id"),
            axis_id=require_safe_id(value["axis_id"], "axis_id"),
            axis_registry_digest=require_digest(
                value["axis_registry_digest"], "axis_registry_digest"
            ),
            factor=factor,
            mutations=mutations,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operator_id": self.operator_id,
            "axis_id": self.axis_id,
            "axis_registry_digest": self.axis_registry_digest,
            "factor": self.factor,
            "mutations": [item.to_dict() for item in self.mutations],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def canonical_environment_instance_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the sole package/server environment-instance identity material."""

    task = require_safe_id(value["task"], "environment instance.task")
    backend = require_safe_id(value["backend"], "environment instance.backend")
    nominal = value["nominal"]
    if not isinstance(nominal, bool):
        raise ContractError("environment instance.nominal must be boolean")
    factor = value["factor"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ContractError("environment instance.factor must be numeric")
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ContractError("environment instance.factor must be finite and positive")
    environment_class = value["environment_class"]
    if not isinstance(environment_class, str) or "." not in environment_class:
        raise ContractError("environment instance.environment_class must be fully qualified")
    operator = value["operator_digest"]
    binding = value["axis_binding_digest"]
    operator = None if operator is None else require_digest(operator, "operator_digest")
    binding = None if binding is None else require_digest(binding, "axis_binding_digest")
    nominal_model = require_digest(
        value["expected_nominal_model_digest"], "expected_nominal_model_digest"
    )
    bound_model = require_digest(
        value["expected_bound_model_digest"], "expected_bound_model_digest"
    )
    model_diff = require_digest(value["model_diff_digest"], "model_diff_digest")
    if nominal:
        if factor != 1.0 or operator is not None or binding is not None:
            raise ContractError("nominal environment instance must be factor-one and unbound")
        if nominal_model != bound_model:
            raise ContractError("nominal environment instance model digests differ")
        expected_model_diff = sha256_json(
            canonical_model_diff_projection(
                nominal_model_digest=nominal_model,
                bound_model_digest=bound_model,
                changes=(),
            )
        )
        if model_diff != expected_model_diff:
            raise ContractError("nominal environment instance model_diff_digest mismatch")
    else:
        if factor == 1.0 or operator is None or binding is None:
            raise ContractError("shifted environment instance must bind operator and axis")
        if nominal_model == bound_model:
            raise ContractError("shifted environment instance did not change the model")
    return {
        "schema": ENVIRONMENT_INSTANCE_SCHEMA,
        "task": task,
        "backend": backend,
        "nominal": nominal,
        "factor": factor,
        "environment_class": environment_class,
        "registry_config_digest": require_digest(
            value["registry_config_digest"], "registry_config_digest"
        ),
        "runtime_digest": require_digest(value["runtime_digest"], "runtime_digest"),
        "expected_nominal_model_digest": nominal_model,
        "expected_bound_model_digest": bound_model,
        "operator_digest": operator,
        "axis_binding_digest": binding,
        "model_diff_digest": model_diff,
    }


def derive_environment_instance_digest(value: Mapping[str, Any]) -> str:
    return sha256_json(canonical_environment_instance_projection(value))


def derive_source_anchor_id(
    *, environment_instance_digest: str, axis_binding_digest: str | None
) -> str:
    environment = require_digest(
        environment_instance_digest, "source anchor.environment_instance_digest"
    )
    binding = (
        None
        if axis_binding_digest is None
        else require_digest(axis_binding_digest, "source anchor.axis_binding_digest")
    )
    return sha256_json(
        {
            "schema": SOURCE_ANCHOR_SCHEMA,
            "environment_instance_digest": environment,
            "axis_binding_digest": binding,
            "split_role": "source",
        }
    )


def canonical_model_diff_projection(
    *,
    nominal_model_digest: str,
    bound_model_digest: str,
    changes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Normalize the sole live model-diff material hashed by package/server."""

    nominal = require_digest(nominal_model_digest, "nominal_model_digest")
    bound = require_digest(bound_model_digest, "bound_model_digest")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(changes):
        require_exact_keys(
            raw,
            {"leaf", "before_digest", "after_digest", "changed_flat_indices"},
            f"model diff change[{index}]",
        )
        leaf = raw["leaf"]
        if leaf not in SUPPORTED_MODEL_LEAVES:
            raise ContractError(f"model diff contains an unallowlisted leaf: {leaf!r}")
        before = require_digest(
            raw["before_digest"], f"model diff change[{index}].before_digest"
        )
        after = require_digest(
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
            raise ContractError(
                "changed_flat_indices must be sorted unique nonnegative integers"
            )
        if before == after:
            raise ContractError("a model diff change must alter its leaf digest")
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
        raise ContractError("model diff changes must have unique leaves in sorted order")
    if bool(normalized) != (nominal != bound):
        raise ContractError("model digest equality and model diff changes disagree")
    return {
        "schema": MODEL_DIFF_SCHEMA,
        "nominal_model_digest": nominal,
        "bound_model_digest": bound,
        "changes": normalized,
    }


def _manifest_model_diff_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    operator = value["operator"]
    changes = [] if operator is None else [
        {
            "leaf": item["leaf"],
            "before_digest": item["expected_before_digest"],
            "after_digest": item["expected_after_digest"],
            "changed_flat_indices": item["flat_indices"],
        }
        for item in operator["mutations"]
    ]
    return canonical_model_diff_projection(
        nominal_model_digest=value["expected_nominal_model_digest"],
        bound_model_digest=value["expected_bound_model_digest"],
        changes=changes,
    )


def derive_manifest_model_diff_digest(value: Mapping[str, Any]) -> str:
    return sha256_json(_manifest_model_diff_projection(value))


@dataclass(frozen=True)
class AnchorManifest:
    _value: Mapping[str, Any]
    operator: AnchorOperator | None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AnchorManifest":
        value = deepcopy(dict(raw))
        require_exact_keys(
            value,
            {
                "schema",
                "anchor_id",
                "task",
                "backend",
                "nominal",
                "factor",
                "environment_class",
                "registry_config",
                "registry_config_digest",
                "runtime",
                "runtime_digest",
                "expected_nominal_model_digest",
                "expected_bound_model_digest",
                "operator",
                "operator_digest",
                "axis_binding_digest",
                "model_diff_digest",
                "environment_instance_digest",
                "manifest_digest",
            },
            "anchor manifest",
        )
        if value["schema"] != ANCHOR_MANIFEST_SCHEMA:
            raise ContractError(f"unsupported anchor manifest schema: {value['schema']!r}")
        anchor_id = require_digest(value["anchor_id"], "anchor_id")
        require_safe_id(value["task"], "task")
        if value["backend"] != "mujoco_playground.registry":
            raise ContractError("anchor backend must be mujoco_playground.registry")
        if not isinstance(value["nominal"], bool):
            raise ContractError("anchor nominal flag must be boolean")
        factor = value["factor"]
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise ContractError("anchor factor must be numeric")
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ContractError("anchor factor must be finite and positive")
        value["factor"] = factor
        if (
            not isinstance(value["environment_class"], str)
            or "." not in value["environment_class"]
        ):
            raise ContractError("environment_class must be a fully-qualified class name")
        if not isinstance(value["registry_config"], dict):
            raise ContractError("registry_config must be a frozen JSON object")
        config_digest = require_digest(value["registry_config_digest"], "registry_config_digest")
        if config_digest != sha256_json(value["registry_config"]):
            raise ContractError("registry_config_digest mismatch")
        runtime = value["runtime"]
        runtime_keys = {
            "fpo_commit",
            "python_major_minor",
            "jax",
            "jaxlib",
            "mujoco",
            "playground",
        }
        if not isinstance(runtime, dict):
            raise ContractError("runtime must be a frozen JSON object")
        require_exact_keys(runtime, runtime_keys, "anchor runtime")
        require_git_commit(runtime["fpo_commit"], "runtime.fpo_commit")
        if any(not isinstance(runtime[name], str) or not runtime[name] for name in runtime_keys - {"fpo_commit"}):
            raise ContractError("all frozen runtime version fields must be non-empty strings")
        runtime_digest = require_digest(value["runtime_digest"], "runtime_digest")
        if runtime_digest != sha256_json(runtime):
            raise ContractError("runtime_digest mismatch")
        nominal_digest = require_digest(
            value["expected_nominal_model_digest"], "expected_nominal_model_digest"
        )
        bound_digest = require_digest(
            value["expected_bound_model_digest"], "expected_bound_model_digest"
        )
        operator_raw = value["operator"]
        operator: AnchorOperator | None
        if value["nominal"]:
            if factor != 1.0 or operator_raw is not None or value["operator_digest"] is not None:
                raise ContractError("nominal anchor requires factor=1 and null operator/digest")
            if value["axis_binding_digest"] is not None:
                raise ContractError("nominal anchor requires axis_binding_digest=null")
            if nominal_digest != bound_digest:
                raise ContractError("nominal anchor model digests must be identical")
            operator = None
        else:
            if factor == 1.0 or not isinstance(operator_raw, dict):
                raise ContractError("shifted anchor requires non-unit factor and operator")
            operator = AnchorOperator.from_dict(operator_raw)
            if operator.factor != factor:
                raise ContractError("anchor factor and operator factor disagree")
            if value["operator_digest"] != operator.digest:
                raise ContractError("anchor operator_digest mismatch")
            require_digest(value["axis_binding_digest"], "axis_binding_digest")
            if nominal_digest == bound_digest:
                raise ContractError("shifted anchor cannot claim a nominal bound model")
        environment_digest = require_digest(
            value["environment_instance_digest"], "environment_instance_digest"
        )
        model_diff_digest = require_digest(
            value["model_diff_digest"], "model_diff_digest"
        )
        if model_diff_digest != derive_manifest_model_diff_digest(value):
            raise ContractError("model_diff_digest mismatch")
        if environment_digest != derive_environment_instance_digest(value):
            raise ContractError("environment_instance_digest mismatch")
        expected_anchor_id = derive_source_anchor_id(
            environment_instance_digest=environment_digest,
            axis_binding_digest=value["axis_binding_digest"],
        )
        if anchor_id != expected_anchor_id:
            raise ContractError("anchor_id does not match canonical source-anchor payload")
        validate_self_digest(value, key="manifest_digest", where="anchor manifest")
        return cls(_value=MappingProxyType(value), operator=operator)

    @classmethod
    def from_path(cls, path: Path | str) -> "AnchorManifest":
        return cls.from_dict(load_strict_json(path))

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self._value))

    def __getattr__(self, name: str) -> Any:
        try:
            return deepcopy(self._value[name])
        except KeyError as error:
            raise AttributeError(name) from error


def _model_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _path_token(key: Any) -> str:
    for attribute in ("name", "key", "idx"):
        if hasattr(key, attribute):
            return str(getattr(key, attribute))
    return str(key)


def _generic_leaves(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    if isinstance(value, (np.ndarray, np.generic, bool, int, float)):
        return [(".".join(prefix) or "<root>", value)]
    if is_dataclass(value) and not isinstance(value, type):
        result: list[tuple[str, Any]] = []
        for field in fields(value):
            result.extend(_generic_leaves(getattr(value, field.name), prefix + (field.name,)))
        return result
    if isinstance(value, Mapping):
        result = []
        for key in sorted(value, key=str):
            result.extend(_generic_leaves(value[key], prefix + (str(key),)))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for index, item in enumerate(value):
            result.extend(_generic_leaves(item, prefix + (str(index),)))
        return result
    if hasattr(value, "__dict__"):
        result = []
        for name in sorted(vars(value)):
            if not name.startswith("_"):
                result.extend(_generic_leaves(getattr(value, name), prefix + (name,)))
        if result:
            return result
    raise TypeError(f"cannot flatten model value of type {_model_type(value)}")


def _model_leaves(model: Any) -> list[tuple[str, Any]]:
    if type(model).__module__.startswith("mujoco.mjx"):
        try:
            import jax
        except ImportError as error:  # pragma: no cover - production gate
            raise AnchorBindingError("MJX model auditing requires JAX") from error
        path_leaves, _ = jax.tree_util.tree_flatten_with_path(model)
        return [
            (".".join(_path_token(key) for key in path), leaf)
            for path, leaf in path_leaves
        ]
    return _generic_leaves(model)


def _host_array(value: Any) -> np.ndarray:
    try:
        import jax
    except ImportError:
        host = value
    else:
        host = jax.device_get(value)
    array = np.asarray(host)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise AnchorBindingError("model contains a non-numerical leaf")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise NumericalIntegrityError("model contains a non-finite leaf")
    return np.ascontiguousarray(array)


def array_digest(value: Any) -> str:
    array = _host_array(value)
    return sha256_json(
        {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    )


def _changed_flat_indices(left: np.ndarray, right: np.ndarray) -> list[int]:
    if left.dtype != right.dtype or left.shape != right.shape:
        raise AnchorBindingError("model leaf shape/dtype changed")
    left_bytes = left.reshape(-1).view(np.uint8).reshape(left.size, left.dtype.itemsize)
    right_bytes = right.reshape(-1).view(np.uint8).reshape(right.size, right.dtype.itemsize)
    return np.flatnonzero(np.any(left_bytes != right_bytes, axis=1)).tolist()


def snapshot_model(model: Any) -> ModelSnapshot:
    records = []
    for path, value in _model_leaves(model):
        array = _host_array(value)
        records.append(
            ModelLeafRecord(
                path=path,
                dtype=array.dtype.str,
                shape=tuple(int(item) for item in array.shape),
                digest=array_digest(array),
            )
        )
    records.sort(key=lambda item: item.path)
    material = {
        "model_type": _model_type(model),
        "leaves": [item.to_dict() for item in records],
    }
    return ModelSnapshot(
        model_type=material["model_type"],
        leaves=tuple(records),
        digest=sha256_json(material),
    )


def derive_live_model_diff(model: Any, bound_model: Any) -> tuple[dict[str, Any], str]:
    """Compute changed leaves and exact flat indices from two live models."""

    nominal = snapshot_model(model)
    bound = snapshot_model(bound_model)
    if nominal.model_type != bound.model_type:
        raise AnchorBindingError("model type changed during anchor binding")
    if set(nominal.by_path) != set(bound.by_path):
        raise AnchorBindingError("model pytree structure changed during anchor binding")
    nominal_values = dict(_model_leaves(model))
    bound_values = dict(_model_leaves(bound_model))
    changes: list[dict[str, Any]] = []
    for path in sorted(nominal.by_path):
        left = _host_array(nominal_values[path])
        right = _host_array(bound_values[path])
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AnchorBindingError(f"model leaf shape/dtype changed: {path}")
        indices = _changed_flat_indices(left, right)
        if indices:
            changes.append(
                {
                    "leaf": f"_mjx_model.{path}",
                    "before_digest": nominal.by_path[path].digest,
                    "after_digest": bound.by_path[path].digest,
                    "changed_flat_indices": indices,
                }
            )
    projection = canonical_model_diff_projection(
        nominal_model_digest=nominal.digest,
        bound_model_digest=bound.digest,
        changes=changes,
    )
    return projection, sha256_json(projection)


def _replace_selected(value: Any, spec: MutationSpec) -> Any:
    before = _host_array(value)
    if before.dtype.kind != "f" or before.size == 0:
        raise AnchorBindingError(f"{spec.leaf} must be a non-empty floating array")
    indices = np.asarray(spec.flat_indices, dtype=np.int64)
    if int(indices[-1]) >= before.size:
        raise AnchorBindingError(f"{spec.leaf} mutation index exceeds flattened leaf size")
    if array_digest(before) != spec.expected_before_digest:
        raise AnchorBindingError(f"{spec.leaf} before digest mismatch")
    flat = before.reshape(-1).copy()
    selected_before = flat[indices].copy()
    flat[indices] *= np.asarray(spec.multiplier, dtype=before.dtype)
    after = flat.reshape(before.shape)
    if not np.all(np.isfinite(after)):
        raise NumericalIntegrityError(f"{spec.leaf} scaling produced non-finite values")
    if not np.all(np.not_equal(selected_before, after.reshape(-1)[indices])):
        raise AnchorBindingError(f"{spec.leaf} selected indices did not all change")
    mask = np.ones(before.size, dtype=bool)
    mask[indices] = False
    if not np.array_equal(before.reshape(-1)[mask], after.reshape(-1)[mask]):
        raise AnchorBindingError(f"{spec.leaf} changed outside its exact flat-index allowlist")
    if array_digest(after) != spec.expected_after_digest:
        raise AnchorBindingError(f"{spec.leaf} after digest mismatch")
    if type(value).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp.asarray(after)
    return after


@dataclass(frozen=True)
class BindingAudit:
    anchor_id: str
    environment_instance_digest: str
    nominal_model_digest: str
    bound_model_digest: str
    changed_leaves: tuple[str, ...]
    model_diff_digest: str
    source_unchanged: bool
    operator_digest: str | None
    manifest_digest: str


def bind_model_to_anchor(
    model: Any, manifest: AnchorManifest
) -> tuple[Any, BindingAudit]:
    nominal = snapshot_model(model)
    if nominal.digest != manifest.expected_nominal_model_digest:
        raise AnchorBindingError(
            "live nominal model digest disagrees with the frozen anchor manifest"
        )
    if manifest.nominal:
        _, live_model_diff_digest = derive_live_model_diff(model, model)
        if live_model_diff_digest != manifest.model_diff_digest:
            raise AnchorBindingError("live nominal model diff disagrees with manifest")
        return model, BindingAudit(
            anchor_id=manifest.anchor_id,
            environment_instance_digest=manifest.environment_instance_digest,
            nominal_model_digest=nominal.digest,
            bound_model_digest=nominal.digest,
            changed_leaves=(),
            model_diff_digest=live_model_diff_digest,
            source_unchanged=True,
            operator_digest=None,
            manifest_digest=manifest.manifest_digest,
        )
    assert manifest.operator is not None
    replacements: dict[str, Any] = {}
    for mutation in manifest.operator.mutations:
        field = mutation.leaf.removeprefix("_mjx_model.")
        if "." in field or not hasattr(model, field):
            raise AnchorBindingError(f"live model lacks direct allowlisted field {field!r}")
        replacements[field] = _replace_selected(getattr(model, field), mutation)
    tree_replace = getattr(model, "tree_replace", None)
    if not callable(tree_replace):
        raise AnchorBindingError("native model lacks immutable tree_replace")
    bound = tree_replace(replacements)
    if bound is model:
        raise AnchorBindingError("tree_replace returned the nominal model in place")
    nominal_after = snapshot_model(model)
    if nominal_after.digest != nominal.digest:
        raise AnchorBindingError("anchor operator mutated the nominal model in place")
    shifted = snapshot_model(bound)
    if shifted.digest != manifest.expected_bound_model_digest:
        raise AnchorBindingError("actual shifted model digest disagrees with manifest")
    if set(nominal.by_path) != set(shifted.by_path):
        raise AnchorBindingError("model pytree structure changed during anchor binding")
    changed = tuple(
        sorted(
            path
            for path in nominal.by_path
            if nominal.by_path[path] != shifted.by_path[path]
        )
    )
    expected = tuple(
        sorted(item.leaf.removeprefix("_mjx_model.") for item in manifest.operator.mutations)
    )
    if changed != expected:
        raise AnchorBindingError(
            f"model diff escaped exact allowlist: expected={expected}, observed={changed}"
        )
    live_projection, live_model_diff_digest = derive_live_model_diff(model, bound)
    expected_indices = {
        item.leaf: list(item.flat_indices) for item in manifest.operator.mutations
    }
    actual_indices = {
        item["leaf"]: item["changed_flat_indices"]
        for item in live_projection["changes"]
    }
    if actual_indices != expected_indices:
        raise AnchorBindingError(
            "actual model diff indices disagree with the frozen anchor operator"
        )
    if live_model_diff_digest != manifest.model_diff_digest:
        raise AnchorBindingError("actual live model diff disagrees with manifest")
    return bound, BindingAudit(
        anchor_id=manifest.anchor_id,
        environment_instance_digest=manifest.environment_instance_digest,
        nominal_model_digest=nominal.digest,
        bound_model_digest=shifted.digest,
        changed_leaves=tuple(f"_mjx_model.{item}" for item in changed),
        model_diff_digest=live_model_diff_digest,
        source_unchanged=True,
        operator_digest=manifest.operator_digest,
        manifest_digest=manifest.manifest_digest,
    )


def _environment_class(env: Any) -> str:
    return f"{type(env).__module__}.{type(env).__qualname__}"


def verify_bound_environment(env: Any, manifest: AnchorManifest) -> ModelSnapshot:
    if _environment_class(env) != manifest.environment_class:
        raise AnchorBindingError("live native environment class mismatch")
    if not hasattr(env, "_mjx_model"):
        raise AnchorBindingError("live native environment lacks _mjx_model")
    snapshot = snapshot_model(env._mjx_model)
    if not manifest.nominal and snapshot.digest == manifest.expected_nominal_model_digest:
        raise AnchorBindingError(
            "poisoned shifted run: directory/manifest claims shifted but actual env is nominal"
        )
    if snapshot.digest != manifest.expected_bound_model_digest:
        raise AnchorBindingError("actual live environment model is not the frozen anchor")
    if hasattr(type(env), "mjx_model") and env.mjx_model is not env._mjx_model:
        raise AnchorBindingError("mjx_model property is not bound to _mjx_model")
    return snapshot


@dataclass(frozen=True)
class BoundEnvironment:
    env: Any
    manifest: AnchorManifest
    audit: BindingAudit

    def verify(self) -> ModelSnapshot:
        return verify_bound_environment(self.env, self.manifest)


def load_and_bind_anchor(*, registry: Any, manifest: AnchorManifest) -> BoundEnvironment:
    config = registry.get_default_config(manifest.task)
    frozen_view = json_ready(config)
    if frozen_view != manifest.registry_config:
        raise AnchorBindingError("live registry default config differs from frozen anchor config")
    if sha256_json(frozen_view) != manifest.registry_config_digest:
        raise AnchorBindingError("live registry config digest mismatch")
    env = registry.load(manifest.task, config=config)
    if _environment_class(env) != manifest.environment_class:
        raise AnchorBindingError("registry loaded an unexpected concrete environment class")
    if not hasattr(env, "_mjx_model"):
        raise AnchorBindingError("registry environment lacks pinned _mjx_model")
    nominal_model = env._mjx_model
    bound_model, audit = bind_model_to_anchor(nominal_model, manifest)
    if bound_model is not nominal_model:
        env._mjx_model = bound_model
    verify_bound_environment(env, manifest)
    return BoundEnvironment(env=env, manifest=manifest, audit=audit)


__all__ = [
    "ANCHOR_MANIFEST_SCHEMA",
    "AnchorBindingError",
    "AnchorManifest",
    "BoundEnvironment",
    "load_and_bind_anchor",
]
