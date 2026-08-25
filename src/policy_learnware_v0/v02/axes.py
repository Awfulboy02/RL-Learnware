"""Closed, task-specific dynamics-axis registry and functional operators.

Formal callers may name only a registered ``axis_id`` and ``factor_id``.  An
entry binds those names to one of four reviewed operator families and to exact
axis-zero indices in a small, operator-specific MuJoCo leaf allowlist.  The
operators copy the environment object, functionally replace the immutable MJX
model, and audit the complete model pytree before returning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..hashing import sha256_file, sha256_json, sha256_ndarrays


FactorRole = Literal[
    "source",
    "development",
    "confirmatory_heldout",
    "safety_exact_reference",
]

SOURCE_ROLE: FactorRole = "source"
DEVELOPMENT_ROLE: FactorRole = "development"
CONFIRMATORY_ROLE: FactorRole = "confirmatory_heldout"
SAFETY_ROLE: FactorRole = "safety_exact_reference"
_ROLES = frozenset({SOURCE_ROLE, DEVELOPMENT_ROLE, CONFIRMATORY_ROLE, SAFETY_ROLE})
MUTATION_STAGE = "post_registry_load_pre_jit_state_creation"

MASS_INERTIA_OPERATOR = "mass_inertia_scale_v02"
JOINT_DAMPING_OPERATOR = "joint_damping_scale_v02"
CONTACT_FRICTION_OPERATOR = "contact_friction_scale_v02"
ACTUATOR_GAIN_OPERATOR = "actuator_gain_scale_v02"

_OPERATOR_ALLOWED_LEAVES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        MASS_INERTIA_OPERATOR: frozenset({"body_mass", "body_inertia"}),
        JOINT_DAMPING_OPERATOR: frozenset({"dof_damping"}),
        CONTACT_FRICTION_OPERATOR: frozenset({"geom_friction"}),
        ACTUATOR_GAIN_OPERATOR: frozenset({"actuator_gainprm", "actuator_gear"}),
    }
)


class DynamicsAxisError(RuntimeError):
    """A registry entry or dynamics mutation violated the closed contract."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
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


def _positive_factor(value: Any, where: str = "factor") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{where} must be finite and positive")
    return result


def operator_source_digest() -> str:
    return sha256_file(Path(__file__))


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str
    value: float
    roles: frozenset[FactorRole]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_id", _nonempty(self.factor_id, "factor_id"))
        object.__setattr__(self, "value", _positive_factor(self.value, "factor value"))
        roles = frozenset(self.roles)
        if not roles or not roles.issubset(_ROLES):
            raise ValueError(f"invalid factor roles: {sorted(roles)}")
        if CONFIRMATORY_ROLE in roles and (
            SOURCE_ROLE in roles or DEVELOPMENT_ROLE in roles or SAFETY_ROLE in roles
        ):
            raise ValueError("confirmatory held-out factors cannot overlap another role")
        if DEVELOPMENT_ROLE in roles and (SOURCE_ROLE in roles or SAFETY_ROLE in roles):
            raise ValueError("development factors cannot overlap source/safety factors")
        if SAFETY_ROLE in roles and SOURCE_ROLE not in roles:
            raise ValueError("safety_exact_reference is legal only for a source factor")
        object.__setattr__(self, "roles", roles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "value": self.value,
            "roles": sorted(self.roles),
        }


@dataclass(frozen=True)
class LeafSelection:
    """Exact axis-zero rows, optionally restricted to trailing components."""

    leaf: str
    indices: tuple[int, ...]
    components: tuple[int, ...] | None = None
    require_nonzero: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "leaf", _nonempty(self.leaf, "leaf"))
        indices = tuple(self.indices)
        if not indices or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in indices):
            raise ValueError("leaf selection indices must be non-empty non-negative integers")
        if len(indices) != len(set(indices)) or indices != tuple(sorted(indices)):
            raise ValueError("leaf selection indices must be unique and sorted")
        object.__setattr__(self, "indices", indices)
        if self.components is not None:
            components = tuple(self.components)
            if not components or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in components
            ):
                raise ValueError("components must be non-empty non-negative integers")
            if len(components) != len(set(components)) or components != tuple(sorted(components)):
                raise ValueError("components must be unique and sorted")
            object.__setattr__(self, "components", components)
        if not isinstance(self.require_nonzero, bool):
            raise ValueError("require_nonzero must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaf": self.leaf,
            "indices": list(self.indices),
            "components": None if self.components is None else list(self.components),
            "require_nonzero": self.require_nonzero,
        }


@dataclass(frozen=True)
class AxisRegistryEntry:
    axis_id: str
    task_id: str
    backend_id: str
    operator_id: str
    operator_version: str
    operator_digest: str
    selections: tuple[LeafSelection, ...]
    factors: tuple[FactorDefinition, ...]
    mutation_stage: str = MUTATION_STAGE

    def __post_init__(self) -> None:
        for name in ("axis_id", "task_id", "backend_id", "operator_version"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.operator_id not in _OPERATOR_ALLOWED_LEAVES:
            raise ValueError(f"unregistered operator_id: {self.operator_id!r}")
        object.__setattr__(self, "operator_digest", _digest(self.operator_digest, "operator_digest"))
        if self.mutation_stage != MUTATION_STAGE:
            raise ValueError(f"mutation_stage must be {MUTATION_STAGE}")
        selections = tuple(self.selections)
        if not selections:
            raise ValueError("an axis requires at least one exact leaf selection")
        leaves = tuple(item.leaf for item in selections)
        if len(leaves) != len(set(leaves)):
            raise ValueError("an axis cannot register the same model leaf twice")
        allowed = _OPERATOR_ALLOWED_LEAVES[self.operator_id]
        unknown = set(leaves) - allowed
        if unknown:
            raise ValueError(
                f"operator {self.operator_id!r} forbids model leaves {sorted(unknown)}"
            )
        self._validate_operator_shape(selections)
        object.__setattr__(self, "selections", selections)

        factors = tuple(self.factors)
        if not factors:
            raise ValueError("an axis requires registered literal factors")
        ids = tuple(item.factor_id for item in factors)
        if len(ids) != len(set(ids)):
            raise ValueError("factor IDs must be unique within an axis")
        # Distinct IDs may never smuggle the same numeric value into disjoint splits.
        by_value: dict[float, set[FactorRole]] = {}
        for item in factors:
            by_value.setdefault(item.value, set()).update(item.roles)
        for value, roles in by_value.items():
            if CONFIRMATORY_ROLE in roles and len(roles) != 1:
                raise ValueError(f"confirmatory factor value {value} overlaps another split")
            if DEVELOPMENT_ROLE in roles and (SOURCE_ROLE in roles or SAFETY_ROLE in roles):
                raise ValueError(f"development factor value {value} overlaps source")
        object.__setattr__(self, "factors", factors)

    def _validate_operator_shape(self, selections: tuple[LeafSelection, ...]) -> None:
        by_leaf = {item.leaf: item for item in selections}
        if self.operator_id == MASS_INERTIA_OPERATOR:
            if set(by_leaf) != {"body_mass", "body_inertia"}:
                raise ValueError("mass/inertia operator requires exactly body_mass and body_inertia")
            if by_leaf["body_mass"].indices != by_leaf["body_inertia"].indices:
                raise ValueError("mass and inertia must scale the same body indices")
            if by_leaf["body_mass"].components is not None:
                raise ValueError("body_mass cannot select trailing components")
            if by_leaf["body_inertia"].components is not None:
                raise ValueError("body_inertia must scale all principal components")
        elif self.operator_id == JOINT_DAMPING_OPERATOR:
            if set(by_leaf) != {"dof_damping"} or by_leaf["dof_damping"].components is not None:
                raise ValueError("joint damping requires exactly row indices in dof_damping")
        elif self.operator_id == CONTACT_FRICTION_OPERATOR:
            if set(by_leaf) != {"geom_friction"}:
                raise ValueError("contact friction requires exactly geom_friction")
        elif self.operator_id == ACTUATOR_GAIN_OPERATOR:
            if not set(by_leaf).issubset({"actuator_gainprm", "actuator_gear"}):
                raise ValueError("actuator gain operator has an invalid leaf")
            row_sets = {item.indices for item in selections}
            if len(row_sets) != 1:
                raise ValueError("actuator gain/gear leaves must select the same actuators")

    @property
    def factor_map(self) -> Mapping[str, FactorDefinition]:
        return MappingProxyType({item.factor_id: item for item in self.factors})

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "task_id": self.task_id,
            "backend_id": self.backend_id,
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "operator_digest": self.operator_digest,
            "selections": [item.to_dict() for item in self.selections],
            "factors": [item.to_dict() for item in self.factors],
            "mutation_stage": self.mutation_stage,
        }


@dataclass(frozen=True)
class AxisRegistry:
    entries: Mapping[str, AxisRegistryEntry]

    def __post_init__(self) -> None:
        entries = dict(self.entries)
        if not entries:
            raise ValueError("axis registry cannot be empty")
        if any(not isinstance(key, str) or not key for key in entries):
            raise ValueError("axis registry storage keys must be non-empty strings")
        semantic_keys = tuple((entry.task_id, entry.axis_id) for entry in entries.values())
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError("axis registry contains a duplicate task/axis pair")
        object.__setattr__(self, "entries", MappingProxyType(entries))

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v02-axis-registry.v0",
                "entries": {
                    f"{entry.task_id}::{entry.axis_id}": entry.to_dict()
                    for entry in sorted(
                        self.entries.values(),
                        key=lambda item: (item.task_id, item.axis_id),
                    )
                },
            }
        )

    def require(
        self,
        *,
        task_id: str,
        axis_id: str,
        factor_id: str,
        role: FactorRole,
    ) -> tuple[AxisRegistryEntry, FactorDefinition]:
        matches = tuple(
            entry
            for entry in self.entries.values()
            if entry.task_id == task_id and entry.axis_id == axis_id
        )
        if not matches:
            raise KeyError(f"unknown or unregistered axis_id: {axis_id!r}")
        if len(matches) != 1:  # guarded by __post_init__, retained fail closed.
            raise ValueError("axis registry contains an ambiguous task/axis pair")
        entry = matches[0]
        if factor_id not in entry.factor_map:
            raise KeyError(f"unknown or unregistered factor_id: {factor_id!r}")
        factor = entry.factor_map[factor_id]
        if role not in factor.roles:
            raise ValueError(
                f"factor {factor_id!r} is not registered for split role {role!r}"
            )
        return entry, factor

    def validate_formal_scope(self, task_ids: Sequence[str]) -> None:
        tasks = tuple(task_ids)
        if not tasks or len(tasks) != len(set(tasks)):
            raise ValueError("formal task IDs must be non-empty and unique")
        if len(tasks) != 6:
            raise ValueError("v0.2 formal scope requires exactly six tasks")
        for task in tasks:
            entries = tuple(item for item in self.entries.values() if item.task_id == task)
            if len(entries) != 2:
                raise ValueError(f"formal task {task!r} must register exactly two axes")
            for entry in entries:
                source_values = {item.value for item in entry.factors if SOURCE_ROLE in item.roles}
                if len(source_values) != 3 or 1.0 not in source_values:
                    raise ValueError(
                        f"axis {entry.axis_id!r} requires three source values including 1.0"
                    )
                nominal = [
                    item for item in entry.factors
                    if item.value == 1.0 and SOURCE_ROLE in item.roles
                ]
                if not nominal or not any(SAFETY_ROLE in item.roles for item in nominal):
                    raise ValueError(
                        f"axis {entry.axis_id!r} nominal source must allow safety_exact_reference"
                    )
        unknown_tasks = {item.task_id for item in self.entries.values()} - set(tasks)
        if unknown_tasks:
            raise ValueError(f"registry contains tasks outside formal scope: {sorted(unknown_tasks)}")


@dataclass(frozen=True)
class DynamicsOperatorAudit:
    axis_id: str
    operator_id: str
    operator_version: str
    task_id: str
    factor: float
    base_model_digest: str
    shifted_model_digest: str
    changed_leaves: tuple[str, ...]
    unchanged_leaves: tuple[str, ...]
    selected_element_count: int
    changed_element_count: int
    source_object_unchanged: bool
    exact_allowlist: bool
    coupling_check: bool
    finite: bool
    passed: bool
    reason: str | None = None

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-dynamics-operator-audit.v0",
            "axis_id": self.axis_id,
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "task_id": self.task_id,
            "factor": self.factor,
            "base_model_digest": self.base_model_digest,
            "shifted_model_digest": self.shifted_model_digest,
            "changed_leaves": list(self.changed_leaves),
            "unchanged_leaves": list(self.unchanged_leaves),
            "selected_element_count": self.selected_element_count,
            "changed_element_count": self.changed_element_count,
            "source_object_unchanged": self.source_object_unchanged,
            "exact_allowlist": self.exact_allowlist,
            "coupling_check": self.coupling_check,
            "finite": self.finite,
            "passed": self.passed,
            "reason": self.reason,
        }


@runtime_checkable
class DynamicsAxisOperator(Protocol):
    operator_id: str
    operator_version: str

    def supports(self, task_id: str, backend_id: str) -> bool: ...

    def apply(self, nominal_env: Any, factor: float) -> Any: ...

    def audit(self, nominal_env: Any, shifted_env: Any, factor: float) -> DynamicsOperatorAudit: ...


def _model(environment: Any) -> Any:
    if not hasattr(environment, "_mjx_model"):
        raise DynamicsAxisError("environment lacks the reviewed _mjx_model binding")
    return environment._mjx_model


def _array(value: Any) -> np.ndarray:
    try:
        import jax
    except ImportError:
        host = value
    else:
        host = jax.device_get(value)
    result = np.asarray(host)
    if result.dtype.hasobject or result.dtype.kind != "f" or not np.all(np.isfinite(result)):
        raise DynamicsAxisError("model leaf must be a finite floating array")
    return np.ascontiguousarray(result)


def _snapshot_array(value: Any) -> np.ndarray:
    """Host-copy any numerical MJX pytree leaf for whole-model auditing.

    MJX models legitimately contain integer and boolean topology leaves.  They
    must participate in the byte audit even though only floating leaves are
    eligible for registered scaling operators.
    """

    try:
        import jax
    except ImportError:
        host = value
    else:
        host = jax.device_get(value)
    result = np.asarray(host)
    if result.dtype.hasobject or result.dtype.kind not in "biufc":
        raise DynamicsAxisError("model snapshot leaf must be numerical")
    if result.dtype.kind in "fc" and not np.all(np.isfinite(result)):
        raise DynamicsAxisError("model snapshot leaf must be finite")
    return np.ascontiguousarray(result)


def _device_array(value: np.ndarray, like: Any) -> Any:
    if type(like).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp.asarray(value)
    return np.array(value, copy=True)


def _get_leaf(model: Any, name: str) -> Any:
    if not hasattr(model, name):
        raise DynamicsAxisError(f"MJX model lacks registered leaf {name!r}")
    return getattr(model, name)


def _path_token(key: Any) -> str:
    for attribute in ("name", "key", "idx"):
        if hasattr(key, attribute):
            return str(getattr(key, attribute))
    return str(key)


def _generic_model_leaves(value: Any) -> list[tuple[str, np.ndarray]]:
    if hasattr(value, "__dict__"):
        rows: list[tuple[str, np.ndarray]] = []
        for name in sorted(vars(value)):
            if name.startswith("_"):
                continue
            item = getattr(value, name)
            if isinstance(item, (np.ndarray, np.generic, int, float)):
                rows.append((name, np.ascontiguousarray(np.asarray(item))))
        if rows:
            return rows
    raise DynamicsAxisError(f"cannot enumerate model leaves for {type(value)!r}")


def _model_leaves(model: Any) -> list[tuple[str, np.ndarray]]:
    if type(model).__module__.startswith("mujoco.mjx"):
        try:
            import jax
        except ImportError as exc:
            raise DynamicsAxisError("MJX model auditing requires JAX") from exc
        rows, _ = jax.tree_util.tree_flatten_with_path(model)
        return [
            (".".join(_path_token(item) for item in path), _snapshot_array(value))
            for path, value in rows
        ]
    return _generic_model_leaves(model)


def _snapshot(model: Any) -> tuple[str, Mapping[str, str]]:
    records: dict[str, str] = {}
    for path, array in _model_leaves(model):
        if path in records:
            raise DynamicsAxisError(f"duplicate model leaf path: {path}")
        records[path] = sha256_ndarrays({path: array})
    digest = sha256_json(
        {
            "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
            "leaves": [
                {"path": path, "sha256": value}
                for path, value in sorted(records.items())
            ],
        }
    )
    return digest, MappingProxyType(records)


def model_digest(model: Any) -> str:
    """Return the complete content digest used by v0.2 instance bindings."""

    return _snapshot(model)[0]


def model_leaf_digests(model: Any) -> Mapping[str, str]:
    """Return a read-only complete leaf digest map for independent audits."""

    return _snapshot(model)[1]


def _selection_mask(array: np.ndarray, selection: LeafSelection) -> np.ndarray:
    if array.ndim == 0:
        raise DynamicsAxisError(f"registered leaf {selection.leaf!r} is scalar")
    if max(selection.indices) >= array.shape[0]:
        raise DynamicsAxisError(f"selection index exceeds {selection.leaf!r} shape")
    mask = np.zeros(array.shape, dtype=np.bool_)
    if selection.components is None:
        mask[np.asarray(selection.indices, dtype=np.int64)] = True
    else:
        if array.ndim != 2 or max(selection.components) >= array.shape[1]:
            raise DynamicsAxisError(
                f"component selection is incompatible with {selection.leaf!r} shape"
            )
        mask[np.ix_(selection.indices, selection.components)] = True
    if selection.require_nonzero and np.any(array[mask] == 0.0):
        raise DynamicsAxisError(
            f"registered selection in {selection.leaf!r} contains zero values"
        )
    if not np.any(mask):
        raise DynamicsAxisError(f"registered selection in {selection.leaf!r} is empty")
    return mask


class _RegisteredScaleOperator:
    operator_id = ""
    operator_version = "1"

    def __init__(self, entry: AxisRegistryEntry) -> None:
        if entry.operator_id != self.operator_id:
            raise ValueError(
                f"entry operator {entry.operator_id!r} does not match {self.operator_id!r}"
            )
        if entry.operator_version != self.operator_version:
            raise ValueError("operator version mismatch")
        if entry.operator_digest != operator_source_digest():
            raise ValueError("operator source digest mismatch")
        self.entry = entry

    def supports(self, task_id: str, backend_id: str) -> bool:
        return self.entry.task_id == task_id and self.entry.backend_id == backend_id

    def _replacements(self, model: Any, factor: float) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for selection in self.entry.selections:
            original = _get_leaf(model, selection.leaf)
            array = _array(original)
            mask = _selection_mask(array, selection)
            shifted = np.array(array, copy=True)
            shifted[mask] *= np.asarray(factor, dtype=shifted.dtype)
            if not np.all(np.isfinite(shifted)):
                raise DynamicsAxisError("scaled model leaf is non-finite")
            result[selection.leaf] = _device_array(shifted, original)
        return result

    def apply(self, nominal_env: Any, factor: float) -> Any:
        factor_value = _positive_factor(factor)
        model = _model(nominal_env)
        source_before, _ = _snapshot(model)
        tree_replace = getattr(model, "tree_replace", None)
        if not callable(tree_replace):
            raise DynamicsAxisError("MJX model lacks immutable tree_replace")
        shifted_model = tree_replace(self._replacements(model, factor_value))
        if shifted_model is model:
            raise DynamicsAxisError("tree_replace returned the source model in place")
        source_after, _ = _snapshot(model)
        if source_after != source_before:
            raise DynamicsAxisError("operator modified the source model")
        shifted_env = copy.copy(nominal_env)
        if shifted_env is nominal_env:
            raise DynamicsAxisError("environment copy returned the source object")
        shifted_env._mjx_model = shifted_model
        if hasattr(shifted_env, "mjx_model") and shifted_env.mjx_model is not shifted_model:
            raise DynamicsAxisError("shifted environment property did not rebind to the model")
        return shifted_env

    def _coupling_check(
        self,
        before: Mapping[str, np.ndarray],
        after: Mapping[str, np.ndarray],
        factor: float,
    ) -> bool:
        return True

    def audit(self, nominal_env: Any, shifted_env: Any, factor: float) -> DynamicsOperatorAudit:
        factor_value = _positive_factor(factor)
        base_model = _model(nominal_env)
        shifted_model = _model(shifted_env)
        base_digest, base_leaves = _snapshot(base_model)
        shifted_digest, shifted_leaves = _snapshot(shifted_model)
        source_after, _ = _snapshot(base_model)
        if set(base_leaves) != set(shifted_leaves):
            raise DynamicsAxisError("model pytree structure changed")
        changed = tuple(
            path for path in sorted(base_leaves) if base_leaves[path] != shifted_leaves[path]
        )
        unchanged = tuple(path for path in sorted(base_leaves) if path not in changed)
        expected_leaves = tuple(
            sorted(() if factor_value == 1.0 else (item.leaf for item in self.entry.selections))
        )
        exact_allowlist = changed == expected_leaves

        before_arrays = {item.leaf: _array(_get_leaf(base_model, item.leaf)) for item in self.entry.selections}
        after_arrays = {item.leaf: _array(_get_leaf(shifted_model, item.leaf)) for item in self.entry.selections}
        selected_count = 0
        changed_count = 0
        finite = True
        reason: str | None = None
        for selection in self.entry.selections:
            before = before_arrays[selection.leaf]
            after = after_arrays[selection.leaf]
            if before.shape != after.shape or before.dtype != after.dtype:
                reason = f"shape_or_dtype_changed:{selection.leaf}"
                exact_allowlist = False
                continue
            mask = _selection_mask(before, selection)
            selected_count += int(np.count_nonzero(mask))
            changed_count += int(np.count_nonzero(before != after))
            finite &= bool(np.all(np.isfinite(after)))
            if not np.array_equal(before[~mask], after[~mask]):
                reason = f"unselected_elements_changed:{selection.leaf}"
                exact_allowlist = False
            expected = np.array(before, copy=True)
            expected[mask] *= np.asarray(factor_value, dtype=expected.dtype)
            tolerance = float(np.finfo(before.dtype).eps) * 8.0
            if not np.allclose(after, expected, rtol=tolerance, atol=0.0):
                reason = f"selected_scaling_mismatch:{selection.leaf}"
                exact_allowlist = False
        coupling = self._coupling_check(before_arrays, after_arrays, factor_value)
        source_unchanged = source_after == base_digest
        expected_changed_count = 0 if factor_value == 1.0 else selected_count
        count_ok = changed_count == expected_changed_count
        if not count_ok and reason is None:
            reason = "changed_element_count_mismatch"
        passed = bool(
            exact_allowlist
            and coupling
            and source_unchanged
            and finite
            and count_ok
            and reason is None
        )
        return DynamicsOperatorAudit(
            axis_id=self.entry.axis_id,
            operator_id=self.operator_id,
            operator_version=self.operator_version,
            task_id=self.entry.task_id,
            factor=factor_value,
            base_model_digest=base_digest,
            shifted_model_digest=shifted_digest,
            changed_leaves=changed,
            unchanged_leaves=unchanged,
            selected_element_count=selected_count,
            changed_element_count=changed_count,
            source_object_unchanged=source_unchanged,
            exact_allowlist=exact_allowlist,
            coupling_check=coupling,
            finite=finite,
            passed=passed,
            reason=reason,
        )


class MassInertiaScaleOperator(_RegisteredScaleOperator):
    operator_id = MASS_INERTIA_OPERATOR

    def _coupling_check(
        self,
        before: Mapping[str, np.ndarray],
        after: Mapping[str, np.ndarray],
        factor: float,
    ) -> bool:
        selections = {item.leaf: item for item in self.entry.selections}
        for leaf in ("body_mass", "body_inertia"):
            mask = _selection_mask(before[leaf], selections[leaf])
            expected = before[leaf][mask] * factor
            if not np.allclose(after[leaf][mask], expected, rtol=1.0e-6, atol=0.0):
                return False
        return True


class JointDampingScaleOperator(_RegisteredScaleOperator):
    operator_id = JOINT_DAMPING_OPERATOR


class ContactFrictionScaleOperator(_RegisteredScaleOperator):
    operator_id = CONTACT_FRICTION_OPERATOR


class ActuatorGainScaleOperator(_RegisteredScaleOperator):
    operator_id = ACTUATOR_GAIN_OPERATOR


def make_operator(entry: AxisRegistryEntry) -> DynamicsAxisOperator:
    classes = {
        MASS_INERTIA_OPERATOR: MassInertiaScaleOperator,
        JOINT_DAMPING_OPERATOR: JointDampingScaleOperator,
        CONTACT_FRICTION_OPERATOR: ContactFrictionScaleOperator,
        ACTUATOR_GAIN_OPERATOR: ActuatorGainScaleOperator,
    }
    try:
        operator_class = classes[entry.operator_id]
    except KeyError as exc:  # AxisRegistryEntry already rejects this; keep fail closed.
        raise DynamicsAxisError(f"no operator implementation for {entry.operator_id!r}") from exc
    return operator_class(entry)


__all__ = [
    "ACTUATOR_GAIN_OPERATOR",
    "CONFIRMATORY_ROLE",
    "CONTACT_FRICTION_OPERATOR",
    "DEVELOPMENT_ROLE",
    "DynamicsAxisError",
    "DynamicsAxisOperator",
    "DynamicsOperatorAudit",
    "FactorDefinition",
    "FactorRole",
    "JOINT_DAMPING_OPERATOR",
    "LeafSelection",
    "MASS_INERTIA_OPERATOR",
    "MUTATION_STAGE",
    "SAFETY_ROLE",
    "SOURCE_ROLE",
    "ActuatorGainScaleOperator",
    "AxisRegistry",
    "AxisRegistryEntry",
    "ContactFrictionScaleOperator",
    "JointDampingScaleOperator",
    "MassInertiaScaleOperator",
    "make_operator",
    "model_digest",
    "model_leaf_digests",
    "operator_source_digest",
]
