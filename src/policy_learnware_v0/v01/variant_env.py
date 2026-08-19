"""Isolated MuJoCo Playground dynamics variants for the v0.1 experiment.

The production operator in this module is deliberately pinned to Playground
0.0.5.  It has one, and only one, mutation path::

    environment._mjx_model.dof_damping

The replacement is functional (``tree_replace``), is applied before either
``reset`` or ``step`` is JIT compiled, and is followed by a complete pytree
leaf audit.  There are intentionally no ``sys``/``model``/``mjx_model``
fallback mutation paths: a runtime that does not match the audited server
layout fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import importlib.metadata
import inspect
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..envs.mujoco_playground import (
    MujocoPlaygroundEnvAdapter,
    MujocoPlaygroundUnavailableError,
    mujoco_playground_package_version,
)
from ..hashing import sha256_file, sha256_json, sha256_ndarrays
from ..schemas import EnvSchema


SHIFT_ID = "global_nonzero_dof_damping_scale"
PINNED_PLAYGROUND_VERSION = "0.0.5"
PINNED_MUJOCO_VERSION = "3.3.6"
PINNED_JAX_VERSION = "0.7.2"
ALLOWLISTED_ENV_LEAF = "_mjx_model.dof_damping"
MODEL_LEAF = "dof_damping"
MUTATION_STAGE = "post_registry_load_pre_jit"

_PINNED_TASK_CLASSES = MappingProxyType(
    {
        "WalkerWalk": (
            "mujoco_playground._src.dm_control_suite.walker",
            "PlanarWalker",
        ),
        "FingerTurnEasy": (
            "mujoco_playground._src.dm_control_suite.finger",
            "Turn",
        ),
    }
)


class VariantEnvironmentError(RuntimeError):
    """The live runtime cannot satisfy the frozen v0.1 shift contract."""


@dataclass(frozen=True)
class ModelLeafDigest:
    """Exact byte digest for one model-pytree leaf."""

    path: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ModelSnapshot:
    """Hash-addressed snapshot of every dynamic pytree leaf."""

    model_type: str
    leaves: tuple[ModelLeafDigest, ...]
    digest: str

    def __post_init__(self) -> None:
        paths = tuple(item.path for item in self.leaves)
        if not self.model_type or not self.leaves:
            raise ValueError("model snapshot requires a type and at least one leaf")
        if len(paths) != len(set(paths)) or paths != tuple(sorted(paths)):
            raise ValueError("model snapshot paths must be unique and sorted")
        expected = sha256_json(
            {
                "model_type": self.model_type,
                "leaves": [item.to_dict() for item in self.leaves],
            }
        )
        if self.digest != expected:
            raise ValueError("model snapshot digest mismatch")

    @property
    def by_path(self) -> Mapping[str, ModelLeafDigest]:
        return MappingProxyType({item.path: item for item in self.leaves})


@dataclass(frozen=True)
class ModelDiffAudit:
    """Evidence that the approved operator touched no unregistered leaf."""

    shift_id: str
    factor: float
    allowlisted_model_leaf: str
    base_model_digest: str
    shifted_model_digest: str
    changed_leaves: tuple[str, ...]
    changed_index_count: int
    nominal_nonzero_count: int
    before_leaf_digest: str
    after_leaf_digest: str
    source_unchanged: bool
    shape: tuple[int, ...]
    dtype: str
    operator_digest: str
    schema: str = "policy-learnware.v01-model-diff-audit.v0"

    def __post_init__(self) -> None:
        if self.shift_id != SHIFT_ID:
            raise ValueError("unsupported shift id in model-diff audit")
        if self.allowlisted_model_leaf != ALLOWLISTED_ENV_LEAF:
            raise ValueError("model-diff audit has an unapproved model leaf")
        if not math.isfinite(float(self.factor)) or float(self.factor) <= 0.0:
            raise ValueError("model-diff factor must be finite and positive")
        if self.changed_index_count < 0 or self.nominal_nonzero_count <= 0:
            raise ValueError("invalid model-diff index counts")
        if self.changed_index_count > self.nominal_nonzero_count:
            raise ValueError("changed indices exceed nominal nonzero damping entries")
        if not self.source_unchanged:
            raise ValueError("the source model was modified in place")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "shift_id": self.shift_id,
            "factor": float(self.factor),
            "allowlisted_model_leaf": self.allowlisted_model_leaf,
            "base_model_digest": self.base_model_digest,
            "shifted_model_digest": self.shifted_model_digest,
            "changed_leaves": list(self.changed_leaves),
            "changed_index_count": int(self.changed_index_count),
            "nominal_nonzero_count": int(self.nominal_nonzero_count),
            "before_leaf_digest": self.before_leaf_digest,
            "after_leaf_digest": self.after_leaf_digest,
            "source_unchanged": bool(self.source_unchanged),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "operator_digest": self.operator_digest,
        }


@dataclass(frozen=True)
class TrajectoryIdentityAudit:
    episode_count: int
    steps_per_episode: int
    schema_identity: bool
    action_identity: bool
    flag_identity: bool
    observation_within_tolerance: bool
    reward_within_tolerance: bool
    maximum_observation_absolute_error: float
    maximum_reward_absolute_error: float
    trajectory_atol: float
    trajectory_rtol: float
    reward_atol: float
    passed: bool
    reason: str | None = None
    schema: str = "policy-learnware.v01-trajectory-identity-audit.v0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_count": self.episode_count,
            "steps_per_episode": self.steps_per_episode,
            "schema_identity": self.schema_identity,
            "action_identity": self.action_identity,
            "flag_identity": self.flag_identity,
            "observation_within_tolerance": self.observation_within_tolerance,
            "reward_within_tolerance": self.reward_within_tolerance,
            "maximum_observation_absolute_error": self.maximum_observation_absolute_error,
            "maximum_reward_absolute_error": self.maximum_reward_absolute_error,
            "trajectory_atol": self.trajectory_atol,
            "trajectory_rtol": self.trajectory_rtol,
            "reward_atol": self.reward_atol,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FiniteTerminationAudit:
    episode_count: int
    steps_per_episode: int
    all_finite: bool
    no_early_termination: bool
    reward_minimum: float | None
    reward_maximum: float | None
    passed: bool
    reason: str | None = None
    schema: str = "policy-learnware.v01-finite-termination-audit.v0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "episode_count": self.episode_count,
            "steps_per_episode": self.steps_per_episode,
            "all_finite": self.all_finite,
            "no_early_termination": self.no_early_termination,
            "reward_minimum": self.reward_minimum,
            "reward_maximum": self.reward_maximum,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FiveFactorFiniteAudit:
    factors: tuple[float, ...]
    results: Mapping[str, FiniteTerminationAudit]
    passed: bool
    schema: str = "policy-learnware.v01-five-factor-finite-audit.v0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "factors": list(self.factors),
            "results": {
                key: value.to_dict() for key, value in sorted(self.results.items())
            },
            "passed": self.passed,
        }


@dataclass(frozen=True)
class InstanceIsolationAudit:
    factor_sequence: tuple[float, ...]
    fresh_environment_objects: bool
    base_model_identity: bool
    schema_identity: bool
    nominal_model_identity: bool
    shifted_models_distinct: bool
    nominal_trajectory_identity: TrajectoryIdentityAudit
    passed: bool
    schema: str = "policy-learnware.v01-instance-isolation-audit.v0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "factor_sequence": list(self.factor_sequence),
            "fresh_environment_objects": self.fresh_environment_objects,
            "base_model_identity": self.base_model_identity,
            "schema_identity": self.schema_identity,
            "nominal_model_identity": self.nominal_model_identity,
            "shifted_models_distinct": self.shifted_models_distinct,
            "nominal_trajectory_identity": self.nominal_trajectory_identity.to_dict(),
            "passed": self.passed,
        }


def _model_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _path_token(key: Any) -> str:
    """Normalise JAX KeyPath entries without depending on their repr."""

    if hasattr(key, "name"):
        return str(key.name)
    if hasattr(key, "key"):
        return str(key.key)
    if hasattr(key, "idx"):
        return str(key.idx)
    return str(key)


def _generic_leaves(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    """Small dataclass/mapping flattener used by dependency-light unit tests."""

    if isinstance(value, np.ndarray) or isinstance(value, np.generic):
        return [(".".join(prefix) or "<root>", value)]
    if isinstance(value, (bool, int, float)):
        return [(".".join(prefix) or "<root>", np.asarray(value))]
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
            if name.startswith("_"):
                continue
            result.extend(_generic_leaves(getattr(value, name), prefix + (name,)))
        if result:
            return result
    raise TypeError(f"cannot flatten model leaf of type {_model_type(value)}")


def _pytree_leaves(model: Any) -> list[tuple[str, Any]]:
    """Return the exact JAX pytree leaves for MJX, with a test-only fallback."""

    module = type(model).__module__
    if module.startswith("mujoco.mjx"):
        try:
            import jax
        except ImportError as error:  # pragma: no cover - production dependency gate
            raise VariantEnvironmentError("MJX model auditing requires JAX") from error
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
    if array.dtype.hasobject:
        raise TypeError("model pytree contains an object-dtype leaf")
    return np.ascontiguousarray(array)


def snapshot_model(model: Any) -> ModelSnapshot:
    """Hash all model pytree leaves, including dtype, shape, path and bytes."""

    records: list[ModelLeafDigest] = []
    for path, value in _pytree_leaves(model):
        array = _host_array(value)
        records.append(
            ModelLeafDigest(
                path=path,
                dtype=array.dtype.str,
                shape=tuple(int(item) for item in array.shape),
                sha256=sha256_ndarrays({path: array}),
            )
        )
    records.sort(key=lambda item: item.path)
    payload = {
        "model_type": _model_type(model),
        "leaves": [item.to_dict() for item in records],
    }
    return ModelSnapshot(
        model_type=payload["model_type"],
        leaves=tuple(records),
        digest=sha256_json(payload),
    )


def damping_operator_digest() -> str:
    """Digest the complete isolated operator module bound by ShiftRegistry."""

    return sha256_file(Path(__file__))


def _scaled_damping(damping: Any, factor: float) -> Any:
    before = _host_array(damping)
    if before.ndim != 1 or before.size == 0:
        raise VariantEnvironmentError("dof_damping must be a non-empty vector")
    if before.dtype.kind != "f":
        raise VariantEnvironmentError("dof_damping must have floating dtype")
    if not np.all(np.isfinite(before)):
        raise VariantEnvironmentError("dof_damping contains non-finite values")
    if not np.any(before != 0):
        raise VariantEnvironmentError("dof_damping has no nonzero entry to scale")

    factor_value = np.asarray(factor, dtype=before.dtype)
    if type(damping).__module__.startswith(("jax", "jaxlib")):
        import jax.numpy as jnp

        return jnp.where(damping != 0, damping * factor_value, damping)
    return np.where(before != 0, before * factor_value, before).astype(
        before.dtype, copy=False
    )


def apply_global_nonzero_dof_damping_scale(
    model: Any,
    factor: float,
) -> tuple[Any, ModelDiffAudit]:
    """Return an audited model with every originally nonzero damping scaled.

    This function never mutates ``model``.  The object must implement MJX's
    immutable ``tree_replace`` interface.  Lightweight test doubles may expose
    the same method, but the production factory additionally pins the concrete
    Playground classes and package versions.
    """

    if isinstance(factor, bool) or not isinstance(factor, (int, float, np.number)):
        raise ValueError("shift factor must be numeric")
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("shift factor must be finite and positive")
    if not hasattr(model, MODEL_LEAF):
        raise VariantEnvironmentError("pinned MJX model lacks dof_damping")
    tree_replace = getattr(model, "tree_replace", None)
    if not callable(tree_replace):
        raise VariantEnvironmentError("pinned MJX model lacks tree_replace")

    base = snapshot_model(model)
    before = _host_array(getattr(model, MODEL_LEAF))
    before_leaf_digest = sha256_ndarrays({MODEL_LEAF: before})
    replacement = _scaled_damping(getattr(model, MODEL_LEAF), factor)
    shifted_model = tree_replace({MODEL_LEAF: replacement})
    if shifted_model is model:
        raise VariantEnvironmentError("tree_replace returned the source model in place")
    shifted = snapshot_model(shifted_model)
    source_after = snapshot_model(model)

    base_paths = set(base.by_path)
    shifted_paths = set(shifted.by_path)
    if base_paths != shifted_paths:
        raise VariantEnvironmentError("model pytree structure changed during shift")
    changed = tuple(
        sorted(
            path
            for path in base_paths
            if base.by_path[path] != shifted.by_path[path]
        )
    )
    expected_changed = () if factor == 1.0 else (MODEL_LEAF,)
    if changed != expected_changed:
        raise VariantEnvironmentError(
            "model diff touched unexpected leaves: "
            f"expected={expected_changed}, observed={changed}"
        )

    after = _host_array(getattr(shifted_model, MODEL_LEAF))
    if after.shape != before.shape or after.dtype != before.dtype:
        raise VariantEnvironmentError("dof_damping shape or dtype changed")
    zero_mask = before == 0
    if not np.array_equal(after[zero_mask], before[zero_mask]):
        raise VariantEnvironmentError("zero damping entries changed")
    expected = np.where(
        zero_mask,
        before,
        before * np.asarray(factor, dtype=before.dtype),
    ).astype(before.dtype, copy=False)
    tolerance = float(np.finfo(before.dtype).eps) * 4.0
    if not np.allclose(after, expected, rtol=tolerance, atol=0.0):
        raise VariantEnvironmentError("nonzero damping entries were not scaled correctly")
    changed_indices = int(np.count_nonzero(before != after))
    nominal_nonzero = int(np.count_nonzero(~zero_mask))
    expected_changed_indices = 0 if factor == 1.0 else nominal_nonzero
    if changed_indices != expected_changed_indices:
        raise VariantEnvironmentError("unexpected number of changed damping indices")
    source_unchanged = source_after.digest == base.digest
    if not source_unchanged:
        raise VariantEnvironmentError("shift operator modified the source model")

    return shifted_model, ModelDiffAudit(
        shift_id=SHIFT_ID,
        factor=factor,
        allowlisted_model_leaf=ALLOWLISTED_ENV_LEAF,
        base_model_digest=base.digest,
        shifted_model_digest=shifted.digest,
        changed_leaves=tuple(f"_mjx_model.{path}" for path in changed),
        changed_index_count=changed_indices,
        nominal_nonzero_count=nominal_nonzero,
        before_leaf_digest=before_leaf_digest,
        after_leaf_digest=sha256_ndarrays({MODEL_LEAF: after}),
        source_unchanged=source_unchanged,
        shape=tuple(int(item) for item in before.shape),
        dtype=before.dtype.str,
        operator_digest=damping_operator_digest(),
    )


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise VariantEnvironmentError(
            f"required runtime distribution is unavailable: {distribution}"
        ) from error


def _verify_pinned_runtime() -> dict[str, str]:
    versions = {
        "playground": mujoco_playground_package_version() or "unavailable",
        "mujoco": _package_version("mujoco"),
        "jax": _package_version("jax"),
    }
    expected = {
        "playground": PINNED_PLAYGROUND_VERSION,
        "mujoco": PINNED_MUJOCO_VERSION,
        "jax": PINNED_JAX_VERSION,
    }
    mismatches = {
        name: (versions[name], required)
        for name, required in expected.items()
        if versions[name] != required
    }
    if mismatches:
        detail = ", ".join(
            f"{name}={actual!r}, expected {required!r}"
            for name, (actual, required) in sorted(mismatches.items())
        )
        raise VariantEnvironmentError(f"pinned variant runtime mismatch: {detail}")
    return versions


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _serializable(value.to_dict())
    if hasattr(value, "items"):
        return {str(key): _serializable(item) for key, item in value.items()}
    return repr(value)


class VariantEnvAdapter(MujocoPlaygroundEnvAdapter):
    """The normal v0 adapter API backed by one audited shifted MJX model."""

    def __init__(
        self,
        *,
        task: str,
        factor: float,
        variant_id: str,
        expected_horizon: int,
        expected_action_repeat: int,
        jit: bool,
        expected_operator_digest: str | None,
        shift_manifest_digest: str,
    ) -> None:
        versions = _verify_pinned_runtime()
        try:
            import jax
            from mujoco_playground import registry
        except ImportError as error:  # pragma: no cover - server dependency gate
            raise MujocoPlaygroundUnavailableError(
                "VariantEnvAdapter requires JAX and mujoco_playground"
            ) from error
        expected_class = _PINNED_TASK_CLASSES.get(task)
        if expected_class is None:
            raise VariantEnvironmentError(f"task {task!r} is not approved for v0.1")

        self._jax = jax
        self._registry = registry
        self._task = task
        self._variant_id = str(variant_id)
        if not self._variant_id:
            raise ValueError("variant_id must be non-empty")
        self._registry_config_object = registry.get_default_config(task)
        self._env = registry.load(task, config=self._registry_config_object)
        actual_class = (type(self._env).__module__, type(self._env).__qualname__)
        if actual_class != expected_class:
            raise VariantEnvironmentError(
                f"pinned environment class mismatch: {actual_class} != {expected_class}"
            )
        if not hasattr(self._env, "_mjx_model"):
            raise VariantEnvironmentError("pinned environment lacks _mjx_model")
        if not hasattr(type(self._env), "mjx_model"):
            raise VariantEnvironmentError("pinned environment lacks mjx_model property")
        if self._env.mjx_model is not self._env._mjx_model:
            raise VariantEnvironmentError("mjx_model property is not bound to _mjx_model")

        shifted_model, audit = apply_global_nonzero_dof_damping_scale(
            self._env._mjx_model, factor
        )
        if expected_operator_digest is not None and audit.operator_digest != str(
            expected_operator_digest
        ):
            raise VariantEnvironmentError("registered shift operator digest mismatch")
        self._env._mjx_model = shifted_model
        if self._env.mjx_model is not shifted_model:
            raise VariantEnvironmentError("shifted model rebind did not take effect")
        self._model_diff_audit = audit
        self._runtime_versions = MappingProxyType(versions)
        self._shift_manifest_digest = str(shift_manifest_digest)

        # No reset/step compilation is permitted before the audited rebind above.
        self._reset_fn = jax.jit(self._env.reset) if jit else self._env.reset
        self._step_fn = jax.jit(self._env.step) if jit else self._env.step
        state = self._reset_fn(self._key(0))
        observation = self._flat_observation(state.obs)
        action_dim = int(self._env.mjx_model.nu)
        ranges = np.asarray(
            jax.device_get(self._env.mjx_model.actuator_ctrlrange), dtype=np.float32
        )
        if ranges.shape != (action_dim, 2):
            raise VariantEnvironmentError("pinned actuator_ctrlrange shape changed")
        horizon = int(self._config_value("episode_length", default=-1))
        action_repeat = int(self._config_value("action_repeat", default=-1))
        if horizon != int(expected_horizon):
            raise VariantEnvironmentError(
                f"registry horizon {horizon} != expected {expected_horizon}"
            )
        if action_repeat != int(expected_action_repeat):
            raise VariantEnvironmentError(
                f"registry action_repeat {action_repeat} != expected {expected_action_repeat}"
            )
        control_dt = float(self._env.dt)
        if not math.isfinite(control_dt) or control_dt <= 0.0:
            raise VariantEnvironmentError("pinned environment has invalid control dt")
        source_path = inspect.getsourcefile(type(self._env))
        implementation_digest = (
            sha256_file(source_path)
            if source_path is not None and Path(source_path).is_file()
            else sha256_json(
                {
                    "module": type(self._env).__module__,
                    "class": type(self._env).__qualname__,
                }
            )
        )
        native_observation = np.asarray(jax.device_get(state.obs))
        flatten_fingerprint = sha256_json(
            {
                "rule": "np.asarray(state.obs).reshape(-1,C)-v0",
                "native_shape": list(native_observation.shape),
                "flat_dim": int(observation.size),
                "dtype": str(observation.dtype),
            }
        )
        self._schema = EnvSchema(
            backend=self.backend,
            task=task,
            observation_dim=int(observation.size),
            action_dim=action_dim,
            action_low=ranges[:, 0],
            action_high=ranges[:, 1],
            horizon=horizon,
            action_repeat=action_repeat,
            control_dt=control_dt,
            flatten_fingerprint=flatten_fingerprint,
            implementation_digest=implementation_digest,
            observation_dtype=str(observation.dtype),
            action_dtype="float32",
        )

    @property
    def variant_id(self) -> str:
        return self._variant_id

    @property
    def model_diff_audit(self) -> ModelDiffAudit:
        return self._model_diff_audit

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        return self._runtime_versions

    @property
    def measurement_schema_view(self) -> Any:
        from .schemas import MeasurementSchemaView

        return MeasurementSchemaView.from_env_schema(self.schema)

    def create_instance_record(
        self, *, finite_termination_audit_summary: Mapping[str, Any]
    ) -> Any:
        """Build the private instance record after Gate-0 finite checks run."""

        from .schemas import EnvironmentInstanceRecord

        audit = self.model_diff_audit
        return EnvironmentInstanceRecord.create(
            variant_id=self.variant_id,
            env_schema_digest=self.schema.digest,
            measurement_schema_view_digest=self.measurement_schema_view.digest,
            shift_manifest_digest=self._shift_manifest_digest,
            base_model_digest=audit.base_model_digest,
            shifted_model_digest=audit.shifted_model_digest,
            changed_leaf=ALLOWLISTED_ENV_LEAF,
            changed_index_count=audit.changed_index_count,
            before_leaf_digest=audit.before_leaf_digest,
            after_leaf_digest=audit.after_leaf_digest,
            operator_digest=audit.operator_digest,
            runtime_versions=dict(self.runtime_versions),
            finite_termination_audit_summary=dict(finite_termination_audit_summary),
        )


def _rollout_audit_inputs(
    adapter: Any,
    reset_seeds: Sequence[int],
    action_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = np.asarray(reset_seeds, dtype=np.int64)
    actions = np.asarray(action_tensor, dtype=np.float32)
    if seeds.ndim != 1 or seeds.size == 0 or np.any(seeds < 0):
        raise ValueError("reset_seeds must be a non-empty nonnegative vector")
    expected = (seeds.size, actions.shape[1], int(adapter.schema.action_dim)) if actions.ndim == 3 else None
    if actions.ndim != 3 or actions.shape != expected or actions.shape[1] <= 0:
        raise ValueError(
            "action_tensor must have shape [len(reset_seeds),steps,action_dim]"
        )
    if actions.shape[1] > int(adapter.schema.horizon):
        raise ValueError("audit action tensor exceeds the registered horizon")
    if not np.all(np.isfinite(actions)):
        raise ValueError("audit action tensor contains non-finite values")
    low = np.asarray(adapter.schema.action_low, dtype=np.float32)
    high = np.asarray(adapter.schema.action_high, dtype=np.float32)
    if np.any(actions < low) or np.any(actions > high):
        raise ValueError("audit action tensor exceeds registered action bounds")
    return seeds, np.array(actions, copy=True)


def audit_trajectory_identity(
    nominal_adapter: Any,
    factor_one_adapter: Any,
    *,
    reset_seeds: Sequence[int],
    action_tensor: np.ndarray,
    trajectory_atol: float = 1.0e-6,
    trajectory_rtol: float = 1.0e-6,
    reward_atol: float = 1.0e-6,
) -> TrajectoryIdentityAudit:
    """Compare two adapters under exactly the same seeds and action tensor."""

    for name, value in (
        ("trajectory_atol", trajectory_atol),
        ("trajectory_rtol", trajectory_rtol),
        ("reward_atol", reward_atol),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    seeds, actions = _rollout_audit_inputs(
        nominal_adapter, reset_seeds, action_tensor
    )
    _rollout_audit_inputs(factor_one_adapter, reset_seeds, action_tensor)
    left_schema = getattr(nominal_adapter.schema, "digest", None)
    right_schema = getattr(factor_one_adapter.schema, "digest", None)
    schema_identity = bool(left_schema == right_schema)
    action_identity = True
    flag_identity = True
    observation_within = True
    reward_within = True
    max_observation_error = 0.0
    max_reward_error = 0.0
    reason: str | None = None
    try:
        for episode, seed in enumerate(seeds):
            left_state, left_observation = nominal_adapter.reset(int(seed))
            right_state, right_observation = factor_one_adapter.reset(int(seed))
            left_observation = np.asarray(left_observation, dtype=np.float32)
            right_observation = np.asarray(right_observation, dtype=np.float32)
            if left_observation.shape != right_observation.shape:
                observation_within = False
                reason = "reset_observation_shape_mismatch"
                break
            reset_error = float(
                np.max(np.abs(left_observation - right_observation), initial=0.0)
            )
            max_observation_error = max(max_observation_error, reset_error)
            observation_within &= bool(
                np.allclose(
                    left_observation,
                    right_observation,
                    atol=trajectory_atol,
                    rtol=trajectory_rtol,
                )
            )
            for action in actions[episode]:
                frozen_action = np.array(action, copy=True)
                left_state, left = nominal_adapter.step(
                    left_state, np.array(frozen_action, copy=True)
                )
                right_state, right = factor_one_adapter.step(
                    right_state, np.array(frozen_action, copy=True)
                )
                action_identity &= bool(np.array_equal(action, frozen_action))
                left_obs = np.asarray(left.observation, dtype=np.float32)
                right_obs = np.asarray(right.observation, dtype=np.float32)
                if left_obs.shape != right_obs.shape:
                    observation_within = False
                    reason = "step_observation_shape_mismatch"
                    break
                observation_error = float(
                    np.max(np.abs(left_obs - right_obs), initial=0.0)
                )
                reward_error = abs(float(left.reward) - float(right.reward))
                max_observation_error = max(max_observation_error, observation_error)
                max_reward_error = max(max_reward_error, reward_error)
                observation_within &= bool(
                    np.allclose(
                        left_obs,
                        right_obs,
                        atol=trajectory_atol,
                        rtol=trajectory_rtol,
                    )
                )
                reward_within &= reward_error <= float(reward_atol)
                flag_identity &= (
                    bool(left.terminated) == bool(right.terminated)
                    and bool(left.truncated) == bool(right.truncated)
                )
            if reason is not None:
                break
    except Exception as error:  # Gate-0 must persist a diagnostic before blocking.
        reason = f"{type(error).__name__}: {error}"
        observation_within = False
        reward_within = False
        flag_identity = False
    passed = bool(
        schema_identity
        and action_identity
        and flag_identity
        and observation_within
        and reward_within
        and reason is None
    )
    return TrajectoryIdentityAudit(
        episode_count=int(seeds.size),
        steps_per_episode=int(actions.shape[1]),
        schema_identity=schema_identity,
        action_identity=action_identity,
        flag_identity=flag_identity,
        observation_within_tolerance=observation_within,
        reward_within_tolerance=reward_within,
        maximum_observation_absolute_error=max_observation_error,
        maximum_reward_absolute_error=max_reward_error,
        trajectory_atol=float(trajectory_atol),
        trajectory_rtol=float(trajectory_rtol),
        reward_atol=float(reward_atol),
        passed=passed,
        reason=reason,
    )


def audit_finite_termination(
    adapter: Any,
    *,
    reset_seeds: Sequence[int],
    action_tensor: np.ndarray,
) -> FiniteTerminationAudit:
    """Check all states/rewards and reject termination before supplied horizon."""

    seeds, actions = _rollout_audit_inputs(adapter, reset_seeds, action_tensor)
    all_finite = True
    no_early_termination = True
    rewards: list[float] = []
    reason: str | None = None
    try:
        for episode, seed in enumerate(seeds):
            state, observation = adapter.reset(int(seed))
            all_finite &= bool(np.all(np.isfinite(np.asarray(observation))))
            for step_index, action in enumerate(actions[episode]):
                state, result = adapter.step(state, np.array(action, copy=True))
                all_finite &= bool(
                    np.all(np.isfinite(np.asarray(result.observation)))
                    and math.isfinite(float(result.reward))
                )
                rewards.append(float(result.reward))
                if (
                    bool(result.terminated) or bool(result.truncated)
                ) and step_index + 1 < actions.shape[1]:
                    no_early_termination = False
    except Exception as error:  # Persist failure evidence for Gate 0.
        reason = f"{type(error).__name__}: {error}"
        all_finite = False
        no_early_termination = False
    passed = bool(all_finite and no_early_termination and reason is None)
    return FiniteTerminationAudit(
        episode_count=int(seeds.size),
        steps_per_episode=int(actions.shape[1]),
        all_finite=all_finite,
        no_early_termination=no_early_termination,
        reward_minimum=(None if not rewards else float(min(rewards))),
        reward_maximum=(None if not rewards else float(max(rewards))),
        passed=passed,
        reason=reason,
    )


def audit_five_factor_finite(
    adapters_by_factor: Mapping[float, Any],
    *,
    reset_seeds: Sequence[int],
    action_tensor: np.ndarray,
) -> FiveFactorFiniteAudit:
    """Run the frozen five-factor finite/termination audit for one task."""

    expected = (0.5, 0.75, 1.0, 1.5, 2.0)
    observed = tuple(sorted(float(value) for value in adapters_by_factor))
    if observed != expected:
        raise ValueError(f"five-factor audit requires {expected}, got {observed}")
    results = {
        format(factor, ".17g"): audit_finite_termination(
            adapters_by_factor[factor],
            reset_seeds=reset_seeds,
            action_tensor=action_tensor,
        )
        for factor in expected
    }
    return FiveFactorFiniteAudit(
        factors=expected,
        results=results,
        passed=all(item.passed for item in results.values()),
    )


def audit_instance_isolation(
    ordered_instances: Sequence[tuple[float, Any]],
    *,
    reset_seeds: Sequence[int],
    action_tensor: np.ndarray,
    trajectory_atol: float = 1.0e-6,
    trajectory_rtol: float = 1.0e-6,
    reward_atol: float = 1.0e-6,
) -> InstanceIsolationAudit:
    """Audit ``nominal → 0.5 → 2.0 → nominal`` fresh-instance isolation."""

    expected = (1.0, 0.5, 2.0, 1.0)
    factors = tuple(float(factor) for factor, _ in ordered_instances)
    if factors != expected:
        raise ValueError(f"isolation audit requires factor sequence {expected}")
    adapters = tuple(adapter for _, adapter in ordered_instances)
    fresh = len({id(adapter.environment) for adapter in adapters}) == len(adapters)
    audits = tuple(adapter.model_diff_audit for adapter in adapters)
    base_identity = len({audit.base_model_digest for audit in audits}) == 1
    schema_identity = len({adapter.schema.digest for adapter in adapters}) == 1
    nominal_model_identity = bool(
        audits[0].base_model_digest
        == audits[0].shifted_model_digest
        == audits[3].base_model_digest
        == audits[3].shifted_model_digest
    )
    shifted_distinct = bool(
        audits[1].shifted_model_digest != audits[0].shifted_model_digest
        and audits[2].shifted_model_digest != audits[0].shifted_model_digest
        and audits[1].shifted_model_digest != audits[2].shifted_model_digest
    )
    trajectory = audit_trajectory_identity(
        adapters[0],
        adapters[3],
        reset_seeds=reset_seeds,
        action_tensor=action_tensor,
        trajectory_atol=trajectory_atol,
        trajectory_rtol=trajectory_rtol,
        reward_atol=reward_atol,
    )
    passed = bool(
        fresh
        and base_identity
        and schema_identity
        and nominal_model_identity
        and shifted_distinct
        and trajectory.passed
    )
    return InstanceIsolationAudit(
        factor_sequence=factors,
        fresh_environment_objects=fresh,
        base_model_identity=base_identity,
        schema_identity=schema_identity,
        nominal_model_identity=nominal_model_identity,
        shifted_models_distinct=shifted_distinct,
        nominal_trajectory_identity=trajectory,
        passed=passed,
    )


class VariantEnvFactory:
    """Closed factory accepting only a validated registry manifest."""

    def __init__(
        self,
        shift_registry: Any | None = None,
        *,
        expected_base_protocol_id: str | None = None,
    ) -> None:
        if shift_registry is None:
            from .registry import default_shift_registry

            shift_registry = default_shift_registry()
        self._shift_registry = shift_registry
        self._expected_base_protocol_id = (
            None
            if expected_base_protocol_id is None
            else str(expected_base_protocol_id)
        )

    def create(
        self,
        *,
        task: str,
        shift_manifest: Any,
        variant_id: str,
        expected_horizon: int = 1000,
        expected_action_repeat: int = 1,
        jit: bool = True,
    ) -> VariantEnvAdapter:
        manifest_task = str(getattr(shift_manifest, "task", ""))
        manifest_shift = str(getattr(shift_manifest, "shift_id", ""))
        try:
            manifest_factor = float(getattr(shift_manifest, "factor"))
        except (TypeError, ValueError) as error:
            raise VariantEnvironmentError("ShiftManifest has no valid factor") from error
        if manifest_task != task:
            raise VariantEnvironmentError("task differs from ShiftManifest")
        if manifest_shift != SHIFT_ID:
            raise VariantEnvironmentError("ShiftManifest uses an unsupported shift")
        entry = self._shift_registry.require(manifest_shift, task, manifest_factor)
        manifest_registry_digest = str(
            getattr(shift_manifest, "registry_digest", "")
        )
        if manifest_registry_digest != str(getattr(entry, "registry_digest", "")):
            raise VariantEnvironmentError("ShiftManifest registry digest mismatch")
        manifest_protocol_id = str(
            getattr(shift_manifest, "base_protocol_id", "")
        )
        if (
            self._expected_base_protocol_id is not None
            and manifest_protocol_id != self._expected_base_protocol_id
        ):
            raise VariantEnvironmentError("ShiftManifest base protocol mismatch")
        backend = str(getattr(entry, "backend", ""))
        if backend != MujocoPlaygroundEnvAdapter.backend:
            raise VariantEnvironmentError("ShiftRegistry backend mismatch")
        allowlisted = str(getattr(entry, "allowlisted_model_leaf", ""))
        if allowlisted != ALLOWLISTED_ENV_LEAF:
            raise VariantEnvironmentError("ShiftRegistry model leaf mismatch")
        stage = str(getattr(entry, "mutation_stage", ""))
        if stage != MUTATION_STAGE:
            raise VariantEnvironmentError("ShiftRegistry mutation stage mismatch")
        selection_rule = str(getattr(entry, "selection_rule", ""))
        if selection_rule != "original_value_nonzero":
            raise VariantEnvironmentError("ShiftRegistry selection rule mismatch")
        expected_digest = getattr(entry, "operator_source_sha256", None)
        return VariantEnvAdapter(
            task=task,
            factor=manifest_factor,
            variant_id=variant_id,
            expected_horizon=expected_horizon,
            expected_action_repeat=expected_action_repeat,
            jit=jit,
            expected_operator_digest=(
                None if expected_digest in (None, "") else str(expected_digest)
            ),
            shift_manifest_digest=str(getattr(shift_manifest, "digest", "")),
        )


def create_variant_env(**kwargs: Any) -> VariantEnvAdapter:
    """Convenience wrapper used by orchestration code."""

    return VariantEnvFactory().create(**kwargs)


__all__ = [
    "ALLOWLISTED_ENV_LEAF",
    "FiniteTerminationAudit",
    "FiveFactorFiniteAudit",
    "InstanceIsolationAudit",
    "MODEL_LEAF",
    "MUTATION_STAGE",
    "ModelDiffAudit",
    "ModelLeafDigest",
    "ModelSnapshot",
    "PINNED_JAX_VERSION",
    "PINNED_MUJOCO_VERSION",
    "PINNED_PLAYGROUND_VERSION",
    "SHIFT_ID",
    "TrajectoryIdentityAudit",
    "VariantEnvAdapter",
    "VariantEnvFactory",
    "VariantEnvironmentError",
    "apply_global_nonzero_dof_damping_scale",
    "audit_finite_termination",
    "audit_five_factor_finite",
    "audit_instance_isolation",
    "audit_trajectory_identity",
    "create_variant_env",
    "damping_operator_digest",
    "snapshot_model",
]
