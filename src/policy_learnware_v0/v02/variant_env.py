"""Fresh v0.2 variant construction and dependency-light Gate-0 audits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json
from .axes import (
    AxisRegistry,
    DynamicsOperatorAudit,
    FactorRole,
    SAFETY_ROLE,
    SOURCE_ROLE,
    make_operator,
)
from .schemas import (
    AxisAnchorBinding,
    EnvironmentInstanceRecord,
    SourceAnchorRecord,
    canonical_array_digest,
    canonical_model_diff_projection,
    canonical_model_snapshot,
    derive_live_model_diff,
)


class VariantEnvironmentError(RuntimeError):
    """A fresh-instance, identity, runtime, or rollout contract failed."""


class NativeAdapterFactory(Protocol):
    def __call__(self, native_env: Any, task_id: str, *, jit: bool) -> Any: ...


def _digest(value: str, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{where} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a SHA-256 digest") from exc
    return value.lower()


def _schema_contract(schema: Any) -> Mapping[str, Any]:
    fields = (
        "backend",
        "task",
        "observation_dim",
        "action_dim",
        "horizon",
        "action_repeat",
        "control_dt",
        "flatten_fingerprint",
        "implementation_digest",
        "observation_dtype",
        "action_dtype",
    )
    payload: dict[str, Any] = {}
    for field in fields:
        if not hasattr(schema, field):
            raise VariantEnvironmentError(f"environment schema lacks {field!r}")
        value = getattr(schema, field)
        if isinstance(value, np.generic):
            value = value.item()
        payload[field] = value
    for field in ("action_low", "action_high"):
        if not hasattr(schema, field):
            raise VariantEnvironmentError(f"environment schema lacks {field!r}")
        array = np.asarray(getattr(schema, field))
        payload[field] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    return MappingProxyType(payload)


def _selection_flat_indices(value: Any, selection: Any) -> list[int]:
    try:
        import jax
    except ImportError:
        host = value
    else:
        host = jax.device_get(value)
    array = np.asarray(host)
    if array.ndim == 0 or max(selection.indices) >= array.shape[0]:
        raise VariantEnvironmentError(
            f"registered selection is incompatible with {selection.leaf!r}"
        )
    mask = np.zeros(array.shape, dtype=np.bool_)
    rows = np.asarray(selection.indices, dtype=np.int64)
    if selection.components is None:
        mask[rows] = True
    else:
        if array.ndim != 2 or max(selection.components) >= array.shape[1]:
            raise VariantEnvironmentError(
                f"registered component selection is incompatible with {selection.leaf!r}"
            )
        mask[np.ix_(rows, selection.components)] = True
    return np.flatnonzero(mask.reshape(-1)).tolist()


def _freeze_json(value: Any) -> Any:
    value = canonicalize(value)
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class VariantBuild:
    task_id: str
    backend_id: str
    axis_id: str
    factor_id: str
    factor_value: float
    role: FactorRole
    native_environment: Any
    adapter: Any
    operator_audit: DynamicsOperatorAudit
    registry_config_digest: str
    runtime_digest: str
    environment_class: str
    expected_nominal_model_digest: str
    expected_bound_model_digest: str
    operator_digest: str | None
    model_diff_digest: str
    anchor_operator: Mapping[str, Any] | None
    environment_instance_digest: str
    axis_binding: AxisAnchorBinding | None
    axis_binding_digest: str | None
    source_anchor_id: str | None

    def __post_init__(self) -> None:
        for field in (
            "registry_config_digest",
            "runtime_digest",
            "expected_nominal_model_digest",
            "expected_bound_model_digest",
            "model_diff_digest",
            "environment_instance_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.operator_digest is not None:
            object.__setattr__(
                self, "operator_digest", _digest(self.operator_digest, "operator_digest")
            )
        if self.axis_binding_digest is not None:
            object.__setattr__(
                self,
                "axis_binding_digest",
                _digest(self.axis_binding_digest, "axis_binding_digest"),
            )
        if (self.axis_binding is None) != (self.axis_binding_digest is None):
            raise ValueError("axis_binding and axis_binding_digest nullability disagree")
        if self.axis_binding is not None:
            if self.axis_binding.axis_binding_digest != self.axis_binding_digest:
                raise ValueError("axis_binding_digest does not match AxisAnchorBinding")
            if self.axis_binding.model_diff_digest != self.model_diff_digest:
                raise ValueError("AxisAnchorBinding does not bind the live model diff")
        if self.source_anchor_id is not None:
            object.__setattr__(
                self,
                "source_anchor_id",
                _digest(self.source_anchor_id, "source_anchor_id"),
            )
        if self.factor_value == 1.0 and self.axis_binding_digest is not None:
            raise ValueError("canonical nominal instance must have axis_binding_digest=None")
        if self.factor_value != 1.0 and self.axis_binding_digest is None:
            raise ValueError("nonnominal instance must bind exactly one axis")
        if self.factor_value == 1.0 and (
            self.operator_digest is not None or self.anchor_operator is not None
        ):
            raise ValueError("canonical nominal instance cannot carry an operator")
        if self.factor_value != 1.0 and (
            self.operator_digest is None or self.anchor_operator is None
        ):
            raise ValueError("nonnominal instance must carry its materialized operator")
        if self.anchor_operator is not None:
            operator = canonicalize(self.anchor_operator)
            if sha256_json(operator) != self.operator_digest:
                raise ValueError("operator_digest does not match anchor_operator")
            object.__setattr__(self, "anchor_operator", _freeze_json(operator))
        changes = [] if self.anchor_operator is None else [
            {
                "leaf": item["leaf"],
                "before_digest": item["expected_before_digest"],
                "after_digest": item["expected_after_digest"],
                "changed_flat_indices": item["flat_indices"],
            }
            for item in self.anchor_operator["mutations"]
        ]
        projection = canonical_model_diff_projection(
            nominal_model_digest=self.expected_nominal_model_digest,
            bound_model_digest=self.expected_bound_model_digest,
            changes=changes,
        )
        if sha256_json(projection) != self.model_diff_digest:
            raise ValueError("model_diff_digest does not match materialized operator changes")
        instance = EnvironmentInstanceRecord.create(
            task=self.task_id,
            backend=self.backend_id,
            nominal=self.factor_value == 1.0,
            factor=self.factor_value,
            environment_class=self.environment_class,
            registry_config_digest=self.registry_config_digest,
            runtime_digest=self.runtime_digest,
            expected_nominal_model_digest=self.expected_nominal_model_digest,
            expected_bound_model_digest=self.expected_bound_model_digest,
            operator_digest=self.operator_digest,
            axis_binding_digest=self.axis_binding_digest,
            model_diff_digest=self.model_diff_digest,
        )
        if instance.environment_instance_digest != self.environment_instance_digest:
            raise ValueError("environment_instance_digest does not match canonical build fields")
        expected_anchor = (
            SourceAnchorRecord.create(
                environment_instance_digest=self.environment_instance_digest,
                axis_binding_digest=self.axis_binding_digest,
            ).anchor_id
            if self.role in {SOURCE_ROLE, SAFETY_ROLE}
            else None
        )
        if self.source_anchor_id != expected_anchor:
            raise ValueError("source_anchor_id does not match the build role and identity")


class VariantEnvironmentFactory:
    """Load a fresh nominal native env, apply one registered axis, then adapt/JIT."""

    def __init__(
        self,
        *,
        registry: AxisRegistry,
        nominal_loader: Callable[[str], Any],
        adapter_factory: NativeAdapterFactory,
        registry_config_digests: Mapping[str, str],
        runtime_digest: str,
    ) -> None:
        self.registry = registry
        self.nominal_loader = nominal_loader
        self.adapter_factory = adapter_factory
        self.registry_config_digests = MappingProxyType(
            {
                str(task): _digest(value, f"registry_config_digests[{task!r}]")
                for task, value in registry_config_digests.items()
            }
        )
        self.runtime_digest = _digest(runtime_digest, "runtime_digest")

    def create(
        self,
        *,
        task_id: str,
        axis_id: str,
        factor_id: str,
        role: FactorRole,
        jit: bool,
    ) -> VariantBuild:
        if task_id not in self.registry_config_digests:
            raise VariantEnvironmentError(f"task has no frozen registry config: {task_id!r}")
        entry, factor = self.registry.require(
            task_id=task_id,
            axis_id=axis_id,
            factor_id=factor_id,
            role=role,
        )
        nominal = self.nominal_loader(task_id)
        if nominal is None:
            raise VariantEnvironmentError("nominal loader returned no environment")
        operator = make_operator(entry)
        shifted = operator.apply(nominal, factor.value)
        if shifted is nominal:
            raise VariantEnvironmentError("variant operator did not return a fresh environment")
        audit = operator.audit(nominal, shifted, factor.value)
        if not audit.passed:
            raise VariantEnvironmentError(f"operator audit failed: {audit.reason}")

        # Crucially, adaptation/JIT occurs only after the audited native rebind.
        adapter = self.adapter_factory(shifted, task_id, jit=jit)
        if hasattr(adapter, "environment") and adapter.environment is not shifted:
            raise VariantEnvironmentError("adapter is not bound to the audited shifted environment")
        schema = getattr(adapter, "schema", None)
        if schema is None:
            raise VariantEnvironmentError("adapter has no environment schema")
        nominal_snapshot = canonical_model_snapshot(nominal._mjx_model)
        shifted_snapshot = canonical_model_snapshot(shifted._mjx_model)
        model_diff, model_diff_digest = derive_live_model_diff(
            nominal._mjx_model, shifted._mjx_model
        )
        registry_config_digest = self.registry_config_digests[task_id]
        environment_class = f"{type(shifted).__module__}.{type(shifted).__qualname__}"
        if factor.value == 1.0:
            axis_binding = None
            axis_binding_digest = None
            anchor_operator = None
            operator_digest = None
        else:
            mutation_by_leaf = {item["leaf"]: item for item in model_diff["changes"]}
            mutations: list[dict[str, Any]] = []
            for selection in sorted(entry.selections, key=lambda item: item.leaf):
                leaf = f"_mjx_model.{selection.leaf}"
                if leaf not in mutation_by_leaf:
                    raise VariantEnvironmentError(
                        f"canonical model diff lacks registered changed leaf {leaf!r}"
                    )
                change = mutation_by_leaf[leaf]
                before = getattr(nominal._mjx_model, selection.leaf)
                after = getattr(shifted._mjx_model, selection.leaf)
                expected_indices = _selection_flat_indices(before, selection)
                if change["changed_flat_indices"] != expected_indices:
                    raise VariantEnvironmentError(
                        f"canonical model diff indices disagree for {leaf!r}"
                    )
                mutations.append(
                    {
                        "leaf": leaf,
                        "flat_indices": expected_indices,
                        "multiplier": factor.value,
                        "expected_before_digest": canonical_array_digest(before),
                        "expected_after_digest": canonical_array_digest(after),
                    }
                )
            anchor_operator = {
                "schema": "policy-learnware.v02-anchor-operator.v0",
                "operator_id": entry.operator_id,
                "axis_id": axis_id,
                "axis_registry_digest": self.registry.digest,
                "factor": factor.value,
                "mutations": mutations,
            }
            operator_digest = sha256_json(anchor_operator)
            axis_binding = AxisAnchorBinding.create(
                axis_id=axis_id,
                factor_id=factor_id,
                operator_digest=entry.operator_digest,
                model_diff_digest=model_diff_digest,
            )
            axis_binding_digest = axis_binding.axis_binding_digest
        instance = EnvironmentInstanceRecord.create(
            task=task_id,
            backend=entry.backend_id,
            nominal=factor.value == 1.0,
            factor=factor.value,
            environment_class=environment_class,
            registry_config_digest=registry_config_digest,
            runtime_digest=self.runtime_digest,
            expected_nominal_model_digest=nominal_snapshot["digest"],
            expected_bound_model_digest=shifted_snapshot["digest"],
            operator_digest=operator_digest,
            axis_binding_digest=axis_binding_digest,
            model_diff_digest=model_diff_digest,
        )
        environment_instance_digest = instance.environment_instance_digest
        source_anchor_id = None
        if role in {SOURCE_ROLE, SAFETY_ROLE}:
            source_anchor_id = SourceAnchorRecord.create(
                environment_instance_digest=environment_instance_digest,
                axis_binding_digest=axis_binding_digest,
            ).anchor_id
        return VariantBuild(
            task_id=task_id,
            backend_id=entry.backend_id,
            axis_id=axis_id,
            factor_id=factor_id,
            factor_value=factor.value,
            role=role,
            native_environment=shifted,
            adapter=adapter,
            operator_audit=audit,
            registry_config_digest=registry_config_digest,
            runtime_digest=self.runtime_digest,
            environment_class=environment_class,
            expected_nominal_model_digest=nominal_snapshot["digest"],
            expected_bound_model_digest=shifted_snapshot["digest"],
            operator_digest=operator_digest,
            model_diff_digest=model_diff_digest,
            anchor_operator=anchor_operator,
            environment_instance_digest=environment_instance_digest,
            axis_binding=axis_binding,
            axis_binding_digest=axis_binding_digest,
            source_anchor_id=source_anchor_id,
        )


@dataclass(frozen=True)
class RolloutAudit:
    episode_count: int
    steps_per_episode: int
    all_finite: bool
    no_early_termination: bool
    paired_observation_identity: bool | None
    paired_reward_identity: bool | None
    paired_flag_identity: bool | None
    maximum_observation_absolute_error: float | None
    maximum_reward_absolute_error: float | None
    passed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-rollout-audit.v0",
            **self.__dict__,
        }


def audit_rollouts(
    adapter: Any,
    *,
    reset_seeds: Sequence[int],
    action_tensor: np.ndarray,
    nominal_adapter: Any | None = None,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> RolloutAudit:
    seeds = np.asarray(reset_seeds, dtype=np.int64)
    actions = np.asarray(action_tensor, dtype=np.float32)
    if seeds.ndim != 1 or seeds.size == 0 or np.any(seeds < 0):
        raise ValueError("reset_seeds must be a non-empty non-negative vector")
    action_dim = int(adapter.schema.action_dim)
    if actions.ndim != 3 or actions.shape[0] != seeds.size or actions.shape[2] != action_dim:
        raise ValueError("action_tensor must have shape [episodes,steps,action_dim]")
    if actions.shape[1] <= 0 or actions.shape[1] > int(adapter.schema.horizon):
        raise ValueError("action_tensor has an invalid step count")
    if not np.all(np.isfinite(actions)):
        raise ValueError("action_tensor must be finite")
    low = np.asarray(adapter.schema.action_low, dtype=np.float32)
    high = np.asarray(adapter.schema.action_high, dtype=np.float32)
    if np.any(actions < low) or np.any(actions > high):
        raise ValueError("action_tensor exceeds registered action bounds")
    if nominal_adapter is not None and dict(_schema_contract(nominal_adapter.schema)) != dict(
        _schema_contract(adapter.schema)
    ):
        raise VariantEnvironmentError("nominal and factor-one schema contracts differ")

    finite = True
    no_early = True
    observation_identity: bool | None = True if nominal_adapter is not None else None
    reward_identity: bool | None = True if nominal_adapter is not None else None
    flag_identity: bool | None = True if nominal_adapter is not None else None
    max_observation_error: float | None = 0.0 if nominal_adapter is not None else None
    max_reward_error: float | None = 0.0 if nominal_adapter is not None else None
    reason: str | None = None
    try:
        for episode, seed in enumerate(seeds):
            state, observation = adapter.reset(int(seed))
            observation = np.asarray(observation)
            finite &= bool(np.all(np.isfinite(observation)))
            if nominal_adapter is not None:
                reference_state, reference_observation = nominal_adapter.reset(int(seed))
                reference_observation = np.asarray(reference_observation)
                error = float(np.max(np.abs(observation - reference_observation), initial=0.0))
                max_observation_error = max(float(max_observation_error), error)
                observation_identity &= bool(
                    np.allclose(observation, reference_observation, atol=atol, rtol=rtol)
                )
            for step_index, action in enumerate(actions[episode]):
                state, result = adapter.step(state, np.array(action, copy=True))
                result_observation = np.asarray(result.observation)
                finite &= bool(
                    np.all(np.isfinite(result_observation))
                    and math.isfinite(float(result.reward))
                )
                ended = bool(result.terminated) or bool(result.truncated)
                if ended and step_index + 1 < actions.shape[1]:
                    no_early = False
                if nominal_adapter is not None:
                    reference_state, reference = nominal_adapter.step(
                        reference_state, np.array(action, copy=True)
                    )
                    reference_observation = np.asarray(reference.observation)
                    error = float(
                        np.max(
                            np.abs(result_observation - reference_observation), initial=0.0
                        )
                    )
                    reward_error = abs(float(result.reward) - float(reference.reward))
                    max_observation_error = max(float(max_observation_error), error)
                    max_reward_error = max(float(max_reward_error), reward_error)
                    observation_identity &= bool(
                        np.allclose(
                            result_observation,
                            reference_observation,
                            atol=atol,
                            rtol=rtol,
                        )
                    )
                    reward_identity &= reward_error <= atol
                    flag_identity &= (
                        bool(result.terminated) == bool(reference.terminated)
                        and bool(result.truncated) == bool(reference.truncated)
                    )
    except Exception as exc:  # Gate 0 retains a diagnostic instead of losing the failure.
        finite = False
        no_early = False
        reason = f"{type(exc).__name__}: {exc}"
    paired_ok = bool(
        nominal_adapter is None
        or (observation_identity and reward_identity and flag_identity)
    )
    return RolloutAudit(
        episode_count=int(seeds.size),
        steps_per_episode=int(actions.shape[1]),
        all_finite=finite,
        no_early_termination=no_early,
        paired_observation_identity=observation_identity,
        paired_reward_identity=reward_identity,
        paired_flag_identity=flag_identity,
        maximum_observation_absolute_error=max_observation_error,
        maximum_reward_absolute_error=max_reward_error,
        passed=bool(finite and no_early and paired_ok and reason is None),
        reason=reason,
    )


@dataclass(frozen=True)
class Gate0Audit:
    environment_instance_digest: str
    operator_audit_digest: str
    schema_contract_identity: bool
    factor_role_valid: bool
    scalar_rollout: RolloutAudit
    jit_finite: bool
    batched_rollout_finite: bool
    source_object_unchanged: bool
    exact_allowlist: bool
    coupled_physics: bool
    passed: bool
    reasons: tuple[str, ...]

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-gate0-audit.v0",
            "environment_instance_digest": self.environment_instance_digest,
            "operator_audit_digest": self.operator_audit_digest,
            "schema_contract_identity": self.schema_contract_identity,
            "factor_role_valid": self.factor_role_valid,
            "scalar_rollout": self.scalar_rollout.to_dict(),
            "jit_finite": self.jit_finite,
            "batched_rollout_finite": self.batched_rollout_finite,
            "source_object_unchanged": self.source_object_unchanged,
            "exact_allowlist": self.exact_allowlist,
            "coupled_physics": self.coupled_physics,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def build_gate0_audit(
    build: VariantBuild,
    *,
    nominal_adapter: Any,
    reset_seeds: Sequence[int],
    action_tensor: np.ndarray,
    jit_finite: bool,
    batched_rollout_finite: bool,
) -> Gate0Audit:
    schema_identity = dict(_schema_contract(nominal_adapter.schema)) == dict(
        _schema_contract(build.adapter.schema)
    )
    factor_role_valid = build.role in (
        SOURCE_ROLE,
        SAFETY_ROLE,
        "development",
        "confirmatory_heldout",
    )
    scalar = audit_rollouts(
        build.adapter,
        reset_seeds=reset_seeds,
        action_tensor=action_tensor,
        nominal_adapter=(nominal_adapter if build.factor_value == 1.0 else None),
    )
    operator = build.operator_audit
    reasons: list[str] = []
    checks = {
        "schema_contract_mismatch": schema_identity,
        "factor_role_mismatch": factor_role_valid,
        "scalar_rollout_failed": scalar.passed,
        "jit_rollout_failed": bool(jit_finite),
        "batched_rollout_failed": bool(batched_rollout_finite),
        "source_object_modified": operator.source_object_unchanged,
        "operator_allowlist_failed": operator.exact_allowlist,
        "coupled_physics_failed": operator.coupling_check,
        "operator_audit_failed": operator.passed,
    }
    reasons.extend(name for name, passed in checks.items() if not passed)
    return Gate0Audit(
        environment_instance_digest=build.environment_instance_digest,
        operator_audit_digest=operator.digest,
        schema_contract_identity=schema_identity,
        factor_role_valid=factor_role_valid,
        scalar_rollout=scalar,
        jit_finite=bool(jit_finite),
        batched_rollout_finite=bool(batched_rollout_finite),
        source_object_unchanged=operator.source_object_unchanged,
        exact_allowlist=operator.exact_allowlist,
        coupled_physics=operator.coupling_check,
        passed=not reasons,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class InstanceOrderAudit:
    request_sequence: tuple[tuple[str, str], ...]
    fresh_environment_objects: bool
    repeated_identity_stable: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-instance-order-audit.v0",
            "request_sequence": [list(item) for item in self.request_sequence],
            "fresh_environment_objects": self.fresh_environment_objects,
            "repeated_identity_stable": self.repeated_identity_stable,
            "passed": self.passed,
        }


def audit_instance_order(
    factory: VariantEnvironmentFactory,
    *,
    task_id: str,
    requests: Sequence[tuple[str, str]],
    role: FactorRole,
) -> InstanceOrderAudit:
    sequence = tuple(requests)
    if not sequence:
        raise ValueError("instance order audit requires at least one request")
    builds = tuple(
        factory.create(
            task_id=task_id,
            axis_id=axis_id,
            factor_id=factor_id,
            role=role,
            jit=False,
        )
        for axis_id, factor_id in sequence
    )
    fresh = len({id(item.native_environment) for item in builds}) == len(builds)
    identities: dict[tuple[str, str], str] = {}
    stable = True
    for request, build in zip(sequence, builds, strict=True):
        previous = identities.setdefault(request, build.environment_instance_digest)
        stable &= previous == build.environment_instance_digest
    return InstanceOrderAudit(
        request_sequence=sequence,
        fresh_environment_objects=fresh,
        repeated_identity_stable=stable,
        passed=bool(fresh and stable),
    )


__all__ = [
    "Gate0Audit",
    "InstanceOrderAudit",
    "NativeAdapterFactory",
    "RolloutAudit",
    "VariantBuild",
    "VariantEnvironmentError",
    "VariantEnvironmentFactory",
    "audit_instance_order",
    "audit_rollouts",
    "build_gate0_audit",
]
