from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from policy_learnware_v0.envs.base import SyntheticEnvAdapter
from policy_learnware_v0.v02.axes import (
    JOINT_DAMPING_OPERATOR,
    SAFETY_ROLE,
    SOURCE_ROLE,
    AxisRegistry,
    AxisRegistryEntry,
    FactorDefinition,
    LeafSelection,
    operator_source_digest,
)
from policy_learnware_v0.v02.variant_env import (
    VariantEnvironmentFactory,
    audit_instance_order,
    build_gate0_audit,
)


@dataclass(frozen=True)
class FakeModel:
    dof_damping: np.ndarray
    untouched: np.ndarray

    def tree_replace(self, changes: dict[str, np.ndarray]) -> "FakeModel":
        return replace(self, **changes)


class FakeNativeEnvironment:
    def __init__(self) -> None:
        self._mjx_model = FakeModel(
            dof_damping=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
            untouched=np.asarray([9.0], dtype=np.float32),
        )

    @property
    def mjx_model(self) -> FakeModel:
        return self._mjx_model


class BoundSyntheticAdapter(SyntheticEnvAdapter):
    def __init__(self, environment: FakeNativeEnvironment, task: str) -> None:
        super().__init__(task=task, observation_dim=3, action_dim=2, horizon=4)
        self._environment = environment

    @property
    def environment(self) -> FakeNativeEnvironment:
        return self._environment


def _registry() -> AxisRegistry:
    factors = (
        FactorDefinition("low", 0.5, frozenset({SOURCE_ROLE})),
        FactorDefinition("nominal", 1.0, frozenset({SOURCE_ROLE, SAFETY_ROLE})),
        FactorDefinition("high", 2.0, frozenset({SOURCE_ROLE})),
    )
    entries = {
        axis: AxisRegistryEntry(
            axis_id=axis,
            task_id="TaskA",
            backend_id="mujoco_playground.registry",
            operator_id=JOINT_DAMPING_OPERATOR,
            operator_version="1",
            operator_digest=operator_source_digest(),
            selections=(LeafSelection("dof_damping", indices, require_nonzero=True),),
            factors=factors,
        )
        for axis, indices in (("axis-a", (1,)), ("axis-b", (2,)))
    }
    return AxisRegistry(entries)


def _factory() -> VariantEnvironmentFactory:
    return VariantEnvironmentFactory(
        registry=_registry(),
        nominal_loader=lambda task: FakeNativeEnvironment(),
        adapter_factory=lambda env, task, jit: BoundSyntheticAdapter(env, task),
        registry_config_digests={"TaskA": "1" * 64},
        runtime_digest="2" * 64,
    )


def test_nominal_anchor_is_shared_across_axes_and_nonnominal_is_axis_bound() -> None:
    factory = _factory()
    left = factory.create(
        task_id="TaskA", axis_id="axis-a", factor_id="nominal", role=SOURCE_ROLE, jit=False
    )
    right = factory.create(
        task_id="TaskA", axis_id="axis-b", factor_id="nominal", role=SOURCE_ROLE, jit=False
    )
    shifted = factory.create(
        task_id="TaskA", axis_id="axis-a", factor_id="high", role=SOURCE_ROLE, jit=False
    )
    assert left.environment_instance_digest == right.environment_instance_digest
    assert left.source_anchor_id == right.source_anchor_id
    assert len(left.source_anchor_id) == 64
    assert left.axis_binding_digest is None
    assert shifted.axis_binding_digest is not None
    assert len(shifted.source_anchor_id) == 64
    assert shifted.source_anchor_id != left.source_anchor_id
    assert shifted.environment_instance_digest != left.environment_instance_digest


def test_gate0_audits_identity_finite_allowlist_and_runtime_paths() -> None:
    factory = _factory()
    nominal_native = FakeNativeEnvironment()
    nominal_adapter = BoundSyntheticAdapter(nominal_native, "TaskA")
    build = factory.create(
        task_id="TaskA", axis_id="axis-a", factor_id="nominal", role=SOURCE_ROLE, jit=False
    )
    actions = np.zeros((2, 3, 2), dtype=np.float32)
    gate = build_gate0_audit(
        build,
        nominal_adapter=nominal_adapter,
        reset_seeds=(10, 11),
        action_tensor=actions,
        jit_finite=True,
        batched_rollout_finite=True,
    )
    assert gate.passed
    assert gate.scalar_rollout.paired_observation_identity
    assert gate.scalar_rollout.paired_reward_identity


def test_instance_creation_order_is_fresh_and_digest_stable() -> None:
    audit = audit_instance_order(
        _factory(),
        task_id="TaskA",
        requests=(
            ("axis-a", "nominal"),
            ("axis-a", "low"),
            ("axis-b", "high"),
            ("axis-a", "nominal"),
        ),
        role=SOURCE_ROLE,
    )
    assert audit.passed
