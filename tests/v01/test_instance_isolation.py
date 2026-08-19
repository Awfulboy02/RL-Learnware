from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from policy_learnware_v0.envs.base import SyntheticEnvAdapter
from policy_learnware_v0.v01.variant_env import (
    apply_global_nonzero_dof_damping_scale,
    audit_instance_isolation,
    snapshot_model,
)


@dataclass(frozen=True)
class SyntheticMjxModel:
    dof_damping: np.ndarray
    body_mass: np.ndarray

    def tree_replace(self, replacements: dict[str, np.ndarray]) -> "SyntheticMjxModel":
        return replace(self, **replacements)


def _fresh_nominal() -> SyntheticMjxModel:
    return SyntheticMjxModel(
        dof_damping=np.asarray([0.0, 0.1, 0.1, 0.2], dtype=np.float32),
        body_mass=np.asarray([1.0, 2.0], dtype=np.float32),
    )


class SyntheticAuditedAdapter:
    def __init__(self, factor: float) -> None:
        self._delegate = SyntheticEnvAdapter(task="IsolationTask", horizon=3)
        self.environment = object()
        _, self.model_diff_audit = apply_global_nonzero_dof_damping_scale(
            _fresh_nominal(), factor
        )

    @property
    def schema(self):
        return self._delegate.schema

    def reset(self, seed: int):
        return self._delegate.reset(seed)

    def step(self, state, action):
        return self._delegate.step(state, action)


def test_nominal_shifted_shifted_nominal_sequence_has_no_cross_instance_pollution() -> None:
    nominal_before = _fresh_nominal()
    nominal_digest = snapshot_model(nominal_before).digest

    low, low_audit = apply_global_nonzero_dof_damping_scale(_fresh_nominal(), 0.5)
    high, high_audit = apply_global_nonzero_dof_damping_scale(_fresh_nominal(), 2.0)
    nominal_after = _fresh_nominal()

    assert snapshot_model(nominal_before).digest == nominal_digest
    assert snapshot_model(nominal_after).digest == nominal_digest
    assert low_audit.base_model_digest == nominal_digest
    assert high_audit.base_model_digest == nominal_digest
    assert snapshot_model(low).digest == low_audit.shifted_model_digest
    assert snapshot_model(high).digest == high_audit.shifted_model_digest
    assert low_audit.shifted_model_digest != high_audit.shifted_model_digest
    np.testing.assert_array_equal(
        nominal_after.dof_damping,
        np.asarray([0.0, 0.1, 0.1, 0.2], dtype=np.float32),
    )


def test_repeated_instantiation_is_digest_reproducible() -> None:
    first, first_audit = apply_global_nonzero_dof_damping_scale(_fresh_nominal(), 1.5)
    second, second_audit = apply_global_nonzero_dof_damping_scale(_fresh_nominal(), 1.5)
    assert snapshot_model(first).digest == snapshot_model(second).digest
    assert first_audit.to_dict() == second_audit.to_dict()


def test_reusable_nominal_before_after_isolation_audit() -> None:
    instances = tuple(
        (factor, SyntheticAuditedAdapter(factor))
        for factor in (1.0, 0.5, 2.0, 1.0)
    )
    report = audit_instance_isolation(
        instances,
        reset_seeds=[7, 8],
        action_tensor=np.zeros((2, 3, 2), dtype=np.float32),
    )
    assert report.passed
    assert report.fresh_environment_objects
    assert report.base_model_identity
    assert report.schema_identity
    assert report.nominal_model_identity
    assert report.shifted_models_distinct
    assert report.nominal_trajectory_identity.passed
