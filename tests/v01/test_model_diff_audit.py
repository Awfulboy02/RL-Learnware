from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from policy_learnware_v0.envs.base import SyntheticEnvAdapter
from policy_learnware_v0.v01.variant_env import (
    ALLOWLISTED_ENV_LEAF,
    VariantEnvironmentError,
    apply_global_nonzero_dof_damping_scale,
    audit_finite_termination,
    audit_five_factor_finite,
    snapshot_model,
)


@dataclass(frozen=True)
class SyntheticMjxModel:
    dof_damping: np.ndarray
    body_mass: np.ndarray

    def tree_replace(self, replacements: dict[str, np.ndarray]) -> "SyntheticMjxModel":
        return replace(self, **replacements)


@dataclass(frozen=True)
class TamperingModel(SyntheticMjxModel):
    def tree_replace(self, replacements: dict[str, np.ndarray]) -> "TamperingModel":
        return replace(
            self,
            **replacements,
            body_mass=np.asarray(self.body_mass) * np.float32(2.0),
        )


def test_non_nominal_diff_changes_only_originally_nonzero_damping() -> None:
    damping = np.asarray([0.0, 0.1, -0.0, 0.4], dtype=np.float32)
    model = SyntheticMjxModel(
        dof_damping=damping.copy(), body_mass=np.asarray([1.0], dtype=np.float32)
    )
    base = snapshot_model(model)

    shifted, audit = apply_global_nonzero_dof_damping_scale(model, 0.5)

    np.testing.assert_array_equal(model.dof_damping, damping)
    np.testing.assert_array_equal(
        shifted.dof_damping,
        np.asarray([0.0, 0.05, -0.0, 0.2], dtype=np.float32),
    )
    np.testing.assert_array_equal(shifted.body_mass, model.body_mass)
    assert snapshot_model(model).digest == base.digest
    assert audit.changed_leaves == (ALLOWLISTED_ENV_LEAF,)
    assert audit.changed_index_count == 2
    assert audit.nominal_nonzero_count == 2
    assert audit.base_model_digest != audit.shifted_model_digest
    assert audit.before_leaf_digest != audit.after_leaf_digest


def test_full_pytree_audit_rejects_a_second_changed_leaf() -> None:
    model = TamperingModel(
        dof_damping=np.asarray([0.0, 0.1], dtype=np.float32),
        body_mass=np.asarray([1.0], dtype=np.float32),
    )
    with pytest.raises(VariantEnvironmentError, match="unexpected leaves"):
        apply_global_nonzero_dof_damping_scale(model, 2.0)


@pytest.mark.parametrize("factor", [0.0, -1.0, np.inf, np.nan, True, "2.0"])
def test_invalid_factor_fails_closed(factor: object) -> None:
    model = SyntheticMjxModel(
        np.asarray([0.0, 0.1], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="factor"):
        apply_global_nonzero_dof_damping_scale(model, factor)  # type: ignore[arg-type]


def test_no_nonzero_damping_fails_closed() -> None:
    model = SyntheticMjxModel(
        np.zeros(3, dtype=np.float32), np.asarray([1.0], dtype=np.float32)
    )
    with pytest.raises(VariantEnvironmentError, match="no nonzero"):
        apply_global_nonzero_dof_damping_scale(model, 2.0)


def test_reusable_five_factor_finite_audit_covers_exact_grid() -> None:
    adapters = {
        factor: SyntheticEnvAdapter(task="FiniteTask", horizon=3)
        for factor in (0.5, 0.75, 1.0, 1.5, 2.0)
    }
    actions = np.zeros((2, 3, 2), dtype=np.float32)
    audit = audit_five_factor_finite(
        adapters, reset_seeds=[1, 2], action_tensor=actions
    )
    assert audit.passed
    assert audit.factors == (0.5, 0.75, 1.0, 1.5, 2.0)
    assert all(result.passed for result in audit.results.values())


def test_single_variant_finite_audit_rejects_early_end() -> None:
    adapter = SyntheticEnvAdapter(task="FiniteTask", horizon=3)
    # Auditing two steps is a valid strict prefix and must not invent an end.
    audit = audit_finite_termination(
        adapter,
        reset_seeds=[3],
        action_tensor=np.zeros((1, 2, 2), dtype=np.float32),
    )
    assert audit.passed
    assert audit.no_early_termination
