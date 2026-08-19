from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from policy_learnware_v0.envs.base import SyntheticEnvAdapter
from policy_learnware_v0.v01.variant_env import (
    apply_global_nonzero_dof_damping_scale,
    audit_trajectory_identity,
    snapshot_model,
)


@dataclass(frozen=True)
class SyntheticMjxModel:
    dof_damping: np.ndarray
    body_mass: np.ndarray

    def tree_replace(self, replacements: dict[str, np.ndarray]) -> "SyntheticMjxModel":
        return replace(self, **replacements)


def test_nominal_factor_is_exact_model_identity_without_aliasing() -> None:
    model = SyntheticMjxModel(
        dof_damping=np.asarray([0.0, 0.1, 0.2, 0.0], dtype=np.float32),
        body_mass=np.asarray([1.0, 2.0], dtype=np.float32),
    )
    before = snapshot_model(model)

    shifted, audit = apply_global_nonzero_dof_damping_scale(model, 1.0)

    assert shifted is not model
    assert snapshot_model(model).digest == before.digest
    assert snapshot_model(shifted).digest == before.digest
    assert audit.base_model_digest == audit.shifted_model_digest
    assert audit.changed_leaves == ()
    assert audit.changed_index_count == 0
    assert audit.nominal_nonzero_count == 2
    assert audit.before_leaf_digest == audit.after_leaf_digest
    assert audit.source_unchanged is True


def test_snapshot_binds_leaf_dtype_shape_and_content() -> None:
    base = SyntheticMjxModel(
        np.asarray([0.0, 0.1], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )
    different_dtype = SyntheticMjxModel(
        np.asarray([0.0, 0.1], dtype=np.float64),
        np.asarray([1.0], dtype=np.float32),
    )
    different_other_leaf = SyntheticMjxModel(
        np.asarray([0.0, 0.1], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
    )
    assert snapshot_model(base).digest != snapshot_model(different_dtype).digest
    assert snapshot_model(base).digest != snapshot_model(different_other_leaf).digest


def test_reusable_trajectory_identity_audit_uses_same_actions_and_seeds() -> None:
    nominal = SyntheticEnvAdapter(task="IdentityTask", horizon=4)
    factor_one = SyntheticEnvAdapter(task="IdentityTask", horizon=4)
    actions = np.asarray(
        [
            [[0.1, -0.2], [0.2, -0.1], [0.0, 0.0], [-0.2, 0.2]],
            [[-0.1, 0.2], [0.3, 0.1], [0.0, -0.1], [0.2, -0.2]],
        ],
        dtype=np.float32,
    )
    audit = audit_trajectory_identity(
        nominal,
        factor_one,
        reset_seeds=[11, 12],
        action_tensor=actions,
    )
    assert audit.passed
    assert audit.schema_identity
    assert audit.action_identity
    assert audit.flag_identity
    assert audit.maximum_observation_absolute_error == 0.0
    assert audit.maximum_reward_absolute_error == 0.0
