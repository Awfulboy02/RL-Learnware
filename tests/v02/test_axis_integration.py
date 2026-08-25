from __future__ import annotations

from types import SimpleNamespace

import pytest

from policy_learnware_v0.v02.axis_catalog import build_candidate_axis_catalog
from policy_learnware_v0.v02.axis_integration import (
    AxisConfigBindingError,
    axis_registry_from_config,
)
from policy_learnware_v0.v02.config import AxisConfig, SourceFactorConfig


def _config(
    candidate,
    *,
    operator_id: str | None = None,
    stage: str = "audit_smoke",
):
    axis = AxisConfig(
        axis_id=candidate.axis_id,
        operator_id=operator_id or candidate.operator_id,
        operator_digest=candidate.operator_digest,
        leaf_allowlist=tuple(
            f"_mjx_model.{selection.leaf}" for selection in candidate.selections
        ),
        static_within_episode=True,
    )
    factors = (
        SourceFactorConfig("low", 0.9, ("source",), "1" * 64, "2" * 64),
        SourceFactorConfig("nominal", 1.0, ("source",), "3" * 64, None),
        SourceFactorConfig("high", 1.1, ("source",), "4" * 64, "5" * 64),
    )
    return SimpleNamespace(
        tasks=(candidate.task_id,),
        dynamics_axes={candidate.task_id: (axis,)},
        source_factors={candidate.task_id: {candidate.axis_id: factors}},
        development_targets=(),
        confirmatory_targets=(),
        safety_exact_targets=(),
        stage=stage,
    )


def test_config_bridge_builds_executable_registered_factors() -> None:
    candidates, _ = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    candidate = candidates.entries["cheetah_actuator_gain"]
    bound = axis_registry_from_config(_config(candidate), candidates)
    entry, factor = bound.require(
        task_id="CheetahRun",
        axis_id="cheetah_actuator_gain",
        factor_id="high",
        role="source",
    )
    assert entry.operator_id == candidate.operator_id
    assert factor.value == 1.1


def test_config_bridge_rejects_operator_or_leaf_drift() -> None:
    candidates, _ = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    candidate = candidates.entries["cheetah_actuator_gain"]
    with pytest.raises(AxisConfigBindingError, match="operator_id"):
        axis_registry_from_config(_config(candidate, operator_id="unreviewed"), candidates)

    config = _config(candidate)
    bad_axis = AxisConfig(
        axis_id=candidate.axis_id,
        operator_id=candidate.operator_id,
        operator_digest=candidate.operator_digest,
        leaf_allowlist=("_mjx_model.dof_damping",),
        static_within_episode=True,
    )
    config.dynamics_axes[candidate.task_id] = (bad_axis,)
    with pytest.raises(AxisConfigBindingError, match="leaf_allowlist"):
        axis_registry_from_config(config, candidates)


def test_bridge_marks_nominal_as_uninstantiated_exact_recurrence_capability() -> None:
    candidates, _ = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    candidate = candidates.entries["cheetah_actuator_gain"]
    bound = axis_registry_from_config(_config(candidate), candidates)
    _, nominal = bound.require(
        task_id="CheetahRun",
        axis_id="cheetah_actuator_gain",
        factor_id="nominal",
        role="safety_exact_reference",
    )
    assert nominal.value == 1.0
