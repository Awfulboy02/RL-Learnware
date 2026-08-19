from __future__ import annotations

import copy

import pytest

from policy_learnware_v0.v01.registry import (
    ALLOWLISTED_MODEL_LEAF,
    ShiftRegistry,
    default_shift_registry,
)


def test_closed_registry_accepts_only_approved_task_factor_and_operator() -> None:
    registry = default_shift_registry(operator_digest="a" * 64)
    resolved = registry.require(
        "global_nonzero_dof_damping_scale", "WalkerWalk", 0.5
    )
    assert resolved.registry_digest == registry.digest
    assert resolved.allowlisted_model_leaf == "_mjx_model.dof_damping"
    assert resolved.selection_rule == "original_value_nonzero"
    assert resolved.operator_source_sha256 == "a" * 64

    with pytest.raises(KeyError):
        registry.require("friction", "WalkerWalk", 1.0)
    with pytest.raises(ValueError):
        registry.require("global_nonzero_dof_damping_scale", "WalkerRun", 1.0)
    with pytest.raises(ValueError):
        registry.require("global_nonzero_dof_damping_scale", "WalkerWalk", 1.1)
    with pytest.raises(ValueError):
        registry.require("global_nonzero_dof_damping_scale", "WalkerWalk", True)


def test_registry_round_trip_and_arbitrary_attribute_path_fail_closed() -> None:
    registry = default_shift_registry(operator_digest="b" * 64)
    assert ShiftRegistry.from_dict(registry.to_dict()).digest == registry.digest
    payload = copy.deepcopy(registry.to_dict())
    payload["entries"]["global_nonzero_dof_damping_scale"]["allowlisted_model_leaf"] = "../../mass"
    with pytest.raises(ValueError):
        ShiftRegistry.from_dict(payload)
    assert ALLOWLISTED_MODEL_LEAF == "_mjx_model.dof_damping"


def test_operator_source_digest_changes_registry_digest() -> None:
    first = default_shift_registry(operator_digest="1" * 64)
    second = default_shift_registry(operator_digest="2" * 64)
    assert first.digest != second.digest
