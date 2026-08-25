from __future__ import annotations

import pytest

from policy_learnware_v0.v02.axis_catalog import (
    V02_TASKS,
    build_candidate_axis_catalog,
    reviewed_axis_subset,
)
from policy_learnware_v0.v02.axes import make_operator


def test_candidate_catalog_requires_explicit_straddling_factors() -> None:
    for values in ((), (1.0,), (0.9, 1.0, 1.0), (1.0, 1.0, 1.1)):
        with pytest.raises(ValueError):
            build_candidate_axis_catalog(values)


def test_candidate_catalog_retains_review_alternatives_and_exact_names() -> None:
    registry, evidence = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    assert set(registry.entries) == set(evidence)
    assert len(registry.entries) == 17
    assert {entry.task_id for entry in registry.entries.values()} == set(V02_TASKS)
    assert evidence["finger_spinner_mass_inertia"].selected_names == ("spinner",)
    assert registry.entries["fish_body_mass_inertia"].selections[0].indices == (
        2,
        3,
        4,
        5,
        6,
    )
    assert all(make_operator(entry).supports(entry.task_id, entry.backend_id) for entry in registry.entries.values())
    # A candidate catalog is not a formal two-axis-per-task registry.
    with pytest.raises(ValueError):
        registry.validate_formal_scope(V02_TASKS)


def test_reviewed_subset_is_exactly_two_axes_per_task() -> None:
    candidates, _ = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    selected = {
        "CartpoleSwingup": ("cartpole_mass_inertia", "cartpole_actuator_gain"),
        "CheetahRun": ("cheetah_mass_inertia", "cheetah_actuator_gain"),
        "FingerTurnEasy": ("finger_spinner_mass_inertia", "finger_joint_damping"),
        "FishSwim": ("fish_body_mass_inertia", "fish_actuator_gain"),
        "HopperHop": ("hopper_mass_inertia", "hopper_actuator_gain"),
        "WalkerWalk": ("walker_mass_inertia", "walker_actuator_gain"),
    }
    formal = reviewed_axis_subset(candidates, selected)
    assert len(formal.entries) == 12
    formal.validate_formal_scope(V02_TASKS)


def test_reviewed_subset_rejects_cross_task_or_missing_review() -> None:
    candidates, _ = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    with pytest.raises(ValueError):
        reviewed_axis_subset(candidates, {"CartpoleSwingup": ("cartpole_mass_inertia", "cartpole_actuator_gain")})

    selected = {
        task: ("cartpole_mass_inertia", "cartpole_actuator_gain")
        for task in V02_TASKS
    }
    with pytest.raises(ValueError):
        reviewed_axis_subset(candidates, selected)
