from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from policy_learnware_v0.v02.axes import (
    ACTUATOR_GAIN_OPERATOR,
    CONFIRMATORY_ROLE,
    CONTACT_FRICTION_OPERATOR,
    DEVELOPMENT_ROLE,
    JOINT_DAMPING_OPERATOR,
    MASS_INERTIA_OPERATOR,
    SAFETY_ROLE,
    SOURCE_ROLE,
    AxisRegistry,
    AxisRegistryEntry,
    FactorDefinition,
    LeafSelection,
    make_operator,
    operator_source_digest,
)


@dataclass(frozen=True)
class FakeModel:
    body_mass: np.ndarray
    body_inertia: np.ndarray
    dof_damping: np.ndarray
    geom_friction: np.ndarray
    actuator_gainprm: np.ndarray
    actuator_gear: np.ndarray
    untouched: np.ndarray
    topology: np.ndarray

    def tree_replace(self, changes: dict[str, np.ndarray]) -> "FakeModel":
        return replace(self, **changes)


class FakeEnvironment:
    def __init__(self, model: FakeModel) -> None:
        self._mjx_model = model

    @property
    def mjx_model(self) -> FakeModel:
        return self._mjx_model


def _model() -> FakeModel:
    return FakeModel(
        body_mass=np.asarray([0.0, 2.0, 3.0], dtype=np.float32),
        body_inertia=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.5, 2.0], [2.0, 2.5, 3.0]], dtype=np.float32),
        dof_damping=np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        geom_friction=np.asarray([[1.0, 0.1, 0.01], [2.0, 0.2, 0.02]], dtype=np.float32),
        actuator_gainprm=np.asarray([[1.0, 0.0], [2.0, 0.5]], dtype=np.float32),
        actuator_gear=np.asarray([[1.0, 0.0], [3.0, 1.0]], dtype=np.float32),
        untouched=np.asarray([7.0, 8.0], dtype=np.float32),
        topology=np.asarray([0, 1, 1], dtype=np.int32),
    )


def _factors() -> tuple[FactorDefinition, ...]:
    return (
        FactorDefinition("low", 0.5, frozenset({SOURCE_ROLE})),
        FactorDefinition("nominal", 1.0, frozenset({SOURCE_ROLE, SAFETY_ROLE})),
        FactorDefinition("high", 2.0, frozenset({SOURCE_ROLE})),
        FactorDefinition("dev", 0.75, frozenset({DEVELOPMENT_ROLE})),
        FactorDefinition("confirm", 1.5, frozenset({CONFIRMATORY_ROLE})),
    )


def _entry(operator_id: str, selections: tuple[LeafSelection, ...], *, axis: str = "axis-a", task: str = "TaskA") -> AxisRegistryEntry:
    return AxisRegistryEntry(
        axis_id=axis,
        task_id=task,
        backend_id="mujoco_playground.registry",
        operator_id=operator_id,
        operator_version="1",
        operator_digest=operator_source_digest(),
        selections=selections,
        factors=_factors(),
    )


@pytest.mark.parametrize(
    ("operator_id", "selections", "expected_leaves"),
    (
        (
            MASS_INERTIA_OPERATOR,
            (LeafSelection("body_mass", (1,)), LeafSelection("body_inertia", (1,))),
            ("body_inertia", "body_mass"),
        ),
        (
            JOINT_DAMPING_OPERATOR,
            (LeafSelection("dof_damping", (1, 2), require_nonzero=True),),
            ("dof_damping",),
        ),
        (
            CONTACT_FRICTION_OPERATOR,
            (LeafSelection("geom_friction", (0,), components=(0, 1, 2), require_nonzero=True),),
            ("geom_friction",),
        ),
        (
            ACTUATOR_GAIN_OPERATOR,
            (LeafSelection("actuator_gainprm", (1,), components=(0, 1)),),
            ("actuator_gainprm",),
        ),
    ),
)
def test_registered_operators_are_functional_exact_and_audited(
    operator_id: str,
    selections: tuple[LeafSelection, ...],
    expected_leaves: tuple[str, ...],
) -> None:
    nominal = FakeEnvironment(_model())
    original = _model()
    operator = make_operator(_entry(operator_id, selections))
    shifted = operator.apply(nominal, 2.0)
    audit = operator.audit(nominal, shifted, 2.0)
    assert shifted is not nominal
    assert nominal.mjx_model is not shifted.mjx_model
    assert audit.passed
    assert audit.changed_leaves == expected_leaves
    assert np.array_equal(nominal.mjx_model.untouched, original.untouched)
    assert np.array_equal(shifted.mjx_model.untouched, original.untouched)


def test_nominal_factor_returns_fresh_identity_instance() -> None:
    nominal = FakeEnvironment(_model())
    entry = _entry(
        JOINT_DAMPING_OPERATOR,
        (LeafSelection("dof_damping", (1, 2), require_nonzero=True),),
    )
    operator = make_operator(entry)
    identity = operator.apply(nominal, 1.0)
    audit = operator.audit(nominal, identity, 1.0)
    assert identity is not nominal
    assert audit.passed
    assert audit.changed_leaves == ()
    assert audit.changed_element_count == 0


def test_mass_inertia_coupling_and_closed_leaf_allowlist_fail_closed() -> None:
    with pytest.raises(ValueError, match="same body indices"):
        _entry(
            MASS_INERTIA_OPERATOR,
            (LeafSelection("body_mass", (1,)), LeafSelection("body_inertia", (2,))),
        )
    with pytest.raises(ValueError, match="forbids"):
        _entry(
            JOINT_DAMPING_OPERATOR,
            (LeafSelection("body_mass", (1,)),),
        )


def test_factor_roles_are_literal_and_heldout_splits_cannot_overlap() -> None:
    with pytest.raises(ValueError, match="cannot overlap"):
        FactorDefinition(
            "leaked", 0.5, frozenset({SOURCE_ROLE, CONFIRMATORY_ROLE})
        )
    entry = _entry(
        JOINT_DAMPING_OPERATOR,
        (LeafSelection("dof_damping", (1,), require_nonzero=True),),
    )
    registry = AxisRegistry({entry.axis_id: entry})
    assert registry.require(
        task_id="TaskA", axis_id="axis-a", factor_id="high", role=SOURCE_ROLE
    )[1].value == 2.0
    with pytest.raises(KeyError):
        registry.require(
            task_id="TaskA", axis_id="axis-a", factor_id="1.234", role=SOURCE_ROLE
        )
    with pytest.raises(ValueError, match="split role"):
        registry.require(
            task_id="TaskA", axis_id="axis-a", factor_id="confirm", role=SOURCE_ROLE
        )


def test_formal_scope_requires_six_tasks_two_axes_and_three_sources() -> None:
    entries: dict[str, AxisRegistryEntry] = {}
    for task_index in range(6):
        task = f"Task{task_index}"
        for axis_index in range(2):
            axis = f"{task}-axis-{axis_index}"
            entries[axis] = _entry(
                JOINT_DAMPING_OPERATOR,
                (LeafSelection("dof_damping", (1,), require_nonzero=True),),
                axis=axis,
                task=task,
            )
    registry = AxisRegistry(entries)
    registry.validate_formal_scope(tuple(f"Task{index}" for index in range(6)))
    with pytest.raises(ValueError, match="six tasks"):
        registry.validate_formal_scope(("Task0", "Task1"))


def test_zero_required_selection_and_source_mutation_are_rejected() -> None:
    entry = _entry(
        JOINT_DAMPING_OPERATOR,
        (LeafSelection("dof_damping", (0,), require_nonzero=True),),
    )
    with pytest.raises(Exception, match="zero"):
        make_operator(entry).apply(FakeEnvironment(_model()), 2.0)


def test_registry_supports_task_local_axis_ids_without_digest_ambiguity() -> None:
    left = _entry(
        JOINT_DAMPING_OPERATOR,
        (LeafSelection("dof_damping", (1,), require_nonzero=True),),
        axis="axis-a",
        task="TaskA",
    )
    right = _entry(
        JOINT_DAMPING_OPERATOR,
        (LeafSelection("dof_damping", (2,), require_nonzero=True),),
        axis="axis-a",
        task="TaskB",
    )
    registry = AxisRegistry({"TaskA::axis-a": left, "TaskB::axis-a": right})
    resolved, _ = registry.require(
        task_id="TaskB",
        axis_id="axis-a",
        factor_id="nominal",
        role=SOURCE_ROLE,
    )
    assert resolved.task_id == "TaskB"
    assert registry.digest == AxisRegistry({"left": left, "right": right}).digest
