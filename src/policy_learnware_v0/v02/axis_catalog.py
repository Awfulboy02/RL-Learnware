"""Auditable *candidate* axis catalog for the six v0.2 DMC tasks.

This module deliberately does not define formal factors or choose between the
``or`` alternatives in the coding plan.  A caller must provide the three
provisional source factors explicitly, and a reviewed selection ledger must
reduce this candidate catalog to exactly two axes per task before
``AxisRegistry.validate_formal_scope`` can pass.

The row indices below are tied to the pinned MuJoCo Playground model layout.
Human-readable names are retained as review evidence; the executable contract
remains the exact leaf/index allowlist in :class:`AxisRegistryEntry`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

from .axes import (
    ACTUATOR_GAIN_OPERATOR,
    CONTACT_FRICTION_OPERATOR,
    JOINT_DAMPING_OPERATOR,
    MASS_INERTIA_OPERATOR,
    SAFETY_ROLE,
    SOURCE_ROLE,
    AxisRegistry,
    AxisRegistryEntry,
    FactorDefinition,
    LeafSelection,
    operator_source_digest,
)


V02_TASKS = (
    "CartpoleSwingup",
    "CheetahRun",
    "FingerTurnEasy",
    "FishSwim",
    "HopperHop",
    "WalkerWalk",
)
BACKEND_ID = "mujoco_playground.registry"


@dataclass(frozen=True)
class CandidateAxisEvidence:
    """Human-readable evidence paired with one executable candidate entry."""

    axis_id: str
    task_id: str
    semantic_label: str
    selected_names: tuple[str, ...]
    plan_position: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.axis_id, self.task_id, self.semantic_label, self.plan_position)
        ):
            raise ValueError("candidate-axis evidence strings must be non-empty")
        if not self.selected_names or any(not item for item in self.selected_names):
            raise ValueError("candidate-axis evidence requires selected model names")

    def to_dict(self) -> dict[str, object]:
        return {
            "axis_id": self.axis_id,
            "task_id": self.task_id,
            "semantic_label": self.semantic_label,
            "selected_names": list(self.selected_names),
            "plan_position": self.plan_position,
        }


def _source_factors(values: Sequence[float]) -> tuple[FactorDefinition, ...]:
    factors = tuple(float(value) for value in values)
    if len(factors) != 3 or factors[1] != 1.0:
        raise ValueError("candidate source factors must be exactly [low, 1.0, high]")
    if any(not math.isfinite(value) or value <= 0.0 for value in factors):
        raise ValueError("candidate source factors must be finite and positive")
    if not factors[0] < 1.0 < factors[2]:
        raise ValueError("candidate source factors must straddle the nominal identity")
    return (
        FactorDefinition("source_low", factors[0], frozenset({SOURCE_ROLE})),
        FactorDefinition(
            "source_nominal",
            1.0,
            frozenset({SOURCE_ROLE, SAFETY_ROLE}),
        ),
        FactorDefinition("source_high", factors[2], frozenset({SOURCE_ROLE})),
    )


def _mass_entry(
    *, axis_id: str, task_id: str, indices: tuple[int, ...], factors: tuple[FactorDefinition, ...]
) -> AxisRegistryEntry:
    selections = (
        LeafSelection("body_mass", indices, require_nonzero=True),
        LeafSelection("body_inertia", indices, require_nonzero=True),
    )
    return _entry(
        axis_id=axis_id,
        task_id=task_id,
        operator_id=MASS_INERTIA_OPERATOR,
        selections=selections,
        factors=factors,
    )


def _gain_entry(
    *, axis_id: str, task_id: str, indices: tuple[int, ...], factors: tuple[FactorDefinition, ...]
) -> AxisRegistryEntry:
    # Scale gainprm[*, 0] once.  Scaling gain and gear together would silently
    # square the intended multiplicative force factor.
    return _entry(
        axis_id=axis_id,
        task_id=task_id,
        operator_id=ACTUATOR_GAIN_OPERATOR,
        selections=(
            LeafSelection(
                "actuator_gainprm",
                indices,
                components=(0,),
                require_nonzero=True,
            ),
        ),
        factors=factors,
    )


def _damping_entry(
    *, axis_id: str, task_id: str, indices: tuple[int, ...], factors: tuple[FactorDefinition, ...]
) -> AxisRegistryEntry:
    return _entry(
        axis_id=axis_id,
        task_id=task_id,
        operator_id=JOINT_DAMPING_OPERATOR,
        selections=(LeafSelection("dof_damping", indices, require_nonzero=True),),
        factors=factors,
    )


def _friction_entry(
    *, axis_id: str, task_id: str, indices: tuple[int, ...], factors: tuple[FactorDefinition, ...]
) -> AxisRegistryEntry:
    return _entry(
        axis_id=axis_id,
        task_id=task_id,
        operator_id=CONTACT_FRICTION_OPERATOR,
        selections=(
            LeafSelection(
                "geom_friction",
                indices,
                components=(0, 1, 2),
                require_nonzero=True,
            ),
        ),
        factors=factors,
    )


def _entry(
    *,
    axis_id: str,
    task_id: str,
    operator_id: str,
    selections: tuple[LeafSelection, ...],
    factors: tuple[FactorDefinition, ...],
) -> AxisRegistryEntry:
    return AxisRegistryEntry(
        axis_id=axis_id,
        task_id=task_id,
        backend_id=BACKEND_ID,
        operator_id=operator_id,
        # The executable operator implementation currently exposes version 1;
        # candidate/formal status belongs in the selection ledger, not in the
        # runtime operator-version field.
        operator_version="1",
        operator_digest=operator_source_digest(),
        selections=selections,
        factors=factors,
    )


def build_candidate_axis_catalog(
    source_factors: Sequence[float],
) -> tuple[AxisRegistry, Mapping[str, CandidateAxisEvidence]]:
    """Build the pinned-layout candidate catalog without making a formal choice.

    ``source_factors`` is intentionally mandatory.  Supplying discovery values
    here records an engineering attempt; it does not satisfy the human freeze
    required by the coding plan.
    """

    factors = _source_factors(source_factors)
    entries: dict[str, AxisRegistryEntry] = {}
    evidence: dict[str, CandidateAxisEvidence] = {}

    def add(entry: AxisRegistryEntry, *, label: str, names: tuple[str, ...], position: str) -> None:
        if entry.axis_id in entries:
            raise AssertionError(f"duplicate built-in candidate axis: {entry.axis_id}")
        entries[entry.axis_id] = entry
        evidence[entry.axis_id] = CandidateAxisEvidence(
            axis_id=entry.axis_id,
            task_id=entry.task_id,
            semantic_label=label,
            selected_names=names,
            plan_position=position,
        )

    add(
        _mass_entry(
            axis_id="cartpole_mass_inertia",
            task_id="CartpoleSwingup",
            indices=(1, 2),
            factors=factors,
        ),
        label="cart and pole coupled mass/inertia",
        names=("cart", "pole_1"),
        position="recommended axis A",
    )
    add(
        _gain_entry(
            axis_id="cartpole_actuator_gain",
            task_id="CartpoleSwingup",
            indices=(0,),
            factors=factors,
        ),
        label="slide actuator gain",
        names=("slide",),
        position="candidate axis B",
    )
    add(
        _damping_entry(
            axis_id="cartpole_hinge_damping",
            task_id="CartpoleSwingup",
            indices=(1,),
            factors=factors,
        ),
        label="pole hinge damping",
        names=("hinge_1",),
        position="candidate axis B fallback",
    )

    add(
        _mass_entry(
            axis_id="cheetah_mass_inertia",
            task_id="CheetahRun",
            indices=(1, 2, 3, 4, 5, 6, 7),
            factors=factors,
        ),
        label="all articulated cheetah bodies coupled mass/inertia",
        names=("torso", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"),
        position="recommended axis A",
    )
    add(
        _gain_entry(
            axis_id="cheetah_actuator_gain",
            task_id="CheetahRun",
            indices=(0, 1, 2, 3, 4, 5),
            factors=factors,
        ),
        label="all cheetah actuator gains",
        names=("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"),
        position="recommended axis B",
    )

    add(
        _mass_entry(
            axis_id="finger_spinner_mass_inertia",
            task_id="FingerTurnEasy",
            indices=(3,),
            factors=factors,
        ),
        label="spinner object coupled mass/inertia",
        names=("spinner",),
        position="recommended axis A",
    )
    add(
        _damping_entry(
            axis_id="finger_joint_damping",
            task_id="FingerTurnEasy",
            indices=(0, 1, 2),
            factors=factors,
        ),
        label="finger and spinner hinge damping",
        names=("proximal", "distal", "hinge"),
        position="candidate axis B",
    )
    add(
        _friction_entry(
            axis_id="finger_object_contact_friction",
            task_id="FingerTurnEasy",
            indices=(4, 5, 6),
            factors=factors,
        ),
        label="fingertip/spinner contact friction",
        names=("fingertip", "cap1", "cap2"),
        position="candidate axis B fallback",
    )

    add(
        _mass_entry(
            axis_id="fish_body_mass_inertia",
            task_id="FishSwim",
            indices=(2, 3, 4, 5, 6),
            factors=factors,
        ),
        label="fish body excluding target coupled mass/inertia",
        names=("torso", "tail1", "tail2", "finright", "finleft"),
        position="recommended axis A",
    )
    add(
        _gain_entry(
            axis_id="fish_actuator_gain",
            task_id="FishSwim",
            indices=(0, 1, 2, 3, 4),
            factors=factors,
        ),
        label="all fish actuator gains",
        names=("tail", "tail_twist", "fins_flap", "finleft_pitch", "finright_pitch"),
        position="candidate axis B",
    )
    add(
        _damping_entry(
            axis_id="fish_joint_damping",
            task_id="FishSwim",
            indices=(6, 7, 8, 9, 10, 11, 12),
            factors=factors,
        ),
        label="articulated tail and fin damping",
        names=(
            "tail1",
            "tail_twist",
            "tail2",
            "finright_roll",
            "finright_pitch",
            "finleft_roll",
            "finleft_pitch",
        ),
        position="candidate axis B fallback",
    )

    add(
        _mass_entry(
            axis_id="hopper_mass_inertia",
            task_id="HopperHop",
            indices=(1, 2, 3, 4, 5),
            factors=factors,
        ),
        label="all articulated hopper bodies coupled mass/inertia",
        names=("torso", "pelvis", "thigh", "calf", "foot"),
        position="recommended axis A",
    )
    add(
        _gain_entry(
            axis_id="hopper_actuator_gain",
            task_id="HopperHop",
            indices=(0, 1, 2, 3),
            factors=factors,
        ),
        label="all hopper actuator gains",
        names=("waist", "hip", "knee", "ankle"),
        position="candidate axis B",
    )
    add(
        _friction_entry(
            axis_id="hopper_ground_contact_friction",
            task_id="HopperHop",
            indices=(0, 6),
            factors=factors,
        ),
        label="floor and foot contact friction",
        names=("floor", "foot"),
        position="candidate axis B fallback",
    )

    add(
        _mass_entry(
            axis_id="walker_mass_inertia",
            task_id="WalkerWalk",
            indices=(1, 2, 3, 4, 5, 6, 7),
            factors=factors,
        ),
        label="all articulated walker bodies coupled mass/inertia",
        names=(
            "torso",
            "right_thigh",
            "right_leg",
            "right_foot",
            "left_thigh",
            "left_leg",
            "left_foot",
        ),
        position="recommended axis A",
    )
    add(
        _gain_entry(
            axis_id="walker_actuator_gain",
            task_id="WalkerWalk",
            indices=(0, 1, 2, 3, 4, 5),
            factors=factors,
        ),
        label="all walker actuator gains",
        names=("right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"),
        position="candidate axis B",
    )
    add(
        _friction_entry(
            axis_id="walker_ground_contact_friction",
            task_id="WalkerWalk",
            indices=(0, 4, 7),
            factors=factors,
        ),
        label="floor and both foot contact friction",
        names=("floor", "right_foot", "left_foot"),
        position="candidate axis B fallback",
    )

    return AxisRegistry(entries), MappingProxyType(evidence)


def reviewed_axis_subset(
    candidate_registry: AxisRegistry,
    selected_axis_ids: Mapping[str, Sequence[str]],
) -> AxisRegistry:
    """Reduce candidates to a reviewed two-axis-per-task formal registry."""

    if set(selected_axis_ids) != set(V02_TASKS):
        raise ValueError("reviewed selection must name exactly the six v0.2 tasks")
    selected: dict[str, AxisRegistryEntry] = {}
    for task in V02_TASKS:
        axis_ids = tuple(selected_axis_ids[task])
        if len(axis_ids) != 2 or len(set(axis_ids)) != 2:
            raise ValueError(f"{task} must select exactly two distinct candidate axes")
        for axis_id in axis_ids:
            if axis_id not in candidate_registry.entries:
                raise KeyError(f"unknown candidate axis: {axis_id!r}")
            entry = candidate_registry.entries[axis_id]
            if entry.task_id != task:
                raise ValueError(f"candidate axis {axis_id!r} does not belong to {task}")
            selected[axis_id] = entry
    registry = AxisRegistry(selected)
    registry.validate_formal_scope(V02_TASKS)
    return registry


__all__ = [
    "BACKEND_ID",
    "CandidateAxisEvidence",
    "V02_TASKS",
    "build_candidate_axis_catalog",
    "reviewed_axis_subset",
]
