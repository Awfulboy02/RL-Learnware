"""Closed registry for the single approved v0.1 dynamics intervention."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..hashing import sha256_file, sha256_json
from .config import APPROVED_GRID, APPROVED_SHIFT_ID, APPROVED_TASKS


SHIFT_REGISTRY_ENTRY_SCHEMA = "policy-learnware.v01-shift-registry-entry.v0"
SHIFT_REGISTRY_SCHEMA = "policy-learnware.v01-shift-registry.v0"
MUTATION_STAGE = "post_registry_load_pre_jit"
ALLOWLISTED_MODEL_LEAF = "_mjx_model.dof_damping"
SELECTION_RULE = "original_value_nonzero"
REQUIRED_INVARIANTS = (
    "only_allowlisted_model_leaf_changes",
    "zero_entries_remain_zero",
    "nonzero_entries_equal_factor_times_nominal",
    "shape_dtype_and_index_order_unchanged",
    "fresh_instance_no_in_place_source_mutation",
    "schema_reward_reset_horizon_action_repeat_unchanged",
    "static_within_episode",
)


def _digest(value: str, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{where} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a SHA-256 hex digest") from exc
    return value.lower()


@dataclass(frozen=True)
class ShiftRegistryEntry:
    shift_id: str
    backend: str
    allowed_tasks: tuple[str, ...]
    nominal_factor: float
    allowed_factors: tuple[float, ...]
    mutation_stage: str
    allowlisted_model_leaf: str
    selection_rule: str
    operator_source_sha256: str
    invariants: tuple[str, ...]
    schema: str = SHIFT_REGISTRY_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHIFT_REGISTRY_ENTRY_SCHEMA:
            raise ValueError(f"unsupported ShiftRegistryEntry schema: {self.schema!r}")
        if self.shift_id != APPROVED_SHIFT_ID:
            raise ValueError(f"unregistered shift_id: {self.shift_id!r}")
        if self.backend != "mujoco_playground.registry":
            raise ValueError("v0.1 shift backend must be mujoco_playground.registry")
        if tuple(self.allowed_tasks) != ("WalkerWalk", "FingerTurnEasy"):
            raise ValueError("registry task order and membership are frozen")
        if self.nominal_factor != 1.0 or tuple(self.allowed_factors) != APPROVED_GRID:
            raise ValueError("registry factor grid is frozen")
        if self.mutation_stage != MUTATION_STAGE:
            raise ValueError(f"mutation_stage must be {MUTATION_STAGE}")
        if self.allowlisted_model_leaf != ALLOWLISTED_MODEL_LEAF:
            raise ValueError("arbitrary model attribute paths are forbidden")
        if self.selection_rule != SELECTION_RULE:
            raise ValueError(f"selection_rule must be {SELECTION_RULE}")
        object.__setattr__(self, "operator_source_sha256", _digest(
            self.operator_source_sha256, "operator_source_sha256"
        ))
        if tuple(self.invariants) != REQUIRED_INVARIANTS:
            raise ValueError("registry invariants are incomplete or reordered")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "shift_id": self.shift_id,
            "backend": self.backend,
            "allowed_tasks": list(self.allowed_tasks),
            "nominal_factor": self.nominal_factor,
            "allowed_factors": list(self.allowed_factors),
            "mutation_stage": self.mutation_stage,
            "allowlisted_model_leaf": self.allowlisted_model_leaf,
            "selection_rule": self.selection_rule,
            "operator_source_sha256": self.operator_source_sha256,
            "invariants": list(self.invariants),
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShiftRegistryEntry":
        fields = {
            "schema", "shift_id", "backend", "allowed_tasks", "nominal_factor",
            "allowed_factors", "mutation_stage", "allowlisted_model_leaf", "selection_rule",
            "operator_source_sha256", "invariants",
        }
        missing = fields - set(value)
        unknown = set(value) - fields
        if missing or unknown:
            raise ValueError(
                f"invalid ShiftRegistryEntry keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(
            schema=str(value["schema"]),
            shift_id=str(value["shift_id"]),
            backend=str(value["backend"]),
            allowed_tasks=tuple(str(item) for item in value["allowed_tasks"]),
            nominal_factor=float(value["nominal_factor"]),
            allowed_factors=tuple(float(item) for item in value["allowed_factors"]),
            mutation_stage=str(value["mutation_stage"]),
            allowlisted_model_leaf=str(value["allowlisted_model_leaf"]),
            selection_rule=str(value["selection_rule"]),
            operator_source_sha256=str(value["operator_source_sha256"]),
            invariants=tuple(str(item) for item in value["invariants"]),
        )


@dataclass(frozen=True)
class ResolvedShift:
    """An approved entry paired with the digest of its enclosing registry."""

    entry: ShiftRegistryEntry
    registry_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "registry_digest", _digest(self.registry_digest, "registry_digest"))

    def __getattr__(self, name: str) -> Any:
        # Keeps the operator-facing API concise while preserving the no-cycle
        # distinction between an entry digest and the enclosing registry digest.
        return getattr(self.entry, name)


@dataclass(frozen=True)
class ShiftRegistry:
    entries: Mapping[str, ShiftRegistryEntry]
    schema: str = SHIFT_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SHIFT_REGISTRY_SCHEMA:
            raise ValueError(f"unsupported ShiftRegistry schema: {self.schema!r}")
        if set(self.entries) != {APPROVED_SHIFT_ID}:
            raise ValueError("ShiftRegistry is closed to the single approved intervention")
        entry = self.entries[APPROVED_SHIFT_ID]
        if entry.shift_id != APPROVED_SHIFT_ID:
            raise ValueError("registry key must match entry.shift_id")
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "entries": {
                shift_id: entry.to_dict()
                for shift_id, entry in sorted(self.entries.items())
            },
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def require(self, shift_id: str, task: str, factor: float) -> ResolvedShift:
        if not isinstance(shift_id, str) or shift_id not in self.entries:
            raise KeyError(f"unknown or unregistered shift_id: {shift_id!r}")
        if not isinstance(task, str) or task not in APPROVED_TASKS:
            raise ValueError(f"unsupported task for v0.1 shift: {task!r}")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise ValueError("shift factor must be numeric")
        factor_value = float(factor)
        entry = self.entries[shift_id]
        if task not in entry.allowed_tasks:
            raise ValueError(f"shift {shift_id!r} is not registered for task {task!r}")
        if factor_value not in entry.allowed_factors:
            raise ValueError(
                f"factor {factor_value!r} is not in the frozen grid {entry.allowed_factors}"
            )
        return ResolvedShift(entry=entry, registry_digest=self.digest)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShiftRegistry":
        fields = {"schema", "entries"}
        missing = fields - set(value)
        unknown = set(value) - fields
        if missing or unknown or not isinstance(value["entries"], Mapping):
            raise ValueError("invalid ShiftRegistry payload")
        return cls(
            schema=str(value["schema"]),
            entries={
                str(name): ShiftRegistryEntry.from_dict(raw)
                for name, raw in value["entries"].items()
            },
        )


def operator_source_digest() -> str:
    """Digest the pinned operator source without importing its heavy runtime."""

    source = Path(__file__).with_name("variant_env.py")
    if not source.is_file():
        raise FileNotFoundError(
            "v0.1 variant_env.py is absent; a formal registry cannot be materialized"
        )
    return sha256_file(source)


def default_shift_registry(*, operator_digest: str | None = None) -> ShiftRegistry:
    digest = operator_source_digest() if operator_digest is None else _digest(
        operator_digest, "operator_digest"
    )
    entry = ShiftRegistryEntry(
        shift_id=APPROVED_SHIFT_ID,
        backend="mujoco_playground.registry",
        allowed_tasks=("WalkerWalk", "FingerTurnEasy"),
        nominal_factor=1.0,
        allowed_factors=APPROVED_GRID,
        mutation_stage=MUTATION_STAGE,
        allowlisted_model_leaf=ALLOWLISTED_MODEL_LEAF,
        selection_rule=SELECTION_RULE,
        operator_source_sha256=digest,
        invariants=REQUIRED_INVARIANTS,
    )
    return ShiftRegistry(entries={entry.shift_id: entry})


__all__ = [
    "ALLOWLISTED_MODEL_LEAF", "MUTATION_STAGE", "REQUIRED_INVARIANTS",
    "ResolvedShift", "SELECTION_RULE", "SHIFT_REGISTRY_ENTRY_SCHEMA",
    "SHIFT_REGISTRY_SCHEMA", "ShiftRegistry", "ShiftRegistryEntry",
    "default_shift_registry", "operator_source_digest",
]
