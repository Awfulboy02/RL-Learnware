"""Strict bridge from frozen experiment config to executable axis registry.

Config parsing and native model mutation intentionally remain separate.  This
bridge is the mandatory compatibility check between them: a config cannot
invent an operator, leaf, task/axis pair, or factor that is absent from the
reviewed registry catalog.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .axes import (
    CONFIRMATORY_ROLE,
    DEVELOPMENT_ROLE,
    SAFETY_ROLE,
    SOURCE_ROLE,
    AxisRegistry,
    AxisRegistryEntry,
    FactorDefinition,
)


class AxisConfigBindingError(ValueError):
    """A frozen config and reviewed executable registry disagree."""


def _candidate(
    registry: AxisRegistry,
    *,
    task_id: str,
    axis_id: str,
) -> AxisRegistryEntry:
    matches = tuple(
        entry
        for entry in registry.entries.values()
        if entry.task_id == task_id and entry.axis_id == axis_id
    )
    if not matches:
        raise AxisConfigBindingError(
            f"config references unreviewed task/axis pair {task_id!r}/{axis_id!r}"
        )
    if len(matches) != 1:
        raise AxisConfigBindingError(
            f"reviewed catalog has ambiguous task/axis pair {task_id!r}/{axis_id!r}"
        )
    return matches[0]


def _registered_leaf_paths(entry: AxisRegistryEntry) -> tuple[str, ...]:
    return tuple(sorted(f"_mjx_model.{selection.leaf}" for selection in entry.selections))


def _factor_definitions(config: Any, *, task_id: str, axis_id: str) -> tuple[FactorDefinition, ...]:
    source_rows = tuple(config.source_factors[task_id][axis_id])
    safety_rows = tuple(config.safety_exact_targets)
    safety_refs = {
        row.source_anchor_ref
        for row in safety_rows
        if row.task_id == task_id
        and row.source_anchor_ref is not None
        and (row.axis_id is None or row.axis_id == axis_id)
    }
    definitions: dict[str, FactorDefinition] = {}
    for row in source_rows:
        roles = {SOURCE_ROLE}
        # Freeze-ready records the exact-recurrence *capability* on the shared
        # nominal source anchor without instantiating a Paper-I safety target.
        # A later joint manifest may bind that capability through
        # ``source_anchor_ref``; development/smoke configs may already carry a
        # concrete reference and are checked by the same rule.
        if row.is_nominal or row.source_anchor_id in safety_refs:
            roles.add(SAFETY_ROLE)
        definitions[row.factor_id] = FactorDefinition(
            row.factor_id,
            row.value,
            frozenset(roles),
        )

    target_sets = (
        (tuple(config.development_targets), DEVELOPMENT_ROLE),
        (tuple(config.confirmatory_targets), CONFIRMATORY_ROLE),
    )
    for rows, expected_role in target_sets:
        for row in rows:
            if row.task_id != task_id or row.axis_id != axis_id:
                continue
            if tuple(row.roles) != (expected_role,):
                raise AxisConfigBindingError(
                    f"target {row.target_id!r} has a role inconsistent with its split"
                )
            if row.factor_id in definitions:
                raise AxisConfigBindingError(
                    f"factor ID {row.factor_id!r} is reused across source/target splits"
                )
            definitions[row.factor_id] = FactorDefinition(
                row.factor_id,
                row.factor_value,
                frozenset({expected_role}),
            )
    return tuple(
        sorted(definitions.values(), key=lambda item: (item.value, item.factor_id))
    )


def axis_registry_from_config(
    config: Any,
    reviewed_catalog: AxisRegistry,
) -> AxisRegistry:
    """Bind every configured task/axis/factor to reviewed executable literals.

    The function accepts the immutable ``V02ExperimentConfig`` interface but
    avoids importing it at runtime, keeping native-axis code dependency-light.
    Formal configs additionally pass the exact six-by-two registry check.
    """

    required = (
        "tasks",
        "dynamics_axes",
        "source_factors",
        "development_targets",
        "confirmatory_targets",
        "safety_exact_targets",
        "stage",
    )
    missing = tuple(name for name in required if not hasattr(config, name))
    if missing:
        raise AxisConfigBindingError(f"config lacks axis bridge fields: {missing}")

    entries: dict[str, AxisRegistryEntry] = {}
    for task_id in tuple(config.tasks):
        for configured in tuple(config.dynamics_axes[task_id]):
            candidate = _candidate(
                reviewed_catalog,
                task_id=task_id,
                axis_id=configured.axis_id,
            )
            checks = {
                "operator_id": configured.operator_id == candidate.operator_id,
                "operator_digest": configured.operator_digest == candidate.operator_digest,
                "leaf_allowlist": tuple(sorted(configured.leaf_allowlist))
                == _registered_leaf_paths(candidate),
                "static_within_episode": configured.static_within_episode is True,
            }
            failed = tuple(name for name, passed in checks.items() if not passed)
            if failed:
                raise AxisConfigBindingError(
                    f"config/registry mismatch for {task_id}/{configured.axis_id}: {failed}"
                )
            factors = _factor_definitions(
                config,
                task_id=task_id,
                axis_id=configured.axis_id,
            )
            bound = replace(candidate, factors=factors)
            entries[f"{task_id}::{configured.axis_id}"] = bound

    registry = AxisRegistry(entries)
    if config.stage == "v02_freeze_ready":
        registry.validate_formal_scope(tuple(config.tasks))
        if config.confirmatory_targets or config.safety_exact_targets:
            raise AxisConfigBindingError(
                "v02_freeze_ready cannot instantiate Paper-I sealed target IDs"
            )
    return registry


__all__ = ["AxisConfigBindingError", "axis_registry_from_config"]
