"""Exact formal training-grid validation against the canonical config freeze."""

from __future__ import annotations

from typing import Any, Mapping

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .anchor_binding import AnchorManifest
    from .provenance import (
        ContractError,
        FORMAL_EXECUTION_PURPOSE,
        validate_formal_freeze_binding,
        validate_training_plan,
        validate_training_protocol,
    )
except ImportError:  # pragma: no cover - executable entry points
    from anchor_binding import AnchorManifest
    from provenance import (
        ContractError,
        FORMAL_EXECUTION_PURPOSE,
        validate_formal_freeze_binding,
        validate_training_plan,
        validate_training_protocol,
    )


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where} must be a positive integer")
    return value


def planned_environment_steps(protocol: Mapping[str, Any]) -> int:
    """Return the exact native transitions executed by the frozen outer loop."""

    validated = validate_training_protocol(protocol)
    config = validated["trainer_config"]
    num_envs = _positive_int(config.get("num_envs"), "trainer_config.num_envs")
    num_minibatches = _positive_int(
        config.get("num_minibatches"), "trainer_config.num_minibatches"
    )
    batch_size = _positive_int(config.get("batch_size"), "trainer_config.batch_size")
    unroll_length = _positive_int(
        config.get("unroll_length"), "trainer_config.unroll_length"
    )
    transitions_per_outer = num_minibatches * batch_size * unroll_length
    if transitions_per_outer % num_envs != 0:
        raise ContractError(
            "formal trainer transition geometry must divide exactly by num_envs"
        )
    maximum = _positive_int(
        validated["max_outer_iterations"], "max_outer_iterations"
    )
    return transitions_per_outer * maximum


def validate_formal_training_projection(
    plan: Mapping[str, Any],
    formal_freeze_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact config-owned anchor × seed grid and trainer semantics."""

    validated_plan = validate_training_plan(plan)
    if validated_plan["execution_purpose"] != FORMAL_EXECUTION_PURPOSE:
        raise ContractError("formal training projection requires v02_freeze_ready purpose")
    binding = validate_formal_freeze_binding(formal_freeze_binding)
    if validated_plan["formal_protocol_freeze_digest"] != binding["binding_digest"]:
        raise ContractError("formal plan is bound to another protocol freeze")
    if validated_plan["config_digest"] != binding["config_digest"]:
        raise ContractError("formal plan config differs from its protocol freeze")

    contract = binding["training_contract"]
    expected_anchors = set(contract["source_anchor_ids"])
    expected_anchor_semantics = {
        row["source_anchor_id"]: row for row in contract["source_anchors"]
    }
    expected_seeds = tuple(contract["training_seeds"])
    expected_grid = {
        (anchor_id, seed)
        for anchor_id in expected_anchors
        for seed in expected_seeds
    }
    observed_grid: set[tuple[str, int]] = set()
    protocol_digests: set[str] = set()
    for job in validated_plan["jobs"]:
        anchor = AnchorManifest.from_path(job["anchor_manifest_path"])
        if anchor.manifest_digest != job["anchor_manifest_digest"]:
            raise ContractError("formal job anchor manifest digest drifted")
        semantics = expected_anchor_semantics.get(anchor.anchor_id)
        if semantics is None:
            raise ContractError("formal job uses an anchor absent from config")
        if anchor.task != semantics["task"]:
            raise ContractError("formal anchor task differs from config")
        if anchor.nominal is not semantics["nominal"]:
            raise ContractError("formal anchor nominal flag differs from config")
        if anchor.factor != float(semantics["factor"]):
            raise ContractError("formal anchor factor differs from config")
        if anchor.axis_binding_digest != semantics["axis_binding_digest"]:
            raise ContractError("formal anchor axis binding differs from config")
        if anchor.nominal:
            if anchor.operator is not None:
                raise ContractError("formal nominal anchor unexpectedly has an operator")
        else:
            assert anchor.operator is not None  # guaranteed by AnchorManifest
            if anchor.operator.axis_id != semantics["axis_id"]:
                raise ContractError("formal anchor axis differs from config")
            if anchor.operator.operator_id != semantics["operator_id"]:
                raise ContractError("formal anchor operator differs from config")
            mutation_leaves = sorted(item.leaf for item in anchor.operator.mutations)
            if mutation_leaves != semantics["leaf_allowlist"]:
                raise ContractError("formal anchor mutation leaves differ from config")
        observed_grid.add((anchor.anchor_id, job["seed"]))
        protocol = job["training_protocol"]
        protocol_digests.add(job["training_protocol_digest"])
        if protocol["algorithm"] != contract["primary_algorithm"]:
            raise ContractError("formal protocol algorithm differs from config")
        planned_steps = planned_environment_steps(protocol)
        if planned_steps != contract["training_steps"]:
            raise ContractError("formal planned environment steps differ from config")
        if protocol["trainer_config"]["num_timesteps"] != contract["training_steps"]:
            raise ContractError("formal trainer num_timesteps differs from config")
        if protocol["checkpoint_rule"] != contract["checkpoint_rule"]:
            raise ContractError("formal checkpoint rule differs from config")

    if len(protocol_digests) != 1:
        raise ContractError("formal grid must use exactly one training protocol")
    if len(validated_plan["jobs"]) != len(expected_grid) or observed_grid != expected_grid:
        missing = sorted(expected_grid - observed_grid)
        unexpected = sorted(observed_grid - expected_grid)
        raise ContractError(
            "formal anchor/seed grid differs from config; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    return validated_plan


__all__ = ["planned_environment_steps", "validate_formal_training_projection"]
