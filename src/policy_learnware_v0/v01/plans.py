"""Pre-result, candidate-independent v0.1 measurement work plans."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..hashing import sha256_json


PAIR_PLAN_SCHEMA = "policy-learnware.v01-pair-plan.v0"


def build_pair_plan(
    variants: Sequence[Mapping[str, Any]],
    *,
    banks: int,
    gate_prefix: int,
    routing_prefix: int,
    within_bank_pairs: Sequence[Sequence[int]],
    nominal_factor: float = 1.0,
) -> dict[str, Any]:
    """Materialize the exact sparse plan before any TaskSpec result exists.

    ``variants`` is a private freeze-time projection containing only
    ``task``, ``factor`` and the already-derived opaque ``variant_id``.  Neither
    task nor factor is copied into the returned measurement artifact.
    """

    if banks <= 0 or gate_prefix <= 0 or routing_prefix < gate_prefix:
        raise ValueError("invalid pair-plan episode counts")
    pairs = tuple(tuple(int(value) for value in pair) for pair in within_bank_pairs)
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("every within-bank pair must contain exactly two banks")
    flattened = [value for pair in pairs for value in pair]
    if sorted(flattened) != list(range(banks)):
        raise ValueError("within-bank pairs must partition every bank exactly once")
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    seen_ids: set[str] = set()
    for record in variants:
        task = str(record["task"])
        variant_id = str(record["variant_id"])
        factor = float(record["factor"])
        if not task or not variant_id or variant_id in seen_ids:
            raise ValueError("variant records have missing/duplicate identity")
        seen_ids.add(variant_id)
        by_task.setdefault(task, []).append(
            {"task": task, "variant_id": variant_id, "factor": factor}
        )
    within: list[dict[str, Any]] = []
    between: list[dict[str, Any]] = []
    routing: list[dict[str, Any]] = []
    for task in sorted(by_task):
        records = sorted(by_task[task], key=lambda item: item["factor"])
        nominal = [item for item in records if item["factor"] == nominal_factor]
        if len(nominal) != 1:
            raise ValueError(f"{task} must contain exactly one nominal variant")
        nominal_id = str(nominal[0]["variant_id"])
        for record in records:
            variant_id = str(record["variant_id"])
            for left_bank, right_bank in pairs:
                within.append(
                    {
                        "left_variant_id": variant_id,
                        "left_bank": left_bank,
                        "right_variant_id": variant_id,
                        "right_bank": right_bank,
                        "prefix": gate_prefix,
                    }
                )
            for bank in range(banks):
                routing.append(
                    {
                        "variant_id": variant_id,
                        "bank": bank,
                        "prefix": routing_prefix,
                    }
                )
                if variant_id != nominal_id:
                    between.append(
                        {
                            "left_variant_id": nominal_id,
                            "left_bank": bank,
                            "right_variant_id": variant_id,
                            "right_bank": bank,
                            "prefix": gate_prefix,
                        }
                    )
    payload: dict[str, Any] = {
        "schema": PAIR_PLAN_SCHEMA,
        "within": within,
        "between": between,
        "routing": routing,
    }
    payload["plan_digest"] = sha256_json(payload)
    return payload


def verify_pair_plan(payload: Mapping[str, Any]) -> str:
    if set(payload) != {"schema", "within", "between", "routing", "plan_digest"}:
        raise ValueError("pair plan has missing or unknown fields")
    if payload["schema"] != PAIR_PLAN_SCHEMA:
        raise ValueError("unsupported pair-plan schema")
    digest = str(payload["plan_digest"])
    material = {key: payload[key] for key in ("schema", "within", "between", "routing")}
    if sha256_json(material) != digest:
        raise ValueError("pair plan digest mismatch")
    for family in ("within", "between"):
        for record in payload[family]:
            if set(record) != {
                "left_variant_id",
                "left_bank",
                "right_variant_id",
                "right_bank",
                "prefix",
            }:
                raise ValueError(f"invalid {family} pair-plan record")
    for record in payload["routing"]:
        if set(record) != {"variant_id", "bank", "prefix"}:
            raise ValueError("invalid routing pair-plan record")
    return digest


__all__ = ["PAIR_PLAN_SCHEMA", "build_pair_plan", "verify_pair_plan"]
