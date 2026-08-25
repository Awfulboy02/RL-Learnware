"""Cold/warm query-cost records with strict component reconciliation.

Oracle evaluation cost is intentionally absent from the deployable query
record.  A query record covers reset through selected-only deployment and is
bound to one frozen cost contract.  Cold and warm records are reconciled by
query identity; missing or duplicate modes fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json


ThermalMode = Literal["cold", "warm"]
COST_RECORD_SCHEMA = "policy-learnware.v02-query-cost-record.v0"
QUERY_COST_COMPONENTS = (
    "environment_setup",
    "probe_collection",
    "canonicalization",
    "encoder_inference",
    "rkme_construction",
    "market_matching",
    "selected_bundle_resolution",
    "selected_only_deployment",
)


class CostContractError(ValueError):
    """A timing record cannot be reconciled to the frozen cost contract."""


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise CostContractError(f"{where} must be a mapping")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise CostContractError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CostContractError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise CostContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise CostContractError(f"{where} must be a SHA-256 digest") from error
    return result


def _nonnegative_float(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CostContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CostContractError(f"{where} must be finite and non-negative")
    return result


def _finite_float(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise CostContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CostContractError(f"{where} must be finite")
    return result


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise CostContractError(f"{where} must be an integer")
    result = int(value)
    if result < 0:
        raise CostContractError(f"{where} must be non-negative")
    return result


def _components(value: Mapping[str, float], where: str) -> Mapping[str, float]:
    _strict(value, set(QUERY_COST_COMPONENTS), where)
    parsed = {
        name: _nonnegative_float(value[name], f"{where}.{name}")
        for name in QUERY_COST_COMPONENTS
    }
    return MappingProxyType(parsed)


@dataclass(frozen=True)
class CostRecord:
    """One immutable cold or warm end-to-end query timing record."""

    query_id: str
    mode: ThermalMode
    cost_contract_digest: str
    execution_attempt_id: str
    components_seconds: Mapping[str, float]
    total_seconds: float
    target_environment_steps: int
    target_gradient_steps: int = 0
    schema: str = COST_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COST_RECORD_SCHEMA:
            raise CostContractError(f"unsupported CostRecord schema: {self.schema!r}")
        query_id = _nonempty(self.query_id, "query_id")
        if self.mode not in {"cold", "warm"}:
            raise CostContractError("mode must be 'cold' or 'warm'")
        contract = _digest(self.cost_contract_digest, "cost_contract_digest")
        attempt = _nonempty(self.execution_attempt_id, "execution_attempt_id")
        components = _components(self.components_seconds, "components_seconds")
        total = _nonnegative_float(self.total_seconds, "total_seconds")
        component_total = math.fsum(components.values())
        if not math.isclose(total, component_total, rel_tol=1e-12, abs_tol=1e-12):
            raise CostContractError(
                "total_seconds does not reconcile with the registered query components"
            )
        environment_steps = _nonnegative_int(
            self.target_environment_steps, "target_environment_steps"
        )
        gradient_steps = _nonnegative_int(self.target_gradient_steps, "target_gradient_steps")
        if gradient_steps != 0:
            raise CostContractError("deployable v0.2 query cost requires zero target gradients")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "cost_contract_digest", contract)
        object.__setattr__(self, "execution_attempt_id", attempt)
        object.__setattr__(self, "components_seconds", components)
        object.__setattr__(self, "total_seconds", total)
        object.__setattr__(self, "target_environment_steps", environment_steps)
        object.__setattr__(self, "target_gradient_steps", gradient_steps)

    @property
    def component_total_seconds(self) -> float:
        return math.fsum(self.components_seconds.values())

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "query_id": self.query_id,
            "mode": self.mode,
            "cost_contract_digest": self.cost_contract_digest,
            "execution_attempt_id": self.execution_attempt_id,
            "components_seconds": dict(self.components_seconds),
            "total_seconds": self.total_seconds,
            "target_environment_steps": self.target_environment_steps,
            "target_gradient_steps": self.target_gradient_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CostRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "CostRecord")
        return cls(**{name: value[name] for name in fields})

    @classmethod
    def create(
        cls,
        *,
        query_id: str,
        mode: ThermalMode,
        cost_contract_digest: str,
        execution_attempt_id: str,
        components_seconds: Mapping[str, float],
        target_environment_steps: int,
        target_gradient_steps: int = 0,
    ) -> "CostRecord":
        parsed = _components(components_seconds, "components_seconds")
        return cls(
            query_id=query_id,
            mode=mode,
            cost_contract_digest=cost_contract_digest,
            execution_attempt_id=execution_attempt_id,
            components_seconds=parsed,
            total_seconds=math.fsum(parsed.values()),
            target_environment_steps=target_environment_steps,
            target_gradient_steps=target_gradient_steps,
        )


@dataclass(frozen=True)
class ModeCostSummary:
    mode: ThermalMode
    query_count: int
    total_seconds: float
    mean_total_seconds: float
    mean_components_seconds: Mapping[str, float]
    target_environment_steps: int

    def __post_init__(self) -> None:
        if self.mode not in {"cold", "warm"}:
            raise CostContractError("mode summary must be cold or warm")
        count = _nonnegative_int(self.query_count, "query_count")
        if count == 0:
            raise CostContractError("mode summary cannot be empty")
        total = _nonnegative_float(self.total_seconds, "total_seconds")
        mean_total = _nonnegative_float(self.mean_total_seconds, "mean_total_seconds")
        components = _components(self.mean_components_seconds, "mean_components_seconds")
        if not math.isclose(
            mean_total, math.fsum(components.values()), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CostContractError("mean total does not reconcile with mean components")
        if not math.isclose(total, count * mean_total, rel_tol=1e-12, abs_tol=1e-12):
            raise CostContractError("mode total does not reconcile with query mean")
        steps = _nonnegative_int(self.target_environment_steps, "target_environment_steps")
        object.__setattr__(self, "query_count", count)
        object.__setattr__(self, "total_seconds", total)
        object.__setattr__(self, "mean_total_seconds", mean_total)
        object.__setattr__(self, "mean_components_seconds", components)
        object.__setattr__(self, "target_environment_steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "query_count": self.query_count,
            "total_seconds": self.total_seconds,
            "mean_total_seconds": self.mean_total_seconds,
            "mean_components_seconds": dict(self.mean_components_seconds),
            "target_environment_steps": self.target_environment_steps,
        }


@dataclass(frozen=True)
class ColdWarmCostReconciliation:
    cost_contract_digest: str
    query_ids: tuple[str, ...]
    cold: ModeCostSummary
    warm: ModeCostSummary
    mean_cold_minus_warm_seconds: float
    cold_to_warm_ratio: float | None

    def __post_init__(self) -> None:
        contract = _digest(self.cost_contract_digest, "cost_contract_digest")
        query_ids = tuple(_nonempty(item, "query_ids[]") for item in self.query_ids)
        if not query_ids or query_ids != tuple(sorted(set(query_ids))):
            raise CostContractError("query_ids must be non-empty, sorted, and unique")
        if not isinstance(self.cold, ModeCostSummary) or self.cold.mode != "cold":
            raise CostContractError("cold summary is invalid")
        if not isinstance(self.warm, ModeCostSummary) or self.warm.mode != "warm":
            raise CostContractError("warm summary is invalid")
        if self.cold.query_count != len(query_ids) or self.warm.query_count != len(query_ids):
            raise CostContractError("mode summaries do not cover all reconciled queries")
        delta = _finite_float(
            self.mean_cold_minus_warm_seconds, "mean_cold_minus_warm_seconds"
        )
        expected_delta = self.cold.mean_total_seconds - self.warm.mean_total_seconds
        if not math.isclose(delta, expected_delta, rel_tol=1e-12, abs_tol=1e-12):
            raise CostContractError("cold/warm delta does not reconcile")
        ratio = self.cold_to_warm_ratio
        if self.warm.mean_total_seconds == 0.0:
            if ratio is not None:
                raise CostContractError("cold/warm ratio must be null when warm time is zero")
        else:
            expected_ratio = self.cold.mean_total_seconds / self.warm.mean_total_seconds
            parsed_ratio = _nonnegative_float(ratio, "cold_to_warm_ratio")
            if not math.isclose(parsed_ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
                raise CostContractError("cold/warm ratio does not reconcile")
            object.__setattr__(self, "cold_to_warm_ratio", parsed_ratio)
        object.__setattr__(self, "cost_contract_digest", contract)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "mean_cold_minus_warm_seconds", delta)

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-cold-warm-cost-reconciliation.v0",
            "cost_contract_digest": self.cost_contract_digest,
            "query_ids": list(self.query_ids),
            "cold": self.cold.to_dict(),
            "warm": self.warm.to_dict(),
            "mean_cold_minus_warm_seconds": self.mean_cold_minus_warm_seconds,
            "cold_to_warm_ratio": self.cold_to_warm_ratio,
        }


def _mode_summary(records: Sequence[CostRecord], mode: ThermalMode) -> ModeCostSummary:
    selected = tuple(record for record in records if record.mode == mode)
    if not selected:
        raise CostContractError(f"no {mode} cost records were supplied")
    component_means = {
        component: math.fsum(record.components_seconds[component] for record in selected)
        / len(selected)
        for component in QUERY_COST_COMPONENTS
    }
    total = math.fsum(record.total_seconds for record in selected)
    return ModeCostSummary(
        mode=mode,
        query_count=len(selected),
        total_seconds=total,
        mean_total_seconds=total / len(selected),
        mean_components_seconds=component_means,
        target_environment_steps=sum(record.target_environment_steps for record in selected),
    )


def reconcile_cold_warm_costs(
    records: Sequence[CostRecord],
    *,
    expected_query_ids: Sequence[str] | None = None,
) -> ColdWarmCostReconciliation:
    """Require exactly one cold and one warm record for every query."""

    rows = tuple(records)
    if not rows or any(not isinstance(row, CostRecord) for row in rows):
        raise CostContractError("records must be a non-empty CostRecord sequence")
    contracts = {row.cost_contract_digest for row in rows}
    if len(contracts) != 1:
        raise CostContractError("cold/warm records use different cost contracts")
    keyed: dict[tuple[str, str], CostRecord] = {}
    for row in rows:
        key = (row.query_id, row.mode)
        if key in keyed:
            raise CostContractError("duplicate query/mode cost record")
        keyed[key] = row
    cold_ids = {query_id for query_id, mode in keyed if mode == "cold"}
    warm_ids = {query_id for query_id, mode in keyed if mode == "warm"}
    if expected_query_ids is None:
        expected = cold_ids | warm_ids
    else:
        parsed = tuple(_nonempty(item, "expected_query_ids[]") for item in expected_query_ids)
        if not parsed or len(parsed) != len(set(parsed)):
            raise CostContractError("expected_query_ids must be non-empty and unique")
        expected = set(parsed)
    if cold_ids != expected or warm_ids != expected:
        raise CostContractError("every expected query must have exactly one cold and warm record")
    for query_id in expected:
        cold = keyed[(query_id, "cold")]
        warm = keyed[(query_id, "warm")]
        if cold.target_environment_steps != warm.target_environment_steps:
            raise CostContractError(
                "cold and warm records must execute the same target environment steps"
            )
    cold_summary = _mode_summary(rows, "cold")
    warm_summary = _mode_summary(rows, "warm")
    delta = cold_summary.mean_total_seconds - warm_summary.mean_total_seconds
    ratio = (
        None
        if warm_summary.mean_total_seconds == 0.0
        else cold_summary.mean_total_seconds / warm_summary.mean_total_seconds
    )
    return ColdWarmCostReconciliation(
        cost_contract_digest=next(iter(contracts)),
        query_ids=tuple(sorted(expected)),
        cold=cold_summary,
        warm=warm_summary,
        mean_cold_minus_warm_seconds=delta,
        cold_to_warm_ratio=ratio,
    )


__all__ = [
    "COST_RECORD_SCHEMA",
    "QUERY_COST_COMPONENTS",
    "ColdWarmCostReconciliation",
    "CostContractError",
    "CostRecord",
    "ModeCostSummary",
    "ThermalMode",
    "reconcile_cold_warm_costs",
]
