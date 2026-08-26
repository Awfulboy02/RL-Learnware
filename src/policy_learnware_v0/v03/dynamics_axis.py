"""Frozen numeric dynamics axes and representation-local order diagnostics.

The ordinary signal metrics deliberately treat a dynamics context as a
categorical label.  This module is the separate, preregistered readout for a
continuous dynamics factor.  It never compares across task, goal,
embodiment, or ABI: those four fields define the scope of one numeric axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from ..hashing import sha256_json
from .signal_metrics import SignalDistanceRow, SignalMetricRecord


DYNAMICS_AXIS_ENTRY_SCHEMA = "policy-learnware.v03-dynamics-axis-entry.v0"
DYNAMICS_AXIS_REGISTRY_SCHEMA = "policy-learnware.v03-dynamics-axis-registry.v0"
DYNAMICS_QUERY_DIAGNOSTIC_SCHEMA = (
    "policy-learnware.v03-dynamics-query-diagnostic.v0"
)
DYNAMICS_AXIS_DIAGNOSTICS_SCHEMA = (
    "policy-learnware.v03-dynamics-axis-diagnostics.v0"
)
DYNAMICS_PUBLIC_QUERY_JOIN_SCHEMA = (
    "policy-learnware.v03-dynamics-public-query-join.v0"
)

DynamicsAxisRole = Literal["ANCHOR", "INTERPOLATION", "EXTRAPOLATION"]
DYNAMICS_AXIS_ROLES = frozenset({"ANCHOR", "INTERPOLATION", "EXTRAPOLATION"})


class DynamicsAxisError(ValueError):
    """A frozen dynamics axis or its diagnostic join is invalid."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DynamicsAxisError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result != result.lower():
        raise DynamicsAxisError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise DynamicsAxisError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise DynamicsAxisError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DynamicsAxisError(f"{where} must be finite")
    return result


@dataclass(frozen=True)
class DynamicsAxisEntry:
    """One globally named dynamics context on one frozen numeric axis."""

    dynamics_context_id: str
    axis_id: str
    task_id: str
    embodiment_id: str
    abi_contract_id: str
    goal_contract_id: str
    factor_value: float
    role: DynamicsAxisRole
    schema: str = DYNAMICS_AXIS_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DYNAMICS_AXIS_ENTRY_SCHEMA:
            raise DynamicsAxisError("unsupported DynamicsAxisEntry schema")
        for name in (
            "dynamics_context_id",
            "axis_id",
            "task_id",
            "embodiment_id",
            "abi_contract_id",
            "goal_contract_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self, "factor_value", _finite(self.factor_value, "factor_value")
        )
        if self.role not in DYNAMICS_AXIS_ROLES:
            raise DynamicsAxisError("unknown dynamics-axis role")

    @property
    def scope_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.axis_id,
            self.task_id,
            self.embodiment_id,
            self.abi_contract_id,
            self.goal_contract_id,
        )

    @property
    def entry_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class DynamicsAxisRegistry:
    """Digest-frozen context-to-factor registry used before oracle unlock."""

    entries: tuple[DynamicsAxisEntry, ...]
    registry_digest: str | None = None
    schema: str = DYNAMICS_AXIS_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DYNAMICS_AXIS_REGISTRY_SCHEMA:
            raise DynamicsAxisError("unsupported DynamicsAxisRegistry schema")
        entries = tuple(self.entries)
        if not entries or not all(isinstance(item, DynamicsAxisEntry) for item in entries):
            raise DynamicsAxisError("dynamics registry requires typed entries")
        context_ids = [item.dynamics_context_id for item in entries]
        if len(set(context_ids)) != len(context_ids):
            raise DynamicsAxisError(
                "dynamics_context_id must map to exactly one frozen factor"
            )
        by_scope: dict[tuple[str, str, str, str, str], list[DynamicsAxisEntry]] = {}
        for entry in entries:
            by_scope.setdefault(entry.scope_key, []).append(entry)
        for scope, group in by_scope.items():
            factors = [item.factor_value for item in group]
            if len(set(factors)) != len(factors):
                raise DynamicsAxisError(
                    f"numeric factors must be unique within dynamics axis {scope!r}"
                )
            anchors = sorted(
                item.factor_value for item in group if item.role == "ANCHOR"
            )
            if len(anchors) < 2:
                raise DynamicsAxisError(
                    f"dynamics axis {scope!r} requires at least two anchors"
                )
            lower, upper = anchors[0], anchors[-1]
            for item in group:
                if item.role == "INTERPOLATION" and not lower < item.factor_value < upper:
                    raise DynamicsAxisError(
                        "INTERPOLATION factor must lie strictly inside anchor range"
                    )
                if item.role == "EXTRAPOLATION" and not (
                    item.factor_value < lower or item.factor_value > upper
                ):
                    raise DynamicsAxisError(
                        "EXTRAPOLATION factor must lie outside anchor range"
                    )
        entries = tuple(
            sorted(entries, key=lambda item: (item.scope_key, item.factor_value))
        )
        object.__setattr__(self, "entries", entries)
        expected = sha256_json(self._payload_without_digest())
        if self.registry_digest is None:
            object.__setattr__(self, "registry_digest", expected)
        elif _digest(self.registry_digest, "registry_digest") != expected:
            raise DynamicsAxisError("dynamics registry digest does not match contents")

    def entry(self, dynamics_context_id: str) -> DynamicsAxisEntry:
        context_id = _nonempty(dynamics_context_id, "dynamics_context_id")
        matches = tuple(
            item for item in self.entries if item.dynamics_context_id == context_id
        )
        if len(matches) != 1:
            raise DynamicsAxisError(
                f"dynamics context {context_id!r} is absent from the frozen registry"
            )
        return matches[0]

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "entries": [item.to_dict() for item in self.entries],
            "scope_rule": "same-task+goal+embodiment+abi",
            "source_candidate_role": "ANCHOR",
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "registry_digest": self.registry_digest}


def dynamics_query_alias_manifest_digest(
    dynamics_context_by_opaque_query_id: Mapping[str, str],
    registry: DynamicsAxisRegistry,
) -> str:
    """Canonical private alias manifest committed by ``PublicQueryPlan``."""

    if not isinstance(registry, DynamicsAxisRegistry):
        raise DynamicsAxisError("query alias manifest requires dynamics registry")
    mapping = {
        _nonempty(query_id, "opaque query ID"): _nonempty(
            context_id, "dynamics context ID"
        )
        for query_id, context_id in sorted(
            dynamics_context_by_opaque_query_id.items()
        )
    }
    if not mapping:
        raise DynamicsAxisError("dynamics query alias mapping cannot be empty")
    for context_id in mapping.values():
        registry.entry(context_id)
    return sha256_json(
        {
            "schema": "policy-learnware.v03-dynamics-query-alias-manifest.v0",
            "dynamics_axis_registry_digest": registry.registry_digest,
            "dynamics_context_by_opaque_query_id": mapping,
        }
    )


@dataclass(frozen=True)
class DynamicsPublicQueryJoin:
    """Private join proving the public 30/24/12 regimes match axis roles."""

    public_query_plan_digest: str
    query_alias_manifest_digest: str
    dynamics_axis_registry_digest: str
    dynamics_context_by_opaque_query_id: Mapping[str, str]
    join_digest: str | None = None
    schema: str = DYNAMICS_PUBLIC_QUERY_JOIN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DYNAMICS_PUBLIC_QUERY_JOIN_SCHEMA:
            raise DynamicsAxisError("unsupported DynamicsPublicQueryJoin schema")
        for name in (
            "public_query_plan_digest",
            "query_alias_manifest_digest",
            "dynamics_axis_registry_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        mapping = {
            _nonempty(query_id, "opaque query ID"): _nonempty(
                context_id, "dynamics context ID"
            )
            for query_id, context_id in sorted(
                self.dynamics_context_by_opaque_query_id.items()
            )
        }
        if not mapping:
            raise DynamicsAxisError("dynamics public-query join cannot be empty")
        object.__setattr__(
            self,
            "dynamics_context_by_opaque_query_id",
            MappingProxyType(mapping),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.join_digest is None:
            object.__setattr__(self, "join_digest", expected)
        elif _digest(self.join_digest, "join_digest") != expected:
            raise DynamicsAxisError("dynamics public-query join digest mismatch")

    @classmethod
    def bind(
        cls,
        *,
        public_query_plan: Any,
        registry: DynamicsAxisRegistry,
        dynamics_context_by_opaque_query_id: Mapping[str, str],
    ) -> "DynamicsPublicQueryJoin":
        from .preflight import PublicQueryPlan

        if not isinstance(public_query_plan, PublicQueryPlan):
            raise DynamicsAxisError("query join requires PublicQueryPlan")
        if not isinstance(registry, DynamicsAxisRegistry):
            raise DynamicsAxisError("query join requires DynamicsAxisRegistry")
        mapping = dict(dynamics_context_by_opaque_query_id)
        if set(mapping) != set(public_query_plan.opaque_query_ids):
            raise DynamicsAxisError(
                "dynamics query aliases differ from the frozen public query plan"
            )
        expected_role = {
            "EXACT": "ANCHOR",
            "INTERPOLATION": "INTERPOLATION",
            "EXTRAPOLATION": "EXTRAPOLATION",
        }
        for query_id, regime in public_query_plan.regime_by_opaque_query_id.items():
            entry = registry.entry(mapping[query_id])
            if entry.role != expected_role[regime]:
                raise DynamicsAxisError(
                    f"public query regime {regime} disagrees with dynamics-axis role"
                )
        alias_digest = dynamics_query_alias_manifest_digest(mapping, registry)
        if alias_digest != public_query_plan.query_alias_manifest_digest:
            raise DynamicsAxisError(
                "dynamics query aliases differ from query-plan alias manifest"
            )
        return cls(
            public_query_plan_digest=str(public_query_plan.plan_digest),
            query_alias_manifest_digest=alias_digest,
            dynamics_axis_registry_digest=str(registry.registry_digest),
            dynamics_context_by_opaque_query_id=mapping,
        )

    def validate(self, *, public_query_plan: Any, registry: DynamicsAxisRegistry) -> None:
        rebuilt = type(self).bind(
            public_query_plan=public_query_plan,
            registry=registry,
            dynamics_context_by_opaque_query_id=(
                self.dynamics_context_by_opaque_query_id
            ),
        )
        if rebuilt.to_dict() != self.to_dict():
            raise DynamicsAxisError("dynamics public-query join drifted")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "dynamics_context_by_opaque_query_id": dict(
                self.dynamics_context_by_opaque_query_id
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "join_digest": self.join_digest}

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "policy-learnware.v03-public-dynamics-query-join.v0",
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "query_count": len(self.dynamics_context_by_opaque_query_id),
            "private_query_to_context_aliases_withheld": True,
            "private_join_digest": self.join_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


@dataclass(frozen=True)
class DynamicsQueryDiagnostic:
    query_bank_id: str
    query_dynamics_context_id: str
    role: DynamicsAxisRole
    factor_value: float
    candidate_source_count: int
    nearest_source_bank_ids: tuple[str, ...]
    selected_source_bank_id: str
    selected_factor_value: float
    neighborhood_top1: float
    factor_absolute_error: float
    order_correct_pair_mass: float
    order_pair_count: int
    schema: str = DYNAMICS_QUERY_DIAGNOSTIC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DYNAMICS_QUERY_DIAGNOSTIC_SCHEMA:
            raise DynamicsAxisError("unsupported DynamicsQueryDiagnostic schema")
        for name in (
            "query_bank_id",
            "query_dynamics_context_id",
            "selected_source_bank_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.role not in DYNAMICS_AXIS_ROLES:
            raise DynamicsAxisError("unknown query dynamics-axis role")
        for name in (
            "factor_value",
            "selected_factor_value",
            "neighborhood_top1",
            "factor_absolute_error",
            "order_correct_pair_mass",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if (
            isinstance(self.candidate_source_count, bool)
            or not isinstance(self.candidate_source_count, int)
            or self.candidate_source_count < 2
        ):
            raise DynamicsAxisError("candidate_source_count must be at least two")
        if (
            isinstance(self.order_pair_count, bool)
            or not isinstance(self.order_pair_count, int)
            or self.order_pair_count < 0
        ):
            raise DynamicsAxisError("order_pair_count must be non-negative")
        nearest = tuple(sorted(_nonempty(item, "nearest source ID") for item in self.nearest_source_bank_ids))
        if not nearest or len(set(nearest)) != len(nearest):
            raise DynamicsAxisError("nearest source IDs must be unique and non-empty")
        if self.neighborhood_top1 not in {0.0, 1.0}:
            raise DynamicsAxisError("neighborhood_top1 must be binary")
        if self.factor_absolute_error < 0.0:
            raise DynamicsAxisError("factor_absolute_error must be non-negative")
        if not 0.0 <= self.order_correct_pair_mass <= float(self.order_pair_count):
            raise DynamicsAxisError("order pair mass is outside its denominator")
        object.__setattr__(self, "nearest_source_bank_ids", nearest)

    @property
    def order_accuracy(self) -> float | None:
        if self.order_pair_count == 0:
            return None
        return self.order_correct_pair_mass / self.order_pair_count

    @property
    def diagnostic_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **{field: getattr(self, field) for field in self.__dataclass_fields__},
            "order_accuracy": self.order_accuracy,
        }


@dataclass(frozen=True)
class DynamicsAxisDiagnostics:
    execution_mode: str
    formal_authorization_digest: str | None
    signal_plan_digest: str
    signal_execution_protocol_digest: str
    identity_registry_digest: str
    metric_record_digest: str
    registry_digest: str
    query_diagnostics: tuple[DynamicsQueryDiagnostic, ...]
    metric_values: Mapping[str, float]
    diagnostics_digest: str | None = None
    schema: str = DYNAMICS_AXIS_DIAGNOSTICS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DYNAMICS_AXIS_DIAGNOSTICS_SCHEMA:
            raise DynamicsAxisError("unsupported DynamicsAxisDiagnostics schema")
        if self.execution_mode not in {"DEVELOPMENT_SMOKE", "FORMAL"}:
            raise DynamicsAxisError("unknown dynamics diagnostics execution mode")
        if self.execution_mode == "FORMAL":
            if self.formal_authorization_digest is None:
                raise DynamicsAxisError(
                    "formal dynamics diagnostics require authorization digest"
                )
            object.__setattr__(
                self,
                "formal_authorization_digest",
                _digest(
                    self.formal_authorization_digest,
                    "formal_authorization_digest",
                ),
            )
        elif self.formal_authorization_digest is not None:
            raise DynamicsAxisError(
                "development dynamics diagnostics cannot carry formal authorization"
            )
        for name in (
            "signal_plan_digest",
            "signal_execution_protocol_digest",
            "identity_registry_digest",
            "metric_record_digest",
            "registry_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        rows = tuple(self.query_diagnostics)
        if not rows or not all(isinstance(item, DynamicsQueryDiagnostic) for item in rows):
            raise DynamicsAxisError("dynamics diagnostics require typed query rows")
        if len({item.query_bank_id for item in rows}) != len(rows):
            raise DynamicsAxisError("dynamics query diagnostics contain duplicate queries")
        rows = tuple(sorted(rows, key=lambda item: item.query_bank_id))
        metrics = {
            _nonempty(key, "dynamics metric ID"): _finite(value, f"metric_values[{key}]")
            for key, value in sorted(self.metric_values.items())
        }
        expected_metrics = _summarize(rows)
        if metrics != expected_metrics:
            raise DynamicsAxisError("dynamics metric values do not match query diagnostics")
        object.__setattr__(self, "query_diagnostics", rows)
        object.__setattr__(self, "metric_values", MappingProxyType(metrics))
        expected = sha256_json(self._payload_without_digest())
        if self.diagnostics_digest is None:
            object.__setattr__(self, "diagnostics_digest", expected)
        elif _digest(self.diagnostics_digest, "diagnostics_digest") != expected:
            raise DynamicsAxisError("dynamics diagnostics digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_mode": self.execution_mode,
            "formal_authorization_digest": self.formal_authorization_digest,
            "signal_plan_digest": self.signal_plan_digest,
            "signal_execution_protocol_digest": (
                self.signal_execution_protocol_digest
            ),
            "identity_registry_digest": self.identity_registry_digest,
            "metric_record_digest": self.metric_record_digest,
            "registry_digest": self.registry_digest,
            "query_diagnostic_digests": [
                item.diagnostic_digest for item in self.query_diagnostics
            ],
            "metric_values": dict(self.metric_values),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "query_diagnostics": [item.to_dict() for item in self.query_diagnostics],
            "diagnostics_digest": self.diagnostics_digest,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Aggregate-only projection; bank/task/context rows remain private."""

        payload = {
            "schema": "policy-learnware.v03-public-dynamics-axis-diagnostics.v0",
            "execution_mode": self.execution_mode,
            "formal_authorization_digest": self.formal_authorization_digest,
            "signal_plan_digest": self.signal_plan_digest,
            "signal_execution_protocol_digest": (
                self.signal_execution_protocol_digest
            ),
            "identity_registry_digest": self.identity_registry_digest,
            "metric_record_digest": self.metric_record_digest,
            "dynamics_axis_registry_digest": self.registry_digest,
            "metric_values": dict(self.metric_values),
            "private_query_and_axis_rows_withheld": True,
            "private_diagnostics_digest": self.diagnostics_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


def _validate_row_identity(
    row: SignalDistanceRow,
    entry: DynamicsAxisEntry,
    *,
    side: Literal["query", "source"],
) -> None:
    actual = (
        getattr(row, f"{side}_task_id"),
        getattr(row, f"{side}_embodiment_id"),
        getattr(row, f"{side}_abi_contract_id"),
        getattr(row, f"{side}_goal_contract_id"),
    )
    expected = (
        entry.task_id,
        entry.embodiment_id,
        entry.abi_contract_id,
        entry.goal_contract_id,
    )
    if actual != expected:
        raise DynamicsAxisError(
            f"{side} row identity differs from frozen dynamics-axis scope"
        )


def _query_diagnostic(
    query_rows: tuple[SignalDistanceRow, ...],
    registry: DynamicsAxisRegistry,
) -> DynamicsQueryDiagnostic:
    query_ids = {row.query_bank_id for row in query_rows}
    contexts = {row.query_dynamics_context_id for row in query_rows}
    if len(query_ids) != 1 or len(contexts) != 1:
        raise DynamicsAxisError("query group has inconsistent bank/dynamics identity")
    query_entry = registry.entry(next(iter(contexts)))
    for row in query_rows:
        _validate_row_identity(row, query_entry, side="query")

    candidates: list[tuple[SignalDistanceRow, DynamicsAxisEntry]] = []
    for row in query_rows:
        source_entry = registry.entry(row.source_dynamics_context_id)
        _validate_row_identity(row, source_entry, side="source")
        if source_entry.scope_key == query_entry.scope_key and source_entry.role == "ANCHOR":
            candidates.append((row, source_entry))
    if len(candidates) < 2:
        raise DynamicsAxisError(
            "each dynamics query requires at least two same-scope anchor sources"
        )
    candidates.sort(key=lambda item: (item[0].distance, item[0].source_bank_id))
    gaps = {
        row.source_bank_id: abs(source.factor_value - query_entry.factor_value)
        for row, source in candidates
    }
    minimum_gap = min(gaps.values())
    nearest = tuple(
        source_id
        for source_id, gap in sorted(gaps.items())
        if math.isclose(gap, minimum_gap, rel_tol=0.0, abs_tol=1.0e-12)
    )
    selected_row, selected_entry = candidates[0]
    pair_mass = 0.0
    pair_count = 0
    for left_index, (left_row, left_entry) in enumerate(candidates):
        left_gap = abs(left_entry.factor_value - query_entry.factor_value)
        for right_row, right_entry in candidates[left_index + 1 :]:
            right_gap = abs(right_entry.factor_value - query_entry.factor_value)
            if math.isclose(left_gap, right_gap, rel_tol=0.0, abs_tol=1.0e-12):
                continue
            pair_count += 1
            expected_sign = -1.0 if left_gap < right_gap else 1.0
            observed_delta = left_row.distance - right_row.distance
            if math.isclose(observed_delta, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
                pair_mass += 0.5
            elif (observed_delta < 0.0 and expected_sign < 0.0) or (
                observed_delta > 0.0 and expected_sign > 0.0
            ):
                pair_mass += 1.0
    return DynamicsQueryDiagnostic(
        query_bank_id=next(iter(query_ids)),
        query_dynamics_context_id=query_entry.dynamics_context_id,
        role=query_entry.role,
        factor_value=query_entry.factor_value,
        candidate_source_count=len(candidates),
        nearest_source_bank_ids=nearest,
        selected_source_bank_id=selected_row.source_bank_id,
        selected_factor_value=selected_entry.factor_value,
        neighborhood_top1=float(selected_row.source_bank_id in nearest),
        factor_absolute_error=abs(
            selected_entry.factor_value - query_entry.factor_value
        ),
        order_correct_pair_mass=pair_mass,
        order_pair_count=pair_count,
    )


def _summarize(rows: tuple[DynamicsQueryDiagnostic, ...]) -> dict[str, float]:
    result: dict[str, float] = {"query_count": float(len(rows))}
    groups: dict[str, tuple[DynamicsQueryDiagnostic, ...]] = {
        "all": rows,
        **{
            role.lower(): tuple(item for item in rows if item.role == role)
            for role in sorted(DYNAMICS_AXIS_ROLES)
        },
    }
    for label, group in groups.items():
        result[f"{label}.query_count"] = float(len(group))
        if not group:
            continue
        result[f"{label}.neighborhood_top1"] = float(
            np.mean([item.neighborhood_top1 for item in group])
        )
        result[f"{label}.factor_mae"] = float(
            np.mean([item.factor_absolute_error for item in group])
        )
        denominator = sum(item.order_pair_count for item in group)
        result[f"{label}.order_pair_count"] = float(denominator)
        if denominator > 0:
            result[f"{label}.order_accuracy"] = float(
                sum(item.order_correct_pair_mass for item in group) / denominator
            )
    return result


def build_dynamics_axis_diagnostics(
    *,
    metric_record: SignalMetricRecord,
    registry: DynamicsAxisRegistry,
    execution_mode: str,
    signal_plan_digest: str,
    signal_execution_protocol_digest: str,
    identity_registry_digest: str,
    formal_authorization: Any | None = None,
) -> DynamicsAxisDiagnostics:
    """Evaluate neighborhood, order and factor MAE inside frozen axis scopes."""

    if not isinstance(metric_record, SignalMetricRecord):
        raise DynamicsAxisError("dynamics diagnostics require SignalMetricRecord")
    if not isinstance(registry, DynamicsAxisRegistry):
        raise DynamicsAxisError("dynamics diagnostics require DynamicsAxisRegistry")
    if execution_mode not in {"DEVELOPMENT_SMOKE", "FORMAL"}:
        raise DynamicsAxisError("unknown dynamics diagnostics execution mode")
    plan_digest = _digest(signal_plan_digest, "signal_plan_digest")
    execution_digest = _digest(
        signal_execution_protocol_digest,
        "signal_execution_protocol_digest",
    )
    identity_digest = _digest(
        identity_registry_digest, "identity_registry_digest"
    )
    if execution_mode == "FORMAL":
        from .signal_atlas import FormalSignalAtlasAuthorization

        if not isinstance(formal_authorization, FormalSignalAtlasAuthorization):
            raise DynamicsAxisError(
                "formal dynamics diagnostics require signal-atlas authorization"
            )
        if (
            formal_authorization.plan_digest != plan_digest
            or formal_authorization.execution_protocol_digest != execution_digest
            or formal_authorization.identity_registry_digest != identity_digest
        ):
            raise DynamicsAxisError(
                "formal dynamics authorization belongs to another signal freeze"
            )
        try:
            formal_authorization.validate_dynamics_axis_registry(registry)
        except Exception as error:
            raise DynamicsAxisError(str(error)) from error
    elif formal_authorization is not None:
        raise DynamicsAxisError(
            "development dynamics diagnostics must remain outside formal authorization"
        )
    by_query: dict[str, list[SignalDistanceRow]] = {}
    for row in metric_record.rows:
        by_query.setdefault(row.query_bank_id, []).append(row)
    diagnostics = tuple(
        _query_diagnostic(tuple(group), registry)
        for _, group in sorted(by_query.items())
    )
    return DynamicsAxisDiagnostics(
        execution_mode=execution_mode,
        formal_authorization_digest=(
            None
            if formal_authorization is None
            else str(formal_authorization.authorization_digest)
        ),
        signal_plan_digest=plan_digest,
        signal_execution_protocol_digest=execution_digest,
        identity_registry_digest=identity_digest,
        metric_record_digest=metric_record.record_digest,
        registry_digest=str(registry.registry_digest),
        query_diagnostics=diagnostics,
        metric_values=_summarize(diagnostics),
    )


__all__ = [
    "DYNAMICS_AXIS_DIAGNOSTICS_SCHEMA",
    "DYNAMICS_AXIS_ENTRY_SCHEMA",
    "DYNAMICS_AXIS_REGISTRY_SCHEMA",
    "DYNAMICS_AXIS_ROLES",
    "DYNAMICS_PUBLIC_QUERY_JOIN_SCHEMA",
    "DYNAMICS_QUERY_DIAGNOSTIC_SCHEMA",
    "DynamicsAxisDiagnostics",
    "DynamicsAxisEntry",
    "DynamicsAxisError",
    "DynamicsAxisRegistry",
    "DynamicsAxisRole",
    "DynamicsPublicQueryJoin",
    "DynamicsQueryDiagnostic",
    "build_dynamics_axis_diagnostics",
    "dynamics_query_alias_manifest_digest",
]
