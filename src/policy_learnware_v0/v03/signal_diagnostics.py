"""Pre-registered geometry and confusion diagnostics for the v0.3 atlas.

The retrieval record intentionally withholds private taxonomy in its public
projection.  This module computes the diagnostics that must exist *while the
represented arrays are still available*: representation effective rank,
collapse, and axis-scoped confusion.  Public projections contain aggregates
only; bank IDs and taxonomy labels remain in the private cell artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .signal_metrics import SignalDistanceRow, SignalMetricRecord
from .signal_runtime import RepresentedBank


BANK_GEOMETRY_DIAGNOSTIC_SCHEMA = (
    "policy-learnware.v03-bank-geometry-diagnostic.v0"
)
AXIS_CONFUSION_RECORD_SCHEMA = "policy-learnware.v03-axis-confusion-record.v0"
SIGNAL_CELL_DIAGNOSTICS_SCHEMA = "policy-learnware.v03-signal-cell-diagnostics.v0"
PUBLIC_SIGNAL_CELL_DIAGNOSTICS_SCHEMA = (
    "policy-learnware.v03-public-signal-cell-diagnostics.v0"
)
GEOMETRY_VARIANCE_EPSILON = 1.0e-12


class SignalDiagnosticError(ValueError):
    """A diagnostic is inconsistent with represented arrays or rankings."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SignalDiagnosticError(f"{where} must be a canonical non-empty string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result.lower() != result:
        raise SignalDiagnosticError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise SignalDiagnosticError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SignalDiagnosticError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SignalDiagnosticError(f"{where} must be finite")
    return result


@dataclass(frozen=True)
class BankGeometryDiagnostic:
    bank_id: str
    represented_bank_digest: str
    data_role: str
    sample_count: int
    output_dim: int
    numerical_rank: int
    effective_rank: float
    total_centered_variance: float
    zero_variance_fraction: float
    collapsed: bool
    schema: str = BANK_GEOMETRY_DIAGNOSTIC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BANK_GEOMETRY_DIAGNOSTIC_SCHEMA:
            raise SignalDiagnosticError("unsupported bank geometry schema")
        object.__setattr__(self, "bank_id", _nonempty(self.bank_id, "bank_id"))
        object.__setattr__(
            self,
            "represented_bank_digest",
            _digest(self.represented_bank_digest, "represented_bank_digest"),
        )
        object.__setattr__(self, "data_role", _nonempty(self.data_role, "data_role"))
        for name in ("sample_count", "output_dim", "numerical_rank"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "numerical_rank" else 1):
                raise SignalDiagnosticError(f"{name} is invalid")
        if self.numerical_rank > min(self.sample_count, self.output_dim):
            raise SignalDiagnosticError("numerical rank exceeds matrix dimensions")
        for name in (
            "effective_rank",
            "total_centered_variance",
            "zero_variance_fraction",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 0.0 <= self.zero_variance_fraction <= 1.0:
            raise SignalDiagnosticError("zero_variance_fraction must lie in [0, 1]")
        if not 0.0 <= self.effective_rank <= float(self.output_dim) + 1.0e-10:
            raise SignalDiagnosticError("effective_rank is outside output dimension")
        if self.total_centered_variance < 0.0 or type(self.collapsed) is not bool:
            raise SignalDiagnosticError("geometry collapse fields are invalid")
        if self.collapsed != (
            self.total_centered_variance <= GEOMETRY_VARIANCE_EPSILON
        ):
            raise SignalDiagnosticError("collapsed flag disagrees with frozen epsilon")

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BankGeometryDiagnostic":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise SignalDiagnosticError("bank geometry fields differ")
        return cls(**{field: value[field] for field in cls.__dataclass_fields__})


def bank_geometry_diagnostic(bank: RepresentedBank) -> BankGeometryDiagnostic:
    if not isinstance(bank, RepresentedBank):
        raise SignalDiagnosticError("geometry requires a typed represented bank")
    values = np.asarray(bank.values, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    total = float(np.sum(energy))
    if total <= GEOMETRY_VARIANCE_EPSILON:
        effective_rank = 0.0
        numerical_rank = 0
    else:
        probabilities = energy[energy > 0.0] / total
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )
        tolerance = (
            max(centered.shape)
            * float(singular[0])
            * np.finfo(np.float64).eps
        )
        numerical_rank = int(np.sum(singular > tolerance))
    column_variances = np.mean(np.square(centered), axis=0)
    return BankGeometryDiagnostic(
        bank_id=bank.feature_bank.receipt.bank_id,
        represented_bank_digest=str(bank.represented_bank_digest),
        data_role=bank.feature_bank.receipt.data_role,
        sample_count=int(values.shape[0]),
        output_dim=int(values.shape[1]),
        numerical_rank=numerical_rank,
        effective_rank=effective_rank,
        total_centered_variance=total,
        zero_variance_fraction=float(
            np.mean(column_variances <= GEOMETRY_VARIANCE_EPSILON)
        ),
        collapsed=total <= GEOMETRY_VARIANCE_EPSILON,
    )


@dataclass(frozen=True)
class AxisConfusionRecord:
    axis_id: str
    true_identity: str
    predicted_identity: str
    query_count: int
    schema: str = AXIS_CONFUSION_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AXIS_CONFUSION_RECORD_SCHEMA:
            raise SignalDiagnosticError("unsupported axis confusion schema")
        if self.axis_id not in {"TASK_GLOBAL", "GOAL_CONDITIONAL", "DYNAMICS_CONDITIONAL"}:
            raise SignalDiagnosticError("unknown confusion axis")
        for name in ("true_identity", "predicted_identity"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.query_count) is not int or self.query_count <= 0:
            raise SignalDiagnosticError("query_count must be positive")

    @property
    def correct(self) -> bool:
        return self.true_identity == self.predicted_identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axis_id": self.axis_id,
            "true_identity": self.true_identity,
            "predicted_identity": self.predicted_identity,
            "query_count": self.query_count,
            "correct": self.correct,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AxisConfusionRecord":
        expected = set(cls.__dataclass_fields__) | {"correct"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SignalDiagnosticError("axis confusion fields differ")
        result = cls(**{field: value[field] for field in cls.__dataclass_fields__})
        if value["correct"] is not result.correct:
            raise SignalDiagnosticError("axis confusion correct projection differs")
        return result


def _confusions(
    metric: SignalMetricRecord,
) -> tuple[AxisConfusionRecord, ...]:
    groups: dict[str, list[SignalDistanceRow]] = {}
    for row in metric.rows:
        groups.setdefault(row.query_bank_id, []).append(row)
    counts: dict[tuple[str, str, str], int] = {}
    for rows in groups.values():
        ranked = sorted(rows, key=lambda item: (item.distance, item.source_bank_id))
        query = ranked[0]
        key = ("TASK_GLOBAL", query.query_task_id, ranked[0].source_task_id)
        counts[key] = counts.get(key, 0) + 1

        goals = [
            row
            for row in ranked
            if row.source_embodiment_id == row.query_embodiment_id
            and row.source_abi_contract_id == row.query_abi_contract_id
        ]
        if (
            len({row.source_goal_contract_id for row in goals}) >= 2
            and any(
                row.source_goal_contract_id == row.query_goal_contract_id
                for row in goals
            )
        ):
            key = (
                "GOAL_CONDITIONAL",
                query.query_goal_contract_id,
                goals[0].source_goal_contract_id,
            )
            counts[key] = counts.get(key, 0) + 1

        dynamics = [
            row
            for row in ranked
            if row.source_task_id == row.query_task_id
            and row.source_goal_contract_id == row.query_goal_contract_id
            and row.source_embodiment_id == row.query_embodiment_id
            and row.source_abi_contract_id == row.query_abi_contract_id
        ]
        if (
            len({row.source_dynamics_context_id for row in dynamics}) >= 2
            and any(
                row.source_dynamics_context_id == row.query_dynamics_context_id
                for row in dynamics
            )
        ):
            key = (
                "DYNAMICS_CONDITIONAL",
                query.query_dynamics_context_id,
                dynamics[0].source_dynamics_context_id,
            )
            counts[key] = counts.get(key, 0) + 1
    return tuple(
        AxisConfusionRecord(axis, truth, predicted, count)
        for (axis, truth, predicted), count in sorted(counts.items())
    )


def axis_confusion_records(
    metric_record: SignalMetricRecord,
) -> tuple[AxisConfusionRecord, ...]:
    """Return the deterministic private confusion rows for one metric record."""

    if not isinstance(metric_record, SignalMetricRecord):
        raise SignalDiagnosticError("confusion rows require SignalMetricRecord")
    return _confusions(metric_record)


def _axis_summary(
    confusion: Sequence[AxisConfusionRecord], axis_id: str
) -> dict[str, float]:
    rows = tuple(item for item in confusion if item.axis_id == axis_id)
    count = sum(item.query_count for item in rows)
    correct = sum(item.query_count for item in rows if item.correct)
    return {
        "eligible_query_count": float(count),
        "correct_query_count": float(correct),
        **({"accuracy": float(correct / count)} if count else {}),
    }


@dataclass(frozen=True)
class SignalCellDiagnostics:
    metric_record_digest: str
    representation_coordinate_digest: str
    bank_geometries: tuple[BankGeometryDiagnostic, ...]
    confusion_records: tuple[AxisConfusionRecord, ...]
    diagnostics_digest: str | None = None
    schema: str = SIGNAL_CELL_DIAGNOSTICS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CELL_DIAGNOSTICS_SCHEMA:
            raise SignalDiagnosticError("unsupported signal diagnostics schema")
        for name in ("metric_record_digest", "representation_coordinate_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        geometries = tuple(self.bank_geometries)
        if not geometries or not all(
            isinstance(item, BankGeometryDiagnostic) for item in geometries
        ):
            raise SignalDiagnosticError("diagnostics require typed bank geometries")
        bank_ids = tuple(item.bank_id for item in geometries)
        if len(set(bank_ids)) != len(bank_ids):
            raise SignalDiagnosticError("diagnostics contain duplicate bank IDs")
        confusion = tuple(self.confusion_records)
        if not confusion or not all(
            isinstance(item, AxisConfusionRecord) for item in confusion
        ):
            raise SignalDiagnosticError("diagnostics require task confusion")
        object.__setattr__(
            self, "bank_geometries", tuple(sorted(geometries, key=lambda item: item.bank_id))
        )
        object.__setattr__(
            self,
            "confusion_records",
            tuple(
                sorted(
                    confusion,
                    key=lambda item: (
                        item.axis_id,
                        item.true_identity,
                        item.predicted_identity,
                    ),
                )
            ),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.diagnostics_digest is None:
            object.__setattr__(self, "diagnostics_digest", expected)
        elif _digest(self.diagnostics_digest, "diagnostics_digest") != expected:
            raise SignalDiagnosticError("signal diagnostics digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "metric_record_digest": self.metric_record_digest,
            "representation_coordinate_digest": self.representation_coordinate_digest,
            "geometry_variance_epsilon": GEOMETRY_VARIANCE_EPSILON,
            "bank_geometries": [item.to_dict() for item in self.bank_geometries],
            "confusion_records": [item.to_dict() for item in self.confusion_records],
        }

    def to_private_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "diagnostics_digest": self.diagnostics_digest}

    @classmethod
    def from_private_dict(cls, value: Mapping[str, Any]) -> "SignalCellDiagnostics":
        expected = {
            "schema",
            "metric_record_digest",
            "representation_coordinate_digest",
            "geometry_variance_epsilon",
            "bank_geometries",
            "confusion_records",
            "diagnostics_digest",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise SignalDiagnosticError("private signal diagnostics fields differ")
        if value["geometry_variance_epsilon"] != GEOMETRY_VARIANCE_EPSILON:
            raise SignalDiagnosticError("geometry epsilon differs from protocol")
        if not isinstance(value["bank_geometries"], list) or not isinstance(
            value["confusion_records"], list
        ):
            raise SignalDiagnosticError("diagnostic rows must be lists")
        return cls(
            schema=value["schema"],
            metric_record_digest=value["metric_record_digest"],
            representation_coordinate_digest=value[
                "representation_coordinate_digest"
            ],
            bank_geometries=tuple(
                BankGeometryDiagnostic.from_dict(item)
                for item in value["bank_geometries"]
            ),
            confusion_records=tuple(
                AxisConfusionRecord.from_dict(item)
                for item in value["confusion_records"]
            ),
            diagnostics_digest=value["diagnostics_digest"],
        )

    def to_public_dict(self) -> dict[str, Any]:
        ranks = np.asarray(
            [item.effective_rank for item in self.bank_geometries], dtype=np.float64
        )
        payload = {
            "schema": PUBLIC_SIGNAL_CELL_DIAGNOSTICS_SCHEMA,
            "metric_record_digest": self.metric_record_digest,
            "representation_coordinate_digest": self.representation_coordinate_digest,
            "bank_count": len(self.bank_geometries),
            "effective_rank_mean": float(np.mean(ranks)),
            "effective_rank_min": float(np.min(ranks)),
            "effective_rank_max": float(np.max(ranks)),
            "collapsed_bank_count": sum(item.collapsed for item in self.bank_geometries),
            "axis_summaries": {
                axis: _axis_summary(self.confusion_records, axis)
                for axis in (
                    "TASK_GLOBAL",
                    "GOAL_CONDITIONAL",
                    "DYNAMICS_CONDITIONAL",
                )
            },
            "private_bank_and_taxonomy_rows_withheld": True,
            "private_diagnostics_digest": self.diagnostics_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}

    def validate_metric_record(self, metric_record: SignalMetricRecord) -> None:
        """Join private diagnostic rows back to the exact distance record."""

        if not isinstance(metric_record, SignalMetricRecord):
            raise SignalDiagnosticError("diagnostic join requires SignalMetricRecord")
        if (
            self.metric_record_digest != metric_record.record_digest
            or self.representation_coordinate_digest
            != metric_record.representation_coordinate_digest
        ):
            raise SignalDiagnosticError("diagnostics belong to another metric record")
        expected_banks = {
            row.query_bank_id for row in metric_record.rows
        } | {row.source_bank_id for row in metric_record.rows}
        if {item.bank_id for item in self.bank_geometries} != expected_banks:
            raise SignalDiagnosticError(
                "diagnostic bank membership differs from metric rows"
            )
        if tuple(item.to_dict() for item in self.confusion_records) != tuple(
            item.to_dict() for item in _confusions(metric_record)
        ):
            raise SignalDiagnosticError(
                "diagnostic confusion differs from distance rankings"
            )


def build_signal_cell_diagnostics(
    *,
    source_banks: Sequence[RepresentedBank],
    query_banks: Sequence[RepresentedBank],
    metric_record: SignalMetricRecord,
) -> SignalCellDiagnostics:
    if not isinstance(metric_record, SignalMetricRecord):
        raise SignalDiagnosticError("diagnostics require a typed metric record")
    banks = (*tuple(source_banks), *tuple(query_banks))
    if not banks or not all(isinstance(item, RepresentedBank) for item in banks):
        raise SignalDiagnosticError("diagnostics require typed represented banks")
    observed_source_ids = {
        item.feature_bank.receipt.bank_id for item in source_banks
    }
    observed_query_ids = {item.feature_bank.receipt.bank_id for item in query_banks}
    if (
        {row.source_bank_id for row in metric_record.rows} != observed_source_ids
        or {row.query_bank_id for row in metric_record.rows} != observed_query_ids
    ):
        raise SignalDiagnosticError(
            "diagnostic banks differ from the metric-record membership"
        )
    confusion = _confusions(metric_record)
    metric_values: Mapping[str, float] = metric_record.metric_values or MappingProxyType({})
    checks = {
        "TASK_GLOBAL": "task_top1",
        "GOAL_CONDITIONAL": "goal_top1",
        "DYNAMICS_CONDITIONAL": "dynamics_top1",
    }
    for axis, metric_id in checks.items():
        summary = _axis_summary(confusion, axis)
        if "accuracy" in summary:
            if metric_id not in metric_values or not math.isclose(
                summary["accuracy"], metric_values[metric_id], rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise SignalDiagnosticError(
                    f"{axis} confusion disagrees with {metric_id}"
                )
        elif metric_id in metric_values:
            raise SignalDiagnosticError(
                f"{metric_id} exists without eligible confusion rows"
            )
    result = SignalCellDiagnostics(
        metric_record_digest=metric_record.record_digest,
        representation_coordinate_digest=metric_record.representation_coordinate_digest,
        bank_geometries=tuple(bank_geometry_diagnostic(item) for item in banks),
        confusion_records=confusion,
    )
    result.validate_metric_record(metric_record)
    return result


__all__ = [
    "AXIS_CONFUSION_RECORD_SCHEMA",
    "BANK_GEOMETRY_DIAGNOSTIC_SCHEMA",
    "GEOMETRY_VARIANCE_EPSILON",
    "PUBLIC_SIGNAL_CELL_DIAGNOSTICS_SCHEMA",
    "SIGNAL_CELL_DIAGNOSTICS_SCHEMA",
    "AxisConfusionRecord",
    "BankGeometryDiagnostic",
    "SignalCellDiagnostics",
    "SignalDiagnosticError",
    "bank_geometry_diagnostic",
    "axis_confusion_records",
    "build_signal_cell_diagnostics",
]
