"""Canonical 39-cell v0.3 signal-matrix and applicability ledger.

The matrix contains 39 *logical* cells and exactly 37 numeric cells.  The two
one-step temporal-shuffle cells (R0 and R5) are structural ``N/A`` records:
they cannot create fit jobs, cannot carry a numeric placeholder, and are never
included in metric denominators.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .representation_ladder import (
    R0_PADDED_RAW,
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    R_HIST_RANDOM_TANH,
)
from .transition_views import (
    REGISTERED_VIEW_IDS,
    V_FULL_LEGACY,
    V_RANDOM_ENCODER,
    V_REWARD_FREE_TRANSITION,
    V_TEMPORAL_SHUFFLE,
)


SIGNAL_CELL_SCHEMA = "policy-learnware.v03-signal-cell.v0"
SIGNAL_MATRIX_PLAN_SCHEMA = "policy-learnware.v03-signal-matrix-plan.v0"
SIGNAL_FIT_JOB_SCHEMA = "policy-learnware.v03-signal-fit-job.v0"
SIGNAL_CELL_RECORD_SCHEMA = "policy-learnware.v03-signal-cell-record.v0"
SIGNAL_MATRIX_LEDGER_SCHEMA = "policy-learnware.v03-signal-matrix-ledger.v0"

C_RF_SHUFFLED_NEXT = "C_RF_SHUFFLED_NEXT"

CORE_INPUT_VIEW_IDS = tuple(
    view_id for view_id in REGISTERED_VIEW_IDS if view_id != V_RANDOM_ENCODER
)
MECHANISM_CONDITIONS = (
    V_FULL_LEGACY,
    V_REWARD_FREE_TRANSITION,
    C_RF_SHUFFLED_NEXT,
)
MECHANISM_REPRESENTATIONS = (
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
)
REPRESENTATION_SEEDS = (0, 1, 2)

CellBlock = Literal["CORE_PAIRED", "HISTORICAL_CONTROL", "MECHANISM_STAIRCASE"]
Applicability = Literal["NUMERIC", "STRUCTURAL_NA"]
CellStatus = Literal["COMPUTED", "STRUCTURAL_NA"]


class SignalMatrixError(ValueError):
    """The signal matrix, fit plan, or result ledger is not canonical."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise SignalMatrixError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SignalMatrixError(f"{where} must be a lowercase SHA-256 digest") from error
    return value


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SignalMatrixError(f"{where} must be a canonical non-empty string")
    return value


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise SignalMatrixError("fit seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise SignalMatrixError("fit seed must be a non-negative integer")
    return result


def _strict(value: Mapping[str, Any], fields: set[str], where: str) -> None:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SignalMatrixError(f"{where} must be a string-keyed mapping")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise SignalMatrixError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _cell_id(block: str, condition_id: str, representation_id: str) -> str:
    return f"{block}::{condition_id}::{representation_id}"


@dataclass(frozen=True)
class SignalCell:
    cell_id: str
    block: CellBlock
    condition_id: str
    representation_id: str
    applicability: Applicability
    applicability_reason: str | None
    optimization_fit_required: bool
    cell_digest: str | None = None
    schema: str = SIGNAL_CELL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CELL_SCHEMA:
            raise SignalMatrixError("unsupported signal cell schema")
        if self.block not in {
            "CORE_PAIRED",
            "HISTORICAL_CONTROL",
            "MECHANISM_STAIRCASE",
        }:
            raise SignalMatrixError("unknown signal cell block")
        _nonempty(self.condition_id, "condition_id")
        _nonempty(self.representation_id, "representation_id")
        expected_id = _cell_id(self.block, self.condition_id, self.representation_id)
        if self.cell_id != expected_id:
            raise SignalMatrixError("cell_id is not the canonical cell identity")
        if self.applicability not in {"NUMERIC", "STRUCTURAL_NA"}:
            raise SignalMatrixError("unknown cell applicability")
        if type(self.optimization_fit_required) is not bool:
            raise SignalMatrixError("optimization_fit_required must be boolean")
        if self.applicability == "STRUCTURAL_NA":
            if not self.applicability_reason:
                raise SignalMatrixError("structural N/A requires a reason")
            if self.optimization_fit_required:
                raise SignalMatrixError("structural N/A cannot require optimization")
        elif self.applicability_reason is not None:
            raise SignalMatrixError("numeric cell cannot carry an N/A reason")
        expected = sha256_json(self._payload_without_digest())
        if self.cell_digest is None:
            object.__setattr__(self, "cell_digest", expected)
        elif _digest(self.cell_digest, "cell_digest") != expected:
            raise SignalMatrixError("cell_digest does not match cell")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cell_id": self.cell_id,
            "block": self.block,
            "condition_id": self.condition_id,
            "representation_id": self.representation_id,
            "applicability": self.applicability,
            "applicability_reason": self.applicability_reason,
            "optimization_fit_required": self.optimization_fit_required,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "cell_digest": self.cell_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalCell":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "signal cell")
        return cls(**{name: value[name] for name in fields})


def _canonical_cells() -> tuple[SignalCell, ...]:
    cells: list[SignalCell] = []
    for view_id in CORE_INPUT_VIEW_IDS:
        for representation_id in (R0_PADDED_RAW, R5_VIEW_SPECIFIC_CORRO_REFIT):
            structural_na = view_id == V_TEMPORAL_SHUFFLE
            cells.append(
                SignalCell(
                    cell_id=_cell_id("CORE_PAIRED", view_id, representation_id),
                    block="CORE_PAIRED",
                    condition_id=view_id,
                    representation_id=representation_id,
                    applicability=("STRUCTURAL_NA" if structural_na else "NUMERIC"),
                    applicability_reason=(
                        "one-step empirical distributions are invariant to row order"
                        if structural_na
                        else None
                    ),
                    optimization_fit_required=(
                        representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
                        and not structural_na
                    ),
                )
            )
    cells.append(
        SignalCell(
            cell_id=_cell_id(
                "HISTORICAL_CONTROL", V_RANDOM_ENCODER, R_HIST_RANDOM_TANH
            ),
            block="HISTORICAL_CONTROL",
            condition_id=V_RANDOM_ENCODER,
            representation_id=R_HIST_RANDOM_TANH,
            applicability="NUMERIC",
            applicability_reason=None,
            optimization_fit_required=False,
        )
    )
    for condition_id in MECHANISM_CONDITIONS:
        for representation_id in MECHANISM_REPRESENTATIONS:
            cells.append(
                SignalCell(
                    cell_id=_cell_id(
                        "MECHANISM_STAIRCASE", condition_id, representation_id
                    ),
                    block="MECHANISM_STAIRCASE",
                    condition_id=condition_id,
                    representation_id=representation_id,
                    applicability="NUMERIC",
                    applicability_reason=None,
                    optimization_fit_required=(
                        representation_id == R5L_SUPERVISED_LINEAR
                    ),
                )
            )
    return tuple(cells)


@dataclass(frozen=True)
class SignalMatrixPlan:
    cells: tuple[SignalCell, ...]
    representation_seeds: tuple[int, ...] = REPRESENTATION_SEEDS
    plan_digest: str | None = None
    schema: str = SIGNAL_MATRIX_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_MATRIX_PLAN_SCHEMA:
            raise SignalMatrixError("unsupported signal matrix plan schema")
        cells = tuple(self.cells)
        if not all(isinstance(cell, SignalCell) for cell in cells):
            raise SignalMatrixError("signal matrix plan requires typed cells")
        canonical = _canonical_cells()
        if tuple(cell.to_dict() for cell in cells) != tuple(
            cell.to_dict() for cell in canonical
        ):
            raise SignalMatrixError("signal matrix cells differ from the frozen 39-cell plan")
        seeds = tuple(_seed(seed) for seed in self.representation_seeds)
        if seeds != REPRESENTATION_SEEDS:
            raise SignalMatrixError("representation seeds must be exactly (0, 1, 2)")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "representation_seeds", seeds)
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise SignalMatrixError("plan_digest does not match signal matrix")

    @property
    def logical_cell_count(self) -> int:
        return len(self.cells)

    @property
    def numeric_cell_count(self) -> int:
        return sum(cell.applicability == "NUMERIC" for cell in self.cells)

    @property
    def structural_na_count(self) -> int:
        return sum(cell.applicability == "STRUCTURAL_NA" for cell in self.cells)

    @property
    def numeric_cells(self) -> tuple[SignalCell, ...]:
        return tuple(cell for cell in self.cells if cell.applicability == "NUMERIC")

    def cell(self, cell_id: str) -> SignalCell:
        matches = tuple(cell for cell in self.cells if cell.cell_id == cell_id)
        if len(matches) != 1:
            raise SignalMatrixError(f"unknown signal cell: {cell_id!r}")
        return matches[0]

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cells": [cell.to_dict() for cell in self.cells],
            "representation_seeds": list(self.representation_seeds),
            "logical_cell_count": self.logical_cell_count,
            "numeric_cell_count": self.numeric_cell_count,
            "structural_na_count": self.structural_na_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalMatrixPlan":
        fields = {
            "schema",
            "cells",
            "representation_seeds",
            "logical_cell_count",
            "numeric_cell_count",
            "structural_na_count",
            "plan_digest",
        }
        _strict(value, fields, "signal matrix plan")
        plan = cls(
            cells=tuple(SignalCell.from_dict(cell) for cell in value["cells"]),
            representation_seeds=tuple(value["representation_seeds"]),
            plan_digest=value["plan_digest"],
            schema=value["schema"],
        )
        expected_counts = (
            plan.logical_cell_count,
            plan.numeric_cell_count,
            plan.structural_na_count,
        )
        observed_counts = (
            value["logical_cell_count"],
            value["numeric_cell_count"],
            value["structural_na_count"],
        )
        if observed_counts != expected_counts:
            raise SignalMatrixError("serialized signal matrix counts are inconsistent")
        return plan


def build_signal_matrix_plan() -> SignalMatrixPlan:
    plan = SignalMatrixPlan(cells=_canonical_cells())
    if (
        plan.logical_cell_count != 39
        or plan.numeric_cell_count != 37
        or plan.structural_na_count != 2
    ):  # pragma: no cover - protects future registry edits
        raise SignalMatrixError("frozen signal matrix cardinality drifted")
    return plan


@dataclass(frozen=True)
class SignalFitJob:
    job_id: str
    plan_digest: str
    cell_id: str
    cell_digest: str
    condition_id: str
    representation_id: str
    seed: int
    job_digest: str | None = None
    schema: str = SIGNAL_FIT_JOB_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_FIT_JOB_SCHEMA:
            raise SignalMatrixError("unsupported signal fit job schema")
        for name in ("plan_digest", "cell_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _nonempty(self.cell_id, "cell_id")
        _nonempty(self.condition_id, "condition_id")
        if self.representation_id not in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            raise SignalMatrixError("only R5/R5L create optimization fit jobs")
        object.__setattr__(self, "seed", _seed(self.seed))
        expected_job_id = f"FIT::{self.cell_id}::seed-{self.seed}"
        if self.job_id != expected_job_id:
            raise SignalMatrixError("job_id is not canonical")
        expected = sha256_json(self._payload_without_digest())
        if self.job_digest is None:
            object.__setattr__(self, "job_digest", expected)
        elif _digest(self.job_digest, "job_digest") != expected:
            raise SignalMatrixError("job_digest does not match fit job")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "job_id": self.job_id,
            "plan_digest": self.plan_digest,
            "cell_id": self.cell_id,
            "cell_digest": self.cell_digest,
            "condition_id": self.condition_id,
            "representation_id": self.representation_id,
            "seed": self.seed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "job_digest": self.job_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalFitJob":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "signal fit job")
        return cls(**{name: value[name] for name in fields})


def build_optimization_fit_jobs(plan: SignalMatrixPlan) -> tuple[SignalFitJob, ...]:
    if not isinstance(plan, SignalMatrixPlan):
        raise SignalMatrixError("fit plan must be typed")
    jobs: list[SignalFitJob] = []
    for cell in plan.cells:
        if not cell.optimization_fit_required:
            continue
        if cell.applicability != "NUMERIC":  # pragma: no cover - cell guards too
            raise SignalMatrixError("structural N/A leaked into optimization jobs")
        for seed in plan.representation_seeds:
            jobs.append(
                SignalFitJob(
                    job_id=f"FIT::{cell.cell_id}::seed-{seed}",
                    plan_digest=str(plan.plan_digest),
                    cell_id=cell.cell_id,
                    cell_digest=str(cell.cell_digest),
                    condition_id=cell.condition_id,
                    representation_id=cell.representation_id,
                    seed=seed,
                )
            )
    r5 = sum(job.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT for job in jobs)
    r5l = sum(job.representation_id == R5L_SUPERVISED_LINEAR for job in jobs)
    if r5 != 36 or r5l != 9 or len(jobs) != 45:
        raise SignalMatrixError("optimization fit job cardinality drifted")
    return tuple(jobs)


@dataclass(frozen=True)
class SignalCellRecord:
    plan_digest: str
    cell_id: str
    cell_digest: str
    status: CellStatus
    metrics: Mapping[str, float] | None
    numeric_artifact_digest: str | None
    record_digest: str | None = None
    schema: str = SIGNAL_CELL_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CELL_RECORD_SCHEMA:
            raise SignalMatrixError("unsupported signal cell record schema")
        for name in ("plan_digest", "cell_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _nonempty(self.cell_id, "cell_id")
        if self.status not in {"COMPUTED", "STRUCTURAL_NA"}:
            raise SignalMatrixError("unknown signal cell record status")
        if self.status == "STRUCTURAL_NA":
            if self.metrics is not None or self.numeric_artifact_digest is not None:
                raise SignalMatrixError(
                    "structural N/A cannot carry zero/NaN metrics or numeric artifacts"
                )
        else:
            if not isinstance(self.metrics, Mapping) or not self.metrics:
                raise SignalMatrixError("computed cell requires non-empty metrics")
            parsed: dict[str, float] = {}
            for name, value in self.metrics.items():
                _nonempty(name, "metric name")
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise SignalMatrixError("computed metrics must be numeric")
                number = float(value)
                if not math.isfinite(number):
                    raise SignalMatrixError("computed metrics must be finite")
                parsed[name] = number
            object.__setattr__(self, "metrics", MappingProxyType(dict(sorted(parsed.items()))))
            if self.numeric_artifact_digest is None:
                raise SignalMatrixError("computed cell requires a numeric artifact digest")
            object.__setattr__(
                self,
                "numeric_artifact_digest",
                _digest(self.numeric_artifact_digest, "numeric_artifact_digest"),
            )
        expected = sha256_json(self._payload_without_digest())
        if self.record_digest is None:
            object.__setattr__(self, "record_digest", expected)
        elif _digest(self.record_digest, "record_digest") != expected:
            raise SignalMatrixError("record_digest does not match signal cell record")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "cell_id": self.cell_id,
            "cell_digest": self.cell_digest,
            "status": self.status,
            "metrics": None if self.metrics is None else dict(self.metrics),
            "numeric_artifact_digest": self.numeric_artifact_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalCellRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "signal cell record")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class SignalMatrixLedger:
    plan: SignalMatrixPlan
    records: tuple[SignalCellRecord, ...]
    ledger_digest: str | None = None
    schema: str = SIGNAL_MATRIX_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_MATRIX_LEDGER_SCHEMA:
            raise SignalMatrixError("unsupported signal matrix ledger schema")
        if not isinstance(self.plan, SignalMatrixPlan):
            raise SignalMatrixError("ledger plan must be typed")
        records = tuple(self.records)
        if len(records) != len(self.plan.cells) or not all(
            isinstance(record, SignalCellRecord) for record in records
        ):
            raise SignalMatrixError("ledger requires exactly one typed record per cell")
        by_id = {record.cell_id: record for record in records}
        if len(by_id) != len(records) or set(by_id) != {
            cell.cell_id for cell in self.plan.cells
        }:
            raise SignalMatrixError("ledger record coverage differs from signal plan")
        ordered: list[SignalCellRecord] = []
        for cell in self.plan.cells:
            record = by_id[cell.cell_id]
            if (
                record.plan_digest != self.plan.plan_digest
                or record.cell_digest != cell.cell_digest
            ):
                raise SignalMatrixError("ledger record is bound to another plan/cell")
            expected_status = (
                "STRUCTURAL_NA"
                if cell.applicability == "STRUCTURAL_NA"
                else "COMPUTED"
            )
            if record.status != expected_status:
                raise SignalMatrixError("record status disagrees with applicability ledger")
            ordered.append(record)
        object.__setattr__(self, "records", tuple(ordered))
        expected = sha256_json(self._payload_without_digest())
        if self.ledger_digest is None:
            object.__setattr__(self, "ledger_digest", expected)
        elif _digest(self.ledger_digest, "ledger_digest") != expected:
            raise SignalMatrixError("ledger_digest does not match records")

    @property
    def numeric_records(self) -> tuple[SignalCellRecord, ...]:
        return tuple(record for record in self.records if record.status == "COMPUTED")

    def metric_denominator(self, metric_name: str) -> int:
        _nonempty(metric_name, "metric_name")
        return sum(
            record.metrics is not None and metric_name in record.metrics
            for record in self.numeric_records
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan.plan_digest,
            "record_digests": [record.record_digest for record in self.records],
            "logical_cell_count": len(self.records),
            "numeric_record_count": len(self.numeric_records),
            "structural_na_count": len(self.records) - len(self.numeric_records),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "records": [record.to_dict() for record in self.records],
            "ledger_digest": self.ledger_digest,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, plan: SignalMatrixPlan
    ) -> "SignalMatrixLedger":
        fields = {
            "schema",
            "plan_digest",
            "record_digests",
            "logical_cell_count",
            "numeric_record_count",
            "structural_na_count",
            "records",
            "ledger_digest",
        }
        _strict(value, fields, "signal matrix ledger")
        if value["plan_digest"] != plan.plan_digest:
            raise SignalMatrixError("serialized ledger is bound to another plan")
        ledger = cls(
            plan=plan,
            records=tuple(SignalCellRecord.from_dict(item) for item in value["records"]),
            ledger_digest=value["ledger_digest"],
            schema=value["schema"],
        )
        expected = ledger._payload_without_digest()
        for name in (
            "record_digests",
            "logical_cell_count",
            "numeric_record_count",
            "structural_na_count",
        ):
            if value[name] != expected[name]:
                raise SignalMatrixError(f"serialized ledger {name} is inconsistent")
        return ledger


__all__ = [
    "CORE_INPUT_VIEW_IDS",
    "C_RF_SHUFFLED_NEXT",
    "MECHANISM_CONDITIONS",
    "MECHANISM_REPRESENTATIONS",
    "REPRESENTATION_SEEDS",
    "SignalCell",
    "SignalCellRecord",
    "SignalFitJob",
    "SignalMatrixError",
    "SignalMatrixLedger",
    "SignalMatrixPlan",
    "build_optimization_fit_jobs",
    "build_signal_matrix_plan",
]
