"""Typed end-to-end cost ledger for the frozen v0.3 protocol.

The signal and policy tables report scientific values; this module records the
resources required to produce them.  A formal ledger has one row for every
pre-registered component in plan section 14.4.  Missing work is never encoded
as zero and a caller cannot silently omit a slow stage from the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

from ..hashing import sha256_json
from .schemas import checked_digest, checked_safe_id, strict_mapping


COST_COMPONENT_SCHEMA = "policy-learnware.v03-cost-component.v0"
COST_LEDGER_SCHEMA = "policy-learnware.v03-cost-ledger.v0"
PUBLIC_COST_LEDGER_SCHEMA = "policy-learnware.v03-public-cost-ledger.v0"

COST_COMPONENT_IDS = (
    "PROBE_COLLECTION",
    "CANONICALIZATION",
    "REPRESENTATION_FIT",
    "ENCODE",
    "SOURCE_REDUCTION",
    "QUERY_KME",
    "DISTANCE",
    "END_TO_END_COLD",
    "END_TO_END_WARM",
)
CostScope = Literal["DEVELOPMENT", "FORMAL"]


class V03CostError(ValueError):
    """A cost measurement is incomplete, non-finite, or unbound."""


def _digest(value: Any, where: str) -> str:
    try:
        return checked_digest(value, where)
    except ValueError as error:
        raise V03CostError(str(error)) from error


def _id(value: Any, where: str) -> str:
    try:
        return checked_safe_id(value, where)
    except ValueError as error:
        raise V03CostError(str(error)) from error


def _nonnegative_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V03CostError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise V03CostError(f"{where} must be finite and non-negative")
    return result


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V03CostError(f"{where} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class CostComponentRecord:
    component_id: str
    measurement_receipt_digest: str
    input_artifact_set_digest: str
    output_artifact_set_digest: str
    wall_seconds: float
    gpu_seconds: float
    peak_memory_bytes: int
    artifact_bytes: int
    environment_steps: int = 0
    invocation_count: int = 1
    device_class: str = "cpu"
    schema: str = COST_COMPONENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COST_COMPONENT_SCHEMA:
            raise V03CostError("unsupported CostComponentRecord schema")
        if self.component_id not in COST_COMPONENT_IDS:
            raise V03CostError("unknown v0.3 cost component")
        for name in (
            "measurement_receipt_digest",
            "input_artifact_set_digest",
            "output_artifact_set_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("wall_seconds", "gpu_seconds"):
            object.__setattr__(
                self, name, _nonnegative_float(getattr(self, name), name)
            )
        for name in (
            "peak_memory_bytes",
            "artifact_bytes",
            "environment_steps",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        if (
            isinstance(self.invocation_count, bool)
            or not isinstance(self.invocation_count, int)
            or self.invocation_count <= 0
        ):
            raise V03CostError("invocation_count must be positive")
        object.__setattr__(self, "device_class", _id(self.device_class, "device_class"))
        if self.component_id == "PROBE_COLLECTION":
            if self.environment_steps <= 0:
                raise V03CostError("probe collection requires positive environment steps")
        elif self.environment_steps != 0:
            raise V03CostError(
                "only PROBE_COLLECTION may report environment steps"
            )
        if self.component_id == "END_TO_END_WARM" and self.invocation_count < 2:
            raise V03CostError(
                "warm latency requires at least two measured invocations"
            )

    @property
    def record_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CostComponentRecord":
        fields = set(cls.__dataclass_fields__)
        try:
            data = strict_mapping(value, fields, "cost component")
        except ValueError as error:
            raise V03CostError(str(error)) from error
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class V03CostLedger:
    run_id: str
    execution_scope: CostScope
    freeze_manifest_digest: str | None
    cost_protocol_digest: str
    prefix_cost_evidence_digest: str
    components: tuple[CostComponentRecord, ...]
    ledger_digest: str | None = None
    schema: str = COST_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COST_LEDGER_SCHEMA:
            raise V03CostError("unsupported V03CostLedger schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        if self.execution_scope not in {"DEVELOPMENT", "FORMAL"}:
            raise V03CostError("execution_scope must be DEVELOPMENT or FORMAL")
        if self.execution_scope == "FORMAL":
            if self.freeze_manifest_digest is None:
                raise V03CostError("formal cost ledger requires a freeze manifest")
            object.__setattr__(
                self,
                "freeze_manifest_digest",
                _digest(self.freeze_manifest_digest, "freeze_manifest_digest"),
            )
        elif self.freeze_manifest_digest is not None:
            raise V03CostError(
                "development cost ledger cannot claim a formal freeze"
            )
        for name in ("cost_protocol_digest", "prefix_cost_evidence_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        components = tuple(self.components)
        if not all(isinstance(item, CostComponentRecord) for item in components):
            raise V03CostError("cost ledger requires typed component rows")
        by_id = {item.component_id: item for item in components}
        if len(by_id) != len(components):
            raise V03CostError("cost ledger contains duplicate components")
        if self.execution_scope == "FORMAL" and set(by_id) != set(COST_COMPONENT_IDS):
            missing = sorted(set(COST_COMPONENT_IDS) - set(by_id))
            extra = sorted(set(by_id) - set(COST_COMPONENT_IDS))
            raise V03CostError(
                f"formal cost ledger requires exact component coverage; missing={missing}, extra={extra}"
            )
        if not components:
            raise V03CostError("cost ledger cannot be empty")
        object.__setattr__(
            self,
            "components",
            tuple(by_id[item] for item in COST_COMPONENT_IDS if item in by_id),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.ledger_digest is None:
            object.__setattr__(self, "ledger_digest", expected)
        elif _digest(self.ledger_digest, "ledger_digest") != expected:
            raise V03CostError("cost ledger digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "execution_scope": self.execution_scope,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "cost_protocol_digest": self.cost_protocol_digest,
            "prefix_cost_evidence_digest": self.prefix_cost_evidence_digest,
            "component_record_digests": [
                item.record_digest for item in self.components
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "components": [item.to_dict() for item in self.components],
            "ledger_digest": self.ledger_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V03CostLedger":
        fields = {
            "schema",
            "run_id",
            "execution_scope",
            "freeze_manifest_digest",
            "cost_protocol_digest",
            "prefix_cost_evidence_digest",
            "component_record_digests",
            "components",
            "ledger_digest",
        }
        try:
            data = strict_mapping(value, fields, "cost ledger")
        except ValueError as error:
            raise V03CostError(str(error)) from error
        if not isinstance(data["components"], list) or not isinstance(
            data["component_record_digests"], list
        ):
            raise V03CostError("cost ledger component fields must be lists")
        components = tuple(
            CostComponentRecord.from_dict(item) for item in data["components"]
        )
        if data["component_record_digests"] != [
            item.record_digest for item in components
        ]:
            raise V03CostError("cost component digest projection differs")
        return cls(
            run_id=data["run_id"],
            execution_scope=data["execution_scope"],
            freeze_manifest_digest=data["freeze_manifest_digest"],
            cost_protocol_digest=data["cost_protocol_digest"],
            prefix_cost_evidence_digest=data["prefix_cost_evidence_digest"],
            components=components,
            ledger_digest=data["ledger_digest"],
            schema=data["schema"],
        )

    def to_public_dict(self) -> dict[str, Any]:
        wall = {item.component_id: item.wall_seconds for item in self.components}
        gpu = {item.component_id: item.gpu_seconds for item in self.components}
        payload = {
            "schema": PUBLIC_COST_LEDGER_SCHEMA,
            "run_id": self.run_id,
            "execution_scope": self.execution_scope,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "cost_protocol_digest": self.cost_protocol_digest,
            "prefix_cost_evidence_digest": self.prefix_cost_evidence_digest,
            "component_ids": [item.component_id for item in self.components],
            "wall_seconds_by_component": wall,
            "gpu_seconds_by_component": gpu,
            "probe_environment_steps": next(
                (
                    item.environment_steps
                    for item in self.components
                    if item.component_id == "PROBE_COLLECTION"
                ),
                0,
            ),
            "peak_memory_bytes": max(
                item.peak_memory_bytes for item in self.components
            ),
            "total_artifact_bytes": sum(item.artifact_bytes for item in self.components),
            "private_physical_paths_withheld": True,
            "private_ledger_digest": self.ledger_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


def frozen_cost_protocol_digest() -> str:
    """Return the canonical section-14.4 component/measurement protocol."""

    return sha256_json(
        {
            "schema": "policy-learnware.v03-cost-protocol.v0",
            "component_ids": list(COST_COMPONENT_IDS),
            "latency": {
                "cold": "first isolated end-to-end invocation",
                "warm": "median after one discarded warmup; invocation_count>=2",
            },
            "memory": "process peak resident-or-device bytes; maximum across components",
            "artifact_size": "exact immutable output bytes",
            "probe_steps": "native environment transitions",
            "missing_component_policy": "FORMAL_FAIL_CLOSED_NOT_ZERO",
        }
    )


__all__ = [
    "COST_COMPONENT_IDS",
    "COST_COMPONENT_SCHEMA",
    "COST_LEDGER_SCHEMA",
    "PUBLIC_COST_LEDGER_SCHEMA",
    "CostComponentRecord",
    "V03CostError",
    "V03CostLedger",
    "frozen_cost_protocol_digest",
]
