"""Pure v0.3 query/source distance stage and recomputation scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from ..hashing import sha256_json
from ..rkme.distance import empirical_to_reduced_distance
from ..rkme.gaussian import GaussianKernel
from .contracts import (
    EmpiricalQuerySpec,
    MarketBoundSourceRepresentationIndex,
    QuerySpec,
    RankingKey,
    ReducedQuerySpec,
    SourceReducedSpec,
    SourceRepresentationIndex,
    V03ContractError,
)


DistanceForm = Literal["mmd", "mmd2"]
JOINT_DISTANCE_RUN_SCHEMA = "policy-learnware.v03-joint-distance-run.v0"
JOINT_RECOMPUTE_REPORT_SCHEMA = "policy-learnware.v03-joint-recompute-report.v0"
MINIMAL_NUMERIC_GATE_SCHEMA = "policy-learnware.v03-minimal-numeric-gate.v0"


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise V03ContractError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise V03ContractError(f"{where} must be a lowercase SHA-256 digest") from error
    return value


def _negative_tolerance(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V03ContractError("negative_tolerance must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise V03ContractError("negative_tolerance must be finite and non-negative")
    return result


def tie_break_digest(tokens: Mapping[str, str]) -> str:
    parsed: dict[str, str] = {}
    if not isinstance(tokens, Mapping) or not tokens:
        raise V03ContractError("tie-break token map cannot be empty")
    for opaque_learnware_id, token in tokens.items():
        if not isinstance(opaque_learnware_id, str) or not opaque_learnware_id:
            raise V03ContractError("tie-break opaque IDs must be non-empty")
        parsed[opaque_learnware_id] = _digest(
            token,
            f"tie_break_tokens[{opaque_learnware_id!r}]",
        )
    if len(set(parsed.values())) != len(parsed):
        raise V03ContractError("tie-break tokens must be unique")
    return sha256_json(
        {
            "schema": "policy-learnware.v03-tie-break-map.v0",
            "tokens": dict(sorted(parsed.items())),
        }
    )


@dataclass(frozen=True)
class QuerySourceDistance:
    distance: float
    squared_distance: float
    raw_squared_distance: float
    clamped: bool
    distance_form: DistanceForm
    query_mode: str

    def __post_init__(self) -> None:
        for name in ("distance", "squared_distance", "raw_squared_distance"):
            if not math.isfinite(float(getattr(self, name))):
                raise V03ContractError(f"{name} must be finite")
        if self.distance < 0.0 or self.squared_distance < 0.0:
            raise V03ContractError("distances must be non-negative")
        if not math.isclose(
            self.distance * self.distance,
            self.squared_distance,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise V03ContractError("distance and squared_distance disagree")
        if type(self.clamped) is not bool:
            raise V03ContractError("clamped must be boolean")
        expected_squared = max(self.raw_squared_distance, 0.0)
        if not math.isclose(
            self.squared_distance,
            expected_squared,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise V03ContractError(
                "squared_distance must be the non-negative raw squared distance"
            )
        if self.clamped != (self.raw_squared_distance < 0.0):
            raise V03ContractError("clamped flag disagrees with raw_squared_distance")
        if self.distance_form not in {"mmd", "mmd2"}:
            raise V03ContractError("unsupported distance_form")
        if self.query_mode not in {"QUERY_EMPIRICAL", "QUERY_REDUCED"}:
            raise V03ContractError("unsupported query_mode")

    @property
    def value(self) -> float:
        return self.distance if self.distance_form == "mmd" else self.squared_distance

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "squared_distance": self.squared_distance,
            "raw_squared_distance": self.raw_squared_distance,
            "clamped": self.clamped,
            "distance_form": self.distance_form,
            "query_mode": self.query_mode,
            "value": self.value,
        }


def _validate_query_source_bindings(
    query: QuerySpec, source: SourceReducedSpec
) -> None:
    checks = {
        "representation_protocol_id": (
            query.representation_protocol_id,
            source.representation_protocol_id,
        ),
        "measurement_protocol_id": (
            query.measurement_protocol_id,
            source.measurement_protocol_id,
        ),
        "canonical_view_digest": (
            query.canonical_view_digest,
            source.canonical_view_digest,
        ),
        "kernel_evaluator_digest": (
            query.spec_key.kernel_evaluator_digest,
            source.spec_key.kernel_evaluator_digest,
        ),
        "sample_weighting_digest": (
            query.spec_key.sample_weighting_digest,
            source.spec_key.sample_weighting_digest,
        ),
        "kernel_bandwidth": (query.kernel_bandwidth, source.kernel_bandwidth),
        "latent_dim": (query.latent_dim, source.latent_dim),
    }
    mismatches = {name: values for name, values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise V03ContractError(f"query/source bindings differ: {mismatches}")


def _reduced_to_reduced_distance(
    query: ReducedQuerySpec,
    source: SourceReducedSpec,
    *,
    negative_tolerance: float,
) -> tuple[float, float, float, bool]:
    kernel = GaussianKernel(query.kernel_bandwidth)
    cross = float(
        query.reduced_kme.beta
        @ kernel.gram(query.reduced_kme.supports, source.reduced_kme.supports)
        @ source.reduced_kme.beta
    )
    raw = float(
        query.reduced_kme.rkme_norm2
        - 2.0 * cross
        + source.reduced_kme.rkme_norm2
    )
    scale = max(
        1.0,
        abs(query.reduced_kme.rkme_norm2),
        abs(source.reduced_kme.rkme_norm2),
        abs(2.0 * cross),
    )
    if raw < -negative_tolerance * scale:
        raise ArithmeticError(f"reduced query/source MMD squared is materially negative ({raw})")
    squared = max(raw, 0.0)
    return math.sqrt(squared), squared, raw, raw < 0.0


def query_to_source_distance(
    query: QuerySpec,
    source: SourceReducedSpec,
    *,
    distance_form: DistanceForm = "mmd",
    block_size: int = 2048,
    negative_tolerance: float = 1.0e-8,
) -> QuerySourceDistance:
    """Dispatch explicitly by query role; this function never auto-falls back."""

    if not isinstance(query, (EmpiricalQuerySpec, ReducedQuerySpec)):
        raise V03ContractError("query has an unsupported v0.3 contract")
    if not isinstance(source, SourceReducedSpec):
        raise V03ContractError("source must be a SourceReducedSpec")
    if distance_form not in {"mmd", "mmd2"}:
        raise V03ContractError("distance_form must be 'mmd' or 'mmd2'")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise V03ContractError("block_size must be a positive integer")
    tolerance = _negative_tolerance(negative_tolerance)
    _validate_query_source_bindings(query, source)
    if isinstance(query, EmpiricalQuerySpec):
        result = empirical_to_reduced_distance(
            query.empirical_kme,
            source.reduced_kme,
            block_size=block_size,
            negative_tolerance=tolerance,
        )
        values = (
            result.distance,
            result.squared_distance,
            result.raw_squared_distance,
            result.clamped,
        )
    else:
        values = _reduced_to_reduced_distance(
            query, source, negative_tolerance=tolerance
        )
    return QuerySourceDistance(
        distance=values[0],
        squared_distance=values[1],
        raw_squared_distance=values[2],
        clamped=values[3],
        distance_form=distance_form,
        query_mode=query.query_mode,
    )


@dataclass(frozen=True)
class JointDistanceRequest:
    query_spec: QuerySpec
    source_index: SourceRepresentationIndex | MarketBoundSourceRepresentationIndex
    ranking_key: RankingKey
    tie_break_tokens: Mapping[str, str]
    distance_form: DistanceForm = "mmd"
    block_size: int = 2048
    negative_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if not isinstance(self.query_spec, (EmpiricalQuerySpec, ReducedQuerySpec)):
            raise V03ContractError("joint request query has the wrong type")
        if not isinstance(
            self.source_index,
            (SourceRepresentationIndex, MarketBoundSourceRepresentationIndex),
        ):
            raise V03ContractError("joint request index has the wrong type")
        if not isinstance(self.ranking_key, RankingKey):
            raise V03ContractError("joint request ranking key has the wrong type")
        if self.query_spec.representation_protocol_id != self.source_index.representation_protocol_id:
            raise V03ContractError("query and source index use different representation protocols")
        if self.ranking_key.query_spec_digest != self.query_spec.query_spec_digest:
            raise V03ContractError("RankingKey is bound to another query")
        if (
            self.ranking_key.representation_index_digest
            != self.source_index.representation_index_digest
        ):
            raise V03ContractError("RankingKey is bound to another source index")
        tokens = dict(self.tie_break_tokens)
        if set(tokens) != set(self.source_index.entries):
            raise V03ContractError("tie-break coverage differs from source index")
        observed_tie_digest = tie_break_digest(tokens)
        if self.ranking_key.tie_break_digest != observed_tie_digest:
            raise V03ContractError("RankingKey tie-break digest is inconsistent")
        if self.distance_form not in {"mmd", "mmd2"}:
            raise V03ContractError("unsupported distance_form")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int) or self.block_size <= 0:
            raise V03ContractError("block_size must be a positive integer")
        object.__setattr__(self, "negative_tolerance", _negative_tolerance(self.negative_tolerance))
        object.__setattr__(self, "tie_break_tokens", MappingProxyType(dict(sorted(tokens.items()))))

    @property
    def request_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-joint-distance-request.v0",
                "query_spec_digest": self.query_spec.query_spec_digest,
                "representation_index_digest": self.source_index.representation_index_digest,
                "ranking_key_digest": self.ranking_key.ranking_key_digest,
                "tie_break_digest": self.ranking_key.tie_break_digest,
                "distance_form": self.distance_form,
                "block_size": self.block_size,
                "negative_tolerance": self.negative_tolerance,
            }
        )


@dataclass(frozen=True)
class JointDistanceRow:
    opaque_learnware_id: str
    rank: int
    result: QuerySourceDistance

    def to_dict(self) -> dict[str, Any]:
        return {
            "opaque_learnware_id": self.opaque_learnware_id,
            "rank": self.rank,
            **self.result.to_dict(),
        }


@dataclass(frozen=True)
class JointDistanceRun:
    request_digest: str
    query_spec_digest: str
    representation_index_digest: str
    ranking_key_digest: str
    query_mode: str
    distance_form: DistanceForm
    rows: tuple[JointDistanceRow, ...]
    clamp_count: int
    run_digest: str | None = None
    schema: str = JOINT_DISTANCE_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != JOINT_DISTANCE_RUN_SCHEMA:
            raise V03ContractError("unsupported JointDistanceRun schema")
        for name in (
            "request_digest",
            "query_spec_digest",
            "representation_index_digest",
            "ranking_key_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not self.rows:
            raise V03ContractError("joint distance run cannot be empty")
        if tuple(row.rank for row in self.rows) != tuple(range(1, len(self.rows) + 1)):
            raise V03ContractError("joint distance ranks must be contiguous")
        if len({row.opaque_learnware_id for row in self.rows}) != len(self.rows):
            raise V03ContractError("joint distance rows contain duplicate IDs")
        if self.clamp_count != sum(row.result.clamped for row in self.rows):
            raise V03ContractError("clamp_count does not match rows")
        expected = sha256_json(self._payload_without_digest())
        if self.run_digest is None:
            object.__setattr__(self, "run_digest", expected)
        elif _digest(self.run_digest, "run_digest") != expected:
            raise V03ContractError("run_digest does not match joint distance rows")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_digest": self.request_digest,
            "query_spec_digest": self.query_spec_digest,
            "representation_index_digest": self.representation_index_digest,
            "ranking_key_digest": self.ranking_key_digest,
            "query_mode": self.query_mode,
            "distance_form": self.distance_form,
            "rows": [row.to_dict() for row in self.rows],
            "clamp_count": self.clamp_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "run_digest": self.run_digest}


def run_joint_distance_stage(request: JointDistanceRequest) -> JointDistanceRun:
    """Run the deterministic public distance/ranking stage over the full index."""

    if not isinstance(request, JointDistanceRequest):
        raise V03ContractError("request must be a JointDistanceRequest")
    scored: list[tuple[float, str, str, QuerySourceDistance]] = []
    for opaque_learnware_id, source in request.source_index.entries.items():
        result = query_to_source_distance(
            request.query_spec,
            source,
            distance_form=request.distance_form,
            block_size=request.block_size,
            negative_tolerance=request.negative_tolerance,
        )
        scored.append(
            (
                result.value,
                request.tie_break_tokens[opaque_learnware_id],
                opaque_learnware_id,
                result,
            )
        )
    scored.sort(key=lambda item: (item[0], item[1]))
    rows = tuple(
        JointDistanceRow(
            opaque_learnware_id=opaque_learnware_id,
            rank=rank,
            result=result,
        )
        for rank, (_value, _token, opaque_learnware_id, result) in enumerate(
            scored,
            start=1,
        )
    )
    return JointDistanceRun(
        request_digest=request.request_digest,
        query_spec_digest=str(request.query_spec.query_spec_digest),
        representation_index_digest=str(request.source_index.representation_index_digest),
        ranking_key_digest=str(request.ranking_key.ranking_key_digest),
        query_mode=request.query_spec.query_mode,
        distance_form=request.distance_form,
        rows=rows,
        clamp_count=sum(row.result.clamped for row in rows),
    )


@dataclass(frozen=True)
class JointRecomputeReport:
    published_run_digest: str
    recomputed_run_digest: str
    matched: bool
    report_digest: str | None = None
    schema: str = JOINT_RECOMPUTE_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != JOINT_RECOMPUTE_REPORT_SCHEMA:
            raise V03ContractError("unsupported JointRecomputeReport schema")
        object.__setattr__(
            self, "published_run_digest", _digest(self.published_run_digest, "published_run_digest")
        )
        object.__setattr__(
            self,
            "recomputed_run_digest",
            _digest(self.recomputed_run_digest, "recomputed_run_digest"),
        )
        if type(self.matched) is not bool:
            raise V03ContractError("matched must be boolean")
        if self.matched != (self.published_run_digest == self.recomputed_run_digest):
            raise V03ContractError("matched must be derived from run digests")
        expected = sha256_json(self._payload_without_digest())
        if self.report_digest is None:
            object.__setattr__(self, "report_digest", expected)
        elif _digest(self.report_digest, "report_digest") != expected:
            raise V03ContractError("report_digest does not match recomputation report")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "published_run_digest": self.published_run_digest,
            "recomputed_run_digest": self.recomputed_run_digest,
            "matched": self.matched,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "report_digest": self.report_digest}


def recompute_joint_distance_run(
    request: JointDistanceRequest, published: JointDistanceRun
) -> JointRecomputeReport:
    """Independent pure-function recomputation hook for a later formal runner."""

    if published.request_digest != request.request_digest:
        raise V03ContractError("published run is bound to another request")
    recomputed = run_joint_distance_stage(request)
    return JointRecomputeReport(
        published_run_digest=str(published.run_digest),
        recomputed_run_digest=str(recomputed.run_digest),
        matched=published.run_digest == recomputed.run_digest,
    )


@dataclass(frozen=True)
class MinimalNumericGateEvidence:
    checks: Mapping[str, bool]
    evidence_digest: str | None = None
    schema: str = MINIMAL_NUMERIC_GATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MINIMAL_NUMERIC_GATE_SCHEMA:
            raise V03ContractError("unsupported MinimalNumericGateEvidence schema")
        checks = dict(sorted(self.checks.items()))
        if not checks or any(type(value) is not bool for value in checks.values()):
            raise V03ContractError("minimal numeric gate checks must be non-empty booleans")
        object.__setattr__(self, "checks", MappingProxyType(checks))
        expected = sha256_json(self._payload_without_digest())
        if self.evidence_digest is None:
            object.__setattr__(self, "evidence_digest", expected)
        elif _digest(self.evidence_digest, "evidence_digest") != expected:
            raise V03ContractError("evidence_digest does not match gate checks")

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checks": dict(self.checks),
            "passed": self.passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "evidence_digest": self.evidence_digest}


def evaluate_minimal_numeric_gate(
    request: JointDistanceRequest,
    run: JointDistanceRun,
    recompute: JointRecomputeReport,
) -> MinimalNumericGateEvidence:
    """Development-only scaffold; this does not authorize any formal v0.3 gate."""

    return MinimalNumericGateEvidence(
        {
            "full_source_coverage": len(run.rows) == len(request.source_index.entries),
            "query_role_bound": run.query_mode == request.query_spec.query_mode,
            "ranking_key_bound": run.ranking_key_digest == request.ranking_key.ranking_key_digest,
            "all_distances_finite": all(
                math.isfinite(row.result.value) for row in run.rows
            ),
            "independent_recompute_match": recompute.matched,
        }
    )


__all__ = [
    "DistanceForm",
    "JointDistanceRequest",
    "JointDistanceRun",
    "JointRecomputeReport",
    "MinimalNumericGateEvidence",
    "QuerySourceDistance",
    "evaluate_minimal_numeric_gate",
    "query_to_source_distance",
    "recompute_joint_distance_run",
    "run_joint_distance_stage",
    "tie_break_digest",
]
