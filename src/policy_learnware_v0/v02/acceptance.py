"""Deterministic CPU end-to-end acceptance fixture for the v0.2 sidecar.

This module is intentionally a *development* fixture.  It exercises the real
v0.2 records and selectors over two synthetic tasks, but it never claims the
six-task formal gate described by the experiment plan.  In particular, the
private oracle is retained only on :class:`CpuAcceptanceRun`; the serialised
acceptance report contains digest bindings rather than private value vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json
from .baselines import (
    CompetenceOnlySelector,
    DevelopmentView,
    EnvironmentOnlySelector,
    FrozenFeatureIndex,
    KnnDevelopmentSelector,
    LegacyTaskSpecSelector,
    LinearDevelopmentSelector,
    RandomAnonymousMarketSelector,
    SourceOnlyLMinSelector,
    TargetQueryView,
    VectorNearestSelector,
    derive_source_only_sigma_artifacts,
    target_probe_evidence_contract,
)
from .competence import ChampionizationResult, SourceEpisodeRow, championize_by_anchor
from .costs import (
    QUERY_COST_COMPONENTS,
    ColdWarmCostReconciliation,
    CostRecord,
    reconcile_cold_warm_costs,
)
from .environment_spec import RepresentationIndex, RepresentationIndexEntry
from .market import V02PolicyMarket, build_policy_market
from .metrics import HierarchicalAggregate, HierarchicalValue, aggregate_hierarchy
from .oracle import (
    FullPoolOracleResult,
    OracleEpisodeRow,
    PublishedSelection,
    aggregate_full_pool_oracle,
)
from .report import DevelopmentPTable, DevelopmentPTableRow, build_development_p_table
from .representation import TraceFeatureVector
from .schemas import EnvironmentSpec, ExecutionABIRecord
from .selectors import PublicMarketView, SelectionRecord
from .training import (
    AdmittedTrainingRecord,
    PolicyTrainingAttestation,
    PolicyTrainingJob,
    plan_training_jobs,
)


AcceptanceScenario = Literal[
    "scientific_pass",
    "no_go_market",
    "no_go_corro",
    "engineering_blocked",
]
AcceptanceStatus = Literal[
    "SCIENTIFIC_PASS",
    "NO_GO_MARKET",
    "NO_GO_CORRO",
    "BLOCKED_ENGINEERING",
]

ACCEPTANCE_REPORT_SCHEMA = "policy-learnware.v02-cpu-acceptance-report.v0"
ACCEPTANCE_GATE_SCHEMA = "policy-learnware.v02-cpu-acceptance-gate.v0"
ACCEPTANCE_SCENARIOS = frozenset(
    {"scientific_pass", "no_go_market", "no_go_corro", "engineering_blocked"}
)
ACCEPTANCE_STATUSES = frozenset(
    {"SCIENTIFIC_PASS", "NO_GO_MARKET", "NO_GO_CORRO", "BLOCKED_ENGINEERING"}
)
ACCEPTANCE_GATE_IDS = (
    "CPU-CORRO",
    "CPU-Engineering",
    "CPU-Market",
    "CPU-Scientific",
)
FULL_METHOD_IDS = (
    "A-Env",
    "B0",
    "B1",
    "B2",
    "B3a",
    "B3b",
    "B4a",
    "B4b",
    "M02/B5",
)
CORRO_FAIL_METHOD_IDS = tuple(item for item in FULL_METHOD_IDS if item != "M02/B5")


class AcceptanceContractError(ValueError):
    """The CPU fixture or its digest-bound report is inconsistent."""


def _fixture_digest(label: str) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-cpu-acceptance-fixture-domain.v0",
            "label": label,
        }
    )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AcceptanceContractError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise AcceptanceContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise AcceptanceContractError(f"{where} must be a SHA-256 digest") from error
    return result


def _optional_digest(value: Any, where: str) -> str | None:
    return None if value is None else _digest(value, where)


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AcceptanceContractError(f"{where} must be a non-negative integer")
    return value


def _strict_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise AcceptanceContractError(f"{where} must be a mapping")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise AcceptanceContractError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


@dataclass(frozen=True)
class AcceptanceGateDecision:
    """One fixture-local gate with exact boolean checks and a content digest."""

    gate_id: str
    checks: Mapping[str, bool]
    gate_digest: str | None = None

    def __post_init__(self) -> None:
        gate_id = _nonempty(self.gate_id, "gate_id")
        checks = dict(self.checks)
        if not checks:
            raise AcceptanceContractError("acceptance gate checks cannot be empty")
        parsed: dict[str, bool] = {}
        for name, passed in checks.items():
            key = _nonempty(name, "gate check name")
            if type(passed) is not bool:
                raise AcceptanceContractError("acceptance gate checks must be boolean")
            parsed[key] = passed
        parsed = dict(sorted(parsed.items()))
        object.__setattr__(self, "gate_id", gate_id)
        object.__setattr__(self, "checks", MappingProxyType(parsed))
        expected = sha256_json(self._payload_without_digest())
        if self.gate_digest is None:
            object.__setattr__(self, "gate_digest", expected)
        elif _digest(self.gate_digest, "gate_digest") != expected:
            raise AcceptanceContractError("gate_digest does not match gate contents")

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": ACCEPTANCE_GATE_SCHEMA,
            "gate_id": self.gate_id,
            "checks": dict(self.checks),
            "passed": self.passed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "gate_digest": self.gate_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcceptanceGateDecision":
        expected = {"schema", "gate_id", "checks", "passed", "gate_digest"}
        _strict_keys(value, expected, "AcceptanceGateDecision")
        if value["schema"] != ACCEPTANCE_GATE_SCHEMA:
            raise AcceptanceContractError("unsupported acceptance gate schema")
        gate = cls(
            gate_id=value["gate_id"],
            checks=value["checks"],
            gate_digest=value["gate_digest"],
        )
        if value["passed"] is not gate.passed:
            raise AcceptanceContractError("serialized gate passed flag is inconsistent")
        return gate


def _derive_status(gates: Mapping[str, AcceptanceGateDecision]) -> AcceptanceStatus:
    engineering = gates["CPU-Engineering"].passed
    market = gates["CPU-Market"].passed
    corro = gates["CPU-CORRO"].passed
    scientific = gates["CPU-Scientific"].passed
    if not engineering:
        return "BLOCKED_ENGINEERING"
    if not market:
        return "NO_GO_MARKET"
    if not corro:
        return "NO_GO_CORRO"
    if scientific:
        return "SCIENTIFIC_PASS"
    raise AcceptanceContractError(
        "engineering/market/CORRO passed but the scientific fixture is incomplete"
    )


@dataclass(frozen=True)
class AcceptanceReport:
    """Public, strict report for the two-task CPU development fixture.

    The report binds all private/scientific products by digest.  It deliberately
    cannot represent a six-task formal completion: the stage and fixture scope
    are fixed and ``formal_completion_claimed`` must be false.
    """

    scenario: AcceptanceScenario
    status: AcceptanceStatus
    gates: Mapping[str, AcceptanceGateDecision]
    fixture_task_count: int
    axes_per_task: int
    anchors_per_task: int
    source_anchor_count: int
    training_seed_count: int
    planned_training_run_count: int
    admitted_training_run_count: int
    champion_count: int
    public_market_entry_count: int
    query_count: int
    method_ids: tuple[str, ...]
    training_plan_digest: str
    admitted_records_digest: str | None
    championization_digest: str | None
    policy_market_id: str | None
    raw_representation_index_id: str | None
    corro_representation_index_id: str | None
    source_sigma_artifact_digest: str | None
    selection_records_digest: str | None
    private_oracle_results_digest: str | None
    metrics_digest: str | None
    cost_reconciliation_digest: str | None
    development_p_table_digest: str | None
    blocked_reason_digest: str | None
    stage: str = "development_discovery"
    formal_task_requirement: int = 6
    formal_completion_claimed: bool = False
    acceptance_report_digest: str | None = None

    def __post_init__(self) -> None:
        if self.scenario not in ACCEPTANCE_SCENARIOS:
            raise AcceptanceContractError(f"unsupported acceptance scenario: {self.scenario!r}")
        if self.status not in ACCEPTANCE_STATUSES:
            raise AcceptanceContractError(f"unsupported acceptance status: {self.status!r}")
        if self.stage != "development_discovery":
            raise AcceptanceContractError("CPU acceptance report is development-only")
        if self.formal_task_requirement != 6:
            raise AcceptanceContractError("formal task requirement must remain six")
        if self.formal_completion_claimed is not False:
            raise AcceptanceContractError("two-task CPU fixture cannot claim formal completion")

        gates = dict(self.gates)
        if tuple(sorted(gates)) != ACCEPTANCE_GATE_IDS:
            raise AcceptanceContractError("acceptance report requires the exact fixture gate set")
        if any(not isinstance(item, AcceptanceGateDecision) for item in gates.values()):
            raise AcceptanceContractError("gates must contain AcceptanceGateDecision objects")
        if any(key != item.gate_id for key, item in gates.items()):
            raise AcceptanceContractError("gate mapping keys must match gate_id")
        object.__setattr__(self, "gates", MappingProxyType(dict(sorted(gates.items()))))
        if self.status != _derive_status(gates):
            raise AcceptanceContractError("acceptance status disagrees with gate decisions")

        expected_status = {
            "scientific_pass": "SCIENTIFIC_PASS",
            "no_go_market": "NO_GO_MARKET",
            "no_go_corro": "NO_GO_CORRO",
            "engineering_blocked": "BLOCKED_ENGINEERING",
        }[self.scenario]
        if self.status != expected_status:
            raise AcceptanceContractError("scenario and acceptance status disagree")

        for name in (
            "fixture_task_count",
            "axes_per_task",
            "anchors_per_task",
            "source_anchor_count",
            "training_seed_count",
            "planned_training_run_count",
            "admitted_training_run_count",
            "champion_count",
            "public_market_entry_count",
            "query_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if (
            self.fixture_task_count != 2
            or self.axes_per_task != 2
            or self.anchors_per_task != 5
            or self.source_anchor_count != 10
            or self.training_seed_count != 2
            or self.planned_training_run_count != 20
        ):
            raise AcceptanceContractError(
                "CPU fixture scope must be 2 tasks x 5 shared-nominal anchors x 2 seeds"
            )
        if self.planned_training_run_count != self.source_anchor_count * self.training_seed_count:
            raise AcceptanceContractError("planned training run matrix does not reconcile")

        methods = tuple(_nonempty(item, "method_ids[]") for item in self.method_ids)
        if methods != tuple(sorted(set(methods))):
            raise AcceptanceContractError("method_ids must be sorted and unique")
        object.__setattr__(self, "method_ids", methods)
        expected_methods = () if self.status == "BLOCKED_ENGINEERING" else (
            CORRO_FAIL_METHOD_IDS if self.status == "NO_GO_CORRO" else FULL_METHOD_IDS
        )
        if methods != expected_methods:
            raise AcceptanceContractError("reported method set disagrees with fixture outcome")

        object.__setattr__(
            self, "training_plan_digest", _digest(self.training_plan_digest, "training_plan_digest")
        )
        for name in (
            "admitted_records_digest",
            "championization_digest",
            "policy_market_id",
            "raw_representation_index_id",
            "corro_representation_index_id",
            "source_sigma_artifact_digest",
            "selection_records_digest",
            "private_oracle_results_digest",
            "metrics_digest",
            "cost_reconciliation_digest",
            "development_p_table_digest",
            "blocked_reason_digest",
        ):
            object.__setattr__(self, name, _optional_digest(getattr(self, name), name))

        if self.status == "BLOCKED_ENGINEERING":
            if any(
                (
                    self.admitted_training_run_count,
                    self.champion_count,
                    self.public_market_entry_count,
                    self.query_count,
                )
            ):
                raise AcceptanceContractError("blocked engineering fixture cannot publish science rows")
            if self.blocked_reason_digest is None:
                raise AcceptanceContractError("blocked engineering fixture requires a reason digest")
            forbidden = (
                self.admitted_records_digest,
                self.championization_digest,
                self.policy_market_id,
                self.raw_representation_index_id,
                self.corro_representation_index_id,
                self.source_sigma_artifact_digest,
                self.selection_records_digest,
                self.private_oracle_results_digest,
                self.metrics_digest,
                self.cost_reconciliation_digest,
                self.development_p_table_digest,
            )
            if any(item is not None for item in forbidden):
                raise AcceptanceContractError("blocked fixture leaked post-engineering artifacts")
        else:
            if (
                self.admitted_training_run_count != 20
                or self.champion_count != 10
                or self.public_market_entry_count != 10
                or self.query_count != 4
            ):
                raise AcceptanceContractError("completed CPU chain has inconsistent counts")
            required = (
                self.admitted_records_digest,
                self.championization_digest,
                self.policy_market_id,
                self.raw_representation_index_id,
                self.corro_representation_index_id,
                self.selection_records_digest,
                self.private_oracle_results_digest,
                self.metrics_digest,
                self.cost_reconciliation_digest,
                self.development_p_table_digest,
            )
            if any(item is None for item in required):
                raise AcceptanceContractError("completed CPU chain is missing a digest binding")
            if self.status == "NO_GO_CORRO":
                if self.source_sigma_artifact_digest is not None:
                    raise AcceptanceContractError("CORRO failure cannot publish a sigma fallback")
            elif self.source_sigma_artifact_digest is None:
                raise AcceptanceContractError("successful CORRO fixture requires source-only sigma")
            if self.blocked_reason_digest is not None:
                raise AcceptanceContractError("non-blocked fixture cannot retain a blocked reason")

        expected_digest = sha256_json(self._payload_without_digest())
        if self.acceptance_report_digest is None:
            object.__setattr__(self, "acceptance_report_digest", expected_digest)
        elif _digest(self.acceptance_report_digest, "acceptance_report_digest") != expected_digest:
            raise AcceptanceContractError("acceptance_report_digest does not match report contents")

    @property
    def digest(self) -> str:
        assert self.acceptance_report_digest is not None
        return self.acceptance_report_digest

    def _payload_without_digest(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema": ACCEPTANCE_REPORT_SCHEMA,
                "scenario": self.scenario,
                "status": self.status,
                "stage": self.stage,
                "formal_task_requirement": self.formal_task_requirement,
                "formal_completion_claimed": self.formal_completion_claimed,
                "scope": {
                    "fixture_task_count": self.fixture_task_count,
                    "axes_per_task": self.axes_per_task,
                    "anchors_per_task": self.anchors_per_task,
                    "source_anchor_count": self.source_anchor_count,
                    "training_seed_count": self.training_seed_count,
                    "planned_training_run_count": self.planned_training_run_count,
                    "admitted_training_run_count": self.admitted_training_run_count,
                    "champion_count": self.champion_count,
                    "public_market_entry_count": self.public_market_entry_count,
                    "query_count": self.query_count,
                },
                "method_ids": list(self.method_ids),
                "gates": {key: value.to_dict() for key, value in self.gates.items()},
                "artifact_digests": {
                    "training_plan": self.training_plan_digest,
                    "admitted_records": self.admitted_records_digest,
                    "championization": self.championization_digest,
                    "policy_market": self.policy_market_id,
                    "raw_representation_index": self.raw_representation_index_id,
                    "corro_representation_index": self.corro_representation_index_id,
                    "source_sigma_artifact": self.source_sigma_artifact_digest,
                    "selection_records": self.selection_records_digest,
                    "private_oracle_results": self.private_oracle_results_digest,
                    "metrics": self.metrics_digest,
                    "cost_reconciliation": self.cost_reconciliation_digest,
                    "development_p_table": self.development_p_table_digest,
                    "blocked_reason": self.blocked_reason_digest,
                },
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "acceptance_report_digest": self.acceptance_report_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AcceptanceReport":
        expected = {
            "schema",
            "scenario",
            "status",
            "stage",
            "formal_task_requirement",
            "formal_completion_claimed",
            "scope",
            "method_ids",
            "gates",
            "artifact_digests",
            "acceptance_report_digest",
        }
        _strict_keys(value, expected, "AcceptanceReport")
        if value["schema"] != ACCEPTANCE_REPORT_SCHEMA:
            raise AcceptanceContractError("unsupported acceptance report schema")
        scope_keys = {
            "fixture_task_count",
            "axes_per_task",
            "anchors_per_task",
            "source_anchor_count",
            "training_seed_count",
            "planned_training_run_count",
            "admitted_training_run_count",
            "champion_count",
            "public_market_entry_count",
            "query_count",
        }
        digest_keys = {
            "training_plan",
            "admitted_records",
            "championization",
            "policy_market",
            "raw_representation_index",
            "corro_representation_index",
            "source_sigma_artifact",
            "selection_records",
            "private_oracle_results",
            "metrics",
            "cost_reconciliation",
            "development_p_table",
            "blocked_reason",
        }
        _strict_keys(value["scope"], scope_keys, "AcceptanceReport.scope")
        _strict_keys(value["artifact_digests"], digest_keys, "AcceptanceReport.artifact_digests")
        gates_value = value["gates"]
        if not isinstance(gates_value, Mapping):
            raise AcceptanceContractError("AcceptanceReport.gates must be a mapping")
        gates = {
            key: AcceptanceGateDecision.from_dict(item)
            for key, item in gates_value.items()
        }
        scope = value["scope"]
        digests = value["artifact_digests"]
        return cls(
            scenario=value["scenario"],
            status=value["status"],
            gates=gates,
            fixture_task_count=scope["fixture_task_count"],
            axes_per_task=scope["axes_per_task"],
            anchors_per_task=scope["anchors_per_task"],
            source_anchor_count=scope["source_anchor_count"],
            training_seed_count=scope["training_seed_count"],
            planned_training_run_count=scope["planned_training_run_count"],
            admitted_training_run_count=scope["admitted_training_run_count"],
            champion_count=scope["champion_count"],
            public_market_entry_count=scope["public_market_entry_count"],
            query_count=scope["query_count"],
            method_ids=tuple(value["method_ids"]),
            training_plan_digest=digests["training_plan"],
            admitted_records_digest=digests["admitted_records"],
            championization_digest=digests["championization"],
            policy_market_id=digests["policy_market"],
            raw_representation_index_id=digests["raw_representation_index"],
            corro_representation_index_id=digests["corro_representation_index"],
            source_sigma_artifact_digest=digests["source_sigma_artifact"],
            selection_records_digest=digests["selection_records"],
            private_oracle_results_digest=digests["private_oracle_results"],
            metrics_digest=digests["metrics"],
            cost_reconciliation_digest=digests["cost_reconciliation"],
            development_p_table_digest=digests["development_p_table"],
            blocked_reason_digest=digests["blocked_reason"],
            stage=value["stage"],
            formal_task_requirement=value["formal_task_requirement"],
            formal_completion_claimed=value["formal_completion_claimed"],
            acceptance_report_digest=value["acceptance_report_digest"],
        )


@dataclass(frozen=True)
class _SyntheticAnchor:
    task_id: str
    axis_id: str
    factor_id: str
    anchor_id: str
    environment_instance_digest: str
    anchor_manifest_digest: str
    coordinate: float
    nominal: bool


@dataclass(frozen=True)
class _SyntheticQuery:
    opaque_query_id: str
    task_id: str
    axis_id: str
    coordinate: float
    private_target_instance_digest: str
    target_evidence_digest: str
    probe_dataset_digest: str


def _training_plan_digest(jobs: Sequence[PolicyTrainingJob]) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-cpu-training-plan.v0",
            "jobs": [item.to_dict() for item in sorted(jobs, key=lambda row: row.job_id)],
        }
    )


def _admitted_digest(records: Mapping[str, AdmittedTrainingRecord]) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-cpu-admitted-records.v0",
            "records": {key: value.digest for key, value in sorted(records.items())},
        }
    )


def _championization_digest(result: ChampionizationResult) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-cpu-championization-result.v0",
            "selection_digest": result.selection_digest,
            "selected_by_anchor": dict(sorted(result.selected_by_anchor.items())),
            "competence_records": {
                key: value.to_dict() for key, value in sorted(result.competence_records.items())
            },
            "rejected_anchors": dict(sorted(result.rejected_anchors.items())),
        }
    )


def _selection_records_digest(
    records: Mapping[str, Mapping[str, SelectionRecord]],
) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-cpu-selection-record-index.v0",
            "queries": {
                query_id: {
                    method_id: record.digest
                    for method_id, record in sorted(by_method.items())
                }
                for query_id, by_method in sorted(records.items())
            },
        }
    )


def _oracle_results_digest(results: Mapping[str, FullPoolOracleResult]) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-cpu-private-oracle-index.v0",
            "visibility": "private-oracle-only",
            "queries": {key: value.digest for key, value in sorted(results.items())},
        }
    )


def _metrics_digest(metrics: Mapping[str, HierarchicalAggregate]) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-cpu-hierarchical-metrics.v0",
            "metric": "pool_regret",
            "methods": {key: value.to_dict() for key, value in sorted(metrics.items())},
        }
    )


@dataclass(frozen=True)
class CpuAcceptanceRun:
    """Typed fixture products; private oracle objects never enter report payloads."""

    report: AcceptanceReport
    training_jobs: tuple[PolicyTrainingJob, ...]
    admitted_records: Mapping[str, AdmittedTrainingRecord]
    championization: ChampionizationResult | None
    market: V02PolicyMarket | None
    raw_representation_index: RepresentationIndex | None
    corro_representation_index: RepresentationIndex | None
    selection_records: Mapping[str, Mapping[str, SelectionRecord]]
    oracle_results: Mapping[str, FullPoolOracleResult]
    metrics: Mapping[str, HierarchicalAggregate]
    cost_reconciliation: ColdWarmCostReconciliation | None
    development_p_table: DevelopmentPTable | None
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.report, AcceptanceReport):
            raise AcceptanceContractError("report must be an AcceptanceReport")
        jobs = tuple(sorted(self.training_jobs, key=lambda row: row.job_id))
        if any(not isinstance(item, PolicyTrainingJob) for item in jobs):
            raise AcceptanceContractError("training_jobs contain an invalid record")
        if _training_plan_digest(jobs) != self.report.training_plan_digest:
            raise AcceptanceContractError("report training plan digest is not bound to jobs")
        object.__setattr__(self, "training_jobs", jobs)

        admitted = dict(self.admitted_records)
        if any(not isinstance(item, AdmittedTrainingRecord) for item in admitted.values()):
            raise AcceptanceContractError("admitted_records contain an invalid record")
        object.__setattr__(self, "admitted_records", MappingProxyType(dict(sorted(admitted.items()))))
        nested = {
            query_id: MappingProxyType(dict(sorted(by_method.items())))
            for query_id, by_method in sorted(self.selection_records.items())
        }
        object.__setattr__(self, "selection_records", MappingProxyType(nested))
        oracle = dict(sorted(self.oracle_results.items()))
        metric_rows = dict(sorted(self.metrics.items()))
        object.__setattr__(self, "oracle_results", MappingProxyType(oracle))
        object.__setattr__(self, "metrics", MappingProxyType(metric_rows))

        if self.report.status == "BLOCKED_ENGINEERING":
            if admitted or nested or oracle or metric_rows:
                raise AcceptanceContractError("blocked run cannot retain scientific products")
            if self.blocked_reason is None:
                raise AcceptanceContractError("blocked run requires a reason")
            if _fixture_digest(f"blocked:{self.blocked_reason}") != self.report.blocked_reason_digest:
                raise AcceptanceContractError("blocked reason is not digest-bound")
            return

        if _admitted_digest(admitted) != self.report.admitted_records_digest:
            raise AcceptanceContractError("report admitted digest is not bound to records")
        if self.championization is None or (
            _championization_digest(self.championization) != self.report.championization_digest
        ):
            raise AcceptanceContractError("report championization digest is not bound")
        if self.market is None or self.market.policy_market_id != self.report.policy_market_id:
            raise AcceptanceContractError("report market ID is not bound")
        if (
            self.raw_representation_index is None
            or self.raw_representation_index.representation_index_id
            != self.report.raw_representation_index_id
        ):
            raise AcceptanceContractError("report raw representation index is not bound")
        if (
            self.corro_representation_index is None
            or self.corro_representation_index.representation_index_id
            != self.report.corro_representation_index_id
        ):
            raise AcceptanceContractError("report CORRO representation index is not bound")
        if _selection_records_digest(nested) != self.report.selection_records_digest:
            raise AcceptanceContractError("report selections are not digest-bound")
        if _oracle_results_digest(oracle) != self.report.private_oracle_results_digest:
            raise AcceptanceContractError("report private oracle index is not digest-bound")
        if _metrics_digest(metric_rows) != self.report.metrics_digest:
            raise AcceptanceContractError("report metrics are not digest-bound")
        if (
            self.cost_reconciliation is None
            or self.cost_reconciliation.digest != self.report.cost_reconciliation_digest
        ):
            raise AcceptanceContractError("report cost reconciliation is not bound")
        if (
            self.development_p_table is None
            or self.development_p_table.digest != self.report.development_p_table_digest
        ):
            raise AcceptanceContractError("report development P-table is not bound")


def _anchors() -> tuple[_SyntheticAnchor, ...]:
    result: list[_SyntheticAnchor] = []
    for task_index, (task_id, base) in enumerate(
        (("cpu-task-alpha", 0.0), ("cpu-task-beta", 10.0))
    ):
        definitions = (
            ("shared-nominal", "nominal", base, True),
            ("axis-x", "low", base - 2.0, False),
            ("axis-x", "high", base + 2.0, False),
            ("axis-y", "low", base - 1.0, False),
            ("axis-y", "high", base + 1.0, False),
        )
        for axis_id, factor_id, coordinate, nominal in definitions:
            label = f"task:{task_index}:{axis_id}:{factor_id}"
            result.append(
                _SyntheticAnchor(
                    task_id=task_id,
                    axis_id=axis_id,
                    factor_id=factor_id,
                    anchor_id=_fixture_digest(f"anchor:{label}"),
                    environment_instance_digest=_fixture_digest(f"environment:{label}"),
                    anchor_manifest_digest=_fixture_digest(f"manifest:{label}"),
                    coordinate=coordinate,
                    nominal=nominal,
                )
            )
    by_task: dict[str, list[_SyntheticAnchor]] = {}
    for anchor in result:
        by_task.setdefault(anchor.task_id, []).append(anchor)
    if (
        len(by_task) != 2
        or any(len(rows) != 5 for rows in by_task.values())
        or any(sum(row.nominal for row in rows) != 1 for rows in by_task.values())
        or any({row.axis_id for row in rows if not row.nominal} != {"axis-x", "axis-y"} for rows in by_task.values())
    ):
        raise AssertionError("internal shared-nominal anchor fixture is malformed")
    return tuple(sorted(result, key=lambda item: item.anchor_id))


def _queries() -> tuple[_SyntheticQuery, ...]:
    definitions = (
        ("cpu-task-alpha", "axis-x", -1.8),
        ("cpu-task-alpha", "axis-y", 0.8),
        ("cpu-task-beta", "axis-x", 8.2),
        ("cpu-task-beta", "axis-y", 10.8),
    )
    rows: list[_SyntheticQuery] = []
    for index, (task_id, axis_id, coordinate) in enumerate(definitions):
        payload = f"query:{index}:{task_id}:{axis_id}:{coordinate}"
        rows.append(
            _SyntheticQuery(
                opaque_query_id="v02q-" + _fixture_digest(payload)[:32],
                task_id=task_id,
                axis_id=axis_id,
                coordinate=coordinate,
                private_target_instance_digest=_fixture_digest(f"target-instance:{payload}"),
                target_evidence_digest=_fixture_digest(f"target-evidence:{payload}"),
                probe_dataset_digest=_fixture_digest(f"target-probe:{payload}"),
            )
        )
    return tuple(rows)


def _attestation(
    job: PolicyTrainingJob,
    *,
    actual_train_environment_instance_digest: str | None = None,
) -> PolicyTrainingAttestation:
    bundle = _fixture_digest(f"bundle:{job.job_id}")
    return PolicyTrainingAttestation(
        job_id=job.job_id,
        job_digest=job.digest,
        attempt_id=f"{job.job_id}-attempt-1",
        attempt_number=1,
        source_anchor_id=job.source_anchor_id,
        anchor_manifest_digest=job.anchor_manifest_digest,
        declared_environment_instance_digest=job.environment_instance_digest,
        actual_train_environment_instance_digest=(
            job.environment_instance_digest
            if actual_train_environment_instance_digest is None
            else actual_train_environment_instance_digest
        ),
        actual_eval_environment_instance_digest=job.environment_instance_digest,
        operator_digest=_fixture_digest(f"operator:{job.source_anchor_id}"),
        model_diff_digest=_fixture_digest(f"model-diff:{job.source_anchor_id}"),
        algorithm=job.algorithm,
        seed=job.seed,
        environment_steps=job.environment_steps,
        checkpoint_rule=job.checkpoint_rule,
        checkpoint_digests={"selected": _fixture_digest(f"checkpoint:{job.job_id}")},
        bundle_digest=bundle,
        bundle_manifest_digest=_fixture_digest(f"bundle-manifest:{job.job_id}"),
        golden_parity_digest=_fixture_digest(f"golden-parity:{job.job_id}"),
        compiled_parity_digest=_fixture_digest(f"compiled-parity:{job.job_id}"),
        finiteness_audit_digest=_fixture_digest(f"finiteness:{job.job_id}"),
        all_arrays_finite=True,
        golden_parity_passed=True,
        compiled_parity_passed=True,
        trainer_commit=job.trainer_commit,
        dependency_digest=job.dependency_digest,
        runtime_digest=job.runtime_digest,
        hardware_digest=_fixture_digest("cpu-hardware"),
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        elapsed_seconds=1.0,
        status="succeeded",
        bundle_path=f"/private/cpu-v02-fixture/{bundle}.bundle",
    )


def _execution_abi() -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=_fixture_digest("observation-tensor-abi"),
        action_tensor_abi_digest=_fixture_digest("action-tensor-abi"),
        action_transform_id="identity-action-transform-v02",
        policy_runtime_id="cpu-fixture-runtime-v02",
        state_abi_id="stateless-policy-v02",
    )


def _environment_spec(
    coordinate: float,
    *,
    representation_protocol_id: str,
    measurement_protocol_id: str,
    canonical_view_digest: str,
    probe_dataset_digest: str,
) -> EnvironmentSpec:
    if not math.isfinite(coordinate):
        raise AcceptanceContractError("synthetic EnvironmentSpec coordinate is non-finite")
    return EnvironmentSpec(
        supports=np.asarray([[coordinate]], dtype=np.float64),
        beta=np.asarray([1.0], dtype=np.float64),
        empirical_norm2=1.0,
        rkme_norm2=1.0,
        reconstruction_error=0.0,
        reducer_digest=_fixture_digest(f"reducer:{representation_protocol_id}"),
        support_budget=1,
        latent_dim=1,
        representation_protocol_id=representation_protocol_id,
        measurement_protocol_id=measurement_protocol_id,
        canonical_view_digest=canonical_view_digest,
        kernel_bandwidth=1.0,
        probe_dataset_digest=probe_dataset_digest,
    )


def _value_for_coordinate(
    *,
    source_coordinate: float,
    target_coordinate: float,
    scenario: AcceptanceScenario,
    opaque_id: str,
    global_champion_id: str,
) -> float:
    if scenario == "no_go_market":
        return 0.95 if opaque_id == global_champion_id else 0.50
    return max(0.05, 1.0 - 0.12 * abs(source_coordinate - target_coordinate))


def _cost_records(queries: Sequence[_SyntheticQuery]) -> tuple[tuple[CostRecord, ...], Mapping[str, str]]:
    contract = _fixture_digest("cpu-query-cost-contract")
    records: list[CostRecord] = []
    cold_digest_by_query: dict[str, str] = {}
    for index, query in enumerate(sorted(queries, key=lambda item: item.opaque_query_id)):
        for mode, multiplier in (("cold", 1.0), ("warm", 0.5)):
            components = {
                component: multiplier * (index + 1) * (position + 1) / 1000.0
                for position, component in enumerate(QUERY_COST_COMPONENTS)
            }
            record = CostRecord.create(
                query_id=query.opaque_query_id,
                mode=mode,
                cost_contract_digest=contract,
                execution_attempt_id=f"{query.opaque_query_id}:{mode}:attempt-1",
                components_seconds=components,
                target_environment_steps=32,
                target_gradient_steps=0,
            )
            records.append(record)
            if mode == "cold":
                cold_digest_by_query[query.opaque_query_id] = record.digest
    return tuple(records), MappingProxyType(cold_digest_by_query)


def _blocked_run(
    *, scenario: AcceptanceScenario, jobs: Sequence[PolicyTrainingJob]
) -> CpuAcceptanceRun:
    try:
        _attestation(
            jobs[0],
            actual_train_environment_instance_digest=_fixture_digest("poison-environment"),
        )
    except ValueError as error:
        reason = str(error)
    else:  # pragma: no cover - defensive assertion around a strict core schema
        raise AssertionError("poisoned successful attestation unexpectedly passed")
    gates = {
        "CPU-Engineering": AcceptanceGateDecision(
            "CPU-Engineering",
            {
                "training_plan_has_20_runs": len(jobs) == 20,
                "attestation_environment_identity": False,
                "all_attestations_admitted": False,
                "ten_anchor_champions": False,
                "anonymous_market_has_10_entries": False,
            },
        ),
        "CPU-Market": AcceptanceGateDecision(
            "CPU-Market", {"engineering_prerequisite": False, "development_selection_value": False}
        ),
        "CPU-CORRO": AcceptanceGateDecision(
            "CPU-CORRO", {"engineering_prerequisite": False, "raw_signal": False, "corro_signal": False}
        ),
        "CPU-Scientific": AcceptanceGateDecision(
            "CPU-Scientific",
            {
                "engineering_prerequisite": False,
                "private_oracle_complete": False,
                "metrics_complete": False,
                "cost_reconciled": False,
                "development_report_complete": False,
            },
        ),
    }
    report = AcceptanceReport(
        scenario=scenario,
        status="BLOCKED_ENGINEERING",
        gates=gates,
        fixture_task_count=2,
        axes_per_task=2,
        anchors_per_task=5,
        source_anchor_count=10,
        training_seed_count=2,
        planned_training_run_count=20,
        admitted_training_run_count=0,
        champion_count=0,
        public_market_entry_count=0,
        query_count=0,
        method_ids=(),
        training_plan_digest=_training_plan_digest(jobs),
        admitted_records_digest=None,
        championization_digest=None,
        policy_market_id=None,
        raw_representation_index_id=None,
        corro_representation_index_id=None,
        source_sigma_artifact_digest=None,
        selection_records_digest=None,
        private_oracle_results_digest=None,
        metrics_digest=None,
        cost_reconciliation_digest=None,
        development_p_table_digest=None,
        blocked_reason_digest=_fixture_digest(f"blocked:{reason}"),
    )
    return CpuAcceptanceRun(
        report=report,
        training_jobs=tuple(jobs),
        admitted_records={},
        championization=None,
        market=None,
        raw_representation_index=None,
        corro_representation_index=None,
        selection_records={},
        oracle_results={},
        metrics={},
        cost_reconciliation=None,
        development_p_table=None,
        blocked_reason=reason,
    )


def run_cpu_acceptance_fixture(
    scenario: AcceptanceScenario = "scientific_pass",
    *,
    shuffle_inputs: bool = False,
) -> CpuAcceptanceRun:
    """Run the 20-training-run v0.2 CPU fixture through the real sidecar APIs.

    ``shuffle_inputs`` reverses every shard-like input before it reaches a
    canonical boundary.  It is deliberately absent from the report payload, so
    replaying the same evidence in another order must produce the same digest.
    """

    if scenario not in ACCEPTANCE_SCENARIOS:
        raise AcceptanceContractError(f"unsupported acceptance scenario: {scenario!r}")
    if type(shuffle_inputs) is not bool:
        raise AcceptanceContractError("shuffle_inputs must be boolean")

    anchors = _anchors()
    anchor_rows = tuple(reversed(anchors)) if shuffle_inputs else anchors
    plan_input = {
        anchor.anchor_id: {
            "environment_instance_digest": anchor.environment_instance_digest,
            "anchor_manifest_digest": anchor.anchor_manifest_digest,
        }
        for anchor in anchor_rows
    }
    jobs = plan_training_jobs(
        plan_input,
        config_digest=_fixture_digest("cpu-acceptance-config"),
        execution_purpose="audit_smoke",
        algorithm="ppo",
        seeds=(101, 202),
        environment_steps=100,
        checkpoint_rule="best-source-selection-mean",
        trainer_config={"hidden_sizes": [16, 16], "learning_rate": 3.0e-4},
        trainer_commit="a" * 40,
        dependency_digest=_fixture_digest("training-dependencies"),
        runtime_digest=_fixture_digest("training-runtime"),
        training_protocol_id=_fixture_digest("training-protocol"),
    )
    if len(jobs) != 20:
        raise AssertionError("CPU fixture must plan exactly 20 training runs")
    jobs_input = tuple(reversed(jobs)) if shuffle_inputs else jobs
    if scenario == "engineering_blocked":
        return _blocked_run(scenario=scenario, jobs=jobs_input)

    admitted: dict[str, AdmittedTrainingRecord] = {}
    attestations: dict[str, PolicyTrainingAttestation] = {}
    for job in jobs_input:
        attestation = _attestation(job)
        admitted[job.job_id] = AdmittedTrainingRecord(job=job, attestation=attestation)
        attestations[job.job_id] = attestation

    selection_rows: list[SourceEpisodeRow] = []
    attestation_rows: list[SourceEpisodeRow] = []
    sorted_anchor_ids = tuple(sorted(anchor.anchor_id for anchor in anchors))
    anchor_rank = {anchor_id: index for index, anchor_id in enumerate(sorted_anchor_ids)}
    for job in jobs_input:
        selected_seed = job.seed == 101
        center = 0.91 if selected_seed else 0.71
        for offset, reset_seed in ((-0.01, 1001), (0.01, 1002)):
            selection_rows.append(
                SourceEpisodeRow(
                    source_anchor_id=job.source_anchor_id,
                    candidate_id=job.job_id,
                    bundle_digest=attestations[job.job_id].bundle_digest,
                    block="source_selection",
                    reset_seed=reset_seed,
                    normalized_return=center + offset,
                )
            )
        if selected_seed:
            competence_center = 0.82 + 0.005 * anchor_rank[job.source_anchor_id]
            for offset, reset_seed in ((-0.005, 2001), (0.005, 2002)):
                attestation_rows.append(
                    SourceEpisodeRow(
                        source_anchor_id=job.source_anchor_id,
                        candidate_id=job.job_id,
                        bundle_digest=attestations[job.job_id].bundle_digest,
                        block="source_attestation",
                        reset_seed=reset_seed,
                        normalized_return=competence_center + offset,
                    )
                )
    if shuffle_inputs:
        selection_rows.reverse()
        attestation_rows.reverse()
    selection_rows.sort(
        key=lambda row: (row.source_anchor_id, row.candidate_id, row.reset_seed)
    )
    attestation_rows.sort(
        key=lambda row: (row.source_anchor_id, row.candidate_id, row.reset_seed)
    )
    championization = championize_by_anchor(
        selection_rows,
        attestation_rows,
        competence_floors={anchor.anchor_id: 0.75 for anchor in anchors},
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id=_fixture_digest("dmc-normalized-return-contract"),
    )
    abi = _execution_abi()
    execution_abis = {
        candidate: abi for candidate in championization.selected_by_anchor.values()
    }
    market = build_policy_market(
        admitted,
        championization,
        execution_abis,
        expected_anchor_count=10,
        market_alias_nonce=_fixture_digest("market-alias-nonce"),
        tie_break_nonce=_fixture_digest("market-tie-break-nonce"),
    )

    raw_protocol = _fixture_digest("raw-representation-protocol")
    corro_protocol = _fixture_digest("corro-fake-representation-protocol")
    measurement_protocol = _fixture_digest("candidate-independent-probe-measurement")
    raw_view_digest = _fixture_digest("raw-canonical-view")
    corro_view_digest = _fixture_digest("corro-canonical-view")
    coordinate_by_opaque = {
        market.anchor_to_opaque_id[anchor.anchor_id]: anchor.coordinate for anchor in anchors
    }
    raw_entries = {
        opaque_id: RepresentationIndexEntry(
            opaque_id,
            _environment_spec(
                coordinate_by_opaque[opaque_id],
                representation_protocol_id=raw_protocol,
                measurement_protocol_id=measurement_protocol,
                canonical_view_digest=raw_view_digest,
                probe_dataset_digest=_fixture_digest(f"source-probe:{opaque_id}"),
            ),
        )
        for opaque_id in sorted(market.entries)
    }
    raw_index = RepresentationIndex(
        policy_market_id=market.policy_market_id,
        representation_protocol_id=raw_protocol,
        entries=raw_entries,
    )
    corro_entries = {
        opaque_id: RepresentationIndexEntry(
            opaque_id,
            _environment_spec(
                0.0 if scenario == "no_go_corro" else 0.5 * coordinate_by_opaque[opaque_id],
                representation_protocol_id=corro_protocol,
                measurement_protocol_id=measurement_protocol,
                canonical_view_digest=corro_view_digest,
                probe_dataset_digest=_fixture_digest(f"source-probe:{opaque_id}"),
            ),
        )
        for opaque_id in sorted(market.entries)
    }
    corro_index = RepresentationIndex(
        policy_market_id=market.policy_market_id,
        representation_protocol_id=corro_protocol,
        entries=corro_entries,
    )
    raw_market = PublicMarketView(market.policy_market_id, market.entries, raw_index)
    corro_market = PublicMarketView(market.policy_market_id, market.entries, corro_index)

    source_sigma = None
    try:
        source_sigma = derive_source_only_sigma_artifacts(
            corro_index,
            partitions={"anonymous-global": tuple(sorted(market.entries))},
            distance_form="mmd",
        )["anonymous-global"]
    except ValueError:
        if scenario != "no_go_corro":
            raise
    if scenario == "no_go_corro" and source_sigma is not None:
        raise AssertionError("collapsed CORRO fixture unexpectedly produced source signal")

    feature_protocol = _fixture_digest("raw-trace-feature-protocol")
    feature_index = FrozenFeatureIndex(
        policy_market_id=market.policy_market_id,
        feature_protocol_id=feature_protocol,
        entries={
            opaque_id: TraceFeatureVector(
                values=np.asarray([coordinate_by_opaque[opaque_id]], dtype=np.float64),
                feature_protocol_id=feature_protocol,
                probe_dataset_digest=_fixture_digest(f"source-feature-probe:{opaque_id}"),
            )
            for opaque_id in sorted(market.entries)
        },
    )
    nominal_anchors = tuple(sorted((item for item in anchors if item.nominal), key=lambda item: item.task_id))
    legacy_specs = {
        f"source-task-spec-{index}": TraceFeatureVector(
            values=np.asarray([anchor.coordinate], dtype=np.float64),
            feature_protocol_id=feature_protocol,
            probe_dataset_digest=_fixture_digest(f"legacy-task-spec:{index}"),
        )
        for index, anchor in enumerate(nominal_anchors)
    }
    nominal_champions = {
        f"source-task-spec-{index}": market.anchor_to_opaque_id[anchor.anchor_id]
        for index, anchor in enumerate(nominal_anchors)
    }

    global_champion_id = min(
        market.entries,
        key=lambda opaque_id: (
            -market.entries[opaque_id].normalized_source_competence,
            market.entries[opaque_id].tie_break_token,
        ),
    )
    policy_ids = tuple(sorted(market.entries))
    development_coordinates = (-2.0, 0.0, 2.0, 8.0, 10.0, 12.0)
    development_context_ids = tuple(f"cpu-development-context-{index}" for index in range(6))
    development_returns = np.asarray(
        [
            [
                _value_for_coordinate(
                    source_coordinate=coordinate_by_opaque[opaque_id],
                    target_coordinate=coordinate,
                    scenario=scenario,
                    opaque_id=opaque_id,
                    global_champion_id=global_champion_id,
                )
                for opaque_id in policy_ids
            ]
            for coordinate in development_coordinates
        ],
        dtype=np.float64,
    )
    development_view = DevelopmentView(
        context_ids=development_context_ids,
        opaque_policy_ids=policy_ids,
        context_features=np.asarray(development_coordinates, dtype=np.float64)[:, None],
        normalized_returns=development_returns,
        training_context_ids=development_context_ids[:4],
        validation_context_ids=development_context_ids[4:],
        evaluation_seed_digests=tuple(
            _fixture_digest(f"development-evaluation-seeds:{context_id}")
            for context_id in development_context_ids
        ),
        policy_market_id=market.policy_market_id,
        feature_protocol_id=feature_protocol,
        split_manifest_digest=_fixture_digest("development-supervision-split"),
        label_contract_digest=_fixture_digest("development-oracle-label-contract"),
        candidate_paired_seeds=True,
    )

    probe_evidence = target_probe_evidence_contract(
        reads_development_policy_returns=False, reads_probe_rewards=False
    )
    legacy_evidence = target_probe_evidence_contract(
        reads_development_policy_returns=False,
        reads_probe_rewards=False,
        reads_source_side_labels=True,
    )
    supervised_evidence = target_probe_evidence_contract(
        reads_development_policy_returns=True, reads_probe_rewards=False
    )
    selectors: dict[str, Any] = {
        "B0": RandomAnonymousMarketSelector(
            method_id="B0",
            selector_seed=17,
            policy_market_id=market.policy_market_id,
        ),
        "B1": CompetenceOnlySelector(
            method_id="B1", policy_market_id=market.policy_market_id
        ),
        "B2": LegacyTaskSpecSelector(
            method_id="B2",
            source_task_specs=legacy_specs,
            nominal_champions=nominal_champions,
            policy_market_id=market.policy_market_id,
            evidence_contract=legacy_evidence,
        ),
        "B3a": VectorNearestSelector(
            method_id="B3a", feature_index=feature_index, evidence_contract=probe_evidence
        ),
        "B3b": EnvironmentOnlySelector(
            method_id="B3b",
            distance_form="mmd",
            policy_market_id=market.policy_market_id,
            representation_index_id=str(raw_index.representation_index_id),
            evidence_contract=probe_evidence,
        ),
        "B4a": KnnDevelopmentSelector(
            method_id="B4a",
            neighbor_count=1,
            policy_market_id=market.policy_market_id,
            evidence_contract=supervised_evidence,
        ),
        "B4b": LinearDevelopmentSelector(
            method_id="B4b",
            ridge=0.1,
            policy_market_id=market.policy_market_id,
            evidence_contract=supervised_evidence,
        ),
        "A-Env": EnvironmentOnlySelector(
            method_id="A-Env",
            distance_form="mmd",
            policy_market_id=market.policy_market_id,
            representation_index_id=str(corro_index.representation_index_id),
            evidence_contract=probe_evidence,
        ),
    }
    if source_sigma is not None:
        selectors["M02/B5"] = SourceOnlyLMinSelector(
            method_id="M02/B5",
            sigma_artifact=source_sigma,
            epsilon=0.05,
            evidence_contract=probe_evidence,
        )
    artifacts = {
        method_id: selector.fit(
            development_view if method_id in {"B4a", "B4b"} else None
        )
        for method_id, selector in sorted(selectors.items())
    }

    queries = _queries()
    cost_rows, cold_cost_digest = _cost_records(queries)
    if shuffle_inputs:
        cost_rows = tuple(reversed(cost_rows))
    cost_reconciliation = reconcile_cold_warm_costs(
        cost_rows, expected_query_ids=tuple(query.opaque_query_id for query in queries)
    )
    selections: dict[str, dict[str, SelectionRecord]] = {}
    raw_query_views: dict[str, TargetQueryView] = {}
    corro_query_views: dict[str, TargetQueryView] = {}
    for query in (tuple(reversed(queries)) if shuffle_inputs else queries):
        trace = TraceFeatureVector(
            values=np.asarray([query.coordinate], dtype=np.float64),
            feature_protocol_id=feature_protocol,
            probe_dataset_digest=query.probe_dataset_digest,
        )
        raw_query = TargetQueryView(
            stage="development_discovery",
            query_spec=_environment_spec(
                query.coordinate,
                representation_protocol_id=raw_protocol,
                measurement_protocol_id=measurement_protocol,
                canonical_view_digest=raw_view_digest,
                probe_dataset_digest=query.probe_dataset_digest,
            ),
            target_evidence_digest=query.target_evidence_digest,
            cost_digest=cold_cost_digest[query.opaque_query_id],
            probe_rewards_included=False,
            trace_feature=trace,
        )
        corro_query = TargetQueryView(
            stage="development_discovery",
            query_spec=_environment_spec(
                0.0 if scenario == "no_go_corro" else 0.5 * query.coordinate,
                representation_protocol_id=corro_protocol,
                measurement_protocol_id=measurement_protocol,
                canonical_view_digest=corro_view_digest,
                probe_dataset_digest=query.probe_dataset_digest,
            ),
            target_evidence_digest=query.target_evidence_digest,
            cost_digest=cold_cost_digest[query.opaque_query_id],
            probe_rewards_included=False,
            trace_feature=trace,
        )
        raw_query_views[query.opaque_query_id] = raw_query
        corro_query_views[query.opaque_query_id] = corro_query
        by_method: dict[str, SelectionRecord] = {}
        for method_id, selector in sorted(selectors.items()):
            use_corro = method_id in {"A-Env", "M02/B5"}
            by_method[method_id] = selector.select(
                corro_query if use_corro else raw_query,
                corro_market if use_corro else raw_market,
                artifacts[method_id],
            )
        selections[query.opaque_query_id] = by_method

    evaluation_protocol = _fixture_digest("cpu-private-oracle-evaluation-protocol")
    oracle_results: dict[str, FullPoolOracleResult] = {}
    query_by_id = {query.opaque_query_id: query for query in queries}
    for query_id in (tuple(reversed(sorted(query_by_id))) if shuffle_inputs else tuple(sorted(query_by_id))):
        query = query_by_id[query_id]
        episode_rows: list[OracleEpisodeRow] = []
        for opaque_id in (tuple(reversed(policy_ids)) if shuffle_inputs else policy_ids):
            value = _value_for_coordinate(
                source_coordinate=coordinate_by_opaque[opaque_id],
                target_coordinate=query.coordinate,
                scenario=scenario,
                opaque_id=opaque_id,
                global_champion_id=global_champion_id,
            )
            for episode_index, offset in ((0, -0.005), (1, 0.005)):
                normalized = min(1.0, max(0.0, value + offset))
                episode_rows.append(
                    OracleEpisodeRow(
                        opaque_query_id=query_id,
                        opaque_learnware_id=opaque_id,
                        episode_index=episode_index,
                        reset_seed=3000 + episode_index,
                        policy_seed=4000 + episode_index,
                        steps=100,
                        raw_return=100.0 * normalized,
                        normalized_return=normalized,
                        terminated=True,
                        truncated=False,
                        runtime_seconds=0.01,
                        private_target_instance_digest=query.private_target_instance_digest,
                        bundle_digest=market.deployment_private[opaque_id].bundle_digest,
                        seed_contract_digest=_fixture_digest(
                            f"oracle-paired-seed:{query_id}:{episode_index}"
                        ),
                        evaluation_protocol_id=evaluation_protocol,
                    )
                )
        if shuffle_inputs:
            episode_rows.reverse()
        episode_rows.sort(key=lambda row: (row.opaque_learnware_id, row.episode_index))
        published = [
            PublishedSelection.from_selection_record(record)
            for _, record in sorted(selections[query_id].items())
        ]
        if shuffle_inputs:
            published.reverse()
        oracle_results[query_id] = aggregate_full_pool_oracle(
            opaque_query_id=query_id,
            private_target_instance_digest=query.private_target_instance_digest,
            evaluation_protocol_id=evaluation_protocol,
            market_ids=policy_ids,
            deployment_registry=market.deployment_private,
            target_execution_abi=abi,
            episode_rows=episode_rows,
            published_selections=published,
            failure_floor=0.0,
            tie_atol=1.0e-12,
            candidate_paired_seeds=True,
        )

    p_rows: list[DevelopmentPTableRow] = []
    metric_leaves: dict[str, list[HierarchicalValue]] = {
        method_id: [] for method_id in sorted(selectors)
    }
    for query_id, oracle_result in sorted(oracle_results.items()):
        query = query_by_id[query_id]
        for method_id, outcome in sorted(oracle_result.outcomes.items()):
            record = selections[query_id][method_id]
            representation_id = (
                str(corro_index.representation_index_id)
                if method_id in {"A-Env", "M02/B5"}
                else str(raw_index.representation_index_id)
            )
            p_rows.append(
                DevelopmentPTableRow.from_oracle_outcome(
                    opaque_query_id=query_id,
                    bank_index=0,
                    prefix=32,
                    representation_id=representation_id,
                    outcome=outcome,
                    epsilon=0.05,
                    target_transition_count=32,
                    target_evidence_digest=record.target_evidence_digest,
                    evidence_contract_digest=sha256_json(record.evidence_contract.to_dict()),
                    cost_digest=record.cost_digest,
                )
            )
            metric_leaves[method_id].append(
                HierarchicalValue(
                    task_id=query.task_id,
                    axis_id=query.axis_id,
                    context_id=query_id,
                    observation_id=f"{query_id}:bank-0",
                    value=outcome.regret,
                )
            )
    if shuffle_inputs:
        p_rows.reverse()
        for rows in metric_leaves.values():
            rows.reverse()
    development_p_table = build_development_p_table(
        p_rows,
        development_split_digest=_fixture_digest("cpu-acceptance-development-split"),
        policy_market_id=market.policy_market_id,
        evaluation_protocol_id=evaluation_protocol,
    )
    metrics = {
        method_id: aggregate_hierarchy(rows)
        for method_id, rows in sorted(metric_leaves.items())
    }

    raw_signal = len({coordinate_by_opaque[item] for item in coordinate_by_opaque}) > 1
    corro_signal = source_sigma is not None
    market_value = metrics["B3b"].macro_mean + 0.05 < metrics["B1"].macro_mean
    engineering_checks = {
        "training_plan_has_20_runs": len(jobs) == 20,
        "attestation_environment_identity": True,
        "all_attestations_admitted": len(admitted) == 20,
        "ten_anchor_champions": len(championization.competence_records) == 10,
        "anonymous_market_has_10_entries": len(market.entries) == 10,
    }
    market_checks = {
        "engineering_prerequisite": all(engineering_checks.values()),
        "all_specialists_above_absolute_floor": not championization.rejected_anchors,
        "development_selection_value": market_value,
        "two_task_two_axis_coverage": len({item.task_id for item in queries}) == 2
        and len({(item.task_id, item.axis_id) for item in queries}) == 4,
    }
    corro_checks = {
        "engineering_prerequisite": all(engineering_checks.values()),
        "raw_signal": raw_signal,
        "corro_signal": corro_signal,
        "no_zero_distance_sigma_fallback": source_sigma is not None
        or scenario == "no_go_corro",
    }
    expected_methods = CORRO_FAIL_METHOD_IDS if scenario == "no_go_corro" else FULL_METHOD_IDS
    oracle_complete = (
        len(oracle_results) == 4
        and all(set(item.outcomes) == set(expected_methods) for item in oracle_results.values())
    )
    scientific_checks = {
        "engineering_prerequisite": all(engineering_checks.values()),
        "market_prerequisite": all(market_checks.values()),
        "corro_prerequisite": all(corro_checks.values()),
        "m02_better_than_competence_only": source_sigma is not None
        and metrics["M02/B5"].macro_mean + 0.05 < metrics["B1"].macro_mean,
        "private_oracle_complete": oracle_complete,
        "metrics_complete": set(metrics) == set(expected_methods),
        "cost_reconciled": cost_reconciliation.query_ids
        == tuple(sorted(query.opaque_query_id for query in queries)),
        "development_report_complete": len(development_p_table.rows)
        == len(queries) * len(expected_methods),
    }
    gates = {
        "CPU-Engineering": AcceptanceGateDecision("CPU-Engineering", engineering_checks),
        "CPU-Market": AcceptanceGateDecision("CPU-Market", market_checks),
        "CPU-CORRO": AcceptanceGateDecision("CPU-CORRO", corro_checks),
        "CPU-Scientific": AcceptanceGateDecision("CPU-Scientific", scientific_checks),
    }
    status = _derive_status(gates)
    report = AcceptanceReport(
        scenario=scenario,
        status=status,
        gates=gates,
        fixture_task_count=2,
        axes_per_task=2,
        anchors_per_task=5,
        source_anchor_count=10,
        training_seed_count=2,
        planned_training_run_count=20,
        admitted_training_run_count=len(admitted),
        champion_count=len(championization.competence_records),
        public_market_entry_count=len(market.entries),
        query_count=len(queries),
        method_ids=tuple(sorted(selectors)),
        training_plan_digest=_training_plan_digest(jobs),
        admitted_records_digest=_admitted_digest(admitted),
        championization_digest=_championization_digest(championization),
        policy_market_id=market.policy_market_id,
        raw_representation_index_id=str(raw_index.representation_index_id),
        corro_representation_index_id=str(corro_index.representation_index_id),
        source_sigma_artifact_digest=(
            None if source_sigma is None else str(source_sigma.artifact_digest)
        ),
        selection_records_digest=_selection_records_digest(selections),
        private_oracle_results_digest=_oracle_results_digest(oracle_results),
        metrics_digest=_metrics_digest(metrics),
        cost_reconciliation_digest=cost_reconciliation.digest,
        development_p_table_digest=development_p_table.digest,
        blocked_reason_digest=None,
    )
    return CpuAcceptanceRun(
        report=report,
        training_jobs=tuple(jobs),
        admitted_records=admitted,
        championization=championization,
        market=market,
        raw_representation_index=raw_index,
        corro_representation_index=corro_index,
        selection_records=selections,
        oracle_results=oracle_results,
        metrics=metrics,
        cost_reconciliation=cost_reconciliation,
        development_p_table=development_p_table,
    )


def run_cpu_acceptance(
    scenario: AcceptanceScenario = "scientific_pass", *, shuffle_inputs: bool = False
) -> CpuAcceptanceRun:
    """Backward-friendly short alias for :func:`run_cpu_acceptance_fixture`."""

    return run_cpu_acceptance_fixture(scenario, shuffle_inputs=shuffle_inputs)


__all__ = [
    "ACCEPTANCE_GATE_IDS",
    "ACCEPTANCE_REPORT_SCHEMA",
    "AcceptanceContractError",
    "AcceptanceGateDecision",
    "AcceptanceReport",
    "AcceptanceScenario",
    "AcceptanceStatus",
    "CpuAcceptanceRun",
    "FULL_METHOD_IDS",
    "run_cpu_acceptance",
    "run_cpu_acceptance_fixture",
]
