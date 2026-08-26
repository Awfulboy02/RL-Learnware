"""Fail-closed v0.3 pre-experiment freeze and recovery contracts.

These records deliberately stop one step before scientific execution.  They
bind the five hard P4 prerequisites, the frozen execution plans and the public
ranking barrier, but they cannot mint review authority or read/write the
confirmatory oracle.  Oracle release remains owned by the joint Paper-I
orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from ..hashing import sha256_json
from .baselines import (
    FORMAL_DEVELOPMENT_CONTEXT_COUNT,
    FORMAL_MODE,
    REQUIRED_BASELINE_METHOD_IDS,
)
from .schemas import checked_digest, checked_safe_id, strict_mapping


HARD_TODO_EVIDENCE_SCHEMA = "policy-learnware.v03-hard-todo-evidence.v0"
PRE_EXPERIMENT_FREEZE_SCHEMA = "policy-learnware.v03-pre-experiment-freeze.v11"
EXECUTION_CHECKPOINT_SCHEMA = "policy-learnware.v03-execution-checkpoint.v0"
PUBLIC_RANKING_BARRIER_SCHEMA = "policy-learnware.v03-public-ranking-barrier.v3"
PUBLIC_RANKING_PUBLICATION_SCHEMA = (
    "policy-learnware.v03-public-ranking-publication.v0"
)
PUBLIC_QUERY_PLAN_SCHEMA = "policy-learnware.v03-public-query-plan.v0"
ORACLE_UNLOCK_HANDOFF_SCHEMA = "policy-learnware.v03-oracle-unlock-handoff.v1"
INDEPENDENT_RECOMPUTE_SCHEMA = "policy-learnware.v03-independent-recompute.v1"

HARD_TODO_IDS = (
    "T-P4-01",
    "T-P4-02",
    "T-P4-03",
    "T-P4-04",
    "T-P4-05",
)
ORACLE_OWNER = "policy-learnware-paper1"
_OPAQUE_QUERY_ID = re.compile(r"^v03q-[0-9a-f]{32}$")
FORMAL_QUERY_REGIME_COUNTS = MappingProxyType(
    {"EXACT": 30, "INTERPOLATION": 24, "EXTRAPOLATION": 12}
)
FORMAL_PRODUCTION_STAGE_IDS = (
    "collect-source-receipts",
    "build-market",
    "build-canonical-banks",
    "build-transition-views",
    "replay-legacy-attribution",
    "fit-representation-controls",
    "build-signal-atlas",
    "build-source-specs",
    "build-query-specs",
    "fit-baselines",
    "run-public-rankings",
)
FORMAL_GATE_PLAN_IDS = (
    "G03-Attribution",
    "G03-Probe",
    "G03-Market",
)
FORMAL_STAGE_ADAPTER_BINDING_SCHEMA = (
    "policy-learnware.v03-formal-stage-adapter-binding.v0"
)
FORMAL_STAGE_ADAPTER_REGISTRY_SCHEMA = (
    "policy-learnware.v03-formal-stage-adapter-registry.v0"
)
FORMAL_STAGE_REQUEST_TEMPLATE_REGISTRY_SCHEMA = (
    "policy-learnware.v03-formal-stage-request-template-registry.v0"
)


class PreflightError(ValueError):
    """A freeze, resume, public barrier or recompute record is invalid."""


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> Mapping[str, Any]:
    try:
        return strict_mapping(value, expected, where)
    except ValueError as error:
        raise PreflightError(str(error)) from error


def _digest(value: Any, where: str) -> str:
    try:
        return checked_digest(value, where)
    except ValueError as error:
        raise PreflightError(str(error)) from error


def _id(value: Any, where: str) -> str:
    try:
        return checked_safe_id(value, where)
    except ValueError as error:
        raise PreflightError(str(error)) from error


def formal_stage_adapter_binding_digest(
    stage_id: str,
    adapter_id: str,
    adapter_contract_digest: str,
) -> str:
    """Return the freeze-bound identity of one reviewed formal stage adapter."""

    if stage_id not in FORMAL_PRODUCTION_STAGE_IDS:
        raise PreflightError(f"unknown formal production stage: {stage_id!r}")
    canonical_adapter_id = _id(adapter_id, "formal adapter_id")
    canonical_contract = _digest(
        adapter_contract_digest, "formal adapter_contract_digest"
    )
    return sha256_json(
        {
            "schema": FORMAL_STAGE_ADAPTER_BINDING_SCHEMA,
            "stage_id": stage_id,
            "adapter_id": canonical_adapter_id,
            "adapter_contract_digest": canonical_contract,
        }
    )


@dataclass(frozen=True)
class HardTodoEvidence:
    todo_id: str
    contract_digest: str
    implementation_digest: str
    unit_test_evidence_digest: str
    synthetic_fixture_evidence_digest: str
    cpu_integration_evidence_digest: str
    status: str = "ENGINEERING_PASS"
    schema: str = HARD_TODO_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != HARD_TODO_EVIDENCE_SCHEMA:
            raise PreflightError("unsupported HardTodoEvidence schema")
        if self.todo_id not in HARD_TODO_IDS:
            raise PreflightError(f"unknown v0.3 hard TODO: {self.todo_id!r}")
        if self.status != "ENGINEERING_PASS":
            raise PreflightError("hard TODO evidence must have ENGINEERING_PASS status")
        for name in (
            "contract_digest",
            "implementation_digest",
            "unit_test_evidence_digest",
            "synthetic_fixture_evidence_digest",
            "cpu_integration_evidence_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def evidence_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "todo_id": self.todo_id,
            "contract_digest": self.contract_digest,
            "implementation_digest": self.implementation_digest,
            "unit_test_evidence_digest": self.unit_test_evidence_digest,
            "synthetic_fixture_evidence_digest": self.synthetic_fixture_evidence_digest,
            "cpu_integration_evidence_digest": self.cpu_integration_evidence_digest,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HardTodoEvidence":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "hard TODO evidence")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class PreExperimentFreezeManifest:
    """The immutable code/protocol boundary before any large formal run.

    ``review_authority_verified`` may only be set by a caller that already has
    an external authority receipt.  The manifest records that receipt; it does
    not derive or self-sign one.
    """

    freeze_id: str
    config_bytes_digest: str
    implementation_tree_digest: str
    clean_commit_digest: str
    review_decisions_digest: str
    review_authority_receipt_digest: str | None
    review_authority_verified: bool
    encoder_extension_gate_enabled: bool
    data_role_manifest_digest: str
    canonicalizer_registry_digest: str
    signal_matrix_digest: str
    signal_contrast_plan_digest: str
    signal_materiality_threshold_digest: str
    formal_signal_readout_plan_digest: str
    preoracle_signal_outcome_plan_digest: str
    signal_identity_registry_digest: str
    signal_execution_protocol_digest: str
    representation_plan_digest: str
    condition_plan_digest: str
    formal_source_fit_schedule_digest: str
    formal_source_membership_digest: str
    signal_work_item_graph_digest: str
    formal_signal_prefix_schedule_digest: str
    dynamics_axis_registry_digest: str
    public_query_plan_digest: str
    baseline_plan_digest: str
    statistics_plan_digest: str
    cost_protocol_digest: str
    source_reduced_query_empirical_protocol_digest: str
    formal_gate_plan_digests: Mapping[str, str]
    formal_stage_request_template_digests: Mapping[str, str]
    hard_todo_evidence: tuple[HardTodoEvidence, ...]
    formal_stage_adapter_binding_digests: Mapping[str, str] = field(
        default_factory=dict
    )
    schema: str = PRE_EXPERIMENT_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRE_EXPERIMENT_FREEZE_SCHEMA:
            raise PreflightError("unsupported PreExperimentFreezeManifest schema")
        object.__setattr__(self, "freeze_id", _id(self.freeze_id, "freeze_id"))
        for name in (
            "config_bytes_digest",
            "implementation_tree_digest",
            "clean_commit_digest",
            "review_decisions_digest",
            "data_role_manifest_digest",
            "canonicalizer_registry_digest",
            "signal_matrix_digest",
            "signal_contrast_plan_digest",
            "signal_materiality_threshold_digest",
            "formal_signal_readout_plan_digest",
            "preoracle_signal_outcome_plan_digest",
            "signal_identity_registry_digest",
            "signal_execution_protocol_digest",
            "representation_plan_digest",
            "condition_plan_digest",
            "formal_source_fit_schedule_digest",
            "formal_source_membership_digest",
            "signal_work_item_graph_digest",
            "formal_signal_prefix_schedule_digest",
            "dynamics_axis_registry_digest",
            "public_query_plan_digest",
            "baseline_plan_digest",
            "statistics_plan_digest",
            "cost_protocol_digest",
            "source_reduced_query_empirical_protocol_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if type(self.review_authority_verified) is not bool:
            raise PreflightError("review_authority_verified must be boolean")
        if type(self.encoder_extension_gate_enabled) is not bool:
            raise PreflightError("encoder_extension_gate_enabled must be boolean")
        if self.encoder_extension_gate_enabled:
            raise PreflightError(
                "v0.3 pre-experiment freeze requires encoder extension gate disabled"
            )
        if self.review_authority_verified:
            if self.review_authority_receipt_digest is None:
                raise PreflightError(
                    "verified review authority requires an external receipt digest"
                )
            object.__setattr__(
                self,
                "review_authority_receipt_digest",
                _digest(
                    self.review_authority_receipt_digest,
                    "review_authority_receipt_digest",
                ),
            )
        elif self.review_authority_receipt_digest is not None:
            raise PreflightError(
                "an unverified freeze cannot carry a review authority receipt"
            )
        gate_plans: dict[str, str] = {}
        if not isinstance(self.formal_gate_plan_digests, Mapping):
            raise PreflightError("formal_gate_plan_digests must be a mapping")
        for gate_id, plan_digest in self.formal_gate_plan_digests.items():
            if gate_id not in FORMAL_GATE_PLAN_IDS:
                raise PreflightError(f"unknown formal gate plan: {gate_id!r}")
            gate_plans[gate_id] = _digest(
                plan_digest, f"formal_gate_plan_digests[{gate_id}]"
            )
        if self.review_authority_verified:
            if set(gate_plans) != set(FORMAL_GATE_PLAN_IDS):
                raise PreflightError(
                    "formal freeze requires exact G03 Attribution/Probe/Market plan digests"
                )
        elif gate_plans:
            raise PreflightError(
                "development freeze cannot carry formal gate plan digests"
            )
        object.__setattr__(
            self,
            "formal_gate_plan_digests",
            MappingProxyType(dict(sorted(gate_plans.items()))),
        )
        request_templates: dict[str, str] = {}
        if not isinstance(self.formal_stage_request_template_digests, Mapping):
            raise PreflightError(
                "formal_stage_request_template_digests must be a mapping"
            )
        for stage_id, template_digest in (
            self.formal_stage_request_template_digests.items()
        ):
            if stage_id not in FORMAL_PRODUCTION_STAGE_IDS:
                raise PreflightError(
                    f"unknown stage in formal request template registry: {stage_id!r}"
                )
            request_templates[stage_id] = _digest(
                template_digest,
                f"formal_stage_request_template_digests[{stage_id}]",
            )
        if self.review_authority_verified:
            if set(request_templates) != set(FORMAL_PRODUCTION_STAGE_IDS):
                raise PreflightError(
                    "formal freeze requires an exact request template for every production stage"
                )
        elif request_templates:
            raise PreflightError(
                "development freeze cannot carry formal stage request templates"
            )
        object.__setattr__(
            self,
            "formal_stage_request_template_digests",
            MappingProxyType(dict(sorted(request_templates.items()))),
        )
        adapter_bindings: dict[str, str] = {}
        if not isinstance(self.formal_stage_adapter_binding_digests, Mapping):
            raise PreflightError(
                "formal_stage_adapter_binding_digests must be a mapping"
            )
        for stage_id, binding_digest in self.formal_stage_adapter_binding_digests.items():
            if stage_id not in FORMAL_PRODUCTION_STAGE_IDS:
                raise PreflightError(
                    f"unknown stage in formal adapter registry: {stage_id!r}"
                )
            adapter_bindings[stage_id] = _digest(
                binding_digest,
                f"formal_stage_adapter_binding_digests[{stage_id}]",
            )
        if self.review_authority_verified:
            if set(adapter_bindings) != set(FORMAL_PRODUCTION_STAGE_IDS):
                raise PreflightError(
                    "formal freeze requires an exact adapter binding for every production stage"
                )
        elif adapter_bindings:
            raise PreflightError(
                "development freeze cannot carry formal stage adapter bindings"
            )
        object.__setattr__(
            self,
            "formal_stage_adapter_binding_digests",
            MappingProxyType(dict(sorted(adapter_bindings.items()))),
        )
        evidence = tuple(self.hard_todo_evidence)
        if not all(isinstance(item, HardTodoEvidence) for item in evidence):
            raise PreflightError("hard_todo_evidence must contain typed evidence")
        identities = tuple(item.todo_id for item in evidence)
        if set(identities) != set(HARD_TODO_IDS) or len(identities) != len(HARD_TODO_IDS):
            raise PreflightError(
                "pre-experiment freeze requires exactly one evidence row for each T-P4-01..05"
            )
        object.__setattr__(
            self,
            "hard_todo_evidence",
            tuple(sorted(evidence, key=lambda item: item.todo_id)),
        )

    @property
    def engineering_ready(self) -> bool:
        return True

    @property
    def formal_run_authorized(self) -> bool:
        return self.review_authority_verified

    @property
    def formal_stage_adapter_registry_digest(self) -> str:
        return sha256_json(
            {
                "schema": FORMAL_STAGE_ADAPTER_REGISTRY_SCHEMA,
                "binding_digests": dict(
                    self.formal_stage_adapter_binding_digests
                ),
            }
        )

    @property
    def formal_stage_request_template_registry_digest(self) -> str:
        return sha256_json(
            {
                "schema": FORMAL_STAGE_REQUEST_TEMPLATE_REGISTRY_SCHEMA,
                "template_digests": dict(
                    self.formal_stage_request_template_digests
                ),
            }
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "freeze_id": self.freeze_id,
            "config_bytes_digest": self.config_bytes_digest,
            "implementation_tree_digest": self.implementation_tree_digest,
            "clean_commit_digest": self.clean_commit_digest,
            "review_decisions_digest": self.review_decisions_digest,
            "review_authority_receipt_digest": self.review_authority_receipt_digest,
            "review_authority_verified": self.review_authority_verified,
            "encoder_extension_gate_enabled": self.encoder_extension_gate_enabled,
            "data_role_manifest_digest": self.data_role_manifest_digest,
            "canonicalizer_registry_digest": self.canonicalizer_registry_digest,
            "signal_matrix_digest": self.signal_matrix_digest,
            "signal_contrast_plan_digest": self.signal_contrast_plan_digest,
            "signal_materiality_threshold_digest": (
                self.signal_materiality_threshold_digest
            ),
            "formal_signal_readout_plan_digest": (
                self.formal_signal_readout_plan_digest
            ),
            "preoracle_signal_outcome_plan_digest": (
                self.preoracle_signal_outcome_plan_digest
            ),
            "signal_identity_registry_digest": self.signal_identity_registry_digest,
            "signal_execution_protocol_digest": self.signal_execution_protocol_digest,
            "representation_plan_digest": self.representation_plan_digest,
            "condition_plan_digest": self.condition_plan_digest,
            "formal_source_fit_schedule_digest": (
                self.formal_source_fit_schedule_digest
            ),
            "formal_source_membership_digest": (
                self.formal_source_membership_digest
            ),
            "signal_work_item_graph_digest": self.signal_work_item_graph_digest,
            "formal_signal_prefix_schedule_digest": (
                self.formal_signal_prefix_schedule_digest
            ),
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "public_query_plan_digest": self.public_query_plan_digest,
            "baseline_plan_digest": self.baseline_plan_digest,
            "statistics_plan_digest": self.statistics_plan_digest,
            "cost_protocol_digest": self.cost_protocol_digest,
            "source_reduced_query_empirical_protocol_digest": (
                self.source_reduced_query_empirical_protocol_digest
            ),
            "formal_gate_plan_digests": dict(self.formal_gate_plan_digests),
            "formal_stage_request_template_digests": dict(
                self.formal_stage_request_template_digests
            ),
            "formal_stage_request_template_registry_digest": (
                self.formal_stage_request_template_registry_digest
            ),
            "hard_todo_evidence": [item.to_dict() for item in self.hard_todo_evidence],
            "formal_stage_adapter_binding_digests": dict(
                self.formal_stage_adapter_binding_digests
            ),
            "formal_stage_adapter_registry_digest": (
                self.formal_stage_adapter_registry_digest
            ),
            "engineering_ready": self.engineering_ready,
            "formal_run_authorized": self.formal_run_authorized,
        }

    @property
    def freeze_manifest_digest(self) -> str:
        return sha256_json(self._payload_without_digest())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "freeze_manifest_digest": self.freeze_manifest_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PreExperimentFreezeManifest":
        fields = set(cls.__dataclass_fields__) | {
            "engineering_ready",
            "formal_run_authorized",
            "freeze_manifest_digest",
            "formal_stage_adapter_registry_digest",
            "formal_stage_request_template_registry_digest",
        }
        data = _strict(value, fields, "pre-experiment freeze manifest")
        manifest = cls(
            **{
                field: (
                    tuple(HardTodoEvidence.from_dict(item) for item in data[field])
                    if field == "hard_todo_evidence"
                    else data[field]
                )
                for field in cls.__dataclass_fields__
            }
        )
        if data["engineering_ready"] is not manifest.engineering_ready:
            raise PreflightError("engineering_ready projection is inconsistent")
        if data["formal_run_authorized"] is not manifest.formal_run_authorized:
            raise PreflightError("formal_run_authorized projection is inconsistent")
        if (
            _digest(
                data["formal_stage_adapter_registry_digest"],
                "formal_stage_adapter_registry_digest",
            )
            != manifest.formal_stage_adapter_registry_digest
        ):
            raise PreflightError(
                "formal stage adapter registry digest is inconsistent"
            )
        if (
            _digest(
                data["formal_stage_request_template_registry_digest"],
                "formal_stage_request_template_registry_digest",
            )
            != manifest.formal_stage_request_template_registry_digest
        ):
            raise PreflightError(
                "formal stage request template registry digest is inconsistent"
            )
        if _digest(data["freeze_manifest_digest"], "freeze_manifest_digest") != manifest.freeze_manifest_digest:
            raise PreflightError("freeze manifest digest does not match contents")
        return manifest


WorkState = Literal["PENDING", "RUNNING", "COMPLETE", "FAILED"]


@dataclass(frozen=True)
class ExecutionCheckpoint:
    """Immutable snapshot used for fail-closed interruption/resume."""

    execution_plan_digest: str
    work_item_states: Mapping[str, WorkState]
    completed_artifact_digests: Mapping[str, str]
    attempt: int
    schema: str = EXECUTION_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_CHECKPOINT_SCHEMA:
            raise PreflightError("unsupported ExecutionCheckpoint schema")
        object.__setattr__(
            self,
            "execution_plan_digest",
            _digest(self.execution_plan_digest, "execution_plan_digest"),
        )
        states = dict(sorted(self.work_item_states.items()))
        if not states:
            raise PreflightError("execution checkpoint requires work items")
        for work_id, state in states.items():
            _id(work_id, "work_item_id")
            if state not in {"PENDING", "RUNNING", "COMPLETE", "FAILED"}:
                raise PreflightError(f"invalid work-item state: {state!r}")
        artifacts = dict(sorted(self.completed_artifact_digests.items()))
        if set(artifacts) != {item for item, state in states.items() if state == "COMPLETE"}:
            raise PreflightError(
                "completed artifact coverage must exactly equal COMPLETE work items"
            )
        for work_id, digest in artifacts.items():
            artifacts[work_id] = _digest(digest, f"completed_artifact_digests[{work_id}]")
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 0:
            raise PreflightError("attempt must be a non-negative integer")
        object.__setattr__(self, "work_item_states", MappingProxyType(states))
        object.__setattr__(self, "completed_artifact_digests", MappingProxyType(artifacts))

    @property
    def checkpoint_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_plan_digest": self.execution_plan_digest,
            "work_item_states": dict(self.work_item_states),
            "completed_artifact_digests": dict(self.completed_artifact_digests),
            "attempt": self.attempt,
        }

    def resume(self) -> "ExecutionCheckpoint":
        """Reset interrupted/failed items while preserving byte-bound results."""

        return ExecutionCheckpoint(
            execution_plan_digest=self.execution_plan_digest,
            work_item_states={
                item: state if state == "COMPLETE" else "PENDING"
                for item, state in self.work_item_states.items()
            },
            completed_artifact_digests=self.completed_artifact_digests,
            attempt=self.attempt + 1,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionCheckpoint":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "execution checkpoint")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class PublicRankingPublication:
    """One formal method×query ranking and every public input it claims."""

    method_id: str
    opaque_query_id: str
    ranking_digest: str
    query_spec_digest: str
    probe_dataset_digest: str
    target_evidence_digest: str
    cost_digest: str
    policy_market_id: str
    representation_index_digest: str
    selector_view_digest: str
    evidence_contract_digest: str
    selector_artifact_digest: str
    development_freeze_digest: str
    query_input_digest: str
    query_mode: str
    execution_mode: str
    development_context_count: int
    publication_digest: str | None = None
    schema: str = PUBLIC_RANKING_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_RANKING_PUBLICATION_SCHEMA:
            raise PreflightError("unsupported PublicRankingPublication schema")
        if self.method_id not in REQUIRED_BASELINE_METHOD_IDS:
            raise PreflightError("ranking publication has an unknown baseline method")
        if (
            not isinstance(self.opaque_query_id, str)
            or _OPAQUE_QUERY_ID.fullmatch(self.opaque_query_id) is None
        ):
            raise PreflightError("ranking publication has an invalid v03 query ID")
        for name in (
            "ranking_digest",
            "query_spec_digest",
            "probe_dataset_digest",
            "target_evidence_digest",
            "cost_digest",
            "policy_market_id",
            "representation_index_digest",
            "selector_view_digest",
            "evidence_contract_digest",
            "selector_artifact_digest",
            "development_freeze_digest",
            "query_input_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.execution_mode != FORMAL_MODE:
            raise PreflightError("oracle barrier accepts formal rankings only")
        if self.query_mode != "QUERY_EMPIRICAL":
            raise PreflightError(
                "formal ranking publication requires QUERY_EMPIRICAL"
            )
        if (
            isinstance(self.development_context_count, bool)
            or not isinstance(self.development_context_count, int)
            or self.development_context_count != FORMAL_DEVELOPMENT_CONTEXT_COUNT
        ):
            raise PreflightError(
                "formal ranking publication requires exactly 24 development contexts"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.publication_digest is None:
            object.__setattr__(self, "publication_digest", expected)
        elif _digest(self.publication_digest, "publication_digest") != expected:
            raise PreflightError("ranking publication digest does not match contents")

    @property
    def publication_key(self) -> tuple[str, str]:
        return self.method_id, self.opaque_query_id

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "opaque_query_id": self.opaque_query_id,
            "ranking_digest": self.ranking_digest,
            "query_spec_digest": self.query_spec_digest,
            "probe_dataset_digest": self.probe_dataset_digest,
            "target_evidence_digest": self.target_evidence_digest,
            "cost_digest": self.cost_digest,
            "policy_market_id": self.policy_market_id,
            "representation_index_digest": self.representation_index_digest,
            "selector_view_digest": self.selector_view_digest,
            "evidence_contract_digest": self.evidence_contract_digest,
            "selector_artifact_digest": self.selector_artifact_digest,
            "development_freeze_digest": self.development_freeze_digest,
            "query_input_digest": self.query_input_digest,
            "query_mode": self.query_mode,
            "execution_mode": self.execution_mode,
            "development_context_count": self.development_context_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "publication_digest": self.publication_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicRankingPublication":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "public ranking publication")
        return cls(**{field: data[field] for field in fields})

    @classmethod
    def from_published_ranking(cls, ranking: Any) -> "PublicRankingPublication":
        # Local import avoids making preflight construction depend on baseline
        # execution, while still requiring the exact typed formal artifact here.
        from .baselines import PublishedFullRanking

        if not isinstance(ranking, PublishedFullRanking):
            raise PreflightError("publication requires a PublishedFullRanking")
        return cls(
            method_id=ranking.method_id,
            opaque_query_id=ranking.opaque_query_id,
            ranking_digest=str(ranking.ranking_digest),
            query_spec_digest=ranking.query_spec_digest,
            probe_dataset_digest=ranking.probe_dataset_digest,
            target_evidence_digest=ranking.target_evidence_digest,
            cost_digest=ranking.cost_digest,
            policy_market_id=ranking.policy_market_id,
            representation_index_digest=ranking.representation_index_digest,
            selector_view_digest=ranking.selector_view_digest,
            evidence_contract_digest=ranking.evidence_contract_digest,
            selector_artifact_digest=ranking.selector_artifact_digest,
            development_freeze_digest=ranking.development_freeze_digest,
            query_input_digest=ranking.query_input_digest,
            query_mode=ranking.query_mode,
            execution_mode=ranking.execution_mode,
            development_context_count=ranking.development_context_count,
        )


def formal_baseline_input_plan_digest(
    publications: Sequence[PublicRankingPublication],
    *,
    expected_opaque_query_ids: Sequence[str],
    query_alias_manifest_digest: str,
) -> str:
    """Digest every formal baseline input while excluding ranking outputs.

    The returned value is suitable for
    ``PreExperimentFreezeManifest.baseline_plan_digest``.  It binds the exact
    24-context selector freeze/artifacts, market/index/evidence contracts and
    public query inputs before any ranking output or oracle access exists.
    """

    queries = tuple(sorted(expected_opaque_query_ids))
    if not queries or len(set(queries)) != len(queries) or any(
        not isinstance(item, str) or _OPAQUE_QUERY_ID.fullmatch(item) is None
        for item in queries
    ):
        raise PreflightError("formal baseline plan requires canonical query IDs")
    alias_digest = _digest(
        query_alias_manifest_digest, "query_alias_manifest_digest"
    )
    rows = tuple(publications)
    if not all(isinstance(item, PublicRankingPublication) for item in rows):
        raise PreflightError("formal baseline plan requires typed publications")
    expected_pairs = {
        (method_id, query_id)
        for method_id in REQUIRED_BASELINE_METHOD_IDS
        for query_id in queries
    }
    if (
        len(rows) != len(expected_pairs)
        or {item.publication_key for item in rows} != expected_pairs
    ):
        raise PreflightError(
            "formal baseline plan must cover the full method×query matrix"
        )
    input_rows = []
    for item in sorted(rows, key=lambda value: value.publication_key):
        payload = item._payload_without_digest()
        payload.pop("ranking_digest")
        input_rows.append(payload)
    return sha256_json(
        {
            "schema": "policy-learnware.v03-formal-baseline-input-plan.v0",
            "expected_method_ids": list(REQUIRED_BASELINE_METHOD_IDS),
            "expected_opaque_query_ids": list(queries),
            "query_alias_manifest_digest": alias_digest,
            "input_rows": input_rows,
        }
    )


@dataclass(frozen=True)
class PublicQueryPlan:
    """Frozen 30/24/12 public query schedule with explicit regime identity."""

    regime_by_opaque_query_id: Mapping[str, str]
    query_alias_manifest_digest: str
    plan_digest: str | None = None
    schema: str = PUBLIC_QUERY_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_QUERY_PLAN_SCHEMA:
            raise PreflightError("unsupported PublicQueryPlan schema")
        alias = _digest(
            self.query_alias_manifest_digest, "query_alias_manifest_digest"
        )
        if not isinstance(self.regime_by_opaque_query_id, Mapping):
            raise PreflightError("public query plan requires a regime mapping")
        regimes: dict[str, str] = {}
        for query_id, regime in sorted(self.regime_by_opaque_query_id.items()):
            if (
                not isinstance(query_id, str)
                or _OPAQUE_QUERY_ID.fullmatch(query_id) is None
            ):
                raise PreflightError("public query plan contains an invalid query ID")
            if regime not in FORMAL_QUERY_REGIME_COUNTS:
                raise PreflightError("public query plan contains an invalid regime")
            regimes[query_id] = regime
        observed_counts = {
            regime: sum(value == regime for value in regimes.values())
            for regime in FORMAL_QUERY_REGIME_COUNTS
        }
        if observed_counts != dict(FORMAL_QUERY_REGIME_COUNTS):
            raise PreflightError(
                "formal public query plan requires exactly 30 exact, 24 interpolation, and 12 extrapolation queries"
            )
        object.__setattr__(
            self, "regime_by_opaque_query_id", MappingProxyType(regimes)
        )
        object.__setattr__(self, "query_alias_manifest_digest", alias)
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "public query plan_digest") != expected:
            raise PreflightError("public query plan digest does not match contents")

    @property
    def opaque_query_ids(self) -> tuple[str, ...]:
        return tuple(self.regime_by_opaque_query_id)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime_by_opaque_query_id": dict(self.regime_by_opaque_query_id),
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicQueryPlan":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "public query plan")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class PublicRankingBarrier:
    run_id: str
    freeze_manifest: PreExperimentFreezeManifest
    query_plan: PublicQueryPlan
    expected_opaque_query_ids: tuple[str, ...]
    expected_method_ids: tuple[str, ...]
    publications: tuple[PublicRankingPublication, ...]
    query_alias_manifest_digest: str
    preoracle_signal_outcome_manifest_digest: str
    oracle_read_count: int = 0
    schema: str = PUBLIC_RANKING_BARRIER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_RANKING_BARRIER_SCHEMA:
            raise PreflightError("unsupported PublicRankingBarrier schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        if not isinstance(self.freeze_manifest, PreExperimentFreezeManifest):
            raise PreflightError(
                "public ranking barrier requires a typed pre-experiment freeze"
            )
        if not self.freeze_manifest.formal_run_authorized:
            raise PreflightError(
                "public ranking barrier requires externally reviewed formal authority"
            )
        if not isinstance(self.query_plan, PublicQueryPlan):
            raise PreflightError("public ranking barrier requires PublicQueryPlan")
        if self.query_plan.plan_digest != self.freeze_manifest.public_query_plan_digest:
            raise PreflightError(
                "public query schedule differs from the externally reviewed freeze"
            )
        object.__setattr__(
            self,
            "query_alias_manifest_digest",
            _digest(self.query_alias_manifest_digest, "query_alias_manifest_digest"),
        )
        object.__setattr__(
            self,
            "preoracle_signal_outcome_manifest_digest",
            _digest(
                self.preoracle_signal_outcome_manifest_digest,
                "preoracle_signal_outcome_manifest_digest",
            ),
        )
        queries = tuple(sorted(self.expected_opaque_query_ids))
        if any(
            not isinstance(item, str) or _OPAQUE_QUERY_ID.fullmatch(item) is None
            for item in queries
        ):
            raise PreflightError("expected query IDs must use canonical v03q identities")
        if not queries or len(set(queries)) != len(queries):
            raise PreflightError("expected opaque query IDs must be non-empty and unique")
        if queries != tuple(sorted(self.query_plan.opaque_query_ids)):
            raise PreflightError(
                "public ranking query coverage differs from the 30/24/12 plan"
            )
        if self.query_alias_manifest_digest != self.query_plan.query_alias_manifest_digest:
            raise PreflightError(
                "public ranking alias manifest differs from the query plan"
            )
        methods = tuple(self.expected_method_ids)
        if methods != REQUIRED_BASELINE_METHOD_IDS:
            raise PreflightError(
                "public ranking barrier must freeze the exact required baseline methods"
            )
        publications = tuple(self.publications)
        if not all(isinstance(item, PublicRankingPublication) for item in publications):
            raise PreflightError("public ranking barrier requires typed publications")
        expected_pairs = {(method, query) for method in methods for query in queries}
        observed_pairs = {item.publication_key for item in publications}
        if len(publications) != len(observed_pairs) or observed_pairs != expected_pairs:
            raise PreflightError(
                "public ranking coverage must equal the full method×query matrix"
            )
        method_order = {method: index for index, method in enumerate(methods)}
        publications = tuple(
            sorted(
                publications,
                key=lambda item: (
                    method_order[item.method_id],
                    item.opaque_query_id,
                ),
            )
        )
        if len({item.policy_market_id for item in publications}) != 1:
            raise PreflightError("all public rankings must use one frozen policy market")
        if len({item.development_freeze_digest for item in publications}) != 1:
            raise PreflightError(
                "all public rankings must use one frozen 24-context development plan"
            )
        for query_id in queries:
            group = tuple(
                item for item in publications if item.opaque_query_id == query_id
            )
            if len({item.probe_dataset_digest for item in group}) != 1:
                raise PreflightError(
                    "one opaque query cannot refer to multiple probe datasets"
                )
            if len({item.target_evidence_digest for item in group}) != 1:
                raise PreflightError(
                    "one opaque query cannot refer to multiple target evidence records"
                )
        method_binding_fields = (
            "policy_market_id",
            "representation_index_digest",
            "selector_view_digest",
            "evidence_contract_digest",
            "selector_artifact_digest",
            "development_freeze_digest",
            "execution_mode",
            "development_context_count",
        )
        for method_id in methods:
            group = tuple(item for item in publications if item.method_id == method_id)
            if any(
                len({getattr(item, field) for item in group}) != 1
                for field in method_binding_fields
            ):
                raise PreflightError(
                    f"method {method_id!r} changes its frozen selector binding across queries"
                )
        if self.oracle_read_count != 0:
            raise PreflightError("public rankings must be published before any oracle read")
        observed_baseline_plan = formal_baseline_input_plan_digest(
            publications,
            expected_opaque_query_ids=queries,
            query_alias_manifest_digest=self.query_alias_manifest_digest,
        )
        if observed_baseline_plan != self.freeze_manifest.baseline_plan_digest:
            raise PreflightError(
                "public ranking inputs differ from the externally reviewed baseline plan"
            )
        object.__setattr__(self, "expected_opaque_query_ids", queries)
        object.__setattr__(self, "expected_method_ids", methods)
        object.__setattr__(self, "publications", publications)

    @property
    def publication_count(self) -> int:
        return len(self.publications)

    @property
    def freeze_manifest_digest(self) -> str:
        return self.freeze_manifest.freeze_manifest_digest

    @property
    def ranking_digests(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                f"{item.method_id}::{item.opaque_query_id}": item.ranking_digest
                for item in self.publications
            }
        )

    @property
    def barrier_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest": self.freeze_manifest.to_dict(),
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "query_plan": self.query_plan.to_dict(),
            "expected_opaque_query_ids": list(self.expected_opaque_query_ids),
            "expected_method_ids": list(self.expected_method_ids),
            "publications": [item.to_dict() for item in self.publications],
            "publication_digests": [
                item.publication_digest for item in self.publications
            ],
            "publication_count": self.publication_count,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "preoracle_signal_outcome_manifest_digest": (
                self.preoracle_signal_outcome_manifest_digest
            ),
            "oracle_read_count": self.oracle_read_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicRankingBarrier":
        fields = set(cls.__dataclass_fields__) | {
            "freeze_manifest_digest",
            "publication_digests",
            "publication_count",
        }
        data = _strict(value, fields, "public ranking barrier")
        barrier = cls(
            **{
                field: (
                    PreExperimentFreezeManifest.from_dict(data[field])
                    if field == "freeze_manifest"
                    else PublicQueryPlan.from_dict(data[field])
                    if field == "query_plan"
                    else tuple(
                        PublicRankingPublication.from_dict(item)
                        for item in data[field]
                    )
                    if field == "publications"
                    else tuple(data[field])
                    if field in {"expected_opaque_query_ids", "expected_method_ids"}
                    else data[field]
                )
                for field in cls.__dataclass_fields__
            }
        )
        if data["freeze_manifest_digest"] != barrier.freeze_manifest_digest:
            raise PreflightError("serialized freeze_manifest_digest is inconsistent")
        if data["publication_count"] != barrier.publication_count:
            raise PreflightError("serialized publication_count is inconsistent")
        if data["publication_digests"] != [
            item.publication_digest for item in barrier.publications
        ]:
            raise PreflightError("serialized publication_digests are inconsistent")
        return barrier


@dataclass(frozen=True)
class OracleUnlockHandoff:
    """Read-only request handed to the external oracle owner.

    It is not an unlock token and confers no v0.3 write capability.
    """

    run_id: str
    freeze_manifest_digest: str
    public_ranking_barrier_digest: str
    requested_owner: str = ORACLE_OWNER
    v03_oracle_write_capability: bool = False
    schema: str = ORACLE_UNLOCK_HANDOFF_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_UNLOCK_HANDOFF_SCHEMA:
            raise PreflightError("unsupported OracleUnlockHandoff schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "freeze_manifest_digest",
            _digest(self.freeze_manifest_digest, "freeze_manifest_digest"),
        )
        object.__setattr__(
            self,
            "public_ranking_barrier_digest",
            _digest(
                self.public_ranking_barrier_digest,
                "public_ranking_barrier_digest",
            ),
        )
        if self.requested_owner != ORACLE_OWNER:
            raise PreflightError(f"oracle owner must be {ORACLE_OWNER}")
        if self.v03_oracle_write_capability is not False:
            raise PreflightError("v0.3 cannot acquire confirmatory-oracle write capability")

    @property
    def handoff_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "public_ranking_barrier_digest": self.public_ranking_barrier_digest,
            "requested_owner": self.requested_owner,
            "v03_oracle_write_capability": self.v03_oracle_write_capability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleUnlockHandoff":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "oracle unlock handoff")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class IndependentRecomputeAttestation:
    run_id: str
    freeze_manifest_digest: str
    public_ranking_barrier_digest: str
    formal_statistics_result_digest: str
    raw_input_manifest_digest: str
    primary_artifact_root_digest: str
    recompute_artifact_root_digest: str
    primary_result_digest: str
    recompute_result_digest: str
    primary_process_nonce_digest: str
    recompute_process_nonce_digest: str
    schema: str = INDEPENDENT_RECOMPUTE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != INDEPENDENT_RECOMPUTE_SCHEMA:
            raise PreflightError("unsupported IndependentRecomputeAttestation schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "freeze_manifest_digest",
            "public_ranking_barrier_digest",
            "formal_statistics_result_digest",
            "raw_input_manifest_digest",
            "primary_artifact_root_digest",
            "recompute_artifact_root_digest",
            "primary_result_digest",
            "recompute_result_digest",
            "primary_process_nonce_digest",
            "recompute_process_nonce_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.primary_artifact_root_digest == self.recompute_artifact_root_digest:
            raise PreflightError("independent recompute must use a distinct artifact root")
        if self.primary_process_nonce_digest == self.recompute_process_nonce_digest:
            raise PreflightError("independent recompute must use a fresh process nonce")
        if self.primary_result_digest != self.recompute_result_digest:
            raise PreflightError("independent recompute result does not match primary result")
        if self.formal_statistics_result_digest != self.primary_result_digest:
            raise PreflightError(
                "independent recompute must attest the exact formal statistics result"
            )

    @property
    def attestation_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "IndependentRecomputeAttestation":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "independent recompute attestation")
        return cls(**{field: data[field] for field in fields})


__all__ = [
    "EXECUTION_CHECKPOINT_SCHEMA",
    "HARD_TODO_EVIDENCE_SCHEMA",
    "HARD_TODO_IDS",
    "INDEPENDENT_RECOMPUTE_SCHEMA",
    "ORACLE_OWNER",
    "ORACLE_UNLOCK_HANDOFF_SCHEMA",
    "PRE_EXPERIMENT_FREEZE_SCHEMA",
    "PUBLIC_QUERY_PLAN_SCHEMA",
    "PUBLIC_RANKING_BARRIER_SCHEMA",
    "PUBLIC_RANKING_PUBLICATION_SCHEMA",
    "ExecutionCheckpoint",
    "FORMAL_QUERY_REGIME_COUNTS",
    "FORMAL_GATE_PLAN_IDS",
    "FORMAL_PRODUCTION_STAGE_IDS",
    "FORMAL_STAGE_ADAPTER_BINDING_SCHEMA",
    "FORMAL_STAGE_ADAPTER_REGISTRY_SCHEMA",
    "FORMAL_STAGE_REQUEST_TEMPLATE_REGISTRY_SCHEMA",
    "HardTodoEvidence",
    "IndependentRecomputeAttestation",
    "OracleUnlockHandoff",
    "PreExperimentFreezeManifest",
    "PreflightError",
    "PublicRankingBarrier",
    "PublicRankingPublication",
    "PublicQueryPlan",
    "formal_baseline_input_plan_digest",
    "formal_stage_adapter_binding_digest",
]
