"""Authority-bound admissions for the three pre-large v0.3 asset gates.

The engineering records in :mod:`attribution`, :mod:`probe_audit`, and
:mod:`source_market` intentionally cannot grant formal authority.  This module
adds the narrow external-review boundary that was missing between those
records and a formal production run.  It does not collect data, train a model,
or read an oracle.

All public projections contain only aggregate counts and content digests.  In
particular, archived paths, task IDs, source-anchor IDs, candidate IDs, and
deployment identities remain private.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .attribution import (
    ATTRIBUTION_REPLAY_PROTOCOL_ID,
    FORMAL_ATTRIBUTION_PREFIX_EPISODE_COUNTS,
    ArchivedLegacyReference,
    AttributionPrefixSchedule,
    AttributionSuite,
    REQUIRED_ATTRIBUTION_VIEW_IDS,
)
from .pool_intake import EXPECTED_ANCHOR_COUNT, EXPECTED_JOB_COUNT
from .preflight import PreExperimentFreezeManifest
from .probe_audit import ProbeAuditReport
from .probes import (
    CP0_EXACT_COMMON,
    CP2_UNSEEN_PROBE,
    FROZEN_PROBE_STYLES,
    ProbeTrainingManifest,
    validate_cp2_holdout,
)
from .representation_ladder import R0_PADDED_RAW, R_HIST_RANDOM_TANH
from .source_market import (
    EvaluatorSourceReceipt,
    SourceChampionizationRecord,
    SourceEvaluationProtocol,
    SourceMarketError,
    V03SourcePolicyMarket,
    derive_market_opaque_id,
    derive_market_tie_break_token,
    formal_market_alias_protocol_digest,
    market_nonce_commitment,
)
from .transition_views import V_RANDOM_ENCODER


FORMAL_GATE_IDS = (
    "G03-Attribution",
    "G03-Probe",
    "G03-Market",
)
FORMAL_GATE_PLAN_KEYS = frozenset(FORMAL_GATE_IDS)

FORMAL_GATE_AUTHORITY_SCHEMA = "policy-learnware.v03-formal-gate-authority.v0"
FORMAL_ATTRIBUTION_PLAN_SCHEMA = (
    "policy-learnware.v03-formal-attribution-plan.v0"
)
FORMAL_ATTRIBUTION_EVIDENCE_SCHEMA = (
    "policy-learnware.v03-formal-attribution-recompute-evidence.v0"
)
FORMAL_ATTRIBUTION_ADMISSION_SCHEMA = (
    "policy-learnware.v03-formal-attribution-admission.v0"
)
FORMAL_PROBE_PLAN_SCHEMA = "policy-learnware.v03-formal-probe-plan.v0"
FORMAL_PROBE_ADMISSION_SCHEMA = "policy-learnware.v03-formal-probe-admission.v0"
FORMAL_MARKET_PLAN_SCHEMA = "policy-learnware.v03-formal-market-plan.v0"
FORMAL_MARKET_EVIDENCE_SCHEMA = "policy-learnware.v03-formal-market-evidence.v0"
FORMAL_MARKET_ADMISSION_SCHEMA = (
    "policy-learnware.v03-formal-market-admission.v0"
)

FORMAL_ATTRIBUTION_INPUT_VIEW_IDS = tuple(
    view_id
    for view_id in REQUIRED_ATTRIBUTION_VIEW_IDS
    if view_id != V_RANDOM_ENCODER
)
if len(FORMAL_ATTRIBUTION_INPUT_VIEW_IDS) != 13:  # pragma: no cover - registry guard
    raise RuntimeError("formal attribution must contain exactly thirteen input views")


class FormalGateError(ValueError):
    """A formal gate record is malformed or differs from its frozen plan."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise FormalGateError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise FormalGateError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise FormalGateError(f"{where} must be a non-empty canonical string")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise FormalGateError(f"{where} must be a mapping")
    if set(value) != expected:
        raise FormalGateError(
            f"{where} fields differ: missing={sorted(expected-set(value))}, "
            f"unknown={sorted(set(value)-expected)}"
        )


def _digest_map(value: Mapping[str, str], where: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise FormalGateError(f"{where} must be a mapping")
    result: dict[str, str] = {}
    for key, digest in value.items():
        result[_text(key, f"{where} key")] = _digest(
            digest, f"{where}[{key!r}]"
        )
    return MappingProxyType(dict(sorted(result.items())))


def _text_map(value: Mapping[str, str], where: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise FormalGateError(f"{where} must be a mapping")
    result = {
        _text(key, f"{where} key"): _text(item, f"{where}[{key!r}]")
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _nested_digest_map(
    value: Mapping[str, Mapping[str, str]], where: str
) -> Mapping[str, Mapping[str, str]]:
    if not isinstance(value, Mapping):
        raise FormalGateError(f"{where} must be a mapping")
    result = {
        _text(key, f"{where} key"): _digest_map(rows, f"{where}[{key!r}]")
        for key, rows in value.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _reasons(value: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_text(item, "failure reason") for item in value)
    if len(set(result)) != len(result):
        raise FormalGateError("failure reasons must be unique")
    return result


def _pair_key(task_id: str, axis_id: str) -> str:
    return f"{_text(task_id, 'task_id')}::{_text(axis_id, 'axis_id')}"


def _assert_freeze_binding(
    freeze: PreExperimentFreezeManifest,
    *,
    gate_id: str,
    plan_digest: str,
) -> None:
    if not isinstance(freeze, PreExperimentFreezeManifest):
        raise FormalGateError("formal admission requires a typed freeze manifest")
    if not freeze.formal_run_authorized:
        raise FormalGateError("formal admission requires external freeze authority")
    observed = freeze.formal_gate_plan_digests.get(gate_id)
    if observed != plan_digest:
        raise FormalGateError(f"freeze binds another {gate_id} plan")


@dataclass(frozen=True)
class FormalGateAuthorityReceipt:
    """Receipt emitted by an external reviewer, never by a development gate."""

    gate_id: Literal["G03-Attribution", "G03-Probe", "G03-Market"]
    plan_digest: str
    evidence_digest: str
    freeze_manifest_digest: str
    authority_id: str
    external_review_record_digest: str
    decision: Literal["AUTHORIZED"] = "AUTHORIZED"
    receipt_digest: str | None = None
    schema: str = FORMAL_GATE_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_GATE_AUTHORITY_SCHEMA:
            raise FormalGateError("unsupported formal gate authority schema")
        if self.gate_id not in FORMAL_GATE_PLAN_KEYS:
            raise FormalGateError("unknown formal gate identity")
        for name in (
            "plan_digest",
            "evidence_digest",
            "freeze_manifest_digest",
            "external_review_record_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _text(self.authority_id, "authority_id")
        if self.decision != "AUTHORIZED":
            raise FormalGateError("formal gate authority must be externally authorized")
        if self.external_review_record_digest in {
            self.plan_digest,
            self.evidence_digest,
            self.freeze_manifest_digest,
        }:
            raise FormalGateError(
                "external review record must be distinct from reviewed artifacts"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.receipt_digest is None:
            object.__setattr__(self, "receipt_digest", expected)
        elif _digest(self.receipt_digest, "receipt_digest") != expected:
            raise FormalGateError("formal authority receipt digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "gate_id": self.gate_id,
            "plan_digest": self.plan_digest,
            "evidence_digest": self.evidence_digest,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "authority_id": self.authority_id,
            "external_review_record_digest": self.external_review_record_digest,
            "decision": self.decision,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalGateAuthorityReceipt":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal gate authority receipt")
        return cls(**{name: value[name] for name in fields})


def _assert_authority(
    authority: FormalGateAuthorityReceipt,
    *,
    gate_id: str,
    plan_digest: str,
    evidence_digest: str,
    freeze: PreExperimentFreezeManifest,
) -> None:
    if not isinstance(authority, FormalGateAuthorityReceipt):
        raise FormalGateError("formal admission requires a typed authority receipt")
    if (
        authority.gate_id != gate_id
        or authority.plan_digest != plan_digest
        or authority.evidence_digest != evidence_digest
        or authority.freeze_manifest_digest != freeze.freeze_manifest_digest
    ):
        raise FormalGateError("formal authority receipt binding mismatch")


@dataclass(frozen=True)
class FormalAttributionPlan:
    required_input_view_ids: tuple[str, ...]
    historical_random_view_id: str
    historical_random_representation_id: str
    prefix_episode_counts: tuple[int, ...]
    prefix_schedule_digest: str
    attribution_replay_protocol_id: str
    archive_protocol_id: str
    archive_manifest_digest: str
    archived_reference_digest: str
    archived_dataset_digest: str
    canonical_bank_digest: str
    encoder_checkpoint_digest: str
    encoder_implementation_digest: str
    legacy_normalizer_digest: str
    independent_recompute_protocol_digest: str
    plan_digest: str | None = None
    schema: str = FORMAL_ATTRIBUTION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_ATTRIBUTION_PLAN_SCHEMA:
            raise FormalGateError("unsupported formal attribution plan schema")
        views = tuple(self.required_input_view_ids)
        if views != FORMAL_ATTRIBUTION_INPUT_VIEW_IDS:
            raise FormalGateError(
                "formal attribution plan requires the exact thirteen input views"
            )
        object.__setattr__(self, "required_input_view_ids", views)
        if self.historical_random_view_id != V_RANDOM_ENCODER:
            raise FormalGateError("formal attribution random view drifted")
        if self.historical_random_representation_id != R_HIST_RANDOM_TANH:
            raise FormalGateError("formal attribution historical random cell drifted")
        prefixes = tuple(self.prefix_episode_counts)
        if prefixes != FORMAL_ATTRIBUTION_PREFIX_EPISODE_COUNTS:
            raise FormalGateError("formal attribution prefix schedule drifted")
        object.__setattr__(self, "prefix_episode_counts", prefixes)
        if self.prefix_schedule_digest != AttributionPrefixSchedule.formal().schedule_digest:
            raise FormalGateError("formal attribution prefix digest drifted")
        if self.attribution_replay_protocol_id != ATTRIBUTION_REPLAY_PROTOCOL_ID:
            raise FormalGateError("formal attribution replay protocol drifted")
        _text(self.archive_protocol_id, "archive_protocol_id")
        for name in (
            "prefix_schedule_digest",
            "attribution_replay_protocol_id",
            "archive_manifest_digest",
            "archived_reference_digest",
            "archived_dataset_digest",
            "canonical_bank_digest",
            "encoder_checkpoint_digest",
            "encoder_implementation_digest",
            "legacy_normalizer_digest",
            "independent_recompute_protocol_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise FormalGateError("formal attribution plan digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "required_input_view_ids": list(self.required_input_view_ids),
            "historical_random_view_id": self.historical_random_view_id,
            "historical_random_representation_id": self.historical_random_representation_id,
            "prefix_episode_counts": list(self.prefix_episode_counts),
            "prefix_schedule_digest": self.prefix_schedule_digest,
            "attribution_replay_protocol_id": self.attribution_replay_protocol_id,
            "archive_protocol_id": self.archive_protocol_id,
            "archive_manifest_digest": self.archive_manifest_digest,
            "archived_reference_digest": self.archived_reference_digest,
            "archived_dataset_digest": self.archived_dataset_digest,
            "canonical_bank_digest": self.canonical_bank_digest,
            "encoder_checkpoint_digest": self.encoder_checkpoint_digest,
            "encoder_implementation_digest": self.encoder_implementation_digest,
            "legacy_normalizer_digest": self.legacy_normalizer_digest,
            "independent_recompute_protocol_digest": self.independent_recompute_protocol_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalAttributionPlan":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal attribution plan")
        return cls(
            **{
                name: tuple(value[name]) if name in {"required_input_view_ids", "prefix_episode_counts"} else value[name]
                for name in fields
            }
        )


@dataclass(frozen=True)
class FormalAttributionRecomputeEvidence:
    primary_suite_digest: str
    archived_reference_digest: str
    prefix_schedule_digest: str
    report_digests_by_view: Mapping[str, str]
    independent_report_digests_by_view: Mapping[str, str]
    independent_recompute_protocol_digest: str
    independent_recompute_receipt_digest: str
    independent_execution_digest: str
    full_legacy_replay_pass: bool
    controls_fail_closed_pass: bool
    contribution_quantified_pass: bool
    shared_schema_explanation_pass: bool
    test_time_ood_claim_bounded_pass: bool
    maximum_legacy_replay_error: float
    failure_reasons: tuple[str, ...]
    evidence_scope: Literal["LEGACY_ARCHIVED"] = "LEGACY_ARCHIVED"
    evidence_digest: str | None = None
    schema: str = FORMAL_ATTRIBUTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_ATTRIBUTION_EVIDENCE_SCHEMA:
            raise FormalGateError("unsupported formal attribution evidence schema")
        if self.evidence_scope != "LEGACY_ARCHIVED":
            raise FormalGateError("formal attribution rejects synthetic evidence")
        for name in (
            "primary_suite_digest",
            "archived_reference_digest",
            "prefix_schedule_digest",
            "independent_recompute_protocol_digest",
            "independent_recompute_receipt_digest",
            "independent_execution_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        primary = _digest_map(self.report_digests_by_view, "report_digests_by_view")
        independent = _digest_map(
            self.independent_report_digests_by_view,
            "independent_report_digests_by_view",
        )
        expected_views = set(REQUIRED_ATTRIBUTION_VIEW_IDS)
        if set(primary) != expected_views or independent != primary:
            raise FormalGateError(
                "independent attribution recompute must reproduce all 13+1 reports"
            )
        object.__setattr__(self, "report_digests_by_view", primary)
        object.__setattr__(self, "independent_report_digests_by_view", independent)
        if len(
            {
                self.primary_suite_digest,
                self.independent_recompute_receipt_digest,
                self.independent_execution_digest,
            }
        ) != 3:
            raise FormalGateError(
                "independent recompute receipts must be distinct from the primary run"
            )
        for name in (
            "full_legacy_replay_pass",
            "controls_fail_closed_pass",
            "contribution_quantified_pass",
            "shared_schema_explanation_pass",
            "test_time_ood_claim_bounded_pass",
        ):
            if type(getattr(self, name)) is not bool:
                raise FormalGateError(f"{name} must be boolean")
        error = self.maximum_legacy_replay_error
        if (
            isinstance(error, bool)
            or not isinstance(error, (int, float, np.integer, np.floating))
            or not np.isfinite(error)
            or error < 0.0
        ):
            raise FormalGateError("maximum_legacy_replay_error is invalid")
        object.__setattr__(self, "maximum_legacy_replay_error", float(error))
        object.__setattr__(self, "failure_reasons", _reasons(self.failure_reasons))
        expected = sha256_json(self._payload_without_digest())
        if self.evidence_digest is None:
            object.__setattr__(self, "evidence_digest", expected)
        elif _digest(self.evidence_digest, "evidence_digest") != expected:
            raise FormalGateError("formal attribution evidence digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_scope": self.evidence_scope,
            "primary_suite_digest": self.primary_suite_digest,
            "archived_reference_digest": self.archived_reference_digest,
            "prefix_schedule_digest": self.prefix_schedule_digest,
            "report_digests_by_view": dict(self.report_digests_by_view),
            "independent_report_digests_by_view": dict(self.independent_report_digests_by_view),
            "independent_recompute_protocol_digest": self.independent_recompute_protocol_digest,
            "independent_recompute_receipt_digest": self.independent_recompute_receipt_digest,
            "independent_execution_digest": self.independent_execution_digest,
            "full_legacy_replay_pass": self.full_legacy_replay_pass,
            "controls_fail_closed_pass": self.controls_fail_closed_pass,
            "contribution_quantified_pass": self.contribution_quantified_pass,
            "shared_schema_explanation_pass": self.shared_schema_explanation_pass,
            "test_time_ood_claim_bounded_pass": self.test_time_ood_claim_bounded_pass,
            "maximum_legacy_replay_error": self.maximum_legacy_replay_error,
            "failure_reasons": list(self.failure_reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "evidence_digest": self.evidence_digest}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "FormalAttributionRecomputeEvidence":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal attribution recompute evidence")
        return cls(
            **{
                name: tuple(value[name]) if name == "failure_reasons" else value[name]
                for name in fields
            }
        )


def build_formal_attribution_recompute_evidence(
    suite: AttributionSuite,
    *,
    independent_report_digests_by_view: Mapping[str, str],
    independent_recompute_protocol_digest: str,
    independent_recompute_receipt_digest: str,
    independent_execution_digest: str,
    test_time_ood_claim_bounded_pass: bool,
) -> FormalAttributionRecomputeEvidence:
    """Wrap two archived replays; development status never grants authority."""

    if not isinstance(suite, AttributionSuite):
        raise FormalGateError("formal attribution evidence requires a typed suite")
    if not suite.prefix_schedule.formal_eligible:
        raise FormalGateError("formal attribution rejects a development prefix schedule")
    reports = {report.view_id: report.digest for report in suite.reports}
    gate = suite.gate_evidence
    return FormalAttributionRecomputeEvidence(
        primary_suite_digest=suite.digest,
        archived_reference_digest=suite.archived_reference_digest,
        prefix_schedule_digest=suite.prefix_schedule.schedule_digest,
        report_digests_by_view=reports,
        independent_report_digests_by_view=independent_report_digests_by_view,
        independent_recompute_protocol_digest=independent_recompute_protocol_digest,
        independent_recompute_receipt_digest=independent_recompute_receipt_digest,
        independent_execution_digest=independent_execution_digest,
        full_legacy_replay_pass=gate.full_legacy_replay_pass,
        controls_fail_closed_pass=gate.controls_fail_closed_pass,
        contribution_quantified_pass=gate.contribution_quantified_pass,
        shared_schema_explanation_pass=gate.shared_schema_explanation_pass,
        test_time_ood_claim_bounded_pass=test_time_ood_claim_bounded_pass,
        maximum_legacy_replay_error=gate.maximum_legacy_replay_error,
        failure_reasons=gate.failure_reasons,
    )


@dataclass(frozen=True)
class FormalAttributionAdmission:
    plan_digest: str
    evidence_digest: str
    authority_receipt_digest: str
    freeze_manifest_digest: str
    status: Literal["PASS", "NO_GO"]
    failure_reasons: tuple[str, ...]
    admission_digest: str | None = None
    schema: str = FORMAL_ATTRIBUTION_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_ATTRIBUTION_ADMISSION_SCHEMA:
            raise FormalGateError("unsupported formal attribution admission schema")
        for name in (
            "plan_digest",
            "evidence_digest",
            "authority_receipt_digest",
            "freeze_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        reasons = _reasons(self.failure_reasons)
        if self.status not in {"PASS", "NO_GO"}:
            raise FormalGateError("unknown formal attribution status")
        if (self.status == "PASS") == bool(reasons):
            raise FormalGateError("formal attribution status/reasons are inconsistent")
        object.__setattr__(self, "failure_reasons", reasons)
        expected = sha256_json(self._payload_without_digest())
        if self.admission_digest is None:
            object.__setattr__(self, "admission_digest", expected)
        elif _digest(self.admission_digest, "admission_digest") != expected:
            raise FormalGateError("formal attribution admission digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "evidence_digest": self.evidence_digest,
            "authority_receipt_digest": self.authority_receipt_digest,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "status": self.status,
            "failure_reasons": list(self.failure_reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "admission_digest": self.admission_digest}

    def public_projection(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-formal-attribution-admission-public.v0",
            "status": self.status,
            "view_count": 14,
            "plan_digest": self.plan_digest,
            "evidence_digest": self.evidence_digest,
            "admission_digest": self.admission_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalAttributionAdmission":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal attribution admission")
        return cls(
            **{
                name: tuple(value[name]) if name == "failure_reasons" else value[name]
                for name in fields
            }
        )


def admit_formal_attribution(
    *,
    plan: FormalAttributionPlan,
    archived_reference: ArchivedLegacyReference,
    suite: AttributionSuite,
    evidence: FormalAttributionRecomputeEvidence,
    authority: FormalGateAuthorityReceipt,
    freeze: PreExperimentFreezeManifest,
) -> FormalAttributionAdmission:
    if not isinstance(plan, FormalAttributionPlan):
        raise FormalGateError("formal attribution requires a typed plan")
    if not isinstance(archived_reference, ArchivedLegacyReference):
        raise FormalGateError("formal attribution requires a typed archive reference")
    if not isinstance(suite, AttributionSuite) or not isinstance(
        evidence, FormalAttributionRecomputeEvidence
    ):
        raise FormalGateError("formal attribution requires typed replay evidence")
    _assert_freeze_binding(
        freeze, gate_id="G03-Attribution", plan_digest=str(plan.plan_digest)
    )
    _assert_authority(
        authority,
        gate_id="G03-Attribution",
        plan_digest=str(plan.plan_digest),
        evidence_digest=str(evidence.evidence_digest),
        freeze=freeze,
    )
    reference_bindings = (
        plan.archive_protocol_id == archived_reference.archive_protocol_id,
        plan.archive_manifest_digest == archived_reference.archive_manifest_digest,
        plan.archived_reference_digest == archived_reference.digest,
        plan.archived_dataset_digest == archived_reference.archived_dataset_digest,
        plan.canonical_bank_digest == archived_reference.canonical_bank_digest,
        plan.encoder_checkpoint_digest == archived_reference.encoder_checkpoint_digest,
        plan.encoder_implementation_digest == archived_reference.encoder_implementation_digest,
    )
    if not all(reference_bindings):
        raise FormalGateError("formal attribution archive binding mismatch")
    if (
        suite.digest != evidence.primary_suite_digest
        or suite.archived_reference_digest != plan.archived_reference_digest
        or evidence.archived_reference_digest != plan.archived_reference_digest
        or evidence.prefix_schedule_digest != plan.prefix_schedule_digest
        or evidence.independent_recompute_protocol_digest
        != plan.independent_recompute_protocol_digest
    ):
        raise FormalGateError("formal attribution suite/evidence binding mismatch")
    reports = {report.view_id: report for report in suite.reports}
    if set(reports) != set(REQUIRED_ATTRIBUTION_VIEW_IDS):
        raise FormalGateError("formal attribution requires exactly 13+1 reports")
    if tuple(plan.required_input_view_ids) != FORMAL_ATTRIBUTION_INPUT_VIEW_IDS:
        raise FormalGateError("formal attribution input view order drifted")
    for view_id, report in reports.items():
        if (
            report.digest != evidence.report_digests_by_view[view_id]
            or report.prefix_schedule_scope != "FORMAL"
            or report.prefix_schedule_digest != plan.prefix_schedule_digest
            or report.encoder_checkpoint_digest != plan.encoder_checkpoint_digest
            or report.encoder_implementation_digest != plan.encoder_implementation_digest
            or report.archived_dataset_digest != plan.archived_dataset_digest
            or report.canonical_bank_digest != plan.canonical_bank_digest
        ):
            raise FormalGateError("formal attribution report binding mismatch")
    reasons = list(evidence.failure_reasons)
    checks = (
        (evidence.full_legacy_replay_pass, "LEGACY_FULL_REPLAY_MISMATCH"),
        (evidence.controls_fail_closed_pass, "ATTRIBUTION_CONTROLS_INCOMPLETE"),
        (evidence.contribution_quantified_pass, "ATTRIBUTION_DELTAS_INCOMPLETE"),
        (evidence.shared_schema_explanation_pass, "SHARED_SCHEMA_UNEXPLAINED"),
        (evidence.test_time_ood_claim_bounded_pass, "OOD_CAUSAL_CLAIM_UNBOUNDED"),
    )
    for passed, reason in checks:
        if not passed and reason not in reasons:
            reasons.append(reason)
    unique_reasons = tuple(dict.fromkeys(reasons))
    return FormalAttributionAdmission(
        plan_digest=str(plan.plan_digest),
        evidence_digest=str(evidence.evidence_digest),
        authority_receipt_digest=str(authority.receipt_digest),
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        status="NO_GO" if unique_reasons else "PASS",
        failure_reasons=unique_reasons,
    )


@dataclass(frozen=True)
class FormalProbePlan:
    required_task_ids: tuple[str, ...]
    required_task_axis_pairs: tuple[tuple[str, str], ...]
    cp0_style_id: str
    cp2_style_id: str
    raw_representation_id: str
    raw_representation_protocol_digest: str
    thresholds_digest: str
    training_manifest_digest: str
    target_bank_digests_by_task: Mapping[str, Mapping[str, str]]
    distance_semantic_bank_digests_by_pair: Mapping[str, Mapping[str, str]]
    plan_digest: str | None = None
    schema: str = FORMAL_PROBE_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_PROBE_PLAN_SCHEMA:
            raise FormalGateError("unsupported formal probe plan schema")
        tasks = tuple(self.required_task_ids)
        if (
            len(tasks) < 2
            or len(set(tasks)) != len(tasks)
            or any(_text(task, "required_task_id") != task for task in tasks)
        ):
            raise FormalGateError("formal probe requires at least two unique tasks")
        object.__setattr__(self, "required_task_ids", tasks)
        pairs = tuple(tuple(pair) for pair in self.required_task_axis_pairs)
        if (
            len(set(pairs)) != len(pairs)
            or any(len(pair) != 2 for pair in pairs)
            or {task for task, _ in pairs} != set(tasks)
            or len({axis for _, axis in pairs}) < 2
        ):
            raise FormalGateError("formal probe task/axis coverage is invalid")
        for task, axis in pairs:
            _text(task, "task axis task")
            _text(axis, "task axis axis")
        object.__setattr__(self, "required_task_axis_pairs", tuple(sorted(pairs)))
        try:
            cp0 = FROZEN_PROBE_STYLES[self.cp0_style_id]
            cp2 = FROZEN_PROBE_STYLES[self.cp2_style_id]
        except KeyError as error:
            raise FormalGateError("formal probe uses an unregistered style") from error
        if cp0.regime != CP0_EXACT_COMMON or cp2.regime != CP2_UNSEEN_PROBE:
            raise FormalGateError("formal probe must bind one CP0 and one CP2 style")
        if self.raw_representation_id != R0_PADDED_RAW:
            raise FormalGateError("formal probe must use R0 padded Raw")
        for name in (
            "raw_representation_protocol_digest",
            "thresholds_digest",
            "training_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        targets = _nested_digest_map(
            self.target_bank_digests_by_task,
            "target_bank_digests_by_task",
        )
        expected_styles = {self.cp0_style_id, self.cp2_style_id}
        if set(targets) != set(tasks) or any(
            set(rows) != expected_styles for rows in targets.values()
        ):
            raise FormalGateError("formal probe target banks must cover exact task x CP0/CP2")
        object.__setattr__(self, "target_bank_digests_by_task", targets)
        semantic = _nested_digest_map(
            self.distance_semantic_bank_digests_by_pair,
            "distance_semantic_bank_digests_by_pair",
        )
        expected_pair_keys = {_pair_key(task, axis) for task, axis in pairs}
        if set(semantic) != expected_pair_keys or any(
            len(rows) < 3 for rows in semantic.values()
        ):
            raise FormalGateError("formal probe distance-bank registry is incomplete")
        object.__setattr__(self, "distance_semantic_bank_digests_by_pair", semantic)
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise FormalGateError("formal probe plan digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "required_task_ids": list(self.required_task_ids),
            "required_task_axis_pairs": [list(pair) for pair in self.required_task_axis_pairs],
            "cp0_style_id": self.cp0_style_id,
            "cp2_style_id": self.cp2_style_id,
            "raw_representation_id": self.raw_representation_id,
            "raw_representation_protocol_digest": self.raw_representation_protocol_digest,
            "thresholds_digest": self.thresholds_digest,
            "training_manifest_digest": self.training_manifest_digest,
            "target_bank_digests_by_task": {
                task: dict(rows) for task, rows in self.target_bank_digests_by_task.items()
            },
            "distance_semantic_bank_digests_by_pair": {
                pair: dict(rows)
                for pair, rows in self.distance_semantic_bank_digests_by_pair.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalProbePlan":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal probe plan")
        return cls(
            **{
                name: (
                    tuple(tuple(pair) for pair in value[name])
                    if name == "required_task_axis_pairs"
                    else tuple(value[name])
                    if name == "required_task_ids"
                    else value[name]
                )
                for name in fields
            }
        )


@dataclass(frozen=True)
class FormalProbeAdmission:
    plan_digest: str
    evidence_digest: str
    authority_receipt_digest: str
    freeze_manifest_digest: str
    status: Literal["PASS", "NO_GO"]
    failure_reasons: tuple[str, ...]
    task_count: int
    axis_count: int
    admission_digest: str | None = None
    schema: str = FORMAL_PROBE_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_PROBE_ADMISSION_SCHEMA:
            raise FormalGateError("unsupported formal probe admission schema")
        for name in (
            "plan_digest",
            "evidence_digest",
            "authority_receipt_digest",
            "freeze_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("task_count", "axis_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise FormalGateError(f"{name} must be an integer >= 2")
        reasons = _reasons(self.failure_reasons)
        if self.status not in {"PASS", "NO_GO"} or (
            (self.status == "PASS") == bool(reasons)
        ):
            raise FormalGateError("formal probe status/reasons are inconsistent")
        object.__setattr__(self, "failure_reasons", reasons)
        expected = sha256_json(self._payload_without_digest())
        if self.admission_digest is None:
            object.__setattr__(self, "admission_digest", expected)
        elif _digest(self.admission_digest, "admission_digest") != expected:
            raise FormalGateError("formal probe admission digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "evidence_digest": self.evidence_digest,
            "authority_receipt_digest": self.authority_receipt_digest,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "status": self.status,
            "failure_reasons": list(self.failure_reasons),
            "task_count": self.task_count,
            "axis_count": self.axis_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "admission_digest": self.admission_digest}

    def public_projection(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-formal-probe-admission-public.v0",
            "status": self.status,
            "task_count": self.task_count,
            "axis_count": self.axis_count,
            "plan_digest": self.plan_digest,
            "evidence_digest": self.evidence_digest,
            "admission_digest": self.admission_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalProbeAdmission":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal probe admission")
        return cls(
            **{
                name: tuple(value[name]) if name == "failure_reasons" else value[name]
                for name in fields
            }
        )


def admit_formal_probe(
    *,
    plan: FormalProbePlan,
    report: ProbeAuditReport,
    training_manifest: ProbeTrainingManifest,
    authority: FormalGateAuthorityReceipt,
    freeze: PreExperimentFreezeManifest,
) -> FormalProbeAdmission:
    if not isinstance(plan, FormalProbePlan) or not isinstance(report, ProbeAuditReport):
        raise FormalGateError("formal probe requires typed plan and report")
    if not isinstance(training_manifest, ProbeTrainingManifest):
        raise FormalGateError("formal probe requires a typed training manifest")
    validate_cp2_holdout(training_manifest)
    _assert_freeze_binding(freeze, gate_id="G03-Probe", plan_digest=str(plan.plan_digest))
    _assert_authority(
        authority,
        gate_id="G03-Probe",
        plan_digest=str(plan.plan_digest),
        evidence_digest=report.digest,
        freeze=freeze,
    )
    if (
        report.thresholds.digest != plan.thresholds_digest
        or report.training_manifest_digest != plan.training_manifest_digest
        or training_manifest.digest != plan.training_manifest_digest
        or tuple(report.freeze_decision.required_task_ids) != plan.required_task_ids
        or tuple(report.freeze_decision.required_task_axis_pairs)
        != plan.required_task_axis_pairs
    ):
        raise FormalGateError("formal probe threshold/training/freeze binding mismatch")
    summaries = {(row.task_id, row.probe_style_id): row for row in report.summaries}
    expected_summary_keys = {
        (task, style)
        for task in plan.required_task_ids
        for style in (plan.cp0_style_id, plan.cp2_style_id)
    }
    if len(summaries) != len(report.summaries) or set(summaries) != expected_summary_keys:
        raise FormalGateError("formal probe requires exact task x CP0/CP2 banks")
    for task, style in expected_summary_keys:
        if summaries[(task, style)].dataset_digest != plan.target_bank_digests_by_task[task][style]:
            raise FormalGateError("formal probe target-bank digest mismatch")
    distances = {(row.task_id, row.axis_id): row for row in report.distance_evidence}
    if len(distances) != len(report.distance_evidence) or set(distances) != set(
        plan.required_task_axis_pairs
    ):
        raise FormalGateError("formal probe requires exact task x axis evidence")
    for pair, row in distances.items():
        if (
            row.representation_id != R0_PADDED_RAW
            or row.representation_protocol_digest
            != plan.raw_representation_protocol_digest
            or dict(row.semantic_bank_digests)
            != dict(plan.distance_semantic_bank_digests_by_pair[_pair_key(*pair)])
        ):
            raise FormalGateError("formal probe Raw/distance-bank binding mismatch")
    reasons = list(report.failure_reasons)
    if report.gate_status != "DEVELOPMENT_PASS" and not reasons:
        reasons.append("PROBE_COVERAGE_NO_GO")
    return FormalProbeAdmission(
        plan_digest=str(plan.plan_digest),
        evidence_digest=report.digest,
        authority_receipt_digest=str(authority.receipt_digest),
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        status="NO_GO" if reasons else "PASS",
        failure_reasons=tuple(dict.fromkeys(reasons)),
        task_count=len(plan.required_task_ids),
        axis_count=len({axis for _, axis in plan.required_task_axis_pairs}),
    )


@dataclass(frozen=True)
class FormalMarketPlan:
    intake_record_digest: str
    source_pool_digest: str
    source_evaluation_protocol_digest: str
    intake_cell_digests_by_candidate: Mapping[str, str]
    source_anchor_id_by_candidate: Mapping[str, str]
    deployment_abi_digests_by_candidate: Mapping[str, str]
    market_alias_protocol_digest: str
    market_alias_commitment_digest: str
    tie_break_commitment_digest: str
    expected_candidate_count: int = EXPECTED_JOB_COUNT
    expected_market_entry_count: int = EXPECTED_ANCHOR_COUNT
    competence_mode: Literal["OBSERVE"] = "OBSERVE"
    plan_digest: str | None = None
    schema: str = FORMAL_MARKET_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_MARKET_PLAN_SCHEMA:
            raise FormalGateError("unsupported formal market plan schema")
        for name in (
            "intake_record_digest",
            "source_pool_digest",
            "source_evaluation_protocol_digest",
            "market_alias_protocol_digest",
            "market_alias_commitment_digest",
            "tie_break_commitment_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.market_alias_commitment_digest == self.tie_break_commitment_digest:
            raise FormalGateError("market alias and tie-break commitments must differ")
        if (
            self.expected_candidate_count != EXPECTED_JOB_COUNT
            or self.expected_market_entry_count != EXPECTED_ANCHOR_COUNT
            or self.competence_mode != "OBSERVE"
        ):
            raise FormalGateError("formal market count/mode contract drifted")
        cells = _digest_map(
            self.intake_cell_digests_by_candidate,
            "intake_cell_digests_by_candidate",
        )
        anchors = _digest_map(
            self.source_anchor_id_by_candidate,
            "source_anchor_id_by_candidate",
        )
        abis = _digest_map(
            self.deployment_abi_digests_by_candidate,
            "deployment_abi_digests_by_candidate",
        )
        if len(cells) != EXPECTED_JOB_COUNT or set(cells) != set(anchors) or set(cells) != set(abis):
            raise FormalGateError("formal market plan requires exact-90 candidate bindings")
        counts: dict[str, int] = {}
        for anchor in anchors.values():
            counts[anchor] = counts.get(anchor, 0) + 1
        if len(counts) != EXPECTED_ANCHOR_COUNT or set(counts.values()) != {3}:
            raise FormalGateError("formal market plan requires 30 anchors x 3 candidates")
        object.__setattr__(self, "intake_cell_digests_by_candidate", cells)
        object.__setattr__(self, "source_anchor_id_by_candidate", anchors)
        object.__setattr__(self, "deployment_abi_digests_by_candidate", abis)
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise FormalGateError("formal market plan digest mismatch")

    @property
    def source_anchor_ids(self) -> frozenset[str]:
        return frozenset(self.source_anchor_id_by_candidate.values())

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "intake_record_digest": self.intake_record_digest,
            "source_pool_digest": self.source_pool_digest,
            "source_evaluation_protocol_digest": self.source_evaluation_protocol_digest,
            "intake_cell_digests_by_candidate": dict(self.intake_cell_digests_by_candidate),
            "source_anchor_id_by_candidate": dict(self.source_anchor_id_by_candidate),
            "deployment_abi_digests_by_candidate": dict(self.deployment_abi_digests_by_candidate),
            "market_alias_protocol_digest": self.market_alias_protocol_digest,
            "market_alias_commitment_digest": self.market_alias_commitment_digest,
            "tie_break_commitment_digest": self.tie_break_commitment_digest,
            "expected_candidate_count": self.expected_candidate_count,
            "expected_market_entry_count": self.expected_market_entry_count,
            "competence_mode": self.competence_mode,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalMarketPlan":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal market plan")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class FormalMarketEvidence:
    intake_record_digest: str
    source_pool_digest: str
    source_evaluation_protocol_digest: str
    selection_receipt_digests_by_candidate: Mapping[str, str]
    attestation_receipt_digests_by_candidate: Mapping[str, str]
    champion_candidate_ids_by_anchor: Mapping[str, str]
    champion_digests_by_anchor: Mapping[str, str]
    competence_observation_digests_by_anchor: Mapping[str, str]
    championization_digest: str
    policy_market_id: str
    public_entry_digests_by_opaque_id: Mapping[str, str]
    deployment_entry_digests_by_opaque_id: Mapping[str, str]
    deployment_candidate_ids_by_opaque_id: Mapping[str, str]
    deployment_abi_digests_by_candidate: Mapping[str, str]
    derived_market_alias_protocol_digest: str
    derived_market_alias_commitment_digest: str
    derived_tie_break_commitment_digest: str
    derived_market_binding_digest: str
    receipts_binding_pass: bool
    market_binding_pass: bool
    observe_mode_pass: bool
    nonce_commitment_binding_pass: bool
    market_derivation_pass: bool
    failure_reasons: tuple[str, ...]
    evidence_digest: str | None = None
    schema: str = FORMAL_MARKET_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_MARKET_EVIDENCE_SCHEMA:
            raise FormalGateError("unsupported formal market evidence schema")
        for name in (
            "intake_record_digest",
            "source_pool_digest",
            "source_evaluation_protocol_digest",
            "championization_digest",
            "policy_market_id",
            "derived_market_alias_protocol_digest",
            "derived_market_alias_commitment_digest",
            "derived_tie_break_commitment_digest",
            "derived_market_binding_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in (
            "selection_receipt_digests_by_candidate",
            "attestation_receipt_digests_by_candidate",
            "champion_digests_by_anchor",
            "competence_observation_digests_by_anchor",
            "public_entry_digests_by_opaque_id",
            "deployment_entry_digests_by_opaque_id",
            "deployment_abi_digests_by_candidate",
        ):
            object.__setattr__(self, name, _digest_map(getattr(self, name), name))
        for name in (
            "champion_candidate_ids_by_anchor",
            "deployment_candidate_ids_by_opaque_id",
        ):
            object.__setattr__(self, name, _text_map(getattr(self, name), name))
        for name in (
            "receipts_binding_pass",
            "market_binding_pass",
            "observe_mode_pass",
            "nonce_commitment_binding_pass",
            "market_derivation_pass",
        ):
            if type(getattr(self, name)) is not bool:
                raise FormalGateError(f"{name} must be boolean")
        reasons = _reasons(self.failure_reasons)
        if reasons and all(
            (
                self.receipts_binding_pass,
                self.market_binding_pass,
                self.observe_mode_pass,
                self.nonce_commitment_binding_pass,
                self.market_derivation_pass,
            )
        ):
            raise FormalGateError("passing market evidence cannot carry failure reasons")
        object.__setattr__(self, "failure_reasons", reasons)
        expected = sha256_json(self._payload_without_digest())
        if self.evidence_digest is None:
            object.__setattr__(self, "evidence_digest", expected)
        elif _digest(self.evidence_digest, "evidence_digest") != expected:
            raise FormalGateError("formal market evidence digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "intake_record_digest": self.intake_record_digest,
            "source_pool_digest": self.source_pool_digest,
            "source_evaluation_protocol_digest": self.source_evaluation_protocol_digest,
            "selection_receipt_digests_by_candidate": dict(self.selection_receipt_digests_by_candidate),
            "attestation_receipt_digests_by_candidate": dict(self.attestation_receipt_digests_by_candidate),
            "champion_candidate_ids_by_anchor": dict(self.champion_candidate_ids_by_anchor),
            "champion_digests_by_anchor": dict(self.champion_digests_by_anchor),
            "competence_observation_digests_by_anchor": dict(self.competence_observation_digests_by_anchor),
            "championization_digest": self.championization_digest,
            "policy_market_id": self.policy_market_id,
            "public_entry_digests_by_opaque_id": dict(self.public_entry_digests_by_opaque_id),
            "deployment_entry_digests_by_opaque_id": dict(self.deployment_entry_digests_by_opaque_id),
            "deployment_candidate_ids_by_opaque_id": dict(self.deployment_candidate_ids_by_opaque_id),
            "deployment_abi_digests_by_candidate": dict(self.deployment_abi_digests_by_candidate),
            "derived_market_alias_protocol_digest": self.derived_market_alias_protocol_digest,
            "derived_market_alias_commitment_digest": self.derived_market_alias_commitment_digest,
            "derived_tie_break_commitment_digest": self.derived_tie_break_commitment_digest,
            "derived_market_binding_digest": self.derived_market_binding_digest,
            "receipts_binding_pass": self.receipts_binding_pass,
            "market_binding_pass": self.market_binding_pass,
            "observe_mode_pass": self.observe_mode_pass,
            "nonce_commitment_binding_pass": self.nonce_commitment_binding_pass,
            "market_derivation_pass": self.market_derivation_pass,
            "failure_reasons": list(self.failure_reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "evidence_digest": self.evidence_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalMarketEvidence":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal market evidence")
        return cls(
            **{
                name: tuple(value[name]) if name == "failure_reasons" else value[name]
                for name in fields
            }
        )


def build_formal_market_evidence(
    *,
    plan: FormalMarketPlan,
    intake_record_digest: str,
    source_pool_digest: str,
    protocol: SourceEvaluationProtocol,
    selection_receipts: Sequence[EvaluatorSourceReceipt],
    attestation_receipts: Sequence[EvaluatorSourceReceipt],
    championization: SourceChampionizationRecord,
    market: V03SourcePolicyMarket,
    market_alias_nonce: str,
    tie_break_nonce: str,
) -> FormalMarketEvidence:
    """Build private evidence and reopen both committed market nonces.

    The two raw nonces are used only to recompute commitments and the exact
    champion-to-alias/tie-token assignment.  They are never copied into the
    returned evidence.
    """

    if not isinstance(plan, FormalMarketPlan) or not isinstance(
        protocol, SourceEvaluationProtocol
    ):
        raise FormalGateError("market evidence requires a typed plan/source protocol")
    if not isinstance(championization, SourceChampionizationRecord) or not isinstance(
        market, V03SourcePolicyMarket
    ):
        raise FormalGateError("market evidence requires typed championization/market")
    selection = tuple(selection_receipts)
    attestation = tuple(attestation_receipts)
    if not all(isinstance(row, EvaluatorSourceReceipt) for row in selection + attestation):
        raise FormalGateError("market evidence contains an untyped source receipt")
    selection_map = {row.candidate_id: str(row.receipt_digest) for row in selection}
    attestation_map = {row.candidate_id: str(row.receipt_digest) for row in attestation}
    receipt_binding = (
        len(selection_map) == len(selection)
        and len(attestation_map) == len(attestation)
        and all(
            row.block == "source_selection"
            and row.intake_record_digest == intake_record_digest
            and row.source_evaluation_protocol_digest
            == protocol.source_evaluation_protocol_digest
            for row in selection
        )
        and all(
            row.block == "source_attestation"
            and row.intake_record_digest == intake_record_digest
            and row.source_evaluation_protocol_digest
            == protocol.source_evaluation_protocol_digest
            for row in attestation
        )
    )
    champions = championization.champions
    try:
        alias_commitment = market_nonce_commitment(
            purpose="market_alias",
            nonce=market_alias_nonce,
            intake_record_digest=intake_record_digest,
        )
        tie_commitment = market_nonce_commitment(
            purpose="market_tie_break",
            nonce=tie_break_nonce,
            intake_record_digest=intake_record_digest,
        )
        alias_protocol_digest = formal_market_alias_protocol_digest(
            intake_record_digest=intake_record_digest,
            source_pool_digest=source_pool_digest,
            alias_commitment_digest=alias_commitment,
            candidate_count=plan.expected_candidate_count,
            market_entry_count=plan.expected_market_entry_count,
        )
        expected_bindings = {
            champion.candidate_id: {
                "source_anchor_id": anchor,
                "opaque_learnware_id": derive_market_opaque_id(
                    candidate_id=champion.candidate_id,
                    market_alias_nonce=market_alias_nonce,
                ),
                "tie_break_token": derive_market_tie_break_token(
                    candidate_id=champion.candidate_id,
                    tie_break_nonce=tie_break_nonce,
                ),
            }
            for anchor, champion in champions.items()
        }
    except SourceMarketError as error:
        raise FormalGateError(f"invalid formal market nonce reopening: {error}") from error
    finally:
        # Make the non-persistence boundary explicit; only derived commitments
        # and bindings survive into the evidence below.
        del market_alias_nonce, tie_break_nonce
    nonce_binding = (
        alias_commitment == plan.market_alias_commitment_digest
        and tie_commitment == plan.tie_break_commitment_digest
        and alias_protocol_digest == plan.market_alias_protocol_digest
    )
    expected_opaque_ids = {
        binding["opaque_learnware_id"] for binding in expected_bindings.values()
    }
    market_derivation = (
        len(expected_bindings) == EXPECTED_ANCHOR_COUNT
        and len(expected_opaque_ids) == EXPECTED_ANCHOR_COUNT
        and set(market.entries) == expected_opaque_ids
        and set(market.deployment_private) == expected_opaque_ids
        and set(market.anchor_to_opaque_learnware_id) == set(champions)
        and all(
            market.entries[binding["opaque_learnware_id"]].tie_break_token
            == binding["tie_break_token"]
            and market.entries[
                binding["opaque_learnware_id"]
            ].normalized_source_competence
            == champions[binding["source_anchor_id"]].competence.normalized_competence
            and market.deployment_private[
                binding["opaque_learnware_id"]
            ].candidate_id
            == candidate_id
            and market.deployment_private[
                binding["opaque_learnware_id"]
            ].source_anchor_id
            == binding["source_anchor_id"]
            and market.anchor_to_opaque_learnware_id[binding["source_anchor_id"]]
            == binding["opaque_learnware_id"]
            for candidate_id, binding in expected_bindings.items()
        )
    )
    market_binding = (
        championization.intake_record_digest == intake_record_digest
        and championization.source_evaluation_protocol_digest
        == protocol.source_evaluation_protocol_digest
        and market.intake_record_digest == intake_record_digest
        and market.championization_digest == championization.championization_digest
        and market_derivation
    )
    observe = (
        championization.competence_mode == "OBSERVE"
        and all(champion.competence.mode == "OBSERVE" for champion in champions.values())
    )
    reasons: list[str] = []
    if not receipt_binding:
        reasons.append("SOURCE_RECEIPT_BINDING_FAILURE")
    if not market_binding:
        reasons.append("SOURCE_MARKET_BINDING_FAILURE")
    if not observe:
        reasons.append("SOURCE_COMPETENCE_NOT_OBSERVE")
    if not nonce_binding:
        reasons.append("MARKET_NONCE_COMMITMENT_MISMATCH")
    if not market_derivation:
        reasons.append("MARKET_ALIAS_TIE_DERIVATION_FAILURE")
    market_derivation_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-formal-market-derived-binding.v0",
            "policy_market_id": market.policy_market_id,
            "bindings_by_candidate": expected_bindings,
        }
    )
    return FormalMarketEvidence(
        intake_record_digest=intake_record_digest,
        source_pool_digest=source_pool_digest,
        source_evaluation_protocol_digest=protocol.source_evaluation_protocol_digest,
        selection_receipt_digests_by_candidate=selection_map,
        attestation_receipt_digests_by_candidate=attestation_map,
        champion_candidate_ids_by_anchor={
            anchor: champion.candidate_id for anchor, champion in champions.items()
        },
        champion_digests_by_anchor={
            anchor: str(champion.champion_digest) for anchor, champion in champions.items()
        },
        competence_observation_digests_by_anchor={
            anchor: str(champion.competence.observation_digest)
            for anchor, champion in champions.items()
        },
        championization_digest=str(championization.championization_digest),
        policy_market_id=market.policy_market_id,
        public_entry_digests_by_opaque_id={
            opaque_id: sha256_json(entry.to_dict())
            for opaque_id, entry in market.entries.items()
        },
        deployment_entry_digests_by_opaque_id={
            opaque_id: sha256_json(entry.to_dict())
            for opaque_id, entry in market.deployment_private.items()
        },
        deployment_candidate_ids_by_opaque_id={
            opaque_id: entry.candidate_id
            for opaque_id, entry in market.deployment_private.items()
        },
        deployment_abi_digests_by_candidate={
            entry.candidate_id: entry.execution_abi.digest
            for entry in market.deployment_private.values()
        },
        derived_market_alias_protocol_digest=alias_protocol_digest,
        derived_market_alias_commitment_digest=alias_commitment,
        derived_tie_break_commitment_digest=tie_commitment,
        derived_market_binding_digest=market_derivation_digest,
        receipts_binding_pass=receipt_binding,
        market_binding_pass=market_binding,
        observe_mode_pass=observe,
        nonce_commitment_binding_pass=nonce_binding,
        market_derivation_pass=market_derivation,
        failure_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class FormalMarketAdmission:
    plan_digest: str
    evidence_digest: str
    authority_receipt_digest: str
    freeze_manifest_digest: str
    status: Literal["ASSET_READY", "NO_GO"]
    failure_reasons: tuple[str, ...]
    candidate_count: int
    market_entry_count: int
    admission_digest: str | None = None
    schema: str = FORMAL_MARKET_ADMISSION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_MARKET_ADMISSION_SCHEMA:
            raise FormalGateError("unsupported formal market admission schema")
        for name in (
            "plan_digest",
            "evidence_digest",
            "authority_receipt_digest",
            "freeze_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("candidate_count", "market_entry_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FormalGateError(f"{name} must be a nonnegative integer")
        reasons = _reasons(self.failure_reasons)
        if self.status not in {"ASSET_READY", "NO_GO"} or (
            (self.status == "ASSET_READY") == bool(reasons)
        ):
            raise FormalGateError("formal market status/reasons are inconsistent")
        object.__setattr__(self, "failure_reasons", reasons)
        expected = sha256_json(self._payload_without_digest())
        if self.admission_digest is None:
            object.__setattr__(self, "admission_digest", expected)
        elif _digest(self.admission_digest, "admission_digest") != expected:
            raise FormalGateError("formal market admission digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "plan_digest": self.plan_digest,
            "evidence_digest": self.evidence_digest,
            "authority_receipt_digest": self.authority_receipt_digest,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "status": self.status,
            "failure_reasons": list(self.failure_reasons),
            "candidate_count": self.candidate_count,
            "market_entry_count": self.market_entry_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "admission_digest": self.admission_digest}

    def public_projection(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-formal-market-admission-public.v0",
            "status": self.status,
            "candidate_count": self.candidate_count,
            "market_entry_count": self.market_entry_count,
            "plan_digest": self.plan_digest,
            "evidence_digest": self.evidence_digest,
            "admission_digest": self.admission_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalMarketAdmission":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "formal market admission")
        return cls(
            **{
                name: tuple(value[name]) if name == "failure_reasons" else value[name]
                for name in fields
            }
        )


def admit_formal_market(
    *,
    plan: FormalMarketPlan,
    evidence: FormalMarketEvidence,
    authority: FormalGateAuthorityReceipt,
    freeze: PreExperimentFreezeManifest,
) -> FormalMarketAdmission:
    if not isinstance(plan, FormalMarketPlan) or not isinstance(
        evidence, FormalMarketEvidence
    ):
        raise FormalGateError("formal market requires typed plan and evidence")
    _assert_freeze_binding(freeze, gate_id="G03-Market", plan_digest=str(plan.plan_digest))
    _assert_authority(
        authority,
        gate_id="G03-Market",
        plan_digest=str(plan.plan_digest),
        evidence_digest=str(evidence.evidence_digest),
        freeze=freeze,
    )
    if (
        evidence.intake_record_digest != plan.intake_record_digest
        or evidence.source_pool_digest != plan.source_pool_digest
        or evidence.source_evaluation_protocol_digest
        != plan.source_evaluation_protocol_digest
    ):
        raise FormalGateError("formal market evidence belongs to another plan")
    reasons = list(evidence.failure_reasons)
    for matches, reason in (
        (
            evidence.derived_market_alias_protocol_digest
            == plan.market_alias_protocol_digest,
            "MARKET_ALIAS_PROTOCOL_MISMATCH",
        ),
        (
            evidence.derived_market_alias_commitment_digest
            == plan.market_alias_commitment_digest,
            "MARKET_ALIAS_COMMITMENT_MISMATCH",
        ),
        (
            evidence.derived_tie_break_commitment_digest
            == plan.tie_break_commitment_digest,
            "TIE_BREAK_COMMITMENT_MISMATCH",
        ),
    ):
        if not matches and reason not in reasons:
            reasons.append(reason)
    expected_candidates = set(plan.intake_cell_digests_by_candidate)
    if set(evidence.selection_receipt_digests_by_candidate) != expected_candidates:
        reasons.append("EXACT_90_SELECTION_RECEIPTS_MISSING")
    champion_map = evidence.champion_candidate_ids_by_anchor
    champions = set(champion_map.values())
    if (
        len(champion_map) != EXPECTED_ANCHOR_COUNT
        or set(champion_map) != plan.source_anchor_ids
        or len(champions) != EXPECTED_ANCHOR_COUNT
        or any(
            candidate not in expected_candidates
            or plan.source_anchor_id_by_candidate[candidate] != anchor
            for anchor, candidate in champion_map.items()
        )
        or set(evidence.champion_digests_by_anchor) != set(champion_map)
        or set(evidence.competence_observation_digests_by_anchor) != set(champion_map)
    ):
        reasons.append("EXACT_30_CHAMPIONS_MISSING")
    if set(evidence.attestation_receipt_digests_by_candidate) != champions:
        reasons.append("EXACT_30_ATTESTATION_RECEIPTS_MISSING")
    public_ids = set(evidence.public_entry_digests_by_opaque_id)
    private_ids = set(evidence.deployment_entry_digests_by_opaque_id)
    if (
        len(public_ids) != EXPECTED_ANCHOR_COUNT
        or public_ids != private_ids
        or public_ids != set(evidence.deployment_candidate_ids_by_opaque_id)
        or set(evidence.deployment_candidate_ids_by_opaque_id.values()) != champions
    ):
        reasons.append("EXACT_30_MARKET_ENTRIES_MISSING")
    expected_abis = {
        candidate: plan.deployment_abi_digests_by_candidate[candidate]
        for candidate in champions
        if candidate in plan.deployment_abi_digests_by_candidate
    }
    if dict(evidence.deployment_abi_digests_by_candidate) != expected_abis:
        reasons.append("DEPLOYMENT_ABI_REGISTRY_MISMATCH")
    for passed, reason in (
        (evidence.receipts_binding_pass, "SOURCE_RECEIPT_BINDING_FAILURE"),
        (evidence.market_binding_pass, "SOURCE_MARKET_BINDING_FAILURE"),
        (evidence.observe_mode_pass, "SOURCE_COMPETENCE_NOT_OBSERVE"),
        (
            evidence.nonce_commitment_binding_pass,
            "MARKET_NONCE_COMMITMENT_MISMATCH",
        ),
        (
            evidence.market_derivation_pass,
            "MARKET_ALIAS_TIE_DERIVATION_FAILURE",
        ),
    ):
        if not passed and reason not in reasons:
            reasons.append(reason)
    unique_reasons = tuple(dict.fromkeys(reasons))
    return FormalMarketAdmission(
        plan_digest=str(plan.plan_digest),
        evidence_digest=str(evidence.evidence_digest),
        authority_receipt_digest=str(authority.receipt_digest),
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        status="NO_GO" if unique_reasons else "ASSET_READY",
        failure_reasons=unique_reasons,
        candidate_count=len(evidence.selection_receipt_digests_by_candidate),
        market_entry_count=len(evidence.public_entry_digests_by_opaque_id),
    )


__all__ = [
    "FORMAL_ATTRIBUTION_INPUT_VIEW_IDS",
    "FORMAL_GATE_IDS",
    "FORMAL_GATE_PLAN_KEYS",
    "FormalAttributionAdmission",
    "FormalAttributionPlan",
    "FormalAttributionRecomputeEvidence",
    "FormalGateAuthorityReceipt",
    "FormalGateError",
    "FormalMarketAdmission",
    "FormalMarketEvidence",
    "FormalMarketPlan",
    "FormalProbeAdmission",
    "FormalProbePlan",
    "admit_formal_attribution",
    "admit_formal_market",
    "admit_formal_probe",
    "build_formal_attribution_recompute_evidence",
    "build_formal_market_evidence",
]
