"""Final, authority-owned claim audit for a completed v0.3 formal run.

The v0.3 driver does not decide which scientific claims are warranted.  That
decision belongs to the external Paper-I review owner.  This module only gives
the owner a strict, digest-bound record that joins every prerequisite used by
the final completion checker.  Consequently a claim state cannot be carried
from another run, freeze, signal bundle, market admission, or recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..hashing import sha256_json
from .schemas import checked_digest, checked_safe_id, strict_mapping


FORMAL_CLAIM_AUDIT_SCHEMA = "policy-learnware.v03-formal-claim-audit.v0"

CompletionState = Literal[
    "BLOCKED_ENGINEERING",
    "COMPLETE_NO_GO_PROBE",
    "COMPLETE_NO_GO_TRANSITION_SIGNAL",
    "COMPLETE_SIGNAL_ATLAS_ONLY",
    "COMPLETE_NO_GO_MLP_GAIN",
    "COMPLETE_TASK_ONLY",
    "COMPLETE_GO_SIGNAL_SPEC",
    "COMPLETE_GO_PAPER_I",
]

COMPLETION_STATES = (
    "BLOCKED_ENGINEERING",
    "COMPLETE_NO_GO_PROBE",
    "COMPLETE_NO_GO_TRANSITION_SIGNAL",
    "COMPLETE_SIGNAL_ATLAS_ONLY",
    "COMPLETE_NO_GO_MLP_GAIN",
    "COMPLETE_TASK_ONLY",
    "COMPLETE_GO_SIGNAL_SPEC",
    "COMPLETE_GO_PAPER_I",
)


class FormalClaimAuditError(ValueError):
    """A final claim audit is malformed, incomplete, or cross-run."""


def _digest(value: Any, where: str) -> str:
    try:
        return checked_digest(value, where)
    except ValueError as error:
        raise FormalClaimAuditError(str(error)) from error


def _id(value: Any, where: str) -> str:
    try:
        return checked_safe_id(value, where)
    except ValueError as error:
        raise FormalClaimAuditError(str(error)) from error


@dataclass(frozen=True)
class FormalClaimAudit:
    """External review decision over the exact final formal evidence chain."""

    run_id: str
    freeze_manifest_digest: str
    attribution_admission_digest: str
    probe_admission_digest: str
    market_admission_digest: str
    signal_readout_bundle_digest: str
    cost_ledger_digest: str
    preoracle_signal_outcome_digest: str
    pre_oracle_signal_manifest_digest: str
    public_ranking_barrier_digest: str
    statistics_result_digest: str
    independent_recompute_attestation_digest: str
    completion_state: CompletionState
    allowed_claim_ids: tuple[str, ...]
    review_authority_receipt_digest: str
    audit_digest: str | None = None
    schema: str = FORMAL_CLAIM_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_CLAIM_AUDIT_SCHEMA:
            raise FormalClaimAuditError("unsupported FormalClaimAudit schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "freeze_manifest_digest",
            "attribution_admission_digest",
            "probe_admission_digest",
            "market_admission_digest",
            "signal_readout_bundle_digest",
            "cost_ledger_digest",
            "preoracle_signal_outcome_digest",
            "pre_oracle_signal_manifest_digest",
            "public_ranking_barrier_digest",
            "statistics_result_digest",
            "independent_recompute_attestation_digest",
            "review_authority_receipt_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.completion_state not in COMPLETION_STATES:
            raise FormalClaimAuditError("unknown v0.3 completion state")
        claim_ids = tuple(sorted(_id(item, "allowed_claim_id") for item in self.allowed_claim_ids))
        if len(set(claim_ids)) != len(claim_ids):
            raise FormalClaimAuditError("allowed claim IDs must be unique")
        if self.completion_state == "BLOCKED_ENGINEERING" and claim_ids:
            raise FormalClaimAuditError(
                "BLOCKED_ENGINEERING cannot authorize scientific claims"
            )
        object.__setattr__(self, "allowed_claim_ids", claim_ids)
        expected = sha256_json(self._payload_without_digest())
        if self.audit_digest is None:
            object.__setattr__(self, "audit_digest", expected)
        elif _digest(self.audit_digest, "audit_digest") != expected:
            raise FormalClaimAuditError("formal claim audit digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "attribution_admission_digest": self.attribution_admission_digest,
            "probe_admission_digest": self.probe_admission_digest,
            "market_admission_digest": self.market_admission_digest,
            "signal_readout_bundle_digest": self.signal_readout_bundle_digest,
            "cost_ledger_digest": self.cost_ledger_digest,
            "preoracle_signal_outcome_digest": (
                self.preoracle_signal_outcome_digest
            ),
            "pre_oracle_signal_manifest_digest": (
                self.pre_oracle_signal_manifest_digest
            ),
            "public_ranking_barrier_digest": self.public_ranking_barrier_digest,
            "statistics_result_digest": self.statistics_result_digest,
            "independent_recompute_attestation_digest": (
                self.independent_recompute_attestation_digest
            ),
            "completion_state": self.completion_state,
            "allowed_claim_ids": list(self.allowed_claim_ids),
            "review_authority_receipt_digest": (
                self.review_authority_receipt_digest
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "audit_digest": self.audit_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalClaimAudit":
        try:
            data = strict_mapping(
                value, set(cls.__dataclass_fields__), "formal claim audit"
            )
        except ValueError as error:
            raise FormalClaimAuditError(str(error)) from error
        return cls(
            **{
                field: (
                    tuple(data[field]) if field == "allowed_claim_ids" else data[field]
                )
                for field in cls.__dataclass_fields__
            }
        )


__all__ = [
    "COMPLETION_STATES",
    "FORMAL_CLAIM_AUDIT_SCHEMA",
    "CompletionState",
    "FormalClaimAudit",
    "FormalClaimAuditError",
]
