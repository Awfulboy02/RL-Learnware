"""Fail-closed v0.2 completion and v0.3 hand-off publication.

This module owns only the development/freeze-ready endpoint.  It can publish a
scientific No-Go or ``READY_FOR_V03_JOINT_CONFIRMATORY`` record, but it has no
API for reading sealed target evidence, unlocking an oracle, or claiming a
Paper-I confirmatory result.

The legacy file-only completion entry point is intentionally non-authorizing:
JSON preserves provenance but cannot preserve the in-process authority of the
fixed evaluators.  The trusted-object entry point also fails closed in this
release because the gate evaluator and raw recompute loader registries are not
yet complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal, Mapping

from ..hashing import canonicalize, sha256_file, sha256_json
from ..io import read_json
from .artifacts import V02ArtifactLayout
from .config import load_v02_formal_config
from .gates import (
    FormalGateEvidenceError,
    FormalV02GateState,
    GATE_ORDER,
    GATE_REQUIREMENTS,
    GateCriterion,
    GateDecision,
    V02GateState,
    evaluate_gate_state,
    validate_formal_gate_state_payload,
)


TheoryStatus = Literal["PENDING", "MINIMAL_FINITE_POOL_CLOSED"]
THEORY_STATUSES = frozenset({"PENDING", "MINIMAL_FINITE_POOL_CLOSED"})
COMPLETION_SCHEMA = "policy-learnware.v02-completion-manifest.v0"
_SAFE_STATUS = re.compile(r"^[A-Z][A-Z0-9_.-]{0,127}$")
_FORBIDDEN_INPUT_PARTS = frozenset(
    {
        "artifacts_paper1_joint",
        "confirmatory_oracle_private",
        "sealed_targets",
        "sealed_target_transitions",
    }
)


class V02CompletionError(ValueError):
    """Persisted evidence cannot authorize a v0.2 completion record."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V02CompletionError(f"{where} must be a JSON object")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    observed = set(value)
    if observed != expected:
        raise V02CompletionError(
            f"{where} fields differ: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )


def _safe_input_file(path: str | Path, where: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise V02CompletionError(f"{where} cannot be a symlink")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise V02CompletionError(f"{where} is not a file: {resolved}")
    parts = {part.lower() for part in resolved.parts}
    forbidden = parts & _FORBIDDEN_INPUT_PARTS
    if forbidden:
        raise V02CompletionError(
            f"v0.2 completion cannot read joint/sealed/oracle path components: "
            f"{sorted(forbidden)}"
        )
    return resolved


def _parse_gate_decision(value: Any, expected_gate: str) -> GateDecision:
    data = _mapping(value, f"gate decision {expected_gate}")
    _strict(
        data,
        {
            "schema",
            "gate",
            "outcome",
            "passed",
            "fail_closed",
            "criteria",
            "missing_checks",
            "unexpected_checks",
        },
        f"gate decision {expected_gate}",
    )
    if data["schema"] != "policy-learnware.v02-gate-decision.v0":
        raise V02CompletionError("unsupported v0.2 gate-decision schema")
    if data["gate"] != expected_gate or data["fail_closed"] is not True:
        raise V02CompletionError("gate identity/fail-closed marker mismatch")
    raw_criteria = data["criteria"]
    if not isinstance(raw_criteria, list):
        raise V02CompletionError("gate criteria must be a list")
    expected_names = GATE_REQUIREMENTS[expected_gate]  # type: ignore[index]
    if len(raw_criteria) != len(expected_names):
        raise V02CompletionError("gate criterion coverage differs from the registry")
    criteria: list[GateCriterion] = []
    for raw, expected_name in zip(raw_criteria, expected_names, strict=True):
        item = _mapping(raw, f"gate criterion {expected_name}")
        _strict(item, {"name", "observed", "passed", "reason"}, "gate criterion")
        if item["name"] != expected_name:
            raise V02CompletionError("gate criteria are not in registered order")
        observed = item["observed"]
        if observed is not None and type(observed) is not bool:
            raise V02CompletionError("persisted gate observations must be bool or null")
        derived_passed = observed is True
        if item["passed"] is not derived_passed:
            raise V02CompletionError("gate criterion passed flag is not derived")
        if derived_passed:
            allowed_reasons = {None}
        elif observed is False:
            allowed_reasons = {"primitive_check_failed"}
        else:
            allowed_reasons = {"missing_required_check", "check_is_not_boolean"}
        if item["reason"] not in allowed_reasons:
            raise V02CompletionError("gate criterion reason is inconsistent")
        criteria.append(
            GateCriterion(
                name=expected_name,
                observed=observed,
                passed=derived_passed,
                reason=item["reason"],
            )
        )
    unexpected = data["unexpected_checks"]
    if (
        not isinstance(unexpected, list)
        or any(not isinstance(item, str) or not item for item in unexpected)
        or len(unexpected) != len(set(unexpected))
        or unexpected != sorted(unexpected)
    ):
        raise V02CompletionError("unexpected gate checks must be sorted unique strings")
    decision = GateDecision(
        gate=expected_gate,  # type: ignore[arg-type]
        criteria=tuple(criteria),
        unexpected_checks=tuple(unexpected),
    )
    if canonicalize(data) != decision.to_dict():
        raise V02CompletionError("persisted gate decision differs from recomputation")
    return decision


def validate_gate_state_payload(value: Any) -> V02GateState:
    """Rebuild a non-formal/core gate state for diagnostics and CPU fixtures.

    Completion deliberately does not call this compatibility parser.  A core
    state contains primitive booleans but no artifact provenance and therefore
    cannot authorize a formal v0.2 endpoint.
    """

    data = _mapping(value, "gate state")
    _strict(
        data,
        {
            "schema",
            "status",
            "ready_for_v03",
            "passed_gates",
            "blocking_gate",
            "invalid_gate_inputs",
            "decisions",
        },
        "gate state",
    )
    if data["schema"] != "policy-learnware.v02-gate-state.v0":
        raise V02CompletionError("unsupported v0.2 gate-state schema")
    raw_decisions = data["decisions"]
    if not isinstance(raw_decisions, list) or len(raw_decisions) != len(GATE_ORDER):
        raise V02CompletionError(
            "gate state must contain every registered gate exactly once"
        )
    decisions = tuple(
        _parse_gate_decision(raw, gate)
        for raw, gate in zip(raw_decisions, GATE_ORDER, strict=True)
    )
    state = evaluate_gate_state(decisions)
    if state.invalid_gate_inputs:
        raise V02CompletionError("invalid gate inputs cannot authorize completion")
    if canonicalize(data) != state.to_dict():
        raise V02CompletionError(
            "persisted gate state differs from independent derivation"
        )
    return state


@dataclass(frozen=True)
class V02CompletionRecord:
    experiment_id: str
    config_digest: str
    config_file_sha256: str
    gate_state: FormalV02GateState
    gate_state_digest: str
    gate_state_file_sha256: str
    recompute_audit_digest: str
    recompute_audit_file_sha256: str
    theory_status: TheoryStatus
    literature_novelty_audit_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.gate_state, FormalV02GateState):
            raise V02CompletionError(
                "completion requires a formal evidence-bound gate state"
            )
        try:
            self.gate_state.require_formal_authority()
        except FormalGateEvidenceError as error:
            raise V02CompletionError(
                "completion requires gate results produced by trusted "
                "in-process content evaluators"
            ) from error
        if self.gate_state.config_digest != self.config_digest:
            raise V02CompletionError(
                "completion config and formal gate-state config differ"
            )
        if self.gate_state.experiment_id != self.experiment_id:
            raise V02CompletionError(
                "completion experiment and formal gate-state experiment differ"
            )
        if self.theory_status not in THEORY_STATUSES:
            raise V02CompletionError(
                f"unsupported theory status: {self.theory_status!r}"
            )
        if (
            not isinstance(self.literature_novelty_audit_status, str)
            or _SAFE_STATUS.fullmatch(self.literature_novelty_audit_status) is None
        ):
            raise V02CompletionError(
                "literature_novelty_audit_status must be an explicit canonical status"
            )
        for name in (
            "config_digest",
            "config_file_sha256",
            "gate_state_digest",
            "gate_state_file_sha256",
            "recompute_audit_digest",
            "recompute_audit_file_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise V02CompletionError(f"{name} must be a SHA-256 digest")
            try:
                int(value, 16)
            except ValueError as error:
                raise V02CompletionError(f"{name} must be a SHA-256 digest") from error
        if self.gate_state_digest != self.gate_state.digest:
            raise V02CompletionError(
                "completion gate_state_digest differs from the formal gate state"
            )

    @property
    def status(self) -> str:
        return self.gate_state.status

    @property
    def v02_software_complete(self) -> bool:
        return self.status != "BLOCKED_ENGINEERING"

    def to_dict(self) -> dict[str, Any]:
        if self.status == "READY_FOR_V03_JOINT_CONFIRMATORY":
            interpretation = (
                "The v0.2 frozen market, L-min, baselines, and replaceable encoder "
                "interface are ready for a future v0.3-owned joint freeze."
            )
        elif self.status.startswith("COMPLETE_NO_GO"):
            interpretation = "The v0.2 engineering protocol completed with a retained scientific No-Go."
        else:
            interpretation = (
                "The v0.2 run is blocked by an engineering or isolation failure."
            )
        return {
            "schema": COMPLETION_SCHEMA,
            "experiment_id": self.experiment_id,
            "stage": "v02_freeze_ready",
            "status": self.status,
            "v02_software_complete": self.v02_software_complete,
            "ready_for_v03": self.gate_state.ready_for_v03,
            "paper1_empirical_claim_authorized": False,
            "sealed_target_state": "NOT_INSTANTIATED_OR_READ",
            "confirmatory_oracle_state": "NOT_READ",
            "interpretation": interpretation,
            "theory_status": self.theory_status,
            "literature_novelty_audit_status": self.literature_novelty_audit_status,
            "evidence": {
                "config_digest": self.config_digest,
                "config_file_sha256": self.config_file_sha256,
                "gate_state_digest": self.gate_state_digest,
                "gate_state_file_sha256": self.gate_state_file_sha256,
                "gate_evidence_manifest_digest": (
                    self.gate_state.gate_evidence_manifest_digest
                ),
                "gate_evidence_manifest_file_sha256": (
                    self.gate_state.gate_evidence_manifest_file_sha256
                ),
                "recompute_audit_digest": self.recompute_audit_digest,
                "recompute_audit_file_sha256": self.recompute_audit_file_sha256,
            },
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def complete_v02_from_trusted_results(
    *,
    config_path: str | Path,
    gate_state: FormalV02GateState,
    recompute_report: Any,
    gate_state_path: str | Path,
    recompute_audit_path: str | Path,
    output_path: str | Path,
    theory_status: str,
    literature_novelty_audit_status: str,
    resume: bool = False,
) -> V02CompletionRecord:
    """Publish only from live trusted evaluator/recompute result objects.

    Persisted artifacts remain mandatory and are re-read byte-for-byte for an
    immutable audit trail, but they do not confer authority.  That authority is
    carried only by the live objects returned by the source-owned gate registry
    and :func:`run_formal_independent_recompute` in the current process.
    """

    from .recompute import (
        IndependentRecomputeReport,
        RecomputeContractError,
        validate_formal_recompute_report_payload,
    )

    config_file = _safe_input_file(config_path, "config_path")
    config = load_v02_formal_config(config_file)
    config_file_digest = sha256_file(config_file)
    layout = V02ArtifactLayout(Path(config.artifact_root), config.experiment_id)
    gate_file = _safe_input_file(gate_state_path, "gate_state_path")
    recompute_file = _safe_input_file(recompute_audit_path, "recompute_audit_path")
    if gate_file != layout.gate_artifact("v02_gate_state.json").resolve():
        raise V02CompletionError("gate state does not use its canonical path")
    if recompute_file != layout.recompute_audit.resolve():
        raise V02CompletionError("recompute audit does not use its canonical path")
    if not isinstance(gate_state, FormalV02GateState):
        raise V02CompletionError(
            "trusted completion requires a typed formal gate state"
        )
    try:
        gate_state.require_formal_authority()
    except FormalGateEvidenceError as error:
        raise V02CompletionError(
            "trusted completion received an untrusted gate-state object"
        ) from error
    if (
        gate_state.experiment_id != config.experiment_id
        or gate_state.config_digest != config.config_digest
    ):
        raise V02CompletionError("live gate state is bound to another run/config")
    gate_payload = read_json(gate_file)
    try:
        rederived_gate = validate_formal_gate_state_payload(
            gate_payload,
            experiment_root=layout.experiment_root,
            expected_experiment_id=config.experiment_id,
            expected_config_digest=config.config_digest,
        )
    except FormalGateEvidenceError as error:
        raise V02CompletionError(
            f"formal gate evidence validation failed: {error}"
        ) from error
    if rederived_gate.digest != gate_state.digest:
        raise V02CompletionError(
            "persisted gate state differs from the live trusted evaluation"
        )

    if not isinstance(recompute_report, IndependentRecomputeReport):
        raise V02CompletionError("trusted completion requires a typed recompute report")
    recompute_payload = read_json(recompute_file)
    try:
        archived_report = validate_formal_recompute_report_payload(
            recompute_payload,
            experiment_root=layout.experiment_root,
            expected_experiment_id=config.experiment_id,
            expected_config_digest=config.config_digest,
            expected_config_file_sha256=config_file_digest,
        )
        recompute_report.require_formal_authority(
            expected_experiment_id=config.experiment_id,
            expected_config_digest=config.config_digest,
            expected_config_file_sha256=config_file_digest,
        )
    except RecomputeContractError as error:
        raise V02CompletionError(
            f"formal recompute evidence validation failed: {error}"
        ) from error
    if archived_report.digest != recompute_report.digest:
        raise V02CompletionError(
            "persisted recompute report differs from the live trusted replay"
        )

    output = Path(output_path).expanduser().resolve()
    if output != layout.completion_manifest.resolve():
        raise V02CompletionError(
            f"completion output must be the canonical path {layout.completion_manifest.resolve()}"
        )
    record = V02CompletionRecord(
        experiment_id=config.experiment_id,
        config_digest=config.config_digest,
        config_file_sha256=config_file_digest,
        gate_state=gate_state,
        gate_state_digest=sha256_json(gate_payload),
        gate_state_file_sha256=sha256_file(gate_file),
        recompute_audit_digest=recompute_report.digest,
        recompute_audit_file_sha256=sha256_file(recompute_file),
        theory_status=theory_status,  # type: ignore[arg-type]
        literature_novelty_audit_status=literature_novelty_audit_status,
    )
    layout.writer("completion").publish_json(
        layout.completion_manifest,
        record.to_dict(),
        resume=resume,
    )
    return record


def complete_v02_from_files(
    *,
    config_path: str | Path,
    gate_state_path: str | Path,
    recompute_audit_path: str | Path,
    output_path: str | Path,
    theory_status: str,
    literature_novelty_audit_status: str,
    resume: bool = False,
) -> V02CompletionRecord:
    """Validate persisted evidence and immutably publish one v0.2 endpoint."""

    config_file = _safe_input_file(config_path, "config_path")
    config = load_v02_formal_config(config_file)
    layout = V02ArtifactLayout(Path(config.artifact_root), config.experiment_id)
    gate_file = _safe_input_file(gate_state_path, "gate_state_path")
    recompute_file = _safe_input_file(recompute_audit_path, "recompute_audit_path")
    expected_gate = layout.gate_artifact("v02_gate_state.json").resolve()
    expected_recompute = layout.recompute_audit.resolve()
    if gate_file != expected_gate:
        raise V02CompletionError(f"gate state must use canonical path {expected_gate}")
    if recompute_file != expected_recompute:
        raise V02CompletionError(
            f"recompute audit must use canonical path {expected_recompute}"
        )
    gate_payload = read_json(gate_file)
    try:
        gate_state = validate_formal_gate_state_payload(
            gate_payload,
            experiment_root=layout.experiment_root,
            expected_experiment_id=config.experiment_id,
            expected_config_digest=config.config_digest,
        )
    except FormalGateEvidenceError as error:
        raise V02CompletionError(
            f"formal gate evidence validation failed: {error}"
        ) from error

    # Delayed import keeps config/completion inspection CPU-only and permits the
    # recompute implementation to own its strict persisted schema.
    from .recompute import (
        RecomputeContractError,
        validate_formal_recompute_report_payload,
    )

    recompute_payload = read_json(recompute_file)
    try:
        report = validate_formal_recompute_report_payload(
            recompute_payload,
            experiment_root=layout.experiment_root,
            expected_experiment_id=config.experiment_id,
            expected_config_digest=config.config_digest,
            expected_config_file_sha256=sha256_file(config_file),
        )
        report.require_passed()
        # JSON can preserve provenance, but not the live authority attached by
        # run_formal_independent_recompute.  The legacy file-only completion API
        # therefore remains a diagnostic/compatibility surface and fails closed
        # until a caller supplies the trusted live objects to the object API.
        report.require_formal_authority(
            expected_experiment_id=config.experiment_id,
            expected_config_digest=config.config_digest,
            expected_config_file_sha256=sha256_file(config_file),
        )
    except RecomputeContractError as error:
        raise V02CompletionError(
            f"formal recompute evidence validation failed: {error}"
        ) from error

    output = Path(output_path).expanduser().resolve()
    expected_output = layout.completion_manifest.resolve()
    if output != expected_output:
        raise V02CompletionError(
            f"completion output must be the canonical path {expected_output}"
        )
    record = V02CompletionRecord(
        experiment_id=config.experiment_id,
        config_digest=config.config_digest,
        config_file_sha256=sha256_file(config_file),
        gate_state=gate_state,
        gate_state_digest=sha256_json(gate_payload),
        gate_state_file_sha256=sha256_file(gate_file),
        recompute_audit_digest=report.digest,
        recompute_audit_file_sha256=sha256_file(recompute_file),
        theory_status=theory_status,  # type: ignore[arg-type]
        literature_novelty_audit_status=literature_novelty_audit_status,
    )
    layout.writer("completion").publish_json(
        layout.completion_manifest,
        record.to_dict(),
        resume=resume,
    )
    return record


__all__ = [
    "COMPLETION_SCHEMA",
    "THEORY_STATUSES",
    "TheoryStatus",
    "V02CompletionError",
    "V02CompletionRecord",
    "complete_v02_from_files",
    "complete_v02_from_trusted_results",
    "validate_gate_state_payload",
    "validate_formal_gate_state_payload",
]
