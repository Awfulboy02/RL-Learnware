"""Fail-closed v0.2 development gates and completion-state transitions.

The five gates in this module cover the v0.2-owned path only.  In particular,
passing them ends at ``READY_FOR_V03_JOINT_CONFIRMATORY``; it never manufactures
a Paper-I confirmatory success state.  The dependency-light core evaluator is
used by CPU acceptance.  Formal evaluation additionally requires an exact
criterion-to-artifact evidence graph and re-reads every byte/hash/config binding;
a naked ``passed`` flag, missing reference, or unknown check cannot authorize
formal progress.  The formal registry is intentionally empty: none of the 35
criteria yet has both a source-owned primitive inverse and an exact binding to
the reviewed config/freeze.  Formal ``READY`` therefore remains unreachable
until those production collectors and bindings are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence

from ..hashing import canonicalize, sha256_file, sha256_json
from ..io import read_json


GateName = Literal[
    "G02-Scope",
    "G02-Engineering",
    "G02-Market",
    "G02-Replace",
    "G02-SplitFreeze",
]
GateOutcome = Literal["PASS", "FAIL"]
V02CompletionStatus = Literal[
    "BLOCKED_ENGINEERING",
    "COMPLETE_NO_GO_MARKET",
    "COMPLETE_NO_GO_CORRO_INCUMBENT",
    "READY_FOR_V03_JOINT_CONFIRMATORY",
]

GATE_DECISION_SCHEMA = "policy-learnware.v02-gate-decision.v0"
GATE_STATE_SCHEMA = "policy-learnware.v02-gate-state.v0"
FORMAL_GATE_DECISION_SCHEMA = "policy-learnware.v02-formal-gate-decision.v0"
FORMAL_GATE_STATE_SCHEMA = "policy-learnware.v02-formal-gate-state.v0"
GATE_EVIDENCE_MANIFEST_SCHEMA = "policy-learnware.v02-gate-evidence-manifest.v0"
GATE_CRITERION_EVIDENCE_SCHEMA = "policy-learnware.v02-gate-criterion-evidence.v0"
GATE_EVALUATOR_REGISTRY_SCHEMA = "policy-learnware.v02-gate-evaluator-registry.v0"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_GATE_PATH_COMPONENT = MappingProxyType(
    {
        "G02-Scope": "scope",
        "G02-Engineering": "engineering",
        "G02-Market": "market",
        "G02-Replace": "replace",
        "G02-SplitFreeze": "split-freeze",
    }
)
_FORBIDDEN_EVIDENCE_PATH_PARTS = frozenset(
    {
        "artifacts_paper1_joint",
        "confirmatory_oracle_private",
        "sealed_targets",
        "sealed_target_transitions",
    }
)
_FORBIDDEN_SOURCE_SCHEMAS = frozenset(
    {
        GATE_CRITERION_EVIDENCE_SCHEMA,
        GATE_EVIDENCE_MANIFEST_SCHEMA,
        FORMAL_GATE_DECISION_SCHEMA,
        FORMAL_GATE_STATE_SCHEMA,
        GATE_DECISION_SCHEMA,
        GATE_STATE_SCHEMA,
        "policy-learnware.v02-completion-manifest.v0",
    }
)


GATE_ORDER: tuple[GateName, ...] = (
    "G02-Scope",
    "G02-Engineering",
    "G02-Market",
    "G02-Replace",
    "G02-SplitFreeze",
)

G02_SCOPE_CHECKS = (
    "scope_frozen",
    "anonymous_full_market",
    "source_anchor_frozen_market",
    "candidate_independent_target_probes",
    "zero_target_policy_update",
    "task_reward_schema_not_selector_input",
)
G02_ENGINEERING_CHECKS = (
    "six_task_runtime",
    "all_registered_axis_audits",
    "source_anchor_manifests",
    "bundle_runtime_parity",
    "artifact_capability_domains",
    "immutable_resume",
    "extension_conformance",
)
G02_MARKET_CHECKS = (
    "all_anchor_specialists_competent",
    "development_gain_meets_threshold",
    "development_gain_ci_excludes_zero",
    "pool_quality_floor",
    "coverage_at_least_two_tasks",
    "coverage_at_least_two_axes",
    "championization_frozen",
)
G02_REPLACE_CHECKS = (
    "raw_encoder_conformance",
    "corro_incumbent_conformance",
    "synthetic_future_encoder_conformance",
    "selector_implementation_unchanged",
    "corro_signal_above_bank_noise",
    "raw_vs_corro_diagnostic_complete",
)
G02_SPLIT_FREEZE_CHECKS = (
    "market_frozen",
    "sealed_target_generator_frozen",
    "probe_interface_and_panel_frozen",
    "incumbent_and_source_reference_rules_frozen",
    "baseline_algorithms_and_hyperparameters_frozen",
    "primary_statistics_and_cost_contracts_frozen",
    "sealed_targets_not_instantiated_or_read",
    "confirmatory_oracle_not_read",
    "information_isolation_audit_passed",
)

GATE_REQUIREMENTS: Mapping[GateName, tuple[str, ...]] = MappingProxyType(
    {
        "G02-Scope": G02_SCOPE_CHECKS,
        "G02-Engineering": G02_ENGINEERING_CHECKS,
        "G02-Market": G02_MARKET_CHECKS,
        "G02-Replace": G02_REPLACE_CHECKS,
        "G02-SplitFreeze": G02_SPLIT_FREEZE_CHECKS,
    }
)


class FormalGateEvidenceError(ValueError):
    """Formal gate provenance is missing, misbound, or byte-inconsistent."""


# This capability is deliberately not serialized.  Loading a JSON document can
# reconstruct an evidence graph for diagnostics, but cannot manufacture proof
# that the source-owned evaluator code actually ran in this process.
_TRUSTED_GATE_EVALUATION_AUTHORITY = object()


def _digest(value: object, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FormalGateEvidenceError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _strict_mapping(value: object, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalGateEvidenceError(f"{where} must be a JSON object")
    observed = set(value)
    if observed != expected:
        raise FormalGateEvidenceError(
            f"{where} fields differ: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )
    return value


def _experiment_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or _SAFE_EXPERIMENT_ID.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise FormalGateEvidenceError("experiment_id is not a safe path segment")
    return value


def _canonical_relative_path(value: object, where: str) -> str:
    """Return an exact POSIX relative path without normalising caller input."""

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise FormalGateEvidenceError(f"{where} is not a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise FormalGateEvidenceError(f"{where} is not a canonical relative path")
    canonical = path.as_posix()
    if canonical != value:
        raise FormalGateEvidenceError(f"{where} is not in canonical POSIX form")
    forbidden = {part.lower() for part in path.parts} & _FORBIDDEN_EVIDENCE_PATH_PARTS
    if forbidden:
        raise FormalGateEvidenceError(
            f"{where} crosses a sealed/joint/oracle-private boundary: {sorted(forbidden)}"
        )
    return canonical


def criterion_evidence_relative_path(gate: GateName, criterion: str) -> str:
    """Canonical location of one formal criterion evidence artifact."""

    if gate not in GATE_REQUIREMENTS:
        raise FormalGateEvidenceError(f"unknown gate: {gate!r}")
    if criterion not in GATE_REQUIREMENTS[gate]:
        raise FormalGateEvidenceError(
            f"criterion {criterion!r} is not registered for {gate}"
        )
    return f"analysis/gates/evidence/{_GATE_PATH_COMPONENT[gate]}/{criterion}.json"


def gate_evidence_manifest_relative_path() -> str:
    return "analysis/gates/v02_gate_evidence_manifest.json"


def _resolve_evidence_path(
    experiment_root: str | Path, relative_path: str, where: str
) -> Path:
    root = Path(experiment_root).expanduser().resolve()
    if not root.is_dir():
        raise FormalGateEvidenceError(f"experiment root is not a directory: {root}")
    canonical = _canonical_relative_path(relative_path, where)
    lexical = root.joinpath(*PurePosixPath(canonical).parts)
    current = root
    for part in PurePosixPath(canonical).parts:
        current = current / part
        if current.is_symlink():
            raise FormalGateEvidenceError(f"{where} cannot traverse a symlink")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise FormalGateEvidenceError(f"{where} escapes the experiment root") from error
    if not resolved.is_file():
        raise FormalGateEvidenceError(f"{where} is not an evidence file: {resolved}")
    return resolved


@dataclass(frozen=True)
class CanonicalEvidenceRef:
    """Byte and semantic hash reference to one JSON artifact under an experiment.

    ``artifact_digest`` is the canonical-JSON digest while ``file_sha256`` binds
    the exact persisted bytes.  A nullable ``config_digest`` is explicit: it is
    required whenever the referenced artifact itself carries that field.
    """

    canonical_path: str
    file_sha256: str
    artifact_digest: str
    artifact_schema: str
    config_digest: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canonical_path",
            _canonical_relative_path(self.canonical_path, "evidence canonical_path"),
        )
        object.__setattr__(
            self, "file_sha256", _digest(self.file_sha256, "evidence file_sha256")
        )
        object.__setattr__(
            self,
            "artifact_digest",
            _digest(self.artifact_digest, "evidence artifact_digest"),
        )
        if (
            not isinstance(self.artifact_schema, str)
            or not self.artifact_schema
            or len(self.artifact_schema) > 256
        ):
            raise FormalGateEvidenceError("evidence artifact_schema is invalid")
        if self.config_digest is not None:
            object.__setattr__(
                self,
                "config_digest",
                _digest(self.config_digest, "evidence config_digest"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_path": self.canonical_path,
            "file_sha256": self.file_sha256,
            "artifact_digest": self.artifact_digest,
            "artifact_schema": self.artifact_schema,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CanonicalEvidenceRef":
        data = _strict_mapping(
            value,
            {
                "canonical_path",
                "file_sha256",
                "artifact_digest",
                "artifact_schema",
                "config_digest",
            },
            "canonical evidence reference",
        )
        return cls(
            canonical_path=data["canonical_path"],
            file_sha256=data["file_sha256"],
            artifact_digest=data["artifact_digest"],
            artifact_schema=data["artifact_schema"],
            config_digest=data["config_digest"],
        )


def build_canonical_evidence_ref(
    path: str | Path,
    *,
    experiment_root: str | Path,
    expected_config_digest: str | None = None,
) -> CanonicalEvidenceRef:
    """Build a reference from an already-persisted canonical JSON artifact."""

    root = Path(experiment_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise FormalGateEvidenceError("evidence artifact cannot be a symlink")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise FormalGateEvidenceError(
            "evidence artifact is outside the experiment root"
        ) from error
    resolved = _resolve_evidence_path(root, relative, "evidence artifact")
    payload = read_json(resolved)
    if not isinstance(payload, Mapping):
        raise FormalGateEvidenceError("evidence artifact must contain a JSON object")
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema:
        raise FormalGateEvidenceError("evidence artifact has no typed schema")
    payload_config = payload.get("config_digest")
    if "config_digest" in payload:
        payload_config = _digest(payload_config, "artifact config_digest")
        if expected_config_digest is not None and payload_config != _digest(
            expected_config_digest, "expected config_digest"
        ):
            raise FormalGateEvidenceError(
                "evidence artifact is bound to another config"
            )
    else:
        payload_config = None
    return CanonicalEvidenceRef(
        canonical_path=relative,
        file_sha256=sha256_file(resolved),
        artifact_digest=sha256_json(payload),
        artifact_schema=schema,
        config_digest=payload_config,
    )


def verify_canonical_evidence_ref(
    reference: CanonicalEvidenceRef,
    *,
    experiment_root: str | Path,
    expected_config_digest: str,
    require_config_binding: bool,
    source_artifact: bool = False,
) -> Mapping[str, Any]:
    """Reload and verify one reference; return the exact parsed JSON object."""

    expected_config = _digest(expected_config_digest, "expected config_digest")
    path = _resolve_evidence_path(
        experiment_root, reference.canonical_path, "evidence reference"
    )
    if sha256_file(path) != reference.file_sha256:
        raise FormalGateEvidenceError(
            f"evidence file bytes changed: {reference.canonical_path}"
        )
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise FormalGateEvidenceError("referenced evidence must be a JSON object")
    if sha256_json(payload) != reference.artifact_digest:
        raise FormalGateEvidenceError(
            f"evidence artifact digest changed: {reference.canonical_path}"
        )
    if payload.get("schema") != reference.artifact_schema:
        raise FormalGateEvidenceError(
            f"evidence schema differs from its reference: {reference.canonical_path}"
        )
    carries_config = "config_digest" in payload
    if carries_config:
        if payload["config_digest"] != expected_config:
            raise FormalGateEvidenceError(
                f"evidence artifact is bound to another config: {reference.canonical_path}"
            )
        if reference.config_digest != expected_config:
            raise FormalGateEvidenceError(
                "evidence ref omits or changes its config binding"
            )
    elif reference.config_digest is not None:
        raise FormalGateEvidenceError(
            "evidence ref claims a config binding absent from the artifact"
        )
    if require_config_binding and not carries_config:
        raise FormalGateEvidenceError(
            f"formal gate artifact lacks config_digest: {reference.canonical_path}"
        )
    if source_artifact:
        if reference.artifact_schema in _FORBIDDEN_SOURCE_SCHEMAS:
            raise FormalGateEvidenceError(
                "a gate/completion artifact cannot serve as its own source evidence"
            )
        if not reference.artifact_schema.startswith("policy-learnware.v02-"):
            raise FormalGateEvidenceError(
                "source evidence schema is outside the typed v0.2 artifact family"
            )
    return payload


@dataclass(frozen=True)
class GateCriterionEvidence:
    """A typed, config-bound criterion result with non-circular source refs."""

    config_digest: str
    gate: GateName
    criterion: str
    passed: bool
    derivation_id: str
    evaluator_digest: str
    source_artifacts: tuple[CanonicalEvidenceRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config_digest",
            _digest(self.config_digest, "criterion config_digest"),
        )
        if self.gate not in GATE_REQUIREMENTS:
            raise FormalGateEvidenceError(f"unknown criterion gate: {self.gate!r}")
        if self.criterion not in GATE_REQUIREMENTS[self.gate]:
            raise FormalGateEvidenceError(
                f"unregistered criterion {self.criterion!r} for {self.gate}"
            )
        if type(self.passed) is not bool:
            raise FormalGateEvidenceError("criterion passed must be a boolean")
        expected_derivation = (
            f"policy-learnware.v02.gate/{self.gate}/{self.criterion}/v0"
        )
        if self.derivation_id != expected_derivation:
            raise FormalGateEvidenceError("criterion derivation_id is not canonical")
        object.__setattr__(
            self,
            "evaluator_digest",
            _digest(self.evaluator_digest, "criterion evaluator_digest"),
        )
        sources = tuple(self.source_artifacts)
        if not sources or any(
            not isinstance(item, CanonicalEvidenceRef) for item in sources
        ):
            raise FormalGateEvidenceError(
                "criterion evidence requires one or more typed source artifact refs"
            )
        if len({item.canonical_path for item in sources}) != len(sources):
            raise FormalGateEvidenceError(
                "criterion source artifact paths must be unique"
            )
        object.__setattr__(self, "source_artifacts", sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_CRITERION_EVIDENCE_SCHEMA,
            "config_digest": self.config_digest,
            "gate": self.gate,
            "criterion": self.criterion,
            "passed": self.passed,
            "derivation_id": self.derivation_id,
            "evaluator_digest": self.evaluator_digest,
            "source_artifacts": [item.to_dict() for item in self.source_artifacts],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "GateCriterionEvidence":
        data = _strict_mapping(
            value,
            {
                "schema",
                "config_digest",
                "gate",
                "criterion",
                "passed",
                "derivation_id",
                "evaluator_digest",
                "source_artifacts",
            },
            "gate criterion evidence",
        )
        if data["schema"] != GATE_CRITERION_EVIDENCE_SCHEMA:
            raise FormalGateEvidenceError("unsupported gate criterion evidence schema")
        raw_sources = data["source_artifacts"]
        if not isinstance(raw_sources, list):
            raise FormalGateEvidenceError("criterion source_artifacts must be a list")
        return cls(
            config_digest=data["config_digest"],
            gate=data["gate"],
            criterion=data["criterion"],
            passed=data["passed"],
            derivation_id=data["derivation_id"],
            evaluator_digest=data["evaluator_digest"],
            source_artifacts=tuple(
                CanonicalEvidenceRef.from_dict(item) for item in raw_sources
            ),
        )


GateContentEvaluator = Callable[[tuple[Mapping[str, Any], ...], str, str], bool]


@dataclass(frozen=True)
class RegisteredGateEvaluator:
    """Source-owned evaluator identity and exact accepted input contract.

    The descriptor is hashed from source-owned literals.  The executable
    callable is intentionally omitted from serialization: evidence authors may
    bind to an evaluator, but cannot upload a replacement predicate.
    """

    gate: GateName
    criterion: str
    evaluator_id: str
    evaluator_version: str
    accepted_source_schemas: tuple[str, ...]
    evaluate: GateContentEvaluator = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.gate not in GATE_REQUIREMENTS:
            raise FormalGateEvidenceError("registered evaluator has an unknown gate")
        if self.criterion not in GATE_REQUIREMENTS[self.gate]:
            raise FormalGateEvidenceError(
                "registered evaluator has an unknown criterion"
            )
        for name, value in (
            ("evaluator_id", self.evaluator_id),
            ("evaluator_version", self.evaluator_version),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise FormalGateEvidenceError(f"registered {name} is invalid")
        schemas = tuple(self.accepted_source_schemas)
        if (
            not schemas
            or len(schemas) != len(set(schemas))
            or any(
                not isinstance(item, str)
                or not item.startswith("policy-learnware.v02-")
                for item in schemas
            )
        ):
            raise FormalGateEvidenceError(
                "registered evaluator source schemas must be exact typed v0.2 schemas"
            )
        if not callable(self.evaluate):
            raise FormalGateEvidenceError("registered evaluator is not callable")
        object.__setattr__(self, "accepted_source_schemas", schemas)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema": GATE_EVALUATOR_REGISTRY_SCHEMA,
            "gate": self.gate,
            "criterion": self.criterion,
            "derivation_id": (
                f"policy-learnware.v02.gate/{self.gate}/{self.criterion}/v0"
            ),
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "accepted_source_schemas": list(self.accepted_source_schemas),
            "source_count": len(self.accepted_source_schemas),
            "caller_predicate_consumed": False,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.descriptor())


def _evaluate_all_registered_axis_audits(
    sources: tuple[Mapping[str, Any], ...],
    expected_experiment_id: str,
    expected_config_digest: str,
) -> bool:
    """Re-derive the axis audit result from exact primitive coverage fields."""

    del expected_experiment_id  # this artifact family is config-, not run-, bound
    if len(sources) != 1:
        raise FormalGateEvidenceError(
            "axis-audit evaluator requires exactly one source artifact"
        )
    data = _strict_mapping(
        sources[0],
        {
            "schema",
            "passed",
            "config_digest",
            "axis_registry_digest",
            "expected_work_units",
            "validated_work_units",
            "violations",
        },
        "axis-audit validation source",
    )
    if data["schema"] != "policy-learnware.v02-axis-audit-validation.v0":
        raise FormalGateEvidenceError("axis-audit evaluator received another schema")
    if data["config_digest"] != expected_config_digest:
        raise FormalGateEvidenceError("axis-audit source is bound to another config")
    _digest(data["axis_registry_digest"], "axis_registry_digest")
    expected = data["expected_work_units"]
    validated = data["validated_work_units"]
    violations = data["violations"]
    if (
        isinstance(expected, bool)
        or not isinstance(expected, int)
        or expected <= 0
        or isinstance(validated, bool)
        or not isinstance(validated, int)
        or validated < 0
        or not isinstance(violations, list)
    ):
        raise FormalGateEvidenceError(
            "axis-audit primitive coverage fields are invalid"
        )
    derived = validated == expected and not violations
    if type(data["passed"]) is not bool or data["passed"] is not derived:
        raise FormalGateEvidenceError(
            "axis-audit uploaded passed flag disagrees with primitive derivation"
        )
    return derived


# Only criteria with a complete, source-owned primitive derivation belong here.
# Missing entries are an intentional hard failure, never a request to consume
# ``GateCriterionEvidence.passed``.  The previously prototyped axis-validation
# evaluator is deliberately not registered: its expected/validated counters
# and violations list are a summary, not the raw per-axis audits bound to the
# reviewed config registry.  A caller could otherwise upload 1/1/[] and mint a
# passing criterion.  The same rule excludes source/championization snapshots
# whose thresholds and anchor census are not yet bound to the formal freeze.
_REGISTERED_GATE_EVALUATORS: Mapping[
    tuple[GateName, str], RegisteredGateEvaluator
] = MappingProxyType({})


def registered_gate_evaluator_descriptor(
    gate: GateName, criterion: str
) -> Mapping[str, Any]:
    """Return the immutable descriptor evidence must bind for one criterion."""

    evaluator = _REGISTERED_GATE_EVALUATORS.get((gate, criterion))
    if evaluator is None:
        raise FormalGateEvidenceError(
            f"no trusted content evaluator is registered for {gate}/{criterion}"
        )
    return MappingProxyType(evaluator.descriptor())


def missing_registered_gate_evaluators() -> tuple[str, ...]:
    """Return criteria that cannot yet receive formal evaluator authority."""

    return tuple(
        f"{gate}/{criterion}"
        for gate in GATE_ORDER
        for criterion in GATE_REQUIREMENTS[gate]
        if (gate, criterion) not in _REGISTERED_GATE_EVALUATORS
    )


def evaluate_registered_gate_criterion(
    evidence: GateCriterionEvidence,
    source_payloads: Sequence[Mapping[str, Any]],
    *,
    expected_experiment_id: str,
    expected_config_digest: str,
) -> bool:
    """Derive one criterion with the immutable source-owned registry."""

    if not isinstance(evidence, GateCriterionEvidence):
        raise FormalGateEvidenceError("criterion evidence has the wrong type")
    expected_id = _experiment_id(expected_experiment_id)
    expected_config = _digest(expected_config_digest, "expected config_digest")
    if evidence.config_digest != expected_config:
        raise FormalGateEvidenceError("criterion evidence is bound to another config")
    evaluator = _REGISTERED_GATE_EVALUATORS.get((evidence.gate, evidence.criterion))
    if evaluator is None:
        raise FormalGateEvidenceError(
            "formal criterion has no trusted content-derived evaluator: "
            f"{evidence.gate}/{evidence.criterion}"
        )
    payloads = tuple(source_payloads)
    if any(not isinstance(payload, Mapping) for payload in payloads):
        raise FormalGateEvidenceError("criterion evaluator sources must be mappings")
    observed_schemas = tuple(payload.get("schema") for payload in payloads)
    if observed_schemas != evaluator.accepted_source_schemas:
        raise FormalGateEvidenceError(
            "formal criterion source schemas/order differ from the registered "
            f"evaluator for {evidence.gate}/{evidence.criterion}"
        )
    if evidence.derivation_id != evaluator.descriptor()["derivation_id"]:
        raise FormalGateEvidenceError("criterion derivation differs from registry")
    if evidence.evaluator_digest != evaluator.digest:
        raise FormalGateEvidenceError(
            "criterion evaluator digest differs from registry"
        )
    derived = evaluator.evaluate(payloads, expected_id, expected_config)
    if type(derived) is not bool:
        raise FormalGateEvidenceError("registered evaluator returned a non-boolean")
    if evidence.passed is not derived:
        raise FormalGateEvidenceError(
            "criterion uploaded passed flag disagrees with registered content derivation"
        )
    return derived


@dataclass(frozen=True)
class GateEvidenceManifest:
    """Exact criterion-to-artifact map for all five formal v0.2 gates."""

    experiment_id: str
    config_digest: str
    criteria: Mapping[GateName, Mapping[str, CanonicalEvidenceRef]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _experiment_id(self.experiment_id))
        object.__setattr__(
            self, "config_digest", _digest(self.config_digest, "manifest config_digest")
        )
        if not isinstance(self.criteria, Mapping) or set(self.criteria) != set(
            GATE_ORDER
        ):
            raise FormalGateEvidenceError(
                "gate evidence manifest must cover every registered gate exactly once"
            )
        normalized: dict[GateName, Mapping[str, CanonicalEvidenceRef]] = {}
        seen_paths: set[str] = set()
        for gate in GATE_ORDER:
            raw = self.criteria[gate]
            expected = GATE_REQUIREMENTS[gate]
            if not isinstance(raw, Mapping) or set(raw) != set(expected):
                raise FormalGateEvidenceError(
                    f"gate evidence coverage differs for {gate}"
                )
            rows: dict[str, CanonicalEvidenceRef] = {}
            for criterion in expected:
                reference = raw[criterion]
                if not isinstance(reference, CanonicalEvidenceRef):
                    raise FormalGateEvidenceError(
                        "gate evidence entries must be typed refs"
                    )
                expected_path = criterion_evidence_relative_path(gate, criterion)
                if reference.canonical_path != expected_path:
                    raise FormalGateEvidenceError(
                        f"criterion evidence must use canonical path {expected_path}"
                    )
                if reference.artifact_schema != GATE_CRITERION_EVIDENCE_SCHEMA:
                    raise FormalGateEvidenceError(
                        "criterion ref must name the criterion-evidence schema"
                    )
                if reference.config_digest != self.config_digest:
                    raise FormalGateEvidenceError(
                        "criterion ref is not bound to the manifest config"
                    )
                if reference.canonical_path in seen_paths:
                    raise FormalGateEvidenceError(
                        "criterion evidence paths cannot be reused"
                    )
                seen_paths.add(reference.canonical_path)
                rows[criterion] = reference
            normalized[gate] = MappingProxyType(rows)
        object.__setattr__(self, "criteria", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_EVIDENCE_MANIFEST_SCHEMA,
            "experiment_id": self.experiment_id,
            "config_digest": self.config_digest,
            "criteria": {
                gate: {
                    criterion: self.criteria[gate][criterion].to_dict()
                    for criterion in GATE_REQUIREMENTS[gate]
                }
                for gate in GATE_ORDER
            },
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "GateEvidenceManifest":
        data = _strict_mapping(
            value,
            {"schema", "experiment_id", "config_digest", "criteria"},
            "gate evidence manifest",
        )
        if data["schema"] != GATE_EVIDENCE_MANIFEST_SCHEMA:
            raise FormalGateEvidenceError("unsupported gate evidence manifest schema")
        raw_criteria = data["criteria"]
        if not isinstance(raw_criteria, Mapping):
            raise FormalGateEvidenceError("gate evidence criteria must be an object")
        criteria: dict[GateName, dict[str, CanonicalEvidenceRef]] = {}
        for gate in GATE_ORDER:
            raw_gate = raw_criteria.get(gate)
            if not isinstance(raw_gate, Mapping):
                raise FormalGateEvidenceError(f"missing evidence group {gate}")
            criteria[gate] = {
                criterion: CanonicalEvidenceRef.from_dict(raw_gate[criterion])
                for criterion in GATE_REQUIREMENTS[gate]
                if criterion in raw_gate
            }
            if set(raw_gate) != set(GATE_REQUIREMENTS[gate]):
                raise FormalGateEvidenceError(
                    f"gate evidence coverage differs for {gate}"
                )
        if set(raw_criteria) != set(GATE_ORDER):
            raise FormalGateEvidenceError("gate evidence contains unknown gate groups")
        return cls(
            experiment_id=data["experiment_id"],
            config_digest=data["config_digest"],
            criteria=criteria,
        )


@dataclass(frozen=True)
class GateCriterion:
    name: str
    observed: bool | None
    passed: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observed": self.observed,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GateDecision:
    """A decision computed from primitive checks, never from a caller pass bit."""

    gate: GateName
    criteria: tuple[GateCriterion, ...]
    unexpected_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.gate not in GATE_REQUIREMENTS:
            raise ValueError(f"unknown v0.2 gate: {self.gate!r}")
        expected = GATE_REQUIREMENTS[self.gate]
        names = tuple(item.name for item in self.criteria)
        if names != expected:
            raise ValueError(
                "GateDecision criteria must exactly follow the registered gate contract"
            )
        if len(set(self.unexpected_checks)) != len(self.unexpected_checks):
            raise ValueError("unexpected gate checks must be unique")

    @property
    def passed(self) -> bool:
        return bool(
            not self.unexpected_checks
            and self.criteria
            and all(item.passed for item in self.criteria)
        )

    @property
    def outcome(self) -> GateOutcome:
        return "PASS" if self.passed else "FAIL"

    @property
    def missing_checks(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.criteria if item.observed is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_DECISION_SCHEMA,
            "gate": self.gate,
            "outcome": self.outcome,
            "passed": self.passed,
            "fail_closed": True,
            "criteria": [item.to_dict() for item in self.criteria],
            "missing_checks": list(self.missing_checks),
            "unexpected_checks": list(self.unexpected_checks),
        }


def evaluate_gate(gate: GateName, checks: Mapping[str, object]) -> GateDecision:
    """Evaluate one gate against its exact primitive-check allowlist.

    Missing, non-boolean, and unknown fields yield a failed decision.  They do
    not raise an exception that an orchestration layer could accidentally treat
    as a skipped gate.
    """

    if gate not in GATE_REQUIREMENTS:
        raise ValueError(f"unknown v0.2 gate: {gate!r}")
    if not isinstance(checks, Mapping):
        checks = {}
    expected = GATE_REQUIREMENTS[gate]
    criteria: list[GateCriterion] = []
    for name in expected:
        if name not in checks:
            criteria.append(GateCriterion(name, None, False, "missing_required_check"))
            continue
        value = checks[name]
        if type(value) is not bool:
            criteria.append(GateCriterion(name, None, False, "check_is_not_boolean"))
            continue
        criteria.append(
            GateCriterion(
                name=name,
                observed=value,
                passed=value,
                reason=None if value else "primitive_check_failed",
            )
        )
    unexpected = tuple(sorted(str(name) for name in set(checks) - set(expected)))
    return GateDecision(gate, tuple(criteria), unexpected)


def evaluate_scope(checks: Mapping[str, object]) -> GateDecision:
    return evaluate_gate("G02-Scope", checks)


def evaluate_engineering(checks: Mapping[str, object]) -> GateDecision:
    return evaluate_gate("G02-Engineering", checks)


def evaluate_market(checks: Mapping[str, object]) -> GateDecision:
    return evaluate_gate("G02-Market", checks)


def evaluate_replace(checks: Mapping[str, object]) -> GateDecision:
    return evaluate_gate("G02-Replace", checks)


def evaluate_split_freeze(checks: Mapping[str, object]) -> GateDecision:
    return evaluate_gate("G02-SplitFreeze", checks)


@dataclass(frozen=True)
class V02GateState:
    """Ordered v0.2 gate state with scientific No-Go distinguished from blockage."""

    status: V02CompletionStatus
    decisions: tuple[GateDecision, ...]
    passed_gates: tuple[GateName, ...]
    blocking_gate: GateName | None
    invalid_gate_inputs: tuple[str, ...] = ()

    @property
    def ready_for_v03(self) -> bool:
        return self.status == "READY_FOR_V03_JOINT_CONFIRMATORY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_STATE_SCHEMA,
            "status": self.status,
            "ready_for_v03": self.ready_for_v03,
            "passed_gates": list(self.passed_gates),
            "blocking_gate": self.blocking_gate,
            "invalid_gate_inputs": list(self.invalid_gate_inputs),
            "decisions": [item.to_dict() for item in self.decisions],
        }


def evaluate_gate_state(
    decisions: Mapping[str, GateDecision] | Sequence[GateDecision],
) -> V02GateState:
    """Advance only through a contiguous prefix of passing registered gates."""

    invalid: list[str] = []
    by_name: dict[str, GateDecision] = {}
    values = (
        tuple(decisions.values())
        if isinstance(decisions, Mapping)
        else tuple(decisions)
    )
    if isinstance(decisions, Mapping):
        for key, decision in decisions.items():
            if not isinstance(decision, GateDecision) or key != getattr(
                decision, "gate", None
            ):
                invalid.append(f"invalid_mapping_entry:{key}")
    for index, decision in enumerate(values):
        if not isinstance(decision, GateDecision):
            invalid.append(f"non_decision_at:{index}")
            continue
        if decision.gate in by_name:
            invalid.append(f"duplicate_gate:{decision.gate}")
            continue
        by_name[decision.gate] = decision

    normalized = tuple(
        by_name.get(gate, evaluate_gate(gate, {})) for gate in GATE_ORDER
    )
    if invalid:
        return V02GateState(
            status="BLOCKED_ENGINEERING",
            decisions=normalized,
            passed_gates=(),
            blocking_gate="G02-Scope",
            invalid_gate_inputs=tuple(sorted(invalid)),
        )

    passed: list[GateName] = []
    for decision in normalized:
        if decision.passed:
            passed.append(decision.gate)
            continue
        status: V02CompletionStatus
        if decision.gate == "G02-Market":
            status = "COMPLETE_NO_GO_MARKET"
        elif decision.gate == "G02-Replace":
            status = "COMPLETE_NO_GO_CORRO_INCUMBENT"
        else:
            status = "BLOCKED_ENGINEERING"
        return V02GateState(
            status=status,
            decisions=normalized,
            passed_gates=tuple(passed),
            blocking_gate=decision.gate,
        )
    return V02GateState(
        status="READY_FOR_V03_JOINT_CONFIRMATORY",
        decisions=normalized,
        passed_gates=GATE_ORDER,
        blocking_gate=None,
    )


@dataclass(frozen=True)
class FormalV02GateState:
    """Formal gate state whose every primitive is backed by verified artifacts."""

    experiment_id: str
    config_digest: str
    gate_evidence_manifest: GateEvidenceManifest
    gate_evidence_manifest_ref: CanonicalEvidenceRef
    criterion_evidence: Mapping[GateName, Mapping[str, GateCriterionEvidence]]
    core_state: V02GateState
    _evaluation_authority: object | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_id", _experiment_id(self.experiment_id))
        object.__setattr__(
            self,
            "config_digest",
            _digest(self.config_digest, "gate-state config_digest"),
        )
        if not isinstance(self.gate_evidence_manifest, GateEvidenceManifest):
            raise FormalGateEvidenceError(
                "formal gate state has no typed evidence manifest"
            )
        if self.gate_evidence_manifest.experiment_id != self.experiment_id:
            raise FormalGateEvidenceError(
                "gate state and evidence experiment IDs differ"
            )
        if self.gate_evidence_manifest.config_digest != self.config_digest:
            raise FormalGateEvidenceError(
                "gate state and evidence config digests differ"
            )
        reference = self.gate_evidence_manifest_ref
        if not isinstance(reference, CanonicalEvidenceRef):
            raise FormalGateEvidenceError("gate state has no manifest evidence ref")
        if reference.canonical_path != gate_evidence_manifest_relative_path():
            raise FormalGateEvidenceError(
                "gate evidence manifest path is not canonical"
            )
        if reference.artifact_schema != GATE_EVIDENCE_MANIFEST_SCHEMA:
            raise FormalGateEvidenceError(
                "gate evidence manifest ref has the wrong schema"
            )
        if reference.artifact_digest != self.gate_evidence_manifest.digest:
            raise FormalGateEvidenceError("gate evidence manifest digest is misbound")
        if reference.config_digest != self.config_digest:
            raise FormalGateEvidenceError(
                "gate evidence manifest ref is config-misbound"
            )
        if not isinstance(self.core_state, V02GateState):
            raise FormalGateEvidenceError("formal gate state has no derived core state")
        if not isinstance(self.criterion_evidence, Mapping) or set(
            self.criterion_evidence
        ) != set(GATE_ORDER):
            raise FormalGateEvidenceError(
                "formal criterion evidence gate coverage differs"
            )
        normalized: dict[GateName, Mapping[str, GateCriterionEvidence]] = {}
        for gate in GATE_ORDER:
            rows = self.criterion_evidence[gate]
            if not isinstance(rows, Mapping) or set(rows) != set(
                GATE_REQUIREMENTS[gate]
            ):
                raise FormalGateEvidenceError(
                    f"formal criterion evidence coverage differs for {gate}"
                )
            normalized_rows: dict[str, GateCriterionEvidence] = {}
            for criterion in GATE_REQUIREMENTS[gate]:
                evidence = rows[criterion]
                if not isinstance(evidence, GateCriterionEvidence):
                    raise FormalGateEvidenceError(
                        "criterion evidence has the wrong type"
                    )
                if (
                    evidence.gate != gate
                    or evidence.criterion != criterion
                    or evidence.config_digest != self.config_digest
                ):
                    raise FormalGateEvidenceError(
                        "criterion evidence identity is misbound"
                    )
                reference = self.gate_evidence_manifest.criteria[gate][criterion]
                if reference.artifact_digest != evidence.digest:
                    raise FormalGateEvidenceError(
                        "criterion evidence digest differs from manifest binding"
                    )
                normalized_rows[criterion] = evidence
            normalized[gate] = MappingProxyType(normalized_rows)
        object.__setattr__(self, "criterion_evidence", MappingProxyType(normalized))

    @property
    def status(self) -> V02CompletionStatus:
        return self.core_state.status

    @property
    def is_formally_authoritative(self) -> bool:
        """Whether fixed source-owned evaluators produced this live object."""

        return self._evaluation_authority is _TRUSTED_GATE_EVALUATION_AUTHORITY

    def require_formal_authority(self) -> None:
        if not self.is_formally_authoritative:
            raise FormalGateEvidenceError(
                "formal gate state lacks trusted in-process evaluator authority"
            )

    @property
    def ready_for_v03(self) -> bool:
        return self.core_state.ready_for_v03

    @property
    def passed_gates(self) -> tuple[GateName, ...]:
        return self.core_state.passed_gates

    @property
    def blocking_gate(self) -> GateName | None:
        return self.core_state.blocking_gate

    @property
    def gate_evidence_manifest_digest(self) -> str:
        return self.gate_evidence_manifest.digest

    @property
    def gate_evidence_manifest_file_sha256(self) -> str:
        return self.gate_evidence_manifest_ref.file_sha256

    def to_dict(self) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        for decision in self.core_state.decisions:
            criteria: list[dict[str, Any]] = []
            for item in decision.criteria:
                row = item.to_dict()
                row["evidence_ref"] = self.gate_evidence_manifest.criteria[
                    decision.gate
                ][item.name].to_dict()
                criteria.append(row)
            decisions.append(
                {
                    "schema": FORMAL_GATE_DECISION_SCHEMA,
                    "gate": decision.gate,
                    "outcome": decision.outcome,
                    "passed": decision.passed,
                    "fail_closed": True,
                    "criteria": criteria,
                    "missing_checks": list(decision.missing_checks),
                    "unexpected_checks": list(decision.unexpected_checks),
                }
            )
        return {
            "schema": FORMAL_GATE_STATE_SCHEMA,
            "experiment_id": self.experiment_id,
            "config_digest": self.config_digest,
            "gate_evidence_manifest_digest": self.gate_evidence_manifest_digest,
            "gate_evidence_manifest_file_sha256": (
                self.gate_evidence_manifest_file_sha256
            ),
            "gate_evidence_manifest_ref": self.gate_evidence_manifest_ref.to_dict(),
            "status": self.status,
            "ready_for_v03": self.ready_for_v03,
            "passed_gates": list(self.passed_gates),
            "blocking_gate": self.blocking_gate,
            "invalid_gate_inputs": list(self.core_state.invalid_gate_inputs),
            "decisions": decisions,
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def _evaluate_formal_gate_manifest(
    manifest: GateEvidenceManifest,
    *,
    manifest_ref: CanonicalEvidenceRef,
    experiment_root: str | Path,
) -> FormalV02GateState:
    checks: dict[GateName, dict[str, bool]] = {}
    evidence_by_gate: dict[GateName, dict[str, GateCriterionEvidence]] = {}
    for gate in GATE_ORDER:
        checks[gate] = {}
        evidence_by_gate[gate] = {}
        for criterion in GATE_REQUIREMENTS[gate]:
            reference = manifest.criteria[gate][criterion]
            raw = verify_canonical_evidence_ref(
                reference,
                experiment_root=experiment_root,
                expected_config_digest=manifest.config_digest,
                require_config_binding=True,
            )
            evidence = GateCriterionEvidence.from_dict(raw)
            if (
                evidence.config_digest != manifest.config_digest
                or evidence.gate != gate
                or evidence.criterion != criterion
            ):
                raise FormalGateEvidenceError(
                    f"criterion evidence identity is misbound for {gate}/{criterion}"
                )
            if canonicalize(raw) != evidence.to_dict():
                raise FormalGateEvidenceError(
                    f"criterion evidence is not canonical for {gate}/{criterion}"
                )
            source_payloads: list[Mapping[str, Any]] = []
            for source in evidence.source_artifacts:
                source_payloads.append(
                    verify_canonical_evidence_ref(
                        source,
                        experiment_root=experiment_root,
                        expected_config_digest=manifest.config_digest,
                        require_config_binding=False,
                        source_artifact=True,
                    )
                )
            derived = evaluate_registered_gate_criterion(
                evidence,
                source_payloads,
                expected_experiment_id=manifest.experiment_id,
                expected_config_digest=manifest.config_digest,
            )
            checks[gate][criterion] = derived
            evidence_by_gate[gate][criterion] = evidence
    decisions = tuple(evaluate_gate(gate, checks[gate]) for gate in GATE_ORDER)
    return FormalV02GateState(
        experiment_id=manifest.experiment_id,
        config_digest=manifest.config_digest,
        gate_evidence_manifest=manifest,
        gate_evidence_manifest_ref=manifest_ref,
        criterion_evidence=evidence_by_gate,
        core_state=evaluate_gate_state(decisions),
        _evaluation_authority=_TRUSTED_GATE_EVALUATION_AUTHORITY,
    )


def evaluate_formal_gate_state_from_file(
    manifest_path: str | Path,
    *,
    experiment_root: str | Path,
    expected_experiment_id: str,
    expected_config_digest: str,
) -> FormalV02GateState:
    """Verify the canonical evidence graph and derive a formal gate state.

    This verifies persisted provenance and deterministic bindings.  It is not a
    signature system and therefore does not claim to protect against an actor
    who is authorized to replace every referenced local artifact and hash.
    """

    expected_id = _experiment_id(expected_experiment_id)
    expected_config = _digest(expected_config_digest, "expected config_digest")
    root = Path(experiment_root).expanduser().resolve()
    supplied = Path(manifest_path).expanduser().resolve()
    expected_path = root.joinpath(
        *PurePosixPath(gate_evidence_manifest_relative_path()).parts
    ).resolve()
    if supplied != expected_path:
        raise FormalGateEvidenceError(
            f"gate evidence manifest must use canonical path {expected_path}"
        )
    manifest_ref = build_canonical_evidence_ref(
        supplied,
        experiment_root=root,
        expected_config_digest=expected_config,
    )
    if manifest_ref.artifact_schema != GATE_EVIDENCE_MANIFEST_SCHEMA:
        raise FormalGateEvidenceError("gate evidence manifest has the wrong schema")
    raw = verify_canonical_evidence_ref(
        manifest_ref,
        experiment_root=root,
        expected_config_digest=expected_config,
        require_config_binding=True,
    )
    manifest = GateEvidenceManifest.from_dict(raw)
    if canonicalize(raw) != manifest.to_dict():
        raise FormalGateEvidenceError("gate evidence manifest is not canonical")
    if manifest.experiment_id != expected_id:
        raise FormalGateEvidenceError(
            "gate evidence manifest has another experiment ID"
        )
    if manifest.config_digest != expected_config:
        raise FormalGateEvidenceError(
            "gate evidence manifest has another config digest"
        )
    return _evaluate_formal_gate_manifest(
        manifest,
        manifest_ref=manifest_ref,
        experiment_root=root,
    )


def validate_formal_gate_state_payload(
    value: object,
    *,
    experiment_root: str | Path,
    expected_experiment_id: str,
    expected_config_digest: str,
) -> FormalV02GateState:
    """Re-read all evidence and compare a persisted formal state to derivation."""

    data = _strict_mapping(
        value,
        {
            "schema",
            "experiment_id",
            "config_digest",
            "gate_evidence_manifest_digest",
            "gate_evidence_manifest_file_sha256",
            "gate_evidence_manifest_ref",
            "status",
            "ready_for_v03",
            "passed_gates",
            "blocking_gate",
            "invalid_gate_inputs",
            "decisions",
        },
        "formal gate state",
    )
    if data["schema"] != FORMAL_GATE_STATE_SCHEMA:
        raise FormalGateEvidenceError(
            "completion requires the formal evidence-bound gate-state schema"
        )
    expected_id = _experiment_id(expected_experiment_id)
    expected_config = _digest(expected_config_digest, "expected config_digest")
    if data["experiment_id"] != expected_id or data["config_digest"] != expected_config:
        raise FormalGateEvidenceError(
            "formal gate state is bound to another run/config"
        )
    embedded_ref = CanonicalEvidenceRef.from_dict(data["gate_evidence_manifest_ref"])
    if embedded_ref.canonical_path != gate_evidence_manifest_relative_path():
        raise FormalGateEvidenceError(
            "embedded gate evidence manifest path is not canonical"
        )
    if data["gate_evidence_manifest_digest"] != embedded_ref.artifact_digest:
        raise FormalGateEvidenceError(
            "formal gate state manifest digest differs from its embedded ref"
        )
    if data["gate_evidence_manifest_file_sha256"] != embedded_ref.file_sha256:
        raise FormalGateEvidenceError(
            "formal gate state manifest file hash differs from its embedded ref"
        )
    state = evaluate_formal_gate_state_from_file(
        Path(experiment_root)
        .expanduser()
        .resolve()
        .joinpath(*PurePosixPath(embedded_ref.canonical_path).parts),
        experiment_root=experiment_root,
        expected_experiment_id=expected_id,
        expected_config_digest=expected_config,
    )
    if embedded_ref != state.gate_evidence_manifest_ref:
        raise FormalGateEvidenceError("embedded gate evidence manifest ref is stale")
    if data["gate_evidence_manifest_digest"] != state.gate_evidence_manifest_digest:
        raise FormalGateEvidenceError("formal gate state manifest digest is stale")
    if (
        data["gate_evidence_manifest_file_sha256"]
        != state.gate_evidence_manifest_file_sha256
    ):
        raise FormalGateEvidenceError("formal gate state manifest file hash is stale")
    if canonicalize(data) != state.to_dict():
        raise FormalGateEvidenceError(
            "persisted formal gate state differs from evidence-derived state"
        )
    return state


__all__ = [
    "FORMAL_GATE_DECISION_SCHEMA",
    "FORMAL_GATE_STATE_SCHEMA",
    "G02_ENGINEERING_CHECKS",
    "G02_MARKET_CHECKS",
    "G02_REPLACE_CHECKS",
    "G02_SCOPE_CHECKS",
    "G02_SPLIT_FREEZE_CHECKS",
    "GATE_ORDER",
    "GATE_REQUIREMENTS",
    "GATE_CRITERION_EVIDENCE_SCHEMA",
    "GATE_DECISION_SCHEMA",
    "GATE_EVALUATOR_REGISTRY_SCHEMA",
    "GATE_EVIDENCE_MANIFEST_SCHEMA",
    "GATE_STATE_SCHEMA",
    "CanonicalEvidenceRef",
    "FormalGateEvidenceError",
    "FormalV02GateState",
    "GateCriterionEvidence",
    "GateEvidenceManifest",
    "GateCriterion",
    "GateDecision",
    "GateName",
    "GateOutcome",
    "RegisteredGateEvaluator",
    "V02CompletionStatus",
    "V02GateState",
    "build_canonical_evidence_ref",
    "criterion_evidence_relative_path",
    "evaluate_engineering",
    "evaluate_registered_gate_criterion",
    "evaluate_gate",
    "evaluate_gate_state",
    "evaluate_formal_gate_state_from_file",
    "gate_evidence_manifest_relative_path",
    "missing_registered_gate_evaluators",
    "registered_gate_evaluator_descriptor",
    "evaluate_market",
    "evaluate_replace",
    "evaluate_scope",
    "evaluate_split_freeze",
    "validate_formal_gate_state_payload",
    "verify_canonical_evidence_ref",
]
