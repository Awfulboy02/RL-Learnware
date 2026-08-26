"""Typed, fail-closed driver for the v0.3 production stage chain.

The scientific implementations in :mod:`policy_learnware_v0.v03` deliberately
do not know how a particular cluster launches environments, trainers, or batch
jobs.  This module is the narrow boundary between those implementations and an
externally supplied execution adapter.  It provides three properties that a
plain collection of CLI parser entries would not provide:

* every execution is bound to an exact pre-experiment freeze and exact input
  bytes;
* an adapter must be injected by the server launcher and its identity and
  contract digest must match the reviewed stage manifest;
* successful outputs are re-read by digest before an immutable execution
  receipt is published, and resume never calls the adapter again.

The driver cannot create review authority, cannot request oracle access, and
does not contain a fallback adapter.  Formal execution therefore requires an
externally authorized :class:`PreExperimentFreezeManifest` and an injected
adapter.  Development fixtures use the same path with a development namespace
and an unverified engineering freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence

from ..hashing import canonical_json_bytes, sha256_file, sha256_json
from .artifacts import V03ArtifactLayout
from .claim_audit import FormalClaimAudit
from .costs import (
    COST_COMPONENT_SCHEMA,
    COST_LEDGER_SCHEMA,
    CostComponentRecord,
    V03CostLedger,
)
from .preflight import (
    FORMAL_PRODUCTION_STAGE_IDS,
    IndependentRecomputeAttestation,
    OracleUnlockHandoff,
    PreExperimentFreezeManifest,
    PublicRankingBarrier,
    formal_stage_adapter_binding_digest,
)
from .statistics import FormalStatisticsResult
from .schemas import checked_digest, checked_safe_id, strict_mapping


STAGE_INPUT_SCHEMA = "policy-learnware.v03-stage-input.v0"
STAGE_DEPENDENCY_SCHEMA = "policy-learnware.v03-stage-dependency.v0"
FORMAL_STAGE_REQUEST_TEMPLATE_SCHEMA = (
    "policy-learnware.v03-formal-stage-request-template.v0"
)
STAGE_EXECUTION_MANIFEST_SCHEMA = "policy-learnware.v03-stage-execution-manifest.v1"
STAGE_OUTPUT_ARTIFACT_SCHEMA = "policy-learnware.v03-stage-output-artifact.v0"
STAGE_SEMANTIC_OUTPUT_SCHEMA = "policy-learnware.v03-stage-semantic-output.v0"
STAGE_ADAPTER_RESULT_SCHEMA = "policy-learnware.v03-stage-adapter-result.v0"
STAGE_EXECUTION_RECEIPT_SCHEMA = "policy-learnware.v03-stage-execution-receipt.v1"
PIPELINE_COMPLETION_MANIFEST_SCHEMA = (
    "policy-learnware.v03-pipeline-completion-manifest.v2"
)
PIPELINE_COMPLETION_RECEIPT_SCHEMA = (
    "policy-learnware.v03-pipeline-completion-receipt.v2"
)

DEVELOPMENT_MODE = "DEVELOPMENT"
FORMAL_MODE = "FORMAL"
ExecutionMode = Literal["DEVELOPMENT", "FORMAL"]

PRODUCTION_STAGE_IDS = FORMAL_PRODUCTION_STAGE_IDS

# A linear dependency spine is intentionally explicit.  Extra physical inputs
# are bound by StageInputBinding; stage receipts establish that the reviewed
# orchestration order itself was followed.
REQUIRED_PREDECESSOR = MappingProxyType(
    {
        PRODUCTION_STAGE_IDS[index]: (
            () if index == 0 else (PRODUCTION_STAGE_IDS[index - 1],)
        )
        for index in range(len(PRODUCTION_STAGE_IDS))
    }
)

_ALLOWED_OUTPUT_DOMAINS = MappingProxyType(
    {
        "collect-source-receipts": frozenset(
            {"source_market", "source_market_private", "source_reference_measurement"}
        ),
        "build-market": frozenset(
            {"source_market", "market_public", "deployment_private"}
        ),
        "build-canonical-banks": frozenset({"raw_banks", "canonical_banks"}),
        "build-transition-views": frozenset({"views"}),
        "replay-legacy-attribution": frozenset({"attribution", "analysis"}),
        "fit-representation-controls": frozenset(
            {
                "representation_controls",
                "encoder_training_banks",
                "encoder_checkpoints",
                "encoder_training_private",
                "representation_indices",
            }
        ),
        "build-signal-atlas": frozenset(
            {"signal_atlas", "signal_atlas_private", "pair_controls"}
        ),
        "build-source-specs": frozenset({"source_specs", "semantic_caches"}),
        "build-query-specs": frozenset({"query_specs", "semantic_caches"}),
        "fit-baselines": frozenset(
            {"baseline_tables", "selector_views", "selector_outputs"}
        ),
        "run-public-rankings": frozenset({"anonymous_rankings"}),
    }
)

REQUIRED_COMPLETION_EXTERNAL_ARTIFACT_IDS = tuple(
    sorted(
        {
            "formal-attribution-admission",
            "formal-claim-audit",
            "formal-cost-ledger",
            "formal-market-admission",
            "formal-probe-admission",
            "formal-signal-readout-bundle",
            "independent-recompute-attestation",
            "oracle-unlock-handoff",
            "pre-oracle-signal-manifest",
            "public-ranking-barrier",
            "statistics-result",
        }
    )
)


class StageExecutionError(ValueError):
    """A production stage is unbound, unauthorized, tampered, or incomplete."""


@dataclass(frozen=True)
class StageSemanticRequirement:
    """One required principal result for a formal production stage."""

    semantic_id: str
    domain: str
    payload_schema: str
    minimum_record_count: int
    exact_record_count: int | None = None

    def __post_init__(self) -> None:
        try:
            semantic_id = checked_safe_id(self.semantic_id, "semantic_id")
            domain = checked_safe_id(self.domain, "semantic domain")
        except ValueError as error:
            raise StageExecutionError(str(error)) from error
        object.__setattr__(self, "semantic_id", semantic_id)
        object.__setattr__(self, "domain", domain)
        if (
            not isinstance(self.payload_schema, str)
            or not self.payload_schema
            or self.payload_schema.strip() != self.payload_schema
        ):
            raise StageExecutionError("semantic payload schema must be canonical")
        if (
            isinstance(self.minimum_record_count, bool)
            or not isinstance(self.minimum_record_count, int)
            or self.minimum_record_count <= 0
        ):
            raise StageExecutionError("semantic minimum record count must be positive")
        if self.exact_record_count is not None and (
            isinstance(self.exact_record_count, bool)
            or not isinstance(self.exact_record_count, int)
            or self.exact_record_count < self.minimum_record_count
        ):
            raise StageExecutionError("semantic exact record count is invalid")

    def validate_count(self, value: int) -> None:
        if value < self.minimum_record_count:
            raise StageExecutionError(
                f"semantic output {self.semantic_id!r} has incomplete record coverage"
            )
        if self.exact_record_count is not None and value != self.exact_record_count:
            raise StageExecutionError(
                f"semantic output {self.semantic_id!r} requires exactly "
                f"{self.exact_record_count} records"
            )


# The adapter may create additional private/binary files, but a formal receipt
# is issued only for these exact, canonical semantic summaries.  Each summary
# binds the digests of the underlying stage evidence, so an arbitrary file in
# an allowed directory can never satisfy the stage contract.
STAGE_SEMANTIC_REQUIREMENTS = MappingProxyType(
    {
        "collect-source-receipts": (
            StageSemanticRequirement(
                "source-receipt-set",
                "source_market",
                STAGE_SEMANTIC_OUTPUT_SCHEMA,
                30,
                30,
            ),
        ),
        "build-market": (
            StageSemanticRequirement(
                "public-policy-market",
                "market_public",
                STAGE_SEMANTIC_OUTPUT_SCHEMA,
                30,
                30,
            ),
            StageSemanticRequirement(
                "private-deployment-registry",
                "deployment_private",
                STAGE_SEMANTIC_OUTPUT_SCHEMA,
                30,
                30,
            ),
        ),
        "build-canonical-banks": (
            StageSemanticRequirement(
                "canonical-bank-set", "canonical_banks", STAGE_SEMANTIC_OUTPUT_SCHEMA, 1
            ),
        ),
        "build-transition-views": (
            StageSemanticRequirement(
                "transition-view-set", "views", STAGE_SEMANTIC_OUTPUT_SCHEMA, 13, 13
            ),
        ),
        "replay-legacy-attribution": (
            StageSemanticRequirement(
                "legacy-attribution-table",
                "attribution",
                STAGE_SEMANTIC_OUTPUT_SCHEMA,
                14,
                14,
            ),
        ),
        "fit-representation-controls": (
            StageSemanticRequirement(
                "representation-fit-receipt-set",
                "representation_controls",
                STAGE_SEMANTIC_OUTPUT_SCHEMA,
                45,
                45,
            ),
        ),
        "build-signal-atlas": (
            StageSemanticRequirement(
                "formal-signal-readout-bundle",
                "signal_atlas_private",
                "policy-learnware.v03-formal-signal-readout-bundle.v0",
                39,
                39,
            ),
            StageSemanticRequirement(
                "public-signal-readout-bundle",
                "signal_atlas",
                "policy-learnware.v03-public-signal-readout-bundle.v0",
                39,
                39,
            ),
        ),
        "build-source-specs": (
            StageSemanticRequirement(
                "source-spec-set", "source_specs", STAGE_SEMANTIC_OUTPUT_SCHEMA, 30, 30
            ),
        ),
        "build-query-specs": (
            StageSemanticRequirement(
                "query-spec-set", "query_specs", STAGE_SEMANTIC_OUTPUT_SCHEMA, 66, 66
            ),
        ),
        "fit-baselines": (
            StageSemanticRequirement(
                "baseline-method-set", "baseline_tables", STAGE_SEMANTIC_OUTPUT_SCHEMA, 9, 9
            ),
        ),
        "run-public-rankings": (
            StageSemanticRequirement(
                "public-ranking-matrix",
                "anonymous_rankings",
                STAGE_SEMANTIC_OUTPUT_SCHEMA,
                594,
                594,
            ),
        ),
    }
)


def _strict(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    try:
        return strict_mapping(value, expected, where)
    except ValueError as error:
        raise StageExecutionError(str(error)) from error


def _digest(value: Any, where: str) -> str:
    try:
        return checked_digest(value, where)
    except ValueError as error:
        raise StageExecutionError(str(error)) from error


def _safe_id(value: Any, where: str) -> str:
    try:
        return checked_safe_id(value, where)
    except ValueError as error:
        raise StageExecutionError(str(error)) from error


def _regular_file(path: str | Path, where: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.is_symlink():
        raise StageExecutionError(f"{where} must be a regular non-symlink file")
    return candidate.resolve()


def _canonical_json_file(path: Path, where: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StageExecutionError(f"cannot load {where}: {error}") from error
    if not isinstance(value, Mapping):
        raise StageExecutionError(f"{where} must contain one JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise StageExecutionError(f"{where} is not canonical JSON")
    return value


@dataclass(frozen=True)
class FormalStageRequestTemplate:
    """Review-time request identity excluding run-local predecessor receipts.

    Paths are intentionally excluded: cluster mount points may differ.  The
    template instead freezes the adapter, parameter digest, and every static
    input's logical ID and byte-content digest.  Predecessor outputs are bound
    independently by the immutable linear receipt chain.
    """

    stage_id: str
    adapter_id: str
    adapter_contract_digest: str
    parameters_digest: str
    static_input_content_digests: Mapping[str, str]
    schema: str = FORMAL_STAGE_REQUEST_TEMPLATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_STAGE_REQUEST_TEMPLATE_SCHEMA:
            raise StageExecutionError("unsupported formal stage request template")
        if self.stage_id not in PRODUCTION_STAGE_IDS:
            raise StageExecutionError("formal request template has unknown stage")
        object.__setattr__(self, "adapter_id", _safe_id(self.adapter_id, "adapter_id"))
        for name in ("adapter_contract_digest", "parameters_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.static_input_content_digests, Mapping):
            raise StageExecutionError("static input digest registry must be a mapping")
        static_inputs: dict[str, str] = {}
        for input_id, content_digest in self.static_input_content_digests.items():
            canonical_id = _safe_id(input_id, "static input ID")
            static_inputs[canonical_id] = _digest(
                content_digest, f"static input {canonical_id} content digest"
            )
        object.__setattr__(
            self,
            "static_input_content_digests",
            MappingProxyType(dict(sorted(static_inputs.items()))),
        )

    @property
    def request_template_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_id": self.stage_id,
            "adapter_id": self.adapter_id,
            "adapter_contract_digest": self.adapter_contract_digest,
            "parameters_digest": self.parameters_digest,
            "static_input_content_digests": dict(
                self.static_input_content_digests
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalStageRequestTemplate":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "formal stage request template")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class StageSemanticOutput:
    """Canonical principal result emitted by one reviewed production stage."""

    stage_id: str
    semantic_id: str
    run_id: str
    freeze_manifest_digest: str
    evidence_digests: Mapping[str, str]
    record_count: int
    semantic_digest: str | None = None
    schema: str = STAGE_SEMANTIC_OUTPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_SEMANTIC_OUTPUT_SCHEMA:
            raise StageExecutionError("unsupported stage semantic output schema")
        if self.stage_id not in PRODUCTION_STAGE_IDS:
            raise StageExecutionError("semantic output has unknown production stage")
        object.__setattr__(self, "semantic_id", _safe_id(self.semantic_id, "semantic_id"))
        object.__setattr__(self, "run_id", _safe_id(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "freeze_manifest_digest",
            _digest(self.freeze_manifest_digest, "freeze_manifest_digest"),
        )
        if not isinstance(self.evidence_digests, Mapping) or not self.evidence_digests:
            raise StageExecutionError("semantic output requires evidence digests")
        evidence: dict[str, str] = {}
        for evidence_id, digest in self.evidence_digests.items():
            canonical_id = _safe_id(evidence_id, "semantic evidence ID")
            evidence[canonical_id] = _digest(
                digest, f"semantic evidence {canonical_id} digest"
            )
        object.__setattr__(
            self, "evidence_digests", MappingProxyType(dict(sorted(evidence.items())))
        )
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or self.record_count <= 0
        ):
            raise StageExecutionError("semantic output record count must be positive")
        expected = sha256_json(self._payload_without_digest())
        if self.semantic_digest is None:
            object.__setattr__(self, "semantic_digest", expected)
        elif _digest(self.semantic_digest, "semantic_digest") != expected:
            raise StageExecutionError("stage semantic output digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_id": self.stage_id,
            "semantic_id": self.semantic_id,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "evidence_digests": dict(self.evidence_digests),
            "record_count": self.record_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "semantic_digest": self.semantic_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageSemanticOutput":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "stage semantic output")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class StageInputBinding:
    input_id: str
    path: str
    sha256: str
    schema: str = STAGE_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_INPUT_SCHEMA:
            raise StageExecutionError("unsupported stage input schema")
        object.__setattr__(self, "input_id", _safe_id(self.input_id, "input_id"))
        if not isinstance(self.path, str) or not self.path.strip():
            raise StageExecutionError("stage input path must be non-empty")
        object.__setattr__(self, "sha256", _digest(self.sha256, "input sha256"))

    @property
    def binding_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "input_id": self.input_id,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageInputBinding":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "stage input")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class StageDependencyBinding:
    stage_id: str
    receipt_path: str
    receipt_file_sha256: str
    receipt_digest: str
    schema: str = STAGE_DEPENDENCY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_DEPENDENCY_SCHEMA:
            raise StageExecutionError("unsupported stage dependency schema")
        if self.stage_id not in PRODUCTION_STAGE_IDS:
            raise StageExecutionError(f"unknown dependency stage: {self.stage_id!r}")
        if not isinstance(self.receipt_path, str) or not self.receipt_path.strip():
            raise StageExecutionError("dependency receipt path must be non-empty")
        object.__setattr__(
            self,
            "receipt_file_sha256",
            _digest(self.receipt_file_sha256, "receipt_file_sha256"),
        )
        object.__setattr__(
            self, "receipt_digest", _digest(self.receipt_digest, "receipt_digest")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_id": self.stage_id,
            "receipt_path": self.receipt_path,
            "receipt_file_sha256": self.receipt_file_sha256,
            "receipt_digest": self.receipt_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageDependencyBinding":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "stage dependency")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class StageExecutionManifest:
    stage_id: str
    execution_id: str
    execution_mode: ExecutionMode
    run_id: str
    freeze_manifest_digest: str
    adapter_id: str
    adapter_contract_digest: str
    parameters_digest: str
    inputs: tuple[StageInputBinding, ...]
    dependencies: tuple[StageDependencyBinding, ...]
    oracle_access_requested: bool = False
    schema: str = STAGE_EXECUTION_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_EXECUTION_MANIFEST_SCHEMA:
            raise StageExecutionError("unsupported stage execution manifest schema")
        if self.stage_id not in PRODUCTION_STAGE_IDS:
            raise StageExecutionError(f"unknown production stage: {self.stage_id!r}")
        object.__setattr__(self, "execution_id", _safe_id(self.execution_id, "execution_id"))
        object.__setattr__(self, "run_id", _safe_id(self.run_id, "run_id"))
        if self.execution_mode not in {DEVELOPMENT_MODE, FORMAL_MODE}:
            raise StageExecutionError("execution_mode must be DEVELOPMENT or FORMAL")
        for name in ("freeze_manifest_digest", "adapter_contract_digest", "parameters_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "adapter_id", _safe_id(self.adapter_id, "adapter_id"))
        if type(self.oracle_access_requested) is not bool:
            raise StageExecutionError("oracle_access_requested must be boolean")
        if self.oracle_access_requested:
            raise StageExecutionError("v0.3 production stages cannot request oracle access")

        inputs = tuple(sorted(self.inputs, key=lambda item: item.input_id))
        if not inputs or not all(isinstance(item, StageInputBinding) for item in inputs):
            raise StageExecutionError("stage execution requires typed physical inputs")
        if len({item.input_id for item in inputs}) != len(inputs):
            raise StageExecutionError("stage input IDs must be unique")
        object.__setattr__(self, "inputs", inputs)

        dependencies = tuple(sorted(self.dependencies, key=lambda item: item.stage_id))
        if not all(isinstance(item, StageDependencyBinding) for item in dependencies):
            raise StageExecutionError("dependencies must be typed stage receipt bindings")
        observed = tuple(item.stage_id for item in dependencies)
        expected = tuple(sorted(REQUIRED_PREDECESSOR[self.stage_id]))
        if observed != expected:
            raise StageExecutionError(
                f"{self.stage_id} requires exact predecessor receipts {expected!r}"
            )
        object.__setattr__(self, "dependencies", dependencies)

    @property
    def manifest_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_id": self.stage_id,
            "execution_id": self.execution_id,
            "execution_mode": self.execution_mode,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "adapter_id": self.adapter_id,
            "adapter_contract_digest": self.adapter_contract_digest,
            "parameters_digest": self.parameters_digest,
            "inputs": [item.to_dict() for item in self.inputs],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "oracle_access_requested": self.oracle_access_requested,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageExecutionManifest":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "stage execution manifest")
        return cls(
            **{
                field: (
                    tuple(StageInputBinding.from_dict(item) for item in data[field])
                    if field == "inputs"
                    else tuple(
                        StageDependencyBinding.from_dict(item) for item in data[field]
                    )
                    if field == "dependencies"
                    else data[field]
                )
                for field in fields
            }
        )


@dataclass(frozen=True)
class StageOutputArtifact:
    domain: str
    path: str
    sha256: str
    semantic_id: str | None = None
    payload_schema: str | None = None
    schema: str = STAGE_OUTPUT_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_OUTPUT_ARTIFACT_SCHEMA:
            raise StageExecutionError("unsupported stage output artifact schema")
        object.__setattr__(self, "domain", _safe_id(self.domain, "output domain"))
        if not isinstance(self.path, str) or not self.path.strip():
            raise StageExecutionError("output artifact path must be non-empty")
        object.__setattr__(self, "sha256", _digest(self.sha256, "output sha256"))
        if (self.semantic_id is None) != (self.payload_schema is None):
            raise StageExecutionError(
                "output semantic_id and payload_schema must be provided together"
            )
        if self.semantic_id is not None:
            object.__setattr__(
                self, "semantic_id", _safe_id(self.semantic_id, "semantic_id")
            )
            if (
                not isinstance(self.payload_schema, str)
                or not self.payload_schema
                or self.payload_schema.strip() != self.payload_schema
            ):
                raise StageExecutionError("output payload_schema must be canonical")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domain": self.domain,
            "path": self.path,
            "sha256": self.sha256,
            "semantic_id": self.semantic_id,
            "payload_schema": self.payload_schema,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageOutputArtifact":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "stage output artifact")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class StageAdapterResult:
    output_payload_digest: str
    artifacts: tuple[StageOutputArtifact, ...]
    record_counts: Mapping[str, int]
    schema: str = STAGE_ADAPTER_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_ADAPTER_RESULT_SCHEMA:
            raise StageExecutionError("unsupported stage adapter result schema")
        object.__setattr__(
            self,
            "output_payload_digest",
            _digest(self.output_payload_digest, "output_payload_digest"),
        )
        artifacts = tuple(sorted(self.artifacts, key=lambda item: (item.domain, item.path)))
        if not artifacts or not all(isinstance(item, StageOutputArtifact) for item in artifacts):
            raise StageExecutionError("adapter result requires typed output artifacts")
        keys = tuple((item.domain, item.path) for item in artifacts)
        if len(set(keys)) != len(keys):
            raise StageExecutionError("adapter result contains duplicate artifacts")
        object.__setattr__(self, "artifacts", artifacts)
        counts: dict[str, int] = {}
        if not isinstance(self.record_counts, Mapping):
            raise StageExecutionError("record_counts must be a mapping")
        for key, value in self.record_counts.items():
            canonical_key = _safe_id(key, "record_counts key")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StageExecutionError("record counts must be non-negative integers")
            counts[canonical_key] = value
        object.__setattr__(self, "record_counts", MappingProxyType(dict(sorted(counts.items()))))

    @property
    def result_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_payload_digest": self.output_payload_digest,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "record_counts": dict(self.record_counts),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageAdapterResult":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "stage adapter result")
        return cls(
            output_payload_digest=data["output_payload_digest"],
            artifacts=tuple(StageOutputArtifact.from_dict(item) for item in data["artifacts"]),
            record_counts=data["record_counts"],
            schema=data["schema"],
        )


@dataclass(frozen=True)
class StageExecutionReceipt:
    stage_id: str
    execution_id: str
    execution_mode: ExecutionMode
    run_id: str
    freeze_manifest_digest: str
    manifest_digest: str
    adapter_id: str
    adapter_contract_digest: str
    verified_input_set_digest: str
    verified_dependency_set_digest: str
    execution_manifest: StageExecutionManifest
    adapter_result: StageAdapterResult
    status: str = "COMPLETE"
    oracle_accessed: bool = False
    schema: str = STAGE_EXECUTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE_EXECUTION_RECEIPT_SCHEMA:
            raise StageExecutionError("unsupported stage execution receipt schema")
        if self.stage_id not in PRODUCTION_STAGE_IDS:
            raise StageExecutionError("receipt has unknown production stage")
        for name in ("execution_id", "run_id", "adapter_id"):
            object.__setattr__(self, name, _safe_id(getattr(self, name), name))
        if self.execution_mode not in {DEVELOPMENT_MODE, FORMAL_MODE}:
            raise StageExecutionError("receipt has invalid execution mode")
        for name in (
            "freeze_manifest_digest",
            "manifest_digest",
            "adapter_contract_digest",
            "verified_input_set_digest",
            "verified_dependency_set_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.execution_manifest, StageExecutionManifest):
            raise StageExecutionError("receipt requires a typed execution manifest")
        manifest = self.execution_manifest
        expected_input_digest = sha256_json(
            {item.input_id: item.binding_digest for item in manifest.inputs}
        )
        expected_dependency_digest = sha256_json(
            {
                item.stage_id: item.receipt_digest
                for item in manifest.dependencies
            }
        )
        if (
            self.stage_id != manifest.stage_id
            or self.execution_id != manifest.execution_id
            or self.execution_mode != manifest.execution_mode
            or self.run_id != manifest.run_id
            or self.freeze_manifest_digest != manifest.freeze_manifest_digest
            or self.manifest_digest != manifest.manifest_digest
            or self.adapter_id != manifest.adapter_id
            or self.adapter_contract_digest != manifest.adapter_contract_digest
            or self.verified_input_set_digest != expected_input_digest
            or self.verified_dependency_set_digest != expected_dependency_digest
        ):
            raise StageExecutionError(
                "stage receipt identity or verified binding differs from execution manifest"
            )
        if not isinstance(self.adapter_result, StageAdapterResult):
            raise StageExecutionError("receipt requires a typed adapter result")
        if self.status != "COMPLETE" or self.oracle_accessed is not False:
            raise StageExecutionError("stage receipt must be COMPLETE and oracle-free")

    @property
    def receipt_digest(self) -> str:
        return sha256_json(self._payload_without_digest())

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_id": self.stage_id,
            "execution_id": self.execution_id,
            "execution_mode": self.execution_mode,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "manifest_digest": self.manifest_digest,
            "adapter_id": self.adapter_id,
            "adapter_contract_digest": self.adapter_contract_digest,
            "verified_input_set_digest": self.verified_input_set_digest,
            "verified_dependency_set_digest": self.verified_dependency_set_digest,
            "execution_manifest": self.execution_manifest.to_dict(),
            "adapter_result": self.adapter_result.to_dict(),
            "status": self.status,
            "oracle_accessed": self.oracle_accessed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageExecutionReceipt":
        fields = set(cls.__dataclass_fields__) | {"receipt_digest"}
        data = _strict(value, fields, "stage execution receipt")
        receipt = cls(
            **{
                field: (
                    StageAdapterResult.from_dict(data[field])
                    if field == "adapter_result"
                    else StageExecutionManifest.from_dict(data[field])
                    if field == "execution_manifest"
                    else data[field]
                )
                for field in cls.__dataclass_fields__
            }
        )
        if _digest(data["receipt_digest"], "receipt_digest") != receipt.receipt_digest:
            raise StageExecutionError("stage execution receipt digest mismatch")
        return receipt


@dataclass(frozen=True)
class PipelineCompletionManifest:
    """Exact physical preconditions for the final aggregate checker.

    The manifest binds, but does not create, the public-ranking barrier, the
    Paper-I oracle handoff, formal statistics output, and independent
    recompute attestation.  Those files remain owned by their respective
    typed producers.
    """

    completion_id: str
    run_id: str
    freeze_manifest_digest: str
    stage_receipts: tuple[StageDependencyBinding, ...]
    external_artifacts: tuple[StageInputBinding, ...]
    schema: str = PIPELINE_COMPLETION_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PIPELINE_COMPLETION_MANIFEST_SCHEMA:
            raise StageExecutionError("unsupported pipeline completion manifest schema")
        object.__setattr__(self, "completion_id", _safe_id(self.completion_id, "completion_id"))
        object.__setattr__(self, "run_id", _safe_id(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "freeze_manifest_digest",
            _digest(self.freeze_manifest_digest, "freeze_manifest_digest"),
        )
        receipts = tuple(sorted(self.stage_receipts, key=lambda item: item.stage_id))
        if (
            not all(isinstance(item, StageDependencyBinding) for item in receipts)
            or tuple(item.stage_id for item in receipts)
            != tuple(sorted(PRODUCTION_STAGE_IDS))
        ):
            raise StageExecutionError(
                "completion requires exactly one receipt for every production stage"
            )
        object.__setattr__(self, "stage_receipts", receipts)
        external = tuple(sorted(self.external_artifacts, key=lambda item: item.input_id))
        if (
            not all(isinstance(item, StageInputBinding) for item in external)
            or tuple(item.input_id for item in external)
            != REQUIRED_COMPLETION_EXTERNAL_ARTIFACT_IDS
        ):
            raise StageExecutionError(
                "completion requires the exact reviewed formal evidence chain"
            )
        object.__setattr__(self, "external_artifacts", external)

    @property
    def manifest_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "completion_id": self.completion_id,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "stage_receipts": [item.to_dict() for item in self.stage_receipts],
            "external_artifacts": [item.to_dict() for item in self.external_artifacts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineCompletionManifest":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "pipeline completion manifest")
        return cls(
            completion_id=data["completion_id"],
            run_id=data["run_id"],
            freeze_manifest_digest=data["freeze_manifest_digest"],
            stage_receipts=tuple(
                StageDependencyBinding.from_dict(item) for item in data["stage_receipts"]
            ),
            external_artifacts=tuple(
                StageInputBinding.from_dict(item) for item in data["external_artifacts"]
            ),
            schema=data["schema"],
        )


@dataclass(frozen=True)
class PipelineCompletionReceipt:
    completion_id: str
    run_id: str
    freeze_manifest_digest: str
    completion_manifest_digest: str
    stage_receipt_digests: Mapping[str, str]
    external_artifact_set_digest: str
    status: str = "COMPLETE_PRECONDITIONS_VERIFIED"
    oracle_read_by_v03_driver: bool = False
    schema: str = PIPELINE_COMPLETION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PIPELINE_COMPLETION_RECEIPT_SCHEMA:
            raise StageExecutionError("unsupported pipeline completion receipt schema")
        for name in ("completion_id", "run_id"):
            object.__setattr__(self, name, _safe_id(getattr(self, name), name))
        for name in (
            "freeze_manifest_digest",
            "completion_manifest_digest",
            "external_artifact_set_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        receipts = dict(sorted(self.stage_receipt_digests.items()))
        if tuple(receipts) != tuple(sorted(PRODUCTION_STAGE_IDS)):
            raise StageExecutionError("completion receipt stage coverage is incomplete")
        for stage_id, digest in receipts.items():
            receipts[stage_id] = _digest(digest, f"stage receipt {stage_id}")
        object.__setattr__(self, "stage_receipt_digests", MappingProxyType(receipts))
        if (
            self.status != "COMPLETE_PRECONDITIONS_VERIFIED"
            or self.oracle_read_by_v03_driver is not False
        ):
            raise StageExecutionError("invalid pipeline completion status")

    @property
    def receipt_digest(self) -> str:
        return sha256_json(self._payload_without_digest())

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "completion_id": self.completion_id,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "completion_manifest_digest": self.completion_manifest_digest,
            "stage_receipt_digests": dict(self.stage_receipt_digests),
            "external_artifact_set_digest": self.external_artifact_set_digest,
            "status": self.status,
            "oracle_read_by_v03_driver": self.oracle_read_by_v03_driver,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineCompletionReceipt":
        fields = set(cls.__dataclass_fields__) | {"receipt_digest"}
        data = _strict(value, fields, "pipeline completion receipt")
        receipt = cls(
            **{field: data[field] for field in cls.__dataclass_fields__}
        )
        if _digest(data["receipt_digest"], "receipt_digest") != receipt.receipt_digest:
            raise StageExecutionError("pipeline completion receipt digest mismatch")
        return receipt


@dataclass(frozen=True)
class StageExecutionContext:
    manifest: StageExecutionManifest
    freeze_manifest: PreExperimentFreezeManifest
    layout: V03ArtifactLayout
    verified_inputs: Mapping[str, Path]
    verified_dependencies: Mapping[str, StageExecutionReceipt]


class StageExecutionAdapter(Protocol):
    """Server-owned implementation injected into the reviewed driver."""

    @property
    def adapter_id(self) -> str: ...

    @property
    def adapter_contract_digest(self) -> str: ...

    def execute(self, context: StageExecutionContext) -> StageAdapterResult: ...


def _verify_inputs(manifest: StageExecutionManifest) -> tuple[Mapping[str, Path], str]:
    verified: dict[str, Path] = {}
    bindings: dict[str, str] = {}
    for item in manifest.inputs:
        path = _regular_file(item.path, f"input {item.input_id}")
        actual = sha256_file(path)
        if actual != item.sha256:
            raise StageExecutionError(f"stage input digest mismatch: {item.input_id}")
        verified[item.input_id] = path
        bindings[item.input_id] = item.binding_digest
    return MappingProxyType(verified), sha256_json(bindings)


def _load_dependency(
    binding: StageDependencyBinding,
    *,
    manifest: StageExecutionManifest,
) -> StageExecutionReceipt:
    path = _regular_file(binding.receipt_path, f"dependency {binding.stage_id}")
    if sha256_file(path) != binding.receipt_file_sha256:
        raise StageExecutionError(f"dependency receipt byte digest mismatch: {binding.stage_id}")
    receipt = StageExecutionReceipt.from_dict(
        _canonical_json_file(path, f"dependency receipt {binding.stage_id}")
    )
    if receipt.receipt_digest != binding.receipt_digest:
        raise StageExecutionError(f"dependency semantic digest mismatch: {binding.stage_id}")
    if receipt.stage_id != binding.stage_id:
        raise StageExecutionError(f"dependency stage identity mismatch: {binding.stage_id}")
    if (
        receipt.freeze_manifest_digest != manifest.freeze_manifest_digest
        or receipt.execution_mode != manifest.execution_mode
        or receipt.run_id != manifest.run_id
    ):
        raise StageExecutionError("dependency crosses freeze, mode, or run boundary")
    return receipt


def _verify_dependencies(
    manifest: StageExecutionManifest,
    *,
    verified_inputs: Mapping[str, Path],
) -> tuple[Mapping[str, StageExecutionReceipt], str]:
    verified = {
        binding.stage_id: _load_dependency(binding, manifest=manifest)
        for binding in manifest.dependencies
    }
    input_bindings_by_physical_path: dict[Path, StageInputBinding] = {}
    for item in manifest.inputs:
        physical_path = verified_inputs[item.input_id]
        if physical_path in input_bindings_by_physical_path:
            raise StageExecutionError(
                "stage manifest cannot bind the same physical input path twice"
            )
        input_bindings_by_physical_path[physical_path] = item
    for dependency_binding in manifest.dependencies:
        dependency = verified[dependency_binding.stage_id]
        receipt_path = _regular_file(
            dependency_binding.receipt_path,
            f"dependency {dependency_binding.stage_id}",
        )
        dependency_run_root = receipt_path.parents[2]
        for output in dependency.adapter_result.artifacts:
            output_path = _regular_file(
                dependency_run_root / output.path,
                f"dependency output {dependency_binding.stage_id}:{output.path}",
            )
            input_binding = input_bindings_by_physical_path.get(output_path)
            if input_binding is None or input_binding.sha256 != output.sha256:
                raise StageExecutionError(
                    "successor inputs must bind every predecessor output by exact path and sha256"
                )
    digest = sha256_json(
        {stage_id: receipt.receipt_digest for stage_id, receipt in sorted(verified.items())}
    )
    return MappingProxyType(verified), digest


def _verify_formal_adapter_authorization(
    manifest: StageExecutionManifest,
    freeze_manifest: PreExperimentFreezeManifest,
) -> None:
    try:
        expected = freeze_manifest.formal_stage_adapter_binding_digests[
            manifest.stage_id
        ]
    except KeyError as error:
        raise StageExecutionError(
            f"formal freeze does not authorize an adapter for {manifest.stage_id}"
        ) from error
    observed = formal_stage_adapter_binding_digest(
        manifest.stage_id,
        manifest.adapter_id,
        manifest.adapter_contract_digest,
    )
    if observed != expected:
        raise StageExecutionError(
            "formal adapter identity/contract is not allowlisted by the freeze"
        )


def _observed_formal_stage_request_template(
    manifest: StageExecutionManifest,
    *,
    verified_inputs: Mapping[str, Path],
    verified_dependencies: Mapping[str, StageExecutionReceipt],
) -> FormalStageRequestTemplate:
    predecessor_paths: set[Path] = set()
    for dependency_binding in manifest.dependencies:
        dependency = verified_dependencies[dependency_binding.stage_id]
        receipt_path = _regular_file(
            dependency_binding.receipt_path,
            f"dependency {dependency_binding.stage_id}",
        )
        dependency_run_root = receipt_path.parents[2]
        for output in dependency.adapter_result.artifacts:
            predecessor_paths.add(
                _regular_file(
                    dependency_run_root / output.path,
                    f"dependency output {dependency_binding.stage_id}:{output.path}",
                )
            )
    static_inputs = {
        item.input_id: item.sha256
        for item in manifest.inputs
        if verified_inputs[item.input_id] not in predecessor_paths
    }
    return FormalStageRequestTemplate(
        stage_id=manifest.stage_id,
        adapter_id=manifest.adapter_id,
        adapter_contract_digest=manifest.adapter_contract_digest,
        parameters_digest=manifest.parameters_digest,
        static_input_content_digests=static_inputs,
    )


def _verify_formal_request_authorization(
    manifest: StageExecutionManifest,
    freeze_manifest: PreExperimentFreezeManifest,
    *,
    verified_inputs: Mapping[str, Path],
    verified_dependencies: Mapping[str, StageExecutionReceipt],
) -> None:
    registry = getattr(
        freeze_manifest, "formal_stage_request_template_digests", None
    )
    if not isinstance(registry, Mapping):
        raise StageExecutionError(
            "formal freeze lacks the exact stage request template registry"
        )
    try:
        expected = registry[manifest.stage_id]
    except KeyError as error:
        raise StageExecutionError(
            f"formal freeze lacks request template for {manifest.stage_id}"
        ) from error
    observed = _observed_formal_stage_request_template(
        manifest,
        verified_inputs=verified_inputs,
        verified_dependencies=verified_dependencies,
    )
    if observed.request_template_digest != expected:
        raise StageExecutionError(
            "formal stage adapter/parameters/static inputs differ from the reviewed request template"
        )


def _verify_receipt_against_request(
    receipt: StageExecutionReceipt,
    *,
    manifest: StageExecutionManifest,
    freeze_manifest: PreExperimentFreezeManifest,
    input_set_digest: str,
    dependency_set_digest: str,
) -> None:
    if (
        receipt.stage_id != manifest.stage_id
        or receipt.execution_id != manifest.execution_id
        or receipt.execution_mode != manifest.execution_mode
        or receipt.run_id != manifest.run_id
        or receipt.adapter_id != manifest.adapter_id
        or receipt.adapter_contract_digest != manifest.adapter_contract_digest
        or receipt.execution_manifest != manifest
        or receipt.manifest_digest != manifest.manifest_digest
        or receipt.freeze_manifest_digest != freeze_manifest.freeze_manifest_digest
        or receipt.verified_input_set_digest != input_set_digest
        or receipt.verified_dependency_set_digest != dependency_set_digest
    ):
        raise StageExecutionError(
            "stage receipt does not match the complete execution request"
        )


def _verify_output_artifact(
    item: StageOutputArtifact,
    *,
    stage_id: str,
    layout: V03ArtifactLayout,
    execution_mode: ExecutionMode,
    freeze_manifest: PreExperimentFreezeManifest,
    run_id: str,
) -> tuple[str, str, int, frozenset[str]] | None:
    if item.domain != "cost" and item.domain not in _ALLOWED_OUTPUT_DOMAINS[stage_id]:
        raise StageExecutionError(
            f"{stage_id} cannot claim output domain {item.domain!r}"
        )
    if item.domain not in layout.domains:
        raise StageExecutionError(
            f"output domain {item.domain!r} is unavailable in {layout.namespace}"
        )
    path = layout.assert_domain(layout.run_root / item.path, item.domain)
    if not path.is_file() or path.is_symlink():
        raise StageExecutionError(f"stage output is not a regular artifact: {item.path}")
    if sha256_file(path) != item.sha256:
        raise StageExecutionError(f"stage output digest mismatch: {item.path}")
    if item.domain == "cost":
        payload = _canonical_json_file(path, "stage cost artifact")
        try:
            if payload.get("schema") == COST_COMPONENT_SCHEMA:
                CostComponentRecord.from_dict(payload)
            elif payload.get("schema") == COST_LEDGER_SCHEMA:
                ledger = V03CostLedger.from_dict(payload)
                if (
                    ledger.cost_protocol_digest
                    != freeze_manifest.cost_protocol_digest
                ):
                    raise StageExecutionError(
                        "cost ledger differs from the frozen cost protocol"
                    )
                if execution_mode == FORMAL_MODE and (
                    ledger.execution_scope != "FORMAL"
                    or ledger.run_id != run_id
                    or ledger.freeze_manifest_digest
                    != freeze_manifest.freeze_manifest_digest
                ):
                    raise StageExecutionError(
                        "formal stage cost ledger crosses run or freeze boundary"
                    )
            else:
                raise StageExecutionError(
                    "cost output must be a typed cost component or cost ledger"
                )
        except ValueError as error:
            raise StageExecutionError(f"invalid typed cost output: {error}") from error
        return None
    if execution_mode == FORMAL_MODE:
        if (item.semantic_id is None) != (item.payload_schema is None):
            raise StageExecutionError(
                "formal stage output semantic_id and payload_schema must appear together"
            )
        # A formal adapter may emit immutable underlying evidence in addition
        # to the exact principal semantic summaries.  It cannot satisfy a
        # semantic requirement by the file alone: the typed summary must name
        # its byte digest, and the aggregate verifier below requires a unique
        # in-result evidence witness for every generic semantic output.
        if item.semantic_id is None:
            return None
        requirements = {
            requirement.semantic_id: requirement
            for requirement in STAGE_SEMANTIC_REQUIREMENTS[stage_id]
        }
        try:
            requirement = requirements[item.semantic_id]
        except KeyError as error:
            raise StageExecutionError(
                f"{stage_id} emitted an unregistered semantic output"
            ) from error
        if item.domain != requirement.domain:
            raise StageExecutionError(
                f"semantic output {item.semantic_id!r} requires domain "
                f"{requirement.domain!r}"
            )
        if item.payload_schema != requirement.payload_schema:
            raise StageExecutionError(
                f"semantic output {item.semantic_id!r} has an unreviewed payload schema"
            )
        payload = _canonical_json_file(path, "formal stage semantic output")
        if payload.get("schema") != item.payload_schema:
            raise StageExecutionError(
                "formal output payload schema differs from its artifact binding"
            )
        if item.payload_schema == STAGE_SEMANTIC_OUTPUT_SCHEMA:
            semantic = StageSemanticOutput.from_dict(payload)
            if (
                semantic.stage_id != stage_id
                or semantic.semantic_id != item.semantic_id
                or semantic.run_id != run_id
                or semantic.freeze_manifest_digest
                != freeze_manifest.freeze_manifest_digest
            ):
                raise StageExecutionError(
                    "formal semantic output crosses stage, run, or freeze boundary"
                )
            semantic_digest = str(semantic.semantic_digest)
            record_count = semantic.record_count
        elif item.payload_schema == (
            "policy-learnware.v03-formal-signal-readout-bundle.v0"
        ):
            bundle = _parse_formal_signal_readout_bundle_manifest(payload)
            if (
                bundle["freeze_manifest_digest"]
                != freeze_manifest.freeze_manifest_digest
                or bundle["readout_plan_digest"]
                != freeze_manifest.formal_signal_readout_plan_digest
            ):
                raise StageExecutionError(
                    "formal signal readout bundle differs from the reviewed freeze"
                )
            semantic_digest = str(bundle["bundle_digest"])
            record_count = 39
        elif item.payload_schema == (
            "policy-learnware.v03-public-signal-readout-bundle.v0"
        ):
            bundle = _parse_public_signal_readout_bundle(payload)
            if (
                bundle["freeze_manifest_digest"]
                != freeze_manifest.freeze_manifest_digest
                or bundle["readout_plan_digest"]
                != freeze_manifest.formal_signal_readout_plan_digest
            ):
                raise StageExecutionError(
                    "public signal readout bundle differs from the reviewed freeze"
                )
            semantic_digest = str(bundle["public_projection_digest"])
            record_count = 39
        else:  # pragma: no cover - registry and parser table must evolve together
            raise StageExecutionError("formal output schema has no semantic parser")
        requirement.validate_count(record_count)
        bound_evidence = (
            frozenset(semantic.evidence_digests.values())
            if item.payload_schema == STAGE_SEMANTIC_OUTPUT_SCHEMA
            else frozenset()
        )
        return item.semantic_id, semantic_digest, record_count, bound_evidence
    if stage_id == "run-public-rankings":
        # Ranking material is public and must be canonical JSON.  Domain-level
        # publishers perform the stronger method-specific field audit.
        _canonical_json_file(path, "public ranking artifact")
    return None


def formal_stage_output_payload_digest(
    artifacts: Sequence[StageOutputArtifact],
    semantic_output_digests: Mapping[str, str],
) -> str:
    """Canonical digest adapters must place in a formal StageAdapterResult."""

    return sha256_json(
        {
            "artifact_byte_digests": {
                f"{item.domain}:{item.path}": item.sha256
                for item in sorted(artifacts, key=lambda value: (value.domain, value.path))
            },
            "semantic_output_digests": {
                semantic_id: digest
                for semantic_id, digest in sorted(semantic_output_digests.items())
            },
        }
    )


def _verify_stage_adapter_result(
    result: StageAdapterResult,
    *,
    stage_id: str,
    execution_mode: ExecutionMode,
    layout: V03ArtifactLayout,
    freeze_manifest: PreExperimentFreezeManifest,
    run_id: str,
) -> None:
    semantic_output_digests: dict[str, str] = {}
    semantic_record_counts: dict[str, int] = {}
    generic_evidence_digests: dict[str, frozenset[str]] = {}
    underlying_artifact_digests: set[str] = set()
    for item in result.artifacts:
        verified_semantic = _verify_output_artifact(
            item,
            stage_id=stage_id,
            layout=layout,
            execution_mode=execution_mode,
            freeze_manifest=freeze_manifest,
            run_id=run_id,
        )
        if verified_semantic is None:
            if (
                execution_mode == FORMAL_MODE
                and item.domain != "cost"
                and item.semantic_id is None
                and item.payload_schema is None
            ):
                underlying_artifact_digests.add(item.sha256)
            continue
        semantic_id, semantic_digest, record_count, evidence_digests = (
            verified_semantic
        )
        if semantic_id in semantic_output_digests:
            raise StageExecutionError("formal stage emitted duplicate semantic output")
        semantic_output_digests[semantic_id] = semantic_digest
        semantic_record_counts[semantic_id] = record_count
        if evidence_digests:
            generic_evidence_digests[semantic_id] = evidence_digests
    if execution_mode != FORMAL_MODE:
        return
    expected_ids = {
        requirement.semantic_id
        for requirement in STAGE_SEMANTIC_REQUIREMENTS[stage_id]
    }
    if set(semantic_output_digests) != expected_ids:
        missing = sorted(expected_ids - set(semantic_output_digests))
        extra = sorted(set(semantic_output_digests) - expected_ids)
        raise StageExecutionError(
            f"formal stage semantic output coverage mismatch; missing={missing}, extra={extra}"
        )
    if dict(result.record_counts) != dict(sorted(semantic_record_counts.items())):
        raise StageExecutionError(
            "formal stage record_counts differ from typed semantic outputs"
        )
    evidence_candidates = {
        semantic_id: tuple(sorted(digests & underlying_artifact_digests))
        for semantic_id, digests in generic_evidence_digests.items()
    }
    if any(not values for values in evidence_candidates.values()):
        raise StageExecutionError(
            "formal generic semantic output lacks a bound underlying artifact byte digest"
        )
    # Require an injective semantic->artifact witness.  This prevents two
    # principal outputs from being justified solely by one unrelated file.
    assigned: dict[str, str] = {}

    def _assign(semantic_id: str, seen: set[str]) -> bool:
        for digest in evidence_candidates[semantic_id]:
            if digest in seen:
                continue
            seen.add(digest)
            owner = assigned.get(digest)
            if owner is None or _assign(owner, seen):
                assigned[digest] = semantic_id
                return True
        return False

    if any(
        not _assign(semantic_id, set())
        for semantic_id in sorted(evidence_candidates)
    ):
        raise StageExecutionError(
            "formal generic semantic outputs require distinct underlying evidence artifacts"
        )
    expected_payload_digest = formal_stage_output_payload_digest(
        result.artifacts, semantic_output_digests
    )
    if result.output_payload_digest != expected_payload_digest:
        raise StageExecutionError(
            "formal stage output payload digest differs from verified semantic artifacts"
        )


def _receipt_path(layout: V03ArtifactLayout, execution_id: str) -> Path:
    return layout.artifact("scope", "stage_executions", f"{execution_id}.json")


def execute_stage(
    manifest: StageExecutionManifest,
    freeze_manifest: PreExperimentFreezeManifest,
    *,
    artifacts_root: str | Path,
    adapter: StageExecutionAdapter,
    resume: bool = False,
) -> tuple[StageExecutionReceipt, Path, str, bool]:
    """Execute or byte-exactly resume one reviewed production stage.

    Returns ``(receipt, receipt_path, receipt_file_sha256, resumed)``.
    """

    if manifest.freeze_manifest_digest != freeze_manifest.freeze_manifest_digest:
        raise StageExecutionError("stage manifest is bound to a different freeze")
    if manifest.execution_mode == FORMAL_MODE:
        if not freeze_manifest.formal_run_authorized:
            raise StageExecutionError("formal stage requires external review authority")
        _verify_formal_adapter_authorization(manifest, freeze_manifest)
        layout = V03ArtifactLayout.joint(artifacts_root, manifest.run_id)
    else:
        if freeze_manifest.formal_run_authorized:
            raise StageExecutionError(
                "development stage cannot consume a formal-authority freeze"
            )
        layout = V03ArtifactLayout.development(artifacts_root, manifest.run_id)

    if adapter is None:  # type: ignore[unreachable]
        raise StageExecutionError("production execution requires an injected stage adapter")
    if adapter.adapter_id != manifest.adapter_id:
        raise StageExecutionError("injected adapter identity differs from stage manifest")
    if adapter.adapter_contract_digest != manifest.adapter_contract_digest:
        raise StageExecutionError("injected adapter contract differs from stage manifest")

    input_paths, input_set_digest = _verify_inputs(manifest)
    dependency_receipts, dependency_set_digest = _verify_dependencies(
        manifest,
        verified_inputs=input_paths,
    )
    if manifest.execution_mode == FORMAL_MODE:
        _verify_formal_request_authorization(
            manifest,
            freeze_manifest,
            verified_inputs=input_paths,
            verified_dependencies=dependency_receipts,
        )
    receipt_path = _receipt_path(layout, manifest.execution_id)

    if receipt_path.exists():
        if not resume:
            raise StageExecutionError("stage execution receipt already exists; use --resume")
        existing = StageExecutionReceipt.from_dict(
            _canonical_json_file(receipt_path, "stage execution receipt")
        )
        _verify_receipt_against_request(
            existing,
            manifest=manifest,
            freeze_manifest=freeze_manifest,
            input_set_digest=input_set_digest,
            dependency_set_digest=dependency_set_digest,
        )
        _verify_stage_adapter_result(
            existing.adapter_result,
            stage_id=manifest.stage_id,
            execution_mode=manifest.execution_mode,
            layout=layout,
            freeze_manifest=freeze_manifest,
            run_id=manifest.run_id,
        )
        return existing, receipt_path, sha256_file(receipt_path), True

    context = StageExecutionContext(
        manifest=manifest,
        freeze_manifest=freeze_manifest,
        layout=layout,
        verified_inputs=input_paths,
        verified_dependencies=dependency_receipts,
    )
    result = adapter.execute(context)
    if not isinstance(result, StageAdapterResult):
        raise StageExecutionError("stage adapter must return StageAdapterResult")
    _verify_stage_adapter_result(
        result,
        stage_id=manifest.stage_id,
        execution_mode=manifest.execution_mode,
        layout=layout,
        freeze_manifest=freeze_manifest,
        run_id=manifest.run_id,
    )

    receipt = StageExecutionReceipt(
        stage_id=manifest.stage_id,
        execution_id=manifest.execution_id,
        execution_mode=manifest.execution_mode,
        run_id=manifest.run_id,
        freeze_manifest_digest=freeze_manifest.freeze_manifest_digest,
        manifest_digest=manifest.manifest_digest,
        adapter_id=manifest.adapter_id,
        adapter_contract_digest=manifest.adapter_contract_digest,
        verified_input_set_digest=input_set_digest,
        verified_dependency_set_digest=dependency_set_digest,
        execution_manifest=manifest,
        adapter_result=result,
    )
    receipt_sha256 = layout.writer("scope").publish_json(
        receipt_path, receipt.to_dict(), resume=False
    )
    return receipt, receipt_path, receipt_sha256, False


def load_stage_execution_manifest(path: str | Path) -> StageExecutionManifest:
    source = _regular_file(path, "stage execution manifest")
    return StageExecutionManifest.from_dict(
        _canonical_json_file(source, "stage execution manifest")
    )


def load_preexperiment_freeze(path: str | Path) -> PreExperimentFreezeManifest:
    source = _regular_file(path, "pre-experiment freeze manifest")
    return PreExperimentFreezeManifest.from_dict(
        _canonical_json_file(source, "pre-experiment freeze manifest")
    )


def execute_stage_from_files(
    *,
    expected_stage_id: str,
    stage_manifest_path: str | Path,
    freeze_manifest_path: str | Path,
    artifacts_root: str | Path,
    adapters: Mapping[str, StageExecutionAdapter] | None,
    resume: bool = False,
) -> tuple[StageExecutionReceipt, Path, str, bool]:
    manifest = load_stage_execution_manifest(stage_manifest_path)
    if manifest.stage_id != expected_stage_id:
        raise StageExecutionError(
            f"CLI command {expected_stage_id!r} received manifest for {manifest.stage_id!r}"
        )
    if adapters is None or manifest.adapter_id not in adapters:
        raise StageExecutionError(
            "production execution requires a server-injected adapter registry; "
            "the CLI has no fallback or self-authorizing adapter"
        )
    freeze = load_preexperiment_freeze(freeze_manifest_path)
    return execute_stage(
        manifest,
        freeze,
        artifacts_root=artifacts_root,
        adapter=adapters[manifest.adapter_id],
        resume=resume,
    )


_READOUT_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "readout_plan_digest",
        "freeze_manifest_digest",
        "formal_authorization_digest",
        "atlas_run_digest",
        "atlas_public_projection_digest",
        "prefix_run_digests",
        "prefix_public_projection_digests",
        "dynamics_diagnostic_digests",
        "dynamics_public_projection_digests",
        "dynamics_public_query_join_digest",
        "dynamics_query_join_public_projection_digest",
        "contrast_gate_evaluation_digest",
        "contrast_gate_public_projection_digest",
        "pair_control_evidence_set_digest",
        "attribution_gate_evidence_digest",
        "bundle_digest",
    }
)

_PUBLIC_READOUT_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "readout_plan_digest",
        "freeze_manifest_digest",
        "formal_authorization_digest",
        "atlas",
        "prefix_readouts",
        "dynamics_readouts",
        "dynamics_query_join",
        "contrast_gate",
        "pair_control_evidence_set_digest",
        "pair_control_evidence_count",
        "attribution_gate_evidence_digest",
        "private_bank_task_context_and_alias_rows_withheld",
        "private_bundle_digest",
        "public_projection_digest",
    }
)

def _parse_formal_signal_readout_bundle_manifest(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    # FormalSignalReadoutBundle.to_dict() is intentionally a compact manifest,
    # not a reconstruction of private raw rows.  Validate that manifest here
    # without importing or materializing the private atlas graph.
    from .signal_readout import FORMAL_SIGNAL_READOUT_BUNDLE_SCHEMA

    data = dict(_strict(value, set(_READOUT_BUNDLE_FIELDS), "signal readout bundle"))
    if data["schema"] != FORMAL_SIGNAL_READOUT_BUNDLE_SCHEMA:
        raise StageExecutionError("unsupported formal signal readout bundle schema")
    for name in _READOUT_BUNDLE_FIELDS - {
        "schema",
        "prefix_run_digests",
        "prefix_public_projection_digests",
        "dynamics_diagnostic_digests",
        "dynamics_public_projection_digests",
    }:
        data[name] = _digest(data[name], f"signal readout {name}")
    for name in (
        "prefix_run_digests",
        "prefix_public_projection_digests",
        "dynamics_diagnostic_digests",
        "dynamics_public_projection_digests",
    ):
        values = data[name]
        if not isinstance(values, Mapping) or not values:
            raise StageExecutionError(f"signal readout {name} must be non-empty")
        data[name] = {
            _safe_id(key, f"signal readout {name} key"): _digest(
                digest, f"signal readout {name}[{key}]"
            )
            for key, digest in values.items()
        }
    supplied = data["bundle_digest"]
    body = dict(data)
    body.pop("bundle_digest")
    if supplied != sha256_json(body):
        raise StageExecutionError("formal signal readout bundle digest mismatch")
    return MappingProxyType(data)


def _parse_public_signal_readout_bundle(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    from .signal_readout import (
        PUBLIC_SIGNAL_READOUT_BUNDLE_SCHEMA,
        _assert_public_projection_has_no_private_keys,
    )

    data = dict(
        _strict(
            value, set(_PUBLIC_READOUT_BUNDLE_FIELDS), "public signal readout bundle"
        )
    )
    if data["schema"] != PUBLIC_SIGNAL_READOUT_BUNDLE_SCHEMA:
        raise StageExecutionError("unsupported public signal readout bundle schema")
    for name in (
        "readout_plan_digest",
        "freeze_manifest_digest",
        "formal_authorization_digest",
        "pair_control_evidence_set_digest",
        "attribution_gate_evidence_digest",
        "private_bundle_digest",
        "public_projection_digest",
    ):
        data[name] = _digest(data[name], f"public signal readout {name}")
    if data["private_bank_task_context_and_alias_rows_withheld"] is not True:
        raise StageExecutionError("public signal readout does not withhold private rows")
    if (
        isinstance(data["pair_control_evidence_count"], bool)
        or not isinstance(data["pair_control_evidence_count"], int)
        or data["pair_control_evidence_count"] <= 0
    ):
        raise StageExecutionError("public signal readout pair evidence count is invalid")
    for name in (
        "atlas",
        "prefix_readouts",
        "dynamics_readouts",
        "dynamics_query_join",
        "contrast_gate",
    ):
        if not isinstance(data[name], Mapping) or not data[name]:
            raise StageExecutionError(f"public signal readout {name} must be non-empty")
    supplied = data["public_projection_digest"]
    body = dict(data)
    body.pop("public_projection_digest")
    if supplied != sha256_json(body):
        raise StageExecutionError("public signal readout projection digest mismatch")
    try:
        _assert_public_projection_has_no_private_keys(data)
    except ValueError as error:
        raise StageExecutionError(str(error)) from error
    return MappingProxyType(data)


def _late_typed_record(
    value: Mapping[str, Any], *, module_name: str, class_name: str, where: str
) -> Any:
    """Load a strict typed record after all concurrently developed modules exist."""

    try:
        module = import_module(module_name, package=__package__)
        record_type = getattr(module, class_name)
        parser = getattr(record_type, "from_dict")
        return parser(value)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError) as error:
        raise StageExecutionError(f"invalid typed {where}: {error}") from error


def verify_pipeline_completion(
    manifest: PipelineCompletionManifest,
    freeze_manifest: PreExperimentFreezeManifest,
    *,
    artifacts_root: str | Path,
    resume: bool = False,
) -> tuple[PipelineCompletionReceipt, Path, str, bool]:
    """Verify the complete stage chain and externally produced final artifacts."""

    if manifest.freeze_manifest_digest != freeze_manifest.freeze_manifest_digest:
        raise StageExecutionError("completion manifest is bound to a different freeze")
    if not freeze_manifest.formal_run_authorized:
        raise StageExecutionError("pipeline completion requires external review authority")
    layout = V03ArtifactLayout.joint(artifacts_root, manifest.run_id)

    receipts: dict[str, StageExecutionReceipt] = {}
    receipt_bindings: dict[str, StageDependencyBinding] = {}
    for binding in manifest.stage_receipts:
        path = _regular_file(binding.receipt_path, f"completion stage {binding.stage_id}")
        if sha256_file(path) != binding.receipt_file_sha256:
            raise StageExecutionError(
                f"completion stage receipt byte mismatch: {binding.stage_id}"
            )
        receipt = StageExecutionReceipt.from_dict(
            _canonical_json_file(path, f"completion stage receipt {binding.stage_id}")
        )
        if (
            receipt.stage_id != binding.stage_id
            or receipt.receipt_digest != binding.receipt_digest
        ):
            raise StageExecutionError(
                f"completion stage receipt identity mismatch: {binding.stage_id}"
            )
        if (
            receipt.execution_mode != FORMAL_MODE
            or receipt.run_id != manifest.run_id
            or receipt.freeze_manifest_digest != manifest.freeze_manifest_digest
        ):
            raise StageExecutionError(
                "completion stage receipt crosses formal freeze or run boundary"
            )
        _verify_stage_adapter_result(
            receipt.adapter_result,
            stage_id=receipt.stage_id,
            execution_mode=receipt.execution_mode,
            layout=layout,
            freeze_manifest=freeze_manifest,
            run_id=receipt.run_id,
        )
        receipts[binding.stage_id] = receipt
        receipt_bindings[binding.stage_id] = binding

    execution_ids = tuple(item.execution_id for item in receipts.values())
    if len(set(execution_ids)) != len(execution_ids):
        raise StageExecutionError("completion stage execution IDs must be unique")
    for stage_id in PRODUCTION_STAGE_IDS:
        receipt = receipts[stage_id]
        stage_manifest = receipt.execution_manifest
        if (
            stage_manifest.execution_mode != FORMAL_MODE
            or stage_manifest.run_id != manifest.run_id
            or stage_manifest.freeze_manifest_digest
            != freeze_manifest.freeze_manifest_digest
        ):
            raise StageExecutionError(
                "completion execution manifest crosses formal freeze or run boundary"
            )
        _verify_formal_adapter_authorization(stage_manifest, freeze_manifest)
        for dependency in stage_manifest.dependencies:
            predecessor = receipt_bindings[dependency.stage_id]
            if dependency != predecessor:
                raise StageExecutionError(
                    "successor manifest does not bind the exact predecessor receipt bytes"
                )
        verified_inputs, input_set_digest = _verify_inputs(stage_manifest)
        verified_dependencies, dependency_set_digest = _verify_dependencies(
            stage_manifest,
            verified_inputs=verified_inputs,
        )
        _verify_formal_request_authorization(
            stage_manifest,
            freeze_manifest,
            verified_inputs=verified_inputs,
            verified_dependencies=verified_dependencies,
        )
        _verify_receipt_against_request(
            receipt,
            manifest=stage_manifest,
            freeze_manifest=freeze_manifest,
            input_set_digest=input_set_digest,
            dependency_set_digest=dependency_set_digest,
        )

    external_bindings: dict[str, str] = {}
    external_values: dict[str, Mapping[str, Any]] = {}
    for item in manifest.external_artifacts:
        path = _regular_file(item.path, f"completion external artifact {item.input_id}")
        if sha256_file(path) != item.sha256:
            raise StageExecutionError(
                f"completion external artifact digest mismatch: {item.input_id}"
            )
        external_values[item.input_id] = _canonical_json_file(
            path, f"completion external artifact {item.input_id}"
        )
        external_bindings[item.input_id] = item.binding_digest

    try:
        attribution = _late_typed_record(
            external_values["formal-attribution-admission"],
            module_name=".formal_gates",
            class_name="FormalAttributionAdmission",
            where="formal attribution admission",
        )
        probe = _late_typed_record(
            external_values["formal-probe-admission"],
            module_name=".formal_gates",
            class_name="FormalProbeAdmission",
            where="formal probe admission",
        )
        market = _late_typed_record(
            external_values["formal-market-admission"],
            module_name=".formal_gates",
            class_name="FormalMarketAdmission",
            where="formal market admission",
        )
        readout = _parse_formal_signal_readout_bundle_manifest(
            external_values["formal-signal-readout-bundle"]
        )
        cost = V03CostLedger.from_dict(external_values["formal-cost-ledger"])
        signal_publication = _late_typed_record(
            external_values["pre-oracle-signal-manifest"],
            module_name=".preoracle_signal",
            class_name="PreOracleSignalOutcomePublication",
            where="pre-oracle signal publication",
        )
        barrier = PublicRankingBarrier.from_dict(
            external_values["public-ranking-barrier"]
        )
        handoff = OracleUnlockHandoff.from_dict(
            external_values["oracle-unlock-handoff"]
        )
        statistics = FormalStatisticsResult.from_dict(
            external_values["statistics-result"]
        )
        recompute = IndependentRecomputeAttestation.from_dict(
            external_values["independent-recompute-attestation"]
        )
        claim = FormalClaimAudit.from_dict(external_values["formal-claim-audit"])
    except (TypeError, ValueError, KeyError) as error:
        raise StageExecutionError(
            f"completion external artifact is not a valid typed v0.3 record: {error}"
        ) from error

    gate_bindings = (
        (
            attribution,
            "G03-Attribution",
            "PASS",
            "formal attribution",
        ),
        (probe, "G03-Probe", "PASS", "formal probe"),
        (market, "G03-Market", "ASSET_READY", "formal market"),
    )
    for admission, gate_id, expected_status, where in gate_bindings:
        if (
            admission.status != expected_status
            or admission.plan_digest
            != freeze_manifest.formal_gate_plan_digests[gate_id]
            or admission.freeze_manifest_digest
            != freeze_manifest.freeze_manifest_digest
        ):
            raise StageExecutionError(
                f"{where} admission is not a passing record for the reviewed freeze"
            )
    if market.candidate_count != 90 or market.market_entry_count != 30:
        raise StageExecutionError(
            "formal market admission requires the frozen 90-candidate/30-entry market"
        )

    if (
        readout["freeze_manifest_digest"]
        != freeze_manifest.freeze_manifest_digest
        or readout["readout_plan_digest"]
        != freeze_manifest.formal_signal_readout_plan_digest
        or readout["attribution_gate_evidence_digest"]
        != attribution.evidence_digest
    ):
        raise StageExecutionError(
            "formal signal readout differs from the freeze or admitted attribution evidence"
        )
    atlas_readout_artifacts = tuple(
        item
        for item in receipts["build-signal-atlas"].adapter_result.artifacts
        if item.semantic_id == "formal-signal-readout-bundle"
    )
    if len(atlas_readout_artifacts) != 1:
        raise StageExecutionError(
            "build-signal-atlas receipt lacks its unique formal readout artifact"
        )
    stage_readout_artifact = atlas_readout_artifacts[0]
    stage_readout_path = layout.assert_domain(
        layout.run_root / stage_readout_artifact.path,
        stage_readout_artifact.domain,
    )
    stage_readout = _parse_formal_signal_readout_bundle_manifest(
        _canonical_json_file(stage_readout_path, "build-signal-atlas readout bundle")
    )
    if stage_readout["bundle_digest"] != readout["bundle_digest"]:
        raise StageExecutionError(
            "completion signal readout differs from the build-signal-atlas stage receipt"
        )
    if (
        cost.execution_scope != "FORMAL"
        or cost.run_id != manifest.run_id
        or cost.freeze_manifest_digest != freeze_manifest.freeze_manifest_digest
        or cost.cost_protocol_digest != freeze_manifest.cost_protocol_digest
    ):
        raise StageExecutionError(
            "formal cost ledger crosses the run/freeze or differs from the frozen protocol"
        )

    signal_manifest = signal_publication.signal_outcome_manifest
    if (
        signal_publication.run_id != manifest.run_id
        or signal_publication.freeze_manifest_digest
        != freeze_manifest.freeze_manifest_digest
        or signal_publication.signal_extraction_plan_digest
        != freeze_manifest.preoracle_signal_outcome_plan_digest
        or signal_publication.formal_signal_readout_bundle_digest
        != readout["bundle_digest"]
        or signal_publication.public_query_plan_digest
        != freeze_manifest.public_query_plan_digest
        or signal_publication.signal_atlas_digest != readout["atlas_run_digest"]
        or signal_publication.signal_prefix_schedule_digest
        != freeze_manifest.formal_signal_prefix_schedule_digest
        or signal_publication.signal_outcome_manifest_digest
        != signal_manifest.manifest_digest
        or signal_manifest.opaque_query_ids
        != tuple(sorted(barrier.expected_opaque_query_ids))
        or signal_publication.query_alias_manifest_digest
        != barrier.query_alias_manifest_digest
        or barrier.preoracle_signal_outcome_manifest_digest
        != signal_publication.signal_outcome_manifest_digest
    ):
        raise StageExecutionError(
            "pre-oracle signal publication differs from the frozen readout/query/barrier chain"
        )
    if (
        barrier.run_id != manifest.run_id
        or barrier.freeze_manifest_digest != manifest.freeze_manifest_digest
        or handoff.run_id != manifest.run_id
        or handoff.freeze_manifest_digest != manifest.freeze_manifest_digest
        or statistics.run_id != manifest.run_id
        or statistics.preexperiment_freeze_manifest_digest
        != manifest.freeze_manifest_digest
        or recompute.run_id != manifest.run_id
        or recompute.freeze_manifest_digest != manifest.freeze_manifest_digest
    ):
        raise StageExecutionError(
            "completion external records cross the formal freeze or run boundary"
        )
    if (
        handoff.public_ranking_barrier_digest != barrier.barrier_digest
        or statistics.public_ranking_barrier_digest != barrier.barrier_digest
        or statistics.oracle_unlock_handoff_digest != handoff.handoff_digest
        or recompute.public_ranking_barrier_digest != barrier.barrier_digest
    ):
        raise StageExecutionError(
            "completion external records do not share the exact ranking barrier"
        )
    if (
        statistics.statistics_plan_digest != freeze_manifest.statistics_plan_digest
        or recompute.formal_statistics_result_digest != statistics.result_digest
        or recompute.raw_input_manifest_digest != statistics.statistics_input_digest
    ):
        raise StageExecutionError(
            "completion statistics/recompute records differ from the frozen result chain"
        )
    if (
        claim.run_id != manifest.run_id
        or claim.freeze_manifest_digest != freeze_manifest.freeze_manifest_digest
        or claim.attribution_admission_digest != attribution.admission_digest
        or claim.probe_admission_digest != probe.admission_digest
        or claim.market_admission_digest != market.admission_digest
        or claim.signal_readout_bundle_digest != readout["bundle_digest"]
        or claim.cost_ledger_digest != cost.ledger_digest
        or claim.preoracle_signal_outcome_digest
        != signal_publication.preoracle_signal_outcome_digest
        or claim.pre_oracle_signal_manifest_digest
        != signal_publication.signal_outcome_manifest_digest
        or claim.public_ranking_barrier_digest != barrier.barrier_digest
        or claim.statistics_result_digest != statistics.result_digest
        or claim.independent_recompute_attestation_digest
        != recompute.attestation_digest
        or claim.review_authority_receipt_digest
        != freeze_manifest.review_authority_receipt_digest
        or claim.completion_state
        in {"BLOCKED_ENGINEERING", "COMPLETE_NO_GO_PROBE"}
    ):
        raise StageExecutionError(
            "formal claim audit does not bind the exact completed evidence chain"
        )

    receipt = PipelineCompletionReceipt(
        completion_id=manifest.completion_id,
        run_id=manifest.run_id,
        freeze_manifest_digest=manifest.freeze_manifest_digest,
        completion_manifest_digest=manifest.manifest_digest,
        stage_receipt_digests={
            stage_id: item.receipt_digest for stage_id, item in receipts.items()
        },
        external_artifact_set_digest=sha256_json(external_bindings),
    )
    path = layout.artifact("joint_completion")
    if path.exists() and resume:
        existing = PipelineCompletionReceipt.from_dict(
            _canonical_json_file(path, "pipeline completion receipt")
        )
        if existing.receipt_digest != receipt.receipt_digest:
            raise StageExecutionError("completion resume content mismatch")
        return existing, path, sha256_file(path), True
    digest = layout.writer("joint_completion").publish_json(
        path, receipt.to_dict(), resume=False
    )
    return receipt, path, digest, False


def verify_pipeline_completion_from_files(
    *,
    completion_manifest_path: str | Path,
    freeze_manifest_path: str | Path,
    artifacts_root: str | Path,
    resume: bool = False,
) -> tuple[PipelineCompletionReceipt, Path, str, bool]:
    source = _regular_file(completion_manifest_path, "pipeline completion manifest")
    manifest = PipelineCompletionManifest.from_dict(
        _canonical_json_file(source, "pipeline completion manifest")
    )
    freeze = load_preexperiment_freeze(freeze_manifest_path)
    return verify_pipeline_completion(
        manifest,
        freeze,
        artifacts_root=artifacts_root,
        resume=resume,
    )


__all__ = [
    "DEVELOPMENT_MODE",
    "FORMAL_MODE",
    "FORMAL_STAGE_REQUEST_TEMPLATE_SCHEMA",
    "PipelineCompletionManifest",
    "PipelineCompletionReceipt",
    "PRODUCTION_STAGE_IDS",
    "REQUIRED_COMPLETION_EXTERNAL_ARTIFACT_IDS",
    "REQUIRED_PREDECESSOR",
    "STAGE_SEMANTIC_OUTPUT_SCHEMA",
    "STAGE_SEMANTIC_REQUIREMENTS",
    "FormalStageRequestTemplate",
    "StageAdapterResult",
    "StageDependencyBinding",
    "StageExecutionAdapter",
    "StageExecutionContext",
    "StageExecutionError",
    "StageExecutionManifest",
    "StageExecutionReceipt",
    "StageInputBinding",
    "StageOutputArtifact",
    "StageSemanticOutput",
    "execute_stage",
    "execute_stage_from_files",
    "load_preexperiment_freeze",
    "load_stage_execution_manifest",
    "formal_stage_output_payload_digest",
    "verify_pipeline_completion",
    "verify_pipeline_completion_from_files",
]
