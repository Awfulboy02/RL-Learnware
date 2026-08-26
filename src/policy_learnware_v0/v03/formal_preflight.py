"""Read-only launch preflight for an externally authorized v0.3 run.

The production stage driver cannot know the byte identities of future
predecessor receipts before those stages have run.  This module therefore
checks the immutable launch-time surface: the authorized freeze, one reviewed
request template for each of the eleven stages, the exact adapter bindings,
the fixed predecessor topology, and every pre-existing static input byte.

It deliberately has no adapter registry, artifact writer, or authority
constructor.  Runtime predecessor receipts and their output bytes remain
fail-closed in :mod:`policy_learnware_v0.v03.orchestration` when each stage is
actually executed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from .orchestration import (
    PRODUCTION_STAGE_IDS,
    REQUIRED_PREDECESSOR,
    FormalStageRequestTemplate,
    StageInputBinding,
)
from .preflight import (
    PreExperimentFreezeManifest,
    formal_stage_adapter_binding_digest,
)
from .schemas import checked_digest, checked_safe_id, strict_mapping


FORMAL_STAGE_LAUNCH_MANIFEST_SCHEMA = (
    "policy-learnware.v03-formal-stage-launch-manifest.v0"
)
FORMAL_PIPELINE_LAUNCH_MANIFEST_SCHEMA = (
    "policy-learnware.v03-formal-pipeline-launch-manifest.v0"
)
FORMAL_LAUNCH_PREFLIGHT_REPORT_SCHEMA = (
    "policy-learnware.v03-formal-launch-preflight-report.v0"
)
FORMAL_LAUNCH_REVIEW_AUTHORITY_SCHEMA = (
    "policy-learnware.v03-formal-launch-review-authority.v0"
)


class FormalPreflightError(ValueError):
    """The reviewed formal launch surface is missing, stale, or ambiguous."""


def _strict(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    try:
        return strict_mapping(value, expected, where)
    except ValueError as error:
        raise FormalPreflightError(str(error)) from error


def _digest(value: Any, where: str) -> str:
    try:
        return checked_digest(value, where)
    except ValueError as error:
        raise FormalPreflightError(str(error)) from error


def _safe_id(value: Any, where: str) -> str:
    try:
        return checked_safe_id(value, where)
    except ValueError as error:
        raise FormalPreflightError(str(error)) from error


def _absolute_path(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise FormalPreflightError(f"{where} must be a non-empty canonical path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise FormalPreflightError(f"{where} must be absolute")
    return str(path.resolve(strict=False))


def _entrypoint(value: Any, where: str) -> str:
    if not isinstance(value, str) or value.count(":") != 1:
        raise FormalPreflightError(f"{where} must use module:attribute form")
    module_name, attribute_name = value.split(":", 1)
    if (
        not module_name
        or any(not part.isidentifier() for part in module_name.split("."))
        or not attribute_name.isidentifier()
    ):
        raise FormalPreflightError(f"{where} must use module:attribute form")
    return value


def _regular_file(value: str | Path, where: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file() or path.is_symlink():
        raise FormalPreflightError(f"{where} must be a regular non-symlink file")
    return path.resolve()


def _regular_directory(value: str | Path, where: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise FormalPreflightError(f"{where} must be a regular non-symlink directory")
    return path.resolve()


def _entrypoint_module_source(
    module_root: Path,
    entrypoint: str,
    where: str,
) -> Path:
    """Resolve one reviewed ``module:attribute`` without following tree symlinks."""

    module_name, _ = entrypoint.split(":", 1)
    module_parts = module_name.split(".")
    parent = module_root
    for part in module_parts[:-1]:
        parent /= part
        if not parent.is_dir() or parent.is_symlink():
            raise FormalPreflightError(
                f"{where} package path must contain only regular directories"
            )
    source = parent / f"{module_parts[-1]}.py"
    resolved = _regular_file(source, where)
    try:
        resolved.relative_to(module_root)
    except ValueError as error:  # defensive after rejecting symlinked tree entries
        raise FormalPreflightError(f"{where} escapes adapter module root") from error
    return resolved


def _canonical_json_file(path: Path, where: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalPreflightError(f"cannot load {where}: {error}") from error
    if not isinstance(value, Mapping):
        raise FormalPreflightError(f"{where} must contain one JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise FormalPreflightError(f"{where} is not canonical JSON")
    return value


def _canonical_json_file_once(
    value: str | Path, where: str
) -> tuple[Path, Mapping[str, Any], str]:
    """Read one immutable snapshot and derive both byte and semantic views."""

    path = _regular_file(value, where)
    try:
        raw = path.read_bytes()
        decoded = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalPreflightError(f"cannot load {where}: {error}") from error
    if not isinstance(decoded, Mapping):
        raise FormalPreflightError(f"{where} must contain one JSON object")
    try:
        canonical = canonical_json_bytes(decoded) + b"\n"
    except (TypeError, ValueError) as error:
        raise FormalPreflightError(f"{where} is not canonical JSON: {error}") from error
    if raw != canonical:
        raise FormalPreflightError(f"{where} is not canonical JSON")
    return path, decoded, sha256_bytes(raw)


@dataclass(frozen=True)
class FormalStageLaunchManifest:
    """Launch-time inputs for one future formal stage.

    ``predecessor_stage_ids`` freezes topology only.  Receipt paths and output
    byte digests do not exist yet and are bound by ``StageExecutionManifest``
    immediately before the stage runs.
    """

    stage_id: str
    execution_id: str
    request_template_path: str
    request_template_file_sha256: str
    adapter_source_path: str
    adapter_source_file_sha256: str
    adapter_entrypoint: str
    static_inputs: tuple[StageInputBinding, ...]
    predecessor_stage_ids: tuple[str, ...]
    manifest_digest: str | None = None
    schema: str = FORMAL_STAGE_LAUNCH_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_STAGE_LAUNCH_MANIFEST_SCHEMA:
            raise FormalPreflightError("unsupported formal stage launch manifest")
        if self.stage_id not in PRODUCTION_STAGE_IDS:
            raise FormalPreflightError("formal stage launch has unknown stage")
        object.__setattr__(
            self, "execution_id", _safe_id(self.execution_id, "execution_id")
        )
        object.__setattr__(
            self,
            "request_template_path",
            _absolute_path(self.request_template_path, "request_template_path"),
        )
        object.__setattr__(
            self,
            "request_template_file_sha256",
            _digest(
                self.request_template_file_sha256,
                "request_template_file_sha256",
            ),
        )
        object.__setattr__(
            self,
            "adapter_source_path",
            _absolute_path(self.adapter_source_path, "adapter_source_path"),
        )
        object.__setattr__(
            self,
            "adapter_source_file_sha256",
            _digest(self.adapter_source_file_sha256, "adapter_source_file_sha256"),
        )
        object.__setattr__(
            self,
            "adapter_entrypoint",
            _entrypoint(self.adapter_entrypoint, "adapter_entrypoint"),
        )
        inputs = tuple(sorted(self.static_inputs, key=lambda item: item.input_id))
        if not all(isinstance(item, StageInputBinding) for item in inputs):
            raise FormalPreflightError(
                "formal stage launch requires typed static input bindings"
            )
        if len({item.input_id for item in inputs}) != len(inputs):
            raise FormalPreflightError("formal stage static input IDs must be unique")
        normalized_inputs = tuple(
            StageInputBinding(
                input_id=item.input_id,
                path=_absolute_path(
                    item.path, f"static input {item.input_id} path"
                ),
                sha256=item.sha256,
            )
            for item in inputs
        )
        object.__setattr__(self, "static_inputs", normalized_inputs)
        predecessors = tuple(self.predecessor_stage_ids)
        expected = tuple(REQUIRED_PREDECESSOR[self.stage_id])
        if predecessors != expected:
            raise FormalPreflightError(
                f"{self.stage_id} requires exact predecessor topology {expected!r}"
            )
        object.__setattr__(self, "predecessor_stage_ids", predecessors)
        expected_digest = sha256_json(self._payload_without_digest())
        if self.manifest_digest is None:
            object.__setattr__(self, "manifest_digest", expected_digest)
        elif _digest(self.manifest_digest, "manifest_digest") != expected_digest:
            raise FormalPreflightError("formal stage launch manifest digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "stage_id": self.stage_id,
            "execution_id": self.execution_id,
            "request_template_path": self.request_template_path,
            "request_template_file_sha256": self.request_template_file_sha256,
            "adapter_source_path": self.adapter_source_path,
            "adapter_source_file_sha256": self.adapter_source_file_sha256,
            "adapter_entrypoint": self.adapter_entrypoint,
            "static_inputs": [item.to_dict() for item in self.static_inputs],
            "predecessor_stage_ids": list(self.predecessor_stage_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalStageLaunchManifest":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "formal stage launch manifest")
        if data["manifest_digest"] is None:
            raise FormalPreflightError(
                "persisted formal stage launch manifest requires manifest_digest"
            )
        return cls(
            **{
                field: (
                    tuple(StageInputBinding.from_dict(item) for item in data[field])
                    if field == "static_inputs"
                    else tuple(data[field])
                    if field == "predecessor_stage_ids"
                    else data[field]
                )
                for field in fields
            }
        )


def formal_pipeline_launch_surface_digest(
    *,
    run_id: str,
    adapter_module_root: str | Path,
    stages: tuple[FormalStageLaunchManifest, ...],
) -> str:
    """Digest the acyclic launch surface that an external reviewer authorizes.

    The freeze digest and authority-receipt path are deliberately excluded so
    the receipt can bind this surface without a hash cycle.  The authorized
    freeze binds the receipt's semantic digest, completing the chain
    ``freeze -> authority receipt -> launch surface``.
    """

    normalized_run = _safe_id(run_id, "run_id")
    normalized_root = _absolute_path(
        str(Path(adapter_module_root).expanduser()), "adapter_module_root"
    )
    normalized_stages = tuple(stages)
    if (
        not all(isinstance(item, FormalStageLaunchManifest) for item in normalized_stages)
        or tuple(item.stage_id for item in normalized_stages) != PRODUCTION_STAGE_IDS
    ):
        raise FormalPreflightError(
            "formal launch surface requires all eleven typed stages in reviewed order"
        )
    return sha256_json(
        {
            "schema": "policy-learnware.v03-formal-launch-surface.v0",
            "run_id": normalized_run,
            "adapter_module_root": normalized_root,
            "stages": [item.to_dict() for item in normalized_stages],
        }
    )


def formal_freeze_authorization_surface_digest(
    freeze: PreExperimentFreezeManifest,
) -> str:
    """Digest the acyclic scientific/configuration surface under review.

    The authority receipt and the authorization projections are excluded so an
    external reviewer can sign this surface before its receipt is inserted into
    the final freeze.  Every substantive frozen protocol, implementation,
    registry, gate, query, statistics, cost, and hard-TODO binding remains in
    the surface.  The receipt can therefore bind this digest and the launch
    surface without introducing a hash cycle.
    """

    if not isinstance(freeze, PreExperimentFreezeManifest):
        raise FormalPreflightError(
            "freeze authorization surface requires a typed freeze manifest"
        )
    payload = freeze.to_dict()
    for name in (
        "review_authority_receipt_digest",
        "review_authority_verified",
        "engineering_ready",
        "formal_run_authorized",
        "freeze_manifest_digest",
    ):
        payload.pop(name)
    return sha256_json(
        {
            "schema": "policy-learnware.v03-formal-freeze-authorization-surface.v0",
            "freeze": payload,
        }
    )


@dataclass(frozen=True)
class FormalPipelineLaunchManifest:
    run_id: str
    freeze_manifest_digest: str
    review_authority_receipt_path: str
    review_authority_receipt_file_sha256: str
    adapter_module_root: str
    stages: tuple[FormalStageLaunchManifest, ...]
    manifest_digest: str | None = None
    schema: str = FORMAL_PIPELINE_LAUNCH_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_PIPELINE_LAUNCH_MANIFEST_SCHEMA:
            raise FormalPreflightError("unsupported formal pipeline launch manifest")
        object.__setattr__(self, "run_id", _safe_id(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "freeze_manifest_digest",
            _digest(self.freeze_manifest_digest, "freeze_manifest_digest"),
        )
        object.__setattr__(
            self,
            "review_authority_receipt_path",
            _absolute_path(
                self.review_authority_receipt_path,
                "review_authority_receipt_path",
            ),
        )
        object.__setattr__(
            self,
            "review_authority_receipt_file_sha256",
            _digest(
                self.review_authority_receipt_file_sha256,
                "review_authority_receipt_file_sha256",
            ),
        )
        object.__setattr__(
            self,
            "adapter_module_root",
            _absolute_path(self.adapter_module_root, "adapter_module_root"),
        )
        stages = tuple(self.stages)
        if not all(isinstance(item, FormalStageLaunchManifest) for item in stages):
            raise FormalPreflightError(
                "formal pipeline launch requires typed stage manifests"
            )
        if tuple(item.stage_id for item in stages) != PRODUCTION_STAGE_IDS:
            raise FormalPreflightError(
                "formal pipeline launch requires all eleven stages in reviewed order"
            )
        if len({item.execution_id for item in stages}) != len(stages):
            raise FormalPreflightError("formal stage execution IDs must be unique")
        if not stages[0].static_inputs:
            raise FormalPreflightError(
                "the first formal stage requires at least one static input"
            )
        object.__setattr__(self, "stages", stages)
        expected = sha256_json(self._payload_without_digest())
        if self.manifest_digest is None:
            object.__setattr__(self, "manifest_digest", expected)
        elif _digest(self.manifest_digest, "manifest_digest") != expected:
            raise FormalPreflightError("formal pipeline launch manifest digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "review_authority_receipt_path": self.review_authority_receipt_path,
            "review_authority_receipt_file_sha256": (
                self.review_authority_receipt_file_sha256
            ),
            "adapter_module_root": self.adapter_module_root,
            "stages": [item.to_dict() for item in self.stages],
            "launch_surface_digest": self.launch_surface_digest,
        }

    @property
    def launch_surface_digest(self) -> str:
        return formal_pipeline_launch_surface_digest(
            run_id=self.run_id,
            adapter_module_root=self.adapter_module_root,
            stages=self.stages,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalPipelineLaunchManifest":
        fields = set(cls.__dataclass_fields__) | {"launch_surface_digest"}
        data = _strict(value, fields, "formal pipeline launch manifest")
        if data["manifest_digest"] is None:
            raise FormalPreflightError(
                "persisted formal pipeline launch manifest requires manifest_digest"
            )
        manifest = cls(
            run_id=data["run_id"],
            freeze_manifest_digest=data["freeze_manifest_digest"],
            review_authority_receipt_path=data["review_authority_receipt_path"],
            review_authority_receipt_file_sha256=data[
                "review_authority_receipt_file_sha256"
            ],
            adapter_module_root=data["adapter_module_root"],
            stages=tuple(
                FormalStageLaunchManifest.from_dict(item) for item in data["stages"]
            ),
            manifest_digest=data["manifest_digest"],
            schema=data["schema"],
        )
        if _digest(
            data["launch_surface_digest"], "launch_surface_digest"
        ) != manifest.launch_surface_digest:
            raise FormalPreflightError(
                "formal pipeline launch surface digest is inconsistent"
            )
        return manifest


@dataclass(frozen=True)
class FormalLaunchPreflightReport:
    run_id: str
    freeze_manifest_digest: str
    launch_manifest_digest: str
    launch_surface_digest: str
    stage_request_template_registry_digest: str
    stage_adapter_registry_digest: str
    review_authority_receipt_file_sha256: str
    adapter_module_root: str
    request_template_file_digests: Mapping[str, str]
    adapter_source_file_digests: Mapping[str, str]
    adapter_entrypoints: Mapping[str, str]
    static_input_binding_digests: Mapping[str, str]
    predecessor_chain_digest: str
    checked_request_template_file_count: int
    checked_adapter_source_file_count: int
    checked_static_input_file_count: int
    stage_count: int = len(PRODUCTION_STAGE_IDS)
    adapter_executed: bool = False
    artifacts_written: bool = False
    status: str = "FORMAL_STATIC_BINDINGS_READY"
    report_digest: str | None = None
    schema: str = FORMAL_LAUNCH_PREFLIGHT_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_LAUNCH_PREFLIGHT_REPORT_SCHEMA:
            raise FormalPreflightError("unsupported formal launch preflight report")
        object.__setattr__(self, "run_id", _safe_id(self.run_id, "run_id"))
        for name in (
            "freeze_manifest_digest",
            "launch_manifest_digest",
            "launch_surface_digest",
            "stage_request_template_registry_digest",
            "stage_adapter_registry_digest",
            "review_authority_receipt_file_sha256",
            "predecessor_chain_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "adapter_module_root",
            _absolute_path(self.adapter_module_root, "adapter_module_root"),
        )
        for name in (
            "request_template_file_digests",
            "adapter_source_file_digests",
            "static_input_binding_digests",
        ):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping):
                raise FormalPreflightError(f"{name} must be a mapping")
            values = {
                _safe_id(key, f"{name} key"): _digest(value, f"{name}[{key!r}]")
                for key, value in raw.items()
            }
            object.__setattr__(
                self, name, MappingProxyType(dict(sorted(values.items())))
            )
        if not isinstance(self.adapter_entrypoints, Mapping):
            raise FormalPreflightError("adapter_entrypoints must be a mapping")
        entrypoints = {
            _safe_id(key, "adapter_entrypoints key"): _entrypoint(
                value, f"adapter_entrypoints[{key!r}]"
            )
            for key, value in self.adapter_entrypoints.items()
        }
        object.__setattr__(
            self,
            "adapter_entrypoints",
            MappingProxyType(dict(sorted(entrypoints.items()))),
        )
        for name in (
            "checked_request_template_file_count",
            "checked_adapter_source_file_count",
            "checked_static_input_file_count",
            "stage_count",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise FormalPreflightError(f"{name} has an invalid count")
        if (
            self.stage_count != len(PRODUCTION_STAGE_IDS)
            or self.checked_request_template_file_count != len(PRODUCTION_STAGE_IDS)
            or self.checked_adapter_source_file_count != len(PRODUCTION_STAGE_IDS)
            or set(self.request_template_file_digests) != set(PRODUCTION_STAGE_IDS)
            or set(self.adapter_source_file_digests) != set(PRODUCTION_STAGE_IDS)
            or set(self.adapter_entrypoints) != set(PRODUCTION_STAGE_IDS)
            or self.checked_static_input_file_count
            > len(self.static_input_binding_digests)
            or self.adapter_executed is not False
            or self.artifacts_written is not False
            or self.status != "FORMAL_STATIC_BINDINGS_READY"
        ):
            raise FormalPreflightError("formal preflight report status is invalid")
        expected = sha256_json(self._payload_without_digest())
        if self.report_digest is None:
            object.__setattr__(self, "report_digest", expected)
        elif _digest(self.report_digest, "report_digest") != expected:
            raise FormalPreflightError("formal preflight report digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "launch_manifest_digest": self.launch_manifest_digest,
            "launch_surface_digest": self.launch_surface_digest,
            "stage_request_template_registry_digest": (
                self.stage_request_template_registry_digest
            ),
            "stage_adapter_registry_digest": self.stage_adapter_registry_digest,
            "review_authority_receipt_file_sha256": (
                self.review_authority_receipt_file_sha256
            ),
            "adapter_module_root": self.adapter_module_root,
            "request_template_file_digests": dict(
                self.request_template_file_digests
            ),
            "adapter_source_file_digests": dict(
                self.adapter_source_file_digests
            ),
            "adapter_entrypoints": dict(self.adapter_entrypoints),
            "static_input_binding_digests": dict(
                self.static_input_binding_digests
            ),
            "predecessor_chain_digest": self.predecessor_chain_digest,
            "checked_request_template_file_count": (
                self.checked_request_template_file_count
            ),
            "checked_adapter_source_file_count": (
                self.checked_adapter_source_file_count
            ),
            "checked_static_input_file_count": self.checked_static_input_file_count,
            "stage_count": self.stage_count,
            "adapter_executed": self.adapter_executed,
            "artifacts_written": self.artifacts_written,
            "status": self.status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "report_digest": self.report_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalLaunchPreflightReport":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "formal launch preflight report")
        if data["report_digest"] is None:
            raise FormalPreflightError(
                "persisted formal launch preflight report requires report_digest"
            )
        return cls(**{field: data[field] for field in fields})


def load_formal_pipeline_launch_manifest(
    path: str | Path,
) -> FormalPipelineLaunchManifest:
    source = _regular_file(path, "formal pipeline launch manifest")
    return FormalPipelineLaunchManifest.from_dict(
        _canonical_json_file(source, "formal pipeline launch manifest")
    )


def verify_formal_launch_preflight(
    manifest: FormalPipelineLaunchManifest,
    freeze: PreExperimentFreezeManifest,
) -> FormalLaunchPreflightReport:
    """Verify all immutable launch inputs without importing or calling adapters."""

    if not isinstance(manifest, FormalPipelineLaunchManifest):
        raise FormalPreflightError("formal preflight requires a typed launch manifest")
    if not isinstance(freeze, PreExperimentFreezeManifest):
        raise FormalPreflightError("formal preflight requires a typed freeze manifest")
    if not freeze.formal_run_authorized or not freeze.review_authority_verified:
        raise FormalPreflightError(
            "formal preflight requires externally reviewed launch authority"
        )
    if manifest.freeze_manifest_digest != freeze.freeze_manifest_digest:
        raise FormalPreflightError("formal launch manifest belongs to another freeze")

    adapter_module_root = _regular_directory(
        manifest.adapter_module_root,
        "adapter module root",
    )

    _, authority_record, authority_file_sha = _canonical_json_file_once(
        manifest.review_authority_receipt_path,
        "external review authority receipt",
    )
    if authority_file_sha != manifest.review_authority_receipt_file_sha256:
        raise FormalPreflightError("external review authority receipt byte drift")
    if freeze.review_authority_receipt_digest is None or sha256_json(
        authority_record
    ) != freeze.review_authority_receipt_digest:
        raise FormalPreflightError(
            "external review authority receipt differs from the authorized freeze"
        )
    authority = _strict(
        authority_record,
        {
            "schema",
            "decision",
            "authority_id",
            "formal_freeze_authorization_surface_digest",
            "formal_launch_surface_digest",
        },
        "external review authority receipt",
    )
    if authority["schema"] != FORMAL_LAUNCH_REVIEW_AUTHORITY_SCHEMA:
        raise FormalPreflightError("unsupported formal launch review authority schema")
    if authority["decision"] != "AUTHORIZED":
        raise FormalPreflightError("formal launch review authority is not authorized")
    _safe_id(authority["authority_id"], "formal launch authority_id")
    if _digest(
        authority["formal_freeze_authorization_surface_digest"],
        "formal freeze authority surface digest",
    ) != formal_freeze_authorization_surface_digest(freeze):
        raise FormalPreflightError(
            "external review authority receipt binds another freeze authorization surface"
        )
    if _digest(
        authority["formal_launch_surface_digest"],
        "formal launch authority surface digest",
    ) != manifest.launch_surface_digest:
        raise FormalPreflightError(
            "external review authority receipt binds another launch surface"
        )

    request_files: dict[str, str] = {}
    adapter_sources: dict[str, str] = {}
    adapter_entrypoints: dict[str, str] = {}
    static_bindings: dict[str, str] = {}
    physical_static_paths: dict[Path, tuple[str, str]] = {}
    for stage in manifest.stages:
        _, template_value, template_file_sha = _canonical_json_file_once(
            stage.request_template_path,
            f"request template for {stage.stage_id}",
        )
        if template_file_sha != stage.request_template_file_sha256:
            raise FormalPreflightError(
                f"request template byte digest mismatch: {stage.stage_id}"
            )
        template = FormalStageRequestTemplate.from_dict(
            template_value
        )
        if template.stage_id != stage.stage_id:
            raise FormalPreflightError("request template stage identity mismatch")
        try:
            expected_template = freeze.formal_stage_request_template_digests[
                stage.stage_id
            ]
            expected_adapter = freeze.formal_stage_adapter_binding_digests[
                stage.stage_id
            ]
        except KeyError as error:  # defensive: authorized freeze constructor is exact
            raise FormalPreflightError(
                f"authorized freeze lacks launch binding for {stage.stage_id}"
            ) from error
        if template.request_template_digest != expected_template:
            raise FormalPreflightError(
                f"request template differs from authorized freeze: {stage.stage_id}"
            )
        observed_adapter = formal_stage_adapter_binding_digest(
            stage.stage_id,
            template.adapter_id,
            template.adapter_contract_digest,
        )
        if observed_adapter != expected_adapter:
            raise FormalPreflightError(
                f"adapter identity/contract differs from authorized freeze: {stage.stage_id}"
            )
        adapter_source = _regular_file(
            stage.adapter_source_path,
            f"adapter source for {stage.stage_id}",
        )
        resolved_module_source = _entrypoint_module_source(
            adapter_module_root,
            stage.adapter_entrypoint,
            f"entrypoint module source for {stage.stage_id}",
        )
        if resolved_module_source != adapter_source:
            raise FormalPreflightError(
                "adapter entrypoint module does not resolve to reviewed source: "
                f"{stage.stage_id}"
            )
        adapter_source_sha = sha256_file(adapter_source)
        if adapter_source_sha != stage.adapter_source_file_sha256:
            raise FormalPreflightError(
                f"adapter source byte digest mismatch: {stage.stage_id}"
            )

        observed_static: dict[str, str] = {}
        stage_paths: set[Path] = set()
        for item in stage.static_inputs:
            path = _regular_file(
                item.path,
                f"static input {stage.stage_id}:{item.input_id}",
            )
            if path in stage_paths:
                raise FormalPreflightError(
                    f"{stage.stage_id} binds one physical static input more than once"
                )
            stage_paths.add(path)
            actual_sha = sha256_file(path)
            if actual_sha != item.sha256:
                raise FormalPreflightError(
                    f"static input byte digest mismatch: {stage.stage_id}:{item.input_id}"
                )
            previous = physical_static_paths.get(path)
            if previous is not None and previous[1] != actual_sha:
                raise FormalPreflightError(
                    "one physical static input has inconsistent cross-stage digests"
                )
            physical_static_paths[path] = (f"{stage.stage_id}:{item.input_id}", actual_sha)
            observed_static[item.input_id] = actual_sha
            static_bindings[f"{stage.stage_id}:{item.input_id}"] = item.binding_digest
        if observed_static != dict(template.static_input_content_digests):
            raise FormalPreflightError(
                f"static input set differs from request template: {stage.stage_id}"
            )
        request_files[stage.stage_id] = template_file_sha
        adapter_sources[stage.stage_id] = adapter_source_sha
        adapter_entrypoints[stage.stage_id] = stage.adapter_entrypoint

    predecessor_chain_digest = sha256_json(
        {
            stage.stage_id: list(stage.predecessor_stage_ids)
            for stage in manifest.stages
        }
    )
    return FormalLaunchPreflightReport(
        run_id=manifest.run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        launch_manifest_digest=str(manifest.manifest_digest),
        launch_surface_digest=manifest.launch_surface_digest,
        stage_request_template_registry_digest=(
            freeze.formal_stage_request_template_registry_digest
        ),
        stage_adapter_registry_digest=freeze.formal_stage_adapter_registry_digest,
        review_authority_receipt_file_sha256=authority_file_sha,
        adapter_module_root=str(adapter_module_root),
        request_template_file_digests=request_files,
        adapter_source_file_digests=adapter_sources,
        adapter_entrypoints=adapter_entrypoints,
        static_input_binding_digests=static_bindings,
        predecessor_chain_digest=predecessor_chain_digest,
        checked_request_template_file_count=len(request_files),
        checked_adapter_source_file_count=len(adapter_sources),
        checked_static_input_file_count=len(physical_static_paths),
    )


def verify_formal_launch_preflight_from_files(
    *,
    launch_manifest_path: str | Path,
    freeze_manifest_path: str | Path,
) -> FormalLaunchPreflightReport:
    """Load canonical files and perform the zero-write formal launch preflight."""

    from .orchestration import load_preexperiment_freeze

    launch = load_formal_pipeline_launch_manifest(launch_manifest_path)
    freeze = load_preexperiment_freeze(freeze_manifest_path)
    return verify_formal_launch_preflight(launch, freeze)


__all__ = [
    "FORMAL_LAUNCH_PREFLIGHT_REPORT_SCHEMA",
    "FORMAL_LAUNCH_REVIEW_AUTHORITY_SCHEMA",
    "FORMAL_PIPELINE_LAUNCH_MANIFEST_SCHEMA",
    "FORMAL_STAGE_LAUNCH_MANIFEST_SCHEMA",
    "FormalLaunchPreflightReport",
    "FormalPipelineLaunchManifest",
    "FormalPreflightError",
    "FormalStageLaunchManifest",
    "formal_freeze_authorization_surface_digest",
    "formal_pipeline_launch_surface_digest",
    "load_formal_pipeline_launch_manifest",
    "verify_formal_launch_preflight",
    "verify_formal_launch_preflight_from_files",
]
