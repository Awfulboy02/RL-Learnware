"""Production boundary for v0.3 source-only policy evaluation.

The source market contracts deliberately do not know how a frozen policy is
loaded or how a source environment is instantiated.  This module supplies the
missing execution boundary:

* the frozen v0.2 server plan is the only authority that joins an exact-90
  intake cell to its anchor manifest;
* a backend must validate immutable bundle bytes and return a typed execution
  ABI before a work unit can be published;
* the runner invokes the backend once for every literal reset seed and derives
  normalized returns from raw episode returns through a frozen return
  contract; and
* a receipt can only be produced by rebuilding a raw shard from a successful,
  digest-bound attempt record.

Training summaries, target evidence, and caller-supplied aggregate returns are
not inputs to this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..hashing import sha256_json
from ..v02.schemas import ExecutionABIRecord
from .pool_intake import (
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_JOB_COUNT,
    EXPECTED_SEEDS,
    PoolIntakeCell,
    PoolIntakeError,
    V03PoolIntakeRecord,
    assert_frozen_v02_intake_authority,
)
from .source_market import (
    EvaluatorSourceReceipt,
    EvaluationBlock,
    RawSourceEpisodeShard,
    SOURCE_ROLLOUT_DATASET_SCHEMA,
    SourceEvaluationProtocol,
    SourceEvaluationWorkUnit,
    build_source_evaluation_work_unit,
    receipt_from_source_episode_shard,
)


class SourceEvaluatorError(ValueError):
    """The plan, backend, return projection, or execution record is invalid."""


class SourceEvaluationAttemptFailed(SourceEvaluatorError):
    """A source episode failed; the attached record is evidence, not a receipt."""

    def __init__(self, attempt_record: "SourceEvaluationAttemptRecord") -> None:
        self.attempt_record = attempt_record
        super().__init__(
            f"source evaluation failed for {attempt_record.candidate_id}: "
            f"{attempt_record.failure_code}: {attempt_record.failure_message}"
        )


CANONICAL_SOURCE_ANCHOR_SCHEMA = "policy-learnware.v03-canonical-source-anchor.v0"
FROZEN_PLAN_JOB_BINDING_SCHEMA = "policy-learnware.v03-frozen-plan-job-binding.v0"
FROZEN_SERVER_PLAN_BINDING_SCHEMA = "policy-learnware.v03-frozen-server-plan-binding.v0"
SOURCE_CANDIDATE_REQUEST_SCHEMA = "policy-learnware.v03-source-candidate-request.v0"
VALIDATED_SOURCE_BINDING_SCHEMA = "policy-learnware.v03-validated-source-binding.v0"
BACKEND_EPISODE_RESULT_SCHEMA = "policy-learnware.v03-backend-episode-result.v0"
DMC_RETURN_CONTRACT_SCHEMA = "policy-learnware.v03-dmc-fixed-horizon-return-contract.v0"
SOURCE_EPISODE_ATTEMPT_SCHEMA = "policy-learnware.v03-source-episode-attempt.v0"
SOURCE_EVALUATION_ATTEMPT_SCHEMA = "policy-learnware.v03-source-evaluation-attempt.v0"
SOURCE_EVALUATION_RUN_SCHEMA = "policy-learnware.v03-source-evaluation-run.v0"
SOURCE_WORK_UNIT_MANIFEST_SCHEMA = "policy-learnware.v03-source-work-unit-manifest.v0"

AttemptState = Literal["SUCCEEDED", "FAILED"]
EpisodeState = Literal["SUCCEEDED", "FAILED"]

_JOB_ID = re.compile(r"^v02j-[0-9a-f]{24}$")
_FORBIDDEN_EVIDENCE_FIELDS = frozenset(
    {
        "training_summary",
        "training_return",
        "target_evidence",
        "target_return",
        "oracle_return",
        "oracle_evidence",
        "normalized_returns",
        "mean_return",
    }
)


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise SourceEvaluatorError(f"{where} must be a mapping")
    observed = set(value)
    if observed != expected:
        raise SourceEvaluatorError(
            f"{where} fields differ: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SourceEvaluatorError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result.lower() != result:
        raise SourceEvaluatorError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise SourceEvaluatorError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SourceEvaluatorError(f"{where} must be a positive integer")
    return value


def _seed(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceEvaluatorError(f"{where} must be a non-negative integer")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise SourceEvaluatorError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SourceEvaluatorError(f"{where} must be finite")
    return result


def _absolute_existing(path: str | Path, *, where: str, kind: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise SourceEvaluatorError(f"{where} must be absolute")
    if supplied.is_symlink():
        raise SourceEvaluatorError(f"{where} may not be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise SourceEvaluatorError(f"{where} does not exist") from error
    if kind == "file" and not resolved.is_file():
        raise SourceEvaluatorError(f"{where} must be a file")
    if kind == "dir" and not resolved.is_dir():
        raise SourceEvaluatorError(f"{where} must be a directory")
    return resolved


def _strict_json(path: Path, where: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise SourceEvaluatorError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SourceEvaluatorError(f"{where} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceEvaluatorError(f"cannot read strict JSON {where}: {error}") from error
    if not isinstance(value, dict):
        raise SourceEvaluatorError(f"{where} must be a JSON object")
    return value


def _reject_forbidden_evidence(value: Any, *, where: str) -> None:
    if isinstance(value, Mapping):
        overlap = _FORBIDDEN_EVIDENCE_FIELDS & set(value)
        if overlap:
            raise SourceEvaluatorError(
                f"{where} contains forbidden summary/target/oracle evidence: {sorted(overlap)}"
            )
        for key, child in value.items():
            _reject_forbidden_evidence(child, where=f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_evidence(child, where=f"{where}[{index}]")


@dataclass(frozen=True)
class CanonicalSourceAnchor:
    """Canonical bytes selected by one frozen server-plan job."""

    manifest_path: str
    manifest_digest: str
    source_anchor_id: str
    environment_instance_digest: str
    axis_binding_digest: str | None
    runtime_digest: str
    manifest_content: Mapping[str, Any]
    schema: str = CANONICAL_SOURCE_ANCHOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_SOURCE_ANCHOR_SCHEMA:
            raise SourceEvaluatorError("unsupported CanonicalSourceAnchor schema")
        path = _absolute_existing(self.manifest_path, where="anchor manifest path", kind="file")
        object.__setattr__(self, "manifest_path", str(path))
        for name in (
            "manifest_digest",
            "source_anchor_id",
            "environment_instance_digest",
            "runtime_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.axis_binding_digest is not None:
            object.__setattr__(
                self,
                "axis_binding_digest",
                _digest(self.axis_binding_digest, "axis_binding_digest"),
            )
        content = dict(self.manifest_content)
        _reject_forbidden_evidence(content, where="anchor manifest")
        required = {
            "manifest_digest",
            "anchor_id",
            "environment_instance_digest",
            "axis_binding_digest",
            "runtime",
            "runtime_digest",
        }
        if not required.issubset(content):
            raise SourceEvaluatorError(
                f"anchor manifest lacks canonical fields: {sorted(required-set(content))}"
            )
        observed_digest = sha256_json(
            {key: item for key, item in content.items() if key != "manifest_digest"}
        )
        if observed_digest != self.manifest_digest or content["manifest_digest"] != self.manifest_digest:
            raise SourceEvaluatorError("anchor manifest content/self digest drifted")
        if (
            content["anchor_id"] != self.source_anchor_id
            or content["environment_instance_digest"] != self.environment_instance_digest
            or content["axis_binding_digest"] != self.axis_binding_digest
            or content["runtime_digest"] != self.runtime_digest
            or not isinstance(content["runtime"], Mapping)
            or sha256_json(content["runtime"]) != self.runtime_digest
        ):
            raise SourceEvaluatorError("anchor manifest identity/runtime projection drifted")
        if _strict_json(path, "anchor manifest bytes") != content:
            raise SourceEvaluatorError("anchor manifest content differs from immutable path bytes")
        object.__setattr__(self, "manifest_content", MappingProxyType(content))

    @property
    def binding_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "manifest_path": self.manifest_path,
            "manifest_digest": self.manifest_digest,
            "source_anchor_id": self.source_anchor_id,
            "environment_instance_digest": self.environment_instance_digest,
            "axis_binding_digest": self.axis_binding_digest,
            "runtime_digest": self.runtime_digest,
            "manifest_content": dict(self.manifest_content),
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "CanonicalSourceAnchor":
        resolved = _absolute_existing(path, where="anchor manifest path", kind="file")
        content = _strict_json(resolved, "anchor manifest")
        try:
            return cls(
                manifest_path=str(resolved),
                manifest_digest=content["manifest_digest"],
                source_anchor_id=content["anchor_id"],
                environment_instance_digest=content["environment_instance_digest"],
                axis_binding_digest=content["axis_binding_digest"],
                runtime_digest=content["runtime_digest"],
                manifest_content=content,
            )
        except KeyError as error:
            raise SourceEvaluatorError(
                f"anchor manifest lacks canonical field {error.args[0]!r}"
            ) from error


@dataclass(frozen=True)
class FrozenPlanJobBinding:
    candidate_id: str
    job_digest: str
    seed: int
    training_protocol_digest: str
    anchor: CanonicalSourceAnchor
    schema: str = FROZEN_PLAN_JOB_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FROZEN_PLAN_JOB_BINDING_SCHEMA:
            raise SourceEvaluatorError("unsupported FrozenPlanJobBinding schema")
        if not _JOB_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceEvaluatorError("plan job candidate_id is not a frozen v0.2 job ID")
        object.__setattr__(self, "job_digest", _digest(self.job_digest, "job_digest"))
        object.__setattr__(
            self,
            "training_protocol_digest",
            _digest(self.training_protocol_digest, "training_protocol_digest"),
        )
        if self.seed not in EXPECTED_SEEDS:
            raise SourceEvaluatorError("plan job seed must be one of 0/1/2")
        if not isinstance(self.anchor, CanonicalSourceAnchor):
            raise SourceEvaluatorError("plan job requires a canonical source anchor")

    @property
    def binding_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "job_digest": self.job_digest,
            "seed": self.seed,
            "training_protocol_digest": self.training_protocol_digest,
            "anchor": self.anchor.to_dict(),
        }


@dataclass(frozen=True)
class FrozenServerPlanBinding:
    """Typed output of the frozen v0.2 plan/anchor authority."""

    plan_path: str
    plan_digest: str
    jobs: Mapping[str, FrozenPlanJobBinding]
    schema: str = FROZEN_SERVER_PLAN_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FROZEN_SERVER_PLAN_BINDING_SCHEMA:
            raise SourceEvaluatorError("unsupported FrozenServerPlanBinding schema")
        path = _absolute_existing(self.plan_path, where="frozen server plan path", kind="file")
        object.__setattr__(self, "plan_path", str(path))
        object.__setattr__(self, "plan_digest", _digest(self.plan_digest, "plan_digest"))
        jobs = dict(self.jobs)
        if (
            len(jobs) != EXPECTED_JOB_COUNT
            or any(not isinstance(job, FrozenPlanJobBinding) for job in jobs.values())
            or set(jobs) != {job.candidate_id for job in jobs.values()}
        ):
            raise SourceEvaluatorError("frozen server-plan binding must contain exactly 90 keyed jobs")
        anchors = {job.anchor.source_anchor_id for job in jobs.values()}
        units = {(job.anchor.source_anchor_id, job.seed) for job in jobs.values()}
        expected = {(anchor, seed) for anchor in anchors for seed in EXPECTED_SEEDS}
        if len(anchors) != EXPECTED_ANCHOR_COUNT or units != expected:
            raise SourceEvaluatorError("frozen server plan is not an exact 30-anchor x 3-seed grid")
        raw_plan = _strict_json(path, "frozen server plan")
        observed_plan_digest = raw_plan.get("plan_digest")
        if (
            observed_plan_digest != self.plan_digest
            or sha256_json(
                {key: value for key, value in raw_plan.items() if key != "plan_digest"}
            )
            != self.plan_digest
        ):
            raise SourceEvaluatorError("frozen server-plan path/self digest drifted")
        raw_jobs = raw_plan.get("jobs")
        if not isinstance(raw_jobs, list) or len(raw_jobs) != EXPECTED_JOB_COUNT:
            raise SourceEvaluatorError("frozen server-plan bytes do not contain exactly 90 jobs")
        projected: dict[str, tuple[Any, ...]] = {}
        required = {
            "job_id",
            "job_digest",
            "seed",
            "training_protocol_digest",
            "anchor_manifest_path",
            "anchor_manifest_digest",
        }
        for raw in raw_jobs:
            if not isinstance(raw, Mapping) or not required.issubset(raw):
                raise SourceEvaluatorError("frozen server-plan job lacks binding fields")
            job_id = raw["job_id"]
            if not isinstance(job_id, str) or job_id in projected:
                raise SourceEvaluatorError("frozen server-plan bytes contain duplicate jobs")
            supplied_anchor = Path(raw["anchor_manifest_path"]).expanduser()
            if not supplied_anchor.is_absolute() or supplied_anchor.is_symlink():
                raise SourceEvaluatorError("frozen server-plan anchor path is non-canonical")
            try:
                canonical_anchor_path = str(supplied_anchor.resolve(strict=True))
            except OSError as error:
                raise SourceEvaluatorError("frozen server-plan anchor path is missing") from error
            projected[job_id] = (
                raw["job_digest"],
                raw["seed"],
                raw["training_protocol_digest"],
                canonical_anchor_path,
                raw["anchor_manifest_digest"],
            )
        expected_projection = {
            job_id: (
                job.job_digest,
                job.seed,
                job.training_protocol_digest,
                job.anchor.manifest_path,
                job.anchor.manifest_digest,
            )
            for job_id, job in jobs.items()
        }
        if projected != expected_projection:
            raise SourceEvaluatorError("typed server-plan jobs differ from immutable plan bytes")
        object.__setattr__(self, "jobs", MappingProxyType(dict(sorted(jobs.items()))))

    @property
    def binding_digest(self) -> str:
        return sha256_json(
            {
                "schema": self.schema,
                "plan_path": self.plan_path,
                "plan_digest": self.plan_digest,
                "jobs": {key: job.to_dict() for key, job in self.jobs.items()},
            }
        )


class FrozenV02ServerPlanAuthority:
    """Load a real server plan through the frozen v0.2 validators."""

    def load(self, plan_path: str | Path) -> FrozenServerPlanBinding:
        path = _absolute_existing(plan_path, where="frozen server plan path", kind="file")
        try:
            from server.repro_fpo_ppo_v02.anchor_binding import AnchorManifest
            from server.repro_fpo_ppo_v02.provenance import (
                load_strict_json,
                validate_training_plan,
            )
        except ImportError as error:  # pragma: no cover - deployment/package gate
            raise SourceEvaluatorError(
                "frozen v0.2 server-plan validators are unavailable"
            ) from error
        try:
            validated = validate_training_plan(load_strict_json(path))
            jobs: dict[str, FrozenPlanJobBinding] = {}
            for row in validated["jobs"]:
                manifest_path = _absolute_existing(
                    row["anchor_manifest_path"],
                    where=f"plan job {row['job_id']} anchor manifest path",
                    kind="file",
                )
                manifest = AnchorManifest.from_path(manifest_path)
                if manifest.manifest_digest != row["anchor_manifest_digest"]:
                    raise SourceEvaluatorError(
                        f"plan job {row['job_id']} anchor manifest bytes drifted"
                    )
                canonical = CanonicalSourceAnchor(
                    manifest_path=str(manifest_path),
                    manifest_digest=manifest.manifest_digest,
                    source_anchor_id=manifest.anchor_id,
                    environment_instance_digest=manifest.environment_instance_digest,
                    axis_binding_digest=manifest.axis_binding_digest,
                    runtime_digest=manifest.runtime_digest,
                    manifest_content=manifest.to_dict(),
                )
                jobs[row["job_id"]] = FrozenPlanJobBinding(
                    candidate_id=row["job_id"],
                    job_digest=row["job_digest"],
                    seed=row["seed"],
                    training_protocol_digest=row["training_protocol_digest"],
                    anchor=canonical,
                )
        except SourceEvaluatorError:
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise SourceEvaluatorError(f"frozen v0.2 server plan is invalid: {error}") from error
        return FrozenServerPlanBinding(
            plan_path=str(path),
            plan_digest=validated["plan_digest"],
            jobs=jobs,
        )


@dataclass(frozen=True)
class SourceCandidateRequest:
    """All immutable private inputs a backend may use to validate a candidate."""

    evaluator_implementation_digest: str
    intake_cell_digest: str
    candidate_id: str
    source_anchor_id: str
    attempt_number: int
    attempt_digest: str
    bundle_path: str
    bundle_digest: str
    outer_iteration: int
    environment_steps: int
    source_environment_digest: str
    anchor: CanonicalSourceAnchor
    request_digest: str | None = None
    schema: str = SOURCE_CANDIDATE_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_CANDIDATE_REQUEST_SCHEMA:
            raise SourceEvaluatorError("unsupported SourceCandidateRequest schema")
        if not _JOB_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceEvaluatorError("candidate request has an invalid candidate_id")
        for name in (
            "evaluator_implementation_digest",
            "intake_cell_digest",
            "source_anchor_id",
            "attempt_digest",
            "bundle_digest",
            "source_environment_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "attempt_number", _positive_int(self.attempt_number, "attempt_number"))
        object.__setattr__(self, "outer_iteration", _positive_int(self.outer_iteration, "outer_iteration"))
        object.__setattr__(self, "environment_steps", _positive_int(self.environment_steps, "environment_steps"))
        bundle = _absolute_existing(self.bundle_path, where="candidate bundle path", kind="dir")
        object.__setattr__(self, "bundle_path", str(bundle))
        if not isinstance(self.anchor, CanonicalSourceAnchor):
            raise SourceEvaluatorError("candidate request requires a canonical source anchor")
        if (
            self.anchor.source_anchor_id != self.source_anchor_id
            or self.anchor.environment_instance_digest != self.source_environment_digest
        ):
            raise SourceEvaluatorError("candidate request anchor/environment binding drifted")
        expected = sha256_json(self._payload_without_digest())
        if self.request_digest is None:
            object.__setattr__(self, "request_digest", expected)
        elif _digest(self.request_digest, "request_digest") != expected:
            raise SourceEvaluatorError("candidate request digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evaluator_implementation_digest": self.evaluator_implementation_digest,
            "intake_cell_digest": self.intake_cell_digest,
            "candidate_id": self.candidate_id,
            "source_anchor_id": self.source_anchor_id,
            "attempt_number": self.attempt_number,
            "attempt_digest": self.attempt_digest,
            "bundle_path": self.bundle_path,
            "bundle_digest": self.bundle_digest,
            "outer_iteration": self.outer_iteration,
            "environment_steps": self.environment_steps,
            "source_environment_digest": self.source_environment_digest,
            "anchor": self.anchor.to_dict(),
        }


@dataclass(frozen=True)
class ValidatedSourceBinding:
    """Backend-owned proof that bundle bytes have one executable ABI."""

    request_digest: str
    candidate_id: str
    evaluator_implementation_digest: str
    bundle_path: str
    bundle_digest: str
    anchor_manifest_path: str
    anchor_manifest_digest: str
    anchor_runtime_digest: str
    source_environment_digest: str
    execution_abi: ExecutionABIRecord
    binding_digest: str | None = None
    schema: str = VALIDATED_SOURCE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != VALIDATED_SOURCE_BINDING_SCHEMA:
            raise SourceEvaluatorError("unsupported ValidatedSourceBinding schema")
        if not _JOB_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceEvaluatorError("validated binding has an invalid candidate_id")
        for name in (
            "request_digest",
            "evaluator_implementation_digest",
            "bundle_digest",
            "anchor_manifest_digest",
            "anchor_runtime_digest",
            "source_environment_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "bundle_path",
            str(_absolute_existing(self.bundle_path, where="validated bundle path", kind="dir")),
        )
        object.__setattr__(
            self,
            "anchor_manifest_path",
            str(
                _absolute_existing(
                    self.anchor_manifest_path,
                    where="validated anchor manifest path",
                    kind="file",
                )
            ),
        )
        if not isinstance(self.execution_abi, ExecutionABIRecord):
            raise SourceEvaluatorError("validated binding requires a typed execution ABI")
        expected = sha256_json(self._payload_without_digest())
        if self.binding_digest is None:
            object.__setattr__(self, "binding_digest", expected)
        elif _digest(self.binding_digest, "binding_digest") != expected:
            raise SourceEvaluatorError("validated source binding digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_digest": self.request_digest,
            "candidate_id": self.candidate_id,
            "evaluator_implementation_digest": self.evaluator_implementation_digest,
            "bundle_path": self.bundle_path,
            "bundle_digest": self.bundle_digest,
            "anchor_manifest_path": self.anchor_manifest_path,
            "anchor_manifest_digest": self.anchor_manifest_digest,
            "anchor_runtime_digest": self.anchor_runtime_digest,
            "source_environment_digest": self.source_environment_digest,
            "execution_abi": self.execution_abi.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "binding_digest": self.binding_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidatedSourceBinding":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "ValidatedSourceBinding")
        try:
            abi = ExecutionABIRecord.from_dict(value["execution_abi"])
        except (TypeError, ValueError) as error:
            raise SourceEvaluatorError("validated binding contains an invalid execution ABI") from error
        return cls(**{name: abi if name == "execution_abi" else value[name] for name in fields})


@dataclass(frozen=True)
class BackendEpisodeResult:
    """One raw backend outcome.  It deliberately has no normalized return."""

    reset_seed: int
    runtime_digest: str
    state: EpisodeState
    raw_return: float | None
    steps: int | None
    terminated: bool | None
    truncated: bool | None
    failure_code: str | None = None
    failure_message: str | None = None
    schema: str = BACKEND_EPISODE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BACKEND_EPISODE_RESULT_SCHEMA:
            raise SourceEvaluatorError("unsupported BackendEpisodeResult schema")
        object.__setattr__(self, "reset_seed", _seed(self.reset_seed, "reset_seed"))
        object.__setattr__(self, "runtime_digest", _digest(self.runtime_digest, "runtime_digest"))
        if self.state == "SUCCEEDED":
            object.__setattr__(self, "raw_return", _finite(self.raw_return, "raw_return"))
            object.__setattr__(self, "steps", _positive_int(self.steps, "steps"))
            if type(self.terminated) is not bool or type(self.truncated) is not bool:
                raise SourceEvaluatorError("successful backend outcome requires boolean end flags")
            if self.failure_code is not None or self.failure_message is not None:
                raise SourceEvaluatorError("successful backend outcome cannot carry a failure")
        elif self.state == "FAILED":
            if any(
                value is not None
                for value in (self.raw_return, self.steps, self.terminated, self.truncated)
            ):
                raise SourceEvaluatorError("failed backend outcome cannot fabricate episode values")
            object.__setattr__(self, "failure_code", _nonempty(self.failure_code, "failure_code"))
            object.__setattr__(
                self,
                "failure_message",
                _nonempty(self.failure_message, "failure_message"),
            )
        else:
            raise SourceEvaluatorError("backend episode state must be SUCCEEDED or FAILED")

    @classmethod
    def succeeded(
        cls,
        *,
        reset_seed: int,
        runtime_digest: str,
        raw_return: float,
        steps: int,
        terminated: bool,
        truncated: bool,
    ) -> "BackendEpisodeResult":
        return cls(
            reset_seed=reset_seed,
            runtime_digest=runtime_digest,
            state="SUCCEEDED",
            raw_return=raw_return,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
        )

    @classmethod
    def failed(
        cls,
        *,
        reset_seed: int,
        runtime_digest: str,
        failure_code: str,
        failure_message: str,
    ) -> "BackendEpisodeResult":
        return cls(
            reset_seed=reset_seed,
            runtime_digest=runtime_digest,
            state="FAILED",
            raw_return=None,
            steps=None,
            terminated=None,
            truncated=None,
            failure_code=failure_code,
            failure_message=failure_message,
        )


@runtime_checkable
class SourceEvaluatorBackend(Protocol):
    """Strict private backend; implementations may cache loaded policy/env state."""

    evaluator_implementation_digest: str

    def validate_candidate(self, request: SourceCandidateRequest) -> ValidatedSourceBinding: ...

    def evaluate_episode(
        self,
        binding: ValidatedSourceBinding,
        *,
        reset_seed: int,
    ) -> BackendEpisodeResult: ...


@runtime_checkable
class SourceReturnProjector(Protocol):
    return_contract_digest: str

    def project(
        self,
        *,
        raw_return: float,
        steps: int,
        terminated: bool,
        truncated: bool,
    ) -> float: ...


@dataclass(frozen=True)
class DmcFixedHorizonReturnContract:
    """Normalize a fixed-horizon DMC return without clipping."""

    horizon: int
    per_step_lower: float = 0.0
    per_step_upper: float = 1.0
    return_contract_digest: str | None = None
    schema: str = DMC_RETURN_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DMC_RETURN_CONTRACT_SCHEMA:
            raise SourceEvaluatorError("unsupported DMC return-contract schema")
        object.__setattr__(self, "horizon", _positive_int(self.horizon, "horizon"))
        lower = _finite(self.per_step_lower, "per_step_lower")
        upper = _finite(self.per_step_upper, "per_step_upper")
        if not lower < upper:
            raise SourceEvaluatorError("return-contract upper bound must exceed lower bound")
        object.__setattr__(self, "per_step_lower", lower)
        object.__setattr__(self, "per_step_upper", upper)
        expected = sha256_json(self._payload_without_digest())
        if self.return_contract_digest is None:
            object.__setattr__(self, "return_contract_digest", expected)
        elif _digest(self.return_contract_digest, "return_contract_digest") != expected:
            raise SourceEvaluatorError("return contract digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "horizon": self.horizon,
            "per_step_lower": self.per_step_lower,
            "per_step_upper": self.per_step_upper,
            "projection": "affine_fixed_horizon_no_clip",
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "return_contract_digest": self.return_contract_digest}

    def project(
        self,
        *,
        raw_return: float,
        steps: int,
        terminated: bool,
        truncated: bool,
    ) -> float:
        raw = _finite(raw_return, "raw_return")
        if _positive_int(steps, "steps") != self.horizon:
            raise SourceEvaluatorError("source episode violates the fixed-horizon return contract")
        if type(terminated) is not bool or type(truncated) is not bool:
            raise SourceEvaluatorError("source episode end flags must be boolean")
        lower = self.horizon * self.per_step_lower
        upper = self.horizon * self.per_step_upper
        normalized = (raw - lower) / (upper - lower)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise SourceEvaluatorError(
                "raw source return lies outside the frozen return-contract bounds"
            )
        return float(normalized)


def _candidate_request(
    *,
    cell: PoolIntakeCell,
    anchor: CanonicalSourceAnchor,
    protocol: SourceEvaluationProtocol,
) -> SourceCandidateRequest:
    return SourceCandidateRequest(
        evaluator_implementation_digest=protocol.evaluator_implementation_digest,
        intake_cell_digest=cell.intake_cell_digest,
        candidate_id=cell.job_id,
        source_anchor_id=cell.source_anchor_id,
        attempt_number=cell.attempt_number,
        attempt_digest=cell.attempt_digest,
        bundle_path=cell.bundle_path,
        bundle_digest=cell.bundle_digest,
        outer_iteration=cell.outer_iteration,
        environment_steps=cell.environment_steps,
        source_environment_digest=protocol.source_environment_digests[cell.source_anchor_id],
        anchor=anchor,
    )


def _request_from_work_unit(work_unit: SourceEvaluationWorkUnit) -> SourceCandidateRequest:
    anchor = CanonicalSourceAnchor.from_path(work_unit.anchor_manifest_path)
    return SourceCandidateRequest(
        evaluator_implementation_digest=work_unit.evaluator_implementation_digest,
        intake_cell_digest=work_unit.intake_cell_digest,
        candidate_id=work_unit.candidate_id,
        source_anchor_id=work_unit.source_anchor_id,
        attempt_number=work_unit.attempt_number,
        attempt_digest=work_unit.attempt_digest,
        bundle_path=work_unit.bundle_path,
        bundle_digest=work_unit.bundle_digest,
        outer_iteration=work_unit.outer_iteration,
        environment_steps=work_unit.environment_steps,
        source_environment_digest=work_unit.source_environment_digest,
        anchor=anchor,
    )


def _validate_backend_identity(backend: SourceEvaluatorBackend, expected_digest: str) -> str:
    if not isinstance(backend, SourceEvaluatorBackend):
        raise SourceEvaluatorError("source evaluator backend does not implement the protocol")
    observed = _digest(
        backend.evaluator_implementation_digest,
        "backend evaluator_implementation_digest",
    )
    if observed != expected_digest:
        raise SourceEvaluatorError("source evaluator backend implementation digest drifted")
    return observed


def _require_binding_matches_request(
    request: SourceCandidateRequest,
    binding: ValidatedSourceBinding,
) -> None:
    if not isinstance(binding, ValidatedSourceBinding):
        raise SourceEvaluatorError("backend did not return a typed validated source binding")
    expected = {
        "request_digest": request.request_digest,
        "candidate_id": request.candidate_id,
        "evaluator_implementation_digest": request.evaluator_implementation_digest,
        "bundle_path": request.bundle_path,
        "bundle_digest": request.bundle_digest,
        "anchor_manifest_path": request.anchor.manifest_path,
        "anchor_manifest_digest": request.anchor.manifest_digest,
        "anchor_runtime_digest": request.anchor.runtime_digest,
        "source_environment_digest": request.source_environment_digest,
    }
    drift = {
        name: {"expected": expected_value, "observed": getattr(binding, name)}
        for name, expected_value in expected.items()
        if getattr(binding, name) != expected_value
    }
    if drift:
        raise SourceEvaluatorError(f"backend validated binding drifted: {sorted(drift)}")


def plan_source_selection_work_units(
    intake: V03PoolIntakeRecord,
    protocol: SourceEvaluationProtocol,
    plan: FrozenServerPlanBinding,
    backend: SourceEvaluatorBackend,
) -> Mapping[str, SourceEvaluationWorkUnit]:
    """Generate the exact 90 selection work units without manual cell mapping."""

    if not isinstance(intake, V03PoolIntakeRecord) or intake.pool_state != "POOL_READY":
        raise SourceEvaluatorError("source planning requires a typed POOL_READY intake")
    if not isinstance(protocol, SourceEvaluationProtocol):
        raise SourceEvaluatorError("source planning requires a typed source protocol")
    if protocol.intake_record_digest != intake.intake_record_digest:
        raise SourceEvaluatorError("source protocol belongs to another intake")
    if not isinstance(plan, FrozenServerPlanBinding):
        raise SourceEvaluatorError("source planning requires a typed frozen server plan")
    if plan.plan_digest != intake.server_plan_digest:
        raise SourceEvaluatorError("frozen server-plan digest differs from the intake authority")
    if set(plan.jobs) != set(intake.cells):
        raise SourceEvaluatorError("frozen server plan does not cover the exact-90 intake")
    _validate_backend_identity(backend, protocol.evaluator_implementation_digest)

    units: dict[str, SourceEvaluationWorkUnit] = {}
    for candidate_id, cell in intake.cells.items():
        planned = plan.jobs[candidate_id]
        if (
            planned.job_digest != cell.job_digest
            or planned.seed != cell.seed
            or planned.anchor.source_anchor_id != cell.source_anchor_id
            or planned.anchor.environment_instance_digest
            != protocol.source_environment_digests.get(cell.source_anchor_id)
        ):
            raise SourceEvaluatorError(
                f"frozen server-plan cell {candidate_id} differs from intake/protocol"
            )
        request = _candidate_request(cell=cell, anchor=planned.anchor, protocol=protocol)
        binding = backend.validate_candidate(request)
        _require_binding_matches_request(request, binding)
        unit = build_source_evaluation_work_unit(
            intake,
            protocol,
            candidate_id,
            block="source_selection",
            anchor_manifest_path=planned.anchor.manifest_path,
            execution_abi=binding.execution_abi,
        )
        units[candidate_id] = unit
    if len(units) != EXPECTED_JOB_COUNT or set(units) != set(intake.cells):
        raise SourceEvaluatorError("source-selection planner failed exact-90 coverage")
    return MappingProxyType(dict(sorted(units.items())))


def plan_source_selection_from_server_plan(
    intake: V03PoolIntakeRecord,
    protocol: SourceEvaluationProtocol,
    *,
    server_plan_path: str | Path,
    backend: SourceEvaluatorBackend,
    authority: FrozenV02ServerPlanAuthority | None = None,
) -> Mapping[str, SourceEvaluationWorkUnit]:
    """CLI-ready production entry point using the frozen v0.2 authority."""

    # The generic planner remains fixture-friendly.  The production entry point
    # must never accept a merely self-consistent, caller-minted POOL_READY value.
    try:
        assert_frozen_v02_intake_authority(intake)
    except PoolIntakeError as error:
        raise SourceEvaluatorError(
            f"source planning intake lacks frozen production authority: {error}"
        ) from error
    plan = (authority or FrozenV02ServerPlanAuthority()).load(server_plan_path)
    return plan_source_selection_work_units(intake, protocol, plan, backend)


def source_work_unit_manifest(
    units: Mapping[str, SourceEvaluationWorkUnit],
) -> dict[str, Any]:
    """Return a strict JSON-ready manifest for an artifact writer/CLI."""

    rows = dict(units)
    if not rows or set(rows) != {unit.candidate_id for unit in rows.values()}:
        raise SourceEvaluatorError("source work-unit mapping is empty or mis-keyed")
    blocks = {unit.block for unit in rows.values()}
    protocols = {unit.source_evaluation_protocol_digest for unit in rows.values()}
    intakes = {unit.intake_record_digest for unit in rows.values()}
    if len(blocks) != 1 or len(protocols) != 1 or len(intakes) != 1:
        raise SourceEvaluatorError("source work-unit manifest mixes authorities")
    payload = {
        "schema": SOURCE_WORK_UNIT_MANIFEST_SCHEMA,
        "block": next(iter(blocks)),
        "source_evaluation_protocol_digest": next(iter(protocols)),
        "intake_record_digest": next(iter(intakes)),
        "work_unit_count": len(rows),
        "work_units": {key: unit.to_dict() for key, unit in sorted(rows.items())},
    }
    return {**payload, "manifest_digest": sha256_json(payload)}


@dataclass(frozen=True)
class SourceEpisodeAttempt:
    reset_seed: int
    runtime_digest: str
    state: EpisodeState
    raw_return: float | None
    normalized_return: float | None
    steps: int | None
    terminated: bool | None
    truncated: bool | None
    failure_code: str | None = None
    failure_message: str | None = None
    episode_record_digest: str | None = None
    schema: str = SOURCE_EPISODE_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_EPISODE_ATTEMPT_SCHEMA:
            raise SourceEvaluatorError("unsupported SourceEpisodeAttempt schema")
        object.__setattr__(self, "reset_seed", _seed(self.reset_seed, "reset_seed"))
        object.__setattr__(self, "runtime_digest", _digest(self.runtime_digest, "runtime_digest"))
        if self.state == "SUCCEEDED":
            object.__setattr__(self, "raw_return", _finite(self.raw_return, "raw_return"))
            normalized = _finite(self.normalized_return, "normalized_return")
            if not 0.0 <= normalized <= 1.0:
                raise SourceEvaluatorError("normalized_return must lie in [0, 1]")
            object.__setattr__(self, "normalized_return", normalized)
            object.__setattr__(self, "steps", _positive_int(self.steps, "steps"))
            if type(self.terminated) is not bool or type(self.truncated) is not bool:
                raise SourceEvaluatorError("successful source episode requires boolean end flags")
            if self.failure_code is not None or self.failure_message is not None:
                raise SourceEvaluatorError("successful source episode cannot carry a failure")
        elif self.state == "FAILED":
            if self.raw_return is not None:
                object.__setattr__(self, "raw_return", _finite(self.raw_return, "raw_return"))
            if self.normalized_return is not None:
                raise SourceEvaluatorError("failed source episode cannot claim a normalized return")
            if self.steps is not None:
                object.__setattr__(self, "steps", _positive_int(self.steps, "steps"))
            if self.terminated is not None and type(self.terminated) is not bool:
                raise SourceEvaluatorError("failed source episode terminated flag must be boolean")
            if self.truncated is not None and type(self.truncated) is not bool:
                raise SourceEvaluatorError("failed source episode truncated flag must be boolean")
            object.__setattr__(self, "failure_code", _nonempty(self.failure_code, "failure_code"))
            object.__setattr__(
                self,
                "failure_message",
                _nonempty(self.failure_message, "failure_message"),
            )
        else:
            raise SourceEvaluatorError("source episode state must be SUCCEEDED or FAILED")
        expected = sha256_json(self._payload_without_digest())
        if self.episode_record_digest is None:
            object.__setattr__(self, "episode_record_digest", expected)
        elif _digest(self.episode_record_digest, "episode_record_digest") != expected:
            raise SourceEvaluatorError("source episode-record digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reset_seed": self.reset_seed,
            "runtime_digest": self.runtime_digest,
            "state": self.state,
            "raw_return": self.raw_return,
            "normalized_return": self.normalized_return,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "episode_record_digest": self.episode_record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceEpisodeAttempt":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceEpisodeAttempt")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class SourceEvaluationAttemptRecord:
    work_unit_digest: str
    candidate_id: str
    block: EvaluationBlock
    evaluator_implementation_digest: str
    validated_binding_digest: str
    runtime_digest: str
    return_contract_digest: str
    evaluation_attempt_number: int
    expected_reset_seeds: tuple[int, ...]
    episodes: tuple[SourceEpisodeAttempt, ...]
    state: AttemptState
    failure_code: str | None = None
    failure_message: str | None = None
    attempt_record_digest: str | None = None
    schema: str = SOURCE_EVALUATION_ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_EVALUATION_ATTEMPT_SCHEMA:
            raise SourceEvaluatorError("unsupported SourceEvaluationAttemptRecord schema")
        if not _JOB_ID.fullmatch(_nonempty(self.candidate_id, "candidate_id")):
            raise SourceEvaluatorError("source attempt candidate_id is invalid")
        if self.block not in {"source_selection", "source_attestation"}:
            raise SourceEvaluatorError("source attempt block is invalid")
        for name in (
            "work_unit_digest",
            "evaluator_implementation_digest",
            "validated_binding_digest",
            "runtime_digest",
            "return_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "evaluation_attempt_number",
            _positive_int(self.evaluation_attempt_number, "evaluation_attempt_number"),
        )
        seeds = tuple(_seed(seed, "expected_reset_seeds[]") for seed in self.expected_reset_seeds)
        if not seeds or seeds != tuple(sorted(set(seeds))):
            raise SourceEvaluatorError("attempt reset seeds must be sorted, unique, and non-empty")
        episodes = tuple(self.episodes)
        if any(not isinstance(row, SourceEpisodeAttempt) for row in episodes):
            raise SourceEvaluatorError("attempt episodes must be typed")
        if tuple(row.reset_seed for row in episodes) != seeds[: len(episodes)]:
            raise SourceEvaluatorError("attempt episodes are not the literal seed prefix")
        if any(row.runtime_digest != self.runtime_digest for row in episodes):
            raise SourceEvaluatorError("attempt episode runtime digest drifted")
        if self.state == "SUCCEEDED":
            if len(episodes) != len(seeds) or any(row.state != "SUCCEEDED" for row in episodes):
                raise SourceEvaluatorError("successful attempt must cover every literal seed")
            if self.failure_code is not None or self.failure_message is not None:
                raise SourceEvaluatorError("successful attempt cannot carry a failure")
        elif self.state == "FAILED":
            if not episodes or episodes[-1].state != "FAILED" or any(
                row.state != "SUCCEEDED" for row in episodes[:-1]
            ):
                raise SourceEvaluatorError("failed attempt must end at its first failed seed")
            object.__setattr__(self, "failure_code", _nonempty(self.failure_code, "failure_code"))
            object.__setattr__(
                self,
                "failure_message",
                _nonempty(self.failure_message, "failure_message"),
            )
            if (
                episodes[-1].failure_code != self.failure_code
                or episodes[-1].failure_message != self.failure_message
            ):
                raise SourceEvaluatorError("attempt failure differs from its final episode")
        else:
            raise SourceEvaluatorError("source attempt state must be SUCCEEDED or FAILED")
        object.__setattr__(self, "expected_reset_seeds", seeds)
        object.__setattr__(self, "episodes", episodes)
        expected = sha256_json(self._payload_without_digest())
        if self.attempt_record_digest is None:
            object.__setattr__(self, "attempt_record_digest", expected)
        elif _digest(self.attempt_record_digest, "attempt_record_digest") != expected:
            raise SourceEvaluatorError("source attempt-record digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "work_unit_digest": self.work_unit_digest,
            "candidate_id": self.candidate_id,
            "block": self.block,
            "evaluator_implementation_digest": self.evaluator_implementation_digest,
            "validated_binding_digest": self.validated_binding_digest,
            "runtime_digest": self.runtime_digest,
            "return_contract_digest": self.return_contract_digest,
            "evaluation_attempt_number": self.evaluation_attempt_number,
            "expected_reset_seeds": list(self.expected_reset_seeds),
            "episodes": [row.to_dict() for row in self.episodes],
            "state": self.state,
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "attempt_record_digest": self.attempt_record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceEvaluationAttemptRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SourceEvaluationAttemptRecord")
        return cls(
            **{
                name: (
                    tuple(SourceEpisodeAttempt.from_dict(row) for row in value[name])
                    if name == "episodes"
                    else tuple(value[name])
                    if name == "expected_reset_seeds"
                    else value[name]
                )
                for name in fields
            }
        )


def rebuild_raw_source_episode_shard(
    work_unit: SourceEvaluationWorkUnit,
    attempt: SourceEvaluationAttemptRecord,
    return_contract: SourceReturnProjector,
) -> RawSourceEpisodeShard:
    """Recompute every normalized value from raw attempt rows before receipt use."""

    if not isinstance(work_unit, SourceEvaluationWorkUnit):
        raise SourceEvaluatorError("raw-shard rebuild requires a typed work unit")
    if not isinstance(attempt, SourceEvaluationAttemptRecord) or attempt.state != "SUCCEEDED":
        raise SourceEvaluatorError("raw-shard rebuild requires a successful typed attempt")
    if not isinstance(return_contract, SourceReturnProjector):
        raise SourceEvaluatorError("raw-shard rebuild requires a return projector")
    contract_digest = _digest(return_contract.return_contract_digest, "return_contract_digest")
    if (
        attempt.work_unit_digest != work_unit.work_unit_digest
        or attempt.candidate_id != work_unit.candidate_id
        or attempt.block != work_unit.block
        or attempt.evaluator_implementation_digest != work_unit.evaluator_implementation_digest
        or attempt.runtime_digest != work_unit.anchor_runtime_digest
        or attempt.return_contract_digest != work_unit.return_contract_digest
        or contract_digest != work_unit.return_contract_digest
        or attempt.expected_reset_seeds != work_unit.reset_seeds
    ):
        raise SourceEvaluatorError("source attempt differs from work-unit/return authorities")
    raw: list[float] = []
    normalized: list[float] = []
    for row in attempt.episodes:
        assert row.raw_return is not None
        assert row.steps is not None
        assert row.terminated is not None
        assert row.truncated is not None
        recomputed = return_contract.project(
            raw_return=row.raw_return,
            steps=row.steps,
            terminated=row.terminated,
            truncated=row.truncated,
        )
        if row.normalized_return != recomputed:
            raise SourceEvaluatorError("stored normalized return differs from raw-return recomputation")
        raw.append(row.raw_return)
        normalized.append(recomputed)
    return RawSourceEpisodeShard(
        work_unit_digest=work_unit.work_unit_digest,
        attempt_record_digest=attempt.attempt_record_digest,
        validated_binding_digest=attempt.validated_binding_digest,
        evaluation_attempt_number=attempt.evaluation_attempt_number,
        block=work_unit.block,
        candidate_id=work_unit.candidate_id,
        runtime_digest=attempt.runtime_digest,
        reset_seeds=work_unit.reset_seeds,
        raw_episode_returns=tuple(raw),
        normalized_returns=tuple(normalized),
        return_contract_digest=work_unit.return_contract_digest,
    )


@dataclass(frozen=True)
class SourceEvaluationRun:
    attempt: SourceEvaluationAttemptRecord
    raw_episode_shard: RawSourceEpisodeShard
    receipt: EvaluatorSourceReceipt
    run_digest: str | None = None
    schema: str = SOURCE_EVALUATION_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_EVALUATION_RUN_SCHEMA:
            raise SourceEvaluatorError("unsupported SourceEvaluationRun schema")
        if not isinstance(self.attempt, SourceEvaluationAttemptRecord):
            raise SourceEvaluatorError("source evaluation run requires a typed attempt")
        if self.attempt.state != "SUCCEEDED":
            raise SourceEvaluatorError("source evaluation run requires a successful attempt")
        if not isinstance(self.raw_episode_shard, RawSourceEpisodeShard):
            raise SourceEvaluatorError("source evaluation run requires a typed raw shard")
        if not isinstance(self.receipt, EvaluatorSourceReceipt):
            raise SourceEvaluatorError("source evaluation run requires a typed receipt")

        attempt_raw = tuple(row.raw_return for row in self.attempt.episodes)
        attempt_normalized = tuple(
            row.normalized_return for row in self.attempt.episodes
        )
        expected_dataset_digest = sha256_json(
            {
                "schema": SOURCE_ROLLOUT_DATASET_SCHEMA,
                "work_unit_digest": self.attempt.work_unit_digest,
                "attempt_record_digest": self.attempt.attempt_record_digest,
                "validated_binding_digest": self.attempt.validated_binding_digest,
                "evaluation_attempt_number": self.attempt.evaluation_attempt_number,
                "raw_episode_shard_digest": self.raw_episode_shard.episode_shard_digest,
            }
        )
        if (
            self.attempt.work_unit_digest != self.raw_episode_shard.work_unit_digest
            or self.attempt.work_unit_digest != self.receipt.work_unit_digest
            or self.attempt.candidate_id != self.raw_episode_shard.candidate_id
            or self.attempt.candidate_id != self.receipt.candidate_id
            or self.attempt.block != self.raw_episode_shard.block
            or self.attempt.block != self.receipt.block
            or self.attempt.runtime_digest != self.raw_episode_shard.runtime_digest
            or self.attempt.runtime_digest != self.receipt.runtime_digest
            or self.attempt.return_contract_digest
            != self.raw_episode_shard.return_contract_digest
            or self.attempt.return_contract_digest
            != self.receipt.return_contract_digest
            or self.attempt.evaluator_implementation_digest
            != self.receipt.evaluator_implementation_digest
            or self.attempt.expected_reset_seeds
            != self.raw_episode_shard.reset_seeds
            or self.attempt.expected_reset_seeds != self.receipt.reset_seeds
            or attempt_raw != self.raw_episode_shard.raw_episode_returns
            or attempt_normalized != self.raw_episode_shard.normalized_returns
            or attempt_normalized != self.receipt.normalized_returns
            or self.attempt.attempt_record_digest
            != self.raw_episode_shard.attempt_record_digest
            or self.attempt.attempt_record_digest != self.receipt.attempt_record_digest
            or self.attempt.validated_binding_digest
            != self.raw_episode_shard.validated_binding_digest
            or self.attempt.validated_binding_digest
            != self.receipt.validated_binding_digest
            or self.attempt.evaluation_attempt_number
            != self.raw_episode_shard.evaluation_attempt_number
            or self.attempt.evaluation_attempt_number
            != self.receipt.evaluation_attempt_number
            or self.raw_episode_shard.episode_shard_digest
            != self.receipt.raw_episode_shard_digest
            or self.receipt.dataset_digest != expected_dataset_digest
        ):
            raise SourceEvaluatorError("source evaluation run contains inconsistent evidence")
        expected = sha256_json(self._payload_without_digest())
        if self.run_digest is None:
            object.__setattr__(self, "run_digest", expected)
        elif _digest(self.run_digest, "run_digest") != expected:
            raise SourceEvaluatorError("source evaluation run digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "attempt": self.attempt.to_dict(),
            "raw_episode_shard": self.raw_episode_shard.to_dict(),
            "receipt": self.receipt.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "run_digest": self.run_digest}


def _episode_failure(
    *,
    reset_seed: int,
    runtime_digest: str,
    code: str,
    message: str,
    raw_return: float | None = None,
    steps: int | None = None,
    terminated: bool | None = None,
    truncated: bool | None = None,
) -> SourceEpisodeAttempt:
    return SourceEpisodeAttempt(
        reset_seed=reset_seed,
        runtime_digest=runtime_digest,
        state="FAILED",
        raw_return=raw_return,
        normalized_return=None,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        failure_code=code,
        failure_message=message,
    )


def _raise_failed_attempt(
    *,
    work_unit: SourceEvaluationWorkUnit,
    binding: ValidatedSourceBinding,
    return_contract_digest: str,
    evaluation_attempt_number: int,
    episodes: Sequence[SourceEpisodeAttempt],
) -> None:
    final = episodes[-1]
    raise SourceEvaluationAttemptFailed(
        SourceEvaluationAttemptRecord(
            work_unit_digest=work_unit.work_unit_digest,
            candidate_id=work_unit.candidate_id,
            block=work_unit.block,
            evaluator_implementation_digest=work_unit.evaluator_implementation_digest,
            validated_binding_digest=binding.binding_digest,
            runtime_digest=binding.anchor_runtime_digest,
            return_contract_digest=return_contract_digest,
            evaluation_attempt_number=evaluation_attempt_number,
            expected_reset_seeds=work_unit.reset_seeds,
            episodes=tuple(episodes),
            state="FAILED",
            failure_code=final.failure_code,
            failure_message=final.failure_message,
        )
    )


def run_source_evaluation_work_unit(
    work_unit: SourceEvaluationWorkUnit,
    *,
    backend: SourceEvaluatorBackend,
    return_contract: SourceReturnProjector,
    evaluation_attempt_number: int = 1,
) -> SourceEvaluationRun:
    """Execute each literal seed and return a receipt only after raw recomputation."""

    if not isinstance(work_unit, SourceEvaluationWorkUnit):
        raise SourceEvaluatorError("source runner requires a typed work unit")
    attempt_number = _positive_int(evaluation_attempt_number, "evaluation_attempt_number")
    _validate_backend_identity(backend, work_unit.evaluator_implementation_digest)
    if not isinstance(return_contract, SourceReturnProjector):
        raise SourceEvaluatorError("source runner requires a return projector")
    contract_digest = _digest(return_contract.return_contract_digest, "return_contract_digest")
    if contract_digest != work_unit.return_contract_digest:
        raise SourceEvaluatorError("source runner return-contract digest drifted")
    request = _request_from_work_unit(work_unit)
    binding = backend.validate_candidate(request)
    _require_binding_matches_request(request, binding)
    if binding.execution_abi != work_unit.execution_abi:
        raise SourceEvaluatorError("backend execution ABI drifted after work-unit freeze")

    episodes: list[SourceEpisodeAttempt] = []
    for reset_seed in work_unit.reset_seeds:
        try:
            result = backend.evaluate_episode(binding, reset_seed=reset_seed)
        except Exception as error:  # backend boundary; converted into private failure evidence
            row = _episode_failure(
                reset_seed=reset_seed,
                runtime_digest=binding.anchor_runtime_digest,
                code="BACKEND_EXCEPTION",
                message=f"{type(error).__name__}: {error}",
            )
            episodes.append(row)
            _raise_failed_attempt(
                work_unit=work_unit,
                binding=binding,
                return_contract_digest=contract_digest,
                evaluation_attempt_number=attempt_number,
                episodes=episodes,
            )
        if not isinstance(result, BackendEpisodeResult):
            row = _episode_failure(
                reset_seed=reset_seed,
                runtime_digest=binding.anchor_runtime_digest,
                code="BACKEND_PROTOCOL_VIOLATION",
                message="backend returned a non-typed episode result",
            )
            episodes.append(row)
            _raise_failed_attempt(
                work_unit=work_unit,
                binding=binding,
                return_contract_digest=contract_digest,
                evaluation_attempt_number=attempt_number,
                episodes=episodes,
            )
        if result.reset_seed != reset_seed or result.runtime_digest != binding.anchor_runtime_digest:
            row = _episode_failure(
                reset_seed=reset_seed,
                runtime_digest=binding.anchor_runtime_digest,
                code="BACKEND_PROTOCOL_VIOLATION",
                message="backend seed/runtime differs from the literal work unit",
            )
            episodes.append(row)
            _raise_failed_attempt(
                work_unit=work_unit,
                binding=binding,
                return_contract_digest=contract_digest,
                evaluation_attempt_number=attempt_number,
                episodes=episodes,
            )
        if result.state == "FAILED":
            row = _episode_failure(
                reset_seed=reset_seed,
                runtime_digest=result.runtime_digest,
                code=result.failure_code or "BACKEND_FAILURE",
                message=result.failure_message or "backend episode failed",
            )
            episodes.append(row)
            _raise_failed_attempt(
                work_unit=work_unit,
                binding=binding,
                return_contract_digest=contract_digest,
                evaluation_attempt_number=attempt_number,
                episodes=episodes,
            )
        assert result.raw_return is not None
        assert result.steps is not None
        assert result.terminated is not None
        assert result.truncated is not None
        try:
            normalized = return_contract.project(
                raw_return=result.raw_return,
                steps=result.steps,
                terminated=result.terminated,
                truncated=result.truncated,
            )
        except Exception as error:
            row = _episode_failure(
                reset_seed=reset_seed,
                runtime_digest=result.runtime_digest,
                code="RETURN_CONTRACT_VIOLATION",
                message=f"{type(error).__name__}: {error}",
                raw_return=result.raw_return,
                steps=result.steps,
                terminated=result.terminated,
                truncated=result.truncated,
            )
            episodes.append(row)
            _raise_failed_attempt(
                work_unit=work_unit,
                binding=binding,
                return_contract_digest=contract_digest,
                evaluation_attempt_number=attempt_number,
                episodes=episodes,
            )
        episodes.append(
            SourceEpisodeAttempt(
                reset_seed=reset_seed,
                runtime_digest=result.runtime_digest,
                state="SUCCEEDED",
                raw_return=result.raw_return,
                normalized_return=normalized,
                steps=result.steps,
                terminated=result.terminated,
                truncated=result.truncated,
            )
        )

    attempt = SourceEvaluationAttemptRecord(
        work_unit_digest=work_unit.work_unit_digest,
        candidate_id=work_unit.candidate_id,
        block=work_unit.block,
        evaluator_implementation_digest=work_unit.evaluator_implementation_digest,
        validated_binding_digest=binding.binding_digest,
        runtime_digest=binding.anchor_runtime_digest,
        return_contract_digest=contract_digest,
        evaluation_attempt_number=attempt_number,
        expected_reset_seeds=work_unit.reset_seeds,
        episodes=tuple(episodes),
        state="SUCCEEDED",
    )
    shard = rebuild_raw_source_episode_shard(work_unit, attempt, return_contract)
    receipt = receipt_from_source_episode_shard(work_unit, shard, attempt)
    return SourceEvaluationRun(attempt=attempt, raw_episode_shard=shard, receipt=receipt)


__all__ = [
    "BackendEpisodeResult",
    "CanonicalSourceAnchor",
    "DmcFixedHorizonReturnContract",
    "FrozenPlanJobBinding",
    "FrozenServerPlanBinding",
    "FrozenV02ServerPlanAuthority",
    "SourceCandidateRequest",
    "SourceEvaluationAttemptFailed",
    "SourceEvaluationAttemptRecord",
    "SourceEvaluationRun",
    "SourceEvaluatorBackend",
    "SourceEvaluatorError",
    "SourceEpisodeAttempt",
    "SourceReturnProjector",
    "ValidatedSourceBinding",
    "plan_source_selection_from_server_plan",
    "plan_source_selection_work_units",
    "rebuild_raw_source_episode_shard",
    "run_source_evaluation_work_unit",
    "source_work_unit_manifest",
]
