"""Digest-bound source-anchor training plans, attestations, and admission."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..hashing import canonicalize, sha256_json


TrainingStatus = Literal["planned", "running", "succeeded", "recovered", "failed"]
ExecutionPurpose = Literal[
    "audit_smoke", "development_discovery", "v02_freeze_ready"
]
EXECUTION_PURPOSES = frozenset(
    {"audit_smoke", "development_discovery", "v02_freeze_ready"}
)


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise ValueError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a SHA-256 digest") from exc
    return result


def _git_commit(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 40:
        raise ValueError(f"{where} must be a full 40-character Git commit")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be hexadecimal") from exc
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _freeze_mapping(value: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{where} must be a non-empty mapping")
    canonical = canonicalize(value)

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({str(key): freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(canonical)


@dataclass(frozen=True)
class PolicyTrainingJob:
    job_id: str
    config_digest: str
    execution_purpose: ExecutionPurpose
    source_anchor_id: str
    environment_instance_digest: str
    anchor_manifest_digest: str
    algorithm: str
    trainer_config: Mapping[str, Any]
    seed: int
    environment_steps: int
    checkpoint_rule: str
    trainer_commit: str
    dependency_digest: str
    runtime_digest: str
    training_protocol_id: str

    def __post_init__(self) -> None:
        for name in ("job_id", "algorithm", "checkpoint_rule", "execution_purpose"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.execution_purpose not in EXECUTION_PURPOSES:
            raise ValueError(
                f"unsupported training execution purpose: {self.execution_purpose!r}"
            )
        if self.algorithm not in {"ppo", "fpo"}:
            raise ValueError("primary training algorithm must be ppo or fpo")
        for name in (
            "source_anchor_id",
            "config_digest",
            "environment_instance_digest",
            "anchor_manifest_digest",
            "dependency_digest",
            "runtime_digest",
            "training_protocol_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "trainer_commit", _git_commit(self.trainer_commit, "trainer_commit"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("training seed must be a non-negative integer")
        object.__setattr__(self, "environment_steps", _positive_int(self.environment_steps, "environment_steps"))
        object.__setattr__(self, "trainer_config", _freeze_mapping(self.trainer_config, "trainer_config"))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema": "policy-learnware.v02-policy-training-job.v0",
                "job_id": self.job_id,
                "config_digest": self.config_digest,
                "execution_purpose": self.execution_purpose,
                "source_anchor_id": self.source_anchor_id,
                "environment_instance_digest": self.environment_instance_digest,
                "anchor_manifest_digest": self.anchor_manifest_digest,
                "algorithm": self.algorithm,
                "trainer_config": self.trainer_config,
                "seed": self.seed,
                "environment_steps": self.environment_steps,
                "checkpoint_rule": self.checkpoint_rule,
                "trainer_commit": self.trainer_commit,
                "dependency_digest": self.dependency_digest,
                "runtime_digest": self.runtime_digest,
                "training_protocol_id": self.training_protocol_id,
            }
        )


@dataclass(frozen=True)
class PolicyTrainingAttestation:
    job_id: str
    job_digest: str
    attempt_id: str
    attempt_number: int
    source_anchor_id: str
    anchor_manifest_digest: str
    declared_environment_instance_digest: str
    actual_train_environment_instance_digest: str
    actual_eval_environment_instance_digest: str
    operator_digest: str | None
    model_diff_digest: str
    algorithm: str
    seed: int
    environment_steps: int
    checkpoint_rule: str
    checkpoint_digests: Mapping[str, str]
    bundle_digest: str
    bundle_manifest_digest: str
    golden_parity_digest: str
    compiled_parity_digest: str
    finiteness_audit_digest: str
    all_arrays_finite: bool
    golden_parity_passed: bool
    compiled_parity_passed: bool
    trainer_commit: str
    dependency_digest: str
    runtime_digest: str
    hardware_digest: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    status: TrainingStatus
    failure_reason: str | None = None
    bundle_path: str | None = None
    server_plan_binding_digest: str | None = None
    server_training_plan_digest: str | None = None
    server_job_digest: str | None = None
    server_attempt_digest: str | None = None
    server_run_manifest_digest: str | None = None
    server_training_record_digest: str | None = None
    planned_outer_iterations: int | None = None
    completed_outer_iterations: int | None = None
    promoted_outer_iteration: int | None = None
    planned_environment_steps: int | None = None
    completed_environment_steps: int | None = None
    promoted_environment_steps: int | None = None
    failure_type: str | None = None
    failure_trace_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "job_id",
            "attempt_id",
            "algorithm",
            "checkpoint_rule",
            "started_at",
            "finished_at",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "job_digest",
            "source_anchor_id",
            "anchor_manifest_digest",
            "declared_environment_instance_digest",
            "actual_train_environment_instance_digest",
            "actual_eval_environment_instance_digest",
            "model_diff_digest",
            "bundle_digest",
            "bundle_manifest_digest",
            "golden_parity_digest",
            "compiled_parity_digest",
            "finiteness_audit_digest",
            "dependency_digest",
            "runtime_digest",
            "hardware_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.operator_digest is not None:
            object.__setattr__(
                self,
                "operator_digest",
                _digest(self.operator_digest, "operator_digest"),
            )
        server_digests = (
            "server_plan_binding_digest",
            "server_training_plan_digest",
            "server_job_digest",
            "server_attempt_digest",
            "server_run_manifest_digest",
            "server_training_record_digest",
        )
        present = tuple(getattr(self, name) is not None for name in server_digests)
        if any(present) and not all(present):
            raise ValueError(
                "server-bound training provenance must supply every server digest"
            )
        for name in server_digests:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _digest(value, name))
        object.__setattr__(self, "trainer_commit", _git_commit(self.trainer_commit, "trainer_commit"))
        object.__setattr__(self, "attempt_number", _positive_int(self.attempt_number, "attempt_number"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("attested training seed must be non-negative")
        object.__setattr__(self, "environment_steps", _positive_int(self.environment_steps, "environment_steps"))
        checkpoints = dict(self.checkpoint_digests)
        if not checkpoints:
            raise ValueError("successful training attestation requires checkpoint digests")
        object.__setattr__(
            self,
            "checkpoint_digests",
            MappingProxyType(
                {str(name): _digest(value, f"checkpoint_digests[{name!r}]") for name, value in sorted(checkpoints.items())}
            ),
        )
        for name in ("all_arrays_finite", "golden_parity_passed", "compiled_parity_passed"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        if self.status not in {
            "planned",
            "running",
            "succeeded",
            "recovered",
            "failed",
        }:
            raise ValueError(f"invalid training status: {self.status!r}")
        outer_values = (
            self.planned_outer_iterations,
            self.completed_outer_iterations,
            self.promoted_outer_iteration,
        )
        step_values = (
            self.planned_environment_steps,
            self.completed_environment_steps,
            self.promoted_environment_steps,
        )
        if any(value is not None for value in outer_values):
            if not all(value is not None for value in outer_values):
                raise ValueError("terminal outer-iteration provenance must be complete")
            for name, value in zip(
                (
                    "planned_outer_iterations",
                    "completed_outer_iterations",
                    "promoted_outer_iteration",
                ),
                outer_values,
                strict=True,
            ):
                object.__setattr__(self, name, _positive_int(value, name))
        if any(value is not None for value in step_values):
            if not all(value is not None for value in step_values):
                raise ValueError("terminal environment-step provenance must be complete")
            for name, value in zip(
                (
                    "planned_environment_steps",
                    "completed_environment_steps",
                    "promoted_environment_steps",
                ),
                step_values,
                strict=True,
            ):
                object.__setattr__(self, name, _positive_int(value, name))
        if (any(value is not None for value in outer_values)) is not (
            any(value is not None for value in step_values)
        ):
            raise ValueError("outer and environment-step terminal provenance must travel together")
        if outer_values[0] is not None:
            if self.planned_environment_steps % self.planned_outer_iterations != 0:
                raise ValueError("terminal training geometry must be integral")
            steps_per_outer = (
                self.planned_environment_steps // self.planned_outer_iterations
            )
            if (
                self.completed_environment_steps
                != self.completed_outer_iterations * steps_per_outer
                or self.promoted_environment_steps
                != self.promoted_outer_iteration * steps_per_outer
            ):
                raise ValueError("terminal outer/step provenance geometry is inconsistent")
        if self.failure_type is not None:
            object.__setattr__(self, "failure_type", _nonempty(self.failure_type, "failure_type"))
        if self.failure_trace_digest is not None:
            object.__setattr__(
                self,
                "failure_trace_digest",
                _digest(self.failure_trace_digest, "failure_trace_digest"),
            )
        if self.status in {"succeeded", "recovered"}:
            if self.failure_reason is not None:
                if self.status == "succeeded":
                    raise ValueError("succeeded training cannot have a failure_reason")
            if not (
                self.all_arrays_finite
                and self.golden_parity_passed
                and self.compiled_parity_passed
            ):
                raise ValueError("non-finite or parity-failed bundle cannot be attested succeeded")
            identities = {
                self.declared_environment_instance_digest,
                self.actual_train_environment_instance_digest,
                self.actual_eval_environment_instance_digest,
            }
            if len(identities) != 1:
                raise ValueError("train/eval actual environment digest differs from the anchor")
            if self.status == "succeeded":
                if self.failure_type is not None or self.failure_trace_digest is not None:
                    raise ValueError("succeeded training cannot retain recovery failure metadata")
                if outer_values[0] is not None and not (
                    self.planned_outer_iterations
                    == self.completed_outer_iterations
                    == self.promoted_outer_iteration
                ):
                    raise ValueError("succeeded training must promote the completed final outer")
                if step_values[0] is not None and not (
                    self.planned_environment_steps
                    == self.completed_environment_steps
                    == self.promoted_environment_steps
                    == self.environment_steps
                ):
                    raise ValueError("succeeded training budget provenance is inconsistent")
            else:
                if not self.failure_reason:
                    raise ValueError("recovered training must retain the numerical failure message")
                if self.checkpoint_rule != "fixed_ladder":
                    raise ValueError("recovered training requires a fixed_ladder checkpoint rule")
                if self.failure_type != "NumericalIntegrityError":
                    raise ValueError("only NumericalIntegrityError may produce a recovered attestation")
                if self.failure_trace_digest is None:
                    raise ValueError("recovered training must retain a failure trace digest")
                if outer_values[0] is None or step_values[0] is None:
                    raise ValueError("recovered training requires complete terminal budget provenance")
                if not (
                    self.promoted_outer_iteration
                    <= self.completed_outer_iterations
                    < self.planned_outer_iterations
                ):
                    raise ValueError("recovered outer iterations are inconsistent")
                if not (
                    self.promoted_environment_steps
                    <= self.completed_environment_steps
                    < self.planned_environment_steps
                ):
                    raise ValueError("recovered environment-step budgets are inconsistent")
                if self.environment_steps != self.promoted_environment_steps:
                    raise ValueError("recovered attestation environment_steps must be the promoted bundle budget")
        elif self.status == "failed" and not self.failure_reason:
            raise ValueError("failed training must retain a failure_reason")
        if self.bundle_path is not None:
            object.__setattr__(self, "bundle_path", _nonempty(self.bundle_path, "bundle_path"))

    @property
    def is_server_bound(self) -> bool:
        """Whether all package↔server provenance digests are present.

        CPU acceptance fixtures may intentionally construct package-only
        attestations.  Production admission from the anchor-aware backend must
        instead go through the strict server bridge, which fills this complete
        digest set atomically.
        """

        return self.server_plan_binding_digest is not None

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema": "policy-learnware.v02-policy-training-attestation.v1",
                **{name: getattr(self, name) for name in self.__dataclass_fields__},
            }
        )


@dataclass(frozen=True)
class AdmittedTrainingRecord:
    job: PolicyTrainingJob
    attestation: PolicyTrainingAttestation

    def __post_init__(self) -> None:
        if self.attestation.status not in {"succeeded", "recovered"}:
            raise ValueError("only succeeded or numerically recovered attestations may be admitted")
        checks = {
            "job_id": self.job.job_id == self.attestation.job_id,
            "job_digest": self.job.digest == self.attestation.job_digest,
            "source_anchor_id": self.job.source_anchor_id == self.attestation.source_anchor_id,
            "anchor_manifest_digest": self.job.anchor_manifest_digest == self.attestation.anchor_manifest_digest,
            "environment_instance_digest": self.job.environment_instance_digest == self.attestation.declared_environment_instance_digest,
            "algorithm": self.job.algorithm == self.attestation.algorithm,
            "seed": self.job.seed == self.attestation.seed,
            "environment_steps": (
                self.job.environment_steps == self.attestation.environment_steps
                if self.attestation.status == "succeeded"
                else self.job.environment_steps
                == self.attestation.planned_environment_steps
            ),
            "checkpoint_rule": self.job.checkpoint_rule == self.attestation.checkpoint_rule,
            "trainer_commit": self.job.trainer_commit == self.attestation.trainer_commit,
            "dependency_digest": self.job.dependency_digest == self.attestation.dependency_digest,
            "runtime_digest": self.job.runtime_digest == self.attestation.runtime_digest,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"training attestation differs from frozen job: {failed}")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v02-admitted-training-record.v0",
                "job_digest": self.job.digest,
                "attestation_digest": self.attestation.digest,
            }
        )


def admitted_training_records_digest(
    records: Mapping[str, AdmittedTrainingRecord],
) -> str:
    """Digest the *entire* admitted candidate set, including bundle bindings.

    The mapping key is part of the contract and must equal the frozen training
    job ID.  Championization uses this digest so a caller cannot silently swap,
    add, or remove a candidate after source evaluation.
    """

    if not isinstance(records, Mapping) or not records:
        raise ValueError("admitted training records must be a non-empty mapping")
    payload: dict[str, Any] = {}
    for candidate_id, record in sorted(records.items()):
        candidate = _nonempty(candidate_id, "admitted candidate ID")
        if not isinstance(record, AdmittedTrainingRecord):
            raise ValueError("admitted training records must contain typed records")
        if candidate != record.job.job_id:
            raise ValueError("admitted candidate key differs from its frozen job ID")
        payload[candidate] = {
            "record_digest": record.digest,
            "source_anchor_id": record.job.source_anchor_id,
            "training_seed": record.job.seed,
            "bundle_digest": record.attestation.bundle_digest,
        }
    return sha256_json(
        {
            "schema": "policy-learnware.v02-admitted-training-record-index.v0",
            "records": payload,
        }
    )


def validate_admitted_training_grid(
    records: Mapping[str, AdmittedTrainingRecord],
    *,
    expected_anchor_ids: Iterable[str],
    expected_seeds: Iterable[int],
    algorithm: str,
    environment_steps: int,
    checkpoint_rule: str,
) -> Mapping[str, tuple[str, ...]]:
    """Validate an exact ``source anchor x seed`` admitted-training matrix.

    This is deliberately parameterized by the reviewed configuration.  It does
    not choose algorithms, budgets, seeds, checkpoints, or source anchors.
    """

    admitted_training_records_digest(records)
    anchors = tuple(sorted({_digest(item, "expected source anchor ID") for item in expected_anchor_ids}))
    if not anchors:
        raise ValueError("expected source anchor IDs cannot be empty")
    raw_seeds = tuple(expected_seeds)
    if (
        not raw_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in raw_seeds)
        or len(raw_seeds) != len(set(raw_seeds))
    ):
        raise ValueError("expected training seeds must be unique non-negative integers")
    seeds = tuple(sorted(raw_seeds))
    algorithm_id = _nonempty(algorithm, "expected algorithm").lower()
    if algorithm_id not in {"ppo", "fpo"}:
        raise ValueError("expected algorithm must be PPO/FPO")
    steps = _positive_int(environment_steps, "expected environment_steps")
    checkpoint = _nonempty(checkpoint_rule, "expected checkpoint_rule")

    expected_units = {(anchor, seed) for anchor in anchors for seed in seeds}
    observed_units: dict[tuple[str, int], str] = {}
    by_anchor: dict[str, list[str]] = {anchor: [] for anchor in anchors}
    for candidate_id, record in sorted(records.items()):
        job = record.job
        checks = {
            "algorithm": job.algorithm == algorithm_id,
            "environment_steps": job.environment_steps == steps,
            "checkpoint_rule": job.checkpoint_rule == checkpoint,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"admitted candidate {candidate_id!r} differs from reviewed training config: {failed}"
            )
        unit = (job.source_anchor_id, job.seed)
        if unit not in expected_units:
            raise ValueError("admitted candidate is outside the reviewed anchor/seed grid")
        if unit in observed_units:
            raise ValueError("multiple admitted candidates occupy one reviewed anchor/seed unit")
        observed_units[unit] = candidate_id
        by_anchor[job.source_anchor_id].append(candidate_id)
    if set(observed_units) != expected_units:
        missing = sorted(expected_units - set(observed_units))
        raise ValueError(f"admitted training grid is incomplete; missing={missing}")
    return MappingProxyType(
        {anchor: tuple(sorted(candidates)) for anchor, candidates in sorted(by_anchor.items())}
    )


@dataclass(frozen=True)
class HistoricalBundleReuseAudit:
    bundle_digest: str
    task_reward_reset_horizon_action_repeat_equal: bool
    nominal_trajectory_identity: bool
    algorithm_budget_checkpoint_comparable: bool
    seed_allowed: bool
    checksum_golden_compiled_runtime_parity: bool
    independent_v02_evaluation_planned: bool
    anchor_sidecar_bound: bool
    passed: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_digest", _digest(self.bundle_digest, "bundle_digest"))
        fields = tuple(
            name
            for name in self.__dataclass_fields__
            if name not in {"bundle_digest", "passed", "reasons"}
        )
        if any(not isinstance(getattr(self, name), bool) for name in fields):
            raise ValueError("reuse audit checks must be boolean")
        expected = all(getattr(self, name) for name in fields)
        if self.passed != expected:
            raise ValueError("reuse audit passed flag must equal the conjunction of all checks")
        if self.passed and self.reasons:
            raise ValueError("passed reuse audit cannot retain failure reasons")
        if not self.passed and not self.reasons:
            raise ValueError("failed reuse audit must retain reasons")


def plan_training_jobs(
    anchors: Mapping[str, Mapping[str, str]],
    *,
    config_digest: str,
    execution_purpose: ExecutionPurpose,
    algorithm: str,
    seeds: Sequence[int],
    environment_steps: int,
    checkpoint_rule: str,
    trainer_config: Mapping[str, Any],
    trainer_commit: str,
    dependency_digest: str,
    runtime_digest: str,
    training_protocol_id: str,
) -> tuple[PolicyTrainingJob, ...]:
    """Create a deterministic, purpose- and config-bound anchor × seed matrix."""

    config_id = _digest(config_digest, "config_digest")
    purpose = _nonempty(execution_purpose, "execution_purpose")
    if purpose not in EXECUTION_PURPOSES:
        raise ValueError(f"unsupported training execution purpose: {purpose!r}")

    seed_values = tuple(seeds)
    if not seed_values or len(seed_values) != len(set(seed_values)):
        raise ValueError("training seeds must be non-empty and unique")
    jobs: list[PolicyTrainingJob] = []
    for anchor_id, manifest in sorted(anchors.items()):
        required = {"environment_instance_digest", "anchor_manifest_digest"}
        if set(manifest) != required:
            raise ValueError("anchor plan rows must contain exact digest keys")
        for seed in seed_values:
            job_id = (
                "v02j-"
                + sha256_json(
                    {
                        "anchor_id": anchor_id,
                        "config_digest": config_id,
                        "execution_purpose": purpose,
                        "algorithm": algorithm,
                        "seed": seed,
                        "training_protocol_id": training_protocol_id,
                    }
                )[:24]
            )
            jobs.append(
                PolicyTrainingJob(
                    job_id=job_id,
                    config_digest=config_id,
                    execution_purpose=purpose,
                    source_anchor_id=anchor_id,
                    environment_instance_digest=manifest["environment_instance_digest"],
                    anchor_manifest_digest=manifest["anchor_manifest_digest"],
                    algorithm=algorithm,
                    trainer_config=trainer_config,
                    seed=seed,
                    environment_steps=environment_steps,
                    checkpoint_rule=checkpoint_rule,
                    trainer_commit=trainer_commit,
                    dependency_digest=dependency_digest,
                    runtime_digest=runtime_digest,
                    training_protocol_id=training_protocol_id,
                )
            )
    return tuple(jobs)


__all__ = [
    "AdmittedTrainingRecord",
    "EXECUTION_PURPOSES",
    "ExecutionPurpose",
    "HistoricalBundleReuseAudit",
    "PolicyTrainingAttestation",
    "PolicyTrainingJob",
    "TrainingStatus",
    "admitted_training_records_digest",
    "plan_training_jobs",
    "validate_admitted_training_grid",
]
