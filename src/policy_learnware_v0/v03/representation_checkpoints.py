"""Immutable publication and restart-safe restore for v0.3 R5/R5L fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..hashing import sha256_json
from .artifacts import V03ArtifactReader, V03ArtifactWriter
from .corro_trainers import (
    CorroTrainerAdapter,
    CorroTrainerError,
    FormalCorroTrainerContract,
)
from .representation_ladder import (
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    FittedRepresentation,
    FormalTrainedRepresentationReceipt,
    RepresentationBatch,
    RepresentationManifest,
    RepresentationRestorer,
    TrainingRequest,
    restore_trained_representation,
)
from .representation_plan import (
    RepresentationExecutionPlan,
    RepresentationPlanError,
)
from .source_fit import (
    FormalSourceFitBatch,
    FormalSourceFitSchedule,
    SourceFitProvenanceError,
)
from .signal_matrix import SignalFitJob


TRAINED_REPRESENTATION_CHECKPOINT_SCHEMA = (
    "policy-learnware.v03-trained-representation-checkpoint.v3"
)

DEVELOPMENT_SMOKE_MODE = "DEVELOPMENT_SMOKE"
FORMAL_MODE = "FORMAL"
EXECUTION_MODES = frozenset({DEVELOPMENT_SMOKE_MODE, FORMAL_MODE})


class RepresentationCheckpointError(ValueError):
    """A trained representation checkpoint is not immutable or restorable."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise RepresentationCheckpointError(
            f"{where} must be a lowercase SHA-256 digest"
        )
    try:
        int(value, 16)
    except ValueError as error:
        raise RepresentationCheckpointError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _strict(value: Mapping[str, Any], fields: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise RepresentationCheckpointError(f"{where} must be a mapping")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing or unknown:
        raise RepresentationCheckpointError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


@dataclass(frozen=True)
class TrainedRepresentationCheckpointManifest:
    representation_manifest: RepresentationManifest
    training_request: TrainingRequest
    representation_execution_plan_digest: str
    optimization_digest: str
    execution_mode: str
    formal_source_fit_batch_digest: str | None
    formal_trainer_contract_digest: str | None
    formal_fit_job_digest: str | None
    formal_source_fit_schedule_digest: str | None
    checkpoint_artifact_digest: str
    checkpoint_artifact_name: str
    checkpoint_manifest_digest: str | None = None
    schema: str = TRAINED_REPRESENTATION_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRAINED_REPRESENTATION_CHECKPOINT_SCHEMA:
            raise RepresentationCheckpointError(
                "unsupported trained representation checkpoint schema"
            )
        if not isinstance(self.representation_manifest, RepresentationManifest):
            raise RepresentationCheckpointError(
                "checkpoint manifest requires RepresentationManifest"
            )
        if not isinstance(self.training_request, TrainingRequest):
            raise RepresentationCheckpointError(
                "checkpoint manifest requires TrainingRequest"
            )
        if self.representation_manifest.representation_id not in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            raise RepresentationCheckpointError("only R5/R5L checkpoints are trainable")
        if (
            self.representation_manifest.representation_id
            != self.training_request.representation_id
            or self.representation_manifest.protocol_digest
            != self.training_request.request_digest
        ):
            raise RepresentationCheckpointError(
                "training request differs from representation manifest"
            )
        object.__setattr__(
            self,
            "representation_execution_plan_digest",
            _digest(
                self.representation_execution_plan_digest,
                "representation_execution_plan_digest",
            ),
        )
        object.__setattr__(
            self,
            "optimization_digest",
            _digest(self.optimization_digest, "optimization_digest"),
        )
        if self.execution_mode not in EXECUTION_MODES:
            raise RepresentationCheckpointError(
                "checkpoint execution_mode must be DEVELOPMENT_SMOKE or FORMAL"
            )
        if self.execution_mode == FORMAL_MODE:
            if self.formal_source_fit_batch_digest is None:
                raise RepresentationCheckpointError(
                    "formal checkpoint requires a source-fit batch digest"
                )
            if self.formal_trainer_contract_digest is None:
                raise RepresentationCheckpointError(
                    "formal checkpoint requires a trainer contract digest"
                )
            if (
                self.formal_fit_job_digest is None
                or self.formal_source_fit_schedule_digest is None
            ):
                raise RepresentationCheckpointError(
                    "formal checkpoint requires fit-job schedule authority"
                )
        elif (
            self.formal_source_fit_batch_digest is not None
            or self.formal_trainer_contract_digest is not None
            or self.formal_fit_job_digest is not None
            or self.formal_source_fit_schedule_digest is not None
        ):
            raise RepresentationCheckpointError(
                "development checkpoint cannot carry formal fit authority"
            )
        if self.formal_source_fit_batch_digest is not None:
            object.__setattr__(
                self,
                "formal_source_fit_batch_digest",
                _digest(
                    self.formal_source_fit_batch_digest,
                    "formal_source_fit_batch_digest",
                ),
            )
        if self.formal_trainer_contract_digest is not None:
            object.__setattr__(
                self,
                "formal_trainer_contract_digest",
                _digest(
                    self.formal_trainer_contract_digest,
                    "formal_trainer_contract_digest",
                ),
            )
        if self.formal_fit_job_digest is not None:
            object.__setattr__(
                self,
                "formal_fit_job_digest",
                _digest(self.formal_fit_job_digest, "formal_fit_job_digest"),
            )
        if self.formal_source_fit_schedule_digest is not None:
            object.__setattr__(
                self,
                "formal_source_fit_schedule_digest",
                _digest(
                    self.formal_source_fit_schedule_digest,
                    "formal_source_fit_schedule_digest",
                ),
            )
        artifact_digest = _digest(
            self.checkpoint_artifact_digest, "checkpoint_artifact_digest"
        )
        if artifact_digest != self.representation_manifest.checkpoint_digest:
            raise RepresentationCheckpointError(
                "checkpoint artifact digest differs from representation manifest"
            )
        if (
            not isinstance(self.checkpoint_artifact_name, str)
            or not self.checkpoint_artifact_name
            or self.checkpoint_artifact_name.strip() != self.checkpoint_artifact_name
            or "/" in self.checkpoint_artifact_name
            or "\\" in self.checkpoint_artifact_name
            or self.checkpoint_artifact_name in {".", ".."}
        ):
            raise RepresentationCheckpointError(
                "checkpoint_artifact_name must be one safe filename"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.checkpoint_manifest_digest is None:
            object.__setattr__(self, "checkpoint_manifest_digest", expected)
        elif _digest(
            self.checkpoint_manifest_digest, "checkpoint_manifest_digest"
        ) != expected:
            raise RepresentationCheckpointError("checkpoint manifest digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "representation_manifest": self.representation_manifest.to_dict(),
            "training_request": self.training_request.to_dict(),
            "representation_execution_plan_digest": (
                self.representation_execution_plan_digest
            ),
            "optimization_digest": self.optimization_digest,
            "execution_mode": self.execution_mode,
            "formal_source_fit_batch_digest": self.formal_source_fit_batch_digest,
            "formal_trainer_contract_digest": self.formal_trainer_contract_digest,
            "formal_fit_job_digest": self.formal_fit_job_digest,
            "formal_source_fit_schedule_digest": (
                self.formal_source_fit_schedule_digest
            ),
            "checkpoint_artifact_digest": self.checkpoint_artifact_digest,
            "checkpoint_artifact_name": self.checkpoint_artifact_name,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "checkpoint_manifest_digest": self.checkpoint_manifest_digest,
        }

    def formal_fit_receipt(self) -> FormalTrainedRepresentationReceipt:
        if self.execution_mode != FORMAL_MODE:
            raise RepresentationCheckpointError(
                "development checkpoint cannot issue a formal fit receipt"
            )
        if (
            self.formal_source_fit_batch_digest is None
            or self.formal_trainer_contract_digest is None
            or self.formal_fit_job_digest is None
            or self.formal_source_fit_schedule_digest is None
            or self.checkpoint_manifest_digest is None
        ):
            raise RepresentationCheckpointError(
                "formal checkpoint manifest is missing authority fields"
            )
        return FormalTrainedRepresentationReceipt(
            representation_id=self.representation_manifest.representation_id,
            representation_coordinate_digest=str(
                self.representation_manifest.coordinate_digest
            ),
            checkpoint_artifact_digest=self.checkpoint_artifact_digest,
            checkpoint_manifest_digest=self.checkpoint_manifest_digest,
            training_request_digest=self.training_request.request_digest,
            representation_execution_plan_digest=(
                self.representation_execution_plan_digest
            ),
            formal_source_fit_batch_digest=self.formal_source_fit_batch_digest,
            formal_trainer_contract_digest=self.formal_trainer_contract_digest,
            formal_fit_job_digest=self.formal_fit_job_digest,
            formal_source_fit_schedule_digest=(
                self.formal_source_fit_schedule_digest
            ),
        )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "TrainedRepresentationCheckpointManifest":
        fields = {
            "schema",
            "representation_manifest",
            "training_request",
            "representation_execution_plan_digest",
            "optimization_digest",
            "execution_mode",
            "formal_source_fit_batch_digest",
            "formal_trainer_contract_digest",
            "formal_fit_job_digest",
            "formal_source_fit_schedule_digest",
            "checkpoint_artifact_digest",
            "checkpoint_artifact_name",
            "checkpoint_manifest_digest",
        }
        _strict(value, fields, "trained representation checkpoint manifest")
        return cls(
            representation_manifest=RepresentationManifest.from_dict(
                value["representation_manifest"]
            ),
            training_request=TrainingRequest.from_dict(value["training_request"]),
            representation_execution_plan_digest=value[
                "representation_execution_plan_digest"
            ],
            optimization_digest=value["optimization_digest"],
            execution_mode=value["execution_mode"],
            formal_source_fit_batch_digest=value[
                "formal_source_fit_batch_digest"
            ],
            formal_trainer_contract_digest=value[
                "formal_trainer_contract_digest"
            ],
            formal_fit_job_digest=value["formal_fit_job_digest"],
            formal_source_fit_schedule_digest=value[
                "formal_source_fit_schedule_digest"
            ],
            checkpoint_artifact_digest=value["checkpoint_artifact_digest"],
            checkpoint_artifact_name=value["checkpoint_artifact_name"],
            checkpoint_manifest_digest=value["checkpoint_manifest_digest"],
            schema=value["schema"],
        )


@dataclass(frozen=True)
class TrainedRepresentationCheckpointPublication:
    manifest: TrainedRepresentationCheckpointManifest
    checkpoint_path: Path
    manifest_path: Path
    checkpoint_file_digest: str
    manifest_file_digest: str


def _require_representation_domain(value: Any) -> None:
    if getattr(value, "domain", None) != "representation_controls":
        raise RepresentationCheckpointError(
            "trained representation checkpoints require representation_controls capability"
        )


def _validate_execution_binding(
    *,
    representation_plan: RepresentationExecutionPlan,
    execution_mode: str,
    training_request: TrainingRequest,
    representation_manifest: RepresentationManifest,
    formal_source_fit: FormalSourceFitBatch | None,
    formal_trainer: CorroTrainerAdapter | None,
    formal_fit_job: SignalFitJob | None,
    formal_source_fit_schedule: FormalSourceFitSchedule | None,
) -> tuple[
    str | None,
    FormalCorroTrainerContract | None,
    str | None,
    str | None,
]:
    if not isinstance(representation_plan, RepresentationExecutionPlan):
        raise RepresentationCheckpointError(
            "checkpoint requires a typed RepresentationExecutionPlan"
        )
    if execution_mode not in EXECUTION_MODES:
        raise RepresentationCheckpointError(
            "checkpoint execution_mode must be DEVELOPMENT_SMOKE or FORMAL"
        )
    try:
        representation_plan.validate_training_request(training_request)
        representation_plan.validate_manifest(representation_manifest)
    except RepresentationPlanError as error:
        raise RepresentationCheckpointError(str(error)) from error
    if execution_mode == DEVELOPMENT_SMOKE_MODE:
        if (
            formal_source_fit is not None
            or formal_trainer is not None
            or formal_fit_job is not None
            or formal_source_fit_schedule is not None
        ):
            raise RepresentationCheckpointError(
                "development checkpoint cannot consume formal fit authority"
            )
        return None, None, None, None
    if formal_source_fit is None:
        raise RepresentationCheckpointError(
            "formal checkpoint requires FormalSourceFitBatch"
        )
    if not isinstance(formal_source_fit, FormalSourceFitBatch):
        raise RepresentationCheckpointError(
            "formal_source_fit must be a typed FormalSourceFitBatch"
        )
    try:
        formal_source_fit.require_manifest_binding(representation_manifest)
    except SourceFitProvenanceError as error:
        raise RepresentationCheckpointError(str(error)) from error
    if not isinstance(formal_trainer, CorroTrainerAdapter):
        raise RepresentationCheckpointError(
            "formal checkpoint requires production CorroTrainerAdapter"
        )
    if not isinstance(formal_fit_job, SignalFitJob) or not isinstance(
        formal_source_fit_schedule, FormalSourceFitSchedule
    ):
        raise RepresentationCheckpointError(
            "formal checkpoint requires typed 45-job source-fit schedule"
        )
    try:
        scheduled_authority = formal_source_fit_schedule.authority_for(
            formal_fit_job
        )
    except SourceFitProvenanceError as error:
        raise RepresentationCheckpointError(str(error)) from error
    if (
        scheduled_authority.authority_digest
        != formal_source_fit.authority.authority_digest
        or formal_fit_job.condition_id
        != formal_source_fit.authority.condition_id
        or formal_fit_job.representation_id
        != training_request.representation_id
        or formal_fit_job.seed != training_request.seed
        or formal_fit_job.plan_digest != representation_plan.signal_matrix_digest
    ):
        raise RepresentationCheckpointError(
            "formal checkpoint differs from its frozen fit job/source schedule"
        )
    train_split, validation_split = formal_source_fit.corro_source_splits()
    try:
        contract = formal_trainer.formal_contract(
            request=training_request,
            source_fit_batch_digest=str(formal_source_fit.batch_digest),
            expected_train_split=train_split,
            expected_validation_split=validation_split,
            expected_optimization=representation_plan.optimization,
        )
        representation_plan.validate_formal_trainer_contract(
            contract, training_request
        )
    except (CorroTrainerError, RepresentationPlanError) as error:
        raise RepresentationCheckpointError(str(error)) from error
    return (
        formal_source_fit.batch_digest,
        contract,
        str(formal_fit_job.job_digest),
        str(formal_source_fit_schedule.schedule_digest),
    )


def _validate_formal_fitted_artifact(
    *,
    fitted: FittedRepresentation,
    training_request: TrainingRequest,
    formal_source_fit: FormalSourceFitBatch,
    formal_trainer: CorroTrainerAdapter,
    contract: FormalCorroTrainerContract,
) -> None:
    """Restore once through the production adapter before formal publication."""

    checkpoint_bytes = fitted.checkpoint_bytes
    if checkpoint_bytes is None:
        raise RepresentationCheckpointError("formal fit has no checkpoint bytes")
    if fitted.manifest.implementation_digest != contract.trainer_implementation_digest:
        raise RepresentationCheckpointError(
            "formal fit was not produced by the frozen CORRO backend"
        )
    if fitted.manifest.params_digest != contract.expected_manifest_parameter_digest(
        checkpoint_bytes
    ):
        raise RepresentationCheckpointError(
            "formal fit parameters are not bound to source splits and optimization"
        )
    try:
        restore_trained_representation(
            manifest=fitted.manifest,
            checkpoint_bytes=checkpoint_bytes,
            request=training_request,
            restorer=formal_trainer,
            verification_source=formal_source_fit.training_batch,
            labels=formal_source_fit.training_task_labels,
        )
    except (CorroTrainerError, SourceFitProvenanceError, ValueError) as error:
        raise RepresentationCheckpointError(
            "formal checkpoint failed production-backend restore verification"
        ) from error


def freeze_trained_representation_checkpoint(
    *,
    fitted: FittedRepresentation,
    training_request: TrainingRequest,
    representation_plan: RepresentationExecutionPlan,
    execution_mode: str,
    formal_source_fit: FormalSourceFitBatch | None = None,
    formal_trainer: CorroTrainerAdapter | None = None,
    formal_fit_job: SignalFitJob | None = None,
    formal_source_fit_schedule: FormalSourceFitSchedule | None = None,
    writer: V03ArtifactWriter,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    resume: bool = False,
) -> TrainedRepresentationCheckpointPublication:
    """Atomically publish exact checkpoint bytes and their typed manifest."""

    if not isinstance(fitted, FittedRepresentation):
        raise RepresentationCheckpointError("fitted must be FittedRepresentation")
    if fitted.checkpoint_bytes is None:
        raise RepresentationCheckpointError(
            "trained representation did not retain checkpoint bytes"
        )
    if not isinstance(training_request, TrainingRequest):
        raise RepresentationCheckpointError("training_request must be typed")
    if not isinstance(writer, V03ArtifactWriter):
        raise RepresentationCheckpointError("writer must be V03ArtifactWriter")
    _require_representation_domain(writer)
    (
        source_fit_digest,
        trainer_contract,
        fit_job_digest,
        source_fit_schedule_digest,
    ) = _validate_execution_binding(
        representation_plan=representation_plan,
        execution_mode=execution_mode,
        training_request=training_request,
        representation_manifest=fitted.manifest,
        formal_source_fit=formal_source_fit,
        formal_trainer=formal_trainer,
        formal_fit_job=formal_fit_job,
        formal_source_fit_schedule=formal_source_fit_schedule,
    )
    if execution_mode == FORMAL_MODE:
        assert isinstance(formal_source_fit, FormalSourceFitBatch)
        assert isinstance(formal_trainer, CorroTrainerAdapter)
        assert isinstance(trainer_contract, FormalCorroTrainerContract)
        _validate_formal_fitted_artifact(
            fitted=fitted,
            training_request=training_request,
            formal_source_fit=formal_source_fit,
            formal_trainer=formal_trainer,
            contract=trainer_contract,
        )
    checkpoint = Path(checkpoint_path)
    manifest_destination = Path(manifest_path)
    checkpoint_digest = writer.publish_bytes(
        checkpoint, fitted.checkpoint_bytes, resume=resume
    )
    manifest = TrainedRepresentationCheckpointManifest(
        representation_manifest=fitted.manifest,
        training_request=training_request,
        representation_execution_plan_digest=str(representation_plan.plan_digest),
        optimization_digest=representation_plan.optimization_digest,
        execution_mode=execution_mode,
        formal_source_fit_batch_digest=source_fit_digest,
        formal_trainer_contract_digest=(
            None if trainer_contract is None else trainer_contract.contract_digest
        ),
        formal_fit_job_digest=fit_job_digest,
        formal_source_fit_schedule_digest=source_fit_schedule_digest,
        checkpoint_artifact_digest=checkpoint_digest,
        checkpoint_artifact_name=checkpoint.name,
    )
    manifest_digest = writer.publish_json(
        manifest_destination, manifest.to_dict(), resume=resume
    )
    return TrainedRepresentationCheckpointPublication(
        manifest=manifest,
        checkpoint_path=checkpoint,
        manifest_path=manifest_destination,
        checkpoint_file_digest=checkpoint_digest,
        manifest_file_digest=manifest_digest,
    )


def load_trained_representation_checkpoint(
    *,
    reader: V03ArtifactReader,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    expected_checkpoint_file_digest: str,
    expected_manifest_file_digest: str,
    representation_plan: RepresentationExecutionPlan,
    execution_mode: str,
    formal_source_fit: FormalSourceFitBatch | None = None,
    formal_fit_job: SignalFitJob | None = None,
    formal_source_fit_schedule: FormalSourceFitSchedule | None = None,
    restorer: RepresentationRestorer,
    verification_source: RepresentationBatch,
    labels: np.ndarray,
) -> tuple[TrainedRepresentationCheckpointManifest, FittedRepresentation]:
    """Load exact bytes and rebuild a deterministic transform after restart."""

    if not isinstance(reader, V03ArtifactReader):
        raise RepresentationCheckpointError("reader must be V03ArtifactReader")
    _require_representation_domain(reader)
    raw_manifest = reader.load_json(
        manifest_path, expected_sha256=_digest(
            expected_manifest_file_digest, "expected_manifest_file_digest"
        )
    )
    manifest = TrainedRepresentationCheckpointManifest.from_dict(raw_manifest)
    formal_restorer = restorer if isinstance(restorer, CorroTrainerAdapter) else None
    (
        source_fit_digest,
        trainer_contract,
        fit_job_digest,
        source_fit_schedule_digest,
    ) = _validate_execution_binding(
        representation_plan=representation_plan,
        execution_mode=execution_mode,
        training_request=manifest.training_request,
        representation_manifest=manifest.representation_manifest,
        formal_source_fit=formal_source_fit,
        formal_trainer=formal_restorer,
        formal_fit_job=formal_fit_job,
        formal_source_fit_schedule=formal_source_fit_schedule,
    )
    if manifest.optimization_digest != representation_plan.optimization_digest:
        raise RepresentationCheckpointError(
            "checkpoint optimization digest differs from representation plan"
        )
    if (
        manifest.representation_execution_plan_digest
        != representation_plan.plan_digest
    ):
        raise RepresentationCheckpointError(
            "checkpoint representation plan digest differs from caller"
        )
    if manifest.execution_mode != execution_mode:
        raise RepresentationCheckpointError(
            "checkpoint execution mode differs from caller"
        )
    if manifest.formal_source_fit_batch_digest != source_fit_digest:
        raise RepresentationCheckpointError(
            "checkpoint source-fit batch differs from caller"
        )
    expected_trainer_contract_digest = (
        None if trainer_contract is None else trainer_contract.contract_digest
    )
    if manifest.formal_trainer_contract_digest != expected_trainer_contract_digest:
        raise RepresentationCheckpointError(
            "checkpoint trainer contract differs from caller"
        )
    if (
        manifest.formal_fit_job_digest != fit_job_digest
        or manifest.formal_source_fit_schedule_digest
        != source_fit_schedule_digest
    ):
        raise RepresentationCheckpointError(
            "checkpoint fit-job schedule differs from caller"
        )
    if Path(checkpoint_path).name != manifest.checkpoint_artifact_name:
        raise RepresentationCheckpointError(
            "checkpoint path name differs from typed checkpoint manifest"
        )
    checkpoint_digest = _digest(
        expected_checkpoint_file_digest, "expected_checkpoint_file_digest"
    )
    if checkpoint_digest != manifest.checkpoint_artifact_digest:
        raise RepresentationCheckpointError(
            "expected checkpoint digest differs from typed manifest"
        )
    checkpoint_bytes = reader.load_bytes(
        checkpoint_path, expected_sha256=checkpoint_digest
    )
    fitted = restore_trained_representation(
        manifest=manifest.representation_manifest,
        checkpoint_bytes=checkpoint_bytes,
        request=manifest.training_request,
        restorer=restorer,
        verification_source=verification_source,
        labels=labels,
    )
    return manifest, fitted


__all__ = [
    "DEVELOPMENT_SMOKE_MODE",
    "FORMAL_MODE",
    "TRAINED_REPRESENTATION_CHECKPOINT_SCHEMA",
    "RepresentationCheckpointError",
    "TrainedRepresentationCheckpointManifest",
    "TrainedRepresentationCheckpointPublication",
    "freeze_trained_representation_checkpoint",
    "load_trained_representation_checkpoint",
]
