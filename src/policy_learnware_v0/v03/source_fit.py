"""Formal source-only provenance for data-fitted v0.3 representations.

``RepresentationBatch(role="SOURCE_FIT")`` remains a convenient development
value object, but the role string alone is not authority for a formal fit.  A
formal batch in this module can only be assembled from canonical feature banks
registered as ``source_representation_train`` and
``source_representation_validation`` in one :class:`DataRoleManifest`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from .condition_plan import ConditionExecutionPlan, ConditionPlanError
from .data_roles import (
    DataRoleManifest,
    DataRoleRecord,
    assert_process_can_read,
)
from .representation_ladder import (
    R2_SOURCE_PCA_WHITEN,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    RepresentationBatch,
    RepresentationManifest,
)
from .signal_matrix import (
    SignalFitJob,
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from .signal_runtime import FormalFeatureBank


FORMAL_SOURCE_FIT_BANK_SCHEMA = "policy-learnware.v03-formal-source-fit-bank.v0"
FORMAL_SOURCE_FIT_AUTHORITY_SCHEMA = (
    "policy-learnware.v03-formal-source-fit-authority.v1"
)
FORMAL_SOURCE_FIT_BATCH_SCHEMA = "policy-learnware.v03-formal-source-fit-batch.v0"
FORMAL_SOURCE_FIT_SCHEDULE_SCHEMA = "policy-learnware.v03-formal-source-fit-schedule.v0"

SOURCE_REPRESENTATION_TRAIN = "source_representation_train"
SOURCE_REPRESENTATION_VALIDATION = "source_representation_validation"
DATA_FITTED_REPRESENTATION_IDS = frozenset(
    {
        R2_SOURCE_PCA_WHITEN,
        R5_VIEW_SPECIFIC_CORRO_REFIT,
        R5L_SUPERVISED_LINEAR,
    }
)


class SourceFitProvenanceError(ValueError):
    """A source-fit role, split, canonical coordinate, or digest is invalid."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise SourceFitProvenanceError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SourceFitProvenanceError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SourceFitProvenanceError(f"{where} must be a non-empty canonical string")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise SourceFitProvenanceError(f"{where} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise SourceFitProvenanceError(f"{where} must be a positive integer")
    return result


def _task_identity_digest(bank: FormalFeatureBank) -> str:
    identity = bank.identity
    return sha256_json(
        {
            "schema": "policy-learnware.v03-source-fit-task-identity.v0",
            "task_private_id": identity.task_private_id,
            "embodiment_id": identity.embodiment_id,
            "abi_contract_id": identity.abi_contract_id,
            "goal_contract_id": identity.goal_contract_id,
            "dynamics_context_id": identity.dynamics_context_id,
            "equivalence_class_id": identity.equivalence_class_id,
        }
    )


def _feature_arrays_digest(bank: FormalFeatureBank) -> str:
    return sha256_ndarrays(
        {"values": bank.values, "episode_offsets": bank.episode_offsets}
    )


@dataclass(frozen=True)
class FormalSourceFitBankBinding:
    """Persistable binding for one task in one source fit split."""

    role: str
    task_private_id: str
    bank_id: str
    data_role_record_digest: str
    receipt_digest: str
    raw_dataset_digest: str
    native_bank_digest: str
    canonical_transition_digest: str
    feature_bank_digest: str
    feature_arrays_digest: str
    task_identity_digest: str
    row_count: int
    binding_digest: str | None = None
    schema: str = FORMAL_SOURCE_FIT_BANK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SOURCE_FIT_BANK_SCHEMA:
            raise SourceFitProvenanceError("unsupported formal source-fit bank schema")
        if self.role not in {
            SOURCE_REPRESENTATION_TRAIN,
            SOURCE_REPRESENTATION_VALIDATION,
        }:
            raise SourceFitProvenanceError("formal source-fit bank has the wrong role")
        for name in ("task_private_id", "bank_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "data_role_record_digest",
            "receipt_digest",
            "raw_dataset_digest",
            "native_bank_digest",
            "canonical_transition_digest",
            "feature_bank_digest",
            "feature_arrays_digest",
            "task_identity_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "row_count", _positive_int(self.row_count, "row_count"))
        expected = sha256_json(self._payload_without_digest())
        if self.binding_digest is None:
            object.__setattr__(self, "binding_digest", expected)
        elif _digest(self.binding_digest, "binding_digest") != expected:
            raise SourceFitProvenanceError(
                "formal source-fit bank digest does not match contents"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "role": self.role,
            "task_private_id": self.task_private_id,
            "bank_id": self.bank_id,
            "data_role_record_digest": self.data_role_record_digest,
            "receipt_digest": self.receipt_digest,
            "raw_dataset_digest": self.raw_dataset_digest,
            "native_bank_digest": self.native_bank_digest,
            "canonical_transition_digest": self.canonical_transition_digest,
            "feature_bank_digest": self.feature_bank_digest,
            "feature_arrays_digest": self.feature_arrays_digest,
            "task_identity_digest": self.task_identity_digest,
            "row_count": self.row_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "binding_digest": self.binding_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalSourceFitBankBinding":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise SourceFitProvenanceError("invalid formal source-fit bank fields")
        return cls(**{name: value[name] for name in fields})


def _cross_split_disjoint(
    train: Sequence[FormalSourceFitBankBinding],
    validation: Sequence[FormalSourceFitBankBinding],
) -> None:
    fields = (
        "bank_id",
        "raw_dataset_digest",
        "native_bank_digest",
        "canonical_transition_digest",
        "receipt_digest",
        "feature_bank_digest",
        "feature_arrays_digest",
    )
    for field_name in fields:
        left = {getattr(item, field_name) for item in train}
        right = {getattr(item, field_name) for item in validation}
        if left & right:
            raise SourceFitProvenanceError(
                "source train/validation physical or digest overlap detected "
                f"for {field_name}"
            )


def _membership_binding_payload(
    binding: FormalSourceFitBankBinding,
) -> dict[str, Any]:
    """Return the condition-independent source-row identity of one bank.

    ``feature_bank_digest`` and ``feature_arrays_digest`` are intentionally not
    part of this payload: every condition transforms the same admitted source
    rows into different features.  The canonical bank and receipt commitments
    below are the physical-row identity shared by all 45 fit jobs.
    """

    if not isinstance(binding, FormalSourceFitBankBinding):
        raise SourceFitProvenanceError(
            "source membership requires typed formal bank bindings"
        )
    return {
        "role": binding.role,
        "task_private_id": binding.task_private_id,
        "bank_id": binding.bank_id,
        "data_role_record_digest": binding.data_role_record_digest,
        "receipt_digest": binding.receipt_digest,
        "raw_dataset_digest": binding.raw_dataset_digest,
        "native_bank_digest": binding.native_bank_digest,
        "canonical_transition_digest": binding.canonical_transition_digest,
        "task_identity_digest": binding.task_identity_digest,
        "row_count": binding.row_count,
    }


def _source_membership_digest_from_bindings(
    *,
    data_role_manifest_digest: str,
    train_role_digest: str,
    validation_role_digest: str,
    split_nonce_digest: str,
    train_bindings: Sequence[FormalSourceFitBankBinding],
    validation_bindings: Sequence[FormalSourceFitBankBinding],
) -> str:
    """Derive the source membership solely from authority-owned commitments."""

    train = tuple(sorted(train_bindings, key=lambda item: item.task_private_id))
    validation = tuple(
        sorted(validation_bindings, key=lambda item: item.task_private_id)
    )
    return sha256_json(
        {
            "schema": "policy-learnware.v03-formal-source-membership.v0",
            "data_role_manifest_digest": _digest(
                data_role_manifest_digest, "data_role_manifest_digest"
            ),
            "train_role_digest": _digest(train_role_digest, "train_role_digest"),
            "validation_role_digest": _digest(
                validation_role_digest, "validation_role_digest"
            ),
            "split_nonce_digest": _digest(
                split_nonce_digest, "split_nonce_digest"
            ),
            "train": [_membership_binding_payload(item) for item in train],
            "validation": [
                _membership_binding_payload(item) for item in validation
            ],
        }
    )


@dataclass(frozen=True)
class FormalSourceFitAuthority:
    """Serializable authority proving one condition-specific source split."""

    data_role_manifest_digest: str
    train_role_digest: str
    validation_role_digest: str
    split_nonce_digest: str
    condition_id: str
    condition_transform_digest: str
    condition_execution_plan_digest: str
    source_membership_digest: str
    canonicalizer_digest: str
    native_shape_registry_digest: str
    normalizer_digest: str
    measurement_protocol_digest: str
    input_dim: int
    task_private_ids: tuple[str, ...]
    train_bindings: tuple[FormalSourceFitBankBinding, ...]
    validation_bindings: tuple[FormalSourceFitBankBinding, ...]
    authority_digest: str | None = None
    schema: str = FORMAL_SOURCE_FIT_AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SOURCE_FIT_AUTHORITY_SCHEMA:
            raise SourceFitProvenanceError("unsupported formal source-fit authority schema")
        for name in (
            "data_role_manifest_digest",
            "train_role_digest",
            "validation_role_digest",
            "split_nonce_digest",
            "condition_transform_digest",
            "condition_execution_plan_digest",
            "source_membership_digest",
            "canonicalizer_digest",
            "native_shape_registry_digest",
            "normalizer_digest",
            "measurement_protocol_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "condition_id", _nonempty(self.condition_id, "condition_id"))
        object.__setattr__(self, "input_dim", _positive_int(self.input_dim, "input_dim"))
        task_ids = tuple(sorted(_nonempty(item, "task_private_ids[]") for item in self.task_private_ids))
        if len(task_ids) < 2 or len(set(task_ids)) != len(task_ids):
            raise SourceFitProvenanceError(
                "formal source fit requires at least two unique source tasks"
            )
        raw_train = tuple(self.train_bindings)
        raw_validation = tuple(self.validation_bindings)
        if not all(
            isinstance(item, FormalSourceFitBankBinding)
            for item in (*raw_train, *raw_validation)
        ):
            raise SourceFitProvenanceError("formal source fit requires typed bank bindings")
        train = tuple(sorted(raw_train, key=lambda item: item.task_private_id))
        validation = tuple(
            sorted(raw_validation, key=lambda item: item.task_private_id)
        )
        if any(item.role != SOURCE_REPRESENTATION_TRAIN for item in train):
            raise SourceFitProvenanceError("training binding has the wrong data role")
        if any(item.role != SOURCE_REPRESENTATION_VALIDATION for item in validation):
            raise SourceFitProvenanceError("validation binding has the wrong data role")
        if tuple(item.task_private_id for item in train) != task_ids or tuple(
            item.task_private_id for item in validation
        ) != task_ids:
            raise SourceFitProvenanceError(
                "source train/validation task sets must match the authority"
            )
        for task_id, train_item, validation_item in zip(
            task_ids, train, validation, strict=True
        ):
            if train_item.task_identity_digest != validation_item.task_identity_digest:
                raise SourceFitProvenanceError(
                    f"{task_id}: source train/validation task identity differs"
                )
        _cross_split_disjoint(train, validation)
        object.__setattr__(self, "task_private_ids", task_ids)
        object.__setattr__(self, "train_bindings", train)
        object.__setattr__(self, "validation_bindings", validation)
        expected_membership = _source_membership_digest_from_bindings(
            data_role_manifest_digest=self.data_role_manifest_digest,
            train_role_digest=self.train_role_digest,
            validation_role_digest=self.validation_role_digest,
            split_nonce_digest=self.split_nonce_digest,
            train_bindings=train,
            validation_bindings=validation,
        )
        if self.source_membership_digest != expected_membership:
            raise SourceFitProvenanceError(
                "source_membership_digest does not match the admitted source bindings"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.authority_digest is None:
            object.__setattr__(self, "authority_digest", expected)
        elif _digest(self.authority_digest, "authority_digest") != expected:
            raise SourceFitProvenanceError(
                "formal source-fit authority digest does not match contents"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "data_role_manifest_digest": self.data_role_manifest_digest,
            "train_role_digest": self.train_role_digest,
            "validation_role_digest": self.validation_role_digest,
            "split_nonce_digest": self.split_nonce_digest,
            "condition_id": self.condition_id,
            "condition_transform_digest": self.condition_transform_digest,
            "condition_execution_plan_digest": self.condition_execution_plan_digest,
            "source_membership_digest": self.source_membership_digest,
            "canonicalizer_digest": self.canonicalizer_digest,
            "native_shape_registry_digest": self.native_shape_registry_digest,
            "normalizer_digest": self.normalizer_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "input_dim": self.input_dim,
            "task_private_ids": list(self.task_private_ids),
            "train_bindings": [item.to_dict() for item in self.train_bindings],
            "validation_bindings": [item.to_dict() for item in self.validation_bindings],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "authority_digest": self.authority_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalSourceFitAuthority":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise SourceFitProvenanceError("invalid formal source-fit authority fields")
        return cls(
            **{
                name: (
                    tuple(FormalSourceFitBankBinding.from_dict(item) for item in value[name])
                    if name in {"train_bindings", "validation_bindings"}
                    else tuple(value[name])
                    if name == "task_private_ids"
                    else value[name]
                )
                for name in fields
            }
        )


@dataclass(frozen=True)
class FormalSourceFitSchedule:
    """Exact 45-job join to condition authorities and one source membership.

    This is the schedule-level guard that prevents each worker from quietly
    rebuilding its own source split.  R5/R5L seeds may change model parameters,
    and conditions may change feature transforms, but all 45 jobs must consume
    the same canonical source rows and the exact frozen condition transform.
    """

    condition_plan: ConditionExecutionPlan
    job_authorities: Mapping[str, FormalSourceFitAuthority]
    source_membership_digest: str | None = None
    schedule_digest: str | None = None
    schema: str = FORMAL_SOURCE_FIT_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SOURCE_FIT_SCHEDULE_SCHEMA:
            raise SourceFitProvenanceError(
                "unsupported formal source-fit schedule schema"
            )
        if not isinstance(self.condition_plan, ConditionExecutionPlan):
            raise SourceFitProvenanceError(
                "formal source-fit schedule requires ConditionExecutionPlan"
            )
        condition_plan_digest = str(self.condition_plan.plan_digest)
        expected_jobs = build_optimization_fit_jobs(build_signal_matrix_plan())
        expected_by_id = {item.job_id: item for item in expected_jobs}
        supplied = dict(sorted(self.job_authorities.items()))
        if set(supplied) != set(expected_by_id) or not all(
            isinstance(item, FormalSourceFitAuthority)
            for item in supplied.values()
        ):
            raise SourceFitProvenanceError(
                "formal source-fit schedule must cover the exact 45 fit jobs"
            )
        memberships = set()
        for job_id, authority in supplied.items():
            job = expected_by_id[job_id]
            if authority.condition_id != job.condition_id:
                raise SourceFitProvenanceError(
                    f"{job_id}: source-fit condition differs from fit job"
                )
            if authority.condition_execution_plan_digest != condition_plan_digest:
                raise SourceFitProvenanceError(
                    f"{job_id}: source-fit condition plan differs from schedule"
                )
            if authority.condition_transform_digest != self.condition_plan.transform_digest(
                job.condition_id
            ):
                raise SourceFitProvenanceError(
                    f"{job_id}: source-fit transform differs from schedule"
                )
            memberships.add(authority.source_membership_digest)
        if len(memberships) != 1:
            raise SourceFitProvenanceError(
                "formal fit jobs do not share one frozen source membership"
            )
        membership_digest = next(iter(memberships))
        if self.source_membership_digest is None:
            object.__setattr__(self, "source_membership_digest", membership_digest)
        elif _digest(
            self.source_membership_digest, "source_membership_digest"
        ) != membership_digest:
            raise SourceFitProvenanceError(
                "formal source-fit schedule membership digest mismatch"
            )
        object.__setattr__(self, "job_authorities", MappingProxyType(supplied))
        expected_digest = sha256_json(self._payload_without_digest())
        if self.schedule_digest is None:
            object.__setattr__(self, "schedule_digest", expected_digest)
        elif _digest(self.schedule_digest, "schedule_digest") != expected_digest:
            raise SourceFitProvenanceError(
                "formal source-fit schedule digest mismatch"
            )

    @classmethod
    def from_condition_authorities(
        cls,
        *,
        condition_plan: ConditionExecutionPlan,
        authorities: Mapping[str, FormalSourceFitAuthority],
    ) -> "FormalSourceFitSchedule":
        if not isinstance(condition_plan, ConditionExecutionPlan):
            raise SourceFitProvenanceError(
                "formal source-fit schedule requires ConditionExecutionPlan"
            )
        values = dict(authorities)
        jobs = build_optimization_fit_jobs(build_signal_matrix_plan())
        required_conditions = {item.condition_id for item in jobs}
        if set(values) != required_conditions or not all(
            isinstance(item, FormalSourceFitAuthority) for item in values.values()
        ):
            raise SourceFitProvenanceError(
                "condition authorities must exactly cover the 45-job condition set"
            )
        for condition_id, authority in values.items():
            if (
                authority.condition_id != condition_id
                or authority.condition_execution_plan_digest
                != condition_plan.plan_digest
                or authority.condition_transform_digest
                != condition_plan.transform_digest(condition_id)
            ):
                raise SourceFitProvenanceError(
                    f"{condition_id}: authority differs from condition freeze"
                )
        return cls(
            condition_plan=condition_plan,
            job_authorities={
                job.job_id: values[job.condition_id] for job in jobs
            },
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "condition_plan_digest": self.condition_plan.plan_digest,
            "source_membership_digest": self.source_membership_digest,
            "job_authority_digests": {
                job_id: authority.authority_digest
                for job_id, authority in self.job_authorities.items()
            },
        }

    def authority_for(self, job: SignalFitJob) -> FormalSourceFitAuthority:
        if not isinstance(job, SignalFitJob):
            raise SourceFitProvenanceError("fit schedule lookup requires SignalFitJob")
        try:
            authority = self.job_authorities[job.job_id]
        except KeyError as error:
            raise SourceFitProvenanceError("fit job is absent from source schedule") from error
        canonical = {
            item.job_id: item
            for item in build_optimization_fit_jobs(build_signal_matrix_plan())
        }[job.job_id]
        if job.to_dict() != canonical.to_dict():
            raise SourceFitProvenanceError("fit job differs from frozen schedule")
        return authority


def _matched_record(
    bank: FormalFeatureBank,
    records: Sequence[DataRoleRecord],
    role: str,
) -> DataRoleRecord:
    matches = tuple(
        item
        for item in records
        if item.dataset_digest == bank.receipt.raw_dataset_digest
        and bank.receipt.task_private_id in item.task_private_ids
    )
    if len(matches) != 1:
        raise SourceFitProvenanceError(
            f"{role} bank must match exactly one data-role record by task and dataset digest"
        )
    return matches[0]


def _bindings(
    banks: Sequence[FormalFeatureBank],
    records: Sequence[DataRoleRecord],
    role: str,
) -> tuple[FormalSourceFitBankBinding, ...]:
    values = tuple(banks)
    if not values or not all(isinstance(item, FormalFeatureBank) for item in values):
        raise SourceFitProvenanceError("formal source fit requires typed feature banks")
    if any(item.receipt.data_role != role for item in values):
        raise SourceFitProvenanceError(f"formal source-fit bank must have role={role}")
    if len({item.receipt.task_private_id for item in values}) != len(values):
        raise SourceFitProvenanceError("one source split may contain only one bank per task")
    bindings = []
    used_record_digests = set()
    for bank in values:
        record = _matched_record(bank, records, role)
        used_record_digests.add(record.role_record_digest)
        bindings.append(
            FormalSourceFitBankBinding(
                role=role,
                task_private_id=bank.receipt.task_private_id,
                bank_id=bank.receipt.bank_id,
                data_role_record_digest=record.role_record_digest,
                receipt_digest=str(bank.receipt.receipt_digest),
                raw_dataset_digest=bank.receipt.raw_dataset_digest,
                native_bank_digest=bank.receipt.native_bank_digest,
                canonical_transition_digest=bank.receipt.canonical_transition_digest,
                feature_bank_digest=str(bank.feature_bank_digest),
                feature_arrays_digest=_feature_arrays_digest(bank),
                task_identity_digest=_task_identity_digest(bank),
                row_count=int(bank.values.shape[0]),
            )
        )
    if used_record_digests != {item.role_record_digest for item in records}:
        raise SourceFitProvenanceError(
            f"formal {role} banks do not cover the exact data-role records"
        )
    registered_tasks = {task for item in records for task in item.task_private_ids}
    if {item.task_private_id for item in bindings} != registered_tasks:
        raise SourceFitProvenanceError(
            f"formal {role} banks do not cover the exact registered task set"
        )
    return tuple(bindings)


def _membership_digest(
    *,
    manifest: DataRoleManifest,
    split_nonce_digest: str,
    train_bindings: Sequence[FormalSourceFitBankBinding],
    validation_bindings: Sequence[FormalSourceFitBankBinding],
) -> str:
    """Condition-independent identity of the rows admitted to every fit job.

    Feature values and view-transform digests are deliberately excluded: two
    condition jobs must transform the same canonical source rows, but their
    resulting features are expected to differ.  The resulting digest can be
    frozen once by an orchestrator and supplied to every condition-specific
    fit to prevent silent source-membership drift across jobs.
    """

    return _source_membership_digest_from_bindings(
        data_role_manifest_digest=manifest.manifest_digest,
        train_role_digest=manifest.role_digest(SOURCE_REPRESENTATION_TRAIN),
        validation_role_digest=manifest.role_digest(
            SOURCE_REPRESENTATION_VALIDATION
        ),
        split_nonce_digest=split_nonce_digest,
        train_bindings=train_bindings,
        validation_bindings=validation_bindings,
    )


@dataclass(frozen=True)
class FormalSourceFitBatch:
    """Runtime train/validation matrices backed by a formal authority."""

    authority: FormalSourceFitAuthority
    train_feature_banks: tuple[FormalFeatureBank, ...] = field(repr=False)
    validation_feature_banks: tuple[FormalFeatureBank, ...] = field(repr=False)
    training_batch: RepresentationBatch = field(init=False)
    validation_batch: RepresentationBatch = field(init=False)
    training_task_labels: np.ndarray = field(init=False, repr=False, compare=False)
    validation_task_labels: np.ndarray = field(init=False, repr=False, compare=False)
    batch_digest: str = field(init=False)
    schema: str = FORMAL_SOURCE_FIT_BATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SOURCE_FIT_BATCH_SCHEMA:
            raise SourceFitProvenanceError("unsupported formal source-fit batch schema")
        if not isinstance(self.authority, FormalSourceFitAuthority):
            raise SourceFitProvenanceError("formal source-fit batch requires typed authority")
        train = tuple(sorted(self.train_feature_banks, key=lambda item: item.receipt.task_private_id))
        validation = tuple(
            sorted(
                self.validation_feature_banks,
                key=lambda item: item.receipt.task_private_id,
            )
        )
        if not all(isinstance(item, FormalFeatureBank) for item in (*train, *validation)):
            raise SourceFitProvenanceError("formal source-fit batch requires feature banks")
        if tuple(item.receipt.task_private_id for item in train) != self.authority.task_private_ids:
            raise SourceFitProvenanceError(
                "formal training banks must cover every authority task exactly once"
            )
        if tuple(
            item.receipt.task_private_id for item in validation
        ) != self.authority.task_private_ids:
            raise SourceFitProvenanceError(
                "formal validation banks must cover every authority task exactly once"
            )
        expected_train = {
            item.task_private_id: item.feature_bank_digest
            for item in self.authority.train_bindings
        }
        expected_validation = {
            item.task_private_id: item.feature_bank_digest
            for item in self.authority.validation_bindings
        }
        observed_train = {
            item.receipt.task_private_id: item.feature_bank_digest for item in train
        }
        observed_validation = {
            item.receipt.task_private_id: item.feature_bank_digest for item in validation
        }
        if observed_train != expected_train or observed_validation != expected_validation:
            raise SourceFitProvenanceError(
                "formal source-fit feature banks differ from the frozen authority"
            )
        train_values = np.ascontiguousarray(
            np.concatenate([item.values for item in train], axis=0), dtype=np.float64
        )
        validation_values = np.ascontiguousarray(
            np.concatenate([item.values for item in validation], axis=0),
            dtype=np.float64,
        )
        train_dataset_digest = sha256_json(
            {
                "schema": "policy-learnware.v03-formal-source-fit-dataset.v0",
                "authority_digest": self.authority.authority_digest,
                "role": SOURCE_REPRESENTATION_TRAIN,
                "binding_digests": [
                    item.binding_digest for item in self.authority.train_bindings
                ],
            }
        )
        validation_dataset_digest = sha256_json(
            {
                "schema": "policy-learnware.v03-formal-source-fit-dataset.v0",
                "authority_digest": self.authority.authority_digest,
                "role": SOURCE_REPRESENTATION_VALIDATION,
                "binding_digests": [
                    item.binding_digest for item in self.authority.validation_bindings
                ],
            }
        )
        training_batch = RepresentationBatch(
            train_values, train_dataset_digest, "SOURCE_FIT"
        )
        validation_batch = RepresentationBatch(
            validation_values, validation_dataset_digest, "SOURCE_FIT"
        )
        task_index = {
            task_id: index for index, task_id in enumerate(self.authority.task_private_ids)
        }
        training_labels = np.concatenate(
            [
                np.full(item.values.shape[0], task_index[item.receipt.task_private_id], dtype=np.int64)
                for item in train
            ]
        )
        validation_labels = np.concatenate(
            [
                np.full(item.values.shape[0], task_index[item.receipt.task_private_id], dtype=np.int64)
                for item in validation
            ]
        )
        for array in (training_labels, validation_labels):
            array.setflags(write=False)
        object.__setattr__(self, "train_feature_banks", train)
        object.__setattr__(self, "validation_feature_banks", validation)
        object.__setattr__(self, "training_batch", training_batch)
        object.__setattr__(self, "validation_batch", validation_batch)
        object.__setattr__(self, "training_task_labels", training_labels)
        object.__setattr__(self, "validation_task_labels", validation_labels)
        object.__setattr__(self, "batch_digest", sha256_json(self._payload()))

    @property
    def training_label_digest(self) -> str:
        return sha256_ndarrays({"labels": self.training_task_labels})

    @property
    def validation_label_digest(self) -> str:
        return sha256_ndarrays({"labels": self.validation_task_labels})

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority_digest": self.authority.authority_digest,
            "training_batch_digest": self.training_batch.batch_digest,
            "validation_batch_digest": self.validation_batch.batch_digest,
            "training_label_digest": self.training_label_digest,
            "validation_label_digest": self.validation_label_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "batch_digest": self.batch_digest}

    def expected_manifest_source_fit_digest(
        self, manifest: RepresentationManifest
    ) -> str:
        if not isinstance(manifest, RepresentationManifest):
            raise SourceFitProvenanceError("source-fit check requires RepresentationManifest")
        if manifest.representation_id == R2_SOURCE_PCA_WHITEN:
            return self.training_batch.batch_digest
        if manifest.representation_id in {
            R5_VIEW_SPECIFIC_CORRO_REFIT,
            R5L_SUPERVISED_LINEAR,
        }:
            return sha256_json(
                {
                    "schema": "policy-learnware.v03-supervised-source-fit.v0",
                    "source_batch_digest": self.training_batch.batch_digest,
                    "label_digest": self.training_label_digest,
                    "training_request_digest": manifest.protocol_digest,
                }
            )
        raise SourceFitProvenanceError(
            "formal source-fit check applies only to data-fitted R2/R5/R5L"
        )

    def require_manifest_binding(self, manifest: RepresentationManifest) -> None:
        if manifest.input_dim != self.authority.input_dim:
            raise SourceFitProvenanceError(
                "representation input width differs from formal source-fit authority"
            )
        expected = self.expected_manifest_source_fit_digest(manifest)
        if manifest.source_fit_digest != expected:
            raise SourceFitProvenanceError(
                "representation source_fit_digest is not bound to the formal source-fit batch"
            )

    def require_condition_plan(self, condition_plan: ConditionExecutionPlan) -> None:
        """Rejoin a persisted source-fit batch to the exact transform freeze."""

        if not isinstance(condition_plan, ConditionExecutionPlan):
            raise SourceFitProvenanceError(
                "source-fit condition check requires ConditionExecutionPlan"
            )
        if (
            self.authority.condition_execution_plan_digest
            != condition_plan.plan_digest
        ):
            raise SourceFitProvenanceError(
                "source-fit condition plan differs from the frozen authority"
            )
        try:
            expected_transform = condition_plan.transform_digest(
                self.authority.condition_id
            )
        except ConditionPlanError as error:
            raise SourceFitProvenanceError(str(error)) from error
        if expected_transform != self.authority.condition_transform_digest:
            raise SourceFitProvenanceError(
                "source-fit condition transform differs from the frozen authority"
            )
        try:
            for bank in (
                *self.train_feature_banks,
                *self.validation_feature_banks,
            ):
                condition_plan.validate_feature_bank(bank)
        except ConditionPlanError as error:
            raise SourceFitProvenanceError(str(error)) from error

    def corro_source_splits(self) -> tuple[Any, Any]:
        """Build the exact episode-aware source splits consumed by R5/R5L."""

        from .corro_trainers import CorroSourceSplit, CorroTaskDataset

        def build(role: str, banks: tuple[FormalFeatureBank, ...]) -> Any:
            return CorroSourceSplit(
                role=role,
                tasks=tuple(
                    CorroTaskDataset(
                        task_id=item.receipt.task_private_id,
                        packed=item.values,
                        episode_offsets=item.episode_offsets,
                    )
                    for item in banks
                ),
            )

        return (
            build(SOURCE_REPRESENTATION_TRAIN, self.train_feature_banks),
            build(
                SOURCE_REPRESENTATION_VALIDATION,
                self.validation_feature_banks,
            ),
        )


def build_formal_source_fit_batch(
    manifest: DataRoleManifest,
    *,
    train_feature_banks: Sequence[FormalFeatureBank],
    validation_feature_banks: Sequence[FormalFeatureBank],
    condition_plan: ConditionExecutionPlan,
    expected_source_membership_digest: str | None = None,
) -> FormalSourceFitBatch:
    """Join the data-role manifest to canonical train/validation feature banks."""

    if not isinstance(manifest, DataRoleManifest):
        raise SourceFitProvenanceError("formal source fit requires DataRoleManifest")
    if not isinstance(condition_plan, ConditionExecutionPlan):
        raise SourceFitProvenanceError(
            "formal source fit requires a typed ConditionExecutionPlan"
        )
    assert_process_can_read("representation_trainer", SOURCE_REPRESENTATION_TRAIN)
    assert_process_can_read("representation_trainer", SOURCE_REPRESENTATION_VALIDATION)
    train_records = manifest.records_for(SOURCE_REPRESENTATION_TRAIN)
    validation_records = manifest.records_for(SOURCE_REPRESENTATION_VALIDATION)
    if not train_records or not validation_records:
        raise SourceFitProvenanceError(
            "formal source fit requires registered train and validation records"
        )
    split_nonces = {
        item.split_nonce_digest for item in (*train_records, *validation_records)
    }
    if len(split_nonces) != 1:
        raise SourceFitProvenanceError(
            "source train/validation records must share one split nonce"
        )
    train = tuple(train_feature_banks)
    validation = tuple(validation_feature_banks)
    train_bindings = _bindings(
        train, train_records, SOURCE_REPRESENTATION_TRAIN
    )
    validation_bindings = _bindings(
        validation, validation_records, SOURCE_REPRESENTATION_VALIDATION
    )
    all_banks = (*train, *validation)
    try:
        for bank in all_banks:
            condition_plan.validate_feature_bank(bank)
    except ConditionPlanError as error:
        raise SourceFitProvenanceError(str(error)) from error

    def one(values: set[Any], where: str) -> Any:
        if len(values) != 1:
            raise SourceFitProvenanceError(
                f"source train/validation must share one {where}"
            )
        return next(iter(values))

    task_ids = tuple(sorted(item.receipt.task_private_id for item in train))
    if task_ids != tuple(sorted(item.receipt.task_private_id for item in validation)):
        raise SourceFitProvenanceError("source train/validation task sets must match")
    condition_id = one({item.condition_id for item in all_banks}, "condition")
    condition_transform_digest = one(
        {item.condition_transform_digest for item in all_banks},
        "condition transform",
    )
    canonicalizer_digest = one(
        {item.receipt.canonicalizer_digest for item in all_banks}, "canonicalizer"
    )
    registry_digest = one(
        {item.receipt.native_shape_registry_digest for item in all_banks},
        "native-shape registry",
    )
    normalizer_digest = one(
        {item.receipt.normalizer_digest for item in all_banks}, "normalizer"
    )
    measurement_digest = one(
        {item.identity.measurement_protocol_digest for item in all_banks},
        "measurement protocol",
    )
    input_dim = one({int(item.values.shape[1]) for item in all_banks}, "input width")
    source_membership_digest = _membership_digest(
        manifest=manifest,
        split_nonce_digest=next(iter(split_nonces)),
        train_bindings=train_bindings,
        validation_bindings=validation_bindings,
    )
    if expected_source_membership_digest is not None and _digest(
        expected_source_membership_digest, "expected_source_membership_digest"
    ) != source_membership_digest:
        raise SourceFitProvenanceError(
            "formal source rows differ from the frozen cross-job membership"
        )
    authority = FormalSourceFitAuthority(
        data_role_manifest_digest=manifest.manifest_digest,
        train_role_digest=manifest.role_digest(SOURCE_REPRESENTATION_TRAIN),
        validation_role_digest=manifest.role_digest(
            SOURCE_REPRESENTATION_VALIDATION
        ),
        split_nonce_digest=next(iter(split_nonces)),
        condition_id=condition_id,
        condition_transform_digest=condition_transform_digest,
        condition_execution_plan_digest=str(condition_plan.plan_digest),
        source_membership_digest=source_membership_digest,
        canonicalizer_digest=canonicalizer_digest,
        native_shape_registry_digest=registry_digest,
        normalizer_digest=normalizer_digest,
        measurement_protocol_digest=measurement_digest,
        input_dim=input_dim,
        task_private_ids=task_ids,
        train_bindings=train_bindings,
        validation_bindings=validation_bindings,
    )
    return FormalSourceFitBatch(authority, train, validation)


def development_source_fit_batch(
    values: Any, *, dataset_digest: str
) -> RepresentationBatch:
    """Explicit non-formal helper retained for unit/development smoke tests."""

    return RepresentationBatch(values, dataset_digest, "SOURCE_FIT")


__all__ = [
    "DATA_FITTED_REPRESENTATION_IDS",
    "FORMAL_SOURCE_FIT_AUTHORITY_SCHEMA",
    "FORMAL_SOURCE_FIT_BANK_SCHEMA",
    "FORMAL_SOURCE_FIT_BATCH_SCHEMA",
    "FORMAL_SOURCE_FIT_SCHEDULE_SCHEMA",
    "FormalSourceFitAuthority",
    "FormalSourceFitBankBinding",
    "FormalSourceFitBatch",
    "FormalSourceFitSchedule",
    "SOURCE_REPRESENTATION_TRAIN",
    "SOURCE_REPRESENTATION_VALIDATION",
    "SourceFitProvenanceError",
    "build_formal_source_fit_batch",
    "development_source_fit_batch",
]
