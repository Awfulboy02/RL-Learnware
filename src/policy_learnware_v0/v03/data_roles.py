"""Physical/logical dataset-role separation and strict LOTO leakage guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ..hashing import sha256_json
from .schemas import (
    LOTOFoldRecord,
    V03SchemaError,
    checked_digest,
    checked_ids,
    checked_safe_id,
    strict_mapping,
)


DATA_ROLE_RECORD_SCHEMA = "policy-learnware.v03-data-role-record.v0"
DATA_ROLE_MANIFEST_SCHEMA = "policy-learnware.v03-data-role-manifest.v0"

DatasetRole = Literal[
    "source_encoder_train",
    "source_encoder_validation",
    "source_reference_spec",
    "development_query",
    "development_oracle",
    "confirmatory_query",
    "confirmatory_oracle",
]
DATASET_ROLES = frozenset(
    {
        "source_encoder_train",
        "source_encoder_validation",
        "source_reference_spec",
        "development_query",
        "development_oracle",
        "confirmatory_query",
        "confirmatory_oracle",
    }
)

PROCESS_ROLE_READS: Mapping[str, frozenset[str]] = {
    "encoder_trainer": frozenset(
        {"source_encoder_train", "source_encoder_validation"}
    ),
    "source_spec_builder": frozenset({"source_reference_spec"}),
    "query_encoder": frozenset({"development_query", "confirmatory_query"}),
    "development_analysis": frozenset(
        {"development_query", "development_oracle"}
    ),
    "joint_oracle": frozenset({"confirmatory_query", "confirmatory_oracle"}),
    "analysis_recompute": DATASET_ROLES,
    "selector": frozenset(),
}


class DataRoleError(V03SchemaError):
    """Dataset identity, role, seed, process access, or LOTO isolation is invalid."""


@dataclass(frozen=True)
class DataRoleRecord:
    role: DatasetRole
    dataset_id: str
    dataset_digest: str
    task_private_ids: tuple[str, ...]
    seed_tokens: tuple[str, ...]
    split_nonce_digest: str

    def __post_init__(self) -> None:
        if self.role not in DATASET_ROLES:
            raise DataRoleError(f"unknown dataset role: {self.role!r}")
        object.__setattr__(self, "dataset_id", checked_safe_id(self.dataset_id, "dataset_id"))
        object.__setattr__(
            self, "dataset_digest", checked_digest(self.dataset_digest, "dataset_digest")
        )
        object.__setattr__(
            self,
            "task_private_ids",
            checked_ids(self.task_private_ids, "task_private_ids"),
        )
        object.__setattr__(
            self,
            "seed_tokens",
            checked_ids(self.seed_tokens, "seed_tokens", allow_empty=False),
        )
        object.__setattr__(
            self,
            "split_nonce_digest",
            checked_digest(self.split_nonce_digest, "split_nonce_digest"),
        )

    def material_dict(self) -> dict[str, Any]:
        return {
            "schema": DATA_ROLE_RECORD_SCHEMA,
            "role": self.role,
            "dataset_id": self.dataset_id,
            "dataset_digest": self.dataset_digest,
            "task_private_ids": list(self.task_private_ids),
            "seed_tokens": list(self.seed_tokens),
            "split_nonce_digest": self.split_nonce_digest,
        }

    @property
    def role_record_digest(self) -> str:
        return sha256_json(self.material_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.material_dict(), "role_record_digest": self.role_record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataRoleRecord":
        fields = {
            "schema",
            "role",
            "dataset_id",
            "dataset_digest",
            "task_private_ids",
            "seed_tokens",
            "split_nonce_digest",
            "role_record_digest",
        }
        data = strict_mapping(value, fields, "data-role record")
        if data["schema"] != DATA_ROLE_RECORD_SCHEMA:
            raise DataRoleError("unknown data-role record schema")
        try:
            record = cls(
                role=data["role"],
                dataset_id=data["dataset_id"],
                dataset_digest=data["dataset_digest"],
                task_private_ids=tuple(data["task_private_ids"]),
                seed_tokens=tuple(data["seed_tokens"]),
                split_nonce_digest=data["split_nonce_digest"],
            )
        except (TypeError, KeyError) as exc:
            raise DataRoleError("invalid data-role record value") from exc
        if checked_digest(data["role_record_digest"], "role_record_digest") != record.role_record_digest:
            raise DataRoleError("data-role record digest does not match payload")
        return record


@dataclass(frozen=True)
class DataRoleManifest:
    manifest_id: str
    records: tuple[DataRoleRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_id", checked_safe_id(self.manifest_id, "manifest_id")
        )
        records = tuple(self.records)
        if not records or not all(isinstance(record, DataRoleRecord) for record in records):
            raise DataRoleError("data-role manifest requires typed records")
        identities = [(record.role, record.dataset_id) for record in records]
        if len(set(identities)) != len(identities):
            raise DataRoleError("duplicate role/dataset identity in data-role manifest")
        digest_roles: dict[str, set[str]] = {}
        for record in records:
            digest_roles.setdefault(record.dataset_digest, set()).add(record.role)
        aliases = {
            digest: roles for digest, roles in digest_roles.items() if len(roles) > 1
        }
        if aliases:
            raise DataRoleError(
                "the same physical dataset digest is assigned to multiple logical roles"
            )
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda item: (item.role, item.dataset_id))),
        )
        self._validate_seed_separation()

    def _validate_seed_separation(self) -> None:
        by_role: dict[str, set[str]] = {role: set() for role in DATASET_ROLES}
        for record in self.records:
            by_role[record.role].update(record.seed_tokens)
        train_validation_overlap = (
            by_role["source_encoder_train"] & by_role["source_encoder_validation"]
        )
        if train_validation_overlap:
            raise DataRoleError(
                "source encoder train and validation seed tokens overlap: "
                f"{sorted(train_validation_overlap)}"
            )
        reference = by_role["source_reference_spec"]
        encoder_fit_seeds = (
            by_role["source_encoder_train"]
            | by_role["source_encoder_validation"]
        )
        fit_reference_overlap = encoder_fit_seeds & reference
        if fit_reference_overlap:
            raise DataRoleError(
                "encoder train/validation and source-reference seed tokens overlap: "
                f"{sorted(fit_reference_overlap)}"
            )
        for query_role in ("development_query", "confirmatory_query"):
            overlap = reference & by_role[query_role]
            if overlap:
                raise DataRoleError(
                    f"source-reference and {query_role} seed tokens overlap: {sorted(overlap)}"
                )
        overlap = by_role["development_query"] & by_role["confirmatory_query"]
        if overlap:
            raise DataRoleError(
                f"development and confirmatory query seeds overlap: {sorted(overlap)}"
            )

    def records_for(self, role: DatasetRole) -> tuple[DataRoleRecord, ...]:
        if role not in DATASET_ROLES:
            raise DataRoleError(f"unknown dataset role: {role!r}")
        return tuple(record for record in self.records if record.role == role)

    def role_digest(self, role: DatasetRole) -> str:
        records = self.records_for(role)
        if not records:
            raise DataRoleError(f"no records registered for role {role!r}")
        return sha256_json(
            {
                "schema": "policy-learnware.v03-data-role-projection.v0",
                "role": role,
                "record_digests": sorted(record.role_record_digest for record in records),
            }
        )

    def material_dict(self) -> dict[str, Any]:
        return {
            "schema": DATA_ROLE_MANIFEST_SCHEMA,
            "manifest_id": self.manifest_id,
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def manifest_digest(self) -> str:
        return sha256_json(self.material_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.material_dict(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataRoleManifest":
        fields = {"schema", "manifest_id", "records", "manifest_digest"}
        data = strict_mapping(value, fields, "data-role manifest")
        if data["schema"] != DATA_ROLE_MANIFEST_SCHEMA:
            raise DataRoleError("unknown data-role manifest schema")
        if isinstance(data["records"], (str, bytes)) or not isinstance(
            data["records"], Sequence
        ):
            raise DataRoleError("data-role manifest records must be a sequence")
        manifest = cls(
            manifest_id=data["manifest_id"],
            records=tuple(DataRoleRecord.from_dict(record) for record in data["records"]),
        )
        if checked_digest(data["manifest_digest"], "manifest_digest") != manifest.manifest_digest:
            raise DataRoleError("data-role manifest digest does not match payload")
        return manifest


def assert_process_can_read(process_role: str, dataset_role: DatasetRole) -> None:
    if process_role not in PROCESS_ROLE_READS:
        raise DataRoleError(f"unknown process role: {process_role!r}")
    if dataset_role not in DATASET_ROLES:
        raise DataRoleError(f"unknown dataset role: {dataset_role!r}")
    if dataset_role not in PROCESS_ROLE_READS[process_role]:
        raise DataRoleError(
            f"process {process_role!r} cannot read dataset role {dataset_role!r}"
        )


def validate_loto_isolation(
    fold: LOTOFoldRecord,
    manifest: DataRoleManifest,
    *,
    target_query_role: Literal["development_query", "confirmatory_query"],
) -> None:
    """Validate one fold before training or held-out query encoding begins."""

    train = manifest.records_for("source_encoder_train")
    validation = manifest.records_for("source_encoder_validation")
    reference = manifest.records_for("source_reference_spec")
    queries = manifest.records_for(target_query_role)
    if not train or not validation or not reference or not queries:
        raise DataRoleError("LOTO manifest is missing a required train/validation/ref/query role")

    held_out = fold.held_out_task_private_id
    train_tasks = {task for record in (*train, *validation) for task in record.task_private_ids}
    if held_out in train_tasks:
        raise DataRoleError("LOTO held-out task leaked into train/validation records")
    if train_tasks != set(fold.train_task_private_ids):
        raise DataRoleError("LOTO train/validation task set does not match fold record")

    query_tasks = {task for record in queries for task in record.task_private_ids}
    if query_tasks != {held_out}:
        raise DataRoleError("LOTO target-query records must contain only the held-out task")
    reference_tasks = {task for record in reference for task in record.task_private_ids}
    if held_out not in reference_tasks:
        raise DataRoleError("LOTO source references must include the held-out task")

    if set(fold.train_dataset_digests) != {record.dataset_digest for record in train}:
        raise DataRoleError("LOTO training dataset digests do not match role manifest")
    if set(fold.validation_dataset_digests) != {
        record.dataset_digest for record in validation
    }:
        raise DataRoleError("LOTO validation dataset digests do not match role manifest")
    if fold.source_reference_role_digest != manifest.role_digest("source_reference_spec"):
        raise DataRoleError("LOTO source-reference role digest mismatch")
    if fold.target_query_role_digest != manifest.role_digest(target_query_role):
        raise DataRoleError("LOTO target-query role digest mismatch")

    split_nonces = {record.split_nonce_digest for record in (*train, *validation, *queries)}
    if split_nonces != {fold.split_nonce_digest}:
        raise DataRoleError("LOTO split nonce mismatch")


def assert_confirmatory_queries_unseen(
    manifest: DataRoleManifest, development_read_digests: Sequence[str]
) -> None:
    read = {checked_digest(item, "development_read_digests[]") for item in development_read_digests}
    confirmatory = {
        record.dataset_digest for record in manifest.records_for("confirmatory_query")
    }
    overlap = read & confirmatory
    if overlap:
        raise DataRoleError(
            "confirmatory query was read during development and must be downgraded"
        )


__all__ = [
    "DATASET_ROLES",
    "DATA_ROLE_MANIFEST_SCHEMA",
    "DATA_ROLE_RECORD_SCHEMA",
    "DataRoleError",
    "DataRoleManifest",
    "DataRoleRecord",
    "DatasetRole",
    "PROCESS_ROLE_READS",
    "assert_confirmatory_queries_unseen",
    "assert_process_can_read",
    "validate_loto_isolation",
]
