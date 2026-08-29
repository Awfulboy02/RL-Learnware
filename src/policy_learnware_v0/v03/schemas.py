"""Strict, digest-bound foundation records for the v0.3 sidecar.

The records in this module deliberately contain only protocol identities and
opaque public identifiers.  Private task/axis/factor material is kept in the
LOTO audit record and is never accepted by the anonymous selector projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

from ..hashing import sha256_json


ANONYMOUS_SELECTOR_VIEW_ENTRY_SCHEMA = (
    "policy-learnware.v03-anonymous-selector-view-entry.v0"
)
SELECTION_FAILURE_RECORD_SCHEMA = "policy-learnware.v03-selection-failure-record.v0"

ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST = frozenset(
    {
        "schema",
        "opaque_learnware_id",
        "environment_spec_digest",
        "normalized_source_competence",
        "tie_break_token",
        "entry_digest",
    }
)
PUBLIC_FORBIDDEN_FIELDS = frozenset(
    {
        "task",
        "task_id",
        "task_private_id",
        "task_schema",
        "schema_id",
        "schema_digest",
        "observation_dim",
        "action_dim",
        "dims",
        "axis",
        "axis_id",
        "factor",
        "factor_id",
        "factor_value",
        "anchor_id",
        "canonical_environment_id",
        "bundle_path",
        "policy_path",
        "opaque_id",
        "opaque_target_id",
    }
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_OPAQUE_LEARNWARE_ID = re.compile(r"^lw-[0-9a-f]{32}$")
_OPAQUE_QUERY_ID = re.compile(r"^v03q-[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class V03SchemaError(ValueError):
    """A v0.3 record is ambiguous, non-canonical, or digest-inconsistent."""


def strict_mapping(
    value: Any,
    expected: set[str] | frozenset[str],
    where: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise V03SchemaError(f"{where} must be a string-keyed mapping")
    missing = set(expected) - set(value)
    unknown = set(value) - set(expected)
    if missing or unknown:
        raise V03SchemaError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def checked_digest(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise V03SchemaError(f"{where} must be a SHA-256 hex digest")
    if not _HEX64.fullmatch(value):
        raise V03SchemaError(f"{where} must be a lowercase SHA-256 hex digest")
    return value


def checked_safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise V03SchemaError(f"{where} must be a non-empty safe identifier")
    if value in {".", ".."}:
        raise V03SchemaError(f"{where} must be a non-empty safe identifier")
    return value


def validate_public_mapping(
    value: Mapping[str, Any],
    *,
    allowlist: set[str] | frozenset[str],
    required: set[str] | frozenset[str] | None = None,
    where: str = "public mapping",
) -> None:
    """Reject unknown metadata and recursively reject private identity fields."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise V03SchemaError(f"{where} must be a string-keyed mapping")
    unknown = set(value) - set(allowlist)
    missing = set(required or ()) - set(value)
    if unknown or missing:
        raise V03SchemaError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    def scan(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                normalized = str(key).lower()
                if normalized in PUBLIC_FORBIDDEN_FIELDS:
                    raise V03SchemaError(f"forbidden public field at {path}.{key}")
                scan(nested, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                scan(nested, f"{path}[{index}]")

    scan(value, where)


@dataclass(frozen=True)
class AnonymousSelectorViewEntry:
    opaque_learnware_id: str
    environment_spec_digest: str
    normalized_source_competence: float | None
    tie_break_token: str

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_learnware_id, str) or not _OPAQUE_LEARNWARE_ID.fullmatch(
            self.opaque_learnware_id
        ):
            raise V03SchemaError("opaque_learnware_id has invalid canonical format")
        object.__setattr__(
            self,
            "environment_spec_digest",
            checked_digest(self.environment_spec_digest, "environment_spec_digest"),
        )
        if self.normalized_source_competence is not None:
            if isinstance(self.normalized_source_competence, bool) or not isinstance(
                self.normalized_source_competence, (int, float)
            ):
                raise V03SchemaError("normalized_source_competence must be finite or null")
            competence = float(self.normalized_source_competence)
            if not math.isfinite(competence) or not 0.0 <= competence <= 1.0:
                raise V03SchemaError(
                    "normalized_source_competence must lie in [0, 1] or be null"
                )
            object.__setattr__(self, "normalized_source_competence", competence)
        object.__setattr__(
            self, "tie_break_token", checked_digest(self.tie_break_token, "tie_break_token")
        )

    def material_dict(self) -> dict[str, Any]:
        return {
            "schema": ANONYMOUS_SELECTOR_VIEW_ENTRY_SCHEMA,
            "opaque_learnware_id": self.opaque_learnware_id,
            "environment_spec_digest": self.environment_spec_digest,
            "normalized_source_competence": self.normalized_source_competence,
            "tie_break_token": self.tie_break_token,
        }

    @property
    def entry_digest(self) -> str:
        return sha256_json(self.material_dict())

    def to_dict(self) -> dict[str, Any]:
        result = {**self.material_dict(), "entry_digest": self.entry_digest}
        validate_public_mapping(
            result,
            allowlist=ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST,
            required=ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST,
            where="anonymous selector view entry",
        )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnonymousSelectorViewEntry":
        validate_public_mapping(
            value,
            allowlist=ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST,
            required=ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST,
            where="anonymous selector view entry",
        )
        if value["schema"] != ANONYMOUS_SELECTOR_VIEW_ENTRY_SCHEMA:
            raise V03SchemaError("unknown anonymous selector view entry schema")
        record = cls(
            opaque_learnware_id=value["opaque_learnware_id"],
            environment_spec_digest=value["environment_spec_digest"],
            normalized_source_competence=value["normalized_source_competence"],
            tie_break_token=value["tie_break_token"],
        )
        if checked_digest(value["entry_digest"], "entry_digest") != record.entry_digest:
            raise V03SchemaError("anonymous selector entry digest does not match payload")
        return record


@dataclass(frozen=True)
class SelectionFailureRecord:
    opaque_query_id: str
    selected_opaque_learnware_id: str
    status: str
    ranking_digest: str
    abi_audit_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_query_id, str) or not _OPAQUE_QUERY_ID.fullmatch(
            self.opaque_query_id
        ):
            raise V03SchemaError("opaque_query_id has invalid canonical format")
        if not isinstance(self.selected_opaque_learnware_id, str) or not _OPAQUE_LEARNWARE_ID.fullmatch(
            self.selected_opaque_learnware_id
        ):
            raise V03SchemaError("selected_opaque_learnware_id has invalid canonical format")
        if self.status not in {"SELECTED_INCOMPATIBLE_ABI", "EXECUTION_FAILED"}:
            raise V03SchemaError("unknown selection failure status")
        object.__setattr__(self, "ranking_digest", checked_digest(self.ranking_digest, "ranking_digest"))
        object.__setattr__(self, "abi_audit_digest", checked_digest(self.abi_audit_digest, "abi_audit_digest"))

    def material_dict(self) -> dict[str, Any]:
        return {
            "schema": SELECTION_FAILURE_RECORD_SCHEMA,
            "opaque_query_id": self.opaque_query_id,
            "selected_opaque_learnware_id": self.selected_opaque_learnware_id,
            "status": self.status,
            "ranking_digest": self.ranking_digest,
            "abi_audit_digest": self.abi_audit_digest,
        }

    @property
    def failure_record_digest(self) -> str:
        return sha256_json(self.material_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.material_dict(), "failure_record_digest": self.failure_record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionFailureRecord":
        fields = {
            "schema",
            "opaque_query_id",
            "selected_opaque_learnware_id",
            "status",
            "ranking_digest",
            "abi_audit_digest",
            "failure_record_digest",
        }
        data = strict_mapping(value, fields, "selection failure record")
        if data["schema"] != SELECTION_FAILURE_RECORD_SCHEMA:
            raise V03SchemaError("unknown selection failure record schema")
        record = cls(
            opaque_query_id=data["opaque_query_id"],
            selected_opaque_learnware_id=data["selected_opaque_learnware_id"],
            status=data["status"],
            ranking_digest=data["ranking_digest"],
            abi_audit_digest=data["abi_audit_digest"],
        )
        expected = checked_digest(
            data["failure_record_digest"], "failure_record_digest"
        )
        if expected != record.failure_record_digest:
            raise V03SchemaError("selection failure record digest does not match payload")
        return record


__all__ = [
    "ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST",
    "ANONYMOUS_SELECTOR_VIEW_ENTRY_SCHEMA",
    "AnonymousSelectorViewEntry",
    "PUBLIC_FORBIDDEN_FIELDS",
    "SELECTION_FAILURE_RECORD_SCHEMA",
    "SelectionFailureRecord",
    "V03SchemaError",
    "checked_digest",
    "checked_safe_id",
    "strict_mapping",
    "validate_public_mapping",
]
