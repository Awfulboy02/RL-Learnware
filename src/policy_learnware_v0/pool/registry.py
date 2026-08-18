"""Private policy registry, isolated from the selector-visible pool."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..hashing import sha256_json
from .learnware import LearnwarePool, PoolValidationError


@dataclass(frozen=True)
class RegistryRecord:
    opaque_id: str
    protocol_id: str
    policy_bundle: Path
    policy_bundle_digest: str
    native_observation_dim: int
    native_action_dim: int
    source_task: str
    provenance: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.policy_bundle_digest:
            raise PoolValidationError("policy bundle digest is required")
        if self.native_observation_dim <= 0 or self.native_action_dim <= 0:
            raise PoolValidationError("native policy dimensions must be positive")
        object.__setattr__(self, "policy_bundle", Path(self.policy_bundle))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class DeploymentRegistry:
    """Private lookup used only after selection has completed."""

    def __init__(
        self,
        records: tuple[RegistryRecord, ...],
        *,
        pool_id: str,
        pool_digest: str,
    ) -> None:
        if not pool_id:
            raise PoolValidationError("deployment registry pool_id is required")
        if len(pool_digest) != 64:
            raise PoolValidationError("deployment registry pool_digest must be SHA-256")
        try:
            int(pool_digest, 16)
        except ValueError as error:
            raise PoolValidationError(
                "deployment registry pool_digest must be SHA-256"
            ) from error
        by_id = {record.opaque_id: record for record in records}
        if len(by_id) != len(records):
            raise PoolValidationError("deployment registry has duplicate opaque ids")
        self._records = MappingProxyType(by_id)
        self.pool_id = pool_id
        self.pool_digest = pool_digest

    def get(self, opaque_id: str) -> RegistryRecord:
        try:
            return self._records[opaque_id]
        except KeyError as error:
            raise KeyError(f"unknown selected opaque id: {opaque_id}") from error

    def validate_against(self, pool: LearnwarePool) -> None:
        if self.pool_id != pool.pool_id:
            raise PoolValidationError("private registry pool id mismatch")
        if self.pool_digest != sha256_json(pool.public_manifest()):
            raise PoolValidationError("private registry public-pool digest mismatch")
        public_ids = {entry.opaque_id for entry in pool.entries}
        private_ids = set(self._records)
        if public_ids != private_ids:
            raise PoolValidationError(
                f"public/private entry mismatch: missing={sorted(public_ids-private_ids)}, "
                f"extra={sorted(private_ids-public_ids)}"
            )
        for record in self._records.values():
            if record.protocol_id != pool.protocol_id:
                raise PoolValidationError("private registry protocol mismatch")

    def __len__(self) -> int:
        return len(self._records)

    def private_payload(self) -> dict[str, Any]:
        """Serialize for the deployment process, never for selector loading."""

        return {
            "schema": "policy-learnware.deployment-registry.v0",
            "pool_id": self.pool_id,
            "pool_digest": self.pool_digest,
            "records": [
                {
                    "opaque_id": record.opaque_id,
                    "protocol_id": record.protocol_id,
                    "policy_bundle": str(record.policy_bundle),
                    "policy_bundle_digest": record.policy_bundle_digest,
                    "native_observation_dim": record.native_observation_dim,
                    "native_action_dim": record.native_action_dim,
                    "source_task": record.source_task,
                    "provenance": dict(record.provenance),
                }
                for record in sorted(self._records.values(), key=lambda item: item.opaque_id)
            ],
        }


def save_private_registry(
    registry: DeploymentRegistry,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite private registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = registry.private_payload()
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def load_private_registry(
    path: str | Path,
    *,
    public_pool: LearnwarePool | None = None,
) -> DeploymentRegistry:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PoolValidationError(f"cannot load private deployment registry: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "policy-learnware.deployment-registry.v0"
    ):
        raise PoolValidationError("unsupported private deployment registry")
    records: list[RegistryRecord] = []
    for raw in payload.get("records", []):
        if not isinstance(raw, dict):
            raise PoolValidationError("invalid private registry record")
        records.append(
            RegistryRecord(
                opaque_id=str(raw["opaque_id"]),
                protocol_id=str(raw["protocol_id"]),
                policy_bundle=Path(raw["policy_bundle"]),
                policy_bundle_digest=str(raw["policy_bundle_digest"]),
                native_observation_dim=int(raw["native_observation_dim"]),
                native_action_dim=int(raw["native_action_dim"]),
                source_task=str(raw["source_task"]),
                provenance=raw.get("provenance", {}),
            )
        )
    registry = DeploymentRegistry(
        tuple(records),
        pool_id=str(payload.get("pool_id", "")),
        pool_digest=str(payload.get("pool_digest", "")),
    )
    if public_pool is not None:
        registry.validate_against(public_pool)
    return registry
