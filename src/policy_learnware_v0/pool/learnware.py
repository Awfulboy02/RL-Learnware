"""The selector-visible portion of a Policy Learnware v0 pool."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..io import deterministic_npz_bytes


_OPAQUE_ID = re.compile(r"^lw-[0-9a-f]{20,64}$")


class PoolValidationError(ValueError):
    """A pool violates the closed v0 protocol."""


def _field(value: Any, name: str, *aliases: str) -> Any:
    names = (name, *aliases)
    if isinstance(value, Mapping):
        for candidate in names:
            if candidate in value:
                return value[candidate]
    else:
        for candidate in names:
            if hasattr(value, candidate):
                return getattr(value, candidate)
    raise PoolValidationError(f"TaskSpec has no field among {names}")


def _readonly(array: Any, *, ndim: int, label: str) -> np.ndarray:
    result = np.array(array, dtype=np.float64, copy=True)
    if result.ndim != ndim:
        raise PoolValidationError(f"{label} must have rank {ndim}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise PoolValidationError(f"{label} contains non-finite values")
    result.setflags(write=False)
    return result


def _digest_spec(
    protocol_id: str,
    supports: np.ndarray,
    beta: np.ndarray,
    norm2: float,
    kernel_bandwidth: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(protocol_id.encode("utf-8"))
    for array in (supports, beta):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(float(norm2).hex().encode("ascii"))
    digest.update(float(kernel_bandwidth).hex().encode("ascii"))
    return digest.hexdigest()


def _gaussian_norm2(
    supports: np.ndarray, beta: np.ndarray, bandwidth: float
) -> float:
    squared = np.maximum(
        np.sum(supports * supports, axis=1)[:, None]
        + np.sum(supports * supports, axis=1)[None, :]
        - 2.0 * supports @ supports.T,
        0.0,
    )
    gram = np.exp(-squared / (2.0 * bandwidth * bandwidth))
    return float(beta @ gram @ beta)


@dataclass(frozen=True, eq=False)
class SelectorTaskSpec:
    """Sanitized RKME view: no task label, policy metadata, or return."""

    supports: np.ndarray
    beta: np.ndarray
    rkme_norm2: float
    protocol_id: str
    kernel_bandwidth: float
    task_spec_digest: str

    @classmethod
    def from_rkme(
        cls,
        rkme: Any,
        *,
        protocol_id: str | None = None,
        kernel_bandwidth: float | None = None,
    ) -> "SelectorTaskSpec":
        supports = _readonly(_field(rkme, "supports"), ndim=2, label="supports")
        beta = _readonly(_field(rkme, "beta", "weights"), ndim=1, label="beta")
        if supports.shape[0] != beta.shape[0] or supports.shape[0] == 0:
            raise PoolValidationError("supports and beta have inconsistent support counts")
        norm2 = float(_field(rkme, "rkme_norm2", "norm2"))
        if not np.isfinite(norm2) or norm2 < -1.0e-10:
            raise PoolValidationError("rkme_norm2 must be finite and non-negative")
        try:
            embedded_protocol = str(_field(rkme, "protocol_id"))
        except PoolValidationError as error:
            raise PoolValidationError(
                "TaskSpec artifact must carry its own protocol_id"
            ) from error
        if not embedded_protocol:
            raise PoolValidationError("TaskSpec artifact protocol_id cannot be empty")
        if protocol_id is not None and embedded_protocol != protocol_id:
            raise PoolValidationError("TaskSpec protocol mismatch")
        try:
            embedded_bandwidth = float(_field(rkme, "kernel_bandwidth", "bandwidth"))
        except PoolValidationError as error:
            raise PoolValidationError(
                "TaskSpec artifact must carry its own kernel bandwidth"
            ) from error
        if not np.isfinite(embedded_bandwidth) or embedded_bandwidth <= 0:
            raise PoolValidationError("TaskSpec kernel bandwidth is required and must be positive")
        if (
            kernel_bandwidth is not None
            and not np.isclose(embedded_bandwidth, kernel_bandwidth, rtol=1.0e-12, atol=0.0)
        ):
            raise PoolValidationError("TaskSpec kernel bandwidth mismatch")
        computed_norm2 = _gaussian_norm2(supports, beta, embedded_bandwidth)
        scale = max(1.0, abs(norm2), abs(computed_norm2))
        if abs(norm2 - computed_norm2) > 1.0e-8 * scale:
            raise PoolValidationError(
                "TaskSpec rkme_norm2 disagrees with beta^T K_uu beta"
            )
        digest = _digest_spec(
            embedded_protocol, supports, beta, computed_norm2, embedded_bandwidth
        )
        return cls(
            supports,
            beta,
            max(computed_norm2, 0.0),
            embedded_protocol,
            embedded_bandwidth,
            digest,
        )

    @property
    def latent_dim(self) -> int:
        return int(self.supports.shape[1])

    @property
    def support_budget(self) -> int:
        return int(self.supports.shape[0])


@dataclass(frozen=True)
class SelectorEntry:
    """A public entry.  ``opaque_id`` deliberately encodes no task name."""

    opaque_id: str
    protocol_id: str
    task_spec: SelectorTaskSpec

    def __post_init__(self) -> None:
        if _OPAQUE_ID.fullmatch(self.opaque_id) is None:
            raise PoolValidationError("opaque_id must match lw-[0-9a-f]{20,64}")
        if self.protocol_id != self.task_spec.protocol_id:
            raise PoolValidationError("entry and TaskSpec protocol ids differ")


@dataclass(frozen=True)
class LearnwarePool:
    """Everything the selector is allowed to inspect."""

    pool_id: str
    protocol_id: str
    kernel_bandwidth: float
    entries: tuple[SelectorEntry, ...]

    def __post_init__(self) -> None:
        if not self.pool_id:
            raise PoolValidationError("pool_id is required")
        bandwidth = float(self.kernel_bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise PoolValidationError("kernel_bandwidth must be finite and positive")
        if not self.entries:
            raise PoolValidationError("pool cannot be empty")
        identifiers = [entry.opaque_id for entry in self.entries]
        if len(identifiers) != len(set(identifiers)):
            raise PoolValidationError("pool contains duplicate opaque ids")
        if any(entry.protocol_id != self.protocol_id for entry in self.entries):
            raise PoolValidationError("all pool entries must use the pool protocol")
        if any(
            not np.isclose(
                entry.task_spec.kernel_bandwidth,
                bandwidth,
                rtol=1.0e-12,
                atol=0.0,
            )
            for entry in self.entries
        ):
            raise PoolValidationError("all TaskSpecs must use the pool kernel bandwidth")
        latent_dims = {entry.task_spec.latent_dim for entry in self.entries}
        support_budgets = {entry.task_spec.support_budget for entry in self.entries}
        if len(latent_dims) != 1:
            raise PoolValidationError("all TaskSpecs must share one latent dimension")
        if len(support_budgets) != 1:
            raise PoolValidationError("all TaskSpecs must share one support budget")

    @property
    def latent_dim(self) -> int:
        return self.entries[0].task_spec.latent_dim

    @property
    def support_budget(self) -> int:
        return self.entries[0].task_spec.support_budget

    def validate_expected_size(self, expected_entries: int) -> None:
        if len(self.entries) != int(expected_entries):
            raise PoolValidationError(
                f"pool has {len(self.entries)} entries, expected {expected_entries}"
            )

    def public_manifest(self) -> dict[str, Any]:
        """Serialize only selector-safe identifiers and RKME fingerprints."""

        return {
            "schema": "policy-learnware.selector-pool.v0",
            "pool_id": self.pool_id,
            "protocol_id": self.protocol_id,
            "kernel_bandwidth": float(self.kernel_bandwidth),
            "latent_dim": self.latent_dim,
            "support_budget": self.support_budget,
            "entries": [
                {
                    "opaque_id": entry.opaque_id,
                    "protocol_id": entry.protocol_id,
                    "task_spec_digest": entry.task_spec.task_spec_digest,
                    "task_spec_file": f"task_specs/{entry.opaque_id}.npz",
                }
                for entry in self.entries
            ],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_public_pool(
    pool: LearnwarePool,
    directory: str | Path,
) -> Path:
    """Atomically publish a self-contained selector-only pool artifact."""

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite existing pool artifact: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent))
    published = False
    try:
        spec_dir = temporary / "task_specs"
        spec_dir.mkdir()
        payload = pool.public_manifest()
        entries_payload = payload["entries"]
        for entry, record in zip(pool.entries, entries_payload):
            spec_path = spec_dir / f"{entry.opaque_id}.npz"
            encoded_spec = deterministic_npz_bytes(
                {
                    "supports": entry.task_spec.supports,
                    "beta": entry.task_spec.beta,
                    "rkme_norm2": np.asarray(entry.task_spec.rkme_norm2),
                    "protocol_id": np.asarray(entry.task_spec.protocol_id),
                    "kernel_bandwidth": np.asarray(entry.task_spec.kernel_bandwidth),
                }
            )
            with spec_path.open("wb") as handle:
                handle.write(encoded_spec)
                handle.flush()
                os.fsync(handle.fileno())
            record["file_sha256"] = _sha256_file(spec_path)
        manifest_path = temporary / "pool_manifest.json"
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8") + b"\n"
        with manifest_path.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory)
        published = True
        return directory
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def load_public_pool(directory: str | Path) -> LearnwarePool:
    """Reload and verify a selector-only pool in a fresh process."""

    directory = Path(directory)
    try:
        with (directory / "pool_manifest.json").open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise PoolValidationError(f"cannot load public pool manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != "policy-learnware.selector-pool.v0":
        raise PoolValidationError("unsupported public pool manifest")
    protocol_id = str(manifest["protocol_id"])
    bandwidth = float(manifest["kernel_bandwidth"])
    entries: list[SelectorEntry] = []
    for record in manifest.get("entries", []):
        if not isinstance(record, dict):
            raise PoolValidationError("invalid public pool entry")
        relative = Path(str(record["task_spec_file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise PoolValidationError("unsafe TaskSpec path in public pool")
        spec_path = directory / relative
        if _sha256_file(spec_path) != record.get("file_sha256"):
            raise PoolValidationError(f"TaskSpec file checksum mismatch: {relative}")
        with np.load(spec_path, allow_pickle=False) as archive:
            raw = {
                "supports": archive["supports"],
                "beta": archive["beta"],
                "rkme_norm2": float(archive["rkme_norm2"]),
                "protocol_id": str(archive["protocol_id"]),
                "kernel_bandwidth": float(archive["kernel_bandwidth"]),
            }
        spec = SelectorTaskSpec.from_rkme(
            raw, protocol_id=protocol_id, kernel_bandwidth=bandwidth
        )
        if spec.task_spec_digest != record.get("task_spec_digest"):
            raise PoolValidationError("TaskSpec semantic digest mismatch")
        entries.append(SelectorEntry(str(record["opaque_id"]), protocol_id, spec))
    pool = LearnwarePool(
        pool_id=str(manifest["pool_id"]),
        protocol_id=protocol_id,
        kernel_bandwidth=bandwidth,
        entries=tuple(entries),
    )
    if pool.latent_dim != int(manifest["latent_dim"]):
        raise PoolValidationError("public pool latent dimension mismatch")
    if pool.support_budget != int(manifest["support_budget"]):
        raise PoolValidationError("public pool support budget mismatch")
    return pool
