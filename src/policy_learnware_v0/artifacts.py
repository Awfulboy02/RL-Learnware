"""Canonical, path-safe artifact layout for one immutable v0 pool build.

The layout deliberately keeps selector-visible material (``selector_pool``)
separate from private policy inventory, championization, and deployment records.
Callers receive paths from this object and publish through the atomic helpers;
the CLI never accepts an arbitrary output path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .io import (
    ArtifactExistsError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    deterministic_npz_bytes,
    read_json,
)


class ArtifactLayoutError(ValueError):
    """An artifact path or immutable resume request violates the layout."""


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def _safe_segment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactLayoutError(f"{label} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ArtifactLayoutError(f"unsafe {label}: {value!r}")
    if not _SAFE_SEGMENT.fullmatch(value):
        raise ArtifactLayoutError(
            f"{label} may contain only ASCII letters, digits, dot, underscore, and hyphen"
        )
    return value


@dataclass(frozen=True)
class ArtifactLayout:
    """All paths belonging to ``artifacts/<pool_id>``.

    Construction is side-effect free, which makes it safe for CLI dry-runs.
    Directories are created only by an atomic publication helper or an existing
    domain-specific save routine receiving one of these managed paths.
    """

    artifacts_root: Path
    pool_id: str

    def __post_init__(self) -> None:
        root = Path(self.artifacts_root).expanduser().resolve()
        pool_id = _safe_segment(self.pool_id, "pool_id")
        object.__setattr__(self, "artifacts_root", root)
        object.__setattr__(self, "pool_id", pool_id)

    @property
    def pool_root(self) -> Path:
        return self.artifacts_root / self.pool_id

    @property
    def protocol_dir(self) -> Path:
        return self.pool_root / "protocol"

    @property
    def datasets_dir(self) -> Path:
        return self.pool_root / "datasets"

    @property
    def task_specs_dir(self) -> Path:
        return self.pool_root / "task_specs"

    @property
    def policy_dir(self) -> Path:
        return self.pool_root / "policy"

    @property
    def parity_reports_dir(self) -> Path:
        return self.policy_dir / "parity_reports"

    @property
    def championization_candidates_dir(self) -> Path:
        """Private, independently publishable candidate-return shards."""

        return self.policy_dir / "championization_candidates"

    @property
    def championization_lock(self) -> Path:
        return self.policy_dir / ".championization.lock"

    @property
    def deployment_pair_evaluations_dir(self) -> Path:
        """Private, resumable final-return evaluations for unique selected pairs."""

        return self.policy_dir / "deployment_pair_evaluations"

    @property
    def learnwares_dir(self) -> Path:
        return self.pool_root / "learnwares"

    @property
    def selector_pool_dir(self) -> Path:
        """Self-contained public pool; never contains the private registry."""

        return self.pool_root / "selector_pool"

    @property
    def queries_dir(self) -> Path:
        return self.pool_root / "queries"

    @property
    def reports_dir(self) -> Path:
        return self.pool_root / "reports"

    @property
    def env_schemas(self) -> Path:
        return self.protocol_dir / "env_schemas.json"

    @property
    def env_golden_io(self) -> Path:
        return self.protocol_dir / "env_golden_io.npz"

    @property
    def environment_manifest(self) -> Path:
        return self.protocol_dir / "environment_manifest.json"

    @property
    def normalization(self) -> Path:
        return self.protocol_dir / "normalization.npz"

    @property
    def normalization_manifest(self) -> Path:
        return self.protocol_dir / "normalization_manifest.json"

    @property
    def encoder_checkpoint(self) -> Path:
        return self.protocol_dir / "encoder.msgpack"

    @property
    def encoder_config(self) -> Path:
        return self.protocol_dir / "encoder_config.json"

    @property
    def encoder_manifest(self) -> Path:
        return self.protocol_dir / "encoder_manifest.json"

    @property
    def kernel(self) -> Path:
        return self.protocol_dir / "kernel.json"

    @property
    def kernel_manifest(self) -> Path:
        return self.protocol_dir / "kernel_manifest.json"

    @property
    def frozen_protocol(self) -> Path:
        return self.protocol_dir / "protocol.json"

    @property
    def protocol_manifest(self) -> Path:
        return self.protocol_dir / "manifest.json"

    @property
    def policy_inventory(self) -> Path:
        return self.policy_dir / "inventory.json"

    @property
    def bundle_verification(self) -> Path:
        return self.policy_dir / "verification.json"

    @property
    def championization_returns(self) -> Path:
        return self.policy_dir / "championization_returns.json"

    @property
    def championization(self) -> Path:
        return self.policy_dir / "championization.json"

    @property
    def private_registry(self) -> Path:
        return self.policy_dir / "deployment_registry.json"

    @property
    def pool_manifest(self) -> Path:
        return self.pool_root / "pool_manifest.json"

    @property
    def smoke_report(self) -> Path:
        return self.reports_dir / "smoke.json"

    @property
    def unreduced_diagnostics(self) -> Path:
        return self.reports_dir / "unreduced_diagnostics.json"

    @property
    def mmd_matrix(self) -> Path:
        return self.reports_dir / "mmd_matrix.csv"

    @property
    def reduced_unreduced_ranking(self) -> Path:
        return self.reports_dir / "reduced_unreduced_ranking.json"

    @property
    def retrieval_metrics(self) -> Path:
        return self.reports_dir / "retrieval_metrics.json"

    @property
    def retrieval_execution_attestation(self) -> Path:
        return self.reports_dir / "retrieval_execution_attestation.json"

    @property
    def deployment_metrics(self) -> Path:
        return self.reports_dir / "deployment_metrics.json"

    @property
    def summary(self) -> Path:
        return self.reports_dir / "summary.md"

    def dataset_dir(self, split: str, *, bank: int | None = None) -> Path:
        split = _safe_segment(split, "dataset split")
        directory = self.datasets_dir / split
        if bank is not None:
            if isinstance(bank, bool) or not isinstance(bank, int) or bank < 0:
                raise ArtifactLayoutError("dataset bank must be a non-negative integer")
            directory = directory / f"bank_{bank:03d}"
        return directory

    def dataset_npz(self, split: str, task: str, *, bank: int | None = None) -> Path:
        return self.dataset_dir(split, bank=bank) / f"{_safe_segment(task, 'task')}.npz"

    def dataset_manifest(
        self, split: str, task: str, *, bank: int | None = None
    ) -> Path:
        return self.dataset_dir(split, bank=bank) / f"{_safe_segment(task, 'task')}.json"

    def task_spec_dir(self, task: str) -> Path:
        return self.task_specs_dir / _safe_segment(task, "task")

    def empirical_summary(self, task: str) -> Path:
        return self.task_spec_dir(task) / "empirical_summary.json"

    def task_rkme(self, task: str) -> Path:
        return self.task_spec_dir(task) / "task_rkme.npz"

    def task_rkme_manifest(self, task: str) -> Path:
        return self.task_spec_dir(task) / "task_rkme.json"

    def parity_report(self, job_id: str) -> Path:
        return self.parity_reports_dir / f"{_safe_segment(job_id, 'job_id')}.json"

    def championization_candidate(self, job_id: str) -> Path:
        return self.championization_candidates_dir / (
            f"{_safe_segment(job_id, 'job_id')}.json"
        )

    def deployment_pair_evaluation(self, pair_id: str) -> Path:
        return self.deployment_pair_evaluations_dir / (
            f"{_safe_segment(pair_id, 'deployment pair id')}.json"
        )

    def learnware_manifest(self, task: str) -> Path:
        return self.learnwares_dir / _safe_segment(task, "task") / "learnware.json"

    def query_dir(self, query_id: str) -> Path:
        return self.queries_dir / _safe_segment(query_id, "query_id")

    def selection_result(self, query_id: str) -> Path:
        return self.query_dir(query_id) / "selection_result.json"

    def deployment_result(self, query_id: str) -> Path:
        return self.query_dir(query_id) / "deployment_result.json"

    def assert_managed(self, path: str | Path) -> Path:
        """Resolve and reject any output outside this pool's artifact root."""

        candidate = Path(path).expanduser().resolve()
        try:
            candidate.relative_to(self.pool_root)
        except ValueError as error:
            raise ArtifactLayoutError(
                f"output path is outside managed pool root {self.pool_root}: {candidate}"
            ) from error
        if candidate == self.pool_root:
            raise ArtifactLayoutError("atomic file publication cannot target the pool directory")
        return candidate

    def publish_json(
        self,
        path: str | Path,
        payload: Any,
        *,
        resume: bool = False,
    ) -> str:
        destination = self.assert_managed(path)
        if destination.exists() and resume:
            expected = canonical_json_bytes(payload) + b"\n"
            actual = destination.read_bytes()
            if actual != expected:
                raise ArtifactLayoutError(
                    f"resume artifact differs from requested content: {destination}"
                )
            return sha256_bytes(actual)
        return atomic_write_json(destination, payload, overwrite=False)

    def publish_npz(
        self,
        path: str | Path,
        arrays: Mapping[str, np.ndarray],
        *,
        resume: bool = False,
    ) -> str:
        destination = self.assert_managed(path)
        if destination.exists() and resume:
            expected = deterministic_npz_bytes(arrays)
            actual = destination.read_bytes()
            if actual != expected:
                raise ArtifactLayoutError(
                    f"resume artifact differs from requested content: {destination}"
                )
            return sha256_bytes(actual)
        return atomic_write_npz(destination, arrays, overwrite=False)

    def publish_text(
        self,
        path: str | Path,
        text: str,
        *,
        resume: bool = False,
    ) -> str:
        destination = self.assert_managed(path)
        payload = text.encode("utf-8")
        if destination.exists() and resume:
            if destination.read_bytes() != payload:
                raise ArtifactLayoutError(
                    f"resume artifact differs from requested content: {destination}"
                )
            return sha256_bytes(payload)
        return atomic_write_bytes(destination, payload, overwrite=False)

    def verify_manifest_files(self, manifest_path: str | Path) -> Mapping[str, Any]:
        """Verify a CLI manifest whose ``files`` values carry path and SHA-256."""

        manifest = read_json(self.assert_managed(manifest_path))
        if not isinstance(manifest, Mapping):
            raise ArtifactLayoutError("artifact manifest must be a JSON object")
        files = manifest.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ArtifactLayoutError("artifact manifest has no files mapping")
        for label, record in files.items():
            if not isinstance(label, str) or not isinstance(record, Mapping):
                raise ArtifactLayoutError("invalid artifact manifest file record")
            raw_path = record.get("path")
            digest = record.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(digest, str):
                raise ArtifactLayoutError("manifest file record misses path or sha256")
            file_path = Path(raw_path)
            if not file_path.is_absolute():
                file_path = self.pool_root / file_path
            file_path = self.assert_managed(file_path)
            if not file_path.is_file():
                raise ArtifactLayoutError(f"manifest payload is missing: {file_path}")
            if sha256_file(file_path) != digest:
                raise ArtifactLayoutError(f"manifest payload digest mismatch: {file_path}")
        return manifest

    def relative(self, path: str | Path) -> str:
        return str(self.assert_managed(path).relative_to(self.pool_root))


__all__ = [
    "ArtifactExistsError",
    "ArtifactLayout",
    "ArtifactLayoutError",
]
