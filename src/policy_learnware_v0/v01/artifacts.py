"""Independent, path-safe, immutable artifact layout for v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import numpy as np

from ..hashing import canonical_json_bytes, sha256_bytes
from ..io import atomic_write_bytes, atomic_write_json, atomic_write_npz, deterministic_npz_bytes


ArtifactDomain = Literal[
    "frozen", "benchmark_private", "measurement", "oracle_private", "analysis", "completion"
]
_DOMAINS = frozenset({
    "frozen", "benchmark_private", "measurement", "oracle_private", "analysis", "completion"
})
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class V01ArtifactLayoutError(ValueError):
    """A v0.1 output path or immutable resume request is invalid."""


def _safe_segment(value: str, where: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_SEGMENT.fullmatch(value):
        raise V01ArtifactLayoutError(f"unsafe {where}: {value!r}")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise V01ArtifactLayoutError(f"unsafe {where}: {value!r}")
    return value


def _index(value: int, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V01ArtifactLayoutError(f"{where} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class V01ArtifactLayout:
    artifacts_root: Path
    experiment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root).expanduser().resolve())
        object.__setattr__(self, "experiment_id", _safe_segment(self.experiment_id, "experiment_id"))

    @property
    def experiment_root(self) -> Path:
        return self.artifacts_root / self.experiment_id

    @property
    def frozen_dir(self) -> Path:
        return self.experiment_root / "frozen"

    @property
    def benchmark_private_dir(self) -> Path:
        return self.experiment_root / "benchmark_private"

    @property
    def measurement_dir(self) -> Path:
        return self.experiment_root / "measurement"

    @property
    def oracle_private_dir(self) -> Path:
        return self.experiment_root / "oracle_private"

    @property
    def analysis_dir(self) -> Path:
        return self.experiment_root / "analysis"

    @property
    def run_lock(self) -> Path:
        return self.experiment_root / ".run.lock"

    @property
    def run_manifest(self) -> Path:
        return self.frozen_dir / "run_manifest.json"

    @property
    def measurement_protocol(self) -> Path:
        return self.frozen_dir / "measurement_protocol.json"

    @property
    def oracle_protocol(self) -> Path:
        return self.frozen_dir / "oracle_protocol.json"

    @property
    def base_protocol_ref(self) -> Path:
        return self.frozen_dir / "base_protocol_ref.json"

    @property
    def shift_registry_ref(self) -> Path:
        return self.frozen_dir / "shift_registry_ref.json"

    @property
    def frozen_measurement_contract(self) -> Path:
        return self.frozen_dir / "measurement_contract.json"

    @property
    def oracle_contract(self) -> Path:
        return self.frozen_dir / "oracle_contract.json"

    @property
    def audit_plan(self) -> Path:
        return self.frozen_dir / "audit_plan.json"

    @property
    def contexts(self) -> Path:
        return self.benchmark_private_dir / "contexts.json"

    @property
    def source_task_map(self) -> Path:
        return self.benchmark_private_dir / "source_task_map.json"

    def private_variant_dir(self, task: str, variant_id: str) -> Path:
        return self.benchmark_private_dir / "variants" / _safe_segment(task, "task") / _safe_segment(variant_id, "variant_id")

    def shift_manifest(self, task: str, variant_id: str) -> Path:
        return self.private_variant_dir(task, variant_id) / "shift_manifest.json"

    def instance_record(self, task: str, variant_id: str) -> Path:
        return self.private_variant_dir(task, variant_id) / "instance.json"

    def model_diff(self, task: str, variant_id: str) -> Path:
        return self.private_variant_dir(task, variant_id) / "model_diff.json"

    def identity_audit(self, task: str, variant_id: str) -> Path:
        return self.private_variant_dir(task, variant_id) / "identity_audit.json"

    def collection_attestation(self, variant_id: str, bank: int) -> Path:
        return self.benchmark_private_dir / "collection_attestations" / _safe_segment(variant_id, "variant_id") / f"bank_{_index(bank, 'bank'):03d}.json"

    @property
    def measurement_run_ref(self) -> Path:
        return self.measurement_dir / "run_ref.json"

    @property
    def measurement_contract(self) -> Path:
        return self.measurement_dir / "measurement_contract.json"

    @property
    def pair_plan(self) -> Path:
        return self.measurement_dir / "pair_plan.json"

    def schema_view(self, schema_view_id: str) -> Path:
        return self.measurement_dir / "schema_views" / f"{_safe_segment(schema_view_id, 'schema_view_id')}.json"

    def dataset_dir(self, variant_id: str, bank: int) -> Path:
        return self.measurement_dir / "datasets" / _safe_segment(variant_id, "variant_id") / f"bank_{_index(bank, 'bank'):03d}"

    def dataset_npz(self, variant_id: str, bank: int) -> Path:
        return self.dataset_dir(variant_id, bank) / "dataset.npz"

    def dataset_manifest(self, variant_id: str, bank: int) -> Path:
        return self.dataset_dir(variant_id, bank) / "manifest.json"

    def semantic_cache(self, variant_id: str, bank: int) -> Path:
        return self.measurement_dir / "semantic_cache" / _safe_segment(variant_id, "variant_id") / f"bank_{_index(bank, 'bank'):03d}.npz"

    def semantic_cache_manifest(self, variant_id: str, bank: int) -> Path:
        return self.measurement_dir / "semantic_cache" / _safe_segment(variant_id, "variant_id") / f"bank_{_index(bank, 'bank'):03d}.json"

    def execution_attempt(self, attempt_id: str) -> Path:
        return self.measurement_dir / "execution_attempts" / f"{_safe_segment(attempt_id, 'attempt_id')}.json"

    @property
    def taskspec_matrix_npz(self) -> Path:
        return self.measurement_dir / "taskspec_matrix.npz"

    @property
    def taskspec_matrix_axes(self) -> Path:
        return self.measurement_dir / "taskspec_matrix_axes.json"

    @property
    def taskspec_primitive_manifest(self) -> Path:
        return self.measurement_dir / "taskspec_primitive_manifest.json"

    @property
    def taskspec_matrix_csv(self) -> Path:
        return self.measurement_dir / "taskspec_matrix.csv"

    @property
    def routing_matrix_csv(self) -> Path:
        return self.measurement_dir / "routing_matrix.csv"

    @property
    def candidates(self) -> Path:
        return self.oracle_private_dir / "candidates.json"

    def oracle_shard(self, task: str, variant_id: str, candidate_id: str) -> Path:
        return self.oracle_private_dir / "shards" / _safe_segment(task, "task") / _safe_segment(variant_id, "variant_id") / f"{_safe_segment(candidate_id, 'candidate_id')}.json"

    @property
    def oracle_episodes_npz(self) -> Path:
        return self.oracle_private_dir / "oracle_episodes.npz"

    @property
    def oracle_episodes_axes(self) -> Path:
        return self.oracle_private_dir / "oracle_episodes_axes.json"

    @property
    def oracle_episodes_csv(self) -> Path:
        return self.oracle_private_dir / "oracle_episodes.csv"

    @property
    def oracle_aggregates_json(self) -> Path:
        return self.oracle_private_dir / "oracle_aggregates.json"

    @property
    def oracle_aggregates_csv(self) -> Path:
        return self.oracle_private_dir / "oracle_aggregates.csv"

    def analysis_artifact(self, filename: str) -> Path:
        return self.analysis_dir / _safe_segment(filename, "analysis filename")

    @property
    def completion_manifest(self) -> Path:
        return self.experiment_root / "completion_manifest.json"

    @property
    def preflight_completion_manifest(self) -> Path:
        return self.experiment_root / "preflight_completion_manifest.json"

    def _domain_root(self, domain: ArtifactDomain) -> Path:
        if domain not in _DOMAINS:
            raise V01ArtifactLayoutError(f"unknown artifact capability domain: {domain!r}")
        if domain == "completion":
            return self.experiment_root
        return {
            "frozen": self.frozen_dir,
            "benchmark_private": self.benchmark_private_dir,
            "measurement": self.measurement_dir,
            "oracle_private": self.oracle_private_dir,
            "analysis": self.analysis_dir,
        }[domain]

    def assert_managed(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve()
        root = self.experiment_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise V01ArtifactLayoutError(f"path escapes v0.1 experiment root: {candidate}") from exc
        if candidate == root:
            raise V01ArtifactLayoutError("publication cannot target the experiment directory")
        return candidate

    def assert_domain(self, path: str | Path, domain: ArtifactDomain) -> Path:
        candidate = self.assert_managed(path)
        root = self._domain_root(domain).resolve()
        if domain == "completion":
            if candidate not in {self.completion_manifest.resolve(), self.preflight_completion_manifest.resolve()}:
                raise V01ArtifactLayoutError("completion capability may publish only a completion manifest")
            return candidate
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise V01ArtifactLayoutError(
                f"{domain} capability cannot write outside {root}: {candidate}"
            ) from exc
        if candidate == root:
            raise V01ArtifactLayoutError("publication cannot target a domain directory")
        return candidate

    def writer(self, domain: ArtifactDomain) -> "V01ArtifactWriter":
        self._domain_root(domain)  # validate eagerly
        return V01ArtifactWriter(layout=self, domain=domain)

    def relative(self, path: str | Path) -> str:
        return str(self.assert_managed(path).relative_to(self.experiment_root))


@dataclass(frozen=True)
class V01ArtifactWriter:
    """A least-privilege write capability for exactly one artifact domain."""

    layout: V01ArtifactLayout
    domain: ArtifactDomain

    def _path(self, path: str | Path) -> Path:
        return self.layout.assert_domain(path, self.domain)

    def publish_json(self, path: str | Path, payload: Any, *, resume: bool = False) -> str:
        destination = self._path(path)
        expected = canonical_json_bytes(payload) + b"\n"
        if destination.exists() and resume:
            actual = destination.read_bytes()
            if actual != expected:
                raise V01ArtifactLayoutError(f"resume content mismatch: {destination}")
            return sha256_bytes(actual)
        return atomic_write_json(destination, payload, overwrite=False)

    def publish_npz(
        self, path: str | Path, arrays: Mapping[str, np.ndarray], *, resume: bool = False
    ) -> str:
        destination = self._path(path)
        expected = deterministic_npz_bytes(arrays)
        if destination.exists() and resume:
            actual = destination.read_bytes()
            if actual != expected:
                raise V01ArtifactLayoutError(f"resume content mismatch: {destination}")
            return sha256_bytes(actual)
        return atomic_write_npz(destination, arrays, overwrite=False)

    def publish_text(self, path: str | Path, text: str, *, resume: bool = False) -> str:
        destination = self._path(path)
        payload = text.encode("utf-8")
        if destination.exists() and resume:
            actual = destination.read_bytes()
            if actual != payload:
                raise V01ArtifactLayoutError(f"resume content mismatch: {destination}")
            return sha256_bytes(actual)
        return atomic_write_bytes(destination, payload, overwrite=False)


__all__ = [
    "ArtifactDomain", "V01ArtifactLayout", "V01ArtifactLayoutError", "V01ArtifactWriter",
]
