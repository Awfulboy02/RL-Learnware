"""Path-safe, capability-scoped, immutable artifact publication for v0.2.

The v0.2 protocol separates public measurement/selection artifacts from
benchmark, training, deployment, and oracle-private data.  Writers created by
this module carry exactly one domain capability and cannot cross that physical
boundary.  A resume is accepted only when the bytes already on disk are
identical to the bytes that would be published now.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import numpy as np

from ..hashing import canonical_json_bytes, sha256_bytes
from ..io import atomic_write_bytes, atomic_write_json, atomic_write_npz, deterministic_npz_bytes


ArtifactDomain = Literal[
    "frozen",
    "benchmark_private",
    "training_private",
    "market_public",
    "representation_indices",
    "deployment_private",
    "measurement",
    "selector_outputs",
    "analysis",
    "completion",
]

_DOMAIN_NAMES = frozenset(
    {
        "frozen",
        "benchmark_private",
        "training_private",
        "market_public",
        "representation_indices",
        "deployment_private",
        "measurement",
        "selector_outputs",
        "analysis",
        "completion",
    }
)
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class V02ArtifactLayoutError(ValueError):
    """An artifact path, capability request, or immutable resume is invalid."""


def _safe_segment(value: str, where: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_SEGMENT.fullmatch(value):
        raise V02ArtifactLayoutError(f"unsafe {where}: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute() or len(candidate.parts) != 1 or value in {".", ".."}:
        raise V02ArtifactLayoutError(f"unsafe {where}: {value!r}")
    return value


def _index(value: int, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V02ArtifactLayoutError(f"{where} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class V02ArtifactLayout:
    """Canonical v0.2 experiment layout with least-privilege writers."""

    artifacts_root: Path
    experiment_id: str

    def __post_init__(self) -> None:
        root = Path(self.artifacts_root).expanduser().resolve()
        object.__setattr__(self, "artifacts_root", root)
        object.__setattr__(
            self, "experiment_id", _safe_segment(self.experiment_id, "experiment_id")
        )

    @property
    def experiment_root(self) -> Path:
        return self.artifacts_root / self.experiment_id

    def domain_dir(self, domain: ArtifactDomain) -> Path:
        if domain not in _DOMAIN_NAMES:
            raise V02ArtifactLayoutError(f"unknown artifact capability domain: {domain!r}")
        if domain == "completion":
            return self.experiment_root
        return self.experiment_root / domain

    @property
    def frozen_dir(self) -> Path:
        return self.domain_dir("frozen")

    @property
    def benchmark_private_dir(self) -> Path:
        return self.domain_dir("benchmark_private")

    @property
    def training_private_dir(self) -> Path:
        return self.domain_dir("training_private")

    @property
    def market_public_dir(self) -> Path:
        return self.domain_dir("market_public")

    @property
    def representation_indices_dir(self) -> Path:
        return self.domain_dir("representation_indices")

    @property
    def deployment_private_dir(self) -> Path:
        return self.domain_dir("deployment_private")

    @property
    def measurement_dir(self) -> Path:
        return self.domain_dir("measurement")

    @property
    def selector_outputs_dir(self) -> Path:
        return self.domain_dir("selector_outputs")

    @property
    def analysis_dir(self) -> Path:
        return self.domain_dir("analysis")

    @property
    def completion_manifest(self) -> Path:
        return self.experiment_root / "completion_manifest.json"

    @property
    def preflight_completion_manifest(self) -> Path:
        return self.experiment_root / "preflight_completion_manifest.json"

    @property
    def run_lock(self) -> Path:
        return self.experiment_root / ".run.lock"

    def frozen_artifact(self, filename: str) -> Path:
        return self.frozen_dir / _safe_segment(filename, "frozen filename")

    def variant_artifact(self, opaque_env_id: str, filename: str) -> Path:
        return (
            self.benchmark_private_dir
            / "variants"
            / _safe_segment(opaque_env_id, "opaque_env_id")
            / _safe_segment(filename, "variant filename")
        )

    def training_job_artifact(self, job_id: str, filename: str) -> Path:
        return (
            self.training_private_dir
            / "jobs"
            / _safe_segment(job_id, "job_id")
            / _safe_segment(filename, "training filename")
        )

    def public_learnware_artifact(self, learnware_id: str, filename: str) -> Path:
        return (
            self.market_public_dir
            / "learnwares"
            / _safe_segment(learnware_id, "learnware_id")
            / _safe_segment(filename, "learnware filename")
        )

    def representation_artifact(
        self, representation_index_id: str, learnware_id: str, filename: str
    ) -> Path:
        return (
            self.representation_indices_dir
            / _safe_segment(representation_index_id, "representation_index_id")
            / "learnwares"
            / _safe_segment(learnware_id, "learnware_id")
            / _safe_segment(filename, "representation filename")
        )

    def target_dataset_artifact(
        self, opaque_target_id: str, bank: int, filename: str
    ) -> Path:
        return (
            self.measurement_dir
            / "target_queries"
            / _safe_segment(opaque_target_id, "opaque_target_id")
            / "datasets"
            / f"bank_{_index(bank, 'bank'):03d}"
            / _safe_segment(filename, "dataset filename")
        )

    def selector_artifact(
        self, method_id: str, opaque_target_id: str, filename: str
    ) -> Path:
        return (
            self.selector_outputs_dir
            / _safe_segment(method_id, "method_id")
            / _safe_segment(opaque_target_id, "opaque_target_id")
            / _safe_segment(filename, "selector filename")
        )

    def analysis_artifact(self, filename: str) -> Path:
        return self.analysis_dir / _safe_segment(filename, "analysis filename")

    def gate_artifact(self, filename: str) -> Path:
        return self.analysis_dir / "gates" / _safe_segment(filename, "gate filename")

    @property
    def recompute_audit(self) -> Path:
        return self.analysis_artifact("recompute_audit.json")

    def _reject_symlink_components(self, lexical: Path) -> None:
        """Reject existing symlinks below the experiment root, even if they point inward."""

        root = self.experiment_root
        try:
            relative = lexical.absolute().relative_to(root.absolute())
        except ValueError as exc:
            raise V02ArtifactLayoutError(
                f"path escapes v0.2 experiment root: {lexical}"
            ) from exc
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise V02ArtifactLayoutError(f"symlink artifact path is forbidden: {current}")

    def assert_managed(self, path: str | Path) -> Path:
        lexical = Path(path).expanduser()
        if not lexical.is_absolute():
            lexical = lexical.absolute()
        self._reject_symlink_components(lexical)
        candidate = lexical.resolve()
        root = self.experiment_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise V02ArtifactLayoutError(
                f"path escapes v0.2 experiment root: {candidate}"
            ) from exc
        if candidate == root:
            raise V02ArtifactLayoutError("publication cannot target the experiment directory")
        return candidate

    def assert_domain(self, path: str | Path, domain: ArtifactDomain) -> Path:
        candidate = self.assert_managed(path)
        root = self.domain_dir(domain).resolve()
        if domain == "completion":
            allowed = {
                self.completion_manifest.resolve(),
                self.preflight_completion_manifest.resolve(),
            }
            if candidate not in allowed:
                raise V02ArtifactLayoutError(
                    "completion capability may publish only a completion manifest"
                )
            return candidate
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise V02ArtifactLayoutError(
                f"{domain} capability cannot write outside {root}: {candidate}"
            ) from exc
        if candidate == root:
            raise V02ArtifactLayoutError("publication cannot target a domain directory")
        return candidate

    def writer(self, domain: ArtifactDomain) -> "V02ArtifactWriter":
        self.domain_dir(domain)
        return V02ArtifactWriter(layout=self, domain=domain)

    def relative(self, path: str | Path) -> str:
        return str(self.assert_managed(path).relative_to(self.experiment_root.resolve()))


@dataclass(frozen=True)
class V02ArtifactWriter:
    """A write capability constrained to exactly one v0.2 artifact domain."""

    layout: V02ArtifactLayout
    domain: ArtifactDomain

    def _path(self, path: str | Path) -> Path:
        return self.layout.assert_domain(path, self.domain)

    @staticmethod
    def _resume_bytes(destination: Path, expected: bytes) -> str:
        actual = destination.read_bytes()
        if actual != expected:
            raise V02ArtifactLayoutError(f"resume content mismatch: {destination}")
        return sha256_bytes(actual)

    def publish_json(self, path: str | Path, payload: Any, *, resume: bool = False) -> str:
        destination = self._path(path)
        expected = canonical_json_bytes(payload) + b"\n"
        if destination.exists() and resume:
            return self._resume_bytes(destination, expected)
        return atomic_write_json(destination, payload, overwrite=False)

    def publish_npz(
        self,
        path: str | Path,
        arrays: Mapping[str, np.ndarray],
        *,
        resume: bool = False,
    ) -> str:
        destination = self._path(path)
        expected = deterministic_npz_bytes(arrays)
        if destination.exists() and resume:
            return self._resume_bytes(destination, expected)
        return atomic_write_npz(destination, arrays, overwrite=False)

    def publish_text(self, path: str | Path, value: str, *, resume: bool = False) -> str:
        if not isinstance(value, str):
            raise TypeError("text artifact must be a string")
        return self.publish_bytes(path, value.encode("utf-8"), resume=resume)

    def publish_bytes(self, path: str | Path, value: bytes, *, resume: bool = False) -> str:
        destination = self._path(path)
        expected = bytes(value)
        if destination.exists() and resume:
            return self._resume_bytes(destination, expected)
        return atomic_write_bytes(destination, expected, overwrite=False)


__all__ = [
    "ArtifactDomain",
    "V02ArtifactLayout",
    "V02ArtifactLayoutError",
    "V02ArtifactWriter",
]
