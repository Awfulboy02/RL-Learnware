"""Immutable, path-confined artifact stores for v0.3 development and joint runs.

The joint v0.3 layout intentionally has no confirmatory-oracle capability.
That namespace is owned by the shared ``policy-learnware-paper1`` orchestrator.
All publications are atomic and immutable; resume is byte-exact, and readers
require an expected content digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping

import numpy as np

from ..hashing import canonical_json_bytes, sha256_bytes, sha256_file
from ..io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    deterministic_npz_bytes,
    read_npz,
)


ArtifactNamespace = Literal["development", "joint"]

DEVELOPMENT_DOMAINS = frozenset(
    {
        "frozen_protocol_drafts",
        "v02_pool_intake",
        "source_market_private",
        "market_public",
        "attribution",
        "probe_discovery",
        "encoder_training_banks",
        "encoder_checkpoints",
        "tuning_trials",
        "development_queries",
        "development_oracle_private",
        "completion",
    }
)

JOINT_V03_DOMAINS = frozenset(
    {
        "frozen",
        "encoder_training_private",
        "source_reference_measurement",
        "representation_indices",
        "selector_views",
        "measurement",
        "selector_outputs",
        "public_completion",
        "deployment_private",
        "analysis",
        "joint_completion",
    }
)

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


class V03ArtifactError(ValueError):
    """An artifact path, capability, immutable resume, or digest is invalid."""


def _safe_segment(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_SEGMENT.fullmatch(value):
        raise V03ArtifactError(f"unsafe {where}: {value!r}")
    candidate = Path(value)
    if candidate.is_absolute() or len(candidate.parts) != 1 or value in {".", ".."}:
        raise V03ArtifactError(f"unsafe {where}: {value!r}")
    return value


@dataclass(frozen=True)
class V03ArtifactLayout:
    artifacts_root: Path
    run_id: str
    namespace: ArtifactNamespace

    def __post_init__(self) -> None:
        root = Path(self.artifacts_root).expanduser().resolve()
        object.__setattr__(self, "artifacts_root", root)
        object.__setattr__(self, "run_id", _safe_segment(self.run_id, "run_id"))
        if self.namespace not in {"development", "joint"}:
            raise V03ArtifactError(f"unknown v0.3 artifact namespace: {self.namespace!r}")

    @classmethod
    def development(cls, artifacts_root: str | Path, development_id: str) -> "V03ArtifactLayout":
        return cls(Path(artifacts_root), development_id, "development")

    @classmethod
    def joint(cls, artifacts_root: str | Path, joint_experiment_id: str) -> "V03ArtifactLayout":
        return cls(Path(artifacts_root), joint_experiment_id, "joint")

    @property
    def run_root(self) -> Path:
        return self.artifacts_root / self.run_id

    @property
    def domains(self) -> frozenset[str]:
        return DEVELOPMENT_DOMAINS if self.namespace == "development" else JOINT_V03_DOMAINS

    def domain_dir(self, domain: str) -> Path:
        if domain not in self.domains:
            if domain == "confirmatory_oracle_private":
                raise V03ArtifactError(
                    "v0.3 has no confirmatory-oracle write capability; use the joint orchestrator"
                )
            raise V03ArtifactError(f"unknown {self.namespace} artifact capability: {domain!r}")
        if domain in {"completion", "joint_completion"}:
            return self.run_root
        return self.run_root / domain

    @property
    def completion_manifest(self) -> Path:
        name = (
            "completion_manifest.json"
            if self.namespace == "development"
            else "joint_completion_manifest.json"
        )
        return self.run_root / name

    def artifact(self, domain: str, *segments: str) -> Path:
        if domain in {"completion", "joint_completion"}:
            if segments:
                raise V03ArtifactError("completion capability has a single canonical path")
            return self.completion_manifest
        if not segments:
            raise V03ArtifactError("artifact path requires at least one filename segment")
        path = self.domain_dir(domain)
        for index, segment in enumerate(segments):
            path /= _safe_segment(segment, f"artifact segment[{index}]")
        return path

    def encoder_checkpoint_artifact(
        self, fold_id: str, encoder_id: str, filename: str
    ) -> Path:
        domain = (
            "encoder_checkpoints"
            if self.namespace == "development"
            else "encoder_training_private"
        )
        return self.artifact(domain, "folds", fold_id, encoder_id, filename)

    def _reject_symlink_components(self, lexical: Path) -> None:
        root = self.run_root.absolute()
        if root.is_symlink():
            raise V03ArtifactError(f"symlink run root is forbidden: {root}")
        try:
            relative = lexical.absolute().relative_to(root)
        except ValueError as exc:
            raise V03ArtifactError(f"path escapes v0.3 run root: {lexical}") from exc
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise V03ArtifactError(f"symlink artifact path is forbidden: {current}")

    def assert_managed(self, path: str | Path) -> Path:
        lexical = Path(path).expanduser()
        if not lexical.is_absolute():
            lexical = lexical.absolute()
        self._reject_symlink_components(lexical)
        candidate = lexical.resolve()
        root = self.run_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise V03ArtifactError(f"path escapes v0.3 run root: {candidate}") from exc
        if candidate == root:
            raise V03ArtifactError("publication cannot target the run directory")
        return candidate

    def assert_domain(self, path: str | Path, domain: str) -> Path:
        candidate = self.assert_managed(path)
        domain_root = self.domain_dir(domain).resolve()
        if domain in {"completion", "joint_completion"}:
            if candidate != self.completion_manifest.resolve():
                raise V03ArtifactError(
                    "completion capability may publish only its canonical manifest"
                )
            return candidate
        try:
            candidate.relative_to(domain_root)
        except ValueError as exc:
            raise V03ArtifactError(
                f"{domain} capability cannot access outside {domain_root}: {candidate}"
            ) from exc
        if candidate == domain_root:
            raise V03ArtifactError("artifact access cannot target a domain directory")
        return candidate

    def writer(self, domain: str) -> "V03ArtifactWriter":
        self.domain_dir(domain)
        return V03ArtifactWriter(self, domain)

    def reader(self, domain: str) -> "V03ArtifactReader":
        self.domain_dir(domain)
        return V03ArtifactReader(self, domain)

    def relative(self, path: str | Path) -> str:
        return str(self.assert_managed(path).relative_to(self.run_root.resolve()))


@dataclass(frozen=True)
class V03ArtifactWriter:
    layout: V03ArtifactLayout
    domain: str

    def _path(self, path: str | Path) -> Path:
        return self.layout.assert_domain(path, self.domain)

    @staticmethod
    def _resume(destination: Path, expected: bytes) -> str:
        if not destination.is_file() or destination.is_symlink():
            raise V03ArtifactError(f"resume target is not a regular artifact: {destination}")
        actual = destination.read_bytes()
        if actual != expected:
            raise V03ArtifactError(f"resume content mismatch: {destination}")
        return sha256_bytes(actual)

    def publish_json(self, path: str | Path, value: Any, *, resume: bool = False) -> str:
        destination = self._path(path)
        expected = canonical_json_bytes(value) + b"\n"
        if destination.exists() and resume:
            return self._resume(destination, expected)
        return atomic_write_json(destination, value, overwrite=False)

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
            return self._resume(destination, expected)
        return atomic_write_npz(destination, arrays, overwrite=False)

    def publish_bytes(
        self, path: str | Path, value: bytes, *, resume: bool = False
    ) -> str:
        destination = self._path(path)
        expected = bytes(value)
        if destination.exists() and resume:
            return self._resume(destination, expected)
        return atomic_write_bytes(destination, expected, overwrite=False)

    def publish_text(self, path: str | Path, value: str, *, resume: bool = False) -> str:
        if not isinstance(value, str):
            raise TypeError("text artifact must be a string")
        return self.publish_bytes(path, value.encode("utf-8"), resume=resume)


@dataclass(frozen=True)
class V03ArtifactReader:
    layout: V03ArtifactLayout
    domain: str

    def _verified_path(self, path: str | Path, expected_sha256: str) -> Path:
        destination = self.layout.assert_domain(path, self.domain)
        if not destination.is_file() or destination.is_symlink():
            raise V03ArtifactError(f"artifact is not a regular file: {destination}")
        if not isinstance(expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_sha256
        ):
            raise V03ArtifactError(
                "expected_sha256 must be a lowercase SHA-256 hex digest"
            )
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise V03ArtifactError(
                f"artifact digest mismatch for {destination}: expected "
                f"{expected_sha256}, got {actual}"
            )
        return destination

    def load_json(self, path: str | Path, *, expected_sha256: str) -> Any:
        destination = self._verified_path(path, expected_sha256)
        with destination.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        expected_bytes = canonical_json_bytes(value) + b"\n"
        if destination.read_bytes() != expected_bytes:
            raise V03ArtifactError(f"JSON artifact is not canonical: {destination}")
        return value

    def load_npz(
        self, path: str | Path, *, expected_sha256: str
    ) -> dict[str, np.ndarray]:
        destination = self._verified_path(path, expected_sha256)
        arrays = read_npz(destination)
        for array in arrays.values():
            array.setflags(write=False)
        return arrays

    def load_bytes(self, path: str | Path, *, expected_sha256: str) -> bytes:
        return self._verified_path(path, expected_sha256).read_bytes()


__all__ = [
    "ArtifactNamespace",
    "DEVELOPMENT_DOMAINS",
    "JOINT_V03_DOMAINS",
    "V03ArtifactError",
    "V03ArtifactLayout",
    "V03ArtifactReader",
    "V03ArtifactWriter",
]
