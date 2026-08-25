"""Immutable formal-protocol freeze binding for Policy Learnware v0.2.

The strict configuration object is necessary but not sufficient for a formal
run: reading the YAML again after it has changed would silently create a new
protocol under the same experiment directory.  This module freezes the exact
config bytes and the v0.2 implementation tree before any formal work is
allowed, and re-verifies both on every downstream formal command.

The record deliberately contains no Paper-I joint/confirmatory material.  It
only authorizes the v0.2 development/freeze-ready sidecar and caps the terminal
status at ``READY_FOR_V03_JOINT_CONFIRMATORY``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from ..hashing import sha256_file, sha256_json
from .artifacts import V02ArtifactLayout
from .config import V02ExperimentConfig, load_v02_formal_config


FORMAL_FREEZE_SCHEMA = "policy-learnware.v02-formal-protocol-freeze.v0"
FORMAL_FREEZE_FILENAME = "v02_freeze_manifest.json"
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class FormalFreezeError(ValueError):
    """A formal command is not bound to one immutable config/software freeze."""


def _repo_root() -> Path:
    # .../repo/src/policy_learnware_v0/v02/freeze.py -> repo
    return Path(__file__).resolve().parents[3]


def _reject_symlink_ancestry(path: Path, repository_root: Path) -> None:
    try:
        relative = path.relative_to(repository_root)
    except ValueError as error:
        raise FormalFreezeError("implementation path escapes repository root") from error
    current = repository_root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise FormalFreezeError(
                f"formal implementation closure forbids symlinks: {current}"
            )


def _implementation_files(repository_root: Path) -> tuple[Path, ...]:
    """Resolve the complete repository-owned v0.2 execution closure.

    v0.2 deliberately reuses the package's hashing, IO, probe, RKME,
    environment and policy-loading modules.  Binding only the ``v02``
    directory would therefore leave executable dependencies outside the
    freeze.  The server shell entrypoints are executable protocol code too.
    We conservatively bind every Python module in the package, every Python
    and shell file in the v0.2 server backend, and the package/entrypoint
    metadata.  External FPO/runtime/vendor bytes are bound separately by the
    server provenance contract.
    """

    roots = (
        repository_root / "src" / "policy_learnware_v0",
        repository_root / "server" / "repro_fpo_ppo_v02",
    )
    forbidden_import_artifacts = {".pyc", ".pyo", ".so", ".pyd", ".dylib"}
    files: list[Path] = []
    for index, root in enumerate(roots):
        _reject_symlink_ancestry(root, repository_root)
        if not root.is_dir():
            raise FormalFreezeError(f"required v0.2 implementation root is missing: {root}")
        suffixes = {".py"} if index == 0 else {".py", ".sh"}
        for path in root.rglob("*"):
            if path.is_symlink():
                raise FormalFreezeError(
                    f"formal implementation closure forbids symlinks: {path}"
                )
            if "__pycache__" in path.parts or path.suffix.lower() in forbidden_import_artifacts:
                raise FormalFreezeError(
                    "formal implementation closure forbids local bytecode/native "
                    f"import artifacts: {path}"
                )
            if path.suffix in suffixes and path.is_file():
                files.append(path)
    metadata = repository_root / "pyproject.toml"
    _reject_symlink_ancestry(metadata, repository_root)
    if not metadata.is_file():
        raise FormalFreezeError(f"required package metadata is missing: {metadata}")
    files.append(metadata)
    audit_scripts = (
        repository_root / "scripts" / "audit_v02_recompute.py",
        repository_root / "scripts" / "run_v02_cpu_acceptance.py",
    )
    for script in audit_scripts:
        _reject_symlink_ancestry(script, repository_root)
        if not script.is_file():
            raise FormalFreezeError(f"required v0.2 audit entrypoint is missing: {script}")
        files.append(script)
    if not files:
        raise FormalFreezeError("v0.2 implementation tree is empty")
    return tuple(sorted(files))


def _live_git_release_state(repository_root: Path) -> tuple[str, bool]:
    """Return the current full commit and cleanliness, failing closed.

    A commit recorded in an old manifest is not evidence that the process is
    still executing that commit.  Formal create *and* verify therefore query
    the live repository rather than trusting caller-supplied Git fields.
    """

    root = repository_root.resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        porcelain = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise FormalFreezeError(f"cannot verify live Git release state: {error}") from error
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise FormalFreezeError("live Git HEAD is not a full commit")
    return commit, not bool(porcelain)


def implementation_tree_manifest(
    repository_root: str | Path | None = None,
) -> Mapping[str, str]:
    """Return path-independent SHA bindings for all tracked v0.2 Python code."""

    root = _repo_root() if repository_root is None else Path(repository_root).resolve()
    result: dict[str, str] = {}
    for path in _implementation_files(root):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise FormalFreezeError("implementation file escapes repository root") from error
        result[relative] = sha256_file(path)
    return dict(sorted(result.items()))


def implementation_tree_digest(
    repository_root: str | Path | None = None,
) -> str:
    return sha256_json(
        {
            "schema": "policy-learnware.v02-implementation-tree.v0",
            "files": implementation_tree_manifest(repository_root),
        }
    )


@dataclass(frozen=True)
class FormalProtocolFreeze:
    experiment_id: str
    config_digest: str
    config_file_sha256: str
    benchmark_projection_digest: str
    training_projection_digest: str
    probe_projection_digest: str
    analysis_projection_digest: str
    implementation_tree_digest: str
    software_commit: str
    worktree_clean_at_freeze: bool

    def __post_init__(self) -> None:
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise FormalFreezeError("freeze experiment_id must be non-empty")
        for name in (
            "config_digest",
            "config_file_sha256",
            "benchmark_projection_digest",
            "training_projection_digest",
            "probe_projection_digest",
            "analysis_projection_digest",
            "implementation_tree_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST.fullmatch(value.lower()) is None:
                raise FormalFreezeError(f"{name} must be a SHA-256 digest")
            object.__setattr__(self, name, value.lower())
        if (
            not isinstance(self.software_commit, str)
            or _GIT_COMMIT.fullmatch(self.software_commit.lower()) is None
        ):
            raise FormalFreezeError("software_commit must be a full Git commit")
        object.__setattr__(self, "software_commit", self.software_commit.lower())
        if self.worktree_clean_at_freeze is not True:
            raise FormalFreezeError("formal freeze requires a clean Git worktree")

    @classmethod
    def create(
        cls,
        config: V02ExperimentConfig,
        *,
        config_path: str | Path,
        software_commit: str,
        worktree_clean_at_freeze: bool,
        repository_root: str | Path | None = None,
    ) -> "FormalProtocolFreeze":
        if not isinstance(config, V02ExperimentConfig) or config.stage != "v02_freeze_ready":
            raise FormalFreezeError("formal freeze requires a v02_freeze_ready config")
        source = Path(config_path).resolve()
        if not source.is_file() or source.is_symlink():
            raise FormalFreezeError("formal config must be a regular non-symlink file")
        if (
            not isinstance(software_commit, str)
            or _GIT_COMMIT.fullmatch(software_commit.lower()) is None
        ):
            raise FormalFreezeError("software_commit must be a full Git commit")
        root = _repo_root() if repository_root is None else Path(repository_root).resolve()
        live_commit, live_clean = _live_git_release_state(root)
        if software_commit.lower() != live_commit:
            raise FormalFreezeError(
                "software_commit does not match the live repository HEAD"
            )
        if worktree_clean_at_freeze is not True or not live_clean:
            raise FormalFreezeError("formal freeze requires a clean Git worktree")
        return cls(
            experiment_id=config.experiment_id,
            config_digest=config.config_digest,
            config_file_sha256=sha256_file(source),
            benchmark_projection_digest=sha256_json(config.benchmark_projection),
            training_projection_digest=sha256_json(config.training_projection),
            probe_projection_digest=sha256_json(config.probe_projection),
            analysis_projection_digest=sha256_json(config.analysis_projection),
            implementation_tree_digest=implementation_tree_digest(root),
            software_commit=software_commit,
            worktree_clean_at_freeze=worktree_clean_at_freeze,
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FORMAL_FREEZE_SCHEMA,
            "experiment_id": self.experiment_id,
            "stage": "v02_freeze_ready",
            "config_digest": self.config_digest,
            "config_file_sha256": self.config_file_sha256,
            "benchmark_projection_digest": self.benchmark_projection_digest,
            "training_projection_digest": self.training_projection_digest,
            "probe_projection_digest": self.probe_projection_digest,
            "analysis_projection_digest": self.analysis_projection_digest,
            "implementation_tree_digest": self.implementation_tree_digest,
            "software_commit": self.software_commit,
            "worktree_clean_at_freeze": self.worktree_clean_at_freeze,
            "sealed_target_state": "NOT_INSTANTIATED_OR_READ",
            "confirmatory_oracle_state": "NOT_READ",
            "maximum_authorized_status": "READY_FOR_V03_JOINT_CONFIRMATORY",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalProtocolFreeze":
        if not isinstance(value, Mapping):
            raise FormalFreezeError("formal freeze must be a JSON object")
        fields = {
            "schema",
            "experiment_id",
            "stage",
            "config_digest",
            "config_file_sha256",
            "benchmark_projection_digest",
            "training_projection_digest",
            "probe_projection_digest",
            "analysis_projection_digest",
            "implementation_tree_digest",
            "software_commit",
            "worktree_clean_at_freeze",
            "sealed_target_state",
            "confirmatory_oracle_state",
            "maximum_authorized_status",
        }
        if set(value) != fields:
            raise FormalFreezeError("formal freeze fields differ from the canonical schema")
        constants = {
            "schema": FORMAL_FREEZE_SCHEMA,
            "stage": "v02_freeze_ready",
            "sealed_target_state": "NOT_INSTANTIATED_OR_READ",
            "confirmatory_oracle_state": "NOT_READ",
            "maximum_authorized_status": "READY_FOR_V03_JOINT_CONFIRMATORY",
        }
        if any(value[name] != expected for name, expected in constants.items()):
            raise FormalFreezeError("formal freeze boundary/status constant mismatch")
        return cls(
            experiment_id=value["experiment_id"],
            config_digest=value["config_digest"],
            config_file_sha256=value["config_file_sha256"],
            benchmark_projection_digest=value["benchmark_projection_digest"],
            training_projection_digest=value["training_projection_digest"],
            probe_projection_digest=value["probe_projection_digest"],
            analysis_projection_digest=value["analysis_projection_digest"],
            implementation_tree_digest=value["implementation_tree_digest"],
            software_commit=value["software_commit"],
            worktree_clean_at_freeze=value["worktree_clean_at_freeze"],
        )

    def verify(
        self,
        config: V02ExperimentConfig,
        *,
        config_path: str | Path,
        repository_root: str | Path | None = None,
    ) -> None:
        root = _repo_root() if repository_root is None else Path(repository_root).resolve()
        live_commit, live_clean = _live_git_release_state(root)
        if live_commit != self.software_commit:
            raise FormalFreezeError(
                "formal freeze software_commit no longer matches live Git HEAD"
            )
        if not live_clean:
            raise FormalFreezeError(
                "formal freeze requires the live Git worktree to remain clean"
            )
        expected = FormalProtocolFreeze.create(
            config,
            config_path=config_path,
            software_commit=self.software_commit,
            worktree_clean_at_freeze=True,
            repository_root=root,
        )
        for name in (
            "experiment_id",
            "config_digest",
            "config_file_sha256",
            "benchmark_projection_digest",
            "training_projection_digest",
            "probe_projection_digest",
            "analysis_projection_digest",
            "implementation_tree_digest",
        ):
            if getattr(self, name) != getattr(expected, name):
                raise FormalFreezeError(f"formal freeze {name} no longer matches live inputs")


def canonical_formal_freeze_path(config: V02ExperimentConfig) -> Path:
    layout = V02ArtifactLayout(Path(config.artifact_root), config.experiment_id)
    return layout.frozen_artifact(FORMAL_FREEZE_FILENAME).resolve()


def load_verified_formal_freeze(
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[V02ExperimentConfig, FormalProtocolFreeze]:
    source = Path(config_path).resolve()
    config = load_v02_formal_config(source)
    freeze_path = canonical_formal_freeze_path(config)
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise FormalFreezeError(
            f"formal protocol freeze is missing at canonical path {freeze_path}"
        )
    try:
        payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalFreezeError(f"cannot read formal protocol freeze: {error}") from error
    freeze = FormalProtocolFreeze.from_dict(payload)
    freeze.verify(config, config_path=source, repository_root=repository_root)
    return config, freeze


__all__ = [
    "FORMAL_FREEZE_FILENAME",
    "FORMAL_FREEZE_SCHEMA",
    "FormalFreezeError",
    "FormalProtocolFreeze",
    "canonical_formal_freeze_path",
    "implementation_tree_digest",
    "implementation_tree_manifest",
    "load_verified_formal_freeze",
]
