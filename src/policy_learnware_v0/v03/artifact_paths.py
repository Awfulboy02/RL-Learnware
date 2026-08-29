"""Portable paths for the external v0.3 evidence trees.

Historical manifests remain byte-identical.  This module only locates their
new container and, when asked, applies the separately audited relocation map.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping


ARTIFACTS_ROOT_ENV = "RL_LEARNWARE_ARTIFACTS_ROOT"
V03_MAIN_RUN = "v03-main-20260827-r0"
V03_SIGNAL_RUN = "v03-signal-ranking-20260827-r1"
V031_RAW_RUN = "v031-raw-transition-controls-20260828-r1"


class ArtifactPathError(ValueError):
    """An external artifact root or relocation entry is unusable."""


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ArtifactPathError("cannot derive the repository root; pass --artifacts-root")


def resolve_artifacts_root(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    repository_root: str | Path | None = None,
) -> Path:
    """Resolve explicit path, then the shared environment variable, then default."""

    env = os.environ if environ is None else environ
    supplied = explicit
    if supplied is None:
        supplied = env.get(ARTIFACTS_ROOT_ENV)
    if supplied is None:
        repo = Path(repository_root).expanduser() if repository_root else _repository_root()
        supplied = repo.resolve().parent / "artifacts"
    return Path(supplied).expanduser().resolve()


@dataclass(frozen=True)
class V03ArtifactLayout:
    """Canonical locations inside the common artifact root."""

    root: Path

    @classmethod
    def discover(cls, explicit: str | Path | None = None) -> "V03ArtifactLayout":
        return cls(resolve_artifacts_root(explicit))

    @property
    def runs(self) -> Path:
        return self.root / "v03" / "runs"

    @property
    def main_run(self) -> Path:
        return self.runs / V03_MAIN_RUN

    @property
    def signal_run(self) -> Path:
        return self.runs / V03_SIGNAL_RUN

    @property
    def v031_run(self) -> Path:
        return self.runs / V031_RAW_RUN

    @property
    def public_policy_market(self) -> Path:
        return self.main_run / "source-market" / "public_policy_market.json"

    @property
    def deployment_private_registry(self) -> Path:
        return self.main_run / "source-market" / "deployment_private_registry.json"

    @property
    def context_index(self) -> Path:
        return self.signal_run / "probes" / "context_index.json"

    @property
    def signal_fit_root(self) -> Path:
        return self.signal_run / "signal" / "fits"

    @property
    def development_oracle(self) -> Path:
        return self.signal_run / "baseline" / "oracle"


def resolve_recorded_path(
    recorded: str | Path,
    *,
    relocation_manifest: str | Path | None,
    artifacts_root: str | Path | None = None,
) -> Path:
    """Resolve one missing immutable path through an external relocation manifest.

    The manifest is intentionally small: its ``entries`` (or ``relocations``)
    contain ``old_path`` and ``new_path``.  Relative new paths are rooted at the
    common artifacts directory.  Longest-prefix matching also relocates files
    nested below a moved directory.
    """

    original = Path(recorded).expanduser()
    if original.exists():
        return original.resolve()
    if relocation_manifest is None:
        raise ArtifactPathError(
            f"recorded asset is missing: {original}; pass a verified relocation manifest"
        )
    manifest_path = Path(relocation_manifest).expanduser().resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactPathError(f"cannot read relocation manifest: {manifest_path}") from error
    rows = value.get("entries", value.get("relocations")) if isinstance(value, dict) else None
    if not isinstance(rows, list):
        raise ArtifactPathError("relocation manifest requires an entries list")
    matches: list[tuple[int, Path]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        old_raw, new_raw = row.get("old_path"), row.get("new_path")
        if not isinstance(old_raw, str) or not isinstance(new_raw, str):
            continue
        old = Path(old_raw).expanduser()
        try:
            suffix = original.relative_to(old)
        except ValueError:
            continue
        new = Path(new_raw).expanduser()
        if not new.is_absolute():
            new = resolve_artifacts_root(artifacts_root) / new
        matches.append((len(old.parts), new / suffix))
    if not matches:
        raise ArtifactPathError(f"no relocation covers recorded asset: {original}")
    candidate = max(matches, key=lambda item: item[0])[1].resolve()
    if not candidate.exists():
        raise ArtifactPathError(f"relocated asset is missing: {candidate}")
    return candidate


__all__ = [
    "ARTIFACTS_ROOT_ENV",
    "ArtifactPathError",
    "V03ArtifactLayout",
    "resolve_artifacts_root",
    "resolve_recorded_path",
]
