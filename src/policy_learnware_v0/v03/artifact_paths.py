"""Portable, fail-closed paths for the external v0.3 evidence trees.

Historical manifests remain byte-identical. This module only locates their
new container and applies the separately audited root relocation manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


ARTIFACTS_ROOT_ENV = "RL_LEARNWARE_ARTIFACTS_ROOT"
RELOCATION_MANIFEST_SCHEMA = "rl-learnware-relocation/v1"
INVENTORY_ALGORITHM = "sha256sum-relative-v1"
V03_MAIN_RUN = "v03-main-20260827-r0"
V03_SIGNAL_RUN = "v03-signal-ranking-20260827-r1"
V031_RAW_RUN = "v031-raw-transition-controls-20260828-r1"

_MANIFEST_KEYS = frozenset({"schema", "mappings"})
_MAPPING_REQUIRED_KEYS = frozenset(
    {
        "kind",
        "source",
        "target",
        "content_manifest_sha256",
        "file_count",
        "total_bytes",
        "role",
        "access_class",
        "status",
    }
)
_MAPPING_OPTIONAL_KEYS = frozenset({"completeness", "known_missing"})
_MAPPING_KINDS = frozenset({"directory", "file", "prefix"})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


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
    if not str(supplied).strip():
        raise ArtifactPathError("artifacts root cannot be empty")
    raw = Path(supplied).expanduser()
    if raw.is_symlink():
        raise ArtifactPathError(f"artifacts root cannot be a symlink: {raw}")
    return raw.resolve()


@dataclass(frozen=True)
class V03ArtifactLayout:
    """Canonical locations inside the common artifact root."""

    root: Path

    @classmethod
    def discover(cls, explicit: str | Path | None = None) -> "V03ArtifactLayout":
        return cls(resolve_artifacts_root(explicit))

    @property
    def relocation_manifest(self) -> Path:
        return self.root / "relocation_manifest.json"

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


@dataclass(frozen=True)
class _Relocation:
    kind: str
    source: Path
    target: Path
    content_manifest_sha256: str
    file_count: int
    total_bytes: int


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactPathError(f"duplicate JSON key in relocation manifest: {key}")
        value[key] = item
    return value


def _normal_absolute(raw: str, *, field: str) -> Path:
    if not raw or os.path.normpath(raw) != raw:
        raise ArtifactPathError(f"{field} must be normalized")
    value = Path(raw)
    if not value.is_absolute() or ".." in value.parts:
        raise ArtifactPathError(f"{field} must be a safe absolute path")
    return value


def _normal_relative_target(raw: str, *, field: str) -> PurePosixPath:
    if not raw or "\\" in raw or "\x00" in raw or "\n" in raw:
        raise ArtifactPathError(f"{field} must be a safe POSIX relative path")
    value = PurePosixPath(raw)
    if (
        value.is_absolute()
        or value == PurePosixPath(".")
        or ".." in value.parts
        or value.as_posix() != raw
    ):
        raise ArtifactPathError(f"{field} must be a normalized POSIX relative path")
    return value


def _reject_symlinks_below(path: Path, *, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ArtifactPathError(f"relocated path escapes artifacts root: {path}") from error
    if path == root:
        raise ArtifactPathError("a relocation target must be strictly below artifacts root")
    if root.is_symlink():
        raise ArtifactPathError(f"artifacts root cannot be a symlink: {root}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactPathError(f"relocated path contains a symlink: {current}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _target_inventory(path_raw: str) -> tuple[str, int, int]:
    """Return sha256sum-relative-v1 digest, regular-file count and file bytes."""

    root = Path(path_raw)
    if root.is_symlink() or not root.exists():
        raise ArtifactPathError(f"relocation target is missing or a symlink: {root}")
    if root.is_file():
        files = [(root.name, root)]
    elif root.is_dir():
        files: list[tuple[str, Path]] = []
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            parent = Path(directory)
            for name in dirnames:
                entry = parent / name
                if entry.is_symlink() or not entry.is_dir():
                    raise ArtifactPathError(f"relocation tree contains a special entry: {entry}")
            for name in filenames:
                entry = parent / name
                if entry.is_symlink() or not entry.is_file():
                    raise ArtifactPathError(f"relocation tree contains a special entry: {entry}")
                files.append((entry.relative_to(root).as_posix(), entry))
    else:
        raise ArtifactPathError(f"relocation target is a special entry: {root}")

    encoded: list[tuple[bytes, Path]] = []
    for relative, path in files:
        try:
            relative_bytes = relative.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ArtifactPathError(f"relocation path is not UTF-8 encodable: {path}") from error
        if relative.startswith("./") or b"\n" in relative_bytes:
            raise ArtifactPathError(f"relocation path is not inventory-safe: {path}")
        encoded.append((relative_bytes, path))
    encoded.sort(key=lambda item: item[0])

    manifest_digest = hashlib.sha256()
    total_bytes = 0
    for relative_bytes, path in encoded:
        leaf_digest = _sha256_file(path)
        manifest_digest.update(leaf_digest.encode("ascii"))
        manifest_digest.update(b"  ")
        manifest_digest.update(relative_bytes)
        manifest_digest.update(b"\n")
        total_bytes += path.stat().st_size
    return manifest_digest.hexdigest(), len(encoded), total_bytes


def _read_verified_mappings(
    manifest_path: Path, *, artifacts_root: Path
) -> tuple[_Relocation, ...]:
    try:
        value = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
        )
    except ArtifactPathError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactPathError(f"cannot read relocation manifest: {manifest_path}") from error
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise ArtifactPathError("relocation manifest has an invalid top-level schema")
    if value["schema"] != RELOCATION_MANIFEST_SCHEMA:
        raise ArtifactPathError("relocation manifest schema is unsupported")
    rows = value["mappings"]
    if not isinstance(rows, list):
        raise ArtifactPathError("relocation manifest requires a mappings list")

    verified: list[_Relocation] = []
    seen_sources: set[Path] = set()
    allowed = _MAPPING_REQUIRED_KEYS | _MAPPING_OPTIONAL_KEYS
    for index, row in enumerate(rows):
        label = f"mappings[{index}]"
        if (
            not isinstance(row, dict)
            or not _MAPPING_REQUIRED_KEYS <= set(row)
            or not set(row) <= allowed
        ):
            raise ArtifactPathError(f"{label} has an invalid schema")
        if row["kind"] not in _MAPPING_KINDS:
            raise ArtifactPathError(f"{label}.kind is unsupported")
        if not isinstance(row["source"], str) or not isinstance(row["target"], str):
            raise ArtifactPathError(f"{label} source and target must be strings")
        source = _normal_absolute(row["source"], field=f"{label}.source")
        target_relative = _normal_relative_target(row["target"], field=f"{label}.target")
        target = artifacts_root.joinpath(*target_relative.parts)
        if source in seen_sources:
            raise ArtifactPathError(f"duplicate relocation source: {source}")
        seen_sources.add(source)
        digest = row["content_manifest_sha256"]
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise ArtifactPathError(f"{label}.content_manifest_sha256 is invalid")
        for field in ("file_count", "total_bytes"):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
                raise ArtifactPathError(f"{label}.{field} must be a non-negative integer")
        for field in ("role", "access_class", "status"):
            if not isinstance(row[field], str) or row[field] != row[field].strip() or not row[field]:
                raise ArtifactPathError(f"{label}.{field} must be a normalized non-empty string")
        if "completeness" in row and row["completeness"] not in {"complete", "incomplete"}:
            raise ArtifactPathError(f"{label}.completeness is invalid")
        if "known_missing" in row and (
            not isinstance(row["known_missing"], list)
            or not all(isinstance(item, str) and item for item in row["known_missing"])
        ):
            raise ArtifactPathError(f"{label}.known_missing must be a string list")
        if row["status"] == "verified":
            verified.append(
                _Relocation(
                    kind=row["kind"],
                    source=source,
                    target=target,
                    content_manifest_sha256=digest,
                    file_count=row["file_count"],
                    total_bytes=row["total_bytes"],
                )
            )
    return tuple(verified)


def _validate_target(mapping: _Relocation, *, artifacts_root: Path) -> None:
    _reject_symlinks_below(mapping.target, root=artifacts_root)
    if mapping.kind == "file" and not mapping.target.is_file():
        raise ArtifactPathError(f"file relocation target is not a file: {mapping.target}")
    if mapping.kind in {"directory", "prefix"} and not mapping.target.is_dir():
        raise ArtifactPathError(f"directory relocation target is not a directory: {mapping.target}")
    observed = _target_inventory(str(mapping.target))
    expected = (
        mapping.content_manifest_sha256,
        mapping.file_count,
        mapping.total_bytes,
    )
    if observed != expected:
        raise ArtifactPathError(
            f"relocation inventory differs for {mapping.target}: "
            f"expected {expected}, observed {observed}"
        )


def resolve_recorded_path(
    recorded: str | Path,
    *,
    relocation_manifest: str | Path | None,
    artifacts_root: str | Path | None = None,
) -> Path:
    """Resolve an immutable absolute path through the verified root manifest.

    Supplying a manifest disables the legacy existing-path fallback. This
    prevents a stale source tree from bypassing the audited relocation.
    """

    original = _normal_absolute(
        os.path.expanduser(os.fspath(recorded)), field="recorded path"
    )
    if relocation_manifest is None:
        if not original.exists():
            raise ArtifactPathError(
                f"recorded asset is missing: {original}; pass a verified relocation manifest"
            )
        current = Path(original.anchor)
        for part in original.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise ArtifactPathError(f"recorded path contains a symlink: {current}")
        return original.resolve()

    root = resolve_artifacts_root(artifacts_root)
    expected_manifest = root / "relocation_manifest.json"
    supplied_manifest = Path(relocation_manifest).expanduser()
    if supplied_manifest.is_symlink() or supplied_manifest.resolve() != expected_manifest.resolve():
        raise ArtifactPathError(
            f"relocation manifest must be the authoritative root manifest: {expected_manifest}"
        )
    mappings = _read_verified_mappings(expected_manifest, artifacts_root=root)
    matches: list[tuple[int, _Relocation, Path]] = []
    for mapping in mappings:
        if mapping.kind == "file":
            if original == mapping.source:
                matches.append((len(mapping.source.parts), mapping, Path()))
            continue
        try:
            suffix = original.relative_to(mapping.source)
        except ValueError:
            continue
        matches.append((len(mapping.source.parts), mapping, suffix))
    if not matches:
        raise ArtifactPathError(f"no verified relocation covers recorded asset: {original}")
    longest = max(length for length, _, _ in matches)
    winners = [item for item in matches if item[0] == longest]
    if len(winners) != 1:
        raise ArtifactPathError(f"ambiguous relocation covers recorded asset: {original}")
    _, mapping, suffix = winners[0]
    _validate_target(mapping, artifacts_root=root)
    candidate = mapping.target if mapping.kind == "file" else mapping.target / suffix
    _reject_symlinks_below(candidate, root=root)
    if not candidate.exists():
        raise ArtifactPathError(f"relocated asset is missing: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ArtifactPathError(f"relocated path escapes artifacts root: {resolved}") from error
    if resolved == root:
        raise ArtifactPathError("a relocated asset must be strictly below artifacts root")
    return resolved


__all__ = [
    "ARTIFACTS_ROOT_ENV",
    "INVENTORY_ALGORITHM",
    "RELOCATION_MANIFEST_SCHEMA",
    "ArtifactPathError",
    "V03ArtifactLayout",
    "resolve_artifacts_root",
    "resolve_recorded_path",
]
