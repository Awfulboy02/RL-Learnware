"""Strict provenance for the legacy dependency tree used by GPU runners."""

from __future__ import annotations

import base64
import csv
from email.parser import Parser
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Mapping

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .provenance import (
        ContractError,
        VENDOR_PROVENANCE_SCHEMA,
        sha256_file,
        sha256_json,
        validate_vendor_provenance,
    )
except ImportError:  # pragma: no cover - exercised by executable entry points
    from provenance import (
        ContractError,
        VENDOR_PROVENANCE_SCHEMA,
        sha256_file,
        sha256_json,
        validate_vendor_provenance,
    )


_IGNORED_CACHE_DIRECTORIES = frozenset({"__pycache__"})
_IGNORED_CACHE_SUFFIXES = frozenset({".pyc", ".pyo"})


def _decode_record_sha256(value: str) -> bytes:
    algorithm, separator, encoded = value.partition("=")
    if separator != "=" or algorithm != "sha256" or not encoded:
        raise ContractError("vendored wandb RECORD must use sha256 hashes")
    try:
        return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, TypeError) as error:
        raise ContractError("vendored wandb RECORD contains an invalid sha256") from error


def _validate_wandb_distribution(root: Path) -> str:
    candidates = sorted(
        path
        for path in root.glob("wandb-*.dist-info")
        if path.is_dir() and not path.is_symlink()
    )
    if len(candidates) != 1:
        raise ContractError(
            "vendor directory must contain exactly one pinned wandb-*.dist-info"
        )
    distribution = candidates[0]
    metadata_path = distribution / "METADATA"
    record_path = distribution / "RECORD"
    package_path = root / "wandb" / "__init__.py"
    for path, label in (
        (metadata_path, "wandb METADATA"),
        (record_path, "wandb RECORD"),
        (package_path, "wandb package"),
    ):
        if not path.is_file() or path.is_symlink():
            raise ContractError(f"vendor directory is missing a regular {label} file")

    try:
        metadata = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise ContractError("cannot read vendored wandb METADATA") from error
    if metadata.get("Name", "").strip().lower() != "wandb":
        raise ContractError("vendored wandb METADATA has the wrong distribution name")
    version = metadata.get("Version", "").strip()
    if not version or any(character.isspace() for character in version):
        raise ContractError("vendored wandb METADATA has no strict version")

    expected_relative = package_path.relative_to(root).as_posix()
    matching: list[tuple[str, str]] = []
    try:
        with record_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) != 3:
                    raise ContractError("vendored wandb RECORD contains a malformed row")
                if row[0] == expected_relative:
                    matching.append((row[1], row[2]))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ContractError("cannot read vendored wandb RECORD") from error
    if len(matching) != 1:
        raise ContractError("vendored wandb RECORD does not uniquely bind wandb/__init__.py")
    encoded_digest, recorded_size = matching[0]
    try:
        expected_size = int(recorded_size)
    except ValueError as error:
        raise ContractError("vendored wandb RECORD has an invalid file size") from error
    package_bytes = package_path.read_bytes()
    if expected_size != len(package_bytes):
        raise ContractError("vendored wandb package size differs from RECORD")
    if _decode_record_sha256(encoded_digest) != hashlib.sha256(package_bytes).digest():
        raise ContractError("vendored wandb package digest differs from RECORD")
    return version


def inspect_vendor_directory(path: Path | str) -> dict[str, Any]:
    """Validate and digest a pinned dependency tree, excluding runtime caches.

    Every regular non-cache file is hashed. Symlinks and special files are
    rejected so the digest cannot depend on content outside the selected tree.
    """

    source = Path(path)
    if source.is_symlink():
        raise ContractError("vendor directory itself must not be a symlink")
    try:
        root = source.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"vendor directory does not exist: {source}") from error
    if not root.is_dir():
        raise ContractError(f"vendor dependency path is not a directory: {root}")
    wandb_version = _validate_wandb_distribution(root)

    paths: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = current_path / name
            if candidate.is_symlink():
                raise ContractError(f"vendor tree contains a symlink: {candidate}")
            if name not in _IGNORED_CACHE_DIRECTORIES:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            candidate = current_path / name
            if candidate.suffix in _IGNORED_CACHE_SUFFIXES:
                continue
            if candidate.is_symlink():
                raise ContractError(f"vendor tree contains a symlink: {candidate}")
            mode = candidate.stat().st_mode
            if not stat.S_ISREG(mode):
                raise ContractError(f"vendor tree contains a non-regular file: {candidate}")
            paths.append(candidate)
    paths.sort(key=lambda item: item.relative_to(root).as_posix())
    if not paths:
        raise ContractError("vendor dependency directory contains no pinned files")

    inventory: list[dict[str, Any]] = []
    total_bytes = 0
    for candidate in paths:
        size = candidate.stat().st_size
        total_bytes += size
        inventory.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(candidate),
            }
        )
    provenance = {
        "schema": VENDOR_PROVENANCE_SCHEMA,
        "path": str(root),
        "tree_digest": sha256_json(
            {
                "schema": "policy-learnware.v02-vendor-tree.v0",
                "files": inventory,
            }
        ),
        "file_count": len(inventory),
        "total_bytes": total_bytes,
        "wandb_version": wandb_version,
    }
    return validate_vendor_provenance(provenance)


def require_vendor_pythonpath_first(
    vendor: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail unless the pinned directory precedes inherited Python paths."""

    validated = validate_vendor_provenance(vendor)
    environment = os.environ if environ is None else environ
    pythonpath = environment.get("PYTHONPATH", "")
    entries = [entry for entry in pythonpath.split(os.pathsep) if entry]
    if not entries:
        raise ContractError("runner PYTHONPATH does not contain the pinned vendor directory")
    try:
        first = Path(entries[0]).resolve(strict=True)
    except OSError as error:
        raise ContractError("runner PYTHONPATH begins with a missing directory") from error
    if first != Path(validated["path"]):
        raise ContractError("pinned vendor directory is not first on runner PYTHONPATH")


__all__ = ["inspect_vendor_directory", "require_vendor_pythonpath_first"]
