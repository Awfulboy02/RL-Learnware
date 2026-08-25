"""Content-address the exact server and legacy source bytes used by a run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .provenance import (
        ContractError,
        IMPLEMENTATION_FILE_LABELS,
        IMPLEMENTATION_PROVENANCE_SCHEMA,
        sha256_file,
        sha256_json,
        validate_implementation_provenance,
    )
except ImportError:  # pragma: no cover - exercised by executable entry points
    from provenance import (
        ContractError,
        IMPLEMENTATION_FILE_LABELS,
        IMPLEMENTATION_PROVENANCE_SCHEMA,
        sha256_file,
        sha256_json,
        validate_implementation_provenance,
    )


def _regular_file(path: Path | str, label: str) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ContractError(f"implementation source must not be a symlink: {label}")
    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"implementation source is missing: {label}: {source}") from error
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ContractError(f"implementation source is not a non-empty file: {label}")
    return resolved


def _package_source_root(server_root: Path) -> Path:
    here = Path(__file__).resolve()
    candidates = (
        here.parents[2] / "src",
        here.parents[1] / "policy_learnware_v0" / "src",
    )
    required = Path("policy_learnware_v0/policy/loader.py")
    matches = [candidate.resolve() for candidate in candidates if (candidate / required).is_file()]
    if len(matches) != 1:
        raise ContractError(
            "cannot uniquely locate the versioned policy_learnware_v0 source tree"
        )
    return matches[0]


def inspect_implementation_inventory(
    *,
    runner_path: Path | str,
    legacy_policy_io_path: Path | str,
) -> dict[str, Any]:
    """Return a stable logical-name → source-byte digest inventory.

    Logical labels make the digest stable across the tracked and sibling server
    deployment layouts. The actual runner and legacy exporter are explicit
    inputs; every other file has exactly one versioned location relative to
    this module or the package source root.
    """

    server_root = Path(__file__).resolve().parent
    package_source = _package_source_root(server_root)
    paths = {
        "v02/runner.py": _regular_file(runner_path, "v02/runner.py"),
        "v02/queue_master.py": _regular_file(
            server_root / "queue_master.py", "v02/queue_master.py"
        ),
        "v02/package_bridge.py": _regular_file(
            server_root / "package_bridge.py", "v02/package_bridge.py"
        ),
        "v02/anchor_binding.py": _regular_file(
            server_root / "anchor_binding.py", "v02/anchor_binding.py"
        ),
        "v02/provenance.py": _regular_file(
            server_root / "provenance.py", "v02/provenance.py"
        ),
        "v02/vendor.py": _regular_file(server_root / "vendor.py", "v02/vendor.py"),
        "v02/implementation.py": _regular_file(
            server_root / "implementation.py", "v02/implementation.py"
        ),
        "v02/formal_plan.py": _regular_file(
            server_root / "formal_plan.py", "v02/formal_plan.py"
        ),
        "legacy/policy_io.py": _regular_file(
            legacy_policy_io_path, "legacy/policy_io.py"
        ),
        "package/policy/bundle.py": _regular_file(
            package_source / "policy_learnware_v0/policy/bundle.py",
            "package/policy/bundle.py",
        ),
        "package/policy/evaluate.py": _regular_file(
            package_source / "policy_learnware_v0/policy/evaluate.py",
            "package/policy/evaluate.py",
        ),
        "package/policy/loader.py": _regular_file(
            package_source / "policy_learnware_v0/policy/loader.py",
            "package/policy/loader.py",
        ),
        "package/policy/parity.py": _regular_file(
            package_source / "policy_learnware_v0/policy/parity.py",
            "package/policy/parity.py",
        ),
        "package/v02/training.py": _regular_file(
            package_source / "policy_learnware_v0/v02/training.py",
            "package/v02/training.py",
        ),
    }
    if set(paths) != IMPLEMENTATION_FILE_LABELS:
        raise AssertionError("implementation inventory labels drifted from the validator")
    files = {
        label: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for label, path in sorted(paths.items())
    }
    material = {"schema": IMPLEMENTATION_PROVENANCE_SCHEMA, "files": files}
    return validate_implementation_provenance(
        {**material, "implementation_digest": sha256_json(material)}
    )


__all__ = ["inspect_implementation_inventory"]
