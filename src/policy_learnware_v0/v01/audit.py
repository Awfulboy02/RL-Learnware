"""Information-isolation and immutable-artifact audits for v0.1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np

from ..hashing import sha256_file, sha256_json


@dataclass(frozen=True)
class AuditViolation:
    path: str
    location: str
    forbidden_field: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "location": self.location,
            "forbidden_field": self.forbidden_field,
        }


def _normalise_fields(fields: Iterable[str]) -> frozenset[str]:
    result = frozenset(str(field).strip().lower() for field in fields)
    if not result or "" in result:
        raise ValueError("forbidden fields must be non-empty strings")
    return result


def _string_tokens(value: str) -> frozenset[str]:
    """Return conservative identifier tokens from a persisted string value.

    Measurement isolation applies to payload values as well as field names.  A
    token scan (rather than a substring scan) avoids treating schema words such
    as ``taskspec`` as the forbidden key ``task`` while still detecting values
    such as ``candidate_id`` or path fragments containing ``bundle-path``.
    """

    return frozenset(
        token.lower()
        for token in re.split(r"[^A-Za-z0-9_]+", value)
        if token
    )


def _scan_string(
    value: str,
    *,
    path: Path,
    location: str,
    forbidden: frozenset[str],
    violations: list[AuditViolation],
) -> None:
    for token in sorted(_string_tokens(value) & forbidden):
        violations.append(AuditViolation(str(path), location, token))


def _walk_json(
    value: Any,
    *,
    path: Path,
    location: str,
    forbidden: frozenset[str],
    violations: list[AuditViolation],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child = f"{location}.{key_text}" if location else key_text
            if key_text.lower() in forbidden:
                violations.append(AuditViolation(str(path), child, key_text.lower()))
            _walk_json(
                item,
                path=path,
                location=child,
                forbidden=forbidden,
                violations=violations,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_json(
                item,
                path=path,
                location=f"{location}[{index}]",
                forbidden=forbidden,
                violations=violations,
            )
    elif isinstance(value, str):
        _scan_string(
            value,
            path=path,
            location=location or "$",
            forbidden=forbidden,
            violations=violations,
        )


def scan_measurement_tree(
    measurement_root: str | Path, forbidden_fields: Iterable[str]
) -> tuple[AuditViolation, ...]:
    """Recursively scan JSON keys, CSV headers and NPZ member names."""

    root = Path(measurement_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"measurement root does not exist: {root}")
    forbidden = _normalise_fields(forbidden_fields)
    violations: list[AuditViolation] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            relative = path.resolve().relative_to(root)
        except ValueError as error:
            raise ValueError(f"measurement path escapes root: {path}") from error
        for part in relative.parts:
            stem_tokens = part.replace(".", "_").replace("-", "_").split("_")
            for token in stem_tokens:
                if token.lower() in forbidden:
                    violations.append(
                        AuditViolation(str(relative), "filename", token.lower())
                    )
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"cannot parse measurement JSON {relative}: {error}") from error
            _walk_json(
                value,
                path=relative,
                location="",
                forbidden=forbidden,
                violations=violations,
            )
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            header = rows[0] if rows else []
            for index, name in enumerate(header):
                if name.strip().lower() in forbidden:
                    violations.append(
                        AuditViolation(str(relative), f"header[{index}]", name.lower())
                    )
            for row_index, row in enumerate(rows[1:], start=1):
                for column_index, value in enumerate(row):
                    _scan_string(
                        value,
                        path=relative,
                        location=f"row[{row_index}][{column_index}]",
                        forbidden=forbidden,
                        violations=violations,
                    )
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    if name.lower() in forbidden:
                        violations.append(
                            AuditViolation(str(relative), f"member:{name}", name.lower())
                        )
                    if archive[name].dtype.hasobject:
                        raise TypeError(f"measurement NPZ {relative} contains object member {name}")
                    if archive[name].dtype.kind in {"U", "S"}:
                        for index, value in np.ndenumerate(archive[name]):
                            text = (
                                value.decode("utf-8", errors="strict")
                                if isinstance(value, bytes)
                                else str(value)
                            )
                            _scan_string(
                                text,
                                path=relative,
                                location=f"member:{name}{index}",
                                forbidden=forbidden,
                                violations=violations,
                            )
    return tuple(violations)


def assert_measurement_isolation(
    measurement_root: str | Path, forbidden_fields: Iterable[str]
) -> dict[str, Any]:
    violations = scan_measurement_tree(measurement_root, forbidden_fields)
    return {
        "schema": "policy-learnware.v01-measurement-isolation-audit.v0",
        "passed": not violations,
        "measurement_root_digest": measurement_tree_digest(measurement_root),
        "violations": [item.to_dict() for item in violations],
    }


def assert_measurement_schema_allowlist(
    measurement_root: str | Path,
) -> dict[str, Any]:
    """Enforce the versioned top-level schema/member contract for every file."""

    root = Path(measurement_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"measurement root does not exist: {root}")
    json_keys = {
        "run_ref.json": {
            "schema", "measurement_protocol_id", "measurement_run_id",
            "measurement_protocol_sha256", "base_protocol_ref",
            "measurement_contract_digest", "pair_plan_digest",
            "schema_view_digests", "formal", "git", "runtime_versions",
            "measurement_component_digests",
        },
        "measurement_contract.json": {
            "schema", "measurement_protocol_id", "base_protocol_id",
            "probe_banks", "episodes_per_bank", "prefix_grid", "gate_prefix",
            "pair_plan_digest", "variant_ids", "schema_view_digests", "visibility",
        },
        "pair_plan.json": {"schema", "within", "between", "routing", "plan_digest"},
        "taskspec_matrix_axes.json": {
            "schema", "plan_digest", "pair_rows", "routing_rows",
            "self_norm_rows", "clamp_count",
        },
        "taskspec_primitive_manifest.json": {
            "schema", "plan_digest", "primitive_digest",
            "taskspec_matrix_sha256", "semantic_manifest_sha256",
            "semantic_content_digest",
        },
    }
    schema_view_keys = {
        "schema", "observation_dim", "action_dim", "observation_dtype",
        "action_dtype", "action_low", "action_high", "horizon",
        "action_repeat", "control_dt", "flatten_fingerprint_without_task",
    }
    dataset_manifest_keys = {
        "variant_id", "bank", "episode_count", "transition_count",
        "reset_seeds", "probe_seeds", "dataset_digest", "base_protocol_id",
        "measurement_contract_digest", "measurement_schema_view_digest", "schema",
    }
    semantic_manifest_keys = {
        "schema", "variant_id", "bank", "dataset_digest",
        "measurement_schema_view_digest", "base_binding_digest",
        "normalization_sha256", "encoder_checkpoint_sha256",
        "encoder_config_sha256", "cache_sha256",
    }
    dataset_members = {
        "observation", "action", "reward", "next_observation", "terminated",
        "truncated", "episode_offsets", "reset_seeds", "probe_seeds",
    }
    semantic_members = {"points", "weights", "episode_offsets"}
    matrix_members = {"family", "d_phi", "raw_mmd2", "mmd2"}
    csv_headers = {
        "taskspec_matrix.csv": (
            "family", "pair_index", "left_variant_id", "left_bank",
            "right_variant_id", "right_bank", "prefix", "raw_mmd2", "mmd2",
            "d_phi", "roundoff_clamped", "cross_term",
        ),
        "routing_matrix.csv": (
            "routing_index", "variant_id", "bank", "prefix", "selected_source_id",
        ),
    }
    violations: list[dict[str, Any]] = []
    file_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        file_count += 1
        try:
            path.resolve().relative_to(root)
        except ValueError:
            violations.append(
                {"path": str(relative), "reason": "artifact_path_escapes_measurement"}
            )
            # Never parse or hash bytes reached through an escaping symlink.
            continue
        expected: set[str] | tuple[str, ...] | None = None
        execution_attempt_json = (
            len(relative.parts) == 2
            and relative.parts[0] == "execution_attempts"
            and path.suffix == ".json"
        )
        if relative.as_posix() in json_keys:
            expected = json_keys[relative.as_posix()]
        elif len(relative.parts) == 2 and relative.parts[0] == "schema_views" and path.suffix == ".json":
            expected = schema_view_keys
        elif (
            len(relative.parts) == 4
            and relative.parts[0] == "datasets"
            and relative.name == "manifest.json"
        ):
            expected = dataset_manifest_keys
        elif len(relative.parts) == 3 and relative.parts[0] == "semantic_cache" and path.suffix == ".json":
            expected = semantic_manifest_keys

        if path.suffix == ".json":
            if execution_attempt_json:
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    from .execution_profile import validate_execution_attempt_payload

                    validated = validate_execution_attempt_payload(value)
                    if validated["execution_attempt_id"] != path.stem:
                        raise ValueError("execution attempt filename/ID mismatch")
                except (OSError, json.JSONDecodeError, ValueError) as error:
                    violations.append(
                        {
                            "path": str(relative),
                            "reason": "execution_attempt_schema_mismatch",
                            "detail": type(error).__name__,
                        }
                    )
                continue
            if expected is None:
                violations.append(
                    {"path": str(relative), "reason": "unregistered_json_artifact"}
                )
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping) or set(value) != set(expected):
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "json_schema_key_mismatch",
                        "expected": sorted(expected),
                        "observed": sorted(value) if isinstance(value, Mapping) else None,
                    }
                )
        elif path.suffix == ".npz":
            if len(relative.parts) == 4 and relative.parts[0] == "datasets":
                expected_members = dataset_members
            elif len(relative.parts) == 3 and relative.parts[0] == "semantic_cache":
                expected_members = semantic_members
            elif relative.as_posix() == "taskspec_matrix.npz":
                expected_members = matrix_members
            else:
                expected_members = set()
            with np.load(path, allow_pickle=False) as archive:
                observed_members = set(archive.files)
            if observed_members != expected_members:
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "npz_member_allowlist_mismatch",
                        "expected": sorted(expected_members),
                        "observed": sorted(observed_members),
                    }
                )
        elif path.suffix == ".csv":
            expected_header = csv_headers.get(relative.as_posix())
            with path.open("r", encoding="utf-8", newline="") as handle:
                observed_header = tuple(next(csv.reader(handle), []))
            if expected_header is None or observed_header != expected_header:
                violations.append(
                    {
                        "path": str(relative),
                        "reason": "csv_header_allowlist_mismatch",
                        "expected": expected_header,
                        "observed": observed_header,
                    }
                )
        else:
            violations.append(
                {"path": str(relative), "reason": "unregistered_artifact_type"}
            )
    return {
        "schema": "policy-learnware.v01-measurement-schema-allowlist-audit.v0",
        "passed": not violations,
        "file_count": file_count,
        "violations": violations,
    }


def measurement_tree_digest(measurement_root: str | Path) -> str:
    root = Path(measurement_root).resolve()
    files = {
        str(path.resolve().relative_to(root)): sha256_file(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }
    return sha256_json(files)


def assert_no_oracle_dependencies(paths: Iterable[str | Path]) -> None:
    for raw in paths:
        parts = {part.lower() for part in Path(raw).parts}
        if "oracle_private" in parts or "benchmark_private" in parts:
            raise PermissionError(
                "TaskSpec computation cannot read oracle_private or benchmark_private"
            )


__all__ = [
    "AuditViolation",
    "assert_measurement_schema_allowlist",
    "assert_measurement_isolation",
    "assert_no_oracle_dependencies",
    "measurement_tree_digest",
    "scan_measurement_tree",
]
