"""Execution-only TaskSpec attempt records and P5 resource profiles.

These records deliberately live outside the mathematical protocol identity.
They bind an execution attempt to immutable measurement inputs and published
outputs, while allowing block size/backend changes after an OOM.  A failed
attempt may be retained for audit, but can never authorize matrix merge or a
``NO_GO_COMPUTE`` preflight decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import resource
import secrets
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..hashing import sha256_file, sha256_json, sha256_ndarrays


ATTEMPT_SCHEMA = "policy-learnware.v01-taskspec-execution-attempt.v0"
EXTRAPOLATION_SCHEMA = "policy-learnware.v01-p5-resource-extrapolation.v0"
ATTEMPT_PREFIX = "v01xa-"

# The only smoke contract approved for a P5 resource decision.  This digest is
# the typed/canonical ``V01ExperimentConfig.config_digest`` rather than a YAML
# byte hash, so comments and formatting cannot change the identity.
APPROVED_P5_SMOKE_CONFIG_DIGEST = (
    "c88abc2bac693fb50a0bf3d6fcc3f8e028d3aa1e9efa5b3c0c5b7ddbe8b88c24"
)

# Formal matrix/audit shapes frozen by the approved coding plan.  Targets are
# recomputed below for the attempt's backend and block size; these are not run
# identity inputs.
FORMAL_VARIANT_COUNT = 10
FORMAL_DATASET_COUNT = 100
FORMAL_PAIR_SELF_COUNT = 100
FORMAL_PAIR_CROSS_COUNT = 130
FORMAL_ROUTING_COUNT = 100
FORMAL_GATE_PREFIX_TRANSITIONS = 16_000
FORMAL_ROUTING_PREFIX_TRANSITIONS = 64_000
FORMAL_MATRIX_SEMANTIC_TRANSITIONS = 6_400_000
FORMAL_RECOMPUTE_SELF_COUNT = 6
FORMAL_RECOMPUTE_CROSS_COUNT = 4
FORMAL_RECOMPUTE_ROUTING_COUNT = 2
# audit-recompute rebuilds all 100 semantic caches from raw datasets.
FORMAL_RECOMPUTE_SEMANTIC_TRANSITIONS = 6_400_000

_ATTEMPT_ID = re.compile(r"^v01xa-[0-9a-f]{24}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_VARIANT_ID = re.compile(r"^v01v-[0-9a-f]{20}$")
_SAFE_DEVICE_STRING = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+()=-]{0,79}$")
_SAFE_RUNTIME_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
_SAFE_RUNTIME_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+:/()=#~,-]{0,159}$")
_KNOWN_DEVICE_PLATFORMS = frozenset(
    {"cpu", "gpu", "tpu", "metal", "unavailable", "unknown"}
)
_SAFE_ERROR_TYPES = frozenset(
    {
        "ArtifactConflict",
        "InputValidationFailure",
        "MemoryFailure",
        "RuntimeFailure",
        "UnexpectedFailure",
    }
)
_SAFE_REASON_CODES = frozenset(
    {
        "ARTIFACT_CONFLICT",
        "INPUT_VALIDATION_FAILED",
        "OUT_OF_MEMORY",
        "RUNTIME_ERROR",
        "UNEXPECTED_ERROR",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "execution_attempt_id",
        "status",
        "measurement_run_id",
        "input_binding",
        "input_digest",
        "semantic_input_binding",
        "semantic_input_digest",
        "execution",
        "workload",
        "resource_profile",
        "resource_extrapolation",
        "output_artifact_sha256",
        "output_digest",
        "failure",
        "attempt_digest",
    }
)
_INPUT_KEYS = frozenset(
    {
        "run_ref_sha256",
        "measurement_contract_sha256",
        "pair_plan_sha256",
        "base_binding_digest",
        "base_assets_digest",
        "dataset_artifact_sha256",
        "dataset_content_digest",
    }
)
_SEMANTIC_KEYS = frozenset(
    {"semantic_manifest_sha256", "semantic_content_digest"}
)
_EXECUTION_KEYS = frozenset(
    {"block_size", "computation_backend", "device", "runtime_versions"}
)
_DEVICE_KEYS = frozenset(
    {"requested_backend", "platforms", "device_kinds", "device_count"}
)
_WORKLOAD_KEYS = frozenset(
    {
        "semantic_dataset_count",
        "semantic_transition_count",
        "semantic_cache_hits",
        "semantic_cache_misses",
        "pair_count",
        "routing_count",
        "source_support_sizes",
        "mathematical_self_kernel_entries",
        "mathematical_pair_cross_kernel_entries",
        "mathematical_routing_cross_kernel_entries",
        "mathematical_total_kernel_entries",
        "padded_self_block_entries",
        "padded_pair_cross_block_entries",
        "padded_routing_cross_block_entries",
        "padded_total_block_entries",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "wall_time_seconds",
        "rss_start_bytes",
        "rss_peak_bytes",
        "rss_peak_delta_bytes",
        "device_peak_memory_bytes",
        "device_peak_memory_profile_available",
    }
)
_EXTRAPOLATION_KEYS = frozenset(
    {
        "schema",
        "method",
        "formal_target_backend",
        "formal_target_block_size",
        "observed_padded_block_entries",
        "observed_wall_time_seconds",
        "observed_padded_block_entries_per_second",
        "formal_matrix_mathematical_kernel_entries",
        "formal_recompute_mathematical_kernel_entries",
        "formal_matrix_padded_block_entries",
        "formal_recompute_padded_block_entries",
        "formal_total_padded_block_entries",
        "formal_matrix_semantic_transitions",
        "formal_recompute_semantic_transitions",
        "kernel_scaling_ratio",
        "semantic_scaling_ratio",
        "conservative_scaling_ratio",
        "kernel_only_projected_wall_time_seconds",
        "projected_formal_wall_time_seconds",
        "assumptions",
    }
)
_FAILURE_KEYS = frozenset({"error_type", "reason_code"})
_OUTPUT_NAMES = (
    "routing_matrix.csv",
    "taskspec_matrix.csv",
    "taskspec_matrix.npz",
    "taskspec_matrix_axes.json",
    "taskspec_primitive_manifest.json",
)


class ExecutionProfileError(ValueError):
    """An execution attempt/profile is malformed, stale, or unsuccessful."""


def _require_object(value: Any, keys: frozenset[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = sorted(value) if isinstance(value, Mapping) else None
        raise ExecutionProfileError(
            f"{where} exact-schema mismatch: expected={sorted(keys)}, observed={observed}"
        )
    return value


def _require_digest(value: Any, where: str) -> str:
    digest = str(value).lower()
    if not _DIGEST.fullmatch(digest):
        raise ExecutionProfileError(f"{where} is not a SHA-256 digest")
    return digest


def _require_nonnegative_int(value: Any, where: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExecutionProfileError(f"{where} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ExecutionProfileError(f"{where} must be {qualifier}")
    return int(value)


def _require_nonnegative_float(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionProfileError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < (0.0 if not positive else sys.float_info.min):
        qualifier = "positive finite" if positive else "finite and non-negative"
        raise ExecutionProfileError(f"{where} must be {qualifier}")
    return result


def _contained_file(root: Path, relative: str, where: str) -> Path:
    """Resolve a public input without following a path outside measurement."""

    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ExecutionProfileError(f"{where} relative path is invalid")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ExecutionProfileError(f"{where} escapes the measurement root") from error
    if not candidate.is_file():
        raise ExecutionProfileError(f"{where} is missing or not a regular file")
    return candidate


def _contained_directory(root: Path, relative: str, where: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ExecutionProfileError(f"{where} relative path is invalid")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ExecutionProfileError(f"{where} escapes the measurement root") from error
    if not candidate.is_dir():
        raise ExecutionProfileError(f"{where} is missing or not a directory")
    return candidate


def new_execution_attempt_id() -> str:
    """Return a non-semantic, collision-resistant execution-attempt ID."""

    return ATTEMPT_PREFIX + secrets.token_hex(12)


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the server runtime report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _device_snapshot(computation_backend: str) -> tuple[dict[str, Any], int | None]:
    backend = str(computation_backend)
    if backend == "numpy":
        return (
            {
                "requested_backend": backend,
                "platforms": ["cpu"],
                "device_kinds": [platform.machine() or "unknown-cpu"],
                "device_count": 1,
            },
            None,
        )
    if backend != "jax":
        raise ExecutionProfileError("computation backend must be numpy or jax")
    try:
        import jax

        devices = tuple(jax.devices())
    except Exception:
        return (
            {
                "requested_backend": backend,
                "platforms": ["unavailable"],
                "device_kinds": ["unavailable"],
                "device_count": 0,
            },
            None,
        )
    platforms = sorted(
        {str(getattr(device, "platform", "unknown")).lower() for device in devices}
    )
    kinds = sorted({str(getattr(device, "device_kind", "unknown")) for device in devices})
    peaks: list[int] = []
    for device in devices:
        try:
            stats = device.memory_stats()
        except Exception:
            stats = None
        if not isinstance(stats, Mapping):
            continue
        for key in ("peak_bytes_in_use", "peak_bytes_reserved"):
            value = stats.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                peaks.append(int(value))
    return (
        {
            "requested_backend": backend,
            "platforms": platforms or ["unknown"],
            "device_kinds": kinds or ["unknown"],
            "device_count": len(devices),
        },
        max(peaks) if peaks else None,
    )


def runtime_versions() -> dict[str, str]:
    """Small path-free runtime projection persisted in public measurement."""

    result = {"python": platform.python_version(), "platform": platform.platform()}
    for distribution in (
        "jax",
        "jaxlib",
        "flax",
        "numpy",
        "scipy",
        "mujoco",
        "playground",
        "mujoco-playground",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def build_initial_input_binding(measurement_root: str | Path) -> tuple[str, dict[str, Any]]:
    """Bind raw public inputs without hashing the mutable measurement tree."""

    root = Path(measurement_root).expanduser().resolve()
    fixed_paths = {
        "run_ref_sha256": _contained_file(root, "run_ref.json", "run reference"),
        "measurement_contract_sha256": _contained_file(
            root, "measurement_contract.json", "measurement contract"
        ),
        "pair_plan_sha256": _contained_file(root, "pair_plan.json", "pair plan"),
    }
    try:
        run_ref = json.loads(fixed_paths["run_ref_sha256"].read_text(encoding="utf-8"))
        contract = json.loads(
            fixed_paths["measurement_contract_sha256"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionProfileError("cannot read frozen public execution inputs") from error
    if not isinstance(run_ref, Mapping) or not isinstance(contract, Mapping):
        raise ExecutionProfileError("frozen public execution inputs must be objects")
    components = run_ref.get("measurement_component_digests")
    if not isinstance(components, Mapping):
        raise ExecutionProfileError("run reference lacks measurement components")
    base_binding = _require_digest(components.get("base_binding"), "base binding")
    base_assets = _require_digest(components.get("base_assets"), "base assets")
    variant_ids = contract.get("variant_ids")
    banks = contract.get("probe_banks")
    if not isinstance(variant_ids, list):
        raise ExecutionProfileError("measurement contract variant IDs are malformed")
    bank_count = _require_nonnegative_int(banks, "probe bank count", positive=True)
    artifact_map: dict[str, str] = {}
    content_map: dict[str, str] = {}
    for variant_id in variant_ids:
        if not isinstance(variant_id, str) or not _OPAQUE_VARIANT_ID.fullmatch(variant_id):
            raise ExecutionProfileError("measurement contract contains an invalid variant ID")
        for bank in range(bank_count):
            unit = f"{variant_id}/bank_{bank:03d}"
            data_relative = f"datasets/{unit}/dataset.npz"
            manifest_relative = f"datasets/{unit}/manifest.json"
            data_path = _contained_file(root, data_relative, "dataset input")
            manifest_path = _contained_file(
                root, manifest_relative, "dataset manifest"
            )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ExecutionProfileError("cannot read a registered dataset input") from error
            if not isinstance(manifest, Mapping):
                raise ExecutionProfileError("dataset manifest must be an object")
            artifact_map[data_relative] = sha256_file(data_path)
            artifact_map[manifest_relative] = sha256_file(manifest_path)
            content_map[unit] = _require_digest(
                manifest.get("dataset_digest"), "dataset content digest"
            )
    binding = {
        **{name: sha256_file(path) for name, path in fixed_paths.items()},
        "base_binding_digest": base_binding,
        "base_assets_digest": base_assets,
        "dataset_artifact_sha256": dict(sorted(artifact_map.items())),
        "dataset_content_digest": dict(sorted(content_map.items())),
    }
    return _require_digest(run_ref.get("measurement_run_id"), "measurement run ID"), binding


def build_semantic_input_binding(measurement_root: str | Path) -> dict[str, Any]:
    """Bind every derived semantic cache used by the successful matrix."""

    root = Path(measurement_root).expanduser().resolve()
    primitive_path = _contained_file(
        root, "taskspec_primitive_manifest.json", "TaskSpec primitive manifest"
    )
    try:
        primitive = json.loads(primitive_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionProfileError("cannot read TaskSpec primitive manifest") from error
    if not isinstance(primitive, Mapping):
        raise ExecutionProfileError("TaskSpec primitive manifest must be an object")
    manifest_map = primitive.get("semantic_manifest_sha256")
    content_map = primitive.get("semantic_content_digest")
    try:
        contract = json.loads(
            _contained_file(
                root, "measurement_contract.json", "measurement contract"
            ).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionProfileError("cannot read measurement contract") from error
    if not isinstance(contract, Mapping):
        raise ExecutionProfileError("measurement contract must be an object")
    variant_ids = contract.get("variant_ids")
    bank_count = _require_nonnegative_int(
        contract.get("probe_banks"), "probe bank count", positive=True
    )
    if not isinstance(variant_ids, list) or any(
        not isinstance(item, str) or not _OPAQUE_VARIANT_ID.fullmatch(item)
        for item in variant_ids
    ):
        raise ExecutionProfileError("measurement contract variant IDs are malformed")
    expected_units = {
        f"{variant_id}/bank_{bank:03d}"
        for variant_id in variant_ids
        for bank in range(bank_count)
    }
    if (
        not isinstance(manifest_map, Mapping)
        or not isinstance(content_map, Mapping)
        or set(manifest_map) != set(content_map)
        or set(manifest_map) != expected_units
    ):
        raise ExecutionProfileError("semantic input coverage is incomplete")
    verified_manifests: dict[str, str] = {}
    verified_content: dict[str, str] = {}
    for unit in sorted(manifest_map):
        if not isinstance(unit, str) or "/bank_" not in unit:
            raise ExecutionProfileError("semantic input unit ID is malformed")
        variant_id, bank_name = unit.split("/", 1)
        if not _OPAQUE_VARIANT_ID.fullmatch(variant_id):
            raise ExecutionProfileError("semantic input variant ID is malformed")
        try:
            bank = int(bank_name.removeprefix("bank_"))
        except ValueError as error:
            raise ExecutionProfileError("semantic input bank ID is malformed") from error
        manifest_path = _contained_file(
            root,
            f"semantic_cache/{variant_id}/bank_{bank:03d}.json",
            "semantic cache manifest",
        )
        cache_path = _contained_file(
            root,
            f"semantic_cache/{variant_id}/bank_{bank:03d}.npz",
            "semantic cache",
        )
        expected_manifest = _require_digest(
            manifest_map[unit], "semantic manifest digest"
        )
        if sha256_file(manifest_path) != expected_manifest:
            raise ExecutionProfileError("semantic manifest digest mismatch")
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
        except (OSError, ValueError) as error:
            raise ExecutionProfileError("cannot read semantic cache") from error
        if set(arrays) != {"points", "weights", "episode_offsets"}:
            raise ExecutionProfileError("semantic cache member mismatch")
        expected_content = _require_digest(
            content_map[unit], "semantic content digest"
        )
        if sha256_ndarrays(arrays) != expected_content:
            raise ExecutionProfileError("semantic content digest mismatch")
        verified_manifests[unit] = expected_manifest
        verified_content[unit] = expected_content
    return {
        "semantic_manifest_sha256": verified_manifests,
        "semantic_content_digest": verified_content,
    }


def output_artifact_digests(measurement_root: str | Path) -> dict[str, str]:
    root = Path(measurement_root).expanduser().resolve()
    return {
        name: sha256_file(_contained_file(root, name, "TaskSpec output"))
        for name in _OUTPUT_NAMES
    }


def estimate_taskspec_workload(
    samples: Mapping[tuple[str, int], Any],
    pair_plan: Mapping[str, Any],
    sources: Mapping[str, Any],
    *,
    block_size: int,
    computation_backend: str,
    semantic_cache_hits: int = 0,
    semantic_cache_misses: int | None = None,
) -> dict[str, int]:
    """Count mathematical terms and backend/block-level Gram evaluations."""

    support_sizes = tuple(
        sorted(int(np.asarray(source.supports).shape[0]) for source in sources.values())
    )
    return _estimate_workload_from_samples(
        samples,
        pair_plan,
        source_support_sizes=support_sizes,
        block_size=block_size,
        computation_backend=computation_backend,
        semantic_cache_hits=semantic_cache_hits,
        semantic_cache_misses=(
            len(samples) if semantic_cache_misses is None else semantic_cache_misses
        ),
    )


def _block_lengths(count: int, block_size: int) -> tuple[int, ...]:
    if count <= 0 or block_size <= 0:
        raise ExecutionProfileError("kernel count/block size must be positive")
    full, remainder = divmod(count, block_size)
    return (block_size,) * full + ((remainder,) if remainder else ())


def _self_block_entries(count: int, block_size: int, backend: str) -> int:
    lengths = _block_lengths(count, block_size)
    if backend == "jax":
        blocks = len(lengths)
        return blocks * (blocks + 1) // 2 * block_size * block_size
    if backend == "numpy":
        return sum(
            left * right
            for index, left in enumerate(lengths)
            for right in lengths[index:]
        )
    raise ExecutionProfileError("computation backend must be numpy or jax")


def _cross_block_entries(
    left_count: int, right_count: int, block_size: int, backend: str
) -> int:
    left = _block_lengths(left_count, block_size)
    right = _block_lengths(right_count, block_size)
    if backend == "jax":
        return len(left) * len(right) * block_size * block_size
    if backend == "numpy":
        return sum(left) * sum(right)
    raise ExecutionProfileError("computation backend must be numpy or jax")


def _estimate_workload_from_samples(
    samples: Mapping[tuple[str, int], Any],
    pair_plan: Mapping[str, Any],
    *,
    source_support_sizes: tuple[int, ...],
    block_size: int,
    computation_backend: str,
    semantic_cache_hits: int,
    semantic_cache_misses: int,
) -> dict[str, Any]:
    backend = str(computation_backend)
    block = _require_nonnegative_int(block_size, "block size", positive=True)
    if backend not in {"numpy", "jax"}:
        raise ExecutionProfileError("computation backend must be numpy or jax")
    if not source_support_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in source_support_sizes
    ):
        raise ExecutionProfileError("source support sizes must be positive integers")

    prefix_sizes: dict[tuple[str, int, int], int] = {}

    def size(variant_id: str, bank: int, prefix: int) -> int:
        key = (str(variant_id), int(bank), int(prefix))
        if key not in prefix_sizes:
            sample = samples[(key[0], key[1])]
            offsets = np.asarray(sample.episode_offsets, dtype=np.int64)
            if key[2] <= 0 or key[2] >= offsets.size:
                raise ExecutionProfileError("pair plan prefix exceeds semantic sample")
            prefix_sizes[key] = int(offsets[key[2]])
        return prefix_sizes[key]

    self_keys: set[tuple[str, int, int]] = set()
    mathematical_pair_cross = 0
    padded_pair_cross = 0
    pair_count = 0
    for family in ("within", "between"):
        records = pair_plan.get(family)
        if not isinstance(records, list):
            raise ExecutionProfileError("pair plan family is malformed")
        for record in records:
            left = (
                str(record["left_variant_id"]),
                int(record["left_bank"]),
                int(record["prefix"]),
            )
            right = (
                str(record["right_variant_id"]),
                int(record["right_bank"]),
                int(record["prefix"]),
            )
            self_keys.update((left, right))
            left_size = size(*left)
            right_size = size(*right)
            mathematical_pair_cross += left_size * right_size
            padded_pair_cross += _cross_block_entries(
                left_size, right_size, block, backend
            )
            pair_count += 1
    mathematical_self = sum(size(*key) ** 2 for key in self_keys)
    padded_self = sum(
        _self_block_entries(size(*key), block, backend) for key in self_keys
    )
    source_supports = sum(source_support_sizes)
    mathematical_routing = 0
    padded_routing = 0
    routing_records = pair_plan.get("routing")
    if not isinstance(routing_records, list):
        raise ExecutionProfileError("routing plan is malformed")
    for record in routing_records:
        target_size = size(
            str(record["variant_id"]), int(record["bank"]), int(record["prefix"])
        )
        mathematical_routing += target_size * source_supports
        padded_routing += sum(
            _cross_block_entries(target_size, support, block, backend)
            for support in source_support_sizes
        )
    transition_count = sum(int(np.asarray(sample.points).shape[0]) for sample in samples.values())
    mathematical_total = (
        mathematical_self + mathematical_pair_cross + mathematical_routing
    )
    padded_total = padded_self + padded_pair_cross + padded_routing
    hits = _require_nonnegative_int(semantic_cache_hits, "semantic cache hits")
    misses = _require_nonnegative_int(semantic_cache_misses, "semantic cache misses")
    if hits + misses != len(samples):
        raise ExecutionProfileError("semantic cache hit/miss coverage mismatch")
    if mathematical_total <= 0 or padded_total <= 0:
        raise ExecutionProfileError("observed kernel-entry workload must be positive")
    return {
        "semantic_dataset_count": len(samples),
        "semantic_transition_count": transition_count,
        "semantic_cache_hits": hits,
        "semantic_cache_misses": misses,
        "pair_count": pair_count,
        "routing_count": len(routing_records),
        "source_support_sizes": list(source_support_sizes),
        "mathematical_self_kernel_entries": mathematical_self,
        "mathematical_pair_cross_kernel_entries": mathematical_pair_cross,
        "mathematical_routing_cross_kernel_entries": mathematical_routing,
        "mathematical_total_kernel_entries": mathematical_total,
        "padded_self_block_entries": padded_self,
        "padded_pair_cross_block_entries": padded_pair_cross,
        "padded_routing_cross_block_entries": padded_routing,
        "padded_total_block_entries": padded_total,
    }


def rebuild_live_workload(
    measurement_root: str | Path,
    *,
    source_support_sizes: tuple[int, ...],
    block_size: int,
    computation_backend: str,
    semantic_cache_hits: int,
    semantic_cache_misses: int,
) -> dict[str, Any]:
    """Rebuild workload from the live public pair plan and semantic caches."""

    root = Path(measurement_root).expanduser().resolve()
    try:
        contract = json.loads(
            _contained_file(root, "measurement_contract.json", "measurement contract").read_text(
                encoding="utf-8"
            )
        )
        pair_plan = json.loads(
            _contained_file(root, "pair_plan.json", "pair plan").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionProfileError("cannot read workload inputs") from error
    if not isinstance(contract, Mapping) or not isinstance(pair_plan, Mapping):
        raise ExecutionProfileError("workload inputs must be objects")
    variants = contract.get("variant_ids")
    banks = _require_nonnegative_int(
        contract.get("probe_banks"), "probe bank count", positive=True
    )
    if not isinstance(variants, list) or any(
        not isinstance(item, str) or not _OPAQUE_VARIANT_ID.fullmatch(item)
        for item in variants
    ):
        raise ExecutionProfileError("workload variant coverage is malformed")
    samples: dict[tuple[str, int], Any] = {}
    for variant_id in variants:
        for bank in range(banks):
            cache = _contained_file(
                root,
                f"semantic_cache/{variant_id}/bank_{bank:03d}.npz",
                "semantic cache",
            )
            try:
                with np.load(cache, allow_pickle=False) as archive:
                    if set(archive.files) != {"points", "weights", "episode_offsets"}:
                        raise ExecutionProfileError("semantic cache member mismatch")
                    point_count = int(archive["points"].shape[0])
                    offsets = np.asarray(archive["episode_offsets"], dtype=np.int64)
            except (OSError, ValueError) as error:
                raise ExecutionProfileError("cannot read semantic cache workload") from error
            samples[(variant_id, bank)] = SimpleNamespace(
                points=np.empty((point_count, 0), dtype=np.float64),
                episode_offsets=offsets,
            )
    return _estimate_workload_from_samples(
        samples,
        pair_plan,
        source_support_sizes=source_support_sizes,
        block_size=block_size,
        computation_backend=computation_backend,
        semantic_cache_hits=semantic_cache_hits,
        semantic_cache_misses=semantic_cache_misses,
    )


def _formal_targets(
    source_support_sizes: tuple[int, ...], block_size: int, backend: str
) -> dict[str, int]:
    support_total = sum(source_support_sizes)
    n_gate = FORMAL_GATE_PREFIX_TRANSITIONS
    n_route = FORMAL_ROUTING_PREFIX_TRANSITIONS
    matrix_math = (
        FORMAL_PAIR_SELF_COUNT * n_gate * n_gate
        + FORMAL_PAIR_CROSS_COUNT * n_gate * n_gate
        + FORMAL_ROUTING_COUNT * n_route * support_total
    )
    recompute_math = (
        FORMAL_RECOMPUTE_SELF_COUNT * n_gate * n_gate
        + FORMAL_RECOMPUTE_CROSS_COUNT * n_gate * n_gate
        + FORMAL_RECOMPUTE_ROUTING_COUNT * n_route * support_total
    )
    matrix_padded = (
        FORMAL_PAIR_SELF_COUNT * _self_block_entries(n_gate, block_size, backend)
        + FORMAL_PAIR_CROSS_COUNT
        * _cross_block_entries(n_gate, n_gate, block_size, backend)
        + FORMAL_ROUTING_COUNT
        * sum(
            _cross_block_entries(n_route, support, block_size, backend)
            for support in source_support_sizes
        )
    )
    recompute_padded = (
        FORMAL_RECOMPUTE_SELF_COUNT
        * _self_block_entries(n_gate, block_size, backend)
        + FORMAL_RECOMPUTE_CROSS_COUNT
        * _cross_block_entries(n_gate, n_gate, block_size, backend)
        + FORMAL_RECOMPUTE_ROUTING_COUNT
        * sum(
            _cross_block_entries(n_route, support, block_size, backend)
            for support in source_support_sizes
        )
    )
    return {
        "matrix_math": matrix_math,
        "recompute_math": recompute_math,
        "matrix_padded": matrix_padded,
        "recompute_padded": recompute_padded,
    }


def _resource_extrapolation(
    workload: Mapping[str, Any],
    wall_time: float,
    *,
    block_size: int,
    computation_backend: str,
) -> dict[str, Any]:
    observed = _require_nonnegative_int(
        workload.get("padded_total_block_entries"),
        "observed padded block entries",
        positive=True,
    )
    transitions = _require_nonnegative_int(
        workload.get("semantic_transition_count"),
        "observed semantic transitions",
        positive=True,
    )
    support_sizes = tuple(int(item) for item in workload["source_support_sizes"])
    targets = _formal_targets(support_sizes, int(block_size), str(computation_backend))
    rate = observed / wall_time
    kernel_ratio = (targets["matrix_padded"] + targets["recompute_padded"]) / observed
    semantic_ratio = (
        FORMAL_MATRIX_SEMANTIC_TRANSITIONS
        + FORMAL_RECOMPUTE_SEMANTIC_TRANSITIONS
    ) / transitions
    conservative_ratio = max(kernel_ratio, semantic_ratio)
    return {
        "schema": EXTRAPOLATION_SCHEMA,
        "method": "conservative_backend_block_and_semantic_scaling_v1",
        "formal_target_backend": str(computation_backend),
        "formal_target_block_size": int(block_size),
        "observed_padded_block_entries": observed,
        "observed_wall_time_seconds": wall_time,
        "observed_padded_block_entries_per_second": rate,
        "formal_matrix_mathematical_kernel_entries": targets["matrix_math"],
        "formal_recompute_mathematical_kernel_entries": targets["recompute_math"],
        "formal_matrix_padded_block_entries": targets["matrix_padded"],
        "formal_recompute_padded_block_entries": targets["recompute_padded"],
        "formal_total_padded_block_entries": (
            targets["matrix_padded"] + targets["recompute_padded"]
        ),
        "formal_matrix_semantic_transitions": FORMAL_MATRIX_SEMANTIC_TRANSITIONS,
        "formal_recompute_semantic_transitions": FORMAL_RECOMPUTE_SEMANTIC_TRANSITIONS,
        "kernel_scaling_ratio": kernel_ratio,
        "semantic_scaling_ratio": semantic_ratio,
        "conservative_scaling_ratio": conservative_ratio,
        "kernel_only_projected_wall_time_seconds": wall_time * kernel_ratio,
        "projected_formal_wall_time_seconds": wall_time * conservative_ratio,
        "assumptions": [
            "same_backend_and_device_family",
            "backend_block_evaluations_scale_linearly",
            "semantic_encoding_scales_linearly",
            "primary_projection_uses_larger_component_ratio",
            "includes_full_semantic_rebuild_and_preregistered_raw_audit",
            "engineering_resource_evidence_only",
        ],
    }


@dataclass(frozen=True)
class AttemptTimer:
    attempt_id: str
    started_monotonic: float
    rss_start_bytes: int

    @classmethod
    def start(cls) -> "AttemptTimer":
        return cls(new_execution_attempt_id(), time.monotonic(), _max_rss_bytes())


def classify_failure(error: Exception) -> dict[str, str]:
    """Map arbitrary exceptions to a bounded, path-free public failure code."""

    name = type(error).__name__.lower()
    if isinstance(error, MemoryError) or "outofmemory" in name or "resourceexhausted" in name:
        return {"error_type": "MemoryFailure", "reason_code": "OUT_OF_MEMORY"}
    if isinstance(error, FileExistsError) or "artifact" in name and "exist" in name:
        return {"error_type": "ArtifactConflict", "reason_code": "ARTIFACT_CONFLICT"}
    if isinstance(error, (ValueError, TypeError, FileNotFoundError, ExecutionProfileError)):
        return {
            "error_type": "InputValidationFailure",
            "reason_code": "INPUT_VALIDATION_FAILED",
        }
    if isinstance(error, RuntimeError):
        return {"error_type": "RuntimeFailure", "reason_code": "RUNTIME_ERROR"}
    return {"error_type": "UnexpectedFailure", "reason_code": "UNEXPECTED_ERROR"}


def build_execution_attempt(
    *,
    timer: AttemptTimer,
    measurement_run_id: str,
    input_binding: Mapping[str, Any],
    block_size: int,
    computation_backend: str,
    workload: Mapping[str, Any] | None,
    measurement_root: str | Path,
    success: bool,
    failure: Mapping[str, str] | None = None,
    runtime: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a self-digesting immutable attempt after success or failure."""

    wall_time = max(time.monotonic() - timer.started_monotonic, sys.float_info.min)
    rss_peak = _max_rss_bytes()
    device, device_peak = _device_snapshot(computation_backend)
    status = "SUCCESS" if success else "FAILED"
    if success:
        if workload is None:
            raise ExecutionProfileError("successful attempt lacks workload evidence")
        semantic_binding: Mapping[str, Any] | None = build_semantic_input_binding(
            measurement_root
        )
        output_map = output_artifact_digests(measurement_root)
        output_digest: str | None = sha256_json(output_map)
        extrapolation: Mapping[str, Any] | None = _resource_extrapolation(
            workload,
            wall_time,
            block_size=block_size,
            computation_backend=computation_backend,
        )
        failure_payload: Mapping[str, str] | None = None
    else:
        semantic_binding = None
        output_map = {}
        output_digest = None
        extrapolation = None
        failure_payload = _require_object(failure, _FAILURE_KEYS, "failure")
    payload: dict[str, Any] = {
        "schema": ATTEMPT_SCHEMA,
        "execution_attempt_id": timer.attempt_id,
        "status": status,
        "measurement_run_id": _require_digest(
            measurement_run_id, "measurement run ID"
        ),
        "input_binding": dict(input_binding),
        "input_digest": sha256_json(input_binding),
        "semantic_input_binding": semantic_binding,
        "semantic_input_digest": (
            sha256_json(semantic_binding) if semantic_binding is not None else None
        ),
        "execution": {
            "block_size": _require_nonnegative_int(
                block_size, "block size", positive=True
            ),
            "computation_backend": str(computation_backend),
            "device": device,
            "runtime_versions": dict(runtime or runtime_versions()),
        },
        "workload": dict(workload) if workload is not None else None,
        "resource_profile": {
            "wall_time_seconds": wall_time,
            "rss_start_bytes": timer.rss_start_bytes,
            "rss_peak_bytes": rss_peak,
            "rss_peak_delta_bytes": max(0, rss_peak - timer.rss_start_bytes),
            "device_peak_memory_bytes": device_peak,
            "device_peak_memory_profile_available": device_peak is not None,
        },
        "resource_extrapolation": extrapolation,
        "output_artifact_sha256": output_map,
        "output_digest": output_digest,
        "failure": failure_payload,
    }
    payload["attempt_digest"] = sha256_json(payload)
    validate_execution_attempt_payload(payload)
    return payload


def validate_execution_attempt_payload(payload: Any) -> Mapping[str, Any]:
    """Validate exact nested schemas and the attempt's self digest."""

    value = _require_object(payload, _TOP_LEVEL_KEYS, "execution attempt")
    if value.get("schema") != ATTEMPT_SCHEMA:
        raise ExecutionProfileError("unsupported execution attempt schema")
    attempt_id = str(value.get("execution_attempt_id", ""))
    if not _ATTEMPT_ID.fullmatch(attempt_id):
        raise ExecutionProfileError("execution attempt ID is malformed")
    _require_digest(value.get("measurement_run_id"), "measurement run ID")
    input_binding = _require_object(value.get("input_binding"), _INPUT_KEYS, "input binding")
    for name in (
        "run_ref_sha256",
        "measurement_contract_sha256",
        "pair_plan_sha256",
        "base_binding_digest",
        "base_assets_digest",
    ):
        _require_digest(input_binding.get(name), name)
    for map_name in ("dataset_artifact_sha256", "dataset_content_digest"):
        digest_map = input_binding.get(map_name)
        if not isinstance(digest_map, Mapping) or not digest_map:
            raise ExecutionProfileError(f"{map_name} must be a non-empty digest map")
        for key, digest in digest_map.items():
            if not isinstance(key, str) or not key:
                raise ExecutionProfileError(f"{map_name} has an invalid key")
            _require_digest(digest, map_name)
    if _require_digest(value.get("input_digest"), "input digest") != sha256_json(input_binding):
        raise ExecutionProfileError("input digest mismatch")

    execution = _require_object(value.get("execution"), _EXECUTION_KEYS, "execution")
    _require_nonnegative_int(execution.get("block_size"), "block size", positive=True)
    if execution.get("computation_backend") not in {"numpy", "jax"}:
        raise ExecutionProfileError("unsupported computation backend")
    device = _require_object(execution.get("device"), _DEVICE_KEYS, "device")
    if device.get("requested_backend") != execution.get("computation_backend"):
        raise ExecutionProfileError("device/backend mismatch")
    device_count = _require_nonnegative_int(device.get("device_count"), "device count")
    if device_count > 256:
        raise ExecutionProfileError("device count exceeds the bounded profile schema")
    for field in ("platforms", "device_kinds"):
        entries = device.get(field)
        if not isinstance(entries, list) or not entries or any(
            not isinstance(item, str)
            or not _SAFE_DEVICE_STRING.fullmatch(item)
            for item in entries
        ):
            raise ExecutionProfileError(f"device {field} is malformed")
        if len(entries) > 16:
            raise ExecutionProfileError(f"device {field} exceeds bounded coverage")
        if entries != sorted(set(entries)):
            raise ExecutionProfileError(f"device {field} must be sorted and unique")
    if not set(device["platforms"]).issubset(_KNOWN_DEVICE_PLATFORMS):
        raise ExecutionProfileError("device platform is not recognized")
    if execution["computation_backend"] == "numpy" and (
        device_count != 1 or device["platforms"] != ["cpu"]
    ):
        raise ExecutionProfileError("numpy execution device projection is invalid")
    versions = execution.get("runtime_versions")
    if not isinstance(versions, Mapping) or not versions or any(
        not isinstance(key, str)
        or not _SAFE_RUNTIME_KEY.fullmatch(key)
        or not isinstance(version, str)
        or not _SAFE_RUNTIME_VALUE.fullmatch(version)
        for key, version in versions.items()
    ):
        raise ExecutionProfileError("runtime versions are malformed")

    resources = _require_object(
        value.get("resource_profile"), _RESOURCE_KEYS, "resource profile"
    )
    _require_nonnegative_float(
        resources.get("wall_time_seconds"), "wall time", positive=True
    )
    for field in ("rss_start_bytes", "rss_peak_bytes", "rss_peak_delta_bytes"):
        _require_nonnegative_int(resources.get(field), field)
    if resources["rss_peak_bytes"] < resources["rss_start_bytes"]:
        raise ExecutionProfileError("RSS peak precedes the attempt baseline")
    if resources["rss_peak_delta_bytes"] != (
        resources["rss_peak_bytes"] - resources["rss_start_bytes"]
    ):
        raise ExecutionProfileError("RSS peak delta is inconsistent")
    available = resources.get("device_peak_memory_profile_available")
    if type(available) is not bool:
        raise ExecutionProfileError("device memory availability flag must be boolean")
    peak_device = resources.get("device_peak_memory_bytes")
    if available:
        _require_nonnegative_int(peak_device, "device peak memory")
    elif peak_device is not None:
        raise ExecutionProfileError("unavailable device peak memory must be null")

    status = value.get("status")
    if status == "SUCCESS":
        if execution["computation_backend"] == "jax" and device_count <= 0:
            raise ExecutionProfileError("successful JAX attempt has no execution device")
        semantic = _require_object(
            value.get("semantic_input_binding"), _SEMANTIC_KEYS, "semantic input binding"
        )
        for name in _SEMANTIC_KEYS:
            digest_map = semantic.get(name)
            if not isinstance(digest_map, Mapping) or not digest_map:
                raise ExecutionProfileError(f"{name} must be a non-empty digest map")
            for digest in digest_map.values():
                _require_digest(digest, name)
        if _require_digest(
            value.get("semantic_input_digest"), "semantic input digest"
        ) != sha256_json(semantic):
            raise ExecutionProfileError("semantic input digest mismatch")
        workload = _require_object(value.get("workload"), _WORKLOAD_KEYS, "workload")
        support_sizes = workload.get("source_support_sizes")
        if not isinstance(support_sizes, list) or not support_sizes:
            raise ExecutionProfileError("source support sizes are missing")
        for support in support_sizes:
            _require_nonnegative_int(support, "source support size", positive=True)
        if support_sizes != sorted(support_sizes):
            raise ExecutionProfileError("source support sizes must be sorted")
        positive_fields = {
            "semantic_dataset_count",
            "semantic_transition_count",
            "pair_count",
            "routing_count",
            "mathematical_total_kernel_entries",
            "padded_total_block_entries",
        }
        for field in _WORKLOAD_KEYS - {"source_support_sizes"}:
            _require_nonnegative_int(
                workload.get(field), field, positive=(field in positive_fields)
            )
        if workload["semantic_cache_hits"] + workload["semantic_cache_misses"] != workload[
            "semantic_dataset_count"
        ]:
            raise ExecutionProfileError("semantic cache coverage is inconsistent")
        mathematical_sum = (
            workload["mathematical_self_kernel_entries"]
            + workload["mathematical_pair_cross_kernel_entries"]
            + workload["mathematical_routing_cross_kernel_entries"]
        )
        padded_sum = (
            workload["padded_self_block_entries"]
            + workload["padded_pair_cross_block_entries"]
            + workload["padded_routing_cross_block_entries"]
        )
        if workload["mathematical_total_kernel_entries"] != mathematical_sum:
            raise ExecutionProfileError("mathematical workload sum mismatch")
        if workload["padded_total_block_entries"] != padded_sum:
            raise ExecutionProfileError("padded workload sum mismatch")
        extrapolation = _require_object(
            value.get("resource_extrapolation"),
            _EXTRAPOLATION_KEYS,
            "resource extrapolation",
        )
        expected_extrapolation = _resource_extrapolation(
            workload,
            float(resources["wall_time_seconds"]),
            block_size=int(execution["block_size"]),
            computation_backend=str(execution["computation_backend"]),
        )
        if dict(extrapolation) != expected_extrapolation:
            raise ExecutionProfileError("resource extrapolation is not derivable from the profile")
        output_map = value.get("output_artifact_sha256")
        if not isinstance(output_map, Mapping) or tuple(sorted(output_map)) != _OUTPUT_NAMES:
            raise ExecutionProfileError("successful output coverage is incomplete")
        for digest in output_map.values():
            _require_digest(digest, "output artifact digest")
        if _require_digest(value.get("output_digest"), "output digest") != sha256_json(output_map):
            raise ExecutionProfileError("output digest mismatch")
        if value.get("failure") is not None:
            raise ExecutionProfileError("successful attempt cannot contain failure data")
    elif status == "FAILED":
        if value.get("semantic_input_binding") is not None or value.get("semantic_input_digest") is not None:
            raise ExecutionProfileError("failed attempt cannot claim semantic input completion")
        if value.get("workload") is not None or value.get("resource_extrapolation") is not None:
            raise ExecutionProfileError("failed attempt cannot claim a successful profile")
        if value.get("output_artifact_sha256") != {} or value.get("output_digest") is not None:
            raise ExecutionProfileError("failed attempt cannot authorize outputs")
        failure = _require_object(value.get("failure"), _FAILURE_KEYS, "failure")
        if failure.get("error_type") not in _SAFE_ERROR_TYPES:
            raise ExecutionProfileError("failure error type is not allowlisted")
        if failure.get("reason_code") not in _SAFE_REASON_CODES:
            raise ExecutionProfileError("failure reason code is not allowlisted")
    else:
        raise ExecutionProfileError("execution status must be SUCCESS or FAILED")

    expected_attempt_digest = sha256_json(
        {key: item for key, item in value.items() if key != "attempt_digest"}
    )
    if _require_digest(value.get("attempt_digest"), "attempt digest") != expected_attempt_digest:
        raise ExecutionProfileError("attempt self digest mismatch")
    return value


def verify_execution_attempt(
    measurement_root: str | Path,
    attempt_id: str,
    *,
    require_success: bool,
    source_support_sizes: tuple[int, ...],
) -> Mapping[str, Any]:
    """Revalidate an attempt against live public inputs and output bytes."""

    if not _ATTEMPT_ID.fullmatch(str(attempt_id)):
        raise ExecutionProfileError("execution attempt ID is malformed")
    root = Path(measurement_root).expanduser().resolve()
    path = _contained_file(
        root,
        f"execution_attempts/{attempt_id}.json",
        "execution attempt",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionProfileError("execution attempt is missing or unreadable") from error
    value = validate_execution_attempt_payload(payload)
    if value["execution_attempt_id"] != attempt_id:
        raise ExecutionProfileError("execution attempt filename/ID mismatch")
    measurement_run_id, live_input = build_initial_input_binding(root)
    if value["measurement_run_id"] != measurement_run_id:
        raise ExecutionProfileError("execution attempt measurement-run mismatch")
    if value["input_binding"] != live_input or value["input_digest"] != sha256_json(live_input):
        raise ExecutionProfileError("execution attempt input binding is stale")
    if require_success and value["status"] != "SUCCESS":
        raise ExecutionProfileError("execution attempt is not successful")
    try:
        run_ref = json.loads(
            _contained_file(root, "run_ref.json", "run reference").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionProfileError("cannot read run reference runtime") from error
    expected_runtime = run_ref.get("runtime_versions") if isinstance(run_ref, Mapping) else None
    if (
        not isinstance(expected_runtime, Mapping)
        or value["execution"]["runtime_versions"] != expected_runtime
        or dict(expected_runtime) != runtime_versions()
    ):
        raise ExecutionProfileError("execution attempt runtime differs from frozen/current runtime")
    if value["status"] == "SUCCESS":
        live_semantic = build_semantic_input_binding(root)
        if (
            value["semantic_input_binding"] != live_semantic
            or value["semantic_input_digest"] != sha256_json(live_semantic)
        ):
            raise ExecutionProfileError("execution attempt semantic binding is stale")
        live_outputs = output_artifact_digests(root)
        if (
            value["output_artifact_sha256"] != live_outputs
            or value["output_digest"] != sha256_json(live_outputs)
        ):
            raise ExecutionProfileError("execution attempt output binding is stale")
        observed_support_sizes = tuple(
            int(item) for item in value["workload"]["source_support_sizes"]
        )
        expected_support_sizes = tuple(
            sorted(int(item) for item in source_support_sizes)
        )
        if observed_support_sizes != expected_support_sizes:
            raise ExecutionProfileError("execution attempt source-support coverage is stale")
        live_workload = rebuild_live_workload(
            root,
            source_support_sizes=expected_support_sizes,
            block_size=int(value["execution"]["block_size"]),
            computation_backend=str(value["execution"]["computation_backend"]),
            semantic_cache_hits=int(value["workload"]["semantic_cache_hits"]),
            semantic_cache_misses=int(value["workload"]["semantic_cache_misses"]),
        )
        if value["workload"] != live_workload:
            raise ExecutionProfileError("execution attempt workload differs from live plan/caches")
    return value


def verify_any_successful_execution_attempt(
    measurement_root: str | Path,
    *,
    source_support_sizes: tuple[int, ...],
) -> Mapping[str, Any]:
    """Return a deterministic verified SUCCESS attempt or fail closed."""

    root = Path(measurement_root).expanduser().resolve()
    attempts = _contained_directory(
        root, "execution_attempts", "TaskSpec execution-attempt directory"
    )
    failures: list[str] = []
    for path in sorted(attempts.glob("v01xa-*.json")):
        try:
            value = verify_execution_attempt(
                root,
                path.stem,
                require_success=True,
                source_support_sizes=source_support_sizes,
            )
        except ExecutionProfileError as error:
            failures.append(type(error).__name__)
            continue
        return value
    detail = "" if not failures else f" ({len(failures)} rejected attempts)"
    raise ExecutionProfileError(f"no verified successful TaskSpec execution attempt{detail}")


def verified_profile_projection(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fixed, path-free subset allowed in a NO_GO preflight."""

    value = validate_execution_attempt_payload(profile)
    if value["status"] != "SUCCESS":
        raise ExecutionProfileError("only a successful profile may be projected")
    return {
        "execution_attempt_id": value["execution_attempt_id"],
        "execution_attempt_digest": value["attempt_digest"],
        "measurement_input_digest": value["input_digest"],
        "semantic_input_digest": value["semantic_input_digest"],
        "measurement_output_digest": value["output_digest"],
        "execution": dict(value["execution"]),
        "workload": dict(value["workload"]),
        "resource_profile": dict(value["resource_profile"]),
        "resource_extrapolation": dict(value["resource_extrapolation"]),
    }


__all__ = [
    "ATTEMPT_SCHEMA",
    "EXTRAPOLATION_SCHEMA",
    "ExecutionProfileError",
    "AttemptTimer",
    "APPROVED_P5_SMOKE_CONFIG_DIGEST",
    "build_execution_attempt",
    "build_initial_input_binding",
    "build_semantic_input_binding",
    "classify_failure",
    "estimate_taskspec_workload",
    "new_execution_attempt_id",
    "output_artifact_digests",
    "runtime_versions",
    "rebuild_live_workload",
    "validate_execution_attempt_payload",
    "verify_any_successful_execution_attempt",
    "verify_execution_attempt",
    "verified_profile_projection",
]
