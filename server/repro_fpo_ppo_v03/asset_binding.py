"""Bind real v0.2 assets for v0.3 without running an episode or training.

This command is deliberately narrower than a formal runner.  It replays the
frozen P5R intake authority, reloads the frozen v0.2 server plan, validates all
90 accepted policy bundles through the production FPO/JAX driver's
``validate_candidate`` boundary, and publishes immutable inputs for the later
P5M run.  It cannot issue formal authority and it never calls the driver's
rollout method.

Run production binding with ``python -B`` and put the attested vendor directory
first on ``PYTHONPATH``; those are requirements of the frozen runtime driver.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from policy_learnware_v0.io import atomic_write_json, deterministic_npz_bytes
from policy_learnware_v0.probe.dataset import DatasetManifest, load_dataset_artifact
from policy_learnware_v0.schemas import FrozenProtocol
from policy_learnware_v0.v03.formal_gates import FormalMarketPlan
from policy_learnware_v0.v03.fpo_source_backend import (
    FpoJaxRuntimeDriver,
    FpoJaxSourceEvaluatorBackend,
    FrozenV02FpoJaxRuntimeDriver,
)
from policy_learnware_v0.v03.pool_intake import (
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_JOB_COUNT,
    V03PoolIntakeRecord,
    load_verified_frozen_v02_intake,
)
from policy_learnware_v0.v03.source_evaluator import (
    DmcFixedHorizonReturnContract,
    FrozenServerPlanBinding,
    FrozenV02ServerPlanAuthority,
    plan_source_selection_work_units,
    source_work_unit_manifest,
)
from policy_learnware_v0.v03.source_market import (
    MARKET_ALIAS_ASSIGNMENT,
    MARKET_ALIAS_PROTOCOL_SCHEMA,
    SourceEvaluationProtocol,
    formal_market_alias_protocol_digest,
    market_nonce_commitment,
)


class AssetBindingError(ValueError):
    """A production asset, path, digest, or publication contract drifted."""


ASSET_BINDINGS_READY = "ASSET_BINDINGS_READY"
LEGACY_ASSET_INVENTORY_SCHEMA = (
    "policy-learnware.v03-legacy-asset-binding-inventory.v0"
)
ASSET_BINDING_RECEIPT_SCHEMA = (
    "policy-learnware.v03-production-asset-binding-receipt.v0"
)
SEED_NAMESPACE_SCHEMA = "policy-learnware.v03-source-seed-namespace.v0"
SEED_DERIVATION_PROTOCOL_SCHEMA = (
    "policy-learnware.v03-source-seed-derivation-protocol.v0"
)
V02_SELECTION_LEDGER_SCHEMA = "policy-learnware.v02-formal-selection-ledger.v0"

# These are reviewed v0.3 literals, not CLI knobs.  Keeping the complete tuples
# in code prevents a caller from silently changing either sample size or the
# reset-seed population while retaining the same source-evaluation label.
PRODUCTION_SELECTION_RESET_SEEDS = (
    100_000,
    100_001,
    100_002,
    100_003,
    100_004,
    100_005,
    100_006,
    100_007,
    100_008,
    100_009,
    100_010,
    100_011,
    100_012,
    100_013,
    100_014,
    100_015,
    100_016,
    100_017,
    100_018,
    100_019,
    100_020,
    100_021,
    100_022,
    100_023,
    100_024,
)
PRODUCTION_ATTESTATION_RESET_SEEDS = (
    200_000,
    200_001,
    200_002,
    200_003,
    200_004,
    200_005,
    200_006,
    200_007,
    200_008,
    200_009,
    200_010,
    200_011,
    200_012,
    200_013,
    200_014,
    200_015,
    200_016,
    200_017,
    200_018,
    200_019,
    200_020,
    200_021,
    200_022,
    200_023,
    200_024,
    200_025,
    200_026,
    200_027,
    200_028,
    200_029,
    200_030,
    200_031,
    200_032,
    200_033,
    200_034,
    200_035,
    200_036,
    200_037,
    200_038,
    200_039,
    200_040,
    200_041,
    200_042,
    200_043,
    200_044,
    200_045,
    200_046,
    200_047,
    200_048,
    200_049,
)
PRODUCTION_COMPETENCE_FLOOR = 0.5
PRODUCTION_MEAN_TOLERANCE = 0.01
PRODUCTION_LCB_Z = 1.645
PRODUCTION_COMPETENCE_MODE = "OBSERVE"
PRODUCTION_RETURN_HORIZON = 1000
PRODUCTION_PER_STEP_LOWER = 0.0
PRODUCTION_PER_STEP_UPPER = 1.0
PRODUCTION_SELECTION_LEDGER_FILE_SHA256 = (
    "7da7512f4aa31e8801a3ea76eaa02b4628a40d6f0fa4844bfae059c3d4a79431"
)
PRODUCTION_SELECTION_LEDGER_DIGEST = (
    "a6eca797fc3f733859c3a349f60e0336eebfb6ad379a3ce61cca088450e5b22d"
)
PRODUCTION_SELECTION_LEDGER_EXPERIMENT_ID = "v02-reacher-formal-2r-20260825-r2"
PRODUCTION_SELECTION_LEDGER_CONFIG_DIGEST = (
    "c2b52e3ff4d9fa58ff94deb55c581f342cbb37b78265e1f51b9c2640a80acbba"
)


def _seed_derivation_protocol_payload() -> dict[str, Any]:
    return {
        "schema": SEED_DERIVATION_PROTOCOL_SCHEMA,
        "derivation": "reviewed_literal_contiguous_uint32_blocks_v0",
        "selection": {
            "namespace": "source_selection",
            "start_inclusive": 100_000,
            "stop_exclusive": 100_025,
            "reset_seeds": list(PRODUCTION_SELECTION_RESET_SEEDS),
        },
        "attestation": {
            "namespace": "source_attestation",
            "start_inclusive": 200_000,
            "stop_exclusive": 200_050,
            "reset_seeds": list(PRODUCTION_ATTESTATION_RESET_SEEDS),
        },
        "disjoint": True,
    }


PRODUCTION_SEED_DERIVATION_PROTOCOL_DIGEST = sha256_json(
    _seed_derivation_protocol_payload()
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_NONCE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_LEGACY_SPLITS = frozenset(
    {
        "encoder_train",
        "encoder_validation",
        "kernel_calibration",
        "separability_calibration",
        "source_taskspec",
        "target_query",
    }
)
_OUTPUT_NAMES = MappingProxyType(
    {
        "source_evaluation_protocol": "source_evaluation_protocol.json",
        "source_selection_work_units": "source_selection_work_units.json",
        "formal_market_plan": "formal_market_plan.json",
        "legacy_asset_inventory": "legacy_asset_inventory.json",
        "asset_binding_receipt": "asset_binding_receipt.json",
    }
)


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise AssetBindingError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, where: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssetBindingError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise AssetBindingError(f"{where} must be finite and non-negative")
    return result


def _absolute(path: str | Path, where: str, *, kind: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise AssetBindingError(f"{where} must be absolute and may not be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise AssetBindingError(f"{where} does not exist") from error
    if kind == "file" and not resolved.is_file():
        raise AssetBindingError(f"{where} must be a file")
    if kind == "dir" and not resolved.is_dir():
        raise AssetBindingError(f"{where} must be a directory")
    return resolved


def _confined_path(root: Path, relative: Any, where: str, *, kind: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise AssetBindingError(f"{where} must be a non-empty relative path")
    parts = Path(relative).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise AssetBindingError(f"{where} is not a canonical confined path")
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AssetBindingError(f"{where} traverses a symlink")
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as error:
        raise AssetBindingError(f"{where} does not exist") from error
    if not resolved.is_relative_to(root):
        raise AssetBindingError(f"{where} escapes the legacy asset root")
    if kind == "file" and not resolved.is_file():
        raise AssetBindingError(f"{where} must be a file")
    if kind == "dir" and not resolved.is_dir():
        raise AssetBindingError(f"{where} must be a directory")
    return resolved


def _strict_json(path: Path, where: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AssetBindingError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise AssetBindingError(f"{where} contains non-finite constant {value}")

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetBindingError(f"cannot read {where}: {error}") from error
    if not isinstance(result, dict):
        raise AssetBindingError(f"{where} must be a JSON object")
    return result


def _seeds(values: Sequence[int], where: str) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise AssetBindingError(f"{where} must be a seed sequence") from error
    if (
        not result
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in result)
        or result != tuple(sorted(set(result)))
    ):
        raise AssetBindingError(
            f"{where} must be a non-empty, sorted, unique non-negative seed block"
        )
    return result


def _nonce(value: Any, where: str) -> str:
    if not isinstance(value, str) or _NONCE.fullmatch(value) is None:
        raise AssetBindingError(f"{where} must be an explicit 256-bit lowercase hex nonce")
    if len(set(value)) < 2:
        raise AssetBindingError(f"{where} may not be an all-one-symbol placeholder")
    return value


@dataclass(frozen=True)
class ProductionAssetBindingConfig:
    intake_record_path: str | Path
    intake_record_sha256: str
    trusted_experiment_root: str | Path
    server_plan_path: str | Path
    server_plan_sha256: str
    selection_ledger_path: str | Path
    fpo_root: str | Path
    vendor_dir: str | Path
    legacy_v0_root: str | Path
    output_dir: str | Path
    market_alias_private_nonce_file: str | Path
    tie_break_private_nonce_file: str | Path
    selection_reset_seeds: tuple[int, ...] = PRODUCTION_SELECTION_RESET_SEEDS
    attestation_reset_seeds: tuple[int, ...] = PRODUCTION_ATTESTATION_RESET_SEEDS
    competence_floor: float = PRODUCTION_COMPETENCE_FLOOR
    mean_tolerance: float = PRODUCTION_MEAN_TOLERANCE
    lcb_z: float = PRODUCTION_LCB_Z
    return_horizon: int = PRODUCTION_RETURN_HORIZON
    per_step_lower: float = PRODUCTION_PER_STEP_LOWER
    per_step_upper: float = PRODUCTION_PER_STEP_UPPER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intake_record_sha256",
            _digest(self.intake_record_sha256, "intake_record_sha256"),
        )
        object.__setattr__(
            self,
            "server_plan_sha256",
            _digest(self.server_plan_sha256, "server_plan_sha256"),
        )
        selection = _seeds(self.selection_reset_seeds, "selection_reset_seeds")
        attestation = _seeds(self.attestation_reset_seeds, "attestation_reset_seeds")
        if selection != PRODUCTION_SELECTION_RESET_SEEDS:
            raise AssetBindingError(
                "selection_reset_seeds must equal the reviewed 25-seed production tuple"
            )
        if attestation != PRODUCTION_ATTESTATION_RESET_SEEDS:
            raise AssetBindingError(
                "attestation_reset_seeds must equal the reviewed 50-seed production tuple"
            )
        object.__setattr__(self, "selection_reset_seeds", selection)
        object.__setattr__(self, "attestation_reset_seeds", attestation)
        floor = _finite(self.competence_floor, "competence_floor", nonnegative=True)
        if floor != PRODUCTION_COMPETENCE_FLOOR:
            raise AssetBindingError("competence_floor must equal the reviewed literal 0.5")
        object.__setattr__(self, "competence_floor", floor)
        mean_tolerance = _finite(
            self.mean_tolerance, "mean_tolerance", nonnegative=True
        )
        if mean_tolerance != PRODUCTION_MEAN_TOLERANCE:
            raise AssetBindingError(
                "mean_tolerance must equal the reviewed literal 0.01"
            )
        object.__setattr__(self, "mean_tolerance", mean_tolerance)
        lcb_z = _finite(self.lcb_z, "lcb_z", nonnegative=True)
        if lcb_z != PRODUCTION_LCB_Z:
            raise AssetBindingError("lcb_z must equal the reviewed literal 1.645")
        object.__setattr__(self, "lcb_z", lcb_z)
        if self.return_horizon != PRODUCTION_RETURN_HORIZON:
            raise AssetBindingError("return_horizon must equal the reviewed literal 1000")
        lower = _finite(self.per_step_lower, "per_step_lower")
        upper = _finite(self.per_step_upper, "per_step_upper")
        if (
            lower != PRODUCTION_PER_STEP_LOWER
            or upper != PRODUCTION_PER_STEP_UPPER
        ):
            raise AssetBindingError(
                "per-step return bounds must equal the reviewed literals [0.0, 1.0]"
            )
        object.__setattr__(self, "per_step_lower", lower)
        object.__setattr__(self, "per_step_upper", upper)
        alias_file = _private_nonce_file(
            self.market_alias_private_nonce_file, "market_alias_private_nonce_file"
        )
        tie_file = _private_nonce_file(
            self.tie_break_private_nonce_file, "tie_break_private_nonce_file"
        )
        if alias_file == tie_file:
            raise AssetBindingError("market alias and tie-break nonce files must differ")
        object.__setattr__(self, "market_alias_private_nonce_file", alias_file)
        object.__setattr__(self, "tie_break_private_nonce_file", tie_file)


def _private_nonce_file(path: str | Path, where: str) -> Path:
    result = _absolute(path, where, kind="file")
    mode = stat.S_IMODE(result.stat().st_mode)
    if mode != 0o600:
        raise AssetBindingError(f"{where} must have exact mode 0600")
    return result


def _read_private_nonces(config: ProductionAssetBindingConfig) -> tuple[str, str]:
    def read(raw_path: Path, where: str) -> str:
        path = _private_nonce_file(raw_path, where)
        before = path.stat()
        try:
            text = path.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise AssetBindingError(f"cannot read {where}") from error
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AssetBindingError(f"{where} changed while it was read")
        if text.endswith("\n"):
            text = text[:-1]
        return _nonce(text, where)

    alias = read(
        Path(config.market_alias_private_nonce_file), "market alias private nonce"
    )
    tie = read(Path(config.tie_break_private_nonce_file), "tie-break private nonce")
    if alias == tie:
        raise AssetBindingError("market alias and tie-break private nonces must differ")
    return alias, tie


def _verify_manifest_files(
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    where: str,
) -> dict[str, dict[str, str]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise AssetBindingError(f"{where}.files must be a non-empty mapping")
    result: dict[str, dict[str, str]] = {}
    for label, raw in sorted(files.items()):
        if not isinstance(label, str) or not isinstance(raw, Mapping):
            raise AssetBindingError(f"{where}.files contains an invalid entry")
        if set(raw) != {"path", "sha256"}:
            raise AssetBindingError(f"{where}.files[{label!r}] has drifted fields")
        expected = _digest(raw["sha256"], f"{where}.files[{label!r}].sha256")
        path = _confined_path(root, raw["path"], f"{where}.files[{label!r}].path", kind="file")
        actual = sha256_file(path)
        if actual != expected:
            raise AssetBindingError(f"{where}.files[{label!r}] digest mismatch")
        result[label] = {
            "path": str(path),
            "relative_path": str(path.relative_to(root)),
            "sha256": actual,
        }
    return result


def build_legacy_asset_inventory(legacy_v0_root: str | Path) -> dict[str, Any]:
    """Validate and bind the frozen v0 encoder, normalizer, and all datasets."""

    root = _absolute(legacy_v0_root, "legacy_v0_root", kind="dir")
    required_manifests = {
        "protocol": "protocol/manifest.json",
        "encoder": "protocol/encoder_manifest.json",
        "normalization": "protocol/normalization_manifest.json",
        "kernel": "protocol/kernel_manifest.json",
        "environment": "protocol/environment_manifest.json",
    }
    expected_schemas = {
        "protocol": "policy-learnware.protocol-artifacts.v0",
        "encoder": "policy-learnware.encoder-artifact.v0",
        "normalization": "policy-learnware.normalization-artifact.v0",
        "kernel": "policy-learnware.kernel-artifact.v0",
        "environment": "policy-learnware.environment-artifacts.v0",
    }
    parsed: dict[str, dict[str, Any]] = {}
    manifest_rows: dict[str, dict[str, Any]] = {}
    for label, relative in required_manifests.items():
        path = _confined_path(root, relative, f"legacy {label} manifest", kind="file")
        value = _strict_json(path, f"legacy {label} manifest")
        if value.get("complete") is not True:
            raise AssetBindingError(f"legacy {label} manifest is not complete")
        if value.get("schema") != expected_schemas[label]:
            raise AssetBindingError(f"legacy {label} manifest schema drifted")
        parsed[label] = value
        manifest_rows[label] = {
            "path": str(path),
            "relative_path": relative,
            "sha256": sha256_file(path),
            "schema": value.get("schema"),
            "files": _verify_manifest_files(root, path, value, where=f"legacy {label} manifest"),
        }

    protocol_file = _confined_path(
        root, "protocol/protocol.json", "legacy frozen protocol", kind="file"
    )
    try:
        protocol = FrozenProtocol.from_dict(_strict_json(protocol_file, "legacy frozen protocol"))
    except (TypeError, ValueError) as error:
        raise AssetBindingError(f"legacy frozen protocol is invalid: {error}") from error
    top = parsed["protocol"]
    draft_digest = _digest(top.get("protocol_draft_hash"), "legacy protocol_draft_hash")
    if any(
        row.get("protocol_draft_hash") != draft_digest
        for label, row in parsed.items()
        if label != "protocol"
    ):
        raise AssetBindingError("legacy protocol artifact draft hashes differ")
    if top.get("protocol_id") != protocol.protocol_id:
        raise AssetBindingError("legacy protocol manifest and frozen protocol ID differ")
    component_labels = {
        "encoder": "encoder",
        "normalization": "normalization",
        "kernel": "kernel",
        "environment": "environment_manifest",
    }
    for manifest_label, component_label in component_labels.items():
        if protocol.component_digests.get(component_label) != manifest_rows[manifest_label]["sha256"]:
            raise AssetBindingError(
                f"legacy frozen protocol component {component_label!r} digest drifted"
            )

    datasets_dir = _confined_path(root, "datasets", "legacy datasets root", kind="dir")
    dataset_rows: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    tasks_by_split: dict[str, set[str]] = {}
    query_banks: dict[str, set[str]] = {}
    dataset_identities: set[tuple[str, str, str]] = set()
    manifest_paths = sorted(datasets_dir.rglob("*.json"))
    if not manifest_paths:
        raise AssetBindingError("legacy datasets root contains no manifests")
    for manifest_path in manifest_paths:
        if manifest_path.is_symlink():
            raise AssetBindingError("legacy dataset manifest may not be a symlink")
        relative = manifest_path.relative_to(root)
        manifest_path = _confined_path(
            root,
            str(relative),
            f"legacy dataset manifest {relative}",
            kind="file",
        )
        npz_path = manifest_path.with_suffix(".npz")
        _confined_path(root, str(npz_path.relative_to(root)), "legacy dataset NPZ", kind="file")
        raw = _strict_json(manifest_path, f"legacy dataset manifest {relative}")
        try:
            declared = DatasetManifest.from_dict(raw)
            dataset, validated = load_dataset_artifact(npz_path, manifest_path)
        except (OSError, TypeError, ValueError) as error:
            raise AssetBindingError(f"legacy dataset {relative} is invalid: {error}") from error
        if declared != validated:
            raise AssetBindingError(f"legacy dataset {relative} changed during validation")
        if declared.protocol_draft_hash != draft_digest:
            raise AssetBindingError(f"legacy dataset {relative} belongs to another draft")
        if npz_path.read_bytes() != deterministic_npz_bytes(dataset.to_arrays(copy=False)):
            raise AssetBindingError(f"legacy dataset {relative} NPZ is not canonical")
        if declared.split not in _REQUIRED_LEGACY_SPLITS:
            raise AssetBindingError(f"legacy dataset {relative} uses an unknown split")
        if manifest_path.name != f"{declared.task}.json":
            raise AssetBindingError(
                f"legacy dataset {relative} filename differs from its declared task"
            )
        expected_prefix = Path("datasets") / declared.split
        if declared.split == "target_query":
            if len(relative.parts) != 4 or relative.parts[:2] != ("datasets", "target_query"):
                raise AssetBindingError("target_query dataset path must include exactly one bank")
            bank = relative.parts[2]
            query_banks.setdefault(bank, set()).add(declared.task)
            identity = (declared.split, bank, declared.task)
        elif relative.parent != expected_prefix:
            raise AssetBindingError(f"legacy dataset {relative} does not match its split")
        else:
            identity = (declared.split, "", declared.task)
        if identity in dataset_identities:
            raise AssetBindingError(
                f"legacy dataset {relative} duplicates a split/bank/task identity"
            )
        dataset_identities.add(identity)
        tasks_by_split.setdefault(declared.split, set()).add(declared.task)
        counts = split_counts.setdefault(declared.split, {"manifests": 0, "episodes": 0, "transitions": 0})
        counts["manifests"] += 1
        counts["episodes"] += declared.episode_count
        counts["transitions"] += declared.transition_count
        key = str(relative)
        dataset_rows[key] = {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "npz_path": str(npz_path.resolve()),
            "npz_sha256": sha256_file(npz_path),
            "dataset_digest": dataset.digest,
            "split": declared.split,
            "task": declared.task,
            "episode_count": declared.episode_count,
            "transition_count": declared.transition_count,
        }
        del dataset

    observed_splits = set(split_counts)
    missing_splits = _REQUIRED_LEGACY_SPLITS - observed_splits
    if missing_splits:
        raise AssetBindingError(f"legacy datasets miss required splits: {sorted(missing_splits)}")
    try:
        tasks = tuple(protocol.config["environment"]["tasks"])
        expected_query_banks = int(protocol.config["episodes"]["target_query_banks"])
    except (KeyError, TypeError, ValueError) as error:
        raise AssetBindingError("legacy protocol lacks task/query-bank literals") from error
    if not tasks or len(set(tasks)) != len(tasks) or any(not isinstance(task, str) or not task for task in tasks):
        raise AssetBindingError("legacy protocol task list is invalid")
    task_set = set(tasks)
    for split in _REQUIRED_LEGACY_SPLITS - {"target_query"}:
        if tasks_by_split.get(split) != task_set:
            raise AssetBindingError(f"legacy split {split} does not cover the frozen task set")
    if len(query_banks) != expected_query_banks or any(rows != task_set for rows in query_banks.values()):
        raise AssetBindingError("legacy target_query banks do not match the frozen task x bank grid")
    if parsed["environment"].get("tasks") != list(tasks):
        raise AssetBindingError("legacy environment manifest task order drifted")

    source_manifest_shas = {
        task: dataset_rows[f"datasets/source_taskspec/{task}.json"]["manifest_sha256"]
        for task in sorted(task_set)
    }
    if top.get("source_dataset_manifest_digests") != source_manifest_shas:
        raise AssetBindingError("legacy protocol manifest source-dataset digests drifted")
    if protocol.component_digests.get("source_dataset_manifests") != sha256_json(source_manifest_shas):
        raise AssetBindingError("legacy frozen protocol source-dataset binding drifted")

    semantic_by_split_task: dict[str, dict[str, str]] = {}
    for row in dataset_rows.values():
        if row["split"] != "target_query":
            semantic_by_split_task.setdefault(row["split"], {})[row["task"]] = row["dataset_digest"]
    encoder_sources = parsed["encoder"].get("source_dataset_digests")
    expected_encoder_sources = {
        split: semantic_by_split_task[split]
        for split in ("encoder_train", "encoder_validation")
    }
    if encoder_sources != expected_encoder_sources:
        raise AssetBindingError("legacy encoder source-dataset binding drifted")
    if parsed["normalization"].get("source_dataset_digests") != semantic_by_split_task["encoder_train"]:
        raise AssetBindingError("legacy normalizer source-dataset binding drifted")
    if parsed["kernel"].get("source_dataset_digests") != semantic_by_split_task["kernel_calibration"]:
        raise AssetBindingError("legacy kernel source-dataset binding drifted")
    checkpoint_sha = manifest_rows["encoder"]["files"]["checkpoint"]["sha256"]
    normalization_sha = manifest_rows["normalization"]["files"]["normalization"]["sha256"]
    if parsed["encoder"].get("normalization_sha256") != normalization_sha:
        raise AssetBindingError("legacy encoder does not bind the frozen normalizer")
    if parsed["kernel"].get("encoder_sha256") != checkpoint_sha:
        raise AssetBindingError("legacy kernel does not bind the frozen encoder")

    totals = {
        "manifest_count": len(dataset_rows),
        "episode_count": sum(row["episode_count"] for row in dataset_rows.values()),
        "transition_count": sum(row["transition_count"] for row in dataset_rows.values()),
    }
    payload = {
        "schema": LEGACY_ASSET_INVENTORY_SCHEMA,
        "legacy_v0_root": str(root),
        "protocol_id": protocol.protocol_id,
        "packed_layout": dict(protocol.packed_layout),
        "task_ids": sorted(task_set),
        "target_query_bank_ids": sorted(query_banks),
        "protocol_manifests": manifest_rows,
        "datasets": dict(sorted(dataset_rows.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "totals": totals,
    }
    return {**payload, "inventory_digest": sha256_json(payload)}


def _seed_namespace_digest(
    *,
    intake_record_digest: str,
    selection_ledger_digest: str,
    namespace: str,
    seeds: tuple[int, ...],
) -> str:
    return sha256_json(
        {
            "schema": SEED_NAMESPACE_SCHEMA,
            "intake_record_digest": intake_record_digest,
            "selection_ledger_digest": selection_ledger_digest,
            "seed_derivation_protocol_digest": (
                PRODUCTION_SEED_DERIVATION_PROTOCOL_DIGEST
            ),
            "namespace": namespace,
            "reset_seeds": list(seeds),
        }
    )


def _selection_ledger_binding(
    config: ProductionAssetBindingConfig,
    *,
    trusted_experiment_root: Path,
    server_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and bind the reviewed v0.2 selection ledger exactly once."""

    path = _absolute(config.selection_ledger_path, "selection ledger", kind="file")
    if path.name != "v02_selection_ledger.json" or path.parent.name != "configs":
        raise AssetBindingError(
            "selection ledger must be the canonical configs/v02_selection_ledger.json"
        )
    before = path.stat()
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise AssetBindingError("cannot read selection ledger") from error
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise AssetBindingError("selection ledger changed while it was read")
    if sha256_bytes(raw_bytes) != PRODUCTION_SELECTION_LEDGER_FILE_SHA256:
        raise AssetBindingError(
            "selection ledger SHA-256 differs from reviewed production authority"
        )

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AssetBindingError(
                    f"selection ledger contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise AssetBindingError(
            f"selection ledger contains non-finite constant {value}"
        )

    try:
        ledger = json.loads(
            raw_bytes,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssetBindingError(f"selection ledger is invalid JSON: {error}") from error
    if not isinstance(ledger, dict):
        raise AssetBindingError("selection ledger must be a JSON object")
    if raw_bytes != canonical_json_bytes(ledger) + b"\n":
        raise AssetBindingError("selection ledger is not canonical JSON")
    if ledger.get("schema") != V02_SELECTION_LEDGER_SCHEMA:
        raise AssetBindingError("selection ledger schema drifted")
    declared_digest = _digest(
        ledger.get("ledger_digest"), "selection ledger ledger_digest"
    )
    computed_digest = sha256_json(
        {key: value for key, value in ledger.items() if key != "ledger_digest"}
    )
    if declared_digest != computed_digest:
        raise AssetBindingError("selection ledger semantic self-digest mismatch")
    if declared_digest != PRODUCTION_SELECTION_LEDGER_DIGEST:
        raise AssetBindingError(
            "selection ledger semantic digest differs from reviewed production authority"
        )
    experiment_id = ledger.get("experiment_id")
    if experiment_id != PRODUCTION_SELECTION_LEDGER_EXPERIMENT_ID:
        raise AssetBindingError(
            "selection ledger experiment differs from reviewed production authority"
        )
    if experiment_id not in trusted_experiment_root.parts:
        raise AssetBindingError(
            "selection ledger experiment is absent from trusted_experiment_root lineage"
        )
    config_digest = ledger.get("config_digest")
    if config_digest != PRODUCTION_SELECTION_LEDGER_CONFIG_DIGEST:
        raise AssetBindingError(
            "selection ledger config differs from reviewed production authority"
        )
    if config_digest != server_plan.get("config_digest"):
        raise AssetBindingError(
            "selection ledger config differs from the frozen server plan"
        )
    admission = ledger.get("admission_decision")
    required_admission = {
        "selection_episodes": len(PRODUCTION_SELECTION_RESET_SEEDS),
        "attestation_episodes": len(PRODUCTION_ATTESTATION_RESET_SEEDS),
        "competence_lcb_floor": PRODUCTION_COMPETENCE_FLOOR,
        "champion_mean_tolerance": PRODUCTION_MEAN_TOLERANCE,
        "lcb_z": PRODUCTION_LCB_Z,
        "competence_mode": PRODUCTION_COMPETENCE_MODE,
    }
    if admission != required_admission:
        raise AssetBindingError(
            "selection ledger admission_decision differs from reviewed production literals"
        )
    return {
        "path": str(path),
        "file_sha256": PRODUCTION_SELECTION_LEDGER_FILE_SHA256,
        "semantic_digest": declared_digest,
        "experiment_id": experiment_id,
        "config_digest": config_digest,
        "admission_decision": required_admission,
    }


def _require_file_matches(path: str | Path, expected_sha256: str, where: str) -> Path:
    resolved = _absolute(path, where, kind="file")
    if sha256_file(resolved) != expected_sha256:
        raise AssetBindingError(f"{where} SHA-256 differs from the explicit authority")
    return resolved


def _require_confined_existing(path: str | Path, root: Path, where: str) -> Path:
    resolved = _absolute(path, where, kind="file")
    if not resolved.is_relative_to(root):
        raise AssetBindingError(f"{where} escapes trusted_experiment_root")
    return resolved


def _prepare_documents(
    config: ProductionAssetBindingConfig,
    *,
    intake: V03PoolIntakeRecord,
    plan: FrozenServerPlanBinding,
    runtime_driver: FpoJaxRuntimeDriver,
    backend: FpoJaxSourceEvaluatorBackend,
) -> dict[str, dict[str, Any]]:
    """Fixture-friendly core; callers must provide already validated authorities."""

    if not isinstance(intake, V03PoolIntakeRecord) or intake.pool_state != "POOL_READY":
        raise AssetBindingError("asset binding requires a typed POOL_READY intake")
    if not isinstance(plan, FrozenServerPlanBinding):
        raise AssetBindingError("asset binding requires a typed frozen server plan")
    if plan.plan_digest != intake.server_plan_digest or set(plan.jobs) != set(intake.cells):
        raise AssetBindingError("server plan differs from the frozen P5R intake")
    intake_path = _require_file_matches(
        config.intake_record_path, config.intake_record_sha256, "frozen P5R intake record"
    )
    intake_bytes = _strict_json(intake_path, "frozen P5R intake record")
    if intake_bytes != intake.to_dict() or intake_path.read_bytes() != canonical_json_bytes(intake_bytes) + b"\n":
        raise AssetBindingError("typed P5R intake differs from canonical persisted bytes")
    plan_path = _require_file_matches(
        config.server_plan_path, config.server_plan_sha256, "frozen server plan"
    )
    if plan_path != Path(plan.plan_path):
        raise AssetBindingError("typed frozen server plan belongs to another path")
    plan_bytes = _strict_json(plan_path, "frozen server plan")
    if plan_path.read_bytes() != canonical_json_bytes(plan_bytes) + b"\n":
        raise AssetBindingError("frozen server plan is not canonical JSON")
    if plan_bytes.get("plan_digest") != intake.server_plan_digest:
        raise AssetBindingError("frozen server plan semantic digest differs from P5R intake")
    trusted_root = _absolute(
        config.trusted_experiment_root, "trusted_experiment_root", kind="dir"
    )
    selection_ledger = _selection_ledger_binding(
        config,
        trusted_experiment_root=trusted_root,
        server_plan=plan_bytes,
    )
    if not isinstance(runtime_driver, FpoJaxRuntimeDriver):
        raise AssetBindingError("runtime driver does not implement FpoJaxRuntimeDriver")
    if not isinstance(backend, FpoJaxSourceEvaluatorBackend):
        raise AssetBindingError("asset binding requires the FpoJaxSourceEvaluatorBackend")
    alias_nonce, tie_nonce = _read_private_nonces(config)
    alias_commitment = market_nonce_commitment(
        purpose="market_alias",
        nonce=alias_nonce,
        intake_record_digest=intake.intake_record_digest,
    )
    tie_commitment = market_nonce_commitment(
        purpose="market_tie_break",
        nonce=tie_nonce,
        intake_record_digest=intake.intake_record_digest,
    )
    del alias_nonce, tie_nonce

    return_contract = DmcFixedHorizonReturnContract(
        horizon=config.return_horizon,
        per_step_lower=config.per_step_lower,
        per_step_upper=config.per_step_upper,
    )
    source_environments = {
        anchor_id: next(
            job.anchor.environment_instance_digest
            for job in plan.jobs.values()
            if job.anchor.source_anchor_id == anchor_id
        )
        for anchor_id in sorted({job.anchor.source_anchor_id for job in plan.jobs.values()})
    }
    protocol = SourceEvaluationProtocol(
        intake_record_digest=intake.intake_record_digest,
        evaluator_implementation_digest=backend.evaluator_implementation_digest,
        return_contract_digest=return_contract.return_contract_digest,
        selection_seed_namespace_digest=_seed_namespace_digest(
            intake_record_digest=intake.intake_record_digest,
            selection_ledger_digest=selection_ledger["semantic_digest"],
            namespace="source_selection",
            seeds=config.selection_reset_seeds,
        ),
        attestation_seed_namespace_digest=_seed_namespace_digest(
            intake_record_digest=intake.intake_record_digest,
            selection_ledger_digest=selection_ledger["semantic_digest"],
            namespace="source_attestation",
            seeds=config.attestation_reset_seeds,
        ),
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=config.attestation_reset_seeds,
        selection_episodes_per_candidate=len(config.selection_reset_seeds),
        attestation_episodes_per_champion=len(config.attestation_reset_seeds),
        source_environment_digests=source_environments,
        competence_floors={anchor: config.competence_floor for anchor in source_environments},
        mean_tolerance=config.mean_tolerance,
        lcb_z=config.lcb_z,
    )

    # This is the only candidate operation in this tool.  The planner invokes
    # validate_candidate exactly once for each of the 90 cells and never calls
    # evaluate_episode/evaluate_seed_block.
    units = plan_source_selection_work_units(intake, protocol, plan, backend)
    units_manifest = source_work_unit_manifest(units)
    if units_manifest["work_unit_count"] != EXPECTED_JOB_COUNT:
        raise AssetBindingError("validate-only planner did not produce exact-90 work units")
    abi_digests = {
        candidate_id: unit.execution_abi.digest for candidate_id, unit in units.items()
    }
    if len(abi_digests) != EXPECTED_JOB_COUNT:
        raise AssetBindingError("validate-only ABI census is incomplete")

    alias_protocol_payload = {
        "schema": MARKET_ALIAS_PROTOCOL_SCHEMA,
        "intake_record_digest": intake.intake_record_digest,
        "source_pool_digest": intake.source_pool_digest,
        "candidate_count": EXPECTED_JOB_COUNT,
        "market_entry_count": EXPECTED_ANCHOR_COUNT,
        "assignment": MARKET_ALIAS_ASSIGNMENT,
        "alias_commitment_digest": alias_commitment,
    }
    alias_protocol_digest = formal_market_alias_protocol_digest(
        intake_record_digest=intake.intake_record_digest,
        source_pool_digest=intake.source_pool_digest,
        alias_commitment_digest=alias_commitment,
    )
    if alias_protocol_digest != sha256_json(alias_protocol_payload):
        raise AssetBindingError("shared market alias protocol implementation drifted")
    market_plan = FormalMarketPlan(
        intake_record_digest=intake.intake_record_digest,
        source_pool_digest=intake.source_pool_digest,
        source_evaluation_protocol_digest=protocol.source_evaluation_protocol_digest,
        intake_cell_digests_by_candidate={
            candidate_id: cell.intake_cell_digest for candidate_id, cell in intake.cells.items()
        },
        source_anchor_id_by_candidate={
            candidate_id: cell.source_anchor_id for candidate_id, cell in intake.cells.items()
        },
        deployment_abi_digests_by_candidate=abi_digests,
        market_alias_protocol_digest=alias_protocol_digest,
        market_alias_commitment_digest=alias_commitment,
        tie_break_commitment_digest=tie_commitment,
    )
    legacy = build_legacy_asset_inventory(config.legacy_v0_root)
    return {
        "source_evaluation_protocol": protocol.to_dict(),
        "source_selection_work_units": units_manifest,
        "formal_market_plan": market_plan.to_dict(),
        "legacy_asset_inventory": legacy,
        "_receipt_material": {
            "selection_ledger": selection_ledger,
            "seed_derivation_protocol": {
                **_seed_derivation_protocol_payload(),
                "protocol_digest": PRODUCTION_SEED_DERIVATION_PROTOCOL_DIGEST,
            },
            "return_contract": return_contract.to_dict(),
            "market_alias_protocol": {
                **alias_protocol_payload,
                "market_alias_protocol_digest": alias_protocol_digest,
            },
            "market_alias_commitment_digest": alias_commitment,
            "tie_break_commitment_digest": tie_commitment,
            "intake_path": str(intake_path),
            "plan_path": str(plan_path),
            "plan_digest": plan.plan_digest,
            "plan_binding_digest": plan.binding_digest,
        },
    }


def _output_directory(path: str | Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise AssetBindingError("output_dir must be an absolute, non-symlink path")
    if supplied.exists():
        raise AssetBindingError("output_dir already exists; immutable binding refuses resume/overwrite")
    parent = _absolute(supplied.parent, "output_dir parent", kind="dir")
    return parent / supplied.name


def _publish_documents(
    config: ProductionAssetBindingConfig,
    documents: dict[str, dict[str, Any]],
    *,
    runtime_driver: FpoJaxRuntimeDriver,
    backend: FpoJaxSourceEvaluatorBackend,
) -> dict[str, Any]:
    output_dir = _output_directory(config.output_dir)
    material = documents["_receipt_material"]
    labels = (
        "source_evaluation_protocol",
        "source_selection_work_units",
        "formal_market_plan",
        "legacy_asset_inventory",
    )
    receipt_payload_base = {
        "schema": ASSET_BINDING_RECEIPT_SCHEMA,
        "status": ASSET_BINDINGS_READY,
        "formal_run_authorized": False,
        "operation": "VALIDATE_ONLY",
        "rollout_executed": False,
        "training_executed": False,
        "formal_authority_granted": False,
        "output_directory": str(output_dir),
        "candidate_validation_count": EXPECTED_JOB_COUNT,
        "source_anchor_count": EXPECTED_ANCHOR_COUNT,
        "intake": {
            "path": material["intake_path"],
            "file_sha256": config.intake_record_sha256,
            "intake_record_digest": documents["source_evaluation_protocol"]["intake_record_digest"],
            "source_pool_digest": documents["formal_market_plan"]["source_pool_digest"],
        },
        "server_plan": {
            "path": material["plan_path"],
            "file_sha256": config.server_plan_sha256,
            "semantic_digest": material["plan_digest"],
            "binding_digest": material["plan_binding_digest"],
        },
        "selection_ledger": material["selection_ledger"],
        "seed_derivation_protocol": material["seed_derivation_protocol"],
        "runtime": {
            "fpo_root": str(_absolute(config.fpo_root, "fpo_root", kind="dir")),
            "vendor_dir": str(_absolute(config.vendor_dir, "vendor_dir", kind="dir")),
            "runtime_driver_digest": runtime_driver.runtime_driver_digest,
            "evaluator_implementation_digest": backend.evaluator_implementation_digest,
        },
        "return_contract": material["return_contract"],
        "market_alias_protocol": material["market_alias_protocol"],
        "market_alias_commitment_digest": material["market_alias_commitment_digest"],
        "tie_break_commitment_digest": material["tie_break_commitment_digest"],
        "private_nonces_persisted": False,
    }
    with tempfile.TemporaryDirectory(
        dir=output_dir.parent,
        prefix=f".{output_dir.name}.staging-",
    ) as staging_name:
        staging = Path(staging_name)
        artifact_bindings: dict[str, dict[str, str]] = {}
        for label in labels:
            staged_path = staging / _OUTPUT_NAMES[label]
            final_path = output_dir / _OUTPUT_NAMES[label]
            file_sha = atomic_write_json(
                staged_path, documents[label], overwrite=False
            )
            semantic_digest_key = {
                "source_evaluation_protocol": "source_evaluation_protocol_digest",
                "source_selection_work_units": "manifest_digest",
                "formal_market_plan": "plan_digest",
                "legacy_asset_inventory": "inventory_digest",
            }[label]
            artifact_bindings[label] = {
                "path": str(final_path),
                "file_sha256": file_sha,
                "semantic_digest": documents[label][semantic_digest_key],
            }
        receipt_payload = {
            **receipt_payload_base,
            "artifacts": artifact_bindings,
        }
        receipt = {
            **receipt_payload,
            "binding_receipt_digest": sha256_json(receipt_payload),
        }
        staged_receipt = staging / _OUTPUT_NAMES["asset_binding_receipt"]
        receipt_sha = atomic_write_json(staged_receipt, receipt, overwrite=False)
        try:
            os.replace(staging, output_dir)
        except OSError as error:
            raise AssetBindingError(
                "cannot atomically publish the immutable asset binding directory"
            ) from error
    receipt_path = output_dir / _OUTPUT_NAMES["asset_binding_receipt"]
    return {
        **receipt,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_file_sha256": receipt_sha,
    }


def bind_production_assets(config: ProductionAssetBindingConfig) -> dict[str, Any]:
    """Execute the production validate-only binding path and publish its receipt."""

    if not isinstance(config, ProductionAssetBindingConfig):
        raise AssetBindingError("bind_production_assets requires typed configuration")
    trusted_root = _absolute(
        config.trusted_experiment_root, "trusted_experiment_root", kind="dir"
    )
    plan_path = _require_file_matches(
        config.server_plan_path, config.server_plan_sha256, "frozen server plan"
    )
    _require_confined_existing(plan_path, trusted_root, "frozen server plan")
    intake = load_verified_frozen_v02_intake(
        config.intake_record_path,
        expected_artifact_sha256=config.intake_record_sha256,
        trusted_experiment_root=trusted_root,
    )
    plan = FrozenV02ServerPlanAuthority().load(plan_path)
    # These exact production classes are intentionally instantiated here.  The
    # driver constructor verifies physical FPO/vendor provenance but does not
    # import JAX or reserve a device.
    runtime_driver = FrozenV02FpoJaxRuntimeDriver(
        fpo_root=config.fpo_root,
        vendor_dir=config.vendor_dir,
    )
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=runtime_driver,
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=config.attestation_reset_seeds,
    )
    documents = _prepare_documents(
        config,
        intake=intake,
        plan=plan,
        runtime_driver=runtime_driver,
        backend=backend,
    )
    return _publish_documents(
        config, documents, runtime_driver=runtime_driver, backend=backend
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and bind v0.3 production assets without rollout/training"
    )
    parser.add_argument("--intake-record", required=True, type=Path)
    parser.add_argument("--intake-record-sha256", required=True)
    parser.add_argument("--trusted-experiment-root", required=True, type=Path)
    parser.add_argument("--server-plan", required=True, type=Path)
    parser.add_argument("--server-plan-sha256", required=True)
    parser.add_argument("--selection-ledger", required=True, type=Path)
    parser.add_argument("--fpo-root", required=True, type=Path)
    parser.add_argument("--vendor-dir", required=True, type=Path)
    parser.add_argument("--legacy-v0-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--market-alias-private-nonce-file", required=True, type=Path)
    parser.add_argument("--tie-break-private-nonce-file", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ProductionAssetBindingConfig(
        intake_record_path=args.intake_record,
        intake_record_sha256=args.intake_record_sha256,
        trusted_experiment_root=args.trusted_experiment_root,
        server_plan_path=args.server_plan,
        server_plan_sha256=args.server_plan_sha256,
        selection_ledger_path=args.selection_ledger,
        fpo_root=args.fpo_root,
        vendor_dir=args.vendor_dir,
        legacy_v0_root=args.legacy_v0_root,
        output_dir=args.output_dir,
        market_alias_private_nonce_file=args.market_alias_private_nonce_file,
        tie_break_private_nonce_file=args.tie_break_private_nonce_file,
    )
    receipt = bind_production_assets(config)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by deployment CLI
    raise SystemExit(main())


__all__ = [
    "ASSET_BINDING_RECEIPT_SCHEMA",
    "ASSET_BINDINGS_READY",
    "LEGACY_ASSET_INVENTORY_SCHEMA",
    "PRODUCTION_ATTESTATION_RESET_SEEDS",
    "PRODUCTION_COMPETENCE_FLOOR",
    "PRODUCTION_LCB_Z",
    "PRODUCTION_MEAN_TOLERANCE",
    "PRODUCTION_RETURN_HORIZON",
    "PRODUCTION_SEED_DERIVATION_PROTOCOL_DIGEST",
    "PRODUCTION_SELECTION_LEDGER_CONFIG_DIGEST",
    "PRODUCTION_SELECTION_LEDGER_DIGEST",
    "PRODUCTION_SELECTION_LEDGER_EXPERIMENT_ID",
    "PRODUCTION_SELECTION_LEDGER_FILE_SHA256",
    "PRODUCTION_SELECTION_RESET_SEEDS",
    "AssetBindingError",
    "ProductionAssetBindingConfig",
    "bind_production_assets",
    "build_legacy_asset_inventory",
    "main",
]
