#!/usr/bin/env python3
"""Read-only provenance validators for the frozen v0.2 exact-90 handoff.

Validation intentionally has no JAX or MuJoCo import, so the frozen evidence
remains independently checkable on a CPU-only host.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


DIGEST_RE = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

TRAINING_PROTOCOL_SCHEMA = "policy-learnware.v02-training-protocol.v0"
TRAINING_JOB_SCHEMA = "policy-learnware.v02-training-job.v0"
TRAINING_PLAN_SCHEMA = "policy-learnware.v02-training-plan.v0"
ATTEMPT_SCHEMA = "policy-learnware.v02-training-attempt.v0"
TRAINING_RECORD_SCHEMA = "policy-learnware.v02-training-record.v1"
QUEUE_RESULT_SCHEMA = "policy-learnware.v02-queue-result.v0"
EXECUTION_EVIDENCE_SCHEMA = "policy-learnware.v02-execution-evidence.v0"
VENDOR_PROVENANCE_SCHEMA = "policy-learnware.v02-vendor-directory.v0"
IMPLEMENTATION_PROVENANCE_SCHEMA = "policy-learnware.v02-implementation-inventory.v0"
FORMAL_FREEZE_BINDING_SCHEMA = "policy-learnware.v02-formal-freeze-binding.v0"
FORMAL_TRAINING_CONTRACT_SCHEMA = "policy-learnware.v02-formal-training-contract.v0"
RUN_MANIFEST_SCHEMA = "policy-learnware.v02-anchor-training-run.v0"

FPO_SOURCE_ATTESTATION_KEYS = frozenset(
    {
        "fpo_commit",
        "expected_fpo_commit",
        "fpo_commit_matches_expected",
        "fpo_tracked_dirty",
        "fpo_tracked_changes",
        "fpo_head_tree_digest",
        "fpo_worktree_tree_digest",
        "fpo_execution_tree_digest",
        "fpo_source_file_count",
        "fpo_index_flags",
        "fpo_untracked_paths",
    }
)

RUN_RUNTIME_KEYS = frozenset(
    {
        "runner_schema",
        "runner_file",
        "fpo_root",
        *FPO_SOURCE_ATTESTATION_KEYS,
        "runtime_contract",
        "runtime_digest",
        "vendor",
        "implementation",
        "legacy_policy_io_path",
        "pythonpath_vendor_precedence_verified",
        "wandb_mode",
        "python_dont_write_bytecode",
        "host",
        "pid",
        "platform",
        "python",
        "cuda_visible_devices",
        "xla_python_client_preallocate",
        "jax_backend",
        "jax_devices",
        "hardware_contract",
        "hardware_digest",
        "execution_evidence",
        "command",
        "started_at",
    }
)

RUN_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "job",
        "job_digest",
        "attempt_digest",
        "config_digest",
        "execution_purpose",
        "anchor_manifest",
        "anchor_manifest_digest",
        "environment_instance_digest",
        "model_diff_digest",
        "binding_audit",
        "training_protocol_digest",
        "config",
        "num_envs",
        "iterations_per_env",
        "transitions_per_outer",
        "planned_environment_steps",
        "execution_mode",
        "formal_eligible",
        "execution_evidence_digest",
        "runtime",
        "run_manifest_digest",
    }
)

IMPLEMENTATION_FILE_LABELS = frozenset(
    {
        "v02/runner.py",
        "v02/queue_master.py",
        "v02/package_bridge.py",
        "v02/anchor_binding.py",
        "v02/provenance.py",
        "v02/vendor.py",
        "v02/implementation.py",
        "v02/formal_plan.py",
        "legacy/policy_io.py",
        "package/policy/bundle.py",
        "package/policy/evaluate.py",
        "package/policy/loader.py",
        "package/policy/parity.py",
        "package/v02/training.py",
    }
)

FORMAL_GPU_EXECUTION_MODE = "formal_gpu"
AUDIT_SMOKE_EXECUTION_MODE = "audit_smoke"
AUDIT_SMOKE_EXECUTION_PURPOSE = "audit_smoke"
DEVELOPMENT_EXECUTION_PURPOSE = "development_discovery"
FORMAL_EXECUTION_PURPOSE = "v02_freeze_ready"
EXECUTION_PURPOSES = frozenset(
    {
        AUDIT_SMOKE_EXECUTION_PURPOSE,
        DEVELOPMENT_EXECUTION_PURPOSE,
        FORMAL_EXECUTION_PURPOSE,
    }
)


class ContractError(ValueError):
    """A frozen input or published record violates its strict contract."""


class NumericalIntegrityError(RuntimeError):
    """A metric, state tensor, golden pair, or bundle contains non-finite data."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-standard/non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def load_strict_json(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read strict JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"top-level JSON value must be an object: {source}")
    assert_json_value(value, where=str(source))
    return value


def json_ready(value: Any) -> Any:
    """Convert runtime/config values to strict JSON without hiding NaN/Inf."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NumericalIntegrityError(f"non-finite float cannot enter provenance: {value}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError("JSON object keys must already be strings")
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if hasattr(value, "to_dict"):
        return json_ready(value.to_dict())
    if getattr(value, "shape", None) == () and hasattr(value, "item"):
        return json_ready(value.item())
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    raise ContractError(f"value of type {type(value).__name__} is not JSON-serializable")


def assert_json_value(value: Any, *, where: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{where} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_value(item, where=f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{where} contains a non-string object key")
            assert_json_value(item, where=f"{where}.{key}")
        return
    raise ContractError(f"{where} contains unsupported type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    material = json_ready(value)
    assert_json_value(material)
    return json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise ContractError(f"{where} must be a lowercase SHA-256 digest")
    return value


def require_git_commit(value: Any, where: str = "git commit") -> str:
    if not isinstance(value, str) or GIT_COMMIT_RE.fullmatch(value) is None:
        raise ContractError(f"{where} must be a lowercase full Git object ID")
    return value


def require_safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or SAFE_ID_RE.fullmatch(value) is None:
        raise ContractError(f"{where} is not a safe non-empty identifier")
    return value


def require_execution_purpose(value: Any, where: str = "execution_purpose") -> str:
    if not isinstance(value, str) or value not in EXECUTION_PURPOSES:
        raise ContractError(
            f"{where} must be one of {sorted(EXECUTION_PURPOSES)}"
        )
    return value


def require_exact_keys(
    value: Mapping[str, Any], expected: Iterable[str], where: str
) -> None:
    expected_set = set(expected)
    missing = expected_set - set(value)
    unknown = set(value) - expected_set
    if missing or unknown:
        raise ContractError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def validate_self_digest(
    value: Mapping[str, Any], *, key: str, where: str
) -> str:
    digest = require_digest(value.get(key), f"{where}.{key}")
    material = {name: item for name, item in value.items() if name != key}
    expected = sha256_json(material)
    if digest != expected:
        raise ContractError(f"{where}.{key} mismatch: {digest} != {expected}")
    return digest


def validate_run_manifest_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the strict, self-digested runner envelope shared by all consumers."""

    require_exact_keys(value, RUN_MANIFEST_KEYS, "anchor training run manifest")
    if value["schema"] != RUN_MANIFEST_SCHEMA:
        raise ContractError("unsupported anchor training run manifest schema")
    validate_self_digest(value, key="run_manifest_digest", where="run manifest")
    return dict(value)


def validate_run_runtime_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(value, RUN_RUNTIME_KEYS, "run manifest runtime")
    return dict(value)


def validate_run_manifest_server_binding(
    value: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a run envelope to its server job, attempt, anchor, and geometry."""

    run = validate_run_manifest_envelope(value)
    if not isinstance(run["runtime"], Mapping):
        raise ContractError("run manifest runtime must be an object")
    validate_run_runtime_envelope(run["runtime"])
    frozen_job = validate_training_job(job)
    frozen_attempt = validate_attempt(attempt)
    if frozen_attempt["job"] != frozen_job:
        raise ContractError("run attempt embeds a different server job")
    expected = {
        "job": frozen_job,
        "job_digest": frozen_job["job_digest"],
        "attempt_digest": frozen_attempt["attempt_digest"],
        "config_digest": frozen_job["config_digest"],
        "execution_purpose": frozen_job["execution_purpose"],
        "anchor_manifest": dict(anchor),
        "anchor_manifest_digest": anchor.get("manifest_digest"),
        "environment_instance_digest": anchor.get("environment_instance_digest"),
        "model_diff_digest": anchor.get("model_diff_digest"),
        "training_protocol_digest": frozen_job["training_protocol_digest"],
        "config": frozen_job["training_protocol"]["trainer_config"],
        "execution_mode": frozen_attempt["execution_mode"],
        "formal_eligible": frozen_attempt["formal_eligible"],
    }
    failed = [key for key, item in expected.items() if run[key] != item]
    if failed:
        raise ContractError(f"run manifest server binding mismatch: {failed}")

    operator = anchor.get("operator")
    if operator is None:
        changed_leaves: list[str] = []
    elif isinstance(operator, Mapping) and isinstance(operator.get("mutations"), list):
        changed_leaves = sorted(str(row["leaf"]) for row in operator["mutations"])
    else:
        raise ContractError("anchor operator is malformed in run binding")
    audit = run["binding_audit"]
    if not isinstance(audit, Mapping):
        raise ContractError("run manifest binding_audit must be an object")
    require_exact_keys(
        audit,
        {
            "anchor_id",
            "environment_instance_digest",
            "nominal_model_digest",
            "bound_model_digest",
            "changed_leaves",
            "model_diff_digest",
            "source_unchanged",
            "operator_digest",
            "manifest_digest",
        },
        "run manifest binding audit",
    )
    expected_audit = {
        "anchor_id": anchor.get("anchor_id"),
        "environment_instance_digest": anchor.get("environment_instance_digest"),
        "nominal_model_digest": anchor.get("expected_nominal_model_digest"),
        "bound_model_digest": anchor.get("expected_bound_model_digest"),
        "changed_leaves": changed_leaves,
        "model_diff_digest": anchor.get("model_diff_digest"),
        "source_unchanged": True,
        "operator_digest": anchor.get("operator_digest"),
        "manifest_digest": anchor.get("manifest_digest"),
    }
    failed_audit = [key for key, item in expected_audit.items() if audit[key] != item]
    if failed_audit:
        raise ContractError(
            f"run manifest binding audit mismatch: {failed_audit}"
        )

    config = frozen_job["training_protocol"]["trainer_config"]
    num_envs = _positive_int(config.get("num_envs"), "trainer_config.num_envs")
    num_minibatches = _positive_int(
        config.get("num_minibatches"), "trainer_config.num_minibatches"
    )
    batch_size = _positive_int(config.get("batch_size"), "trainer_config.batch_size")
    unroll_length = _positive_int(
        config.get("unroll_length"), "trainer_config.unroll_length"
    )
    transitions = num_minibatches * batch_size * unroll_length
    if transitions % num_envs:
        raise ContractError("trainer transition geometry does not divide num_envs")
    iterations = transitions // num_envs
    maximum = _positive_int(
        frozen_job["training_protocol"]["max_outer_iterations"],
        "training protocol max_outer_iterations",
    )
    expected_geometry = {
        "num_envs": num_envs,
        "iterations_per_env": iterations,
        "transitions_per_outer": transitions,
        "planned_environment_steps": transitions * maximum,
    }
    failed_geometry = [
        key for key, item in expected_geometry.items() if run[key] != item
    ]
    if failed_geometry:
        raise ContractError(f"run manifest geometry mismatch: {failed_geometry}")
    return run


def validate_fpo_source_attestation(
    value: Mapping[str, Any],
    *,
    expected_commit: str,
    require_exact: bool = False,
) -> dict[str, Any]:
    """Validate a live, externally anchored, byte-level FPO source proof."""

    if require_exact:
        require_exact_keys(
            value, FPO_SOURCE_ATTESTATION_KEYS, "FPO source attestation"
        )
    missing = FPO_SOURCE_ATTESTATION_KEYS - set(value)
    if missing:
        raise ContractError(
            f"FPO source attestation misses fields: {sorted(missing)}"
        )
    result = {key: value[key] for key in FPO_SOURCE_ATTESTATION_KEYS}
    frozen = require_git_commit(expected_commit, "expected FPO commit")
    actual = require_git_commit(result["fpo_commit"], "FPO source commit")
    claimed = require_git_commit(
        result["expected_fpo_commit"], "FPO source expected commit"
    )
    if (
        actual != frozen
        or claimed != frozen
        or result["fpo_commit_matches_expected"] is not True
        or result["fpo_tracked_dirty"] is not False
        or result["fpo_tracked_changes"] != []
        or result["fpo_index_flags"] != []
        or result["fpo_untracked_paths"] != []
    ):
        raise ContractError("FPO source attestation is not clean and freeze-matched")
    head = require_digest(
        result["fpo_head_tree_digest"], "FPO source HEAD tree digest"
    )
    worktree = require_digest(
        result["fpo_worktree_tree_digest"], "FPO source worktree tree digest"
    )
    require_digest(
        result["fpo_execution_tree_digest"], "FPO source execution tree digest"
    )
    if head != worktree:
        raise ContractError("FPO source worktree bytes differ from HEAD")
    _positive_int(result["fpo_source_file_count"], "FPO source file count")
    return result


def with_self_digest(value: Mapping[str, Any], *, key: str) -> dict[str, Any]:
    if key in value:
        raise ContractError(f"cannot finalize payload that already contains {key!r}")
    result = json_ready(dict(value))
    result[key] = sha256_json(result)
    return result


def _validate_formal_freeze_record(value: Mapping[str, Any]) -> None:
    """Preserve the historical typed freeze-record contract without its writer."""

    require_exact_keys(
        value,
        {
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
        },
        "formal freeze record",
    )
    expected_literals = {
        "schema": "policy-learnware.v02-formal-protocol-freeze.v0",
        "stage": "v02_freeze_ready",
        "sealed_target_state": "NOT_INSTANTIATED_OR_READ",
        "confirmatory_oracle_state": "NOT_READ",
        "maximum_authorized_status": "READY_FOR_V03_JOINT_CONFIRMATORY",
    }
    if any(value[name] != expected for name, expected in expected_literals.items()):
        raise ContractError("formal freeze boundary/status constant mismatch")
    if not isinstance(value["experiment_id"], str) or not value["experiment_id"]:
        raise ContractError("formal freeze experiment_id must be non-empty")
    for name in (
        "config_digest",
        "config_file_sha256",
        "benchmark_projection_digest",
        "training_projection_digest",
        "probe_projection_digest",
        "analysis_projection_digest",
        "implementation_tree_digest",
    ):
        require_digest(value[name], f"formal freeze record.{name}")
    commit = require_git_commit(
        value["software_commit"], "formal freeze record.software_commit"
    )
    if len(commit) != 40:
        raise ContractError("formal freeze software_commit must be a full SHA-1 commit")
    if value["worktree_clean_at_freeze"] is not True:
        raise ContractError("formal freeze requires a clean Git worktree")


def validate_formal_freeze_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the serialized config/freeze projection embedded in a plan."""

    if not isinstance(value, Mapping):
        raise ContractError("formal freeze binding must be an object")
    require_exact_keys(
        value,
        {
            "schema",
            "config_path",
            "freeze_manifest_path",
            "config_digest",
            "freeze_record",
            "freeze_digest",
            "training_contract",
            "binding_digest",
        },
        "formal freeze binding",
    )
    if value["schema"] != FORMAL_FREEZE_BINDING_SCHEMA:
        raise ContractError("unsupported formal freeze binding schema")
    for name in ("config_path", "freeze_manifest_path"):
        raw = value[name]
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise ContractError(f"formal freeze binding {name} must be an absolute path")
        resolved = Path(raw).resolve()
        if raw != str(resolved):
            raise ContractError(f"formal freeze binding {name} must be canonical")
    config_digest = require_digest(
        value["config_digest"], "formal freeze binding.config_digest"
    )
    record = value["freeze_record"]
    if not isinstance(record, dict):
        raise ContractError("formal freeze binding.freeze_record must be an object")
    _validate_formal_freeze_record(record)
    if record.get("config_digest") != config_digest:
        raise ContractError("formal freeze record config digest differs from its binding")
    freeze_digest = require_digest(
        value["freeze_digest"], "formal freeze binding.freeze_digest"
    )
    if freeze_digest != sha256_json(record):
        raise ContractError("formal freeze record digest mismatch")
    contract = value["training_contract"]
    if not isinstance(contract, Mapping):
        raise ContractError("formal freeze binding.training_contract must be an object")
    require_exact_keys(
        contract,
        {
            "schema",
            "source_anchor_ids",
            "source_anchors",
            "training_seeds",
            "primary_algorithm",
            "training_steps",
            "checkpoint_rule",
            "training_projection_digest",
            "contract_digest",
        },
        "formal training contract",
    )
    if contract["schema"] != FORMAL_TRAINING_CONTRACT_SCHEMA:
        raise ContractError("unsupported formal training contract schema")
    anchor_ids = contract["source_anchor_ids"]
    if (
        not isinstance(anchor_ids, list)
        or len(anchor_ids) != 30
        or anchor_ids != sorted(set(anchor_ids))
    ):
        raise ContractError("formal training contract requires 30 sorted unique anchors")
    for index, anchor_id in enumerate(anchor_ids):
        require_digest(anchor_id, f"formal training contract.source_anchor_ids[{index}]")
    source_anchors = contract["source_anchors"]
    if not isinstance(source_anchors, list) or len(source_anchors) != len(anchor_ids):
        raise ContractError("formal training contract requires one semantic row per anchor")
    semantic_ids: list[str] = []
    row_keys = {
        "source_anchor_id",
        "task",
        "nominal",
        "factor",
        "factor_id",
        "axis_id",
        "operator_id",
        "axis_binding_digest",
        "leaf_allowlist",
    }
    for index, row in enumerate(source_anchors):
        where = f"formal training contract.source_anchors[{index}]"
        if not isinstance(row, Mapping):
            raise ContractError(f"{where} must be an object")
        require_exact_keys(row, row_keys, where)
        anchor_id = require_digest(row["source_anchor_id"], f"{where}.source_anchor_id")
        semantic_ids.append(anchor_id)
        require_safe_id(row["task"], f"{where}.task")
        require_safe_id(row["factor_id"], f"{where}.factor_id")
        nominal = row["nominal"]
        if not isinstance(nominal, bool):
            raise ContractError(f"{where}.nominal must be boolean")
        factor = row["factor"]
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(float(factor))
            or float(factor) <= 0.0
        ):
            raise ContractError(f"{where}.factor must be finite and positive")
        leaves = row["leaf_allowlist"]
        if (
            not isinstance(leaves, list)
            or leaves != sorted(set(leaves))
            or any(not isinstance(leaf, str) or not leaf.startswith("_mjx_model.") for leaf in leaves)
        ):
            raise ContractError(f"{where}.leaf_allowlist must be sorted unique model leaves")
        if nominal:
            if (
                float(factor) != 1.0
                or row["axis_id"] is not None
                or row["operator_id"] is not None
                or row["axis_binding_digest"] is not None
                or leaves
            ):
                raise ContractError(f"{where} has invalid nominal semantics")
        else:
            if float(factor) == 1.0 or not leaves:
                raise ContractError(f"{where} has invalid shifted semantics")
            require_safe_id(row["axis_id"], f"{where}.axis_id")
            require_safe_id(row["operator_id"], f"{where}.operator_id")
            require_digest(row["axis_binding_digest"], f"{where}.axis_binding_digest")
    if semantic_ids != anchor_ids:
        raise ContractError(
            "formal source-anchor semantic rows must match sorted source_anchor_ids"
        )
    seeds = contract["training_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or seeds != sorted(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ContractError("formal training contract requires three sorted unique seeds")
    if contract["primary_algorithm"] not in {"ppo", "fpo"}:
        raise ContractError("formal training contract algorithm must be ppo or fpo")
    _positive_int(contract["training_steps"], "formal training contract.training_steps")
    if contract["checkpoint_rule"] not in {"fixed_final", "fixed_ladder"}:
        raise ContractError(
            "formal checkpoint_rule must be exactly fixed_final or fixed_ladder"
        )
    projection_digest = require_digest(
        contract["training_projection_digest"],
        "formal training contract.training_projection_digest",
    )
    if record.get("training_projection_digest") != projection_digest:
        raise ContractError(
            "formal training contract projection digest differs from freeze record"
        )
    validate_self_digest(
        contract, key="contract_digest", where="formal training contract"
    )
    validate_self_digest(
        value, key="binding_digest", where="formal freeze binding"
    )
    return dict(value)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where} must be a positive integer")
    return value


def validate_training_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "schema",
            "algorithm",
            "trainer_config",
            "max_outer_iterations",
            "export_outer_iterations",
            "evaluation",
            "parity",
            "checkpoint_rule",
            "protocol_digest",
        },
        "training protocol",
    )
    if value["schema"] != TRAINING_PROTOCOL_SCHEMA:
        raise ContractError(f"unsupported training protocol schema: {value['schema']!r}")
    if value["algorithm"] not in {"ppo", "fpo"}:
        raise ContractError("training algorithm must be explicitly ppo or fpo")
    config = value["trainer_config"]
    if not isinstance(config, dict) or not config:
        raise ContractError("trainer_config must be a non-empty complete mapping")
    if "num_timesteps" not in config:
        raise ContractError("trainer_config must explicitly contain num_timesteps")
    _positive_int(config["num_timesteps"], "trainer_config.num_timesteps")
    maximum = _positive_int(value["max_outer_iterations"], "max_outer_iterations")
    exports = value["export_outer_iterations"]
    if (
        not isinstance(exports, list)
        or not exports
        or any(isinstance(item, bool) or not isinstance(item, int) for item in exports)
        or exports != sorted(set(exports))
        or exports[0] <= 0
        or exports[-1] > maximum
    ):
        raise ContractError("export_outer_iterations must be sorted unique positive integers")
    rule = value["checkpoint_rule"]
    if rule not in {"fixed_final", "fixed_ladder"}:
        raise ContractError("checkpoint_rule must be explicit fixed_final or fixed_ladder")
    if maximum not in exports:
        raise ContractError("the frozen final outer iteration must be exported")
    if rule == "fixed_final" and exports != [maximum]:
        raise ContractError("fixed_final requires exactly the final outer iteration")
    evaluation = value["evaluation"]
    if not isinstance(evaluation, dict):
        raise ContractError("evaluation must be an object")
    require_exact_keys(evaluation, {"enabled", "num_envs", "base_seed"}, "evaluation")
    if not isinstance(evaluation["enabled"], bool):
        raise ContractError("evaluation.enabled must be boolean")
    _positive_int(evaluation["num_envs"], "evaluation.num_envs")
    if (
        isinstance(evaluation["base_seed"], bool)
        or not isinstance(evaluation["base_seed"], int)
        or evaluation["base_seed"] < 0
    ):
        raise ContractError("evaluation.base_seed must be a nonnegative integer")
    assert_json_value(config, where="trainer_config")
    parity = value["parity"]
    if not isinstance(parity, dict):
        raise ContractError("parity must be an object")
    require_exact_keys(
        parity,
        {"atol", "rtol", "golden_sample_count", "compiled_sample_count"},
        "parity",
    )
    for name in ("atol", "rtol"):
        tolerance = parity[name]
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise ContractError(f"parity.{name} must be finite and nonnegative")
    if _positive_int(parity["golden_sample_count"], "parity.golden_sample_count") != 8:
        raise ContractError("parity.golden_sample_count must be exactly 8 for the legacy bundle ABI")
    compiled_count = _positive_int(
        parity["compiled_sample_count"], "parity.compiled_sample_count"
    )
    if compiled_count > parity["golden_sample_count"]:
        raise ContractError("compiled parity cannot use more samples than golden parity")
    validate_self_digest(value, key="protocol_digest", where="training protocol")
    return dict(value)


def validate_training_job(value: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "schema",
            "job_id",
            "config_digest",
            "execution_purpose",
            "formal_protocol_freeze_digest",
            "anchor_manifest_path",
            "anchor_manifest_digest",
            "training_protocol",
            "training_protocol_digest",
            "seed",
            "job_digest",
        },
        "training job",
    )
    if value["schema"] != TRAINING_JOB_SCHEMA:
        raise ContractError(f"unsupported training job schema: {value['schema']!r}")
    require_safe_id(value["job_id"], "job_id")
    require_digest(value["config_digest"], "training job.config_digest")
    purpose = require_execution_purpose(
        value["execution_purpose"], "training job.execution_purpose"
    )
    freeze_digest = value["formal_protocol_freeze_digest"]
    if purpose == FORMAL_EXECUTION_PURPOSE:
        require_digest(
            freeze_digest, "training job.formal_protocol_freeze_digest"
        )
    elif freeze_digest is not None:
        raise ContractError(
            "non-formal training job must have formal_protocol_freeze_digest=null"
        )
    if not isinstance(value["anchor_manifest_path"], str) or not value["anchor_manifest_path"]:
        raise ContractError("anchor_manifest_path must be a non-empty frozen path")
    require_digest(value["anchor_manifest_digest"], "anchor_manifest_digest")
    protocol = value["training_protocol"]
    if not isinstance(protocol, dict):
        raise ContractError("training_protocol must be embedded as an object")
    validate_training_protocol(protocol)
    if value["training_protocol_digest"] != protocol["protocol_digest"]:
        raise ContractError("embedded training protocol digest mismatch")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int) or value["seed"] < 0:
        raise ContractError("training seed must be a nonnegative integer")
    validate_self_digest(value, key="job_digest", where="training job")
    return dict(value)


def validate_training_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "schema",
            "config_digest",
            "execution_purpose",
            "formal_protocol_freeze",
            "formal_protocol_freeze_digest",
            "jobs",
            "expected_job_count",
            "plan_digest",
        },
        "training plan",
    )
    if value["schema"] != TRAINING_PLAN_SCHEMA:
        raise ContractError(f"unsupported training plan schema: {value['schema']!r}")
    config_digest = require_digest(
        value["config_digest"], "training plan.config_digest"
    )
    purpose = require_execution_purpose(
        value["execution_purpose"], "training plan.execution_purpose"
    )
    freeze_raw = value["formal_protocol_freeze"]
    freeze_digest = value["formal_protocol_freeze_digest"]
    if purpose == FORMAL_EXECUTION_PURPOSE:
        if not isinstance(freeze_raw, Mapping):
            raise ContractError("formal training plan requires a freeze binding")
        freeze = validate_formal_freeze_binding(freeze_raw)
        if freeze["config_digest"] != config_digest:
            raise ContractError("formal training plan freeze config digest mismatch")
        if freeze_digest != freeze["binding_digest"]:
            raise ContractError("formal training plan freeze binding digest mismatch")
    else:
        if freeze_raw is not None or freeze_digest is not None:
            raise ContractError("non-formal training plan cannot carry a formal freeze binding")
    jobs = value["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ContractError("training plan must contain at least one job")
    if value["expected_job_count"] != len(jobs):
        raise ContractError("training plan expected_job_count mismatch")
    seen_ids: set[str] = set()
    seen_semantics: set[tuple[str, str, int]] = set()
    for raw in jobs:
        if not isinstance(raw, dict):
            raise ContractError("every training plan job must be an object")
        job = validate_training_job(raw)
        if job["config_digest"] != config_digest:
            raise ContractError("training plan job config_digest drifted from plan")
        if job["execution_purpose"] != purpose:
            raise ContractError("training plan job execution_purpose drifted from plan")
        if job["formal_protocol_freeze_digest"] != freeze_digest:
            raise ContractError("training plan job formal freeze binding drifted from plan")
        if job["job_id"] in seen_ids:
            raise ContractError(f"duplicate training job_id: {job['job_id']}")
        semantic = (
            job["anchor_manifest_digest"],
            job["training_protocol_digest"],
            job["seed"],
        )
        if semantic in seen_semantics:
            raise ContractError("duplicate anchor/protocol/seed training unit")
        seen_ids.add(job["job_id"])
        seen_semantics.add(semantic)
    validate_self_digest(value, key="plan_digest", where="training plan")
    return dict(value)


def validate_attempt(value: Mapping[str, Any]) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "schema",
            "plan_digest",
            "job",
            "job_digest",
            "attempt_id",
            "attempt_number",
            "execution_attempt_id",
            "gpu",
            "config_digest",
            "execution_purpose",
            "execution_mode",
            "formal_eligible",
            "implementation",
            "created_at",
            "attempt_digest",
        },
        "training attempt",
    )
    if value["schema"] != ATTEMPT_SCHEMA:
        raise ContractError(f"unsupported attempt schema: {value['schema']!r}")
    require_digest(value["plan_digest"], "attempt.plan_digest")
    if not isinstance(value["job"], dict):
        raise ContractError("attempt.job must be an object")
    job = validate_training_job(value["job"])
    if value["job_digest"] != job["job_digest"]:
        raise ContractError("attempt job_digest mismatch")
    if value["config_digest"] != job["config_digest"]:
        raise ContractError("attempt config_digest mismatch")
    if value["execution_purpose"] != job["execution_purpose"]:
        raise ContractError("attempt execution_purpose mismatch")
    require_safe_id(value["attempt_id"], "attempt_id")
    _positive_int(value["attempt_number"], "attempt_number")
    require_safe_id(value["execution_attempt_id"], "execution_attempt_id")
    if not isinstance(value["gpu"], str) or not value["gpu"].isdigit():
        raise ContractError("attempt.gpu must be one physical GPU index string")
    _validate_execution_mode_projection(
        value["execution_mode"],
        value["formal_eligible"],
        value["execution_purpose"],
        where="training attempt",
    )
    validate_implementation_provenance(value["implementation"])
    if not isinstance(value["created_at"], str) or not value["created_at"]:
        raise ContractError("attempt.created_at must be non-empty")
    validate_self_digest(value, key="attempt_digest", where="training attempt")
    return dict(value)


def _validate_execution_mode_projection(
    execution_mode: Any,
    formal_eligible: Any,
    execution_purpose: Any,
    *,
    where: str,
) -> tuple[str, bool]:
    if execution_mode not in {
        FORMAL_GPU_EXECUTION_MODE,
        AUDIT_SMOKE_EXECUTION_MODE,
    }:
        raise ContractError(f"{where}.execution_mode is unsupported")
    purpose = require_execution_purpose(
        execution_purpose, f"{where}.execution_purpose"
    )
    if not isinstance(formal_eligible, bool):
        raise ContractError(f"{where}.formal_eligible must be boolean")
    if purpose == AUDIT_SMOKE_EXECUTION_PURPOSE:
        if execution_mode != AUDIT_SMOKE_EXECUTION_MODE:
            raise ContractError(
                f"{where}.audit_smoke purpose requires audit_smoke execution mode"
            )
    elif execution_mode != FORMAL_GPU_EXECUTION_MODE:
        raise ContractError(
            f"{where}.{purpose} purpose requires GPU execution mode"
        )
    expected = (
        execution_mode == FORMAL_GPU_EXECUTION_MODE
        and purpose == FORMAL_EXECUTION_PURPOSE
    )
    if formal_eligible is not expected:
        raise ContractError(
            f"{where}.formal_eligible is inconsistent with execution_mode"
        )
    return execution_mode, formal_eligible


def validate_execution_evidence(
    value: Mapping[str, Any],
    *,
    expected_job_digest: str | None = None,
    expected_attempt_digest: str | None = None,
    expected_hardware_digest: str | None = None,
    expected_config_digest: str | None = None,
    expected_execution_purpose: str | None = None,
    expected_attempt_root: Path | str | None = None,
    require_formal: bool = False,
) -> dict[str, Any]:
    """Validate the digest-bound execution-mode and physical-backend claim.

    Passing ``--allow-non-gpu`` is an irreversible evidence downgrade even if
    the process happens to see a GPU.  Formal admission additionally requires
    the real JAX backend to be ``gpu`` and the attempt to live at the exact
    immutable root named by the runner.
    """

    require_exact_keys(
        value,
        {
            "schema",
            "config_digest",
            "execution_purpose",
            "execution_mode",
            "formal_eligible",
            "allow_non_gpu",
            "jax_backend",
            "jax_devices",
            "cuda_visible_devices",
            "hardware_digest",
            "job_digest",
            "attempt_digest",
            "attempt_root",
            "execution_evidence_digest",
        },
        "execution evidence",
    )
    if value["schema"] != EXECUTION_EVIDENCE_SCHEMA:
        raise ContractError("unsupported execution evidence schema")
    mode, eligible = _validate_execution_mode_projection(
        value["execution_mode"],
        value["formal_eligible"],
        value["execution_purpose"],
        where="execution evidence",
    )
    config_digest = require_digest(
        value["config_digest"], "execution evidence.config_digest"
    )
    purpose = require_execution_purpose(
        value["execution_purpose"], "execution evidence.execution_purpose"
    )
    if not isinstance(value["allow_non_gpu"], bool):
        raise ContractError("execution evidence.allow_non_gpu must be boolean")
    if value["allow_non_gpu"] is not (
        purpose == AUDIT_SMOKE_EXECUTION_PURPOSE
        and mode == AUDIT_SMOKE_EXECUTION_MODE
    ):
        raise ContractError(
            "execution evidence allow_non_gpu is inconsistent with execution_mode"
        )
    backend = value["jax_backend"]
    if not isinstance(backend, str) or not backend:
        raise ContractError("execution evidence.jax_backend must be non-empty")
    devices = value["jax_devices"]
    if (
        not isinstance(devices, list)
        or not devices
        or any(not isinstance(item, str) or not item for item in devices)
    ):
        raise ContractError("execution evidence.jax_devices must be non-empty strings")
    visible = value["cuda_visible_devices"]
    if visible is not None and (not isinstance(visible, str) or not visible):
        raise ContractError(
            "execution evidence.cuda_visible_devices must be null or non-empty"
        )
    require_digest(value["hardware_digest"], "execution evidence.hardware_digest")
    require_digest(value["job_digest"], "execution evidence.job_digest")
    require_digest(value["attempt_digest"], "execution evidence.attempt_digest")
    root = value["attempt_root"]
    if not isinstance(root, str) or not root or not Path(root).is_absolute():
        raise ContractError("execution evidence.attempt_root must be an absolute path")

    if mode == FORMAL_GPU_EXECUTION_MODE:
        if value["allow_non_gpu"] or backend != "gpu":
            raise ContractError("GPU execution evidence requires the GPU backend")
        if visible is None or not visible.isdigit():
            raise ContractError(
                "formal execution evidence requires one explicit physical GPU index"
            )
    if require_formal and (
        mode != FORMAL_GPU_EXECUTION_MODE
        or eligible is not True
        or purpose != FORMAL_EXECUTION_PURPOSE
        or value["allow_non_gpu"] is not False
        or backend != "gpu"
    ):
        raise ContractError("non-formal/debug execution evidence is not admissible")

    expected = {
        "job_digest": expected_job_digest,
        "attempt_digest": expected_attempt_digest,
        "hardware_digest": expected_hardware_digest,
        "config_digest": expected_config_digest,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and value[key] != require_digest(
            expected_value, f"expected_{key}"
        ):
            raise ContractError(f"execution evidence {key} mismatch")
    if (
        expected_execution_purpose is not None
        and purpose
        != require_execution_purpose(
            expected_execution_purpose, "expected_execution_purpose"
        )
    ):
        raise ContractError("execution evidence execution_purpose mismatch")
    if expected_attempt_root is not None:
        expected_root = Path(expected_attempt_root).resolve()
        if Path(root).resolve() != expected_root or root != str(expected_root):
            raise ContractError("execution evidence attempt_root mismatch")
    validate_self_digest(
        value,
        key="execution_evidence_digest",
        where="execution evidence",
    )
    return dict(value)


def validate_vendor_provenance(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the immutable projection of the runner's vendored dependencies.

    The tree digest is content-only; the absolute path is recorded separately so
    queue and runner evidence cannot silently substitute a different directory.
    Runtime cache files are excluded by the tree materializer in ``vendor.py``.
    """

    require_exact_keys(
        value,
        {
            "schema",
            "path",
            "tree_digest",
            "file_count",
            "total_bytes",
            "wandb_version",
        },
        "vendor provenance",
    )
    if value["schema"] != VENDOR_PROVENANCE_SCHEMA:
        raise ContractError("unsupported vendor provenance schema")
    path = value["path"]
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ContractError("vendor provenance.path must be an absolute path")
    if str(Path(path)) != path:
        raise ContractError("vendor provenance.path must be normalized")
    require_digest(value["tree_digest"], "vendor provenance.tree_digest")
    _positive_int(value["file_count"], "vendor provenance.file_count")
    _positive_int(value["total_bytes"], "vendor provenance.total_bytes")
    version = value["wandb_version"]
    if not isinstance(version, str) or not version or any(character.isspace() for character in version):
        raise ContractError("vendor provenance.wandb_version must be non-empty and whitespace-free")
    result = dict(value)
    if expected is not None:
        validated_expected = validate_vendor_provenance(expected)
        if canonical_json_bytes(result) != canonical_json_bytes(validated_expected):
            raise ContractError("vendor provenance differs from the expected pinned directory")
    return result


def validate_implementation_provenance(
    value: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact source-byte inventory used for one attempt."""

    require_exact_keys(
        value,
        {"schema", "files", "implementation_digest"},
        "implementation provenance",
    )
    if value["schema"] != IMPLEMENTATION_PROVENANCE_SCHEMA:
        raise ContractError("unsupported implementation provenance schema")
    files = value["files"]
    if not isinstance(files, dict) or set(files) != IMPLEMENTATION_FILE_LABELS:
        raise ContractError("implementation provenance file inventory is not exact")
    for label, metadata in files.items():
        if not isinstance(metadata, dict):
            raise ContractError(f"implementation file {label} metadata must be an object")
        require_exact_keys(metadata, {"bytes", "sha256"}, f"implementation file {label}")
        _positive_int(metadata["bytes"], f"implementation file {label}.bytes")
        require_digest(metadata["sha256"], f"implementation file {label}.sha256")
    digest = require_digest(
        value["implementation_digest"],
        "implementation provenance.implementation_digest",
    )
    material = {"schema": value["schema"], "files": files}
    if digest != sha256_json(material):
        raise ContractError("implementation provenance digest mismatch")
    result = dict(value)
    if expected is not None:
        validated_expected = validate_implementation_provenance(expected)
        if canonical_json_bytes(result) != canonical_json_bytes(validated_expected):
            raise ContractError("implementation provenance differs from expected source bytes")
    return result


def assert_finite_array(value: Any, *, where: str) -> None:
    import numpy as np

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise NumericalIntegrityError(f"{where} has object dtype")
    if array.dtype.kind not in "biufc":
        raise NumericalIntegrityError(f"{where} has non-numerical dtype {array.dtype}")
    if array.dtype.kind in "fc" and not bool(np.all(np.isfinite(array))):
        count = int(array.size - np.count_nonzero(np.isfinite(array)))
        raise NumericalIntegrityError(f"{where} contains {count} non-finite values")


def assert_finite_mapping(value: Any, *, where: str = "metrics") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_mapping(item, where=f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_mapping(item, where=f"{where}[{index}]")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise NumericalIntegrityError(f"{where} is non-finite")
        return
    if value is None:
        raise NumericalIntegrityError(f"{where} is null")
    assert_finite_array(value, where=where)


def validate_policy_bundle(
    bundle_dir: Path | str, *, require_evaluation: bool
) -> dict[str, Any]:
    """Validate legacy bundle bytes plus all actor/obs/golden numerical arrays."""

    import numpy as np

    root = Path(bundle_dir)
    manifest_path = root / "bundle_manifest.json"
    manifest = load_strict_json(manifest_path)
    required = {
        "algorithm",
        "complete",
        "created_at",
        "environment_steps",
        "files",
        "outer_iteration",
        "schema",
        "seed",
        "task",
    }
    require_exact_keys(manifest, required, "legacy bundle manifest")
    if manifest["schema"] != "policy-learnware.policy-bundle.v0" or manifest["complete"] is not True:
        raise ContractError("legacy bundle must be complete policy-bundle.v0")
    files = manifest["files"]
    expected_files = {"actor.npz", "golden_io.npz", "obs_stats.npz", "policy_spec.json", "provenance.json"}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ContractError("legacy bundle file inventory is not exact")
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != expected_files | {"bundle_manifest.json"}:
        raise ContractError(
            "legacy bundle directory inventory is not exact: "
            f"observed={sorted(actual_entries)}"
        )
    for name, metadata in files.items():
        if not isinstance(metadata, dict):
            raise ContractError(f"bundle metadata for {name} must be an object")
        require_exact_keys(metadata, {"bytes", "sha256"}, f"bundle file {name}")
        path = root / name
        if not path.is_file():
            raise ContractError(f"bundle file is missing: {path}")
        if path.stat().st_size != metadata["bytes"]:
            raise ContractError(f"bundle file size mismatch: {path}")
        if sha256_file(path) != require_digest(metadata["sha256"], f"{name}.sha256"):
            raise ContractError(f"bundle file digest mismatch: {path}")
    for name in ("actor.npz", "obs_stats.npz", "golden_io.npz"):
        try:
            with np.load(root / name, allow_pickle=False) as archive:
                if not archive.files:
                    raise NumericalIntegrityError(f"{name} is empty")
                for key in archive.files:
                    assert_finite_array(archive[key], where=f"{name}:{key}")
        except (OSError, ValueError) as error:
            raise ContractError(f"cannot load numerical bundle file {name}: {error}") from error
    spec = load_strict_json(root / "policy_spec.json")
    observation_size = spec.get("observation_size")
    action_size = spec.get("action_size")
    if (
        isinstance(observation_size, bool)
        or not isinstance(observation_size, int)
        or observation_size <= 0
        or isinstance(action_size, bool)
        or not isinstance(action_size, int)
        or action_size <= 0
    ):
        raise ContractError("policy_spec must declare positive observation/action sizes")
    try:
        with np.load(root / "obs_stats.npz", allow_pickle=False) as stats:
            if set(stats.files) != {"count", "mean", "var_sum", "std"}:
                raise ContractError("obs_stats numerical inventory is not exact")
            if np.asarray(stats["count"]).shape != ():
                raise ContractError("obs_stats.count must be scalar")
            for name in ("mean", "var_sum", "std"):
                if np.asarray(stats[name]).shape != (observation_size,):
                    raise ContractError(f"obs_stats.{name} has the wrong shape")
            if bool(np.any(np.asarray(stats["std"]) <= 0.0)):
                raise NumericalIntegrityError("obs_stats.std must be strictly positive")
        with np.load(root / "golden_io.npz", allow_pickle=False) as golden:
            if set(golden.files) != {
                "observation",
                "prng_key_data",
                "raw_action",
                "environment_action",
            }:
                raise ContractError("golden_io numerical inventory is not exact")
            observation = np.asarray(golden["observation"])
            raw_action = np.asarray(golden["raw_action"])
            environment_action = np.asarray(golden["environment_action"])
            key_data = np.asarray(golden["prng_key_data"])
            if observation.shape != (8, observation_size):
                raise ContractError("golden observations must have exact legacy shape [8,D]")
            if raw_action.shape != (8, action_size) or environment_action.shape != raw_action.shape:
                raise ContractError("golden action tensors have the wrong legacy shape")
            if key_data.shape != (2,) or key_data.dtype != np.dtype(np.uint32):
                raise ContractError("golden PRNG key data must have shape [2] and uint32 dtype")
            if not bool(
                np.allclose(
                    environment_action,
                    np.tanh(raw_action),
                    atol=1.0e-6,
                    rtol=1.0e-6,
                )
            ):
                raise NumericalIntegrityError("golden environment action is not tanh(raw_action)")
    except (OSError, ValueError) as error:
        if isinstance(error, (ContractError, NumericalIntegrityError)):
            raise
        raise ContractError(f"cannot validate legacy numerical ABI: {error}") from error
    provenance = load_strict_json(root / "provenance.json")
    evaluation = provenance.get("evaluation")
    if require_evaluation:
        if not isinstance(evaluation, dict) or not evaluation:
            raise NumericalIntegrityError("required bundle evaluation is absent")
        assert_finite_mapping(evaluation, where="bundle.evaluation")
    elif evaluation is not None:
        assert_finite_mapping(evaluation, where="bundle.evaluation")
    return {
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bundle_manifest_digest": sha256_json(manifest),
        "files": {name: files[name]["sha256"] for name in sorted(files)},
    }


def validate_success_record(
    path: Path | str,
    *,
    expected_job_digest: str,
    expected_attempt_digest: str | None = None,
    expected_anchor_manifest_digest: str | None = None,
    expected_environment_instance_digest: str | None = None,
    expected_training_protocol_digest: str | None = None,
    expected_config_digest: str | None = None,
    expected_execution_purpose: str | None = None,
) -> dict[str, Any]:
    value = load_strict_json(path)
    required = {
        "schema",
        "state",
        "config_digest",
        "execution_purpose",
        "job_digest",
        "attempt_digest",
        "anchor_manifest_digest",
        "environment_instance_digest",
        "training_protocol_digest",
        "algorithm",
        "seed",
        "execution_mode",
        "formal_eligible",
        "implementation",
        "execution_evidence_digest",
        "checkpoint_bundles",
        "planned_outer_iterations",
        "completed_outer_iterations",
        "promoted_outer_iteration",
        "planned_environment_steps",
        "completed_environment_steps",
        "promoted_environment_steps",
        "terminal_failure",
        "started_at",
        "finished_at",
        "wall_seconds",
        "record_digest",
    }
    require_exact_keys(value, required, "training success record")
    if value["schema"] != TRAINING_RECORD_SCHEMA or value["state"] not in {
        "succeeded",
        "recovered",
    }:
        raise ContractError("training record is not an admissible v0.2 terminal record")
    if value["job_digest"] != require_digest(expected_job_digest, "expected_job_digest"):
        raise ContractError("training record belongs to another semantic job")
    require_digest(value["config_digest"], "training record.config_digest")
    purpose = require_execution_purpose(
        value["execution_purpose"], "training record.execution_purpose"
    )
    for key in (
        "job_digest",
        "attempt_digest",
        "anchor_manifest_digest",
        "environment_instance_digest",
        "training_protocol_digest",
    ):
        require_digest(value[key], f"training record.{key}")
    expected = {
        "attempt_digest": expected_attempt_digest,
        "anchor_manifest_digest": expected_anchor_manifest_digest,
        "environment_instance_digest": expected_environment_instance_digest,
        "training_protocol_digest": expected_training_protocol_digest,
        "config_digest": expected_config_digest,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and value[key] != require_digest(
            expected_value, f"expected_{key}"
        ):
            raise ContractError(f"training record {key} mismatch")
    if (
        expected_execution_purpose is not None
        and purpose
        != require_execution_purpose(
            expected_execution_purpose, "expected_execution_purpose"
        )
    ):
        raise ContractError("training record execution_purpose mismatch")
    if value["algorithm"] not in {"ppo", "fpo"}:
        raise ContractError("training record algorithm must be ppo or fpo")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int) or value["seed"] < 0:
        raise ContractError("training record seed must be a nonnegative integer")
    _validate_execution_mode_projection(
        value["execution_mode"],
        value["formal_eligible"],
        purpose,
        where="training record",
    )
    validate_implementation_provenance(value["implementation"])
    require_digest(
        value["execution_evidence_digest"],
        "training record.execution_evidence_digest",
    )
    if not isinstance(value["checkpoint_bundles"], list) or not value["checkpoint_bundles"]:
        raise ContractError("admissible training record has no checkpoint bundles")
    planned_outer = _positive_int(
        value["planned_outer_iterations"], "planned_outer_iterations"
    )
    completed_outer = _positive_int(
        value["completed_outer_iterations"], "completed_outer_iterations"
    )
    promoted_outer = _positive_int(
        value["promoted_outer_iteration"], "promoted_outer_iteration"
    )
    planned_steps = _positive_int(
        value["planned_environment_steps"], "planned_environment_steps"
    )
    completed_steps = _positive_int(
        value["completed_environment_steps"], "completed_environment_steps"
    )
    promoted_steps = _positive_int(
        value["promoted_environment_steps"], "promoted_environment_steps"
    )
    if planned_steps % planned_outer != 0:
        raise ContractError("training record planned geometry is not integral")
    per_outer = planned_steps // planned_outer
    if completed_steps != completed_outer * per_outer:
        raise ContractError("training record completed-step geometry drifted")
    if promoted_steps != promoted_outer * per_outer:
        raise ContractError("training record promoted-step geometry drifted")
    terminal_failure = value["terminal_failure"]
    if value["state"] == "succeeded":
        if terminal_failure is not None:
            raise ContractError("succeeded training cannot retain terminal failure metadata")
        if not (
            planned_outer == completed_outer == promoted_outer
            and planned_steps == completed_steps == promoted_steps
        ):
            raise ContractError("succeeded training did not promote its completed final budget")
    else:
        if not (promoted_outer <= completed_outer < planned_outer):
            raise ContractError("recovered training outer-iteration provenance is inconsistent")
        if not (promoted_steps <= completed_steps < planned_steps):
            raise ContractError("recovered training step-budget provenance is inconsistent")
        if not isinstance(terminal_failure, dict):
            raise ContractError("recovered training must retain terminal failure metadata")
        require_exact_keys(
            terminal_failure,
            {"type", "message", "traceback_file", "traceback_sha256"},
            "recovered terminal failure",
        )
        if terminal_failure["type"] != "NumericalIntegrityError":
            raise ContractError("only NumericalIntegrityError is recoverable")
        if not isinstance(terminal_failure["message"], str) or not terminal_failure["message"]:
            raise ContractError("recovered numerical failure message must be non-empty")
        if terminal_failure["traceback_file"] != "recovery_traceback.txt":
            raise ContractError("recovered failure trace path is not canonical")
        require_digest(
            terminal_failure["traceback_sha256"],
            "recovered failure traceback_sha256",
        )
        trace_path = Path(path).resolve().parent / terminal_failure["traceback_file"]
        if not trace_path.is_file():
            raise ContractError("recovered failure trace file is missing")
        if sha256_file(trace_path) != terminal_failure["traceback_sha256"]:
            raise ContractError("recovered failure trace digest mismatch")
    observed_outers: list[int] = []
    for index, checkpoint in enumerate(value["checkpoint_bundles"]):
        if not isinstance(checkpoint, dict):
            raise ContractError(f"checkpoint_bundles[{index}] must be an object")
        require_exact_keys(
            checkpoint,
            {
                "outer_iteration",
                "environment_steps",
                "path",
                "bundle_manifest_sha256",
                "bundle_manifest_digest",
                "files",
                "bundle_digest",
                "config_digest",
                "execution_purpose",
                "execution_mode",
                "formal_eligible",
                "execution_evidence_digest",
                "finiteness_audit",
                "golden_parity",
                "compiled_parity",
            },
            f"checkpoint_bundles[{index}]",
        )
        observed_outers.append(_positive_int(checkpoint["outer_iteration"], "outer_iteration"))
        checkpoint_steps = _positive_int(
            checkpoint["environment_steps"], "environment_steps"
        )
        if checkpoint_steps != observed_outers[-1] * per_outer:
            raise ContractError("checkpoint environment steps drifted from record geometry")
        if not isinstance(checkpoint["path"], str) or not checkpoint["path"]:
            raise ContractError("checkpoint path must be a non-empty string")
        require_digest(checkpoint["bundle_manifest_sha256"], "bundle_manifest_sha256")
        require_digest(checkpoint["bundle_manifest_digest"], "bundle_manifest_digest")
        files = checkpoint["files"]
        expected_files = {
            "actor.npz",
            "golden_io.npz",
            "obs_stats.npz",
            "policy_spec.json",
            "provenance.json",
        }
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ContractError("checkpoint bundle digest inventory is not exact")
        for name, digest in files.items():
            require_digest(digest, f"checkpoint files.{name}")
        if checkpoint["bundle_digest"] != checkpoint["bundle_manifest_sha256"]:
            raise ContractError("checkpoint bundle_digest must be the immutable manifest SHA-256")
        for key in (
            "config_digest",
            "execution_purpose",
            "execution_mode",
            "formal_eligible",
            "execution_evidence_digest",
        ):
            if checkpoint[key] != value[key]:
                raise ContractError(
                    f"checkpoint {key} differs from training record evidence"
                )
        for name in ("finiteness_audit", "golden_parity", "compiled_parity"):
            report = checkpoint[name]
            if not isinstance(report, dict):
                raise ContractError(f"checkpoint {name} must be a digest-bound object")
            if report.get("passed") is not True:
                raise ContractError(f"checkpoint {name} did not pass")
            digest = report.get("report_digest")
            require_digest(digest, f"checkpoint {name}.report_digest")
            material = {key: item for key, item in report.items() if key != "report_digest"}
            if digest != sha256_json(material):
                raise ContractError(f"checkpoint {name} report digest mismatch")
        golden = checkpoint["golden_parity"]
        compiled = checkpoint["compiled_parity"]
        if golden.get("raw_checked") is not True:
            raise ContractError("golden parity must include raw-action replay")
        if compiled.get("next_keys_equal") is not True:
            raise ContractError("compiled parity must preserve next PRNG keys")
    if observed_outers != sorted(set(observed_outers)):
        raise ContractError("checkpoint outer iterations must be sorted and unique")
    if observed_outers[-1] != promoted_outer:
        raise ContractError("promoted checkpoint is not the final validated checkpoint")
    for key in ("started_at", "finished_at"):
        if not isinstance(value[key], str) or not value[key]:
            raise ContractError(f"training record {key} must be non-empty")
    assert_finite_mapping(value["wall_seconds"], where="training record.wall_seconds")
    if isinstance(value["wall_seconds"], bool) or float(value["wall_seconds"]) < 0.0:
        raise ContractError("training record wall_seconds must be nonnegative")
    validate_self_digest(value, key="record_digest", where="training record")
    return value


def validate_queue_result(
    value: Mapping[str, Any],
    *,
    expected_job_digest: str,
    expected_attempt_digest: str,
    expected_config_digest: str | None = None,
    expected_execution_purpose: str | None = None,
) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
            "schema",
            "state",
            "job_digest",
            "attempt_digest",
            "gpu",
            "config_digest",
            "execution_purpose",
            "execution_mode",
            "formal_eligible",
            "pid",
            "returncode",
            "started_at",
            "finished_at",
            "elapsed_seconds",
            "validation_error",
            "command",
            "vendor",
            "implementation",
            "result_digest",
        },
        "queue result",
    )
    if value["schema"] != QUEUE_RESULT_SCHEMA:
        raise ContractError(f"unsupported queue result schema: {value['schema']!r}")
    if value["state"] not in {"succeeded", "failed", "interrupted"}:
        raise ContractError("queue result state is invalid")
    if value["job_digest"] != require_digest(expected_job_digest, "expected_job_digest"):
        raise ContractError("queue result belongs to another semantic job")
    if value["attempt_digest"] != require_digest(
        expected_attempt_digest, "expected_attempt_digest"
    ):
        raise ContractError("queue result belongs to another execution attempt")
    if not isinstance(value["gpu"], str) or not value["gpu"].isdigit():
        raise ContractError("queue result gpu must be one physical GPU index string")
    config_digest = require_digest(
        value["config_digest"], "queue result.config_digest"
    )
    purpose = require_execution_purpose(
        value["execution_purpose"], "queue result.execution_purpose"
    )
    if (
        expected_config_digest is not None
        and config_digest
        != require_digest(expected_config_digest, "expected_config_digest")
    ):
        raise ContractError("queue result config_digest mismatch")
    if (
        expected_execution_purpose is not None
        and purpose
        != require_execution_purpose(
            expected_execution_purpose, "expected_execution_purpose"
        )
    ):
        raise ContractError("queue result execution_purpose mismatch")
    _validate_execution_mode_projection(
        value["execution_mode"],
        value["formal_eligible"],
        purpose,
        where="queue result",
    )
    _positive_int(value["pid"], "queue result.pid")
    if isinstance(value["returncode"], bool) or not isinstance(value["returncode"], int):
        raise ContractError("queue result returncode must be an integer")
    for key in ("started_at", "finished_at"):
        if not isinstance(value[key], str) or not value[key]:
            raise ContractError(f"queue result {key} must be non-empty")
    assert_finite_mapping(value["elapsed_seconds"], where="queue result.elapsed_seconds")
    if isinstance(value["elapsed_seconds"], bool) or float(value["elapsed_seconds"]) < 0.0:
        raise ContractError("queue result elapsed_seconds must be nonnegative")
    if value["validation_error"] is not None and not isinstance(value["validation_error"], str):
        raise ContractError("queue result validation_error must be null or a string")
    if (
        not isinstance(value["command"], list)
        or not value["command"]
        or any(not isinstance(item, str) or not item for item in value["command"])
    ):
        raise ContractError("queue result command must be a non-empty string list")
    validate_vendor_provenance(value["vendor"])
    validate_implementation_provenance(value["implementation"])
    if value["state"] == "succeeded" and (
        value["returncode"] != 0 or value["validation_error"] is not None
    ):
        raise ContractError("successful queue result must have returncode=0 and no validation error")
    validate_self_digest(value, key="result_digest", where="queue result")
    return dict(value)


__all__ = [
    "ContractError",
    "FORMAL_EXECUTION_PURPOSE",
    "FORMAL_GPU_EXECUTION_MODE",
    "NumericalIntegrityError",
    "json_ready",
    "load_strict_json",
    "require_digest",
    "require_git_commit",
    "require_exact_keys",
    "require_safe_id",
    "sha256_file",
    "sha256_json",
    "utc_now",
    "validate_attempt",
    "validate_execution_evidence",
    "validate_fpo_source_attestation",
    "validate_implementation_provenance",
    "validate_policy_bundle",
    "validate_queue_result",
    "validate_run_manifest_server_binding",
    "validate_self_digest",
    "validate_success_record",
    "validate_training_job",
    "validate_training_plan",
    "validate_vendor_provenance",
    "with_self_digest",
]
