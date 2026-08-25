"""Fail-closed, file-driven orchestration for v0.2 development artifacts.

This namespace intentionally ends at the v0.2 freeze-ready handoff.  It has no
command or argument capable of collecting Paper-I sealed targets, publishing
joint rankings, unlocking the confirmatory oracle, or evaluating that oracle.
Those capabilities belong exclusively to the separate Paper-I orchestrator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from ..hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from ..io import atomic_write_json
from .audit import (
    PUBLIC_MARKET_ENTRY_FIELDS,
    PublicArtifactRule,
    artifact_tree_digest,
    audit_public_artifacts,
    audit_public_market_entries,
)
from .axis_catalog import V02_TASKS, build_candidate_axis_catalog
from .axis_integration import axis_registry_from_config
from .config import (
    FORMAL_V02_METHOD_IDS,
    V02ConfigError,
    load_v02_config_draft,
    load_v02_experiment_config,
    load_v02_formal_config,
)
from .freeze import (
    FormalFreezeError,
    FormalProtocolFreeze,
    canonical_formal_freeze_path,
    load_verified_formal_freeze,
)


RECOMPUTE_CHECKS = (
    "full_digest_coverage",
    "full_selector_replay",
    "full_statistical_recompute",
    "raw_numeric_subset_coverage",
    "cost_recompute",
    "information_isolation",
)
RECOMPUTE_SECTIONS = (
    "source",
    "gate0",
    "representations",
    "selectors",
    "oracle",
    "statistics",
    "costs",
    "information",
)

# Method identities are part of the public scientific protocol.  A caller may
# choose hyperparameters only in a reviewed config/fit plan; it may not relabel
# one implementation (for example B0 random) as another comparator.  Methods
# backed by an external adapter remain non-executable until their conformance
# receipt has been admitted.
METHOD_KIND_REGISTRY: Mapping[str, str] = {
    "B0": "random",
    "B1": "competence",
    "B2": "legacy_taskspec",
    "B3a": "vector_nearest",
    "B3b": "environment_only",
    "B4a": "knn_development",
    "B4b": "linear_development",
    "A-Env": "environment_only",
    "M02/B5": "lmin",
}
FORMAL_METHOD_IDS = frozenset(METHOD_KIND_REGISTRY)
if FORMAL_METHOD_IDS != FORMAL_V02_METHOD_IDS:  # source-owned registry drift guard
    raise RuntimeError("v0.2 config and CLI method registries differ")

_RESTRICTED_PATH_TOKENS = (
    "artifacts_paper1_joint",
    "paper1_joint",
    "paper1-joint",
    "joint_confirmatory",
    "joint-confirmatory",
    "confirmatory_oracle_private",
    "oracle_unlock",
    "sealed_target",
    "sealed-target",
)


class V02CommandError(RuntimeError):
    """A CLI input or artifact violates a v0.2 orchestration boundary."""


def _load_config_for_command(path: str | Path) -> Any:
    """Load a stage config and require the immutable freeze on formal paths."""

    config = load_v02_experiment_config(path)
    if config.stage == "v02_freeze_ready":
        try:
            verified, _freeze = load_verified_formal_freeze(path)
        except (FormalFreezeError, V02ConfigError) as error:
            raise V02CommandError(f"formal protocol freeze verification failed: {error}") from error
        if verified.config_digest != config.config_digest:
            raise V02CommandError("formal freeze loader returned another config")
        return verified
    return config


def _strict_keys(value: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise V02CommandError(f"{where} must be a string-keyed object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise V02CommandError(
            f"invalid {where} keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise V02CommandError(f"{where} must be a SHA-256 digest")
    result = value.lower()
    try:
        int(result, 16)
    except ValueError as error:
        raise V02CommandError(f"{where} must be a SHA-256 digest") from error
    return result


def _guard_path(value: str | Path, where: str, *, must_exist: bool = False) -> Path:
    path = Path(value).expanduser()
    lowered = path.as_posix().lower()
    if any(token in lowered for token in _RESTRICTED_PATH_TOKENS):
        raise V02CommandError(
            f"{where} points into a Paper-I joint/sealed capability domain"
        )
    if path.is_symlink():
        raise V02CommandError(f"{where} cannot be a symlink")
    resolved = path.resolve()
    resolved_lowered = resolved.as_posix().lower()
    if any(token in resolved_lowered for token in _RESTRICTED_PATH_TOKENS):
        raise V02CommandError(
            f"{where} resolves into a Paper-I joint/sealed capability domain"
        )
    if must_exist and not resolved.exists():
        raise V02CommandError(f"{where} does not exist: {resolved}")
    return resolved


def _load_json(path: str | Path, where: str) -> Mapping[str, Any]:
    source = _guard_path(path, where, must_exist=True)
    if not source.is_file():
        raise V02CommandError(f"{where} must be a regular JSON file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V02CommandError(f"cannot read {where}: {error}") from error
    if not isinstance(value, Mapping):
        raise V02CommandError(f"{where} must contain a JSON object")
    return value


def _evidence_file_ref(path: str | Path, where: str) -> dict[str, str]:
    source = _guard_path(path, where, must_exist=True)
    if not source.is_file():
        raise V02CommandError(f"{where} must be a regular file")
    return {"path": str(source), "file_sha256": sha256_file(source)}


def _verify_evidence_file_ref(raw: Any, where: str) -> Path:
    reference = _strict_keys(raw, {"path", "file_sha256"}, f"{where} reference")
    expected = _digest(reference["file_sha256"], f"{where} file_sha256")
    source = _guard_path(reference["path"], where, must_exist=True)
    if not source.is_file() or sha256_file(source) != expected:
        raise V02CommandError(f"{where} bytes differ from the admitted evidence receipt")
    return source


def _load_evidence_file_ref(raw: Any, where: str) -> tuple[Path, Mapping[str, Any]]:
    source = _verify_evidence_file_ref(raw, where)
    return source, _load_json(source, where)


def _resolve_relative_file(root: Path, relative: Any, where: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise V02CommandError(f"{where} must be a non-empty relative path")
    lexical = Path(relative)
    if lexical.is_absolute() or ".." in lexical.parts or "\\" in relative:
        raise V02CommandError(f"{where} must be traversal-free and relative")
    current = root
    for part in lexical.parts:
        current = current / part
        if current.is_symlink():
            raise V02CommandError(f"{where} traverses a symlink")
    resolved = current.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise V02CommandError(f"{where} escapes its declared root") from error
    if not resolved.is_file():
        raise V02CommandError(f"{where} does not identify a regular file")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _server_training_bridge() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Load the tracked sibling bridge from this exact repository checkout."""

    repository_root = Path(__file__).resolve().parents[3]
    bridge_path = repository_root / "server" / "repro_fpo_ppo_v02" / "package_bridge.py"
    if not bridge_path.is_file():
        raise V02CommandError(
            "the versioned anchor-aware server training bridge is unavailable"
        )
    root_text = str(repository_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        from server.repro_fpo_ppo_v02.package_bridge import (
            admit_server_success_batch,
            project_policy_training_plan,
        )
    except (ImportError, ValueError) as error:
        raise V02CommandError(f"cannot load the server training bridge: {error}") from error
    return project_policy_training_plan, admit_server_success_batch


def _publish(path: str | Path | None, payload: Mapping[str, Any], *, resume: bool) -> str | None:
    if path is None:
        if resume:
            raise V02CommandError("--resume requires --output")
        return None
    destination = _guard_path(path, "output")
    expected = canonical_json_bytes(payload) + b"\n"
    if destination.exists():
        if not resume:
            raise V02CommandError(f"immutable output already exists: {destination}")
        if not destination.is_file() or destination.read_bytes() != expected:
            raise V02CommandError("resume output differs from independently recomputed bytes")
        return sha256_file(destination)
    if resume:
        raise V02CommandError("--resume cannot accept a missing output artifact")
    return atomic_write_json(destination, payload, overwrite=False)


def _emit(payload: Mapping[str, Any], stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    stream.write(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")


def _typed_payload(
    value: Any,
    *,
    schema: str,
    fields: set[str],
    where: str,
) -> Mapping[str, Any]:
    data = _strict_keys(value, fields | {"schema"}, where)
    if data["schema"] != schema:
        raise V02CommandError(f"unsupported {where} schema")
    return {name: data[name] for name in fields}


def _mapping_rows(value: Any, where: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise V02CommandError(f"{where} must be a non-empty list")
    if any(not isinstance(item, Mapping) for item in value):
        raise V02CommandError(f"{where} must contain only objects")
    return tuple(value)


def _publish_many(
    artifacts: Sequence[tuple[Path, Mapping[str, Any]]], *, resume: bool
) -> Mapping[str, str]:
    """Preflight a small immutable publication set before writing any member."""

    normalized = tuple((_guard_path(path, "output"), payload) for path, payload in artifacts)
    paths = tuple(path for path, _ in normalized)
    if len(paths) != len(set(paths)):
        raise V02CommandError("multi-artifact outputs must be distinct")
    expected = {
        path: canonical_json_bytes(payload) + b"\n" for path, payload in normalized
    }
    for path in paths:
        if path.exists():
            if not resume:
                raise V02CommandError(f"immutable output already exists: {path}")
            if not path.is_file() or path.read_bytes() != expected[path]:
                raise V02CommandError(f"resume content mismatch: {path}")
    digests: dict[str, str] = {}
    for path, payload in normalized:
        if not path.exists():
            atomic_write_json(path, payload, overwrite=False)
        digests[str(path)] = sha256_file(path)
    return digests


def _layout_for_config(config: Any) -> Any:
    from .artifacts import V02ArtifactLayout

    return V02ArtifactLayout(Path(config.artifact_root), config.experiment_id)


def _require_path(path: Path, expected: Path, where: str) -> None:
    if path.resolve() != expected.resolve():
        raise V02CommandError(f"{where} must use canonical path {expected.resolve()}")


def _configured_anchor_ids(config: Any) -> frozenset[str]:
    return frozenset(
        factor.source_anchor_id
        for task in config.tasks
        for axis in config.dynamics_axes[task]
        for factor in config.source_factors[task][axis.axis_id]
    )


def _verify_server_anchor_semantics(
    config: Any,
    *,
    anchor_plan: Mapping[str, Any],
    anchor_paths: Mapping[str, Path],
) -> None:
    """Bind reviewed config factors to the exact materialized server anchors."""

    # Import only after the tracked bridge has made this checkout's server
    # package available.  The server type independently validates every
    # self-digest, runtime, model, operator and source-anchor identity field.
    _server_training_bridge()
    from server.repro_fpo_ppo_v02.anchor_binding import AnchorManifest

    if set(anchor_paths) != set(anchor_plan):
        raise V02CommandError("server anchor paths differ from the package anchor plan")
    by_anchor: dict[str, list[tuple[str, Any, Any]]] = {}
    for task in config.tasks:
        for axis in config.dynamics_axes[task]:
            for factor in config.source_factors[task][axis.axis_id]:
                by_anchor.setdefault(factor.source_anchor_id, []).append(
                    (task, axis, factor)
                )
    if set(by_anchor) != set(anchor_paths):
        raise V02CommandError("config/source anchor semantic coverage differs")
    registry = axis_registry_from_config(config, _candidate_catalog_for_config(config))
    for anchor_id, path in sorted(anchor_paths.items()):
        try:
            manifest = AnchorManifest.from_path(path)
        except Exception as error:
            raise V02CommandError(f"invalid server anchor manifest {anchor_id}: {error}") from error
        plan_row = _strict_keys(
            anchor_plan[anchor_id],
            {"environment_instance_digest", "anchor_manifest_digest"},
            f"anchor plan row {anchor_id}",
        )
        rows = by_anchor[anchor_id]
        tasks = {task for task, _axis, _factor in rows}
        if len(tasks) != 1:
            raise V02CommandError("one source anchor cannot span multiple tasks")
        task = next(iter(tasks))
        factors = {factor.value for _task, _axis, factor in rows}
        nominal = all(factor.is_nominal for _task, _axis, factor in rows)
        checks = {
            "anchor_id": manifest.anchor_id == anchor_id,
            "task": manifest.task == task,
            "nominal": manifest.nominal is nominal,
            "factor": len(factors) == 1 and manifest.factor == next(iter(factors)),
            "environment_instance_digest": (
                manifest.environment_instance_digest
                == plan_row["environment_instance_digest"]
            ),
            "anchor_manifest_digest": manifest.manifest_digest == plan_row["anchor_manifest_digest"],
        }
        if nominal:
            checks.update(
                {
                    "nominal_operator": manifest.operator is None,
                    "nominal_axis_binding": manifest.axis_binding_digest is None,
                }
            )
        else:
            if len(rows) != 1 or manifest.operator is None:
                raise V02CommandError("shifted source anchor must bind exactly one reviewed axis")
            _task, axis, factor = rows[0]
            # AnchorManifest validates mutation leaves against the server's
            # fully-qualified ``_mjx_model.*`` allowlist.  Preserve those
            # canonical paths here; prefixing them again would make every
            # valid shifted anchor fail the package/config semantic check.
            manifest_leaves = {
                mutation.leaf for mutation in manifest.operator.mutations
            }
            checks.update(
                {
                    "axis_id": manifest.operator.axis_id == axis.axis_id,
                    "operator_id": manifest.operator.operator_id == axis.operator_id,
                    "axis_registry_digest": (
                        manifest.operator.axis_registry_digest == registry.digest
                    ),
                    "axis_binding_digest": (
                        manifest.axis_binding_digest == factor.axis_binding_digest
                    ),
                    "leaf_allowlist": manifest_leaves == set(axis.leaf_allowlist),
                }
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise V02CommandError(
                f"server anchor {anchor_id} differs from reviewed config/plan: {failed}"
            )


def _require_registered_method_contract(config: Any) -> None:
    observed = set(config.method_ids)
    unknown = observed - FORMAL_METHOD_IDS
    if unknown:
        raise V02CommandError(
            f"config contains unregistered v0.2 method IDs: {sorted(unknown)}"
        )
    if config.stage == "v02_freeze_ready" and observed != FORMAL_METHOD_IDS:
        raise V02CommandError(
            "formal v0.2 must freeze the exact B0/B1/B2/B3a/B3b/B4a/B4b/"
            "A-Env/M02-B5 comparator registry"
        )


def _validate_config(args: argparse.Namespace) -> Mapping[str, Any]:
    config_path = _guard_path(args.config, "config", must_exist=True)
    if args.draft:
        if args.expect_stage is not None:
            raise V02CommandError("--expect-stage cannot be combined with --draft")
        draft = load_v02_config_draft(config_path)
        result: dict[str, Any] = {
            "schema": "policy-learnware.v02-config-validation.v0",
            "passed": True,
            "config_digest": draft.config_digest,
            "executable": not draft.unresolved_fields,
            "unresolved_fields": list(draft.unresolved_fields),
            "mode": "draft-structure-only",
        }
    else:
        config = load_v02_experiment_config(config_path)
        if args.expect_stage is not None and config.stage != args.expect_stage:
            raise V02CommandError(
                f"config stage {config.stage!r} differs from --expect-stage"
            )
        if config.stage == "v02_freeze_ready":
            config = load_v02_formal_config(config_path)
        result = {
            "schema": "policy-learnware.v02-config-validation.v0",
            "passed": True,
            "config_digest": config.config_digest,
            "executable": True,
            "unresolved_fields": [],
            "mode": "strict-executable",
            "experiment_id": config.experiment_id,
            "stage": config.stage,
            "task_count": len(config.tasks),
            "axis_count": sum(len(config.dynamics_axes[task]) for task in config.tasks),
            "source_anchor_count": len(
                {
                    factor.source_anchor_id
                    for task in config.tasks
                    for axis in config.dynamics_axes[task]
                    for factor in config.source_factors[task][axis.axis_id]
                }
            ),
        }
    _publish(args.output, result, resume=args.resume)
    return result


def _candidate_catalog_for_config(config: Any) -> Any:
    """Build the reviewed catalog from explicit config factors, never defaults."""

    triples = tuple(
        tuple(factor.value for factor in config.source_factors[task][axis.axis_id])
        for task in config.tasks
        for axis in config.dynamics_axes[task]
        if len(config.source_factors[task][axis.axis_id]) == 3
    )
    if not triples:
        raise V02CommandError(
            "axis audit requires at least one explicit low/nominal/high source-factor triple"
        )
    candidate_catalog, _ = build_candidate_axis_catalog(triples[0])
    return candidate_catalog


def _freeze_run(args: argparse.Namespace) -> Mapping[str, Any]:
    config_path = _guard_path(args.config, "config", must_exist=True)
    config = load_v02_formal_config(config_path)
    layout = _layout_for_config(config)
    output = _guard_path(args.output, "output")
    _require_path(output, canonical_formal_freeze_path(config), "freeze output")
    repository_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise V02CommandError(f"cannot verify formal Git release state: {error}") from error
    if porcelain:
        raise V02CommandError(
            "formal freeze requires a clean Git worktree; commit or remove every change first"
        )
    try:
        record = FormalProtocolFreeze.create(
            config,
            config_path=config_path,
            software_commit=commit,
            worktree_clean_at_freeze=True,
            repository_root=repository_root,
        )
    except FormalFreezeError as error:
        raise V02CommandError(f"cannot construct formal protocol freeze: {error}") from error
    result = record.to_dict()
    _publish(output, result, resume=args.resume)
    return result


_ABI_RUNTIME_CHECKS = {
    "observation_shape_compatible",
    "action_shape_compatible",
    "runtime_loadable",
    "golden_action_finite",
    "compiled_action_finite",
    "selected_only_private_lookup",
}


def _audit_environment_abi(args: argparse.Namespace) -> Mapping[str, Any]:
    from .schemas import ExecutionABIRecord

    payload = _strict_keys(
        _load_json(args.manifest, "environment ABI manifest"),
        {"schema", "records"},
        "environment ABI manifest",
    )
    if payload["schema"] != "policy-learnware.v02-execution-abi-audit-manifest.v0":
        raise V02CommandError("unsupported environment ABI audit manifest schema")
    records = _mapping_rows(payload["records"], "environment ABI records")
    seen: set[str] = set()
    violations: list[dict[str, str]] = []
    record_digests: dict[str, str] = {}
    for index, raw in enumerate(records):
        row = _strict_keys(
            raw,
            {"opaque_learnware_id", "execution_abi", "runtime_checks"},
            f"environment ABI records[{index}]",
        )
        opaque_id = row["opaque_learnware_id"]
        if not isinstance(opaque_id, str) or not opaque_id:
            raise V02CommandError("environment ABI record has an invalid opaque ID")
        if opaque_id in seen:
            violations.append({"opaque_learnware_id": opaque_id, "reason": "duplicate_record"})
            continue
        seen.add(opaque_id)
        abi = ExecutionABIRecord.from_dict(row["execution_abi"])
        checks = _strict_keys(
            row["runtime_checks"], _ABI_RUNTIME_CHECKS, "execution ABI runtime checks"
        )
        failed = tuple(
            name for name in sorted(_ABI_RUNTIME_CHECKS) if checks[name] is not True
        )
        if failed:
            violations.extend(
                {"opaque_learnware_id": opaque_id, "reason": name} for name in failed
            )
        record_digests[opaque_id] = abi.digest
    result = {
        "schema": "policy-learnware.v02-execution-abi-audit.v0",
        "passed": not violations,
        "record_count": len(seen),
        "execution_abi_digests": dict(sorted(record_digests.items())),
        "public_compatibility_filter_used": False,
        "selected_only_private_lookup": all(
            item["reason"] != "selected_only_private_lookup" for item in violations
        ),
        "violations": violations,
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _policy_job(raw: Any) -> Any:
    from .training import PolicyTrainingJob

    fields = set(PolicyTrainingJob.__dataclass_fields__)
    values = _typed_payload(
        raw,
        schema="policy-learnware.v02-policy-training-job.v0",
        fields=fields,
        where="policy training job",
    )
    return PolicyTrainingJob(**values)


def _policy_attestation(raw: Any) -> Any:
    from .training import PolicyTrainingAttestation

    fields = set(PolicyTrainingAttestation.__dataclass_fields__)
    values = _typed_payload(
        raw,
        schema="policy-learnware.v02-policy-training-attestation.v0",
        fields=fields,
        where="policy training attestation",
    )
    attestation = PolicyTrainingAttestation(**values)
    if attestation.bundle_path is not None:
        _guard_path(attestation.bundle_path, "attested bundle path", must_exist=True)
    return attestation


def _plan_training(args: argparse.Namespace) -> Mapping[str, Any]:
    from .training import plan_training_jobs

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    anchors = _strict_keys(
        _load_json(args.anchors, "anchor plan"),
        {"schema", "config_digest", "anchors"},
        "anchor plan",
    )
    if anchors["schema"] != "policy-learnware.v02-training-anchor-plan.v0":
        raise V02CommandError("unsupported training anchor plan schema")
    if anchors["config_digest"] != config.config_digest:
        raise V02CommandError("training anchor plan is bound to another config")
    if not isinstance(anchors["anchors"], Mapping) or not anchors["anchors"]:
        raise V02CommandError("training anchor plan cannot be empty")
    configured_anchor_ids = {
        factor.source_anchor_id
        for task in config.tasks
        for axis in config.dynamics_axes[task]
        for factor in config.source_factors[task][axis.axis_id]
    }
    supplied_anchor_ids = set(anchors["anchors"])
    unexpected = supplied_anchor_ids - configured_anchor_ids
    if unexpected:
        raise V02CommandError(
            f"training anchor plan contains anchors outside the frozen config: {sorted(unexpected)}"
        )
    if config.stage != "audit_smoke" and supplied_anchor_ids != configured_anchor_ids:
        missing = configured_anchor_ids - supplied_anchor_ids
        raise V02CommandError(
            f"training anchor plan does not cover the frozen source market: {sorted(missing)}"
        )
    if config.stage == "v02_freeze_ready" and len(supplied_anchor_ids) != 30:
        raise V02CommandError("formal training plan must contain exactly 30 source anchors")
    trainer = _strict_keys(
        _load_json(args.trainer_contract, "trainer contract"),
        {
            "schema",
            "config_digest",
            "trainer_config",
            "trainer_commit",
            "dependency_digest",
            "runtime_digest",
            "training_protocol_id",
        },
        "trainer contract",
    )
    if trainer["schema"] != "policy-learnware.v02-trainer-contract.v0":
        raise V02CommandError("unsupported trainer contract schema")
    if trainer["config_digest"] != config.config_digest:
        raise V02CommandError("trainer contract is bound to another config")
    jobs = plan_training_jobs(
        anchors["anchors"],
        config_digest=config.config_digest,
        execution_purpose=config.stage,
        algorithm=config.primary_algorithm.lower(),
        seeds=config.training_seeds,
        environment_steps=config.training_steps,
        checkpoint_rule=config.checkpoint_rule,
        trainer_config=trainer["trainer_config"],
        trainer_commit=trainer["trainer_commit"],
        dependency_digest=trainer["dependency_digest"],
        runtime_digest=trainer["runtime_digest"],
        training_protocol_id=trainer["training_protocol_id"],
    )
    result = {
        "schema": "policy-learnware.v02-policy-training-job-plan.v0",
        "config_digest": config.config_digest,
        "anchor_plan_digest": sha256_json(anchors),
        "trainer_contract_digest": sha256_json(trainer),
        "job_count": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
    }
    projection_values = (
        args.server_projection_inputs,
        args.server_plan_output,
        args.server_binding_output,
    )
    projection_requested = any(value is not None for value in projection_values)
    if projection_requested and not all(value is not None for value in projection_values):
        raise V02CommandError(
            "server projection requires --server-projection-inputs, "
            "--server-plan-output, and --server-binding-output together"
        )
    if config.stage == "v02_freeze_ready" and not projection_requested:
        raise V02CommandError(
            "formal plan-training requires the runnable package-to-server projection"
        )
    if projection_requested:
        projection = _strict_keys(
            _load_json(args.server_projection_inputs, "server projection inputs"),
            {
                "schema",
                "config_digest",
                "anchor_manifest_paths",
                "training_protocol_paths",
            },
            "server projection inputs",
        )
        if projection["schema"] != "policy-learnware.v02-server-training-projection-inputs.v0":
            raise V02CommandError("unsupported server training projection input schema")
        if projection["config_digest"] != config.config_digest:
            raise V02CommandError("server projection inputs are bound to another config")
        raw_anchor_paths = projection["anchor_manifest_paths"]
        raw_protocol_paths = projection["training_protocol_paths"]
        if not isinstance(raw_anchor_paths, Mapping) or not raw_anchor_paths:
            raise V02CommandError("server anchor manifest path mapping cannot be empty")
        if not isinstance(raw_protocol_paths, Mapping) or not raw_protocol_paths:
            raise V02CommandError("server training protocol path mapping cannot be empty")
        anchor_paths = {
            _digest(anchor_id, "server anchor mapping key"): _guard_path(
                path, f"anchor manifest for {anchor_id}", must_exist=True
            )
            for anchor_id, path in raw_anchor_paths.items()
        }
        _verify_server_anchor_semantics(
            config,
            anchor_plan=anchors["anchors"],
            anchor_paths=anchor_paths,
        )
        protocols = {
            _digest(protocol_id, "server protocol mapping key"): _load_json(
                path, f"training protocol for {protocol_id}"
            )
            for protocol_id, path in raw_protocol_paths.items()
        }
        project_policy_training_plan, _ = _server_training_bridge()
        formal_protocol_freeze = None
        if config.stage == "v02_freeze_ready":
            from server.repro_fpo_ppo_v02.provenance import (
                load_and_bind_formal_freeze,
            )

            try:
                formal_protocol_freeze = load_and_bind_formal_freeze(config_path)
            except Exception as error:
                raise V02CommandError(
                    f"server formal freeze binding failed: {error}"
                ) from error
        try:
            server_plan, binding = project_policy_training_plan(
                jobs,
                anchor_manifest_paths=anchor_paths,
                training_protocols=protocols,
                formal_protocol_freeze=formal_protocol_freeze,
            )
        except Exception as error:
            raise V02CommandError(f"server training projection failed: {error}") from error
        _publish_many(
            (
                (_guard_path(args.output, "output"), result),
                (_guard_path(args.server_plan_output, "server plan output"), server_plan),
                (
                    _guard_path(args.server_binding_output, "server binding output"),
                    binding,
                ),
            ),
            resume=args.resume,
        )
    else:
        _publish(args.output, result, resume=args.resume)
    return result


def _admit_training_records(args: argparse.Namespace) -> Mapping[str, Any]:
    from .training import AdmittedTrainingRecord, admitted_training_records_digest

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    jobs_payload = _strict_keys(
        _load_json(args.jobs, "training jobs"),
        {
            "schema",
            "config_digest",
            "anchor_plan_digest",
            "trainer_contract_digest",
            "job_count",
            "jobs",
        },
        "training jobs",
    )
    if jobs_payload["schema"] != "policy-learnware.v02-policy-training-job-plan.v0":
        raise V02CommandError("unsupported training job plan schema")
    if jobs_payload["config_digest"] != config.config_digest:
        raise V02CommandError("training job plan is bound to another config")
    jobs = tuple(_policy_job(row) for row in _mapping_rows(jobs_payload["jobs"], "jobs"))
    if jobs_payload["job_count"] != len(jobs):
        raise V02CommandError("training job_count differs from typed jobs")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise V02CommandError("training job plan contains duplicate job IDs")
    by_job = {job.job_id: job for job in jobs}
    admission_provenance: dict[str, Any]
    if args.server_evidence is not None:
        server_evidence_path = _guard_path(
            args.server_evidence, "server training evidence", must_exist=True
        )
        envelope = _strict_keys(
            _load_json(server_evidence_path, "server training evidence"),
            {
                "schema",
                "package_job_plan_digest",
                "server_plan_path",
                "plan_binding_path",
                "evidence_by_job",
            },
            "server training evidence",
        )
        if envelope["schema"] != "policy-learnware.v02-server-training-evidence-set.v0":
            raise V02CommandError("unsupported server training evidence schema")
        if envelope["package_job_plan_digest"] != sha256_json(jobs_payload):
            raise V02CommandError("server evidence is bound to another package job plan")
        server_plan_ref = _evidence_file_ref(
            envelope["server_plan_path"], "server training plan"
        )
        binding_ref = _evidence_file_ref(
            envelope["plan_binding_path"], "server plan binding"
        )
        server_plan = _load_json(server_plan_ref["path"], "server training plan")
        binding = _load_json(binding_ref["path"], "server plan binding")
        raw_evidence = envelope["evidence_by_job"]
        if not isinstance(raw_evidence, Mapping) or set(raw_evidence) != set(by_job):
            raise V02CommandError("server evidence coverage differs from the package job plan")
        evidence_by_job: dict[str, Mapping[str, Any]] = {}
        provenance_by_job: dict[str, Any] = {}
        for job_id in sorted(by_job):
            raw = _strict_keys(
                raw_evidence[job_id],
                {
                    "attempt_manifest_path",
                    "anchor_manifest_path",
                    "run_manifest_path",
                    "training_record_path",
                    "checkpoint_paths",
                },
                f"server evidence for {job_id}",
            )
            checkpoint_paths = raw["checkpoint_paths"]
            if not isinstance(checkpoint_paths, Mapping) or not checkpoint_paths:
                raise V02CommandError(f"checkpoint paths for {job_id} cannot be empty")
            parsed_checkpoints: dict[int, Path] = {}
            checkpoint_refs: dict[str, Any] = {}
            for outer_text, path in checkpoint_paths.items():
                try:
                    outer = int(outer_text)
                except (TypeError, ValueError) as error:
                    raise V02CommandError(
                        f"checkpoint outer iteration for {job_id} must be an integer"
                    ) from error
                if outer <= 0 or str(outer) != str(outer_text):
                    raise V02CommandError(
                        f"checkpoint outer iteration for {job_id} is not canonical"
                    )
                parsed_checkpoints[outer] = _guard_path(
                    path, f"checkpoint {outer} for {job_id}", must_exist=True
                )
                checkpoint_refs[str(outer)] = _evidence_file_ref(
                    parsed_checkpoints[outer], f"checkpoint {outer} for {job_id}"
                )
            attempt_ref = _evidence_file_ref(
                raw["attempt_manifest_path"], f"attempt manifest for {job_id}"
            )
            anchor_ref = _evidence_file_ref(
                raw["anchor_manifest_path"], f"anchor manifest for {job_id}"
            )
            run_ref = _evidence_file_ref(
                raw["run_manifest_path"], f"run manifest for {job_id}"
            )
            record_ref = _evidence_file_ref(
                raw["training_record_path"], f"training record for {job_id}"
            )
            evidence_by_job[job_id] = {
                "attempt_manifest": _load_json(
                    attempt_ref["path"], f"attempt manifest for {job_id}"
                ),
                "anchor_manifest": _load_json(
                    anchor_ref["path"], f"anchor manifest for {job_id}"
                ),
                "run_manifest": _load_json(
                    run_ref["path"], f"run manifest for {job_id}"
                ),
                "training_record": _load_json(
                    record_ref["path"], f"training record for {job_id}"
                ),
                "checkpoint_paths": parsed_checkpoints,
            }
            provenance_by_job[job_id] = {
                "attempt_manifest": attempt_ref,
                "anchor_manifest": anchor_ref,
                "run_manifest": run_ref,
                "training_record": record_ref,
                "checkpoint_paths": dict(sorted(checkpoint_refs.items())),
            }
        _, admit_server_success_batch = _server_training_bridge()
        try:
            admitted = admit_server_success_batch(
                package_jobs=jobs,
                binding_manifest=binding,
                server_plan=server_plan,
                evidence_by_job=evidence_by_job,
            )
        except Exception as error:
            raise V02CommandError(f"server training admission failed: {error}") from error
        admission_provenance = {
            "schema": "policy-learnware.v02-raw-server-admission-provenance.v0",
            "mode": "raw_server_revalidated",
            "package_job_plan_digest": sha256_json(jobs_payload),
            "source_envelope": _evidence_file_ref(
                server_evidence_path, "server training evidence"
            ),
            "server_plan": server_plan_ref,
            "plan_binding": binding_ref,
            "evidence_by_job": dict(sorted(provenance_by_job.items())),
        }
    else:
        if config.stage == "v02_freeze_ready":
            raise V02CommandError(
                "formal admission requires explicit raw server evidence, not attestation JSON"
            )
        attestations_payload = _strict_keys(
            _load_json(args.attestations, "training attestations"),
            {"schema", "job_plan_digest", "records"},
            "training attestations",
        )
        if attestations_payload["schema"] != "policy-learnware.v02-training-attestations.v0":
            raise V02CommandError("unsupported training attestation schema")
        if attestations_payload["job_plan_digest"] != sha256_json(jobs_payload):
            raise V02CommandError("training attestations are bound to another job plan")
        attestations = tuple(
            _policy_attestation(row)
            for row in _mapping_rows(attestations_payload["records"], "attestations")
        )
        if len({item.job_id for item in attestations}) != len(attestations):
            raise V02CommandError("training attestations contain duplicate job IDs")
        if {item.job_id for item in attestations} != set(by_job):
            raise V02CommandError("attestation coverage differs from the frozen job plan")
        admitted = {
            item.job_id: AdmittedTrainingRecord(by_job[item.job_id], item)
            for item in attestations
        }
        admission_provenance = {
            "schema": "policy-learnware.v02-nonformal-attestation-provenance.v0",
            "mode": "attestation_only_nonformal",
            "package_job_plan_digest": sha256_json(jobs_payload),
            "attestation_source": _evidence_file_ref(
                args.attestations, "training attestations"
            ),
        }
    admission_provenance["admitted_records_digest"] = (
        admitted_training_records_digest(admitted)
    )
    result = {
        "schema": "policy-learnware.v02-admitted-training-records.v0",
        "config_digest": jobs_payload["config_digest"],
        "job_plan_digest": sha256_json(jobs_payload),
        "record_count": len(admitted),
        "admission_provenance": admission_provenance,
        "records": {
            job_id: {
                "job": record.job.to_dict(),
                "attestation": record.attestation.to_dict(),
                "record_digest": record.digest,
            }
            for job_id, record in sorted(admitted.items())
        },
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _source_row(raw: Any) -> Any:
    from .competence import SourceEpisodeRow

    data = _strict_keys(
        raw, set(SourceEpisodeRow.__dataclass_fields__), "source episode row"
    )
    return SourceEpisodeRow(**data)


def _evaluate_source_competence(args: argparse.Namespace) -> Mapping[str, Any]:
    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal source competence requires raw evaluator-owned episode shards; "
            "the JSON row adapter is development/audit only"
        )
    payload = _strict_keys(
        _load_json(args.rows, "source episode rows"),
        {"schema", "config_digest", "rows"},
        "source episode rows",
    )
    if payload["schema"] != "policy-learnware.v02-source-episode-rows.v0":
        raise V02CommandError("unsupported source episode row schema")
    if payload["config_digest"] != config.config_digest:
        raise V02CommandError("source episode rows are bound to another config")
    rows = tuple(_source_row(row) for row in _mapping_rows(payload["rows"], "source rows"))
    identities = [
        (row.block, row.source_anchor_id, row.candidate_id, row.reset_seed) for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise V02CommandError("source episode evidence contains duplicate work units")
    selection_seeds = {row.reset_seed for row in rows if row.block == "source_selection"}
    attestation_seeds = {row.reset_seed for row in rows if row.block == "source_attestation"}
    if selection_seeds & attestation_seeds:
        raise V02CommandError("source selection and attestation seed blocks overlap")
    blocks = {name: [row for row in rows if row.block == name] for name in (
        "source_selection", "source_attestation"
    )}
    result = {
        "schema": "policy-learnware.v02-source-episode-evidence-audit.v0",
        "passed": all(blocks.values()),
        "config_digest": payload["config_digest"],
        "row_count": len(rows),
        "row_digest": sha256_json([row.to_dict() for row in rows]),
        "selection_row_count": len(blocks["source_selection"]),
        "attestation_row_count": len(blocks["source_attestation"]),
        "seed_blocks_disjoint": True,
        "source_anchor_ids": sorted({row.source_anchor_id for row in rows}),
        "candidate_ids": sorted({row.candidate_id for row in rows}),
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _championize_anchors(args: argparse.Namespace) -> Mapping[str, Any]:
    from .competence import admit_formal_championization, championize_by_anchor

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal championization is blocked until source evaluator-owned raw "
            "episode receipts and reviewed LCB/tolerance literals are admitted"
        )
    payload = _strict_keys(
        _load_json(args.manifest, "championization inputs"),
        {
            "schema",
            "config_digest",
            "selection_rows",
            "attestation_rows",
            "competence_floors",
            "mean_tolerance",
            "lcb_z",
            "return_contract_id",
        },
        "championization inputs",
    )
    if payload["schema"] != "policy-learnware.v02-championization-inputs.v0":
        raise V02CommandError("unsupported championization input schema")
    if payload["config_digest"] != config.config_digest:
        raise V02CommandError("championization inputs are bound to another config")
    selection = tuple(
        _source_row(row)
        for row in _mapping_rows(payload["selection_rows"], "selection rows")
    )
    attestation = tuple(
        _source_row(row)
        for row in _mapping_rows(payload["attestation_rows"], "attestation rows")
    )
    if config.stage == "v02_freeze_ready":
        if getattr(args, "admitted_records", None) is None:
            raise V02CommandError(
                "formal championization requires --admitted-records"
            )
        admitted_payload, admitted = _admitted_records(
            args.admitted_records, require_server_revalidation=True
        )
        if admitted_payload["config_digest"] != config.config_digest:
            raise V02CommandError(
                "admitted training records are bound to another config"
            )
        expected_floors = dict(config.source_competence_floor_by_anchor)
        if payload["competence_floors"] != expected_floors:
            raise V02CommandError(
                "formal competence floors must be derived from the reviewed config"
            )
        result_typed = admit_formal_championization(
            config,
            admitted,
            selection,
            attestation,
            mean_tolerance=payload["mean_tolerance"],
            lcb_z=payload["lcb_z"],
            return_contract_id=payload["return_contract_id"],
        )
    else:
        if getattr(args, "admitted_records", None) is not None:
            admitted_payload, admitted = _admitted_records(args.admitted_records)
            if admitted_payload["config_digest"] != config.config_digest:
                raise V02CommandError(
                    "admitted training records are bound to another config"
                )
            for row in selection + attestation:
                record = admitted.get(row.candidate_id)
                if record is None or record.job.source_anchor_id != row.source_anchor_id:
                    raise V02CommandError(
                        "source evidence references an unadmitted candidate/anchor"
                    )
                if record.attestation.bundle_digest != row.bundle_digest:
                    raise V02CommandError(
                        "source evidence bundle differs from admitted training"
                    )
        result_typed = championize_by_anchor(
            selection,
            attestation,
            competence_floors=payload["competence_floors"],
            mean_tolerance=payload["mean_tolerance"],
            lcb_z=payload["lcb_z"],
            return_contract_id=payload["return_contract_id"],
        )
    formal_admission = result_typed.formal_admission
    result = {
        "schema": "policy-learnware.v02-championization-result.v0",
        "config_digest": payload["config_digest"],
        "input_digest": sha256_json(payload),
        "mean_tolerance": float(payload["mean_tolerance"]),
        "passed": not result_typed.rejected_anchors,
        "selection_digest": result_typed.selection_digest,
        "selected_by_anchor": dict(result_typed.selected_by_anchor),
        "selection_summaries": [row.to_dict() for row in result_typed.selection_summaries],
        "competence_records": {
            key: value.to_dict()
            for key, value in sorted(result_typed.competence_records.items())
        },
        "rejected_anchors": dict(result_typed.rejected_anchors),
        "attested_bundle_digests": dict(result_typed.attested_bundle_digests),
        "formal_admission": (
            None if formal_admission is None else formal_admission.to_dict()
        ),
        "formal_admission_digest": (
            None if formal_admission is None else formal_admission.digest
        ),
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _admitted_records(
    path: str | Path, *, require_server_revalidation: bool = False
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    from .training import AdmittedTrainingRecord, admitted_training_records_digest

    payload = _strict_keys(
        _load_json(path, "admitted training records"),
        {
            "schema",
            "config_digest",
            "job_plan_digest",
            "record_count",
            "admission_provenance",
            "records",
        },
        "admitted training records",
    )
    if payload["schema"] != "policy-learnware.v02-admitted-training-records.v0":
        raise V02CommandError("unsupported admitted training record schema")
    if not isinstance(payload["records"], Mapping) or not payload["records"]:
        raise V02CommandError("admitted training records cannot be empty")
    parsed: dict[str, Any] = {}
    for candidate_id, raw in payload["records"].items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise V02CommandError("admitted training record key must be non-empty")
        row = _strict_keys(
            raw, {"job", "attestation", "record_digest"}, "admitted training record"
        )
        record = AdmittedTrainingRecord(
            _policy_job(row["job"]), _policy_attestation(row["attestation"])
        )
        if record.job.job_id != candidate_id or row["record_digest"] != record.digest:
            raise V02CommandError("admitted training record identity/digest mismatch")
        parsed[candidate_id] = record
    if payload["record_count"] != len(parsed):
        raise V02CommandError("admitted record_count mismatch")
    records_digest = admitted_training_records_digest(parsed)
    provenance = payload["admission_provenance"]
    if not isinstance(provenance, Mapping):
        raise V02CommandError("admission_provenance must be an object")
    mode = provenance.get("mode")
    if mode == "raw_server_revalidated":
        source = _strict_keys(
            provenance,
            {
                "schema",
                "mode",
                "package_job_plan_digest",
                "source_envelope",
                "server_plan",
                "plan_binding",
                "evidence_by_job",
                "admitted_records_digest",
            },
            "raw server admission provenance",
        )
        if source["schema"] != "policy-learnware.v02-raw-server-admission-provenance.v0":
            raise V02CommandError("unsupported raw server admission provenance schema")
        if (
            source["package_job_plan_digest"] != payload["job_plan_digest"]
            or source["admitted_records_digest"] != records_digest
        ):
            raise V02CommandError("raw server provenance differs from the admitted record set")
        _envelope_path, envelope = _load_evidence_file_ref(
            source["source_envelope"], "server training evidence envelope"
        )
        envelope = _strict_keys(
            envelope,
            {
                "schema",
                "package_job_plan_digest",
                "server_plan_path",
                "plan_binding_path",
                "evidence_by_job",
            },
            "server training evidence envelope",
        )
        if (
            envelope["schema"]
            != "policy-learnware.v02-server-training-evidence-set.v0"
            or envelope["package_job_plan_digest"] != payload["job_plan_digest"]
        ):
            raise V02CommandError("server evidence envelope is misbound")
        _server_plan_path, server_plan = _load_evidence_file_ref(
            source["server_plan"], "server training plan"
        )
        _binding_path, binding = _load_evidence_file_ref(
            source["plan_binding"], "server plan binding"
        )
        raw_jobs = source["evidence_by_job"]
        if not isinstance(raw_jobs, Mapping) or set(raw_jobs) != set(parsed):
            raise V02CommandError("raw server provenance job coverage differs")
        evidence_by_job: dict[str, Mapping[str, Any]] = {}
        for job_id in sorted(parsed):
            raw = _strict_keys(
                raw_jobs[job_id],
                {
                    "attempt_manifest",
                    "anchor_manifest",
                    "run_manifest",
                    "training_record",
                    "checkpoint_paths",
                },
                f"raw server provenance for {job_id}",
            )
            _attempt_path, attempt = _load_evidence_file_ref(
                raw["attempt_manifest"], f"attempt manifest for {job_id}"
            )
            _anchor_path, anchor = _load_evidence_file_ref(
                raw["anchor_manifest"], f"anchor manifest for {job_id}"
            )
            _run_path, run = _load_evidence_file_ref(
                raw["run_manifest"], f"run manifest for {job_id}"
            )
            _record_path, record = _load_evidence_file_ref(
                raw["training_record"], f"training record for {job_id}"
            )
            checkpoints = raw["checkpoint_paths"]
            if not isinstance(checkpoints, Mapping) or not checkpoints:
                raise V02CommandError("raw server checkpoint evidence cannot be empty")
            checkpoint_paths: dict[int, Path] = {}
            for outer_text, checkpoint_ref in checkpoints.items():
                try:
                    outer = int(outer_text)
                except (TypeError, ValueError) as error:
                    raise V02CommandError("checkpoint iteration is not an integer") from error
                if outer <= 0 or str(outer) != str(outer_text):
                    raise V02CommandError("checkpoint iteration is not canonical")
                checkpoint_path = _verify_evidence_file_ref(
                    checkpoint_ref, f"checkpoint {outer} for {job_id}"
                )
                checkpoint_paths[outer] = checkpoint_path
            evidence_by_job[job_id] = {
                "attempt_manifest": attempt,
                "anchor_manifest": anchor,
                "run_manifest": run,
                "training_record": record,
                "checkpoint_paths": checkpoint_paths,
            }
        _, admit_server_success_batch = _server_training_bridge()
        try:
            revalidated = admit_server_success_batch(
                package_jobs=tuple(parsed[job_id].job for job_id in sorted(parsed)),
                binding_manifest=binding,
                server_plan=server_plan,
                evidence_by_job=evidence_by_job,
            )
        except Exception as error:
            raise V02CommandError(
                f"raw server admission revalidation failed: {error}"
            ) from error
        if {
            job_id: record.digest for job_id, record in revalidated.items()
        } != {job_id: record.digest for job_id, record in parsed.items()}:
            raise V02CommandError("revalidated raw server records differ from publication")
        if any(not record.attestation.is_server_bound for record in parsed.values()):
            raise V02CommandError("raw server admission lacks complete server bindings")
    elif mode == "attestation_only_nonformal":
        source = _strict_keys(
            provenance,
            {
                "schema",
                "mode",
                "package_job_plan_digest",
                "attestation_source",
                "admitted_records_digest",
            },
            "nonformal attestation provenance",
        )
        if (
            source["schema"]
            != "policy-learnware.v02-nonformal-attestation-provenance.v0"
            or source["package_job_plan_digest"] != payload["job_plan_digest"]
            or source["admitted_records_digest"] != records_digest
        ):
            raise V02CommandError("nonformal attestation provenance is misbound")
        _load_evidence_file_ref(
            source["attestation_source"], "nonformal training attestations"
        )
    else:
        raise V02CommandError("unsupported admitted training provenance mode")
    if require_server_revalidation and mode != "raw_server_revalidated":
        raise V02CommandError(
            "formal downstream publication requires revalidated raw server evidence"
        )
    return payload, parsed


def _championization_result(path: str | Path) -> tuple[Mapping[str, Any], Any]:
    from .competence import (
        CandidateSelectionSummary,
        ChampionizationResult,
        FormalChampionizationAdmission,
    )
    from .schemas import SourceCompetenceRecord

    payload = _strict_keys(
        _load_json(path, "championization result"),
        {
            "schema",
            "config_digest",
            "input_digest",
            "mean_tolerance",
            "passed",
            "selection_digest",
            "selected_by_anchor",
            "selection_summaries",
            "competence_records",
            "rejected_anchors",
            "attested_bundle_digests",
            "formal_admission",
            "formal_admission_digest",
        },
        "championization result",
    )
    if payload["schema"] != "policy-learnware.v02-championization-result.v0":
        raise V02CommandError("unsupported championization result schema")
    summaries = tuple(
        CandidateSelectionSummary(
            **_strict_keys(
                row,
                set(CandidateSelectionSummary.__dataclass_fields__),
                "candidate selection summary",
            )
        )
        for row in _mapping_rows(payload["selection_summaries"], "selection summaries")
    )
    if not isinstance(payload["competence_records"], Mapping):
        raise V02CommandError("competence_records must be an object")
    competence = {
        str(anchor): SourceCompetenceRecord.from_dict(record)
        for anchor, record in payload["competence_records"].items()
    }
    if not isinstance(payload["selected_by_anchor"], Mapping) or not isinstance(
        payload["rejected_anchors"], Mapping
    ):
        raise V02CommandError("championization identity maps must be objects")
    raw_admission = payload["formal_admission"]
    formal_admission = None
    if raw_admission is not None:
        admission_values = _typed_payload(
            raw_admission,
            schema="policy-learnware.v02-formal-championization-admission.v0",
            fields=set(FormalChampionizationAdmission.__dataclass_fields__),
            where="formal championization admission",
        )
        formal_admission = FormalChampionizationAdmission(
            expected_anchor_ids=tuple(admission_values["expected_anchor_ids"]),
            expected_candidate_ids=tuple(admission_values["expected_candidate_ids"]),
            **{
                key: value
                for key, value in admission_values.items()
                if key not in {"expected_anchor_ids", "expected_candidate_ids"}
            },
        )
        if payload["formal_admission_digest"] != formal_admission.digest:
            raise V02CommandError("formal championization admission digest mismatch")
    elif payload["formal_admission_digest"] is not None:
        raise V02CommandError("formal championization admission digest lacks its record")
    if not isinstance(payload["attested_bundle_digests"], Mapping):
        raise V02CommandError("attested_bundle_digests must be an object")
    typed = ChampionizationResult(
        selected_by_anchor=payload["selected_by_anchor"],
        selection_summaries=summaries,
        competence_records=competence,
        rejected_anchors=payload["rejected_anchors"],
        selection_digest=_digest(payload["selection_digest"], "selection_digest"),
        attested_bundle_digests=payload["attested_bundle_digests"],
        formal_admission=formal_admission,
    )
    expected_selection_digest = sha256_json(
        {
            "schema": "policy-learnware.v02-championization.v0",
            "mean_tolerance": float(payload["mean_tolerance"]),
            "rows": [row.to_dict() for row in summaries],
            "selected": dict(sorted(typed.selected_by_anchor.items())),
        }
    )
    if typed.selection_digest != expected_selection_digest:
        raise V02CommandError("championization selection digest is not independently derived")
    if any(
        record.championization_digest != typed.selection_digest
        for record in typed.competence_records.values()
    ):
        raise V02CommandError("competence record is bound to another championization")
    if payload["passed"] is not (not typed.rejected_anchors):
        raise V02CommandError("championization passed disagrees with rejected anchors")
    return payload, typed


def _build_market(args: argparse.Namespace) -> Mapping[str, Any]:
    from .market import build_policy_market
    from .schemas import ExecutionABIRecord

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal market publication requires evaluator-owned championization "
            "evidence; the file adapter cannot authorize a formal market"
        )
    admitted_payload, admitted = _admitted_records(
        args.admitted_records,
        require_server_revalidation=(config.stage == "v02_freeze_ready"),
    )
    champion_payload, champion = _championization_result(args.championization)
    if admitted_payload["config_digest"] != config.config_digest or champion_payload[
        "config_digest"
    ] != config.config_digest:
        raise V02CommandError("market inputs are bound to another config")
    abi_payload = _strict_keys(
        _load_json(args.execution_abis, "private execution ABIs"),
        {"schema", "config_digest", "entries"},
        "private execution ABIs",
    )
    if abi_payload["schema"] != "policy-learnware.v02-private-execution-abis.v0":
        raise V02CommandError("unsupported private execution ABI schema")
    if abi_payload["config_digest"] != config.config_digest or not isinstance(
        abi_payload["entries"], Mapping
    ):
        raise V02CommandError("private execution ABIs are invalid or misbound")
    execution_abis = {
        str(candidate): ExecutionABIRecord.from_dict(raw)
        for candidate, raw in abi_payload["entries"].items()
    }
    parameters = _strict_keys(
        _load_json(args.parameters, "market parameters"),
        {"schema", "config_digest", "expected_anchor_count", "market_alias_nonce", "tie_break_nonce"},
        "market parameters",
    )
    if parameters["schema"] != "policy-learnware.v02-market-build-parameters.v0":
        raise V02CommandError("unsupported market build parameter schema")
    if parameters["config_digest"] != config.config_digest:
        raise V02CommandError("market parameters are bound to another config")
    configured_anchor_ids = _configured_anchor_ids(config)
    declared_count = parameters["expected_anchor_count"]
    if isinstance(declared_count, bool) or not isinstance(declared_count, int):
        raise V02CommandError("market expected_anchor_count must be an integer")
    if config.stage == "audit_smoke":
        if declared_count <= 0 or declared_count > len(configured_anchor_ids):
            raise V02CommandError("audit-smoke market count exceeds configured anchors")
        market = build_policy_market(
            admitted,
            champion,
            execution_abis,
            expected_anchor_count=declared_count,
            market_alias_nonce=parameters["market_alias_nonce"],
            tie_break_nonce=parameters["tie_break_nonce"],
        )
    else:
        if declared_count != len(configured_anchor_ids):
            raise V02CommandError(
                "market count differs from the config-derived source-anchor set"
            )
        market = build_policy_market(
            admitted,
            champion,
            execution_abis,
            expected_anchor_ids=configured_anchor_ids,
            market_alias_nonce=parameters["market_alias_nonce"],
            tie_break_nonce=parameters["tie_break_nonce"],
        )
    layout = _layout_for_config(config)
    public_output = _guard_path(args.public_output, "public market output")
    private_output = _guard_path(args.private_output, "deployment registry output")
    binding_output = _guard_path(args.binding_output, "anchor binding output")
    _require_path(
        public_output, layout.market_public_dir / "policy_market.json", "public market output"
    )
    _require_path(
        private_output,
        layout.deployment_private_dir / "deployment_registry.json",
        "deployment registry output",
    )
    _require_path(
        binding_output,
        layout.benchmark_private_dir / "anchor_to_opaque_id.json",
        "anchor binding output",
    )
    public_manifest = market.public_manifest()
    deployment_manifest = market.deployment_manifest()
    binding = {
        "schema": "policy-learnware.v02-private-anchor-market-bindings.v0",
        "policy_market_id": market.policy_market_id,
        "anchor_to_opaque_id": dict(sorted(market.anchor_to_opaque_id.items())),
    }
    receipt_output = (
        layout.benchmark_private_dir / "market_publication_receipt.json"
    ).resolve()
    formal_freeze_digest = None
    if config.stage == "v02_freeze_ready":
        try:
            _verified, formal_freeze = load_verified_formal_freeze(config_path)
        except FormalFreezeError as error:
            raise V02CommandError(f"formal market freeze verification failed: {error}") from error
        formal_freeze_digest = formal_freeze.digest
    expected_file_digest = lambda payload: sha256_bytes(  # noqa: E731
        canonical_json_bytes(payload) + b"\n"
    )
    publication_receipt = {
        "schema": "policy-learnware.v02-market-publication-receipt.v0",
        "config_digest": config.config_digest,
        "formal_protocol_freeze_digest": formal_freeze_digest,
        "policy_market_id": market.policy_market_id,
        "entry_count": len(market.entries),
        "public_manifest_file_sha256": expected_file_digest(public_manifest),
        "deployment_manifest_file_sha256": expected_file_digest(deployment_manifest),
        "anchor_binding_file_sha256": expected_file_digest(binding),
        "public_manifest_contains_execution_abi": False,
    }
    digests = _publish_many(
        (
            (public_output, public_manifest),
            (private_output, deployment_manifest),
            (binding_output, binding),
            (receipt_output, publication_receipt),
        ),
        resume=args.resume,
    )
    return {
        **publication_receipt,
        "passed": True,
        "publication_receipt_file_sha256": digests[str(receipt_output)],
    }


_PROBE_WORK_FIELDS = {
    "work_unit_id",
    "context_kind",
    "context_id",
    "environment_instance_digest",
    "probe_protocol_id",
    "seed_digest",
    "prefixes",
}


def _collect_probes(args: argparse.Namespace) -> Mapping[str, Any]:
    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    manifest = _strict_keys(
        _load_json(args.manifest, "probe collection manifest"),
        {"schema", "config_digest", "collector_adapter_id", "collector_adapter_digest", "work_units"},
        "probe collection manifest",
    )
    if manifest["schema"] != "policy-learnware.v02-probe-collection-inputs.v0":
        raise V02CommandError("unsupported probe collection input schema")
    if manifest["config_digest"] != config.config_digest:
        raise V02CommandError("probe collection inputs are bound to another config")
    _digest(manifest["collector_adapter_digest"], "collector_adapter_digest")
    if not isinstance(manifest["collector_adapter_id"], str) or not manifest[
        "collector_adapter_id"
    ]:
        raise V02CommandError("collector_adapter_id must be non-empty")
    rows = _mapping_rows(manifest["work_units"], "probe work units")
    normalized: list[Mapping[str, Any]] = []
    identities: set[str] = set()
    for index, raw in enumerate(rows):
        row = _strict_keys(raw, _PROBE_WORK_FIELDS, f"probe work_units[{index}]")
        if row["context_kind"] not in {"source", "development"}:
            raise V02CommandError("v0.2 probe collection permits only source/development contexts")
        if row["probe_protocol_id"] != config.probe_protocol_id:
            raise V02CommandError("probe work unit uses another probe protocol")
        _digest(row["environment_instance_digest"], "environment_instance_digest")
        _digest(row["seed_digest"], "seed_digest")
        prefixes = tuple(row["prefixes"]) if isinstance(row["prefixes"], list) else ()
        if not prefixes or any(type(value) is not int for value in prefixes):
            raise V02CommandError("probe prefixes must be explicit integers")
        if not set(prefixes).issubset(config.encoder_eval_prefixes):
            raise V02CommandError("probe work unit requests an unregistered prefix")
        identity = row["work_unit_id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise V02CommandError("probe work_unit_id must be non-empty and unique")
        identities.add(identity)
        normalized.append(dict(row))
    result = {
        "schema": "policy-learnware.v02-frozen-probe-work-plan.v0",
        "config_digest": config.config_digest,
        "input_digest": sha256_json(manifest),
        "collector_adapter_id": manifest["collector_adapter_id"],
        "collector_adapter_digest": manifest["collector_adapter_digest"],
        "work_unit_count": len(normalized),
        "work_units": normalized,
        "execution_state": "REQUIRES_REGISTERED_COLLECTOR_ADAPTER",
        "collection_completed": False,
        "sealed_contexts_authorized": False,
        "blocked_requirements": [
            "execute collector_adapter_id in the registered JAX/environment runtime",
            "publish immutable trace shards and rerun digest admission",
        ],
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _representation_index(raw: Any) -> Any:
    from .environment_spec import RepresentationIndex, RepresentationIndexEntry
    from .schemas import EnvironmentSpec

    payload = _strict_keys(
        raw,
        {"schema", "representation_index_id", "policy_market_id", "representation_protocol_id", "entries"},
        "representation index",
    )
    if payload["schema"] != "policy-learnware.v02-representation-index.v0" or not isinstance(
        payload["entries"], Mapping
    ):
        raise V02CommandError("unsupported or invalid representation index")
    entries = {}
    for opaque_id, raw_entry in payload["entries"].items():
        entry = _strict_keys(
            raw_entry, {"opaque_id", "environment_spec"}, "representation index entry"
        )
        spec = EnvironmentSpec.from_dict(entry["environment_spec"])
        entries[str(opaque_id)] = RepresentationIndexEntry(entry["opaque_id"], spec)
    return RepresentationIndex(
        policy_market_id=payload["policy_market_id"],
        representation_protocol_id=payload["representation_protocol_id"],
        entries=entries,
        representation_index_id=payload["representation_index_id"],
    )


def _build_environment_specs(args: argparse.Namespace) -> Mapping[str, Any]:
    from .environment_spec import RepresentationIndex, RepresentationIndexEntry
    from .schemas import EnvironmentSpec

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal EnvironmentSpecs must be rebuilt from admitted raw probe shards; "
            "the precomputed-spec adapter is development/audit only"
        )
    manifest = _strict_keys(
        _load_json(args.manifest, "environment spec inputs"),
        {"schema", "config_digest", "policy_market_id", "representation_protocol_id", "entries"},
        "environment spec inputs",
    )
    if manifest["schema"] != "policy-learnware.v02-environment-spec-inputs.v0":
        raise V02CommandError("unsupported EnvironmentSpec input schema")
    if manifest["config_digest"] != config.config_digest or not isinstance(
        manifest["entries"], Mapping
    ) or not manifest["entries"]:
        raise V02CommandError("EnvironmentSpec inputs are empty or misbound")
    entries = {
        str(opaque_id): RepresentationIndexEntry(
            str(opaque_id), EnvironmentSpec.from_dict(raw_spec)
        )
        for opaque_id, raw_spec in manifest["entries"].items()
    }
    index = RepresentationIndex(
        policy_market_id=manifest["policy_market_id"],
        representation_protocol_id=manifest["representation_protocol_id"],
        entries=entries,
    )
    result = index.to_dict()
    _publish(args.output, result, resume=args.resume)
    return result


def _sigma_artifact(raw: Any) -> Any:
    from .baselines import SourceOnlySigmaArtifact

    data = _strict_keys(
        raw,
        {
            "schema",
            "policy_market_id",
            "representation_index_id",
            "partition_id",
            "source_ids",
            "source_spec_digests",
            "distance_form",
            "sigma",
            "derivation",
            "zero_distance_fallback",
            "artifact_digest",
        },
        "source-only sigma artifact",
    )
    if (
        data["schema"] != "policy-learnware.v02-source-only-sigma.v0"
        or data["derivation"] != "median(nonzero_source_pair_distances)"
        or data["zero_distance_fallback"] is not None
    ):
        raise V02CommandError("invalid source-only sigma artifact contract")
    return SourceOnlySigmaArtifact(
        policy_market_id=data["policy_market_id"],
        representation_index_id=data["representation_index_id"],
        partition_id=data["partition_id"],
        source_ids=tuple(data["source_ids"]),
        source_spec_digests=tuple(data["source_spec_digests"]),
        distance_form=data["distance_form"],
        sigma=data["sigma"],
        artifact_digest=data["artifact_digest"],
    )


def _frozen_selector_artifact(raw: Any) -> Any:
    from .baselines import FrozenSelectorArtifact
    from .selectors import EvidenceContract

    fields = {
        "method_id",
        "evidence_contract",
        "fit_capability",
        "training_data_digest",
        "payload",
        "development_freeze_ref",
        "artifact_digest",
    }
    data = _typed_payload(
        raw,
        schema="policy-learnware.v02-frozen-selector-artifact.v0",
        fields=fields,
        where="frozen selector artifact",
    )
    return FrozenSelectorArtifact(
        method_id=data["method_id"],
        evidence_contract=EvidenceContract.from_dict(data["evidence_contract"]),
        fit_capability=data["fit_capability"],
        training_data_digest=data["training_data_digest"],
        payload=data["payload"],
        development_freeze_ref=data["development_freeze_ref"],
        artifact_digest=data["artifact_digest"],
    )


def _fit_baselines(args: argparse.Namespace) -> Mapping[str, Any]:
    from .baselines import (
        CompetenceOnlySelector,
        RandomAnonymousMarketSelector,
        SourceOnlyLMinSelector,
    )
    from .selectors import EvidenceContract

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    plan = _strict_keys(
        _load_json(args.manifest, "baseline fit manifest"),
        {"schema", "config_digest", "policy_market_id", "methods"},
        "baseline fit manifest",
    )
    if plan["schema"] != "policy-learnware.v02-baseline-fit-plan.v0":
        raise V02CommandError("unsupported baseline fit plan schema")
    if plan["config_digest"] != config.config_digest:
        raise V02CommandError("baseline fit plan is bound to another config")
    policy_market_id = _digest(plan["policy_market_id"], "policy_market_id")
    _require_registered_method_contract(config)
    methods = _mapping_rows(plan["methods"], "baseline methods")
    artifacts: dict[str, Any] = {}
    pending: dict[str, Any] = {}
    for index, raw in enumerate(methods):
        row = _strict_keys(
            raw, {"method_id", "kind", "parameters"}, f"baseline methods[{index}]"
        )
        method_id = row["method_id"]
        if not isinstance(method_id, str) or not method_id or method_id in artifacts or method_id in pending:
            raise V02CommandError("baseline method IDs must be non-empty and unique")
        if method_id not in config.method_ids:
            raise V02CommandError(f"baseline method {method_id!r} is not frozen in config")
        kind = row["kind"]
        expected_kind = METHOD_KIND_REGISTRY[method_id]
        if kind != expected_kind and kind != "registered_adapter":
            raise V02CommandError(
                f"baseline {method_id!r} must use role {expected_kind!r}, not {kind!r}"
            )
        if kind == "registered_adapter" and expected_kind in {
            "random",
            "competence",
            "lmin",
        }:
            raise V02CommandError(
                f"built-in method {method_id!r} cannot be replaced by an adapter"
            )
        if kind == "random":
            parameters = _strict_keys(row["parameters"], {"selector_seed"}, "random parameters")
            selector = RandomAnonymousMarketSelector(
                method_id=method_id,
                selector_seed=parameters["selector_seed"],
                policy_market_id=policy_market_id,
            )
            artifact = selector.fit(None)
        elif kind == "competence":
            _strict_keys(row["parameters"], set(), "competence parameters")
            selector = CompetenceOnlySelector(
                method_id=method_id, policy_market_id=policy_market_id
            )
            artifact = selector.fit(None)
        elif kind == "lmin":
            parameters = _strict_keys(
                row["parameters"], {"sigma_artifact", "epsilon", "evidence_contract"}, "L-min parameters"
            )
            selector = SourceOnlyLMinSelector(
                method_id=method_id,
                sigma_artifact=_sigma_artifact(parameters["sigma_artifact"]),
                epsilon=parameters["epsilon"],
                evidence_contract=EvidenceContract.from_dict(parameters["evidence_contract"]),
            )
            artifact = selector.fit(None)
        elif kind == "registered_adapter":
            parameters = _strict_keys(
                row["parameters"],
                {
                    "adapter_id",
                    "adapter_digest",
                    "input_digest",
                    "evidence_contract",
                    "method_role",
                    "conformance_receipt",
                },
                "registered adapter parameters",
            )
            _digest(parameters["adapter_digest"], "adapter_digest")
            _digest(parameters["input_digest"], "input_digest")
            if parameters["method_role"] != expected_kind:
                raise V02CommandError(
                    f"registered adapter for {method_id!r} declares another method role"
                )
            evidence = EvidenceContract.from_dict(parameters["evidence_contract"])
            evidence.require_public_selector_safe()
            receipt = _strict_keys(
                parameters["conformance_receipt"],
                {
                    "schema",
                    "method_id",
                    "method_role",
                    "adapter_digest",
                    "fixture_digest",
                    "full_anonymous_market",
                    "unique_tie_break_token",
                    "evidence_contract_digest",
                    "passed",
                },
                "registered adapter conformance receipt",
            )
            receipt_passed = bool(
                receipt["schema"]
                == "policy-learnware.v02-baseline-adapter-conformance.v0"
                and receipt["method_id"] == method_id
                and receipt["method_role"] == expected_kind
                and receipt["adapter_digest"] == parameters["adapter_digest"]
                and receipt["full_anonymous_market"] is True
                and receipt["unique_tie_break_token"] is True
                and receipt["evidence_contract_digest"]
                == sha256_json(evidence.to_dict())
            )
            _digest(receipt["fixture_digest"], "adapter conformance fixture_digest")
            if receipt["passed"] is not receipt_passed or not receipt_passed:
                raise V02CommandError(
                    f"registered adapter for {method_id!r} lacks a passing typed conformance receipt"
                )
            pending[method_id] = {
                "kind": kind,
                "method_role": expected_kind,
                "execution_state": "REQUIRES_REGISTERED_BASELINE_ADAPTER",
                "adapter_id": parameters["adapter_id"],
                "adapter_digest": parameters["adapter_digest"],
                "input_digest": parameters["input_digest"],
                "evidence_contract_digest": sha256_json(evidence.to_dict()),
            }
            continue
        else:
            raise V02CommandError(f"unsupported baseline kind: {kind!r}")
        artifacts[method_id] = {"kind": kind, "artifact": artifact.to_dict()}
    observed_methods = set(artifacts) | set(pending)
    if observed_methods != set(config.method_ids):
        raise V02CommandError(
            "baseline fit plan must cover every frozen method exactly once"
        )
    result = {
        "schema": "policy-learnware.v02-baseline-fit-result.v0",
        "config_digest": config.config_digest,
        "plan_digest": sha256_json(plan),
        "execution_state": "FITTED" if not pending else "FROZEN_WORK_PLAN",
        "all_methods_fitted": not pending,
        "artifacts": dict(sorted(artifacts.items())),
        "pending_work": dict(sorted(pending.items())),
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _public_market(path: str | Path) -> tuple[str, Mapping[str, Any]]:
    from .schemas import PublicMarketEntry

    payload = _strict_keys(
        _load_json(path, "public policy market"),
        {"schema", "policy_market_id", "entries"},
        "public policy market",
    )
    if payload["schema"] != "policy-learnware.v02-public-policy-market.v0" or not isinstance(
        payload["entries"], Mapping
    ):
        raise V02CommandError("unsupported or invalid public policy market")
    entries = {
        str(opaque_id): PublicMarketEntry.from_dict(raw)
        for opaque_id, raw in payload["entries"].items()
    }
    return payload["policy_market_id"], entries


def _verify_formal_market_receipt(
    *, config: Any, config_path: Path, public_market_path: Path
) -> Mapping[str, Any]:
    """Rebind the public market bytes to the exact config/protocol freeze."""

    layout = _layout_for_config(config)
    receipt_path = (
        layout.benchmark_private_dir / "market_publication_receipt.json"
    ).resolve()
    receipt = _strict_keys(
        _load_json(receipt_path, "market publication receipt"),
        {
            "schema",
            "config_digest",
            "formal_protocol_freeze_digest",
            "policy_market_id",
            "entry_count",
            "public_manifest_file_sha256",
            "deployment_manifest_file_sha256",
            "anchor_binding_file_sha256",
            "public_manifest_contains_execution_abi",
        },
        "market publication receipt",
    )
    if receipt["schema"] != "policy-learnware.v02-market-publication-receipt.v0":
        raise V02CommandError("unsupported market publication receipt schema")
    try:
        _verified, formal_freeze = load_verified_formal_freeze(config_path)
    except FormalFreezeError as error:
        raise V02CommandError(f"formal market freeze verification failed: {error}") from error
    checks = {
        "config_digest": receipt["config_digest"] == config.config_digest,
        "formal_protocol_freeze_digest": (
            receipt["formal_protocol_freeze_digest"] == formal_freeze.digest
        ),
        "public_manifest_file_sha256": (
            receipt["public_manifest_file_sha256"] == sha256_file(public_market_path)
        ),
        "public_manifest_contains_execution_abi": (
            receipt["public_manifest_contains_execution_abi"] is False
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise V02CommandError(
            f"formal public market differs from its canonical publication: {failed}"
        )
    for name in ("deployment_manifest_file_sha256", "anchor_binding_file_sha256"):
        _digest(receipt[name], f"market receipt {name}")
    return receipt


def _selection_record(raw: Any) -> Any:
    from .selectors import EvidenceContract, RankingRow, SelectionRecord

    data = _typed_payload(
        raw,
        schema="policy-learnware.v02-selection-record.v0",
        fields={
            "method_id",
            "status",
            "selected_id",
            "ranking",
            "target_evidence_digest",
            "selector_artifact_digest",
            "cost_digest",
            "evidence_contract",
        },
        where="selection record",
    )
    ranking = tuple(
        RankingRow(
            **_strict_keys(
                row, set(RankingRow.__dataclass_fields__), "selection ranking row"
            )
        )
        for row in _mapping_rows(data["ranking"], "selection ranking")
    )
    return SelectionRecord(
        method_id=data["method_id"],
        status=data["status"],
        selected_id=data["selected_id"],
        ranking=ranking,
        target_evidence_digest=data["target_evidence_digest"],
        selector_artifact_digest=data["selector_artifact_digest"],
        cost_digest=data["cost_digest"],
        evidence_contract=EvidenceContract.from_dict(data["evidence_contract"]),
    )


def _run_selectors(args: argparse.Namespace) -> Mapping[str, Any]:
    from .baselines import (
        CompetenceOnlySelector,
        RandomAnonymousMarketSelector,
        SourceOnlyLMinSelector,
        TargetQueryView,
    )
    from .baselines import SourceOnlySigmaArtifact
    from .schemas import EnvironmentSpec
    from .selectors import PublicMarketView

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal selector execution requires raw-probe-derived index/query receipts; "
            "the direct JSON adapter is development/audit only"
        )
    _require_registered_method_contract(config)
    market_path = _guard_path(args.market, "public market", must_exist=True)
    if config.stage == "v02_freeze_ready":
        _require_path(
            market_path,
            _layout_for_config(config).market_public_dir / "policy_market.json",
            "formal selector market",
        )
    policy_market_id, entries = _public_market(market_path)
    if config.stage == "v02_freeze_ready":
        receipt = _verify_formal_market_receipt(
            config=config, config_path=config_path, public_market_path=market_path
        )
        if (
            receipt["policy_market_id"] != policy_market_id
            or receipt["entry_count"] != len(entries)
        ):
            raise V02CommandError(
                "formal public market ID/count differs from its publication receipt"
            )
    if config.stage != "audit_smoke" and len(entries) != len(_configured_anchor_ids(config)):
        raise V02CommandError(
            "selector market entry count differs from the frozen source-anchor market"
        )
    index = _representation_index(_load_json(args.representation_index, "representation index"))
    market = PublicMarketView(policy_market_id, entries, index)
    artifacts_payload = _strict_keys(
        _load_json(args.baseline_artifacts, "baseline artifacts"),
        {"schema", "config_digest", "plan_digest", "execution_state", "all_methods_fitted", "artifacts", "pending_work"},
        "baseline artifacts",
    )
    if artifacts_payload["schema"] != "policy-learnware.v02-baseline-fit-result.v0":
        raise V02CommandError("unsupported baseline artifact schema")
    if artifacts_payload["config_digest"] != config.config_digest:
        raise V02CommandError("baseline artifacts are bound to another config")
    if artifacts_payload["all_methods_fitted"] is not True or artifacts_payload["pending_work"]:
        raise V02CommandError("selector execution cannot consume an incomplete baseline work plan")
    if not isinstance(artifacts_payload["artifacts"], Mapping):
        raise V02CommandError("baseline artifacts must be an object")
    if set(artifacts_payload["artifacts"]) != set(config.method_ids):
        raise V02CommandError(
            "selector artifacts must exactly cover every frozen method"
        )
    selectors: dict[str, tuple[Any, Any]] = {}
    for method_id, raw in artifacts_payload["artifacts"].items():
        row = _strict_keys(raw, {"kind", "artifact"}, "baseline artifact entry")
        artifact = _frozen_selector_artifact(row["artifact"])
        if artifact.method_id != method_id:
            raise V02CommandError("baseline artifact mapping key differs from method ID")
        if row["kind"] != METHOD_KIND_REGISTRY[method_id]:
            raise V02CommandError(
                f"selector artifact for {method_id!r} is labeled with another implementation role"
            )
        if row["kind"] == "random":
            selector = RandomAnonymousMarketSelector(
                method_id=method_id,
                selector_seed=artifact.payload["selector_seed"],
                policy_market_id=artifact.payload["selector_binding"][
                    "policy_market_id"
                ],
            )
        elif row["kind"] == "competence":
            selector = CompetenceOnlySelector(
                method_id=method_id,
                policy_market_id=artifact.payload["selector_binding"][
                    "policy_market_id"
                ],
            )
        elif row["kind"] == "lmin":
            source_ids = tuple(artifact.payload["source_ids"])
            sigma = SourceOnlySigmaArtifact(
                policy_market_id=artifact.payload["policy_market_id"],
                representation_index_id=artifact.payload["representation_index_id"],
                partition_id=artifact.payload["partition_id"],
                source_ids=source_ids,
                source_spec_digests=tuple(
                    str(index.entries[opaque_id].environment_spec.environment_spec_digest)
                    for opaque_id in source_ids
                ),
                distance_form=artifact.payload["distance_form"],
                sigma=artifact.payload["sigma"],
                artifact_digest=artifact.payload["source_only_sigma_artifact_digest"],
            )
            selector = SourceOnlyLMinSelector(
                method_id=method_id,
                sigma_artifact=sigma,
                epsilon=artifact.payload["epsilon"],
                evidence_contract=artifact.evidence_contract,
            )
        else:
            raise V02CommandError("selector execution requires a built-in executable artifact")
        selectors[method_id] = (selector, artifact)
    queries_payload = _strict_keys(
        _load_json(args.queries, "selector queries"),
        {"schema", "config_digest", "queries"},
        "selector queries",
    )
    if queries_payload["schema"] != "policy-learnware.v02-development-selector-queries.v0":
        raise V02CommandError("v0.2 CLI accepts only development selector queries")
    if queries_payload["config_digest"] != config.config_digest:
        raise V02CommandError("selector queries are bound to another config")
    outputs: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for raw in _mapping_rows(queries_payload["queries"], "selector queries"):
        row = _strict_keys(
            raw,
            {"query_id", "query_spec", "target_evidence_digest", "cost_digest", "probe_rewards_included"},
            "selector query",
        )
        query_id = row["query_id"]
        if not isinstance(query_id, str) or not query_id or query_id in query_ids:
            raise V02CommandError("selector query IDs must be non-empty and unique")
        query_ids.add(query_id)
        query = TargetQueryView(
            stage="development_discovery",
            query_spec=EnvironmentSpec.from_dict(row["query_spec"]),
            target_evidence_digest=row["target_evidence_digest"],
            cost_digest=row["cost_digest"],
            probe_rewards_included=row["probe_rewards_included"],
        )
        for method_id, (selector, artifact) in sorted(selectors.items()):
            selection = selector.select(query, market, artifact)
            outputs.append({"query_id": query_id, "selection": selection.to_dict()})
    if config.stage != "audit_smoke":
        expected_queries = {target.target_id for target in config.development_targets}
        if query_ids != expected_queries:
            raise V02CommandError(
                "selector queries must exactly cover the frozen development contexts"
            )
    result = {
        "schema": "policy-learnware.v02-development-selector-results.v0",
        "config_digest": config.config_digest,
        "policy_market_id": policy_market_id,
        "representation_index_id": index.representation_index_id,
        "query_count": len(query_ids),
        "method_count": len(selectors),
        "results": outputs,
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _compute_metrics(args: argparse.Namespace) -> Mapping[str, Any]:
    from .metrics import compute_ranking_metrics, compute_selection_metrics

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal metrics must use recompute_development_oracle with raw typed "
            "OracleEpisodeRows; uploaded normalized value maps are forbidden"
        )
    selections = _strict_keys(
        _load_json(args.selections, "selector results"),
        {
            "schema",
            "config_digest",
            "policy_market_id",
            "representation_index_id",
            "query_count",
            "method_count",
            "results",
        },
        "selector results",
    )
    if selections["schema"] != "policy-learnware.v02-development-selector-results.v0":
        raise V02CommandError("metrics accept only v0.2 development selector results")
    if selections["config_digest"] != config.config_digest:
        raise V02CommandError("selector results are bound to another config")
    parsed_selections: dict[tuple[str, str], Any] = {}
    for raw in _mapping_rows(selections["results"], "selector results"):
        row = _strict_keys(raw, {"query_id", "selection"}, "selector result")
        selection = _selection_record(row["selection"])
        key = (row["query_id"], selection.method_id)
        if key in parsed_selections:
            raise V02CommandError("duplicate selector query/method result")
        parsed_selections[key] = selection
    values = _strict_keys(
        _load_json(args.values, "development value inputs"),
        {"schema", "config_digest", "units"},
        "development value inputs",
    )
    if values["schema"] != "policy-learnware.v02-development-metric-inputs.v0":
        raise V02CommandError("metrics accept only development value inputs")
    if values["config_digest"] != selections["config_digest"]:
        raise V02CommandError("metric inputs and selections use different configs")
    units = _mapping_rows(values["units"], "metric units")
    results: list[dict[str, Any]] = []
    observed: set[tuple[str, str]] = set()
    for raw in units:
        row = _strict_keys(
            raw,
            {
                "query_id",
                "method_id",
                "normalized_returns_by_policy",
                "executable_policy_ids",
                "incompatible_failure_value",
                "epsilon",
                "tie_tolerance",
            },
            "metric unit",
        )
        key = (row["query_id"], row["method_id"])
        if key in observed or key not in parsed_selections:
            raise V02CommandError("metric unit coverage has a duplicate or unexpected key")
        observed.add(key)
        selection = parsed_selections[key]
        selection_metrics = compute_selection_metrics(
            selected_policy_id=selection.selected_id,
            normalized_returns_by_policy=row["normalized_returns_by_policy"],
            executable_policy_ids=row["executable_policy_ids"],
            incompatible_failure_value=row["incompatible_failure_value"],
            epsilon=row["epsilon"],
            tie_tolerance=row["tie_tolerance"],
        )
        executable = set(row["executable_policy_ids"])
        predicted = tuple(
            item.opaque_learnware_id
            for item in selection.ranking
            if item.opaque_learnware_id in executable
        )
        ranking_metrics = compute_ranking_metrics(
            predicted,
            row["normalized_returns_by_policy"],
            tie_tolerance=row["tie_tolerance"],
        )
        results.append(
            {
                "query_id": row["query_id"],
                "method_id": row["method_id"],
                "selection_record_digest": selection.digest,
                "selection_metrics": selection_metrics.to_dict(),
                "ranking_metrics": ranking_metrics.to_dict(),
            }
        )
    if observed != set(parsed_selections):
        raise V02CommandError("metric units do not cover every selector result")
    result = {
        "schema": "policy-learnware.v02-development-metrics.v0",
        "config_digest": selections["config_digest"],
        "unit_count": len(results),
        "results": results,
    }
    _publish(args.output, result, resume=args.resume)
    return result


def _evaluate_gates(args: argparse.Namespace) -> Mapping[str, Any]:
    from .gates import evaluate_formal_gate_state_from_file

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    layout = _layout_for_config(config)
    manifest_path = _guard_path(
        args.evidence_manifest, "gate evidence manifest", must_exist=True
    )
    _require_path(
        manifest_path,
        layout.gate_artifact("v02_gate_evidence_manifest.json"),
        "gate evidence manifest",
    )
    try:
        state = evaluate_formal_gate_state_from_file(
            manifest_path,
            experiment_root=layout.experiment_root,
            expected_experiment_id=config.experiment_id,
            expected_config_digest=config.config_digest,
        )
    except Exception as error:
        raise V02CommandError(f"formal gate evidence failed: {error}") from error
    result = state.to_dict()
    output = _guard_path(args.output, "gate state output")
    _require_path(
        output,
        layout.gate_artifact("v02_gate_state.json"),
        "gate state output",
    )
    _publish(output, result, resume=args.resume)
    return result


def _validate_public_surface_payload(raw: Any, *, config: Any) -> bool:
    payload = _strict_keys(
        raw,
        {"schema", "passed", "policy_market_id", "market", "artifacts"},
        "public surface audit",
    )
    if payload["schema"] != "policy-learnware.v02-public-surface-audit.v0":
        raise V02CommandError("unsupported public surface audit schema")
    market = _strict_keys(
        payload["market"],
        {"schema", "passed", "entry_count", "allowed_entry_fields", "violations"},
        "public market audit",
    )
    artifacts = _strict_keys(
        payload["artifacts"],
        {"schema", "passed", "root", "file_count", "tree_digest", "violations"},
        "public artifact audit",
    )
    market_passed = bool(
        market["schema"] == "policy-learnware.v02-public-market-allowlist-audit.v0"
        and isinstance(market["entry_count"], int)
        and not isinstance(market["entry_count"], bool)
        and market["entry_count"] > 0
        and market["allowed_entry_fields"] == sorted(PUBLIC_MARKET_ENTRY_FIELDS)
        and market["violations"] == []
    )
    artifact_root: Path | None = None
    observed_tree_digest: str | None = None
    try:
        artifact_root = _guard_path(
            artifacts["root"], "audited public artifact root", must_exist=True
        )
        if not artifact_root.is_dir():
            raise V02CommandError("audited public artifact root must be a directory")
        observed_tree_digest = artifact_tree_digest(artifact_root)
    except (TypeError, ValueError, V02CommandError):
        artifact_root = None
    artifact_passed = bool(
        artifacts["schema"] == "policy-learnware.v02-public-artifact-audit.v0"
        and isinstance(artifacts["file_count"], int)
        and not isinstance(artifacts["file_count"], bool)
        and artifacts["file_count"] > 0
        and isinstance(artifacts["tree_digest"], str)
        and observed_tree_digest == artifacts["tree_digest"]
        and artifacts["violations"] == []
    )
    if isinstance(artifacts["tree_digest"], str):
        _digest(artifacts["tree_digest"], "public artifact tree digest")
    if market["passed"] is not market_passed or artifacts["passed"] is not artifact_passed:
        raise V02CommandError("public audit pass bits disagree with primitive evidence")
    if config.stage == "v02_freeze_ready" and market["entry_count"] != len(
        _configured_anchor_ids(config)
    ):
        raise V02CommandError(
            "formal public audit does not cover the complete source-anchor market"
        )
    derived = market_passed and artifact_passed
    if payload["passed"] is not derived:
        raise V02CommandError("public surface pass bit disagrees with component audits")
    return derived


def _validate_oracle_independence_payload(raw: Any) -> bool:
    payload = _strict_keys(
        raw,
        {
            "schema",
            "passed",
            "baseline_selection_digest",
            "replay_callback_capabilities",
            "oracle_root_passed_to_replay",
            "scenarios",
            "violations",
        },
        "oracle independence audit",
    )
    if payload["schema"] != "policy-learnware.v02-oracle-independence-audit.v0":
        raise V02CommandError("unsupported oracle independence audit schema")
    baseline = _digest(payload["baseline_selection_digest"], "baseline_selection_digest")
    if payload["replay_callback_capabilities"] != [
        "market_public_root",
        "measurement_root",
        "selector_outputs_root",
    ] or payload["oracle_root_passed_to_replay"] is not False:
        raise V02CommandError("oracle independence callback received a forbidden capability")
    scenarios = _strict_keys(payload["scenarios"], {"missing", "poison"}, "oracle scenarios")
    derived = payload["violations"] == []
    for label in ("missing", "poison"):
        scenario = _strict_keys(
            scenarios[label],
            {
                "passed",
                "returned_selection_digest",
                "before_tree_digests",
                "after_tree_digests",
                "oracle_state_unchanged",
            },
            f"oracle scenario {label}",
        )
        returned = _digest(
            scenario["returned_selection_digest"],
            f"oracle scenario {label} returned selection digest",
        )
        if not isinstance(scenario["before_tree_digests"], Mapping) or not isinstance(
            scenario["after_tree_digests"], Mapping
        ):
            raise V02CommandError("oracle scenario tree digests must be objects")
        for tree in (scenario["before_tree_digests"], scenario["after_tree_digests"]):
            if set(tree) != {"market_public", "measurement", "selector_outputs"}:
                raise V02CommandError("oracle scenario tree coverage is incomplete")
            for name, digest in tree.items():
                _digest(digest, f"oracle scenario {label} {name} tree digest")
        scenario_passed = bool(
            returned == baseline
            and scenario["before_tree_digests"]["selector_outputs"] == baseline
            and scenario["before_tree_digests"] == scenario["after_tree_digests"]
            and scenario["oracle_state_unchanged"] is True
        )
        if scenario["passed"] is not scenario_passed:
            raise V02CommandError("oracle scenario pass bit disagrees with primitive evidence")
        derived &= scenario_passed
    if payload["passed"] is not derived:
        raise V02CommandError("oracle independence pass bit disagrees with scenarios")
    return derived


def _audit_information(args: argparse.Namespace) -> Mapping[str, Any]:
    from .audit import audit_evidence_contract
    from .selectors import EvidenceContract

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal information isolation requires live oracle poison/missing replay; "
            "the uploaded audit adapter is nonformal only"
        )
    payload = _strict_keys(
        _load_json(args.manifest, "information audit inputs"),
        {
            "schema",
            "config_digest",
            "public_surface_audit",
            "evidence_contracts",
            "oracle_independence_audit",
        },
        "information audit inputs",
    )
    if payload["schema"] != "policy-learnware.v02-information-audit-inputs.v0":
        raise V02CommandError("unsupported information audit input schema")
    if payload["config_digest"] != config.config_digest:
        raise V02CommandError("information audit inputs are bound to another config")
    public_passed = _validate_public_surface_payload(
        payload["public_surface_audit"], config=config
    )
    if not isinstance(payload["evidence_contracts"], Mapping) or not payload[
        "evidence_contracts"
    ]:
        raise V02CommandError("information audit requires evidence contracts")
    _require_registered_method_contract(config)
    if set(payload["evidence_contracts"]) != set(config.method_ids):
        raise V02CommandError(
            "information audit evidence contracts must exactly cover frozen methods"
        )
    evidence_results = {}
    for method_id, raw in sorted(payload["evidence_contracts"].items()):
        contract = EvidenceContract.from_dict(raw)
        audit = audit_evidence_contract(contract)
        evidence_results[str(method_id)] = audit.to_dict()
    evidence_passed = all(value["passed"] for value in evidence_results.values())
    oracle_passed = _validate_oracle_independence_payload(
        payload["oracle_independence_audit"]
    )
    result = {
        "schema": "policy-learnware.v02-information-isolation-audit.v0",
        "passed": public_passed and evidence_passed and oracle_passed,
        "config_digest": config.config_digest,
        "public_surface_passed": public_passed,
        "evidence_contracts_passed": evidence_passed,
        "oracle_independence_passed": oracle_passed,
        "oracle_evidence_consumed_by_selector": False,
        "evidence_contract_audits": evidence_results,
    }
    output = _guard_path(args.output, "information audit output")
    layout = _layout_for_config(config)
    _require_path(
        output,
        layout.analysis_dir / "information_isolation.json",
        "information audit output",
    )
    _publish(output, result, resume=args.resume)
    return result


def _development_p_table(raw: Any) -> Any:
    from .report import DevelopmentPTableRow, build_development_p_table

    payload = _strict_keys(
        raw,
        {
            "schema",
            "visibility",
            "stage",
            "development_split_digest",
            "policy_market_id",
            "evaluation_protocol_id",
            "rows",
        },
        "development P-table",
    )
    if (
        payload["schema"] != "policy-learnware.v02-development-p-table.v0"
        or payload["visibility"] != "development-analysis-only"
        or payload["stage"] != "development_discovery"
    ):
        raise V02CommandError("invalid development P-table boundary")
    rows = tuple(
        DevelopmentPTableRow(
            **_strict_keys(
                row, set(DevelopmentPTableRow.__dataclass_fields__), "development P-table row"
            )
        )
        for row in _mapping_rows(payload["rows"], "development P-table rows")
    )
    table = build_development_p_table(
        rows,
        development_split_digest=payload["development_split_digest"],
        policy_market_id=payload["policy_market_id"],
        evaluation_protocol_id=payload["evaluation_protocol_id"],
    )
    if sha256_json(payload) != table.digest:
        raise V02CommandError("development P-table differs from typed reconstruction")
    return table


def _private_o_table(raw: Any) -> Any:
    from .report import PrivateOTableRow, build_private_o_table

    payload = _strict_keys(
        raw,
        {"schema", "visibility", "policy_market_id", "evaluation_protocol_id", "rows"},
        "private O-table",
    )
    if (
        payload["schema"] != "policy-learnware.v02-private-o-table.v0"
        or payload["visibility"] != "private-oracle-analysis-only"
    ):
        raise V02CommandError("invalid private O-table boundary")
    parsed = []
    row_fields = {
        "opaque_query_id",
        "true_task_id",
        "true_axis_id",
        "true_factor",
        "regime",
        "physical_nearest_anchor_id",
        "true_distance_lmin",
        "source_global_champion",
        "executable_ids",
        "incompatible_ids",
        "full_candidate_value_vector",
        "best_in_pool_ids",
        "best_in_pool_value",
        "pool_viability_q_star",
        "failure_floor",
        "selected_method_regret_decomposition",
        "episode_rows_digest",
        "execution_abi_census_digest",
        "oracle_result_digest",
    }
    for raw_row in _mapping_rows(payload["rows"], "private O-table rows"):
        row = _strict_keys(raw_row, row_fields, "private O-table row")
        nearest = _strict_keys(
            row["true_distance_lmin"], {"selected_id", "selected_value", "regret"}, "true-distance L-min"
        )
        champion = _strict_keys(
            row["source_global_champion"], {"selected_id", "selected_value", "regret_g0"}, "source/global champion"
        )
        parsed.append(
            PrivateOTableRow(
                opaque_query_id=row["opaque_query_id"],
                true_task_id=row["true_task_id"],
                true_axis_id=row["true_axis_id"],
                true_factor=row["true_factor"],
                regime=row["regime"],
                physical_nearest_anchor_id=row["physical_nearest_anchor_id"],
                true_distance_lmin_selected_id=nearest["selected_id"],
                true_distance_lmin_value=nearest["selected_value"],
                true_distance_lmin_regret=nearest["regret"],
                source_global_champion_id=champion["selected_id"],
                source_global_champion_value=champion["selected_value"],
                source_global_champion_regret=champion["regret_g0"],
                executable_ids=tuple(row["executable_ids"]),
                incompatible_ids=tuple(row["incompatible_ids"]),
                full_candidate_value_vector=row["full_candidate_value_vector"],
                best_in_pool_ids=tuple(row["best_in_pool_ids"]),
                best_in_pool_value=row["best_in_pool_value"],
                pool_viability=row["pool_viability_q_star"],
                failure_floor=row["failure_floor"],
                selected_method_regret_decomposition=row["selected_method_regret_decomposition"],
                episode_rows_digest=row["episode_rows_digest"],
                execution_abi_census_digest=row["execution_abi_census_digest"],
                oracle_result_digest=row["oracle_result_digest"],
            )
        )
    table = build_private_o_table(
        parsed,
        policy_market_id=payload["policy_market_id"],
        evaluation_protocol_id=payload["evaluation_protocol_id"],
    )
    if sha256_json(payload) != table.digest:
        raise V02CommandError("private O-table differs from typed reconstruction")
    return table


def _reference_e_table(raw: Any) -> Any:
    from .report import ReferenceETableRow, build_reference_e_table

    payload = _strict_keys(
        raw,
        {"schema", "visibility", "evaluation_protocol_id", "included_reference_kinds", "rows"},
        "reference E-table",
    )
    if (
        payload["schema"] != "policy-learnware.v02-reference-e-table.v0"
        or payload["visibility"] != "representation-reference-analysis"
    ):
        raise V02CommandError("invalid reference E-table boundary")
    rows = []
    row_fields = {
        "reference_kind",
        "representation_id",
        "representation_version",
        "training_split_digest",
        "canonical_event_view_digest",
        "probe_protocol_id",
        "component_digests",
        "heldout_metrics",
        "repeated_bank",
        "prefix_curve",
        "sample_efficiency_auc",
        "fixed_lmin",
        "encoding_latency_seconds",
    }
    for raw_row in _mapping_rows(payload["rows"], "reference E-table rows"):
        row = _strict_keys(raw_row, row_fields, "reference E-table row")
        components = _strict_keys(
            row["component_digests"], {"checkpoint", "normalizer", "latent_contract", "kernel", "reducer"}, "E-table components"
        )
        heldout = _strict_keys(row["heldout_metrics"], {"neighborhood", "order"}, "E-table heldout metrics")
        repeated = _strict_keys(row["repeated_bank"], {"stability", "signal_to_noise_ratio"}, "E-table bank metrics")
        fixed = _strict_keys(row["fixed_lmin"], {"selected_return", "regret"}, "E-table fixed L-min")
        latency = _strict_keys(row["encoding_latency_seconds"], {"cold", "warm"}, "E-table latency")
        curve = _mapping_rows(row["prefix_curve"], "E-table prefix curve")
        for point in curve:
            _strict_keys(point, {"prefix", "selected_return", "regret"}, "E-table prefix point")
        rows.append(
            ReferenceETableRow(
                reference_kind=row["reference_kind"],
                representation_id=row["representation_id"],
                representation_version=row["representation_version"],
                training_split_digest=row["training_split_digest"],
                canonical_event_view_digest=row["canonical_event_view_digest"],
                probe_protocol_id=row["probe_protocol_id"],
                checkpoint_digest=components["checkpoint"],
                normalizer_digest=components["normalizer"],
                latent_contract_digest=components["latent_contract"],
                kernel_digest=components["kernel"],
                reducer_digest=components["reducer"],
                heldout_neighborhood_score=heldout["neighborhood"],
                heldout_order_score=heldout["order"],
                repeated_bank_stability=repeated["stability"],
                signal_to_noise_ratio=repeated["signal_to_noise_ratio"],
                prefix_budgets=tuple(point["prefix"] for point in curve),
                prefix_selected_returns=tuple(point["selected_return"] for point in curve),
                prefix_regrets=tuple(point["regret"] for point in curve),
                sample_efficiency_auc=row["sample_efficiency_auc"],
                fixed_lmin_selected_return=fixed["selected_return"],
                fixed_lmin_regret=fixed["regret"],
                cold_encoding_seconds=latency["cold"],
                warm_encoding_seconds=latency["warm"],
            )
        )
    table = build_reference_e_table(rows, evaluation_protocol_id=payload["evaluation_protocol_id"])
    if sorted({row.reference_kind for row in rows}) != payload["included_reference_kinds"]:
        raise V02CommandError("reference E-table kind index differs from rows")
    if sha256_json(payload) != table.digest:
        raise V02CommandError("reference E-table differs from typed reconstruction")
    return table


def _build_report(args: argparse.Namespace) -> Mapping[str, Any]:
    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    layout = _layout_for_config(config)
    p_path = _guard_path(args.p_table, "development P-table", must_exist=True)
    o_path = _guard_path(args.o_table, "private O-table", must_exist=True)
    e_path = _guard_path(args.e_table, "reference E-table", must_exist=True)
    _require_path(p_path, layout.analysis_dir / "development_p_table.json", "development P-table")
    _require_path(o_path, layout.benchmark_private_dir / "development_o_table.json", "private O-table")
    _require_path(e_path, layout.analysis_dir / "reference_e_table.json", "reference E-table")
    p_table = _development_p_table(_load_json(p_path, "development P-table"))
    o_table = _private_o_table(_load_json(o_path, "private O-table"))
    e_table = _reference_e_table(_load_json(e_path, "reference E-table"))
    if len({p_table.evaluation_protocol_id, o_table.evaluation_protocol_id, e_table.evaluation_protocol_id}) != 1:
        raise V02CommandError("P/O/reference-E tables use different evaluation protocols")
    if p_table.policy_market_id != o_table.policy_market_id:
        raise V02CommandError("P/O tables use different policy markets")
    result = {
        "schema": "policy-learnware.v02-development-report-index.v0",
        "config_digest": config.config_digest,
        "policy_market_id": p_table.policy_market_id,
        "evaluation_protocol_id": p_table.evaluation_protocol_id,
        "development_p_table_digest": p_table.digest,
        "private_o_table_digest": o_table.digest,
        "reference_e_table_digest": e_table.digest,
        "private_o_table_embedded_in_public_report": False,
        "paper1_joint_tables_present": False,
    }
    output = _guard_path(args.output, "report index output")
    _require_path(output, layout.analysis_dir / "v02_report_index.json", "report index output")
    _publish(output, result, resume=args.resume)
    return result


_OPERATOR_AUDIT_FIELDS = {
    "schema",
    "axis_id",
    "operator_id",
    "operator_version",
    "task_id",
    "factor",
    "base_model_digest",
    "shifted_model_digest",
    "changed_leaves",
    "unchanged_leaves",
    "selected_element_count",
    "changed_element_count",
    "source_object_unchanged",
    "exact_allowlist",
    "coupling_check",
    "finite",
    "passed",
    "reason",
}
_AXIS_RECORD_FIELDS = {
    "task_id",
    "axis_id",
    "factor_id",
    "factor_value",
    "operator_audit",
    "operator_audit_digest",
    "runtime_checks",
}
_RUNTIME_AXIS_CHECKS = {
    "fresh_instance_isolation",
    "reset_finite",
    "step_finite",
    "jit_reset",
    "jit_step",
}


def _audit_axis_record(raw: Any, expected: Any, entry: Any) -> tuple[str, ...]:
    violations: list[str] = []
    try:
        record = _strict_keys(raw, _AXIS_RECORD_FIELDS, "axis audit record")
        operator = _strict_keys(
            record["operator_audit"], _OPERATOR_AUDIT_FIELDS, "operator_audit"
        )
        runtime = _strict_keys(
            record["runtime_checks"], _RUNTIME_AXIS_CHECKS, "runtime_checks"
        )
    except V02CommandError as error:
        return (str(error),)
    identity = (
        record["task_id"] == entry.task_id
        and record["axis_id"] == entry.axis_id
        and record["factor_id"] == expected.factor_id
        and isinstance(record["factor_value"], (int, float))
        and not isinstance(record["factor_value"], bool)
        and float(record["factor_value"]) == expected.value
        and operator["task_id"] == entry.task_id
        and operator["axis_id"] == entry.axis_id
        and operator["operator_id"] == entry.operator_id
        and operator["operator_version"] == entry.operator_version
        and isinstance(operator["factor"], (int, float))
        and not isinstance(operator["factor"], bool)
        and float(operator["factor"]) == expected.value
    )
    if not identity:
        violations.append("task_axis_factor_or_operator_identity_mismatch")
    if operator["schema"] != "policy-learnware.v02-dynamics-operator-audit.v0":
        violations.append("unsupported_operator_audit_schema")
    for name in ("base_model_digest", "shifted_model_digest"):
        try:
            _digest(operator[name], f"operator_audit.{name}")
        except V02CommandError:
            violations.append(f"invalid_{name}")
    changed = operator["changed_leaves"]
    unchanged = operator["unchanged_leaves"]
    if (
        not isinstance(changed, list)
        or not all(isinstance(item, str) for item in changed)
        or len(changed) != len(set(changed))
        or not isinstance(unchanged, list)
        or not all(isinstance(item, str) for item in unchanged)
        or len(unchanged) != len(set(unchanged))
        or set(changed) & set(unchanged)
    ):
        violations.append("invalid_changed_unchanged_leaf_partition")
        changed = []
    expected_changed = (
        [] if expected.value == 1.0 else sorted(item.leaf for item in entry.selections)
    )
    if sorted(changed) != expected_changed:
        violations.append("changed_leaves_differ_from_reviewed_allowlist")
    counts_valid = (
        isinstance(operator["selected_element_count"], int)
        and not isinstance(operator["selected_element_count"], bool)
        and operator["selected_element_count"] > 0
        and isinstance(operator["changed_element_count"], int)
        and not isinstance(operator["changed_element_count"], bool)
        and operator["changed_element_count"]
        == (0 if expected.value == 1.0 else operator["selected_element_count"])
    )
    if not counts_valid:
        violations.append("changed_element_count_mismatch")
    digest_identity = (
        operator["base_model_digest"] == operator["shifted_model_digest"]
        if expected.value == 1.0
        else operator["base_model_digest"] != operator["shifted_model_digest"]
    )
    if not digest_identity:
        violations.append("nominal_identity_or_shifted_model_diff_failed")
    primitive_names = (
        "source_object_unchanged",
        "exact_allowlist",
        "coupling_check",
        "finite",
    )
    if any(type(operator[name]) is not bool or operator[name] is not True for name in primitive_names):
        violations.append("operator_primitive_check_failed")
    computed_passed = not violations and operator["reason"] is None
    if type(operator["passed"]) is not bool or operator["passed"] != computed_passed:
        violations.append("uploaded_passed_disagrees_with_recomputed_primitives")
    if any(type(runtime[name]) is not bool or runtime[name] is not True for name in runtime):
        violations.append("runtime_finite_jit_or_isolation_check_failed")
    try:
        declared_digest = _digest(
            record["operator_audit_digest"], "operator_audit_digest"
        )
    except V02CommandError:
        declared_digest = ""
        violations.append("invalid_operator_audit_digest")
    if declared_digest != sha256_json(operator):
        violations.append("operator_audit_digest_mismatch")
    return tuple(violations)


def _audit_axes(args: argparse.Namespace) -> Mapping[str, Any]:
    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    candidate_catalog = _candidate_catalog_for_config(config)
    registry = axis_registry_from_config(config, candidate_catalog)
    if config.stage == "v02_freeze_ready":
        registry.validate_formal_scope(V02_TASKS)
    manifest = _strict_keys(
        _load_json(args.audit_manifest, "axis audit manifest"),
        {"schema", "config_digest", "axis_registry_digest", "records"},
        "axis audit manifest",
    )
    if manifest["schema"] != "policy-learnware.v02-axis-audit-manifest.v0":
        raise V02CommandError("unsupported axis audit manifest schema")
    if manifest["config_digest"] != config.config_digest:
        raise V02CommandError("axis audit manifest is bound to another config")
    if manifest["axis_registry_digest"] != registry.digest:
        raise V02CommandError("axis audit manifest is bound to another registry")
    records = manifest["records"]
    if not isinstance(records, list):
        raise V02CommandError("axis audit records must be a list")
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    duplicate_keys: list[str] = []
    for raw in records:
        if isinstance(raw, Mapping):
            key = (str(raw.get("task_id")), str(raw.get("axis_id")), str(raw.get("factor_id")))
            if key in by_key:
                duplicate_keys.append("/".join(key))
            else:
                by_key[key] = raw
    expected: dict[tuple[str, str, str], tuple[Any, Any]] = {}
    for entry in registry.entries.values():
        for factor in entry.factors:
            expected[(entry.task_id, entry.axis_id, factor.factor_id)] = (factor, entry)
    missing = sorted(set(expected) - set(by_key))
    extra = sorted(set(by_key) - set(expected))
    violations: list[dict[str, Any]] = []
    for key, (factor, entry) in sorted(expected.items()):
        if key not in by_key:
            continue
        for reason in _audit_axis_record(by_key[key], factor, entry):
            violations.append({"work_unit": "/".join(key), "reason": reason})
    violations.extend(
        {"work_unit": "/".join(key), "reason": "missing_audit_record"}
        for key in missing
    )
    violations.extend(
        {"work_unit": "/".join(key), "reason": "unexpected_audit_record"}
        for key in extra
    )
    violations.extend(
        {"work_unit": key, "reason": "duplicate_audit_record"}
        for key in duplicate_keys
    )
    result = {
        "schema": "policy-learnware.v02-axis-audit-validation.v0",
        "passed": not violations,
        "config_digest": config.config_digest,
        "axis_registry_digest": registry.digest,
        "expected_work_units": len(expected),
        "validated_work_units": len(expected) - len(missing),
        "violations": violations,
    }
    _publish(args.output, result, resume=args.resume)
    return result


_RULE_FIELDS = {
    "pattern",
    "kind",
    "json_keys",
    "npz_members",
    "permitted_forbidden_keys",
    "permitted_forbidden_string_tokens",
    "permitted_forbidden_npz_members",
}


def _string_set(value: Any, where: str) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise V02CommandError(f"{where} must be a list of strings")
    if len(value) != len(set(value)):
        raise V02CommandError(f"{where} contains duplicates")
    return frozenset(value)


def _parse_public_rules(path: str | Path) -> tuple[PublicArtifactRule, ...]:
    payload = _strict_keys(
        _load_json(path, "public artifact rules"),
        {"schema", "rules"},
        "public artifact rules",
    )
    if payload["schema"] != "policy-learnware.v02-public-artifact-rules.v0":
        raise V02CommandError("unsupported public artifact rules schema")
    if not isinstance(payload["rules"], list) or not payload["rules"]:
        raise V02CommandError("public artifact rules cannot be empty")
    rules: list[PublicArtifactRule] = []
    for index, raw in enumerate(payload["rules"]):
        value = _strict_keys(raw, _RULE_FIELDS, f"rules[{index}]")
        exception_fields = (
            "permitted_forbidden_keys",
            "permitted_forbidden_string_tokens",
            "permitted_forbidden_npz_members",
        )
        if any(value[name] not in ([], ()) for name in exception_fields):
            raise V02CommandError(
                "public audit forbidden-token exceptions are source-owned; "
                "the v0.2 CLI does not accept caller-defined exemptions"
            )
        rules.append(
            PublicArtifactRule(
                pattern=value["pattern"],
                kind=value["kind"],
                json_keys=_string_set(value["json_keys"], f"rules[{index}].json_keys"),
                npz_members=_string_set(value["npz_members"], f"rules[{index}].npz_members"),
                permitted_forbidden_keys=frozenset(),
                permitted_forbidden_string_tokens=frozenset(),
                permitted_forbidden_npz_members=frozenset(),
            )
        )
    return tuple(rules)


def _audit_public(args: argparse.Namespace) -> Mapping[str, Any]:
    public_root = _guard_path(args.public_root, "public root", must_exist=True)
    if not public_root.is_dir():
        raise V02CommandError("public root must be a directory")
    market_path = _guard_path(args.market_manifest, "market manifest", must_exist=True)
    if not _is_within(market_path, public_root):
        raise V02CommandError("market manifest must be inside the audited public root")
    output = _guard_path(args.output, "output")
    if _is_within(output, public_root):
        raise V02CommandError("audit output cannot mutate the public tree being audited")
    manifest = _strict_keys(
        _load_json(market_path, "market manifest"),
        {"schema", "policy_market_id", "entries"},
        "market manifest",
    )
    if manifest["schema"] != "policy-learnware.v02-public-policy-market.v0":
        raise V02CommandError("unsupported public market manifest schema")
    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, Mapping):
        raise V02CommandError("public market entries must be an object")
    entries: dict[str, Mapping[str, Any]] = {}
    for opaque_id, raw in raw_entries.items():
        if isinstance(raw, Mapping) and set(raw) == PUBLIC_MARKET_ENTRY_FIELDS | {"schema"}:
            if raw["schema"] != "policy-learnware.v02-public-market-entry.v0":
                entries[str(opaque_id)] = raw
            else:
                entries[str(opaque_id)] = {
                    key: raw[key] for key in PUBLIC_MARKET_ENTRY_FIELDS
                }
        else:
            entries[str(opaque_id)] = raw
    market_audit = audit_public_market_entries(entries)
    artifact_audit = audit_public_artifacts(public_root, _parse_public_rules(args.rules))
    result = {
        "schema": "policy-learnware.v02-public-surface-audit.v0",
        "passed": market_audit.passed and artifact_audit.passed,
        "policy_market_id": manifest["policy_market_id"],
        "market": market_audit.to_dict(),
        "artifacts": artifact_audit.to_dict(),
    }
    _publish(output, result, resume=args.resume)
    return result


_RECOMPUTE_SPEC_FIELDS = {
    "source_files",
    "recomputed_file",
    "expected_output_digest",
}


def _recompute(args: argparse.Namespace) -> Mapping[str, Any]:
    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    if config.stage == "v02_freeze_ready":
        raise V02CommandError(
            "formal completion cannot use the checksum-only recompute adapter; "
            "construct IndependentRecomputeInputs from raw evidence and run "
            "run_independent_recompute so selectors, oracle, statistics, costs, "
            "and isolation are actually replayed"
        )
    raw_root = _guard_path(args.raw_root, "raw root", must_exist=True)
    recompute_root = _guard_path(args.recompute_root, "recompute root", must_exist=True)
    if not raw_root.is_dir() or not recompute_root.is_dir():
        raise V02CommandError("raw and recompute roots must be directories")
    if _is_within(raw_root, recompute_root) or _is_within(recompute_root, raw_root):
        raise V02CommandError("raw and independent recompute roots must be disjoint")
    output = _guard_path(args.output, "output")
    if config.stage == "v02_freeze_ready":
        _require_path(
            output,
            _layout_for_config(config).recompute_audit,
            "formal recompute audit output",
        )
    if _is_within(output, raw_root) or _is_within(output, recompute_root):
        raise V02CommandError("recompute audit output must be outside both audited roots")
    manifest = _strict_keys(
        _load_json(args.manifest, "recompute manifest"),
        {
            "schema",
            "config_digest",
            "coverage_contract_digest",
            "raw_tree_digest",
            "sections",
        },
        "recompute manifest",
    )
    if manifest["schema"] != "policy-learnware.v02-independent-recompute-plan.v0":
        raise V02CommandError("unsupported independent recompute plan schema")
    if manifest["config_digest"] != config.config_digest:
        raise V02CommandError("recompute plan is bound to another config")
    coverage_contract_digest = _digest(
        manifest["coverage_contract_digest"], "coverage_contract_digest"
    )
    declared_tree = _digest(manifest["raw_tree_digest"], "raw_tree_digest")
    observed_tree = artifact_tree_digest(raw_root)
    raw_tree_matches = declared_tree == observed_tree
    sections = _strict_keys(
        manifest["sections"], set(RECOMPUTE_SECTIONS), "recompute sections"
    )
    section_passed: dict[str, bool] = {}
    section_digests: dict[str, str] = {}
    errors: list[str] = []
    output_paths: set[Path] = set()
    for name in RECOMPUTE_SECTIONS:
        spec = _strict_keys(sections[name], _RECOMPUTE_SPEC_FIELDS, f"sections.{name}")
        source_files = spec["source_files"]
        if not isinstance(source_files, Mapping) or not source_files:
            raise V02CommandError(f"sections.{name}.source_files must be non-empty")
        source_matches = True
        for relative, expected_digest in source_files.items():
            source = _resolve_relative_file(
                raw_root, relative, f"sections.{name}.source_files[{relative!r}]"
            )
            source_matches &= sha256_file(source) == _digest(
                expected_digest, f"sections.{name}.source_files[{relative!r}]"
            )
        recomputed = _resolve_relative_file(
            recompute_root, spec["recomputed_file"], f"sections.{name}.recomputed_file"
        )
        if recomputed in output_paths:
            raise V02CommandError("recompute checks must publish distinct output files")
        output_paths.add(recomputed)
        expected_output = _digest(
            spec["expected_output_digest"], f"sections.{name}.expected_output_digest"
        )
        observed_output = sha256_file(recomputed)
        passed = raw_tree_matches and source_matches and observed_output == expected_output
        section_passed[name] = passed
        section_digests[name] = observed_output
        if not raw_tree_matches:
            errors.append(f"{name}: raw tree digest mismatch")
        if not source_matches:
            errors.append(f"{name}: one or more primitive source digests mismatch")
        if observed_output != expected_output:
            errors.append(f"{name}: independent recompute output digest mismatch")
    # Construct through the same strict type consumed by v0.2 completion.  The
    # report has no caller-supplied pass bit: it is derived from the six
    # primitive checks and the absence of errors.
    from .recompute import IndependentRecomputeReport

    primitive_checks = {
        "full_digest_coverage": all(section_passed.values()),
        "full_selector_replay": section_passed["selectors"],
        "full_statistical_recompute": (
            section_passed["oracle"] and section_passed["statistics"]
        ),
        "raw_numeric_subset_coverage": (
            section_passed["source"]
            and section_passed["gate0"]
            and section_passed["representations"]
        ),
        "cost_recompute": section_passed["costs"],
        "information_isolation": section_passed["information"],
    }
    report = IndependentRecomputeReport(
        coverage_contract_digest=coverage_contract_digest,
        checks=primitive_checks,
        section_digests=section_digests,
        errors=tuple(errors),
    )
    result = report.to_dict()
    _publish(output, result, resume=args.resume)
    return result


def _complete(args: argparse.Namespace) -> Mapping[str, Any]:
    """Delegate completion semantics to the dedicated strict contract module."""

    config_path = _guard_path(args.config, "config", must_exist=True)
    config = _load_config_for_command(config_path)
    layout = _layout_for_config(config)
    gate_state_path = _guard_path(args.gate_state, "gate state", must_exist=True)
    recompute_path = _guard_path(
        args.recompute_audit, "recompute audit", must_exist=True
    )
    output_path = _guard_path(args.output, "output")
    _require_path(
        gate_state_path,
        layout.gate_artifact("v02_gate_state.json"),
        "gate state",
    )
    _require_path(
        recompute_path,
        layout.recompute_audit,
        "recompute audit",
    )
    _require_path(output_path, layout.completion_manifest, "completion output")
    # Delayed import keeps basic validation/audit commands usable while the
    # completion contract evolves, and prevents this CLI from duplicating its
    # scientific state machine.
    from .completion import complete_v02_from_files

    result = complete_v02_from_files(
        config_path=config_path,
        gate_state_path=gate_state_path,
        recompute_audit_path=recompute_path,
        output_path=output_path,
        theory_status=args.theory_status,
        literature_novelty_audit_status=args.literature_novelty_audit_status,
        resume=args.resume,
    )
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "to_dict") and callable(result.to_dict):
        return result.to_dict()
    return {
        "schema": "policy-learnware.v02-complete-command-result.v0",
        "passed": True,
        "completion_manifest": str(result),
    }


def _output_flags(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--output", type=Path, required=required)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-learnware-v02",
        description="v0.2 development/freeze-ready orchestration (no Paper-I sealed access)",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--draft", action="store_true")
    validate.add_argument(
        "--expect-stage",
        choices=("audit_smoke", "development_discovery", "v02_freeze_ready"),
    )
    _output_flags(validate, required=False)
    validate.set_defaults(handler=_validate_config)

    environment_abi = subparsers.add_parser("audit-environment-abi")
    environment_abi.add_argument("--manifest", type=Path, required=True)
    _output_flags(environment_abi, required=True)
    environment_abi.set_defaults(handler=_audit_environment_abi)

    freeze = subparsers.add_parser("freeze-run")
    freeze.add_argument("--config", type=Path, required=True)
    _output_flags(freeze, required=True)
    freeze.set_defaults(handler=_freeze_run)

    training = subparsers.add_parser("plan-training")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--anchors", type=Path, required=True)
    training.add_argument("--trainer-contract", type=Path, required=True)
    training.add_argument("--server-projection-inputs", type=Path)
    training.add_argument("--server-plan-output", type=Path)
    training.add_argument("--server-binding-output", type=Path)
    _output_flags(training, required=True)
    training.set_defaults(handler=_plan_training)

    admission = subparsers.add_parser("admit-training-records")
    admission.add_argument("--config", type=Path, required=True)
    admission.add_argument("--jobs", type=Path, required=True)
    evidence_group = admission.add_mutually_exclusive_group(required=True)
    evidence_group.add_argument("--attestations", type=Path)
    evidence_group.add_argument("--server-evidence", type=Path)
    _output_flags(admission, required=True)
    admission.set_defaults(handler=_admit_training_records)

    source = subparsers.add_parser("evaluate-source-competence")
    source.add_argument("--config", type=Path, required=True)
    source.add_argument("--rows", type=Path, required=True)
    _output_flags(source, required=True)
    source.set_defaults(handler=_evaluate_source_competence)

    champion = subparsers.add_parser("championize-anchors")
    champion.add_argument("--config", type=Path, required=True)
    champion.add_argument("--manifest", type=Path, required=True)
    champion.add_argument("--admitted-records", type=Path)
    _output_flags(champion, required=True)
    champion.set_defaults(handler=_championize_anchors)

    market = subparsers.add_parser("build-market")
    market.add_argument("--config", type=Path, required=True)
    market.add_argument("--admitted-records", type=Path, required=True)
    market.add_argument("--championization", type=Path, required=True)
    market.add_argument("--execution-abis", type=Path, required=True)
    market.add_argument("--parameters", type=Path, required=True)
    market.add_argument("--public-output", type=Path, required=True)
    market.add_argument("--private-output", type=Path, required=True)
    market.add_argument("--binding-output", type=Path, required=True)
    market.add_argument("--resume", action="store_true")
    market.set_defaults(handler=_build_market)

    probes = subparsers.add_parser("collect-probes")
    probes.add_argument("--config", type=Path, required=True)
    probes.add_argument("--manifest", type=Path, required=True)
    _output_flags(probes, required=True)
    probes.set_defaults(handler=_collect_probes)

    specs = subparsers.add_parser("build-environment-specs")
    specs.add_argument("--config", type=Path, required=True)
    specs.add_argument("--manifest", type=Path, required=True)
    _output_flags(specs, required=True)
    specs.set_defaults(handler=_build_environment_specs)

    baselines = subparsers.add_parser("fit-baselines")
    baselines.add_argument("--config", type=Path, required=True)
    baselines.add_argument("--manifest", type=Path, required=True)
    _output_flags(baselines, required=True)
    baselines.set_defaults(handler=_fit_baselines)

    selectors = subparsers.add_parser("run-selectors")
    selectors.add_argument("--config", type=Path, required=True)
    selectors.add_argument("--market", type=Path, required=True)
    selectors.add_argument("--representation-index", type=Path, required=True)
    selectors.add_argument("--baseline-artifacts", type=Path, required=True)
    selectors.add_argument("--queries", type=Path, required=True)
    _output_flags(selectors, required=True)
    selectors.set_defaults(handler=_run_selectors)

    metrics = subparsers.add_parser("compute-metrics")
    metrics.add_argument("--config", type=Path, required=True)
    metrics.add_argument("--selections", type=Path, required=True)
    metrics.add_argument("--values", type=Path, required=True)
    _output_flags(metrics, required=True)
    metrics.set_defaults(handler=_compute_metrics)

    gates = subparsers.add_parser("evaluate-gates")
    gates.add_argument("--config", type=Path, required=True)
    gates.add_argument(
        "--evidence-manifest",
        "--checks",
        dest="evidence_manifest",
        type=Path,
        required=True,
        help=(
            "canonical typed gate evidence manifest; --checks is retained only "
            "as a spelling alias and no longer accepts naked booleans"
        ),
    )
    _output_flags(gates, required=True)
    gates.set_defaults(handler=_evaluate_gates)

    information = subparsers.add_parser("audit-information")
    information.add_argument("--config", type=Path, required=True)
    information.add_argument("--manifest", type=Path, required=True)
    _output_flags(information, required=True)
    information.set_defaults(handler=_audit_information)

    report = subparsers.add_parser("build-report")
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--p-table", type=Path, required=True)
    report.add_argument("--o-table", type=Path, required=True)
    report.add_argument("--e-table", type=Path, required=True)
    _output_flags(report, required=True)
    report.set_defaults(handler=_build_report)

    axes = subparsers.add_parser("audit-axes")
    axes.add_argument("--config", type=Path, required=True)
    axes.add_argument("--audit-manifest", type=Path, required=True)
    _output_flags(axes, required=True)
    axes.set_defaults(handler=_audit_axes)

    public = subparsers.add_parser("audit-public")
    public.add_argument("--public-root", type=Path, required=True)
    public.add_argument("--market-manifest", type=Path, required=True)
    public.add_argument("--rules", type=Path, required=True)
    _output_flags(public, required=True)
    public.set_defaults(handler=_audit_public)

    recompute = subparsers.add_parser("recompute", aliases=["audit-recompute"])
    recompute.add_argument("--config", type=Path, required=True)
    recompute.add_argument("--manifest", type=Path, required=True)
    recompute.add_argument("--raw-root", type=Path, required=True)
    recompute.add_argument("--recompute-root", type=Path, required=True)
    _output_flags(recompute, required=True)
    recompute.set_defaults(handler=_recompute)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--config", type=Path, required=True)
    complete.add_argument("--gate-state", type=Path, required=True)
    complete.add_argument("--recompute-audit", type=Path, required=True)
    complete.add_argument(
        "--theory-status",
        choices=("PENDING", "MINIMAL_FINITE_POOL_CLOSED"),
        required=True,
    )
    complete.add_argument("--literature-novelty-audit-status", required=True)
    _output_flags(complete, required=True)
    complete.set_defaults(handler=_complete)
    return parser


HANDLERS: Mapping[str, Callable[[argparse.Namespace], Mapping[str, Any]]] = {
    "validate-config": _validate_config,
    "audit-environment-abi": _audit_environment_abi,
    "freeze-run": _freeze_run,
    "plan-training": _plan_training,
    "admit-training-records": _admit_training_records,
    "evaluate-source-competence": _evaluate_source_competence,
    "championize-anchors": _championize_anchors,
    "build-market": _build_market,
    "collect-probes": _collect_probes,
    "build-environment-specs": _build_environment_specs,
    "fit-baselines": _fit_baselines,
    "run-selectors": _run_selectors,
    "compute-metrics": _compute_metrics,
    "evaluate-gates": _evaluate_gates,
    "audit-information": _audit_information,
    "build-report": _build_report,
    "audit-axes": _audit_axes,
    "audit-public": _audit_public,
    "recompute": _recompute,
    "audit-recompute": _recompute,
    "complete": _complete,
}


def _command_completed(payload: Mapping[str, Any]) -> bool:
    """Derive process success without treating an omitted ``passed`` as success.

    Most publication commands are complete once their immutable artifact was
    written and therefore need no scientific pass bit.  Work-plan commands are
    different: a pending registered adapter/collector is an actionable block
    and must return a non-zero process status even though recording the plan
    itself succeeded.
    """

    if "passed" in payload:
        if type(payload["passed"]) is not bool:
            raise V02CommandError("command payload passed field must be boolean")
        return payload["passed"] is True
    if payload.get("collection_completed") is False:
        return False
    execution_state = payload.get("execution_state")
    if isinstance(execution_state, str) and (
        execution_state.startswith("REQUIRES_")
        or execution_state == "FROZEN_WORK_PLAN"
    ):
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler not in HANDLERS.values():
        _emit(
            {
                "schema": "policy-learnware.v02-command-error.v0",
                "passed": False,
                "error_type": "V02CommandError",
                "error": "unregistered v0.2 command handler",
            },
            sys.stderr,
        )
        return 2
    try:
        payload = handler(args)
    except Exception as error:  # fail closed at the process boundary
        _emit(
            {
                "schema": "policy-learnware.v02-command-error.v0",
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            sys.stderr,
        )
        return 2
    try:
        completed = _command_completed(payload)
    except Exception as error:
        _emit(
            {
                "schema": "policy-learnware.v02-command-error.v0",
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            sys.stderr,
        )
        return 2
    _emit(payload)
    return 0 if completed else 2


__all__ = [
    "HANDLERS",
    "RECOMPUTE_CHECKS",
    "RECOMPUTE_SECTIONS",
    "V02CommandError",
    "FORMAL_METHOD_IDS",
    "METHOD_KIND_REGISTRY",
    "build_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
