"""Read-only validators required by the frozen v0.2 exact-90 handoff.

The exact-90 pool is an immutable output of the historical training queue.  It
must remain independently verifiable without importing the queue scheduler,
runner, package admission bridge, or implementation-inventory builder that
created it.  This module therefore contains only the strict geometry, bundle,
and completed-attempt checks consumed by :mod:`pool_acceptance`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .anchor_binding import AnchorManifest
from .provenance import (
    ContractError,
    load_strict_json,
    validate_execution_evidence,
    validate_fpo_source_attestation,
    validate_implementation_provenance,
    validate_policy_bundle,
    validate_queue_result,
    validate_run_manifest_server_binding,
    validate_success_record,
    validate_vendor_provenance,
)


RecordedPathResolver = Callable[[str | Path], Path]


def _physical_path(
    value: str | Path, resolver: RecordedPathResolver | None
) -> Path:
    return Path(value).resolve() if resolver is None else Path(resolver(value)).resolve()


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where} must be a positive integer")
    return value


def derive_iterations_per_env(config: Mapping[str, Any]) -> int:
    """Reproduce upstream ``PpoConfig.iterations_per_env`` exactly."""

    if not isinstance(config, Mapping):
        raise ContractError("trainer_config must be an object")
    num_envs = _positive_int(config.get("num_envs"), "trainer_config.num_envs")
    num_minibatches = _positive_int(
        config.get("num_minibatches"), "trainer_config.num_minibatches"
    )
    batch_size = _positive_int(config.get("batch_size"), "trainer_config.batch_size")
    unroll_length = _positive_int(
        config.get("unroll_length"), "trainer_config.unroll_length"
    )
    transition_count = num_minibatches * batch_size * unroll_length
    if transition_count % num_envs != 0:
        raise ContractError(
            "trainer transition geometry must divide exactly by trainer_config.num_envs"
        )
    iterations = transition_count // num_envs
    return _positive_int(iterations, "derived trainer_config.iterations_per_env")


def _validate_checkpoint_bytes(
    *,
    checkpoint: Mapping[str, Any],
    explicit_path: Path | str,
    server_job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    anchor: AnchorManifest,
    execution: Mapping[str, Any],
    source_attestation: Mapping[str, Any],
    attempt_root: Path,
    path_resolver: RecordedPathResolver | None = None,
) -> Path:
    """Bind one selected checkpoint record to its immutable bundle bytes."""

    path = _physical_path(explicit_path, path_resolver)
    if _physical_path(checkpoint["path"], path_resolver) != path:
        raise ContractError("explicit checkpoint path differs from the success record")
    expected_path = attempt_root / "checkpoints" / (
        f"outer_{checkpoint['outer_iteration']:06d}"
    )
    if path != expected_path:
        raise ContractError("checkpoint is outside the canonical formal attempt root")
    observed = validate_policy_bundle(
        path,
        require_evaluation=bool(server_job["training_protocol"]["evaluation"]["enabled"]),
    )
    for key in ("bundle_manifest_sha256", "bundle_manifest_digest", "files"):
        if observed[key] != checkpoint[key]:
            raise ContractError(f"checkpoint {key} differs from immutable bundle bytes")
    manifest = load_strict_json(path / "bundle_manifest.json")
    expected_manifest = {
        "algorithm": server_job["training_protocol"]["algorithm"],
        "task": anchor.task,
        "seed": server_job["seed"],
        "outer_iteration": checkpoint["outer_iteration"],
        "environment_steps": checkpoint["environment_steps"],
    }
    if any(manifest.get(key) != expected for key, expected in expected_manifest.items()):
        raise ContractError("checkpoint bundle manifest semantics drifted from the job")
    provenance = load_strict_json(path / "provenance.json")
    recorded_attempt_root = str(attempt_root)
    if path_resolver is not None:
        raw_attempt_root = execution.get("attempt_root")
        if not isinstance(raw_attempt_root, str):
            raise ContractError("relocated execution evidence lacks recorded attempt_root")
        if _physical_path(raw_attempt_root, path_resolver) != attempt_root.resolve():
            raise ContractError("checkpoint execution root does not resolve to this attempt")
        recorded_attempt_root = raw_attempt_root
    expected_provenance = {
        "config_digest": server_job["config_digest"],
        "execution_purpose": server_job["execution_purpose"],
        "job_digest": server_job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "operator_digest": anchor.operator_digest,
        "model_diff_digest": anchor.model_diff_digest,
        "actual_bound_model_digest": anchor.expected_bound_model_digest,
        **dict(source_attestation),
        "runtime_digest": anchor.runtime_digest,
        "implementation": attempt["implementation"],
        "execution_mode": execution["execution_mode"],
        "formal_eligible": execution["formal_eligible"],
        "execution_evidence_digest": execution["execution_evidence_digest"],
        # Preserve the original absolute receipt string.  The physical root is
        # separately bound through ``path_resolver`` below.
        "attempt_root": recorded_attempt_root,
    }
    failed = [
        key for key, expected in expected_provenance.items() if provenance.get(key) != expected
    ]
    if failed:
        raise ContractError(f"checkpoint provenance binding mismatch: {failed}")
    return path


def validate_completed_attempt(
    attempt_dir: Path,
    job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    expected_vendor: Mapping[str, Any],
    expected_implementation: Mapping[str, Any],
    path_resolver: RecordedPathResolver | None = None,
) -> dict[str, Any]:
    """Cross-check terminal record, runtime evidence, ladder, and bundle bytes."""

    anchor = AnchorManifest.from_path(
        _physical_path(job["anchor_manifest_path"], path_resolver)
    )
    record = validate_success_record(
        attempt_dir / "training_record.json",
        expected_job_digest=str(job["job_digest"]),
        expected_attempt_digest=str(attempt["attempt_digest"]),
        expected_anchor_manifest_digest=anchor.manifest_digest,
        expected_environment_instance_digest=anchor.environment_instance_digest,
        expected_training_protocol_digest=str(job["training_protocol_digest"]),
        expected_config_digest=str(job["config_digest"]),
        expected_execution_purpose=str(job["execution_purpose"]),
    )
    validate_implementation_provenance(
        attempt["implementation"], expected=expected_implementation
    )
    validate_implementation_provenance(
        record["implementation"], expected=expected_implementation
    )
    if record["algorithm"] != job["training_protocol"]["algorithm"]:
        raise ContractError("training record algorithm drifted from the frozen job")
    if record["seed"] != job["seed"]:
        raise ContractError("training record seed drifted from the frozen job")
    for key in (
        "config_digest",
        "execution_purpose",
        "execution_mode",
        "formal_eligible",
    ):
        if record[key] != attempt[key]:
            raise ContractError(f"training record {key} drifted from the attempt")
    run_manifest = validate_run_manifest_server_binding(
        load_strict_json(attempt_dir / "run_manifest.json"),
        job=job,
        attempt=attempt,
        anchor=anchor.to_dict(),
    )
    runtime = run_manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ContractError("run manifest runtime evidence is missing")
    vendor = runtime.get("vendor")
    if not isinstance(vendor, Mapping):
        raise ContractError("run manifest vendor provenance is missing")
    validated_vendor = validate_vendor_provenance(vendor, expected=expected_vendor)
    implementation = runtime.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ContractError("run manifest implementation provenance is missing")
    validate_implementation_provenance(
        implementation, expected=expected_implementation
    )
    if runtime.get("pythonpath_vendor_precedence_verified") is not True:
        raise ContractError("run manifest did not verify vendor PYTHONPATH precedence")
    if runtime.get("wandb_mode") != "disabled":
        raise ContractError("run manifest did not preserve WANDB_MODE=disabled")
    if runtime.get("python_dont_write_bytecode") != "1":
        raise ContractError("run manifest did not disable vendor bytecode writes")
    source_attestation = validate_fpo_source_attestation(
        runtime, expected_commit=str(anchor.runtime["fpo_commit"])
    )
    execution = runtime.get("execution_evidence")
    if not isinstance(execution, Mapping):
        raise ContractError("run manifest execution evidence is missing")
    hardware_digest = runtime.get("hardware_digest")
    evidence = validate_execution_evidence(
        execution,
        expected_job_digest=str(job["job_digest"]),
        expected_attempt_digest=str(attempt["attempt_digest"]),
        expected_hardware_digest=str(hardware_digest),
        expected_config_digest=str(job["config_digest"]),
        expected_execution_purpose=str(job["execution_purpose"]),
        # Validate the immutable recorded value first; then bind that value to
        # the relocated physical directory below.
        expected_attempt_root=Path(execution["attempt_root"]),
        require_formal=bool(attempt["formal_eligible"]),
    )
    if _physical_path(evidence["attempt_root"], path_resolver) != attempt_dir.resolve():
        raise ContractError("execution attempt root does not resolve to this attempt")
    expected_execution_projection = {
        "config_digest": evidence["config_digest"],
        "execution_purpose": evidence["execution_purpose"],
        "execution_mode": evidence["execution_mode"],
        "formal_eligible": evidence["formal_eligible"],
        "execution_evidence_digest": evidence["execution_evidence_digest"],
    }
    for key, expected in expected_execution_projection.items():
        if run_manifest.get(key) != expected or record.get(key) != expected:
            raise ContractError(f"run/record {key} execution binding mismatch")
    command = runtime.get("command")
    if not isinstance(command, list):
        raise ContractError("run manifest command is missing")
    has_allow_flag = "--allow-non-gpu" in command
    if has_allow_flag is not bool(evidence["allow_non_gpu"]):
        raise ContractError("run command disagrees with execution-mode evidence")
    try:
        purpose_index = command.index("--execution-purpose")
    except ValueError as error:
        raise ContractError("run command omitted --execution-purpose") from error
    if (
        command.count("--execution-purpose") != 1
        or purpose_index + 1 >= len(command)
        or command[purpose_index + 1] != job["execution_purpose"]
    ):
        raise ContractError("run command execution purpose drifted from the job")
    try:
        vendor_index = command.index("--vendor-dir")
    except ValueError as error:
        raise ContractError("run command omitted --vendor-dir") from error
    if (
        command.count("--vendor-dir") != 1
        or vendor_index + 1 >= len(command)
        or command[vendor_index + 1] != validated_vendor["path"]
    ):
        raise ContractError("run command vendor directory drifted from provenance")
    try:
        exporter_index = command.index("--legacy-policy-io")
    except ValueError as error:
        raise ContractError("run command omitted --legacy-policy-io") from error
    if (
        command.count("--legacy-policy-io") != 1
        or exporter_index + 1 >= len(command)
        or command[exporter_index + 1] != runtime.get("legacy_policy_io_path")
    ):
        raise ContractError("run command legacy exporter path drifted from provenance")
    observed_outers = [item["outer_iteration"] for item in record["checkpoint_bundles"]]
    frozen_outers = job["training_protocol"]["export_outer_iterations"]
    expected_outers = (
        frozen_outers
        if record["state"] == "succeeded"
        else [
            outer
            for outer in frozen_outers
            if outer <= record["completed_outer_iterations"]
        ]
    )
    if observed_outers != expected_outers:
        raise ContractError(
            "training record checkpoint set drifted from the frozen terminal prefix"
        )
    checkpoint_root = (attempt_dir / "checkpoints").resolve()
    require_evaluation = bool(job["training_protocol"]["evaluation"]["enabled"])
    for item in record["checkpoint_bundles"]:
        bundle = _physical_path(item["path"], path_resolver)
        try:
            bundle.relative_to(checkpoint_root)
        except ValueError as error:
            raise ContractError(
                "recorded checkpoint escapes its immutable attempt root"
            ) from error
        observed = validate_policy_bundle(bundle, require_evaluation=require_evaluation)
        for key in ("bundle_manifest_sha256", "bundle_manifest_digest", "files"):
            if observed[key] != item[key]:
                raise ContractError(
                    f"recorded checkpoint {key} disagrees with bundle bytes"
                )
        if item["bundle_digest"] != observed["bundle_manifest_sha256"]:
            raise ContractError(
                "recorded checkpoint bundle_digest disagrees with bundle bytes"
            )
        finiteness = item["finiteness_audit"]
        if (
            finiteness.get("all_arrays_finite") is not True
            or finiteness.get("bundle_manifest_sha256")
            != observed["bundle_manifest_sha256"]
            or finiteness.get("validated_file_digests") != observed["files"]
        ):
            raise ContractError(
                "checkpoint finiteness audit is not bound to the validated bytes"
            )
        parity_contract = job["training_protocol"]["parity"]
        golden = item["golden_parity"]
        compiled = item["compiled_parity"]
        if (
            golden.get("atol") != parity_contract["atol"]
            or golden.get("rtol") != parity_contract["rtol"]
            or golden.get("sample_count") != parity_contract["golden_sample_count"]
            or compiled.get("atol") != parity_contract["atol"]
            or compiled.get("rtol") != parity_contract["rtol"]
            or compiled.get("sample_count") != parity_contract["compiled_sample_count"]
        ):
            raise ContractError(
                "checkpoint parity evidence differs from the frozen protocol"
            )
        bundle_manifest = load_strict_json(bundle / "bundle_manifest.json")
        expected_bundle_semantics = {
            "algorithm": job["training_protocol"]["algorithm"],
            "task": anchor.task,
            "seed": job["seed"],
            "outer_iteration": item["outer_iteration"],
            "environment_steps": item["environment_steps"],
        }
        for key, expected in expected_bundle_semantics.items():
            if bundle_manifest.get(key) != expected:
                raise ContractError(
                    f"checkpoint bundle {key} drifted from frozen job"
                )
        bundle_provenance = load_strict_json(bundle / "provenance.json")
        expected_bundle_bindings = {
            "config_digest": job["config_digest"],
            "execution_purpose": job["execution_purpose"],
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor.manifest_digest,
            "environment_instance_digest": anchor.environment_instance_digest,
            "operator_digest": anchor.operator_digest,
            "model_diff_digest": anchor.model_diff_digest,
            "actual_bound_model_digest": anchor.expected_bound_model_digest,
            "runtime_digest": anchor.runtime_digest,
            **source_attestation,
            "execution_mode": evidence["execution_mode"],
            "formal_eligible": evidence["formal_eligible"],
            "execution_evidence_digest": evidence["execution_evidence_digest"],
            "attempt_root": evidence["attempt_root"],
            "implementation": expected_implementation,
        }
        for key, expected in expected_bundle_bindings.items():
            if bundle_provenance.get(key) != expected:
                raise ContractError(
                    f"checkpoint provenance {key} binding mismatch"
                )
    status = load_strict_json(attempt_dir / "status.json")
    required_status = {
        "state": "completed" if record["state"] == "succeeded" else "recovered",
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "last_completed_outer": record["completed_outer_iterations"],
        "environment_steps": record["completed_environment_steps"],
        "planned_outer_iterations": record["planned_outer_iterations"],
        "planned_environment_steps": record["planned_environment_steps"],
        "promoted_outer_iteration": record["promoted_outer_iteration"],
        "promoted_environment_steps": record["promoted_environment_steps"],
        "terminal_failure": record["terminal_failure"],
        "training_record_digest": record["record_digest"],
    }
    for key, expected in required_status.items():
        if status.get(key) != expected:
            raise ContractError(f"completed runner status {key} mismatch")
    if status.get("exported_outer_iterations") != observed_outers:
        raise ContractError("completed runner status export list mismatch")
    return record


__all__ = [
    "_validate_checkpoint_bytes",
    "derive_iterations_per_env",
    "validate_completed_attempt",
]
