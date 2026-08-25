"""Fail-closed package-to-server training provenance bridge.

The package and the anchor-aware runner deliberately use different job
schemas.  This module binds them with a separate immutable manifest instead of
pretending their job digests are interchangeable.  Every input is passed
explicitly; no run, checkpoint, or manifest is discovered by directory scan.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .anchor_binding import AnchorManifest
from .formal_plan import validate_formal_training_projection
from .implementation import inspect_implementation_inventory
from .provenance import (
    ContractError,
    FPO_SOURCE_ATTESTATION_KEYS,
    FORMAL_EXECUTION_PURPOSE,
    FORMAL_GPU_EXECUTION_MODE,
    TRAINING_JOB_SCHEMA,
    TRAINING_RECORD_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    assert_finite_mapping,
    finalize_training_job,
    finalize_training_plan,
    load_strict_json,
    revalidate_formal_freeze_binding,
    require_digest,
    require_exact_keys,
    sha256_json,
    validate_attempt,
    validate_execution_evidence,
    validate_fpo_source_attestation,
    validate_implementation_provenance,
    validate_policy_bundle,
    validate_queue_result,
    validate_run_manifest_envelope,
    validate_self_digest,
    validate_training_job,
    validate_training_plan,
    validate_training_protocol,
    validate_vendor_provenance,
    with_self_digest,
)


def _load_package_contracts() -> tuple[Any, Any, Any]:
    """Load the package only from one of the two versioned repository layouts."""

    try:
        from policy_learnware_v0.v02.training import (
            AdmittedTrainingRecord,
            PolicyTrainingAttestation,
            PolicyTrainingJob,
        )
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        candidates = (
            here.parents[2] / "src",
            here.parents[1] / "policy_learnware_v0" / "src",
        )
        for source in candidates:
            if (source / "policy_learnware_v0" / "v02" / "training.py").is_file():
                sys.path.insert(0, str(source))
                break
        else:
            raise ContractError(
                "the versioned policy_learnware_v0 training package is unavailable"
            ) from None
        from policy_learnware_v0.v02.training import (
            AdmittedTrainingRecord,
            PolicyTrainingAttestation,
            PolicyTrainingJob,
        )
    return PolicyTrainingJob, PolicyTrainingAttestation, AdmittedTrainingRecord


PolicyTrainingJob, PolicyTrainingAttestation, AdmittedTrainingRecord = (
    _load_package_contracts()
)


PLAN_BINDING_SCHEMA = "policy-learnware.v02-package-server-training-plan-binding.v0"
PACKAGE_PLAN_SCHEMA = "policy-learnware.v02-policy-training-job-set.v0"
_BINDING_ROW_KEYS = {
    "package_job_id",
    "package_job_digest",
    "config_digest",
    "execution_purpose",
    "formal_protocol_freeze_digest",
    "server_job_id",
    "server_job_digest",
    "source_anchor_id",
    "anchor_manifest_digest",
    "environment_instance_digest",
    "axis_binding_digest",
    "operator_digest",
    "expected_nominal_model_digest",
    "expected_bound_model_digest",
    "training_protocol_digest",
    "algorithm",
    "trainer_config_digest",
    "seed",
    "environment_steps",
    "checkpoint_rule",
    "trainer_commit",
    "dependency_digest",
    "runtime_digest",
}


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where} must be a positive integer")
    return value


def _package_plan_digest(jobs: Sequence[Any]) -> str:
    rows = [job.to_dict() for job in sorted(jobs, key=lambda item: item.job_id)]
    return sha256_json(
        {"schema": PACKAGE_PLAN_SCHEMA, "job_count": len(rows), "jobs": rows}
    )


def _revalidate_formal_plan_authority(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Re-enter canonical formal authority at every package trust boundary."""

    validated = validate_training_plan(plan)
    if validated["execution_purpose"] == FORMAL_EXECUTION_PURPOSE:
        live_binding = revalidate_formal_freeze_binding(
            validated["formal_protocol_freeze"]
        )
        validate_formal_training_projection(validated, live_binding)
    return validated


def derive_iterations_per_env(config: Mapping[str, Any]) -> int:
    """Reproduce the upstream computed ``PpoConfig.iterations_per_env`` exactly."""

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


def _planned_environment_steps(protocol: Mapping[str, Any]) -> int:
    config = protocol["trainer_config"]
    if not isinstance(config, Mapping):
        raise ContractError("training protocol trainer_config must be an object")
    num_envs = _positive_int(config.get("num_envs"), "trainer_config.num_envs")
    iterations = derive_iterations_per_env(config)
    maximum = _positive_int(
        protocol["max_outer_iterations"], "max_outer_iterations"
    )
    return num_envs * iterations * maximum


def _verify_projection(
    package_job: Any,
    *,
    anchor: AnchorManifest,
    protocol: Mapping[str, Any],
    server_job: Mapping[str, Any],
) -> None:
    checks = {
        "job_id": server_job["job_id"] == package_job.job_id,
        "config_digest": (
            server_job["config_digest"] == package_job.config_digest
        ),
        "execution_purpose": (
            server_job["execution_purpose"] == package_job.execution_purpose
        ),
        "source_anchor_id": anchor.anchor_id == package_job.source_anchor_id,
        "anchor_manifest_digest": (
            anchor.manifest_digest == package_job.anchor_manifest_digest
            == server_job["anchor_manifest_digest"]
        ),
        "environment_instance_digest": (
            anchor.environment_instance_digest
            == package_job.environment_instance_digest
        ),
        "training_protocol_digest": (
            protocol["protocol_digest"] == package_job.training_protocol_id
            == server_job["training_protocol_digest"]
        ),
        "embedded_training_protocol": server_job["training_protocol"] == protocol,
        "algorithm": protocol["algorithm"] == package_job.algorithm,
        "trainer_config": protocol["trainer_config"] == package_job.to_dict()[
            "trainer_config"
        ],
        "seed": server_job["seed"] == package_job.seed,
        "environment_steps": (
            _planned_environment_steps(protocol) == package_job.environment_steps
        ),
        "checkpoint_rule": (
            protocol["checkpoint_rule"] == package_job.checkpoint_rule
        ),
        "trainer_commit": anchor.runtime["fpo_commit"] == package_job.trainer_commit,
        "runtime_digest": anchor.runtime_digest == package_job.runtime_digest,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ContractError(
            f"package job cannot be projected to the frozen server contract: {failed}"
        )


def project_policy_training_plan(
    package_jobs: Sequence[Any],
    *,
    anchor_manifest_paths: Mapping[str, Path | str],
    training_protocols: Mapping[str, Mapping[str, Any]],
    formal_protocol_freeze: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project package jobs into a runner plan and an immutable bridge manifest.

    ``anchor_manifest_paths`` and ``training_protocols`` must have exact
    coverage.  The function never searches a directory for a plausible anchor
    or protocol and never supplies trainer/budget/checkpoint defaults.
    """

    jobs = tuple(package_jobs)
    if not jobs or any(not isinstance(job, PolicyTrainingJob) for job in jobs):
        raise ContractError("package_jobs must be a non-empty PolicyTrainingJob sequence")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ContractError("package training plan contains duplicate job IDs")
    if len({job.digest for job in jobs}) != len(jobs):
        raise ContractError("package training plan contains duplicate job digests")

    anchor_ids = {job.source_anchor_id for job in jobs}
    protocol_ids = {job.training_protocol_id for job in jobs}
    if set(anchor_manifest_paths) != anchor_ids:
        raise ContractError("anchor_manifest_paths coverage differs from package jobs")
    if set(training_protocols) != protocol_ids:
        raise ContractError("training_protocols coverage differs from package jobs")

    anchors: dict[str, tuple[Path, AnchorManifest]] = {}
    for anchor_id, raw_path in anchor_manifest_paths.items():
        path = Path(raw_path).resolve()
        anchor = AnchorManifest.from_path(path)
        if anchor.anchor_id != anchor_id:
            raise ContractError("anchor manifest mapping key is not its source anchor ID")
        anchors[anchor_id] = (path, anchor)

    protocols: dict[str, dict[str, Any]] = {}
    for digest, raw_protocol in training_protocols.items():
        protocol = validate_training_protocol(raw_protocol)
        if protocol["protocol_digest"] != require_digest(
            digest, "training_protocols key"
        ):
            raise ContractError("training protocol mapping key is not protocol_digest")
        protocols[digest] = protocol

    purposes = {job.execution_purpose for job in jobs}
    if len(purposes) != 1:
        raise ContractError("package training plan cannot mix execution purposes")
    purpose = next(iter(purposes))
    if purpose == FORMAL_EXECUTION_PURPOSE:
        if formal_protocol_freeze is None:
            raise ContractError(
                "formal package projection requires a canonical freeze binding"
            )
        formal_protocol_freeze = revalidate_formal_freeze_binding(
            formal_protocol_freeze
        )
    elif formal_protocol_freeze is not None:
        raise ContractError("non-formal package projection cannot carry a formal freeze")
    formal_freeze_digest = (
        None
        if formal_protocol_freeze is None
        else formal_protocol_freeze.get("binding_digest")
    )

    server_jobs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for package_job in sorted(jobs, key=lambda item: item.job_id):
        path, anchor = anchors[package_job.source_anchor_id]
        protocol = protocols[package_job.training_protocol_id]
        server_job = finalize_training_job(
            {
                "schema": TRAINING_JOB_SCHEMA,
                "job_id": package_job.job_id,
                "config_digest": package_job.config_digest,
                "execution_purpose": package_job.execution_purpose,
                "formal_protocol_freeze_digest": formal_freeze_digest,
                "anchor_manifest_path": str(path),
                "anchor_manifest_digest": anchor.manifest_digest,
                "training_protocol": protocol,
                "training_protocol_digest": protocol["protocol_digest"],
                "seed": package_job.seed,
            }
        )
        _verify_projection(
            package_job, anchor=anchor, protocol=protocol, server_job=server_job
        )
        server_jobs.append(server_job)
        rows.append(
            {
                "package_job_id": package_job.job_id,
                "package_job_digest": package_job.digest,
                "config_digest": package_job.config_digest,
                "execution_purpose": package_job.execution_purpose,
                "formal_protocol_freeze_digest": formal_freeze_digest,
                "server_job_id": server_job["job_id"],
                "server_job_digest": server_job["job_digest"],
                "source_anchor_id": anchor.anchor_id,
                "anchor_manifest_digest": anchor.manifest_digest,
                "environment_instance_digest": anchor.environment_instance_digest,
                "axis_binding_digest": anchor.axis_binding_digest,
                "operator_digest": anchor.operator_digest,
                "expected_nominal_model_digest": anchor.expected_nominal_model_digest,
                "expected_bound_model_digest": anchor.expected_bound_model_digest,
                "training_protocol_digest": protocol["protocol_digest"],
                "algorithm": package_job.algorithm,
                "trainer_config_digest": sha256_json(protocol["trainer_config"]),
                "seed": package_job.seed,
                "environment_steps": package_job.environment_steps,
                "checkpoint_rule": package_job.checkpoint_rule,
                "trainer_commit": package_job.trainer_commit,
                "dependency_digest": package_job.dependency_digest,
                "runtime_digest": package_job.runtime_digest,
            }
        )

    server_plan = finalize_training_plan(
        server_jobs, formal_protocol_freeze=formal_protocol_freeze
    )
    _revalidate_formal_plan_authority(server_plan)
    binding = with_self_digest(
        {
            "schema": PLAN_BINDING_SCHEMA,
            "package_plan_digest": _package_plan_digest(jobs),
            "server_plan_digest": server_plan["plan_digest"],
            "job_count": len(rows),
            "rows": rows,
        },
        key="binding_digest",
    )
    validate_plan_binding(
        binding, package_jobs=jobs, server_plan=server_plan
    )
    return server_plan, binding


def validate_plan_binding(
    value: Mapping[str, Any],
    *,
    package_jobs: Sequence[Any],
    server_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact two-way coverage of package and server job digests."""

    require_exact_keys(
        value,
        {
            "schema",
            "package_plan_digest",
            "server_plan_digest",
            "job_count",
            "rows",
            "binding_digest",
        },
        "package/server training plan binding",
    )
    if value["schema"] != PLAN_BINDING_SCHEMA:
        raise ContractError("unsupported package/server training binding schema")
    jobs = tuple(package_jobs)
    if any(not isinstance(job, PolicyTrainingJob) for job in jobs):
        raise ContractError("plan binding package jobs are not typed training jobs")
    plan = _revalidate_formal_plan_authority(server_plan)
    if value["package_plan_digest"] != _package_plan_digest(jobs):
        raise ContractError("plan binding belongs to another package job set")
    if value["server_plan_digest"] != plan["plan_digest"]:
        raise ContractError("plan binding belongs to another server plan")
    rows = value["rows"]
    if not isinstance(rows, list) or value["job_count"] != len(rows):
        raise ContractError("plan binding job_count differs from rows")
    if len(rows) != len(jobs) or len(rows) != len(plan["jobs"]):
        raise ContractError("plan binding does not have exact two-way job coverage")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ContractError(f"plan binding row {index} must be an object")
        require_exact_keys(row, _BINDING_ROW_KEYS, f"plan binding row {index}")
        for key in (
            "package_job_digest",
            "server_job_digest",
            "config_digest",
            "source_anchor_id",
            "anchor_manifest_digest",
            "environment_instance_digest",
            "expected_nominal_model_digest",
            "expected_bound_model_digest",
            "training_protocol_digest",
            "trainer_config_digest",
            "dependency_digest",
            "runtime_digest",
        ):
            require_digest(row[key], f"plan binding row {index}.{key}")
        if row["formal_protocol_freeze_digest"] is not None:
            require_digest(
                row["formal_protocol_freeze_digest"],
                f"plan binding row {index}.formal_protocol_freeze_digest",
            )
        if row["axis_binding_digest"] is not None:
            require_digest(
                row["axis_binding_digest"], f"plan binding row {index}.axis_binding_digest"
            )
        if row["operator_digest"] is not None:
            require_digest(
                row["operator_digest"], f"plan binding row {index}.operator_digest"
            )
    by_package = {job.job_id: job for job in jobs}
    by_server = {job["job_id"]: job for job in plan["jobs"]}
    if len(by_package) != len(jobs) or len(by_server) != len(plan["jobs"]):
        raise ContractError("duplicate job IDs prevent exact bridge coverage")
    if {row["package_job_id"] for row in rows} != set(by_package):
        raise ContractError("plan binding package job IDs are not exact")
    if {row["server_job_id"] for row in rows} != set(by_server):
        raise ContractError("plan binding server job IDs are not exact")
    if len({row["package_job_digest"] for row in rows}) != len(rows):
        raise ContractError("plan binding repeats a package job digest")
    if len({row["server_job_digest"] for row in rows}) != len(rows):
        raise ContractError("plan binding repeats a server job digest")
    for row in rows:
        package_job = by_package[row["package_job_id"]]
        server_job = by_server[row["server_job_id"]]
        expected = {
            "package_job_digest": package_job.digest,
            "server_job_digest": server_job["job_digest"],
            "config_digest": package_job.config_digest,
            "execution_purpose": package_job.execution_purpose,
            "formal_protocol_freeze_digest": server_job[
                "formal_protocol_freeze_digest"
            ],
            "source_anchor_id": package_job.source_anchor_id,
            "anchor_manifest_digest": package_job.anchor_manifest_digest,
            "environment_instance_digest": package_job.environment_instance_digest,
            "training_protocol_digest": package_job.training_protocol_id,
            "algorithm": package_job.algorithm,
            "trainer_config_digest": sha256_json(package_job.to_dict()["trainer_config"]),
            "seed": package_job.seed,
            "environment_steps": package_job.environment_steps,
            "checkpoint_rule": package_job.checkpoint_rule,
            "trainer_commit": package_job.trainer_commit,
            "dependency_digest": package_job.dependency_digest,
            "runtime_digest": package_job.runtime_digest,
        }
        failed = [key for key, expected_value in expected.items() if row[key] != expected_value]
        if failed:
            raise ContractError(f"plan binding row drifted from package job: {failed}")
        if server_job["anchor_manifest_digest"] != row["anchor_manifest_digest"]:
            raise ContractError("plan binding row anchor drifted from server job")
        if server_job["training_protocol_digest"] != row["training_protocol_digest"]:
            raise ContractError("plan binding row protocol drifted from server job")
        if server_job["seed"] != row["seed"]:
            raise ContractError("plan binding row seed drifted from server job")
        if server_job["config_digest"] != row["config_digest"]:
            raise ContractError("plan binding row config drifted from server job")
        if server_job["execution_purpose"] != row["execution_purpose"]:
            raise ContractError("plan binding row purpose drifted from server job")
        if (
            server_job["formal_protocol_freeze_digest"]
            != row["formal_protocol_freeze_digest"]
        ):
            raise ContractError("plan binding row formal freeze drifted from server job")
    validate_self_digest(
        value, key="binding_digest", where="package/server training plan binding"
    )
    return dict(value)


def _validate_run_manifest(
    value: Mapping[str, Any],
    *,
    server_job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    anchor: AnchorManifest,
    package_job: Any,
) -> dict[str, Any]:
    value = validate_run_manifest_envelope(value)
    expected = {
        "job": server_job,
        "job_digest": server_job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "config_digest": package_job.config_digest,
        "execution_purpose": package_job.execution_purpose,
        "anchor_manifest": anchor.to_dict(),
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "model_diff_digest": anchor.model_diff_digest,
        "training_protocol_digest": server_job["training_protocol_digest"],
        "config": server_job["training_protocol"]["trainer_config"],
        "planned_environment_steps": package_job.environment_steps,
    }
    failed = [key for key, expected_value in expected.items() if value[key] != expected_value]
    if failed:
        raise ContractError(f"run manifest drifted from the frozen job: {failed}")
    num_envs = _positive_int(value["num_envs"], "run manifest.num_envs")
    iterations = _positive_int(
        value["iterations_per_env"], "run manifest.iterations_per_env"
    )
    expected_num_envs = _positive_int(
        server_job["training_protocol"]["trainer_config"].get("num_envs"),
        "trainer_config.num_envs",
    )
    expected_iterations = derive_iterations_per_env(
        server_job["training_protocol"]["trainer_config"]
    )
    if num_envs != expected_num_envs or iterations != expected_iterations:
        raise ContractError("run manifest transition geometry drifted from native config")
    if value["transitions_per_outer"] != num_envs * iterations:
        raise ContractError("run manifest transition geometry is inconsistent")
    if value["planned_environment_steps"] != (
        value["transitions_per_outer"]
        * server_job["training_protocol"]["max_outer_iterations"]
    ):
        raise ContractError("run manifest environment-step budget is inconsistent")

    audit = value["binding_audit"]
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
    expected_changes = (
        []
        if anchor.operator is None
        else sorted(item.leaf for item in anchor.operator.mutations)
    )
    expected_audit = {
        "anchor_id": anchor.anchor_id,
        "environment_instance_digest": anchor.environment_instance_digest,
        "nominal_model_digest": anchor.expected_nominal_model_digest,
        "bound_model_digest": anchor.expected_bound_model_digest,
        "changed_leaves": expected_changes,
        "model_diff_digest": anchor.model_diff_digest,
        "source_unchanged": True,
        "operator_digest": anchor.operator_digest,
        "manifest_digest": anchor.manifest_digest,
    }
    failed = [key for key, expected_value in expected_audit.items() if audit[key] != expected_value]
    if failed:
        raise ContractError(f"run binding audit drifted from anchor: {failed}")

    runtime = value["runtime"]
    if not isinstance(runtime, Mapping):
        raise ContractError("run manifest runtime must be an object")
    require_exact_keys(
        runtime,
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
        },
        "run manifest runtime",
    )
    vendor = validate_vendor_provenance(runtime["vendor"])
    implementation = validate_implementation_provenance(
        runtime["implementation"], expected=attempt["implementation"]
    )
    if runtime["pythonpath_vendor_precedence_verified"] is not True:
        raise ContractError("run did not verify pinned vendor PYTHONPATH precedence")
    if runtime["wandb_mode"] != "disabled":
        raise ContractError("run did not disable wandb networking")
    if runtime["python_dont_write_bytecode"] != "1":
        raise ContractError("run did not protect the pinned vendor tree from bytecode writes")
    runner_file = runtime["runner_file"]
    legacy_policy_io_path = runtime["legacy_policy_io_path"]
    if (
        not isinstance(legacy_policy_io_path, str)
        or not Path(legacy_policy_io_path).is_absolute()
        or Path(legacy_policy_io_path).name != "policy_io.py"
    ):
        raise ContractError("run did not name an absolute legacy policy_io.py")
    observed_implementation = inspect_implementation_inventory(
        runner_path=runner_file,
        legacy_policy_io_path=legacy_policy_io_path,
    )
    validate_implementation_provenance(
        observed_implementation, expected=implementation
    )
    if runtime["runtime_contract"] != anchor.runtime:
        raise ContractError("run runtime contract differs from the anchor")
    if runtime["runtime_digest"] != package_job.runtime_digest:
        raise ContractError("run runtime digest differs from the package job")
    source_attestation = validate_fpo_source_attestation(
        runtime, expected_commit=package_job.trainer_commit
    )
    if (
        attempt["execution_mode"] != FORMAL_GPU_EXECUTION_MODE
        or attempt["formal_eligible"] is not True
        or attempt["execution_purpose"] != FORMAL_EXECUTION_PURPOSE
        or package_job.execution_purpose != FORMAL_EXECUTION_PURPOSE
    ):
        raise ContractError(
            "audit-smoke/development attempt is not formal training evidence"
        )
    hardware = runtime["hardware_contract"]
    if not isinstance(hardware, Mapping):
        raise ContractError("run hardware_contract must be an object")
    require_exact_keys(
        hardware,
        {"host", "platform", "jax_backend", "jax_devices", "cuda_visible_devices"},
        "run hardware contract",
    )
    if runtime["hardware_digest"] != sha256_json(hardware):
        raise ContractError("run hardware_digest mismatch")
    duplicates = {
        "host": runtime["host"],
        "platform": runtime["platform"],
        "jax_backend": runtime["jax_backend"],
        "jax_devices": runtime["jax_devices"],
        "cuda_visible_devices": runtime["cuda_visible_devices"],
    }
    if duplicates != hardware:
        raise ContractError("run hardware projection is internally inconsistent")
    execution = runtime["execution_evidence"]
    if not isinstance(execution, Mapping):
        raise ContractError("run execution_evidence must be an object")
    evidence = validate_execution_evidence(
        execution,
        expected_job_digest=server_job["job_digest"],
        expected_attempt_digest=attempt["attempt_digest"],
        expected_hardware_digest=runtime["hardware_digest"],
        expected_config_digest=package_job.config_digest,
        expected_execution_purpose=package_job.execution_purpose,
        require_formal=True,
    )
    projection = {
        "config_digest": evidence["config_digest"],
        "execution_purpose": evidence["execution_purpose"],
        "execution_mode": evidence["execution_mode"],
        "formal_eligible": evidence["formal_eligible"],
        "execution_evidence_digest": evidence["execution_evidence_digest"],
    }
    failed = [key for key, expected_value in projection.items() if value[key] != expected_value]
    if failed:
        raise ContractError(f"run execution-evidence projection drifted: {failed}")
    command = runtime["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ContractError("run command must be a non-empty string list")
    if "--allow-non-gpu" in command:
        raise ContractError("debug runner flag is not admissible as formal evidence")
    try:
        purpose_index = command.index("--execution-purpose")
    except ValueError as error:
        raise ContractError("formal runner command omitted --execution-purpose") from error
    if (
        purpose_index + 1 >= len(command)
        or command[purpose_index + 1] != FORMAL_EXECUTION_PURPOSE
        or command.count("--execution-purpose") != 1
    ):
        raise ContractError("formal runner command execution purpose drifted")
    try:
        vendor_index = command.index("--vendor-dir")
    except ValueError as error:
        raise ContractError("formal runner command omitted --vendor-dir") from error
    if (
        vendor_index + 1 >= len(command)
        or command[vendor_index + 1] != vendor["path"]
        or command.count("--vendor-dir") != 1
    ):
        raise ContractError("formal runner command vendor directory drifted")
    try:
        exporter_index = command.index("--legacy-policy-io")
    except ValueError as error:
        raise ContractError("formal runner command omitted --legacy-policy-io") from error
    if (
        exporter_index + 1 >= len(command)
        or command[exporter_index + 1] != legacy_policy_io_path
        or command.count("--legacy-policy-io") != 1
    ):
        raise ContractError("formal runner command legacy exporter path drifted")
    if (
        not isinstance(runner_file, str)
        or Path(runner_file).name != "runner.py"
        or Path(runner_file).parent.name != "repro_fpo_ppo_v02"
    ):
        raise ContractError("formal evidence did not name the versioned v0.2 runner")
    return dict(value)


def _validate_success_record_mapping(
    value: Mapping[str, Any],
    *,
    server_job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    anchor: AnchorManifest,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    require_exact_keys(
        value,
        {
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
            "started_at",
            "finished_at",
            "wall_seconds",
            "record_digest",
        },
        "training success record",
    )
    if value["schema"] != TRAINING_RECORD_SCHEMA or value["state"] != "succeeded":
        raise ContractError("training record is not a successful v0.2 record")
    expected = {
        "config_digest": server_job["config_digest"],
        "execution_purpose": server_job["execution_purpose"],
        "job_digest": server_job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "training_protocol_digest": server_job["training_protocol_digest"],
        "algorithm": server_job["training_protocol"]["algorithm"],
        "seed": server_job["seed"],
        "execution_mode": execution["execution_mode"],
        "formal_eligible": execution["formal_eligible"],
        "implementation": attempt["implementation"],
        "execution_evidence_digest": execution["execution_evidence_digest"],
    }
    failed = [key for key, expected_value in expected.items() if value[key] != expected_value]
    if failed:
        raise ContractError(f"training success record drifted from job: {failed}")
    checkpoints = value["checkpoint_bundles"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ContractError("successful training record has no checkpoints")
    expected_outers = server_job["training_protocol"]["export_outer_iterations"]
    observed_outers: list[int] = []
    per_outer = _planned_environment_steps(server_job["training_protocol"]) // (
        server_job["training_protocol"]["max_outer_iterations"]
    )
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ContractError(f"checkpoint {index} must be an object")
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
            f"checkpoint {index}",
        )
        outer = _positive_int(checkpoint["outer_iteration"], "checkpoint outer")
        observed_outers.append(outer)
        if checkpoint["environment_steps"] != outer * per_outer:
            raise ContractError("checkpoint environment steps drifted from frozen geometry")
        if not isinstance(checkpoint["path"], str) or not checkpoint["path"]:
            raise ContractError("checkpoint path must be an explicit non-empty string")
        for name in (
            "bundle_manifest_sha256",
            "bundle_manifest_digest",
            "bundle_digest",
        ):
            require_digest(checkpoint[name], f"checkpoint {index}.{name}")
        if checkpoint["bundle_digest"] != checkpoint["bundle_manifest_sha256"]:
            raise ContractError("bundle_digest must be the manifest file SHA-256")
        for key in (
            "config_digest",
            "execution_purpose",
            "execution_mode",
            "formal_eligible",
            "execution_evidence_digest",
        ):
            if checkpoint[key] != execution[key]:
                raise ContractError(
                    f"checkpoint {key} differs from formal execution evidence"
                )
        files = checkpoint["files"]
        expected_files = {
            "actor.npz",
            "golden_io.npz",
            "obs_stats.npz",
            "policy_spec.json",
            "provenance.json",
        }
        if not isinstance(files, Mapping) or set(files) != expected_files:
            raise ContractError("checkpoint file digest inventory is not exact")
        for name, digest in files.items():
            require_digest(digest, f"checkpoint {index} files.{name}")
        for report_name in ("finiteness_audit", "golden_parity", "compiled_parity"):
            report = checkpoint[report_name]
            if not isinstance(report, Mapping) or report.get("passed") is not True:
                raise ContractError(f"checkpoint {report_name} did not pass")
            validate_self_digest(
                report,
                key="report_digest",
                where=f"checkpoint {index} {report_name}",
            )
        if checkpoint["finiteness_audit"].get("all_arrays_finite") is not True:
            raise ContractError("checkpoint finiteness audit is incomplete")
        if checkpoint["golden_parity"].get("raw_checked") is not True:
            raise ContractError("checkpoint golden raw-action parity is incomplete")
        if checkpoint["compiled_parity"].get("next_keys_equal") is not True:
            raise ContractError("checkpoint compiled PRNG-key parity is incomplete")
    if observed_outers != expected_outers:
        raise ContractError("checkpoint export set differs from the frozen protocol")
    for key in ("started_at", "finished_at"):
        if not isinstance(value[key], str) or not value[key]:
            raise ContractError(f"training record {key} must be non-empty")
    assert_finite_mapping(value["wall_seconds"], where="training record.wall_seconds")
    if isinstance(value["wall_seconds"], bool) or float(value["wall_seconds"]) < 0.0:
        raise ContractError("training record wall_seconds must be nonnegative")
    validate_self_digest(value, key="record_digest", where="training success record")
    return dict(value)


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
) -> Path:
    path = Path(explicit_path).resolve()
    if Path(checkpoint["path"]).resolve() != path:
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
        "attempt_root": str(attempt_root),
    }
    failed = [
        key for key, expected in expected_provenance.items() if provenance.get(key) != expected
    ]
    if failed:
        raise ContractError(f"checkpoint provenance binding mismatch: {failed}")
    return path


def _validate_formal_attempt_root(
    *,
    server_job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    anchor: AnchorManifest,
    run_manifest: Mapping[str, Any],
    training_record: Mapping[str, Any],
    checkpoint_paths: Mapping[int, Path | str],
) -> Path:
    """Bind supplied mappings to one canonical immutable queue attempt root."""

    execution = run_manifest["runtime"]["execution_evidence"]
    root = Path(execution["attempt_root"]).resolve()
    validate_execution_evidence(
        execution,
        expected_job_digest=server_job["job_digest"],
        expected_attempt_digest=attempt["attempt_digest"],
        expected_hardware_digest=run_manifest["runtime"]["hardware_digest"],
        expected_config_digest=server_job["config_digest"],
        expected_execution_purpose=server_job["execution_purpose"],
        expected_attempt_root=root,
        require_formal=True,
    )
    expected_name = f"attempt_{attempt['attempt_number']:03d}"
    if (
        root.name != expected_name
        or root.parent.name != server_job["job_id"]
        or root.parent.parent.name != "jobs"
    ):
        raise ContractError("formal evidence is not rooted in a canonical queue attempt")

    expected_files = {
        "attempt_manifest.json": attempt,
        "run_manifest.json": run_manifest,
        "training_record.json": training_record,
    }
    for name, expected in expected_files.items():
        path = root / name
        if load_strict_json(path) != expected:
            raise ContractError(f"supplied {name} differs from immutable attempt bytes")
    if load_strict_json(root.parent / "job_manifest.json") != server_job:
        raise ContractError("formal attempt job root differs from the server job")
    if AnchorManifest.from_path(server_job["anchor_manifest_path"]).to_dict() != anchor.to_dict():
        raise ContractError("supplied anchor manifest differs from its frozen path")

    result = validate_queue_result(
        load_strict_json(root / "queue_result.json"),
        expected_job_digest=server_job["job_digest"],
        expected_attempt_digest=attempt["attempt_digest"],
        expected_config_digest=server_job["config_digest"],
        expected_execution_purpose=server_job["execution_purpose"],
    )
    if result["state"] != "succeeded":
        raise ContractError("formal attempt has no successful queue result")
    validate_vendor_provenance(
        result["vendor"], expected=run_manifest["runtime"]["vendor"]
    )
    validate_implementation_provenance(
        result["implementation"], expected=attempt["implementation"]
    )
    if (
        result["execution_mode"] != FORMAL_GPU_EXECUTION_MODE
        or result["formal_eligible"] is not True
        or result["execution_purpose"] != FORMAL_EXECUTION_PURPOSE
        or result["config_digest"] != server_job["config_digest"]
        or "--allow-non-gpu" in result["command"]
        or result["command"] != run_manifest["runtime"]["command"]
    ):
        raise ContractError("queue result identifies non-formal/debug execution")

    checkpoint_root = root / "checkpoints"
    for outer, raw_path in checkpoint_paths.items():
        expected_path = checkpoint_root / f"outer_{outer:06d}"
        if Path(raw_path).resolve() != expected_path:
            raise ContractError("explicit checkpoint escapes the formal attempt root")
    return root


def attestation_from_server_success(
    *,
    package_job: Any,
    package_jobs: Sequence[Any],
    binding_manifest: Mapping[str, Any],
    server_plan: Mapping[str, Any],
    attempt_manifest: Mapping[str, Any],
    anchor_manifest: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    training_record: Mapping[str, Any],
    checkpoint_paths: Mapping[int, Path | str],
) -> Any:
    """Validate explicit runner evidence and construct a package attestation."""

    if not isinstance(package_job, PolicyTrainingJob):
        raise ContractError("package_job must be a PolicyTrainingJob")
    plan = validate_training_plan(server_plan)
    binding = validate_plan_binding(
        binding_manifest, package_jobs=package_jobs, server_plan=plan
    )
    typed_matches = [
        item
        for item in package_jobs
        if isinstance(item, PolicyTrainingJob)
        and item.job_id == package_job.job_id
        and item.digest == package_job.digest
    ]
    if len(typed_matches) != 1:
        raise ContractError("selected package job is not uniquely present in package_jobs")
    rows = [
        row
        for row in binding["rows"]
        if isinstance(row, Mapping)
        and row.get("package_job_id") == package_job.job_id
        and row.get("package_job_digest") == package_job.digest
    ]
    if len(rows) != 1:
        raise ContractError("package job has no unique row in the plan binding")
    row = rows[0]
    require_exact_keys(row, _BINDING_ROW_KEYS, "selected plan binding row")
    server_jobs = [
        job
        for job in plan["jobs"]
        if job["job_id"] == row["server_job_id"]
        and job["job_digest"] == row["server_job_digest"]
    ]
    if len(server_jobs) != 1:
        raise ContractError("plan binding has no unique server job")
    server_job = validate_training_job(server_jobs[0])
    anchor = AnchorManifest.from_dict(anchor_manifest)
    protocol = validate_training_protocol(server_job["training_protocol"])
    _verify_projection(
        package_job, anchor=anchor, protocol=protocol, server_job=server_job
    )
    row_expected = {
        "source_anchor_id": anchor.anchor_id,
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "axis_binding_digest": anchor.axis_binding_digest,
        "operator_digest": anchor.operator_digest,
        "expected_nominal_model_digest": anchor.expected_nominal_model_digest,
        "expected_bound_model_digest": anchor.expected_bound_model_digest,
    }
    if any(row[key] != expected for key, expected in row_expected.items()):
        raise ContractError("selected plan binding row differs from anchor manifest")

    attempt = validate_attempt(attempt_manifest)
    if attempt["plan_digest"] != plan["plan_digest"]:
        raise ContractError("attempt belongs to another server plan")
    if attempt["job"] != server_job or attempt["job_digest"] != server_job["job_digest"]:
        raise ContractError("attempt embeds another server job")
    run = _validate_run_manifest(
        run_manifest,
        server_job=server_job,
        attempt=attempt,
        anchor=anchor,
        package_job=package_job,
    )
    execution = run["runtime"]["execution_evidence"]
    source_attestation = validate_fpo_source_attestation(
        run["runtime"], expected_commit=package_job.trainer_commit
    )
    record = _validate_success_record_mapping(
        training_record,
        server_job=server_job,
        attempt=attempt,
        anchor=anchor,
        execution=execution,
    )
    outers = {item["outer_iteration"] for item in record["checkpoint_bundles"]}
    if set(checkpoint_paths) != outers:
        raise ContractError("explicit checkpoint path coverage differs from the record")
    attempt_root = _validate_formal_attempt_root(
        server_job=server_job,
        attempt=attempt,
        anchor=anchor,
        run_manifest=run,
        training_record=record,
        checkpoint_paths=checkpoint_paths,
    )
    verified_paths: dict[int, Path] = {}
    for checkpoint in record["checkpoint_bundles"]:
        outer = checkpoint["outer_iteration"]
        verified_paths[outer] = _validate_checkpoint_bytes(
            checkpoint=checkpoint,
            explicit_path=checkpoint_paths[outer],
            server_job=server_job,
            attempt=attempt,
            anchor=anchor,
            execution=execution,
            source_attestation=source_attestation,
            attempt_root=attempt_root,
        )
    final = record["checkpoint_bundles"][-1]
    if final["outer_iteration"] != server_job["training_protocol"]["max_outer_iterations"]:
        raise ContractError("final attested bundle is not the frozen final checkpoint")
    if final["environment_steps"] != package_job.environment_steps:
        raise ContractError("final bundle budget differs from the package job")
    checkpoints = {
        f"outer_{item['outer_iteration']:06d}": item["bundle_digest"]
        for item in record["checkpoint_bundles"]
    }
    return PolicyTrainingAttestation(
        job_id=package_job.job_id,
        job_digest=package_job.digest,
        attempt_id=attempt["attempt_id"],
        attempt_number=attempt["attempt_number"],
        source_anchor_id=package_job.source_anchor_id,
        anchor_manifest_digest=package_job.anchor_manifest_digest,
        declared_environment_instance_digest=package_job.environment_instance_digest,
        actual_train_environment_instance_digest=run["environment_instance_digest"],
        actual_eval_environment_instance_digest=record["environment_instance_digest"],
        operator_digest=anchor.operator_digest,
        model_diff_digest=run["model_diff_digest"],
        algorithm=package_job.algorithm,
        seed=package_job.seed,
        environment_steps=package_job.environment_steps,
        checkpoint_rule=package_job.checkpoint_rule,
        checkpoint_digests=checkpoints,
        bundle_digest=final["bundle_digest"],
        bundle_manifest_digest=final["bundle_manifest_digest"],
        golden_parity_digest=final["golden_parity"]["report_digest"],
        compiled_parity_digest=final["compiled_parity"]["report_digest"],
        finiteness_audit_digest=final["finiteness_audit"]["report_digest"],
        all_arrays_finite=True,
        golden_parity_passed=True,
        compiled_parity_passed=True,
        trainer_commit=package_job.trainer_commit,
        dependency_digest=package_job.dependency_digest,
        runtime_digest=package_job.runtime_digest,
        hardware_digest=run["runtime"]["hardware_digest"],
        started_at=record["started_at"],
        finished_at=record["finished_at"],
        elapsed_seconds=record["wall_seconds"],
        status="succeeded",
        bundle_path=str(verified_paths[final["outer_iteration"]]),
        server_plan_binding_digest=binding["binding_digest"],
        server_training_plan_digest=plan["plan_digest"],
        server_job_digest=server_job["job_digest"],
        server_attempt_digest=attempt["attempt_digest"],
        server_run_manifest_digest=run["run_manifest_digest"],
        server_training_record_digest=record["record_digest"],
    )


def admit_server_success(**evidence: Any) -> Any:
    """Construct and immediately admit one fully server-bound success record."""

    package_job = evidence.get("package_job")
    attestation = attestation_from_server_success(**evidence)
    if not attestation.is_server_bound:
        raise ContractError("server bridge produced an unbound attestation")
    return AdmittedTrainingRecord(job=package_job, attestation=attestation)


def admit_server_success_batch(
    *,
    package_jobs: Sequence[Any],
    binding_manifest: Mapping[str, Any],
    server_plan: Mapping[str, Any],
    evidence_by_job: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Admit an exact package-plan-sized set of explicit server evidence.

    Each evidence row has exactly ``attempt_manifest``, ``anchor_manifest``,
    ``run_manifest``, ``training_record``, and ``checkpoint_paths``.  The
    caller must load those named artifacts explicitly; this function never
    searches a runs root or substitutes a later successful attempt.
    """

    jobs = tuple(package_jobs)
    by_id = {job.job_id: job for job in jobs if isinstance(job, PolicyTrainingJob)}
    if len(by_id) != len(jobs):
        raise ContractError("batch package jobs must be typed with unique job IDs")
    if set(evidence_by_job) != set(by_id):
        raise ContractError("server evidence coverage differs from the package plan")
    _revalidate_formal_plan_authority(server_plan)
    validate_plan_binding(
        binding_manifest, package_jobs=jobs, server_plan=server_plan
    )
    expected_fields = {
        "attempt_manifest",
        "anchor_manifest",
        "run_manifest",
        "training_record",
        "checkpoint_paths",
    }
    admitted: dict[str, Any] = {}
    for job_id in sorted(by_id):
        evidence = evidence_by_job[job_id]
        if not isinstance(evidence, Mapping):
            raise ContractError(f"server evidence for {job_id} must be an object")
        require_exact_keys(evidence, expected_fields, f"server evidence for {job_id}")
        admitted[job_id] = admit_server_success(
            package_job=by_id[job_id],
            package_jobs=jobs,
            binding_manifest=binding_manifest,
            server_plan=server_plan,
            attempt_manifest=evidence["attempt_manifest"],
            anchor_manifest=evidence["anchor_manifest"],
            run_manifest=evidence["run_manifest"],
            training_record=evidence["training_record"],
            checkpoint_paths=evidence["checkpoint_paths"],
        )
    return admitted


__all__ = [
    "PACKAGE_PLAN_SCHEMA",
    "PLAN_BINDING_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "admit_server_success",
    "admit_server_success_batch",
    "attestation_from_server_success",
    "derive_iterations_per_env",
    "project_policy_training_plan",
    "validate_plan_binding",
]
