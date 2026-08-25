from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from policy_learnware_v0.v02.training import PolicyTrainingJob
from server.repro_fpo_ppo_v02 import package_bridge as package_bridge_module
from server.repro_fpo_ppo_v02.anchor_binding import (
    ANCHOR_MANIFEST_SCHEMA,
    ANCHOR_OPERATOR_SCHEMA,
    finalize_anchor_manifest,
)
from server.repro_fpo_ppo_v02.package_bridge import (
    admit_server_success,
    admit_server_success_batch,
    attestation_from_server_success,
    derive_iterations_per_env,
    project_policy_training_plan,
)
from server.repro_fpo_ppo_v02.implementation import inspect_implementation_inventory
from server.repro_fpo_ppo_v02.provenance import (
    AUDIT_SMOKE_EXECUTION_MODE,
    ATTEMPT_SCHEMA,
    EXECUTION_EVIDENCE_SCHEMA,
    FORMAL_GPU_EXECUTION_MODE,
    FORMAL_EXECUTION_PURPOSE,
    QUEUE_RESULT_SCHEMA,
    TRAINING_PROTOCOL_SCHEMA,
    TRAINING_RECORD_SCHEMA,
    VENDOR_PROVENANCE_SCHEMA,
    ContractError,
    atomic_write_json,
    finalize_attempt,
    finalize_training_protocol,
    sha256_file,
    sha256_json,
    validate_formal_freeze_binding,
    validate_training_plan,
    with_self_digest,
)
from server.repro_fpo_ppo_v02.tests.helpers import make_formal_freeze_binding


_REAL_REVALIDATE_FORMAL_FREEZE = (
    package_bridge_module.revalidate_formal_freeze_binding
)
_REAL_VALIDATE_FORMAL_PROJECTION = (
    package_bridge_module.validate_formal_training_projection
)


@pytest.fixture(autouse=True)
def _synthetic_formal_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let one-job fixtures exercise evidence below the formal authority gate."""

    monkeypatch.setattr(
        package_bridge_module,
        "revalidate_formal_freeze_binding",
        validate_formal_freeze_binding,
    )
    monkeypatch.setattr(
        package_bridge_module,
        "validate_formal_training_projection",
        lambda plan, _binding: validate_training_plan(plan),
    )


def _anchor(path: Path, *, nominal: bool = False) -> dict[str, Any]:
    runtime = {
        "fpo_commit": "b" * 40,
        "python_major_minor": "3.11",
        "jax": "synthetic",
        "jaxlib": "synthetic",
        "mujoco": "synthetic",
        "playground": "synthetic",
    }
    operator = None
    factor = 1.0
    axis_binding = None
    bound_model = "1" * 64
    if not nominal:
        factor = 2.0
        axis_binding = "c" * 64
        bound_model = "2" * 64
        operator = {
            "schema": ANCHOR_OPERATOR_SCHEMA,
            "operator_id": "synthetic-damping-x2",
            "axis_id": "synthetic-damping",
            "axis_registry_digest": "a" * 64,
            "factor": factor,
            "mutations": [
                {
                    "leaf": "_mjx_model.dof_damping",
                    "flat_indices": [0],
                    "multiplier": factor,
                    "expected_before_digest": "3" * 64,
                    "expected_after_digest": "4" * 64,
                }
            ],
        }
    value = finalize_anchor_manifest(
        {
            "schema": ANCHOR_MANIFEST_SCHEMA,
            "task": "SyntheticTask",
            "backend": "mujoco_playground.registry",
            "nominal": nominal,
            "factor": factor,
            "environment_class": "synthetic.FakeEnv",
            "registry_config": {"episode_length": 64, "action_repeat": 1},
            "runtime": runtime,
            "expected_nominal_model_digest": "1" * 64,
            "expected_bound_model_digest": bound_model,
            "operator": operator,
            "axis_binding_digest": axis_binding,
        }
    )
    atomic_write_json(path, value, overwrite=False)
    return value


def _protocol() -> dict[str, Any]:
    return finalize_training_protocol(
        {
            "schema": TRAINING_PROTOCOL_SCHEMA,
            "algorithm": "ppo",
            "trainer_config": {
                "num_timesteps": 128,
                "num_envs": 8,
                "num_minibatches": 2,
                "batch_size": 8,
                "unroll_length": 8,
            },
            "max_outer_iterations": 1,
            "export_outer_iterations": [1],
            "evaluation": {"enabled": False, "num_envs": 1, "base_seed": 0},
            "parity": {
                "atol": 1.0e-6,
                "rtol": 1.0e-6,
                "golden_sample_count": 8,
                "compiled_sample_count": 2,
            },
            "checkpoint_rule": "fixed_final",
        }
    )


def _package_job(anchor: dict[str, Any], protocol: dict[str, Any]) -> PolicyTrainingJob:
    return PolicyTrainingJob(
        job_id="v02-package-job-001",
        config_digest="0" * 64,
        execution_purpose=FORMAL_EXECUTION_PURPOSE,
        source_anchor_id=anchor["anchor_id"],
        environment_instance_digest=anchor["environment_instance_digest"],
        anchor_manifest_digest=anchor["manifest_digest"],
        algorithm="ppo",
        trainer_config=protocol["trainer_config"],
        seed=7,
        environment_steps=128,
        checkpoint_rule="fixed_final",
        trainer_commit=anchor["runtime"]["fpo_commit"],
        dependency_digest="d" * 64,
        runtime_digest=anchor["runtime_digest"],
        training_protocol_id=protocol["protocol_digest"],
    )


def _formal_binding(tmp_path: Path, anchor_id: str) -> dict[str, Any]:
    anchor_ids = [anchor_id]
    for index in range(29):
        candidate = sha256_json({"synthetic_formal_anchor": index})
        if candidate not in anchor_ids:
            anchor_ids.append(candidate)
    return make_formal_freeze_binding(
        tmp_path,
        anchor_ids=anchor_ids,
        seeds=[7, 8, 9],
        config_digest="0" * 64,
        algorithm="ppo",
        training_steps=128,
        checkpoint_rule="fixed_final",
    )


def _bundle(
    root: Path,
    *,
    package_job: PolicyTrainingJob,
    server_job: dict[str, Any],
    attempt: dict[str, Any],
    anchor: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True)
    np.savez(root / "actor.npz", layer=np.asarray([1.0, 2.0], dtype=np.float32))
    np.savez(
        root / "obs_stats.npz",
        count=np.asarray(1.0),
        mean=np.zeros(2, dtype=np.float32),
        var_sum=np.ones(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )
    raw_action = np.ones((8, 1), dtype=np.float32)
    np.savez(
        root / "golden_io.npz",
        observation=np.zeros((8, 2), dtype=np.float32),
        prng_key_data=np.asarray([0, 7], dtype=np.uint32),
        raw_action=raw_action,
        environment_action=np.tanh(raw_action),
    )
    atomic_write_json(
        root / "policy_spec.json",
        {"observation_size": 2, "action_size": 1},
        overwrite=False,
    )
    atomic_write_json(
        root / "provenance.json",
        {
            "evaluation": None,
            "config_digest": server_job["config_digest"],
            "execution_purpose": server_job["execution_purpose"],
            "job_digest": server_job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor["manifest_digest"],
        "environment_instance_digest": anchor["environment_instance_digest"],
        "operator_digest": anchor["operator_digest"],
        "model_diff_digest": anchor["model_diff_digest"],
        "actual_bound_model_digest": anchor["expected_bound_model_digest"],
            "runtime_digest": package_job.runtime_digest,
            "implementation": attempt["implementation"],
            "execution_mode": execution["execution_mode"],
            "formal_eligible": execution["formal_eligible"],
            "execution_evidence_digest": execution["execution_evidence_digest"],
            "attempt_root": execution["attempt_root"],
        },
        overwrite=False,
    )
    file_names = {
        "actor.npz",
        "golden_io.npz",
        "obs_stats.npz",
        "policy_spec.json",
        "provenance.json",
    }
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in sorted(file_names)
    }
    manifest = {
        "schema": "policy-learnware.policy-bundle.v0",
        "complete": True,
        "created_at": "synthetic",
        "algorithm": "ppo",
        "task": anchor["task"],
        "seed": package_job.seed,
        "outer_iteration": 1,
        "environment_steps": package_job.environment_steps,
        "files": files,
    }
    atomic_write_json(root / "bundle_manifest.json", manifest, overwrite=False)
    manifest_file_digest = sha256_file(root / "bundle_manifest.json")
    inventory = {name: files[name]["sha256"] for name in sorted(files)}
    finiteness = with_self_digest(
        {
            "passed": True,
            "all_arrays_finite": True,
            "bundle_manifest_sha256": manifest_file_digest,
            "validated_file_digests": inventory,
        },
        key="report_digest",
    )
    golden = with_self_digest(
        {
            "passed": True,
            "raw_checked": True,
            "raw_max_abs_error": 0.0,
            "environment_max_abs_error": 0.0,
            "atol": 1.0e-6,
            "rtol": 1.0e-6,
            "sample_count": 8,
        },
        key="report_digest",
    )
    compiled = with_self_digest(
        {
            "passed": True,
            "max_abs_error": 0.0,
            "atol": 1.0e-6,
            "rtol": 1.0e-6,
            "sample_count": 2,
            "next_keys_equal": True,
        },
        key="report_digest",
    )
    return {
        "outer_iteration": 1,
        "environment_steps": package_job.environment_steps,
        "path": str(root.resolve()),
        "bundle_manifest_sha256": manifest_file_digest,
        "bundle_manifest_digest": sha256_json(manifest),
        "files": inventory,
        "bundle_digest": manifest_file_digest,
        "config_digest": server_job["config_digest"],
        "execution_purpose": server_job["execution_purpose"],
        "execution_mode": execution["execution_mode"],
        "formal_eligible": execution["formal_eligible"],
        "execution_evidence_digest": execution["execution_evidence_digest"],
        "finiteness_audit": finiteness,
        "golden_parity": golden,
        "compiled_parity": compiled,
    }


def _evidence(
    tmp_path: Path,
    *,
    nominal: bool = False,
    execution_mode: str = FORMAL_GPU_EXECUTION_MODE,
    execution_purpose: str = FORMAL_EXECUTION_PURPOSE,
    jax_backend: str = "gpu",
    canonical_root: bool = True,
) -> dict[str, Any]:
    anchor_path = tmp_path / "anchor.json"
    anchor = _anchor(anchor_path, nominal=nominal)
    protocol = _protocol()
    package_job = _package_job(anchor, protocol)
    package_job = replace(package_job, execution_purpose=execution_purpose)
    formal_freeze = (
        _formal_binding(tmp_path, anchor["anchor_id"])
        if execution_purpose == FORMAL_EXECUTION_PURPOSE
        else None
    )
    server_plan, binding = project_policy_training_plan(
        (package_job,),
        anchor_manifest_paths={package_job.source_anchor_id: anchor_path},
        training_protocols={package_job.training_protocol_id: protocol},
        formal_protocol_freeze=formal_freeze,
    )
    server_job = server_plan["jobs"][0]
    formal_eligible = (
        execution_mode == FORMAL_GPU_EXECUTION_MODE
        and execution_purpose == FORMAL_EXECUTION_PURPOSE
    )
    allow_non_gpu = execution_mode == AUDIT_SMOKE_EXECUTION_MODE
    repository_root = Path(__file__).resolve().parents[2]
    runner_path = (
        repository_root / "server/repro_fpo_ppo_v02/runner.py"
    ).resolve()
    legacy_policy_io = tmp_path / "legacy" / "policy_io.py"
    legacy_policy_io.parent.mkdir()
    legacy_policy_io.write_text(
        "def export_policy_bundle(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    implementation = inspect_implementation_inventory(
        runner_path=runner_path,
        legacy_policy_io_path=legacy_policy_io,
    )
    attempt_root = (
        tmp_path / "runs" / "jobs" / server_job["job_id"] / "attempt_001"
        if canonical_root
        else tmp_path / "nonformal-root" / "attempt_001"
    ).resolve()
    attempt = finalize_attempt(
        {
            "schema": ATTEMPT_SCHEMA,
            "plan_digest": server_plan["plan_digest"],
            "job": server_job,
            "job_digest": server_job["job_digest"],
            "attempt_id": "v02-package-job-001.a001",
            "attempt_number": 1,
            "execution_attempt_id": "exec-synthetic-001",
            "gpu": "0",
            "config_digest": server_job["config_digest"],
            "execution_purpose": server_job["execution_purpose"],
            "execution_mode": execution_mode,
            "formal_eligible": formal_eligible,
            "implementation": implementation,
            "created_at": "2026-08-24T00:00:00Z",
        }
    )
    changed = [] if nominal else ["_mjx_model.dof_damping"]
    hardware = {
        "host": "synthetic-host",
        "platform": "synthetic-platform",
        "jax_backend": jax_backend,
        "jax_devices": [f"Synthetic{jax_backend.title()}Device(id=0)"],
        "cuda_visible_devices": "0",
    }
    execution = with_self_digest(
        {
            "schema": EXECUTION_EVIDENCE_SCHEMA,
            "config_digest": server_job["config_digest"],
            "execution_purpose": server_job["execution_purpose"],
            "execution_mode": execution_mode,
            "formal_eligible": formal_eligible,
            "allow_non_gpu": allow_non_gpu,
            "jax_backend": hardware["jax_backend"],
            "jax_devices": hardware["jax_devices"],
            "cuda_visible_devices": hardware["cuda_visible_devices"],
            "hardware_digest": sha256_json(hardware),
            "job_digest": server_job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "attempt_root": str(attempt_root),
        },
        key="execution_evidence_digest",
    )
    vendor = {
        "schema": VENDOR_PROVENANCE_SCHEMA,
        "path": "/frozen/repro_fpo_ppo/_vendor",
        "tree_digest": "7" * 64,
        "file_count": 128,
        "total_bytes": 4096,
        "wandb_version": "0.21.0",
    }
    command = [
        str(runner_path),
        "--attempt-manifest",
        str(attempt_root / "attempt_manifest.json"),
        "--run-dir",
        str(attempt_root),
        "--fpo-root",
        "/frozen/fpo",
        "--vendor-dir",
        vendor["path"],
        "--legacy-policy-io",
        str(legacy_policy_io.resolve()),
        "--execution-purpose",
        execution_purpose,
    ]
    if allow_non_gpu:
        command.append("--allow-non-gpu")
    runtime = {
        "runner_schema": "policy-learnware.v02-anchor-aware-runner.v0",
        "runner_file": str(runner_path),
        "fpo_root": "/frozen/fpo",
        "fpo_commit": package_job.trainer_commit,
        "runtime_contract": anchor["runtime"],
        "runtime_digest": package_job.runtime_digest,
        "vendor": vendor,
        "implementation": implementation,
        "legacy_policy_io_path": str(legacy_policy_io.resolve()),
        "pythonpath_vendor_precedence_verified": True,
        "wandb_mode": "disabled",
        "python_dont_write_bytecode": "1",
        "host": hardware["host"],
        "pid": 123,
        "platform": hardware["platform"],
        "python": "3.11 synthetic",
        "cuda_visible_devices": hardware["cuda_visible_devices"],
        "xla_python_client_preallocate": "false",
        "jax_backend": hardware["jax_backend"],
        "jax_devices": hardware["jax_devices"],
        "hardware_contract": hardware,
        "hardware_digest": sha256_json(hardware),
        "execution_evidence": execution,
        "command": command,
        "started_at": "2026-08-24T00:00:00Z",
    }
    run_manifest = with_self_digest(
        {
            "schema": "policy-learnware.v02-anchor-training-run.v0",
            "job": server_job,
            "job_digest": server_job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "config_digest": server_job["config_digest"],
            "execution_purpose": server_job["execution_purpose"],
            "anchor_manifest": anchor,
            "anchor_manifest_digest": anchor["manifest_digest"],
            "environment_instance_digest": anchor["environment_instance_digest"],
            "model_diff_digest": anchor["model_diff_digest"],
            "binding_audit": {
                "anchor_id": anchor["anchor_id"],
                "environment_instance_digest": anchor["environment_instance_digest"],
                "nominal_model_digest": anchor["expected_nominal_model_digest"],
                "bound_model_digest": anchor["expected_bound_model_digest"],
                "changed_leaves": changed,
                "model_diff_digest": anchor["model_diff_digest"],
                "source_unchanged": True,
                "operator_digest": anchor["operator_digest"],
                "manifest_digest": anchor["manifest_digest"],
            },
            "training_protocol_digest": protocol["protocol_digest"],
            "config": protocol["trainer_config"],
            "num_envs": 8,
            "iterations_per_env": 16,
            "transitions_per_outer": 128,
            "planned_environment_steps": 128,
            "execution_mode": execution["execution_mode"],
            "formal_eligible": execution["formal_eligible"],
            "execution_evidence_digest": execution["execution_evidence_digest"],
            "runtime": runtime,
        },
        key="run_manifest_digest",
    )
    checkpoint_path = attempt_root / "checkpoints" / "outer_000001"
    checkpoint = _bundle(
        checkpoint_path,
        package_job=package_job,
        server_job=server_job,
        attempt=attempt,
        anchor=anchor,
        execution=execution,
    )
    record = with_self_digest(
        {
            "schema": TRAINING_RECORD_SCHEMA,
            "state": "succeeded",
            "config_digest": server_job["config_digest"],
            "execution_purpose": server_job["execution_purpose"],
            "job_digest": server_job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor["manifest_digest"],
            "environment_instance_digest": anchor["environment_instance_digest"],
            "training_protocol_digest": protocol["protocol_digest"],
            "algorithm": "ppo",
            "seed": package_job.seed,
            "execution_mode": execution["execution_mode"],
            "formal_eligible": execution["formal_eligible"],
            "implementation": implementation,
            "execution_evidence_digest": execution["execution_evidence_digest"],
            "checkpoint_bundles": [checkpoint],
            "started_at": "2026-08-24T00:00:01Z",
            "finished_at": "2026-08-24T00:00:02Z",
            "wall_seconds": 1.0,
        },
        key="record_digest",
    )
    atomic_write_json(attempt_root / "attempt_manifest.json", attempt, overwrite=False)
    atomic_write_json(attempt_root / "run_manifest.json", run_manifest, overwrite=False)
    atomic_write_json(attempt_root / "training_record.json", record, overwrite=False)
    atomic_write_json(attempt_root.parent / "job_manifest.json", server_job, overwrite=False)
    queue_result = with_self_digest(
        {
            "schema": QUEUE_RESULT_SCHEMA,
            "state": "succeeded",
            "job_digest": server_job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "gpu": "0",
            "config_digest": server_job["config_digest"],
            "execution_purpose": server_job["execution_purpose"],
            "execution_mode": execution["execution_mode"],
            "formal_eligible": execution["formal_eligible"],
            "pid": 123,
            "returncode": 0,
            "started_at": "2026-08-24T00:00:00Z",
            "finished_at": "2026-08-24T00:00:03Z",
            "elapsed_seconds": 3.0,
            "validation_error": None,
            "command": command,
            "vendor": vendor,
            "implementation": implementation,
        },
        key="result_digest",
    )
    atomic_write_json(attempt_root / "queue_result.json", queue_result, overwrite=False)
    return {
        "package_job": package_job,
        "package_jobs": (package_job,),
        "binding_manifest": binding,
        "server_plan": server_plan,
        "attempt_manifest": attempt,
        "anchor_manifest": anchor,
        "run_manifest": run_manifest,
        "training_record": record,
        "checkpoint_paths": {1: checkpoint_path},
    }


def _redigest(value: dict[str, Any], key: str) -> dict[str, Any]:
    material = {name: item for name, item in value.items() if name != key}
    return {**material, key: sha256_json(material)}


def test_projection_and_success_bridge_bind_every_provenance_layer(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    package_job = evidence["package_job"]
    server_job = evidence["server_plan"]["jobs"][0]
    assert server_job["job_digest"] != package_job.digest

    attestation = attestation_from_server_success(**evidence)
    assert attestation.is_server_bound
    assert attestation.server_plan_binding_digest == evidence["binding_manifest"]["binding_digest"]
    assert attestation.server_training_plan_digest == evidence["server_plan"]["plan_digest"]
    assert attestation.server_job_digest == server_job["job_digest"]
    assert attestation.server_attempt_digest == evidence["attempt_manifest"]["attempt_digest"]
    assert attestation.server_run_manifest_digest == evidence["run_manifest"]["run_manifest_digest"]
    assert attestation.server_training_record_digest == evidence["training_record"]["record_digest"]
    assert attestation.hardware_digest == evidence["run_manifest"]["runtime"]["hardware_digest"]
    assert attestation.model_diff_digest == evidence["anchor_manifest"]["model_diff_digest"]
    assert admit_server_success(**evidence).attestation == attestation


def test_formal_projection_requires_live_canonical_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor_path = tmp_path / "anchor.json"
    anchor = _anchor(anchor_path)
    protocol = _protocol()
    job = _package_job(anchor, protocol)
    monkeypatch.setattr(
        package_bridge_module,
        "revalidate_formal_freeze_binding",
        _REAL_REVALIDATE_FORMAL_FREEZE,
    )
    monkeypatch.setattr(
        package_bridge_module,
        "validate_formal_training_projection",
        _REAL_VALIDATE_FORMAL_PROJECTION,
    )
    with pytest.raises(ContractError, match="formal config path"):
        project_policy_training_plan(
            (job,),
            anchor_manifest_paths={job.source_anchor_id: anchor_path},
            training_protocols={job.training_protocol_id: protocol},
            formal_protocol_freeze=_formal_binding(tmp_path, job.source_anchor_id),
        )


def test_formal_projection_rejects_single_job_against_30_by_3_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor_path = tmp_path / "anchor.json"
    anchor = _anchor(anchor_path)
    protocol = _protocol()
    job = _package_job(anchor, protocol)
    initial = _formal_binding(tmp_path, job.source_anchor_id)
    semantics = [dict(row) for row in initial["training_contract"]["source_anchors"]]
    row = next(item for item in semantics if item["source_anchor_id"] == anchor["anchor_id"])
    row.update(
        {
            "task": anchor["task"],
            "nominal": False,
            "factor": anchor["factor"],
            "factor_id": "shifted",
            "axis_id": anchor["operator"]["axis_id"],
            "operator_id": anchor["operator"]["operator_id"],
            "axis_binding_digest": anchor["axis_binding_digest"],
            "leaf_allowlist": sorted(
                mutation["leaf"] for mutation in anchor["operator"]["mutations"]
            ),
        }
    )
    binding = make_formal_freeze_binding(
        tmp_path,
        anchor_ids=list(initial["training_contract"]["source_anchor_ids"]),
        anchor_semantics=semantics,
        seeds=[7, 8, 9],
        config_digest="0" * 64,
        algorithm="ppo",
        training_steps=128,
        checkpoint_rule="fixed_final",
    )
    monkeypatch.setattr(
        package_bridge_module,
        "validate_formal_training_projection",
        _REAL_VALIDATE_FORMAL_PROJECTION,
    )
    with pytest.raises(ContractError, match="grid differs"):
        project_policy_training_plan(
            (job,),
            anchor_manifest_paths={job.source_anchor_id: anchor_path},
            training_protocols={job.training_protocol_id: protocol},
            formal_protocol_freeze=binding,
        )


def test_single_and_batch_admission_reenter_canonical_formal_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path)

    def reject_synthetic(_binding: object) -> dict:
        raise ContractError("sentinel canonical freeze rejection")

    monkeypatch.setattr(
        package_bridge_module,
        "revalidate_formal_freeze_binding",
        reject_synthetic,
    )
    with pytest.raises(ContractError, match="sentinel canonical"):
        attestation_from_server_success(**evidence)

    row = {
        key: evidence[key]
        for key in (
            "attempt_manifest",
            "anchor_manifest",
            "run_manifest",
            "training_record",
            "checkpoint_paths",
        )
    }
    with pytest.raises(ContractError, match="sentinel canonical"):
        admit_server_success_batch(
            package_jobs=evidence["package_jobs"],
            binding_manifest=evidence["binding_manifest"],
            server_plan=evidence["server_plan"],
            evidence_by_job={evidence["package_job"].job_id: row},
        )


def test_bridge_derives_upstream_computed_iterations_from_real_config_shape() -> None:
    config = _protocol()["trainer_config"]
    assert "iterations_per_env" not in config
    assert derive_iterations_per_env(config) == 16

    nondivisible = {**config, "num_envs": 7}
    with pytest.raises(ContractError, match="divide exactly"):
        derive_iterations_per_env(nondivisible)
    nonpositive = {**config, "num_minibatches": 0}
    with pytest.raises(ContractError, match="positive integer"):
        derive_iterations_per_env(nonpositive)


def test_nominal_anchor_uses_null_operator_without_losing_binding(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, nominal=True)
    attestation = attestation_from_server_success(**evidence)
    assert attestation.operator_digest is None
    assert attestation.is_server_bound


def test_projection_rejects_budget_or_runtime_guessing(tmp_path: Path) -> None:
    anchor_path = tmp_path / "anchor.json"
    anchor = _anchor(anchor_path)
    protocol = _protocol()
    job = _package_job(anchor, protocol)
    wrong_budget = PolicyTrainingJob(
        **{**job.__dict__, "environment_steps": 129}
    )
    with pytest.raises(ContractError, match="environment_steps"):
        project_policy_training_plan(
            (wrong_budget,),
            anchor_manifest_paths={wrong_budget.source_anchor_id: anchor_path},
            training_protocols={wrong_budget.training_protocol_id: protocol},
            formal_protocol_freeze=_formal_binding(
                tmp_path, wrong_budget.source_anchor_id
            ),
        )


def test_dependency_digest_is_bound_even_though_server_job_schema_omits_it(
    tmp_path: Path,
) -> None:
    anchor_path = tmp_path / "anchor.json"
    anchor = _anchor(anchor_path)
    protocol = _protocol()
    first = _package_job(anchor, protocol)
    second = replace(first, dependency_digest="e" * 64)
    first_plan, first_binding = project_policy_training_plan(
        (first,),
        anchor_manifest_paths={first.source_anchor_id: anchor_path},
        training_protocols={first.training_protocol_id: protocol},
        formal_protocol_freeze=_formal_binding(tmp_path, first.source_anchor_id),
    )
    second_plan, second_binding = project_policy_training_plan(
        (second,),
        anchor_manifest_paths={second.source_anchor_id: anchor_path},
        training_protocols={second.training_protocol_id: protocol},
        formal_protocol_freeze=_formal_binding(tmp_path, second.source_anchor_id),
    )
    assert first_plan == second_plan
    assert first_binding["binding_digest"] != second_binding["binding_digest"]


def test_batch_admission_requires_exact_plan_coverage(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    row = {
        key: evidence[key]
        for key in (
            "attempt_manifest",
            "anchor_manifest",
            "run_manifest",
            "training_record",
            "checkpoint_paths",
        )
    }
    admitted = admit_server_success_batch(
        package_jobs=evidence["package_jobs"],
        binding_manifest=evidence["binding_manifest"],
        server_plan=evidence["server_plan"],
        evidence_by_job={evidence["package_job"].job_id: row},
    )
    assert tuple(admitted) == (evidence["package_job"].job_id,)
    with pytest.raises(ContractError, match="coverage"):
        admit_server_success_batch(
            package_jobs=evidence["package_jobs"],
            binding_manifest=evidence["binding_manifest"],
            server_plan=evidence["server_plan"],
            evidence_by_job={},
        )


def test_bridge_rejects_rehashed_cross_object_poisoning(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    record = dict(evidence["training_record"])
    record["environment_instance_digest"] = "f" * 64
    evidence["training_record"] = _redigest(record, "record_digest")
    with pytest.raises(ContractError, match="drifted from job"):
        attestation_from_server_success(**evidence)

    evidence = _evidence(tmp_path / "model-diff")
    run = dict(evidence["run_manifest"])
    run["model_diff_digest"] = "f" * 64
    evidence["run_manifest"] = _redigest(run, "run_manifest_digest")
    with pytest.raises(ContractError, match="run manifest drifted"):
        attestation_from_server_success(**evidence)

    evidence = _evidence(tmp_path / "geometry")
    run = dict(evidence["run_manifest"])
    run["num_envs"] = 16
    run["iterations_per_env"] = 8
    evidence["run_manifest"] = _redigest(run, "run_manifest_digest")
    with pytest.raises(ContractError, match="geometry drifted from native config"):
        attestation_from_server_success(**evidence)


def test_bridge_rejects_missing_explicit_checkpoint_or_parity_failure(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence["checkpoint_paths"] = {}
    with pytest.raises(ContractError, match="path coverage"):
        attestation_from_server_success(**evidence)

    evidence = _evidence(tmp_path / "second")
    record = dict(evidence["training_record"])
    checkpoint = dict(record["checkpoint_bundles"][0])
    golden = dict(checkpoint["golden_parity"])
    golden["passed"] = False
    checkpoint["golden_parity"] = _redigest(golden, "report_digest")
    record["checkpoint_bundles"] = [checkpoint]
    evidence["training_record"] = _redigest(record, "record_digest")
    with pytest.raises(ContractError, match="golden_parity did not pass"):
        attestation_from_server_success(**evidence)


def test_bridge_rejects_hardware_and_bundle_path_drift(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    run = dict(evidence["run_manifest"])
    runtime = dict(run["runtime"])
    runtime["hardware_digest"] = "0" * 64
    run["runtime"] = runtime
    evidence["run_manifest"] = _redigest(run, "run_manifest_digest")
    with pytest.raises(ContractError, match="hardware_digest mismatch"):
        attestation_from_server_success(**evidence)

    evidence = _evidence(tmp_path / "second")
    evidence["checkpoint_paths"] = {1: tmp_path / "wrong-bundle"}
    with pytest.raises(ContractError, match="explicit checkpoint"):
        attestation_from_server_success(**evidence)


def test_formal_bridge_rejects_audit_smoke_and_cpu_evidence(tmp_path: Path) -> None:
    audit = _evidence(
        tmp_path / "audit",
        execution_mode=AUDIT_SMOKE_EXECUTION_MODE,
        execution_purpose="audit_smoke",
        jax_backend="cpu",
    )
    with pytest.raises(ContractError, match="audit-smoke/development attempt"):
        admit_server_success(**audit)
    batch_row = {
        key: audit[key]
        for key in (
            "attempt_manifest",
            "anchor_manifest",
            "run_manifest",
            "training_record",
            "checkpoint_paths",
        )
    }
    with pytest.raises(ContractError, match="audit-smoke/development attempt"):
        admit_server_success_batch(
            package_jobs=audit["package_jobs"],
            binding_manifest=audit["binding_manifest"],
            server_plan=audit["server_plan"],
            evidence_by_job={audit["package_job"].job_id: batch_row},
        )

    cpu = _evidence(tmp_path / "cpu", jax_backend="cpu")
    with pytest.raises(ContractError, match="GPU execution evidence requires the GPU"):
        admit_server_success(**cpu)


def test_formal_bridge_rejects_development_purpose_even_on_gpu(tmp_path: Path) -> None:
    development = _evidence(
        tmp_path,
        execution_purpose="development_discovery",
        jax_backend="gpu",
    )
    assert development["attempt_manifest"]["formal_eligible"] is False
    with pytest.raises(ContractError, match="audit-smoke/development attempt"):
        admit_server_success(**development)


def test_formal_bridge_rejects_noncanonical_evidence_root(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, canonical_root=False)
    with pytest.raises(ContractError, match="canonical queue attempt"):
        admit_server_success(**evidence)


def test_formal_bridge_rejects_rehashed_debug_flag_poisoning(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    run = dict(evidence["run_manifest"])
    runtime = dict(run["runtime"])
    runtime["command"] = [*runtime["command"], "--allow-non-gpu"]
    run["runtime"] = runtime
    evidence["run_manifest"] = _redigest(run, "run_manifest_digest")
    with pytest.raises(ContractError, match="debug runner flag"):
        admit_server_success(**evidence)
