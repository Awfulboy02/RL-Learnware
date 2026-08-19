from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.v01 import cli
from policy_learnware_v0.v01.artifacts import V01ArtifactLayout
from policy_learnware_v0.v01.audit import assert_measurement_schema_allowlist
from policy_learnware_v0.v01.config import (
    V01ExperimentConfig,
    load_v01_experiment_config,
)
from policy_learnware_v0.v01.execution_profile import (
    ExecutionProfileError,
    verify_any_successful_execution_attempt,
    verify_execution_attempt,
)
from policy_learnware_v0.v01.plans import build_pair_plan
from policy_learnware_v0.v01.recompute import (
    ExecutableEvidence,
    RecomputeContractError,
    compute_oracle_poison_evidence,
)
from policy_learnware_v0.v01.schemas import (
    MeasurementSchemaView,
    VariantDatasetManifest,
)
from policy_learnware_v0.v01.taskspec import (
    WeightedSemanticSample,
    taskspec_primitive_digest,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _approved_smoke_manifest() -> dict[str, object]:
    config = load_v01_experiment_config(
        Path(__file__).resolve().parents[2] / "configs" / "v01_smoke.yaml"
    )
    return {
        "formal": False,
        "config": config.to_dict(),
        "config_digest": config.config_digest,
    }


def _subcommands():
    parser = cli.build_parser()
    command = next(action for action in parser._actions if action.dest == "command")
    return parser, command.choices


def _actions(parser):
    return {action.dest: action for action in parser._actions}


def _dataset(offset: float, seed: int) -> EpisodeDataset:
    return EpisodeDataset(
        observation=np.asarray([[offset], [offset + 1.0]], dtype=np.float32),
        action=np.asarray([[0.1], [0.2]], dtype=np.float32),
        reward=np.asarray([0.0, 1.0], dtype=np.float32),
        next_observation=np.asarray(
            [[offset + 0.5], [offset + 1.5]], dtype=np.float32
        ),
        terminated=np.asarray([True, True]),
        truncated=np.asarray([False, False]),
        episode_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        reset_seeds=np.asarray([seed, seed + 1], dtype=np.int64),
        probe_seeds=np.asarray([seed + 10, seed + 11], dtype=np.int64),
    )


def _publish_minimal_taskspec_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[V01ArtifactLayout, dict[str, object]]:
    layout = V01ArtifactLayout(tmp_path / "artifacts", "synthetic-v01")
    measurement = layout.measurement_dir
    measurement.mkdir(parents=True)
    variant_id = "v01v-00000000000000000001"
    base_protocol_id = "b" * 64
    base_binding_digest = "d" * 64
    view = MeasurementSchemaView(
        observation_dim=1,
        action_dim=1,
        observation_dtype="float32",
        action_dtype="float32",
        action_low=np.asarray([-1.0], dtype=np.float32),
        action_high=np.asarray([1.0], dtype=np.float32),
        horizon=2,
        action_repeat=1,
        control_dt=0.02,
        flatten_fingerprint_without_task="opaque-fingerprint",
    )
    pair_plan = build_pair_plan(
        [{"task": "hidden", "factor": 1.0, "variant_id": variant_id}],
        banks=2,
        gate_prefix=1,
        routing_prefix=2,
        within_bank_pairs=((0, 1),),
    )
    contract = {
        "schema": "policy-learnware.v01-measurement-contract.v0",
        "measurement_protocol_id": "a" * 64,
        "base_protocol_id": base_protocol_id,
        "probe_banks": 2,
        "episodes_per_bank": 2,
        "prefix_grid": [1, 2],
        "gate_prefix": 1,
        "pair_plan_digest": pair_plan["plan_digest"],
        "variant_ids": [variant_id],
        "schema_view_digests": {variant_id: view.digest},
        "visibility": "opaque-variant-only-no-context-policy-or-outcome",
    }
    _write_json(layout.measurement_contract, contract)
    _write_json(layout.pair_plan, pair_plan)
    _write_json(layout.schema_view(view.schema_view_id), view.to_dict())
    run_ref = {
        "schema": "policy-learnware.v01-measurement-run-ref.v0",
        "measurement_protocol_id": "a" * 64,
        "measurement_run_id": "e" * 64,
        "measurement_protocol_sha256": "f" * 64,
        "base_protocol_ref": {
            "pool_id": "base-pool",
            "protocol_id": base_protocol_id,
            "protocol_draft_hash": "c" * 64,
            "binding_digest": base_binding_digest,
        },
        "measurement_contract_digest": sha256_file(layout.measurement_contract),
        "pair_plan_digest": pair_plan["plan_digest"],
        "schema_view_digests": {variant_id: view.digest},
        "formal": False,
        "git": {
            "commit": "0" * 40,
            "clean": False,
            "porcelain_sha256": "0" * 64,
        },
        "runtime_versions": cli._runtime_versions(),
        "measurement_component_digests": {
            "base_binding": base_binding_digest,
            "base_assets": sha256_json(
                {
                    "normalization": "1" * 64,
                    "encoder_checkpoint": "2" * 64,
                    "encoder_config": "3" * 64,
                }
            ),
            **cli._component_source_digests()[0],
        },
    }
    _write_json(layout.measurement_run_ref, run_ref)
    _write_json(
        layout.base_protocol_ref,
        {
            "pool_id": "base-pool",
            "protocol_id": base_protocol_id,
            "protocol_draft_hash": "c" * 64,
            "binding_digest": base_binding_digest,
        },
    )
    for bank in range(2):
        dataset = _dataset(float(bank), 100 * (bank + 1))
        data_path = layout.dataset_npz(variant_id, bank)
        dataset.save_npz(data_path)
        sidecar = VariantDatasetManifest(
            variant_id=variant_id,
            bank=bank,
            episode_count=dataset.episode_count,
            transition_count=dataset.transition_count,
            reset_seeds=tuple(int(value) for value in dataset.reset_seeds),
            probe_seeds=tuple(int(value) for value in dataset.probe_seeds),
            dataset_digest=dataset.digest,
            base_protocol_id=base_protocol_id,
            measurement_contract_digest=sha256_file(layout.measurement_contract),
            measurement_schema_view_digest=view.digest,
        )
        _write_json(layout.dataset_manifest(variant_id, bank), sidecar.to_dict())

    task_spec = SimpleNamespace(
        supports=np.asarray([[0.0]], dtype=np.float64),
        beta=np.asarray([1.0], dtype=np.float64),
        kernel_bandwidth=1.0,
        rkme_norm2=1.0,
        protocol_id=base_protocol_id,
    )
    base = SimpleNamespace(
        binding_digest=base_binding_digest,
        protocol_id=base_protocol_id,
        asset_digests={
            "normalization": "1" * 64,
            "encoder_checkpoint": "2" * 64,
            "encoder_config": "3" * 64,
        },
        protocol=SimpleNamespace(packed_layout={"max_action_dim": 1}),
        public_pool=SimpleNamespace(
            entries=(SimpleNamespace(opaque_id="source-opaque", task_spec=task_spec),)
        ),
        load_measurement_assets=lambda: SimpleNamespace(
            normalization=object(), encoder=object(), kernel=GaussianKernel(1.0)
        ),
    )

    def encode(dataset, _view, **_kwargs):
        return WeightedSemanticSample.from_points(
            np.asarray(dataset.observation, dtype=np.float64),
            dataset.episode_offsets,
        )

    monkeypatch.setattr(cli, "verify_and_load_base_runtime", lambda *_a, **_k: base)
    import policy_learnware_v0.v01.taskspec as taskspec_module

    monkeypatch.setattr(taskspec_module, "encode_measurement_dataset", encode)
    result = cli._compute_taskspec(
        argparse.Namespace(
            base_artifacts_root=tmp_path / "base",
            measurement_root=measurement,
            block_size=8,
            computation_backend="numpy",
            resume=False,
        )
    )
    return layout, result


def test_compute_taskspec_publishes_independent_primitive_manifest_and_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, result = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    axes = json.loads(layout.taskspec_matrix_axes.read_text(encoding="utf-8"))
    plan = json.loads(layout.pair_plan.read_text(encoding="utf-8"))
    primitive = json.loads(
        layout.taskspec_primitive_manifest.read_text(encoding="utf-8")
    )
    assert set(primitive) == {
        "schema",
        "plan_digest",
        "primitive_digest",
        "taskspec_matrix_sha256",
        "semantic_manifest_sha256",
        "semantic_content_digest",
    }
    assert primitive["schema"] == (
        "policy-learnware.v01-taskspec-primitive-manifest.v0"
    )
    assert primitive["primitive_digest"] == taskspec_primitive_digest(plan, axes)
    assert primitive["primitive_digest"] == result["taskspec_primitive_digest"]
    assert primitive["taskspec_matrix_sha256"] == sha256_file(
        layout.taskspec_matrix_axes
    )
    assert set(primitive["semantic_manifest_sha256"]) == {
        "v01v-00000000000000000001/bank_000",
        "v01v-00000000000000000001/bank_001",
    }
    assert assert_measurement_schema_allowlist(layout.measurement_dir)["passed"]

    primitive["passed"] = True
    _write_json(layout.taskspec_primitive_manifest, primitive)
    allowlist = assert_measurement_schema_allowlist(layout.measurement_dir)
    assert not allowlist["passed"]
    violation = next(
        item
        for item in allowlist["violations"]
        if item["path"] == "taskspec_primitive_manifest.json"
    )
    assert violation["reason"] == "json_schema_key_mismatch"


def test_taskspec_zero_compute_resume_reuses_verified_success_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, first = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    attempts = sorted((layout.measurement_dir / "execution_attempts").glob("*.json"))
    assert len(attempts) == 1

    import policy_learnware_v0.v01.taskspec as taskspec_module

    def forbidden_compute(*_args, **_kwargs):
        raise AssertionError("resume must not run Gaussian matrix compute")

    monkeypatch.setattr(taskspec_module, "compute_taskspec_matrix", forbidden_compute)
    resumed = cli._compute_taskspec(
        argparse.Namespace(
            base_artifacts_root=tmp_path / "base",
            measurement_root=layout.measurement_dir,
            block_size=1,
            computation_backend="numpy",
            resume=True,
        )
    )
    assert resumed["resumed"] is True
    assert resumed["execution_attempt_id"] == first["execution_attempt_id"]
    assert sorted((layout.measurement_dir / "execution_attempts").glob("*.json")) == attempts


def test_failed_attempt_cannot_authorize_existing_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    for path in (layout.measurement_dir / "execution_attempts").glob("*.json"):
        path.unlink()

    def fail_compute(_args):
        raise MemoryError("private path and details must not be persisted")

    monkeypatch.setattr(cli, "_compute_taskspec_impl", fail_compute)
    with pytest.raises(MemoryError):
        cli._compute_taskspec(
            argparse.Namespace(
                base_artifacts_root=tmp_path / "base",
                measurement_root=layout.measurement_dir,
                block_size=4,
                computation_backend="numpy",
                resume=False,
            )
        )
    attempts = sorted((layout.measurement_dir / "execution_attempts").glob("*.json"))
    assert len(attempts) == 1
    payload = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["failure"] == {
        "error_type": "MemoryFailure",
        "reason_code": "OUT_OF_MEMORY",
    }
    assert "private path" not in attempts[0].read_text(encoding="utf-8")
    with pytest.raises(ExecutionProfileError, match="no verified successful"):
        verify_any_successful_execution_attempt(
            layout.measurement_dir, source_support_sizes=(1,)
        )
    with pytest.raises(ExecutionProfileError, match="no verified successful"):
        cli._verified_measurement_aggregation(
            layout, source_support_sizes=(1,)
        )


def test_execution_attempt_tamper_fails_allowlist_and_live_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, result = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    path = layout.execution_attempt(result["execution_attempt_id"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution"]["block_size"] += 1
    _write_json(path, payload)
    with pytest.raises(
        ExecutionProfileError,
        match="resource extrapolation|self digest",
    ):
        verify_execution_attempt(
            layout.measurement_dir,
            result["execution_attempt_id"],
            require_success=True,
            source_support_sizes=(1,),
        )
    audit = assert_measurement_schema_allowlist(layout.measurement_dir)
    assert not audit["passed"]
    assert any(
        item["path"].startswith("execution_attempts/")
        and item["reason"] == "execution_attempt_schema_mismatch"
        for item in audit["violations"]
    )


def test_self_signed_resource_and_live_workload_tamper_still_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, result = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    path = layout.execution_attempt(result["execution_attempt_id"])
    original = json.loads(path.read_text(encoding="utf-8"))

    resource_tamper = json.loads(json.dumps(original))
    resource_tamper["resource_profile"]["wall_time_seconds"] *= 0.5
    resource_tamper["attempt_digest"] = sha256_json(
        {key: value for key, value in resource_tamper.items() if key != "attempt_digest"}
    )
    _write_json(path, resource_tamper)
    with pytest.raises(ExecutionProfileError, match="extrapolation is not derivable"):
        verify_execution_attempt(
            layout.measurement_dir,
            result["execution_attempt_id"],
            require_success=True,
            source_support_sizes=(1,),
        )

    workload_tamper = json.loads(json.dumps(original))
    workload_tamper["workload"]["mathematical_pair_cross_kernel_entries"] += 1
    workload_tamper["workload"]["mathematical_total_kernel_entries"] += 1
    workload_tamper["attempt_digest"] = sha256_json(
        {key: value for key, value in workload_tamper.items() if key != "attempt_digest"}
    )
    _write_json(path, workload_tamper)
    with pytest.raises(ExecutionProfileError, match="live plan/caches"):
        verify_execution_attempt(
            layout.measurement_dir,
            result["execution_attempt_id"],
            require_success=True,
            source_support_sizes=(1,),
        )


def test_runtime_device_and_attempt_path_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, result = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    path = layout.execution_attempt(result["execution_attempt_id"])
    original = json.loads(path.read_text(encoding="utf-8"))

    runtime_tamper = json.loads(json.dumps(original))
    runtime_tamper["execution"]["runtime_versions"]["fake"] = "1.0"
    runtime_tamper["attempt_digest"] = sha256_json(
        {key: value for key, value in runtime_tamper.items() if key != "attempt_digest"}
    )
    _write_json(path, runtime_tamper)
    with pytest.raises(ExecutionProfileError, match="frozen/current runtime"):
        verify_execution_attempt(
            layout.measurement_dir,
            result["execution_attempt_id"],
            require_success=True,
            source_support_sizes=(1,),
        )

    device_tamper = json.loads(json.dumps(original))
    device_tamper["execution"]["device"]["device_kinds"] = ["/private/secret"]
    device_tamper["attempt_digest"] = sha256_json(
        {key: value for key, value in device_tamper.items() if key != "attempt_digest"}
    )
    _write_json(path, device_tamper)
    with pytest.raises(ExecutionProfileError, match="device device_kinds is malformed"):
        verify_execution_attempt(
            layout.measurement_dir,
            result["execution_attempt_id"],
            require_success=True,
            source_support_sizes=(1,),
        )

    outside = tmp_path / "outside-attempt.json"
    _write_json(outside, original)
    path.unlink()
    path.symlink_to(outside)
    allowlist = assert_measurement_schema_allowlist(layout.measurement_dir)
    assert not allowlist["passed"]
    assert any(
        item["reason"] == "artifact_path_escapes_measurement"
        for item in allowlist["violations"]
    )
    with pytest.raises(ExecutionProfileError, match="escapes the measurement root"):
        verify_execution_attempt(
            layout.measurement_dir,
            result["execution_attempt_id"],
            require_success=True,
            source_support_sizes=(1,),
        )


def _resume_taskspec_args(layout: V01ArtifactLayout, tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        base_artifacts_root=tmp_path / "base",
        measurement_root=layout.measurement_dir,
        block_size=8,
        computation_backend="numpy",
        resume=True,
    )


def _forbid_taskspec_numeric_recomputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import policy_learnware_v0.v01.taskspec as taskspec_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete TaskSpec resume must not run encoder/kernel work")

    base = cli.verify_and_load_base_runtime(None)
    monkeypatch.setattr(base, "load_measurement_assets", forbidden)
    monkeypatch.setattr(taskspec_module, "encode_measurement_dataset", forbidden)
    monkeypatch.setattr(taskspec_module, "compute_taskspec_matrix", forbidden)


def test_compute_taskspec_resume_strictly_revalidates_without_numeric_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, first = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    _forbid_taskspec_numeric_recomputation(monkeypatch)

    resumed = cli._compute_taskspec(_resume_taskspec_args(layout, tmp_path))

    assert resumed["resumed"] is True
    assert first["resumed"] is False
    assert {key: value for key, value in resumed.items() if key != "resumed"} == {
        key: value for key, value in first.items() if key != "resumed"
    }


def test_compute_taskspec_resume_rejects_partial_output_before_numeric_recompute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    layout.routing_matrix_csv.unlink()
    _forbid_taskspec_numeric_recomputation(monkeypatch)

    with pytest.raises(cli.V01CommandFailure, match="incomplete TaskSpec output bundle"):
        cli._compute_taskspec(_resume_taskspec_args(layout, tmp_path))


@pytest.mark.parametrize(
    "artifact_name",
    ("taskspec_matrix_npz", "taskspec_matrix_csv", "routing_matrix_csv"),
)
def test_compute_taskspec_resume_rejects_redundant_output_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    path = getattr(layout, artifact_name)
    path.write_bytes(path.read_bytes() + b"poison")
    _forbid_taskspec_numeric_recomputation(monkeypatch)

    with pytest.raises(Exception, match="resume content mismatch|Failed to interpret"):
        cli._compute_taskspec(_resume_taskspec_args(layout, tmp_path))


def test_compute_taskspec_resume_revalidates_raw_dataset_sidecars_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    variant_id = "v01v-00000000000000000001"
    path = layout.dataset_npz(variant_id, 0)
    dataset = EpisodeDataset.load_npz(path)
    arrays = dataset.to_arrays(copy=True)
    arrays["reward"][0] += 1.0
    EpisodeDataset(**arrays).save_npz(path, overwrite=True)
    _forbid_taskspec_numeric_recomputation(monkeypatch)

    with pytest.raises(RecomputeContractError, match="dataset content digest mismatch"):
        cli._compute_taskspec(_resume_taskspec_args(layout, tmp_path))


def test_verified_measurement_aggregation_uses_separate_primitive_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    chain, aggregation, primitive = cli._verified_measurement_aggregation(
        layout, source_support_sizes=(1,)
    )
    axes = json.loads(layout.taskspec_matrix_axes.read_text(encoding="utf-8"))
    assert chain.passed
    assert aggregation.passed
    assert aggregation.output_digest == sha256_json(axes)
    assert primitive["primitive_digest"] == aggregation.primitive_digest

    primitive = dict(primitive)
    primitive["primitive_digest"] = "0" * 64
    _write_json(layout.taskspec_primitive_manifest, primitive)
    with pytest.raises(RecomputeContractError, match="primitive digest binding"):
        cli._verified_measurement_aggregation(
            layout, source_support_sizes=(1,)
        )


def test_compute_taskspec_rejects_stale_public_measurement_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    run_ref = json.loads(layout.measurement_run_ref.read_text(encoding="utf-8"))
    run_ref["measurement_component_digests"]["taskspec_source"] = "0" * 64
    _write_json(layout.measurement_run_ref, run_ref)

    with pytest.raises(cli.V01CommandFailure, match="source digest mismatch"):
        cli._compute_taskspec(
            argparse.Namespace(
                base_artifacts_root=tmp_path / "base",
                measurement_root=layout.measurement_dir,
                block_size=8,
                computation_backend="numpy",
                resume=True,
            )
        )


def test_oracle_poison_runs_production_zero_compute_resume_in_isolated_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, first = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    _forbid_taskspec_numeric_recomputation(monkeypatch)
    calls: list[Path] = []

    def runner(measurement_root: Path):
        calls.append(measurement_root)
        return cli._compute_taskspec(
            argparse.Namespace(
                base_artifacts_root=tmp_path / "base",
                measurement_root=measurement_root,
                block_size=8,
                computation_backend="numpy",
                resume=True,
            )
        )

    evidence = compute_oracle_poison_evidence(
        runner,
        measurement_root=layout.measurement_dir,
        source_support_sizes=(1,),
    )
    assert evidence.passed, evidence.details
    assert len(calls) == 2
    assert calls[0] != layout.measurement_dir
    assert all(item["strict_zero_compute_resume"] for item in evidence.details["scenarios"].values())
    assert len(set(evidence.details["digests"].values())) == 1
    baseline = evidence.details["baseline_release"]
    assert baseline["success_attempt_id"] == first["execution_attempt_id"]
    assert baseline["success_attempt_digest"] == first["execution_attempt_digest"]
    assert baseline["success_attempt_sha256"] == first["execution_attempt_sha256"]
    assert set(baseline["output_artifact_sha256"]) == {
        "routing_matrix.csv",
        "taskspec_matrix.csv",
        "taskspec_matrix.npz",
        "taskspec_matrix_axes.json",
        "taskspec_primitive_manifest.json",
    }
    assert evidence.details["oracle_root_passed_to_runner"] is False


def test_build_report_blocks_smoke_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V01ArtifactLayout(tmp_path / "artifacts", "smoke-run")
    args = argparse.Namespace(
        no_go_compute=False,
        base_artifacts_root=tmp_path / "base",
        block_size=8,
        computation_backend="numpy",
        resume=False,
    )
    monkeypatch.setattr(cli, "_analysis_layout", lambda _args: layout)
    monkeypatch.setattr(
        cli,
        "_frozen_state",
        lambda _layout: (_approved_smoke_manifest(), {}, {}),
    )
    with pytest.raises(cli.V01CommandFailure, match="smoke runs cannot publish"):
        cli._build_report(args)
    assert not layout.completion_manifest.exists()


def test_build_report_no_go_compute_binds_verified_profile_and_only_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    _write_json(layout.run_manifest, {"formal": False, "run": "frozen"})
    gate0_path = layout.benchmark_private_dir / "gate_0_attestation.json"
    gate0 = {"schema": "typed-gate0", "passed": True}
    _write_json(gate0_path, gate0)
    args = argparse.Namespace(
        no_go_compute=True,
        execution_profile_attempt_id=next(
            (layout.measurement_dir / "execution_attempts").glob("*.json")
        ).stem,
        base_artifacts_root=tmp_path / "base",
        block_size=8,
        computation_backend="numpy",
        resume=False,
    )
    monkeypatch.setattr(cli, "_analysis_layout", lambda _args: layout)
    monkeypatch.setattr(
        cli,
        "_frozen_state",
        lambda _layout: (_approved_smoke_manifest(), {}, {}),
    )
    monkeypatch.setattr(cli, "_require_private_gate0", lambda _layout: gate0)
    result = cli._build_report(args)
    payload = json.loads(layout.preflight_completion_manifest.read_text())
    assert result["decision"]["code"] == "NO_GO_COMPUTE"
    assert payload["formal_completion_published"] is False
    assert payload["formal_run"] is False
    profile = payload["execution_profile"]
    assert profile["execution_attempt_id"] == args.execution_profile_attempt_id
    assert profile["measurement_output_digest"]
    assert profile["resource_extrapolation"]["formal_total_padded_block_entries"] > 0
    assert payload["approved_smoke_coverage"]["config_digest"] == (
        _approved_smoke_manifest()["config_digest"]
    )
    assert payload["measurement_audits"]["schema_allowlist"]["passed"] is True
    assert payload["measurement_audits"]["isolation"]["passed"] is True
    assert not layout.completion_manifest.exists()
    assert not layout.analysis_artifact("gate_d.json").exists()


def test_build_report_no_go_compute_fails_closed_without_verified_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, _ = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    _write_json(layout.run_manifest, {"formal": False, "run": "frozen"})
    args = argparse.Namespace(
        no_go_compute=True,
        execution_profile_attempt_id="v01xa-000000000000000000000000",
        base_artifacts_root=tmp_path / "base",
        block_size=8,
        computation_backend="numpy",
        resume=False,
    )
    monkeypatch.setattr(cli, "_analysis_layout", lambda _args: layout)
    monkeypatch.setattr(
        cli,
        "_frozen_state",
        lambda _layout: (_approved_smoke_manifest(), {}, {}),
    )
    with pytest.raises(
        ExecutionProfileError, match="missing or not a regular file|missing or unreadable"
    ):
        cli._build_report(args)
    assert not layout.preflight_completion_manifest.exists()


def test_build_report_no_go_compute_rejects_unapproved_smaller_smoke_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, result = _publish_minimal_taskspec_run(tmp_path, monkeypatch)
    _write_json(layout.run_manifest, {"formal": False, "run": "frozen"})
    raw = dict(_approved_smoke_manifest()["config"])
    raw["probe"] = {
        **raw["probe"],
        "max_episodes_per_bank": 8,
        "prefix_grid": [1, 2, 4, 8],
        "gate_b_unreduced_prefix": 8,
    }
    smaller = V01ExperimentConfig.from_dict(raw)
    manifest = {
        "formal": False,
        "config": smaller.to_dict(),
        "config_digest": smaller.config_digest,
    }
    args = argparse.Namespace(
        no_go_compute=True,
        execution_profile_attempt_id=result["execution_attempt_id"],
        base_artifacts_root=tmp_path / "base",
        block_size=8,
        computation_backend="numpy",
        resume=False,
    )
    monkeypatch.setattr(cli, "_analysis_layout", lambda _args: layout)
    monkeypatch.setattr(cli, "_frozen_state", lambda _layout: (manifest, {}, {}))
    with pytest.raises(cli.V01CommandFailure, match="uniquely approved P5 smoke"):
        cli._build_report(args)
    assert not layout.preflight_completion_manifest.exists()


def test_build_report_no_go_compute_rejects_formal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V01ArtifactLayout(tmp_path / "artifacts", "formal-preflight")
    args = argparse.Namespace(
        no_go_compute=True,
        execution_profile_attempt_id="v01xa-000000000000000000000000",
        base_artifacts_root=tmp_path / "base",
        block_size=8,
        computation_backend="numpy",
        resume=False,
    )
    monkeypatch.setattr(cli, "_analysis_layout", lambda _args: layout)
    monkeypatch.setattr(cli, "_frozen_state", lambda _layout: ({"formal": True}, {}, {}))
    with pytest.raises(cli.V01CommandFailure, match="smoke experiment root"):
        cli._build_report(args)
    assert not layout.preflight_completion_manifest.exists()


def test_build_report_reexecutes_gate_and_recompute_before_trusting_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = V01ArtifactLayout(tmp_path / "artifacts", "formal-run")
    for name in (
        "gate_0.json", "gate_a.json", "gate_b.json", "gate_c_diagnostic.json",
        "gate_d.json", "join_audit.json", "joined_taskspec_context.csv",
        "joined_transfer_context.csv", "recompute_audit.json",
    ):
        path = layout.analysis_artifact(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"passed":true}\n')
    args = argparse.Namespace(
        no_go_compute=False,
        base_artifacts_root=tmp_path / "base",
        block_size=8,
        computation_backend="numpy",
        resume=False,
    )
    monkeypatch.setattr(cli, "_analysis_layout", lambda _args: layout)
    monkeypatch.setattr(cli, "_frozen_state", lambda _layout: ({"formal": True}, {}, {}))
    calls: list[str] = []

    def reject_forged(verification_args: argparse.Namespace) -> dict[str, object]:
        calls.append("gates")
        assert verification_args.resume is True
        raise cli.V01CommandFailure("typed gate revalidation rejected forged report")

    monkeypatch.setattr(cli, "_evaluate_gates", reject_forged)
    monkeypatch.setattr(
        cli,
        "_recompute_audit",
        lambda _args: calls.append("recompute"),
    )
    with pytest.raises(cli.V01CommandFailure, match="typed gate revalidation"):
        cli._build_report(args)
    assert calls == ["gates"]
    assert not layout.completion_manifest.exists()


def test_analysis_cli_requires_recompute_base_and_rejects_pass_attestations() -> None:
    parser, commands = _subcommands()
    recompute_actions = _actions(commands["audit-recompute"])
    assert recompute_actions["base_artifacts_root"].required
    assert recompute_actions["computation_backend"].default == "jax"
    assert recompute_actions["block_size"].default == 2048
    for command_name in ("evaluate-gates", "build-report"):
        actions = _actions(commands[command_name])
        assert actions["base_artifacts_root"].required
        assert actions["computation_backend"].default == "jax"
        assert actions["block_size"].default == 2048

    roots = [
        "--frozen-root", "/r/frozen",
        "--benchmark-private-root", "/r/benchmark_private",
        "--measurement-root", "/r/measurement",
        "--oracle-root", "/r/oracle_private",
        "--analysis-root", "/r/analysis",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["audit-recompute", *roots])

    gate_args = parser.parse_args(
        ["evaluate-gates", *roots, "--base-artifacts-root", "/base"]
    )
    for forbidden in (
        "gate_a_passed",
        "gate_b_passed",
        "gate_d_passed",
        "scientific_passed",
        "scientific_gate_input",
    ):
        assert not hasattr(gate_args, forbidden)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate-gates", *roots, "--base-artifacts-root", "/base",
                "--gate-a-passed", "true",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate-gates", *roots, "--base-artifacts-root", "/base",
                "--scientific-gate-input", "/tmp/pass.json",
            ]
        )


def _evidence(kind: str, passed: bool) -> ExecutableEvidence:
    return ExecutableEvidence(kind, passed, {"computed": True})


def test_cli_gate_d_construction_fails_closed_from_executable_evidence(
    tmp_path: Path,
) -> None:
    layout = V01ArtifactLayout(tmp_path / "artifacts", "gate-d-synthetic")
    _write_json(
        layout.measurement_protocol,
        {
            "config_projection": {
                "measurement_gates": {
                    "leakage": {"forbidden_measurement_fields": ["candidate_id"]}
                }
            }
        },
    )
    aggregation = SimpleNamespace(output_digest="a" * 64)
    capability = _evidence("taskspec_capability_and_source", True)
    visibility = _evidence("measurement_visibility", False)
    protocol = _evidence("protocol_digest_binding", True)
    separation = _evidence("smoke_formal_separation", True)
    poison = _evidence("oracle_poison_independence", True)
    with (
        patch(
            "policy_learnware_v0.v01.recompute.compute_taskspec_capability_evidence",
            return_value=capability,
        ),
        patch(
            "policy_learnware_v0.v01.recompute.compute_measurement_visibility_evidence",
            return_value=visibility,
        ),
        patch(
            "policy_learnware_v0.v01.recompute.compute_protocol_binding_evidence",
            return_value=protocol,
        ),
        patch(
            "policy_learnware_v0.v01.recompute.compute_smoke_formal_separation_evidence",
            return_value=separation,
        ),
        patch(
            "policy_learnware_v0.v01.recompute.compute_oracle_poison_evidence",
            return_value=poison,
        ),
    ):
        report, checks = cli._compute_executable_gate_d(
            layout,
            measurement_chain=SimpleNamespace(passed=True),
            aggregation=aggregation,
            source_support_sizes=(1,),
        )
    assert report["caller_supplied_passed_attestations_consumed"] is False
    assert report["passed"] is False
    assert checks["measurement_artifacts_forbidden_fields_absent"] is False
    assert checks["context_confined_to_private_or_baseline"] is False
    assert checks["visibility_artifacts_untampered"] is False
    assert checks["taskspec_command_has_no_oracle_dependency"] is True
