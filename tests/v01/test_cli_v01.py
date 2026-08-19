from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from policy_learnware_v0.v01.cli import (
    COMMANDS,
    V01CommandFailure,
    _identity_candidates,
    _v0_regression_binding_evidence_valid,
    _run_controlled_v0_regression,
    _v0_regression_backend_probe_passed,
    _v0_regression_test_record_passed,
    build_parser,
    main,
)


PROJECT = Path(__file__).resolve().parents[2]


def _actions(parser):
    return {action.dest: action for action in parser._actions}


def _subcommands():
    parser = build_parser()
    action = next(action for action in parser._actions if action.dest == "command")
    return parser, action.choices


def test_cli_exposes_exact_registered_command_set_and_scoped_roots() -> None:
    _, commands = _subcommands()
    assert tuple(commands) == COMMANDS
    assert "config" in _actions(commands["validate-config"])
    assert "config" in _actions(commands["freeze-run"])
    for name in set(COMMANDS) - {"validate-config", "freeze-run"}:
        assert "config" not in _actions(commands[name])

    taskspec = _actions(commands["compute-taskspec-matrix"])
    assert "base_artifacts_root" in taskspec
    assert "measurement_root" in taskspec
    for forbidden in (
        "artifacts_root",
        "benchmark_private_root",
        "oracle_root",
        "factor",
        "model_path",
        "overwrite",
        "alpha",
        "delta_effect",
        "task",
    ):
        assert forbidden not in taskspec


def test_all_commands_expose_dry_run_and_resume() -> None:
    _, commands = _subcommands()
    for command in commands.values():
        actions = _actions(command)
        assert "dry_run" in actions
        assert "resume" in actions


def test_only_probe_and_oracle_expose_certified_shard_interface() -> None:
    _, commands = _subcommands()
    for name, command in commands.items():
        actions = _actions(command)
        if name in {"collect-probes", "evaluate-oracle"}:
            assert {"devices", "shard_index", "shard_count"} <= set(actions)
        else:
            assert {"devices", "shard_index", "shard_count"}.isdisjoint(actions)


def test_explicit_device_selection_fails_instead_of_being_silently_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "collect-probes",
            "--artifacts-root",
            str(tmp_path),
            "--experiment-id",
            "smoke",
            "--devices",
            "0",
            "--dry-run",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["fail_closed"] is True
    assert "not certified" in payload["message"]


def test_variant_audit_requires_controlled_base_and_policy_roots() -> None:
    _, commands = _subcommands()
    actions = _actions(commands["audit-variants"])
    assert actions["base_artifacts_root"].required
    assert actions["fpo_root"].required
    assert not actions["runs_root"].required
    assert "keep_going" not in actions


def test_identity_candidate_selection_is_exact_fpo_ppo_seed0() -> None:
    def candidate(algorithm: str, seed: int, task: str = "WalkerWalk"):
        return SimpleNamespace(
            algorithm=algorithm, training_seed=seed, task_private=task
        )

    fpo, ppo = _identity_candidates(
        [candidate("ppo", 1), candidate("fpo", 0), candidate("ppo", 0)],
        "WalkerWalk",
    )
    assert fpo.algorithm == "fpo" and fpo.training_seed == 0
    assert ppo.algorithm == "ppo" and ppo.training_seed == 0
    with pytest.raises(V01CommandFailure, match="exactly one"):
        _identity_candidates([candidate("fpo", 0), candidate("ppo", 1)], "WalkerWalk")


def test_controlled_v0_regression_runs_fixed_suite_and_reopens_base() -> None:
    semantic = "a" * 64
    base = SimpleNamespace(
        base_artifacts_root=Path("/base"),
        binding_digest="b" * 64,
        protocol_manifest_sha256="c" * 64,
        pool_manifest_sha256="d" * 64,
        public_pool_manifest_sha256="e" * 64,
        protocol=SimpleNamespace(
            component_digests={"taskspec_semantic_source": semantic}
        ),
    )
    test_record = {
        "schema": "policy-learnware.v01-v0-unittest-result.v0",
        "unit_discovered": 217,
        "integration_discovered": 1,
        "tests_run": 218,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
        "expected_failures": 0,
        "unexpected_successes": 0,
        "successful": True,
    }
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(test_record, sort_keys=True) + "\n",
        stderr="Ran 218 tests in 3.0s\n\nOK (skipped=1)\n",
    )
    backend_probe_record = {
        "schema": "policy-learnware.v01-regression-backend-probe.v0",
        "default_backend": "cpu",
        "device_count": 1,
        "device_platforms": ["cpu"],
    }
    probed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(backend_probe_record, sort_keys=True) + "\n",
        stderr="",
    )
    resume_record = {
        "schema": "policy-learnware.cli-result.v0",
        "status": "ok",
        "command": "build-pool",
        "result": {"resumed": True},
    }
    resumed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=json.dumps(resume_record) + "\n", stderr=""
    )
    base_ref = {
        "pool_id": "pool",
        "protocol_id": "1" * 64,
        "protocol_draft_hash": "2" * 64,
    }
    with (
        patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "7", "JAX_PLATFORMS": "cuda"},
            clear=False,
        ),
        patch(
            "policy_learnware_v0.v01.cli.verify_and_load_base_runtime",
            side_effect=[base, base],
        ) as verify,
        patch(
            "policy_learnware_v0.cli._taskspec_semantic_source_digest",
            return_value=semantic,
        ),
        patch(
            "policy_learnware_v0.v01.cli.subprocess.run",
            side_effect=[probed, completed, resumed],
        ) as run,
    ):
        report, log = _run_controlled_v0_regression(
            base_artifacts_root=Path("/base"), base_ref=base_ref
        )
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
        assert os.environ["JAX_PLATFORMS"] == "cuda"
    assert verify.call_count == 2
    assert "jax.default_backend" in run.call_args_list[0].args[0][-1]
    assert "jax._src.xla_bridge" in run.call_args_list[0].args[0][-1]
    assert "setLevel(logging.CRITICAL)" in run.call_args_list[0].args[0][-1]
    assert run.call_args_list[1].args[0][1] == "-c"
    assert "tests/unit" in run.call_args_list[1].args[0][2]
    assert "tests/integration" in run.call_args_list[1].args[0][2]
    assert "unittest.TextTestRunner" in run.call_args_list[1].args[0][2]
    assert "redirect_stdout(sys.stderr)" in run.call_args_list[1].args[0][2]
    for call in run.call_args_list:
        child_environment = call.kwargs["env"]
        assert child_environment["CUDA_VISIBLE_DEVICES"] == ""
        assert child_environment["JAX_PLATFORMS"] == "cpu"
        assert child_environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "build-pool" in run.call_args_list[2].args[0]
    assert report["passed"] is True
    assert report["passed_test_count"] == 217
    assert report["base_resume_passed"] is True
    assert report["base_resume_json_record"] == resume_record
    assert report["subprocess_environment"] == {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    assert report["backend_probe_record"] == backend_probe_record
    assert report["backend_probe_passed"] is True
    assert report["test_record"] == test_record
    assert report["test_runner_passed"] is True
    assert len(report["base_resume_config_sha256"]) == 64
    assert report["semantic_source_passed"] is True
    assert report["log_sha256"]
    assert completed.stdout in log and resumed.stdout in log
    binding_base_ref = {
        "binding_digest": base.binding_digest,
        "protocol_manifest_sha256": base.protocol_manifest_sha256,
        "pool_manifest_sha256": base.pool_manifest_sha256,
        "public_pool_manifest_sha256": base.public_pool_manifest_sha256,
    }
    assert _v0_regression_binding_evidence_valid(report, binding_base_ref)
    poisoned_command = json.loads(json.dumps(report))
    poisoned_command["backend_probe_command"].append("--forged")
    assert not _v0_regression_binding_evidence_valid(
        poisoned_command, binding_base_ref
    )
    poisoned_record = json.loads(json.dumps(report))
    poisoned_record["backend_probe_record"]["device_count"] = 2
    assert not _v0_regression_binding_evidence_valid(poisoned_record, binding_base_ref)
    poisoned_test_command = json.loads(json.dumps(report))
    poisoned_test_command["command"].append("--forged")
    assert not _v0_regression_binding_evidence_valid(
        poisoned_test_command, binding_base_ref
    )
    poisoned_test_record = json.loads(json.dumps(report))
    poisoned_test_record["test_record"]["tests_run"] = 219
    assert not _v0_regression_binding_evidence_valid(
        poisoned_test_record, binding_base_ref
    )
    poisoned_binding = json.loads(json.dumps(report))
    poisoned_binding["base_resume_digest"] = "0" * 64
    assert not _v0_regression_binding_evidence_valid(
        poisoned_binding, binding_base_ref
    )


def test_v0_regression_backend_probe_rejects_gpu_or_malformed_evidence() -> None:
    cpu = {
        "schema": "policy-learnware.v01-regression-backend-probe.v0",
        "default_backend": "cpu",
        "device_count": 1,
        "device_platforms": ["cpu"],
    }
    assert _v0_regression_backend_probe_passed(cpu, returncode=0)
    assert not _v0_regression_backend_probe_passed(
        {**cpu, "default_backend": "gpu", "device_platforms": ["gpu"]},
        returncode=0,
    )
    assert not _v0_regression_backend_probe_passed(cpu, returncode=1)
    assert not _v0_regression_backend_probe_passed(
        {**cpu, "device_count": 2}, returncode=0
    )


def test_v0_regression_unittest_record_requires_both_suites() -> None:
    record = {
        "schema": "policy-learnware.v01-v0-unittest-result.v0",
        "unit_discovered": 100,
        "integration_discovered": 1,
        "tests_run": 101,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
        "expected_failures": 0,
        "unexpected_successes": 0,
        "successful": True,
    }
    assert _v0_regression_test_record_passed(record, returncode=0)
    assert not _v0_regression_test_record_passed(
        {**record, "integration_discovered": 0, "tests_run": 100},
        returncode=0,
    )
    assert not _v0_regression_test_record_passed(
        {**record, "tests_run": 100}, returncode=0
    )
    assert not _v0_regression_test_record_passed(record, returncode=1)


def test_freeze_dry_run_is_side_effect_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "must-not-exist"
    code = main(
        [
            "freeze-run",
            "--config",
            str(PROJECT / "configs" / "v01_smoke.yaml"),
            "--base-artifacts-root",
            str(tmp_path / "base-can-be-absent-in-dry-run"),
            "--artifacts-root",
            str(output),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not output.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry-run"
    assert payload["writes_performed"] is False
    assert payload["gpu_work_performed"] is False
    assert payload["inputs"]["registered_work"]["variants"] == 5


def test_dry_run_and_resume_together_fail_closed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "validate-config",
            "--config",
            str(PROJECT / "configs" / "v01_smoke.yaml"),
            "--base-artifacts-root",
            str(tmp_path),
            "--dry-run",
            "--resume",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["fail_closed"] is True
    assert "mutually exclusive" in payload["message"]
