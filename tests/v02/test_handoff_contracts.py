from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from server.repro_fpo_ppo_v02 import handoff_contracts as module
from server.repro_fpo_ppo_v02.handoff_contracts import derive_iterations_per_env
from server.repro_fpo_ppo_v02.provenance import ContractError


def test_handoff_geometry_preserves_the_frozen_upstream_contract() -> None:
    config = {
        "num_envs": 2_048,
        "num_minibatches": 32,
        "batch_size": 256,
        "unroll_length": 8,
    }
    assert derive_iterations_per_env(config) == 32

    with pytest.raises(ContractError, match="divide exactly"):
        derive_iterations_per_env({**config, "num_envs": 2_049})
    with pytest.raises(ContractError, match="positive integer"):
        derive_iterations_per_env({**config, "num_envs": True})


def test_checkpoint_contract_binds_canonical_path_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt_root = tmp_path.resolve()
    checkpoint_path = attempt_root / "checkpoints" / "outer_000006"
    checkpoint_path.mkdir(parents=True)
    checkpoint = {
        "path": str(checkpoint_path),
        "outer_iteration": 6,
        "environment_steps": 12_288,
        "bundle_manifest_sha256": "1" * 64,
        "bundle_manifest_digest": "2" * 64,
        "files": {"params.npz": "3" * 64},
    }
    job = {
        "training_protocol": {
            "algorithm": "FPO",
            "evaluation": {"enabled": True},
        },
        "seed": 2,
        "config_digest": "4" * 64,
        "execution_purpose": "v02_freeze_ready",
        "job_digest": "5" * 64,
    }
    attempt = {"attempt_digest": "6" * 64, "implementation": {"tree": "fixed"}}
    anchor = SimpleNamespace(
        task="reacher_easy",
        manifest_digest="7" * 64,
        environment_instance_digest="8" * 64,
        operator_digest="9" * 64,
        model_diff_digest="a" * 64,
        expected_bound_model_digest="b" * 64,
        runtime_digest="c" * 64,
    )
    execution = {
        "execution_mode": "formal_gpu",
        "formal_eligible": True,
        "execution_evidence_digest": "d" * 64,
    }
    source_attestation = {"fpo_execution_tree_digest": "e" * 64}
    observed_bundle = {
        key: checkpoint[key]
        for key in ("bundle_manifest_sha256", "bundle_manifest_digest", "files")
    }
    documents = {
        "bundle_manifest.json": {
            "algorithm": "FPO",
            "task": anchor.task,
            "seed": 2,
            "outer_iteration": 6,
            "environment_steps": 12_288,
        },
        "provenance.json": {
            "config_digest": job["config_digest"],
            "execution_purpose": job["execution_purpose"],
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor.manifest_digest,
            "environment_instance_digest": anchor.environment_instance_digest,
            "operator_digest": anchor.operator_digest,
            "model_diff_digest": anchor.model_diff_digest,
            "actual_bound_model_digest": anchor.expected_bound_model_digest,
            **source_attestation,
            "runtime_digest": anchor.runtime_digest,
            "implementation": attempt["implementation"],
            **execution,
            "attempt_root": str(attempt_root),
        },
    }
    monkeypatch.setattr(module, "validate_policy_bundle", lambda *_args, **_kwargs: observed_bundle)
    monkeypatch.setattr(module, "load_strict_json", lambda path: documents[Path(path).name])

    assert module._validate_checkpoint_bytes(
        checkpoint=checkpoint,
        explicit_path=checkpoint_path,
        server_job=job,
        attempt=attempt,
        anchor=anchor,
        execution=execution,
        source_attestation=source_attestation,
        attempt_root=attempt_root,
    ) == checkpoint_path

    documents["provenance.json"]["runtime_digest"] = "f" * 64
    with pytest.raises(ContractError, match="provenance binding mismatch"):
        module._validate_checkpoint_bytes(
            checkpoint=checkpoint,
            explicit_path=checkpoint_path,
            server_job=job,
            attempt=attempt,
            anchor=anchor,
            execution=execution,
            source_attestation=source_attestation,
            attempt_root=attempt_root,
        )


def test_checkpoint_contract_rebases_recorded_paths_without_rewriting_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    physical_attempt = tmp_path.resolve()
    physical_bundle = physical_attempt / "checkpoints" / "outer_000006"
    physical_bundle.mkdir(parents=True)
    recorded_attempt = Path("/frozen/exact90/jobs/job/attempt_001")
    recorded_bundle = recorded_attempt / "checkpoints" / "outer_000006"
    checkpoint = {
        "path": str(recorded_bundle),
        "outer_iteration": 6,
        "environment_steps": 12_288,
        "bundle_manifest_sha256": "1" * 64,
        "bundle_manifest_digest": "2" * 64,
        "files": {"params.npz": "3" * 64},
    }
    job = {
        "training_protocol": {
            "algorithm": "FPO",
            "evaluation": {"enabled": True},
        },
        "seed": 2,
        "config_digest": "4" * 64,
        "execution_purpose": "v02_freeze_ready",
        "job_digest": "5" * 64,
    }
    attempt = {"attempt_digest": "6" * 64, "implementation": {"tree": "fixed"}}
    anchor = SimpleNamespace(
        task="reacher_easy",
        manifest_digest="7" * 64,
        environment_instance_digest="8" * 64,
        operator_digest="9" * 64,
        model_diff_digest="a" * 64,
        expected_bound_model_digest="b" * 64,
        runtime_digest="c" * 64,
    )
    execution = {
        "execution_mode": "formal_gpu",
        "formal_eligible": True,
        "execution_evidence_digest": "d" * 64,
        "attempt_root": str(recorded_attempt),
    }
    source = {"fpo_execution_tree_digest": "e" * 64}
    observed = {
        key: checkpoint[key]
        for key in ("bundle_manifest_sha256", "bundle_manifest_digest", "files")
    }
    provenance = {
        "config_digest": job["config_digest"],
        "execution_purpose": job["execution_purpose"],
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "operator_digest": anchor.operator_digest,
        "model_diff_digest": anchor.model_diff_digest,
        "actual_bound_model_digest": anchor.expected_bound_model_digest,
        **source,
        "runtime_digest": anchor.runtime_digest,
        "implementation": attempt["implementation"],
        "execution_mode": execution["execution_mode"],
        "formal_eligible": execution["formal_eligible"],
        "execution_evidence_digest": execution["execution_evidence_digest"],
        "attempt_root": str(recorded_attempt),
    }
    monkeypatch.setattr(module, "validate_policy_bundle", lambda *_a, **_k: observed)
    monkeypatch.setattr(
        module,
        "load_strict_json",
        lambda path: (
            {
                "algorithm": "FPO",
                "task": anchor.task,
                "seed": 2,
                "outer_iteration": 6,
                "environment_steps": 12_288,
            }
            if Path(path).name == "bundle_manifest.json"
            else provenance
        ),
    )

    def resolver(path: str | Path) -> Path:
        suffix = Path(path).relative_to(recorded_attempt)
        return physical_attempt / suffix

    assert module._validate_checkpoint_bytes(
        checkpoint=checkpoint,
        explicit_path=recorded_bundle,
        server_job=job,
        attempt=attempt,
        anchor=anchor,
        execution=execution,
        source_attestation=source,
        attempt_root=physical_attempt,
        path_resolver=resolver,
    ) == physical_bundle
    assert checkpoint["path"] == str(recorded_bundle)
    assert provenance["attempt_root"] == str(recorded_attempt)


def test_completed_attempt_contract_rejects_job_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor = SimpleNamespace(
        manifest_digest="1" * 64,
        environment_instance_digest="2" * 64,
    )
    monkeypatch.setattr(
        module.AnchorManifest,
        "from_path",
        lambda _path: anchor,
    )
    monkeypatch.setattr(
        module,
        "validate_success_record",
        lambda *_args, **_kwargs: {
            "implementation": {},
            "algorithm": "PPO",
            "seed": 0,
        },
    )
    monkeypatch.setattr(
        module,
        "validate_implementation_provenance",
        lambda *_args, **_kwargs: None,
    )
    job = {
        "anchor_manifest_path": "anchor.json",
        "job_digest": "3" * 64,
        "training_protocol_digest": "4" * 64,
        "config_digest": "5" * 64,
        "execution_purpose": "v02_freeze_ready",
        "training_protocol": {"algorithm": "FPO"},
        "seed": 0,
    }
    attempt = {"attempt_digest": "6" * 64, "implementation": {}}

    with pytest.raises(ContractError, match="algorithm drifted"):
        module.validate_completed_attempt(
            tmp_path,
            job,
            attempt,
            expected_vendor={},
            expected_implementation={},
        )


def test_pool_acceptance_import_does_not_load_historical_training_stack() -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    pythonpath = [str(repository / "src"), str(repository)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import json
import sys

import server.repro_fpo_ppo_v02.pool_acceptance

prefix = "server.repro_fpo_ppo_v02."
print(json.dumps(sorted(name for name in sys.modules if name.startswith(prefix))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))
    forbidden = {
        "server.repro_fpo_ppo_v02.formal_plan",
        "server.repro_fpo_ppo_v02.implementation",
        "server.repro_fpo_ppo_v02.package_bridge",
        "server.repro_fpo_ppo_v02.queue_master",
        "server.repro_fpo_ppo_v02.runner",
    }
    assert loaded.isdisjoint(forbidden)
    assert loaded == {
        "server.repro_fpo_ppo_v02.anchor_binding",
        "server.repro_fpo_ppo_v02.handoff_contracts",
        "server.repro_fpo_ppo_v02.pool_acceptance",
        "server.repro_fpo_ppo_v02.provenance",
    }
