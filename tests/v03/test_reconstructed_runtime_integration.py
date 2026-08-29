from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

from policy_learnware_v0.v02 import runtime as v02_runtime
from policy_learnware_v0.v03 import fpo_source_backend
from server.repro_fpo_ppo_v03 import (
    asset_binding,
    development_baseline_runner,
    source_market_runner,
)


_V03_RUNTIME_FILES = (
    Path(fpo_source_backend.__file__),
    Path(asset_binding.__file__),
    Path(source_market_runner.__file__),
    Path(development_baseline_runner.__file__),
)


def test_v03_runtime_closure_has_no_private_v02_imports() -> None:
    forbidden_modules = {
        "server.repro_fpo_ppo_v02.runner",
        "server.repro_fpo_ppo_v02.vendor",
    }
    forbidden_names = {
        "_load_upstream",
        "inspect_vendor_directory",
        "require_vendor_pythonpath_first",
    }
    for path in _V03_RUNTIME_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not (forbidden_modules & {alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_modules
                assert not (forbidden_names & {alias.name for alias in node.names})
        assert not (forbidden_names & set(source.split()))
    assert "runtime_factory=runtime_factory" in Path(
        development_baseline_runner.__file__
    ).read_text(encoding="utf-8")


def test_original_vendor_is_default_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    fpo_root = tmp_path / "fpo"
    fpo_root.mkdir()

    with pytest.raises(
        fpo_source_backend.FpoSourceBackendError, match="MISSING_ORIGINAL"
    ):
        fpo_source_backend.FrozenV02FpoJaxRuntimeDriver(fpo_root=fpo_root)

    bind = asset_binding._parser().parse_args(
        [
            "--intake-record",
            "intake.json",
            "--server-plan",
            "plan.json",
            "--fpo-root",
            "fpo",
            "--output-dir",
            "out",
        ]
    )
    source = source_market_runner._parser().parse_args(
        ["--binding-dir", "binding", "--output-dir", "out", "--fpo-root", "fpo"]
    )
    baseline = development_baseline_runner._parser().parse_args(
        ["evaluate", "--output-dir", "out"]
    )
    assert bind.allow_reconstructed_runtime is False
    assert source.allow_reconstructed_runtime is False
    assert baseline.allow_reconstructed_runtime is False


def _fake_upstream() -> SimpleNamespace:
    return SimpleNamespace(
        provenance_class=v02_runtime.RECONSTRUCTED_RUNTIME,
        runtime_receipt=MappingProxyType(
            {
                "schema": "policy-learnware.v02-reconstructed-runtime.v1",
                "runtime_status": v02_runtime.RECONSTRUCTED_RUNTIME,
                "original_runtime_capable": False,
                "training_replay_capable": False,
                "inference_only": True,
                "missing_dependency": "wandb",
                "shim_identity": v02_runtime.INFERENCE_ONLY_WANDB_SHIM_IDENTITY,
                "installed_wandb_bypassed": False,
            }
        ),
        source_attestation=MappingProxyType(
            {
                "fpo_commit": v02_runtime.FPO_COMMIT,
                "fpo_execution_tree_digest": v02_runtime.FPO_EXECUTION_TREE_DIGEST,
                "fpo_source_file_count": v02_runtime.FPO_SOURCE_FILE_COUNT,
            }
        ),
        jax=object(),
        jax_dataclasses=object(),
        jax_numpy=object(),
        dm_control_suite=object(),
        registry=object(),
        fpo=object(),
        ppo=object(),
        rollouts=object(),
    )


def test_explicit_reconstructed_preflight_is_portable_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    attestation = {
        "fpo_commit": v02_runtime.FPO_COMMIT,
        "fpo_head_tree_digest": v02_runtime.FPO_HEAD_TREE_DIGEST,
        "fpo_worktree_tree_digest": v02_runtime.FPO_HEAD_TREE_DIGEST,
        "fpo_execution_tree_digest": v02_runtime.FPO_EXECUTION_TREE_DIGEST,
        "fpo_source_file_count": v02_runtime.FPO_SOURCE_FILE_COUNT,
    }
    monkeypatch.setattr(
        v02_runtime, "verify_fpo_checkout", lambda _path: dict(attestation)
    )
    calls: list[tuple[Path, bool]] = []

    def load(path: Path, *, allow_reconstructed: bool) -> SimpleNamespace:
        calls.append((Path(path), allow_reconstructed))
        return _fake_upstream()

    monkeypatch.setattr(v02_runtime, "load_verified_fpo_upstream", load)
    roots = (tmp_path / "host-a" / "fpo", tmp_path / "host-b" / "fpo")
    for root in roots:
        root.mkdir(parents=True)
    drivers = tuple(
        fpo_source_backend.FrozenV02FpoJaxRuntimeDriver(
            fpo_root=root,
            allow_reconstructed_runtime=True,
        )
        for root in roots
    )
    assert drivers[0].runtime_driver_digest == drivers[1].runtime_driver_digest
    first = dict(drivers[0].preflight())
    assert dict(drivers[0].preflight()) == first
    second = dict(drivers[1].preflight())
    assert len(calls) == 2
    assert all(allowed is True for _, allowed in calls)
    for evidence in (first, second):
        assert evidence["provenance_class"] == "RECONSTRUCTED_RUNTIME"
        assert evidence["preflight_complete"] is True
        assert evidence["runtime_receipt"]["original_runtime_capable"] is False
        assert evidence["runtime_receipt"]["training_replay_capable"] is False
        assert evidence["runtime_receipt"]["inference_only"] is True
        assert evidence["runtime_receipt"]["installed_wandb_bypassed"] is False
        assert evidence["original_vendor"]["status"] == "MISSING_ORIGINAL"


def test_source_market_preflights_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class PreflightFailure(RuntimeError):
        pass

    class Driver:
        def __init__(self, **_kwargs: object) -> None:
            events.append("driver")

        def preflight(self) -> None:
            events.append("preflight")
            raise PreflightFailure("stop before output")

    binding = tmp_path / "binding"
    fpo = tmp_path / "fpo"
    binding.mkdir()
    fpo.mkdir()
    output = tmp_path / "new-output"
    protocol = SimpleNamespace(selection_reset_seeds=(1,), attestation_reset_seeds=(2,))
    monkeypatch.setattr(
        source_market_runner,
        "_load_binding",
        lambda _path: (object(), protocol, {}, object()),
    )
    monkeypatch.setattr(
        source_market_runner,
        "_market_nonce",
        lambda _path, domain: domain,
    )
    monkeypatch.setattr(source_market_runner, "FrozenV02FpoJaxRuntimeDriver", Driver)

    with pytest.raises(PreflightFailure, match="before output"):
        source_market_runner.run_source_market(
            binding_dir=binding,
            output_dir=output,
            fpo_root=fpo,
            allow_reconstructed_runtime=True,
        )
    assert events == ["driver", "preflight"]
    assert not output.exists()


def test_development_baseline_publishes_reconstructed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    output = tmp_path / "baseline"
    (output / "representations").mkdir(parents=True)
    (output / "progress").mkdir()
    (output / "representations" / "build_config.json").write_text(
        json.dumps({"context_rows": []}), encoding="utf-8"
    )
    upstream = _fake_upstream()
    monkeypatch.setattr(
        development_baseline_runner,
        "_load_reconstructed_runtime",
        lambda *_args, **_kwargs: upstream,
    )
    monkeypatch.setattr(
        development_baseline_runner,
        "_market",
        lambda *_args: SimpleNamespace(entries={}, deployment_private={}),
    )
    monkeypatch.setattr(development_baseline_runner, "_anchor_tasks", lambda _path: {})
    monkeypatch.setattr(development_baseline_runner, "_factory", lambda *_args: object())
    args = argparse.Namespace(
        output_dir=output,
        fpo_root=tmp_path / "fpo",
        allow_reconstructed_runtime=True,
        public_policy_market=tmp_path / "public.json",
        deployment_private_registry=tmp_path / "private.json",
        source_anchor_manifests=tmp_path / "anchors",
        v02_config=tmp_path / "v02.yaml",
        cp0_config=tmp_path / "cp0.json",
        environment_factory=None,
        shard_count=1,
        shard_index=0,
        reset_seed_start=730_000,
        episodes=1,
        horizon=1000,
        resume=False,
        relocation_manifest=None,
        artifacts_root=tmp_path,
    )
    summary = development_baseline_runner.evaluate(args)
    assert summary["status"] == "COMPLETE"
    assert summary["runtime"]["provenance_class"] == "RECONSTRUCTED_RUNTIME"
    assert summary["runtime"]["runtime_receipt"]["original_runtime_capable"] is False
    assert summary["runtime"]["runtime_receipt"]["training_replay_capable"] is False
    assert summary["runtime"]["original_vendor"]["status"] == "MISSING_ORIGINAL"
    stored = json.loads(
        (output / "progress" / "evaluate-000-of-001.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == summary
