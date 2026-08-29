from __future__ import annotations

import importlib.machinery
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from policy_learnware_v0.v02 import runtime


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _make_checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "fpo"
    source = root / "playground" / "src" / "flow_policy"
    source.mkdir(parents=True)
    (root / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    (source / "__init__.py").write_text("", encoding="utf-8")
    for name in ("fpo", "ppo", "rollouts"):
        (source / f"{name}.py").write_text(f"NAME = {name!r}\n", encoding="utf-8")
    subprocess.run(["/usr/bin/git", "init", "-q", str(root)], check=True)
    _run_git(root, "config", "user.name", "v02-test")
    _run_git(root, "config", "user.email", "v02-test@example.invalid")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-q", "-m", "fixture")
    return root, _run_git(root, "rev-parse", "HEAD")


def _bind_fixture_as_frozen(
    monkeypatch: pytest.MonkeyPatch, root: Path, commit: str
) -> dict[str, object]:
    observed = runtime.inspect_fpo_checkout(root, expected_commit=commit)
    monkeypatch.setattr(runtime, "FPO_COMMIT", commit)
    monkeypatch.setattr(
        runtime, "FPO_HEAD_TREE_DIGEST", observed["fpo_head_tree_digest"]
    )
    monkeypatch.setattr(
        runtime,
        "FPO_EXECUTION_TREE_DIGEST",
        observed["fpo_execution_tree_digest"],
    )
    monkeypatch.setattr(
        runtime, "FPO_SOURCE_FILE_COUNT", observed["fpo_source_file_count"]
    )
    return observed


def test_verify_fpo_checkout_binds_clean_git_and_execution_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, commit = _make_checkout(tmp_path)
    expected = _bind_fixture_as_frozen(monkeypatch, root, commit)

    assert runtime.verify_fpo_checkout(root) == expected
    assert expected["fpo_head_tree_digest"] == expected["fpo_worktree_tree_digest"]
    assert expected["fpo_index_flags"] == []
    assert expected["fpo_untracked_paths"] == []
    assert expected["fpo_ignored_paths"] == []


def test_public_verifier_and_loader_reject_symlink_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    root, commit = _make_checkout(real_parent)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_root = linked_parent / "fpo"

    with pytest.raises(runtime.RuntimeVerificationError, match="symlink component"):
        runtime.inspect_fpo_checkout(linked_root, expected_commit=commit)
    with pytest.raises(runtime.RuntimeVerificationError, match="symlink component"):
        runtime.verify_fpo_checkout(linked_root)

    calls: list[Path] = []

    def should_not_verify(path: str | Path) -> dict[str, object]:
        calls.append(Path(path))
        return {"ok": True}

    monkeypatch.setattr(runtime, "verify_fpo_checkout", should_not_verify)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    with pytest.raises(runtime.RuntimeVerificationError, match="symlink component"):
        runtime.load_verified_fpo_upstream(
            linked_root, allow_reconstructed=True
        )
    assert calls == []


@pytest.mark.parametrize("tamper", ["tracked", "untracked", "ignored", "index"])
def test_verify_fpo_checkout_rejects_all_checkout_bypasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    root, commit = _make_checkout(tmp_path)
    _bind_fixture_as_frozen(monkeypatch, root, commit)
    source = root / "playground" / "src" / "flow_policy"

    if tamper == "tracked":
        (source / "fpo.py").write_text("NAME = 'tampered'\n", encoding="utf-8")
    elif tamper == "untracked":
        (root / "untracked.txt").write_text("not reviewed\n", encoding="utf-8")
    elif tamper == "ignored":
        cache = source / "__pycache__"
        cache.mkdir()
        (cache / "fpo.cpython-312.pyc").write_bytes(b"not reviewed")
        inspected = runtime.inspect_fpo_checkout(root, expected_commit=commit)
        assert inspected["fpo_ignored_paths"] == [
            "playground/src/flow_policy/__pycache__/fpo.cpython-312.pyc"
        ]
    else:
        _run_git(
            root,
            "update-index",
            "--assume-unchanged",
            "playground/src/flow_policy/fpo.py",
        )
        inspected = runtime.inspect_fpo_checkout(root, expected_commit=commit)
        assert inspected["fpo_index_flags"] == [
            "h playground/src/flow_policy/fpo.py"
        ]

    with pytest.raises(runtime.RuntimeVerificationError):
        runtime.verify_fpo_checkout(root)


def _fake_module(name: str, origin: Path | None = None) -> ModuleType:
    module = ModuleType(name)
    loader = (
        None
        if origin is None
        else importlib.machinery.SourceFileLoader(name, str(origin))
    )
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=loader, origin=None if origin is None else str(origin)
    )
    if origin is not None:
        module.__file__ = str(origin)
    return module


def _fake_imports(root: Path) -> dict[str, ModuleType]:
    source = root / "playground" / "src" / "flow_policy"
    playground = _fake_module("mujoco_playground")
    playground.dm_control_suite = _fake_module(  # type: ignore[attr-defined]
        "mujoco_playground.dm_control_suite"
    )
    playground.registry = _fake_module(  # type: ignore[attr-defined]
        "mujoco_playground.registry"
    )
    return {
        "mujoco_playground": playground,
        "jax": _fake_module("jax"),
        "jax_dataclasses": _fake_module("jax_dataclasses"),
        "jax.numpy": _fake_module("jax.numpy"),
        "flow_policy.fpo": _fake_module("flow_policy.fpo", source / "fpo.py"),
        "flow_policy.ppo": _fake_module("flow_policy.ppo", source / "ppo.py"),
        "flow_policy.rollouts": _fake_module(
            "flow_policy.rollouts", source / "rollouts.py"
        ),
    }


def test_loader_requires_opt_in_and_returns_reconstructed_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    attestation = {"fpo_commit": "a" * 40, "fpo_untracked_paths": []}
    calls: list[Path] = []

    def verify(path: str | Path) -> dict[str, object]:
        calls.append(Path(path))
        return dict(attestation)

    modules = _fake_imports(root)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", verify)
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    with pytest.raises(runtime.ReconstructedRuntimeNotAllowed):
        runtime.load_verified_fpo_upstream(root)

    loaded = runtime.load_verified_fpo_upstream(
        root, allow_reconstructed=True
    )
    assert loaded.provenance_class == runtime.RECONSTRUCTED_RUNTIME
    assert loaded.runtime_receipt == {
        "schema": "policy-learnware.v02-reconstructed-runtime.v1",
        "runtime_status": runtime.RECONSTRUCTED_RUNTIME,
        "original_runtime_capable": False,
        "training_replay_capable": False,
        "inference_only": True,
        "missing_dependency": "wandb",
        "shim_identity": runtime.INFERENCE_ONLY_WANDB_SHIM_IDENTITY,
        "installed_wandb_bypassed": False,
    }
    assert loaded.legacy_module_tuple() == (
        modules["jax"],
        modules["jax_dataclasses"],
        modules["jax.numpy"],
        modules["mujoco_playground"].dm_control_suite,  # type: ignore[attr-defined]
        modules["mujoco_playground"].registry,  # type: ignore[attr-defined]
        modules["flow_policy.fpo"],
        modules["flow_policy.ppo"],
        modules["flow_policy.rollouts"],
    )
    assert len(calls) == 2
    assert str(root / "playground" / "src") not in sys.path
    with pytest.raises(TypeError):
        loaded.source_attestation["fpo_commit"] = "b" * 40  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded.runtime_receipt["runtime_status"] = "ORIGINAL"  # type: ignore[index]


def test_loader_rejects_wrong_flow_policy_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    modules = _fake_imports(root)
    outside = tmp_path / "other" / "fpo.py"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")
    modules["flow_policy.fpo"] = _fake_module("flow_policy.fpo", outside)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda path: {"ok": True})
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    with pytest.raises(runtime.RuntimeVerificationError, match="attested"):
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)


def test_loader_rejects_cached_flow_policy_from_another_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    outside = tmp_path / "other" / "fpo.py"
    outside.parent.mkdir()
    outside.write_text("", encoding="utf-8")
    cached = _fake_module("flow_policy.fpo", outside)
    monkeypatch.setitem(sys.modules, "flow_policy.fpo", cached)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda path: {"ok": True})
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    with pytest.raises(runtime.RuntimeVerificationError, match="refusing cached"):
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)


def test_loader_rejects_same_checkout_cached_module_even_with_trusted_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    source = root / "playground" / "src" / "flow_policy" / "fpo.py"
    cached = _fake_module("flow_policy.fpo", source)
    cached.MUTATED_AFTER_IMPORT = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "flow_policy.fpo", cached)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda path: {"ok": True})
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    with pytest.raises(runtime.RuntimeVerificationError, match="refusing cached"):
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)


def test_loader_uses_scoped_import_only_wandb_shim_and_restores_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    modules = _fake_imports(root)
    orphan = _fake_module("wandb.orphan")
    monkeypatch.setitem(sys.modules, "wandb.orphan", orphan)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.delitem(sys.modules, "wandb.sdk", raising=False)
    monkeypatch.delitem(sys.modules, "wandb.sdk.wandb_run", raising=False)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda path: {"ok": True})
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def import_module(name: str) -> ModuleType:
        if name == "flow_policy.fpo" and "wandb" not in sys.modules:
            raise ModuleNotFoundError("No module named 'wandb'", name="wandb")
        if name == "flow_policy.rollouts":
            modules[name].wandb = sys.modules["wandb"]  # type: ignore[attr-defined]
            modules[name].Run = sys.modules[  # type: ignore[attr-defined]
                "wandb.sdk.wandb_run"
            ].Run
        return modules[name]

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    loaded = runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)

    assert loaded.runtime_receipt == {
        "schema": "policy-learnware.v02-reconstructed-runtime.v1",
        "runtime_status": runtime.RECONSTRUCTED_RUNTIME,
        "original_runtime_capable": False,
        "training_replay_capable": False,
        "inference_only": True,
        "missing_dependency": "wandb",
        "shim_identity": runtime.INFERENCE_ONLY_WANDB_SHIM_IDENTITY,
        "installed_wandb_bypassed": False,
    }
    assert sys.modules["wandb.orphan"] is orphan
    assert "wandb" not in sys.modules
    assert "wandb.sdk" not in sys.modules
    assert "wandb.sdk.wandb_run" not in sys.modules

    shim = loaded.rollouts.wandb
    for behavior in ("init", "log", "save", "Histogram", "Api"):
        with pytest.raises(runtime.ReconstructedRuntimeNotAllowed):
            getattr(shim, behavior)()
    with pytest.raises(runtime.ReconstructedRuntimeNotAllowed):
        getattr(shim, "network")
    with pytest.raises(runtime.ReconstructedRuntimeNotAllowed):
        loaded.rollouts.Run()


def test_wandb_shim_supports_required_import_surface_and_restores_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orphan = _fake_module("wandb.preexisting")
    monkeypatch.setitem(sys.modules, "wandb.preexisting", orphan)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    monkeypatch.delitem(sys.modules, "wandb.sdk", raising=False)
    monkeypatch.delitem(sys.modules, "wandb.sdk.wandb_run", raising=False)
    before = {
        name: module
        for name, module in sys.modules.items()
        if name == "wandb" or name.startswith("wandb.")
    }

    with runtime._temporary_inference_only_wandb():
        import wandb
        from wandb.sdk.wandb_run import Run

        assert wandb is sys.modules["wandb"]
        assert Run is sys.modules["wandb.sdk.wandb_run"].Run
        with pytest.raises(runtime.ReconstructedRuntimeNotAllowed):
            wandb.log({"must": "not escape"})

    after = {
        name: module
        for name, module in sys.modules.items()
        if name == "wandb" or name.startswith("wandb.")
    }
    assert after == before


def test_loader_masks_preinstalled_real_wandb_and_restores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    modules = _fake_imports(root)
    real_wandb = _fake_module("wandb")
    real_sdk = _fake_module("wandb.sdk")
    real_run = _fake_module("wandb.sdk.wandb_run")
    real_wandb.init = lambda *_args, **_kwargs: "would-connect"  # type: ignore[attr-defined]
    real_run.Run = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", real_wandb)
    monkeypatch.setitem(sys.modules, "wandb.sdk", real_sdk)
    monkeypatch.setitem(sys.modules, "wandb.sdk.wandb_run", real_run)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda path: {"ok": True})
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def import_module(name: str) -> ModuleType:
        if name == "flow_policy.rollouts":
            modules[name].wandb = sys.modules["wandb"]  # type: ignore[attr-defined]
            modules[name].Run = sys.modules[  # type: ignore[attr-defined]
                "wandb.sdk.wandb_run"
            ].Run
        return modules[name]

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    loaded = runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)

    assert sys.modules["wandb"] is real_wandb
    assert sys.modules["wandb.sdk"] is real_sdk
    assert sys.modules["wandb.sdk.wandb_run"] is real_run
    assert loaded.rollouts.wandb is not real_wandb
    with pytest.raises(runtime.ReconstructedRuntimeNotAllowed):
        loaded.rollouts.wandb.init()
    with pytest.raises(runtime.ReconstructedRuntimeNotAllowed):
        loaded.rollouts.wandb.finish()
    assert loaded.runtime_receipt == {
        "schema": "policy-learnware.v02-reconstructed-runtime.v1",
        "runtime_status": runtime.RECONSTRUCTED_RUNTIME,
        "original_runtime_capable": False,
        "training_replay_capable": False,
        "inference_only": True,
        "missing_dependency": None,
        "shim_identity": runtime.INFERENCE_ONLY_WANDB_SHIM_IDENTITY,
        "installed_wandb_bypassed": True,
    }

    # A second source load in the same process fails closed because executed
    # flow_policy modules are cached; it must not disturb the real wandb.
    monkeypatch.setitem(sys.modules, "flow_policy.fpo", loaded.fpo)
    with pytest.raises(runtime.RuntimeVerificationError, match="refusing cached"):
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)
    assert sys.modules["wandb"] is real_wandb


def test_loader_import_failure_restores_preinstalled_wandb_and_flow_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    modules = _fake_imports(root)
    real_wandb = _fake_module("wandb")
    sentinel_flow = _fake_module("unrelated")
    monkeypatch.setitem(sys.modules, "wandb", real_wandb)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda path: {"ok": True})
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def import_module(name: str) -> ModuleType:
        if name == "flow_policy.ppo":
            monkeypatch.setitem(sys.modules, "flow_policy.partial", sentinel_flow)
            raise ModuleNotFoundError("No module named 'optax'", name="optax")
        return modules[name]

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    with pytest.raises(runtime.RuntimeVerificationError) as caught:
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == "optax"
    assert sys.modules["wandb"] is real_wandb
    assert "flow_policy.partial" not in sys.modules


def test_loader_does_not_shim_any_other_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    modules = _fake_imports(root)
    monkeypatch.setattr(runtime, "verify_fpo_checkout", lambda path: {"ok": True})
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def import_module(name: str) -> ModuleType:
        if name == "flow_policy.fpo":
            raise ModuleNotFoundError("No module named 'optax'", name="optax")
        return modules[name]

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    with pytest.raises(runtime.RuntimeVerificationError) as caught:
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == "optax"


def test_loader_never_installs_shim_before_source_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    imports: list[str] = []

    def reject(_path: str | Path) -> dict[str, object]:
        raise runtime.RuntimeVerificationError("bad source")

    monkeypatch.setattr(runtime, "verify_fpo_checkout", reject)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: imports.append(name),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    with pytest.raises(runtime.RuntimeVerificationError, match="bad source"):
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)
    assert imports == []


def test_loader_requires_bytecode_suppression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _make_checkout(tmp_path)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(runtime.RuntimeVerificationError, match="dont_write_bytecode"):
        runtime.load_verified_fpo_upstream(root, allow_reconstructed=True)


def test_original_vendor_is_permanently_missing_and_fail_closed() -> None:
    status = runtime.original_vendor_status()
    assert status == {
        "status": "MISSING_ORIGINAL",
        "provenance_class": "MISSING_ORIGINAL",
        "expected_tree_digest": runtime.ORIGINAL_VENDOR_TREE_DIGEST,
        "expected_file_count": 1612,
        "expected_total_bytes": 90_428_849,
    }
    with pytest.raises(TypeError):
        status["status"] = "verified"  # type: ignore[index]
    with pytest.raises(runtime.OriginalVendorUnavailable, match="MISSING_ORIGINAL"):
        runtime.require_original_vendor_runtime()
