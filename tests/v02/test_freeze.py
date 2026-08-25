from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from policy_learnware_v0.v02.config import load_v02_formal_config
from policy_learnware_v0.v02.freeze import (
    FormalFreezeError,
    FormalProtocolFreeze,
    canonical_formal_freeze_path,
    implementation_tree_manifest,
    load_verified_formal_freeze,
)
from tests.v02.test_config import _formal_payload


def _fake_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    package = root / "src" / "policy_learnware_v0" / "v02"
    server = root / "server" / "repro_fpo_ppo_v02"
    package.mkdir(parents=True)
    server.mkdir(parents=True)
    (root / "src" / "policy_learnware_v0" / "hashing.py").write_text(
        "def digest(value): return value\n", encoding="utf-8"
    )
    (package / "contract.py").write_text("VALUE = 1\n", encoding="utf-8")
    (server / "runner.py").write_text("VALUE = 2\n", encoding="utf-8")
    (server / "launch.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "audit_v02_recompute.py").write_text("VALUE = 3\n", encoding="utf-8")
    (scripts / "run_v02_cpu_acceptance.py").write_text("VALUE = 4\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'freeze-fixture'\nversion = '0.0.0'\n", encoding="utf-8"
    )
    (root / ".gitignore").write_text(
        "/src/policy_learnware_v0/ignored_runtime.py\n"
        "/src/policy_learnware_v0/injected.py\n"
        "/src/policy_learnware_v0/native.so\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Freeze Test"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "freeze-test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def _frozen_case(tmp_path: Path) -> tuple[Path, Path, FormalProtocolFreeze]:
    payload = _formal_payload()
    payload["experiment_id"] = "v02-freeze-unit"
    payload["artifact_root"] = str((tmp_path / "artifacts").resolve())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    config = load_v02_formal_config(config_path)
    repository = _fake_repository(tmp_path)
    (repository / "src" / "policy_learnware_v0" / "ignored_runtime.py").write_text(
        "VALUE = 'frozen'\n", encoding="utf-8"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    freeze = FormalProtocolFreeze.create(
        config,
        config_path=config_path,
        software_commit=commit,
        worktree_clean_at_freeze=True,
        repository_root=repository,
    )
    output = canonical_formal_freeze_path(config)
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(freeze.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    return config_path, repository, freeze


def test_formal_freeze_binds_exact_config_and_implementation_tree(tmp_path: Path) -> None:
    config_path, repository, expected = _frozen_case(tmp_path)
    config, observed = load_verified_formal_freeze(
        config_path, repository_root=repository
    )
    assert config.config_digest == expected.config_digest
    assert observed.digest == expected.digest

    (repository / "server" / "repro_fpo_ppo_v02" / "runner.py").write_text(
        "VALUE = 3\n", encoding="utf-8"
    )
    with pytest.raises(FormalFreezeError, match="worktree to remain clean"):
        load_verified_formal_freeze(config_path, repository_root=repository)


def test_formal_freeze_binds_shared_modules_shell_entrypoints_and_metadata(
    tmp_path: Path,
) -> None:
    config_path, repository, _ = _frozen_case(tmp_path)
    manifest = implementation_tree_manifest(repository)
    assert {
        "src/policy_learnware_v0/hashing.py",
        "server/repro_fpo_ppo_v02/launch.sh",
        "scripts/audit_v02_recompute.py",
        "scripts/run_v02_cpu_acceptance.py",
        "pyproject.toml",
    } <= set(manifest)
    for relative in (
        "src/policy_learnware_v0/hashing.py",
        "server/repro_fpo_ppo_v02/launch.sh",
        "pyproject.toml",
    ):
        path = repository / relative
        original = path.read_text(encoding="utf-8")
        path.write_text(original + "# changed\n", encoding="utf-8")
        with pytest.raises(FormalFreezeError, match="worktree to remain clean"):
            load_verified_formal_freeze(config_path, repository_root=repository)
        path.write_text(original, encoding="utf-8")


def test_formal_freeze_rejects_different_live_head(tmp_path: Path) -> None:
    config_path, repository, _ = _frozen_case(tmp_path)
    marker = repository / "release-marker.txt"
    marker.write_text("next\n", encoding="utf-8")
    subprocess.run(["git", "add", str(marker)], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "next"], cwd=repository, check=True)
    with pytest.raises(FormalFreezeError, match="software_commit"):
        load_verified_formal_freeze(config_path, repository_root=repository)


def test_formal_freeze_detects_git_ignored_runtime_mutation(tmp_path: Path) -> None:
    config_path, repository, _ = _frozen_case(tmp_path)
    runtime = repository / "src" / "policy_learnware_v0" / "ignored_runtime.py"
    runtime.write_text("VALUE = 'mutated'\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
    with pytest.raises(FormalFreezeError, match="implementation_tree_digest"):
        load_verified_formal_freeze(config_path, repository_root=repository)


def test_formal_freeze_rejects_symlink_and_native_import_artifacts(tmp_path: Path) -> None:
    config_path, repository, freeze = _frozen_case(tmp_path)
    package = repository / "src" / "policy_learnware_v0"
    external = tmp_path / "external.py"
    external.write_text("VALUE = 'mutable'\n", encoding="utf-8")
    injected = package / "injected.py"
    injected.symlink_to(external)
    with pytest.raises(FormalFreezeError, match="forbids symlinks"):
        FormalProtocolFreeze.create(
            load_v02_formal_config(config_path),
            config_path=config_path,
            software_commit=freeze.software_commit,
            worktree_clean_at_freeze=True,
            repository_root=repository,
        )
    injected.unlink()

    native = package / "native.so"
    native.write_bytes(b"not-a-real-extension")
    with pytest.raises(FormalFreezeError, match="bytecode/native"):
        FormalProtocolFreeze.create(
            load_v02_formal_config(config_path),
            config_path=config_path,
            software_commit=freeze.software_commit,
            worktree_clean_at_freeze=True,
            repository_root=repository,
        )


def test_formal_freeze_create_rejects_spoofed_commit_and_dirty_tree(tmp_path: Path) -> None:
    config_path, repository, freeze = _frozen_case(tmp_path)
    config = load_v02_formal_config(config_path)
    with pytest.raises(FormalFreezeError, match="does not match"):
        FormalProtocolFreeze.create(
            config,
            config_path=config_path,
            software_commit="f" * 40,
            worktree_clean_at_freeze=True,
            repository_root=repository,
        )

    (repository / "untracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(FormalFreezeError, match="clean Git worktree"):
        FormalProtocolFreeze.create(
            config,
            config_path=config_path,
            software_commit=freeze.software_commit,
            worktree_clean_at_freeze=True,
            repository_root=repository,
        )


def test_formal_freeze_rejects_changed_config_bytes_and_unclean_claim(tmp_path: Path) -> None:
    config_path, repository, freeze = _frozen_case(tmp_path)
    config_path.write_text(config_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FormalFreezeError, match="config_file_sha256"):
        load_verified_formal_freeze(config_path, repository_root=repository)

    with pytest.raises(FormalFreezeError, match="clean Git worktree"):
        FormalProtocolFreeze(
            **{
                name: getattr(freeze, name)
                for name in freeze.__dataclass_fields__
                if name != "worktree_clean_at_freeze"
            },
            worktree_clean_at_freeze=False,
        )
