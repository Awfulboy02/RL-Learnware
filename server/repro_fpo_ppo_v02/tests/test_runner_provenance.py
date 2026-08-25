from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from repro_fpo_ppo_v02.provenance import (
    AUDIT_SMOKE_EXECUTION_MODE,
    ContractError,
    NumericalIntegrityError,
)
from repro_fpo_ppo_v02.runner import (
    _inspect_fpo_source,
    _is_recoverable_terminal_error,
    _runtime_provenance,
)


class _FakeJax:
    @staticmethod
    def default_backend() -> str:
        return "gpu"

    @staticmethod
    def devices() -> list[str]:
        return ["SyntheticGpuDevice(id=0)"]


class _Anchor:
    runtime = {"fpo_commit": "a" * 40}
    runtime_digest = "b" * 64


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        allow_non_gpu=True,
        run_dir=root / "attempt",
        fpo_root=root / "fpo",
        legacy_policy_io=root / "policy_io.py",
    )


def _attempt() -> dict[str, object]:
    return {
        "config_digest": "c" * 64,
        "execution_purpose": "audit_smoke",
        "execution_mode": AUDIT_SMOKE_EXECUTION_MODE,
        "formal_eligible": False,
        "job_digest": "d" * 64,
        "attempt_digest": "e" * 64,
    }


def _clean_source() -> dict[str, object]:
    return {
        "fpo_commit": "a" * 40,
        "expected_fpo_commit": "a" * 40,
        "fpo_commit_matches_expected": True,
        "fpo_tracked_dirty": False,
        "fpo_tracked_changes": [],
        "fpo_head_tree_digest": "1" * 64,
        "fpo_worktree_tree_digest": "1" * 64,
        "fpo_execution_tree_digest": "2" * 64,
        "fpo_source_file_count": 1,
        "fpo_index_flags": [],
        "fpo_untracked_paths": [],
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(root: Path) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "audit@example.invalid")
    _git(root, "config", "user.name", "Audit Test")
    (root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "tracked.py")
    _git(root, "commit", "-qm", "fixture")
    return _git(root, "rev-parse", "HEAD")


class RunnerProvenanceTests(unittest.TestCase):
    def test_only_post_checkpoint_ladder_numerical_failure_is_recoverable(self) -> None:
        numerical = NumericalIntegrityError("non-finite actor")
        self.assertTrue(
            _is_recoverable_terminal_error(
                numerical,
                checkpoint_rule="fixed_ladder",
                validated_checkpoint_count=1,
            )
        )
        self.assertFalse(
            _is_recoverable_terminal_error(
                numerical,
                checkpoint_rule="fixed_final",
                validated_checkpoint_count=1,
            )
        )
        self.assertFalse(
            _is_recoverable_terminal_error(
                numerical,
                checkpoint_rule="fixed_ladder",
                validated_checkpoint_count=0,
            )
        )
        self.assertFalse(
            _is_recoverable_terminal_error(
                ContractError("compiled parity failed"),
                checkpoint_rule="fixed_ladder",
                validated_checkpoint_count=1,
            )
        )

    def test_assume_unchanged_cannot_hide_modified_tracked_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fpo"
            commit = _repository(root)
            clean = _inspect_fpo_source(root, expected_commit=commit)
            self.assertFalse(clean["fpo_tracked_dirty"])

            _git(root, "update-index", "--assume-unchanged", "tracked.py")
            (root / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
            observed = _inspect_fpo_source(root, expected_commit=commit)

        self.assertEqual(observed["fpo_commit"], commit)
        self.assertTrue(observed["fpo_commit_matches_expected"])
        self.assertTrue(observed["fpo_tracked_dirty"])
        self.assertIn("tracked.py", observed["fpo_tracked_changes"])
        self.assertEqual(observed["fpo_index_flags"], ["h tracked.py"])

    def test_untracked_source_is_explicitly_rejected_by_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fpo"
            commit = _repository(root)
            (root / "untracked_runtime.py").write_text("VALUE = 9\n", encoding="utf-8")
            observed = _inspect_fpo_source(root, expected_commit=commit)
        self.assertEqual(observed["fpo_untracked_paths"], ["untracked_runtime.py"])

    def test_git_replace_ref_cannot_redirect_frozen_commit_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fpo"
            frozen = _repository(root)
            (root / "tracked.py").write_text("VALUE = 'malicious'\n", encoding="utf-8")
            _git(root, "add", "tracked.py")
            _git(root, "commit", "-qm", "replacement")
            replacement = _git(root, "rev-parse", "HEAD")
            _git(root, "replace", frozen, replacement)
            _git(root, "reset", "--hard", frozen)
            self.assertIn("malicious", (root / "tracked.py").read_text(encoding="utf-8"))

            with self.assertRaisesRegex(RuntimeError, "forbidden Git replace refs"):
                _inspect_fpo_source(root, expected_commit=frozen)

    def test_path_precedence_cannot_replace_trusted_git_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "fpo"
            commit = _repository(root)
            wrapper_root = base / "bin"
            wrapper_root.mkdir()
            marker = base / "wrapper-was-run"
            wrapper = wrapper_root / "git"
            wrapper.write_text(
                f"#!/bin/sh\ntouch {marker}\nexec /usr/bin/git \"$@\"\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            with patch.dict(
                os.environ,
                {"PATH": f"{wrapper_root}:{os.environ.get('PATH', '')}"},
            ):
                observed = _inspect_fpo_source(root, expected_commit=commit)
            self.assertTrue(observed["fpo_commit_matches_expected"])
            self.assertFalse(marker.exists())

    def test_runtime_provenance_carries_clean_freeze_matched_attestation(self) -> None:
        source = _clean_source()
        with tempfile.TemporaryDirectory() as directory:
            observed = _runtime_provenance(
                args=_args(Path(directory)),
                attempt=_attempt(),
                anchor=_Anchor(),
                jax=_FakeJax(),
                source=source,
                vendor={"vendor_digest": "f" * 64},
                implementation={"implementation_digest": "0" * 64},
            )

        for key, value in source.items():
            self.assertEqual(observed[key], value)

    def test_runtime_provenance_rejects_forged_clean_attestation(self) -> None:
        forged = _clean_source()
        forged["fpo_tracked_changes"] = ["M tracked.py"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ContractError, "not clean and freeze-matched"
            ):
                _runtime_provenance(
                    args=_args(Path(directory)),
                    attempt=_attempt(),
                    anchor=_Anchor(),
                    jax=_FakeJax(),
                    source=forged,
                    vendor={},
                    implementation={},
                )


if __name__ == "__main__":
    unittest.main()
