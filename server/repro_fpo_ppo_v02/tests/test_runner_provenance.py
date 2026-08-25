from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from repro_fpo_ppo_v02.provenance import (
    AUDIT_SMOKE_EXECUTION_MODE,
    ContractError,
)
from repro_fpo_ppo_v02.runner import _inspect_fpo_source, _runtime_provenance


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


class RunnerProvenanceTests(unittest.TestCase):
    def test_source_attestation_is_derived_from_live_git_checks(self) -> None:
        with patch(
            "repro_fpo_ppo_v02.runner._git",
            side_effect=[" M tracked.py\nD  removed.py", "b" * 40],
        ):
            observed = _inspect_fpo_source(
                Path("/synthetic/fpo"), expected_commit="a" * 40
            )

        self.assertEqual(observed["fpo_commit"], "b" * 40)
        self.assertEqual(observed["expected_fpo_commit"], "a" * 40)
        self.assertFalse(observed["fpo_commit_matches_expected"])
        self.assertTrue(observed["fpo_tracked_dirty"])
        self.assertEqual(
            observed["fpo_tracked_changes"], [" M tracked.py", "D  removed.py"]
        )

    def test_runtime_provenance_carries_clean_freeze_matched_attestation(self) -> None:
        source = {
            "fpo_commit": "a" * 40,
            "expected_fpo_commit": "a" * 40,
            "fpo_commit_matches_expected": True,
            "fpo_tracked_dirty": False,
            "fpo_tracked_changes": [],
        }
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
        forged = {
            "fpo_commit": "a" * 40,
            "expected_fpo_commit": "a" * 40,
            "fpo_commit_matches_expected": True,
            "fpo_tracked_dirty": False,
            "fpo_tracked_changes": ["M tracked.py"],
        }
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
