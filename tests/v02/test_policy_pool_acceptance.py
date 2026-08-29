from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from server.repro_fpo_ppo_v02 import pool_acceptance as module
from server.repro_fpo_ppo_v02.provenance import (
    ContractError,
    sha256_file,
    with_self_digest,
)


def _jobs() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for anchor in range(30):
        for seed in range(3):
            job_id = f"job-{anchor:02d}-{seed}"
            result[job_id] = {
                "job_id": job_id,
                "job_digest": f"{anchor * 3 + seed + 1:064x}",
                "anchor_manifest_digest": f"{anchor + 1000:064x}",
                "anchor_manifest_path": f"/anchors/{anchor:02d}.json",
                "config_digest": "9" * 64,
                "execution_purpose": "v02_freeze_ready",
                "seed": seed,
                "training_protocol": {
                    "export_outer_iterations": [6, 12],
                    "parity": {
                        "atol": 1.0e-6,
                        "rtol": 1.0e-6,
                        "golden_sample_count": 8,
                        "compiled_sample_count": 2,
                    },
                },
            }
    return result


def _states(jobs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    failed = set(sorted(jobs)[-6:])
    rows = {
        job_id: {
            "state": "failed" if job_id in failed else "succeeded",
            "attempts": 3 if job_id in failed else 1,
        }
        for job_id in jobs
    }
    return rows, failed


def test_failure_parser_accepts_only_action_parity_with_equal_prng_keys() -> None:
    message = module.FAILURE_PREFIX + str(
        {
            "passed": False,
            "max_abs_error": 1.2e-6,
            "atol": 1.0e-6,
            "rtol": 1.0e-6,
            "sample_count": 2,
            "next_keys_equal": True,
        }
    )
    assert module._failed_payload(message)["max_abs_error"] == 1.2e-6

    changed_key = message.replace("'next_keys_equal': True", "'next_keys_equal': False")
    with pytest.raises(ContractError, match="preserve PRNG keys"):
        module._failed_payload(changed_key)
    with pytest.raises(ContractError, match="reviewed reloaded"):
        module._failed_payload("some other ContractError")


def test_builder_derives_latest_attempt_from_count_without_attempt_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = _jobs()
    states, failed = _states(jobs)
    runs_root = tmp_path / "server_runs"
    runs_root.mkdir()
    (runs_root / "queue_status.json").write_text("{}\n", encoding="utf-8")
    for job_id in failed:
        attempt_dir = runs_root / "jobs" / job_id / "attempt_003"
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "attempt_manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_canonical_job_roots",
        lambda _plan, _root: (jobs, {"plan_digest": "a" * 64}),
    )
    monkeypatch.setattr(
        module,
        "_queue_authority",
        lambda _root, _plan, _ids: ({"jobs": states}, {}, {}),
    )
    monkeypatch.setattr(
        module,
        "validate_attempt",
        lambda _value: {
            "attempt_number": 3,
            "attempt_digest": "b" * 64,
        },
    )

    def evidence(attempt_dir: Path) -> dict[str, Any]:
        return {
            "traceback_path": attempt_dir / "traceback.txt",
            "traceback_sha256": "c" * 64,
            "events_path": attempt_dir / "events.jsonl",
            "events_sha256": "d" * 64,
            "failure_message": module.FAILURE_PREFIX + "fixture",
            "failed_compiled_parity": {
                "passed": False,
                "max_abs_error": 1.2e-6,
                "atol": 1.0e-6,
                "rtol": 1.0e-6,
                "sample_count": 2,
                "next_keys_equal": True,
            },
            "checkpoints": [
                {
                    "outer_iteration": 6,
                    "bundle_digest": "e" * 64,
                }
            ],
        }

    monkeypatch.setattr(module, "_failure_evidence", evidence)
    manifest = module.build_compiled_parity_promotion_manifest(
        server_plan={}, runs_root=runs_root
    )
    assert manifest["promotion_count"] == 6
    assert set(manifest["entries"]) == failed
    assert all(
        Path(entry["attempt_dir"]).name == "attempt_003"
        for entry in manifest["entries"].values()
    )


def _stub_full_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, dict[str, Any], set[str]]:
    jobs = _jobs()
    states, failed = _states(jobs)
    runs_root = tmp_path / "server_runs"
    runs_root.mkdir()
    (runs_root / "queue_status.json").write_text("{}\n", encoding="utf-8")
    plan = {"plan_digest": "a" * 64}
    for job_id, state in states.items():
        number = state["attempts"]
        attempt_dir = runs_root / "jobs" / job_id / f"attempt_{number:03d}"
        attempt_dir.mkdir(parents=True)
        state["attempt_dir"] = str(attempt_dir.resolve())
        if state["state"] == "succeeded":
            state["terminal_record_state"] = "recovered"
        attempt = {
            "attempt_number": number,
            "attempt_digest": "b" * 64,
            "plan_digest": plan["plan_digest"],
            "job_digest": jobs[job_id]["job_digest"],
            "job": jobs[job_id],
        }
        (attempt_dir / "attempt_manifest.json").write_text(
            json.dumps(attempt) + "\n", encoding="utf-8"
        )
        (attempt_dir / "queue_result.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_canonical_job_roots",
        lambda _plan, _root, **_kwargs: (jobs, plan),
    )
    monkeypatch.setattr(
        module,
        "_queue_authority",
        lambda _root, _plan, _ids: ({"jobs": states}, {}, {}),
    )
    monkeypatch.setattr(
        module,
        "validate_attempt",
        lambda value: value,
    )
    monkeypatch.setattr(
        module,
        "validate_queue_result",
        lambda *_args, **_kwargs: {
            "state": "succeeded",
            "returncode": 0,
            "vendor": {},
            "implementation": {},
        },
    )
    monkeypatch.setattr(
        module, "validate_vendor_provenance", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(
        module, "validate_implementation_provenance", lambda value, **_kwargs: value
    )

    checkpoint = {
        "path": "/bundle",
        "bundle_digest": "c" * 64,
        "outer_iteration": 6,
        "environment_steps": 128,
        "finiteness_audit": {"report_digest": "d" * 64},
        "golden_parity": {"report_digest": "e" * 64},
        "compiled_parity": {"report_digest": "f" * 64},
    }
    monkeypatch.setattr(
        module,
        "validate_completed_attempt",
        lambda *_args, **_kwargs: {
            "checkpoint_bundles": [checkpoint],
            "record_digest": "1" * 64,
            "state": "recovered",
        },
    )

    class FakeAnchor:
        @classmethod
        def from_path(cls, path: str) -> Any:
            anchor = int(Path(path).stem)
            return SimpleNamespace(anchor_id=f"{anchor + 2000:064x}")

    monkeypatch.setattr(module, "AnchorManifest", FakeAnchor)

    def promoted(
        *, entry: dict[str, Any], job: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        anchor = int(Path(job["anchor_manifest_path"]).stem)
        return {
            "resolution": "compiled_parity_fallback_promotion",
            "job_id": job["job_id"],
            "job_digest": job["job_digest"],
            "source_anchor_id": f"{anchor + 2000:064x}",
            "seed": job["seed"],
            "attempt_number": 3,
            "attempt_digest": "2" * 64,
            "bundle_path": "/promoted",
            "bundle_digest": "3" * 64,
            "outer_iteration": 6,
            "environment_steps": 128,
            "finiteness_audit_digest": "4" * 64,
            "golden_parity_digest": "5" * 64,
            "compiled_parity_digest": "6" * 64,
            "promotion_entry_digest": "7" * 64,
            "failure_trace_digest": "8" * 64,
            "failed_compiled_parity": {},
        }

    monkeypatch.setattr(module, "_validate_promotion", promoted)
    manifest = with_self_digest(
        {
            "schema": module.PROMOTION_SCHEMA,
            "server_plan_digest": plan["plan_digest"],
            "queue_status_sha256": sha256_file(runs_root / "queue_status.json"),
            "policy": "last_canonical_checkpoint_before_reloaded_compiled_parity_failure",
            "promotion_count": 6,
            "created_at": "fixture",
            "entries": {job_id: {"job_id": job_id} for job_id in sorted(failed)},
        },
        key="manifest_digest",
    )
    return plan, runs_root, manifest, failed


def test_acceptance_emits_exact_90_self_digested_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, runs_root, manifest, _failed = _stub_full_acceptance(tmp_path, monkeypatch)
    report = module.accept_policy_pool(
        server_plan=plan,
        runs_root=runs_root,
        promotion_manifest=manifest,
    )
    assert report["decision"] == "PASS"
    assert report["job_count"] == 90
    assert report["anchor_count"] == 30
    assert report["direct_terminal_record_count"] == 84
    assert report["compiled_parity_fallback_promotion_count"] == 6
    assert len(report["cells"]) == 90
    module.validate_self_digest(
        report, key="report_digest", where="fixture acceptance report"
    )


def test_acceptance_rejects_coverage_and_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, runs_root, manifest, failed = _stub_full_acceptance(tmp_path, monkeypatch)
    missing = dict(manifest)
    entries = dict(missing["entries"])
    entries.pop(next(iter(failed)))
    missing["entries"] = entries
    missing["promotion_count"] = 5
    missing = with_self_digest(
        {key: value for key, value in missing.items() if key != "manifest_digest"},
        key="manifest_digest",
    )
    with pytest.raises(ContractError, match="frozen 6-cell overlay"):
        module.accept_policy_pool(
            server_plan=plan,
            runs_root=runs_root,
            promotion_manifest=missing,
        )

    tampered = dict(manifest)
    tampered["created_at"] = "tampered-without-redigest"
    with pytest.raises(ContractError, match="digest mismatch"):
        module.accept_policy_pool(
            server_plan=plan,
            runs_root=runs_root,
            promotion_manifest=tampered,
        )
