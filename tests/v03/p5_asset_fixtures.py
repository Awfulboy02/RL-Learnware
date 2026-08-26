from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from policy_learnware_v0.hashing import sha256_file, sha256_json
from policy_learnware_v0.v02.schemas import SourceAnchorRecord
from policy_learnware_v0.v03.pool_intake import V02HandoffTrustAnchor


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def fixture_source_anchor(index: int) -> SourceAnchorRecord:
    return SourceAnchorRecord.create(
        environment_instance_digest=digest(f"source-environment:{index}"),
        axis_binding_digest=None,
    )


def self_digest(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = sha256_json(result)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def refresh_handoff_trust(
    root: Path,
    handoff: Path,
    *,
    plan_digest: str,
    queue_digest: str,
) -> V02HandoffTrustAnchor:
    promotion_path = handoff / "compiled_parity_promotions.json"
    acceptance_path = handoff / "policy_pool_acceptance.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    return V02HandoffTrustAnchor(
        server_plan_digest=plan_digest,
        queue_status_sha256=queue_digest,
        promotions_file_sha256=sha256_file(promotion_path),
        promotions_manifest_digest=promotion["manifest_digest"],
        acceptance_file_sha256=sha256_file(acceptance_path),
        acceptance_report_digest=acceptance["report_digest"],
        pool_digest=acceptance["pool_digest"],
    )


def exact90_handoff(tmp_path: Path) -> tuple[Path, Path, V02HandoffTrustAnchor]:
    root = tmp_path / "experiment"
    handoff = root / "policy_pool_handoff_fixture"
    handoff.mkdir(parents=True)
    config_digest = digest("formal-config")
    protocol = self_digest(
        {"export_outer_iterations": [6, 12]},
        "protocol_digest",
    )
    anchor_rows: list[tuple[SourceAnchorRecord, Path, str]] = []
    for anchor_index in range(30):
        anchor_record = fixture_source_anchor(anchor_index)
        runtime = {"fixture": "v0.2"}
        anchor_manifest = self_digest(
            {
                "schema": "policy-learnware.v02-anchor-manifest.v0",
                "anchor_id": anchor_record.anchor_id,
                "environment_instance_digest": anchor_record.environment_instance_digest,
                "axis_binding_digest": anchor_record.axis_binding_digest,
                "runtime": runtime,
                "runtime_digest": sha256_json(runtime),
            },
            "manifest_digest",
        )
        anchor_path = root / "source_anchor_manifests" / f"{anchor_record.anchor_id}.json"
        write_json(anchor_path, anchor_manifest)
        anchor_rows.append(
            (anchor_record, anchor_path.resolve(), anchor_manifest["manifest_digest"])
        )

    freeze = self_digest(
        {
            "schema": "policy-learnware.v02-formal-freeze-binding.v0",
            "training_contract": {
                "source_anchor_ids": sorted(row[0].anchor_id for row in anchor_rows)
            },
        },
        "binding_digest",
    )
    jobs: list[dict[str, Any]] = []
    job_context: list[tuple[int, SourceAnchorRecord, dict[str, Any]]] = []
    for anchor_index, (anchor_record, anchor_path, anchor_digest) in enumerate(anchor_rows):
        for seed in range(3):
            index = anchor_index * 3 + seed
            job_id = f"v02j-{index + 1:024x}"
            job = self_digest(
                {
                    "schema": "policy-learnware.v02-training-job.v0",
                    "job_id": job_id,
                    "config_digest": config_digest,
                    "execution_purpose": "v02_freeze_ready",
                    "formal_protocol_freeze_digest": freeze["binding_digest"],
                    "anchor_manifest_path": str(anchor_path),
                    "anchor_manifest_digest": anchor_digest,
                    "training_protocol": protocol,
                    "training_protocol_digest": protocol["protocol_digest"],
                    "seed": seed,
                },
                "job_digest",
            )
            jobs.append(job)
            job_context.append((index, anchor_record, job))

    plan_path = root / "training_private" / "plans" / "server_training_plan.json"
    plan = self_digest(
        {
            "schema": "policy-learnware.v02-training-plan.v0",
            "config_digest": config_digest,
            "execution_purpose": "v02_freeze_ready",
            "formal_protocol_freeze": freeze,
            "formal_protocol_freeze_digest": freeze["binding_digest"],
            "jobs": jobs,
            "expected_job_count": 90,
        },
        "plan_digest",
    )
    write_json(plan_path, plan)
    plan_digest = plan["plan_digest"]

    cells: dict[str, Any] = {}
    promotion_entries: dict[str, Any] = {}
    queue_jobs: dict[str, Any] = {}

    for index, anchor_record, job in job_context:
        job_id = job["job_id"]
        seed = job["seed"]
        promoted = index >= 84
        attempt_number = 3 if promoted else 1
        outer = 6
        steps = 128 * (index + 1)
        attempt = (
            root
            / "training_private"
            / "server_runs"
            / "jobs"
            / job_id
            / f"attempt_{attempt_number:03d}"
        )
        bundle = attempt / "checkpoints" / f"outer_{outer:06d}"
        bundle.mkdir(parents=True)
        payload = f"fixture-policy:{job_id}".encode("utf-8")
        (bundle / "actor.npz").write_bytes(payload)
        bundle_manifest = {
            "schema": "policy-learnware.policy-bundle.v0",
            "complete": True,
            "task": f"FixtureTask{index // 15}",
            "algorithm": "fpo",
            "seed": seed,
            "outer_iteration": outer,
            "environment_steps": steps,
            "created_at": "2026-08-26T00:00:00+00:00",
            "files": {
                "actor.npz": {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            },
        }
        write_json(bundle / "bundle_manifest.json", bundle_manifest)
        bundle_digest = sha256_file(bundle / "bundle_manifest.json")
        attempt_manifest = self_digest(
            {
                "schema": "policy-learnware.v02-training-attempt.v0",
                "attempt_number": attempt_number,
                "job_digest": job["job_digest"],
                "plan_digest": plan_digest,
                "execution_purpose": "v02_freeze_ready",
                "execution_mode": "formal_gpu",
                "formal_eligible": True,
                "job": job,
            },
            "attempt_digest",
        )
        attempt_digest = attempt_manifest["attempt_digest"]
        write_json(attempt / "attempt_manifest.json", attempt_manifest)

        finite = self_digest(
            {"passed": True, "all_arrays_finite": True}, "report_digest"
        )
        golden = self_digest(
            {"passed": True, "raw_checked": True}, "report_digest"
        )
        compiled = self_digest(
            {"passed": True, "next_keys_equal": True}, "report_digest"
        )
        checkpoint = {
            "event": "checkpoint_published",
            "outer_iteration": outer,
            "environment_steps": steps,
            "path": str(bundle.resolve()),
            "bundle_digest": bundle_digest,
            "finiteness_audit": finite,
            "golden_parity": golden,
            "compiled_parity": compiled,
        }
        base = {
            "resolution": (
                "compiled_parity_fallback_promotion"
                if promoted
                else "direct_terminal_record"
            ),
            "job_id": job_id,
            "job_digest": job["job_digest"],
            "source_anchor_id": anchor_record.anchor_id,
            "seed": seed,
            "attempt_number": attempt_number,
            "attempt_digest": attempt_digest,
            "bundle_path": str(bundle.resolve()),
            "bundle_digest": bundle_digest,
            "outer_iteration": outer,
            "environment_steps": steps,
            "finiteness_audit_digest": finite["report_digest"],
            "golden_parity_digest": golden["report_digest"],
            "compiled_parity_digest": compiled["report_digest"],
        }
        if not promoted:
            terminal_state = "recovered" if index < 57 else "succeeded"
            training_record = self_digest(
                {
                    "schema": "policy-learnware.v02-training-record.v1",
                    "state": terminal_state,
                    "job_digest": job["job_digest"],
                    "attempt_digest": attempt_digest,
                    "anchor_manifest_digest": job["anchor_manifest_digest"],
                    "training_protocol_digest": job["training_protocol_digest"],
                    "seed": seed,
                    "promoted_outer_iteration": outer,
                    "promoted_environment_steps": steps,
                    "checkpoint_bundles": [checkpoint],
                },
                "record_digest",
            )
            write_json(attempt / "training_record.json", training_record)
            cells[job_id] = {
                **base,
                "training_record_digest": training_record["record_digest"],
                "terminal_record_state": terminal_state,
            }
            queue_jobs[job_id] = {
                "attempt_dir": str(attempt.resolve()),
                "attempts": attempt_number,
                "state": "succeeded",
                "terminal_record_state": terminal_state,
            }
            continue

        failed_payload = {
            "passed": False,
            "max_abs_error": 1.2e-6,
            "atol": 1.0e-6,
            "rtol": 1.0e-6,
            "sample_count": 2,
            "next_keys_equal": True,
        }
        failure_message = (
            "reloaded compiled-policy parity failed: " + repr(failed_payload)
        )
        failure_event = {
            "event": "run_failed",
            "error_type": "ContractError",
            "error": failure_message,
            "state": "failed",
            "job_digest": job["job_digest"],
            "attempt_digest": attempt_digest,
            "last_completed_outer": 11,
        }
        events_path = attempt / "events.jsonl"
        events_path.write_text(
            json.dumps(checkpoint, sort_keys=True)
            + "\n"
            + json.dumps(failure_event, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        traceback_path = attempt / "traceback.txt"
        traceback_path.write_text(
            "Traceback fixture\nprovenance.ContractError: "
            + failure_message
            + "\n",
            encoding="utf-8",
        )
        entry = self_digest(
            {
                "job_id": job_id,
                "job_digest": job["job_digest"],
                "attempt_number": attempt_number,
                "attempt_digest": attempt_digest,
                "attempt_dir": str(attempt.resolve()),
                "promoted_outer_iteration": outer,
                "promoted_bundle_digest": bundle_digest,
                "failed_outer_iteration": 12,
                "failure_code": "RELOADED_COMPILED_PARITY_FAILED",
                "failure_type": "ContractError",
                "failure_message": failure_message,
                "failed_compiled_parity": failed_payload,
                "traceback_path": str(traceback_path.resolve()),
                "traceback_sha256": sha256_file(traceback_path),
                "events_path": str(events_path.resolve()),
                "events_sha256": sha256_file(events_path),
            },
            "entry_digest",
        )
        promotion_entries[job_id] = entry
        cells[job_id] = {
            **base,
            "promotion_entry_digest": entry["entry_digest"],
            "failure_trace_digest": entry["traceback_sha256"],
            "failed_compiled_parity": failed_payload,
        }
        queue_jobs[job_id] = {
            "attempt_dir": str(attempt.resolve()),
            "attempts": attempt_number,
            "returncode": 1,
            "state": "failed",
            "terminal_record_state": None,
            "validation_error": None,
        }

    queue_path = root / "training_private" / "server_runs" / "queue_status.json"
    queue = {
        "schema": "policy-learnware.v02-queue-status.v0",
        "plan": str(plan_path.resolve()),
        "plan_digest": plan_digest,
        "config_digest": config_digest,
        "execution_purpose": "v02_freeze_ready",
        "formal_protocol_freeze_digest": freeze["binding_digest"],
        "execution_mode": "formal_gpu",
        "formal_eligible": True,
        "gpu_resource_gate": {"enabled": True},
        "gpus": ["0"],
        "master_pid": 1,
        "started_at": "2026-08-26T00:00:00+00:00",
        "updated_at": "2026-08-26T01:00:00+00:00",
        "state": "completed_with_failures",
        "stop_signal": None,
        "vendor": {"fixture": True},
        "implementation": {"fixture": True},
        "jobs": dict(sorted(queue_jobs.items())),
        "running": [],
        "counts": {"failed": 6, "succeeded": 84},
        "terminal_record_counts": {"recovered": 57, "succeeded": 27},
    }
    write_json(queue_path, queue)
    queue_digest = sha256_file(queue_path)

    promotion = self_digest(
        {
            "schema": "policy-learnware.v02-compiled-parity-promotion-set.v0",
            "server_plan_digest": plan_digest,
            "queue_status_sha256": queue_digest,
            "policy": "last_canonical_checkpoint_before_reloaded_compiled_parity_failure",
            "promotion_count": 6,
            "created_at": "2026-08-26T00:00:00+00:00",
            "entries": dict(sorted(promotion_entries.items())),
        },
        "manifest_digest",
    )
    write_json(handoff / "compiled_parity_promotions.json", promotion)
    pool_digest = sha256_json(
        {
            job_id: {
                "job_digest": cell["job_digest"],
                "source_anchor_id": cell["source_anchor_id"],
                "seed": cell["seed"],
                "bundle_digest": cell["bundle_digest"],
                "outer_iteration": cell["outer_iteration"],
                "environment_steps": cell["environment_steps"],
            }
            for job_id, cell in sorted(cells.items())
        }
    )
    acceptance = self_digest(
        {
            "schema": "policy-learnware.v02-policy-pool-acceptance.v0",
            "decision": "PASS",
            "accepted_at": "2026-08-26T00:00:00+00:00",
            "server_plan_digest": plan_digest,
            "queue_status_sha256": queue_digest,
            "promotion_manifest_digest": promotion["manifest_digest"],
            "job_count": 90,
            "anchor_count": 30,
            "seeds": [0, 1, 2],
            "direct_terminal_record_count": 84,
            "compiled_parity_fallback_promotion_count": 6,
            "all_selected_bundles_finite": True,
            "all_selected_bundles_golden_parity_passed": True,
            "all_selected_bundles_compiled_parity_passed": True,
            "pool_digest": pool_digest,
            "cells": dict(sorted(cells.items())),
        },
        "report_digest",
    )
    write_json(handoff / "policy_pool_acceptance.json", acceptance)
    trust = refresh_handoff_trust(
        root, handoff, plan_digest=plan_digest, queue_digest=queue_digest
    )
    return root, handoff, trust
