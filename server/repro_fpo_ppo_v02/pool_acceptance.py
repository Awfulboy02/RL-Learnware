"""Append-only acceptance for the frozen v0.2 90-cell policy pool.

The formal runner deliberately fails before writing ``training_record.json``
when a newly exported checkpoint misses compiled-policy parity.  For the one
reviewed v0.2 pool, six such failures have an immediately preceding immutable
checkpoint that already passed finiteness, golden parity, and compiled parity.

This module never blesses the failed checkpoint and never changes the runner
or tolerance.  It accepts exactly 84 ordinary terminal records with the
existing queue validator plus six explicit, self-digested promotion entries.
Every promoted entry is rebound to the failed traceback/event bytes and to the
last canonical, parity-passing checkpoint in that attempt.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .anchor_binding import AnchorManifest
from .handoff_contracts import (
    _validate_checkpoint_bytes,
    derive_iterations_per_env,
    validate_completed_attempt,
)
from .provenance import (
    ContractError,
    FORMAL_EXECUTION_PURPOSE,
    FORMAL_GPU_EXECUTION_MODE,
    load_strict_json,
    require_exact_keys,
    sha256_file,
    sha256_json,
    utc_now,
    validate_attempt,
    validate_execution_evidence,
    validate_fpo_source_attestation,
    validate_implementation_provenance,
    validate_queue_result,
    validate_run_manifest_server_binding,
    validate_self_digest,
    validate_training_job,
    validate_training_plan,
    validate_vendor_provenance,
    with_self_digest,
)


PROMOTION_SCHEMA = "policy-learnware.v02-compiled-parity-promotion-set.v0"
ACCEPTANCE_SCHEMA = "policy-learnware.v02-policy-pool-acceptance.v0"
FAILURE_PREFIX = "reloaded compiled-policy parity failed: "
FAILURE_CODE = "RELOADED_COMPILED_PARITY_FAILED"
EXPECTED_DIRECT_COUNT = 84
EXPECTED_PROMOTION_COUNT = 6
EXPECTED_JOB_COUNT = 90
EXPECTED_ANCHOR_COUNT = 30
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_TERMINAL_RECORD_COUNTS = {"recovered": 57, "succeeded": 27}

RecordedPathResolver = Callable[[str | Path], Path]

_CHECKPOINT_EVENT_KEYS = {
    "event",
    "at",
    "outer_iteration",
    "environment_steps",
    "path",
    "bundle_manifest_sha256",
    "bundle_manifest_digest",
    "files",
    "bundle_digest",
    "config_digest",
    "execution_purpose",
    "execution_mode",
    "formal_eligible",
    "execution_evidence_digest",
    "finiteness_audit",
    "golden_parity",
    "compiled_parity",
}
_PROMOTION_ENTRY_KEYS = {
    "job_id",
    "job_digest",
    "attempt_number",
    "attempt_digest",
    "attempt_dir",
    "promoted_outer_iteration",
    "promoted_bundle_digest",
    "failed_outer_iteration",
    "failure_code",
    "failure_type",
    "failure_message",
    "failed_compiled_parity",
    "traceback_path",
    "traceback_sha256",
    "events_path",
    "events_sha256",
    "entry_digest",
}


def _physical_path(
    value: str | Path, resolver: RecordedPathResolver | None
) -> Path:
    return Path(value).resolve() if resolver is None else Path(resolver(value)).resolve()


def _strict_json_line(line: str, *, where: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(line, object_pairs_hook=no_duplicates)
    except (json.JSONDecodeError, ValueError) as error:
        if isinstance(error, ContractError):
            raise
        raise ContractError(f"cannot parse {where}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be a JSON object")
    return value


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ContractError(f"event log is missing: {path}")
    rows = [
        _strict_json_line(line, where=f"{path}:line {index}")
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.strip()
    ]
    if not rows:
        raise ContractError("event log is empty")
    return rows


def _failed_payload(message: str) -> dict[str, Any]:
    if not isinstance(message, str) or not message.startswith(FAILURE_PREFIX):
        raise ContractError(
            "failure is not the reviewed reloaded compiled-parity class"
        )
    try:
        value = ast.literal_eval(message[len(FAILURE_PREFIX) :])
    except (SyntaxError, ValueError) as error:
        raise ContractError(
            "compiled-parity failure payload is not a literal mapping"
        ) from error
    if not isinstance(value, dict) or set(value) != {
        "passed",
        "max_abs_error",
        "atol",
        "rtol",
        "sample_count",
        "next_keys_equal",
    }:
        raise ContractError("compiled-parity failure payload inventory is not exact")
    if value["passed"] is not False or value["next_keys_equal"] is not True:
        raise ContractError(
            "compiled-parity failure must preserve PRNG keys and fail actions only"
        )
    for key in ("max_abs_error", "atol", "rtol"):
        raw = value[key]
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            raise ContractError(f"compiled-parity failure {key} must be finite")
        value[key] = float(raw)
    if (
        isinstance(value["sample_count"], bool)
        or not isinstance(value["sample_count"], int)
        or value["sample_count"] <= 0
    ):
        raise ContractError("compiled-parity failure sample_count must be positive")
    if value["max_abs_error"] <= 0.0:
        raise ContractError("compiled-parity failure error must be positive")
    return value


def _canonical_job_roots(
    plan: Mapping[str, Any],
    runs_root: Path,
    path_resolver: RecordedPathResolver | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    validated = validate_training_plan(plan)
    jobs = {job["job_id"]: validate_training_job(job) for job in validated["jobs"]}
    if len(jobs) != len(validated["jobs"]):
        raise ContractError("server plan contains duplicate job IDs")
    if len(jobs) != EXPECTED_JOB_COUNT:
        raise ContractError(
            f"v0.2 policy-pool acceptance requires exactly {EXPECTED_JOB_COUNT} jobs"
        )
    for job in jobs.values():
        anchor = AnchorManifest.from_path(
            _physical_path(job["anchor_manifest_path"], path_resolver)
        )
        if anchor.manifest_digest != job["anchor_manifest_digest"]:
            raise ContractError(
                "live anchor manifest differs from the frozen server job"
            )
    # The server job intentionally stores only the immutable anchor-manifest
    # digest/path.  The package-side source anchor ID is rederived from those
    # bytes later; the manifest digest is the correct server-grid identity.
    units = {(job["anchor_manifest_digest"], job["seed"]) for job in jobs.values()}
    anchors = sorted({job["anchor_manifest_digest"] for job in jobs.values()})
    seeds = sorted({job["seed"] for job in jobs.values()})
    expected = {(anchor, seed) for anchor in anchors for seed in EXPECTED_SEEDS}
    if (
        len(anchors) != EXPECTED_ANCHOR_COUNT
        or seeds != list(EXPECTED_SEEDS)
        or units != expected
    ):
        raise ContractError(
            "server plan is not the frozen exact 30-anchor x seeds-0/1/2 grid"
        )
    root = runs_root.resolve()
    if root.name != "server_runs" or not root.is_dir():
        raise ContractError("runs root must be the canonical server_runs directory")
    for job_id in jobs:
        expected_root = root / "jobs" / job_id
        if not expected_root.is_dir():
            raise ContractError(f"canonical job root is missing: {job_id}")
    return jobs, validated


def _queue_authority(
    runs_root: Path, plan: Mapping[str, Any], job_ids: Iterable[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    status = load_strict_json(runs_root / "queue_status.json")
    required = {
        "schema",
        "state",
        "plan",
        "plan_digest",
        "config_digest",
        "execution_purpose",
        "formal_protocol_freeze_digest",
        "started_at",
        "updated_at",
        "master_pid",
        "stop_signal",
        "gpus",
        "gpu_resource_gate",
        "execution_mode",
        "formal_eligible",
        "vendor",
        "implementation",
        "counts",
        "terminal_record_counts",
        "running",
        "jobs",
    }
    require_exact_keys(status, required, "queue status")
    if (
        status["schema"] != "policy-learnware.v02-queue-status.v0"
        or status["state"] != "completed_with_failures"
        or status["plan_digest"] != plan["plan_digest"]
        or status["config_digest"] != plan["config_digest"]
        or status["formal_protocol_freeze_digest"]
        != plan["formal_protocol_freeze_digest"]
        or status["execution_purpose"] != FORMAL_EXECUTION_PURPOSE
        or status["execution_mode"] != FORMAL_GPU_EXECUTION_MODE
        or status["formal_eligible"] is not True
        or status["running"] != []
        or status["counts"]
        != {"failed": EXPECTED_PROMOTION_COUNT, "succeeded": EXPECTED_DIRECT_COUNT}
        or status["terminal_record_counts"] != EXPECTED_TERMINAL_RECORD_COUNTS
    ):
        raise ContractError("queue status is not the completed frozen formal execution")
    if not isinstance(status["jobs"], Mapping) or set(status["jobs"]) != set(job_ids):
        raise ContractError("queue status job coverage differs from the immutable plan")
    observed_counts = Counter(
        state.get("state")
        for state in status["jobs"].values()
        if isinstance(state, Mapping)
    )
    if observed_counts != Counter(status["counts"]):
        raise ContractError("queue status counts differ from its terminal job states")
    vendor = validate_vendor_provenance(status["vendor"])
    implementation = validate_implementation_provenance(status["implementation"])
    return status, vendor, implementation


def _attempt_from_entry(
    *,
    entry: Mapping[str, Any],
    job: Mapping[str, Any],
    runs_root: Path,
    path_resolver: RecordedPathResolver | None = None,
) -> tuple[Path, dict[str, Any]]:
    expected_dir = (
        runs_root / "jobs" / job["job_id"] / f"attempt_{entry['attempt_number']:03d}"
    ).resolve()
    supplied = _physical_path(entry["attempt_dir"], path_resolver)
    if (
        supplied != expected_dir
        or supplied.name != f"attempt_{entry['attempt_number']:03d}"
    ):
        raise ContractError("promotion attempt path is not canonical")
    attempt = validate_attempt(load_strict_json(supplied / "attempt_manifest.json"))
    if (
        attempt["attempt_number"] != entry["attempt_number"]
        or attempt["attempt_digest"] != entry["attempt_digest"]
        or attempt["job_digest"] != job["job_digest"]
        or attempt["job"] != job
    ):
        raise ContractError("promotion entry is bound to another immutable attempt")
    return supplied, attempt


def _final_attempt_dir(
    *,
    state: Mapping[str, Any],
    job_id: str,
    runs_root: Path,
    path_resolver: RecordedPathResolver | None = None,
) -> Path:
    attempts = state.get("attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ContractError("final queue state has no positive attempt count")
    expected = (runs_root / "jobs" / job_id / f"attempt_{attempts:03d}").resolve()
    supplied = state.get("attempt_dir")
    if supplied is not None and _physical_path(supplied, path_resolver) != expected:
        raise ContractError("final queue attempt_dir differs from its attempt count")
    if not expected.is_dir():
        raise ContractError("final canonical attempt directory is missing")
    return expected


def _failure_evidence(attempt_dir: Path) -> dict[str, Any]:
    traceback_path = attempt_dir / "traceback.txt"
    events_path = attempt_dir / "events.jsonl"
    trace = traceback_path.read_text(encoding="utf-8")
    final_line = next(
        (line for line in reversed(trace.splitlines()) if line.strip()), ""
    )
    expected_prefix = "provenance.ContractError: " + FAILURE_PREFIX
    if not final_line.startswith(expected_prefix):
        raise ContractError(
            "traceback final exception is not exact compiled-policy parity"
        )
    failure_message = final_line[len("provenance.ContractError: ") :]
    payload = _failed_payload(failure_message)
    events = _load_events(events_path)
    failures = [row for row in events if row.get("event") == "run_failed"]
    if len(failures) != 1 or events[-1] != failures[0]:
        raise ContractError("event log must end in exactly one run_failed event")
    failure = failures[0]
    if (
        failure.get("error_type") != "ContractError"
        or failure.get("error") != failure_message
        or _failed_payload(failure["error"]) != payload
        or failure.get("traceback_file") != "traceback.txt"
        or failure.get("state") != "failed"
    ):
        raise ContractError("run_failed event differs from the immutable traceback")
    checkpoints = [row for row in events if row.get("event") == "checkpoint_published"]
    if not checkpoints:
        raise ContractError("compiled-parity failure has no prior published checkpoint")
    return {
        "traceback_path": traceback_path.resolve(),
        "traceback_sha256": sha256_file(traceback_path),
        "events_path": events_path.resolve(),
        "events_sha256": sha256_file(events_path),
        "failure_message": failure_message,
        "failed_compiled_parity": payload,
        "failure_event": failure,
        "checkpoints": checkpoints,
    }


def _expected_environment_steps(job: Mapping[str, Any], outer: int) -> int:
    config = job["training_protocol"]["trainer_config"]
    num_envs = config["num_envs"]
    return int(num_envs) * derive_iterations_per_env(config) * outer


def build_compiled_parity_promotion_manifest(
    *, server_plan: Mapping[str, Any], runs_root: Path
) -> dict[str, Any]:
    """Create an explicit manifest for exactly the six reviewed failed cells."""

    jobs, plan = _canonical_job_roots(server_plan, runs_root)
    status, _vendor, _implementation = _queue_authority(runs_root, plan, jobs)
    failed_ids = sorted(
        job_id
        for job_id, state in status["jobs"].items()
        if state.get("state") == "failed"
    )
    direct_ids = sorted(set(jobs) - set(failed_ids))
    if (
        len(failed_ids) != EXPECTED_PROMOTION_COUNT
        or len(direct_ids) != EXPECTED_DIRECT_COUNT
    ):
        raise ContractError(
            "promotion manifest is restricted to the reviewed 84+6 terminal split"
        )
    entries: dict[str, Any] = {}
    for job_id in failed_ids:
        job = jobs[job_id]
        state = status["jobs"][job_id]
        attempt_dir = _final_attempt_dir(
            state=state, job_id=job_id, runs_root=runs_root.resolve()
        )
        attempt = validate_attempt(
            load_strict_json(attempt_dir / "attempt_manifest.json")
        )
        evidence = _failure_evidence(attempt_dir)
        checkpoints = evidence["checkpoints"]
        promoted = checkpoints[-1]
        export_outers = job["training_protocol"]["export_outer_iterations"]
        promoted_outer = promoted["outer_iteration"]
        later = [outer for outer in export_outers if outer > promoted_outer]
        if not later:
            raise ContractError(
                "failed parity export has no successor to the promoted checkpoint"
            )
        failed_outer = later[0]
        entry = with_self_digest(
            {
                "job_id": job_id,
                "job_digest": job["job_digest"],
                "attempt_number": attempt["attempt_number"],
                "attempt_digest": attempt["attempt_digest"],
                "attempt_dir": str(attempt_dir),
                "promoted_outer_iteration": promoted_outer,
                "promoted_bundle_digest": promoted["bundle_digest"],
                "failed_outer_iteration": failed_outer,
                "failure_code": FAILURE_CODE,
                "failure_type": "ContractError",
                "failure_message": evidence["failure_message"],
                "failed_compiled_parity": evidence["failed_compiled_parity"],
                "traceback_path": str(evidence["traceback_path"]),
                "traceback_sha256": evidence["traceback_sha256"],
                "events_path": str(evidence["events_path"]),
                "events_sha256": evidence["events_sha256"],
            },
            key="entry_digest",
        )
        entries[job_id] = entry
    return with_self_digest(
        {
            "schema": PROMOTION_SCHEMA,
            "server_plan_digest": plan["plan_digest"],
            "queue_status_sha256": sha256_file(runs_root / "queue_status.json"),
            "policy": "last_canonical_checkpoint_before_reloaded_compiled_parity_failure",
            "promotion_count": len(entries),
            "created_at": utc_now(),
            "entries": dict(sorted(entries.items())),
        },
        key="manifest_digest",
    )


def _validate_promotion(
    *,
    entry: Mapping[str, Any],
    job: Mapping[str, Any],
    state: Mapping[str, Any],
    runs_root: Path,
    expected_vendor: Mapping[str, Any],
    expected_implementation: Mapping[str, Any],
    expected_plan_digest: str,
    path_resolver: RecordedPathResolver | None = None,
) -> dict[str, Any]:
    require_exact_keys(entry, _PROMOTION_ENTRY_KEYS, "compiled-parity promotion entry")
    validate_self_digest(
        entry, key="entry_digest", where="compiled-parity promotion entry"
    )
    if (
        entry["job_id"] != job["job_id"]
        or entry["job_digest"] != job["job_digest"]
        or entry["failure_code"] != FAILURE_CODE
        or entry["failure_type"] != "ContractError"
        or state.get("state") != "failed"
        or entry["attempt_number"] != state.get("attempts")
        or _final_attempt_dir(
            state=state,
            job_id=job["job_id"],
            runs_root=runs_root,
            path_resolver=path_resolver,
        )
        != _physical_path(entry["attempt_dir"], path_resolver)
    ):
        raise ContractError("promotion entry differs from the final failed queue cell")
    attempt_dir, attempt = _attempt_from_entry(
        entry=entry,
        job=job,
        runs_root=runs_root,
        path_resolver=path_resolver,
    )
    if attempt["plan_digest"] != expected_plan_digest:
        raise ContractError("promotion attempt differs from the frozen server plan")
    validate_implementation_provenance(
        attempt["implementation"], expected=expected_implementation
    )
    anchor = AnchorManifest.from_path(
        _physical_path(job["anchor_manifest_path"], path_resolver)
    )
    run = validate_run_manifest_server_binding(
        load_strict_json(attempt_dir / "run_manifest.json"),
        job=job,
        attempt=attempt,
        anchor=anchor.to_dict(),
    )
    runtime = run.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ContractError("promotion run runtime evidence is missing")
    validate_vendor_provenance(runtime.get("vendor"), expected=expected_vendor)
    validate_implementation_provenance(
        runtime.get("implementation"), expected=expected_implementation
    )
    source_attestation = validate_fpo_source_attestation(
        runtime, expected_commit=str(anchor.runtime["fpo_commit"])
    )
    execution = validate_execution_evidence(
        runtime.get("execution_evidence"),
        expected_job_digest=job["job_digest"],
        expected_attempt_digest=attempt["attempt_digest"],
        expected_hardware_digest=runtime["hardware_digest"],
        expected_config_digest=job["config_digest"],
        expected_execution_purpose=job["execution_purpose"],
        expected_attempt_root=Path(runtime["execution_evidence"]["attempt_root"]),
        require_formal=True,
    )
    if _physical_path(execution["attempt_root"], path_resolver) != attempt_dir.resolve():
        raise ContractError("promotion execution root does not resolve to this attempt")
    if (
        execution["execution_mode"] != FORMAL_GPU_EXECUTION_MODE
        or execution["formal_eligible"] is not True
    ):
        raise ContractError("promotion attempt is not formal GPU evidence")
    result = validate_queue_result(
        load_strict_json(attempt_dir / "queue_result.json"),
        expected_job_digest=job["job_digest"],
        expected_attempt_digest=attempt["attempt_digest"],
        expected_config_digest=job["config_digest"],
        expected_execution_purpose=job["execution_purpose"],
    )
    validate_vendor_provenance(result["vendor"], expected=expected_vendor)
    validate_implementation_provenance(
        result["implementation"], expected=expected_implementation
    )
    if result["state"] != "failed" or result["returncode"] == 0:
        raise ContractError("promotion attempt does not retain a failed queue result")
    if (
        set(state)
        != {
            "attempt_dir",
            "attempts",
            "returncode",
            "state",
            "terminal_record_state",
            "validation_error",
        }
        or state["returncode"] != result["returncode"]
        or state["terminal_record_state"] is not None
        or state["validation_error"] is not None
    ):
        raise ContractError("failed queue cell differs from its immutable queue result")

    evidence = _failure_evidence(attempt_dir)
    exact = {
        "failure_message": evidence["failure_message"],
        "failed_compiled_parity": evidence["failed_compiled_parity"],
        "traceback_sha256": evidence["traceback_sha256"],
        "events_sha256": evidence["events_sha256"],
    }
    if any(entry[key] != value for key, value in exact.items()):
        raise ContractError(
            "promotion entry failure evidence differs from immutable bytes"
        )
    for key, observed_key in (
        ("traceback_path", "traceback_path"),
        ("events_path", "events_path"),
    ):
        if _physical_path(entry[key], path_resolver) != evidence[observed_key]:
            raise ContractError(
                "promotion entry failure path does not resolve to immutable bytes"
            )
    parity = job["training_protocol"]["parity"]
    failed_payload = evidence["failed_compiled_parity"]
    if (
        failed_payload["atol"] != parity["atol"]
        or failed_payload["rtol"] != parity["rtol"]
        or failed_payload["sample_count"] != parity["compiled_sample_count"]
    ):
        raise ContractError(
            "failed compiled parity differs from frozen tolerance contract"
        )

    checkpoints = evidence["checkpoints"]
    observed_outers = [item.get("outer_iteration") for item in checkpoints]
    if observed_outers != sorted(observed_outers) or len(observed_outers) != len(
        set(observed_outers)
    ):
        raise ContractError("published checkpoint event order is not canonical")
    expected_prefix = job["training_protocol"]["export_outer_iterations"][
        : len(checkpoints)
    ]
    if observed_outers != expected_prefix:
        raise ContractError("published checkpoints are not the frozen ladder prefix")
    promoted = checkpoints[-1]
    if (
        promoted["outer_iteration"] != entry["promoted_outer_iteration"]
        or promoted["bundle_digest"] != entry["promoted_bundle_digest"]
    ):
        raise ContractError(
            "promotion entry does not select the last published checkpoint"
        )
    later = [
        outer
        for outer in job["training_protocol"]["export_outer_iterations"]
        if outer > promoted["outer_iteration"]
    ]
    if not later or entry["failed_outer_iteration"] != later[0]:
        raise ContractError("promotion failed outer is not the next frozen export")
    failure_event = evidence["failure_event"]
    if (
        failure_event.get("last_completed_outer") != entry["failed_outer_iteration"] - 1
        or failure_event.get("exported_outer_iterations") != observed_outers
        or failure_event.get("job_digest") != job["job_digest"]
        or failure_event.get("attempt_digest") != attempt["attempt_digest"]
    ):
        raise ContractError("run_failed event has inconsistent ladder provenance")

    for index, checkpoint in enumerate(checkpoints):
        require_exact_keys(
            checkpoint, _CHECKPOINT_EVENT_KEYS, f"checkpoint event {index}"
        )
        if checkpoint["bundle_digest"] != checkpoint["bundle_manifest_sha256"]:
            raise ContractError(
                "checkpoint bundle_digest must equal the manifest file SHA-256"
            )
        if checkpoint["environment_steps"] != _expected_environment_steps(
            job, checkpoint["outer_iteration"]
        ):
            raise ContractError(
                "checkpoint environment steps drifted from frozen geometry"
            )
        for name in ("finiteness_audit", "golden_parity", "compiled_parity"):
            report = checkpoint[name]
            if not isinstance(report, Mapping) or report.get("passed") is not True:
                raise ContractError(f"promoted ladder checkpoint {name} did not pass")
            validate_self_digest(
                report, key="report_digest", where=f"checkpoint {index} {name}"
            )
        if checkpoint["finiteness_audit"].get("all_arrays_finite") is not True:
            raise ContractError("promoted ladder checkpoint is not finite")
        if checkpoint["golden_parity"].get("raw_checked") is not True:
            raise ContractError("promoted ladder checkpoint lacks raw golden parity")
        if checkpoint["compiled_parity"].get("next_keys_equal") is not True:
            raise ContractError("promoted ladder checkpoint changed PRNG keys")
        if (
            checkpoint["golden_parity"].get("atol") != parity["atol"]
            or checkpoint["golden_parity"].get("rtol") != parity["rtol"]
            or checkpoint["golden_parity"].get("sample_count")
            != parity["golden_sample_count"]
            or checkpoint["compiled_parity"].get("atol") != parity["atol"]
            or checkpoint["compiled_parity"].get("rtol") != parity["rtol"]
            or checkpoint["compiled_parity"].get("sample_count")
            != parity["compiled_sample_count"]
        ):
            raise ContractError("promoted ladder checkpoint parity contract drifted")
        _validate_checkpoint_bytes(
            checkpoint=checkpoint,
            explicit_path=checkpoint["path"],
            server_job=job,
            attempt=attempt,
            anchor=anchor,
            execution=execution,
            source_attestation=source_attestation,
            attempt_root=attempt_dir,
            path_resolver=path_resolver,
        )
    return {
        "resolution": "compiled_parity_fallback_promotion",
        "job_id": job["job_id"],
        "job_digest": job["job_digest"],
        "source_anchor_id": anchor.anchor_id,
        "seed": job["seed"],
        "attempt_number": attempt["attempt_number"],
        "attempt_digest": attempt["attempt_digest"],
        "bundle_path": promoted["path"],
        "bundle_digest": promoted["bundle_digest"],
        "outer_iteration": promoted["outer_iteration"],
        "environment_steps": promoted["environment_steps"],
        "finiteness_audit_digest": promoted["finiteness_audit"]["report_digest"],
        "golden_parity_digest": promoted["golden_parity"]["report_digest"],
        "compiled_parity_digest": promoted["compiled_parity"]["report_digest"],
        "promotion_entry_digest": entry["entry_digest"],
        "failure_trace_digest": entry["traceback_sha256"],
        "failed_compiled_parity": dict(entry["failed_compiled_parity"]),
    }


def accept_policy_pool(
    *,
    server_plan: Mapping[str, Any],
    runs_root: Path,
    promotion_manifest: Mapping[str, Any],
    path_resolver: RecordedPathResolver | None = None,
) -> dict[str, Any]:
    """Revalidate and accept the exact reviewed 84-direct + 6-promotion pool."""

    jobs, plan = _canonical_job_roots(
        server_plan, runs_root, path_resolver=path_resolver
    )
    status, vendor, implementation = _queue_authority(runs_root, plan, jobs)
    require_exact_keys(
        promotion_manifest,
        {
            "schema",
            "server_plan_digest",
            "queue_status_sha256",
            "policy",
            "promotion_count",
            "created_at",
            "entries",
            "manifest_digest",
        },
        "compiled-parity promotion manifest",
    )
    validate_self_digest(
        promotion_manifest,
        key="manifest_digest",
        where="compiled-parity promotion manifest",
    )
    entries = promotion_manifest["entries"]
    if (
        promotion_manifest["schema"] != PROMOTION_SCHEMA
        or promotion_manifest["server_plan_digest"] != plan["plan_digest"]
        or promotion_manifest["queue_status_sha256"]
        != sha256_file(runs_root / "queue_status.json")
        or promotion_manifest["policy"]
        != "last_canonical_checkpoint_before_reloaded_compiled_parity_failure"
        or promotion_manifest["promotion_count"] != EXPECTED_PROMOTION_COUNT
        or not isinstance(entries, Mapping)
        or len(entries) != EXPECTED_PROMOTION_COUNT
    ):
        raise ContractError(
            "promotion manifest is not the reviewed frozen 6-cell overlay"
        )
    failed_ids = {
        job_id
        for job_id, state in status["jobs"].items()
        if state.get("state") == "failed"
    }
    direct_ids = set(jobs) - failed_ids
    if set(entries) != failed_ids or len(direct_ids) != EXPECTED_DIRECT_COUNT:
        raise ContractError("promotion coverage differs from the final queue failures")

    cells: dict[str, Any] = {}
    for job_id in sorted(direct_ids):
        job = jobs[job_id]
        state = status["jobs"][job_id]
        if state.get("state") != "succeeded":
            raise ContractError("non-promoted cell is not a successful queue terminal")
        attempt_dir = _final_attempt_dir(
            state=state,
            job_id=job_id,
            runs_root=runs_root.resolve(),
            path_resolver=path_resolver,
        )
        attempt = validate_attempt(
            load_strict_json(attempt_dir / "attempt_manifest.json")
        )
        if (
            attempt["attempt_number"] != state.get("attempts")
            or attempt["plan_digest"] != plan["plan_digest"]
            or attempt["job_digest"] != job["job_digest"]
            or attempt["job"] != job
        ):
            raise ContractError("successful attempt differs from the frozen plan/job")
        result = validate_queue_result(
            load_strict_json(attempt_dir / "queue_result.json"),
            expected_job_digest=job["job_digest"],
            expected_attempt_digest=attempt["attempt_digest"],
            expected_config_digest=job["config_digest"],
            expected_execution_purpose=job["execution_purpose"],
        )
        validate_vendor_provenance(result["vendor"], expected=vendor)
        validate_implementation_provenance(
            result["implementation"], expected=implementation
        )
        if result["state"] != "succeeded" or result["returncode"] != 0:
            raise ContractError("successful attempt has no successful queue result")
        record = validate_completed_attempt(
            attempt_dir,
            job,
            attempt,
            expected_vendor=vendor,
            expected_implementation=implementation,
            path_resolver=path_resolver,
        )
        promoted = record["checkpoint_bundles"][-1]
        direct_keys = {"attempt_dir", "attempts", "state", "terminal_record_state"}
        retry_keys = direct_keys | {"returncode", "validation_error"}
        if (
            set(state) not in (direct_keys, retry_keys)
            or state["terminal_record_state"] != record["state"]
            or (
                set(state) == retry_keys
                and (
                    state["returncode"] != result["returncode"]
                    or state["validation_error"] is not None
                )
            )
        ):
            raise ContractError(
                "successful queue cell differs from its terminal record"
            )
        cells[job_id] = {
            "resolution": "direct_terminal_record",
            "job_id": job_id,
            "job_digest": job["job_digest"],
            "source_anchor_id": AnchorManifest.from_path(
                _physical_path(job["anchor_manifest_path"], path_resolver)
            ).anchor_id,
            "seed": job["seed"],
            "attempt_number": attempt["attempt_number"],
            "attempt_digest": attempt["attempt_digest"],
            "bundle_path": promoted["path"],
            "bundle_digest": promoted["bundle_digest"],
            "outer_iteration": promoted["outer_iteration"],
            "environment_steps": promoted["environment_steps"],
            "finiteness_audit_digest": promoted["finiteness_audit"]["report_digest"],
            "golden_parity_digest": promoted["golden_parity"]["report_digest"],
            "compiled_parity_digest": promoted["compiled_parity"]["report_digest"],
            "training_record_digest": record["record_digest"],
            "terminal_record_state": record["state"],
        }
    for job_id in sorted(failed_ids):
        cells[job_id] = _validate_promotion(
            entry=entries[job_id],
            job=jobs[job_id],
            state=status["jobs"][job_id],
            runs_root=runs_root,
            expected_vendor=vendor,
            expected_implementation=implementation,
            expected_plan_digest=plan["plan_digest"],
            path_resolver=path_resolver,
        )

    resolutions = Counter(cell["resolution"] for cell in cells.values())
    if (
        len(cells) != EXPECTED_JOB_COUNT
        or resolutions["direct_terminal_record"] != EXPECTED_DIRECT_COUNT
        or resolutions["compiled_parity_fallback_promotion"] != EXPECTED_PROMOTION_COUNT
    ):
        raise ContractError("accepted policy pool is not the exact reviewed 84+6 set")
    accepted_units = {
        (cell["source_anchor_id"], cell["seed"]) for cell in cells.values()
    }
    anchors = sorted({cell["source_anchor_id"] for cell in cells.values()})
    expected_units = {(anchor, seed) for anchor in anchors for seed in EXPECTED_SEEDS}
    if len(anchors) != EXPECTED_ANCHOR_COUNT or accepted_units != expected_units:
        raise ContractError(
            "accepted policy pool is not an exact 30-anchor x 3-seed grid"
        )
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
    return with_self_digest(
        {
            "schema": ACCEPTANCE_SCHEMA,
            "decision": "PASS",
            "accepted_at": utc_now(),
            "server_plan_digest": plan["plan_digest"],
            "queue_status_sha256": sha256_file(runs_root / "queue_status.json"),
            "promotion_manifest_digest": promotion_manifest["manifest_digest"],
            "job_count": len(cells),
            "anchor_count": len(anchors),
            "seeds": list(EXPECTED_SEEDS),
            "direct_terminal_record_count": EXPECTED_DIRECT_COUNT,
            "compiled_parity_fallback_promotion_count": EXPECTED_PROMOTION_COUNT,
            "all_selected_bundles_finite": True,
            "all_selected_bundles_golden_parity_passed": True,
            "all_selected_bundles_compiled_parity_passed": True,
            "pool_digest": pool_digest,
            "cells": dict(sorted(cells.items())),
        },
        key="report_digest",
    )


__all__ = [
    "ACCEPTANCE_SCHEMA",
    "FAILURE_CODE",
    "PROMOTION_SCHEMA",
    "accept_policy_pool",
    "build_compiled_parity_promotion_manifest",
]
