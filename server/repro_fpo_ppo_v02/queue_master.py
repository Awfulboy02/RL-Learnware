#!/usr/bin/env python3
"""One-job-per-physical-GPU queue with immutable semantic attempts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, BinaryIO, Mapping, Sequence

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .anchor_binding import AnchorManifest
    from .formal_plan import validate_formal_training_projection
    from .provenance import (
        AUDIT_SMOKE_EXECUTION_MODE,
        AUDIT_SMOKE_EXECUTION_PURPOSE,
        ATTEMPT_SCHEMA,
        ContractError,
        EXECUTION_PURPOSES,
        FORMAL_EXECUTION_PURPOSE,
        FORMAL_GPU_EXECUTION_MODE,
        append_jsonl,
        atomic_write_json,
        canonical_json_bytes,
        finalize_attempt,
        load_strict_json,
        revalidate_formal_freeze_binding,
        utc_now,
        validate_attempt,
        validate_execution_evidence,
        validate_fpo_source_attestation,
        validate_implementation_provenance,
        validate_policy_bundle,
        validate_queue_result,
        validate_run_manifest_server_binding,
        validate_success_record,
        validate_training_job,
        validate_training_plan,
        validate_vendor_provenance,
        with_self_digest,
    )
    from .implementation import inspect_implementation_inventory
    from .vendor import inspect_vendor_directory
except ImportError:  # pragma: no cover - exercised by executable entry points
    from anchor_binding import AnchorManifest
    from formal_plan import validate_formal_training_projection
    from provenance import (
        AUDIT_SMOKE_EXECUTION_MODE,
        AUDIT_SMOKE_EXECUTION_PURPOSE,
        ATTEMPT_SCHEMA,
        ContractError,
        EXECUTION_PURPOSES,
        FORMAL_EXECUTION_PURPOSE,
        FORMAL_GPU_EXECUTION_MODE,
        append_jsonl,
        atomic_write_json,
        canonical_json_bytes,
        finalize_attempt,
        load_strict_json,
        revalidate_formal_freeze_binding,
        utc_now,
        validate_attempt,
        validate_execution_evidence,
        validate_fpo_source_attestation,
        validate_implementation_provenance,
        validate_policy_bundle,
        validate_queue_result,
        validate_run_manifest_server_binding,
        validate_success_record,
        validate_training_job,
        validate_training_plan,
        validate_vendor_provenance,
        with_self_digest,
    )
    from implementation import inspect_implementation_inventory
    from vendor import inspect_vendor_directory


ATTEMPT_DIR_RE = re.compile(r"attempt_(\d{3})")


def parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or any(not item.isdigit() for item in result):
        raise argparse.ArgumentTypeError("--gpus must be comma-separated nonnegative integers")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("--gpus must not contain duplicates")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_strict_json(path)


def validate_completed_attempt(
    attempt_dir: Path,
    job: Mapping[str, Any],
    attempt: Mapping[str, Any],
    *,
    expected_vendor: Mapping[str, Any],
    expected_implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check runner status, record, anchor, export set, and bundle bytes."""

    anchor = AnchorManifest.from_path(job["anchor_manifest_path"])
    record = validate_success_record(
        attempt_dir / "training_record.json",
        expected_job_digest=str(job["job_digest"]),
        expected_attempt_digest=str(attempt["attempt_digest"]),
        expected_anchor_manifest_digest=anchor.manifest_digest,
        expected_environment_instance_digest=anchor.environment_instance_digest,
        expected_training_protocol_digest=str(job["training_protocol_digest"]),
        expected_config_digest=str(job["config_digest"]),
        expected_execution_purpose=str(job["execution_purpose"]),
    )
    validate_implementation_provenance(
        attempt["implementation"], expected=expected_implementation
    )
    validate_implementation_provenance(
        record["implementation"], expected=expected_implementation
    )
    if record["algorithm"] != job["training_protocol"]["algorithm"]:
        raise ContractError("training record algorithm drifted from the frozen job")
    if record["seed"] != job["seed"]:
        raise ContractError("training record seed drifted from the frozen job")
    for key in (
        "config_digest",
        "execution_purpose",
        "execution_mode",
        "formal_eligible",
    ):
        if record[key] != attempt[key]:
            raise ContractError(f"training record {key} drifted from the attempt")
    run_manifest = validate_run_manifest_server_binding(
        load_strict_json(attempt_dir / "run_manifest.json"),
        job=job,
        attempt=attempt,
        anchor=anchor.to_dict(),
    )
    runtime = run_manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ContractError("run manifest runtime evidence is missing")
    vendor = runtime.get("vendor")
    if not isinstance(vendor, Mapping):
        raise ContractError("run manifest vendor provenance is missing")
    validated_vendor = validate_vendor_provenance(vendor, expected=expected_vendor)
    implementation = runtime.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ContractError("run manifest implementation provenance is missing")
    validate_implementation_provenance(
        implementation, expected=expected_implementation
    )
    if runtime.get("pythonpath_vendor_precedence_verified") is not True:
        raise ContractError("run manifest did not verify vendor PYTHONPATH precedence")
    if runtime.get("wandb_mode") != "disabled":
        raise ContractError("run manifest did not preserve WANDB_MODE=disabled")
    if runtime.get("python_dont_write_bytecode") != "1":
        raise ContractError("run manifest did not disable vendor bytecode writes")
    source_attestation = validate_fpo_source_attestation(
        runtime, expected_commit=str(anchor.runtime["fpo_commit"])
    )
    execution = runtime.get("execution_evidence")
    if not isinstance(execution, Mapping):
        raise ContractError("run manifest execution evidence is missing")
    hardware_digest = runtime.get("hardware_digest")
    evidence = validate_execution_evidence(
        execution,
        expected_job_digest=str(job["job_digest"]),
        expected_attempt_digest=str(attempt["attempt_digest"]),
        expected_hardware_digest=str(hardware_digest),
        expected_config_digest=str(job["config_digest"]),
        expected_execution_purpose=str(job["execution_purpose"]),
        expected_attempt_root=attempt_dir,
        require_formal=bool(attempt["formal_eligible"]),
    )
    expected_execution_projection = {
        "config_digest": evidence["config_digest"],
        "execution_purpose": evidence["execution_purpose"],
        "execution_mode": evidence["execution_mode"],
        "formal_eligible": evidence["formal_eligible"],
        "execution_evidence_digest": evidence["execution_evidence_digest"],
    }
    for key, expected in expected_execution_projection.items():
        if run_manifest.get(key) != expected or record.get(key) != expected:
            raise ContractError(f"run/record {key} execution binding mismatch")
    command = runtime.get("command")
    if not isinstance(command, list):
        raise ContractError("run manifest command is missing")
    has_allow_flag = "--allow-non-gpu" in command
    if has_allow_flag is not bool(evidence["allow_non_gpu"]):
        raise ContractError("run command disagrees with execution-mode evidence")
    try:
        purpose_index = command.index("--execution-purpose")
    except ValueError as error:
        raise ContractError("run command omitted --execution-purpose") from error
    if (
        command.count("--execution-purpose") != 1
        or purpose_index + 1 >= len(command)
        or command[purpose_index + 1] != job["execution_purpose"]
    ):
        raise ContractError("run command execution purpose drifted from the job")
    try:
        vendor_index = command.index("--vendor-dir")
    except ValueError as error:
        raise ContractError("run command omitted --vendor-dir") from error
    if (
        command.count("--vendor-dir") != 1
        or vendor_index + 1 >= len(command)
        or command[vendor_index + 1] != validated_vendor["path"]
    ):
        raise ContractError("run command vendor directory drifted from provenance")
    try:
        exporter_index = command.index("--legacy-policy-io")
    except ValueError as error:
        raise ContractError("run command omitted --legacy-policy-io") from error
    if (
        command.count("--legacy-policy-io") != 1
        or exporter_index + 1 >= len(command)
        or command[exporter_index + 1] != runtime.get("legacy_policy_io_path")
    ):
        raise ContractError("run command legacy exporter path drifted from provenance")
    observed_outers = [item["outer_iteration"] for item in record["checkpoint_bundles"]]
    if observed_outers != job["training_protocol"]["export_outer_iterations"]:
        raise ContractError("training record checkpoint set drifted from the frozen export rule")
    checkpoint_root = (attempt_dir / "checkpoints").resolve()
    require_evaluation = bool(job["training_protocol"]["evaluation"]["enabled"])
    for item in record["checkpoint_bundles"]:
        bundle = Path(item["path"]).resolve()
        try:
            bundle.relative_to(checkpoint_root)
        except ValueError as error:
            raise ContractError("recorded checkpoint escapes its immutable attempt root") from error
        observed = validate_policy_bundle(bundle, require_evaluation=require_evaluation)
        for key in ("bundle_manifest_sha256", "bundle_manifest_digest", "files"):
            if observed[key] != item[key]:
                raise ContractError(f"recorded checkpoint {key} disagrees with bundle bytes")
        if item["bundle_digest"] != observed["bundle_manifest_sha256"]:
            raise ContractError("recorded checkpoint bundle_digest disagrees with bundle bytes")
        finiteness = item["finiteness_audit"]
        if (
            finiteness.get("all_arrays_finite") is not True
            or finiteness.get("bundle_manifest_sha256") != observed["bundle_manifest_sha256"]
            or finiteness.get("validated_file_digests") != observed["files"]
        ):
            raise ContractError("checkpoint finiteness audit is not bound to the validated bytes")
        parity_contract = job["training_protocol"]["parity"]
        golden = item["golden_parity"]
        compiled = item["compiled_parity"]
        if (
            golden.get("atol") != parity_contract["atol"]
            or golden.get("rtol") != parity_contract["rtol"]
            or golden.get("sample_count") != parity_contract["golden_sample_count"]
            or compiled.get("atol") != parity_contract["atol"]
            or compiled.get("rtol") != parity_contract["rtol"]
            or compiled.get("sample_count") != parity_contract["compiled_sample_count"]
        ):
            raise ContractError("checkpoint parity evidence differs from the frozen protocol")
        bundle_manifest = load_strict_json(bundle / "bundle_manifest.json")
        expected_bundle_semantics = {
            "algorithm": job["training_protocol"]["algorithm"],
            "task": anchor.task,
            "seed": job["seed"],
            "outer_iteration": item["outer_iteration"],
            "environment_steps": item["environment_steps"],
        }
        for key, expected in expected_bundle_semantics.items():
            if bundle_manifest.get(key) != expected:
                raise ContractError(f"checkpoint bundle {key} drifted from frozen job")
        bundle_provenance = load_strict_json(bundle / "provenance.json")
        expected_bundle_bindings = {
            "config_digest": job["config_digest"],
            "execution_purpose": job["execution_purpose"],
            "job_digest": job["job_digest"],
            "attempt_digest": attempt["attempt_digest"],
            "anchor_manifest_digest": anchor.manifest_digest,
            "environment_instance_digest": anchor.environment_instance_digest,
            "operator_digest": anchor.operator_digest,
            "model_diff_digest": anchor.model_diff_digest,
            "actual_bound_model_digest": anchor.expected_bound_model_digest,
            "runtime_digest": anchor.runtime_digest,
            **source_attestation,
            "execution_mode": evidence["execution_mode"],
            "formal_eligible": evidence["formal_eligible"],
            "execution_evidence_digest": evidence["execution_evidence_digest"],
            "attempt_root": evidence["attempt_root"],
            "implementation": expected_implementation,
        }
        for key, expected in expected_bundle_bindings.items():
            if bundle_provenance.get(key) != expected:
                raise ContractError(f"checkpoint provenance {key} binding mismatch")
    status = load_strict_json(attempt_dir / "status.json")
    required_status = {
        "state": "completed",
        "job_digest": job["job_digest"],
        "attempt_digest": attempt["attempt_digest"],
        "anchor_manifest_digest": anchor.manifest_digest,
        "environment_instance_digest": anchor.environment_instance_digest,
        "training_record_digest": record["record_digest"],
    }
    for key, expected in required_status.items():
        if status.get(key) != expected:
            raise ContractError(f"completed runner status {key} mismatch")
    if status.get("exported_outer_iterations") != observed_outers:
        raise ContractError("completed runner status export list mismatch")
    return record


class AttemptStore:
    """Resolve prior immutable attempts and atomically publish a new one."""

    def __init__(
        self,
        *,
        runs_root: Path,
        plan_digest: str,
        execution_mode: str,
        formal_eligible: bool,
        vendor: Mapping[str, Any],
        implementation: Mapping[str, Any],
    ) -> None:
        self.runs_root = runs_root
        self.plan_digest = plan_digest
        self.execution_mode = execution_mode
        self.formal_eligible = formal_eligible
        self.vendor = validate_vendor_provenance(vendor)
        self.implementation = validate_implementation_provenance(implementation)

    def job_dir(self, job: Mapping[str, Any]) -> Path:
        return self.runs_root / "jobs" / str(job["job_id"])

    def initialize_job(self, job: Mapping[str, Any]) -> None:
        validated = validate_training_job(job)
        root = self.job_dir(validated)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "job_manifest.json"
        if manifest_path.exists():
            observed = load_strict_json(manifest_path)
            if canonical_json_bytes(observed) != canonical_json_bytes(validated):
                raise ContractError(
                    f"existing job root has different semantic inputs: {manifest_path}"
                )
        else:
            atomic_write_json(manifest_path, validated, overwrite=False)
        pending = sorted(root.glob(".attempt_*.pending-*"))
        if pending:
            raise ContractError(
                f"torn attempt publication requires audit before resume: {pending[0]}"
            )
        for path in self.attempt_dirs(validated):
            attempt = validate_attempt(load_strict_json(path / "attempt_manifest.json"))
            if attempt["job_digest"] != validated["job_digest"]:
                raise ContractError(f"attempt belongs to another semantic job: {path}")
            if attempt["plan_digest"] != self.plan_digest:
                raise ContractError(f"attempt belongs to another immutable plan: {path}")
            if (
                attempt["execution_purpose"] != validated["execution_purpose"]
                or attempt["config_digest"] != validated["config_digest"]
                or attempt["execution_mode"] != self.execution_mode
                or attempt["formal_eligible"] is not self.formal_eligible
            ):
                raise ContractError(
                    f"attempt root mixes formal and audit-smoke evidence: {path}"
                )
            validate_implementation_provenance(
                attempt["implementation"], expected=self.implementation
            )

    def attempt_dirs(self, job: Mapping[str, Any]) -> list[Path]:
        root = self.job_dir(job)
        result = []
        if not root.exists():
            return result
        for path in root.iterdir():
            match = ATTEMPT_DIR_RE.fullmatch(path.name)
            if path.is_dir() and match:
                result.append(path)
        result.sort(key=lambda path: int(ATTEMPT_DIR_RE.fullmatch(path.name).group(1)))
        return result

    def successful_attempt(self, job: Mapping[str, Any]) -> Path | None:
        for path in self.attempt_dirs(job):
            try:
                attempt = validate_attempt(load_strict_json(path / "attempt_manifest.json"))
                result = validate_queue_result(
                    load_strict_json(path / "queue_result.json"),
                    expected_job_digest=str(job["job_digest"]),
                    expected_attempt_digest=str(attempt["attempt_digest"]),
                    expected_config_digest=str(job["config_digest"]),
                    expected_execution_purpose=str(job["execution_purpose"]),
                )
                validate_vendor_provenance(result["vendor"], expected=self.vendor)
                validate_implementation_provenance(
                    result["implementation"], expected=self.implementation
                )
                if (
                    result["config_digest"] != attempt["config_digest"]
                    or result["execution_purpose"] != attempt["execution_purpose"]
                    or result["execution_mode"] != attempt["execution_mode"]
                    or result["formal_eligible"] is not attempt["formal_eligible"]
                ):
                    raise ContractError(
                        "queue result execution mode differs from its attempt"
                    )
                if result["state"] != "succeeded":
                    continue
                validate_completed_attempt(
                    path,
                    job,
                    attempt,
                    expected_vendor=self.vendor,
                    expected_implementation=self.implementation,
                )
            except (ContractError, OSError):
                continue
            return path
        return None

    def allocate(self, job: Mapping[str, Any], *, gpu: str) -> tuple[Path, dict[str, Any]]:
        self.initialize_job(job)
        number = len(self.attempt_dirs(job)) + 1
        if number > 999:
            raise ContractError(f"attempt index exhausted for {job['job_id']}")
        final = self.job_dir(job) / f"attempt_{number:03d}"
        temporary = self.job_dir(job) / f".attempt_{number:03d}.pending-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o750)
        attempt_id = f"{job['job_id']}.a{number:03d}"
        value = finalize_attempt(
            {
                "schema": ATTEMPT_SCHEMA,
                "plan_digest": self.plan_digest,
                "job": dict(job),
                "job_digest": job["job_digest"],
                "attempt_id": attempt_id,
                "attempt_number": number,
                "execution_attempt_id": "exec-" + uuid.uuid4().hex,
                "gpu": gpu,
                "config_digest": job["config_digest"],
                "execution_purpose": job["execution_purpose"],
                "execution_mode": self.execution_mode,
                "formal_eligible": self.formal_eligible,
                "implementation": self.implementation,
                "created_at": utc_now(),
            }
        )
        try:
            atomic_write_json(temporary / "attempt_manifest.json", value, overwrite=False)
            os.rename(temporary, final)
        except BaseException:
            # Preserve a torn publication for explicit audit; initialize_job will
            # fail closed on it rather than silently deleting evidence.
            raise
        return final, value


@dataclass
class RunningJob:
    job: dict[str, Any]
    gpu: str
    attempt_dir: Path
    attempt: dict[str, Any]
    process: subprocess.Popen[bytes]
    stdout_handle: BinaryIO
    stderr_handle: BinaryIO
    command: list[str]
    started_at: str
    started_monotonic: float


class QueueMaster:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.plan = validate_training_plan(load_strict_json(args.plan))
        if args.execution_purpose != self.plan["execution_purpose"]:
            raise ContractError(
                "explicit queue execution purpose differs from the immutable plan"
            )
        if args.allow_non_gpu is not (
            args.execution_purpose == AUDIT_SMOKE_EXECUTION_PURPOSE
        ):
            raise ContractError(
                "--allow-non-gpu must be supplied exactly for audit_smoke execution"
            )
        if args.execution_purpose == FORMAL_EXECUTION_PURPOSE:
            # Re-read the canonical YAML, canonical freeze manifest, projection
            # digests, and implementation/Git release state before creating any
            # queue artifacts.  A caller-supplied config hash is never formal
            # authority by itself.
            revalidated = revalidate_formal_freeze_binding(
                self.plan["formal_protocol_freeze"]
            )
            if revalidated["binding_digest"] != self.plan[
                "formal_protocol_freeze_digest"
            ]:
                raise ContractError("formal queue freeze binding digest drifted")
            validate_formal_training_projection(self.plan, revalidated)
        self.runs_root = args.runs_root.resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.execution_mode = (
            AUDIT_SMOKE_EXECUTION_MODE
            if args.allow_non_gpu
            else FORMAL_GPU_EXECUTION_MODE
        )
        self.formal_eligible = (
            args.execution_purpose == FORMAL_EXECUTION_PURPOSE
            and not args.allow_non_gpu
        )
        canonical_runner = Path(__file__).resolve().parent / "runner.py"
        if self.formal_eligible and args.runner.resolve() != canonical_runner:
            raise ContractError("formal execution requires the versioned v0.2 runner.py")
        self.vendor = inspect_vendor_directory(args.vendor_dir)
        self.legacy_policy_io = args.legacy_policy_io.resolve()
        self.implementation = inspect_implementation_inventory(
            runner_path=args.runner,
            legacy_policy_io_path=self.legacy_policy_io,
        )
        self.store = AttemptStore(
            runs_root=self.runs_root,
            plan_digest=self.plan["plan_digest"],
            execution_mode=self.execution_mode,
            formal_eligible=self.formal_eligible,
            vendor=self.vendor,
            implementation=self.implementation,
        )
        self.lock_handle: BinaryIO | None = None
        self.running: dict[str, RunningJob] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.stop_requested = False
        self.stop_signal: int | None = None
        self.started_at = utc_now()

    def acquire_lock(self) -> None:
        path = self.runs_root / "queue_master.lock"
        handle = path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise RuntimeError(f"another queue already holds {path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
        self.lock_handle = handle

    def release_lock(self) -> None:
        if self.lock_handle is None:
            return
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.lock_handle.close()
            self.lock_handle = None

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        path = self.runs_root / "master.log"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(line, flush=True)

    def validate_external_inputs(self) -> None:
        if not self.args.runner.is_file():
            raise FileNotFoundError(f"runner does not exist: {self.args.runner}")
        if not self.args.fpo_root.is_dir():
            raise FileNotFoundError(f"FPO root does not exist: {self.args.fpo_root}")
        for job in self.plan["jobs"]:
            anchor = AnchorManifest.from_path(job["anchor_manifest_path"])
            if anchor.manifest_digest != job["anchor_manifest_digest"]:
                raise ContractError(f"anchor digest drift for job {job['job_id']}")
            self.store.initialize_job(job)

    def initialize_states(self) -> list[dict[str, Any]]:
        pending: list[dict[str, Any]] = []
        for job in self.plan["jobs"]:
            success = self.store.successful_attempt(job)
            attempts = len(self.store.attempt_dirs(job))
            if success is not None:
                self.states[job["job_id"]] = {
                    "state": "succeeded",
                    "attempts": attempts,
                    "attempt_dir": str(success),
                }
            elif attempts >= self.args.max_attempts:
                self.states[job["job_id"]] = {
                    "state": "failed",
                    "attempts": attempts,
                    "reason": "max_attempts_exhausted",
                }
            else:
                self.states[job["job_id"]] = {"state": "queued", "attempts": attempts}
                pending.append(job)
        return pending

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in self.states.values():
            state = str(value["state"])
            result[state] = result.get(state, 0) + 1
        return dict(sorted(result.items()))

    def write_status(self, *, state: str) -> None:
        running = []
        for gpu, item in sorted(self.running.items(), key=lambda pair: int(pair[0])):
            running.append(
                {
                    "gpu": gpu,
                    "job_id": item.job["job_id"],
                    "pid": item.process.pid,
                    "attempt_dir": str(item.attempt_dir),
                    "attempt_digest": item.attempt["attempt_digest"],
                    "elapsed_seconds": round(time.monotonic() - item.started_monotonic, 3),
                }
            )
        atomic_write_json(
            self.runs_root / "queue_status.json",
            {
                "schema": "policy-learnware.v02-queue-status.v0",
                "state": state,
                "plan": str(self.args.plan.resolve()),
                "plan_digest": self.plan["plan_digest"],
                "config_digest": self.plan["config_digest"],
                "execution_purpose": self.args.execution_purpose,
                "formal_protocol_freeze_digest": self.plan[
                    "formal_protocol_freeze_digest"
                ],
                "started_at": self.started_at,
                "updated_at": utc_now(),
                "master_pid": os.getpid(),
                "stop_signal": self.stop_signal,
                "gpus": list(self.args.gpus),
                "execution_mode": self.execution_mode,
                "formal_eligible": self.formal_eligible,
                "vendor": self.vendor,
                "implementation": self.implementation,
                "counts": self.counts(),
                "running": running,
                "jobs": self.states,
            },
        )

    def launch_job(self, job: dict[str, Any], gpu: str) -> None:
        attempt_dir, attempt = self.store.allocate(job, gpu=gpu)
        command = [
            str(self.args.python),
            "-u",
            str(self.args.runner.resolve()),
            "--attempt-manifest",
            str((attempt_dir / "attempt_manifest.json").resolve()),
            "--run-dir",
            str(attempt_dir.resolve()),
            "--fpo-root",
            str(self.args.fpo_root.resolve()),
            "--vendor-dir",
            self.vendor["path"],
            "--legacy-policy-io",
            str(self.legacy_policy_io),
            "--execution-purpose",
            self.args.execution_purpose,
        ]
        if self.args.allow_non_gpu:
            command.append("--allow-non-gpu")
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
                "WANDB_MODE": "disabled",
            }
        )
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            self.vendor["path"]
            if not inherited_pythonpath
            else self.vendor["path"] + os.pathsep + inherited_pythonpath
        )
        stdout_handle = (attempt_dir / "stdout.log").open("xb")
        stderr_handle = (attempt_dir / "stderr.log").open("xb")
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=environment,
                start_new_session=True,
            )
        except BaseException:
            stdout_handle.close()
            stderr_handle.close()
            raise
        item = RunningJob(
            job=job,
            gpu=gpu,
            attempt_dir=attempt_dir,
            attempt=attempt,
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            command=command,
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
        )
        self.running[gpu] = item
        self.states[job["job_id"]] = {
            "state": "running",
            "gpu": gpu,
            "pid": process.pid,
            "attempts": attempt["attempt_number"],
            "attempt_dir": str(attempt_dir),
        }
        self.log(
            f"started job={job['job_id']} attempt={attempt['attempt_number']} "
            f"gpu={gpu} pid={process.pid}"
        )

    def finish_job(self, gpu: str, *, interrupted: bool = False) -> tuple[bool, dict[str, Any]]:
        item = self.running.pop(gpu)
        returncode = item.process.poll()
        if returncode is None:
            returncode = item.process.wait()
        item.stdout_handle.close()
        item.stderr_handle.close()
        succeeded = returncode == 0 and not interrupted
        validation_error: str | None = None
        if succeeded:
            try:
                validate_completed_attempt(
                    item.attempt_dir,
                    item.job,
                    item.attempt,
                    expected_vendor=self.vendor,
                    expected_implementation=self.implementation,
                )
            except (ContractError, OSError) as error:
                succeeded = False
                validation_error = str(error)
        finished_at = utc_now()
        result = with_self_digest(
            {
                "schema": "policy-learnware.v02-queue-result.v0",
                "state": "succeeded" if succeeded else ("interrupted" if interrupted else "failed"),
                "job_digest": item.job["job_digest"],
                "attempt_digest": item.attempt["attempt_digest"],
                "gpu": gpu,
                "config_digest": item.job["config_digest"],
                "execution_purpose": item.job["execution_purpose"],
                "execution_mode": item.attempt["execution_mode"],
                "formal_eligible": item.attempt["formal_eligible"],
                "pid": item.process.pid,
                "returncode": int(returncode),
                "started_at": item.started_at,
                "finished_at": finished_at,
                "elapsed_seconds": round(time.monotonic() - item.started_monotonic, 6),
                "validation_error": validation_error,
                "command": item.command,
                "vendor": self.vendor,
                "implementation": self.implementation,
            },
            key="result_digest",
        )
        atomic_write_json(item.attempt_dir / "queue_result.json", result, overwrite=False)
        attempts = item.attempt["attempt_number"]
        self.states[item.job["job_id"]] = {
            "state": result["state"],
            "attempts": attempts,
            "attempt_dir": str(item.attempt_dir),
            "returncode": int(returncode),
            "validation_error": validation_error,
        }
        atomic_write_json(
            self.store.job_dir(item.job) / "job_state.json",
            {"updated_at": finished_at, **self.states[item.job["job_id"]]},
        )
        self.log(
            f"{result['state']} job={item.job['job_id']} gpu={gpu} "
            f"returncode={returncode} elapsed={result['elapsed_seconds']}s"
        )
        return succeeded, item.job

    def request_stop(self, signum: int, _frame: Any) -> None:
        if not self.stop_requested:
            self.stop_requested = True
            self.stop_signal = signum
            self.log(f"received signal={signum}; stopping active process groups")

    def terminate_running(self) -> None:
        for item in self.running.values():
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + self.args.terminate_grace_seconds
        while self.running and time.monotonic() < deadline:
            for gpu in list(self.running):
                if self.running[gpu].process.poll() is not None:
                    self.finish_job(gpu, interrupted=True)
            if self.running:
                time.sleep(min(0.1, self.args.poll_seconds))
        for item in self.running.values():
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for gpu in list(self.running):
            self.finish_job(gpu, interrupted=True)

    def run(self) -> int:
        self.acquire_lock()
        previous_handlers: dict[int, Any] = {}
        try:
            self.validate_external_inputs()
            pending = self.initialize_states()
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self.request_stop)
            self.log(
                f"queue start plan={self.plan['plan_digest']} jobs={len(self.plan['jobs'])} "
                f"pending={len(pending)} gpus={','.join(self.args.gpus)} "
                f"vendor={self.vendor['tree_digest']} "
                f"implementation={self.implementation['implementation_digest']}"
            )
            self.write_status(state="running")
            while pending or self.running:
                if self.stop_requested:
                    self.terminate_running()
                    for job in pending:
                        self.states[job["job_id"]]["state"] = "queued_after_stop"
                    self.write_status(state="stopped")
                    return 130
                for gpu in self.args.gpus:
                    if gpu not in self.running and pending:
                        job = pending.pop(0)
                        self.launch_job(job, gpu)
                if not self.running:
                    break
                time.sleep(self.args.poll_seconds)
                for gpu in list(self.running):
                    if self.running[gpu].process.poll() is None:
                        continue
                    succeeded, job = self.finish_job(gpu)
                    attempts = len(self.store.attempt_dirs(job))
                    if not succeeded and attempts < self.args.max_attempts:
                        self.states[job["job_id"]] = {"state": "queued", "attempts": attempts}
                        pending.append(job)
                self.write_status(state="running")
            failed = [
                job_id for job_id, value in self.states.items() if value["state"] != "succeeded"
            ]
            state = "completed" if not failed else "completed_with_failures"
            self.write_status(state=state)
            self.log(f"queue finished state={state} failures={len(failed)}")
            return 0 if not failed else 1
        finally:
            if self.running:
                self.stop_requested = True
                self.terminate_running()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            self.release_lock()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--fpo-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--gpus", type=parse_gpus, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        required=True,
        help="pinned legacy dependency directory prepended to runner PYTHONPATH",
    )
    parser.add_argument(
        "--legacy-policy-io",
        type=Path,
        required=True,
        help="exact read-only legacy policy_io.py exporter used by runner jobs",
    )
    parser.add_argument("--max-attempts", type=_positive_int, required=True)
    parser.add_argument("--poll-seconds", type=_positive_float, default=1.0)
    parser.add_argument("--terminate-grace-seconds", type=_positive_float, default=20.0)
    parser.add_argument("--allow-non-gpu", action="store_true")
    parser.add_argument(
        "--execution-purpose",
        choices=tuple(sorted(EXECUTION_PURPOSES)),
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return QueueMaster(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
