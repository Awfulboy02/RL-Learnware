#!/usr/bin/env python3
"""Dual-host GPU waiter and single-writer queue-launch arbiter.

The waiter is deliberately separate from :mod:`queue_master`.  Two hosts may
observe the same requested physical GPUs and the same shared claim path, but
only the host whose ``os.mkdir`` succeeds may execute the queue launcher.  A
claim is never stolen automatically: an incomplete or stale claim is a
fail-closed condition that must be audited by an operator.

Example (arguments beginning with ``--`` use the ``--launch-arg=VALUE`` form)::

    python gpu_waiter.py \
      --host-id g0544 \
      --gpus 0,1,2,3,4,5,6,7 \
      --nvidia-smi /usr/bin/nvidia-smi \
      --claim-dir /share/experiment/queue-writer.claim \
      --status-path /share/experiment/waiters/g0544.json \
      --claim-busy-action observe \
      --launch-program /share/code/launch.sh \
      --launch-arg=--plan --launch-arg=/share/experiment/plan.json

The production resource contract is fixed here: a GPU is idle only when no
compute application is reported for its UUID, memory use is at most 512 MiB,
and utilization is at most 5 percent.  The same GPU must satisfy the complete
contract in two consecutive probes whose starts are at least 15 seconds apart.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid


SCHEMA = "policy-learnware.v02-dual-host-gpu-waiter.v0"
CLAIM_SCHEMA = "policy-learnware.v02-queue-writer-claim.v0"
STATUS_SCHEMA = "policy-learnware.v02-gpu-waiter-status.v0"
GPU_MEMORY_LIMIT_MIB = 512
GPU_UTILIZATION_LIMIT_PERCENT = 5
REQUIRED_CONSECUTIVE_PROBES = 2
PROBE_INTERVAL_SECONDS = 15.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 10.0
DIAGNOSTIC_TAIL_BYTES = 16_384
HOST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class WaiterError(RuntimeError):
    """Raised when resource or claim evidence is invalid."""


class WaiterInterrupted(RuntimeError):
    """Raised by the installed signal handlers to unwind owned claims."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _block_termination_signals() -> Iterator[None]:
    """Defer SIGINT/SIGTERM across claim ownership hand-offs."""

    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:  # pragma: no cover - server targets are POSIX.
        yield
        return
    blocked = {signal.SIGINT, signal.SIGTERM}
    previous = pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write canonical JSON by fsynced temporary file and atomic replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WaiterError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WaiterError(f"could not read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise WaiterError(f"JSON value must be an object: {path}")
    return value


def parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or any(not item.isdigit() for item in result):
        raise argparse.ArgumentTypeError(
            "--gpus must be comma-separated nonnegative integers"
        )
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("--gpus must not contain duplicates")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _host_id(value: str) -> str:
    if not HOST_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "host id must contain only letters, digits, dot, underscore, or dash"
        )
    return value


def _parse_nonnegative_integer(value: str, *, field: str) -> int:
    stripped = value.strip()
    if not stripped.isdigit():
        raise WaiterError(f"{field} must be a nonnegative integer")
    return int(stripped)


@dataclass(frozen=True)
class GpuResource:
    index: str
    uuid: str
    memory_used_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...]

    @property
    def idle(self) -> bool:
        return (
            not self.compute_pids
            and self.memory_used_mib <= GPU_MEMORY_LIMIT_MIB
            and self.utilization_percent <= GPU_UTILIZATION_LIMIT_PERCENT
        )

    def to_dict(self) -> dict[str, Any]:
        reasons: list[str] = []
        if self.compute_pids:
            reasons.append("compute_application_present")
        if self.memory_used_mib > GPU_MEMORY_LIMIT_MIB:
            reasons.append("memory_above_limit")
        if self.utilization_percent > GPU_UTILIZATION_LIMIT_PERCENT:
            reasons.append("utilization_above_limit")
        return {
            "index": self.index,
            "uuid": self.uuid,
            "memory_used_mib": self.memory_used_mib,
            "utilization_percent": self.utilization_percent,
            "compute_pids": list(self.compute_pids),
            "idle": self.idle,
            "busy_reasons": reasons,
        }


def parse_gpu_rows(
    output: str, *, requested_gpus: Sequence[str]
) -> dict[str, tuple[str, int, int]]:
    """Parse ``index,uuid,memory.used,utilization.gpu`` nounits CSV."""

    requested = tuple(requested_gpus)
    rows: dict[str, tuple[str, int, int]] = {}
    seen_uuids: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            raise WaiterError("nvidia-smi GPU row must contain exactly four columns")
        index, gpu_uuid, memory_used, utilization = parts
        if not index.isdigit() or index in rows:
            raise WaiterError("nvidia-smi GPU indices are invalid or duplicated")
        if not gpu_uuid or gpu_uuid in seen_uuids:
            raise WaiterError("nvidia-smi GPU UUIDs are missing or duplicated")
        memory_value = _parse_nonnegative_integer(memory_used, field="memory.used")
        utilization_value = _parse_nonnegative_integer(
            utilization, field="utilization.gpu"
        )
        if utilization_value > 100:
            raise WaiterError("utilization.gpu must not exceed 100")
        rows[index] = (gpu_uuid, memory_value, utilization_value)
        seen_uuids.add(gpu_uuid)
    missing = sorted(set(requested) - set(rows), key=int)
    if missing:
        raise WaiterError(f"nvidia-smi omitted requested GPUs: {missing}")
    return {index: rows[index] for index in requested}


def parse_compute_app_rows(output: str) -> dict[str, tuple[int, ...]]:
    """Parse ``gpu_uuid,pid`` nounits CSV from the compute-app query."""

    stripped = output.strip()
    if not stripped:
        return {}
    # Some drivers emit this sentence rather than an empty projection.
    if stripped.lower() == "no running processes found":
        return {}
    rows: dict[str, list[int]] = {}
    seen: set[tuple[str, int]] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2:
            raise WaiterError(
                "nvidia-smi compute-app row must contain exactly two columns"
            )
        gpu_uuid, pid_text = parts
        if not gpu_uuid:
            raise WaiterError("nvidia-smi compute-app GPU UUID is missing")
        pid = _parse_nonnegative_integer(pid_text, field="compute-app pid")
        if pid == 0 or (gpu_uuid, pid) in seen:
            raise WaiterError("nvidia-smi compute-app PIDs are zero or duplicated")
        rows.setdefault(gpu_uuid, []).append(pid)
        seen.add((gpu_uuid, pid))
    return {
        gpu_uuid: tuple(sorted(pids)) for gpu_uuid, pids in sorted(rows.items())
    }


def _diagnostic_stream(data: bytes | str | None) -> dict[str, Any]:
    if data is None:
        raw = b""
    elif isinstance(data, bytes):
        raw = data
    else:
        raw = data.encode("utf-8", errors="replace")
    return {
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tail": raw[-DIAGNOSTIC_TAIL_BYTES:].decode("utf-8", errors="replace"),
        "tail_truncated": len(raw) > DIAGNOSTIC_TAIL_BYTES,
    }


def _run_projection(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> subprocess.CompletedProcess[Any]:
    try:
        completed = run_command(
            list(argv),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise WaiterError(
            "nvidia-smi projection timed out: "
            + json.dumps(
                {
                    "argv": list(argv),
                    "timeout_seconds": timeout_seconds,
                    "stdout": _diagnostic_stream(error.stdout),
                    "stderr": _diagnostic_stream(error.stderr),
                },
                sort_keys=True,
            )
        ) from error
    if completed.returncode != 0:
        raise WaiterError(
            "nvidia-smi projection failed: "
            + json.dumps(
                {
                    "argv": list(argv),
                    "returncode": completed.returncode,
                    "stdout": _diagnostic_stream(completed.stdout),
                    "stderr": _diagnostic_stream(completed.stderr),
                },
                sort_keys=True,
            )
        )
    return completed


def collect_gpu_probe(
    *,
    nvidia_smi: str,
    requested_gpus: Sequence[str],
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, GpuResource]:
    """Collect one fail-closed resource and compute-application projection."""

    gpu_command = [
        nvidia_smi,
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    app_command = [
        nvidia_smi,
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = _run_projection(
        gpu_command, timeout_seconds=timeout_seconds, run_command=run_command
    )
    app_result = _run_projection(
        app_command, timeout_seconds=timeout_seconds, run_command=run_command
    )
    gpu_output = (
        gpu_result.stdout.decode("utf-8", errors="strict")
        if isinstance(gpu_result.stdout, bytes)
        else str(gpu_result.stdout)
    )
    app_output = (
        app_result.stdout.decode("utf-8", errors="strict")
        if isinstance(app_result.stdout, bytes)
        else str(app_result.stdout)
    )
    gpu_rows = parse_gpu_rows(gpu_output, requested_gpus=requested_gpus)
    app_rows = parse_compute_app_rows(app_output)
    result: dict[str, GpuResource] = {}
    for index in requested_gpus:
        gpu_uuid, memory_used, utilization = gpu_rows[index]
        result[index] = GpuResource(
            index=index,
            uuid=gpu_uuid,
            memory_used_mib=memory_used,
            utilization_percent=utilization,
            compute_pids=app_rows.get(gpu_uuid, ()),
        )
    return result


def update_idle_streaks(
    previous: Mapping[str, int], probe: Mapping[str, GpuResource]
) -> dict[str, int]:
    if set(previous) != set(probe):
        raise WaiterError("idle streak keys do not match the requested GPU probe")
    return {
        gpu: previous[gpu] + 1 if probe[gpu].idle else 0 for gpu in previous
    }


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path | None

    @property
    def digest(self) -> str:
        return sha256_json(
            {"argv": list(self.argv), "cwd": None if self.cwd is None else str(self.cwd)}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": None if self.cwd is None else str(self.cwd),
            "command_digest": self.digest,
        }


def resolve_command(program: str, arguments: Sequence[str], cwd: Path | None) -> CommandSpec:
    if "/" in program:
        resolved = str(Path(program).resolve(strict=True))
    else:
        found = shutil.which(program)
        if found is None:
            raise WaiterError(f"command was not found: {program}")
        resolved = str(Path(found).resolve(strict=True))
    if not os.access(resolved, os.X_OK):
        raise WaiterError(f"command is not executable: {resolved}")
    if cwd is not None:
        cwd = cwd.resolve(strict=True)
        if not cwd.is_dir():
            raise WaiterError(f"command cwd is not a directory: {cwd}")
    return CommandSpec(argv=(resolved, *arguments), cwd=cwd)


@dataclass(frozen=True)
class ClaimOwnership:
    path: Path
    token: str
    metadata_path: Path


class SharedClaim:
    """A mkdir-based claim that retains released evidence by atomic rename."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.metadata_path = path / "claim.json"

    def try_acquire(self, metadata: Mapping[str, Any]) -> ClaimOwnership | None:
        if not self.path.parent.is_dir():
            raise WaiterError(f"claim parent does not exist: {self.path.parent}")
        token = uuid.uuid4().hex
        try:
            os.mkdir(self.path, mode=0o700)
        except FileExistsError:
            return None
        _fsync_directory(self.path.parent)
        payload = {
            **dict(metadata),
            "schema": CLAIM_SCHEMA,
            "claim_token": token,
            "state": "claimed",
            "claimed_at": utc_now(),
        }
        try:
            atomic_write_json(self.metadata_path, payload)
        except Exception:
            tombstone = self.path.with_name(
                f"{self.path.name}.initialization-failed-{token}"
            )
            os.replace(self.path, tombstone)
            _fsync_directory(self.path.parent)
            raise
        return ClaimOwnership(
            path=self.path, token=token, metadata_path=self.metadata_path
        )

    def read_if_present(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        if not self.path.is_dir():
            raise WaiterError(f"claim path exists but is not a directory: {self.path}")
        if not self.metadata_path.is_file():
            return {
                "state": "incomplete_claim",
                "claim_path": str(self.path),
                "diagnostic": "claim directory exists without claim.json",
            }
        return load_json_object(self.metadata_path)

    def update(
        self, ownership: ClaimOwnership, *, state: str, extra: Mapping[str, Any]
    ) -> dict[str, Any]:
        current = load_json_object(ownership.metadata_path)
        if current.get("claim_token") != ownership.token:
            raise WaiterError("refusing to update a claim owned by another token")
        updated = {
            **current,
            **dict(extra),
            "schema": CLAIM_SCHEMA,
            "claim_token": ownership.token,
            "state": state,
            "updated_at": utc_now(),
        }
        atomic_write_json(ownership.metadata_path, updated)
        return updated

    def release(
        self,
        ownership: ClaimOwnership,
        *,
        reason: str,
        diagnostics: Mapping[str, Any],
    ) -> Path:
        self.update(
            ownership,
            state="released",
            extra={
                "released_at": utc_now(),
                "release_reason": reason,
                "release_diagnostics": dict(diagnostics),
            },
        )
        tombstone = self.path.with_name(
            f"{self.path.name}.released-{ownership.token}"
        )
        if tombstone.exists():
            raise WaiterError(f"released-claim evidence already exists: {tombstone}")
        os.replace(self.path, tombstone)
        _fsync_directory(self.path.parent)
        return tombstone


@dataclass(frozen=True)
class WaiterConfig:
    host_id: str
    gpus: tuple[str, ...]
    nvidia_smi: str
    claim_dir: Path
    status_path: Path
    claim_busy_action: str
    launch: CommandSpec
    smoke: CommandSpec | None
    smoke_timeout_seconds: float
    probe_timeout_seconds: float


def command_environment(
    base: Mapping[str, str],
    *,
    config: WaiterConfig,
    ownership: ClaimOwnership,
    eligible_gpus: Sequence[str],
) -> dict[str, str]:
    return {
        **base,
        # The formal freeze verifier rejects local bytecode because Python may
        # import it instead of the attested source bytes.  The waiter execs the
        # queue interpreter directly, so preserve this constraint across exec.
        "PYTHONDONTWRITEBYTECODE": "1",
        "PLW_V02_WAITER_HOST": config.host_id,
        "PLW_V02_WAITER_CLAIM_TOKEN": ownership.token,
        "PLW_V02_WAITER_CLAIM_DIR": str(ownership.path),
        "PLW_V02_WAITER_ELIGIBLE_GPUS": ",".join(eligible_gpus),
    }


def run_smoke(
    command: CommandSpec,
    *,
    timeout_seconds: float,
    environment: Mapping[str, str],
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[bool, dict[str, Any]]:
    started = monotonic()
    timed_out = False
    try:
        completed = run_command(
            list(command.argv),
            cwd=None if command.cwd is None else str(command.cwd),
            env=dict(environment),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = None
        stdout = error.stdout
        stderr = error.stderr
        execution_error: dict[str, str] | None = None
    except OSError as error:
        timed_out = False
        returncode = None
        stdout = None
        stderr = None
        execution_error = {
            "type": type(error).__name__,
            "message": str(error),
        }
    else:
        execution_error = None
    duration = max(0.0, monotonic() - started)
    diagnostics = {
        **command.to_dict(),
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "returncode": returncode,
        "execution_error": execution_error,
        "duration_seconds": duration,
        "stdout": _diagnostic_stream(stdout),
        "stderr": _diagnostic_stream(stderr),
    }
    return (not timed_out and returncode == 0), diagnostics


class GpuWaiter:
    def __init__(
        self,
        config: WaiterConfig,
        *,
        collect_probe: Callable[..., dict[str, GpuResource]] = collect_gpu_probe,
        smoke_runner: Callable[..., tuple[bool, dict[str, Any]]] = run_smoke,
        execve: Callable[[str, Sequence[str], Mapping[str, str]], Any] = os.execve,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.collect_probe = collect_probe
        self.smoke_runner = smoke_runner
        self.execve = execve
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.claim = SharedClaim(config.claim_dir)
        self.idle_streaks = {gpu: 0 for gpu in config.gpus}
        self.probe_number = 0
        self.probe_errors = 0
        self.ownership: ClaimOwnership | None = None

    def _base_status(self) -> dict[str, Any]:
        return {
            "schema": STATUS_SCHEMA,
            "waiter_schema": SCHEMA,
            "updated_at": utc_now(),
            "host_id": self.config.host_id,
            "pid": os.getpid(),
            "requested_gpus": list(self.config.gpus),
            "claim_path": str(self.config.claim_dir),
            "claim_busy_action": self.config.claim_busy_action,
            "resource_contract": {
                "no_compute_application": True,
                "max_memory_used_mib": GPU_MEMORY_LIMIT_MIB,
                "max_utilization_percent": GPU_UTILIZATION_LIMIT_PERCENT,
                "required_consecutive_probes": REQUIRED_CONSECUTIVE_PROBES,
                "probe_interval_seconds": PROBE_INTERVAL_SECONDS,
            },
            "launch": self.config.launch.to_dict(),
            "smoke": None
            if self.config.smoke is None
            else self.config.smoke.to_dict(),
            "probe_number": self.probe_number,
            "probe_error_count": self.probe_errors,
            "idle_streaks": dict(self.idle_streaks),
        }

    def write_status(self, state: str, **extra: Any) -> None:
        atomic_write_json(
            self.config.status_path,
            {**self._base_status(), "state": state, **extra},
        )

    def _claim_metadata(
        self, *, eligible_gpus: Sequence[str], probe: Mapping[str, GpuResource]
    ) -> dict[str, Any]:
        return {
            "waiter_schema": SCHEMA,
            "host_id": self.config.host_id,
            "pid": os.getpid(),
            "requested_gpus": list(self.config.gpus),
            "eligible_gpus": list(eligible_gpus),
            "idle_streaks": dict(self.idle_streaks),
            "probe_number": self.probe_number,
            "probe": {gpu: probe[gpu].to_dict() for gpu in self.config.gpus},
            "resource_contract": {
                "no_compute_application": True,
                "max_memory_used_mib": GPU_MEMORY_LIMIT_MIB,
                "max_utilization_percent": GPU_UTILIZATION_LIMIT_PERCENT,
                "required_consecutive_probes": REQUIRED_CONSECUTIVE_PROBES,
                "probe_interval_seconds": PROBE_INTERVAL_SECONDS,
            },
            "launch": self.config.launch.to_dict(),
            "smoke": None
            if self.config.smoke is None
            else self.config.smoke.to_dict(),
        }

    def _release_owned_claim(
        self, *, reason: str, diagnostics: Mapping[str, Any]
    ) -> Path | None:
        if self.ownership is None:
            return None
        with _block_termination_signals():
            tombstone = self.claim.release(
                self.ownership, reason=reason, diagnostics=diagnostics
            )
            self.ownership = None
        return tombstone

    def _sleep_after_probe(self, probe_started: float) -> None:
        elapsed = max(0.0, self.monotonic() - probe_started)
        self.sleeper(max(0.0, PROBE_INTERVAL_SECONDS - elapsed))

    def run(self) -> int:
        self.write_status("observing")
        try:
            while True:
                probe_started = self.monotonic()
                self.probe_number += 1
                try:
                    probe = self.collect_probe(
                        nvidia_smi=self.config.nvidia_smi,
                        requested_gpus=self.config.gpus,
                        timeout_seconds=self.config.probe_timeout_seconds,
                    )
                    self.idle_streaks = update_idle_streaks(
                        self.idle_streaks, probe
                    )
                except (OSError, UnicodeError, WaiterError) as error:
                    self.probe_errors += 1
                    self.idle_streaks = {gpu: 0 for gpu in self.config.gpus}
                    self.write_status(
                        "probe_error",
                        latest_probe_error={
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    )
                    self._sleep_after_probe(probe_started)
                    continue

                eligible = tuple(
                    gpu
                    for gpu in self.config.gpus
                    if self.idle_streaks[gpu] >= REQUIRED_CONSECUTIVE_PROBES
                )
                probe_payload = {
                    gpu: probe[gpu].to_dict() for gpu in self.config.gpus
                }
                if not eligible:
                    self.write_status("observing", latest_probe=probe_payload)
                    self._sleep_after_probe(probe_started)
                    continue

                with _block_termination_signals():
                    ownership = self.claim.try_acquire(
                        self._claim_metadata(eligible_gpus=eligible, probe=probe)
                    )
                    if ownership is not None:
                        self.ownership = ownership
                if ownership is None:
                    existing = self.claim.read_if_present()
                    self.write_status(
                        "claim_busy",
                        eligible_gpus=list(eligible),
                        latest_probe=probe_payload,
                        existing_claim=existing,
                    )
                    if self.config.claim_busy_action == "exit":
                        return 3
                    self._sleep_after_probe(probe_started)
                    continue

                environment = command_environment(
                    os.environ,
                    config=self.config,
                    ownership=ownership,
                    eligible_gpus=eligible,
                )
                self.write_status(
                    "claim_acquired",
                    claim_token=ownership.token,
                    eligible_gpus=list(eligible),
                    latest_probe=probe_payload,
                )

                if self.config.smoke is not None:
                    self.claim.update(
                        ownership,
                        state="smoke_running",
                        extra={"smoke_started_at": utc_now()},
                    )
                    self.write_status(
                        "smoke_running",
                        claim_token=ownership.token,
                        eligible_gpus=list(eligible),
                    )
                    passed, diagnostics = self.smoke_runner(
                        self.config.smoke,
                        timeout_seconds=self.config.smoke_timeout_seconds,
                        environment=environment,
                    )
                    if not passed:
                        tombstone = self._release_owned_claim(
                            reason="smoke_failed", diagnostics=diagnostics
                        )
                        self.write_status(
                            "smoke_failed_claim_released",
                            eligible_gpus=list(eligible),
                            smoke_diagnostics=diagnostics,
                            released_claim_path=str(tombstone),
                        )
                        return 20
                    self.claim.update(
                        ownership,
                        state="smoke_passed",
                        extra={
                            "smoke_completed_at": utc_now(),
                            "smoke_diagnostics": diagnostics,
                        },
                    )

                self.claim.update(
                    ownership,
                    state="launch_exec_pending",
                    extra={
                        "launch_exec_at": utc_now(),
                        "eligible_gpus": list(eligible),
                    },
                )
                self.write_status(
                    "launch_exec_pending",
                    claim_token=ownership.token,
                    eligible_gpus=list(eligible),
                )
                if self.config.launch.cwd is not None:
                    os.chdir(self.config.launch.cwd)
                try:
                    self.execve(
                        self.config.launch.argv[0],
                        self.config.launch.argv,
                        environment,
                    )
                except OSError as error:
                    diagnostics = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "launch": self.config.launch.to_dict(),
                    }
                    tombstone = self._release_owned_claim(
                        reason="launch_exec_failed", diagnostics=diagnostics
                    )
                    self.write_status(
                        "launch_exec_failed_claim_released",
                        launch_diagnostics=diagnostics,
                        released_claim_path=str(tombstone),
                    )
                    return 21
                raise WaiterError("execve returned instead of replacing the waiter")
        except (KeyboardInterrupt, WaiterInterrupted) as error:
            diagnostics = {"type": type(error).__name__, "message": str(error)}
            tombstone = self._release_owned_claim(
                reason="waiter_interrupted", diagnostics=diagnostics
            )
            self.write_status(
                "interrupted",
                interruption=diagnostics,
                released_claim_path=None if tombstone is None else str(tombstone),
            )
            return 130
        except Exception as error:
            diagnostics = {"type": type(error).__name__, "message": str(error)}
            tombstone: Path | None = None
            release_error: dict[str, str] | None = None
            if self.ownership is not None:
                try:
                    tombstone = self._release_owned_claim(
                        reason="waiter_internal_error", diagnostics=diagnostics
                    )
                except Exception as claim_error:  # Preserve a fail-closed claim.
                    release_error = {
                        "type": type(claim_error).__name__,
                        "message": str(claim_error),
                    }
            self.write_status(
                "internal_error",
                error=diagnostics,
                claim_release_error=release_error,
                released_claim_path=None if tombstone is None else str(tombstone),
            )
            return 70


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a truly idle requested GPU, win one shared mkdir claim, "
            "optionally smoke-test the host, then exec the queue launcher."
        )
    )
    parser.add_argument("--host-id", required=True, type=_host_id)
    parser.add_argument("--gpus", required=True, type=parse_gpus)
    parser.add_argument("--nvidia-smi", required=True)
    parser.add_argument("--claim-dir", required=True, type=_absolute_path)
    parser.add_argument("--status-path", required=True, type=_absolute_path)
    parser.add_argument(
        "--claim-busy-action", required=True, choices=("exit", "observe")
    )
    parser.add_argument("--launch-program", required=True)
    parser.add_argument(
        "--launch-arg",
        action="append",
        default=[],
        help="one literal launcher argument; use --launch-arg=--option for flags",
    )
    parser.add_argument("--launch-cwd", type=_absolute_path)
    parser.add_argument("--smoke-program")
    parser.add_argument(
        "--smoke-arg",
        action="append",
        default=[],
        help="one literal smoke argument; use --smoke-arg=--option for flags",
    )
    parser.add_argument("--smoke-cwd", type=_absolute_path)
    parser.add_argument(
        "--smoke-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> WaiterConfig:
    nvidia_smi = resolve_command(args.nvidia_smi, (), None).argv[0]
    launch = resolve_command(args.launch_program, args.launch_arg, args.launch_cwd)
    if args.smoke_program is None:
        if args.smoke_arg or args.smoke_cwd is not None:
            raise WaiterError("smoke arguments/cwd require --smoke-program")
        smoke = None
    else:
        smoke = resolve_command(args.smoke_program, args.smoke_arg, args.smoke_cwd)
    if args.claim_dir == args.status_path or args.claim_dir in args.status_path.parents:
        raise WaiterError("status path must be outside the shared claim directory")
    return WaiterConfig(
        host_id=args.host_id,
        gpus=args.gpus,
        nvidia_smi=nvidia_smi,
        claim_dir=args.claim_dir,
        status_path=args.status_path,
        claim_busy_action=args.claim_busy_action,
        launch=launch,
        smoke=smoke,
        smoke_timeout_seconds=args.smoke_timeout_seconds,
        probe_timeout_seconds=args.probe_timeout_seconds,
    )


def _install_signal_handlers() -> None:
    def interrupt(signum: int, _frame: Any) -> None:
        raise WaiterInterrupted(f"received signal {signum}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
    except (OSError, WaiterError) as error:
        parser.error(str(error))
    _install_signal_handlers()
    return GpuWaiter(config).run()


if __name__ == "__main__":
    sys.exit(main())
