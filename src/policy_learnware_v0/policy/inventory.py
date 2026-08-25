"""Discover validated full-run policy bundles from the reproduction queue."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .bundle import BundleValidationError, PolicyBundleMetadata, validate_bundle


_SUCCESS_STATES = frozenset({"success", "succeeded", "complete", "completed"})
_FULL_JOB_ID = re.compile(
    r"^full__(?P<task>.+)__(?P<algorithm>ppo|fpo)__seed(?P<seed>\d+)$"
)


def _load_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _is_success(result: dict[str, Any]) -> bool:
    return str(result.get("state", "")).lower() in _SUCCESS_STATES and result.get(
        "returncode", 0
    ) in (0, None)


def resolve_successful_attempt(job_dir: str | Path) -> tuple[int, Path, dict[str, Any]]:
    """Return the successful attempt named by queue metadata, never a hard-coded attempt."""

    job_dir = Path(job_dir)
    root_result_path = job_dir / "queue_result.json"
    candidates: list[tuple[int, Path, dict[str, Any]]] = []

    if root_result_path.is_file():
        root_result = _load_object(root_result_path)
        if _is_success(root_result):
            attempt = int(root_result.get("attempt", 0))
            attempt_dir = job_dir / f"attempt_{attempt:02d}"
            if attempt > 0 and attempt_dir.is_dir():
                result_path = attempt_dir / "queue_result.json"
                result = _load_object(result_path) if result_path.is_file() else root_result
                if _is_success(result):
                    return attempt, attempt_dir, result

    # Queue metadata can be partially copied during an interrupted retry.  Scan
    # explicit attempt results as a recovery path, choosing the latest success.
    for attempt_dir in job_dir.glob("attempt_[0-9][0-9]"):
        try:
            attempt = int(attempt_dir.name.removeprefix("attempt_"))
            result = _load_object(attempt_dir / "queue_result.json")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if _is_success(result) and int(result.get("attempt", attempt)) == attempt:
            candidates.append((attempt, attempt_dir, result))
    if not candidates:
        raise ValueError(f"no successful queue attempt found in {job_dir}")
    return max(candidates, key=lambda item: item[0])


@dataclass(frozen=True)
class InventoryItem:
    job_id: str
    attempt: int
    metadata: PolicyBundleMetadata
    parity_passed: bool | None = None

    @property
    def bundle_dir(self) -> Path:
        return self.metadata.bundle_dir


@dataclass(frozen=True)
class InventoryRejection:
    job_id: str
    reason: str


@dataclass(frozen=True)
class InventoryReport:
    items: tuple[InventoryItem, ...]
    rejected: tuple[InventoryRejection, ...]
    checkpoint_outer: int

    def require_clean(self) -> "InventoryReport":
        if self.rejected:
            summary = "; ".join(f"{item.job_id}: {item.reason}" for item in self.rejected)
            raise BundleValidationError(f"inventory contains rejected jobs: {summary}")
        return self


def _full_runs_root(runs_root: Path) -> Path:
    if (runs_root / "full").is_dir():
        return runs_root / "full"
    return runs_root


def _load_job(attempt_dir: Path, job_dir: Path) -> dict[str, Any]:
    for path in (attempt_dir / "queue_job.json", job_dir / "queue_job.json"):
        if path.is_file():
            payload = _load_object(path)
            job = payload.get("job", payload)
            if isinstance(job, dict):
                return job
    raise ValueError("queue_job.json is missing or has no job object")


def scan_policy_inventory(
    runs_root: str | Path,
    *,
    checkpoint_outer: int,
    expected_environment_steps: int | None = None,
    parity_verifier: Callable[[PolicyBundleMetadata], Any] | None = None,
    require_parity: bool = False,
    expected_job_ids: Iterable[str] | None = None,
    expected_fpo_commit: str | None = None,
    expected_runtime_digest: str | None = None,
) -> InventoryReport:
    """Scan only the queue's ``full`` phase and resolve each successful attempt.

    ``parity_verifier`` is an explicit hook so discovery remains importable on a
    machine without JAX/MuJoCo.  Production inventory construction should pass
    the golden parity verifier and set ``require_parity=True``.
    """

    if require_parity and parity_verifier is None:
        raise ValueError("require_parity=True requires a parity_verifier")
    full_root = _full_runs_root(Path(runs_root))
    requested_ids = set(expected_job_ids) if expected_job_ids is not None else None
    directories = sorted(path for path in full_root.iterdir() if path.is_dir())
    if requested_ids is not None:
        found = {path.name for path in directories}
        directories = [path for path in directories if path.name in requested_ids]
        missing_ids = sorted(requested_ids.difference(found))
    else:
        missing_ids = []

    accepted: list[InventoryItem] = []
    rejected: list[InventoryRejection] = [
        InventoryRejection(job_id=job_id, reason="job directory is missing")
        for job_id in missing_ids
    ]
    for job_dir in directories:
        try:
            attempt, attempt_dir, _ = resolve_successful_attempt(job_dir)
            job = _load_job(attempt_dir, job_dir)
            phase = str(job.get("phase", "full"))
            if phase != "full":
                raise ValueError(f"queue job phase is {phase!r}, expected 'full'")
            task = str(job["task"])
            algorithm = str(job["algorithm"])
            seed = int(job["seed"])
            if job_dir.name.startswith("full__"):
                match = _FULL_JOB_ID.fullmatch(job_dir.name)
                if match is None:
                    raise ValueError("full-run directory has an invalid job id")
                directory_values = (
                    match.group("task"),
                    match.group("algorithm"),
                    int(match.group("seed")),
                )
                if directory_values != (task, algorithm, seed):
                    raise ValueError(
                        "full-run directory disagrees with queue_job metadata"
                    )
            bundle_dir = attempt_dir / "checkpoints" / f"outer_{checkpoint_outer:06d}"
            metadata = validate_bundle(
                bundle_dir,
                expected_task=task,
                expected_algorithm=algorithm,
                expected_seed=seed,
                expected_outer=checkpoint_outer,
                expected_environment_steps=expected_environment_steps,
                expected_fpo_commit=expected_fpo_commit,
                expected_runtime_digest=expected_runtime_digest,
            )
            parity_passed: bool | None = None
            if parity_verifier is not None:
                report = parity_verifier(metadata)
                parity_passed = bool(
                    report if isinstance(report, bool) else getattr(report, "passed", False)
                )
                if not parity_passed:
                    raise ValueError("golden parity failed")
            accepted.append(
                InventoryItem(
                    job_id=job_dir.name,
                    attempt=attempt,
                    metadata=metadata,
                    parity_passed=parity_passed,
                )
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            rejected.append(
                InventoryRejection(job_id=job_dir.name, reason=f"{type(error).__name__}: {error}")
            )

    accepted.sort(
        key=lambda item: (
            item.metadata.task,
            item.metadata.algorithm,
            item.metadata.training_seed,
            item.metadata.bundle_digest,
        )
    )
    return InventoryReport(tuple(accepted), tuple(rejected), int(checkpoint_outer))
