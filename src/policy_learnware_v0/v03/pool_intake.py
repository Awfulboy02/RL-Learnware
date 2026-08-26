"""Fail-closed, read-only intake for the frozen v0.2 exact-90 handoff.

The v0.2 handoff is an acceptance overlay, not a ``TrainingRunRecord``.  This
module therefore validates the overlay and the bytes it references, then emits
a v0.3-owned immutable record with a distinct schema.  It never scans for
unlisted policies, repairs the v0.2 tree, or upgrades the overlay into a v0.2
training admission.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..hashing import canonical_json_bytes, canonicalize, sha256_file, sha256_json


class PoolIntakeError(ValueError):
    """The frozen handoff or one of its referenced immutable bytes is invalid."""


V02_ACCEPTANCE_SCHEMA = "policy-learnware.v02-policy-pool-acceptance.v0"
V02_PROMOTION_SCHEMA = "policy-learnware.v02-compiled-parity-promotion-set.v0"
V03_INTAKE_CELL_SCHEMA = "policy-learnware.v03-pool-intake-cell.v0"
V03_INTAKE_RECORD_SCHEMA = "policy-learnware.v03-v02-pool-intake.v0"

ACCEPTANCE_FILENAME = "policy_pool_acceptance.json"
PROMOTIONS_FILENAME = "compiled_parity_promotions.json"

EXPECTED_JOB_COUNT = 90
EXPECTED_ANCHOR_COUNT = 30
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_DIRECT_COUNT = 84
EXPECTED_PROMOTION_COUNT = 6

DIRECT_RESOLUTION = "direct_terminal_record"
PROMOTION_RESOLUTION = "compiled_parity_fallback_promotion"
PROMOTION_POLICY = "last_canonical_checkpoint_before_reloaded_compiled_parity_failure"
PROMOTION_FAILURE_CODE = "RELOADED_COMPILED_PARITY_FAILED"
PROMOTION_FAILURE_PREFIX = "reloaded compiled-policy parity failed: "

# The v0.2 replay is useful defence in depth, but it is executable code and
# therefore needs an authority of its own.  Production always loads these
# reviewed bytes from this repository-relative path; callers cannot substitute
# either the path or digest.  The independent v0.3 lineage checks below remain
# mandatory even when the replay result is injected by a hermetic fixture.
FROZEN_V02_VALIDATOR_REPO_PATH = "server/repro_fpo_ppo_v02/pool_acceptance.py"
FROZEN_V02_VALIDATOR_FILE_SHA256 = (
    "d18a3bcc587b2f353dcbe693a1d76519bbda1ffd86bb5b5edaee85bec95efe10"
)

_JOB_ID = re.compile(r"^v02j-[0-9a-f]{24}$")

AcceptanceReplayer = Callable[
    [Path, Path, Mapping[str, Any]],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class V02HandoffTrustAnchor:
    """Out-of-band digests reviewed for one immutable v0.2 handoff."""

    server_plan_digest: str
    queue_status_sha256: str
    promotions_file_sha256: str
    promotions_manifest_digest: str
    acceptance_file_sha256: str
    acceptance_report_digest: str
    pool_digest: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


_ACCEPTANCE_KEYS = {
    "schema",
    "decision",
    "accepted_at",
    "server_plan_digest",
    "queue_status_sha256",
    "promotion_manifest_digest",
    "job_count",
    "anchor_count",
    "seeds",
    "direct_terminal_record_count",
    "compiled_parity_fallback_promotion_count",
    "all_selected_bundles_finite",
    "all_selected_bundles_golden_parity_passed",
    "all_selected_bundles_compiled_parity_passed",
    "pool_digest",
    "cells",
    "report_digest",
}
_PROMOTION_MANIFEST_KEYS = {
    "schema",
    "server_plan_digest",
    "queue_status_sha256",
    "policy",
    "promotion_count",
    "created_at",
    "entries",
    "manifest_digest",
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
_BASE_CELL_KEYS = {
    "resolution",
    "job_id",
    "job_digest",
    "source_anchor_id",
    "seed",
    "attempt_number",
    "attempt_digest",
    "bundle_path",
    "bundle_digest",
    "outer_iteration",
    "environment_steps",
    "finiteness_audit_digest",
    "golden_parity_digest",
    "compiled_parity_digest",
}
_DIRECT_CELL_KEYS = _BASE_CELL_KEYS | {"training_record_digest", "terminal_record_state"}
_PROMOTED_CELL_KEYS = _BASE_CELL_KEYS | {
    "promotion_entry_digest",
    "failure_trace_digest",
    "failed_compiled_parity",
}
_BUNDLE_MANIFEST_KEYS = {
    "schema",
    "complete",
    "task",
    "algorithm",
    "seed",
    "outer_iteration",
    "environment_steps",
    "created_at",
    "files",
}
_TRAINING_PLAN_KEYS = {
    "schema",
    "config_digest",
    "execution_purpose",
    "formal_protocol_freeze",
    "formal_protocol_freeze_digest",
    "jobs",
    "expected_job_count",
    "plan_digest",
}
_TRAINING_JOB_KEYS = {
    "schema",
    "job_id",
    "config_digest",
    "execution_purpose",
    "formal_protocol_freeze_digest",
    "anchor_manifest_path",
    "anchor_manifest_digest",
    "training_protocol",
    "training_protocol_digest",
    "seed",
    "job_digest",
}
_QUEUE_STATUS_KEYS = {
    "schema",
    "plan",
    "plan_digest",
    "config_digest",
    "execution_purpose",
    "formal_protocol_freeze_digest",
    "execution_mode",
    "formal_eligible",
    "gpu_resource_gate",
    "gpus",
    "master_pid",
    "started_at",
    "updated_at",
    "state",
    "stop_signal",
    "vendor",
    "implementation",
    "jobs",
    "running",
    "counts",
    "terminal_record_counts",
}


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise PoolIntakeError(f"{where} must be a mapping")
    observed = set(value)
    if observed != expected:
        raise PoolIntakeError(
            f"{where} fields differ: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PoolIntakeError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result.lower() != result:
        raise PoolIntakeError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise PoolIntakeError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


FROZEN_V02_EXACT90_TRUST_ANCHOR = V02HandoffTrustAnchor(
    server_plan_digest="aa6a969ef70b5ca0d73b2cb3efb119860d3ca796d336c741fc2e9895ce05278c",
    queue_status_sha256="ff44038bb69bf3dea710acd3c20757ddcaefb1a33eb852365f5158a10d42f6f8",
    promotions_file_sha256="e544615f614012afa6ff45020a5bc922c05aab2ebd4dfd47bf36ad88b6aa8679",
    promotions_manifest_digest="9320fb3d8d3720d7dc16cc9077ad612b4de10530a48de64c9fb098d49f79446b",
    acceptance_file_sha256="cb133b4a3a15e739a111fb7245b04f20edf4d064a3c4c3850ba34a1f67f6a32d",
    acceptance_report_digest="5a6eba99a795019f036f2597aba0d4001238e353ba4b0f71ed9272f908fb5c00",
    pool_digest="e478ef1d38b7eea1a38691d4ea2bd25dc0356cd7264f5a5bd6df5e6de5e0d15f",
)


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PoolIntakeError(f"{where} must be a positive integer")
    return value


def _strict_json(path: Path, where: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PoolIntakeError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise PoolIntakeError(f"{where} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PoolIntakeError(f"cannot read strict JSON {where}: {error}") from error
    if not isinstance(value, dict):
        raise PoolIntakeError(f"{where} must be a JSON object")
    return value


def _replay_frozen_v02_acceptance(
    root: Path,
    _handoff: Path,
    promotion_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Re-run the frozen v0.2 authority against the original immutable tree.

    This is deliberately a read-only call into the frozen server validator.  It
    rechecks the canonical plan, final queue ledger, every direct terminal
    record, all promoted-failure lineage, and the referenced checkpoint bytes.
    """

    plan_path = root / "training_private" / "plans" / "server_training_plan.json"
    runs_root = root / "training_private" / "server_runs"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise PoolIntakeError("canonical frozen server training plan is missing or symlinked")
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise PoolIntakeError("canonical frozen server_runs directory is missing or symlinked")
    server_plan = _strict_json(plan_path, "canonical frozen server training plan")
    repository_root = Path(__file__).resolve().parents[3]
    validator_path = repository_root / FROZEN_V02_VALIDATOR_REPO_PATH
    if validator_path.is_symlink() or not validator_path.is_file():
        raise PoolIntakeError("canonical frozen v0.2 validator is missing or symlinked")
    validator_path = validator_path.resolve(strict=True)
    if sha256_file(validator_path) != FROZEN_V02_VALIDATOR_FILE_SHA256:
        raise PoolIntakeError(
            "canonical frozen v0.2 validator bytes differ from reviewed SHA-256"
        )

    # Execute the reviewed file itself instead of trusting import resolution or
    # a pre-existing ``sys.modules`` entry.  Its package is also required to
    # originate beside the reviewed implementation so relative imports cannot
    # silently resolve through another checkout.
    repository_text = str(repository_root)
    previous_sys_path = list(sys.path)
    module_name = (
        "server.repro_fpo_ppo_v02._v03_frozen_pool_acceptance_"
        + FROZEN_V02_VALIDATOR_FILE_SHA256[:12]
    )
    previous_module = sys.modules.get(module_name)
    try:
        sys.path.insert(0, repository_text)
        package = importlib.import_module("server.repro_fpo_ppo_v02")
        package_file = getattr(package, "__file__", None)
        if package_file is None or Path(package_file).resolve().parent != validator_path.parent:
            raise PoolIntakeError(
                "frozen v0.2 validator package resolved outside the canonical repository"
            )
        spec = importlib.util.spec_from_file_location(module_name, validator_path)
        if spec is None or spec.loader is None:
            raise PoolIntakeError("cannot construct canonical frozen v0.2 validator loader")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except (ImportError, OSError) as error:  # pragma: no cover - packaging/deployment gate
        raise PoolIntakeError(
            "frozen v0.2 pool-acceptance validator is unavailable"
        ) from error
    finally:
        sys.path[:] = previous_sys_path
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    accept_policy_pool = module.accept_policy_pool
    implementation_file = getattr(getattr(accept_policy_pool, "__code__", None), "co_filename", "")
    if not implementation_file or Path(implementation_file).resolve() != validator_path:
        raise PoolIntakeError(
            "frozen v0.2 acceptance callable did not originate from reviewed bytes"
        )
    try:
        return accept_policy_pool(
            server_plan=server_plan,
            runs_root=runs_root,
            promotion_manifest=promotion_manifest,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise PoolIntakeError(f"frozen v0.2 acceptance replay failed: {error}") from error


def _require_replay_matches_saved(
    saved: Mapping[str, Any],
    replayed: Mapping[str, Any],
) -> None:
    """Compare replay output while normalizing its sole clock-dependent field."""

    _strict(replayed, _ACCEPTANCE_KEYS, "replayed v0.2 policy-pool acceptance")
    _self_digest(replayed, key="report_digest", where="replayed v0.2 acceptance")
    normalized = dict(canonicalize(replayed))
    normalized["accepted_at"] = saved["accepted_at"]
    normalized["report_digest"] = sha256_json(
        {name: item for name, item in normalized.items() if name != "report_digest"}
    )
    if canonicalize(normalized) != canonicalize(saved):
        raise PoolIntakeError(
            "saved v0.2 acceptance differs from a fresh frozen-validator replay"
        )


def _self_digest(value: Mapping[str, Any], *, key: str, where: str) -> str:
    if key not in value:
        raise PoolIntakeError(f"{where} lacks {key}")
    observed = _digest(value[key], f"{where}.{key}")
    expected = sha256_json({name: item for name, item in value.items() if name != key})
    if observed != expected:
        raise PoolIntakeError(f"{where} self digest mismatch")
    return observed


def _resolved_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise PoolIntakeError("trusted experiment root must be an existing directory")
    return root


def _confined(
    raw: Any,
    *,
    root: Path,
    where: str,
    kind: str,
) -> Path:
    text = _nonempty(raw, where)
    supplied = Path(text)
    if not supplied.is_absolute():
        raise PoolIntakeError(f"{where} must be an absolute path")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise PoolIntakeError(f"{where} does not exist") from error
    if not resolved.is_relative_to(root):
        raise PoolIntakeError(f"{where} escapes the trusted experiment root")
    if kind == "file" and not resolved.is_file():
        raise PoolIntakeError(f"{where} must be a file")
    if kind == "dir" and not resolved.is_dir():
        raise PoolIntakeError(f"{where} must be a directory")
    return resolved


def _failure_payload(message: Any) -> dict[str, Any]:
    text = _nonempty(message, "promotion failure_message")
    if not text.startswith(PROMOTION_FAILURE_PREFIX):
        raise PoolIntakeError("promotion failure is not the reviewed parity-only class")
    try:
        payload = ast.literal_eval(text[len(PROMOTION_FAILURE_PREFIX) :])
    except (SyntaxError, ValueError) as error:
        raise PoolIntakeError("promotion failure payload is not a literal mapping") from error
    expected = {"passed", "max_abs_error", "atol", "rtol", "sample_count", "next_keys_equal"}
    _strict(payload, expected, "promotion failed_compiled_parity")
    if payload["passed"] is not False or payload["next_keys_equal"] is not True:
        raise PoolIntakeError("promotion failure must preserve PRNG keys and fail actions only")
    for name in ("max_abs_error", "atol", "rtol"):
        raw = payload[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise PoolIntakeError(f"promotion failure {name} must be finite")
        payload[name] = float(raw)
    _positive_int(payload["sample_count"], "promotion failure sample_count")
    if payload["max_abs_error"] <= 0.0:
        raise PoolIntakeError("promotion failure max_abs_error must be positive")
    return payload


@dataclass(frozen=True)
class PoolIntakeCell:
    job_id: str
    job_digest: str
    source_anchor_id: str
    seed: int
    resolution: str
    attempt_number: int
    attempt_digest: str
    bundle_path: str
    bundle_digest: str
    outer_iteration: int
    environment_steps: int
    finiteness_audit_digest: str
    golden_parity_digest: str
    compiled_parity_digest: str
    source_record_digest: str
    intake_cell_digest: str | None = None
    schema: str = V03_INTAKE_CELL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != V03_INTAKE_CELL_SCHEMA:
            raise PoolIntakeError("unsupported PoolIntakeCell schema")
        if not _JOB_ID.fullmatch(_nonempty(self.job_id, "job_id")):
            raise PoolIntakeError("job_id is not a frozen v0.2 opaque job ID")
        for name in (
            "job_digest",
            "source_anchor_id",
            "attempt_digest",
            "bundle_digest",
            "finiteness_audit_digest",
            "golden_parity_digest",
            "compiled_parity_digest",
            "source_record_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.seed not in EXPECTED_SEEDS:
            raise PoolIntakeError("pool cell seed must be one of 0/1/2")
        if self.resolution not in {DIRECT_RESOLUTION, PROMOTION_RESOLUTION}:
            raise PoolIntakeError("unsupported pool-cell resolution")
        object.__setattr__(self, "attempt_number", _positive_int(self.attempt_number, "attempt_number"))
        object.__setattr__(self, "outer_iteration", _positive_int(self.outer_iteration, "outer_iteration"))
        object.__setattr__(self, "environment_steps", _positive_int(self.environment_steps, "environment_steps"))
        _nonempty(self.bundle_path, "bundle_path")
        expected = sha256_json(self._payload_without_digest())
        if self.intake_cell_digest is None:
            object.__setattr__(self, "intake_cell_digest", expected)
        elif _digest(self.intake_cell_digest, "intake_cell_digest") != expected:
            raise PoolIntakeError("intake_cell_digest does not match cell contents")

    @property
    def candidate_id(self) -> str:
        return self.job_id

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "job_id": self.job_id,
            "job_digest": self.job_digest,
            "source_anchor_id": self.source_anchor_id,
            "seed": self.seed,
            "resolution": self.resolution,
            "attempt_number": self.attempt_number,
            "attempt_digest": self.attempt_digest,
            "bundle_path": self.bundle_path,
            "bundle_digest": self.bundle_digest,
            "outer_iteration": self.outer_iteration,
            "environment_steps": self.environment_steps,
            "finiteness_audit_digest": self.finiteness_audit_digest,
            "golden_parity_digest": self.golden_parity_digest,
            "compiled_parity_digest": self.compiled_parity_digest,
            "source_record_digest": self.source_record_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "intake_cell_digest": self.intake_cell_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoolIntakeCell":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "PoolIntakeCell")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class V03PoolIntakeRecord:
    trust_anchor_digest: str
    server_plan_digest: str
    queue_status_sha256: str
    promotions_file_sha256: str
    promotions_manifest_digest: str
    acceptance_file_sha256: str
    acceptance_report_digest: str
    source_pool_digest: str
    cells: Mapping[str, PoolIntakeCell]
    pool_state: str = "POOL_READY"
    intake_record_digest: str | None = None
    schema: str = V03_INTAKE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != V03_INTAKE_RECORD_SCHEMA:
            raise PoolIntakeError("unsupported V03PoolIntakeRecord schema")
        if self.pool_state != "POOL_READY":
            raise PoolIntakeError("a signed intake record can only represent POOL_READY")
        for name in (
            "trust_anchor_digest",
            "server_plan_digest",
            "queue_status_sha256",
            "promotions_file_sha256",
            "promotions_manifest_digest",
            "acceptance_file_sha256",
            "acceptance_report_digest",
            "source_pool_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        cells = dict(self.cells)
        if len(cells) != EXPECTED_JOB_COUNT or set(cells) != {cell.job_id for cell in cells.values()}:
            raise PoolIntakeError("v0.3 intake record must contain exactly 90 keyed cells")
        if any(not isinstance(cell, PoolIntakeCell) for cell in cells.values()):
            raise PoolIntakeError("v0.3 intake cells must be typed")
        anchors = {cell.source_anchor_id for cell in cells.values()}
        if len(anchors) != EXPECTED_ANCHOR_COUNT:
            raise PoolIntakeError("v0.3 intake record must contain exactly 30 anchors")
        for anchor in anchors:
            candidates = [cell for cell in cells.values() if cell.source_anchor_id == anchor]
            if len(candidates) != 3 or {cell.seed for cell in candidates} != set(EXPECTED_SEEDS):
                raise PoolIntakeError("every intake anchor must contain seeds 0/1/2 exactly once")
        direct_count = sum(
            cell.resolution == DIRECT_RESOLUTION for cell in cells.values()
        )
        promotion_count = sum(
            cell.resolution == PROMOTION_RESOLUTION for cell in cells.values()
        )
        if (
            direct_count != EXPECTED_DIRECT_COUNT
            or promotion_count != EXPECTED_PROMOTION_COUNT
        ):
            raise PoolIntakeError("v0.3 intake record must preserve the reviewed 84+6 lineage")
        observed_pool_digest = sha256_json(
            {
                job_id: {
                    "job_digest": cell.job_digest,
                    "source_anchor_id": cell.source_anchor_id,
                    "seed": cell.seed,
                    "bundle_digest": cell.bundle_digest,
                    "outer_iteration": cell.outer_iteration,
                    "environment_steps": cell.environment_steps,
                }
                for job_id, cell in sorted(cells.items())
            }
        )
        if observed_pool_digest != self.source_pool_digest:
            raise PoolIntakeError(
                "v0.3 intake source_pool_digest does not match its effective cells"
            )
        object.__setattr__(self, "cells", MappingProxyType(dict(sorted(cells.items()))))
        expected = sha256_json(self._payload_without_digest())
        if self.intake_record_digest is None:
            object.__setattr__(self, "intake_record_digest", expected)
        elif _digest(self.intake_record_digest, "intake_record_digest") != expected:
            raise PoolIntakeError("intake_record_digest does not match record contents")

    @property
    def candidates_by_anchor(self) -> Mapping[str, tuple[PoolIntakeCell, ...]]:
        result = {
            anchor: tuple(
                sorted(
                    (cell for cell in self.cells.values() if cell.source_anchor_id == anchor),
                    key=lambda cell: cell.seed,
                )
            )
            for anchor in sorted({cell.source_anchor_id for cell in self.cells.values()})
        }
        return MappingProxyType(result)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pool_state": self.pool_state,
            "trust_anchor_digest": self.trust_anchor_digest,
            "server_plan_digest": self.server_plan_digest,
            "queue_status_sha256": self.queue_status_sha256,
            "promotions_file_sha256": self.promotions_file_sha256,
            "promotions_manifest_digest": self.promotions_manifest_digest,
            "acceptance_file_sha256": self.acceptance_file_sha256,
            "acceptance_report_digest": self.acceptance_report_digest,
            "source_pool_digest": self.source_pool_digest,
            "cells": {job_id: cell.to_dict() for job_id, cell in self.cells.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "intake_record_digest": self.intake_record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V03PoolIntakeRecord":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "V03PoolIntakeRecord")
        return cls(
            **{
                name: (
                    {job_id: PoolIntakeCell.from_dict(cell) for job_id, cell in value[name].items()}
                    if name == "cells"
                    else value[name]
                )
                for name in fields
            }
        )


def _validate_bundle(cell: Mapping[str, Any], *, root: Path) -> str:
    bundle = _confined(cell["bundle_path"], root=root, where="cell.bundle_path", kind="dir")
    manifest_path = bundle / "bundle_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PoolIntakeError("bundle manifest is missing or symlinked")
    if sha256_file(manifest_path) != cell["bundle_digest"]:
        raise PoolIntakeError("bundle manifest SHA-256 differs from accepted bundle digest")
    manifest = _strict_json(manifest_path, "bundle manifest")
    _strict(manifest, _BUNDLE_MANIFEST_KEYS, "bundle manifest")
    if manifest["schema"] != "policy-learnware.policy-bundle.v0" or manifest["complete"] is not True:
        raise PoolIntakeError("referenced policy bundle is not complete schema v0")
    if manifest["algorithm"] != "fpo":
        raise PoolIntakeError("exact-90 handoff contains a non-FPO bundle")
    if (
        manifest["seed"] != cell["seed"]
        or manifest["outer_iteration"] != cell["outer_iteration"]
        or manifest["environment_steps"] != cell["environment_steps"]
    ):
        raise PoolIntakeError("bundle manifest geometry differs from acceptance cell")
    files = manifest["files"]
    if not isinstance(files, Mapping) or not files:
        raise PoolIntakeError("bundle manifest contains no payload files")
    for name, metadata in files.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise PoolIntakeError("bundle payload name is not a confined basename")
        _strict(metadata, {"bytes", "sha256"}, f"bundle payload {name!r}")
        expected_bytes = _positive_int(metadata["bytes"], f"bundle payload {name!r} bytes")
        expected_digest = _digest(metadata["sha256"], f"bundle payload {name!r} sha256")
        payload = bundle / name
        if payload.is_symlink() or not payload.is_file():
            raise PoolIntakeError(f"bundle payload {name!r} is missing or symlinked")
        if payload.stat().st_size != expected_bytes or sha256_file(payload) != expected_digest:
            raise PoolIntakeError(f"bundle payload {name!r} differs from its manifest")
    return str(bundle)


def _validate_plan_and_queue(
    *,
    root: Path,
    trust_anchor: V02HandoffTrustAnchor,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, str],
]:
    plan_path = root / "training_private" / "plans" / "server_training_plan.json"
    queue_path = root / "training_private" / "server_runs" / "queue_status.json"
    if not plan_path.is_file() or plan_path.is_symlink():
        raise PoolIntakeError("canonical frozen server training plan is missing or symlinked")
    if not queue_path.is_file() or queue_path.is_symlink():
        raise PoolIntakeError("canonical final queue status is missing or symlinked")
    if sha256_file(queue_path) != trust_anchor.queue_status_sha256:
        raise PoolIntakeError("queue_status.json bytes differ from frozen trust anchor")
    plan = _strict_json(plan_path, "server training plan")
    queue = _strict_json(queue_path, "final queue status")
    _strict(plan, _TRAINING_PLAN_KEYS, "server training plan")
    _strict(queue, _QUEUE_STATUS_KEYS, "final queue status")
    if (
        plan["schema"] != "policy-learnware.v02-training-plan.v0"
        or plan["execution_purpose"] != "v02_freeze_ready"
        or plan["expected_job_count"] != EXPECTED_JOB_COUNT
        or _self_digest(plan, key="plan_digest", where="server training plan")
        != trust_anchor.server_plan_digest
    ):
        raise PoolIntakeError("server training plan differs from frozen exact-90 plan")
    config_digest = _digest(plan["config_digest"], "training plan config_digest")
    freeze_digest = _digest(
        plan["formal_protocol_freeze_digest"], "formal protocol freeze digest"
    )
    freeze = plan["formal_protocol_freeze"]
    if not isinstance(freeze, Mapping) or freeze.get("binding_digest") != freeze_digest:
        raise PoolIntakeError("training plan formal freeze binding is invalid")
    _self_digest(freeze, key="binding_digest", where="formal freeze binding")
    jobs_raw = plan["jobs"]
    if not isinstance(jobs_raw, list) or len(jobs_raw) != EXPECTED_JOB_COUNT:
        raise PoolIntakeError("server training plan must contain exactly 90 jobs")
    jobs: dict[str, Mapping[str, Any]] = {}
    anchor_ids: dict[str, str] = {}
    semantic_units: set[tuple[str, str, int]] = set()
    for raw in jobs_raw:
        _strict(raw, _TRAINING_JOB_KEYS, "server training job")
        if raw["schema"] != "policy-learnware.v02-training-job.v0":
            raise PoolIntakeError("unsupported server training job schema")
        job_id = _nonempty(raw["job_id"], "training job_id")
        if not _JOB_ID.fullmatch(job_id) or job_id in jobs:
            raise PoolIntakeError("training plan has invalid or duplicate opaque job IDs")
        if (
            raw["config_digest"] != config_digest
            or raw["execution_purpose"] != "v02_freeze_ready"
            or raw["formal_protocol_freeze_digest"] != freeze_digest
            or raw["seed"] not in EXPECTED_SEEDS
            or _self_digest(raw, key="job_digest", where=f"training job {job_id}")
            != raw["job_digest"]
        ):
            raise PoolIntakeError("training job differs from plan bindings")
        protocol = raw["training_protocol"]
        if not isinstance(protocol, Mapping):
            raise PoolIntakeError("training job lacks embedded training protocol")
        protocol_digest = _self_digest(
            protocol, key="protocol_digest", where="embedded training protocol"
        )
        if raw["training_protocol_digest"] != protocol_digest:
            raise PoolIntakeError("training job protocol digest differs from embedded protocol")
        anchor_supplied = Path(
            _nonempty(raw["anchor_manifest_path"], "training job anchor_manifest_path")
        )
        if not anchor_supplied.is_absolute():
            raise PoolIntakeError("training job anchor_manifest_path must be absolute")
        try:
            anchor_path = anchor_supplied.resolve(strict=True)
        except OSError as error:
            raise PoolIntakeError("training job anchor manifest does not exist") from error
        # Production anchor paths are outside the artifact root, but their exact
        # strings and manifest digests are already covered by the reviewed plan
        # digest.  Do not accept symlinks or any non-file target.
        if anchor_supplied.is_symlink() or not anchor_path.is_file():
            raise PoolIntakeError("source anchor manifest may not be symlinked")
        anchor = _strict_json(anchor_path, "source anchor manifest")
        anchor_manifest_digest = _self_digest(
            anchor, key="manifest_digest", where="source anchor manifest"
        )
        if raw["anchor_manifest_digest"] != anchor_manifest_digest:
            raise PoolIntakeError("training job anchor manifest digest differs from bytes")
        anchor_id = _digest(anchor.get("anchor_id"), "source anchor manifest anchor_id")
        unit = (anchor_manifest_digest, protocol_digest, raw["seed"])
        if unit in semantic_units:
            raise PoolIntakeError("training plan contains duplicate anchor/protocol/seed units")
        semantic_units.add(unit)
        jobs[job_id] = raw
        anchor_ids[job_id] = anchor_id

    training_contract = freeze.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise PoolIntakeError("formal freeze lacks its training contract")
    reviewed_anchor_ids = training_contract.get("source_anchor_ids")
    if not isinstance(reviewed_anchor_ids, list):
        raise PoolIntakeError("formal training contract lacks source_anchor_ids")
    reviewed_anchor_ids = [
        _digest(item, "formal training-contract source_anchor_id")
        for item in reviewed_anchor_ids
    ]
    if (
        len(reviewed_anchor_ids) != EXPECTED_ANCHOR_COUNT
        or len(set(reviewed_anchor_ids)) != EXPECTED_ANCHOR_COUNT
        or set(anchor_ids.values()) != set(reviewed_anchor_ids)
    ):
        raise PoolIntakeError("training plan anchors differ from the formal freeze contract")

    if (
        queue["schema"] != "policy-learnware.v02-queue-status.v0"
        or queue["plan_digest"] != trust_anchor.server_plan_digest
        or queue["config_digest"] != config_digest
        or queue["formal_protocol_freeze_digest"] != freeze_digest
        or queue["execution_purpose"] != "v02_freeze_ready"
        or queue["execution_mode"] != "formal_gpu"
        or queue["formal_eligible"] is not True
        or queue["state"] != "completed_with_failures"
        or queue["counts"] != {"failed": EXPECTED_PROMOTION_COUNT, "succeeded": EXPECTED_DIRECT_COUNT}
        or queue["terminal_record_counts"] != {"recovered": 57, "succeeded": 27}
        or bool(queue["running"])
    ):
        raise PoolIntakeError("final queue status is not the frozen completed 84+6 ledger")
    embedded_plan = _confined(
        queue["plan"], root=root, where="queue plan", kind="file"
    )
    if embedded_plan != plan_path.resolve(strict=True):
        raise PoolIntakeError("queue status references another training plan")
    queue_jobs = queue["jobs"]
    if not isinstance(queue_jobs, Mapping) or set(queue_jobs) != set(jobs):
        raise PoolIntakeError("queue terminal ledger does not cover exactly the 90 plan jobs")
    observed_counts = {"failed": 0, "succeeded": 0}
    observed_terminal = {"recovered": 0, "succeeded": 0}
    states: dict[str, Mapping[str, Any]] = {}
    for job_id, state in queue_jobs.items():
        if not isinstance(state, Mapping):
            raise PoolIntakeError("queue job state must be a mapping")
        status = state.get("state")
        if status == "succeeded":
            allowed = {
                frozenset({"attempt_dir", "attempts", "state", "terminal_record_state"}),
                frozenset(
                    {
                        "attempt_dir",
                        "attempts",
                        "state",
                        "terminal_record_state",
                        "returncode",
                        "validation_error",
                    }
                ),
            }
            if frozenset(state) not in allowed or state.get("terminal_record_state") not in {
                "succeeded",
                "recovered",
            }:
                raise PoolIntakeError("successful queue cell has invalid terminal state")
            observed_terminal[state["terminal_record_state"]] += 1
        elif status == "failed":
            _strict(
                state,
                {
                    "attempt_dir",
                    "attempts",
                    "returncode",
                    "state",
                    "terminal_record_state",
                    "validation_error",
                },
                "failed queue cell",
            )
            if state["terminal_record_state"] is not None or state["validation_error"] is not None:
                raise PoolIntakeError("failed queue cell has fabricated terminal admission")
        else:
            raise PoolIntakeError("queue contains a nonterminal or unreviewed job state")
        attempts = _positive_int(state.get("attempts"), "queue attempt count")
        try:
            expected_attempt = (
                root
                / "training_private"
                / "server_runs"
                / "jobs"
                / job_id
                / f"attempt_{attempts:03d}"
            ).resolve(strict=True)
        except OSError as error:
            raise PoolIntakeError("canonical queue attempt directory is missing") from error
        observed_attempt = _confined(
            state.get("attempt_dir"), root=root, where="queue attempt_dir", kind="dir"
        )
        if observed_attempt != expected_attempt:
            raise PoolIntakeError("queue attempt path is not canonical")
        observed_counts[status] += 1
        states[job_id] = state
    if observed_counts != queue["counts"] or observed_terminal != queue["terminal_record_counts"]:
        raise PoolIntakeError("queue summary counts differ from terminal ledger")
    return jobs, states, anchor_ids


def _validate_direct_lineage(
    cell: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    state: Mapping[str, Any],
    root: Path,
    plan_digest: str,
) -> None:
    if state.get("state") != "succeeded" or state.get("terminal_record_state") != cell["terminal_record_state"]:
        raise PoolIntakeError("direct acceptance cell differs from final queue state")
    if state.get("attempts") != cell["attempt_number"]:
        raise PoolIntakeError("direct acceptance attempt number differs from queue")
    attempt = _confined(state["attempt_dir"], root=root, where="direct attempt_dir", kind="dir")
    attempt_manifest = _strict_json(attempt / "attempt_manifest.json", "direct attempt manifest")
    if (
        attempt_manifest.get("schema") != "policy-learnware.v02-training-attempt.v0"
        or _self_digest(attempt_manifest, key="attempt_digest", where="direct attempt manifest")
        != cell["attempt_digest"]
        or attempt_manifest.get("job_digest") != cell["job_digest"]
        or attempt_manifest.get("plan_digest") != plan_digest
        or attempt_manifest.get("attempt_number") != cell["attempt_number"]
        or attempt_manifest.get("execution_purpose") != "v02_freeze_ready"
        or attempt_manifest.get("execution_mode") != "formal_gpu"
        or attempt_manifest.get("formal_eligible") is not True
        or attempt_manifest.get("job") != job
    ):
        raise PoolIntakeError("direct attempt manifest differs from accepted cell/plan")
    record_path = attempt / "training_record.json"
    record = _strict_json(record_path, "direct training record")
    record_digest = _self_digest(record, key="record_digest", where="direct training record")
    if (
        record.get("schema") != "policy-learnware.v02-training-record.v1"
        or record_digest != cell["training_record_digest"]
        or record.get("state") != cell["terminal_record_state"]
        or record.get("job_digest") != cell["job_digest"]
        or record.get("attempt_digest") != cell["attempt_digest"]
        or record.get("anchor_manifest_digest") != job["anchor_manifest_digest"]
        or record.get("training_protocol_digest") != job["training_protocol_digest"]
        or record.get("seed") != cell["seed"]
        or record.get("promoted_outer_iteration") != cell["outer_iteration"]
        or record.get("promoted_environment_steps") != cell["environment_steps"]
    ):
        raise PoolIntakeError("direct training record differs from accepted effective checkpoint")
    checkpoints = record.get("checkpoint_bundles")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise PoolIntakeError("direct training record contains no checkpoint lineage")
    promoted = checkpoints[-1]
    if not isinstance(promoted, Mapping) or (
        promoted.get("outer_iteration") != cell["outer_iteration"]
        or promoted.get("environment_steps") != cell["environment_steps"]
        or promoted.get("bundle_digest") != cell["bundle_digest"]
        or _confined(promoted.get("path"), root=root, where="direct promoted path", kind="dir")
        != Path(cell["bundle_path"]).resolve(strict=True)
    ):
        raise PoolIntakeError("direct record final checkpoint differs from accepted bundle")
    for name, digest_name in (
        ("finiteness_audit", "finiteness_audit_digest"),
        ("golden_parity", "golden_parity_digest"),
        ("compiled_parity", "compiled_parity_digest"),
    ):
        report = promoted.get(name)
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            raise PoolIntakeError(f"direct promoted checkpoint {name} did not pass")
        if _self_digest(report, key="report_digest", where=f"direct promoted {name}") != cell[digest_name]:
            raise PoolIntakeError(f"direct promoted checkpoint {name} digest differs")


def _event_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PoolIntakeError(f"events.jsonl line {index} is invalid") from error
        if not isinstance(value, dict):
            raise PoolIntakeError(f"events.jsonl line {index} is not an object")
        rows.append(value)
    if not rows:
        raise PoolIntakeError("promotion events.jsonl is empty")
    return rows


def _validate_promotion_lineage(
    entry: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    state: Mapping[str, Any],
    root: Path,
    plan_digest: str,
) -> None:
    _strict(entry, _PROMOTION_ENTRY_KEYS, "promotion entry")
    _self_digest(entry, key="entry_digest", where="promotion entry")
    if (
        entry["job_id"] != cell["job_id"]
        or entry["job_digest"] != cell["job_digest"]
        or entry["attempt_number"] != cell["attempt_number"]
        or entry["attempt_digest"] != cell["attempt_digest"]
        or entry["promoted_outer_iteration"] != cell["outer_iteration"]
        or entry["promoted_bundle_digest"] != cell["bundle_digest"]
        or entry["entry_digest"] != cell["promotion_entry_digest"]
        or entry["traceback_sha256"] != cell["failure_trace_digest"]
        or entry["failure_code"] != PROMOTION_FAILURE_CODE
        or entry["failure_type"] != "ContractError"
    ):
        raise PoolIntakeError("promotion entry differs from accepted promoted cell")
    failed = _failure_payload(entry["failure_message"])
    if canonicalize(failed) != canonicalize(entry["failed_compiled_parity"]):
        raise PoolIntakeError("promotion failure payload differs from failure message")
    if canonicalize(failed) != canonicalize(cell["failed_compiled_parity"]):
        raise PoolIntakeError("accepted promotion failure payload differs from lineage")

    if (
        state.get("state") != "failed"
        or state.get("terminal_record_state") is not None
        or state.get("attempts") != cell["attempt_number"]
        or state.get("attempt_dir") != entry["attempt_dir"]
    ):
        raise PoolIntakeError("promoted acceptance cell differs from final queue state")

    attempt = _confined(entry["attempt_dir"], root=root, where="promotion attempt_dir", kind="dir")
    expected_attempt = root / "training_private" / "server_runs" / "jobs" / cell["job_id"] / f"attempt_{cell['attempt_number']:03d}"
    if attempt != expected_attempt.resolve(strict=True):
        raise PoolIntakeError("promotion attempt path is not canonical")
    traceback_path = _confined(entry["traceback_path"], root=root, where="promotion traceback_path", kind="file")
    events_path = _confined(entry["events_path"], root=root, where="promotion events_path", kind="file")
    if traceback_path != attempt / "traceback.txt" or events_path != attempt / "events.jsonl":
        raise PoolIntakeError("promotion evidence is not inside its canonical attempt")
    if sha256_file(traceback_path) != entry["traceback_sha256"]:
        raise PoolIntakeError("promotion traceback bytes changed")
    if sha256_file(events_path) != entry["events_sha256"]:
        raise PoolIntakeError("promotion event bytes changed")
    final_trace_line = next(
        (line for line in reversed(traceback_path.read_text(encoding="utf-8").splitlines()) if line.strip()),
        "",
    )
    if not final_trace_line.endswith(entry["failure_message"]):
        raise PoolIntakeError("promotion traceback does not end in the reviewed failure")

    attempt_manifest = _strict_json(attempt / "attempt_manifest.json", "promotion attempt manifest")
    _self_digest(attempt_manifest, key="attempt_digest", where="promotion attempt manifest")
    if (
        attempt_manifest.get("schema") != "policy-learnware.v02-training-attempt.v0"
        or attempt_manifest.get("attempt_number") != cell["attempt_number"]
        or attempt_manifest.get("attempt_digest") != cell["attempt_digest"]
        or attempt_manifest.get("job_digest") != cell["job_digest"]
        or attempt_manifest.get("plan_digest") != plan_digest
        or attempt_manifest.get("execution_purpose") != "v02_freeze_ready"
        or attempt_manifest.get("execution_mode") != "formal_gpu"
        or attempt_manifest.get("formal_eligible") is not True
    ):
        raise PoolIntakeError("promotion attempt manifest differs from frozen lineage")
    embedded_job = attempt_manifest.get("job")
    if not isinstance(embedded_job, Mapping) or canonicalize(embedded_job) != canonicalize(job):
        raise PoolIntakeError("promotion attempt embeds another job")
    ladder = embedded_job.get("training_protocol", {}).get("export_outer_iterations")
    if not isinstance(ladder, list) or cell["outer_iteration"] not in ladder:
        raise PoolIntakeError("promoted checkpoint is absent from the frozen ladder")
    index = ladder.index(cell["outer_iteration"])
    if index + 1 >= len(ladder) or ladder[index + 1] != entry["failed_outer_iteration"]:
        raise PoolIntakeError("promotion target is not immediately before failed export")

    rows = _event_rows(events_path)
    failures = [row for row in rows if row.get("event") == "run_failed"]
    checkpoints = [row for row in rows if row.get("event") == "checkpoint_published"]
    if len(failures) != 1 or rows[-1] is not failures[0] or not checkpoints:
        raise PoolIntakeError("promotion event lineage has invalid terminal structure")
    failure = failures[0]
    if (
        failure.get("error_type") != "ContractError"
        or failure.get("error") != entry["failure_message"]
        or failure.get("state") != "failed"
        or failure.get("job_digest") != cell["job_digest"]
        or failure.get("attempt_digest") != cell["attempt_digest"]
        or failure.get("last_completed_outer") != entry["failed_outer_iteration"] - 1
    ):
        raise PoolIntakeError("promotion run_failed event differs from frozen lineage")
    promoted = checkpoints[-1]
    if (
        promoted.get("outer_iteration") != cell["outer_iteration"]
        or promoted.get("bundle_digest") != cell["bundle_digest"]
        or promoted.get("environment_steps") != cell["environment_steps"]
        or _confined(promoted.get("path"), root=root, where="promoted event path", kind="dir")
        != Path(cell["bundle_path"]).resolve(strict=True)
    ):
        raise PoolIntakeError("last published checkpoint differs from promoted cell")
    for name, cell_name in (
        ("finiteness_audit", "finiteness_audit_digest"),
        ("golden_parity", "golden_parity_digest"),
        ("compiled_parity", "compiled_parity_digest"),
    ):
        report = promoted.get(name)
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            raise PoolIntakeError(f"promoted checkpoint {name} did not pass")
        if _self_digest(report, key="report_digest", where=f"promoted {name}") != cell[cell_name]:
            raise PoolIntakeError(f"promoted checkpoint {name} digest differs from cell")


def _parse_cell(
    job_id: str,
    raw: Mapping[str, Any],
    *,
    root: Path,
    promotions: Mapping[str, Any],
    jobs: Mapping[str, Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    anchor_ids: Mapping[str, str],
    plan_digest: str,
) -> PoolIntakeCell:
    job = jobs.get(job_id)
    state = states.get(job_id)
    if not isinstance(job, Mapping) or not isinstance(state, Mapping):
        raise PoolIntakeError("acceptance cell is absent from the frozen plan/queue")
    if raw.get("resolution") == DIRECT_RESOLUTION:
        _strict(raw, _DIRECT_CELL_KEYS, f"acceptance cell {job_id}")
        source_record_digest = _digest(raw["training_record_digest"], "training_record_digest")
        if raw["terminal_record_state"] not in {"succeeded", "recovered"}:
            raise PoolIntakeError("direct cell has unsupported terminal record state")
    elif raw.get("resolution") == PROMOTION_RESOLUTION:
        _strict(raw, _PROMOTED_CELL_KEYS, f"acceptance cell {job_id}")
        entry = promotions.get(job_id)
        if not isinstance(entry, Mapping):
            raise PoolIntakeError("promoted acceptance cell lacks promotion entry")
        source_record_digest = _digest(raw["promotion_entry_digest"], "promotion_entry_digest")
    else:
        raise PoolIntakeError("acceptance cell has unsupported resolution")
    if raw["job_id"] != job_id or not _JOB_ID.fullmatch(job_id):
        raise PoolIntakeError("acceptance cell key/job_id mismatch")
    if (
        raw["job_digest"] != job.get("job_digest")
        or raw["seed"] != job.get("seed")
        or raw["source_anchor_id"] != anchor_ids.get(job_id)
    ):
        raise PoolIntakeError("acceptance cell differs from its frozen training job")
    for name in (
        "job_digest",
        "source_anchor_id",
        "attempt_digest",
        "bundle_digest",
        "finiteness_audit_digest",
        "golden_parity_digest",
        "compiled_parity_digest",
    ):
        _digest(raw[name], f"cell {job_id}.{name}")
    if raw["seed"] not in EXPECTED_SEEDS:
        raise PoolIntakeError("acceptance cell seed is outside 0/1/2")
    _positive_int(raw["attempt_number"], "cell attempt_number")
    _positive_int(raw["outer_iteration"], "cell outer_iteration")
    _positive_int(raw["environment_steps"], "cell environment_steps")
    canonical_bundle = _validate_bundle(raw, root=root)
    expected_bundle = (
        root
        / "training_private"
        / "server_runs"
        / "jobs"
        / job_id
        / f"attempt_{raw['attempt_number']:03d}"
        / "checkpoints"
        / f"outer_{raw['outer_iteration']:06d}"
    ).resolve(strict=True)
    if Path(canonical_bundle) != expected_bundle:
        raise PoolIntakeError("accepted bundle path is not canonical for job/attempt/outer")
    if raw["resolution"] == DIRECT_RESOLUTION:
        _validate_direct_lineage(
            raw,
            job=job,
            state=state,
            root=root,
            plan_digest=plan_digest,
        )
    else:
        _validate_promotion_lineage(
            entry,
            raw,
            job=job,
            state=state,
            root=root,
            plan_digest=plan_digest,
        )
    return PoolIntakeCell(
        job_id=job_id,
        job_digest=raw["job_digest"],
        source_anchor_id=raw["source_anchor_id"],
        seed=raw["seed"],
        resolution=raw["resolution"],
        attempt_number=raw["attempt_number"],
        attempt_digest=raw["attempt_digest"],
        bundle_path=canonical_bundle,
        bundle_digest=raw["bundle_digest"],
        outer_iteration=raw["outer_iteration"],
        environment_steps=raw["environment_steps"],
        finiteness_audit_digest=raw["finiteness_audit_digest"],
        golden_parity_digest=raw["golden_parity_digest"],
        compiled_parity_digest=raw["compiled_parity_digest"],
        source_record_digest=source_record_digest,
    )


def _intake_v02_policy_pool(
    handoff_dir: str | Path,
    *,
    trusted_experiment_root: str | Path,
    trust_anchor: V02HandoffTrustAnchor,
    _acceptance_replayer: AcceptanceReplayer | None = None,
) -> V03PoolIntakeRecord:
    """Internal validator parameterized only to permit hermetic fixtures."""

    if not isinstance(trust_anchor, V02HandoffTrustAnchor):
        raise PoolIntakeError("intake requires a typed out-of-band trust anchor")
    root = _resolved_root(trusted_experiment_root)
    handoff = Path(handoff_dir).expanduser().resolve(strict=True)
    if not handoff.is_dir() or not handoff.is_relative_to(root):
        raise PoolIntakeError("handoff directory escapes the trusted experiment root")
    acceptance_path = handoff / ACCEPTANCE_FILENAME
    promotions_path = handoff / PROMOTIONS_FILENAME
    if acceptance_path.is_symlink() or promotions_path.is_symlink():
        raise PoolIntakeError("handoff manifests may not be symlinks")
    if sha256_file(acceptance_path) != trust_anchor.acceptance_file_sha256:
        raise PoolIntakeError("acceptance file SHA-256 differs from frozen trust anchor")
    if sha256_file(promotions_path) != trust_anchor.promotions_file_sha256:
        raise PoolIntakeError("promotions file SHA-256 differs from frozen trust anchor")

    acceptance = _strict_json(acceptance_path, "v0.2 policy-pool acceptance")
    promotions_manifest = _strict_json(promotions_path, "v0.2 promotion manifest")
    _strict(acceptance, _ACCEPTANCE_KEYS, "v0.2 policy-pool acceptance")
    _strict(promotions_manifest, _PROMOTION_MANIFEST_KEYS, "v0.2 promotion manifest")
    if acceptance["schema"] != V02_ACCEPTANCE_SCHEMA or acceptance["decision"] != "PASS":
        raise PoolIntakeError("v0.2 policy-pool acceptance is not a PASS schema v0")
    if promotions_manifest["schema"] != V02_PROMOTION_SCHEMA:
        raise PoolIntakeError("unsupported v0.2 promotion schema")
    report_digest = _self_digest(acceptance, key="report_digest", where="v0.2 acceptance")
    promotion_digest = _self_digest(
        promotions_manifest, key="manifest_digest", where="v0.2 promotion manifest"
    )
    if report_digest != trust_anchor.acceptance_report_digest:
        raise PoolIntakeError("acceptance self digest differs from frozen trust anchor")
    if promotion_digest != trust_anchor.promotions_manifest_digest:
        raise PoolIntakeError("promotion self digest differs from frozen trust anchor")
    jobs, states, anchor_ids = _validate_plan_and_queue(
        root=root,
        trust_anchor=trust_anchor,
    )
    replay = _acceptance_replayer or _replay_frozen_v02_acceptance
    replayed_acceptance = replay(root, handoff, promotions_manifest)
    if not isinstance(replayed_acceptance, Mapping):
        raise PoolIntakeError("frozen v0.2 acceptance replay did not return a mapping")
    _require_replay_matches_saved(acceptance, replayed_acceptance)
    if (
        acceptance["server_plan_digest"] != trust_anchor.server_plan_digest
        or promotions_manifest["server_plan_digest"] != trust_anchor.server_plan_digest
        or acceptance["queue_status_sha256"] != trust_anchor.queue_status_sha256
        or promotions_manifest["queue_status_sha256"] != trust_anchor.queue_status_sha256
        or acceptance["promotion_manifest_digest"] != promotion_digest
        or acceptance["pool_digest"] != trust_anchor.pool_digest
    ):
        raise PoolIntakeError("handoff plan/queue/promotion/pool bindings differ from trust anchor")
    if (
        acceptance["job_count"] != EXPECTED_JOB_COUNT
        or acceptance["anchor_count"] != EXPECTED_ANCHOR_COUNT
        or acceptance["seeds"] != list(EXPECTED_SEEDS)
        or acceptance["direct_terminal_record_count"] != EXPECTED_DIRECT_COUNT
        or acceptance["compiled_parity_fallback_promotion_count"] != EXPECTED_PROMOTION_COUNT
        or acceptance["all_selected_bundles_finite"] is not True
        or acceptance["all_selected_bundles_golden_parity_passed"] is not True
        or acceptance["all_selected_bundles_compiled_parity_passed"] is not True
    ):
        raise PoolIntakeError("v0.2 acceptance summary is not the reviewed exact-90 geometry")
    if (
        promotions_manifest["policy"] != PROMOTION_POLICY
        or promotions_manifest["promotion_count"] != EXPECTED_PROMOTION_COUNT
    ):
        raise PoolIntakeError("promotion manifest is not the reviewed six-cell policy")
    entries = promotions_manifest["entries"]
    cells_raw = acceptance["cells"]
    if not isinstance(entries, Mapping) or len(entries) != EXPECTED_PROMOTION_COUNT:
        raise PoolIntakeError("promotion entries are not an exact six-cell mapping")
    if not isinstance(cells_raw, Mapping) or len(cells_raw) != EXPECTED_JOB_COUNT:
        raise PoolIntakeError("acceptance cells are not an exact 90-cell mapping")
    promoted_ids = {
        job_id for job_id, cell in cells_raw.items()
        if isinstance(cell, Mapping) and cell.get("resolution") == PROMOTION_RESOLUTION
    }
    direct_ids = {
        job_id for job_id, cell in cells_raw.items()
        if isinstance(cell, Mapping) and cell.get("resolution") == DIRECT_RESOLUTION
    }
    if (
        len(direct_ids) != EXPECTED_DIRECT_COUNT
        or len(promoted_ids) != EXPECTED_PROMOTION_COUNT
        or set(entries) != promoted_ids
        or direct_ids | promoted_ids != set(cells_raw)
    ):
        raise PoolIntakeError("acceptance resolution coverage differs from reviewed 84+6")

    cells = {
        job_id: _parse_cell(
            job_id,
            raw,
            root=root,
            promotions=entries,
            jobs=jobs,
            states=states,
            anchor_ids=anchor_ids,
            plan_digest=trust_anchor.server_plan_digest,
        )
        for job_id, raw in sorted(cells_raw.items())
    }
    units = {(cell.source_anchor_id, cell.seed) for cell in cells.values()}
    anchors = {cell.source_anchor_id for cell in cells.values()}
    expected_units = {(anchor, seed) for anchor in anchors for seed in EXPECTED_SEEDS}
    if len(anchors) != EXPECTED_ANCHOR_COUNT or units != expected_units:
        raise PoolIntakeError("acceptance is not an exact 30-anchor x seeds-0/1/2 grid")
    observed_pool_digest = sha256_json(
        {
            job_id: {
                "job_digest": cell.job_digest,
                "source_anchor_id": cell.source_anchor_id,
                "seed": cell.seed,
                "bundle_digest": cell.bundle_digest,
                "outer_iteration": cell.outer_iteration,
                "environment_steps": cell.environment_steps,
            }
            for job_id, cell in sorted(cells.items())
        }
    )
    if observed_pool_digest != trust_anchor.pool_digest:
        raise PoolIntakeError("recomputed effective pool digest differs from frozen pool")
    return V03PoolIntakeRecord(
        trust_anchor_digest=trust_anchor.digest,
        server_plan_digest=trust_anchor.server_plan_digest,
        queue_status_sha256=trust_anchor.queue_status_sha256,
        promotions_file_sha256=trust_anchor.promotions_file_sha256,
        promotions_manifest_digest=trust_anchor.promotions_manifest_digest,
        acceptance_file_sha256=trust_anchor.acceptance_file_sha256,
        acceptance_report_digest=trust_anchor.acceptance_report_digest,
        source_pool_digest=trust_anchor.pool_digest,
        cells=cells,
    )


def intake_v02_policy_pool(
    handoff_dir: str | Path,
    *,
    trusted_experiment_root: str | Path,
) -> V03PoolIntakeRecord:
    """Validate the one reviewed production handoff and return ``POOL_READY``.

    The trust anchor is intentionally not caller-overridable on the production
    entry point.  Hermetic tests exercise the same implementation through the
    private parameterized helper.
    """

    return _intake_v02_policy_pool(
        handoff_dir,
        trusted_experiment_root=trusted_experiment_root,
        trust_anchor=FROZEN_V02_EXACT90_TRUST_ANCHOR,
    )


def _assert_intake_authority(
    record: V03PoolIntakeRecord,
    *,
    trust_anchor: V02HandoffTrustAnchor,
    trusted_experiment_root: str | Path | None = None,
) -> None:
    """Require one explicit trust anchor and optionally recheck bundle bytes.

    ``V03PoolIntakeRecord`` remains a serializable value object so hermetic tests
    can exercise the protocol.  The public wrapper below fixes ``trust_anchor``
    to the one reviewed production value.
    """

    if not isinstance(record, V03PoolIntakeRecord) or record.pool_state != "POOL_READY":
        raise PoolIntakeError("production intake authority requires typed POOL_READY")
    if not isinstance(trust_anchor, V02HandoffTrustAnchor):
        raise PoolIntakeError("intake authority requires a typed trust anchor")
    anchor = trust_anchor
    expected = {
        "trust_anchor_digest": anchor.digest,
        "server_plan_digest": anchor.server_plan_digest,
        "queue_status_sha256": anchor.queue_status_sha256,
        "promotions_file_sha256": anchor.promotions_file_sha256,
        "promotions_manifest_digest": anchor.promotions_manifest_digest,
        "acceptance_file_sha256": anchor.acceptance_file_sha256,
        "acceptance_report_digest": anchor.acceptance_report_digest,
        "source_pool_digest": anchor.pool_digest,
    }
    drift = {
        name: {"expected": value, "observed": getattr(record, name)}
        for name, value in expected.items()
        if getattr(record, name) != value
    }
    if drift:
        raise PoolIntakeError(
            "v0.3 intake differs from the frozen production trust anchor: "
            f"{sorted(drift)}"
        )
    if trusted_experiment_root is None:
        return
    root = _resolved_root(trusted_experiment_root)
    for cell in record.cells.values():
        canonical_bundle = Path(_validate_bundle(cell.to_dict(), root=root))
        expected_bundle = (
            root
            / "training_private"
            / "server_runs"
            / "jobs"
            / cell.job_id
            / f"attempt_{cell.attempt_number:03d}"
            / "checkpoints"
            / f"outer_{cell.outer_iteration:06d}"
        ).resolve(strict=True)
        if canonical_bundle != expected_bundle:
            raise PoolIntakeError(
                "verified intake bundle path is not canonical for its job/attempt/outer"
            )


def assert_frozen_v02_intake_authority(
    record: V03PoolIntakeRecord,
    *,
    trusted_experiment_root: str | Path | None = None,
) -> None:
    """Require the one reviewed production P5R authority."""

    _assert_intake_authority(
        record,
        trust_anchor=FROZEN_V02_EXACT90_TRUST_ANCHOR,
        trusted_experiment_root=trusted_experiment_root,
    )


def _load_verified_v02_intake(
    artifact_path: str | Path,
    *,
    expected_artifact_sha256: str,
    trusted_experiment_root: str | Path,
    trust_anchor: V02HandoffTrustAnchor,
) -> V03PoolIntakeRecord:
    """Internal loader parameterized only for hermetic authority fixtures."""

    supplied = Path(artifact_path).expanduser()
    if supplied.is_symlink():
        raise PoolIntakeError("v0.3 intake artifact may not be a symlink")
    try:
        path = supplied.resolve(strict=True)
    except OSError as error:
        raise PoolIntakeError("v0.3 intake artifact does not exist") from error
    if not path.is_file():
        raise PoolIntakeError("v0.3 intake artifact must be a regular file")
    expected_sha = _digest(expected_artifact_sha256, "expected_artifact_sha256")
    if sha256_file(path) != expected_sha:
        raise PoolIntakeError("v0.3 intake artifact SHA-256 differs from authority")
    payload = _strict_json(path, "v0.3 persisted pool intake")
    if path.read_bytes() != canonical_json_bytes(payload) + b"\n":
        raise PoolIntakeError("v0.3 intake artifact is not canonical JSON")
    record = V03PoolIntakeRecord.from_dict(payload)
    _assert_intake_authority(
        record,
        trust_anchor=trust_anchor,
        trusted_experiment_root=trusted_experiment_root,
    )
    return record


def load_verified_frozen_v02_intake(
    artifact_path: str | Path,
    *,
    expected_artifact_sha256: str,
    trusted_experiment_root: str | Path,
) -> V03PoolIntakeRecord:
    """Load a persisted P5R record through the production trust boundary."""

    return _load_verified_v02_intake(
        artifact_path,
        expected_artifact_sha256=expected_artifact_sha256,
        trusted_experiment_root=trusted_experiment_root,
        trust_anchor=FROZEN_V02_EXACT90_TRUST_ANCHOR,
    )


__all__ = [
    "FROZEN_V02_EXACT90_TRUST_ANCHOR",
    "PoolIntakeCell",
    "PoolIntakeError",
    "V02HandoffTrustAnchor",
    "V03PoolIntakeRecord",
    "assert_frozen_v02_intake_authority",
    "intake_v02_policy_pool",
    "load_verified_frozen_v02_intake",
]
