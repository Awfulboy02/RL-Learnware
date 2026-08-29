"""Resume-safe core runner for the v0.3 source policy market.

It consumes the existing P5R/P5M binding, runs each available policy once, and
selects one runnable policy per source anchor.  Stored parity, provenance and
quality thresholds are metrics; only load/ABI/finite/real-rollout failures can
exclude a candidate.  A second admission-only attestation rollout is not run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from policy_learnware_v0.hashing import sha256_file, sha256_json
from policy_learnware_v0.io import atomic_write_json, read_json
from policy_learnware_v0.v03.fpo_source_backend import (
    FpoJaxSourceEvaluatorBackend,
    FrozenV02FpoJaxRuntimeDriver,
)
from policy_learnware_v0.v03.pool_intake import V03PoolIntakeRecord
from policy_learnware_v0.v03.source_evaluator import (
    DmcFixedHorizonReturnContract,
    SourceEvaluationAttemptFailed,
    SourceEvaluationAttemptRecord,
    SourceEvaluationRun,
    run_source_evaluation_work_unit,
)
from policy_learnware_v0.v03.source_market import (
    EvaluatorSourceReceipt,
    RawSourceEpisodeShard,
    SourceEvaluationProtocol,
    SourceEvaluationWorkUnit,
    build_source_policy_market,
    championize_from_selection,
    provisionally_select_source_pool,
)


_BINDING_FILES = {
    "source_evaluation_protocol": "source_evaluation_protocol.json",
    "source_selection_work_units": "source_selection_work_units.json",
}


def _object(path: Path, where: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{where} must be a regular non-symlink file")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{where} must contain a JSON object")
    return value


def _existing_directory(path: str | Path, where: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError(f"{where} must be an absolute non-symlink directory")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{where} must be a directory")
    return resolved


def _prepare_output_directory(path: str | Path, *, resume: bool) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError("output_dir must be an absolute non-symlink path")
    if supplied.exists():
        resolved = supplied.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("output_dir exists but is not a directory")
        if not resume:
            raise ValueError("output_dir already exists; pass --resume to continue it")
        return resolved
    parent = supplied.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("output_dir parent must be an existing directory")
    destination = parent / supplied.name
    destination.mkdir(mode=0o700)
    return destination.resolve(strict=True)


def _market_nonce(path: str | Path | None, domain: str) -> str:
    if path is None:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-development-market-namespace.v0",
                "domain": domain,
            }
        )
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError(f"{domain} nonce must be an absolute non-symlink file")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{domain} nonce must be a regular file")
    text = resolved.read_text(encoding="ascii").strip()
    if not text:
        raise ValueError(f"{domain} nonce must not be empty")
    return text


def _load_binding(
    binding_dir: Path,
) -> tuple[
    V03PoolIntakeRecord,
    SourceEvaluationProtocol,
    dict[str, SourceEvaluationWorkUnit],
    DmcFixedHorizonReturnContract,
]:
    receipt = _object(binding_dir / "asset_binding_receipt.json", "asset binding receipt")
    documents = {
        label: _object(binding_dir / filename, label)
        for label, filename in _BINDING_FILES.items()
    }

    protocol = SourceEvaluationProtocol.from_dict(
        documents["source_evaluation_protocol"]
    )
    unit_document = documents["source_selection_work_units"]
    raw_units = unit_document.get("work_units")
    if not isinstance(raw_units, Mapping):
        raise ValueError("source-selection manifest lacks keyed work units")
    units = {
        str(candidate): SourceEvaluationWorkUnit.from_dict(unit)
        for candidate, unit in raw_units.items()
    }
    if not units:
        raise ValueError("source-selection manifest contains no work units")
    intake_binding = receipt.get("intake")
    if not isinstance(intake_binding, Mapping):
        raise ValueError("asset binding receipt lacks intake binding")
    intake_path = Path(str(intake_binding.get("path", ""))).resolve(strict=True)
    intake = V03PoolIntakeRecord.from_dict(_object(intake_path, "P5R intake"))

    return_document = receipt.get("return_contract")
    if not isinstance(return_document, Mapping) or return_document.get("projection") != (
        "affine_fixed_horizon_no_clip"
    ):
        raise ValueError("asset binding receipt has an invalid return contract")
    return_contract = DmcFixedHorizonReturnContract(
        horizon=return_document["horizon"],
        per_step_lower=return_document["per_step_lower"],
        per_step_upper=return_document["per_step_upper"],
        return_contract_digest=return_document["return_contract_digest"],
        schema=return_document["schema"],
    )

    if set(units) != set(intake.cells):
        raise ValueError("pool and source work units name different candidates")
    return intake, protocol, units, return_contract


def _load_run(path: Path, unit: SourceEvaluationWorkUnit) -> SourceEvaluationRun:
    value = _object(path, f"persisted {unit.block} run")
    expected = {"schema", "attempt", "raw_episode_shard", "receipt", "run_digest"}
    if set(value) != expected:
        raise ValueError(f"persisted run {path} has unexpected fields")
    run = SourceEvaluationRun(
        attempt=SourceEvaluationAttemptRecord.from_dict(value["attempt"]),
        raw_episode_shard=RawSourceEpisodeShard.from_dict(value["raw_episode_shard"]),
        receipt=EvaluatorSourceReceipt.from_dict(value["receipt"]),
        run_digest=value["run_digest"],
        schema=value["schema"],
    )
    receipt = run.receipt
    if (
        receipt.candidate_id != unit.candidate_id
        or receipt.block != unit.block
        or receipt.work_unit_digest != unit.work_unit_digest
        or receipt.source_evaluation_protocol_digest
        != unit.source_evaluation_protocol_digest
        or receipt.intake_record_digest != unit.intake_record_digest
        or receipt.reset_seeds != unit.reset_seeds
    ):
        raise ValueError(f"persisted run {path} belongs to another work unit")
    return run


def _write_or_match(path: Path, value: Mapping[str, Any], *, resume: bool) -> str:
    if path.exists():
        if not resume:
            raise ValueError(f"refusing to overwrite existing artifact: {path}")
        if _object(path, str(path)) != dict(value):
            raise ValueError(f"existing artifact differs from deterministic recompute: {path}")
        return sha256_file(path)
    return atomic_write_json(path, value, overwrite=False)


def _progress(
    output_dir: Path,
    *,
    status: str,
    selection_complete: int,
    attestation_complete: int,
    last_candidate_id: str | None = None,
    policy_market_id: str | None = None,
) -> dict[str, Any]:
    value = {
        "status": status,
        "selection_complete": selection_complete,
        "selection_total": 90,
        "attestation_complete": attestation_complete,
        "attestation_total": 0,
        "source_market_ready": status == "SOURCE_MARKET_READY",
        "last_candidate_id": last_candidate_id,
        "policy_market_id": policy_market_id,
    }
    atomic_write_json(output_dir / "progress.json", value, overwrite=True)
    return value


def _execute_or_recover(
    *,
    unit: SourceEvaluationWorkUnit,
    path: Path,
    failure_path: Path,
    backend: FpoJaxSourceEvaluatorBackend,
    return_contract: DmcFixedHorizonReturnContract,
    resume: bool,
) -> SourceEvaluationRun:
    if path.exists():
        if not resume:
            raise ValueError(f"run output already exists: {path}")
        return _load_run(path, unit)
    try:
        run = run_source_evaluation_work_unit(
            unit,
            backend=backend,
            return_contract=return_contract,
        )
    except SourceEvaluationAttemptFailed as error:
        if not failure_path.exists():
            atomic_write_json(
                failure_path,
                error.attempt_record.to_dict(),
                overwrite=False,
            )
        raise
    atomic_write_json(path, run.to_dict(), overwrite=False)
    return run


def _publish_quality_metrics(
    backend: FpoJaxSourceEvaluatorBackend,
    *,
    path: Path,
    candidate_id: str,
    block: str,
    resume: bool,
) -> tuple[Mapping[str, Any], ...]:
    rows = backend.drain_quality_metrics()
    if not rows:
        return ()
    document = {
        "candidate_id": candidate_id,
        "block": block,
        "metrics": [dict(row) for row in rows],
    }
    _write_or_match(path, document, resume=resume)
    return rows


def run_source_market(
    *,
    binding_dir: str | Path,
    output_dir: str | Path,
    fpo_root: str | Path,
    allow_reconstructed_runtime: bool = False,
    market_alias_private_nonce_file: str | Path | None = None,
    tie_break_private_nonce_file: str | Path | None = None,
    max_selection: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute or resume the single-pass source-market build."""

    binding = _existing_directory(binding_dir, "binding_dir")
    intake, protocol, selection_units, return_contract = _load_binding(binding)
    if max_selection is not None and (
        isinstance(max_selection, bool) or not 1 <= max_selection <= 90
    ):
        raise ValueError("max_selection must lie in [1, 90]")

    alias_nonce = _market_nonce(market_alias_private_nonce_file, "market-alias")
    tie_nonce = _market_nonce(tie_break_private_nonce_file, "tie-break")
    if alias_nonce == tie_nonce:
        raise ValueError("market alias and tie-break private nonces must differ")
    frozen_fpo_root = _existing_directory(fpo_root, "fpo_root")
    driver = FrozenV02FpoJaxRuntimeDriver(
        fpo_root=frozen_fpo_root,
        allow_reconstructed_runtime=allow_reconstructed_runtime,
    )
    runtime_evidence = dict(driver.preflight())
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=protocol.selection_reset_seeds,
        attestation_reset_seeds=protocol.attestation_reset_seeds,
    )

    output = _prepare_output_directory(output_dir, resume=resume)
    selection_dir = output / "selection"
    failure_selection = output / "failures" / "selection"
    quality_selection = output / "metrics" / "selection"
    for directory in (
        selection_dir,
        failure_selection,
        quality_selection,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    selection_runs: dict[str, SourceEvaluationRun] = {}
    attempted_now = 0
    _progress(
        output,
        status="SELECTION_IN_PROGRESS",
        selection_complete=0,
        attestation_complete=0,
    )
    for candidate, unit in sorted(selection_units.items()):
        path = selection_dir / f"{candidate}.json"
        existed = path.exists()
        if not existed and max_selection is not None and attempted_now >= max_selection:
            continue
        if not existed:
            attempted_now += 1
        try:
            run = _execute_or_recover(
                unit=unit,
                path=path,
                failure_path=failure_selection / f"{candidate}.json",
                backend=backend,
                return_contract=return_contract,
                resume=resume,
            )
        except SourceEvaluationAttemptFailed:
            _publish_quality_metrics(
                backend,
                path=quality_selection / f"{candidate}.json",
                candidate_id=candidate,
                block="source_selection",
                resume=resume,
            )
            _progress(
                output,
                status="SELECTION_IN_PROGRESS",
                selection_complete=len(selection_runs),
                attestation_complete=0,
                last_candidate_id=candidate,
            )
            continue
        _publish_quality_metrics(
            backend,
            path=quality_selection / f"{candidate}.json",
            candidate_id=candidate,
            block="source_selection",
            resume=resume,
        )
        if not path.exists():  # defensive; atomic publication above must have succeeded
            raise RuntimeError("source-selection run was not published")
        selection_runs[candidate] = run
        _progress(
            output,
            status="SELECTION_IN_PROGRESS",
            selection_complete=len(selection_runs),
            attestation_complete=0,
            last_candidate_id=candidate,
        )

    # Load any existing receipts skipped by the per-invocation smoke cap.
    for candidate, unit in sorted(selection_units.items()):
        path = selection_dir / f"{candidate}.json"
        if candidate not in selection_runs and path.exists():
            selection_runs[candidate] = _load_run(path, unit)
    pending = [
        candidate
        for candidate in selection_units
        if not (selection_dir / f"{candidate}.json").exists()
        and not (failure_selection / f"{candidate}.json").exists()
    ]
    if pending:
        return _progress(
            output,
            status="SELECTION_PAUSED",
            selection_complete=len(selection_runs),
            attestation_complete=0,
        )

    failed_selection = sorted(set(selection_units) - set(selection_runs))
    successful_anchors = {
        selection_units[candidate].source_anchor_id for candidate in selection_runs
    }
    missing_anchors = sorted(set(intake.candidates_by_anchor) - successful_anchors)
    if missing_anchors:
        summary = {
            "status": "SOURCE_MARKET_INCOMPLETE",
            "source_market_ready": False,
            "formal_eligible": False,
            "selection_success_count": len(selection_runs),
            "failed_candidate_ids": failed_selection,
            "missing_source_anchor_ids": missing_anchors,
            "reason": "no candidate completed a real rollout for one or more anchors",
        }
        atomic_write_json(output / "summary.json", summary, overwrite=True)
        _progress(
            output,
            status="SOURCE_MARKET_INCOMPLETE",
            selection_complete=len(selection_runs),
            attestation_complete=0,
        )
        return summary

    selection_receipts = [
        selection_runs[candidate].receipt for candidate in sorted(selection_runs)
    ]
    provisional = provisionally_select_source_pool(
        intake,
        protocol,
        selection_units,
        selection_receipts,
    )
    _write_or_match(
        output / "provisional_selection.json",
        provisional.to_dict(),
        resume=resume,
    )
    championization = championize_from_selection(
        intake,
        protocol,
        provisional,
        selection_receipts,
    )
    execution_abis = {
        candidate: selection_units[candidate].execution_abi
        for candidate in provisional.selected_candidate_ids.values()
    }
    market = build_source_policy_market(
        championization,
        execution_abis,
        market_alias_nonce=alias_nonce,
        tie_break_nonce=tie_nonce,
    )
    del alias_nonce, tie_nonce

    files: dict[str, dict[str, str]] = {}
    for name, value in (
        ("championization.json", championization.to_dict()),
        ("public_policy_market.json", market.public_manifest()),
        ("deployment_private_registry.json", market.deployment_manifest()),
    ):
        files[name] = {
            "file_sha256": _write_or_match(output / name, value, resume=resume)
        }
    quality_rows = []
    for path in sorted((output / "metrics").glob("*/*.json")):
        document = _object(path, "policy quality metrics")
        quality_rows.extend(document.get("metrics", ()))
    warning_count = sum(
        isinstance(row, Mapping) and row.get("severity") == "WARNING"
        for row in quality_rows
    )
    summary = {
        "status": "SOURCE_MARKET_READY",
        "source_market_ready": True,
        "formal_eligible": False,
        "scope": "development/core-runtime",
        "evaluation_mode": "single-pass-real-rollout",
        "training_executed": False,
        "intake_record_digest": intake.intake_record_digest,
        "source_evaluation_protocol_digest": protocol.source_evaluation_protocol_digest,
        "selection_receipt_count": len(selection_receipts),
        "selection_failure_count": len(failed_selection),
        "selection_episode_count": sum(row.episode_count for row in selection_receipts),
        "attestation_receipt_count": 0,
        "attestation_episode_count": 0,
        "champion_count": len(championization.champions),
        "championization_digest": championization.championization_digest,
        "policy_market_id": market.policy_market_id,
        "quality_metric_count": len(quality_rows),
        "quality_warning_count": warning_count,
        "parity_warnings_are_non_blocking": True,
        "runtime": runtime_evidence,
        "files": files,
    }
    _write_or_match(output / "summary.json", summary, resume=resume)
    _progress(
        output,
        status="SOURCE_MARKET_READY",
        selection_complete=len(selection_runs),
        attestation_complete=0,
        policy_market_id=market.policy_market_id,
    )
    return summary


def _positive_selection_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 1 <= parsed <= 90:
        raise argparse.ArgumentTypeError("must lie in [1, 90]")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute/resume the v0.3 single-pass source-market rollout"
    )
    parser.add_argument("--binding-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fpo-root", required=True, type=Path)
    parser.add_argument(
        "--allow-reconstructed-runtime",
        action="store_true",
        help="explicitly permit attested reconstructed inference (never original replay)",
    )
    parser.add_argument(
        "--market-alias-private-nonce-file", type=Path
    )
    parser.add_argument("--tie-break-private-nonce-file", type=Path)
    parser.add_argument(
        "--max-selection",
        type=_positive_selection_limit,
        help="maximum new selection candidates in this invocation (smoke/resume)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_source_market(
        binding_dir=args.binding_dir,
        output_dir=args.output_dir,
        fpo_root=args.fpo_root,
        allow_reconstructed_runtime=args.allow_reconstructed_runtime,
        market_alias_private_nonce_file=args.market_alias_private_nonce_file,
        tie_break_private_nonce_file=args.tie_break_private_nonce_file,
        max_selection=args.max_selection,
        resume=args.resume,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - deployment entry point
    raise SystemExit(main())


__all__ = ["main", "run_source_market"]
