"""Resume-safe production runner for the v0.3 source policy market.

This is deliberately a thin server adapter.  It consumes the immutable P5R/P5M
binding directory, executes the already frozen source work units, and persists
the existing typed records.  It defines no new scientific or artifact schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from policy_learnware_v0.hashing import sha256_file, sha256_json
from policy_learnware_v0.io import atomic_write_json, read_json
from policy_learnware_v0.v03.formal_gates import (
    FormalMarketPlan,
    build_formal_market_evidence,
)
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
    source_work_unit_manifest,
)
from policy_learnware_v0.v03.source_market import (
    EvaluatorSourceReceipt,
    RawSourceEpisodeShard,
    SourceEvaluationProtocol,
    SourceEvaluationWorkUnit,
    build_source_evaluation_work_unit,
    build_source_policy_market,
    finalize_source_championization,
    formal_market_alias_protocol_digest,
    freeze_source_attestation_plan,
    market_nonce_commitment,
    provisionally_select_source_pool,
)


_BINDING_FILES = {
    "source_evaluation_protocol": "source_evaluation_protocol.json",
    "source_selection_work_units": "source_selection_work_units.json",
    "formal_market_plan": "formal_market_plan.json",
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


def _read_private_nonce(path: str | Path, where: str) -> str:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ValueError(f"{where} must be an absolute non-symlink file")
    resolved = supplied.resolve(strict=True)
    if not resolved.is_file() or stat.S_IMODE(resolved.stat().st_mode) != 0o600:
        raise ValueError(f"{where} must be a regular file with exact mode 0600")
    before = resolved.stat()
    text = resolved.read_text(encoding="ascii")
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"{where} changed while it was read")
    if text.endswith("\n"):
        text = text[:-1]
    if len(text) != 64 or text.lower() != text:
        raise ValueError(f"{where} must contain one lowercase SHA-256 value")
    try:
        int(text, 16)
    except ValueError as error:
        raise ValueError(f"{where} must contain one lowercase SHA-256 value") from error
    return text


def _load_binding(
    binding_dir: Path,
) -> tuple[
    dict[str, Any],
    V03PoolIntakeRecord,
    SourceEvaluationProtocol,
    dict[str, SourceEvaluationWorkUnit],
    FormalMarketPlan,
    DmcFixedHorizonReturnContract,
]:
    receipt = _object(binding_dir / "asset_binding_receipt.json", "asset binding receipt")
    if (
        receipt.get("status") != "ASSET_BINDINGS_READY"
        or receipt.get("operation") != "VALIDATE_ONLY"
        or receipt.get("candidate_validation_count") != 90
        or receipt.get("source_anchor_count") != 30
        or receipt.get("rollout_executed") is not False
        or receipt.get("training_executed") is not False
    ):
        raise ValueError("binding receipt is not the reviewed validate-only exact-90 input")
    observed_receipt_digest = receipt.get("binding_receipt_digest")
    if observed_receipt_digest != sha256_json(
        {name: value for name, value in receipt.items() if name != "binding_receipt_digest"}
    ):
        raise ValueError("asset binding receipt self digest drifted")
    if Path(receipt.get("output_directory", "")).resolve(strict=True) != binding_dir:
        raise ValueError("asset binding receipt belongs to another directory")

    documents: dict[str, dict[str, Any]] = {}
    artifact_bindings = receipt.get("artifacts")
    if not isinstance(artifact_bindings, Mapping):
        raise ValueError("asset binding receipt lacks artifact bindings")
    for label, filename in _BINDING_FILES.items():
        path = binding_dir / filename
        binding = artifact_bindings.get(label)
        if not isinstance(binding, Mapping):
            raise ValueError(f"asset binding receipt lacks {label}")
        if (
            Path(str(binding.get("path", ""))).resolve(strict=True) != path
            or sha256_file(path) != binding.get("file_sha256")
        ):
            raise ValueError(f"bound {label} bytes/path drifted")
        documents[label] = _object(path, label)

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
    if source_work_unit_manifest(units) != unit_document:
        raise ValueError("source-selection work-unit manifest drifted")
    plan = FormalMarketPlan.from_dict(documents["formal_market_plan"])

    intake_binding = receipt.get("intake")
    if not isinstance(intake_binding, Mapping):
        raise ValueError("asset binding receipt lacks intake binding")
    intake_path = Path(str(intake_binding.get("path", ""))).resolve(strict=True)
    if sha256_file(intake_path) != intake_binding.get("file_sha256"):
        raise ValueError("bound intake bytes drifted")
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

    if (
        intake.intake_record_digest != protocol.intake_record_digest
        or intake.intake_record_digest != plan.intake_record_digest
        or intake.source_pool_digest != plan.source_pool_digest
        or protocol.source_evaluation_protocol_digest
        != plan.source_evaluation_protocol_digest
        or set(units) != set(intake.cells)
        or any(
            plan.intake_cell_digests_by_candidate[candidate]
            != intake.cells[candidate].intake_cell_digest
            or plan.source_anchor_id_by_candidate[candidate]
            != intake.cells[candidate].source_anchor_id
            or plan.deployment_abi_digests_by_candidate[candidate]
            != unit.execution_abi.digest
            for candidate, unit in units.items()
        )
    ):
        raise ValueError("intake/protocol/work-unit/market-plan lineage drifted")
    return receipt, intake, protocol, units, plan, return_contract


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
        "attestation_total": 30,
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
        atomic_write_json(
            failure_path,
            error.attempt_record.to_dict(),
            overwrite=failure_path.exists(),
        )
        raise
    atomic_write_json(path, run.to_dict(), overwrite=False)
    return run


def run_source_market(
    *,
    binding_dir: str | Path,
    output_dir: str | Path,
    fpo_root: str | Path,
    vendor_dir: str | Path,
    market_alias_private_nonce_file: str | Path,
    tie_break_private_nonce_file: str | Path,
    max_selection: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute or resume the frozen 90-selection/30-attestation market build."""

    binding = _existing_directory(binding_dir, "binding_dir")
    receipt, intake, protocol, selection_units, plan, return_contract = _load_binding(
        binding
    )
    if max_selection is not None and (
        isinstance(max_selection, bool) or not 1 <= max_selection <= 90
    ):
        raise ValueError("max_selection must lie in [1, 90]")

    alias_nonce = _read_private_nonce(
        market_alias_private_nonce_file, "market alias private nonce file"
    )
    tie_nonce = _read_private_nonce(
        tie_break_private_nonce_file, "tie-break private nonce file"
    )
    if alias_nonce == tie_nonce:
        raise ValueError("market alias and tie-break private nonces must differ")
    if (
        market_nonce_commitment(
            purpose="market_alias",
            nonce=alias_nonce,
            intake_record_digest=intake.intake_record_digest,
        )
        != plan.market_alias_commitment_digest
        or market_nonce_commitment(
            purpose="market_tie_break",
            nonce=tie_nonce,
            intake_record_digest=intake.intake_record_digest,
        )
        != plan.tie_break_commitment_digest
        or formal_market_alias_protocol_digest(
            intake_record_digest=intake.intake_record_digest,
            source_pool_digest=intake.source_pool_digest,
            alias_commitment_digest=plan.market_alias_commitment_digest,
        )
        != plan.market_alias_protocol_digest
    ):
        raise ValueError("private market nonces do not reopen the frozen market plan")

    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("asset binding receipt lacks runtime binding")
    frozen_fpo_root = _existing_directory(fpo_root, "fpo_root")
    frozen_vendor = _existing_directory(vendor_dir, "vendor_dir")
    if (
        str(frozen_fpo_root) != runtime.get("fpo_root")
        or str(frozen_vendor) != runtime.get("vendor_dir")
    ):
        raise ValueError("runtime paths differ from the validated asset binding")
    driver = FrozenV02FpoJaxRuntimeDriver(
        fpo_root=frozen_fpo_root,
        vendor_dir=frozen_vendor,
    )
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=protocol.selection_reset_seeds,
        attestation_reset_seeds=protocol.attestation_reset_seeds,
    )
    if (
        driver.runtime_driver_digest != runtime.get("runtime_driver_digest")
        or backend.evaluator_implementation_digest
        != runtime.get("evaluator_implementation_digest")
        or backend.evaluator_implementation_digest
        != protocol.evaluator_implementation_digest
    ):
        raise ValueError("live FPO source backend differs from the validated binding")

    output = _prepare_output_directory(output_dir, resume=resume)
    selection_dir = output / "selection"
    attestation_dir = output / "attestation"
    failure_selection = output / "failures" / "selection"
    failure_attestation = output / "failures" / "attestation"
    for directory in (
        selection_dir,
        attestation_dir,
        failure_selection,
        failure_attestation,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    selection_runs: dict[str, SourceEvaluationRun] = {}
    executed_now = 0
    _progress(
        output,
        status="SELECTION_IN_PROGRESS",
        selection_complete=0,
        attestation_complete=0,
    )
    for candidate, unit in sorted(selection_units.items()):
        path = selection_dir / f"{candidate}.json"
        existed = path.exists()
        if not existed and max_selection is not None and executed_now >= max_selection:
            continue
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
            _progress(
                output,
                status="SELECTION_FAILED",
                selection_complete=len(selection_runs),
                attestation_complete=0,
                last_candidate_id=candidate,
            )
            raise
        if not path.exists():  # defensive; atomic publication above must have succeeded
            raise RuntimeError("source-selection run was not published")
        selection_runs[candidate] = run
        if not existed:
            executed_now += 1
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
    if len(selection_runs) != 90:
        return _progress(
            output,
            status="SELECTION_PAUSED",
            selection_complete=len(selection_runs),
            attestation_complete=0,
        )

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
    attestation_units = {
        candidate: build_source_evaluation_work_unit(
            intake,
            protocol,
            candidate,
            block="source_attestation",
            anchor_manifest_path=selection_units[candidate].anchor_manifest_path,
            execution_abi=selection_units[candidate].execution_abi,
        )
        for candidate in sorted(provisional.selected_candidate_ids.values())
    }
    attestation_plan = freeze_source_attestation_plan(
        intake,
        protocol,
        provisional,
        attestation_units,
    )
    _write_or_match(
        output / "attestation_plan.json",
        attestation_plan.to_dict(),
        resume=resume,
    )

    attestation_runs: dict[str, SourceEvaluationRun] = {}
    _progress(
        output,
        status="ATTESTATION_IN_PROGRESS",
        selection_complete=90,
        attestation_complete=0,
    )
    for candidate, unit in sorted(attestation_plan.units.items()):
        try:
            run = _execute_or_recover(
                unit=unit,
                path=attestation_dir / f"{candidate}.json",
                failure_path=failure_attestation / f"{candidate}.json",
                backend=backend,
                return_contract=return_contract,
                resume=resume,
            )
        except SourceEvaluationAttemptFailed:
            _progress(
                output,
                status="ATTESTATION_FAILED",
                selection_complete=90,
                attestation_complete=len(attestation_runs),
                last_candidate_id=candidate,
            )
            raise
        attestation_runs[candidate] = run
        _progress(
            output,
            status="ATTESTATION_IN_PROGRESS",
            selection_complete=90,
            attestation_complete=len(attestation_runs),
            last_candidate_id=candidate,
        )

    attestation_receipts = [
        attestation_runs[candidate].receipt for candidate in sorted(attestation_runs)
    ]
    championization = finalize_source_championization(
        intake,
        protocol,
        provisional,
        attestation_plan,
        attestation_receipts,
    )
    execution_abis = {
        candidate: attestation_plan.units[candidate].execution_abi
        for candidate in attestation_plan.units
    }
    market = build_source_policy_market(
        championization,
        execution_abis,
        market_alias_nonce=alias_nonce,
        tie_break_nonce=tie_nonce,
    )
    evidence = build_formal_market_evidence(
        plan=plan,
        intake_record_digest=intake.intake_record_digest,
        source_pool_digest=intake.source_pool_digest,
        protocol=protocol,
        selection_receipts=selection_receipts,
        attestation_receipts=attestation_receipts,
        championization=championization,
        market=market,
        market_alias_nonce=alias_nonce,
        tie_break_nonce=tie_nonce,
    )
    del alias_nonce, tie_nonce
    if evidence.failure_reasons:
        raise ValueError(
            "completed source market failed frozen-plan recomputation: "
            + ",".join(evidence.failure_reasons)
        )

    files: dict[str, dict[str, str]] = {}
    for name, value in (
        ("championization.json", championization.to_dict()),
        ("public_policy_market.json", market.public_manifest()),
        ("deployment_private_registry.json", market.deployment_manifest()),
        ("formal_market_evidence.json", evidence.to_dict()),
    ):
        files[name] = {
            "file_sha256": _write_or_match(output / name, value, resume=resume)
        }
    summary = {
        "status": "SOURCE_MARKET_READY",
        "source_market_ready": True,
        "training_executed": False,
        "intake_record_digest": intake.intake_record_digest,
        "source_evaluation_protocol_digest": protocol.source_evaluation_protocol_digest,
        "formal_market_plan_digest": plan.plan_digest,
        "selection_receipt_count": len(selection_receipts),
        "selection_episode_count": sum(row.episode_count for row in selection_receipts),
        "attestation_receipt_count": len(attestation_receipts),
        "attestation_episode_count": sum(
            row.episode_count for row in attestation_receipts
        ),
        "champion_count": len(championization.champions),
        "championization_digest": championization.championization_digest,
        "policy_market_id": market.policy_market_id,
        "formal_market_evidence_digest": evidence.evidence_digest,
        "market_alias_commitment_digest": plan.market_alias_commitment_digest,
        "tie_break_commitment_digest": plan.tie_break_commitment_digest,
        "files": files,
    }
    _write_or_match(output / "summary.json", summary, resume=resume)
    _progress(
        output,
        status="SOURCE_MARKET_READY",
        selection_complete=90,
        attestation_complete=30,
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
        description="Execute/resume the frozen v0.3 exact-90 source-market rollout"
    )
    parser.add_argument("--binding-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fpo-root", required=True, type=Path)
    parser.add_argument("--vendor-dir", required=True, type=Path)
    parser.add_argument(
        "--market-alias-private-nonce-file", required=True, type=Path
    )
    parser.add_argument("--tie-break-private-nonce-file", required=True, type=Path)
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
        vendor_dir=args.vendor_dir,
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
