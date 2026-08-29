"""Build the three files needed by the v0.3 source-market runner.

The binder performs no formal admission, provenance replay, ledger
reconstruction, private-nonce commitment, or legacy-asset inventory.  It
loads the v0.2 pool and server plan, checks that every policy bundle is
runtime-loadable, then writes the evaluation protocol, work units and a small
receipt containing the input path and return normalization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.io import atomic_write_json, read_json
from policy_learnware_v0.v03.fpo_source_backend import (
    FpoJaxSourceEvaluatorBackend,
    FrozenV02FpoJaxRuntimeDriver,
)
from policy_learnware_v0.v03.pool_intake import V03PoolIntakeRecord
from policy_learnware_v0.v03.source_evaluator import (
    DmcFixedHorizonReturnContract,
    FrozenV02ServerPlanAuthority,
    plan_source_selection_work_units,
    source_work_unit_manifest,
)
from policy_learnware_v0.v03.source_market import SourceEvaluationProtocol


SELECTION_RESET_SEEDS = tuple(range(100_000, 100_025))
# The core source market is single pass.  This disjoint block remains only
# because the backward-compatible protocol record still carries the field.
UNUSED_ATTESTATION_RESET_SEEDS = tuple(range(200_000, 200_050))
RETURN_HORIZON = 1000


class AssetBindingError(ValueError):
    """The core pool, plan, runtime or destination is unusable."""


@dataclass(frozen=True)
class ProductionAssetBindingConfig:
    intake_record_path: str | Path
    server_plan_path: str | Path
    fpo_root: str | Path
    output_dir: str | Path
    allow_reconstructed_runtime: bool = False
    selection_reset_seeds: tuple[int, ...] = SELECTION_RESET_SEEDS


def _file(path: str | Path, where: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise AssetBindingError(f"{where} must be an absolute non-symlink file")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise AssetBindingError(f"{where} does not exist") from error
    if not resolved.is_file():
        raise AssetBindingError(f"{where} must be a file")
    return resolved


def _directory(path: str | Path, where: str) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise AssetBindingError(f"{where} must be an absolute non-symlink directory")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise AssetBindingError(f"{where} does not exist") from error
    if not resolved.is_dir():
        raise AssetBindingError(f"{where} must be a directory")
    return resolved


def _load_intake(path: Path) -> V03PoolIntakeRecord:
    value = read_json(path)
    if not isinstance(value, dict):
        raise AssetBindingError("intake record must contain a JSON object")
    try:
        return V03PoolIntakeRecord.from_dict(value)
    except (TypeError, ValueError) as error:
        raise AssetBindingError(f"cannot load v0.2 pool intake: {error}") from error


def bind_production_assets(config: ProductionAssetBindingConfig) -> dict[str, Any]:
    """Validate loadability and publish the minimal source-runner binding."""

    if not isinstance(config, ProductionAssetBindingConfig):
        raise AssetBindingError("config must be ProductionAssetBindingConfig")
    intake_path = _file(config.intake_record_path, "intake_record")
    plan_path = _file(config.server_plan_path, "server_plan")
    fpo_root = _directory(config.fpo_root, "fpo_root")
    output = Path(config.output_dir).expanduser()
    if not output.is_absolute() or output.is_symlink() or output.exists():
        raise AssetBindingError("output_dir must be a new absolute non-symlink path")
    output.parent.resolve(strict=True)

    intake = _load_intake(intake_path)
    plan = FrozenV02ServerPlanAuthority().load(plan_path)
    driver = FrozenV02FpoJaxRuntimeDriver(
        fpo_root=fpo_root,
        allow_reconstructed_runtime=config.allow_reconstructed_runtime,
    )
    runtime_evidence = dict(driver.preflight())
    backend = FpoJaxSourceEvaluatorBackend(
        runtime_driver=driver,
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=UNUSED_ATTESTATION_RESET_SEEDS,
    )
    return_contract = DmcFixedHorizonReturnContract(
        horizon=RETURN_HORIZON,
        per_step_lower=0.0,
        per_step_upper=1.0,
    )
    source_environments = {
        job.anchor.source_anchor_id: job.anchor.environment_instance_digest
        for job in plan.jobs.values()
    }
    protocol = SourceEvaluationProtocol(
        intake_record_digest=intake.intake_record_digest,
        evaluator_implementation_digest=backend.evaluator_implementation_digest,
        return_contract_digest=return_contract.return_contract_digest,
        selection_seed_namespace_digest=sha256_json(
            {"block": "source_selection", "seeds": config.selection_reset_seeds}
        ),
        attestation_seed_namespace_digest=sha256_json(
            {
                "block": "unused_source_attestation",
                "seeds": UNUSED_ATTESTATION_RESET_SEEDS,
            }
        ),
        selection_reset_seeds=config.selection_reset_seeds,
        attestation_reset_seeds=UNUSED_ATTESTATION_RESET_SEEDS,
        selection_episodes_per_candidate=len(config.selection_reset_seeds),
        attestation_episodes_per_champion=len(UNUSED_ATTESTATION_RESET_SEEDS),
        source_environment_digests=source_environments,
        competence_floors={anchor: 0.0 for anchor in source_environments},
        mean_tolerance=0.01,
        lcb_z=1.645,
    )
    units = plan_source_selection_work_units(intake, protocol, plan, backend)

    output.mkdir(mode=0o700)
    protocol_path = output / "source_evaluation_protocol.json"
    units_path = output / "source_selection_work_units.json"
    receipt_path = output / "asset_binding_receipt.json"
    protocol_sha = atomic_write_json(protocol_path, protocol.to_dict(), overwrite=False)
    units_sha = atomic_write_json(
        units_path, source_work_unit_manifest(units), overwrite=False
    )
    receipt = {
        "schema": "policy-learnware.v03-core-asset-binding.v1",
        "status": "READY",
        "intake": {"path": str(intake_path)},
        "return_contract": return_contract.to_dict(),
        "runtime": runtime_evidence,
        "files": {
            "source_evaluation_protocol.json": protocol_sha,
            "source_selection_work_units.json": units_sha,
        },
    }
    atomic_write_json(receipt_path, receipt, overwrite=False)
    return {**receipt, "output_dir": str(output), "candidate_count": len(units)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind a v0.2 policy pool to the minimal v0.3 source runner"
    )
    parser.add_argument("--intake-record", required=True, type=Path)
    parser.add_argument("--server-plan", required=True, type=Path)
    parser.add_argument("--fpo-root", required=True, type=Path)
    parser.add_argument(
        "--allow-reconstructed-runtime",
        action="store_true",
        help="explicitly permit attested reconstructed inference (never original replay)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = bind_production_assets(
        ProductionAssetBindingConfig(
            intake_record_path=args.intake_record,
            server_plan_path=args.server_plan,
            fpo_root=args.fpo_root,
            output_dir=args.output_dir,
            allow_reconstructed_runtime=args.allow_reconstructed_runtime,
        )
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "AssetBindingError",
    "ProductionAssetBindingConfig",
    "SELECTION_RESET_SEEDS",
    "bind_production_assets",
    "main",
]
