"""Fail-closed command surface for the v0.3 foundation sidecar.

The CLI intentionally separates engineering acceptance from formal scientific
gates.  In particular, the production pool-intake command has no option for
overriding the reviewed v0.2 trust anchor, and no command can mint source
evaluation receipts from caller-supplied summary statistics.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from ..hashing import canonical_json, sha256_json
from .acceptance import run_minimal_compute_acceptance
from .artifacts import V03ArtifactLayout
from .baselines import OPTIONAL_BASELINE_STATES, REQUIRED_BASELINE_METHOD_IDS
from .config import load_v03_foundation_config
from .pool_intake import V03PoolIntakeRecord, intake_v02_policy_pool
from .orchestration import (
    PRODUCTION_STAGE_IDS,
    StageExecutionAdapter,
    execute_stage_from_files,
    verify_pipeline_completion_from_files,
)
from .preflight import (
    IndependentRecomputeAttestation,
    OracleUnlockHandoff,
    PreExperimentFreezeManifest,
    PublicRankingBarrier,
)
from .prelarge_acceptance import run_prelarge_acceptance
from .signal_matrix import build_optimization_fit_jobs, build_signal_matrix_plan
from .statistics import (
    FormalStatisticsPlan,
    FrozenStatisticsInput,
    compute_formal_statistics,
)


CLI_SCHEMA = "policy-learnware.v03-cli-result.v0"
CLI_VERSION = "0.3.0"


class V03CommandError(RuntimeError):
    """A command input or immutable output violates the v0.3 boundary."""


def _strict_json(path: str | Path, *, where: str) -> Mapping[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file() or source.is_symlink():
        raise V03CommandError(f"{where} must be a regular JSON file")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise V03CommandError(f"{where} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise V03CommandError(f"{where} contains non-finite constant {value}")

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V03CommandError(f"cannot read {where}: {error}") from error
    if not isinstance(value, Mapping):
        raise V03CommandError(f"{where} must contain one JSON object")
    return value


def _result(command: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CLI_SCHEMA,
        "command": command,
        "status": status,
        "payload": dict(payload),
    }


def _publish_intake(
    record: V03PoolIntakeRecord,
    *,
    artifacts_root: str,
    development_id: str,
    resume: bool,
) -> tuple[str, str]:
    layout = V03ArtifactLayout.development(artifacts_root, development_id)
    path = layout.artifact("v02_pool_intake", "intake_record.json")
    digest = layout.writer("v02_pool_intake").publish_json(
        path,
        record.to_dict(),
        resume=resume,
    )
    return str(path), digest


def _publish_development_json(
    value: Mapping[str, Any],
    *,
    domain: str,
    filename: str,
    artifacts_root: str,
    development_id: str,
    resume: bool,
) -> tuple[str, str]:
    layout = V03ArtifactLayout.development(artifacts_root, development_id)
    path = layout.artifact(domain, filename)
    digest = layout.writer(domain).publish_json(path, value, resume=resume)
    return str(path), digest


def _add_optional_development_publication(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts-root")
    parser.add_argument("--development-id")
    parser.add_argument("--resume", action="store_true")


def _publication_args(args: argparse.Namespace) -> tuple[str, str] | None:
    root = getattr(args, "artifacts_root", None)
    development_id = getattr(args, "development_id", None)
    if (root is None) != (development_id is None):
        raise V03CommandError(
            "--artifacts-root and --development-id must be supplied together"
        )
    return None if root is None else (root, development_id)


def _add_real_stage_arguments(
    parser: argparse.ArgumentParser,
    *,
    artifacts_root_already_added: bool = False,
    resume_already_added: bool = False,
) -> None:
    parser.add_argument(
        "--stage-manifest",
        help="canonical typed stage manifest for real adapter execution",
    )
    parser.add_argument(
        "--freeze-manifest",
        help="canonical external pre-experiment freeze bound by the stage manifest",
    )
    parser.add_argument(
        "--adapter-entrypoint",
        help=(
            "explicit server adapter factory as module:attribute; the returned "
            "adapter remains subject to manifest and formal-freeze identity checks"
        ),
    )
    if not artifacts_root_already_added:
        parser.add_argument("--artifacts-root")
    if not resume_already_added:
        parser.add_argument("--resume", action="store_true")


def _require_real_stage_arguments(args: argparse.Namespace) -> None:
    missing = [
        option
        for option, value in (
            ("--stage-manifest", getattr(args, "stage_manifest", None)),
            ("--freeze-manifest", getattr(args, "freeze_manifest", None)),
            ("--artifacts-root", getattr(args, "artifacts_root", None)),
        )
        if value is None
    ]
    if missing:
        raise V03CommandError(
            "real production execution requires " + ", ".join(missing)
        )


def _load_stage_adapter_entrypoint(spec: str) -> StageExecutionAdapter:
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise V03CommandError(
            "--adapter-entrypoint must use the explicit module:attribute form"
        )
    module_name, attribute_name = spec.split(":", 1)
    if not module_name or not attribute_name or "." in attribute_name:
        raise V03CommandError(
            "--adapter-entrypoint must name one top-level factory attribute"
        )
    try:
        factory = getattr(importlib.import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as error:
        raise V03CommandError(f"cannot load stage adapter entrypoint: {error}") from error
    if not callable(factory):
        raise V03CommandError("stage adapter entrypoint must be a zero-argument factory")
    try:
        adapter = factory()
    except Exception as error:
        raise V03CommandError(f"stage adapter factory failed: {error}") from error
    if (
        not isinstance(getattr(adapter, "adapter_id", None), str)
        or not isinstance(getattr(adapter, "adapter_contract_digest", None), str)
        or not callable(getattr(adapter, "execute", None))
    ):
        raise V03CommandError(
            "stage adapter factory did not return the required adapter contract"
        )
    return adapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-learnware-v03",
        description="Policy Learnware v0.3 foundation and asset-intake tools",
    )
    parser.add_argument("--version", action="version", version=CLI_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-config", help="validate and digest a v0.3 foundation config"
    )
    validate.add_argument("config")

    subparsers.add_parser(
        "accept-numeric", help="run the deterministic C01-C05 numeric smoke"
    )
    subparsers.add_parser(
        "accept-prelarge",
        help="run deterministic P4/P6 acceptance without formal assets or training",
    )

    intake = subparsers.add_parser(
        "intake-v02-policy-pool",
        help="read-only replay and import of the one frozen exact-90 handoff",
    )
    intake.add_argument("--handoff-dir", required=True)
    intake.add_argument("--trusted-experiment-root", required=True)
    intake.add_argument("--artifacts-root", required=True)
    intake.add_argument("--development-id", required=True)
    intake.add_argument("--resume", action="store_true")

    fit_representations = subparsers.add_parser(
        "fit-representation-controls",
        help="dry-run the 45-job plan or execute an externally bound adapter",
    )
    fit_representations.add_argument("--dry-run", action="store_true")
    _add_optional_development_publication(fit_representations)
    _add_real_stage_arguments(
        fit_representations,
        artifacts_root_already_added=True,
        resume_already_added=True,
    )

    signal_atlas = subparsers.add_parser(
        "build-signal-atlas",
        help="dry-run the 39/37 signal plan or execute an externally bound adapter",
    )
    signal_atlas.add_argument("--dry-run", action="store_true")
    _add_optional_development_publication(signal_atlas)
    _add_real_stage_arguments(
        signal_atlas,
        artifacts_root_already_added=True,
        resume_already_added=True,
    )

    baselines = subparsers.add_parser(
        "fit-baselines",
        help="dry-run the baseline registry or execute an externally bound adapter",
    )
    baselines.add_argument("--dry-run", action="store_true")
    _add_optional_development_publication(baselines)
    _add_real_stage_arguments(
        baselines,
        artifacts_root_already_added=True,
        resume_already_added=True,
    )

    for stage_id, help_text in (
        ("collect-source-receipts", "collect evaluator-owned source receipts"),
        ("build-market", "build the private/public anonymous source market"),
        ("build-canonical-banks", "build globally canonical source/query banks"),
        ("build-transition-views", "materialize the frozen transition views"),
        ("replay-legacy-attribution", "replay the fixed legacy prefix schedule"),
        ("build-source-specs", "build source-reduced KME specifications"),
        ("build-query-specs", "build query-empirical KME specifications"),
        ("run-public-rankings", "publish oracle-free anonymous rankings"),
    ):
        stage_parser = subparsers.add_parser(stage_id, help=help_text)
        _add_real_stage_arguments(stage_parser)

    freeze = subparsers.add_parser(
        "freeze-preexperiment",
        help="publish an engineering freeze; formal review authority is external",
    )
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--artifacts-root", required=True)
    freeze.add_argument("--development-id", required=True)
    freeze.add_argument("--resume", action="store_true")

    unlock = subparsers.add_parser(
        "unlock-oracle",
        help="validate the public-ranking barrier and emit a Paper-I handoff only",
    )
    unlock.add_argument("--public-ranking-barrier", required=True)

    statistics = subparsers.add_parser(
        "compute-statistics",
        help=(
            "compute frozen bootstrap/max-T/Holm results from an externally "
            "released typed oracle-input manifest"
        ),
    )
    statistics.add_argument("--plan", required=True)
    statistics.add_argument("--freeze-manifest", required=True)
    statistics.add_argument("--frozen-input", required=True)

    recompute = subparsers.add_parser(
        "recompute",
        help="validate an independent-root/process recompute attestation",
    )
    recompute.add_argument("--attestation", required=True)

    complete = subparsers.add_parser(
        "complete",
        help="verify the formal stage chain and external final-artifact preconditions",
    )
    complete.add_argument("--completion-manifest", required=True)
    complete.add_argument("--freeze-manifest", required=True)
    complete.add_argument("--artifacts-root", required=True)
    complete.add_argument("--resume", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stage_adapters: Mapping[str, StageExecutionAdapter] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_v03_foundation_config(args.config)
            payload = {
                "config_digest": config.config_digest,
                "stage": config.stage,
                # A locally declared digest is not a formal authority receipt.
                # Keep this fail-closed until a separate authority loader exists.
                "review_digest_status": (
                    "DECLARED_UNVERIFIED"
                    if config.review_decisions_digest is not None
                    else "NOT_DECLARED"
                ),
                "formal_scope_ready": False,
                "encoder_extension_gate": {
                    "enabled": config.encoder_extension_gate.enabled,
                    "migration_target": config.encoder_extension_gate.migration_target,
                    # Even an opted-in development gate is interface smoke only.
                    # It cannot activate v0.4 assets inside the v0.3 sidecar.
                    "optional_asset_requirements_active": False,
                    "completion_eligible": False,
                    "confirmatory_artifact_access": False,
                    "formal_authority": False,
                },
            }
            result = _result(args.command, "VALID", payload)
        elif args.command == "accept-numeric":
            report = run_minimal_compute_acceptance()
            result = _result(
                args.command,
                "ENGINEERING_PASS" if report.passed else "BLOCKED",
                report.to_dict(),
            )
        elif args.command == "accept-prelarge":
            report = run_prelarge_acceptance()
            result = _result(args.command, report.status, report.to_dict())
        elif args.command == "intake-v02-policy-pool":
            record = intake_v02_policy_pool(
                args.handoff_dir,
                trusted_experiment_root=args.trusted_experiment_root,
            )
            path, artifact_digest = _publish_intake(
                record,
                artifacts_root=args.artifacts_root,
                development_id=args.development_id,
                resume=bool(args.resume),
            )
            result = _result(
                args.command,
                "POOL_READY",
                {
                    "intake_record_digest": record.intake_record_digest,
                    "source_pool_digest": record.source_pool_digest,
                    "cell_count": len(record.cells),
                    "artifact_path": path,
                    "artifact_sha256": artifact_digest,
                },
            )
        elif args.command in PRODUCTION_STAGE_IDS and not getattr(
            args, "dry_run", False
        ):
            _require_real_stage_arguments(args)
            adapters = stage_adapters
            entrypoint = getattr(args, "adapter_entrypoint", None)
            if entrypoint is not None:
                if stage_adapters is not None:
                    raise V03CommandError(
                        "use either injected stage_adapters or --adapter-entrypoint, not both"
                    )
                loaded_adapter = _load_stage_adapter_entrypoint(entrypoint)
                adapters = {loaded_adapter.adapter_id: loaded_adapter}
            receipt, receipt_path, receipt_sha256, resumed = execute_stage_from_files(
                expected_stage_id=args.command,
                stage_manifest_path=args.stage_manifest,
                freeze_manifest_path=args.freeze_manifest,
                artifacts_root=args.artifacts_root,
                adapters=adapters,
                resume=bool(args.resume),
            )
            result = _result(
                args.command,
                "STAGE_RESUMED" if resumed else "STAGE_COMPLETE",
                {
                    "stage_execution_receipt": receipt.to_dict(),
                    "stage_execution_receipt_digest": receipt.receipt_digest,
                    "artifact_path": str(receipt_path),
                    "artifact_sha256": receipt_sha256,
                    "adapter_executed": not resumed,
                    "oracle_accessed": False,
                },
            )
        elif args.command in {
            "fit-representation-controls",
            "build-signal-atlas",
        }:
            if not args.dry_run:  # required=True is defensive against programmatic Namespace use
                raise V03CommandError("this pre-large command requires --dry-run")
            plan = build_signal_matrix_plan()
            jobs = build_optimization_fit_jobs(plan)
            if args.command == "fit-representation-controls":
                payload: dict[str, Any] = {
                    "execution": "NOT_STARTED",
                    "large_experiment_executed": False,
                    "signal_matrix_plan_digest": plan.plan_digest,
                    "fit_job_count": len(jobs),
                    "r5_fit_job_count": sum(
                        job.representation_id == "R5_VIEW_SPECIFIC_CORRO_REFIT"
                        for job in jobs
                    ),
                    "r5l_fit_job_count": sum(
                        job.representation_id == "R5L_SUPERVISED_LINEAR"
                        for job in jobs
                    ),
                    "fit_job_digests": [job.job_digest for job in jobs],
                }
                domain = "representation_controls"
                filename = "optimization_fit_plan.json"
            else:
                payload = {
                    "execution": "NOT_STARTED",
                    "large_experiment_executed": False,
                    "plan": plan.to_dict(),
                    "fit_job_count": len(jobs),
                    "fit_job_digests": [job.job_digest for job in jobs],
                }
                domain = "signal_atlas"
                filename = "signal_matrix_plan.json"
            publication = _publication_args(args)
            if publication is not None:
                path, artifact_digest = _publish_development_json(
                    payload,
                    domain=domain,
                    filename=filename,
                    artifacts_root=publication[0],
                    development_id=publication[1],
                    resume=bool(args.resume),
                )
                payload.update(
                    {"artifact_path": path, "artifact_sha256": artifact_digest}
                )
            result = _result(args.command, "DRY_RUN_READY", payload)
        elif args.command == "fit-baselines":
            if not args.dry_run:
                raise V03CommandError("this pre-large command requires --dry-run")
            payload = {
                "execution": "NOT_STARTED",
                "large_experiment_executed": False,
                "required_method_ids": list(REQUIRED_BASELINE_METHOD_IDS),
                "optional_method_states": dict(OPTIONAL_BASELINE_STATES),
                "registry_digest": sha256_json(
                    {
                        "required_method_ids": list(REQUIRED_BASELINE_METHOD_IDS),
                        "optional_method_states": dict(OPTIONAL_BASELINE_STATES),
                    }
                ),
                "confirmatory_oracle_access": False,
            }
            publication = _publication_args(args)
            if publication is not None:
                path, artifact_digest = _publish_development_json(
                    payload,
                    domain="baseline_tables",
                    filename="baseline_fit_plan.json",
                    artifacts_root=publication[0],
                    development_id=publication[1],
                    resume=bool(args.resume),
                )
                payload.update(
                    {"artifact_path": path, "artifact_sha256": artifact_digest}
                )
            result = _result(args.command, "DRY_RUN_READY", payload)
        elif args.command == "freeze-preexperiment":
            manifest = PreExperimentFreezeManifest.from_dict(
                _strict_json(args.manifest, where="pre-experiment manifest")
            )
            if manifest.formal_run_authorized or manifest.review_authority_verified:
                raise V03CommandError(
                    "v0.3 CLI cannot accept or mint formal review authority; "
                    "publish the unverified engineering freeze and use the external handoff"
                )
            path, artifact_digest = _publish_development_json(
                manifest.to_dict(),
                domain="scope",
                filename="pre_experiment_freeze.json",
                artifacts_root=args.artifacts_root,
                development_id=args.development_id,
                resume=bool(args.resume),
            )
            result = _result(
                args.command,
                "ENGINEERING_READY_REVIEW_UNVERIFIED",
                {
                    "freeze_manifest_digest": manifest.freeze_manifest_digest,
                    "formal_run_authorized": manifest.formal_run_authorized,
                    "artifact_path": path,
                    "artifact_sha256": artifact_digest,
                },
            )
        elif args.command == "unlock-oracle":
            barrier = PublicRankingBarrier.from_dict(
                _strict_json(
                    args.public_ranking_barrier,
                    where="public ranking barrier",
                )
            )
            handoff = OracleUnlockHandoff(
                run_id=barrier.run_id,
                freeze_manifest_digest=barrier.freeze_manifest_digest,
                public_ranking_barrier_digest=barrier.barrier_digest,
            )
            result = _result(
                args.command,
                "HANDOFF_REQUIRED",
                {
                    "barrier_digest": barrier.barrier_digest,
                    "handoff": handoff.to_dict(),
                    "handoff_digest": handoff.handoff_digest,
                    "oracle_unlocked_by_v03": False,
                },
            )
        elif args.command == "compute-statistics":
            plan = FormalStatisticsPlan.from_dict(
                _strict_json(args.plan, where="formal statistics plan")
            )
            manifest = PreExperimentFreezeManifest.from_dict(
                _strict_json(
                    args.freeze_manifest,
                    where="authorized pre-experiment manifest",
                )
            )
            frozen_input = FrozenStatisticsInput.from_dict(
                _strict_json(
                    args.frozen_input,
                    where="externally released statistics input",
                )
            )
            statistics_result = compute_formal_statistics(
                plan=plan,
                freeze_manifest=manifest,
                frozen_input=frozen_input,
            )
            result = _result(
                args.command,
                "FORMAL_STATISTICS_COMPUTED",
                {
                    "statistics_result": statistics_result.to_dict(),
                    "statistics_result_digest": statistics_result.result_digest,
                    "oracle_read_by_cli": False,
                    "oracle_release_receipt_digest": (
                        frozen_input.oracle_release_receipt_digest
                    ),
                },
            )
        elif args.command == "recompute":
            attestation = IndependentRecomputeAttestation.from_dict(
                _strict_json(args.attestation, where="recompute attestation")
            )
            result = _result(
                args.command,
                "INDEPENDENT_RECOMPUTE_MATCH",
                {
                    "attestation_digest": attestation.attestation_digest,
                    "distinct_artifact_root": True,
                    "fresh_process_nonce": True,
                },
            )
        elif args.command == "complete":
            completion, path, artifact_sha256, resumed = (
                verify_pipeline_completion_from_files(
                    completion_manifest_path=args.completion_manifest,
                    freeze_manifest_path=args.freeze_manifest,
                    artifacts_root=args.artifacts_root,
                    resume=bool(args.resume),
                )
            )
            result = _result(
                args.command,
                "COMPLETE_RESUMED" if resumed else completion.status,
                {
                    "completion_receipt": completion.to_dict(),
                    "completion_receipt_digest": completion.receipt_digest,
                    "artifact_path": str(path),
                    "artifact_sha256": artifact_sha256,
                    "oracle_read_by_v03_driver": False,
                },
            )
        else:  # pragma: no cover - argparse owns the command registry
            raise AssertionError(args.command)
    except Exception as error:
        failure = _result(
            getattr(args, "command", "unknown"),
            "BLOCKED",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        print(canonical_json(failure), file=sys.stderr)
        return 1
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["CLI_SCHEMA", "CLI_VERSION", "V03CommandError", "main"]
