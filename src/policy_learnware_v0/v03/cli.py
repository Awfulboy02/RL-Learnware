"""Fail-closed command surface for the v0.3 foundation sidecar.

The CLI intentionally separates engineering acceptance from formal scientific
gates.  In particular, the production pool-intake command has no option for
overriding the reviewed v0.2 trust anchor, and no command can mint source
evaluation receipts from caller-supplied summary statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from ..hashing import canonical_json
from .acceptance import run_minimal_compute_acceptance
from .artifacts import V03ArtifactLayout
from .config import load_v03_foundation_config
from .pool_intake import V03PoolIntakeRecord, intake_v02_policy_pool


CLI_SCHEMA = "policy-learnware.v03-cli-result.v0"
CLI_VERSION = "0.3.0"


class V03CommandError(RuntimeError):
    """A command input or immutable output violates the v0.3 boundary."""


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

    intake = subparsers.add_parser(
        "intake-v02-policy-pool",
        help="read-only replay and import of the one frozen exact-90 handoff",
    )
    intake.add_argument("--handoff-dir", required=True)
    intake.add_argument("--trusted-experiment-root", required=True)
    intake.add_argument("--artifacts-root", required=True)
    intake.add_argument("--development-id", required=True)
    intake.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
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
