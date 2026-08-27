"""Minimal command surface for the v0.3 numerical core.

The CLI deliberately exposes only two dependency-light operations:

* ``accept-numeric`` exercises the RKME/query/ranking numerical loop.
* ``signal-plan`` reports the current signal cells and optional fit jobs.

Training and production runners remain explicit server entry points.  This
module does not import artifact governance, formal admission, orchestration,
authority, provenance, or completion machinery.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Mapping, Sequence


CLI_SCHEMA = "policy-learnware.v03-cli-result.v0"
CLI_VERSION = "0.3.0"


def _result(command: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CLI_SCHEMA,
        "command": command,
        "status": status,
        "payload": dict(payload),
    }


def _emit(value: Mapping[str, Any], *, stream: Any | None = None) -> None:
    stream = sys.stdout if stream is None else stream
    print(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        file=stream,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-learnware-v03",
        description="Policy Learnware v0.3 RKME and signal-plan utilities",
    )
    parser.add_argument("--version", action="version", version=CLI_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "accept-numeric",
        help="run the deterministic RKME/query/ranking numerical smoke",
    )

    signal_plan = subparsers.add_parser(
        "signal-plan",
        help="show the signal matrix and optimization-fit schedule without training",
    )
    signal_plan.add_argument(
        "--include-cells",
        action="store_true",
        help="include the complete cell definitions in the JSON result",
    )
    signal_plan.add_argument(
        "--include-fit-jobs",
        action="store_true",
        help="include the complete optimization-fit job definitions",
    )
    return parser


def _run_numeric_smoke() -> tuple[str, Mapping[str, Any]]:
    # Keep optional numerical dependencies out of ``--help`` and ``--version``.
    from .acceptance import run_minimal_compute_acceptance

    report = run_minimal_compute_acceptance()
    return ("PASS" if report.passed else "FAIL"), report.to_dict()


def _show_signal_plan(args: argparse.Namespace) -> tuple[str, Mapping[str, Any]]:
    # Signal-plan dependencies are loaded only when this command is requested.
    from .signal_matrix import build_optimization_fit_jobs, build_signal_matrix_plan

    plan = build_signal_matrix_plan()
    jobs = build_optimization_fit_jobs(plan)
    payload: dict[str, Any] = {
        "execution": "NOT_STARTED",
        "large_experiment_executed": False,
        "plan_digest": plan.plan_digest,
        "logical_cell_count": plan.logical_cell_count,
        "numeric_cell_count": plan.numeric_cell_count,
        "structural_na_count": plan.structural_na_count,
        "optimization_fit_job_count": len(jobs),
        "r5_fit_job_count": sum(
            job.representation_id == "R5_VIEW_SPECIFIC_CORRO_REFIT" for job in jobs
        ),
        "r5l_fit_job_count": sum(
            job.representation_id == "R5L_SUPERVISED_LINEAR" for job in jobs
        ),
    }
    if args.include_cells:
        payload["plan"] = plan.to_dict()
    if args.include_fit_jobs:
        payload["fit_jobs"] = [job.to_dict() for job in jobs]
    return "READY", payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "accept-numeric":
            status, payload = _run_numeric_smoke()
        elif args.command == "signal-plan":
            status, payload = _show_signal_plan(args)
        else:  # pragma: no cover - argparse owns the command registry
            raise AssertionError(args.command)
    except Exception as error:
        _emit(
            _result(
                getattr(args, "command", "unknown"),
                "FAILED",
                {"error_type": type(error).__name__, "error": str(error)},
            ),
            stream=sys.stderr,
        )
        return 1

    _emit(_result(args.command, status, payload))
    return 0 if status != "FAIL" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
