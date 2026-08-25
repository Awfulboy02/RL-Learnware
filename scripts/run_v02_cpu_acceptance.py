#!/usr/bin/env python3
"""Run the four deterministic, explicitly non-formal v0.2 CPU scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.io import atomic_write_json
from policy_learnware_v0.v02.acceptance import (
    ACCEPTANCE_SCENARIOS,
    AcceptanceReport,
    run_cpu_acceptance_fixture,
)


SCENARIO_ORDER = (
    "scientific_pass",
    "no_go_market",
    "no_go_corro",
    "engineering_blocked",
)


def run_suite() -> dict[str, Any]:
    reports: dict[str, AcceptanceReport] = {}
    for scenario in SCENARIO_ORDER:
        if scenario not in ACCEPTANCE_SCENARIOS:
            raise RuntimeError(f"acceptance scenario registry drift: {scenario}")
        ordinary = run_cpu_acceptance_fixture(scenario)  # type: ignore[arg-type]
        shuffled = run_cpu_acceptance_fixture(  # type: ignore[arg-type]
            scenario, shuffle_inputs=True
        )
        if ordinary.report.digest != shuffled.report.digest:
            raise RuntimeError(
                f"acceptance scenario {scenario!r} is sensitive to input ordering"
            )
        # Strict archival reload proves every status/count/digest invariant is
        # derived rather than trusted from the serialized pass/status fields.
        reports[scenario] = AcceptanceReport.from_dict(ordinary.report.to_dict())
    payload: dict[str, Any] = {
        "schema": "policy-learnware.v02-cpu-acceptance-suite.v0",
        "scope": "two-task-development-fixture-nonformal",
        "formal_completion_claimed": False,
        "scenario_count": len(reports),
        "all_scenarios_replayed_under_input_permutation": True,
        "reports": {
            scenario: reports[scenario].to_dict() for scenario in SCENARIO_ORDER
        },
    }
    payload["suite_digest"] = sha256_json(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="optional immutable JSON output; stdout is always emitted",
    )
    args = parser.parse_args()
    payload = run_suite()
    if args.output is not None:
        atomic_write_json(args.output.resolve(), payload, overwrite=False)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
