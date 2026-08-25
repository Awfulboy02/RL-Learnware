#!/usr/bin/env python3
"""Expand reviewed anchors and a reviewed trainer protocol into strict jobs.

There are intentionally no task, axis, factor, algorithm, seed, budget, or
competence defaults here.  Every such literal must arrive from a frozen anchor
manifest, protocol file, or explicit ``--seeds`` argument.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Package import for tests/``python -m``; fallback for direct script use.
    from .anchor_binding import AnchorManifest
    from .formal_plan import validate_formal_training_projection
    from .provenance import (
        ContractError,
        EXECUTION_PURPOSES,
        TRAINING_JOB_SCHEMA,
        atomic_write_json,
        finalize_training_job,
        finalize_training_plan,
        load_and_bind_formal_freeze,
        load_strict_json,
        FORMAL_EXECUTION_PURPOSE,
        require_digest,
        require_execution_purpose,
        sha256_json,
        validate_training_protocol,
    )
except ImportError:  # pragma: no cover - exercised by executable entry points
    from anchor_binding import AnchorManifest
    from formal_plan import validate_formal_training_projection
    from provenance import (
        ContractError,
        EXECUTION_PURPOSES,
        TRAINING_JOB_SCHEMA,
        atomic_write_json,
        finalize_training_job,
        finalize_training_plan,
        load_and_bind_formal_freeze,
        load_strict_json,
        FORMAL_EXECUTION_PURPOSE,
        require_digest,
        require_execution_purpose,
        sha256_json,
        validate_training_protocol,
    )


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must be unique nonnegative integers")
    if seeds != tuple(sorted(seeds)):
        raise argparse.ArgumentTypeError("--seeds must be explicitly sorted")
    return seeds


def build_training_plan(
    *,
    anchor_paths: Sequence[Path],
    protocol: Mapping[str, Any],
    seeds: Sequence[int],
    config_digest: str,
    execution_purpose: str,
    formal_config_path: Path | None = None,
) -> dict[str, Any]:
    validated_protocol = validate_training_protocol(protocol)
    config_id = require_digest(config_digest, "config_digest")
    purpose = require_execution_purpose(execution_purpose)
    if purpose == FORMAL_EXECUTION_PURPOSE:
        if formal_config_path is None:
            raise ContractError(
                "v02_freeze_ready plan generation requires --formal-config"
            )
        formal_freeze = load_and_bind_formal_freeze(formal_config_path)
        if formal_freeze["config_digest"] != config_id:
            raise ContractError(
                "--config-digest differs from the verified formal config bytes"
            )
        formal_freeze_digest: str | None = formal_freeze["binding_digest"]
    else:
        if formal_config_path is not None:
            raise ContractError("non-formal plan cannot consume --formal-config")
        formal_freeze = None
        formal_freeze_digest = None
    if not anchor_paths:
        raise ContractError("at least one reviewed anchor manifest is required")
    if not seeds or tuple(seeds) != tuple(sorted(set(seeds))) or any(seed < 0 for seed in seeds):
        raise ContractError("seeds must be a sorted unique nonnegative sequence")
    anchors: list[tuple[Path, AnchorManifest]] = []
    seen: set[str] = set()
    for raw_path in anchor_paths:
        path = Path(raw_path).resolve()
        anchor = AnchorManifest.from_path(path)
        if anchor.anchor_id in seen:
            raise ContractError(f"duplicate source anchor in plan: {anchor.anchor_id}")
        seen.add(anchor.anchor_id)
        anchors.append((path, anchor))
    jobs = []
    for path, anchor in sorted(anchors, key=lambda item: item[1].anchor_id):
        for seed in seeds:
            semantic = {
                "config_digest": config_id,
                "execution_purpose": purpose,
                "anchor_manifest_digest": anchor.manifest_digest,
                "training_protocol_digest": validated_protocol["protocol_digest"],
                "seed": seed,
            }
            job_id = "v02-" + sha256_json(semantic)[:24]
            jobs.append(
                finalize_training_job(
                    {
                        "schema": TRAINING_JOB_SCHEMA,
                        "job_id": job_id,
                        "config_digest": config_id,
                        "execution_purpose": purpose,
                        "formal_protocol_freeze_digest": formal_freeze_digest,
                        "anchor_manifest_path": str(path),
                        "anchor_manifest_digest": anchor.manifest_digest,
                        "training_protocol": validated_protocol,
                        "training_protocol_digest": validated_protocol["protocol_digest"],
                        "seed": seed,
                    }
                )
            )
    plan = finalize_training_plan(
        jobs, formal_protocol_freeze=formal_freeze
    )
    if formal_freeze is not None:
        validate_formal_training_projection(plan, formal_freeze)
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-manifest",
        action="append",
        type=Path,
        required=True,
        help="Reviewed immutable anchor JSON; repeat once per unique source anchor",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Reviewed strict training-protocol JSON (no defaults are supplied)",
    )
    parser.add_argument("--seeds", type=parse_seeds, required=True)
    parser.add_argument("--config-digest", required=True)
    parser.add_argument(
        "--formal-config",
        type=Path,
        help=(
            "canonical v02_freeze_ready YAML; required only for formal plans and "
            "revalidated against its canonical freeze manifest"
        ),
    )
    parser.add_argument(
        "--execution-purpose",
        choices=tuple(sorted(EXECUTION_PURPOSES)),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_strict_json(args.protocol)
    plan = build_training_plan(
        anchor_paths=args.anchor_manifest,
        protocol=protocol,
        seeds=args.seeds,
        config_digest=args.config_digest,
        execution_purpose=args.execution_purpose,
        formal_config_path=args.formal_config,
    )
    atomic_write_json(args.output, plan, overwrite=False)
    print(
        f"wrote {len(plan['jobs'])} immutable jobs; plan_digest={plan['plan_digest']} "
        f"output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
