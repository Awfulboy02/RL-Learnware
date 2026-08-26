#!/usr/bin/env python3
"""Prepare/verify the append-only v0.2 exact-90 policy-pool overlay."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from server.repro_fpo_ppo_v02.pool_acceptance import (
    accept_policy_pool,
    build_compiled_parity_promotion_manifest,
)
from server.repro_fpo_ppo_v02.provenance import (
    atomic_write_json,
    load_strict_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare-promotions", "accept"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--server-plan", type=Path, required=True)
        sub.add_argument("--runs-root", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if command == "accept":
            sub.add_argument("--promotions", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_strict_json(args.server_plan.resolve())
    if args.command == "prepare-promotions":
        result = build_compiled_parity_promotion_manifest(
            server_plan=plan,
            runs_root=args.runs_root.resolve(),
        )
    else:
        result = accept_policy_pool(
            server_plan=plan,
            runs_root=args.runs_root.resolve(),
            promotion_manifest=load_strict_json(args.promotions.resolve()),
        )
    atomic_write_json(args.output.resolve(), result, overwrite=False)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
