#!/usr/bin/env python3
"""Read-only verification and canonical replay for the frozen v0.2 pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from policy_learnware_v0.hashing import sha256_file
from policy_learnware_v0.v02.artifacts import (
    RelocationResolver,
    capability_status,
    verify_handoff_trust_anchors,
)
from server.repro_fpo_ppo_v02.replay import (
    replay_relocated_policy_pool_acceptance,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify v0.2 external assets without mutating frozen evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify-assets", "capabilities", "replay"):
        command = commands.add_parser(name)
        command.add_argument(
            "--artifacts-root",
            type=str,
            help=(
                "External root; overrides RL_LEARNWARE_ARTIFACTS_ROOT. "
                "The only manifest read is <root>/relocation_manifest.json."
            ),
        )
        if name == "replay":
            command.add_argument(
                "--acceptance-path",
                type=Path,
                help=(
                    "Optional explicit frozen receipt. It must equal the canonical "
                    "exact90 acceptance path after root resolution."
                ),
            )
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        receipt = replay_relocated_policy_pool_acceptance(
            artifacts_root=args.artifacts_root,
            acceptance_path=args.acceptance_path,
        )
        result = {
            "schema": "policy-learnware.v02-canonical-replay-result.v1",
            "decision": receipt["decision"],
            "job_count": receipt["job_count"],
            "anchor_count": receipt["anchor_count"],
            "seeds": receipt["seeds"],
            "direct_terminal_record_count": receipt[
                "direct_terminal_record_count"
            ],
            "compiled_parity_fallback_promotion_count": receipt[
                "compiled_parity_fallback_promotion_count"
            ],
            "pool_digest": receipt["pool_digest"],
            "frozen_report_digest": receipt["report_digest"],
        }
    else:
        resolver = RelocationResolver.load(artifacts_root=args.artifacts_root)
        capabilities = capability_status(resolver.layout, resolver.manifest)
        if args.command == "capabilities":
            result = capabilities
        else:
            result = {
                "schema": "policy-learnware.v02-asset-verification.v1",
                "relocation_manifest_sha256": sha256_file(
                    resolver.layout.relocation_manifest
                ),
                "handoff": verify_handoff_trust_anchors(resolver.layout),
                "capabilities": capabilities,
            }
    _print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
