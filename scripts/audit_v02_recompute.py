#!/usr/bin/env python3
"""Capability-scoped entry point for v0.2 independent recomputation."""

from __future__ import annotations

import sys

from policy_learnware_v0.v02.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["audit-recompute", *sys.argv[1:]]))
