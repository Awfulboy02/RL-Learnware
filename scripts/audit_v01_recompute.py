#!/usr/bin/env python3
"""Standalone entry point for the v0.1 stratified recomputation audit.

The implementation lives behind the same capability-scoped CLI used by the
installed package, so invoking this script cannot acquire broader roots than
``policy-learnware-v01 audit-recompute``.
"""

from __future__ import annotations

from policy_learnware_v0.v01.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["audit-recompute", *__import__("sys").argv[1:]]))
