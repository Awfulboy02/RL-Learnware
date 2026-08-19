"""Deterministic v0.1 decision and summary rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class V01Decision:
    code: str
    formal_complete: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "formal_complete": self.formal_complete,
            "explanation": self.explanation,
        }


def _passed(report: Mapping[str, Any], name: str) -> bool:
    if "passed" not in report or not isinstance(report["passed"], bool):
        raise ValueError(f"{name} report lacks a strict boolean passed field")
    return bool(report["passed"])


def decide_v01(
    *,
    gate_0: Mapping[str, Any],
    gate_a: Mapping[str, Any],
    gate_b: Mapping[str, Any],
    gate_d: Mapping[str, Any],
    recompute_audit: Mapping[str, Any],
    no_go_compute: bool = False,
) -> V01Decision:
    """Apply the pre-registered decision table without outcome-dependent edits."""

    if not _passed(gate_0, "Gate 0") or not _passed(gate_d, "Gate D"):
        return V01Decision(
            "BLOCKED_ENGINEERING",
            False,
            "Gate 0 identity/runtime or Gate D information isolation failed.",
        )
    if no_go_compute:
        return V01Decision(
            "NO_GO_COMPUTE",
            False,
            "The exact pre-registered TaskSpec computation was not approved or completed.",
        )
    if not _passed(recompute_audit, "recompute audit"):
        return V01Decision(
            "BLOCKED_ENGINEERING",
            False,
            "The stratified independent recomputation audit failed.",
        )
    if not _passed(gate_a, "Gate A"):
        return V01Decision(
            "NO_GO_CURRENT_POOL_SHIFT",
            True,
            "The approved policy pool and damping grid lack the required transfer effect evidence.",
        )
    if not _passed(gate_b, "Gate B"):
        return V01Decision(
            "GO_PROBLEM_NO_GO_TASKSPEC",
            True,
            "A policy-transfer problem is present, but frozen v0 TaskSpec is not sufficiently shift-sensitive.",
        )
    return V01Decision(
        "GO_V02_TRANSFERSPEC",
        True,
        "Both the pool-level transfer problem and TaskSpec shift sensitivity passed the frozen criteria.",
    )


def render_summary(
    *,
    experiment_id: str,
    measurement_run_id: str,
    oracle_protocol_id: str,
    decision: V01Decision,
    gate_0: Mapping[str, Any],
    gate_a: Mapping[str, Any],
    gate_b: Mapping[str, Any],
    gate_c: Mapping[str, Any],
    gate_d: Mapping[str, Any],
    recompute_audit: Mapping[str, Any],
) -> str:
    """Render a compact, auditable report; Gate C is explicitly diagnostic."""

    def status(value: Mapping[str, Any], *, diagnostic: bool = False) -> str:
        if diagnostic:
            return "diagnostic (no pass/fail)"
        return "PASS" if bool(value.get("passed", False)) else "FAIL"

    return "\n".join(
        [
            "# Policy Learnware v0.1 Dynamics-Shift Report",
            "",
            f"- Experiment: `{experiment_id}`",
            f"- Measurement run: `{measurement_run_id}`",
            f"- Oracle protocol: `{oracle_protocol_id}`",
            f"- Decision: **{decision.code}**",
            f"- Formal run complete: `{str(decision.formal_complete).lower()}`",
            "",
            "## Frozen scope",
            "",
            "This is a diagnostic study of an episode-static, one-dimensional correlated DoF-damping intervention. Observation/action schema, reward, reset, termination, horizon, action repeat, frozen policies, and v0 TaskSpec representation remain fixed. It is not an OOD, physical sim-to-real, TransferSpec, or robust-selector result.",
            "",
            "## Gates",
            "",
            f"- Gate 0 (engineering identity/runtime): {status(gate_0)}",
            f"- Gate A (policy transfer effect): {status(gate_a)}",
            f"- Gate B (TaskSpec dynamics sensitivity): {status(gate_b)}",
            f"- Gate C (TaskSpec/effect association): {status(gate_c, diagnostic=True)}",
            f"- Gate D (information isolation): {status(gate_d)}",
            f"- Stratified recomputation audit: {status(recompute_audit)}",
            "",
            "## Interpretation",
            "",
            decision.explanation,
            "",
            "Scientific Gate A/B failures are retained as valid No-Go outcomes. Gate 0/D or audit failure blocks formal completion and must be repaired in a new immutable run.",
            "",
        ]
    )


__all__ = ["V01Decision", "decide_v01", "render_summary"]
