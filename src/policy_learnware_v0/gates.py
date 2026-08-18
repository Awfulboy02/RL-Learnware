"""Auditable, fail-closed decisions for preregistered v0 experiment gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class GateCheck:
    """One numeric lower-bound check in a gate decision."""

    metric: str
    observed: float
    minimum: float
    passed: bool

    @classmethod
    def at_least(cls, metric: str, observed: float, minimum: float) -> "GateCheck":
        observed = float(observed)
        minimum = float(minimum)
        if not math.isfinite(observed) or not math.isfinite(minimum):
            raise ValueError(f"gate check {metric!r} must be finite")
        return cls(
            metric=metric,
            observed=observed,
            minimum=minimum,
            passed=observed >= minimum,
        )


@dataclass(frozen=True)
class GateDecision:
    """A named conjunction of checks, suitable for a JSON audit artifact."""

    name: str
    checks: tuple[GateCheck, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name cannot be empty")
        if not self.checks:
            raise ValueError("a gate needs at least one check")
        metrics = tuple(check.metric for check in self.checks)
        if any(not metric for metric in metrics) or len(metrics) != len(set(metrics)):
            raise ValueError("gate check metric names must be non-empty and unique")

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy": "all_checks_must_pass",
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def validate_gate_record(value: Any, *, expected_name: str) -> GateDecision:
    """Parse an artifact gate and reject inconsistent/tampered pass flags."""

    if not isinstance(value, Mapping):
        raise ValueError("gate artifact must be an object")
    if value.get("name") != expected_name or value.get("policy") != "all_checks_must_pass":
        raise ValueError(f"unexpected gate identity/policy for {expected_name}")
    raw_checks = value.get("checks")
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
        raise ValueError("gate checks must be a sequence")
    checks: list[GateCheck] = []
    for raw in raw_checks:
        if not isinstance(raw, Mapping):
            raise ValueError("gate check must be an object")
        if set(raw) != {"metric", "observed", "minimum", "passed"}:
            raise ValueError("gate check has missing or unknown fields")
        check = GateCheck.at_least(
            str(raw["metric"]), float(raw["observed"]), float(raw["minimum"])
        )
        if raw["passed"] is not check.passed:
            raise ValueError(f"inconsistent pass flag for {check.metric}")
        checks.append(check)
    decision = GateDecision(expected_name, tuple(checks))
    if value.get("passed") is not decision.passed:
        raise ValueError(f"inconsistent aggregate pass flag for {expected_name}")
    return decision


def unreduced_gate(
    *,
    minimum_between_mmd: float,
    maximum_within_mmd: float,
    split_retrieval_accuracy: float,
    minimum_between_within_ratio: float,
    minimum_absolute_margin: float,
    minimum_split_retrieval_accuracy: float,
) -> GateDecision:
    """Evaluate source/query split separability without changing the KME math."""

    if maximum_within_mmd < 0.0:
        raise ValueError("maximum within-task MMD cannot be negative")
    ratio = (
        math.inf
        if maximum_within_mmd == 0.0 and minimum_between_mmd > 0.0
        else minimum_between_mmd / max(maximum_within_mmd, float.fromhex("0x1p-1022"))
    )
    # JSON artifacts disallow infinity. A zero-within case has the strongest
    # possible separation, so record the largest finite float for auditability.
    ratio = min(ratio, float.fromhex("0x1.fffffffffffffp+1023"))
    return GateDecision(
        "unreduced_separability",
        (
            GateCheck.at_least(
                "between_to_within_mmd_ratio",
                ratio,
                minimum_between_within_ratio,
            ),
            GateCheck.at_least(
                "between_minus_within_mmd",
                minimum_between_mmd - maximum_within_mmd,
                minimum_absolute_margin,
            ),
            GateCheck.at_least(
                "split_retrieval_accuracy",
                split_retrieval_accuracy,
                minimum_split_retrieval_accuracy,
            ),
        ),
    )


def ranking_gate(*, top1_agreement: float, minimum_top1_agreement: float) -> GateDecision:
    return GateDecision(
        "reduced_unreduced_ranking",
        (
            GateCheck.at_least(
                "top1_agreement", top1_agreement, minimum_top1_agreement
            ),
        ),
    )


def retrieval_gate(
    *, max_prefix_accuracy: float, minimum_max_prefix_accuracy: float
) -> GateDecision:
    return GateDecision(
        "exact_recurrent_retrieval",
        (
            GateCheck.at_least(
                "max_prefix_accuracy",
                max_prefix_accuracy,
                minimum_max_prefix_accuracy,
            ),
        ),
    )


def deployment_gate(
    *,
    correct_retrieval_count: int,
    correct_retrieval_deployability_rate: float | None,
    minimum_correct_retrieval_deployability_rate: float,
) -> GateDecision:
    # With no correct retrieval, the conditional metric is undefined and the
    # deployment claim is unauditable. Record 0 and fail closed.
    observed_rate = (
        0.0
        if correct_retrieval_deployability_rate is None
        else correct_retrieval_deployability_rate
    )
    return GateDecision(
        "selected_policy_deployment",
        (
            GateCheck.at_least(
                "correct_retrieval_count", float(correct_retrieval_count), 1.0
            ),
            GateCheck.at_least(
                "correct_retrieval_deployability_rate",
                observed_rate,
                minimum_correct_retrieval_deployability_rate,
            ),
        ),
    )


def deterministic_ranking(distances: Mapping[str, float]) -> tuple[str, ...]:
    """Rank candidates by finite distance and lexical task tie-break."""

    if not distances:
        raise ValueError("ranking requires at least one candidate")
    parsed = {str(key): float(value) for key, value in distances.items()}
    if any(not key for key in parsed) or len(parsed) != len(distances):
        raise ValueError("ranking candidate ids must be unique and non-empty")
    if any(not math.isfinite(value) or value < 0.0 for value in parsed.values()):
        raise ValueError("ranking distances must be finite and non-negative")
    return tuple(sorted(parsed, key=lambda key: (parsed[key], key)))


def pairwise_order_agreement(
    left: Sequence[str], right: Sequence[str]
) -> float:
    """Fraction of candidate pairs ordered identically by two total rankings."""

    left = tuple(left)
    right = tuple(right)
    if len(left) < 2 or set(left) != set(right) or len(set(left)) != len(left):
        raise ValueError("rankings must be permutations of at least two candidates")
    left_position = {value: index for index, value in enumerate(left)}
    right_position = {value: index for index, value in enumerate(right)}
    agreed = 0
    total = 0
    for first_index, first in enumerate(left):
        for second in left[first_index + 1 :]:
            total += 1
            if (left_position[first] < left_position[second]) == (
                right_position[first] < right_position[second]
            ):
                agreed += 1
    return agreed / total


def nonoverlapping_half_ranges(
    episode_count: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return first-half source and second-half query ranges with no overlap."""

    if isinstance(episode_count, bool) or not isinstance(episode_count, int):
        raise ValueError("episode_count must be an integer")
    if episode_count < 2:
        raise ValueError("at least two episodes are required for a strict split")
    midpoint = episode_count // 2
    return (0, midpoint), (midpoint, episode_count)


__all__ = [
    "GateCheck",
    "GateDecision",
    "deployment_gate",
    "deterministic_ranking",
    "pairwise_order_agreement",
    "nonoverlapping_half_ranges",
    "ranking_gate",
    "retrieval_gate",
    "unreduced_gate",
    "validate_gate_record",
]
