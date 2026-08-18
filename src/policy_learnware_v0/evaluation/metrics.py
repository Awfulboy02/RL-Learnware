"""Aggregate selected-policy deployment outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from .deployment import DeploymentResult


@dataclass(frozen=True)
class DeploymentMetrics:
    query_count: int
    deployable_count: int
    deployability_rate: float
    conditional_mean_return: float | None
    failure_counts: tuple[tuple[str, int], ...]


def summarize_deployments(results: Iterable[DeploymentResult]) -> DeploymentMetrics:
    values = tuple(results)
    if not values:
        raise ValueError("at least one deployment result is required")
    deployed = [result for result in values if result.deployable]
    returns = [result.mean_return for result in deployed if result.mean_return is not None]
    counts: dict[str, int] = {}
    for result in values:
        if result.deployment_failure is not None:
            counts[result.deployment_failure] = counts.get(result.deployment_failure, 0) + 1
    return DeploymentMetrics(
        query_count=len(values),
        deployable_count=len(deployed),
        deployability_rate=len(deployed) / len(values),
        conditional_mean_return=fmean(returns) if returns else None,
        failure_counts=tuple(sorted(counts.items())),
    )
