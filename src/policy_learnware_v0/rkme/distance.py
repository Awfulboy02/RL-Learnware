"""Distances between an empirical target KME and source reduced RKMEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .empirical import EmpiricalKME, blockwise_weighted_kernel_sum
from .gaussian import GaussianKernel
from .reducer import ReducedRKME


@dataclass(frozen=True)
class DistanceResult:
    distance: float
    squared_distance: float
    raw_squared_distance: float
    clamped: bool


def empirical_to_reduced_distance(
    target: EmpiricalKME,
    source: ReducedRKME,
    *,
    block_size: int = 2048,
    negative_tolerance: float = 1.0e-8,
) -> DistanceResult:
    """Compute the full source-reduced/target-empirical RKHS distance."""

    if not np.isclose(target.bandwidth, source.bandwidth, rtol=0.0, atol=0.0):
        raise ValueError("target and source use different Gaussian bandwidths")
    if target.protocol_id and source.protocol_id and target.protocol_id != source.protocol_id:
        raise ValueError("target and source use different protocols")
    kernel = GaussianKernel(target.bandwidth)
    cross = blockwise_weighted_kernel_sum(
        target.points,
        target.weights,
        source.supports,
        source.beta,
        kernel,
        block_size=block_size,
    )
    raw_squared = float(target.norm2 - 2.0 * cross + source.rkme_norm2)
    scale = max(1.0, abs(target.norm2), abs(source.rkme_norm2), abs(2.0 * cross))
    if raw_squared < -float(negative_tolerance) * scale:
        raise ArithmeticError(
            f"target/source MMD squared is materially negative ({raw_squared})"
        )
    clamped = raw_squared < 0.0
    squared = max(raw_squared, 0.0)
    return DistanceResult(
        distance=float(np.sqrt(squared)),
        squared_distance=squared,
        raw_squared_distance=raw_squared,
        clamped=clamped,
    )


def distances_to_sources(
    target: EmpiricalKME,
    sources: Mapping[str, ReducedRKME],
    *,
    block_size: int = 2048,
    negative_tolerance: float = 1.0e-8,
) -> tuple[dict[str, DistanceResult], int]:
    """Return deterministic task-sorted distances and the numeric clamp count."""

    results = {
        source_id: empirical_to_reduced_distance(
            target,
            sources[source_id],
            block_size=block_size,
            negative_tolerance=negative_tolerance,
        )
        for source_id in sorted(sources)
    }
    return results, sum(result.clamped for result in results.values())
