"""Frozen-v0 TaskSpec measurements for the v0.1 dynamics-shift diagnostic.

This module deliberately contains no environment context or policy logic.  It
accepts an opaque measurement schema and an :class:`EpisodeDataset`, applies
the already-frozen v0 representation, and exposes exact Gaussian-RKHS terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from ..probe.dataset import EpisodeDataset
from ..hashing import sha256_json
from ..representation.canonicalizer import TransitionCanonicalizer
from ..representation.encoder import TransitionSemanticEncoder
from ..representation.normalization import NormalizationStats
from ..rkme.empirical import (
    EmpiricalKME,
    blockwise_weighted_kernel_sum,
    blockwise_weighted_kernel_sum_jax,
    blockwise_weighted_self_kernel_sum,
    blockwise_weighted_self_kernel_sum_jax,
    episode_balanced_weights,
)
from ..rkme.gaussian import GaussianKernel
from ..rkme.reducer import ReducedRKME
from .plans import verify_pair_plan


@dataclass(frozen=True)
class WeightedSemanticSample:
    """A light-weight KME input whose self norm has not been computed."""

    points: np.ndarray
    weights: np.ndarray
    episode_offsets: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        offsets = np.asarray(self.episode_offsets, dtype=np.int64)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("points must have non-empty shape [T,Q]")
        if weights.shape != (points.shape[0],):
            raise ValueError("weights must have shape [T]")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
            raise ValueError("weights must be nonnegative and sum to one")
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != points.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("invalid episode_offsets")
        if not np.all(np.isfinite(points)):
            raise ValueError("semantic points contain non-finite values")
        points = np.ascontiguousarray(points)
        weights = np.ascontiguousarray(weights)
        offsets = np.ascontiguousarray(offsets)
        points.setflags(write=False)
        weights.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "episode_offsets", offsets)

    @classmethod
    def from_points(
        cls, points: np.ndarray, episode_offsets: np.ndarray
    ) -> "WeightedSemanticSample":
        offsets = np.asarray(episode_offsets, dtype=np.int64)
        return cls(
            points=np.asarray(points),
            weights=episode_balanced_weights(offsets),
            episode_offsets=offsets,
        )

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    def prefix(self, episode_count: int) -> "WeightedSemanticSample":
        if not 1 <= int(episode_count) <= self.episode_count:
            raise ValueError("episode prefix lies outside semantic sample")
        stop = int(self.episode_offsets[int(episode_count)])
        offsets = self.episode_offsets[: int(episode_count) + 1]
        return WeightedSemanticSample.from_points(self.points[:stop], offsets)

    def cache_arrays(self) -> dict[str, np.ndarray]:
        """Return the selector-safe cache payload (no task/context fields)."""

        return {
            "points": self.points,
            "weights": self.weights,
            "episode_offsets": self.episode_offsets,
        }


@dataclass(frozen=True)
class RawMMDResult:
    raw_mmd2: float
    mmd2: float
    d_phi: float
    roundoff_clamped: bool
    left_norm2: float
    right_norm2: float
    cross_term: float


@dataclass(frozen=True)
class TaskSpecMatrixResult:
    """Selector-safe primitive rows produced from a frozen sparse pair plan."""

    plan_digest: str
    pair_rows: tuple[dict[str, Any], ...]
    routing_rows: tuple[dict[str, Any], ...]
    self_norm_rows: tuple[dict[str, Any], ...]
    clamp_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v01-taskspec-matrix.v0",
            "plan_digest": self.plan_digest,
            "pair_rows": list(self.pair_rows),
            "routing_rows": list(self.routing_rows),
            "self_norm_rows": list(self.self_norm_rows),
            "clamp_count": self.clamp_count,
        }


def taskspec_primitives_payload(
    pair_plan: Mapping[str, Any], matrix: Mapping[str, Any] | TaskSpecMatrixResult
) -> dict[str, Any]:
    """Canonical merge primitives published independently of matrix summaries."""

    plan_digest = verify_pair_plan(pair_plan)
    payload = matrix.to_dict() if isinstance(matrix, TaskSpecMatrixResult) else dict(matrix)
    if payload.get("schema") != "policy-learnware.v01-taskspec-matrix.v0":
        raise ValueError("unsupported TaskSpec matrix schema")
    if payload.get("plan_digest") != plan_digest:
        raise ValueError("TaskSpec matrix does not bind the frozen pair plan")
    return {
        "schema": "policy-learnware.v01-taskspec-primitives.v0",
        "plan_digest": plan_digest,
        "self_terms": sorted(
            [dict(row) for row in payload["self_norm_rows"]],
            key=lambda row: (
                str(row["variant_id"]), int(row["bank"]), int(row["prefix"])
            ),
        ),
        "cross_terms": [
            {
                key: row[key]
                for key in (
                    "family", "pair_index", "left_variant_id", "left_bank",
                    "right_variant_id", "right_bank", "prefix", "cross_term",
                )
            }
            for row in payload["pair_rows"]
        ],
        "routing_scores": [
            {
                "routing_index": row["routing_index"],
                "variant_id": row["variant_id"],
                "bank": row["bank"],
                "prefix": row["prefix"],
                "scores": dict(
                    sorted(
                        (str(item["source_id"]), float(item["routing_score"]))
                        for item in row["ranking"]
                    )
                ),
            }
            for row in payload["routing_rows"]
        ],
    }


def taskspec_primitive_digest(
    pair_plan: Mapping[str, Any], matrix: Mapping[str, Any] | TaskSpecMatrixResult
) -> str:
    """Digest all primitive self/cross/routing terms before later auditing."""

    return sha256_json(taskspec_primitives_payload(pair_plan, matrix))


def _kernel_cross(
    left: WeightedSemanticSample,
    right: WeightedSemanticSample,
    kernel: GaussianKernel,
    *,
    block_size: int,
    computation_backend: str,
) -> float:
    function = {
        "numpy": blockwise_weighted_kernel_sum,
        "jax": blockwise_weighted_kernel_sum_jax,
    }.get(computation_backend)
    if function is None:
        raise ValueError("computation_backend must be numpy or jax")
    return float(
        function(
            left.points,
            left.weights,
            right.points,
            right.weights,
            kernel,
            block_size=block_size,
        )
    )


def exact_self_norm(
    sample: WeightedSemanticSample,
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
    computation_backend: str = "numpy",
) -> float:
    function = {
        "numpy": blockwise_weighted_self_kernel_sum,
        "jax": blockwise_weighted_self_kernel_sum_jax,
    }.get(computation_backend)
    if function is None:
        raise ValueError("computation_backend must be numpy or jax")
    return float(
        function(
            sample.points,
            sample.weights,
            kernel,
            block_size=block_size,
        )
    )


def empirical_mmd_with_raw(
    left: WeightedSemanticSample | EmpiricalKME,
    right: WeightedSemanticSample | EmpiricalKME,
    kernel: GaussianKernel,
    *,
    left_norm2: float | None = None,
    right_norm2: float | None = None,
    block_size: int = 2048,
    negative_tolerance: float = 1.0e-8,
    computation_backend: str = "numpy",
) -> RawMMDResult:
    """Compute exact MMD and retain the pre-clamp squared value.

    The material-negative rule is byte-for-byte equivalent in meaning to v0's
    :func:`empirical_mmd2`; only the additional diagnostic fields are new.
    """

    left_sample = WeightedSemanticSample(
        left.points, left.weights, left.episode_offsets
    )
    right_sample = WeightedSemanticSample(
        right.points, right.weights, right.episode_offsets
    )
    resolved_left_norm = (
        float(left.norm2)
        if isinstance(left, EmpiricalKME) and left_norm2 is None
        else (
            exact_self_norm(
                left_sample,
                kernel,
                block_size=block_size,
                computation_backend=computation_backend,
            )
            if left_norm2 is None
            else float(left_norm2)
        )
    )
    resolved_right_norm = (
        float(right.norm2)
        if isinstance(right, EmpiricalKME) and right_norm2 is None
        else (
            exact_self_norm(
                right_sample,
                kernel,
                block_size=block_size,
                computation_backend=computation_backend,
            )
            if right_norm2 is None
            else float(right_norm2)
        )
    )
    cross = _kernel_cross(
        left_sample,
        right_sample,
        kernel,
        block_size=block_size,
        computation_backend=computation_backend,
    )
    raw = float(resolved_left_norm + resolved_right_norm - 2.0 * cross)
    scale = max(
        1.0,
        abs(resolved_left_norm),
        abs(resolved_right_norm),
        abs(2.0 * cross),
    )
    if raw < -float(negative_tolerance) * scale:
        raise ArithmeticError(f"empirical MMD squared is materially negative ({raw})")
    mmd2 = max(raw, 0.0)
    return RawMMDResult(
        raw_mmd2=raw,
        mmd2=mmd2,
        d_phi=float(np.sqrt(mmd2)),
        roundoff_clamped=raw < 0.0,
        left_norm2=resolved_left_norm,
        right_norm2=resolved_right_norm,
        cross_term=cross,
    )


def direct_routing_scores(
    target: WeightedSemanticSample,
    sources: Mapping[str, ReducedRKME],
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
    computation_backend: str = "numpy",
) -> dict[str, float]:
    """Return exact self-cancelling source scores in stable source-id order."""

    if computation_backend == "numpy":
        cross_function = blockwise_weighted_kernel_sum
    elif computation_backend == "jax":
        cross_function = blockwise_weighted_kernel_sum_jax
    else:
        raise ValueError("computation_backend must be numpy or jax")
    result: dict[str, float] = {}
    for source_id in sorted(sources):
        source = sources[source_id]
        if not np.isclose(source.bandwidth, kernel.bandwidth, rtol=0.0, atol=0.0):
            raise ValueError("source RKME and target use different bandwidths")
        cross = cross_function(
            target.points,
            target.weights,
            source.supports,
            source.beta,
            kernel,
            block_size=block_size,
        )
        result[source_id] = float(source.rkme_norm2 - 2.0 * cross)
    return result


def compute_taskspec_matrix(
    samples: Mapping[tuple[str, int], WeightedSemanticSample],
    pair_plan: Mapping[str, Any],
    *,
    kernel: GaussianKernel,
    sources: Mapping[str, ReducedRKME],
    block_size: int = 2048,
    negative_tolerance: float = 1.0e-8,
    computation_backend: str = "numpy",
) -> TaskSpecMatrixResult:
    """Execute exactly the frozen pair/routing plan with term-level reuse."""

    plan_digest = verify_pair_plan(pair_plan)
    prefixes: dict[tuple[str, int, int], WeightedSemanticSample] = {}
    self_norms: dict[tuple[str, int, int], float] = {}

    def prefix_for(variant_id: str, bank: int, prefix: int) -> WeightedSemanticSample:
        key = (variant_id, int(bank), int(prefix))
        if key not in prefixes:
            try:
                full = samples[(variant_id, int(bank))]
            except KeyError as error:
                raise ValueError(
                    f"pair plan references missing semantic sample {variant_id}/{bank}"
                ) from error
            prefixes[key] = full.prefix(int(prefix))
        return prefixes[key]

    def norm_for(variant_id: str, bank: int, prefix: int) -> float:
        key = (variant_id, int(bank), int(prefix))
        if key not in self_norms:
            self_norms[key] = exact_self_norm(
                prefix_for(*key),
                kernel,
                block_size=block_size,
                computation_backend=computation_backend,
            )
        return self_norms[key]

    pair_rows: list[dict[str, Any]] = []
    clamp_count = 0
    for family in ("within", "between"):
        for index, record in enumerate(pair_plan[family]):
            left_id = str(record["left_variant_id"])
            right_id = str(record["right_variant_id"])
            left_bank = int(record["left_bank"])
            right_bank = int(record["right_bank"])
            prefix = int(record["prefix"])
            result = empirical_mmd_with_raw(
                prefix_for(left_id, left_bank, prefix),
                prefix_for(right_id, right_bank, prefix),
                kernel,
                left_norm2=norm_for(left_id, left_bank, prefix),
                right_norm2=norm_for(right_id, right_bank, prefix),
                block_size=block_size,
                negative_tolerance=negative_tolerance,
                computation_backend=computation_backend,
            )
            clamp_count += int(result.roundoff_clamped)
            pair_rows.append(
                {
                    "family": family,
                    "pair_index": index,
                    "left_variant_id": left_id,
                    "left_bank": left_bank,
                    "right_variant_id": right_id,
                    "right_bank": right_bank,
                    "prefix": prefix,
                    "raw_mmd2": result.raw_mmd2,
                    "mmd2": result.mmd2,
                    "d_phi": result.d_phi,
                    "roundoff_clamped": result.roundoff_clamped,
                    "cross_term": result.cross_term,
                }
            )
    routing_rows: list[dict[str, Any]] = []
    for index, record in enumerate(pair_plan["routing"]):
        variant_id = str(record["variant_id"])
        bank = int(record["bank"])
        prefix = int(record["prefix"])
        scores = direct_routing_scores(
            prefix_for(variant_id, bank, prefix),
            sources,
            kernel,
            block_size=block_size,
            computation_backend=computation_backend,
        )
        ranking = sorted(scores, key=lambda source_id: (scores[source_id], source_id))
        routing_rows.append(
            {
                "routing_index": index,
                "variant_id": variant_id,
                "bank": bank,
                "prefix": prefix,
                "selected_source_id": ranking[0],
                "ranking": [
                    {"source_id": source_id, "routing_score": scores[source_id]}
                    for source_id in ranking
                ],
            }
        )
    self_rows = tuple(
        {
            "variant_id": key[0],
            "bank": key[1],
            "prefix": key[2],
            "self_norm2": value,
        }
        for key, value in sorted(self_norms.items())
    )
    return TaskSpecMatrixResult(
        plan_digest=plan_digest,
        pair_rows=tuple(pair_rows),
        routing_rows=tuple(routing_rows),
        self_norm_rows=self_rows,
        clamp_count=clamp_count,
    )


def _schema_facade(view: Any, variant_id: str) -> Any:
    """Build an in-memory v0 canonicalizer facade without persisting a task key."""

    return SimpleNamespace(
        task=str(variant_id),
        observation_dim=int(view.observation_dim),
        action_dim=int(view.action_dim),
        flatten_fingerprint=str(view.flatten_fingerprint_without_task),
    )


def encode_measurement_dataset(
    dataset: EpisodeDataset,
    schema_view: Any,
    *,
    variant_id: str,
    normalizer: NormalizationStats,
    encoder: TransitionSemanticEncoder,
    max_action_dim: int,
    encode_batch_size: int = 8192,
) -> WeightedSemanticSample:
    """Apply the frozen v0 representation using measurement-only inputs."""

    if dataset.observation_dim != int(schema_view.observation_dim):
        raise ValueError("measurement observation width differs from schema view")
    if dataset.action_dim != int(schema_view.action_dim):
        raise ValueError("measurement action width differs from schema view")
    facade = _schema_facade(schema_view, variant_id)
    packed = TransitionCanonicalizer(
        normalizer, max_action_dim=int(max_action_dim)
    ).pack(dataset, facade)
    points = encoder.encode(packed.packed, batch_size=int(encode_batch_size))
    return WeightedSemanticSample.from_points(points, packed.episode_offsets)


def mask_only_distance(left_view: Any, right_view: Any) -> float:
    """Schema/mask negative control for two measurement views.

    Within a registered task family this must be exactly zero.  The explicit
    vector form makes accidental schema drift observable without using context.
    """

    def vector(view: Any) -> np.ndarray:
        return np.asarray(
            [
                float(view.observation_dim),
                float(view.action_dim),
                float(view.horizon),
                float(view.action_repeat),
                float(view.control_dt),
            ],
            dtype=np.float64,
        )

    return float(np.linalg.norm(vector(left_view) - vector(right_view)))


__all__ = [
    "RawMMDResult",
    "TaskSpecMatrixResult",
    "WeightedSemanticSample",
    "compute_taskspec_matrix",
    "direct_routing_scores",
    "empirical_mmd_with_raw",
    "encode_measurement_dataset",
    "exact_self_norm",
    "mask_only_distance",
    "taskspec_primitive_digest",
    "taskspec_primitives_payload",
]
