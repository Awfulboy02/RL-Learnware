"""Execution-only accelerators for exact nested-prefix retrieval evaluation."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..rkme.empirical import episode_balanced_weights
from ..rkme.gaussian import GaussianKernel


@lru_cache(maxsize=8)
def _nested_prefix_program(
    block_count: int,
    block_size: int,
    point_dim: int,
    prefix_count: int,
    sigma_squared: float,
) -> Any:
    """Cache the compiled shape/kernel-specific program across query banks."""

    import jax
    import jax.numpy as jnp

    @jax.jit
    def compute(points_device: Any, weights_device: Any) -> Any:
        def outer(left_index: Any, totals: Any) -> Any:
            left_start = left_index * block_size
            left = jax.lax.dynamic_slice(
                points_device, (left_start, 0), (block_size, point_dim)
            )
            left_weights = jax.lax.dynamic_slice(
                weights_device,
                (0, left_start),
                (prefix_count, block_size),
            )

            def inner(right_index: Any, subtotals: Any) -> Any:
                right_start = right_index * block_size
                right = jax.lax.dynamic_slice(
                    points_device,
                    (right_start, 0),
                    (block_size, point_dim),
                )
                right_weights = jax.lax.dynamic_slice(
                    weights_device,
                    (0, right_start),
                    (prefix_count, block_size),
                )
                squared = jnp.maximum(
                    jnp.sum(left * left, axis=1)[:, None]
                    + jnp.sum(right * right, axis=1)[None, :]
                    - 2.0 * (left @ right.T),
                    0.0,
                )
                gram = jnp.exp(-squared / (2.0 * sigma_squared))
                values_by_prefix = jax.vmap(
                    lambda left_weight, right_weight: left_weight
                    @ gram
                    @ right_weight
                )(left_weights, right_weights)
                multiplier = jnp.where(right_index == left_index, 1.0, 2.0)
                return subtotals + multiplier * values_by_prefix

            return jax.lax.fori_loop(left_index, block_count, inner, totals)

        return jax.lax.fori_loop(
            0,
            block_count,
            outer,
            jnp.zeros((prefix_count,), dtype=jnp.float64),
        )

    return compute


def nested_prefix_self_kernel_sums_jax(
    points: np.ndarray,
    episode_offsets: np.ndarray,
    prefix_episode_counts: tuple[int, ...],
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
) -> dict[int, float]:
    """Evaluate exact self-KME norms for nested prefixes in one block pass.

    Every kernel block of the largest prefix is evaluated once.  Per-prefix
    weight masks recover the same episode-balanced quadratic form that
    ``build_empirical_kme`` would compute independently for each prefix.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - production dependency gate
        raise RuntimeError("JAX nested-prefix self-kernel backend is unavailable") from error
    jax.config.update("jax_enable_x64", True)
    values = np.asarray(points, dtype=np.float64)
    offsets = np.asarray(episode_offsets, dtype=np.int64)
    prefixes = tuple(int(value) for value in prefix_episode_counts)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != values.shape[0]
        or not prefixes
        or tuple(sorted(set(prefixes))) != prefixes
        or prefixes[-1] > offsets.size - 1
        or block_size <= 0
    ):
        raise ValueError("nested prefix points/offsets/prefixes are invalid")
    transition_stops = np.asarray([offsets[count] for count in prefixes], dtype=np.int64)
    weight_rows = np.zeros((len(prefixes), values.shape[0]), dtype=np.float64)
    for row, count in enumerate(prefixes):
        weight_rows[row, : transition_stops[row]] = episode_balanced_weights(
            offsets[: count + 1]
        )
    block_count = (values.shape[0] + block_size - 1) // block_size
    padded_count = block_count * block_size
    padded_points = np.zeros((padded_count, values.shape[1]), dtype=np.float64)
    padded_weights = np.zeros((len(prefixes), padded_count), dtype=np.float64)
    padded_points[: values.shape[0]] = values
    padded_weights[:, : values.shape[0]] = weight_rows
    points_device = jnp.asarray(padded_points)
    weights_device = jnp.asarray(padded_weights)
    sigma_squared = float(kernel.bandwidth) ** 2

    compute = _nested_prefix_program(
        block_count,
        block_size,
        values.shape[1],
        len(prefixes),
        sigma_squared,
    )
    results = np.asarray(
        jax.device_get(compute(points_device, weights_device)), dtype=np.float64
    )
    if results.shape != (len(prefixes),) or not np.all(np.isfinite(results)):
        raise RuntimeError("nested prefix self-kernel computation is invalid")
    return {
        count: max(float(value), 0.0)
        for count, value in zip(prefixes, results, strict=True)
    }


__all__ = ["nested_prefix_self_kernel_sums_jax"]
