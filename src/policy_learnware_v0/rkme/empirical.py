"""Episode-balanced empirical kernel mean embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from .gaussian import GaussianKernel


def _readonly(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def episode_balanced_weights(episode_offsets: np.ndarray) -> np.ndarray:
    """Return alpha[n,h] = 1 / (N H_n) in flattened transition order."""

    offsets = np.asarray(episode_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0:
        raise ValueError("episode_offsets must have shape [N+1] and start at zero")
    lengths = np.diff(offsets)
    if np.any(lengths <= 0):
        raise ValueError("all episodes must contain at least one transition")
    episode_count = int(lengths.size)
    weights = np.concatenate(
        [np.full(int(length), 1.0 / (episode_count * length)) for length in lengths]
    ).astype(np.float64)
    # Correct the final bit of accumulation error without changing episode ratios.
    weights /= np.sum(weights)
    return weights


def dense_weighted_kernel_sum(
    left: np.ndarray,
    left_weights: np.ndarray,
    right: np.ndarray,
    right_weights: np.ndarray,
    kernel: GaussianKernel,
) -> float:
    x = np.asarray(left)
    y = np.asarray(right)
    wx = np.asarray(left_weights, dtype=np.float64)
    wy = np.asarray(right_weights, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("point arrays must have compatible shapes [N,Q], [M,Q]")
    if wx.shape != (x.shape[0],) or wy.shape != (y.shape[0],):
        raise ValueError("kernel weights do not align with points")
    return float(wx @ kernel.gram(x, y) @ wy)


def blockwise_weighted_kernel_sum(
    left: np.ndarray,
    left_weights: np.ndarray,
    right: np.ndarray,
    right_weights: np.ndarray,
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
) -> float:
    """Compute ``w_x^T K_xy w_y`` without materializing the full Gram matrix."""

    x = np.asarray(left)
    y = np.asarray(right)
    wx = np.asarray(left_weights, dtype=np.float64)
    wy = np.asarray(right_weights, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("point arrays must have compatible shapes [N,Q], [M,Q]")
    if wx.shape != (x.shape[0],) or wy.shape != (y.shape[0],):
        raise ValueError("kernel weights do not align with points")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    total = 0.0
    for left_start in range(0, x.shape[0], block_size):
        left_stop = min(left_start + block_size, x.shape[0])
        for right_start in range(0, y.shape[0], block_size):
            right_stop = min(right_start + block_size, y.shape[0])
            gram = kernel.gram(
                x[left_start:left_stop], y[right_start:right_stop]
            )
            total += float(
                wx[left_start:left_stop]
                @ gram
                @ wy[right_start:right_stop]
            )
    return total


def blockwise_weighted_kernel_sum_jax(
    left: np.ndarray,
    left_weights: np.ndarray,
    right: np.ndarray,
    right_weights: np.ndarray,
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
) -> float:
    """Exact JAX/GPU cross-kernel sum without a full Gram allocation."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RuntimeError("JAX cross-kernel backend is unavailable") from error
    jax.config.update("jax_enable_x64", True)
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    wx = np.asarray(left_weights, dtype=np.float64)
    wy = np.asarray(right_weights, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or wx.shape != (x.shape[0],)
        or wy.shape != (y.shape[0],)
        or block_size <= 0
    ):
        raise ValueError("cross-kernel arrays/block_size are invalid")
    left_blocks = (x.shape[0] + block_size - 1) // block_size
    right_blocks = (y.shape[0] + block_size - 1) // block_size

    def pad(array: np.ndarray, count: int) -> np.ndarray:
        result = np.zeros((count * block_size, *array.shape[1:]), dtype=np.float64)
        result[: array.shape[0]] = array
        return result

    x_device = jnp.asarray(pad(x, left_blocks))
    y_device = jnp.asarray(pad(y, right_blocks))
    wx_device = jnp.asarray(pad(wx, left_blocks))
    wy_device = jnp.asarray(pad(wy, right_blocks))
    sigma_squared = float(kernel.bandwidth) ** 2

    @jax.jit
    def compute() -> Any:
        def outer(left_index: Any, total: Any) -> Any:
            left_start = left_index * block_size
            left_block = jax.lax.dynamic_slice(
                x_device, (left_start, 0), (block_size, x.shape[1])
            )
            left_weight = jax.lax.dynamic_slice(
                wx_device, (left_start,), (block_size,)
            )

            def inner(right_index: Any, subtotal: Any) -> Any:
                right_start = right_index * block_size
                right_block = jax.lax.dynamic_slice(
                    y_device, (right_start, 0), (block_size, y.shape[1])
                )
                right_weight = jax.lax.dynamic_slice(
                    wy_device, (right_start,), (block_size,)
                )
                squared = jnp.maximum(
                    jnp.sum(left_block * left_block, axis=1)[:, None]
                    + jnp.sum(right_block * right_block, axis=1)[None, :]
                    - 2.0 * (left_block @ right_block.T),
                    0.0,
                )
                gram = jnp.exp(-squared / (2.0 * sigma_squared))
                return subtotal + left_weight @ gram @ right_weight

            return jax.lax.fori_loop(0, right_blocks, inner, total)

        return jax.lax.fori_loop(
            0, left_blocks, outer, jnp.asarray(0.0, dtype=jnp.float64)
        )

    return float(jax.device_get(compute()))


def blockwise_weighted_self_kernel_sum(
    points: np.ndarray,
    weights: np.ndarray,
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
) -> float:
    """Compute ``w^T K_xx w`` blockwise, exploiting Gram symmetry."""

    x = np.asarray(points)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.shape != (x.shape[0],):
        raise ValueError("self-kernel points and weights are not aligned")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    total = 0.0
    for left_start in range(0, x.shape[0], block_size):
        left_stop = min(left_start + block_size, x.shape[0])
        for right_start in range(left_start, x.shape[0], block_size):
            right_stop = min(right_start + block_size, x.shape[0])
            contribution = float(
                w[left_start:left_stop]
                @ kernel.gram(
                    x[left_start:left_stop], x[right_start:right_stop]
                )
                @ w[right_start:right_stop]
            )
            total += contribution if right_start == left_start else 2.0 * contribution
    return total


def blockwise_weighted_self_kernel_sum_jax(
    points: np.ndarray,
    weights: np.ndarray,
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
) -> float:
    """GPU/JIT implementation of the exact symmetric blockwise quadratic sum."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RuntimeError("JAX self-kernel backend is unavailable") from error
    jax.config.update("jax_enable_x64", True)
    x = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.shape != (x.shape[0],) or block_size <= 0:
        raise ValueError("self-kernel points/weights/block_size are invalid")
    block_count = (x.shape[0] + block_size - 1) // block_size
    padded_count = block_count * block_size
    padded_x = np.zeros((padded_count, x.shape[1]), dtype=np.float64)
    padded_w = np.zeros((padded_count,), dtype=np.float64)
    padded_x[: x.shape[0]] = x
    padded_w[: x.shape[0]] = w
    x_device = jnp.asarray(padded_x)
    w_device = jnp.asarray(padded_w)
    sigma_squared = float(kernel.bandwidth) ** 2

    @jax.jit
    def compute() -> Any:
        def outer(left_index: Any, total: Any) -> Any:
            left_start = left_index * block_size
            left = jax.lax.dynamic_slice(
                x_device, (left_start, 0), (block_size, x.shape[1])
            )
            left_weight = jax.lax.dynamic_slice(
                w_device, (left_start,), (block_size,)
            )

            def inner(right_index: Any, subtotal: Any) -> Any:
                right_start = right_index * block_size
                right = jax.lax.dynamic_slice(
                    x_device, (right_start, 0), (block_size, x.shape[1])
                )
                right_weight = jax.lax.dynamic_slice(
                    w_device, (right_start,), (block_size,)
                )
                squared = jnp.maximum(
                    jnp.sum(left * left, axis=1)[:, None]
                    + jnp.sum(right * right, axis=1)[None, :]
                    - 2.0 * (left @ right.T),
                    0.0,
                )
                gram = jnp.exp(-squared / (2.0 * sigma_squared))
                value = left_weight @ gram @ right_weight
                multiplier = jnp.where(right_index == left_index, 1.0, 2.0)
                return subtotal + multiplier * value

            return jax.lax.fori_loop(left_index, block_count, inner, total)

        return jax.lax.fori_loop(
            0, block_count, outer, jnp.asarray(0.0, dtype=jnp.float64)
        )

    return float(jax.device_get(compute()))


def blockwise_weighted_self_kernel_sum_auto(
    points: np.ndarray,
    weights: np.ndarray,
    kernel: GaussianKernel,
    *,
    block_size: int = 2048,
    jax_threshold: int = 8192,
) -> float:
    if np.asarray(points).shape[0] >= jax_threshold:
        try:
            return blockwise_weighted_self_kernel_sum_jax(
                points, weights, kernel, block_size=block_size
            )
        except RuntimeError:
            pass
    return blockwise_weighted_self_kernel_sum(
        points, weights, kernel, block_size=block_size
    )


@dataclass(frozen=True)
class EmpiricalKME:
    points: np.ndarray
    weights: np.ndarray
    episode_offsets: np.ndarray
    bandwidth: float
    norm2: float
    protocol_id: str = ""
    dataset_digest: str = ""
    source_task: str = ""
    source_dataset_manifest_digest: str = ""

    def __post_init__(self) -> None:
        points = _readonly(self.points, dtype=np.float64, name="points")
        weights = _readonly(self.weights, dtype=np.float64, name="weights")
        offsets = np.asarray(self.episode_offsets, dtype=np.int64)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("points must have non-empty shape [T,Q]")
        if weights.shape != (points.shape[0],):
            raise ValueError("weights must have shape [T]")
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0, atol=1e-10):
            raise ValueError("empirical KME weights must be nonnegative and sum to one")
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != points.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise ValueError("invalid empirical episode_offsets")
        if not np.isfinite(self.bandwidth) or float(self.bandwidth) <= 0:
            raise ValueError("bandwidth must be finite and positive")
        if not np.isfinite(self.norm2) or float(self.norm2) < -1e-10:
            raise ValueError("norm2 must be finite and nonnegative")
        offsets = np.array(offsets, copy=True)
        offsets.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "episode_offsets", offsets)
        object.__setattr__(self, "bandwidth", float(self.bandwidth))
        object.__setattr__(self, "norm2", max(float(self.norm2), 0.0))

    @property
    def empirical_norm2(self) -> float:
        return self.norm2

    @property
    def transition_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        from ..io import atomic_write_npz

        return atomic_write_npz(
            path,
            {
                "points": self.points,
                "weights": self.weights,
                "episode_offsets": self.episode_offsets,
                "bandwidth": np.asarray(self.bandwidth, dtype=np.float64),
                "norm2": np.asarray(self.norm2, dtype=np.float64),
                "protocol_id": np.asarray(self.protocol_id),
                "dataset_digest": np.asarray(self.dataset_digest),
                "source_task": np.asarray(self.source_task),
                "source_dataset_manifest_digest": np.asarray(
                    self.source_dataset_manifest_digest
                ),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "EmpiricalKME":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                points=data["points"],
                weights=data["weights"],
                episode_offsets=data["episode_offsets"],
                bandwidth=float(data["bandwidth"]),
                norm2=float(data["norm2"]),
                protocol_id=str(data["protocol_id"]),
                dataset_digest=str(data["dataset_digest"]),
                source_task=(str(data["source_task"]) if "source_task" in data else ""),
                source_dataset_manifest_digest=(
                    str(data["source_dataset_manifest_digest"])
                    if "source_dataset_manifest_digest" in data
                    else ""
                ),
            )


_EXACT_NORM2_ATTESTATION = object()


def _norm2_binding_digest(empirical: EmpiricalKME) -> str:
    """Bind an in-memory exact-norm attestation to all selector inputs."""

    arrays_digest = sha256_ndarrays(
        {
            "points": empirical.points,
            "weights": empirical.weights,
            "episode_offsets": empirical.episode_offsets,
        }
    )
    return sha256_json(
        {
            "arrays_sha256": arrays_digest,
            "bandwidth": empirical.bandwidth,
            "norm2": empirical.norm2,
            "protocol_id": empirical.protocol_id,
            "dataset_digest": empirical.dataset_digest,
            "source_task": empirical.source_task,
            "source_dataset_manifest_digest": (
                empirical.source_dataset_manifest_digest
            ),
        }
    )


def _attest_exact_norm2(empirical: EmpiricalKME) -> EmpiricalKME:
    # The attribute is deliberately absent from the dataclass fields and public
    # constructor.  Only an internal producer that has just completed the exact
    # self-kernel calculation can attach the module-private sentinel.
    object.__setattr__(
        empirical,
        "_exact_norm2_attestation",
        (_EXACT_NORM2_ATTESTATION, _norm2_binding_digest(empirical)),
    )
    return empirical


def _has_exact_norm2_attestation(value: Any) -> bool:
    if not isinstance(value, EmpiricalKME):
        return False
    attestation = getattr(value, "_exact_norm2_attestation", None)
    return (
        isinstance(attestation, tuple)
        and len(attestation) == 2
        and attestation[0] is _EXACT_NORM2_ATTESTATION
        and attestation[1] == _norm2_binding_digest(value)
    )


def _semantic_points(semantic_events: Any) -> np.ndarray:
    for attribute in ("points", "semantic_events", "embeddings", "z"):
        if hasattr(semantic_events, attribute):
            return np.asarray(getattr(semantic_events, attribute), dtype=np.float64)
    if isinstance(semantic_events, np.ndarray):
        return np.asarray(semantic_events, dtype=np.float64)
    raise TypeError("semantic_events must expose points/embeddings")


def build_empirical_kme(
    semantic_events: Any,
    kernel: GaussianKernel,
    *,
    episode_offsets: np.ndarray | None = None,
    protocol_id: str = "",
    dataset_digest: str = "",
    source_task: str = "",
    source_dataset_manifest_digest: str = "",
    block_size: int = 2048,
    computation_backend: str = "numpy",
) -> EmpiricalKME:
    points = _semantic_points(semantic_events)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("semantic points must have non-empty shape [T,Q]")
    if episode_offsets is None:
        episode_offsets = getattr(semantic_events, "episode_offsets", None)
    if episode_offsets is None:
        episode_offsets = np.asarray([0, points.shape[0]], dtype=np.int64)
    offsets = np.asarray(episode_offsets, dtype=np.int64)
    weights = episode_balanced_weights(offsets)
    if weights.shape[0] != points.shape[0]:
        raise ValueError("episode_offsets do not end at the number of semantic points")
    if computation_backend == "numpy":
        norm2 = blockwise_weighted_self_kernel_sum(
            points, weights, kernel, block_size=block_size
        )
    elif computation_backend == "jax":
        norm2 = blockwise_weighted_self_kernel_sum_jax(
            points, weights, kernel, block_size=block_size
        )
    else:
        raise ValueError("computation_backend must be numpy or jax")
    return _attest_exact_norm2(
        EmpiricalKME(
            points=points,
            weights=weights,
            episode_offsets=offsets,
            bandwidth=kernel.bandwidth,
            norm2=norm2,
            protocol_id=protocol_id,
            dataset_digest=dataset_digest,
            source_task=source_task,
            source_dataset_manifest_digest=source_dataset_manifest_digest,
        )
    )


def empirical_mmd2(
    left: EmpiricalKME,
    right: EmpiricalKME,
    *,
    block_size: int = 2048,
    negative_tolerance: float = 1.0e-8,
    computation_backend: str = "numpy",
) -> float:
    if not np.isclose(left.bandwidth, right.bandwidth, rtol=0.0, atol=0.0):
        raise ValueError("empirical KMEs use different Gaussian bandwidths")
    if left.protocol_id and right.protocol_id and left.protocol_id != right.protocol_id:
        raise ValueError("empirical KMEs use different protocols")
    kernel = GaussianKernel(left.bandwidth)
    if computation_backend == "numpy":
        cross = blockwise_weighted_kernel_sum(
            left.points,
            left.weights,
            right.points,
            right.weights,
            kernel,
            block_size=block_size,
        )
    elif computation_backend == "jax":
        cross = blockwise_weighted_kernel_sum_jax(
            left.points,
            left.weights,
            right.points,
            right.weights,
            kernel,
            block_size=block_size,
        )
    else:
        raise ValueError("computation_backend must be numpy or jax")
    raw = float(left.norm2 + right.norm2 - 2.0 * cross)
    scale = max(1.0, abs(left.norm2), abs(right.norm2), abs(2.0 * cross))
    if raw < -float(negative_tolerance) * scale:
        raise ArithmeticError(
            f"empirical MMD squared is materially negative ({raw})"
        )
    return max(raw, 0.0)
