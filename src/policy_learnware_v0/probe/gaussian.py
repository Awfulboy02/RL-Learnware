"""Clipped isotropic Gaussian probe with explicit RNG state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import ProbeConfig
from ..schemas import EnvSchema


def sample_clipped_gaussian_episode_jax(
    key: Any,
    *,
    steps: int,
    action_dim: int,
    sigma: float,
    action_low: Any,
    action_high: Any,
) -> Any:
    """The sole production RNG rule: one Threefry draw for a full episode."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("production probe sampling requires JAX") from exc
    if steps <= 0 or action_dim <= 0:
        raise ValueError("probe episode dimensions must be positive")
    low = jnp.asarray(action_low, dtype=jnp.float32)
    high = jnp.asarray(action_high, dtype=jnp.float32)
    noise = jax.random.normal(key, (steps, action_dim), dtype=jnp.float32)
    return jnp.clip(float(sigma) * noise, low, high)


@dataclass(frozen=True)
class GaussianRandomProbe:
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("probe sigma must be finite and positive")

    @classmethod
    def from_config(cls, config: ProbeConfig) -> "GaussianRandomProbe":
        if config.type != "clipped_gaussian":
            raise ValueError(f"unsupported probe type: {config.type!r}")
        return cls(sigma=config.sigma)

    def sample_sequence_numpy(
        self,
        *,
        seed: int,
        steps: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> np.ndarray:
        if steps <= 0:
            raise ValueError("steps must be positive")
        low = np.asarray(action_low, dtype=np.float32)
        high = np.asarray(action_high, dtype=np.float32)
        if low.ndim != 1 or high.shape != low.shape or np.any(low >= high):
            raise ValueError("action bounds must be same-shape valid vectors")
        rng = np.random.default_rng(int(seed))
        noise = rng.standard_normal((steps, low.size), dtype=np.float32)
        return np.clip(self.sigma * noise, low[None, :], high[None, :]).astype(
            np.float32, copy=False
        )

    def sample_episode_numpy(
        self, *, seed: int, steps: int, schema: EnvSchema
    ) -> np.ndarray:
        return self.sample_sequence_numpy(
            seed=seed,
            steps=steps,
            action_low=schema.action_low,
            action_high=schema.action_high,
        )

    def sample_jax(
        self, key: Any, action_low: Any, action_high: Any
    ) -> tuple[Any, Any]:
        """Test/helper one-step sampler; production uses the full-episode rule."""

        try:
            import jax
            import jax.numpy as jnp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sample_jax requires JAX") from exc
        next_key, sample_key = jax.random.split(key)
        low = jnp.asarray(action_low, dtype=jnp.float32)
        high = jnp.asarray(action_high, dtype=jnp.float32)
        noise = jax.random.normal(sample_key, low.shape, dtype=jnp.float32)
        return jnp.clip(self.sigma * noise, low, high), next_key
