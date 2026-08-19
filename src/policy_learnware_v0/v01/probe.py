"""Candidate-independent random-probe collection for v0.1 variants."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..probe.dataset import EpisodeDataset
from ..probe.gaussian import sample_clipped_gaussian_episode_jax


class ProbeCollectionError(RuntimeError):
    """The shifted environment violated the frozen measurement contract."""


def _jax_keys(jax: Any, values: np.ndarray) -> Any:
    import jax.numpy as jnp

    seeds = jnp.asarray(np.asarray(values, dtype=np.uint32))
    if hasattr(jax.random, "key"):
        return jax.vmap(jax.random.key)(seeds)
    return jax.vmap(jax.random.PRNGKey)(seeds)  # pragma: no cover


def frozen_probe_action_tensor(
    schema: Any,
    probe_seeds: Sequence[int],
    *,
    sigma: float = 1.0,
) -> Any:
    """Pre-generate the complete ``[N,H,A]`` Threefry action tensor.

    Generation is intentionally separate from environment stepping.  This is
    what makes paired actions across all five variants directly auditable.
    """

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - server dependency gate
        raise ProbeCollectionError("v0.1 production probe collection requires JAX") from error
    values = np.asarray(probe_seeds, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("probe_seeds must be a non-empty nonnegative vector")
    if not np.isfinite(sigma) or float(sigma) <= 0:
        raise ValueError("probe sigma must be finite and positive")
    low = jnp.asarray(schema.action_low, dtype=jnp.float32)
    high = jnp.asarray(schema.action_high, dtype=jnp.float32)
    keys = _jax_keys(jax, values)

    def sample(key: Any) -> Any:
        return sample_clipped_gaussian_episode_jax(
            key,
            steps=int(schema.horizon),
            action_dim=int(schema.action_dim),
            sigma=float(sigma),
            action_low=low,
            action_high=high,
        )

    return jax.vmap(sample)(keys)


def collect_probe_batch(
    adapter: Any,
    *,
    reset_seeds: Sequence[int],
    probe_seeds: Sequence[int],
    sigma: float = 1.0,
    action_tensor: Any | None = None,
) -> EpisodeDataset:
    """Collect paired fixed-horizon episodes with one JIT map/scan executable."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - server dependency gate
        raise ProbeCollectionError("v0.1 production probe collection requires JAX") from error
    schema = adapter.schema
    reset_values = np.asarray(reset_seeds, dtype=np.int64)
    probe_values = np.asarray(probe_seeds, dtype=np.int64)
    if (
        reset_values.ndim != 1
        or reset_values.size == 0
        or probe_values.shape != reset_values.shape
        or np.any(reset_values < 0)
        or np.any(probe_values < 0)
    ):
        raise ValueError("reset/probe seeds must be aligned nonnegative vectors")
    environment = adapter.environment
    if action_tensor is None:
        action_tensor = frozen_probe_action_tensor(
            schema, probe_values, sigma=sigma
        )
    else:
        action_tensor = jnp.asarray(action_tensor, dtype=jnp.float32)
        expected = (
            int(reset_values.size),
            int(schema.horizon),
            int(schema.action_dim),
        )
        if action_tensor.shape != expected:
            raise ValueError(
                f"pre-generated action_tensor has shape {action_tensor.shape}, expected {expected}"
            )
        low = jnp.asarray(schema.action_low, dtype=jnp.float32)
        high = jnp.asarray(schema.action_high, dtype=jnp.float32)
        if not bool(
            np.asarray(
                jax.device_get(
                    jnp.all(jnp.isfinite(action_tensor))
                    & jnp.all(action_tensor >= low)
                    & jnp.all(action_tensor <= high)
                )
            )
        ):
            raise ValueError("pre-generated action_tensor is non-finite or out of bounds")
    reset_keys = _jax_keys(jax, reset_values)
    episode_count = int(reset_values.size)
    initial_state = jax.lax.map(environment.reset, reset_keys)
    actions_by_step = jnp.swapaxes(action_tensor, 0, 1)

    def step(carry: Any, actions: Any) -> tuple[Any, tuple[Any, ...]]:
        observation = jnp.reshape(carry.obs, (episode_count, int(schema.observation_dim)))
        next_state = jax.lax.map(
            lambda pair: environment.step(pair[0], pair[1]), (carry, actions)
        )
        next_observation = jnp.reshape(
            next_state.obs, (episode_count, int(schema.observation_dim))
        )
        done = jnp.asarray(getattr(next_state, "done", False), dtype=jnp.bool_)
        info = getattr(next_state, "info", {}) or {}
        truncation = (
            jnp.asarray(info.get("truncation", jnp.zeros_like(done)), dtype=jnp.bool_)
            if hasattr(info, "get")
            else jnp.zeros_like(done)
        )
        terminated = jnp.logical_and(done, jnp.logical_not(truncation))
        return next_state, (
            observation,
            actions,
            next_state.reward,
            next_observation,
            terminated,
            truncation,
        )

    _, scanned = jax.jit(
        lambda state, actions: jax.lax.scan(step, state, actions)
    )(initial_state, actions_by_step)
    observation, action, reward, next_observation, terminated, truncated = (
        np.asarray(jax.device_get(value)) for value in scanned
    )
    arrays = [observation, action, reward, next_observation, terminated, truncated]
    arrays = [np.swapaxes(value, 0, 1) for value in arrays]
    observation, action, reward, next_observation, terminated, truncated = arrays
    terminated = terminated.astype(np.bool_)
    truncated = truncated.astype(np.bool_)
    ended = np.logical_or(terminated, truncated)
    if np.any(ended[:, :-1]):
        raise ProbeCollectionError("variant ended before the registered fixed horizon")
    if not all(
        np.all(np.isfinite(value))
        for value in (observation, action, reward, next_observation)
    ):
        raise ProbeCollectionError("variant emitted non-finite probe data")
    truncated[:, -1] = np.logical_or(
        truncated[:, -1], np.logical_not(terminated[:, -1])
    )
    horizon = int(schema.horizon)
    return EpisodeDataset(
        observation=observation.reshape(
            episode_count * horizon, int(schema.observation_dim)
        ).astype(np.float32),
        action=action.reshape(episode_count * horizon, int(schema.action_dim)).astype(
            np.float32
        ),
        reward=reward.reshape(-1).astype(np.float32),
        next_observation=next_observation.reshape(
            episode_count * horizon, int(schema.observation_dim)
        ).astype(np.float32),
        terminated=terminated.reshape(-1),
        truncated=truncated.reshape(-1),
        episode_offsets=np.arange(episode_count + 1, dtype=np.int64) * horizon,
        reset_seeds=reset_values,
        probe_seeds=probe_values,
    )


def collect_probe_scalar(
    adapter: Any,
    *,
    reset_seed: int,
    probe_seed: int,
    sigma: float = 1.0,
    action_tensor: Any | None = None,
) -> EpisodeDataset:
    """Reference scalar stepper used only for collection parity audits."""

    if action_tensor is None:
        actions = np.asarray(
            frozen_probe_action_tensor(adapter.schema, [probe_seed], sigma=sigma)[0]
        )
    else:
        supplied = np.asarray(action_tensor, dtype=np.float32)
        if supplied.shape == (
            1,
            int(adapter.schema.horizon),
            int(adapter.schema.action_dim),
        ):
            supplied = supplied[0]
        expected = (int(adapter.schema.horizon), int(adapter.schema.action_dim))
        if supplied.shape != expected:
            raise ValueError(
                f"pre-generated scalar action_tensor has shape {supplied.shape}, expected {expected}"
            )
        low = np.asarray(adapter.schema.action_low, dtype=np.float32)
        high = np.asarray(adapter.schema.action_high, dtype=np.float32)
        if (
            not np.all(np.isfinite(supplied))
            or np.any(supplied < low)
            or np.any(supplied > high)
        ):
            raise ValueError("pre-generated scalar action_tensor is non-finite or out of bounds")
        actions = np.array(supplied, copy=True)
    state, observation = adapter.reset(int(reset_seed))
    observations: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    for index, action in enumerate(actions):
        next_state, result = adapter.step(state, action)
        if (result.terminated or result.truncated) and index + 1 < adapter.schema.horizon:
            raise ProbeCollectionError("variant ended before the registered fixed horizon")
        observations.append(np.asarray(observation, dtype=np.float32))
        rewards.append(float(result.reward))
        next_observation = np.asarray(result.observation, dtype=np.float32)
        next_observations.append(next_observation)
        terminated.append(bool(result.terminated))
        truncated.append(
            bool(result.truncated)
            or (index + 1 == adapter.schema.horizon and not result.terminated)
        )
        state, observation = next_state, next_observation
    horizon = int(adapter.schema.horizon)
    return EpisodeDataset(
        observation=np.stack(observations),
        action=actions.astype(np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        next_observation=np.stack(next_observations),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
        episode_offsets=np.asarray([0, horizon], dtype=np.int64),
        reset_seeds=np.asarray([reset_seed], dtype=np.int64),
        probe_seeds=np.asarray([probe_seed], dtype=np.int64),
    )


__all__ = [
    "ProbeCollectionError",
    "collect_probe_batch",
    "collect_probe_scalar",
    "frozen_probe_action_tensor",
]
