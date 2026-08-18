"""Accelerator-resident frozen-policy evaluation for fixed-horizon MJX tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class FrozenPolicyEvaluationError(RuntimeError):
    """A frozen policy rollout violated the registered evaluation contract."""


@dataclass(frozen=True)
class CompiledPolicyParity:
    passed: bool
    max_abs_error: float
    atol: float
    rtol: float
    sample_count: int
    next_keys_equal: bool


def verify_compiled_policy_parity(
    policy: Any,
    observations: np.ndarray,
    base_key_data: np.ndarray,
    *,
    atol: float,
    rtol: float,
    sample_count: int = 2,
) -> CompiledPolicyParity:
    """Compare the exact evaluator ``jit(lax.map(act))`` with scalar native acts."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - production dependency gate
        raise FrozenPolicyEvaluationError(
            "compiled policy parity requires JAX"
        ) from error

    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise FrozenPolicyEvaluationError("golden observations must have shape [N,D]")
    count = min(int(sample_count), values.shape[0])
    if count <= 0:
        raise FrozenPolicyEvaluationError("compiled parity sample count must be positive")
    wrap = getattr(jax.random, "wrap_key_data", None)
    base_key = (
        wrap(jnp.asarray(base_key_data, dtype=jnp.uint32))
        if wrap is not None
        else jnp.asarray(base_key_data, dtype=jnp.uint32)
    )
    keys = jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
        jnp.arange(count, dtype=jnp.uint32)
    )
    observations_device = jnp.asarray(values[:count], dtype=jnp.float32)

    reference_actions: list[np.ndarray] = []
    reference_next_keys: list[np.ndarray] = []
    for index in range(count):
        action, next_key = policy.act(
            observations_device[index], keys[index], deterministic=True
        )
        reference_actions.append(np.asarray(jax.device_get(action)))
        reference_next_keys.append(
            np.asarray(jax.device_get(jax.random.key_data(next_key)))
        )

    native_state = getattr(policy, "native_state", None)

    def act_one(observation: Any, key: Any) -> tuple[Any, Any]:
        if native_state is None:
            return policy.act(observation, key, deterministic=True)
        raw_action, _ = native_state.sample_action(
            observation, key, deterministic=True
        )
        return jnp.tanh(raw_action), jax.random.split(key, 2)[1]

    compiled_actions, compiled_next_keys = jax.jit(
        lambda observation_batch, key_batch: jax.lax.map(
            lambda pair: act_one(pair[0], pair[1]),
            (observation_batch, key_batch),
        )
    )(observations_device, keys)
    compiled_actions_array = np.asarray(jax.device_get(compiled_actions))
    compiled_key_data = np.asarray(
        jax.device_get(jax.vmap(jax.random.key_data)(compiled_next_keys))
    )
    reference_actions_array = np.stack(reference_actions)
    reference_key_data = np.stack(reference_next_keys)
    max_abs_error = float(
        np.max(
            np.abs(compiled_actions_array - reference_actions_array),
            initial=0.0,
        )
    )
    next_keys_equal = bool(np.array_equal(compiled_key_data, reference_key_data))
    passed = bool(
        next_keys_equal
        and np.allclose(
            compiled_actions_array,
            reference_actions_array,
            atol=float(atol),
            rtol=float(rtol),
        )
    )
    return CompiledPolicyParity(
        passed=passed,
        max_abs_error=max_abs_error,
        atol=float(atol),
        rtol=float(rtol),
        sample_count=count,
        next_keys_equal=next_keys_equal,
    )


def evaluate_frozen_policy_returns_batched(
    policy: Any,
    environment: Any,
    *,
    reset_seeds: Sequence[int],
    policy_seeds: Sequence[int],
    horizon: int,
    observation_dim: int,
    action_dim: int,
) -> tuple[float, ...]:
    """Evaluate a candidate with one ``jit(lax.map + lax.scan)`` executable.

    The registered policy, action transform, per-episode seed streams, fixed
    horizon, episode count, and return-ranking rule are unchanged.  XLA may
    fuse float32 operations differently from scalar dispatch, so the evaluator
    is explicitly versioned and its compiled action path is parity-gated before
    every candidate rollout.
    """

    if len(reset_seeds) != len(policy_seeds) or not reset_seeds:
        raise FrozenPolicyEvaluationError(
            "evaluation seed vectors are empty or misaligned"
        )
    if horizon <= 0 or observation_dim <= 0 or action_dim <= 0:
        raise FrozenPolicyEvaluationError("evaluation dimensions must be positive")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:  # pragma: no cover - production dependency gate
        raise FrozenPolicyEvaluationError(
            "batched frozen-policy evaluation requires JAX"
        ) from error

    episode_count = len(reset_seeds)

    def keys_from_seeds(values: Sequence[int]) -> Any:
        seeds = jnp.asarray(np.asarray(values, dtype=np.uint32))
        if hasattr(jax.random, "key"):
            return jax.vmap(jax.random.key)(seeds)
        return jax.vmap(jax.random.PRNGKey)(seeds)  # pragma: no cover

    reset_keys = keys_from_seeds(reset_seeds)
    initial_policy_keys = keys_from_seeds(policy_seeds)
    native_state = getattr(policy, "native_state", None)

    def act_one(observation: Any, key: Any) -> tuple[Any, Any]:
        if native_state is None:
            return policy.act(observation, key, deterministic=True)
        raw_action, _ = native_state.sample_action(
            observation, key, deterministic=True
        )
        return jnp.tanh(raw_action), jax.random.split(key, 2)[1]

    def mapped_reset(keys: Any) -> Any:
        return jax.lax.map(environment.reset, keys)

    def mapped_act(observations: Any, keys: Any) -> tuple[Any, Any]:
        return jax.lax.map(
            lambda pair: act_one(pair[0], pair[1]),
            (observations, keys),
        )

    def mapped_step(states: Any, actions: Any) -> Any:
        return jax.lax.map(
            lambda pair: environment.step(pair[0], pair[1]),
            (states, actions),
        )

    def rollout(reset_key_batch: Any, policy_key_batch: Any) -> tuple[Any, ...]:
        initial_state = mapped_reset(reset_key_batch)
        initial_observation = jnp.reshape(
            initial_state.obs, (episode_count, observation_dim)
        )
        initial_finite = jnp.all(jnp.isfinite(initial_observation), axis=-1)
        initial_early = jnp.zeros((episode_count,), dtype=jnp.bool_)

        def scan_step(
            carry: tuple[Any, Any, Any, Any], step_index: Any
        ) -> tuple[tuple[Any, Any, Any, Any], Any]:
            state, keys, finite, ended_early = carry
            observation = jnp.reshape(
                state.obs, (episode_count, observation_dim)
            )
            action, next_keys = mapped_act(observation, keys)
            action = jnp.reshape(
                jnp.asarray(action, dtype=jnp.float32),
                (episode_count, action_dim),
            )
            next_state = mapped_step(state, action)
            next_observation = jnp.reshape(
                next_state.obs, (episode_count, observation_dim)
            )
            reward = jnp.reshape(
                jnp.asarray(next_state.reward), (episode_count,)
            )
            done = jnp.reshape(
                jnp.asarray(getattr(next_state, "done", False), dtype=jnp.bool_),
                (episode_count,),
            )
            info = getattr(next_state, "info", {}) or {}
            truncation = (
                jnp.reshape(
                    jnp.asarray(
                        info.get("truncation", jnp.zeros_like(done)),
                        dtype=jnp.bool_,
                    ),
                    (episode_count,),
                )
                if isinstance(info, Mapping)
                else jnp.zeros_like(done)
            )
            ended = jnp.logical_or(done, truncation)
            ended_early = jnp.logical_or(
                ended_early,
                jnp.logical_and(ended, step_index + 1 < horizon),
            )
            finite = jnp.logical_and(
                finite,
                jnp.logical_and(
                    jnp.all(jnp.isfinite(next_observation), axis=-1),
                    jnp.logical_and(
                        jnp.all(jnp.isfinite(action), axis=-1),
                        jnp.isfinite(reward),
                    ),
                ),
            )
            return (next_state, next_keys, finite, ended_early), reward

        (_, _, finite, ended_early), rewards = jax.lax.scan(
            scan_step,
            (
                initial_state,
                policy_key_batch,
                initial_finite,
                initial_early,
            ),
            jnp.arange(horizon, dtype=jnp.int32),
        )
        return rewards, finite, ended_early

    rewards, finite, ended_early = (
        np.asarray(jax.device_get(value))
        for value in jax.jit(rollout)(reset_keys, initial_policy_keys)
    )
    if rewards.shape != (horizon, episode_count):
        raise FrozenPolicyEvaluationError(
            f"batched reward shape {rewards.shape} != {(horizon, episode_count)}"
        )
    if np.any(ended_early):
        raise FrozenPolicyEvaluationError(
            "environment ended before the registered fixed horizon"
        )
    if not np.all(finite) or not np.all(np.isfinite(rewards)):
        raise FrozenPolicyEvaluationError(
            "policy evaluation emitted a non-finite action, observation, or reward"
        )
    return tuple(
        sum(float(reward) for reward in rewards[:, episode_index])
        for episode_index in range(episode_count)
    )


__all__ = [
    "CompiledPolicyParity",
    "FrozenPolicyEvaluationError",
    "evaluate_frozen_policy_returns_batched",
    "verify_compiled_policy_parity",
]
