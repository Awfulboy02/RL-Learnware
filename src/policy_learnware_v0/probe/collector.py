"""Reference probe collector preserving exact transition alignment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import ProbeConfig, ProtocolDraft
from ..envs.base import EnvAdapter
from .dataset import EpisodeDataset
from .gaussian import GaussianRandomProbe
from .seed_plan import SeedPlan


def collect_probe_episodes(
    env: EnvAdapter,
    split: str,
    episode_ids: Sequence[int],
    config: ProbeConfig | ProtocolDraft,
    *,
    seed_plan: SeedPlan | None = None,
    task_index: int | None = None,
    bank_index: int = 0,
    prefer_vectorized: bool = True,
) -> EpisodeDataset:
    """Collect candidate-independent random-probe episodes.

    This backend-neutral path is intentionally simple and serves as the
    correctness reference.  It stores the observation returned by ``step`` as
    terminal ``next_observation`` and never substitutes an autoreset value.
    """

    if isinstance(config, ProtocolDraft):
        protocol = config
        probe_config = protocol.probe
        if seed_plan is None:
            seed_plan = SeedPlan(protocol.project_seed)
        if task_index is None:
            try:
                task_index = protocol.environment.tasks.index(env.schema.task)
            except ValueError as exc:
                raise ValueError(
                    f"environment task {env.schema.task!r} is not in the protocol"
                ) from exc
    else:
        probe_config = config
    if seed_plan is None or task_index is None:
        raise ValueError(
            "ProbeConfig collection requires explicit seed_plan and task_index; "
            "passing ProtocolDraft supplies both automatically"
        )

    identifiers = tuple(int(identifier) for identifier in episode_ids)
    if not identifiers or any(identifier < 0 for identifier in identifiers):
        raise ValueError("episode_ids must be a non-empty sequence of non-negative ids")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("episode_ids must be unique")
    schema = env.schema
    if not np.allclose(schema.action_low, probe_config.action_low) or not np.allclose(
        schema.action_high, probe_config.action_high
    ):
        raise ValueError("registered action bounds disagree with probe config")
    probe = GaussianRandomProbe.from_config(probe_config)

    if split != "target_query" and bank_index != 0:
        raise ValueError("only target_query collection may use bank_index > 0")
    if isinstance(config, ProtocolDraft) and (
        config.probe.rng_backend == "jax_threefry_full_episode_v0"
        and config.environment.backend != "synthetic.test-only"
        and (not prefer_vectorized or not hasattr(env, "collect_clipped_gaussian_batch"))
    ):
        raise RuntimeError(
            "the frozen production probe RNG requires the vectorized JAX collector"
        )

    episode_seed_records = tuple(
        seed_plan.episode(
            split, task_index, episode_id, bank_index=bank_index
        )
        for episode_id in identifiers
    )
    if prefer_vectorized and hasattr(env, "collect_clipped_gaussian_batch"):
        reset_array = np.asarray(
            [record.reset_seed for record in episode_seed_records], dtype=np.int64
        )
        probe_array = np.asarray(
            [record.probe_seed for record in episode_seed_records], dtype=np.int64
        )
        arrays = env.collect_clipped_gaussian_batch(  # type: ignore[attr-defined]
            reset_seeds=reset_array,
            probe_seeds=probe_array,
            sigma=probe.sigma,
        )
        episode_count = len(episode_seed_records)
        arrays.update(
            {
                "episode_offsets": np.arange(
                    episode_count + 1, dtype=np.int64
                )
                * schema.horizon,
                "reset_seeds": reset_array,
                "probe_seeds": probe_array,
            }
        )
        return EpisodeDataset(**arrays)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    offsets = [0]
    reset_seeds: list[int] = []
    probe_seeds: list[int] = []

    for seeds in episode_seed_records:
        reset_seeds.append(seeds.reset_seed)
        probe_seeds.append(seeds.probe_seed)
        state, observation = env.reset(seeds.reset_seed)
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation.shape != (schema.observation_dim,):
            raise RuntimeError("reset observation disagrees with registered schema")
        action_sequence = probe.sample_episode_numpy(
            seed=seeds.probe_seed, steps=schema.horizon, schema=schema
        )

        for step_index in range(schema.horizon):
            action = action_sequence[step_index]
            next_state, result = env.step(state, action)
            next_observation = np.asarray(result.observation, dtype=np.float32).reshape(
                -1
            )
            if next_observation.shape != (schema.observation_dim,):
                raise RuntimeError("step observation disagrees with registered schema")
            terminated = bool(result.terminated)
            truncated = bool(result.truncated)
            if step_index + 1 == schema.horizon and not terminated:
                truncated = True

            observations.append(observation.copy())
            actions.append(action.copy())
            rewards.append(float(result.reward))
            next_observations.append(next_observation.copy())
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)

            state = next_state
            observation = next_observation
            if terminated or truncated:
                break
        offsets.append(len(observations))

    return EpisodeDataset(
        observation=np.asarray(observations, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        next_observation=np.asarray(next_observations, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        episode_offsets=np.asarray(offsets, dtype=np.int64),
        reset_seeds=np.asarray(reset_seeds, dtype=np.int64),
        probe_seeds=np.asarray(probe_seeds, dtype=np.int64),
    )
