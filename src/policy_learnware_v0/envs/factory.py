"""Fail-closed adapter factory."""

from __future__ import annotations

from typing import Any

from ..config import EnvironmentConfig, ProtocolDraft
from .base import EnvAdapter, SyntheticEnvAdapter
from .mujoco_playground import MujocoPlaygroundEnvAdapter


def make_env_adapter(
    task: str,
    config: ProtocolDraft | EnvironmentConfig,
    *,
    allow_synthetic: bool = False,
    **kwargs: Any,
) -> EnvAdapter:
    environment = config.environment if isinstance(config, ProtocolDraft) else config
    if task not in environment.tasks:
        raise ValueError(f"task {task!r} is not registered by this protocol")
    if environment.backend == "mujoco_playground.registry":
        return MujocoPlaygroundEnvAdapter(
            task,
            expected_horizon=environment.horizon,
            expected_action_repeat=environment.action_repeat,
            **kwargs,
        )
    if environment.backend == "synthetic.test-only" and allow_synthetic:
        return SyntheticEnvAdapter(
            task=task,
            observation_dim=environment.max_observation_dim,
            action_dim=environment.max_action_dim,
            horizon=environment.horizon,
            **kwargs,
        )
    raise ValueError(
        f"unsupported environment backend {environment.backend!r}; synthetic "
        "adapters require the explicit allow_synthetic test flag"
    )
