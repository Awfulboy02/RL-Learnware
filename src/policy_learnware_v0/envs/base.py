"""Backend-independent environment interface and a test-only fake backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..hashing import sha256_json
from ..schemas import EnvSchema, StepResult


@runtime_checkable
class EnvAdapter(Protocol):
    @property
    def schema(self) -> EnvSchema: ...

    def reset(self, seed: int) -> tuple[Any, np.ndarray]: ...

    def step(self, state: Any, action: np.ndarray) -> tuple[Any, StepResult]: ...


@dataclass(frozen=True)
class SyntheticState:
    observation: np.ndarray
    step_index: int


class SyntheticEnvAdapter:
    """Small deterministic environment used only by unit/integration tests.

    Its backend name is intentionally not accepted by the research factory
    unless ``allow_synthetic=True`` is explicitly supplied.
    """

    def __init__(
        self,
        *,
        task: str = "SyntheticTask",
        observation_dim: int = 3,
        action_dim: int = 2,
        horizon: int = 5,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        if observation_dim <= 0 or action_dim <= 0 or horizon <= 0:
            raise ValueError("synthetic dimensions and horizon must be positive")
        fingerprint = sha256_json(
            {
                "rule": "reshape-c-order-v0",
                "observation_shape": [observation_dim],
                "observation_dtype": "float32",
            }
        )
        self._schema = EnvSchema(
            backend="synthetic.test-only",
            task=task,
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_low=np.full(action_dim, action_low, dtype=np.float32),
            action_high=np.full(action_dim, action_high, dtype=np.float32),
            horizon=horizon,
            action_repeat=1,
            control_dt=1.0,
            flatten_fingerprint=fingerprint,
            implementation_digest=sha256_json(
                {"implementation": "SyntheticEnvAdapter", "version": 1}
            ),
        )

    @property
    def schema(self) -> EnvSchema:
        return self._schema

    def reset(self, seed: int) -> tuple[SyntheticState, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        observation = rng.normal(0.0, 0.1, self.schema.observation_dim).astype(
            np.float32
        )
        return SyntheticState(observation=observation, step_index=0), observation.copy()

    def step(
        self, state: SyntheticState, action: np.ndarray
    ) -> tuple[SyntheticState, StepResult]:
        native_action = np.asarray(action, dtype=np.float32)
        if native_action.shape != (self.schema.action_dim,):
            raise ValueError(
                f"action shape {native_action.shape} does not match "
                f"({self.schema.action_dim},)"
            )
        if np.any(native_action < self.schema.action_low) or np.any(
            native_action > self.schema.action_high
        ):
            raise ValueError("synthetic action lies outside registered bounds")
        drive = np.zeros(self.schema.observation_dim, dtype=np.float32)
        width = min(self.schema.action_dim, self.schema.observation_dim)
        drive[:width] = native_action[:width]
        next_observation = np.tanh(
            0.85 * state.observation + 0.10 * drive + 0.001 * state.step_index
        ).astype(np.float32)
        step_index = state.step_index + 1
        truncated = step_index >= self.schema.horizon
        reward = float(
            1.0
            - np.mean(np.square(next_observation), dtype=np.float64)
            - 0.05 * np.mean(np.square(native_action), dtype=np.float64)
        )
        next_state = SyntheticState(next_observation, step_index)
        return next_state, StepResult(
            observation=next_observation,
            reward=reward,
            terminated=False,
            truncated=truncated,
            info={"synthetic": True, "step_index": step_index},
        )
