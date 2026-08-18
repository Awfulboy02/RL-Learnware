"""Environment schema and reset/step golden inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import ProtocolDraft
from ..hashing import sha256_json
from ..io import atomic_write_json, atomic_write_npz
from ..schemas import EnvSchema
from .factory import make_env_adapter
from .mujoco_playground import MujocoPlaygroundEnvAdapter


@dataclass(frozen=True)
class EnvironmentInspection:
    schema: EnvSchema
    registry_config: Mapping[str, Any]
    reset_seed: int
    reset_observation: np.ndarray
    fixed_action: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    full_rollout_transition_count: int = 0
    full_rollout_early_done: bool = False
    full_rollout_final_terminated: bool = False
    full_rollout_final_truncated: bool = False

    def summary_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "registry_config": dict(self.registry_config),
            "reset_seed": self.reset_seed,
            "golden": {
                "reset_observation_sha256": sha256_json(
                    self.reset_observation.tolist()
                ),
                "fixed_action_sha256": sha256_json(self.fixed_action.tolist()),
                "next_observation_sha256": sha256_json(
                    self.next_observation.tolist()
                ),
                "reward": self.reward,
                "terminated": self.terminated,
                "truncated": self.truncated,
                "full_rollout_transition_count": self.full_rollout_transition_count,
                "full_rollout_early_done": self.full_rollout_early_done,
                "full_rollout_final_terminated": self.full_rollout_final_terminated,
                "full_rollout_final_truncated": self.full_rollout_final_truncated,
            },
        }


def inspect_environment(
    task: str,
    config: ProtocolDraft,
    *,
    reset_seed: int = 0,
    jit: bool = True,
) -> EnvironmentInspection:
    """Inspect one registered task and fail closed on protocol disagreement."""

    adapter = make_env_adapter(task, config, jit=jit)
    schema = adapter.schema
    if schema.horizon != config.environment.horizon:
        raise RuntimeError(
            f"{task} horizon {schema.horizon} != protocol {config.environment.horizon}"
        )
    if schema.action_repeat != config.environment.action_repeat:
        raise RuntimeError(
            f"{task} action_repeat {schema.action_repeat} != protocol "
            f"{config.environment.action_repeat}"
        )
    if schema.observation_dim > config.environment.max_observation_dim:
        raise RuntimeError(f"{task} observation dimension exceeds canonical maximum")
    if schema.action_dim > config.environment.max_action_dim:
        raise RuntimeError(f"{task} action dimension exceeds canonical maximum")
    if not np.allclose(schema.action_low, config.probe.action_low) or not np.allclose(
        schema.action_high, config.probe.action_high
    ):
        raise RuntimeError(
            f"{task} actuator bounds disagree with the clipped-Gaussian protocol"
        )

    state, observation = adapter.reset(reset_seed)
    fixed_action = np.zeros(schema.action_dim, dtype=np.float32)
    _, result = adapter.step(state, fixed_action)
    registry_config = (
        adapter.registry_config
        if isinstance(adapter, MujocoPlaygroundEnvAdapter)
        else {"test_only": True}
    )
    rollout_count = 0
    early_done = False
    final_terminated = False
    final_truncated = False
    if hasattr(adapter, "collect_clipped_gaussian_batch"):
        rollout = adapter.collect_clipped_gaussian_batch(  # type: ignore[attr-defined]
            reset_seeds=np.asarray([reset_seed], dtype=np.int64),
            probe_seeds=np.asarray([0], dtype=np.int64),
            sigma=0.0,
        )
        rollout_count = int(np.asarray(rollout["reward"]).size)
        rollout_terminated = np.asarray(rollout["terminated"], dtype=np.bool_)
        rollout_truncated = np.asarray(rollout["truncated"], dtype=np.bool_)
        early_done = bool(
            np.any(np.logical_or(rollout_terminated[:-1], rollout_truncated[:-1]))
        )
        final_terminated = bool(rollout_terminated[-1])
        final_truncated = bool(rollout_truncated[-1])
        if rollout_count != schema.horizon or early_done:
            raise RuntimeError(f"{task} violates fixed-horizon rollout semantics")
    return EnvironmentInspection(
        schema=schema,
        registry_config=registry_config,
        reset_seed=reset_seed,
        reset_observation=observation,
        fixed_action=fixed_action,
        next_observation=result.observation,
        reward=result.reward,
        terminated=result.terminated,
        truncated=result.truncated,
        full_rollout_transition_count=rollout_count,
        full_rollout_early_done=early_done,
        full_rollout_final_terminated=final_terminated,
        full_rollout_final_truncated=final_truncated,
    )


def inspect_environments(
    config: ProtocolDraft, *, reset_seed: int = 0, jit: bool = True
) -> dict[str, EnvironmentInspection]:
    return {
        task: inspect_environment(task, config, reset_seed=reset_seed, jit=jit)
        for task in config.environment.tasks
    }


def save_inspections(
    inspections: Mapping[str, EnvironmentInspection],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write JSON schemas/config plus task-keyed golden arrays atomically."""

    output = Path(output_directory)
    summaries = {
        task: inspection.summary_dict()
        for task, inspection in sorted(inspections.items())
    }
    arrays: dict[str, np.ndarray] = {}
    for task, inspection in sorted(inspections.items()):
        arrays[f"{task}__reset_observation"] = inspection.reset_observation
        arrays[f"{task}__fixed_action"] = inspection.fixed_action
        arrays[f"{task}__next_observation"] = inspection.next_observation
        arrays[f"{task}__reward"] = np.asarray(inspection.reward, dtype=np.float32)
        arrays[f"{task}__terminated"] = np.asarray(
            inspection.terminated, dtype=np.bool_
        )
        arrays[f"{task}__truncated"] = np.asarray(
            inspection.truncated, dtype=np.bool_
        )
    return {
        "env_schemas": atomic_write_json(
            output / "env_schemas.json", summaries, overwrite=overwrite
        ),
        "env_golden_io": atomic_write_npz(
            output / "env_golden_io.npz", arrays, overwrite=overwrite
        ),
    }
