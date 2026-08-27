"""Lazy MuJoCo Playground adapter for the six DMC tasks."""

from __future__ import annotations

import importlib.metadata
import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..hashing import sha256_file, sha256_json
from ..schemas import EnvSchema, StepResult
from ..probe.gaussian import sample_clipped_gaussian_episode_jax


class MujocoPlaygroundUnavailableError(RuntimeError):
    pass


MUJOCO_PLAYGROUND_DISTRIBUTION_NAMES = (
    "playground",
    "mujoco-playground",
    "mujoco_playground",
)


def mujoco_playground_package_version() -> str | None:
    """Return the installed Playground distribution version, if present.

    The upstream package that provides the :mod:`mujoco_playground` import is
    published as ``playground``.  The other names are retained as fallbacks for
    environments that install an aliased/repacked distribution.
    """

    for distribution in MUJOCO_PLAYGROUND_DISTRIBUTION_NAMES:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _serializable(value.to_dict())
    if hasattr(value, "items"):
        return {str(key): _serializable(item) for key, item in value.items()}
    return repr(value)


def _attribute_path(root: Any, path: str) -> Any | None:
    value = root
    for name in path.split("."):
        if not hasattr(value, name):
            return None
        value = getattr(value, name)
    return value


class MujocoPlaygroundEnvAdapter:
    """Native functional adapter; importing this module has no JAX side effects."""

    backend = "mujoco_playground.registry"

    def __init__(
        self,
        task: str,
        *,
        native_environment: Any | None = None,
        expected_horizon: int | None = None,
        expected_action_repeat: int | None = None,
        jit: bool = True,
    ) -> None:
        try:
            import jax
            from mujoco_playground import registry
        except ImportError as exc:
            raise MujocoPlaygroundUnavailableError(
                "MujocoPlaygroundEnvAdapter requires JAX and mujoco_playground"
            ) from exc

        self._jax = jax
        self._registry = registry
        self._task = task
        if native_environment is None:
            self._registry_config_object = registry.get_default_config(task)
            self._env = registry.load(task, config=self._registry_config_object)
        else:
            config = getattr(native_environment, "_config", None)
            if config is None:
                raise ValueError(
                    "native_environment must expose the registry config as _config"
                )
            self._registry_config_object = config
            self._env = native_environment
        self._reset_fn = jax.jit(self._env.reset) if jit else self._env.reset
        self._step_fn = jax.jit(self._env.step) if jit else self._env.step

        reset_key = self._key(0)
        state = self._reset_fn(reset_key)
        observation = self._flat_observation(state.obs)
        observation_dim = int(observation.size)
        action_dim = self._infer_action_dim()
        action_low, action_high = self._infer_action_bounds(action_dim)
        actual_horizon = self._config_value("episode_length", default=None)
        actual_action_repeat = self._config_value("action_repeat", default=None)
        if actual_horizon is None or actual_action_repeat is None:
            raise RuntimeError(
                "registry config does not expose episode_length/action_repeat"
            )
        horizon = int(actual_horizon)
        action_repeat = int(actual_action_repeat)
        if expected_horizon is not None and horizon != int(expected_horizon):
            raise RuntimeError(
                f"registry horizon {horizon} != expected {expected_horizon}"
            )
        if (
            expected_action_repeat is not None
            and action_repeat != int(expected_action_repeat)
        ):
            raise RuntimeError(
                "registry action_repeat "
                f"{action_repeat} != expected {expected_action_repeat}"
            )
        control_dt = self._infer_control_dt(action_repeat)

        source_path = inspect.getsourcefile(self._env.__class__)
        implementation_digest = (
            sha256_file(source_path)
            if source_path is not None and Path(source_path).is_file()
            else sha256_json(
                {
                    "module": self._env.__class__.__module__,
                    "class": self._env.__class__.__qualname__,
                }
            )
        )
        flatten_fingerprint = sha256_json(
            {
                "rule": "np.asarray(state.obs).reshape(-1,C)-v0",
                "native_shape": list(np.asarray(self._jax.device_get(state.obs)).shape),
                "flat_dim": observation_dim,
                "dtype": str(observation.dtype),
            }
        )
        self._schema = EnvSchema(
            backend=self.backend,
            task=task,
            observation_dim=observation_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            action_repeat=action_repeat,
            control_dt=control_dt,
            flatten_fingerprint=flatten_fingerprint,
            implementation_digest=implementation_digest,
            observation_dtype=str(observation.dtype),
            action_dtype="float32",
        )

    @property
    def schema(self) -> EnvSchema:
        return self._schema

    @property
    def registry_config(self) -> dict[str, Any]:
        return _serializable(self._registry_config_object)

    @property
    def environment(self) -> Any:
        return self._env

    def _key(self, seed: int) -> Any:
        random = self._jax.random
        if hasattr(random, "key"):
            return random.key(int(seed))
        return random.PRNGKey(int(seed))  # pragma: no cover - old JAX fallback

    def _flat_observation(self, value: Any) -> np.ndarray:
        array = np.asarray(self._jax.device_get(value), dtype=np.float32)
        result = np.ascontiguousarray(array.reshape(-1))
        if not np.all(np.isfinite(result)):
            raise RuntimeError("environment emitted a non-finite observation")
        return result

    def _config_value(self, name: str, *, default: Any) -> Any:
        config = self._registry_config_object
        if isinstance(config, Mapping) and name in config:
            return config[name]
        return getattr(config, name, default)

    def _infer_action_dim(self) -> int:
        for name in ("action_size", "action_dim"):
            value = getattr(self._env, name, None)
            if value is not None:
                return int(value() if callable(value) else value)
        for path in (
            "mjx_model.nu",
            "_mjx_model.nu",
            "sys.nu",
            "model.nu",
        ):
            value = _attribute_path(self._env, path)
            if value is not None:
                return int(value)
        raise RuntimeError("could not infer action dimension from MuJoCo Playground env")

    def _infer_action_bounds(self, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
        for path in (
            "mjx_model.actuator_ctrlrange",
            "_mjx_model.actuator_ctrlrange",
            "sys.actuator.ctrl_range",
            "model.actuator_ctrlrange",
        ):
            value = _attribute_path(self._env, path)
            if value is None:
                continue
            ranges = np.asarray(self._jax.device_get(value), dtype=np.float32)
            if ranges.shape == (action_dim, 2):
                return ranges[:, 0], ranges[:, 1]
        raise RuntimeError(
            "could not inspect actuator control ranges; refusing to assume [-1, 1]"
        )

    def _infer_control_dt(self, action_repeat: int) -> float:
        value = getattr(self._env, "dt", None)
        if value is not None:
            value = value() if callable(value) else value
            return float(value)
        timestep = _attribute_path(self._env, "mjx_model.opt.timestep")
        if timestep is None:
            timestep = _attribute_path(self._env, "_mjx_model.opt.timestep")
        if timestep is None:
            raise RuntimeError("could not infer environment control dt")
        return float(timestep) * action_repeat

    def reset(self, seed: int) -> tuple[Any, np.ndarray]:
        state = self._reset_fn(self._key(seed))
        return state, self._flat_observation(state.obs)

    def step(self, state: Any, action: np.ndarray) -> tuple[Any, StepResult]:
        native_action = np.asarray(action, dtype=np.float32)
        if native_action.shape != (self.schema.action_dim,):
            raise ValueError(
                f"action shape {native_action.shape} does not match "
                f"({self.schema.action_dim},)"
            )
        next_state = self._step_fn(state, native_action)
        info = getattr(next_state, "info", {}) or {}
        truncation_value = info.get("truncation", False) if isinstance(info, Mapping) else False
        truncated = bool(np.asarray(self._jax.device_get(truncation_value)))
        done = bool(np.asarray(self._jax.device_get(getattr(next_state, "done", False))))
        terminated = done and not truncated
        reward = float(np.asarray(self._jax.device_get(next_state.reward)))
        serializable_info = (
            {str(key): _serializable(self._jax.device_get(value)) for key, value in info.items()}
            if isinstance(info, Mapping)
            else {}
        )
        return next_state, StepResult(
            observation=self._flat_observation(next_state.obs),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=serializable_info,
        )

    def collect_clipped_gaussian_batch(
        self,
        *,
        reset_seeds: np.ndarray,
        probe_seeds: np.ndarray,
        sigma: float,
        steps: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Collect fixed-horizon DMC episodes with one JIT ``scan`` + ``vmap``.

        The six v0 tasks have no early terminal condition under the native DMC
        interface.  We nevertheless inspect the flags and fail closed if that
        assumption is violated, rather than accidentally storing post-terminal
        transitions.
        """

        jax = self._jax
        import jax.numpy as jnp
        reset_values = np.asarray(reset_seeds, dtype=np.int64)
        probe_values = np.asarray(probe_seeds, dtype=np.int64)
        if reset_values.ndim != 1 or probe_values.shape != reset_values.shape:
            raise ValueError("reset_seeds and probe_seeds must be same-shape vectors")
        if reset_values.size == 0:
            raise ValueError("at least one episode seed is required")
        rollout_steps = self.schema.horizon if steps is None else steps
        if (
            isinstance(rollout_steps, bool)
            or not isinstance(rollout_steps, (int, np.integer))
            or not 1 <= int(rollout_steps) <= self.schema.horizon
        ):
            raise ValueError("steps must lie in [1, environment horizon]")
        rollout_steps = int(rollout_steps)

        def keys_from_seeds(values: np.ndarray) -> Any:
            device_values = jnp.asarray(values, dtype=jnp.uint32)
            if hasattr(jax.random, "key"):
                return jax.vmap(jax.random.key)(device_values)
            return jax.vmap(jax.random.PRNGKey)(device_values)  # pragma: no cover

        reset_keys = keys_from_seeds(reset_values)
        probe_keys = keys_from_seeds(probe_values)
        batched_reset = jax.jit(jax.vmap(self._env.reset))
        initial_state = batched_reset(reset_keys)
        low = jnp.asarray(self.schema.action_low, dtype=jnp.float32)
        high = jnp.asarray(self.schema.action_high, dtype=jnp.float32)

        def episode_actions(key: Any) -> Any:
            return sample_clipped_gaussian_episode_jax(
                key,
                steps=rollout_steps,
                action_dim=self.schema.action_dim,
                sigma=float(sigma),
                action_low=low,
                action_high=high,
            )

        # [N,H,d_a] -> scan-major [H,N,d_a]
        actions_by_episode = jax.vmap(episode_actions)(probe_keys)
        actions_by_step = jnp.swapaxes(actions_by_episode, 0, 1)
        batched_step = jax.vmap(self._env.step)

        def scan_step(state: Any, action: Any) -> tuple[Any, tuple[Any, ...]]:
            observation = jnp.reshape(state.obs, (reset_values.size, -1))
            next_state = batched_step(state, action)
            next_observation = jnp.reshape(
                next_state.obs, (reset_values.size, -1)
            )
            done = jnp.asarray(getattr(next_state, "done", False), dtype=jnp.bool_)
            info = getattr(next_state, "info", {}) or {}
            truncation = (
                jnp.asarray(info.get("truncation", jnp.zeros_like(done)), dtype=jnp.bool_)
                if isinstance(info, Mapping)
                else jnp.zeros_like(done)
            )
            terminated = jnp.logical_and(done, jnp.logical_not(truncation))
            return next_state, (
                observation,
                action,
                next_state.reward,
                next_observation,
                terminated,
                truncation,
            )

        _, scanned = jax.jit(
            lambda state, actions: jax.lax.scan(scan_step, state, actions)
        )(initial_state, actions_by_step)
        observation, action, reward, next_observation, terminated, truncated = (
            np.asarray(jax.device_get(value)) for value in scanned
        )
        # Convert [H,N,...] to episode-major [N,H,...].
        observation = np.swapaxes(observation, 0, 1)
        action = np.swapaxes(action, 0, 1)
        reward = np.swapaxes(reward, 0, 1)
        next_observation = np.swapaxes(next_observation, 0, 1)
        terminated = np.swapaxes(terminated, 0, 1).astype(np.bool_)
        truncated = np.swapaxes(truncated, 0, 1).astype(np.bool_)
        if np.any(np.logical_or(terminated[:, :-1], truncated[:, :-1])):
            raise RuntimeError(
                "a v0 DMC environment ended before the registered fixed horizon"
            )
        truncated[:, -1] = np.logical_or(
            truncated[:, -1], np.logical_not(terminated[:, -1])
        )
        episode_count = reset_values.size
        return {
            "observation": observation.reshape(
                episode_count * rollout_steps, self.schema.observation_dim
            ).astype(np.float32),
            "action": action.reshape(
                episode_count * rollout_steps, self.schema.action_dim
            ).astype(np.float32),
            "reward": reward.reshape(-1).astype(np.float32),
            "next_observation": next_observation.reshape(
                episode_count * rollout_steps, self.schema.observation_dim
            ).astype(np.float32),
            "terminated": terminated.reshape(-1),
            "truncated": truncated.reshape(-1),
        }

    @staticmethod
    def package_version() -> str:
        return mujoco_playground_package_version() or "unknown"
