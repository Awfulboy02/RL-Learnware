"""CPU parity and reuse evidence for the production probe executor."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np

from policy_learnware_v0.hashing import sha256_bytes
from policy_learnware_v0.io import deterministic_npz_bytes
from policy_learnware_v0.v01.probe import ProbeBatchExecutor


try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised by dependency-light CI
    jax = None
    jnp = None


class _SyntheticState(NamedTuple):
    obs: Any
    reward: Any
    done: Any
    info: dict[str, Any]


class _SyntheticEnvironment:
    """Small batch-independent dynamics with the live environment pytree shape."""

    def __init__(self) -> None:
        # These Python side effects run while JAX traces the environment
        # methods.  Stable counts therefore provide direct retrace evidence.
        self.reset_trace_count = 0
        self.step_trace_count = 0

    def reset(self, key: Any) -> _SyntheticState:
        self.reset_trace_count += 1
        observation = jax.random.uniform(
            key,
            (3,),
            minval=jnp.float32(-0.25),
            maxval=jnp.float32(0.25),
            dtype=jnp.float32,
        )
        return _SyntheticState(
            obs=observation,
            reward=jnp.float32(0.0),
            done=jnp.asarray(False, dtype=jnp.bool_),
            info={"truncation": jnp.asarray(False, dtype=jnp.bool_)},
        )

    def step(self, state: _SyntheticState, action: Any) -> _SyntheticState:
        self.step_trace_count += 1
        action = jnp.asarray(action, dtype=jnp.float32)
        forcing = jnp.asarray(
            (action[0], action[1], action[0] - action[1]), dtype=jnp.float32
        )
        next_observation = (
            jnp.float32(0.875) * state.obs
            + jnp.float32(0.125) * forcing
            + jnp.float32(0.01)
        )
        reward = -jnp.sum(next_observation * next_observation, dtype=jnp.float32)
        return _SyntheticState(
            obs=next_observation,
            reward=reward,
            done=jnp.asarray(False, dtype=jnp.bool_),
            info={"truncation": jnp.asarray(False, dtype=jnp.bool_)},
        )


def _schema() -> Any:
    return SimpleNamespace(
        horizon=9,
        observation_dim=3,
        action_dim=2,
        action_low=np.asarray([-1.0, -0.75], dtype=np.float32),
        action_high=np.asarray([1.0, 0.75], dtype=np.float32),
    )


def _actions(*, episode_count: int, horizon: int, reverse: bool = False) -> np.ndarray:
    values = np.linspace(
        -0.7,
        0.7,
        num=episode_count * horizon * 2,
        dtype=np.float32,
    ).reshape(episode_count, horizon, 2)
    return np.flip(values, axis=(0, 1)).copy() if reverse else values


def _npz_bytes(dataset: Any) -> bytes:
    return deterministic_npz_bytes(dataset.to_arrays(copy=False))


@unittest.skipIf(jax is None, "JAX is unavailable")
class ProbeExecutorOptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cpu_devices = jax.devices("cpu")
        if not cls.cpu_devices:  # pragma: no cover - JAX normally exposes CPU
            raise unittest.SkipTest("JAX CPU backend is unavailable")

    def assertDatasetExact(self, left: Any, right: Any) -> None:
        left_arrays = left.to_arrays(copy=False)
        right_arrays = right.to_arrays(copy=False)
        self.assertEqual(left_arrays.keys(), right_arrays.keys())
        for name in left_arrays:
            self.assertEqual(left_arrays[name].dtype, right_arrays[name].dtype)
            self.assertEqual(left_arrays[name].shape, right_arrays[name].shape)
            np.testing.assert_array_equal(
                left_arrays[name],
                right_arrays[name],
                err_msg=f"array differs: {name}",
            )
        self.assertEqual(left.digest, right.digest)
        left_npz = _npz_bytes(left)
        right_npz = _npz_bytes(right)
        self.assertEqual(left_npz, right_npz)
        self.assertEqual(sha256_bytes(left_npz), sha256_bytes(right_npz))

    def test_production_lax_map_and_vmap_are_artifact_exact(self) -> None:
        schema = _schema()
        adapter = SimpleNamespace(
            schema=schema,
            environment=_SyntheticEnvironment(),
        )
        reset_seeds = np.asarray([7, 19, 101, 4093], dtype=np.int64)
        probe_seeds = np.asarray([11, 23, 103, 4099], dtype=np.int64)
        actions = _actions(
            episode_count=reset_seeds.size,
            horizon=schema.horizon,
        )

        # Pin this audit to CPU even when the test suite runs on a GPU host.
        with jax.default_device(self.cpu_devices[0]):
            lax_map = ProbeBatchExecutor(
                adapter,
                episode_count=reset_seeds.size,
                map_mode="lax_map",
            ).collect(
                reset_seeds=reset_seeds,
                probe_seeds=probe_seeds,
                action_tensor=actions,
            )
            vmap = ProbeBatchExecutor(
                adapter,
                episode_count=reset_seeds.size,
                map_mode="vmap",
            ).collect(
                reset_seeds=reset_seeds,
                probe_seeds=probe_seeds,
                action_tensor=actions,
            )

        self.assertDatasetExact(lax_map, vmap)

    def test_one_executor_reuses_traces_across_banks_and_replays_exactly(self) -> None:
        schema = _schema()
        environment = _SyntheticEnvironment()
        adapter = SimpleNamespace(schema=schema, environment=environment)
        episode_count = 4
        bank_a = {
            "reset_seeds": np.asarray([7, 19, 101, 4093], dtype=np.int64),
            "probe_seeds": np.asarray([11, 23, 103, 4099], dtype=np.int64),
            "action_tensor": _actions(
                episode_count=episode_count,
                horizon=schema.horizon,
            ),
        }
        bank_b = {
            "reset_seeds": np.asarray([29, 31, 107, 4127], dtype=np.int64),
            "probe_seeds": np.asarray([37, 41, 109, 4129], dtype=np.int64),
            "action_tensor": _actions(
                episode_count=episode_count,
                horizon=schema.horizon,
                reverse=True,
            ),
        }

        with jax.default_device(self.cpu_devices[0]):
            executor = ProbeBatchExecutor(
                adapter,
                episode_count=episode_count,
                map_mode="lax_map",
            )
            first_a = executor.collect(**bank_a)
            traces_after_first = (
                environment.reset_trace_count,
                environment.step_trace_count,
            )
            first_b = executor.collect(**bank_b)
            replay_a = executor.collect(**bank_a)

        self.assertGreater(traces_after_first[0], 0)
        self.assertGreater(traces_after_first[1], 0)
        self.assertEqual(
            (
                environment.reset_trace_count,
                environment.step_trace_count,
            ),
            traces_after_first,
            "same-shaped banks retraced the cached reset/scan executables",
        )
        self.assertDatasetExact(first_a, replay_a)
        self.assertNotEqual(first_a.digest, first_b.digest)
        self.assertFalse(np.array_equal(first_a.observation, first_b.observation))
        np.testing.assert_array_equal(first_b.reset_seeds, bank_b["reset_seeds"])
        np.testing.assert_array_equal(first_b.probe_seeds, bank_b["probe_seeds"])


if __name__ == "__main__":  # pragma: no cover - server dependency gate
    unittest.main()
