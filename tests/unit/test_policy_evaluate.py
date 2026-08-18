from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
from typing import NamedTuple
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from policy_learnware_v0.policy.evaluate import (  # noqa: E402
    FrozenPolicyEvaluationError,
    evaluate_frozen_policy_returns_batched,
    verify_compiled_policy_parity,
)


class _State(NamedTuple):
    obs: object
    reward: object
    done: object
    info: dict[str, object]


class _Environment:
    def reset(self, key):
        import jax
        import jax.numpy as jnp

        value = jax.random.key_data(key).reshape(-1)[-1]
        observation = jnp.asarray([value % 7], dtype=jnp.float32)
        return _State(
            observation,
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False),
            {"truncation": jnp.asarray(False)},
        )

    def step(self, state, action):
        import jax.numpy as jnp

        observation = state.obs + action
        return _State(
            observation,
            jnp.asarray(observation[0], dtype=jnp.float32),
            jnp.asarray(False),
            {"truncation": jnp.asarray(False)},
        )


class _Policy:
    def act(self, observation, key, *, deterministic=True):
        import jax
        import jax.numpy as jnp

        assert deterministic
        value = jax.random.uniform(key, (), minval=-0.1, maxval=0.1)
        return jnp.asarray([value], dtype=jnp.float32), jax.random.split(key, 2)[1]


class FrozenPolicyEvaluationTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("jax") is not None, "JAX is not installed")
    def test_batched_evaluation_is_repeatable_and_episode_ordered(self) -> None:
        kwargs = {
            "reset_seeds": [11, 12, 13],
            "policy_seeds": [21, 22, 23],
            "horizon": 4,
            "observation_dim": 1,
            "action_dim": 1,
        }
        first = evaluate_frozen_policy_returns_batched(
            _Policy(), _Environment(), **kwargs
        )
        second = evaluate_frozen_policy_returns_batched(
            _Policy(), _Environment(), **kwargs
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(np.all(np.isfinite(first)))

    @unittest.skipUnless(find_spec("jax") is not None, "JAX is not installed")
    def test_compiled_policy_path_matches_scalar_path(self) -> None:
        report = verify_compiled_policy_parity(
            _Policy(),
            np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
            np.asarray([0, 17], dtype=np.uint32),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        self.assertTrue(report.passed)
        self.assertTrue(report.next_keys_equal)

    def test_misaligned_seed_vectors_fail_closed(self) -> None:
        with self.assertRaises(FrozenPolicyEvaluationError):
            evaluate_frozen_policy_returns_batched(
                _Policy(),
                _Environment(),
                reset_seeds=[1],
                policy_seeds=[],
                horizon=4,
                observation_dim=1,
                action_dim=1,
            )


if __name__ == "__main__":
    unittest.main()
