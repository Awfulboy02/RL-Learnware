from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from policy_learnware_v0.v01.oracle import (
    OracleEvaluationError,
    aggregate_oracle_pair,
    aggregate_effect_point,
    paired_episode_effects,
    evaluator_contract_digest,
    validate_oracle_shard_payload,
)


class OracleV01Test(unittest.TestCase):
    def test_abs_transfer_gap_is_abs_of_mean(self) -> None:
        delta, gap = aggregate_effect_point([1.0, -0.5])
        self.assertEqual(delta, 0.25)
        self.assertEqual(gap, 0.25)
        self.assertNotEqual(gap, float(np.mean(np.abs([1.0, -0.5]))))

    def test_paired_rows_require_candidate_and_seed_alignment(self) -> None:
        nominal = [
            {
                "candidate_id": "c0",
                "episode_index": 0,
                "reset_seed": 11,
                "policy_seed": 12,
                "mean_step_return": 0.4,
            }
        ]
        shifted = [dict(nominal[0], mean_step_return=0.7)]
        np.testing.assert_allclose(paired_episode_effects(shifted, nominal), [0.3])
        with self.assertRaises(OracleEvaluationError):
            paired_episode_effects(
                [dict(shifted[0], policy_seed=99)], nominal
            )

    def test_typed_aggregate_uses_paired_bootstrap(self) -> None:
        digest = "a" * 64
        nominal_id = "v01v-" + "1" * 20
        shifted_id = "v01v-" + "2" * 20
        nominal = []
        shifted = []
        for index, value in enumerate((0.2, 0.4, 0.6)):
            base = {
                "task_private": "WalkerWalk",
                "candidate_id": "candidate",
                "episode_index": index,
                "reset_seed": index + 1,
                "policy_seed": index + 10,
                "instance_digest": digest,
                "bundle_digest": digest,
                "evaluator_contract_digest": digest,
            }
            nominal.append(dict(base, variant_id=nominal_id, mean_step_return=value))
            shifted.append(dict(base, variant_id=shifted_id, mean_step_return=value + 0.1))
        result = aggregate_oracle_pair(
            shifted,
            nominal,
            resamples=50,
            mean_seed=11,
            transfer_seed=12,
        )
        self.assertAlmostEqual(result.delta_return, 0.1)
        self.assertAlmostEqual(result.abs_transfer_gap, 0.1)

    def test_resume_payload_revalidates_every_frozen_binding(self) -> None:
        digest = "a" * 64
        variant_id = "v01v-" + "2" * 20
        record = SimpleNamespace(
            candidate_id="candidate",
            observation_dim=3,
            action_dim=2,
        )
        prepared = SimpleNamespace(
            record=record,
            metadata=SimpleNamespace(bundle_digest=digest),
        )
        adapter = SimpleNamespace(
            schema=SimpleNamespace(observation_dim=3, action_dim=2)
        )
        contract = evaluator_contract_digest(horizon=1000)
        episode = {
            "task_private": "WalkerWalk",
            "variant_id": variant_id,
            "candidate_id": "candidate",
            "episode_index": 0,
            "reset_seed": 7,
            "policy_seed": 8,
            "raw_episodic_sum": 500.0,
            "mean_step_return": 0.5,
            "instance_digest": digest,
            "bundle_digest": digest,
            "evaluator_contract_digest": contract,
        }
        payload = {
            "schema": "policy-learnware.v01-oracle-shard.v0",
            "task_private": "WalkerWalk",
            "variant_id": variant_id,
            "candidate_id": "candidate",
            "instance_digest": digest,
            "bundle_digest": digest,
            "evaluator_contract_digest": contract,
            "episodes": [episode],
        }
        validate_oracle_shard_payload(
            payload,
            prepared,
            adapter,
            task_private="WalkerWalk",
            variant_id=variant_id,
            instance_digest=digest,
            reset_seeds=[7],
            policy_seeds=[8],
            horizon=1000,
        )
        poisoned = {**payload, "episodes": [{**episode, "policy_seed": 9}]}
        with self.assertRaises(OracleEvaluationError):
            validate_oracle_shard_payload(
                poisoned,
                prepared,
                adapter,
                task_private="WalkerWalk",
                variant_id=variant_id,
                instance_digest=digest,
                reset_seeds=[7],
                policy_seeds=[8],
                horizon=1000,
            )


if __name__ == "__main__":
    unittest.main()
