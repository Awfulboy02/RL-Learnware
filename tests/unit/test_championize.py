from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.policy.championize import (  # noqa: E402
    CandidateEvaluation,
    championize,
)


def _candidate(digest: str, returns: tuple[float, ...], *, task: str = "task") -> CandidateEvaluation:
    return CandidateEvaluation(
        task=task,
        algorithm="ppo",
        training_seed=0,
        outer_iteration=6,
        environment_steps=5_898_240,
        bundle_dir=Path("/private/bundle") / digest,
        bundle_digest=digest,
        episode_returns=returns,
    )


class ChampionizationTest(unittest.TestCase):
    def test_rule_is_mean_then_std_then_digest(self) -> None:
        candidates = [
            _candidate("c", (2.0, 0.0)),  # mean 1, std 1
            _candidate("b", (1.0, 1.0)),  # mean 1, std 0
            _candidate("a", (1.0, 1.0)),  # exact tie, digest wins
            _candidate("z", (0.0, 0.0)),
        ]
        result = championize(candidates, checkpoint_outer=6, expected_candidates_per_task=4)
        self.assertEqual(result.champions[0].selected.candidate.bundle_digest, "a")
        self.assertEqual(
            [item.candidate.bundle_digest for item in result.champions[0].ranking],
            ["a", "b", "c", "z"],
        )

    def test_invalid_candidate_is_rejected_before_selection(self) -> None:
        invalid = CandidateEvaluation(
            **{
                **_candidate("bad", (100.0,)).__dict__,
                "parity_passed": False,
            }
        )
        result = championize(
            [_candidate("good", (1.0,)), invalid],
            checkpoint_outer=6,
            expected_candidates_per_task=1,
        )
        self.assertEqual(result.champions[0].selected.candidate.bundle_digest, "good")
        self.assertEqual(result.rejected[0].reason, "golden parity failed")

    def test_fixed_budget_is_enforced(self) -> None:
        wrong_outer = CandidateEvaluation(
            **{**_candidate("bad", (1.0,)).__dict__, "outer_iteration": 7}
        )
        with self.assertRaisesRegex(ValueError, "no valid candidate|coverage"):
            championize(
                [wrong_outer],
                checkpoint_outer=6,
                expected_tasks=["task"],
            )


if __name__ == "__main__":
    unittest.main()
