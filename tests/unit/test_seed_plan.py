from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from policy_learnware_v0.probe.seed_plan import (
    EpisodeSeeds,
    SeedPlan,
    assert_seed_records_disjoint,
)


class SeedPlanTest(unittest.TestCase):
    def test_derivation_is_repeatable_and_domain_separated(self) -> None:
        plan = SeedPlan(20260811)
        record = plan.episode("encoder_train", 2, 17)
        self.assertEqual(record, plan.episode("encoder_train", 2, 17))
        self.assertNotEqual(record.reset_seed, record.probe_seed)
        self.assertNotEqual(
            record.reset_seed,
            plan.episode("encoder_validation", 2, 17).reset_seed,
        )
        self.assertNotEqual(
            record.probe_seed, plan.episode("encoder_train", 3, 17).probe_seed
        )

    def test_all_research_namespaces_have_no_observed_overlap(self) -> None:
        plan = SeedPlan(9)
        records = {
            split: plan.episodes(split, 0, range(100))
            for split in (
                "encoder_train",
                "encoder_validation",
                "kernel_calibration",
                "separability_calibration",
                "source_taskspec",
                "target_query",
                "championization",
                "final_return",
            )
        }
        assert_seed_records_disjoint(records)

    def test_overlap_checker_rejects_reused_seed_pair(self) -> None:
        first = EpisodeSeeds("encoder_train", 0, 0, 10, 11)
        reused = EpisodeSeeds("target_query", 0, 99, 10, 11)
        with self.assertRaises(ValueError):
            assert_seed_records_disjoint({"a": [first], "b": [reused]})

    def test_unknown_namespace_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SeedPlan(1).episode("ad_hoc", 0, 0)

    def test_target_query_banks_are_explicit_and_disjoint(self) -> None:
        plan = SeedPlan(20260811)
        bank_zero = plan.episodes("target_query", 1, range(8), bank_index=0)
        bank_one = plan.episodes("target_query", 1, range(8), bank_index=1)
        self.assertTrue(all(record.bank_index == 1 for record in bank_one))
        self.assertTrue(
            {record.collision_key for record in bank_zero}.isdisjoint(
                record.collision_key for record in bank_one
            )
        )
        with self.assertRaises(ValueError):
            plan.episode("source_taskspec", 0, 0, bank_index=1)


if __name__ == "__main__":
    unittest.main()
