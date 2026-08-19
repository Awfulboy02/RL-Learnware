from __future__ import annotations

import unittest

from policy_learnware_v0.v01.plans import build_pair_plan, verify_pair_plan
from policy_learnware_v0.v01.cli import _audit_subset


class PairPlanV01Test(unittest.TestCase):
    def test_public_plan_contains_no_task_or_factor(self) -> None:
        variants = [
            {"task": "WalkerWalk", "factor": factor, "variant_id": f"v{index}"}
            for index, factor in enumerate((0.5, 1.0, 2.0))
        ]
        plan = build_pair_plan(
            variants,
            banks=2,
            gate_prefix=1,
            routing_prefix=2,
            within_bank_pairs=((0, 1),),
        )
        self.assertEqual(verify_pair_plan(plan), plan["plan_digest"])
        encoded = repr(plan).lower()
        self.assertNotIn("walkerwalk", encoded)
        self.assertNotIn("factor", encoded)
        self.assertNotIn("task", encoded)

    def test_private_raw_audit_subset_is_exact_per_task(self) -> None:
        variants = [
            {
                "task": task,
                "factor": factor,
                "variant_id": f"{task}-{factor}",
            }
            for task in ("WalkerWalk", "FingerTurnEasy")
            for factor in (0.5, 0.75, 1.0, 1.5, 2.0)
        ]
        plan = build_pair_plan(
            variants,
            banks=2,
            gate_prefix=16,
            routing_prefix=64,
            within_bank_pairs=((0, 1),),
        )
        audit = _audit_subset(plan, variants)["raw_numeric_subset"]
        self.assertEqual(len(audit["within"]), 2)
        self.assertEqual(len(audit["between"]), 2)
        self.assertEqual(len(audit["routing"]), 2)
        for row in audit["within"]:
            self.assertEqual(row["left_variant_id"], f"{row['task_private']}-1.0")
            self.assertEqual((row["left_bank"], row["right_bank"]), (0, 1))
        for row in audit["between"]:
            self.assertEqual(row["left_variant_id"], f"{row['task_private']}-1.0")
            self.assertEqual(row["right_variant_id"], f"{row['task_private']}-2.0")
        for row in audit["routing"]:
            self.assertEqual(row["variant_id"], f"{row['task_private']}-2.0")
            self.assertEqual(row["prefix"], 64)


if __name__ == "__main__":
    unittest.main()
