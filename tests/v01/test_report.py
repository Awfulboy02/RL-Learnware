from __future__ import annotations

import unittest

from policy_learnware_v0.v01.report import decide_v01


class ReportV01Test(unittest.TestCase):
    def test_pre_registered_decision_table(self) -> None:
        passed = {"passed": True}
        failed = {"passed": False}
        self.assertEqual(
            decide_v01(
                gate_0=failed,
                gate_a=passed,
                gate_b=passed,
                gate_d=passed,
                recompute_audit=passed,
            ).code,
            "BLOCKED_ENGINEERING",
        )
        self.assertEqual(
            decide_v01(
                gate_0=passed,
                gate_a=failed,
                gate_b=passed,
                gate_d=passed,
                recompute_audit=passed,
            ).code,
            "NO_GO_CURRENT_POOL_SHIFT",
        )
        self.assertEqual(
            decide_v01(
                gate_0=passed,
                gate_a=passed,
                gate_b=failed,
                gate_d=passed,
                recompute_audit=passed,
            ).code,
            "GO_PROBLEM_NO_GO_TASKSPEC",
        )
        self.assertEqual(
            decide_v01(
                gate_0=passed,
                gate_a=passed,
                gate_b=passed,
                gate_d=passed,
                recompute_audit=passed,
            ).code,
            "GO_V02_TRANSFERSPEC",
        )


if __name__ == "__main__":
    unittest.main()
