from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.smoke import run_logic_smoke  # noqa: E402


class LogicSmokeTest(unittest.TestCase):
    def test_training_free_end_to_end_logic(self) -> None:
        result = run_logic_smoke()
        self.assertTrue(result.passed)
        self.assertEqual(result.checks["retrieval"]["retrieval_correct"], 6)
        self.assertEqual(
            result.checks["deployment"]["incompatible_status"],
            "incompatible_native_schema",
        )


if __name__ == "__main__":
    unittest.main()
