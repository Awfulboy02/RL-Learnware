from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.rkme.gaussian import (  # noqa: E402
    GaussianKernel,
    calibrate_bandwidth,
)


class GaussianKernelTest(unittest.TestCase):
    def test_gram_invariants_and_psd(self) -> None:
        rng = np.random.default_rng(3)
        points = rng.normal(size=(20, 4))
        gram = GaussianKernel(1.7).gram(points)
        np.testing.assert_allclose(gram, gram.T, atol=1e-14)
        np.testing.assert_array_equal(np.diag(gram), np.ones(20))
        self.assertGreaterEqual(float(np.linalg.eigvalsh(gram).min()), -1e-10)

    def test_source_balanced_calibration_is_reproducible(self) -> None:
        events = {
            "a": SimpleNamespace(
                points=np.array([[0.0], [1.0], [2.0]]),
                episode_offsets=np.array([0, 1, 3]),
            ),
            "b": SimpleNamespace(
                points=np.array([[10.0], [11.0], [12.0], [13.0]]),
                episode_offsets=np.array([0, 2, 4]),
            ),
        }
        first = calibrate_bandwidth(events, calibration_pairs=100, seed=19)
        second = calibrate_bandwidth(events, calibration_pairs=100, seed=19)
        self.assertGreater(first, 0.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
