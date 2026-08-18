from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from policy_learnware_v0.rkme.empirical import build_empirical_kme  # noqa: E402
from policy_learnware_v0.rkme.gaussian import GaussianKernel  # noqa: E402
from policy_learnware_v0.rkme.reducer import (  # noqa: E402
    ReducedRKME,
    ReducerConfig,
    reduce_kme,
    solve_beta,
)


class ReducerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = GaussianKernel(1.2)
        self.empirical = build_empirical_kme(
            SimpleNamespace(
                points=np.array([[0.0], [0.5], [2.0], [3.0]]),
                episode_offsets=np.array([0, 2, 4]),
            ),
            self.kernel,
            protocol_id="protocol",
            dataset_digest="dataset",
        )

    def test_closed_form_beta_and_error_is_norm(self) -> None:
        reduced = reduce_kme(
            self.empirical,
            ReducerConfig(
                support_budget=2,
                support_steps=0,
                kmeans_steps=2,
                ridge=1e-6,
            ),
        )
        beta, kuu, kuz = solve_beta(
            reduced.supports,
            self.empirical,
            self.kernel,
            ridge=1e-6,
            pinv_rcond=1e-8,
        )
        np.testing.assert_allclose(reduced.beta, beta)
        residual_squared = (
            self.empirical.norm2
            - 2 * beta @ kuz @ self.empirical.weights
            + beta @ kuu @ beta
        )
        self.assertAlmostEqual(reduced.reduction_error, np.sqrt(max(residual_squared, 0.0)))
        self.assertEqual(reduced.beta.shape, (2,))

    def test_npz_round_trip(self) -> None:
        reduced = reduce_kme(
            self.empirical,
            ReducerConfig(support_budget=2, support_steps=0, kmeans_steps=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_rkme.npz"
            reduced.save_npz(path)
            loaded = ReducedRKME.load_npz(path)
        np.testing.assert_array_equal(loaded.supports, reduced.supports)
        np.testing.assert_array_equal(loaded.beta, reduced.beta)
        self.assertEqual(loaded.protocol_id, "protocol")
        self.assertEqual(loaded.reduction_error, reduced.reduction_error)

    @unittest.skipUnless(importlib.util.find_spec("jax") is not None, "JAX unavailable")
    def test_jax_optimizer_path_is_finite(self) -> None:
        reduced = reduce_kme(
            self.empirical,
            ReducerConfig(
                support_budget=2,
                support_steps=2,
                kmeans_steps=2,
                optimizer_backend="jax",
            ),
        )
        self.assertTrue(np.all(np.isfinite(reduced.supports)))
        self.assertTrue(np.all(np.isfinite(reduced.beta)))
        self.assertEqual(len(reduced.objective_trace), 3)


if __name__ == "__main__":
    unittest.main()
