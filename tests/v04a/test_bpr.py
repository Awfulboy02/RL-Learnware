from __future__ import annotations

import hashlib

import numpy as np
import pytest

from policy_learnware_v0.v04a.bpr import (
    BPRGaussianModel,
    BPRModelError,
    summarize_episode,
)
from policy_learnware_v0.v04a.protocol import break_tie


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _toy_model() -> BPRGaussianModel:
    train = {
        "type-cold": np.asarray(
            [
                [-2.2, -2.0, -1.8, -2.1],
                [-1.9, -2.1, -2.2, -1.8],
                [-2.0, -1.8, -2.1, -2.2],
                [-1.8, -2.2, -1.9, -2.0],
            ]
        ),
        "type-hot": np.asarray(
            [
                [2.2, 2.0, 1.8, 2.1],
                [1.9, 2.1, 2.2, 1.8],
                [2.0, 1.8, 2.1, 2.2],
                [1.8, 2.2, 1.9, 2.0],
            ]
        ),
    }
    validation = {
        "type-cold": np.asarray(
            [[-2.05, -1.95, -2.0, -2.1], [-1.9, -2.0, -2.1, -1.95]]
        ),
        "type-hot": np.asarray([[2.05, 1.95, 2.0, 2.1], [1.9, 2.0, 2.1, 1.95]]),
    }
    utility = {
        "type-cold": {"policy-cold": 1.0, "policy-hot": 0.0},
        "type-hot": {"policy-cold": 0.0, "policy-hot": 1.0},
    }
    return BPRGaussianModel.fit(
        train,
        validation,
        utility,
        config_digest=_d("toy-config"),
        protocol_id="fixed-probe-64-v1",
        lambda_grid=(1.0, 0.0, 0.5),
        variance_floor_grid=(1e-2, 1e-5),
        temperature_grid=(2.0, 0.5, 1.0),
    )


def test_episode_summary_is_reward_free_mean_std_delta_action() -> None:
    observation = np.asarray([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    next_observation = np.asarray([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])
    action = np.asarray([[-1.0], [0.0], [1.0]])
    actual = summarize_episode(observation, action, next_observation)
    delta = next_observation - observation
    expected = np.concatenate(
        (
            np.mean(delta, axis=0),
            np.std(delta, axis=0),
            np.mean(action, axis=0),
            np.std(action, axis=0),
        )
    )
    np.testing.assert_allclose(actual, expected)


def test_posterior_expected_utility_and_target_predictive_nll() -> None:
    model = _toy_model()
    cold_evidence = np.asarray([[-2.0, -2.0, -2.0, -2.0], [-2.1, -1.9, -2.0, -2.05]])
    hot_evidence = -cold_evidence

    cold_posterior = model.posterior_dict(cold_evidence)
    hot_posterior = model.posterior_dict(hot_evidence)
    assert cold_posterior["type-cold"] > 0.999
    assert hot_posterior["type-hot"] > 0.999
    assert model.map_type(cold_evidence) == "type-cold"
    assert model.select_candidate(cold_evidence) == "policy-cold"
    assert model.select_candidate(hot_evidence) == "policy-hot"
    assert np.isfinite(model.target_predictive_nll(cold_evidence))
    assert model.validation_nll >= 0.0
    assert model.shrinkage in {0.0, 0.5, 1.0}
    assert model.variance_floor in {1e-5, 1e-2}
    assert model.temperature in {0.5, 1.0, 2.0}


def test_model_serialization_preserves_scores_and_source_pooled_normalizer() -> None:
    model = _toy_model()
    restored = BPRGaussianModel.from_dict(model.to_dict())
    evidence = np.asarray([[2.0, 2.0, 2.0, 2.0]])

    np.testing.assert_allclose(restored.posterior(evidence), model.posterior(evidence))
    assert restored.utility_scores(evidence) == model.utility_scores(evidence)
    assert restored.model_digest == model.model_digest
    # The symmetric toy source pool is centered at zero; this normalizer was
    # fitted from train only, not validation or target evidence.
    np.testing.assert_allclose(model.pooled_mean, np.zeros(4), atol=1e-15)
    assert np.all(model.pooled_scale > 0.0)


def test_extreme_finite_evidence_has_finite_posterior_and_nll() -> None:
    model = _toy_model()
    evidence = np.full((1, model.feature_count), 1.0e200, dtype=np.float64)

    posterior = model.posterior(evidence)
    assert np.all(np.isfinite(posterior))
    assert np.isclose(np.sum(posterior), 1.0, rtol=0.0, atol=1.0e-12)
    assert np.isfinite(model.target_predictive_nll(evidence))


def test_checkpoint_requires_explicit_source_fit_roles() -> None:
    payload = _toy_model().to_dict()
    payload.pop("normalizer_fit_role")
    with pytest.raises(BPRModelError, match="checkpoint keys mismatch"):
        BPRGaussianModel.from_dict(payload)


def test_shrinkage_target_is_pooled_within_type_variance() -> None:
    model = BPRGaussianModel.fit(
        {
            "t0": np.asarray([[-1.0], [-1.0], [-1.0]]),
            "t1": np.asarray([[1.0], [1.0], [1.0]]),
        },
        {"t0": np.asarray([[-1.0]]), "t1": np.asarray([[1.0]])},
        {
            "t0": {"p0": 1.0, "p1": 0.0},
            "t1": {"p0": 0.0, "p1": 1.0},
        },
        config_digest=_d("within-type-shrinkage"),
        protocol_id="fixed-probe-64-v1",
        lambda_grid=(0.0,),
        variance_floor_grid=(1.0e-4,),
        temperature_grid=(1.0,),
    )

    np.testing.assert_allclose(model.type_variances, 1.0e-4)


def test_exact_utility_tie_uses_canonical_candidate_rule() -> None:
    train = {
        "t0": np.asarray([[-1.0], [-0.9], [-1.1]]),
        "t1": np.asarray([[1.0], [0.9], [1.1]]),
    }
    validation = {"t0": np.asarray([[-1.0]]), "t1": np.asarray([[1.0]])}
    utility = {
        "t0": {"p0": 0.5, "p1": 0.5},
        "t1": {"p0": 0.5, "p1": 0.5},
    }
    config_digest = _d("tie-config")
    model = BPRGaussianModel.fit(
        train,
        validation,
        utility,
        config_digest=config_digest,
        protocol_id="fixed-probe-64-v1",
        lambda_grid=(0.5,),
        variance_floor_grid=(1e-4,),
        temperature_grid=(1.0,),
    )
    assert model.select_candidate(np.asarray([[0.0]])) == break_tie(
        config_digest, ("p0", "p1")
    )
