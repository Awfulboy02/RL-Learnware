from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from policy_learnware_v0.v04a.ebpr import (
    EBPRError,
    EBPRFixedProbe,
    EBPR_HYBRID_METHOD_ID,
    EBPR_METHOD_ID,
    TransitionEpisode,
)


TYPE_ALPHA = "source-alpha"
TYPE_BETA = "source-beta"
POLICY_ALPHA = "policy-z-paired-with-alpha"
POLICY_BETA = "policy-a-paired-with-beta"


def _episode(type_id: str, seed: int, transitions: int = 72) -> TransitionEpisode:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(transitions, 2))
    action = rng.normal(size=(transitions, 1))
    if type_id == TYPE_ALPHA:
        delta = np.column_stack(
            (
                0.85 * np.tanh(1.2 * state[:, 0] + 0.45 * action[:, 0]),
                0.35 * state[:, 1] - 0.30 * action[:, 0],
            )
        )
    elif type_id == TYPE_BETA:
        delta = np.column_stack(
            (
                -0.75 * np.tanh(1.1 * state[:, 0] - 0.35 * action[:, 0]),
                -0.25 * state[:, 1] + 0.55 * action[:, 0],
            )
        )
    else:  # pragma: no cover - fixture misuse
        raise AssertionError(type_id)
    delta += rng.normal(scale=0.025, size=delta.shape)
    return TransitionEpisode.from_delta(state, action, delta)


@pytest.fixture(scope="module")
def fitted_model() -> EBPRFixedProbe:
    train, validation = _source_banks()
    return EBPRFixedProbe.fit(
        train,
        validation,
        {
            TYPE_ALPHA: POLICY_ALPHA,
            TYPE_BETA: POLICY_BETA,
        },
        hidden_dim=28,
        ridge=1.0e-3,
        feature_seed=20240828,
        variance_floor_candidates=(1.0e-6, 1.0e-4, 1.0e-2),
        temperature_candidates=(0.001, 0.01, 0.1, 1.0),
        tie_token="toy-frozen-tie-token",
    )


def _source_banks() -> tuple[
    dict[str, tuple[TransitionEpisode, ...]],
    dict[str, tuple[TransitionEpisode, ...]],
]:
    train = {
        TYPE_ALPHA: tuple(_episode(TYPE_ALPHA, seed) for seed in range(10, 14)),
        TYPE_BETA: tuple(_episode(TYPE_BETA, seed) for seed in range(20, 24)),
    }
    validation = {
        TYPE_ALPHA: tuple(_episode(TYPE_ALPHA, seed) for seed in range(110, 113)),
        TYPE_BETA: tuple(_episode(TYPE_BETA, seed) for seed in range(120, 123)),
    }
    return train, validation


def test_conditional_likelihood_recovers_type_and_paired_policy(
    fitted_model: EBPRFixedProbe,
) -> None:
    evidence = (_episode(TYPE_BETA, 999, transitions=90),)

    likelihoods = fitted_model.log_likelihoods(evidence)
    posterior = fitted_model.posterior(evidence)
    selection = fitted_model.select_map(evidence)

    assert likelihoods[TYPE_BETA] > likelihoods[TYPE_ALPHA]
    assert posterior[TYPE_BETA] > 0.99
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert selection.method_id == EBPR_METHOD_ID
    assert selection.selected_type_id == TYPE_BETA
    # The policy mapping is explicit and deliberately has the opposite lexical
    # ordering from the type IDs; MAP must follow the paired mapping.
    assert selection.selected_policy_id == POLICY_BETA
    assert math.isfinite(selection.target_predictive_nll)
    assert (
        fitted_model.calibration["residual_variance_fit_role"]
        == "source_train_residual_only"
    )
    assert (
        fitted_model.calibration["hyperparameter_calibration_role"]
        == "source_validation_only"
    )
    assert fitted_model.calibration["validation_episode_count"] == 6


def _normalized_residual_mse(
    model: EBPRFixedProbe,
    type_id: str,
    episodes: tuple[TransitionEpisode, ...],
) -> np.ndarray:
    residuals = []
    for episode in episodes:
        prediction = model.predict_delta(type_id, episode.state, episode.action)
        residuals.append(
            (episode.delta[:, model.y_valid_mask] - prediction[:, model.y_valid_mask])
            / model.y_scale[model.y_valid_mask]
        )
    joined = np.concatenate(residuals, axis=0)
    return np.mean(joined * joined, axis=0)


def test_residual_variance_is_fit_from_source_train_not_validation() -> None:
    train, clean_validation = _source_banks()
    rng = np.random.default_rng(73)
    noisy_validation: dict[str, tuple[TransitionEpisode, ...]] = {}
    for type_id, episodes in clean_validation.items():
        noisy_validation[type_id] = tuple(
            TransitionEpisode.from_delta(
                episode.state,
                episode.action,
                episode.delta + rng.normal(scale=1.5, size=episode.delta.shape),
            )
            for episode in episodes
        )

    model = EBPRFixedProbe.fit(
        train,
        noisy_validation,
        {TYPE_ALPHA: POLICY_ALPHA, TYPE_BETA: POLICY_BETA},
        hidden_dim=20,
        ridge=1.0e-3,
        feature_seed=17,
        variance_floor_candidates=(1.0e-12,),
        temperature_candidates=(0.1,),
    )

    for type_id in model.type_ids:
        train_mse = _normalized_residual_mse(model, type_id, train[type_id])
        validation_mse = _normalized_residual_mse(
            model, type_id, noisy_validation[type_id]
        )
        assert model.residual_variance_by_type[type_id] == pytest.approx(
            np.maximum(train_mse, model.variance_floor), rel=1.0e-10, abs=1.0e-12
        )
        # The deliberately corrupted validation residuals are orders of
        # magnitude larger.  The old implementation would equal this value.
        assert np.all(validation_mse > 100.0 * model.residual_variance_by_type[type_id])


def test_hybrid_expected_utility_is_explicit_and_operational(
    fitted_model: EBPRFixedProbe,
) -> None:
    evidence = _episode(TYPE_BETA, 1001)
    primary = fitted_model.select_map(evidence)
    hybrid = fitted_model.select_hybrid(
        evidence,
        {
            TYPE_ALPHA: {POLICY_ALPHA: 5.0, POLICY_BETA: -1.0},
            TYPE_BETA: {POLICY_ALPHA: 4.0, POLICY_BETA: 0.0},
        },
    )

    assert primary.selected_policy_id == POLICY_BETA
    assert hybrid.method_id == EBPR_HYBRID_METHOD_ID
    assert hybrid.is_hybrid
    assert hybrid.selected_type_id is None
    assert hybrid.selected_policy_id == POLICY_ALPHA
    assert hybrid.expected_utility is not None
    assert hybrid.expected_utility[POLICY_ALPHA] > hybrid.expected_utility[POLICY_BETA]
    assert hybrid.to_dict()["decision_semantics"] == "INSPIRED_HYBRID_EXPECTED_UTILITY"


def test_log_space_posterior_and_predictive_nll_remain_finite(
    fitted_model: EBPRFixedProbe,
) -> None:
    ordinary = _episode(TYPE_ALPHA, 2001, transitions=48)
    # Repeating evidence produces likelihood magnitudes at which a direct
    # product of Gaussian densities underflows, while log-space scoring stays
    # normalized and finite.
    repeated = tuple(ordinary for _ in range(80))
    posterior = fitted_model.posterior(repeated)

    assert all(math.isfinite(value) for value in posterior.values())
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior[TYPE_ALPHA] > 0.999
    assert math.isfinite(fitted_model.target_predictive_nll(repeated))

    extreme_state = np.full((4, 2), 1.0e120)
    extreme_action = np.full((4, 1), -1.0e120)
    extreme = TransitionEpisode.from_delta(
        extreme_state, extreme_action, np.zeros_like(extreme_state)
    )
    extreme_posterior = fitted_model.posterior(extreme)
    assert all(math.isfinite(value) for value in extreme_posterior.values())
    assert sum(extreme_posterior.values()) == pytest.approx(1.0)
    assert math.isfinite(fitted_model.target_predictive_nll(extreme))


def test_predictive_nll_is_raw_mixture_while_posterior_is_tempered(
    fitted_model: EBPRFixedProbe,
) -> None:
    evidence = (_episode(TYPE_BETA, 2112, transitions=37),)
    prior = {TYPE_ALPHA: 0.25, TYPE_BETA: 0.75}
    scores = fitted_model.log_likelihoods(evidence)
    transition_count = sum(episode.transition_count for episode in evidence)

    raw_logits = np.asarray(
        [
            math.log(prior[type_id]) + scores[type_id]
            for type_id in fitted_model.type_ids
        ]
    )
    raw_log_mixture = float(np.logaddexp.reduce(raw_logits))
    expected_raw_nll = -raw_log_mixture / transition_count
    assert fitted_model.target_predictive_nll(evidence, prior) == pytest.approx(
        expected_raw_nll
    )

    tempered_logits = np.asarray(
        [
            math.log(prior[type_id])
            + fitted_model.posterior_temperature * scores[type_id]
            for type_id in fitted_model.type_ids
        ]
    )
    tempered_log_normalizer = float(np.logaddexp.reduce(tempered_logits))
    expected_posterior = np.exp(tempered_logits - tempered_log_normalizer)
    posterior = fitted_model.posterior(evidence, prior)
    assert np.asarray(
        [posterior[key] for key in fitted_model.type_ids]
    ) == pytest.approx(expected_posterior)
    # This is the exact old (incorrect) diagnostic and must remain distinct.
    old_tempered_nll = -tempered_log_normalizer / transition_count
    assert not math.isclose(
        fitted_model.target_predictive_nll(evidence, prior),
        old_tempered_nll,
        rel_tol=1.0e-6,
        abs_tol=1.0e-9,
    )


def test_checkpoint_round_trip_preserves_predictions_and_mapping(
    fitted_model: EBPRFixedProbe, tmp_path
) -> None:
    evidence = _episode(TYPE_ALPHA, 3001)
    checkpoint = tmp_path / "ebpr-source-model.json"
    fitted_model.save(checkpoint)
    restored = EBPRFixedProbe.load(checkpoint)

    assert restored.to_dict() == fitted_model.to_dict()
    assert restored.paired_policy_by_type == {
        TYPE_ALPHA: POLICY_ALPHA,
        TYPE_BETA: POLICY_BETA,
    }
    assert restored.posterior(evidence) == pytest.approx(
        fitted_model.posterior(evidence)
    )
    assert restored.select_map(evidence).selected_policy_id == POLICY_ALPHA
    assert np.allclose(
        restored.predict_delta(TYPE_ALPHA, evidence.state, evidence.action),
        fitted_model.predict_delta(TYPE_ALPHA, evidence.state, evidence.action),
    )


def test_evidence_contract_rejects_reward_like_or_invalid_inputs(
    fitted_model: EBPRFixedProbe,
) -> None:
    with pytest.raises(EBPRError, match="prior probabilities"):
        fitted_model.posterior(
            _episode(TYPE_ALPHA, 4001),
            {TYPE_ALPHA: 1.0, TYPE_BETA: 0.0},
        )
    with pytest.raises(EBPRError, match="utility rows"):
        fitted_model.select_hybrid(
            _episode(TYPE_ALPHA, 4002),
            {TYPE_ALPHA: {POLICY_ALPHA: 1.0}},
        )


def test_map_type_tie_is_broken_by_paired_opaque_policy_id(
    fitted_model: EBPRFixedProbe,
) -> None:
    belief = {TYPE_ALPHA: 0.5, TYPE_BETA: 0.5}
    selected_type = fitted_model._argmax_with_tie(
        belief, tie_identity_by_key=fitted_model.paired_policy_by_type
    )
    expected = min(
        belief,
        key=lambda type_id: hashlib.sha256(
            (
                fitted_model.tie_token + fitted_model.paired_policy_by_type[type_id]
            ).encode("utf-8")
        ).hexdigest(),
    )
    assert selected_type == expected
