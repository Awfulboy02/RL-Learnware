from __future__ import annotations

import math

import numpy as np
import pytest

from policy_learnware_v0.v01.statistics import (
    UNDEFINED_SPEARMAN_REASON,
    centered_bootstrap_p_value,
    derive_bootstrap_seed,
    empirical_competence_set,
    holm_bonferroni,
    independent_mean_difference_bootstrap,
    independent_sensitivity_difference_bootstrap,
    mean_bootstrap,
    nested_gate_c_spearman,
    paired_transfer_bootstrap,
    spearman_correlation,
    top1_bootstrap_probabilities,
)


def test_bootstrap_seed_is_stable_and_tuple_unambiguous() -> None:
    assert derive_bootstrap_seed("run", "task", "material") == derive_bootstrap_seed(
        "run", "task", "material"
    )
    assert derive_bootstrap_seed("ab", "c") != derive_bootstrap_seed("a", "bc")


def test_paired_transfer_uses_absolute_mean_not_mean_absolute_difference() -> None:
    result = paired_transfer_bootstrap(
        shifted_returns=[1.0, -1.0],
        nominal_returns=[0.0, 0.0],
        resamples=200,
        seed=7,
    )
    assert result.delta.observed == 0.0
    assert result.abs_gap == 0.0
    assert np.mean(np.abs(np.array([1.0, -1.0]))) == 1.0
    assert result.delta.resampling_contract == "paired_episode_indices_within_candidate"


def test_mean_bootstrap_supports_oracle_mean_return_interval() -> None:
    result = mean_bootstrap([0.1, 0.2, 0.3], resamples=100, seed=12)
    assert result.observed == pytest.approx(0.2)
    assert result.interval.low <= result.observed <= result.interval.high
    assert result.resampling_contract == "episode_indices_within_candidate_variant"


def test_paired_bootstrap_is_deterministic_and_rejects_unpaired_shapes() -> None:
    kwargs = dict(
        shifted_returns=[0.5, 0.7, 0.8, 0.9],
        nominal_returns=[0.1, 0.2, 0.4, 0.2],
        resamples=128,
        seed=19,
    )
    first = paired_transfer_bootstrap(**kwargs)
    second = paired_transfer_bootstrap(**kwargs)
    np.testing.assert_array_equal(first.delta.replicates, second.delta.replicates)
    assert first.delta.observed == pytest.approx(0.5)
    with pytest.raises(ValueError, match="identical shape"):
        paired_transfer_bootstrap([1.0, 2.0], [1.0], resamples=10, seed=1)


def test_centered_bootstrap_p_value_matches_preregistered_formula() -> None:
    replicates = np.array([1.0, 2.0, 3.0, 4.0])
    # observed=2: centered deviations are [1,0,1,2], one is >= 2.
    assert centered_bootstrap_p_value(2.0, replicates) == pytest.approx(2 / 5)


def test_independent_candidate_bootstraps_declare_contract() -> None:
    mean_gap = independent_mean_difference_bootstrap(
        [4.0, 5.0, 6.0], [1.0, 2.0, 3.0], resamples=100, seed=3
    )
    assert mean_gap.observed == pytest.approx(3.0)
    assert mean_gap.resampling_contract == "independent_episode_indices_across_candidates"

    sensitivity = independent_sensitivity_difference_bootstrap(
        [2.0, 2.0, 2.0], [-0.5, -0.5, -0.5], resamples=100, seed=4
    )
    assert sensitivity.observed == pytest.approx(1.5)
    assert "independent_candidates" in sensitivity.resampling_contract


def test_holm_bonferroni_records_complete_order_and_monotone_adjustment() -> None:
    adjusted = holm_bonferroni({"h3": 0.04, "h1": 0.01, "h2": 0.03})
    assert adjusted["h1"].adjusted_p_value == pytest.approx(0.03)
    assert adjusted["h2"].adjusted_p_value == pytest.approx(0.06)
    assert adjusted["h3"].adjusted_p_value == pytest.approx(0.06)
    assert {value.correction_order for value in adjusted.values()} == {1, 2, 3}
    assert {value.family_size for value in adjusted.values()} == {3}


def test_holm_ties_use_lexical_hypothesis_id() -> None:
    adjusted = holm_bonferroni({"z": 0.01, "a": 0.01})
    assert adjusted["a"].correction_order == 1
    assert adjusted["z"].correction_order == 2


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan])
def test_holm_rejects_invalid_p_values(value: float) -> None:
    with pytest.raises(ValueError):
        holm_bonferroni({"h": value})


def test_competence_set_uses_point_estimate_and_inclusive_threshold() -> None:
    result = empirical_competence_set({"best": 1.0, "edge": 0.8, "low": 0.79}, alpha=0.8)
    assert result.candidate_ids == ("best", "edge")
    assert result.threshold == pytest.approx(0.8)
    assert result.sufficient


def test_spearman_handles_ties_without_scipy() -> None:
    result = spearman_correlation([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0])
    assert result.rho == pytest.approx(1.0)
    assert result.reason is None


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
        ([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]),
        ([1.0, 2.0, math.nan, 4.0], [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_spearman_undefined_is_null_with_fixed_reason(left, right) -> None:
    result = spearman_correlation(left, right)
    assert result.rho is None
    assert result.reason == UNDEFINED_SPEARMAN_REASON


def test_top1_bootstrap_is_deterministic_and_independent_by_candidate() -> None:
    values = {"b": [0.0, 0.0, 0.0], "a": [1.0, 1.0, 1.0]}
    first = top1_bootstrap_probabilities(values, resamples=100, seed=11)
    second = top1_bootstrap_probabilities(values, resamples=100, seed=11)
    assert first == second
    assert first.empirical_winner == "a"
    assert first.probability("a") == 1.0
    assert sum(first.probabilities.values()) == pytest.approx(1.0)


def test_top1_lexical_tie_break() -> None:
    result = top1_bootstrap_probabilities(
        {"z": [1.0, 1.0], "a": [1.0, 1.0]}, resamples=20, seed=1
    )
    assert result.empirical_winner == "a"
    assert result.probability("a") == 1.0


def test_nested_gate_c_keeps_severity_grid_fixed_and_is_reproducible() -> None:
    # Four severity points, five paired probe banks, two competent candidates,
    # six paired transfer episodes.  Both summaries increase monotonically.
    probe = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    transfer = np.empty((4, 2, 6), dtype=np.float64)
    for severity in range(4):
        transfer[severity, 0] = severity + 1
        transfer[severity, 1] = severity + 1.5
    first = nested_gate_c_spearman(probe, transfer, resamples=100, seed=8)
    second = nested_gate_c_spearman(probe, transfer, resamples=100, seed=8)
    assert first == second
    assert first.point.rho == pytest.approx(1.0)
    assert first.finite_bootstrap_fraction == 1.0
    assert first.interval is not None
    assert first.interval.low == pytest.approx(1.0)


def test_nested_gate_c_constant_point_is_null_not_nan() -> None:
    probe = np.ones((4, 3))
    transfer = np.arange(4, dtype=np.float64)[:, None, None] + np.ones((4, 2, 5))
    result = nested_gate_c_spearman(probe, transfer, resamples=20, seed=5)
    assert result.point.rho is None
    assert result.interval is None
    assert result.reason == UNDEFINED_SPEARMAN_REASON
    assert result.finite_bootstrap_fraction == 0.0
