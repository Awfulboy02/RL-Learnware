from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from policy_learnware_v0.v02.costs import (
    QUERY_COST_COMPONENTS,
    CostContractError,
    CostRecord,
    reconcile_cold_warm_costs,
)
from policy_learnware_v0.v02.metrics import (
    HierarchicalValue,
    MetricContractError,
    aggregate_hierarchy,
    compute_prefix_auc,
    compute_ranking_metrics,
    compute_selection_metrics,
    normalize_episode_returns,
    summarize_normalized_returns,
)
from policy_learnware_v0.v02.statistics import (
    HierarchicalBootstrapResult,
    StatisticalContractError,
    bootstrap_max_t_intervals,
    derive_bootstrap_seed,
    evaluate_noninferiority,
    hierarchical_bootstrap,
    hierarchical_paired_difference_bootstrap,
    holm_bonferroni,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _hierarchy(values: dict[tuple[str, str, str, str], float]) -> tuple[HierarchicalValue, ...]:
    return tuple(
        HierarchicalValue(task, axis, context, observation, value)
        for (task, axis, context, observation), value in sorted(values.items())
    )


def _components(scale: float) -> dict[str, float]:
    return {
        component: scale * (index + 1)
        for index, component in enumerate(QUERY_COST_COMPONENTS)
    }


def _cost(query_id: str, mode: str, scale: float, *, steps: int = 64) -> CostRecord:
    return CostRecord.create(
        query_id=query_id,
        mode=mode,  # type: ignore[arg-type]
        cost_contract_digest=_d("cost-contract"),
        execution_attempt_id=f"attempt-{query_id}-{mode}",
        components_seconds=_components(scale),
        target_environment_steps=steps,
    )


def test_return_normalization_is_readonly_and_reward_contract_checked() -> None:
    normalized = normalize_episode_returns([0.0, 5.0, 10.0], horizon=10)
    np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0])
    assert not normalized.flags.writeable
    summary = summarize_normalized_returns(normalized)
    assert summary.mean == pytest.approx(0.5)
    assert summary.std == pytest.approx(np.std([0.0, 0.5, 1.0]))
    with pytest.raises(MetricContractError, match="reward/horizon"):
        normalize_episode_returns([10.1], horizon=10)
    with pytest.raises(MetricContractError, match=r"\[0, 1\]"):
        summarize_normalized_returns([-0.01, 0.5])


def test_pool_regret_uses_executable_oracle_and_never_falls_back() -> None:
    selected = compute_selection_metrics(
        selected_policy_id="policy-b",
        normalized_returns_by_policy={"policy-a": 0.9, "policy-b": 0.6},
        executable_policy_ids=("policy-a", "policy-b"),
        epsilon=0.3,
    )
    assert selected.selected_normalized_return == pytest.approx(0.6)
    assert selected.oracle_best_policy_ids == ("policy-a",)
    assert selected.pool_regret == pytest.approx(0.3)
    assert selected.epsilon_optimal
    assert not selected.top1_agreement

    incompatible = compute_selection_metrics(
        selected_policy_id="wrong-abi-policy",
        normalized_returns_by_policy={"policy-a": 0.9, "policy-b": 0.6},
        executable_policy_ids=("policy-a", "policy-b"),
        incompatible_failure_value=0.0,
    )
    assert not incompatible.selected_executable
    assert incompatible.selected_policy_id == "wrong-abi-policy"
    assert incompatible.pool_regret == pytest.approx(0.9)
    with pytest.raises(MetricContractError, match="requires incompatible_failure_value"):
        compute_selection_metrics(
            selected_policy_id="wrong-abi-policy",
            normalized_returns_by_policy={"policy-a": 0.9},
        )


def test_ranking_metrics_are_tie_aware_and_require_full_pool() -> None:
    result = compute_ranking_metrics(
        ("a", "b", "c"),
        {"a": 0.8, "b": 0.8, "c": 0.1},
    )
    assert result.concordant_pairs == 2
    assert result.oracle_tie_pairs == 1
    assert result.discordant_pairs == 0
    assert result.pairwise_accuracy == pytest.approx(5 / 6)
    assert result.top1_agreement
    assert result.kendall_tau_b == pytest.approx(2 / math.sqrt(6))
    assert result.spearman_rho == pytest.approx(math.sqrt(3) / 2)

    all_tied = compute_ranking_metrics(("a", "b"), {"a": 0.5, "b": 0.5})
    assert all_tied.kendall_tau_b is None
    assert all_tied.spearman_rho is None
    with pytest.raises(MetricContractError, match="match exactly"):
        compute_ranking_metrics(("a", "b"), {"a": 1.0, "c": 0.0})


def test_prefix_auc_uses_registered_x_scale_and_rejects_reordering() -> None:
    log_curve = compute_prefix_auc([1, 2, 4], [0.0, 0.5, 1.0], x_scale="log2")
    assert log_curve.normalized_auc == pytest.approx(0.5)
    linear_curve = compute_prefix_auc([1, 2, 4], [0.0, 0.5, 1.0], x_scale="linear")
    assert linear_curve.normalized_auc == pytest.approx(7 / 12)
    with pytest.raises(MetricContractError, match="strictly increasing"):
        compute_prefix_auc([1, 4, 2], [0.0, 1.0, 0.5])


def test_hierarchical_aggregate_is_task_then_axis_then_context_equal_weighted() -> None:
    rows = _hierarchy(
        {
            ("task-a", "axis-a", "context-1", "bank-1"): 0.0,
            ("task-a", "axis-a", "context-1", "bank-2"): 0.0,
            ("task-a", "axis-a", "context-2", "bank-1"): 1.0,
            ("task-a", "axis-b", "context-3", "bank-1"): 1.0,
            ("task-b", "axis-a", "context-4", "bank-1"): 0.0,
        }
    )
    result = aggregate_hierarchy(rows)
    # task-a: mean(mean(context-1, context-2), context-3) = .75;
    # task-b: 0; macro task-equal mean = .375.
    assert result.macro_mean == pytest.approx(0.375)
    assert result.task_count == 2
    assert result.axis_count == 3
    assert result.context_count == 4
    assert result.observation_count == 5
    assert [item.mean for item in result.task_aggregates] == pytest.approx([0.75, 0.0])


def test_hierarchical_aggregate_rejects_duplicate_or_reowned_context() -> None:
    row = HierarchicalValue("task", "axis", "context", "bank", 1.0)
    with pytest.raises(MetricContractError, match="unique"):
        aggregate_hierarchy((row, row))
    with pytest.raises(MetricContractError, match="multiple"):
        aggregate_hierarchy(
            (
                row,
                HierarchicalValue("task", "other-axis", "context", "other-bank", 1.0),
            )
        )


def test_hierarchical_bootstrap_is_deterministic_and_preserves_observed_formula() -> None:
    rows = _hierarchy(
        {
            ("task-a", "axis-a", "context-1", "bank-1"): 0.0,
            ("task-a", "axis-a", "context-1", "bank-2"): 1.0,
            ("task-a", "axis-b", "context-2", "bank-1"): 0.5,
            ("task-b", "axis-a", "context-3", "bank-1"): 0.25,
            ("task-b", "axis-a", "context-3", "bank-2"): 0.75,
        }
    )
    seed = derive_bootstrap_seed("experiment", "pool_regret", "primary")
    first = hierarchical_bootstrap(rows, resamples=128, seed=seed)
    second = hierarchical_bootstrap(rows, resamples=128, seed=seed)
    assert first.observed == pytest.approx(aggregate_hierarchy(rows).macro_mean)
    np.testing.assert_array_equal(first.replicates, second.replicates)
    assert not first.replicates.flags.writeable
    assert first.interval.low <= first.interval.high


def test_paired_hierarchical_bootstrap_requires_exact_leaf_keys() -> None:
    left = _hierarchy({("task", "axis", "context", "bank"): 0.8})
    right = _hierarchy({("task", "axis", "context", "bank"): 0.5})
    result = hierarchical_paired_difference_bootstrap(
        left, right, resamples=32, seed=7
    )
    assert result.observed == pytest.approx(0.3)
    np.testing.assert_allclose(result.replicates, 0.3)

    mismatched = _hierarchy({("task", "axis", "context", "other-bank"): 0.5})
    with pytest.raises(StatisticalContractError, match="identical keys"):
        hierarchical_paired_difference_bootstrap(
            left, mismatched, resamples=10, seed=1
        )


def test_holm_is_monotone_and_uses_lexical_tie_order() -> None:
    result = holm_bonferroni({"z": 0.01, "a": 0.01, "m": 0.04})
    assert result["a"].correction_order == 1
    assert result["z"].correction_order == 2
    assert result["m"].correction_order == 3
    assert result["a"].adjusted_p_value == pytest.approx(0.03)
    assert result["z"].adjusted_p_value == pytest.approx(0.03)
    assert result["m"].adjusted_p_value == pytest.approx(0.04)
    with pytest.raises(StatisticalContractError):
        holm_bonferroni({"bad": math.nan})


def test_paired_studentized_max_t_builds_simultaneous_family() -> None:
    first = HierarchicalBootstrapResult(
        observed=0.2,
        replicates=np.asarray([0.1, 0.2, 0.3, 0.15, 0.25]),
        confidence_level=0.8,
        seed=17,
        resampling_plan_digest="a" * 64,
    )
    second = HierarchicalBootstrapResult(
        observed=-0.1,
        replicates=np.asarray([-0.2, -0.1, 0.0, -0.15, -0.05]),
        confidence_level=0.8,
        seed=17,
        resampling_plan_digest="a" * 64,
    )
    result = bootstrap_max_t_intervals({"h2": second, "h1": first})
    assert tuple(result.intervals) == ("h1", "h2")
    assert result.method == "paired_studentized_bootstrap_max-T"
    assert result.intervals["h1"].low <= first.observed <= result.intervals["h1"].high
    assert result.intervals["h2"].low <= second.observed <= result.intervals["h2"].high
    with pytest.raises(StatisticalContractError, match="share paired"):
        bootstrap_max_t_intervals(
            {"h1": first, "h2": HierarchicalBootstrapResult(
                observed=second.observed,
                replicates=second.replicates,
                confidence_level=second.confidence_level,
                seed=18,
                resampling_plan_digest="a" * 64,
            )}
        )


def test_max_t_rejects_same_seed_and_count_with_different_leaf_plan() -> None:
    first = HierarchicalBootstrapResult(
        observed=0.2,
        replicates=np.asarray([0.1, 0.2, 0.3]),
        confidence_level=0.8,
        seed=17,
        resampling_plan_digest="a" * 64,
    )
    stale = HierarchicalBootstrapResult(
        observed=-0.1,
        replicates=np.asarray([-0.2, -0.1, 0.0]),
        confidence_level=0.8,
        seed=17,
        resampling_plan_digest="b" * 64,
    )
    with pytest.raises(StatisticalContractError, match="hierarchy/leaf resampling plan"):
        bootstrap_max_t_intervals({"h1": first, "h2": stale})


def test_one_sided_noninferiority_uses_lower_bound_against_negative_margin() -> None:
    method = _hierarchy({("task", "axis", "context", "bank"): 0.50})
    comparator = _hierarchy({("task", "axis", "context", "bank"): 0.505})
    bootstrap = hierarchical_paired_difference_bootstrap(
        method, comparator, resamples=99, seed=13, confidence_level=0.95
    )
    result = evaluate_noninferiority(bootstrap, margin=0.01)
    assert result.observed_difference == pytest.approx(-0.005)
    assert result.one_sided_lower_bound == pytest.approx(-0.005)
    assert result.null_boundary == pytest.approx(-0.01)
    assert result.passed
    assert result.raw_p_value == pytest.approx(0.01)

    failed_bootstrap = hierarchical_paired_difference_bootstrap(
        _hierarchy({("task", "axis", "context", "bank"): 0.48}),
        comparator,
        resamples=20,
        seed=13,
    )
    assert not evaluate_noninferiority(failed_bootstrap, margin=0.01).passed


def test_cost_record_reconciles_components_is_immutable_and_round_trips() -> None:
    record = _cost("query-a", "cold", 0.1)
    assert record.total_seconds == pytest.approx(sum(_components(0.1).values()))
    assert record.component_total_seconds == pytest.approx(record.total_seconds)
    assert record.target_gradient_steps == 0
    assert len(record.digest) == 64
    restored = CostRecord.from_dict(record.to_dict())
    assert restored == record
    with pytest.raises(TypeError):
        record.components_seconds["probe_collection"] = 0.0  # type: ignore[index]


def test_cost_record_rejects_unknown_components_total_drift_and_gradients() -> None:
    components = _components(0.1)
    components["unknown"] = 1.0
    with pytest.raises(CostContractError, match="unknown"):
        CostRecord.create(
            query_id="query",
            mode="cold",
            cost_contract_digest=_d("cost-contract"),
            execution_attempt_id="attempt",
            components_seconds=components,
            target_environment_steps=10,
        )

    valid = _components(0.1)
    with pytest.raises(CostContractError, match="does not reconcile"):
        CostRecord(
            query_id="query",
            mode="cold",
            cost_contract_digest=_d("cost-contract"),
            execution_attempt_id="attempt",
            components_seconds=valid,
            total_seconds=sum(valid.values()) + 0.1,
            target_environment_steps=10,
        )
    with pytest.raises(CostContractError, match="zero target gradients"):
        CostRecord.create(
            query_id="query",
            mode="cold",
            cost_contract_digest=_d("cost-contract"),
            execution_attempt_id="attempt",
            components_seconds=valid,
            target_environment_steps=10,
            target_gradient_steps=1,
        )


def test_cold_warm_reconciliation_requires_complete_paired_queries() -> None:
    records = (
        _cost("query-a", "cold", 0.2),
        _cost("query-a", "warm", 0.1),
        _cost("query-b", "cold", 0.4),
        _cost("query-b", "warm", 0.2),
    )
    result = reconcile_cold_warm_costs(
        records, expected_query_ids=("query-a", "query-b")
    )
    assert result.query_ids == ("query-a", "query-b")
    assert result.cold.mean_total_seconds == pytest.approx(10.8)
    assert result.warm.mean_total_seconds == pytest.approx(5.4)
    assert result.mean_cold_minus_warm_seconds == pytest.approx(5.4)
    assert result.cold_to_warm_ratio == pytest.approx(2.0)
    assert len(result.digest) == 64

    with pytest.raises(CostContractError, match="exactly one cold and warm"):
        reconcile_cold_warm_costs(records[:-1])
    with pytest.raises(CostContractError, match="duplicate"):
        reconcile_cold_warm_costs(records + (records[0],))


def test_cold_warm_reconciliation_rejects_step_or_contract_drift() -> None:
    with pytest.raises(CostContractError, match="same target environment steps"):
        reconcile_cold_warm_costs(
            (_cost("query", "cold", 0.2, steps=64), _cost("query", "warm", 0.1, steps=63))
        )

    warm = CostRecord.create(
        query_id="query",
        mode="warm",
        cost_contract_digest=_d("another-contract"),
        execution_attempt_id="warm-attempt",
        components_seconds=_components(0.1),
        target_environment_steps=64,
    )
    with pytest.raises(CostContractError, match="different cost contracts"):
        reconcile_cold_warm_costs((_cost("query", "cold", 0.2), warm))


def test_cold_warm_reconciliation_reports_measurement_jitter_without_relabeling() -> None:
    result = reconcile_cold_warm_costs(
        (_cost("query", "cold", 0.1), _cost("query", "warm", 0.2))
    )
    assert result.mean_cold_minus_warm_seconds == pytest.approx(-3.6)
    assert result.cold_to_warm_ratio == pytest.approx(0.5)
