from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_v0.v01.gates import (
    CorrectedEffect,
    GATE_0_REQUIRED_CHECKS,
    GATE_D_REQUIRED_CHECKS,
    RankingReversalEvidence,
    Top1ChangeEvidence,
    evaluate_gate_0,
    evaluate_gate_a,
    evaluate_gate_a_task,
    evaluate_gate_b,
    evaluate_gate_b_task,
    evaluate_gate_c,
    evaluate_gate_d,
)
from policy_learnware_v0.v01.statistics import (
    ConfidenceInterval,
    Top1BootstrapResult,
    UNDEFINED_SPEARMAN_REASON,
)


def _effect(
    hypothesis_id: str,
    context_id: str,
    candidates: tuple[str, ...],
    estimate: float,
    *,
    order: int,
    family_size: int,
    significant: bool = True,
) -> CorrectedEffect:
    if significant:
        if estimate > 0:
            interval = ConfidenceInterval(max(estimate / 2, 0.001), estimate * 1.5, 0.95)
        else:
            interval = ConfidenceInterval(estimate * 1.5, min(estimate / 2, -0.001), 0.95)
        raw, adjusted = 0.001, min(1.0, 0.001 * family_size)
    else:
        interval = ConfidenceInterval(-0.1, 0.1, 0.95)
        raw, adjusted = 0.5, 1.0
    return CorrectedEffect(
        hypothesis_id=hypothesis_id,
        context_id=context_id,
        candidate_ids=candidates,
        estimate=estimate,
        interval=interval,
        raw_p_value=raw,
        adjusted_p_value=adjusted,
        correction_order=order,
        family_size=family_size,
    )


def _top1(context: str, nominal: str = "a", shifted: str = "a") -> Top1ChangeEvidence:
    nominal_result = Top1BootstrapResult(
        empirical_winner=nominal,
        probabilities={"a": 1.0 if nominal == "a" else 0.0, "b": 1.0 if nominal == "b" else 0.0},
        resamples=100,
        seed=1,
    )
    shifted_result = Top1BootstrapResult(
        empirical_winner=shifted,
        probabilities={"a": 1.0 if shifted == "a" else 0.0, "b": 1.0 if shifted == "b" else 0.0},
        resamples=100,
        seed=2,
    )
    return Top1ChangeEvidence(context, nominal_result, shifted_result)


def _gate_a_inputs(*, reversal_significant: bool = True):
    context = "shift-1"
    material = [
        _effect("m-a", context, ("a",), 0.20, order=1, family_size=2),
        _effect("m-b", context, ("b",), 0.10, order=2, family_size=2),
    ]
    heterogeneity = [
        _effect("h-ab", context, ("a", "b"), 0.10, order=1, family_size=1)
    ]
    nominal_gap = _effect(
        "r-ab-nominal",
        "nominal",
        ("a", "b"),
        0.10,
        order=1,
        family_size=2,
        significant=reversal_significant,
    )
    shifted_gap = _effect(
        "r-ab-shift",
        context,
        ("a", "b"),
        -0.10,
        order=2,
        family_size=2,
        significant=reversal_significant,
    )
    return context, material, heterogeneity, [RankingReversalEvidence(context, nominal_gap, shifted_gap)]


def test_gate_0_and_d_fail_closed_on_missing_evidence() -> None:
    gate_0 = evaluate_gate_0({name: True for name in GATE_0_REQUIRED_CHECKS[:-1]})
    assert not gate_0.passed
    assert gate_0.criteria[-1].reason == "missing_evidence"

    gate_d = evaluate_gate_d({name: True for name in GATE_D_REQUIRED_CHECKS})
    assert gate_d.passed
    with pytest.raises(ValueError, match="unknown gate_d"):
        evaluate_gate_d({**{name: True for name in GATE_D_REQUIRED_CHECKS}, "extra": True})


def test_gate_a_requires_all_three_criteria_at_the_same_shift() -> None:
    context, material, heterogeneity, reversals = _gate_a_inputs()
    task = evaluate_gate_a_task(
        task="WalkerWalk",
        nominal_returns={"a": 1.0, "b": 0.9},
        context_ids=[context],
        material_effects=material,
        heterogeneity_effects=heterogeneity,
        ranking_reversals=reversals,
        top1_changes=[_top1(context)],
    )
    assert task.passed
    assert task.competence_set == ("a", "b")
    assert task.contexts[0].passed
    report = evaluate_gate_a([task])
    assert report.passed
    assert report.passed_tasks == ("WalkerWalk",)
    assert not report.replicated_across_both_tasks


def test_gate_a_accepts_reliable_top1_change_as_ranking_alternative() -> None:
    context, material, heterogeneity, reversals = _gate_a_inputs(reversal_significant=False)
    task = evaluate_gate_a_task(
        task="WalkerWalk",
        nominal_returns={"a": 1.0, "b": 0.9},
        context_ids=[context],
        material_effects=material,
        heterogeneity_effects=heterogeneity,
        ranking_reversals=reversals,
        top1_changes=[_top1(context, nominal="a", shifted="b")],
    )
    assert task.passed
    assert task.contexts[0].ranking_evidence_ids == ("top1_change",)


def test_gate_a_filters_only_after_full_family_correction() -> None:
    context, material, heterogeneity, reversals = _gate_a_inputs()
    task = evaluate_gate_a_task(
        task="WalkerWalk",
        nominal_returns={"a": 1.0, "b": 0.7},
        context_ids=[context],
        material_effects=material,
        heterogeneity_effects=heterogeneity,
        ranking_reversals=reversals,
        top1_changes=[_top1(context)],
    )
    assert task.competence_set == ("a",)
    assert task.reason == "insufficient_competent_candidates"
    assert not task.passed

    with pytest.raises(ValueError, match="material family"):
        evaluate_gate_a_task(
            task="WalkerWalk",
            nominal_returns={"a": 1.0, "b": 0.9},
            context_ids=[context],
            material_effects=material[:1],
            heterogeneity_effects=heterogeneity,
            ranking_reversals=reversals,
            top1_changes=[_top1(context)],
        )


def test_gate_b_zero_denominator_positive_signal_passes_with_null_ratio() -> None:
    task = evaluate_gate_b_task(
        task="WalkerWalk",
        within_distances=[0.0, 0.0, 0.0],
        between_distances=[0.1, 0.2, 0.3],
        severity_d_theta=[1.0, 2.0, 0.0, 3.0, 4.0],
        severity_median_d_phi=[0.1, 0.2, 0.0, 0.3, 0.4],
        mask_schema_max_distance=0.0,
    )
    assert task.between_within_ratio is None
    assert task.denominator_below_tolerance
    assert task.ratio_criterion.passed
    assert task.severity.rho == pytest.approx(1.0)
    assert task.passed


def test_gate_b_zero_over_zero_is_no_detectable_signal() -> None:
    task = evaluate_gate_b_task(
        task="WalkerWalk",
        within_distances=[0.0, 0.0],
        between_distances=[0.0, 0.0],
        severity_d_theta=[0.0, 1.0, 2.0, 3.0, 4.0],
        severity_median_d_phi=[0.0, 1.0, 2.0, 3.0, 4.0],
        mask_schema_max_distance=0.0,
    )
    assert task.between_within_ratio is None
    assert not task.ratio_criterion.passed
    assert task.ratio_criterion.reason == "no_detectable_signal"
    assert not task.passed


def test_gate_b_excludes_nominal_and_fails_undefined_spearman() -> None:
    task = evaluate_gate_b_task(
        task="FingerTurnEasy",
        within_distances=[0.01, 0.02],
        between_distances=[0.1, 0.2],
        severity_d_theta=[0.0, 1.0, 2.0, 3.0, 4.0],
        severity_median_d_phi=[999.0, 1.0, 1.0, 1.0, 1.0],
        mask_schema_max_distance=0.0,
    )
    assert task.non_nominal_points == 4
    assert task.severity.rho is None
    assert task.severity.reason == UNDEFINED_SPEARMAN_REASON
    assert not task.severity_criterion.passed


def test_gate_b_overall_requires_every_task_routing_and_gate_d() -> None:
    good = evaluate_gate_b_task(
        task="WalkerWalk",
        within_distances=[0.01, 0.02],
        between_distances=[0.1, 0.2],
        severity_d_theta=[1, 2, 3, 4],
        severity_median_d_phi=[1, 2, 3, 4],
        mask_schema_max_distance=0.0,
    )
    assert evaluate_gate_b([good], routing_accuracy=1.0, gate_d_passed=True).passed
    assert not evaluate_gate_b([good], routing_accuracy=0.94, gate_d_passed=True).passed
    assert not evaluate_gate_b([good], routing_accuracy=1.0, gate_d_passed=False).passed


def test_gate_c_is_diagnostic_and_never_serializes_passed_or_strong() -> None:
    probe = np.arange(1, 5, dtype=float)[:, None] * np.ones((4, 3))
    transfer = np.arange(1, 5, dtype=float)[:, None, None] * np.ones((4, 2, 4))
    diagnostic = evaluate_gate_c(
        task="WalkerWalk",
        probe_distances=probe,
        paired_transfer_differences=transfer,
        resamples=20,
        seed=1,
    )
    artifact = diagnostic.to_dict()
    assert artifact["statistics"]["rho"] == pytest.approx(1.0)
    assert "passed" not in artifact
    assert "strong" not in artifact
