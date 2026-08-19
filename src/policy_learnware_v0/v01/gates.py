"""Pure, fail-closed gate computations for the v0.1 diagnostic study."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .statistics import (
    ConfidenceInterval,
    NestedSpearmanResult,
    SpearmanResult,
    Top1BootstrapResult,
    UNDEFINED_SPEARMAN_REASON,
    empirical_competence_set,
    holm_bonferroni,
    nested_gate_c_spearman,
    significant_after_holm,
    spearman_correlation,
)


GATE_0_REQUIRED_CHECKS = (
    "registry_factor_grid_valid",
    "nominal_model_digest_identity",
    "allowlisted_model_diff_only",
    "environment_contract_identity",
    "measurement_schema_identity",
    "identity_trajectory_and_policy_returns",
    "non_nominal_finite_no_early_termination",
    "instance_isolation",
    "base_protocol_runtime_bundle_bindings",
    "v0_regression_attestation",
)

GATE_D_REQUIRED_CHECKS = (
    "measurement_artifacts_forbidden_fields_absent",
    "taskspec_command_has_no_oracle_dependency",
    "oracle_poison_does_not_change_taskspec_digest",
    "context_confined_to_private_or_baseline",
    "smoke_and_formal_runs_separated",
    "matrix_inputs_match_frozen_protocols",
    "visibility_artifacts_untampered",
)


def _identifier(value: str, *, name: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _probability(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


@dataclass(frozen=True)
class BooleanCriterion:
    name: str
    passed: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, name="criterion name")
        if type(self.passed) is not bool:
            raise ValueError("criterion passed flag must be bool")
        if self.reason is not None and not str(self.reason):
            raise ValueError("criterion reason cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "reason": self.reason}


@dataclass(frozen=True)
class HardGateReport:
    gate: str
    criteria: tuple[BooleanCriterion, ...]

    def __post_init__(self) -> None:
        _identifier(self.gate, name="gate")
        names = tuple(item.name for item in self.criteria)
        if not names or len(names) != len(set(names)):
            raise ValueError("hard gate criteria must be non-empty and unique")

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.criteria)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "hard_gate": True,
            "passed": self.passed,
            "criteria": [item.to_dict() for item in self.criteria],
        }


def _evaluate_hard_gate(
    gate: str,
    checks: Mapping[str, bool],
    *,
    required_checks: Sequence[str],
) -> HardGateReport:
    required = tuple(str(name) for name in required_checks)
    if not required or any(not name for name in required) or len(required) != len(set(required)):
        raise ValueError("required check names must be non-empty and unique")
    unknown = sorted(set(checks) - set(required))
    if unknown:
        raise ValueError(f"unknown {gate} checks: {unknown}")
    criteria: list[BooleanCriterion] = []
    for name in required:
        if name not in checks:
            criteria.append(BooleanCriterion(name, False, "missing_evidence"))
            continue
        value = checks[name]
        if type(value) is not bool:
            raise ValueError(f"{gate} check {name!r} must be bool")
        criteria.append(BooleanCriterion(name, value, None if value else "check_failed"))
    return HardGateReport(gate, tuple(criteria))


def evaluate_gate_0(checks: Mapping[str, bool]) -> HardGateReport:
    """Gate 0: all preregistered engineering checks must be present and true."""

    return _evaluate_hard_gate("gate_0", checks, required_checks=GATE_0_REQUIRED_CHECKS)


def evaluate_gate_d(checks: Mapping[str, bool]) -> HardGateReport:
    """Gate D: all visibility/isolation checks must be present and true."""

    return _evaluate_hard_gate("gate_d", checks, required_checks=GATE_D_REQUIRED_CHECKS)


@dataclass(frozen=True)
class CorrectedEffect:
    """One effect after correction over its complete preregistered family."""

    hypothesis_id: str
    context_id: str
    candidate_ids: tuple[str, ...]
    estimate: float
    interval: ConfidenceInterval
    raw_p_value: float
    adjusted_p_value: float
    correction_order: int
    family_size: int

    def __post_init__(self) -> None:
        _identifier(self.hypothesis_id, name="hypothesis_id")
        _identifier(self.context_id, name="context_id")
        candidates = tuple(_identifier(value, name="candidate_id") for value in self.candidate_ids)
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("candidate_ids must be non-empty and unique")
        estimate = float(self.estimate)
        if not math.isfinite(estimate):
            raise ValueError("effect estimate must be finite")
        raw = _probability(self.raw_p_value, name="raw_p_value")
        adjusted = _probability(self.adjusted_p_value, name="adjusted_p_value")
        if adjusted + 1e-15 < raw:
            raise ValueError("Holm-adjusted p-value cannot be smaller than raw p-value")
        if isinstance(self.family_size, bool) or int(self.family_size) <= 0:
            raise ValueError("family_size must be positive")
        if (
            isinstance(self.correction_order, bool)
            or int(self.correction_order) <= 0
            or int(self.correction_order) > int(self.family_size)
        ):
            raise ValueError("correction_order must be in [1, family_size]")
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "estimate", estimate)
        object.__setattr__(self, "raw_p_value", raw)
        object.__setattr__(self, "adjusted_p_value", adjusted)
        object.__setattr__(self, "correction_order", int(self.correction_order))
        object.__setattr__(self, "family_size", int(self.family_size))

    def significant(self, *, alpha: float = 0.05) -> bool:
        return significant_after_holm(
            self.interval, self.adjusted_p_value, alpha=alpha
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "context_id": self.context_id,
            "candidate_ids": list(self.candidate_ids),
            "estimate": self.estimate,
            "ci": self.interval.to_dict(),
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "correction_order": self.correction_order,
            "family_size": self.family_size,
        }


@dataclass(frozen=True)
class RankingReversalEvidence:
    context_id: str
    nominal_gap: CorrectedEffect
    shifted_gap: CorrectedEffect

    def __post_init__(self) -> None:
        context_id = _identifier(self.context_id, name="context_id")
        if len(self.nominal_gap.candidate_ids) != 2 or len(self.shifted_gap.candidate_ids) != 2:
            raise ValueError("ranking gaps must compare exactly two candidates")
        if self.nominal_gap.candidate_ids != self.shifted_gap.candidate_ids:
            raise ValueError("nominal and shifted ranking gaps must use the same ordered pair")
        if self.shifted_gap.context_id != context_id:
            raise ValueError("shifted gap context must match reversal context")

    def supported(self, *, alpha: float = 0.05) -> bool:
        return (
            self.nominal_gap.estimate * self.shifted_gap.estimate < 0.0
            and self.nominal_gap.significant(alpha=alpha)
            and self.shifted_gap.significant(alpha=alpha)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "candidate_ids": list(self.nominal_gap.candidate_ids),
            "supported": self.supported(),
            "nominal_gap": self.nominal_gap.to_dict(),
            "shifted_gap": self.shifted_gap.to_dict(),
        }


@dataclass(frozen=True)
class Top1ChangeEvidence:
    context_id: str
    nominal: Top1BootstrapResult
    shifted: Top1BootstrapResult

    def __post_init__(self) -> None:
        _identifier(self.context_id, name="context_id")

    def supported(
        self,
        *,
        minimum_probability: float,
        eligible_candidates: Sequence[str],
    ) -> bool:
        threshold = _probability(minimum_probability, name="minimum_probability")
        eligible = set(eligible_candidates)
        nominal_winner = self.nominal.empirical_winner
        shifted_winner = self.shifted.empirical_winner
        return (
            nominal_winner != shifted_winner
            and nominal_winner in eligible
            and shifted_winner in eligible
            and self.nominal.probability(nominal_winner) >= threshold
            and self.shifted.probability(shifted_winner) >= threshold
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "nominal": self.nominal.to_dict(),
            "shifted": self.shifted.to_dict(),
        }


def _canonical_pairs(candidate_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(itertools.combinations(sorted(candidate_ids), 2))


def _validate_corrected_family(
    records: Sequence[CorrectedEffect],
    *,
    expected_keys: set[tuple[Any, ...]],
    key,
    expected_size: int,
    family_name: str,
) -> None:
    if len(records) != expected_size:
        raise ValueError(
            f"{family_name} family must contain {expected_size} hypotheses, got {len(records)}"
        )
    observed_keys = [key(record) for record in records]
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != expected_keys:
        raise ValueError(f"{family_name} family does not match preregistered hypothesis keys")
    if any(record.family_size != expected_size for record in records):
        raise ValueError(f"{family_name} records have incorrect family_size")
    if {record.correction_order for record in records} != set(range(1, expected_size + 1)):
        raise ValueError(f"{family_name} correction orders are incomplete or duplicated")
    if len({record.hypothesis_id for record in records}) != expected_size:
        raise ValueError(f"{family_name} hypothesis ids must be unique")
    expected_holm = holm_bonferroni(
        {record.hypothesis_id: record.raw_p_value for record in records}
    )
    for record in records:
        expected = expected_holm[record.hypothesis_id]
        if (
            record.correction_order != expected.correction_order
            or record.family_size != expected.family_size
            or not math.isclose(
                record.adjusted_p_value,
                expected.adjusted_p_value,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"{family_name} record does not match recomputed Holm correction")


def _validate_gate_a_families(
    *,
    candidates: tuple[str, ...],
    contexts: tuple[str, ...],
    material: Sequence[CorrectedEffect],
    heterogeneity: Sequence[CorrectedEffect],
    reversals: Sequence[RankingReversalEvidence],
) -> None:
    candidate_set = set(candidates)
    context_set = set(contexts)
    pairs = _canonical_pairs(candidates)
    pair_set = set(pairs)

    for record in material:
        if len(record.candidate_ids) != 1:
            raise ValueError("material effects must identify one candidate")
    material_keys = {(context, (candidate,)) for context in contexts for candidate in candidates}
    _validate_corrected_family(
        material,
        expected_keys=material_keys,
        key=lambda item: (item.context_id, item.candidate_ids),
        expected_size=len(material_keys),
        family_name="material",
    )

    for record in heterogeneity:
        if len(record.candidate_ids) != 2 or tuple(sorted(record.candidate_ids)) != record.candidate_ids:
            raise ValueError("heterogeneity effects must use canonical two-candidate pairs")
    heterogeneity_keys = {(context, pair) for context in contexts for pair in pairs}
    _validate_corrected_family(
        heterogeneity,
        expected_keys=heterogeneity_keys,
        key=lambda item: (item.context_id, item.candidate_ids),
        expected_size=len(heterogeneity_keys),
        family_name="heterogeneity",
    )

    reversal_keys = [(record.context_id, record.nominal_gap.candidate_ids) for record in reversals]
    expected_reversal_keys = {(context, pair) for context in contexts for pair in pairs}
    if len(reversal_keys) != len(expected_reversal_keys) or set(reversal_keys) != expected_reversal_keys:
        raise ValueError("ranking reversal evidence must cover every pair and context")
    unique_ranking: dict[str, CorrectedEffect] = {}
    for reversal in reversals:
        if reversal.context_id not in context_set:
            raise ValueError("ranking reversal contains an unknown context")
        if reversal.nominal_gap.candidate_ids not in pair_set:
            raise ValueError("ranking reversal contains an unknown candidate pair")
        for effect in (reversal.nominal_gap, reversal.shifted_gap):
            previous = unique_ranking.get(effect.hypothesis_id)
            if previous is not None and previous != effect:
                raise ValueError("repeated nominal ranking hypothesis is inconsistent")
            unique_ranking[effect.hypothesis_id] = effect
    expected_ranking_size = len(pairs) * (1 + len(contexts))
    ranking_records = tuple(unique_ranking.values())
    if len(ranking_records) != expected_ranking_size:
        raise ValueError("ranking-gap family has incorrect number of unique hypotheses")
    if any(record.family_size != expected_ranking_size for record in ranking_records):
        raise ValueError("ranking-gap records have incorrect family_size")
    if {record.correction_order for record in ranking_records} != set(
        range(1, expected_ranking_size + 1)
    ):
        raise ValueError("ranking-gap correction orders are incomplete or duplicated")
    if any(set(record.candidate_ids) - candidate_set for record in ranking_records):
        raise ValueError("ranking-gap record contains unknown candidate")
    expected_holm = holm_bonferroni(
        {record.hypothesis_id: record.raw_p_value for record in ranking_records}
    )
    for record in ranking_records:
        expected = expected_holm[record.hypothesis_id]
        if (
            record.correction_order != expected.correction_order
            or not math.isclose(
                record.adjusted_p_value,
                expected.adjusted_p_value,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("ranking-gap record does not match recomputed Holm correction")


@dataclass(frozen=True)
class GateAContextResult:
    context_id: str
    material_effect_passed: bool
    sensitivity_heterogeneity_passed: bool
    ranking_evidence_passed: bool
    material_hypothesis_ids: tuple[str, ...]
    heterogeneity_hypothesis_ids: tuple[str, ...]
    ranking_evidence_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.material_effect_passed
            and self.sensitivity_heterogeneity_passed
            and self.ranking_evidence_passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "passed": self.passed,
            "material_effect_passed": self.material_effect_passed,
            "sensitivity_heterogeneity_passed": self.sensitivity_heterogeneity_passed,
            "ranking_evidence_passed": self.ranking_evidence_passed,
            "material_hypothesis_ids": list(self.material_hypothesis_ids),
            "heterogeneity_hypothesis_ids": list(self.heterogeneity_hypothesis_ids),
            "ranking_evidence_ids": list(self.ranking_evidence_ids),
        }


@dataclass(frozen=True)
class GateATaskResult:
    task: str
    competence_set: tuple[str, ...]
    competence_threshold: float
    contexts: tuple[GateAContextResult, ...]
    reason: str | None
    material_effects: tuple[CorrectedEffect, ...]
    heterogeneity_effects: tuple[CorrectedEffect, ...]
    ranking_reversals: tuple[RankingReversalEvidence, ...]
    top1_changes: tuple[Top1ChangeEvidence, ...]

    @property
    def passed(self) -> bool:
        return self.reason is None and any(context.passed for context in self.contexts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "passed": self.passed,
            "reason": self.reason,
            "competence_set": list(self.competence_set),
            "competence_threshold": self.competence_threshold,
            "contexts": [context.to_dict() for context in self.contexts],
            "corrected_families": {
                "material": [record.to_dict() for record in self.material_effects],
                "heterogeneity": [record.to_dict() for record in self.heterogeneity_effects],
                "ranking_reversal_views": [
                    record.to_dict() for record in self.ranking_reversals
                ],
            },
            "top1_change_evidence": [record.to_dict() for record in self.top1_changes],
        }


def evaluate_gate_a_task(
    *,
    task: str,
    nominal_returns: Mapping[str, float],
    context_ids: Sequence[str],
    material_effects: Sequence[CorrectedEffect],
    heterogeneity_effects: Sequence[CorrectedEffect],
    ranking_reversals: Sequence[RankingReversalEvidence],
    top1_changes: Sequence[Top1ChangeEvidence],
    competence_alpha: float = 0.8,
    minimum_material_effect: float = 0.05,
    minimum_sensitivity_heterogeneity: float = 0.05,
    significance_alpha: float = 0.05,
    top1_bootstrap_probability: float = 0.95,
) -> GateATaskResult:
    """Evaluate Gate A after full-pool Holm correction, grouped by shift."""

    task = _identifier(task, name="task")
    contexts = tuple(_identifier(item, name="context_id") for item in context_ids)
    if not contexts or len(contexts) != len(set(contexts)):
        raise ValueError("context_ids must be non-empty and unique")
    candidates = tuple(sorted(str(key) for key in nominal_returns))
    if (
        len(candidates) < 2
        or any(not candidate for candidate in candidates)
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("Gate A requires at least two candidates")
    minimum_material_effect = float(minimum_material_effect)
    minimum_sensitivity_heterogeneity = float(minimum_sensitivity_heterogeneity)
    if (
        not math.isfinite(minimum_material_effect)
        or minimum_material_effect < 0.0
        or not math.isfinite(minimum_sensitivity_heterogeneity)
        or minimum_sensitivity_heterogeneity < 0.0
    ):
        raise ValueError("Gate A effect thresholds must be finite and non-negative")
    significance_alpha = float(significance_alpha)
    if not math.isfinite(significance_alpha) or not 0.0 < significance_alpha < 1.0:
        raise ValueError("significance_alpha must be in (0, 1)")
    top1_bootstrap_probability = _probability(
        top1_bootstrap_probability, name="top1_bootstrap_probability"
    )

    _validate_gate_a_families(
        candidates=candidates,
        contexts=contexts,
        material=material_effects,
        heterogeneity=heterogeneity_effects,
        reversals=ranking_reversals,
    )
    top1_by_context = {record.context_id: record for record in top1_changes}
    if set(top1_by_context) != set(contexts) or len(top1_by_context) != len(top1_changes):
        raise ValueError("top1 change evidence must cover every context exactly once")
    candidate_set = set(candidates)
    for record in top1_changes:
        if (
            set(record.nominal.probabilities) != candidate_set
            or set(record.shifted.probabilities) != candidate_set
        ):
            raise ValueError("top1 evidence must cover the complete candidate pool")

    competence = empirical_competence_set(nominal_returns, alpha=competence_alpha)
    eligible = set(competence.candidate_ids)
    context_results: list[GateAContextResult] = []
    for context in contexts:
        material_hits = tuple(
            record.hypothesis_id
            for record in material_effects
            if record.context_id == context
            and record.candidate_ids[0] in eligible
            and abs(record.estimate) >= minimum_material_effect
            and record.significant(alpha=significance_alpha)
        )
        heterogeneity_hits = tuple(
            record.hypothesis_id
            for record in heterogeneity_effects
            if record.context_id == context
            and set(record.candidate_ids) <= eligible
            and abs(record.estimate) >= minimum_sensitivity_heterogeneity
            and record.significant(alpha=significance_alpha)
        )
        reversal_hits = tuple(
            f"reversal:{'/'.join(record.nominal_gap.candidate_ids)}"
            for record in ranking_reversals
            if record.context_id == context
            and set(record.nominal_gap.candidate_ids) <= eligible
            and record.supported(alpha=significance_alpha)
        )
        top1_supported = top1_by_context[context].supported(
            minimum_probability=top1_bootstrap_probability,
            eligible_candidates=eligible,
        )
        ranking_hits = reversal_hits + (("top1_change",) if top1_supported else ())
        context_results.append(
            GateAContextResult(
                context_id=context,
                material_effect_passed=bool(material_hits),
                sensitivity_heterogeneity_passed=bool(heterogeneity_hits),
                ranking_evidence_passed=bool(ranking_hits),
                material_hypothesis_ids=material_hits,
                heterogeneity_hypothesis_ids=heterogeneity_hits,
                ranking_evidence_ids=ranking_hits,
            )
        )
    reason = None if competence.sufficient else "insufficient_competent_candidates"
    return GateATaskResult(
        task=task,
        competence_set=competence.candidate_ids,
        competence_threshold=competence.threshold,
        contexts=tuple(context_results),
        reason=reason,
        material_effects=tuple(material_effects),
        heterogeneity_effects=tuple(heterogeneity_effects),
        ranking_reversals=tuple(ranking_reversals),
        top1_changes=tuple(top1_changes),
    )


@dataclass(frozen=True)
class GateAReport:
    tasks: tuple[GateATaskResult, ...]

    @property
    def passed_tasks(self) -> tuple[str, ...]:
        return tuple(task.task for task in self.tasks if task.passed)

    @property
    def passed(self) -> bool:
        return bool(self.passed_tasks)

    @property
    def replicated_across_both_tasks(self) -> bool:
        return len(self.tasks) == 2 and len(self.passed_tasks) == 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "gate_a",
            "passed": self.passed,
            "passed_tasks": list(self.passed_tasks),
            "replicated_across_both_tasks": self.replicated_across_both_tasks,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def evaluate_gate_a(task_results: Sequence[GateATaskResult]) -> GateAReport:
    tasks = tuple(task_results)
    names = tuple(task.task for task in tasks)
    if not tasks or len(names) != len(set(names)):
        raise ValueError("Gate A task results must be non-empty and unique")
    return GateAReport(tasks)


@dataclass(frozen=True)
class GateBTaskResult:
    task: str
    between_median: float
    within_q95: float
    between_within_ratio: float | None
    denominator_below_tolerance: bool
    ratio_criterion: BooleanCriterion
    severity: SpearmanResult
    severity_criterion: BooleanCriterion
    mask_schema_max_distance: float
    mask_schema_criterion: BooleanCriterion
    non_nominal_points: int

    @property
    def passed(self) -> bool:
        return (
            self.ratio_criterion.passed
            and self.severity_criterion.passed
            and self.mask_schema_criterion.passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "passed": self.passed,
            "between_median": self.between_median,
            "within_q95": self.within_q95,
            "between_within_ratio": self.between_within_ratio,
            "denominator_below_tolerance": self.denominator_below_tolerance,
            "ratio_criterion": self.ratio_criterion.to_dict(),
            "severity_spearman": self.severity.to_dict(),
            "severity_criterion": self.severity_criterion.to_dict(),
            "mask_schema_max_distance": self.mask_schema_max_distance,
            "mask_schema_criterion": self.mask_schema_criterion.to_dict(),
            "non_nominal_points": self.non_nominal_points,
        }


def evaluate_gate_b_task(
    *,
    task: str,
    within_distances: Sequence[float] | np.ndarray,
    between_distances: Sequence[float] | np.ndarray,
    severity_d_theta: Sequence[float] | np.ndarray,
    severity_median_d_phi: Sequence[float] | np.ndarray,
    mask_schema_max_distance: float,
    minimum_between_within_ratio: float = 1.25,
    minimum_severity_spearman: float = 0.8,
    numerical_zero_tolerance: float = 1e-10,
) -> GateBTaskResult:
    """Evaluate per-task Gate B, including all preregistered edge cases."""

    task = _identifier(task, name="task")
    within = np.asarray(within_distances, dtype=np.float64)
    between = np.asarray(between_distances, dtype=np.float64)
    if within.ndim != 1 or between.ndim != 1 or within.size == 0 or between.size == 0:
        raise ValueError("within_distances and between_distances must be non-empty vectors")
    if (
        not np.all(np.isfinite(within))
        or not np.all(np.isfinite(between))
        or np.any(within < 0.0)
        or np.any(between < 0.0)
    ):
        raise ValueError("TaskSpec distances must be finite and non-negative")
    ratio_threshold = float(minimum_between_within_ratio)
    rho_threshold = float(minimum_severity_spearman)
    tolerance = float(numerical_zero_tolerance)
    mask_distance = float(mask_schema_max_distance)
    if (
        not math.isfinite(ratio_threshold)
        or ratio_threshold < 0.0
        or not math.isfinite(rho_threshold)
        or not -1.0 <= rho_threshold <= 1.0
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or not math.isfinite(mask_distance)
        or mask_distance < 0.0
    ):
        raise ValueError("invalid Gate B threshold or mask distance")

    numerator = float(np.median(between))
    denominator = float(np.quantile(within, 0.95, method="linear"))
    denominator_below = denominator <= tolerance
    if denominator_below and numerator <= tolerance:
        ratio = None
        ratio_criterion = BooleanCriterion(
            "between_within_ratio", False, "no_detectable_signal"
        )
    elif denominator_below:
        ratio = None
        ratio_criterion = BooleanCriterion(
            "between_within_ratio", True, "denominator_below_tolerance_with_positive_signal"
        )
    else:
        ratio = numerator / denominator
        ratio_criterion = BooleanCriterion(
            "between_within_ratio",
            ratio >= ratio_threshold,
            None if ratio >= ratio_threshold else "below_minimum",
        )

    d_theta = np.asarray(severity_d_theta, dtype=np.float64)
    d_phi = np.asarray(severity_median_d_phi, dtype=np.float64)
    if d_theta.ndim != 1 or d_phi.ndim != 1 or d_theta.shape != d_phi.shape:
        raise ValueError("severity d_theta and d_phi must be equal-length vectors")
    if np.any(d_theta[np.isfinite(d_theta)] < 0.0) or np.any(d_phi[np.isfinite(d_phi)] < 0.0):
        raise ValueError("severity distances must be non-negative")
    non_nominal = np.isfinite(d_theta) & (d_theta > tolerance)
    non_nominal_count = int(np.count_nonzero(non_nominal))
    if (
        non_nominal_count < 2
        or not np.all(np.isfinite(d_theta))
        or not np.all(np.isfinite(d_phi[non_nominal]))
    ):
        severity = SpearmanResult(None, UNDEFINED_SPEARMAN_REASON, non_nominal_count)
    else:
        severity = spearman_correlation(d_theta[non_nominal], d_phi[non_nominal])
    severity_pass = severity.rho is not None and severity.rho >= rho_threshold
    severity_criterion = BooleanCriterion(
        "severity_spearman",
        severity_pass,
        None if severity_pass else (severity.reason or "below_minimum"),
    )
    mask_pass = mask_distance <= tolerance
    mask_criterion = BooleanCriterion(
        "mask_schema_negative_control",
        mask_pass,
        None if mask_pass else "above_numerical_zero_tolerance",
    )
    return GateBTaskResult(
        task=task,
        between_median=numerator,
        within_q95=denominator,
        between_within_ratio=ratio,
        denominator_below_tolerance=denominator_below,
        ratio_criterion=ratio_criterion,
        severity=severity,
        severity_criterion=severity_criterion,
        mask_schema_max_distance=mask_distance,
        mask_schema_criterion=mask_criterion,
        non_nominal_points=non_nominal_count,
    )


@dataclass(frozen=True)
class GateBReport:
    tasks: tuple[GateBTaskResult, ...]
    routing_accuracy: float
    minimum_routing_accuracy: float
    gate_d_passed: bool

    @property
    def routing_passed(self) -> bool:
        return self.routing_accuracy >= self.minimum_routing_accuracy

    @property
    def passed(self) -> bool:
        return all(task.passed for task in self.tasks) and self.routing_passed and self.gate_d_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "gate_b",
            "passed": self.passed,
            "tasks": [task.to_dict() for task in self.tasks],
            "routing_accuracy": self.routing_accuracy,
            "minimum_routing_accuracy": self.minimum_routing_accuracy,
            "routing_passed": self.routing_passed,
            "gate_d_passed": self.gate_d_passed,
        }


def evaluate_gate_b(
    task_results: Sequence[GateBTaskResult],
    *,
    routing_accuracy: float,
    minimum_routing_accuracy: float = 0.95,
    gate_d_passed: bool,
) -> GateBReport:
    tasks = tuple(task_results)
    names = tuple(task.task for task in tasks)
    if not tasks or len(names) != len(set(names)):
        raise ValueError("Gate B task results must be non-empty and unique")
    routing = _probability(routing_accuracy, name="routing_accuracy")
    minimum = _probability(minimum_routing_accuracy, name="minimum_routing_accuracy")
    if type(gate_d_passed) is not bool:
        raise ValueError("gate_d_passed must be bool")
    return GateBReport(tasks, routing, minimum, gate_d_passed)


@dataclass(frozen=True)
class GateCDiagnostic:
    """Gate C intentionally has no pass/fail or strength field."""

    task: str
    result: NestedSpearmanResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": "gate_c_diagnostic",
            "task": self.task,
            "statistics": self.result.to_dict(),
        }


def evaluate_gate_c(
    *,
    task: str,
    probe_distances: Sequence[Sequence[float]] | np.ndarray,
    paired_transfer_differences: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
    minimum_finite_fraction: float = 0.95,
) -> GateCDiagnostic:
    return GateCDiagnostic(
        task=_identifier(task, name="task"),
        result=nested_gate_c_spearman(
            probe_distances,
            paired_transfer_differences,
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
            minimum_finite_fraction=minimum_finite_fraction,
        ),
    )


__all__ = [
    "BooleanCriterion",
    "CorrectedEffect",
    "GATE_0_REQUIRED_CHECKS",
    "GATE_D_REQUIRED_CHECKS",
    "GateAContextResult",
    "GateAReport",
    "GateATaskResult",
    "GateBReport",
    "GateBTaskResult",
    "GateCDiagnostic",
    "HardGateReport",
    "RankingReversalEvidence",
    "Top1ChangeEvidence",
    "evaluate_gate_0",
    "evaluate_gate_a",
    "evaluate_gate_a_task",
    "evaluate_gate_b",
    "evaluate_gate_b_task",
    "evaluate_gate_c",
    "evaluate_gate_d",
]
