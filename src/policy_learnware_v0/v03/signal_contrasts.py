"""Canonical, N/A-aware signal contrasts and the G03 transition gate.

This module is deliberately downstream of the 79-work-item signal atlas.  It
does not fit a representation, reduce a query KME, or read policy outcomes.  A
formal evaluation consumes the complete atlas metric records, replays every
paired contrast from the underlying rows, and binds the result to the
externally reviewed freeze.

Absolute MMD values are never subtracted across representations.  R5-minus-R0
comparisons use only retrieval metrics through ``representation_gain_contrast``.
The one-step temporal/history controls remain structural N/A records and are
excluded from every numeric denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .representation_ladder import (
    R0_PADDED_RAW,
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    R_HIST_RANDOM_TANH,
)
from .signal_controls import EXACT_REPEAT_CONTROL_ID, SCHEMA_COLLISION_CONTROL_ID
from .signal_matrix import (
    C_RF_SHUFFLED_NEXT,
    SignalCell,
    SignalMatrixPlan,
    build_signal_matrix_plan,
)
from .signal_metrics import (
    SignalMetricError,
    SignalMetricRecord,
    paired_signal_contrast,
    representation_gain_contrast,
)
from .transition_views import (
    V_DIMS_ONLY,
    V_FULL_LEGACY,
    V_MASK_ONLY,
    V_NO_MASK,
    V_RANDOM_ENCODER,
    V_REWARD_FREE_TRANSITION,
    V_REWARD_ONLY,
    V_SHUFFLED_REWARD,
    V_TEMPORAL_SHUFFLE,
)


SIGNAL_CONTRAST_SPEC_SCHEMA = "policy-learnware.v03-signal-contrast-spec.v0"
SIGNAL_CONTRAST_PLAN_SCHEMA = "policy-learnware.v03-signal-contrast-plan.v0"
SIGNAL_MATERIALITY_THRESHOLDS_SCHEMA = (
    "policy-learnware.v03-signal-materiality-thresholds.v0"
)
SIGNAL_CONTRAST_RESULT_SCHEMA = "policy-learnware.v03-signal-contrast-result.v0"
SIGNAL_CONTRAST_GATE_SCHEMA = "policy-learnware.v03-signal-contrast-gate.v0"

SIGNAL_CONTRAST_FAMILIES = (
    "SCHEMA",
    "REWARD_GOAL",
    "TRANSITION_MECHANISM",
    "TEMPORAL_HISTORY",
    "REPRESENTATION_LADDER",
)
TRANSITION_GATE_METRIC_IDS = ("dynamics_top1", "dynamics_mrr")
INTERPRETABLE_REPRESENTATION_GAIN_METRIC_IDS = (
    "task_top1",
    "task_mrr",
    "context_top1",
    "context_mrr",
)
REQUIRED_PAIR_CONTROL_IDS = (
    SCHEMA_COLLISION_CONTROL_ID,
    EXACT_REPEAT_CONTROL_ID,
)

ContrastKind = Literal["PAIRED", "REPRESENTATION_GAIN", "STRUCTURAL_NA"]
MaterialityRole = Literal[
    "REPORT_ONLY", "TRANSITION_MINIMUM", "STRUCTURAL_NA"
]
GateStatus = Literal["PASS", "NO_GO_TRANSITION_SIGNAL"]

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


class SignalContrastError(ValueError):
    """A contrast plan, threshold freeze, or formal gate is invalid."""


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise SignalContrastError(f"{where} must be a canonical safe ID")
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise SignalContrastError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SignalContrastError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SignalContrastError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SignalContrastError(f"{where} must be finite")
    return result


def _work_key(cell_id: str, seed: int | None) -> str:
    return f"{cell_id.replace('::', '--')}--seed-{'NONE' if seed is None else seed}"


def _cell_id(block: str, condition_id: str, representation_id: str) -> str:
    return f"{block}::{condition_id}::{representation_id}"


def _seed_token(seed: int | None) -> str:
    return "NONE" if seed is None else str(seed)


def _seeds(
    representation_id: str, *, historical_seed: int
) -> tuple[int | None, ...]:
    if representation_id in {
        R1_FIXED_RANDOM_LINEAR,
        R3_MATCHED_RANDOM_MLP,
        R5_VIEW_SPECIFIC_CORRO_REFIT,
        R5L_SUPERVISED_LINEAR,
    }:
        return (0, 1, 2)
    if representation_id == R_HIST_RANDOM_TANH:
        return (historical_seed,)
    if representation_id in {R0_PADDED_RAW, R2_SOURCE_PCA_WHITEN}:
        return (None,)
    raise SignalContrastError(f"unknown representation schedule: {representation_id}")


@dataclass(frozen=True)
class SignalContrastSpec:
    contrast_id: str
    family: str
    kind: ContrastKind
    base_cell_id: str
    control_cell_id: str | None
    base_seed: int | None
    control_seed: int | None
    metric_ids: tuple[str, ...]
    materiality_role: MaterialityRole
    schema: str = SIGNAL_CONTRAST_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CONTRAST_SPEC_SCHEMA:
            raise SignalContrastError("unsupported SignalContrastSpec schema")
        object.__setattr__(self, "contrast_id", _safe_id(self.contrast_id, "contrast_id"))
        if self.family not in SIGNAL_CONTRAST_FAMILIES:
            raise SignalContrastError("unknown signal contrast family")
        if self.kind not in {"PAIRED", "REPRESENTATION_GAIN", "STRUCTURAL_NA"}:
            raise SignalContrastError("unknown signal contrast kind")
        if not isinstance(self.base_cell_id, str) or not self.base_cell_id:
            raise SignalContrastError("base_cell_id must be non-empty")
        if self.control_cell_id is not None and (
            not isinstance(self.control_cell_id, str) or not self.control_cell_id
        ):
            raise SignalContrastError("control_cell_id must be non-empty or null")
        for name in ("base_seed", "control_seed"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise SignalContrastError(f"{name} must be non-negative or null")
        metrics = tuple(_safe_id(item, "metric_id") for item in self.metric_ids)
        if len(set(metrics)) != len(metrics):
            raise SignalContrastError("metric_ids must be unique")
        object.__setattr__(self, "metric_ids", metrics)
        if self.kind == "STRUCTURAL_NA":
            if (
                self.family != "TEMPORAL_HISTORY"
                or self.control_cell_id is not None
                or self.control_seed is not None
                or metrics
                or self.materiality_role != "STRUCTURAL_NA"
            ):
                raise SignalContrastError("structural N/A spec carries numeric material")
        else:
            if self.control_cell_id is None or not metrics:
                raise SignalContrastError("numeric contrast requires control and metrics")
            if self.materiality_role not in {"REPORT_ONLY", "TRANSITION_MINIMUM"}:
                raise SignalContrastError("numeric contrast has an invalid role")
            if (
                self.materiality_role == "TRANSITION_MINIMUM"
                and (
                    self.family != "TRANSITION_MECHANISM"
                    or self.kind != "PAIRED"
                    or metrics != TRANSITION_GATE_METRIC_IDS
                )
            ):
                raise SignalContrastError(
                    "transition materiality applies only to the frozen dynamics metrics"
                )

    @property
    def spec_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contrast_id": self.contrast_id,
            "family": self.family,
            "kind": self.kind,
            "base_cell_id": self.base_cell_id,
            "control_cell_id": self.control_cell_id,
            "base_seed": self.base_seed,
            "control_seed": self.control_seed,
            "metric_ids": list(self.metric_ids),
            "materiality_role": self.materiality_role,
        }


def _paired_specs(
    *,
    family: str,
    base_condition: str,
    control_conditions: Sequence[str],
    metric_ids: tuple[str, ...],
) -> list[SignalContrastSpec]:
    result: list[SignalContrastSpec] = []
    for control_condition in control_conditions:
        for representation_id in (R0_PADDED_RAW, R5_VIEW_SPECIFIC_CORRO_REFIT):
            for seed in _seeds(representation_id, historical_seed=0):
                result.append(
                    SignalContrastSpec(
                        contrast_id=(
                            f"{family}-{base_condition}-VS-{control_condition}-"
                            f"{representation_id}-SEED-{_seed_token(seed)}"
                        ),
                        family=family,
                        kind="PAIRED",
                        base_cell_id=_cell_id(
                            "CORE_PAIRED", base_condition, representation_id
                        ),
                        control_cell_id=_cell_id(
                            "CORE_PAIRED", control_condition, representation_id
                        ),
                        base_seed=seed,
                        control_seed=seed,
                        metric_ids=metric_ids,
                        materiality_role="REPORT_ONLY",
                    )
                )
    return result


def _canonical_specs(signal_plan: SignalMatrixPlan) -> tuple[SignalContrastSpec, ...]:
    specs: list[SignalContrastSpec] = []
    specs.extend(
        _paired_specs(
            family="SCHEMA",
            base_condition=V_FULL_LEGACY,
            control_conditions=(V_MASK_ONLY, V_DIMS_ONLY, V_NO_MASK),
            metric_ids=("task_top1", "task_mrr", "context_top1", "context_mrr"),
        )
    )
    specs.extend(
        _paired_specs(
            family="REWARD_GOAL",
            base_condition=V_FULL_LEGACY,
            control_conditions=(
                V_REWARD_ONLY,
                V_REWARD_FREE_TRANSITION,
                V_SHUFFLED_REWARD,
            ),
            metric_ids=("goal_top1", "goal_mrr"),
        )
    )
    specs.extend(
        _paired_specs(
            family="REWARD_GOAL",
            base_condition=V_REWARD_ONLY,
            control_conditions=(V_SHUFFLED_REWARD,),
            metric_ids=("goal_top1", "goal_mrr"),
        )
    )
    for representation_id in (
        R1_FIXED_RANDOM_LINEAR,
        R2_SOURCE_PCA_WHITEN,
        R3_MATCHED_RANDOM_MLP,
        R5L_SUPERVISED_LINEAR,
    ):
        for seed in _seeds(representation_id, historical_seed=0):
            specs.append(
                SignalContrastSpec(
                    contrast_id=(
                        f"TRANSITION-RF-VS-RF-SHUFFLED-NEXT-{representation_id}-"
                        f"SEED-{_seed_token(seed)}"
                    ),
                    family="TRANSITION_MECHANISM",
                    kind="PAIRED",
                    base_cell_id=_cell_id(
                        "MECHANISM_STAIRCASE",
                        V_REWARD_FREE_TRANSITION,
                        representation_id,
                    ),
                    control_cell_id=_cell_id(
                        "MECHANISM_STAIRCASE",
                        C_RF_SHUFFLED_NEXT,
                        representation_id,
                    ),
                    base_seed=seed,
                    control_seed=seed,
                    metric_ids=TRANSITION_GATE_METRIC_IDS,
                    materiality_role="TRANSITION_MINIMUM",
                )
            )
    for representation_id in (R0_PADDED_RAW, R5_VIEW_SPECIFIC_CORRO_REFIT):
        cell_id = _cell_id(
            "CORE_PAIRED", V_TEMPORAL_SHUFFLE, representation_id
        )
        cell = signal_plan.cell(cell_id)
        specs.append(
            SignalContrastSpec(
                contrast_id=f"TEMPORAL-HISTORY-STRUCTURAL-NA-{representation_id}",
                family="TEMPORAL_HISTORY",
                kind="STRUCTURAL_NA",
                base_cell_id=cell_id,
                control_cell_id=None,
                base_seed=None,
                control_seed=None,
                metric_ids=(),
                materiality_role="STRUCTURAL_NA",
            )
        )
    for cell in signal_plan.cells:
        if (
            cell.block != "CORE_PAIRED"
            or cell.representation_id != R0_PADDED_RAW
            or cell.applicability != "NUMERIC"
        ):
            continue
        learned_cell = _cell_id(
            "CORE_PAIRED", cell.condition_id, R5_VIEW_SPECIFIC_CORRO_REFIT
        )
        for learned_seed in (0, 1, 2):
            specs.append(
                SignalContrastSpec(
                    contrast_id=(
                        f"REPRESENTATION-R5-MINUS-R0-{cell.condition_id}-"
                        f"SEED-{learned_seed}"
                    ),
                    family="REPRESENTATION_LADDER",
                    kind="REPRESENTATION_GAIN",
                    base_cell_id=cell.cell_id,
                    control_cell_id=learned_cell,
                    base_seed=None,
                    control_seed=learned_seed,
                    metric_ids=INTERPRETABLE_REPRESENTATION_GAIN_METRIC_IDS,
                    materiality_role="REPORT_ONLY",
                )
            )
    return tuple(sorted(specs, key=lambda item: item.contrast_id))


def _expected_numeric_work(
    signal_plan: SignalMatrixPlan, *, historical_seed: int
) -> dict[str, tuple[str, int | None]]:
    result: dict[str, tuple[str, int | None]] = {}
    for cell in signal_plan.numeric_cells:
        for seed in _seeds(
            cell.representation_id, historical_seed=historical_seed
        ):
            result[_work_key(cell.cell_id, seed)] = (cell.cell_id, seed)
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class SignalContrastPlan:
    signal_matrix_digest: str
    historical_seed: int
    specs: tuple[SignalContrastSpec, ...]
    expected_numeric_work_keys: tuple[str, ...]
    structural_na_cell_ids: tuple[str, ...]
    required_pair_control_ids: tuple[str, ...] = REQUIRED_PAIR_CONTROL_IDS
    plan_digest: str | None = None
    schema: str = SIGNAL_CONTRAST_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CONTRAST_PLAN_SCHEMA:
            raise SignalContrastError("unsupported SignalContrastPlan schema")
        object.__setattr__(
            self,
            "signal_matrix_digest",
            _digest(self.signal_matrix_digest, "signal_matrix_digest"),
        )
        canonical_signal_plan = build_signal_matrix_plan()
        if self.signal_matrix_digest != canonical_signal_plan.plan_digest:
            raise SignalContrastError("contrast plan requires the canonical signal matrix")
        if (
            isinstance(self.historical_seed, bool)
            or not isinstance(self.historical_seed, int)
            or self.historical_seed < 0
        ):
            raise SignalContrastError("historical_seed must be a non-negative integer")
        specs = tuple(self.specs)
        if not all(isinstance(item, SignalContrastSpec) for item in specs):
            raise SignalContrastError("contrast plan requires typed specs")
        canonical_specs = _canonical_specs(canonical_signal_plan)
        if tuple(item.to_dict() for item in specs) != tuple(
            item.to_dict() for item in canonical_specs
        ):
            raise SignalContrastError("contrast specs differ from the canonical plan")
        expected_work = tuple(
            _expected_numeric_work(
                canonical_signal_plan, historical_seed=self.historical_seed
            )
        )
        if tuple(self.expected_numeric_work_keys) != expected_work or len(expected_work) != 79:
            raise SignalContrastError("contrast plan must bind the exact 79-work atlas")
        expected_na = tuple(
            cell.cell_id
            for cell in canonical_signal_plan.cells
            if cell.applicability == "STRUCTURAL_NA"
        )
        if tuple(self.structural_na_cell_ids) != expected_na:
            raise SignalContrastError("contrast plan structural N/A coverage drifted")
        if tuple(self.required_pair_control_ids) != REQUIRED_PAIR_CONTROL_IDS:
            raise SignalContrastError("contrast plan pair-control coverage drifted")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "expected_numeric_work_keys", expected_work)
        object.__setattr__(self, "structural_na_cell_ids", expected_na)
        object.__setattr__(self, "required_pair_control_ids", REQUIRED_PAIR_CONTROL_IDS)
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise SignalContrastError("contrast plan digest mismatch")

    @property
    def numeric_contrast_count(self) -> int:
        return sum(item.kind != "STRUCTURAL_NA" for item in self.specs)

    @property
    def structural_na_count(self) -> int:
        return sum(item.kind == "STRUCTURAL_NA" for item in self.specs)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "signal_matrix_digest": self.signal_matrix_digest,
            "historical_seed": self.historical_seed,
            "specs": [item.to_dict() for item in self.specs],
            "expected_numeric_work_keys": list(self.expected_numeric_work_keys),
            "structural_na_cell_ids": list(self.structural_na_cell_ids),
            "required_pair_control_ids": list(self.required_pair_control_ids),
            "numeric_contrast_count": self.numeric_contrast_count,
            "structural_na_count": self.structural_na_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}


def build_signal_contrast_plan(*, historical_seed: int = 0) -> SignalContrastPlan:
    signal_plan = build_signal_matrix_plan()
    expected_work = _expected_numeric_work(
        signal_plan, historical_seed=historical_seed
    )
    return SignalContrastPlan(
        signal_matrix_digest=str(signal_plan.plan_digest),
        historical_seed=historical_seed,
        specs=_canonical_specs(signal_plan),
        expected_numeric_work_keys=tuple(expected_work),
        structural_na_cell_ids=tuple(
            cell.cell_id
            for cell in signal_plan.cells
            if cell.applicability == "STRUCTURAL_NA"
        ),
    )


@dataclass(frozen=True)
class SignalMaterialityThresholds:
    contrast_plan_digest: str
    minimum_transition_degradation_by_metric: Mapping[str, float]
    review_decision_digest: str
    aggregation: str = "MEAN_ACROSS_EXACT_TRANSITION_CELL_SEED_CONTRASTS"
    threshold_digest: str | None = None
    schema: str = SIGNAL_MATERIALITY_THRESHOLDS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_MATERIALITY_THRESHOLDS_SCHEMA:
            raise SignalContrastError("unsupported materiality threshold schema")
        for name in ("contrast_plan_digest", "review_decision_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.aggregation != "MEAN_ACROSS_EXACT_TRANSITION_CELL_SEED_CONTRASTS":
            raise SignalContrastError("transition materiality aggregation is frozen")
        values = {
            _safe_id(key, "transition threshold metric"): _finite(
                value, f"minimum_transition_degradation_by_metric[{key}]"
            )
            for key, value in sorted(
                self.minimum_transition_degradation_by_metric.items()
            )
        }
        if tuple(values) != tuple(sorted(TRANSITION_GATE_METRIC_IDS)):
            raise SignalContrastError(
                "thresholds must exactly cover dynamics_top1 and dynamics_mrr"
            )
        if any(value <= 0.0 or value > 1.0 for value in values.values()):
            raise SignalContrastError(
                "transition degradation thresholds must lie in (0, 1]"
            )
        object.__setattr__(
            self,
            "minimum_transition_degradation_by_metric",
            MappingProxyType(values),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.threshold_digest is None:
            object.__setattr__(self, "threshold_digest", expected)
        elif _digest(self.threshold_digest, "threshold_digest") != expected:
            raise SignalContrastError("materiality threshold digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contrast_plan_digest": self.contrast_plan_digest,
            "minimum_transition_degradation_by_metric": dict(
                self.minimum_transition_degradation_by_metric
            ),
            "review_decision_digest": self.review_decision_digest,
            "aggregation": self.aggregation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "threshold_digest": self.threshold_digest}


@dataclass(frozen=True)
class SignalContrastResult:
    contrast_id: str
    family: str
    status: Literal["OBSERVED", "STRUCTURAL_NA"]
    contrast_digest: str
    metric_effects: Mapping[str, float]
    schema: str = SIGNAL_CONTRAST_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CONTRAST_RESULT_SCHEMA:
            raise SignalContrastError("unsupported SignalContrastResult schema")
        object.__setattr__(self, "contrast_id", _safe_id(self.contrast_id, "contrast_id"))
        if self.family not in SIGNAL_CONTRAST_FAMILIES:
            raise SignalContrastError("unknown result family")
        if self.status not in {"OBSERVED", "STRUCTURAL_NA"}:
            raise SignalContrastError("unknown contrast result status")
        object.__setattr__(
            self, "contrast_digest", _digest(self.contrast_digest, "contrast_digest")
        )
        effects = {
            _safe_id(key, "metric_effect"): _finite(value, f"metric_effects[{key}]")
            for key, value in sorted(self.metric_effects.items())
        }
        if (self.status == "STRUCTURAL_NA") != (not effects):
            raise SignalContrastError("N/A status and numeric effects disagree")
        object.__setattr__(self, "metric_effects", MappingProxyType(effects))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contrast_id": self.contrast_id,
            "family": self.family,
            "status": self.status,
            "contrast_digest": self.contrast_digest,
            "metric_effects": dict(self.metric_effects),
        }


@dataclass(frozen=True)
class SignalContrastGateEvaluation:
    contrast_plan_digest: str
    signal_historical_seed: int
    threshold_digest: str
    formal_atlas_authorization_digest: str
    metric_record_set_digest: str
    pair_control_evidence_set_digest: str
    results: tuple[SignalContrastResult, ...]
    family_numeric_denominators: Mapping[str, int]
    family_structural_na_counts: Mapping[str, int]
    transition_mean_degradation_by_metric: Mapping[str, float]
    transition_threshold_pass_by_metric: Mapping[str, bool]
    gate_status: GateStatus
    evaluation_digest: str | None = None
    schema: str = SIGNAL_CONTRAST_GATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_CONTRAST_GATE_SCHEMA:
            raise SignalContrastError("unsupported signal contrast gate schema")
        for name in (
            "contrast_plan_digest",
            "threshold_digest",
            "formal_atlas_authorization_digest",
            "metric_record_set_digest",
            "pair_control_evidence_set_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if (
            isinstance(self.signal_historical_seed, bool)
            or not isinstance(self.signal_historical_seed, int)
            or self.signal_historical_seed < 0
        ):
            raise SignalContrastError("signal_historical_seed is invalid")
        results = tuple(self.results)
        if not results or not all(isinstance(item, SignalContrastResult) for item in results):
            raise SignalContrastError("gate evaluation requires typed results")
        if len({item.contrast_id for item in results}) != len(results):
            raise SignalContrastError("gate evaluation contains duplicate contrasts")
        canonical_plan = build_signal_contrast_plan(
            historical_seed=self.signal_historical_seed
        )
        if self.contrast_plan_digest != canonical_plan.plan_digest or {
            item.contrast_id for item in results
        } != {item.contrast_id for item in canonical_plan.specs}:
            raise SignalContrastError(
                "gate evaluation differs from canonical contrast coverage"
            )
        object.__setattr__(self, "results", tuple(sorted(results, key=lambda item: item.contrast_id)))
        numeric = _checked_family_counts(self.family_numeric_denominators, "numeric")
        structural = _checked_family_counts(
            self.family_structural_na_counts, "structural N/A"
        )
        expected_numeric = {
            family: sum(
                item.family == family and item.status == "OBSERVED"
                for item in results
            )
            for family in SIGNAL_CONTRAST_FAMILIES
        }
        expected_structural = {
            family: sum(
                item.family == family and item.status == "STRUCTURAL_NA"
                for item in results
            )
            for family in SIGNAL_CONTRAST_FAMILIES
        }
        if numeric != dict(sorted(expected_numeric.items())) or structural != dict(
            sorted(expected_structural.items())
        ):
            raise SignalContrastError("family denominators disagree with typed results")
        object.__setattr__(self, "family_numeric_denominators", MappingProxyType(numeric))
        object.__setattr__(self, "family_structural_na_counts", MappingProxyType(structural))
        effects = {
            key: _finite(value, f"transition_mean_degradation_by_metric[{key}]")
            for key, value in sorted(self.transition_mean_degradation_by_metric.items())
        }
        decisions = dict(sorted(self.transition_threshold_pass_by_metric.items()))
        if tuple(effects) != tuple(sorted(TRANSITION_GATE_METRIC_IDS)) or set(decisions) != set(effects):
            raise SignalContrastError("transition gate metric coverage is incomplete")
        if any(type(value) is not bool for value in decisions.values()):
            raise SignalContrastError("transition threshold decisions must be boolean")
        expected_status: GateStatus = (
            "PASS" if all(decisions.values()) else "NO_GO_TRANSITION_SIGNAL"
        )
        if self.gate_status != expected_status:
            raise SignalContrastError("gate status disagrees with transition thresholds")
        object.__setattr__(self, "transition_mean_degradation_by_metric", MappingProxyType(effects))
        object.__setattr__(self, "transition_threshold_pass_by_metric", MappingProxyType(decisions))
        expected = sha256_json(self._payload_without_digest())
        if self.evaluation_digest is None:
            object.__setattr__(self, "evaluation_digest", expected)
        elif _digest(self.evaluation_digest, "evaluation_digest") != expected:
            raise SignalContrastError("signal contrast evaluation digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contrast_plan_digest": self.contrast_plan_digest,
            "signal_historical_seed": self.signal_historical_seed,
            "threshold_digest": self.threshold_digest,
            "formal_atlas_authorization_digest": self.formal_atlas_authorization_digest,
            "metric_record_set_digest": self.metric_record_set_digest,
            "pair_control_evidence_set_digest": self.pair_control_evidence_set_digest,
            "results": [item.to_dict() for item in self.results],
            "family_numeric_denominators": dict(self.family_numeric_denominators),
            "family_structural_na_counts": dict(self.family_structural_na_counts),
            "transition_mean_degradation_by_metric": dict(
                self.transition_mean_degradation_by_metric
            ),
            "transition_threshold_pass_by_metric": dict(
                self.transition_threshold_pass_by_metric
            ),
            "gate_status": self.gate_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "evaluation_digest": self.evaluation_digest}

    def to_public_dict(self) -> dict[str, Any]:
        """Publish only aggregate gate decisions, never metric rows or identities."""

        payload = {
            "schema": "policy-learnware.v03-public-signal-contrast-gate.v0",
            "contrast_plan_digest": self.contrast_plan_digest,
            "threshold_digest": self.threshold_digest,
            "formal_atlas_authorization_digest": (
                self.formal_atlas_authorization_digest
            ),
            "metric_record_set_digest": self.metric_record_set_digest,
            "pair_control_evidence_set_digest": (
                self.pair_control_evidence_set_digest
            ),
            "family_numeric_denominators": dict(self.family_numeric_denominators),
            "family_structural_na_counts": dict(self.family_structural_na_counts),
            "transition_mean_degradation_by_metric": dict(
                self.transition_mean_degradation_by_metric
            ),
            "transition_threshold_pass_by_metric": dict(
                self.transition_threshold_pass_by_metric
            ),
            "gate_status": self.gate_status,
            "private_contrast_rows_withheld": True,
            "private_evaluation_digest": self.evaluation_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


def _checked_family_counts(value: Mapping[str, int], where: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(SIGNAL_CONTRAST_FAMILIES):
        raise SignalContrastError(f"{where} family counts have incomplete coverage")
    result: dict[str, int] = {}
    for family, count in sorted(value.items()):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SignalContrastError(f"{where} family count is invalid")
        result[family] = count
    return result


def _validate_record_schedule(
    plan: SignalContrastPlan,
    metric_records: Mapping[str, SignalMetricRecord],
) -> Mapping[str, SignalMetricRecord]:
    if not isinstance(metric_records, Mapping):
        raise SignalContrastError("metric_records must be a mapping")
    records = dict(metric_records)
    if set(records) != set(plan.expected_numeric_work_keys):
        raise SignalContrastError("metric records must exactly cover the 79-work atlas")
    signal_plan = build_signal_matrix_plan()
    expected = _expected_numeric_work(
        signal_plan, historical_seed=plan.historical_seed
    )
    for work_key, record in records.items():
        if not isinstance(record, SignalMetricRecord):
            raise SignalContrastError("metric_records must contain typed records")
        cell_id, seed = expected[work_key]
        cell = signal_plan.cell(cell_id)
        if (
            record.cell_id != cell_id
            or record.view_or_condition_id != cell.condition_id
            or record.representation_id != cell.representation_id
            or record.representation_seed != seed
        ):
            raise SignalContrastError(
                f"metric record identity differs from work key {work_key!r}"
            )
    return MappingProxyType(dict(sorted(records.items())))


def _validate_pair_control_evidence(
    plan: SignalContrastPlan,
    evidence: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], str]:
    if not isinstance(evidence, Mapping) or set(evidence) != set(
        plan.required_pair_control_ids
    ):
        raise SignalContrastError(
            "pair-control evidence must cover schema-collision and exact-repeat"
        )
    frozen: dict[str, tuple[str, ...]] = {}
    for control_id in plan.required_pair_control_ids:
        values = tuple(
            sorted(
                _digest(item, f"pair_control_evidence[{control_id}]")
                for item in evidence[control_id]
            )
        )
        if not values or len(set(values)) != len(values):
            raise SignalContrastError("pair-control evidence must be non-empty and unique")
        frozen[control_id] = values
    return frozen, sha256_json(
        {
            "schema": "policy-learnware.v03-pair-control-evidence-set.v0",
            "contrast_plan_digest": plan.plan_digest,
            "evidence_digests": {key: list(value) for key, value in frozen.items()},
        }
    )


def pair_control_evidence_set_digest(
    plan: SignalContrastPlan, evidence: Mapping[str, Sequence[str]]
) -> str:
    """Validate and digest exact schema-collision/exact-repeat evidence coverage."""

    if not isinstance(plan, SignalContrastPlan):
        raise SignalContrastError("pair-control evidence requires a typed plan")
    return _validate_pair_control_evidence(plan, evidence)[1]


def signal_metric_record_set_digest(
    plan: SignalContrastPlan,
    metric_records: Mapping[str, SignalMetricRecord],
) -> str:
    """Validate the exact 79 records and return their canonical set digest."""

    if not isinstance(plan, SignalContrastPlan):
        raise SignalContrastError("metric record set requires a typed plan")
    records = _validate_record_schedule(plan, metric_records)
    return sha256_json(
        {
            "schema": "policy-learnware.v03-signal-metric-record-set.v0",
            "contrast_plan_digest": plan.plan_digest,
            "record_digests_by_work_key": {
                key: record.record_digest for key, record in records.items()
            },
        }
    )


def _record_for(
    records: Mapping[str, SignalMetricRecord], cell_id: str, seed: int | None
) -> SignalMetricRecord:
    try:
        return records[_work_key(cell_id, seed)]
    except KeyError as error:  # pragma: no cover - exact schedule already checked
        raise SignalContrastError("contrast record is absent") from error


def evaluate_formal_signal_contrasts(
    *,
    plan: SignalContrastPlan,
    thresholds: SignalMaterialityThresholds,
    metric_records: Mapping[str, SignalMetricRecord],
    pair_control_evidence_digests: Mapping[str, Sequence[str]],
    formal_atlas_authorization: Any,
) -> SignalContrastGateEvaluation:
    """Replay the canonical contrast table under one external formal authority."""

    from .signal_atlas import FormalSignalAtlasAuthorization

    if not isinstance(plan, SignalContrastPlan):
        raise SignalContrastError("formal evaluation requires a typed contrast plan")
    if not isinstance(thresholds, SignalMaterialityThresholds):
        raise SignalContrastError("formal evaluation requires typed thresholds")
    if not isinstance(formal_atlas_authorization, FormalSignalAtlasAuthorization):
        raise SignalContrastError(
            "formal evaluation requires FormalSignalAtlasAuthorization"
        )
    if (
        thresholds.contrast_plan_digest != plan.plan_digest
        or formal_atlas_authorization.signal_contrast_plan_digest != plan.plan_digest
        or formal_atlas_authorization.freeze_manifest.signal_contrast_plan_digest
        != plan.plan_digest
        or formal_atlas_authorization.signal_materiality_threshold_digest
        != thresholds.threshold_digest
        or formal_atlas_authorization.freeze_manifest.signal_materiality_threshold_digest
        != thresholds.threshold_digest
        or thresholds.review_decision_digest
        != formal_atlas_authorization.freeze_manifest.review_decisions_digest
    ):
        raise SignalContrastError(
            "contrast plan/thresholds differ from the externally reviewed freeze"
        )
    records = _validate_record_schedule(plan, metric_records)
    _, pair_evidence_digest = _validate_pair_control_evidence(
        plan, pair_control_evidence_digests
    )
    results: list[SignalContrastResult] = []
    transition_effects: dict[str, list[float]] = {
        metric_id: [] for metric_id in TRANSITION_GATE_METRIC_IDS
    }
    for spec in plan.specs:
        if spec.kind == "STRUCTURAL_NA":
            cell = build_signal_matrix_plan().cell(spec.base_cell_id)
            if cell.applicability != "STRUCTURAL_NA":
                raise SignalContrastError("temporal N/A points to a numeric cell")
            results.append(
                SignalContrastResult(
                    contrast_id=spec.contrast_id,
                    family=spec.family,
                    status="STRUCTURAL_NA",
                    contrast_digest=str(cell.cell_digest),
                    metric_effects={},
                )
            )
            continue
        base = _record_for(records, spec.base_cell_id, spec.base_seed)
        control = _record_for(
            records, str(spec.control_cell_id), spec.control_seed
        )
        try:
            if spec.kind == "PAIRED":
                contrast = paired_signal_contrast(
                    base, control, metric_ids=spec.metric_ids
                )
                effects = dict(contrast.metric_deltas)
                contrast_digest = contrast.contrast_digest
            else:
                contrast = representation_gain_contrast(
                    base, control, metric_ids=spec.metric_ids
                )
                effects = dict(contrast.metric_gains)
                contrast_digest = contrast.contrast_digest
        except SignalMetricError as error:
            raise SignalContrastError(
                f"contrast {spec.contrast_id!r} is not a valid paired comparison: {error}"
            ) from error
        if spec.materiality_role == "TRANSITION_MINIMUM":
            for metric_id in TRANSITION_GATE_METRIC_IDS:
                transition_effects[metric_id].append(effects[metric_id])
        results.append(
            SignalContrastResult(
                contrast_id=spec.contrast_id,
                family=spec.family,
                status="OBSERVED",
                contrast_digest=contrast_digest,
                metric_effects=effects,
            )
        )
    if any(len(values) != 10 for values in transition_effects.values()):
        raise SignalContrastError("transition gate must contain exactly ten paired effects")
    means = {
        metric_id: float(np.mean(values))
        for metric_id, values in transition_effects.items()
    }
    decisions = {
        metric_id: means[metric_id]
        >= thresholds.minimum_transition_degradation_by_metric[metric_id]
        for metric_id in TRANSITION_GATE_METRIC_IDS
    }
    numeric_counts = {
        family: sum(
            item.family == family and item.kind != "STRUCTURAL_NA"
            for item in plan.specs
        )
        for family in SIGNAL_CONTRAST_FAMILIES
    }
    structural_counts = {
        family: sum(
            item.family == family and item.kind == "STRUCTURAL_NA"
            for item in plan.specs
        )
        for family in SIGNAL_CONTRAST_FAMILIES
    }
    metric_set_digest = signal_metric_record_set_digest(plan, records)
    return SignalContrastGateEvaluation(
        contrast_plan_digest=str(plan.plan_digest),
        signal_historical_seed=plan.historical_seed,
        threshold_digest=str(thresholds.threshold_digest),
        formal_atlas_authorization_digest=str(
            formal_atlas_authorization.authorization_digest
        ),
        metric_record_set_digest=metric_set_digest,
        pair_control_evidence_set_digest=pair_evidence_digest,
        results=tuple(results),
        family_numeric_denominators=numeric_counts,
        family_structural_na_counts=structural_counts,
        transition_mean_degradation_by_metric=means,
        transition_threshold_pass_by_metric=decisions,
        gate_status=("PASS" if all(decisions.values()) else "NO_GO_TRANSITION_SIGNAL"),
    )


__all__ = [
    "INTERPRETABLE_REPRESENTATION_GAIN_METRIC_IDS",
    "REQUIRED_PAIR_CONTROL_IDS",
    "SIGNAL_CONTRAST_FAMILIES",
    "SIGNAL_CONTRAST_GATE_SCHEMA",
    "SIGNAL_CONTRAST_PLAN_SCHEMA",
    "SIGNAL_CONTRAST_RESULT_SCHEMA",
    "SIGNAL_CONTRAST_SPEC_SCHEMA",
    "SIGNAL_MATERIALITY_THRESHOLDS_SCHEMA",
    "TRANSITION_GATE_METRIC_IDS",
    "SignalContrastError",
    "SignalContrastGateEvaluation",
    "SignalContrastPlan",
    "SignalContrastResult",
    "SignalContrastSpec",
    "SignalMaterialityThresholds",
    "build_signal_contrast_plan",
    "evaluate_formal_signal_contrasts",
    "pair_control_evidence_set_digest",
    "signal_metric_record_set_digest",
]
