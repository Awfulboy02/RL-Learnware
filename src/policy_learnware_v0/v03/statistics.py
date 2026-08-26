"""Frozen formal statistics for the v0.3 confirmatory lifecycle.

The module deliberately starts *after* the public-ranking barrier and an
external oracle-owner release receipt.  It never loads an oracle itself.  Its
only numeric input is a digest-bound tuple of typed paired rows.  The actual
resampling, simultaneous intervals and multiplicity audit reuse the v0.2
deterministic implementations so the two branches cannot silently disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from ..hashing import sha256_json
from ..v02.metrics import HierarchicalValue, aggregate_hierarchy
from ..v02.statistics import (
    HierarchicalBootstrapResult,
    bootstrap_max_t_intervals,
    centered_one_sided_p_value,
    derive_bootstrap_seed,
    hierarchical_paired_difference_bootstrap,
    holm_bonferroni,
)
from .preflight import ORACLE_OWNER, OracleUnlockHandoff, PreExperimentFreezeManifest
from .schemas import checked_digest, checked_safe_id, strict_mapping


ENDPOINT_SCHEMA = "policy-learnware.v03-statistics-endpoint.v0"
CONTRAST_SCHEMA = "policy-learnware.v03-statistics-contrast.v0"
MULTIPLICITY_PLAN_SCHEMA = "policy-learnware.v03-multiplicity-plan.v0"
STATISTICS_PLAN_SCHEMA = "policy-learnware.v03-statistics-plan.v0"
STATISTICS_INPUT_ROW_SCHEMA = "policy-learnware.v03-statistics-input-row.v0"
STATISTICS_INPUT_SCHEMA = "policy-learnware.v03-statistics-input.v1"
STATISTICS_RESULT_SCHEMA = "policy-learnware.v03-statistics-result.v1"

N_A_POLICY = "EXCLUDE_EXPLICIT_N_A_FROM_DENOMINATOR_AND_REPORT_COVERAGE"
RESAMPLING_CONTRACT = "task_axis_context_episode_bank_hierarchical_equal_weight"
MAX_T_METHOD = "paired_studentized_bootstrap_max-T"
HOLM_METHOD = "Holm-Bonferroni-centered-one-sided-bootstrap-audit"
FORMAL_CONTRAST_FAMILY_IDS = (
    "SCHEMA",
    "REWARD_GOAL",
    "TRANSITION_MECHANISM",
    "TEMPORAL_HISTORY",
    "REPRESENTATION_LADDER",
    "POLICY_SELECTION_LINKAGE",
)


class V03StatisticsError(ValueError):
    """A plan, released input or result violates the frozen protocol."""


def _strict(value: Mapping[str, Any], fields: set[str], where: str) -> Mapping[str, Any]:
    try:
        return strict_mapping(value, fields, where)
    except ValueError as error:
        raise V03StatisticsError(str(error)) from error


def _id(value: Any, where: str) -> str:
    try:
        return checked_safe_id(value, where)
    except ValueError as error:
        raise V03StatisticsError(str(error)) from error


def _digest(value: Any, where: str) -> str:
    try:
        return checked_digest(value, where)
    except ValueError as error:
        raise V03StatisticsError(str(error)) from error


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V03StatisticsError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V03StatisticsError(f"{where} must be finite")
    return result


def _positive_int(value: Any, where: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise V03StatisticsError(f"{where} must be an integer >= {minimum}")
    return value


def _probability(value: Any, where: str) -> float:
    result = _finite(value, where)
    if not 0.0 < result < 1.0:
        raise V03StatisticsError(f"{where} must lie strictly in (0, 1)")
    return result


@dataclass(frozen=True)
class StatisticsEndpoint:
    endpoint_id: str
    metric_id: str
    higher_is_better: bool
    n_a_policy: str = N_A_POLICY
    schema: str = ENDPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENDPOINT_SCHEMA:
            raise V03StatisticsError("unsupported StatisticsEndpoint schema")
        object.__setattr__(self, "endpoint_id", _id(self.endpoint_id, "endpoint_id"))
        object.__setattr__(self, "metric_id", _id(self.metric_id, "metric_id"))
        if type(self.higher_is_better) is not bool:
            raise V03StatisticsError("higher_is_better must be boolean")
        if self.n_a_policy != N_A_POLICY:
            raise V03StatisticsError("formal endpoints must exclude explicit N/A rows")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StatisticsEndpoint":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "statistics endpoint")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class StatisticsContrast:
    hypothesis_id: str
    contrast_family_id: str
    endpoint_id: str
    left_method_id: str
    right_method_id: str
    null_boundary: float = 0.0
    schema: str = CONTRAST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONTRAST_SCHEMA:
            raise V03StatisticsError("unsupported StatisticsContrast schema")
        for name in (
            "hypothesis_id",
            "contrast_family_id",
            "endpoint_id",
            "left_method_id",
            "right_method_id",
        ):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        if self.contrast_family_id not in FORMAL_CONTRAST_FAMILY_IDS:
            raise V03StatisticsError("unknown formal contrast family")
        if self.left_method_id == self.right_method_id:
            raise V03StatisticsError("a contrast requires two distinct methods")
        object.__setattr__(self, "null_boundary", _finite(self.null_boundary, "null_boundary"))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StatisticsContrast":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "statistics contrast")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class MultiplicityFamilyPlan:
    contrast_family_id: str
    hypothesis_ids: tuple[str, ...]
    simultaneous_interval_method: str = MAX_T_METHOD
    multiplicity_audit_method: str = HOLM_METHOD
    schema: str = MULTIPLICITY_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MULTIPLICITY_PLAN_SCHEMA:
            raise V03StatisticsError("unsupported MultiplicityFamilyPlan schema")
        family_id = _id(self.contrast_family_id, "contrast_family_id")
        if family_id not in FORMAL_CONTRAST_FAMILY_IDS:
            raise V03StatisticsError("unknown formal contrast family")
        hypotheses = tuple(_id(item, "hypothesis_ids[]") for item in self.hypothesis_ids)
        if not hypotheses or len(set(hypotheses)) != len(hypotheses):
            raise V03StatisticsError("a multiplicity family requires unique hypotheses")
        if self.simultaneous_interval_method != MAX_T_METHOD:
            raise V03StatisticsError("formal simultaneous intervals must use bootstrap max-T")
        if self.multiplicity_audit_method != HOLM_METHOD:
            raise V03StatisticsError("formal multiplicity audit must use Holm")
        object.__setattr__(self, "contrast_family_id", family_id)
        object.__setattr__(self, "hypothesis_ids", tuple(sorted(hypotheses)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "contrast_family_id": self.contrast_family_id,
            "hypothesis_ids": list(self.hypothesis_ids),
            "simultaneous_interval_method": self.simultaneous_interval_method,
            "multiplicity_audit_method": self.multiplicity_audit_method,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultiplicityFamilyPlan":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "multiplicity family plan")
        return cls(
            **{
                field: tuple(data[field]) if field == "hypothesis_ids" else data[field]
                for field in fields
            }
        )


@dataclass(frozen=True)
class FormalStatisticsPlan:
    endpoints: tuple[StatisticsEndpoint, ...]
    contrasts: tuple[StatisticsContrast, ...]
    multiplicity_families: tuple[MultiplicityFamilyPlan, ...]
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    alpha: float = 0.05
    seed_namespace: str = "policy-learnware-v03-formal-statistics"
    resampling_contract: str = RESAMPLING_CONTRACT
    n_a_policy: str = N_A_POLICY
    formal_confirmatory: bool = False
    registered_n_a_reasons: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    schema: str = STATISTICS_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATISTICS_PLAN_SCHEMA:
            raise V03StatisticsError("unsupported FormalStatisticsPlan schema")
        endpoints = tuple(self.endpoints)
        contrasts = tuple(self.contrasts)
        families = tuple(self.multiplicity_families)
        if not endpoints or not all(isinstance(item, StatisticsEndpoint) for item in endpoints):
            raise V03StatisticsError("statistics plan requires typed endpoints")
        if not contrasts or not all(isinstance(item, StatisticsContrast) for item in contrasts):
            raise V03StatisticsError("statistics plan requires typed contrasts")
        if not families or not all(isinstance(item, MultiplicityFamilyPlan) for item in families):
            raise V03StatisticsError("statistics plan requires typed multiplicity families")
        endpoint_ids = tuple(item.endpoint_id for item in endpoints)
        hypothesis_ids = tuple(item.hypothesis_id for item in contrasts)
        family_ids = tuple(item.contrast_family_id for item in families)
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise V03StatisticsError("endpoint IDs must be unique")
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise V03StatisticsError("hypothesis IDs must be unique")
        if len(set(family_ids)) != len(family_ids):
            raise V03StatisticsError("multiplicity family IDs must be unique")
        if any(item.endpoint_id not in set(endpoint_ids) for item in contrasts):
            raise V03StatisticsError("every contrast must reference a frozen endpoint")
        membership = {
            hypothesis_id: family.contrast_family_id
            for family in families
            for hypothesis_id in family.hypothesis_ids
        }
        if len(membership) != sum(len(item.hypothesis_ids) for item in families):
            raise V03StatisticsError("a hypothesis cannot occur in multiple families")
        if set(membership) != set(hypothesis_ids):
            raise V03StatisticsError("multiplicity plans must cover every hypothesis exactly once")
        if any(membership[item.hypothesis_id] != item.contrast_family_id for item in contrasts):
            raise V03StatisticsError("contrast and multiplicity family assignments disagree")
        object.__setattr__(self, "bootstrap_resamples", _positive_int(self.bootstrap_resamples, "bootstrap_resamples", minimum=2))
        object.__setattr__(self, "confidence_level", _probability(self.confidence_level, "confidence_level"))
        object.__setattr__(self, "alpha", _probability(self.alpha, "alpha"))
        object.__setattr__(self, "seed_namespace", _id(self.seed_namespace, "seed_namespace"))
        if self.resampling_contract != RESAMPLING_CONTRACT:
            raise V03StatisticsError("unexpected hierarchical resampling contract")
        if self.n_a_policy != N_A_POLICY or any(item.n_a_policy != N_A_POLICY for item in endpoints):
            raise V03StatisticsError("N/A rows must be excluded and coverage reported")
        if type(self.formal_confirmatory) is not bool:
            raise V03StatisticsError("formal_confirmatory must be boolean")
        if not isinstance(self.registered_n_a_reasons, Mapping):
            raise V03StatisticsError("registered_n_a_reasons must be a mapping")
        registered_reasons: dict[str, tuple[str, ...]] = {}
        for hypothesis_id, reasons in sorted(self.registered_n_a_reasons.items()):
            canonical_hypothesis = _id(
                hypothesis_id, "registered_n_a_reasons hypothesis"
            )
            if canonical_hypothesis not in set(hypothesis_ids):
                raise V03StatisticsError(
                    "registered N/A reason refers to an unknown hypothesis"
                )
            if isinstance(reasons, (str, bytes)) or not isinstance(
                reasons, (tuple, list)
            ):
                raise V03StatisticsError(
                    "registered N/A reasons must be a sequence"
                )
            canonical_reasons = tuple(
                sorted(_id(reason, "registered N/A reason") for reason in reasons)
            )
            if len(set(canonical_reasons)) != len(canonical_reasons):
                raise V03StatisticsError("registered N/A reasons must be unique")
            registered_reasons[canonical_hypothesis] = canonical_reasons
        if self.formal_confirmatory:
            if set(family_ids) != set(FORMAL_CONTRAST_FAMILY_IDS):
                raise V03StatisticsError(
                    "formal confirmatory plan must cover all six contrast families exactly"
                )
            if set(registered_reasons) != set(hypothesis_ids):
                raise V03StatisticsError(
                    "formal confirmatory plan must preregister N/A reasons for every hypothesis"
                )
        object.__setattr__(self, "endpoints", tuple(sorted(endpoints, key=lambda item: item.endpoint_id)))
        object.__setattr__(self, "contrasts", tuple(sorted(contrasts, key=lambda item: item.hypothesis_id)))
        object.__setattr__(self, "multiplicity_families", tuple(sorted(families, key=lambda item: item.contrast_family_id)))
        object.__setattr__(
            self,
            "registered_n_a_reasons",
            MappingProxyType(registered_reasons),
        )

    @property
    def plan_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "endpoints": [item.to_dict() for item in self.endpoints],
            "contrasts": [item.to_dict() for item in self.contrasts],
            "multiplicity_families": [item.to_dict() for item in self.multiplicity_families],
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence_level": self.confidence_level,
            "alpha": self.alpha,
            "seed_namespace": self.seed_namespace,
            "resampling_contract": self.resampling_contract,
            "n_a_policy": self.n_a_policy,
            "formal_confirmatory": self.formal_confirmatory,
            "registered_n_a_reasons": {
                key: list(value)
                for key, value in self.registered_n_a_reasons.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalStatisticsPlan":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "formal statistics plan")
        return cls(
            endpoints=tuple(StatisticsEndpoint.from_dict(item) for item in data["endpoints"]),
            contrasts=tuple(StatisticsContrast.from_dict(item) for item in data["contrasts"]),
            multiplicity_families=tuple(MultiplicityFamilyPlan.from_dict(item) for item in data["multiplicity_families"]),
            bootstrap_resamples=data["bootstrap_resamples"],
            confidence_level=data["confidence_level"],
            alpha=data["alpha"],
            seed_namespace=data["seed_namespace"],
            resampling_contract=data["resampling_contract"],
            n_a_policy=data["n_a_policy"],
            formal_confirmatory=data["formal_confirmatory"],
            registered_n_a_reasons={
                key: tuple(value)
                for key, value in data["registered_n_a_reasons"].items()
            },
            schema=data["schema"],
        )


RowStatus = Literal["OBSERVED", "N_A"]


@dataclass(frozen=True)
class FrozenContrastInputRow:
    hypothesis_id: str
    task_id: str
    axis_id: str
    context_id: str
    observation_id: str
    status: RowStatus
    left_value: float | None
    right_value: float | None
    n_a_reason: str | None = None
    schema: str = STATISTICS_INPUT_ROW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATISTICS_INPUT_ROW_SCHEMA:
            raise V03StatisticsError("unsupported FrozenContrastInputRow schema")
        for name in ("hypothesis_id", "task_id", "axis_id", "context_id", "observation_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        if self.status == "OBSERVED":
            object.__setattr__(self, "left_value", _finite(self.left_value, "left_value"))
            object.__setattr__(self, "right_value", _finite(self.right_value, "right_value"))
            if self.n_a_reason is not None:
                raise V03StatisticsError("observed rows cannot carry an N/A reason")
        elif self.status == "N_A":
            if self.left_value is not None or self.right_value is not None:
                raise V03StatisticsError("N/A rows cannot carry numeric values")
            object.__setattr__(self, "n_a_reason", _id(self.n_a_reason, "n_a_reason"))
        else:
            raise V03StatisticsError("row status must be OBSERVED or N_A")

    @property
    def hierarchy_key(self) -> tuple[str, str, str, str]:
        return self.task_id, self.axis_id, self.context_id, self.observation_id

    @property
    def row_key(self) -> tuple[str, str, str, str, str]:
        return (self.hypothesis_id, *self.hierarchy_key)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenContrastInputRow":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "frozen statistics input row")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class FrozenStatisticsInput:
    run_id: str
    preexperiment_freeze_manifest_digest: str
    statistics_plan_digest: str
    public_ranking_barrier_digest: str
    oracle_unlock_handoff_digest: str
    oracle_release_receipt_digest: str
    oracle_evidence_manifest_digest: str
    rows: tuple[FrozenContrastInputRow, ...]
    oracle_release_verified: bool = True
    oracle_owner: str = ORACLE_OWNER
    v03_oracle_write_capability: bool = False
    schema: str = STATISTICS_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATISTICS_INPUT_SCHEMA:
            raise V03StatisticsError("unsupported FrozenStatisticsInput schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "preexperiment_freeze_manifest_digest",
            "statistics_plan_digest",
            "public_ranking_barrier_digest",
            "oracle_unlock_handoff_digest",
            "oracle_release_receipt_digest",
            "oracle_evidence_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.oracle_owner != ORACLE_OWNER:
            raise V03StatisticsError(f"oracle owner must be {ORACLE_OWNER}")
        if self.oracle_release_verified is not True:
            raise V03StatisticsError("formal statistics require an external oracle release receipt")
        if self.v03_oracle_write_capability is not False:
            raise V03StatisticsError("v0.3 cannot acquire oracle write capability")
        expected_handoff = OracleUnlockHandoff(
            run_id=self.run_id,
            freeze_manifest_digest=self.preexperiment_freeze_manifest_digest,
            public_ranking_barrier_digest=self.public_ranking_barrier_digest,
        )
        if self.oracle_unlock_handoff_digest != expected_handoff.handoff_digest:
            raise V03StatisticsError("oracle handoff does not bind the public-ranking barrier")
        rows = tuple(self.rows)
        if not rows or not all(isinstance(item, FrozenContrastInputRow) for item in rows):
            raise V03StatisticsError("statistics input requires typed rows")
        keys = tuple(item.row_key for item in rows)
        if len(set(keys)) != len(keys):
            raise V03StatisticsError("statistics input row keys must be unique")
        object.__setattr__(self, "rows", tuple(sorted(rows, key=lambda item: item.row_key)))

    @property
    def input_manifest_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "preexperiment_freeze_manifest_digest": self.preexperiment_freeze_manifest_digest,
            "statistics_plan_digest": self.statistics_plan_digest,
            "public_ranking_barrier_digest": self.public_ranking_barrier_digest,
            "oracle_unlock_handoff_digest": self.oracle_unlock_handoff_digest,
            "oracle_release_receipt_digest": self.oracle_release_receipt_digest,
            "oracle_evidence_manifest_digest": self.oracle_evidence_manifest_digest,
            "rows": [item.to_dict() for item in self.rows],
            "oracle_release_verified": self.oracle_release_verified,
            "oracle_owner": self.oracle_owner,
            "v03_oracle_write_capability": self.v03_oracle_write_capability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenStatisticsInput":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "frozen statistics input")
        return cls(
            **{
                field: tuple(FrozenContrastInputRow.from_dict(item) for item in data[field])
                if field == "rows"
                else data[field]
                for field in fields
            }
        )


@dataclass(frozen=True)
class FormalStatisticsResult:
    run_id: str
    statistics_plan_digest: str
    statistics_input_digest: str
    preexperiment_freeze_manifest_digest: str
    public_ranking_barrier_digest: str
    oracle_unlock_handoff_digest: str
    oracle_evidence_manifest_digest: str
    contrast_results: Mapping[str, Mapping[str, Any]]
    family_results: Mapping[str, Mapping[str, Any]]
    schema: str = STATISTICS_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STATISTICS_RESULT_SCHEMA:
            raise V03StatisticsError("unsupported FormalStatisticsResult schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "statistics_plan_digest",
            "statistics_input_digest",
            "preexperiment_freeze_manifest_digest",
            "public_ranking_barrier_digest",
            "oracle_unlock_handoff_digest",
            "oracle_evidence_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        contrasts = {key: dict(value) for key, value in self.contrast_results.items()}
        families = {key: dict(value) for key, value in self.family_results.items()}
        if not contrasts or not families:
            raise V03StatisticsError("statistics result cannot be empty")
        if any(_id(key, "contrast_results key") != key for key in contrasts):
            raise V03StatisticsError("invalid contrast result key")
        if any(_id(key, "family_results key") != key for key in families):
            raise V03StatisticsError("invalid family result key")
        object.__setattr__(
            self,
            "contrast_results",
            MappingProxyType(
                {
                    key: MappingProxyType(value)
                    for key, value in sorted(contrasts.items())
                }
            ),
        )
        object.__setattr__(
            self,
            "family_results",
            MappingProxyType(
                {
                    key: MappingProxyType(value)
                    for key, value in sorted(families.items())
                }
            ),
        )

    @property
    def result_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "statistics_plan_digest": self.statistics_plan_digest,
            "statistics_input_digest": self.statistics_input_digest,
            "preexperiment_freeze_manifest_digest": self.preexperiment_freeze_manifest_digest,
            "public_ranking_barrier_digest": self.public_ranking_barrier_digest,
            "oracle_unlock_handoff_digest": self.oracle_unlock_handoff_digest,
            "oracle_evidence_manifest_digest": self.oracle_evidence_manifest_digest,
            "contrast_results": {key: dict(value) for key, value in self.contrast_results.items()},
            "family_results": {key: dict(value) for key, value in self.family_results.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalStatisticsResult":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "formal statistics result")
        return cls(**{field: data[field] for field in fields})


def _hierarchical_values(
    rows: Sequence[FrozenContrastInputRow],
    *,
    side: Literal["left", "right"],
    higher_is_better: bool,
) -> tuple[HierarchicalValue, ...]:
    multiplier = 1.0 if higher_is_better else -1.0
    attribute = "left_value" if side == "left" else "right_value"
    return tuple(
        HierarchicalValue(
            task_id=row.task_id,
            axis_id=row.axis_id,
            context_id=row.context_id,
            observation_id=row.observation_id,
            value=multiplier * float(getattr(row, attribute)),
        )
        for row in rows
    )


def _task_sensitivity(
    left: Sequence[HierarchicalValue], right: Sequence[HierarchicalValue]
) -> dict[str, Any]:
    differences = tuple(
        HierarchicalValue(
            task_id=left_row.task_id,
            axis_id=left_row.axis_id,
            context_id=left_row.context_id,
            observation_id=left_row.observation_id,
            value=left_row.value - right_row.value,
        )
        for left_row, right_row in zip(left, right)
    )
    aggregate = aggregate_hierarchy(differences)
    per_task = {
        item.task_id: item.mean for item in aggregate.task_aggregates
    }
    ordered = sorted(per_task.items(), key=lambda item: (item[1], item[0]))
    leave_one_out: dict[str, float] = {}
    if len(per_task) > 1:
        for omitted in sorted(per_task):
            retained = tuple(row for row in differences if row.task_id != omitted)
            leave_one_out[omitted] = aggregate_hierarchy(retained).macro_mean
    return {
        "per_task_effects": dict(sorted(per_task.items())),
        "positive_task_count": sum(value > 0.0 for value in per_task.values()),
        "zero_task_count": sum(value == 0.0 for value in per_task.values()),
        "negative_task_count": sum(value < 0.0 for value in per_task.values()),
        "worst_task_id": ordered[0][0],
        "worst_task_effect": ordered[0][1],
        "best_task_id": ordered[-1][0],
        "best_task_effect": ordered[-1][1],
        "leave_one_task_out_effects": dict(sorted(leave_one_out.items())),
        "leave_one_task_out_status": "COMPUTED" if leave_one_out else "N_A",
    }


def compute_formal_statistics(
    *,
    plan: FormalStatisticsPlan,
    freeze_manifest: PreExperimentFreezeManifest,
    frozen_input: FrozenStatisticsInput,
) -> FormalStatisticsResult:
    """Compute all frozen contrasts without any direct oracle access.

    Positive effects always favour the registered left method.  For a
    lower-is-better endpoint both sides are sign-flipped before applying the
    v0.2 paired difference implementation.
    """

    if not isinstance(plan, FormalStatisticsPlan):
        raise V03StatisticsError("plan must be a FormalStatisticsPlan")
    if not isinstance(freeze_manifest, PreExperimentFreezeManifest):
        raise V03StatisticsError("freeze_manifest has the wrong type")
    if not isinstance(frozen_input, FrozenStatisticsInput):
        raise V03StatisticsError("frozen_input has the wrong type")
    if not freeze_manifest.formal_run_authorized:
        raise V03StatisticsError("formal statistics require external pre-experiment authority")
    if freeze_manifest.statistics_plan_digest != plan.plan_digest:
        raise V03StatisticsError("statistics plan does not match the pre-experiment freeze")
    if frozen_input.statistics_plan_digest != plan.plan_digest:
        raise V03StatisticsError("statistics input does not match the frozen plan")
    if frozen_input.preexperiment_freeze_manifest_digest != freeze_manifest.freeze_manifest_digest:
        raise V03StatisticsError("statistics input does not match the authorized freeze")

    endpoints = {item.endpoint_id: item for item in plan.endpoints}
    contrasts = {item.hypothesis_id: item for item in plan.contrasts}
    grouped_rows: dict[str, tuple[FrozenContrastInputRow, ...]] = {
        hypothesis_id: tuple(row for row in frozen_input.rows if row.hypothesis_id == hypothesis_id)
        for hypothesis_id in contrasts
    }
    unknown = sorted(set(row.hypothesis_id for row in frozen_input.rows) - set(contrasts))
    missing = sorted(key for key, rows in grouped_rows.items() if not rows)
    if unknown or missing:
        raise V03StatisticsError(
            f"statistics rows do not exactly cover frozen hypotheses; missing={missing}, unknown={unknown}"
        )
    # An empty registry preserves compatibility for exploratory/development
    # plans.  Once a plan preregisters any N/A reason, however, every
    # hypothesis must be present in that digest-bound registry and every N/A
    # row must use one of its exact reasons.  This prevents a formal caller
    # from inventing exclusions after seeing released outcomes.
    if plan.registered_n_a_reasons:
        if set(plan.registered_n_a_reasons) != set(contrasts):
            raise V03StatisticsError(
                "formal N/A registry must cover every frozen hypothesis"
            )
        for row in frozen_input.rows:
            allowed = plan.registered_n_a_reasons[row.hypothesis_id]
            if row.status == "N_A" and row.n_a_reason not in allowed:
                raise V03StatisticsError(
                    "statistics row uses an N/A reason absent from the frozen plan"
                )
    if plan.formal_confirmatory:
        all_n_a = tuple(
            hypothesis_id
            for hypothesis_id, rows in grouped_rows.items()
            if not any(row.status == "OBSERVED" for row in rows)
        )
        if all_n_a:
            raise V03StatisticsError(
                "formal confirmatory hypotheses cannot be self-filled as all N/A; "
                f"hypotheses={list(all_n_a)}"
            )

    contrast_results: dict[str, dict[str, Any]] = {}
    family_results: dict[str, dict[str, Any]] = {}
    for family in plan.multiplicity_families:
        bootstraps: dict[str, HierarchicalBootstrapResult] = {}
        raw_p_values: dict[str, float] = {}
        observed_rows: dict[str, tuple[FrozenContrastInputRow, ...]] = {}
        coverage: dict[str, tuple[int, int]] = {}
        for hypothesis_id in family.hypothesis_ids:
            all_rows = grouped_rows[hypothesis_id]
            eligible = tuple(row for row in all_rows if row.status == "OBSERVED")
            observed_rows[hypothesis_id] = eligible
            coverage[hypothesis_id] = (len(eligible), len(all_rows))
        active = tuple(sorted(key for key, rows in observed_rows.items() if rows))
        excluded = tuple(sorted(set(family.hypothesis_ids) - set(active)))

        if active:
            reference_keys = {row.hierarchy_key for row in observed_rows[active[0]]}
            if any({row.hierarchy_key for row in observed_rows[key]} != reference_keys for key in active[1:]):
                raise V03StatisticsError(
                    "max-T family members must share identical eligible paired leaves after N/A exclusion"
                )
            family_seed = derive_bootstrap_seed(plan.seed_namespace, family.contrast_family_id)
            for hypothesis_id in active:
                contrast = contrasts[hypothesis_id]
                endpoint = endpoints[contrast.endpoint_id]
                rows = observed_rows[hypothesis_id]
                left = _hierarchical_values(rows, side="left", higher_is_better=endpoint.higher_is_better)
                right = _hierarchical_values(rows, side="right", higher_is_better=endpoint.higher_is_better)
                bootstrap = hierarchical_paired_difference_bootstrap(
                    left,
                    right,
                    resamples=plan.bootstrap_resamples,
                    seed=family_seed,
                    confidence_level=plan.confidence_level,
                )
                bootstraps[hypothesis_id] = bootstrap
                observed_rows[hypothesis_id] = rows
                raw_p_values[hypothesis_id] = centered_one_sided_p_value(
                    bootstrap.observed,
                    bootstrap.replicates,
                    null_boundary=contrast.null_boundary,
                )
            max_t = bootstrap_max_t_intervals(bootstraps)
            holm = holm_bonferroni(raw_p_values)
            family_results[family.contrast_family_id] = {
                "status": "COMPUTED",
                "registered_hypothesis_ids": list(family.hypothesis_ids),
                "eligible_hypothesis_ids": list(active),
                "excluded_n_a_hypothesis_ids": list(excluded),
                "registered_family_size": len(family.hypothesis_ids),
                "multiplicity_denominator": len(active),
                "n_a_entered_denominator": False,
                "simultaneous_interval_method": MAX_T_METHOD,
                "multiplicity_audit_method": HOLM_METHOD,
                "max_t_critical_value": next(iter(max_t.intervals.values())).critical_value,
                "resampling_plan_digest": max_t.resampling_plan_digest,
            }
            for hypothesis_id in active:
                contrast = contrasts[hypothesis_id]
                bootstrap = bootstraps[hypothesis_id]
                endpoint = endpoints[contrast.endpoint_id]
                rows = observed_rows[hypothesis_id]
                sensitivity = _task_sensitivity(
                    _hierarchical_values(
                        rows,
                        side="left",
                        higher_is_better=endpoint.higher_is_better,
                    ),
                    _hierarchical_values(
                        rows,
                        side="right",
                        higher_is_better=endpoint.higher_is_better,
                    ),
                )
                simultaneous = max_t.intervals[hypothesis_id]
                adjusted = holm[hypothesis_id]
                eligible_count, total_count = coverage[hypothesis_id]
                contrast_results[hypothesis_id] = {
                    "status": "COMPUTED",
                    "contrast_family_id": contrast.contrast_family_id,
                    "endpoint_id": contrast.endpoint_id,
                    "left_method_id": contrast.left_method_id,
                    "right_method_id": contrast.right_method_id,
                    "positive_effect_favours": "LEFT_METHOD",
                    "observed_effect": bootstrap.observed,
                    "marginal_interval": bootstrap.interval.to_dict(),
                    "one_sided_lower_bound": bootstrap.one_sided_lower,
                    "simultaneous_interval": simultaneous.to_dict(),
                    "raw_p_value": raw_p_values[hypothesis_id],
                    "holm_adjusted_p_value": adjusted.adjusted_p_value,
                    "holm_rejected": adjusted.adjusted_p_value <= plan.alpha,
                    "eligible_row_count": eligible_count,
                    "n_a_row_count": total_count - eligible_count,
                    "registered_row_count": total_count,
                    "coverage": eligible_count / total_count,
                    "n_a_entered_denominator": False,
                    "bootstrap_resamples": plan.bootstrap_resamples,
                    "bootstrap_seed": bootstrap.seed,
                    "resampling_plan_digest": bootstrap.resampling_plan_digest,
                    **sensitivity,
                }
        else:
            family_results[family.contrast_family_id] = {
                "status": "N_A",
                "registered_hypothesis_ids": list(family.hypothesis_ids),
                "eligible_hypothesis_ids": [],
                "excluded_n_a_hypothesis_ids": list(excluded),
                "registered_family_size": len(family.hypothesis_ids),
                "multiplicity_denominator": 0,
                "n_a_entered_denominator": False,
                "simultaneous_interval_method": MAX_T_METHOD,
                "multiplicity_audit_method": HOLM_METHOD,
                "max_t_critical_value": None,
                "resampling_plan_digest": None,
                "per_task_effects": {},
                "positive_task_count": 0,
                "zero_task_count": 0,
                "negative_task_count": 0,
                "worst_task_id": None,
                "worst_task_effect": None,
                "best_task_id": None,
                "best_task_effect": None,
                "leave_one_task_out_effects": {},
                "leave_one_task_out_status": "N_A",
            }
        for hypothesis_id in excluded:
            contrast = contrasts[hypothesis_id]
            eligible_count, total_count = coverage[hypothesis_id]
            contrast_results[hypothesis_id] = {
                "status": "N_A",
                "contrast_family_id": contrast.contrast_family_id,
                "endpoint_id": contrast.endpoint_id,
                "left_method_id": contrast.left_method_id,
                "right_method_id": contrast.right_method_id,
                "positive_effect_favours": "LEFT_METHOD",
                "observed_effect": None,
                "marginal_interval": None,
                "one_sided_lower_bound": None,
                "simultaneous_interval": None,
                "raw_p_value": None,
                "holm_adjusted_p_value": None,
                "holm_rejected": None,
                "eligible_row_count": eligible_count,
                "n_a_row_count": total_count - eligible_count,
                "registered_row_count": total_count,
                "coverage": 0.0,
                "n_a_entered_denominator": False,
                "bootstrap_resamples": None,
                "bootstrap_seed": None,
                "resampling_plan_digest": None,
            }

    return FormalStatisticsResult(
        run_id=frozen_input.run_id,
        statistics_plan_digest=plan.plan_digest,
        statistics_input_digest=frozen_input.input_manifest_digest,
        preexperiment_freeze_manifest_digest=freeze_manifest.freeze_manifest_digest,
        public_ranking_barrier_digest=frozen_input.public_ranking_barrier_digest,
        oracle_unlock_handoff_digest=frozen_input.oracle_unlock_handoff_digest,
        oracle_evidence_manifest_digest=frozen_input.oracle_evidence_manifest_digest,
        contrast_results=contrast_results,
        family_results=family_results,
    )


__all__ = [
    "FORMAL_CONTRAST_FAMILY_IDS",
    "FormalStatisticsPlan",
    "FormalStatisticsResult",
    "FrozenContrastInputRow",
    "FrozenStatisticsInput",
    "HOLM_METHOD",
    "MAX_T_METHOD",
    "MultiplicityFamilyPlan",
    "N_A_POLICY",
    "RESAMPLING_CONTRACT",
    "StatisticsContrast",
    "StatisticsEndpoint",
    "V03StatisticsError",
    "compute_formal_statistics",
]
