"""N/A-aware, representation-local metrics for the v0.3 signal atlas.

Distances from different representation coordinate systems are never mixed.
The module consumes complete per-query rankings and reports retrieval and
between/within separation inside one frozen representation only.  It performs
no model fitting and has no oracle access.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json


SIGNAL_DISTANCE_ROW_SCHEMA = "policy-learnware.v03-signal-distance-row.v0"
SIGNAL_METRIC_RECORD_SCHEMA = "policy-learnware.v03-signal-metric-record.v0"
PUBLIC_SIGNAL_METRIC_SCHEMA = "policy-learnware.v03-public-signal-metric.v0"
PAIRED_SIGNAL_CONTRAST_SCHEMA = "policy-learnware.v03-paired-signal-contrast.v0"
REPRESENTATION_GAIN_SCHEMA = "policy-learnware.v03-representation-gain.v0"

# These metrics remain interpretable after fitting the same representation
# family independently for two views.  Raw distance means/margins are excluded:
# their scale is tied to the view-specific coordinate and source-only bandwidth.
DEFAULT_PAIRED_METRIC_IDS = (
    "context_top1",
    "task_top1",
    "context_mrr",
    "between_within_ratio",
)


class SignalMetricError(ValueError):
    """A distance matrix, metric identity or paired contrast is invalid."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SignalMetricError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result.lower() != result:
        raise SignalMetricError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise SignalMetricError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise SignalMetricError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SignalMetricError(f"{where} must be finite")
    return result


@dataclass(frozen=True)
class SignalDistanceRow:
    query_bank_id: str
    source_bank_id: str
    query_receipt_digest: str
    source_receipt_digest: str
    query_raw_dataset_digest: str
    source_raw_dataset_digest: str
    query_task_id: str
    source_task_id: str
    query_context_id: str
    source_context_id: str
    query_embodiment_id: str
    source_embodiment_id: str
    query_abi_contract_id: str
    source_abi_contract_id: str
    query_goal_contract_id: str
    source_goal_contract_id: str
    query_dynamics_context_id: str
    source_dynamics_context_id: str
    query_equivalence_class_id: str | None
    source_equivalence_class_id: str | None
    distance: float
    schema: str = SIGNAL_DISTANCE_ROW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_DISTANCE_ROW_SCHEMA:
            raise SignalMetricError("unsupported SignalDistanceRow schema")
        for name in (
            "query_receipt_digest",
            "source_receipt_digest",
            "query_raw_dataset_digest",
            "source_raw_dataset_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in (
            "query_bank_id",
            "source_bank_id",
            "query_task_id",
            "source_task_id",
            "query_context_id",
            "source_context_id",
            "query_embodiment_id",
            "source_embodiment_id",
            "query_abi_contract_id",
            "source_abi_contract_id",
            "query_goal_contract_id",
            "source_goal_contract_id",
            "query_dynamics_context_id",
            "source_dynamics_context_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "query_equivalence_class_id",
            "source_equivalence_class_id",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        distance = _finite(self.distance, "distance")
        if distance < 0.0:
            raise SignalMetricError("distance must be non-negative")
        object.__setattr__(self, "distance", distance)

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class SignalMetricRecord:
    cell_id: str
    view_or_condition_id: str
    representation_id: str
    representation_coordinate_digest: str
    representation_seed: int | None
    source_index_digest: str
    query_manifest_digest: str
    rows: tuple[SignalDistanceRow, ...]
    expected_source_by_query: Mapping[str, str]
    metric_values: Mapping[str, float] | None = None
    schema: str = SIGNAL_METRIC_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_METRIC_RECORD_SCHEMA:
            raise SignalMetricError("unsupported SignalMetricRecord schema")
        for name in ("cell_id", "view_or_condition_id", "representation_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "representation_coordinate_digest",
            "source_index_digest",
            "query_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.representation_seed is not None:
            if isinstance(self.representation_seed, (bool, np.bool_)) or not isinstance(
                self.representation_seed, (int, np.integer)
            ):
                raise SignalMetricError(
                    "representation_seed must be a non-negative integer or null"
                )
            if int(self.representation_seed) < 0:
                raise SignalMetricError(
                    "representation_seed must be a non-negative integer or null"
                )
            object.__setattr__(self, "representation_seed", int(self.representation_seed))
        rows = tuple(self.rows)
        if not rows or not all(isinstance(row, SignalDistanceRow) for row in rows):
            raise SignalMetricError("signal metric record requires typed distance rows")
        pairs = [(row.query_bank_id, row.source_bank_id) for row in rows]
        if len(set(pairs)) != len(pairs):
            raise SignalMetricError("distance matrix contains duplicate query/source pairs")
        by_query: dict[str, list[SignalDistanceRow]] = {}
        for row in rows:
            by_query.setdefault(row.query_bank_id, []).append(row)
        source_sets = {tuple(sorted(item.source_bank_id for item in group)) for group in by_query.values()}
        if len(source_sets) != 1:
            raise SignalMetricError("every query must rank the same complete source set")
        expected = dict(sorted(self.expected_source_by_query.items()))
        if set(expected) != set(by_query):
            raise SignalMetricError("ground-truth source coverage differs from query matrix")
        source_ids = set(next(iter(source_sets)))
        if any(source_id not in source_ids for source_id in expected.values()):
            raise SignalMetricError("ground-truth source is absent from the ranked source set")
        metrics = _compute_metrics(by_query, expected)
        if self.metric_values is not None:
            supplied = {key: _finite(value, f"metric_values[{key}]") for key, value in self.metric_values.items()}
            if supplied != metrics:
                raise SignalMetricError("metric_values do not match distance rows")
        object.__setattr__(self, "rows", tuple(sorted(rows, key=lambda row: (row.query_bank_id, row.source_bank_id))))
        object.__setattr__(self, "expected_source_by_query", MappingProxyType(expected))
        object.__setattr__(self, "metric_values", MappingProxyType(metrics))

    @property
    def record_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cell_id": self.cell_id,
            "view_or_condition_id": self.view_or_condition_id,
            "representation_id": self.representation_id,
            "representation_coordinate_digest": self.representation_coordinate_digest,
            "representation_seed": self.representation_seed,
            "source_index_digest": self.source_index_digest,
            "query_manifest_digest": self.query_manifest_digest,
            "rows": [row.to_dict() for row in self.rows],
            "expected_source_by_query": dict(self.expected_source_by_query),
            "metric_values": dict(self.metric_values or {}),
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Return the aggregate-only projection; private bank/taxonomy rows stay sealed."""

        payload = {
            "schema": PUBLIC_SIGNAL_METRIC_SCHEMA,
            "cell_id": self.cell_id,
            "view_or_condition_id": self.view_or_condition_id,
            "representation_id": self.representation_id,
            "representation_seed": self.representation_seed,
            "representation_coordinate_digest": self.representation_coordinate_digest,
            "source_index_digest": self.source_index_digest,
            "query_manifest_digest": self.query_manifest_digest,
            "metric_values": dict(self.metric_values or {}),
            "private_distance_rows_withheld": True,
            "private_record_digest": self.record_digest,
        }
        return {
            **payload,
            "public_projection_digest": sha256_json(payload),
        }


def _compute_metrics(
    by_query: Mapping[str, Sequence[SignalDistanceRow]],
    expected: Mapping[str, str],
) -> dict[str, float]:
    exact_source_hits: list[float] = []
    exact_source_reciprocal_ranks: list[float] = []
    context_hits: list[float] = []
    context_reciprocal_ranks: list[float] = []
    task_hits: list[float] = []
    task_reciprocal_ranks: list[float] = []
    embodiment_hits: list[float] = []
    abi_hits: list[float] = []
    # Goal and dynamics are conditional axes, not aliases for whichever source
    # wins the global (and often schema-dominated) ranking.  Their metrics are
    # computed only for queries whose frozen source panel contains a genuine
    # within-scope contrast.  If a panel has no eligible query its accuracy/MRR
    # is intentionally absent (N/A); counts and coverage remain explicit.
    goal_hits: list[float] = []
    goal_reciprocal_ranks: list[float] = []
    dynamics_hits: list[float] = []
    dynamics_reciprocal_ranks: list[float] = []
    dynamics_between_distances: list[float] = []
    within: list[float] = []
    between: list[float] = []
    source_identity_by_id: dict[str, tuple[str, ...]] = {}
    for query_id, group in sorted(by_query.items()):
        query_identities = {
            (
                row.query_task_id,
                row.query_context_id,
                row.query_embodiment_id,
                row.query_abi_contract_id,
                row.query_goal_contract_id,
                row.query_dynamics_context_id,
                row.query_equivalence_class_id,
            )
            for row in group
        }
        if len(query_identities) != 1:
            raise SignalMetricError(
                "one query bank cannot carry multiple structural identities"
            )
        for row in group:
            source_identity = (
                row.source_task_id,
                row.source_context_id,
                row.source_embodiment_id,
                row.source_abi_contract_id,
                row.source_goal_contract_id,
                row.source_dynamics_context_id,
                row.source_equivalence_class_id or "",
            )
            previous = source_identity_by_id.setdefault(
                row.source_bank_id, source_identity
            )
            if previous != source_identity:
                raise SignalMetricError(
                    "one source bank cannot carry multiple structural identities"
                )
        ranked = sorted(group, key=lambda row: (row.distance, row.source_bank_id))
        expected_id = expected[query_id]
        top = ranked[0]
        exact_source_hits.append(float(top.source_bank_id == expected_id))
        exact_source_reciprocal_ranks.append(
            1.0
            / next(
                rank
                for rank, row in enumerate(ranked, start=1)
                if row.source_bank_id == expected_id
            )
        )
        context_hits.append(float(top.source_context_id == top.query_context_id))
        matching_context_ranks = [
            rank
            for rank, row in enumerate(ranked, start=1)
            if row.source_context_id == row.query_context_id
        ]
        context_reciprocal_ranks.append(
            0.0 if not matching_context_ranks else 1.0 / matching_context_ranks[0]
        )
        task_hits.append(float(top.source_task_id == top.query_task_id))
        matching_task_ranks = [
            rank
            for rank, row in enumerate(ranked, start=1)
            if row.source_task_id == row.query_task_id
        ]
        task_reciprocal_ranks.append(
            0.0 if not matching_task_ranks else 1.0 / matching_task_ranks[0]
        )
        embodiment_hits.append(
            float(top.source_embodiment_id == top.query_embodiment_id)
        )
        abi_hits.append(
            float(top.source_abi_contract_id == top.query_abi_contract_id)
        )

        # Same-embodiment/inter-goal readout.  ABI is fixed so an observation
        # or action-width shortcut cannot determine the result.  Dynamics is
        # deliberately allowed to vary as a nuisance; the formal panel must
        # still provide at least two goal contracts and the query's own goal.
        goal_ranked = [
            row
            for row in ranked
            if row.source_embodiment_id == row.query_embodiment_id
            and row.source_abi_contract_id == row.query_abi_contract_id
        ]
        goal_contracts = {row.source_goal_contract_id for row in goal_ranked}
        goal_matching_ranks = [
            rank
            for rank, row in enumerate(goal_ranked, start=1)
            if row.source_goal_contract_id == row.query_goal_contract_id
        ]
        if len(goal_contracts) >= 2 and goal_matching_ranks:
            goal_hits.append(
                float(
                    goal_ranked[0].source_goal_contract_id
                    == goal_ranked[0].query_goal_contract_id
                )
            )
            goal_reciprocal_ranks.append(1.0 / goal_matching_ranks[0])

        # Within-task dynamics readout.  Task, goal, embodiment and ABI are
        # fixed before ranking, so this metric cannot be won by changing the
        # registered goal or the agent/schema.  A query is eligible only when
        # the source panel exposes at least two dynamics contexts and includes
        # its matching context.
        dynamics_ranked = [
            row
            for row in ranked
            if row.source_task_id == row.query_task_id
            and row.source_goal_contract_id == row.query_goal_contract_id
            and row.source_embodiment_id == row.query_embodiment_id
            and row.source_abi_contract_id == row.query_abi_contract_id
        ]
        dynamics_contexts = {
            row.source_dynamics_context_id for row in dynamics_ranked
        }
        dynamics_between_distances.extend(
            row.distance
            for row in dynamics_ranked
            if row.source_dynamics_context_id
            != row.query_dynamics_context_id
        )
        dynamics_matching_ranks = [
            rank
            for rank, row in enumerate(dynamics_ranked, start=1)
            if row.source_dynamics_context_id == row.query_dynamics_context_id
        ]
        if len(dynamics_contexts) >= 2 and dynamics_matching_ranks:
            dynamics_hits.append(
                float(
                    dynamics_ranked[0].source_dynamics_context_id
                    == dynamics_ranked[0].query_dynamics_context_id
                )
            )
            dynamics_reciprocal_ranks.append(1.0 / dynamics_matching_ranks[0])
        equivalence = top.query_equivalence_class_id
        if equivalence is not None:
            for row in ranked:
                target = (
                    within
                    if row.source_equivalence_class_id == equivalence
                    else between
                )
                target.append(row.distance)
    metrics = {
        # The expected-source map supports an exact-bank diagnostic only.  It
        # must never redefine task/goal/dynamics truth: those axes are joined
        # directly to the frozen query identity carried by every row.
        "exact_source_top1": float(np.mean(exact_source_hits)),
        "exact_source_mrr": float(np.mean(exact_source_reciprocal_ranks)),
        "context_top1": float(np.mean(context_hits)),
        "task_top1": float(np.mean(task_hits)),
        "task_mrr": float(np.mean(task_reciprocal_ranks)),
        "embodiment_top1": float(np.mean(embodiment_hits)),
        "abi_top1": float(np.mean(abi_hits)),
        "context_mrr": float(np.mean(context_reciprocal_ranks)),
        "query_count": float(len(by_query)),
        "source_count": float(len(next(iter(by_query.values())))),
        "goal_query_count": float(len(goal_hits)),
        "goal_query_coverage": float(len(goal_hits) / len(by_query)),
        "dynamics_query_count": float(len(dynamics_hits)),
        "dynamics_query_coverage": float(len(dynamics_hits) / len(by_query)),
        "dynamics_between_pair_count": float(len(dynamics_between_distances)),
    }
    if goal_hits:
        metrics.update(
            {
                "goal_top1": float(np.mean(goal_hits)),
                "goal_mrr": float(np.mean(goal_reciprocal_ranks)),
            }
        )
    if dynamics_hits:
        metrics.update(
            {
                "dynamics_top1": float(np.mean(dynamics_hits)),
                "dynamics_mrr": float(np.mean(dynamics_reciprocal_ranks)),
            }
        )
    if dynamics_between_distances:
        metrics["dynamics_between_mean_distance"] = float(
            np.mean(dynamics_between_distances)
        )
    if within and between:
        within_mean = float(np.mean(within))
        between_mean = float(np.mean(between))
        metrics.update(
            {
                "within_mean_distance": within_mean,
                "between_mean_distance": between_mean,
                "between_within_margin": between_mean - within_mean,
                "between_within_ratio": between_mean
                / max(within_mean, np.finfo(np.float64).eps),
            }
        )
    return metrics


@dataclass(frozen=True)
class PairedSignalContrast:
    base_record_digest: str
    control_record_digest: str
    base_cell_id: str
    control_cell_id: str
    metric_deltas: Mapping[str, float]
    schema: str = PAIRED_SIGNAL_CONTRAST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAIRED_SIGNAL_CONTRAST_SCHEMA:
            raise SignalMetricError("unsupported PairedSignalContrast schema")
        for name in ("base_record_digest", "control_record_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("base_cell_id", "control_cell_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        values = {
            key: _finite(value, f"metric_deltas[{key}]")
            for key, value in sorted(self.metric_deltas.items())
        }
        if not values:
            raise SignalMetricError("paired contrast requires shared metrics")
        object.__setattr__(self, "metric_deltas", MappingProxyType(values))

    @property
    def contrast_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_record_digest": self.base_record_digest,
            "control_record_digest": self.control_record_digest,
            "base_cell_id": self.base_cell_id,
            "control_cell_id": self.control_cell_id,
            "metric_deltas": dict(self.metric_deltas),
        }


def paired_signal_contrast(
    base: SignalMetricRecord,
    control: SignalMetricRecord,
    *,
    metric_ids: Sequence[str] = DEFAULT_PAIRED_METRIC_IDS,
) -> PairedSignalContrast:
    if not isinstance(base, SignalMetricRecord) or not isinstance(control, SignalMetricRecord):
        raise SignalMetricError("paired contrast requires typed metric records")
    if base.representation_id != control.representation_id:
        raise SignalMetricError(
            "paired destructive contrast requires the same representation family"
        )
    if base.representation_seed != control.representation_seed:
        raise SignalMetricError(
            "paired destructive contrast requires the same representation seed"
        )
    if base.expected_source_by_query != control.expected_source_by_query:
        raise SignalMetricError("paired contrast query/ground-truth bindings differ")
    base_pairs = _raw_membership(base)
    control_pairs = _raw_membership(control)
    if base_pairs != control_pairs:
        raise SignalMetricError("paired contrast raw bank membership differs")
    requested = tuple(_nonempty(metric, "paired metric ID") for metric in metric_ids)
    if not requested or len(set(requested)) != len(requested):
        raise SignalMetricError("paired metric IDs must be non-empty and unique")
    common = tuple(
        metric
        for metric in requested
        if metric in (base.metric_values or {}) and metric in (control.metric_values or {})
    )
    if len(common) != len(requested):
        raise SignalMetricError("paired metric ID is absent from one metric record")
    return PairedSignalContrast(
        base_record_digest=base.record_digest,
        control_record_digest=control.record_digest,
        base_cell_id=base.cell_id,
        control_cell_id=control.cell_id,
        metric_deltas={
            metric: float((base.metric_values or {})[metric] - (control.metric_values or {})[metric])
            for metric in common
        },
    )


def _raw_membership(record: SignalMetricRecord) -> frozenset[tuple[str, ...]]:
    return frozenset(
        (
            row.query_bank_id,
            row.query_receipt_digest,
            row.query_raw_dataset_digest,
            row.source_bank_id,
            row.source_receipt_digest,
            row.source_raw_dataset_digest,
        )
        for row in record.rows
    )


@dataclass(frozen=True)
class RepresentationGainContrast:
    raw_record_digest: str
    learned_record_digest: str
    view_or_condition_id: str
    learned_seed: int
    metric_gains: Mapping[str, float]
    schema: str = REPRESENTATION_GAIN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REPRESENTATION_GAIN_SCHEMA:
            raise SignalMetricError("unsupported RepresentationGainContrast schema")
        for name in ("raw_record_digest", "learned_record_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "view_or_condition_id",
            _nonempty(self.view_or_condition_id, "view_or_condition_id"),
        )
        if isinstance(self.learned_seed, bool) or not isinstance(self.learned_seed, int) or self.learned_seed < 0:
            raise SignalMetricError("learned_seed must be a non-negative integer")
        gains = {
            key: _finite(value, f"metric_gains[{key}]")
            for key, value in sorted(self.metric_gains.items())
        }
        if not gains:
            raise SignalMetricError("representation gain requires shared metrics")
        object.__setattr__(self, "metric_gains", MappingProxyType(gains))

    @property
    def contrast_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "raw_record_digest": self.raw_record_digest,
            "learned_record_digest": self.learned_record_digest,
            "view_or_condition_id": self.view_or_condition_id,
            "learned_seed": self.learned_seed,
            "metric_gains": dict(self.metric_gains),
        }


def representation_gain_contrast(
    raw: SignalMetricRecord,
    learned: SignalMetricRecord,
    *,
    metric_ids: Sequence[str] = DEFAULT_PAIRED_METRIC_IDS,
) -> RepresentationGainContrast:
    """Compute preregistered R5-minus-R0 gains without mixing MMD scales."""

    if not isinstance(raw, SignalMetricRecord) or not isinstance(
        learned, SignalMetricRecord
    ):
        raise SignalMetricError("representation gain requires typed metric records")
    if raw.representation_id != "R0_PADDED_RAW" or learned.representation_id != "R5_VIEW_SPECIFIC_CORRO_REFIT":
        raise SignalMetricError("representation gain is frozen as R5 minus R0")
    if raw.representation_seed is not None or learned.representation_seed not in {0, 1, 2}:
        raise SignalMetricError("representation gain seed identities are invalid")
    if raw.view_or_condition_id != learned.view_or_condition_id:
        raise SignalMetricError("representation gain requires the same input view")
    if raw.expected_source_by_query != learned.expected_source_by_query:
        raise SignalMetricError("representation gain expected-source bindings differ")
    if _raw_membership(raw) != _raw_membership(learned):
        raise SignalMetricError("representation gain raw bank membership differs")
    requested = tuple(_nonempty(metric, "gain metric ID") for metric in metric_ids)
    if not requested or len(set(requested)) != len(requested):
        raise SignalMetricError("gain metric IDs must be non-empty and unique")
    if any(
        metric not in (raw.metric_values or {})
        or metric not in (learned.metric_values or {})
        for metric in requested
    ):
        raise SignalMetricError("gain metric ID is absent from one metric record")
    return RepresentationGainContrast(
        raw_record_digest=raw.record_digest,
        learned_record_digest=learned.record_digest,
        view_or_condition_id=raw.view_or_condition_id,
        learned_seed=int(learned.representation_seed),
        metric_gains={
            metric: float(
                (learned.metric_values or {})[metric]
                - (raw.metric_values or {})[metric]
            )
            for metric in requested
        },
    )


__all__ = [
    "DEFAULT_PAIRED_METRIC_IDS",
    "PAIRED_SIGNAL_CONTRAST_SCHEMA",
    "PUBLIC_SIGNAL_METRIC_SCHEMA",
    "REPRESENTATION_GAIN_SCHEMA",
    "SIGNAL_DISTANCE_ROW_SCHEMA",
    "SIGNAL_METRIC_RECORD_SCHEMA",
    "PairedSignalContrast",
    "RepresentationGainContrast",
    "SignalDistanceRow",
    "SignalMetricError",
    "SignalMetricRecord",
    "paired_signal_contrast",
    "representation_gain_contrast",
]
