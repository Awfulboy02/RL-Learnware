"""Pure sealed-ranking evaluation for v0.5 certified-policy retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from ..v04a.protocol import (
    BUDGET_EPISODES,
    RankingSeal,
    V04AProtocolError,
    verify_ranking_seal,
)
from .labels import CertificateResolver, CertifiedPolicyManifest


PREDICTION_PAYLOAD_SCHEMA = "policy-learnware.v05-sealed-predictions.v2"
MARKET_30_CERT = "MARKET_30_CERT"
TASK_5_CERT = "TASK_5_CERT"
_ENDPOINTS = frozenset({MARKET_30_CERT, TASK_5_CERT})


class V05MetricError(ValueError):
    """A ranking evaluation input is malformed, unsealed, or incomplete."""


def _canonical_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V05MetricError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    value = _canonical_string(value, where)
    if len(value) != 64 or value != value.lower():
        raise V05MetricError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise V05MetricError(f"{where} must be a lowercase SHA-256 digest") from error
    return value


def _integer(value: Any, where: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise V05MetricError(f"{where} must be an integer")
    result = int(value)
    if result < minimum:
        raise V05MetricError(f"{where} must be >= {minimum}")
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V05MetricError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V05MetricError(f"{where} must be finite")
    return result


def _unique_ranking(values: Any, where: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise V05MetricError(f"{where} must be a non-empty list")
    result = tuple(_canonical_string(item, f"{where} item") for item in values)
    if len(result) != len(set(result)):
        raise V05MetricError(f"{where} contains duplicate rank IDs")
    return result


def _exact_keys(payload: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(payload, Mapping):
        raise V05MetricError(f"{where} must be an object")
    actual = set(payload)
    if actual != expected:
        raise V05MetricError(
            f"{where} has invalid fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True, order=True)
class PredictionRanking:
    """One truth-free ranking row that must be sealed before evaluation."""

    method_id: str
    endpoint: str
    budget_episodes: int
    opaque_query_id: str
    ranked_anchor_ids: tuple[str, ...]
    ranked_policy_ids: tuple[str, ...]
    probe_protocol_digest: str
    reward_free_bank_sha256: str
    canonical_query_bank_digest: str
    source_train_membership_digest: str
    source_validation_membership_digest: str
    target_membership_digest: str
    normalization_digest: str
    config_digest: str
    source_model_manifest_digest: str
    authorized_query_manifest_digest: str
    score_vector_digest: str
    budget_ledger_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "method_id", _canonical_string(self.method_id, "method_id")
        )
        endpoint = _canonical_string(self.endpoint, "endpoint")
        if endpoint not in _ENDPOINTS:
            raise V05MetricError(f"unsupported retrieval endpoint: {endpoint}")
        object.__setattr__(self, "endpoint", endpoint)
        budget = _integer(self.budget_episodes, "budget_episodes")
        if budget not in BUDGET_EPISODES:
            raise V05MetricError(f"budget_episodes must be one of {BUDGET_EPISODES}")
        object.__setattr__(self, "budget_episodes", budget)
        object.__setattr__(
            self,
            "opaque_query_id",
            _canonical_string(self.opaque_query_id, "opaque_query_id"),
        )
        object.__setattr__(
            self,
            "ranked_anchor_ids",
            _unique_ranking(self.ranked_anchor_ids, "ranked_anchor_ids"),
        )
        object.__setattr__(
            self,
            "ranked_policy_ids",
            _unique_ranking(self.ranked_policy_ids, "ranked_policy_ids"),
        )
        for field_name in (
            "probe_protocol_digest",
            "reward_free_bank_sha256",
            "canonical_query_bank_digest",
            "source_train_membership_digest",
            "source_validation_membership_digest",
            "target_membership_digest",
            "normalization_digest",
            "config_digest",
            "source_model_manifest_digest",
            "authorized_query_manifest_digest",
            "score_vector_digest",
            "budget_ledger_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), field_name),
            )

    @property
    def cell_key(self) -> tuple[str, str, int, str]:
        return (
            self.method_id,
            self.endpoint,
            self.budget_episodes,
            self.opaque_query_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "endpoint": self.endpoint,
            "budget_episodes": self.budget_episodes,
            "opaque_query_id": self.opaque_query_id,
            "ranked_anchor_ids": list(self.ranked_anchor_ids),
            "ranked_policy_ids": list(self.ranked_policy_ids),
            "probe_protocol_digest": self.probe_protocol_digest,
            "reward_free_bank_sha256": self.reward_free_bank_sha256,
            "canonical_query_bank_digest": self.canonical_query_bank_digest,
            "source_train_membership_digest": self.source_train_membership_digest,
            "source_validation_membership_digest": (
                self.source_validation_membership_digest
            ),
            "target_membership_digest": self.target_membership_digest,
            "normalization_digest": self.normalization_digest,
            "config_digest": self.config_digest,
            "source_model_manifest_digest": self.source_model_manifest_digest,
            "authorized_query_manifest_digest": self.authorized_query_manifest_digest,
            "score_vector_digest": self.score_vector_digest,
            "budget_ledger_digest": self.budget_ledger_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionRanking":
        fields = {
            "method_id",
            "endpoint",
            "budget_episodes",
            "opaque_query_id",
            "ranked_anchor_ids",
            "ranked_policy_ids",
            "probe_protocol_digest",
            "reward_free_bank_sha256",
            "canonical_query_bank_digest",
            "source_train_membership_digest",
            "source_validation_membership_digest",
            "target_membership_digest",
            "normalization_digest",
            "config_digest",
            "source_model_manifest_digest",
            "authorized_query_manifest_digest",
            "score_vector_digest",
            "budget_ledger_digest",
        }
        _exact_keys(payload, fields, "prediction ranking")
        return cls(**{field: payload[field] for field in fields})


@dataclass(frozen=True, order=True)
class TruthBinding:
    """Private post-seal query-to-source-certificate join row."""

    opaque_query_id: str
    source_anchor_id: str
    task_id: str
    opaque_certified_policy_id: str
    authorized_query_manifest_digest: str
    prediction_seal_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "opaque_query_id",
            "source_anchor_id",
            "task_id",
            "opaque_certified_policy_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_string(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "authorized_query_manifest_digest",
            _digest(
                self.authorized_query_manifest_digest,
                "authorized_query_manifest_digest",
            ),
        )
        object.__setattr__(
            self,
            "prediction_seal_digest",
            _digest(self.prediction_seal_digest, "prediction_seal_digest"),
        )

    @property
    def certified_policy_id(self) -> str:
        return self.opaque_certified_policy_id

    def to_dict(self) -> dict[str, str]:
        return {
            "opaque_query_id": self.opaque_query_id,
            "source_anchor_id": self.source_anchor_id,
            "task_id": self.task_id,
            "opaque_certified_policy_id": self.opaque_certified_policy_id,
            "authorized_query_manifest_digest": self.authorized_query_manifest_digest,
            "prediction_seal_digest": self.prediction_seal_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TruthBinding":
        fields = {
            "opaque_query_id",
            "source_anchor_id",
            "task_id",
            "opaque_certified_policy_id",
            "authorized_query_manifest_digest",
            "prediction_seal_digest",
        }
        _exact_keys(payload, fields, "truth binding")
        return cls(**{field: payload[field] for field in fields})


@dataclass(frozen=True)
class TaskRetrievalMetrics:
    task_id: str
    anchor_count: int
    anchor_hit_at_1: float
    anchor_mrr: float
    anchor_recall_at_k: Mapping[int, float]
    policy_hit_at_1: float
    policy_mrr: float
    policy_recall_at_k: Mapping[int, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "anchor_count": self.anchor_count,
            "anchor_hit_at_1": self.anchor_hit_at_1,
            "anchor_mrr": self.anchor_mrr,
            "anchor_recall_at_k": {
                str(key): value for key, value in self.anchor_recall_at_k.items()
            },
            "policy_hit_at_1": self.policy_hit_at_1,
            "policy_mrr": self.policy_mrr,
            "policy_recall_at_k": {
                str(key): value for key, value in self.policy_recall_at_k.items()
            },
        }


@dataclass(frozen=True)
class RetrievalMetrics:
    """Task-equal macro metrics for one method/endpoint/budget group."""

    method_id: str
    endpoint: str
    budget_episodes: int
    anchor_count: int
    task_count: int
    anchor_hit_at_1: float
    anchor_mrr: float
    anchor_recall_at_k: Mapping[int, float]
    policy_hit_at_1: float
    policy_mrr: float
    policy_recall_at_k: Mapping[int, float]
    per_task: tuple[TaskRetrievalMetrics, ...]

    @property
    def certified_policy_hit_at_1(self) -> float:
        return self.policy_hit_at_1

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "endpoint": self.endpoint,
            "budget_episodes": self.budget_episodes,
            "anchor_count": self.anchor_count,
            "task_count": self.task_count,
            "aggregation": "TASK_EQUAL_MACRO",
            "anchor_hit_at_1": self.anchor_hit_at_1,
            "anchor_mrr": self.anchor_mrr,
            "anchor_recall_at_k": {
                str(key): value for key, value in self.anchor_recall_at_k.items()
            },
            "certified_policy_hit_at_1": self.policy_hit_at_1,
            "certified_policy_mrr": self.policy_mrr,
            "certified_policy_recall_at_k": {
                str(key): value for key, value in self.policy_recall_at_k.items()
            },
            "per_task": [item.to_dict() for item in self.per_task],
        }


@dataclass(frozen=True)
class SealedEvaluation:
    prediction_seal_digest: str
    certificate_manifest_digest: str
    truth_join_digest: str
    metrics: tuple[RetrievalMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_seal_digest": self.prediction_seal_digest,
            "certificate_manifest_digest": self.certificate_manifest_digest,
            "truth_join_digest": self.truth_join_digest,
            "metrics": [item.to_dict() for item in self.metrics],
        }


def prediction_payload(
    rankings: Iterable[PredictionRanking | Mapping[str, Any]],
) -> dict[str, Any]:
    """Build canonical truth-free content suitable for ``seal_rankings``."""

    rows = tuple(
        item
        if isinstance(item, PredictionRanking)
        else PredictionRanking.from_dict(item)
        for item in rankings
    )
    if not rows:
        raise V05MetricError("prediction payload must not be empty")
    keys = [item.cell_key for item in rows]
    if len(keys) != len(set(keys)):
        raise V05MetricError("prediction payload has duplicate ranking cells")
    return {
        "schema": PREDICTION_PAYLOAD_SCHEMA,
        "predictions": [
            item.to_dict() for item in sorted(rows, key=lambda item: item.cell_key)
        ],
    }


def require_prediction_cell_coverage(
    rankings: Iterable[PredictionRanking],
    *,
    expected_method_ids: Sequence[str],
    expected_endpoints: Sequence[str] = (MARKET_30_CERT, TASK_5_CERT),
    expected_budgets: Sequence[int] = BUDGET_EPISODES,
) -> tuple[PredictionRanking, ...]:
    """Require one common, traceable Cartesian panel for every query."""

    rows = tuple(rankings)
    if not rows or any(not isinstance(item, PredictionRanking) for item in rows):
        raise V05MetricError("prediction coverage requires ranking rows")
    methods = tuple(
        _canonical_string(item, "expected method ID") for item in expected_method_ids
    )
    endpoints = tuple(
        _canonical_string(item, "expected endpoint") for item in expected_endpoints
    )
    budgets = tuple(_integer(item, "expected budget") for item in expected_budgets)
    if not methods or len(methods) != len(set(methods)):
        raise V05MetricError("expected method IDs must be non-empty and unique")
    if (
        not endpoints
        or len(endpoints) != len(set(endpoints))
        or not set(endpoints).issubset(_ENDPOINTS)
    ):
        raise V05MetricError("expected endpoints must be supported and unique")
    if (
        not budgets
        or len(budgets) != len(set(budgets))
        or not set(budgets).issubset(BUDGET_EPISODES)
    ):
        raise V05MetricError("expected budgets must be frozen and unique")

    keys = [item.cell_key for item in rows]
    if len(keys) != len(set(keys)):
        raise V05MetricError("prediction coverage contains duplicate ranking cells")
    expected_cells = {
        (method_id, endpoint, budget)
        for method_id in methods
        for endpoint in endpoints
        for budget in budgets
    }
    by_query: dict[str, list[PredictionRanking]] = {}
    for item in rows:
        by_query.setdefault(item.opaque_query_id, []).append(item)
    for query_id, query_rows in by_query.items():
        actual_cells = {
            (item.method_id, item.endpoint, item.budget_episodes) for item in query_rows
        }
        if actual_cells != expected_cells:
            raise V05MetricError(
                f"query {query_id} does not have complete method/endpoint/budget coverage"
            )

    common_source_fields = (
        "probe_protocol_digest",
        "source_train_membership_digest",
        "source_validation_membership_digest",
        "normalization_digest",
        "config_digest",
        "source_model_manifest_digest",
    )
    common_source_binding = tuple(
        getattr(rows[0], field_name) for field_name in common_source_fields
    )
    if any(
        tuple(getattr(item, field_name) for field_name in common_source_fields)
        != common_source_binding
        for item in rows[1:]
    ):
        raise V05MetricError("prediction rows do not share one source/probe binding")
    for query_id, query_rows in by_query.items():
        target_binding = (
            query_rows[0].reward_free_bank_sha256,
            query_rows[0].target_membership_digest,
            query_rows[0].authorized_query_manifest_digest,
        )
        if any(
            (
                item.reward_free_bank_sha256,
                item.target_membership_digest,
                item.authorized_query_manifest_digest,
            )
            != target_binding
            for item in query_rows[1:]
        ):
            raise V05MetricError(
                f"query {query_id} does not share one target bank/membership binding"
            )
        for budget in budgets:
            budget_rows = [
                item for item in query_rows if item.budget_episodes == budget
            ]
            if len({item.canonical_query_bank_digest for item in budget_rows}) != 1:
                raise V05MetricError(
                    f"query {query_id} budget {budget} has inconsistent canonical banks"
                )
    return rows


def _truth_rows(
    truth_join: Iterable[TruthBinding | Mapping[str, Any]],
    resolver: CertificateResolver,
) -> tuple[TruthBinding, ...]:
    rows = tuple(
        item if isinstance(item, TruthBinding) else TruthBinding.from_dict(item)
        for item in truth_join
    )
    if not rows:
        raise V05MetricError("truth join must not be empty")
    query_ids = [item.opaque_query_id for item in rows]
    if len(query_ids) != len(set(query_ids)):
        raise V05MetricError("truth join has duplicate opaque query IDs")
    anchors = [item.source_anchor_id for item in rows]
    if len(anchors) != len(set(anchors)):
        raise V05MetricError(
            "truth join must contain exactly one statistical unit per source anchor"
        )
    if set(anchors) != set(resolver.anchor_ids):
        raise V05MetricError("truth join does not exactly cover certificate anchors")
    for item in rows:
        certificate = resolver.record_for_anchor(item.source_anchor_id)
        if item.task_id != certificate.task_id:
            raise V05MetricError("truth task disagrees with certificate manifest")
        if item.opaque_certified_policy_id != certificate.opaque_certified_policy_id:
            raise V05MetricError("truth policy disagrees with certificate manifest")
    return tuple(sorted(rows, key=lambda item: item.opaque_query_id))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise V05MetricError("cannot average an empty metric")
    return math.fsum(values) / len(values)


def _task_metrics(
    task_id: str,
    rows: Sequence[tuple[PredictionRanking, TruthBinding]],
    recall_ks: tuple[int, ...],
) -> TaskRetrievalMetrics:
    anchor_ranks: list[int] = []
    policy_ranks: list[int] = []
    for prediction, truth in rows:
        anchor_ranks.append(
            prediction.ranked_anchor_ids.index(truth.source_anchor_id) + 1
        )
        policy_ranks.append(
            prediction.ranked_policy_ids.index(truth.opaque_certified_policy_id) + 1
        )
    return TaskRetrievalMetrics(
        task_id=task_id,
        anchor_count=len(rows),
        anchor_hit_at_1=_mean([float(rank == 1) for rank in anchor_ranks]),
        anchor_mrr=_mean([1.0 / rank for rank in anchor_ranks]),
        anchor_recall_at_k={
            k: _mean([float(rank <= k) for rank in anchor_ranks]) for k in recall_ks
        },
        policy_hit_at_1=_mean([float(rank == 1) for rank in policy_ranks]),
        policy_mrr=_mean([1.0 / rank for rank in policy_ranks]),
        policy_recall_at_k={
            k: _mean([float(rank <= k) for rank in policy_ranks]) for k in recall_ks
        },
    )


def _evaluate_group(
    group_rows: Sequence[PredictionRanking],
    truths: tuple[TruthBinding, ...],
    resolver: CertificateResolver,
    recall_ks: tuple[int, ...],
) -> RetrievalMetrics:
    first = group_rows[0]
    truth_by_query = {item.opaque_query_id: item for item in truths}
    prediction_by_query = {item.opaque_query_id: item for item in group_rows}
    if set(prediction_by_query) != set(truth_by_query):
        raise V05MetricError(
            "prediction group does not exactly cover the sealed truth-join queries"
        )

    paired: list[tuple[PredictionRanking, TruthBinding]] = []
    all_anchor_ids = set(resolver.anchor_ids)
    all_policy_ids = set(resolver.policy_ids)
    records = resolver.manifest.bindings
    for query_id in sorted(truth_by_query):
        prediction = prediction_by_query[query_id]
        truth = truth_by_query[query_id]
        if first.endpoint == MARKET_30_CERT:
            expected_anchors = all_anchor_ids
            expected_policies = all_policy_ids
        else:
            truth_certificate = resolver.record_for_anchor(truth.source_anchor_id)
            expected_anchors = {
                item.source_anchor_id
                for item in records
                if item.task_id == truth.task_id
                and item.execution_abi_digest == truth_certificate.execution_abi_digest
            }
            expected_policies = {
                item.opaque_certified_policy_id
                for item in records
                if item.task_id == truth.task_id
                and item.execution_abi_digest == truth_certificate.execution_abi_digest
            }
        if set(prediction.ranked_anchor_ids) != expected_anchors:
            raise V05MetricError(
                "ranked anchors do not exactly cover the endpoint candidate set"
            )
        if set(prediction.ranked_policy_ids) != expected_policies:
            raise V05MetricError(
                "ranked policies do not exactly cover the endpoint candidate set"
            )
        paired.append((prediction, truth))

    by_task: dict[str, list[tuple[PredictionRanking, TruthBinding]]] = {}
    for item in paired:
        by_task.setdefault(item[1].task_id, []).append(item)
    per_task = tuple(
        _task_metrics(task, by_task[task], recall_ks) for task in sorted(by_task)
    )
    return RetrievalMetrics(
        method_id=first.method_id,
        endpoint=first.endpoint,
        budget_episodes=first.budget_episodes,
        anchor_count=len(paired),
        task_count=len(per_task),
        anchor_hit_at_1=_mean([item.anchor_hit_at_1 for item in per_task]),
        anchor_mrr=_mean([item.anchor_mrr for item in per_task]),
        anchor_recall_at_k={
            k: _mean([item.anchor_recall_at_k[k] for item in per_task])
            for k in recall_ks
        },
        policy_hit_at_1=_mean([item.policy_hit_at_1 for item in per_task]),
        policy_mrr=_mean([item.policy_mrr for item in per_task]),
        policy_recall_at_k={
            k: _mean([item.policy_recall_at_k[k] for item in per_task])
            for k in recall_ks
        },
        per_task=per_task,
    )


def evaluate_sealed_predictions(
    prediction_seal: RankingSeal,
    truth_join: Iterable[TruthBinding | Mapping[str, Any]],
    certificate_manifest: CertifiedPolicyManifest,
    *,
    recall_ks: Sequence[int] = (1, 3, 5),
    expected_method_ids: Sequence[str] | None = None,
    expected_endpoints: Sequence[str] = (MARKET_30_CERT, TASK_5_CERT),
    expected_budgets: Sequence[int] = BUDGET_EPISODES,
) -> SealedEvaluation:
    """Join private truth only after verifying a complete prediction seal.

    This is a pure function: it reads the v04a semantic ranking seal, validates
    the frozen certificate projection and full query/candidate coverage, and
    returns metrics without writing artifacts or changing rankings.
    """

    if not isinstance(prediction_seal, RankingSeal):
        raise V05MetricError("evaluation requires a v04a RankingSeal")
    if not isinstance(certificate_manifest, CertifiedPolicyManifest):
        raise V05MetricError("evaluation requires a CertifiedPolicyManifest")
    try:
        sealed_payload = prediction_seal.rankings
        verify_ranking_seal(prediction_seal, sealed_payload)
    except V04AProtocolError as error:
        raise V05MetricError("prediction ranking seal verification failed") from error
    fields = {"schema", "predictions"}
    _exact_keys(sealed_payload, fields, "sealed prediction payload")
    if sealed_payload["schema"] != PREDICTION_PAYLOAD_SCHEMA:
        raise V05MetricError("unsupported sealed prediction payload schema")
    raw_predictions = sealed_payload["predictions"]
    if not isinstance(raw_predictions, list) or not raw_predictions:
        raise V05MetricError("sealed predictions must be a non-empty list")
    predictions = tuple(PredictionRanking.from_dict(item) for item in raw_predictions)
    if expected_method_ids is None:
        from .classifiers import P0_METHOD_IDS

        expected_method_ids = P0_METHOD_IDS
    predictions = require_prediction_cell_coverage(
        predictions,
        expected_method_ids=expected_method_ids,
        expected_endpoints=expected_endpoints,
        expected_budgets=expected_budgets,
    )

    ks = tuple(_integer(item, "recall k") for item in recall_ks)
    if not ks or len(ks) != len(set(ks)):
        raise V05MetricError("recall_ks must be non-empty and unique")
    ks = tuple(sorted(ks))
    resolver = CertificateResolver(certificate_manifest)
    truths = _truth_rows(truth_join, resolver)
    if any(
        item.prediction_seal_digest != prediction_seal.rankings_digest
        for item in truths
    ):
        raise V05MetricError("truth release belongs to another prediction seal")

    predictions_by_query: dict[str, list[PredictionRanking]] = {}
    for item in predictions:
        predictions_by_query.setdefault(item.opaque_query_id, []).append(item)
    truths_by_query = {item.opaque_query_id: item for item in truths}
    if set(predictions_by_query) != set(truths_by_query):
        raise V05MetricError(
            "sealed predictions and truth join cover different opaque queries"
        )
    for query_id, query_rows in predictions_by_query.items():
        truth_digest = truths_by_query[query_id].authorized_query_manifest_digest
        if any(
            item.authorized_query_manifest_digest != truth_digest for item in query_rows
        ):
            raise V05MetricError(
                f"query {query_id} truth/authorized manifest binding differs"
            )

    grouped: dict[tuple[str, str, int], list[PredictionRanking]] = {}
    for item in predictions:
        grouped.setdefault(
            (item.method_id, item.endpoint, item.budget_episodes), []
        ).append(item)
    metrics = tuple(
        _evaluate_group(grouped[key], truths, resolver, ks) for key in sorted(grouped)
    )
    truth_payload = [item.to_dict() for item in truths]
    return SealedEvaluation(
        prediction_seal_digest=prediction_seal.rankings_digest,
        certificate_manifest_digest=certificate_manifest.certificate_manifest_digest,
        truth_join_digest=sha256_json(truth_payload),
        metrics=metrics,
    )


def task_equal_macro(values_by_task: Mapping[str, Sequence[Any]]) -> float:
    """Average within task, then average task means with equal task weight."""

    if not isinstance(values_by_task, Mapping) or not values_by_task:
        raise V05MetricError("values_by_task must be a non-empty object")
    task_means: list[float] = []
    for task, raw_values in sorted(values_by_task.items()):
        _canonical_string(task, "task_id")
        if isinstance(raw_values, (str, bytes)):
            raise V05MetricError("each task value must be a non-empty sequence")
        try:
            values = tuple(_finite(item, "task metric") for item in raw_values)
        except TypeError as error:
            raise V05MetricError(
                "each task value must be a non-empty sequence"
            ) from error
        if not values:
            raise V05MetricError("each task must have metric coverage")
        task_means.append(_mean(values))
    return _mean(task_means)


def normalized_log2_budget_auc(
    budgets_or_values: Mapping[Any, Any] | Sequence[Any],
    values: Sequence[Any] | None = None,
) -> float:
    """Normalized trapezoid AUC on ``x=log2(budget)``.

    Pass either ``{budget: metric}`` or two equally-sized budget/value
    sequences.  At least two distinct positive budgets are required.
    """

    if values is None:
        if not isinstance(budgets_or_values, Mapping):
            raise V05MetricError(
                "one-argument AUC form requires a budget-to-value mapping"
            )
        raw_pairs = tuple(budgets_or_values.items())
    else:
        if isinstance(budgets_or_values, Mapping) or isinstance(
            budgets_or_values, (str, bytes)
        ):
            raise V05MetricError("two-argument AUC budgets must be a sequence")
        raw_budgets = tuple(budgets_or_values)
        raw_values = tuple(values)
        if len(raw_budgets) != len(raw_values):
            raise V05MetricError("AUC budgets and values must have equal length")
        raw_pairs = tuple(zip(raw_budgets, raw_values, strict=True))
    if len(raw_pairs) < 2:
        raise V05MetricError("normalized budget AUC requires at least two budgets")
    pairs = tuple(
        (_integer(budget, "budget"), _finite(value, "budget metric"))
        for budget, value in raw_pairs
    )
    budgets = [item[0] for item in pairs]
    if len(budgets) != len(set(budgets)):
        raise V05MetricError("budget AUC contains duplicate budgets")
    ordered = sorted(pairs)
    x = np.log2(np.asarray([item[0] for item in ordered], dtype=np.float64))
    y = np.asarray([item[1] for item in ordered], dtype=np.float64)
    width = float(x[-1] - x[0])
    if not math.isfinite(width) or width <= 0.0:
        raise V05MetricError("budget AUC has a degenerate log2 domain")
    area = (
        float(np.trapezoid(y, x=x))
        if hasattr(np, "trapezoid")
        else float(np.trapz(y, x=x))
    )
    result = area / width
    if not math.isfinite(result):
        raise V05MetricError("normalized budget AUC is not finite")
    return result


def confusion_matrix(
    truth_labels: Sequence[Any],
    predicted_labels: Sequence[Any],
    *,
    label_order: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return JSON-ready counts with rows=true and columns=predicted.

    An explicit order is preserved exactly.  Otherwise the union of observed
    canonical labels is sorted, making the result independent of row order.
    """

    if isinstance(truth_labels, (str, bytes)) or isinstance(
        predicted_labels, (str, bytes)
    ):
        raise V05MetricError("confusion inputs must be label sequences")
    try:
        truth = tuple(_canonical_string(item, "truth label") for item in truth_labels)
        predicted = tuple(
            _canonical_string(item, "predicted label") for item in predicted_labels
        )
    except TypeError as error:
        raise V05MetricError("confusion inputs must be label sequences") from error
    if not truth or len(truth) != len(predicted):
        raise V05MetricError(
            "confusion inputs must be non-empty sequences of equal length"
        )

    if label_order is None:
        labels = tuple(sorted(set(truth) | set(predicted)))
    else:
        if isinstance(label_order, (str, bytes)):
            raise V05MetricError("confusion label_order must be a sequence")
        try:
            labels = tuple(
                _canonical_string(item, "confusion label") for item in label_order
            )
        except TypeError as error:
            raise V05MetricError("confusion label_order must be a sequence") from error
        if not labels or len(labels) != len(set(labels)):
            raise V05MetricError(
                "confusion label_order must be non-empty and contain no duplicates"
            )
    unknown = (set(truth) | set(predicted)) - set(labels)
    if unknown:
        raise V05MetricError(
            f"confusion label_order does not cover labels: {sorted(unknown)}"
        )

    positions = {label: index for index, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth_label, predicted_label in zip(truth, predicted, strict=True):
        counts[positions[truth_label], positions[predicted_label]] += 1
    return {"label_order": list(labels), "counts": counts.tolist()}


def macro_f1(confusion: Mapping[str, Any]) -> float:
    """Compute class-equal macro-F1 from :func:`confusion_matrix` output.

    A label with neither truth nor prediction mass contributes zero.  Callers
    can compute this once per task and pass the values to ``task_equal_macro``
    when the report requires class-then-task equal weighting.
    """

    _exact_keys(confusion, {"label_order", "counts"}, "confusion")
    raw_labels = confusion["label_order"]
    if not isinstance(raw_labels, (list, tuple)):
        raise V05MetricError("confusion label_order must be a sequence")
    labels = tuple(_canonical_string(item, "confusion label") for item in raw_labels)
    if not labels or len(labels) != len(set(labels)):
        raise V05MetricError(
            "confusion label_order must be non-empty and contain no duplicates"
        )
    try:
        counts = np.asarray(confusion["counts"])
    except (TypeError, ValueError) as error:
        raise V05MetricError(
            "confusion counts must be a square numeric matrix"
        ) from error
    if counts.shape != (len(labels), len(labels)) or not np.issubdtype(
        counts.dtype, np.number
    ):
        raise V05MetricError("confusion counts must be a square numeric matrix")
    numeric = counts.astype(np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0.0)
        or not np.array_equal(numeric, np.floor(numeric))
    ):
        raise V05MetricError("confusion counts must be finite nonnegative integers")

    true_mass = np.sum(numeric, axis=1)
    predicted_mass = np.sum(numeric, axis=0)
    true_positive = np.diag(numeric)
    denominator = true_mass + predicted_mass
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0.0,
    )
    return float(np.mean(per_class))


def summarize_budget_auc_coverage(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_method_ids: Sequence[str],
    expected_endpoints: Sequence[str] = (MARKET_30_CERT, TASK_5_CERT),
    expected_budgets: Sequence[int] = (1, 2, 4),
) -> tuple[dict[str, Any], ...]:
    """Validate one complete metric panel and summarize each budget curve.

    Each input row supplies ``method_id``, ``endpoint``, ``budget_episodes``
    and one finite ``value``.  The default domain is the actually authorized
    development panel, not the seven-budget confirmatory schedule.
    """

    methods = tuple(
        _canonical_string(item, "expected method ID") for item in expected_method_ids
    )
    endpoints = tuple(
        _canonical_string(item, "expected endpoint") for item in expected_endpoints
    )
    budgets = tuple(_integer(item, "expected budget") for item in expected_budgets)
    if not methods or len(methods) != len(set(methods)):
        raise V05MetricError("expected method IDs must be non-empty and unique")
    if (
        not endpoints
        or len(endpoints) != len(set(endpoints))
        or not set(endpoints).issubset(_ENDPOINTS)
    ):
        raise V05MetricError("expected endpoints must be supported and unique")
    if (
        len(budgets) < 2
        or len(budgets) != len(set(budgets))
        or not set(budgets).issubset(BUDGET_EPISODES)
    ):
        raise V05MetricError(
            "expected budgets must contain at least two unique frozen budgets"
        )
    budget_domain = tuple(sorted(budgets))

    try:
        raw_rows = tuple(rows)
    except TypeError as error:
        raise V05MetricError("budget results must be an iterable of rows") from error
    if not raw_rows:
        raise V05MetricError("budget results must not be empty")
    required_fields = {"method_id", "endpoint", "budget_episodes", "value"}
    values_by_cell: dict[tuple[str, str, int], float] = {}
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping) or not required_fields.issubset(row):
            raise V05MetricError(
                f"budget result row {index} must contain {sorted(required_fields)}"
            )
        method_id = _canonical_string(row["method_id"], "method_id")
        endpoint = _canonical_string(row["endpoint"], "endpoint")
        budget = _integer(row["budget_episodes"], "budget_episodes")
        if method_id not in methods:
            raise V05MetricError("budget result contains an unexpected method")
        if endpoint not in endpoints:
            raise V05MetricError("budget result contains an unexpected endpoint")
        if budget not in budget_domain:
            raise V05MetricError("budget result contains an unexpected budget")
        cell = (method_id, endpoint, budget)
        if cell in values_by_cell:
            raise V05MetricError("budget results contain a duplicate cell")
        values_by_cell[cell] = _finite(row["value"], "budget metric")

    expected_cells = {
        (method_id, endpoint, budget)
        for method_id in methods
        for endpoint in endpoints
        for budget in budget_domain
    }
    if set(values_by_cell) != expected_cells:
        missing = sorted(expected_cells - set(values_by_cell))
        raise V05MetricError(f"budget result coverage is incomplete; missing={missing}")

    summaries: list[dict[str, Any]] = []
    for method_id in sorted(methods):
        for endpoint in sorted(endpoints):
            values = [
                values_by_cell[(method_id, endpoint, budget)]
                for budget in budget_domain
            ]
            summaries.append(
                {
                    "method_id": method_id,
                    "endpoint": endpoint,
                    "budget_episodes": list(budget_domain),
                    "values": values,
                    "normalized_log2_auc": normalized_log2_budget_auc(
                        budget_domain, values
                    ),
                    "observed_cell_count": len(values),
                    "expected_cell_count": len(budget_domain),
                    "coverage": 1.0,
                }
            )
    return tuple(summaries)


def _task_equal_macro_f1(
    rows: Sequence[Mapping[str, Any]],
    kind: str,
    label_order: tuple[str, ...],
    labels_by_task: Mapping[str, tuple[str, ...]],
) -> float:
    positions = {label: index for index, label in enumerate(label_order)}
    task_values = {}
    for task_id, task_labels in labels_by_task.items():
        task_rows = tuple(item for item in rows if item["task_id"] == task_id)
        counts = np.asarray(
            confusion_matrix(
                [item[f"truth_{kind}_id"] for item in task_rows],
                [item[f"top1_{kind}_id"] for item in task_rows],
                label_order=label_order,
            )["counts"],
            dtype=np.float64,
        )
        indices = np.asarray(
            [positions[label] for label in task_labels], dtype=np.int64
        )
        denominator = np.sum(counts, axis=1) + np.sum(counts, axis=0)
        scores = np.divide(
            2.0 * np.diag(counts),
            denominator,
            out=np.zeros(len(label_order), dtype=np.float64),
            where=denominator > 0.0,
        )
        task_values[task_id] = (float(np.mean(scores[indices])),)
    return task_equal_macro(task_values)


def build_development_report(
    prediction_seal: RankingSeal,
    truth_join: Iterable[TruthBinding | Mapping[str, Any]],
    certificate_manifest: CertifiedPolicyManifest,
    expected_method_ids: Sequence[str],
    expected_budgets: Sequence[int] = (1, 2, 4),
) -> dict[str, Any]:
    """Build the JSON-ready development report from one immutable seal."""

    raw_truth = tuple(truth_join)
    evaluation = evaluate_sealed_predictions(
        prediction_seal,
        raw_truth,
        certificate_manifest,
        expected_method_ids=expected_method_ids,
        expected_budgets=expected_budgets,
    )
    budgets = tuple(
        sorted(_integer(item, "expected budget") for item in expected_budgets)
    )
    if budgets != (1, 2, 4):
        raise V05MetricError("development report requires exactly budgets (1, 2, 4)")

    resolver = CertificateResolver(certificate_manifest)
    truths = _truth_rows(raw_truth, resolver)
    truth_by_query = {item.opaque_query_id: item for item in truths}
    predictions = tuple(
        PredictionRanking.from_dict(item)
        for item in prediction_seal.rankings["predictions"]
    )
    long_rows: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda item: item.cell_key):
        truth = truth_by_query[prediction.opaque_query_id]
        anchor_rank = prediction.ranked_anchor_ids.index(truth.source_anchor_id) + 1
        truth_policy = truth.opaque_certified_policy_id
        policy_rank = prediction.ranked_policy_ids.index(truth_policy) + 1
        long_rows.append(
            {
                "method_id": prediction.method_id,
                "endpoint": prediction.endpoint,
                "budget_episodes": prediction.budget_episodes,
                "opaque_query_id": prediction.opaque_query_id,
                "task_id": truth.task_id,
                "truth_anchor_id": truth.source_anchor_id,
                "top1_anchor_id": prediction.ranked_anchor_ids[0],
                "anchor_rank": anchor_rank,
                "truth_policy_id": truth_policy,
                "top1_policy_id": prediction.ranked_policy_ids[0],
                "policy_rank": policy_rank,
            }
        )

    anchor_order = tuple(sorted(resolver.anchor_ids))
    policy_order = tuple(sorted(resolver.policy_ids))
    labels_by_task = {
        task_id: (
            tuple(
                item.source_anchor_id
                for item in certificate_manifest.bindings
                if item.task_id == task_id
            ),
            tuple(
                sorted(
                    {
                        item.opaque_certified_policy_id
                        for item in certificate_manifest.bindings
                        if item.task_id == task_id
                    }
                )
            ),
        )
        for task_id in sorted({item.task_id for item in truths})
    }

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in long_rows:
        key = (row["method_id"], row["endpoint"], row["budget_episodes"])
        grouped.setdefault(key, []).append(row)
    confusion_rows: list[dict[str, Any]] = []
    for (method_id, endpoint, budget), rows in sorted(grouped.items()):
        anchor = confusion_matrix(
            [item["truth_anchor_id"] for item in rows],
            [item["top1_anchor_id"] for item in rows],
            label_order=anchor_order,
        )
        policy = confusion_matrix(
            [item["truth_policy_id"] for item in rows],
            [item["top1_policy_id"] for item in rows],
            label_order=policy_order,
        )
        confusion_rows.append(
            {
                "method_id": method_id,
                "endpoint": endpoint,
                "budget_episodes": budget,
                "anchor_confusion": anchor,
                "policy_confusion": policy,
                "anchor_global_macro_f1": macro_f1(anchor),
                "policy_global_macro_f1": macro_f1(policy),
                "anchor_task_equal_macro_f1": _task_equal_macro_f1(
                    rows,
                    "anchor",
                    anchor_order,
                    {task: labels[0] for task, labels in labels_by_task.items()},
                ),
                "policy_task_equal_macro_f1": _task_equal_macro_f1(
                    rows,
                    "policy",
                    policy_order,
                    {task: labels[1] for task, labels in labels_by_task.items()},
                ),
            }
        )

    auc_rows: list[dict[str, Any]] = []
    for metric_name in ("anchor_hit_at_1", "policy_hit_at_1"):
        curve_rows = (
            {
                "method_id": item.method_id,
                "endpoint": item.endpoint,
                "budget_episodes": item.budget_episodes,
                "value": getattr(item, metric_name),
            }
            for item in evaluation.metrics
        )
        summaries = summarize_budget_auc_coverage(
            curve_rows,
            expected_method_ids=expected_method_ids,
            expected_budgets=budgets,
        )
        auc_rows += [{"metric": metric_name, **item} for item in summaries]
    return {
        "schema": "policy-learnware.v05-development-report.v1",
        "evaluation": evaluation.to_dict(),
        "per_query_rows": long_rows,
        "confusion_rows": confusion_rows,
        "budget_auc_rows": auc_rows,
    }


__all__ = [
    "MARKET_30_CERT",
    "PREDICTION_PAYLOAD_SCHEMA",
    "TASK_5_CERT",
    "PredictionRanking",
    "RetrievalMetrics",
    "SealedEvaluation",
    "TaskRetrievalMetrics",
    "TruthBinding",
    "V05MetricError",
    "build_development_report",
    "confusion_matrix",
    "evaluate_sealed_predictions",
    "macro_f1",
    "normalized_log2_budget_auc",
    "prediction_payload",
    "require_prediction_cell_coverage",
    "summarize_budget_auc_coverage",
    "task_equal_macro",
]
