"""Typed development-oracle admission for Policy Learnware v0.2.

This module is the production boundary between immutable public selection and
deployment-private evaluation.  It deliberately has no loader for aggregate
value maps: the only performance evidence it accepts is a complete sequence of
typed :class:`OracleEpisodeRow` objects.  The full anonymous market, private
bundle/ABI registry, development query universe, selector method universe, and
all scientific evaluation literals are frozen in one digest before replay.

Only the v0.2 development split is projected from configuration.  No joint,
confirmatory, sealed, or safety artifact path is accepted or inspected here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .config import V02ExperimentConfig
from .market import DeploymentPrivateEntry, V02PolicyMarket
from .metrics import (
    RankingMetrics,
    SelectionMetrics,
    compute_ranking_metrics,
    compute_selection_metrics,
)
from .oracle import (
    FullPoolOracleResult,
    OracleEpisodeRow,
    PublishedSelection,
    aggregate_full_pool_oracle,
    minimum_executable_set,
)
from .schemas import ExecutionABIRecord
from .selectors import RankingRow, SelectionRecord


DEVELOPMENT_ORACLE_PROTOCOL_SCHEMA = (
    "policy-learnware.v02-development-oracle-protocol.v0"
)
DEVELOPMENT_ORACLE_RESULT_SCHEMA = (
    "policy-learnware.v02-development-oracle-admission.v0"
)


class DevelopmentOracleAdmissionError(ValueError):
    """Raw development evidence or its frozen coverage contract is invalid."""


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DevelopmentOracleAdmissionError(
            f"{where} must be a non-empty canonical string"
        )
    return value


def _digest(value: Any, where: str) -> str:
    result = _identifier(value, where).lower()
    if len(result) != 64:
        raise DevelopmentOracleAdmissionError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise DevelopmentOracleAdmissionError(
            f"{where} must be a SHA-256 digest"
        ) from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise DevelopmentOracleAdmissionError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DevelopmentOracleAdmissionError(f"{where} must be finite")
    return result


def _unit_interval(value: Any, where: str) -> float:
    result = _finite(value, where)
    if not 0.0 <= result <= 1.0:
        raise DevelopmentOracleAdmissionError(f"{where} must lie in [0, 1]")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DevelopmentOracleAdmissionError(f"{where} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise DevelopmentOracleAdmissionError(f"{where} must be a positive integer")
    return result


def _exact_ids(values: Sequence[str], where: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise DevelopmentOracleAdmissionError(f"{where} must be a sequence")
    parsed = tuple(_identifier(item, f"{where}[]") for item in values)
    if not parsed or len(parsed) != len(set(parsed)):
        raise DevelopmentOracleAdmissionError(
            f"{where} must be non-empty and unique"
        )
    return tuple(sorted(parsed))


def _registry_digest(
    policy_market_id: str,
    registry: Mapping[str, DeploymentPrivateEntry],
) -> str:
    """Bind the complete private entries, including bundle and minimum ABI."""

    return sha256_json(
        {
            "schema": "policy-learnware.v02-development-deployment-census.v0",
            "policy_market_id": policy_market_id,
            "entries": {
                opaque_id: registry[opaque_id].to_dict()
                for opaque_id in sorted(registry)
            },
        }
    )


def _development_config_projection(config: V02ExperimentConfig) -> dict[str, Any]:
    """Return only the development-facing part of a v0.2 config.

    In particular, this projection never traverses confirmatory or safety
    target collections, so a development replay cannot accidentally bind to or
    disclose future joint-confirmatory material.
    """

    return {
        "schema": "policy-learnware.v02-development-config-projection.v0",
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "protocol_family_id": config.protocol_family_id,
        "development_targets": [
            target.to_dict() for target in config.development_targets
        ],
        "method_ids": list(config.method_ids),
    }


@dataclass(frozen=True)
class DevelopmentTargetEvaluationProtocol:
    """Frozen private execution contract for one opaque development target."""

    opaque_query_id: str
    private_target_instance_digest: str
    target_evidence_digest: str
    target_execution_abi: ExecutionABIRecord
    seed_contract_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opaque_query_id",
            _identifier(self.opaque_query_id, "opaque_query_id"),
        )
        object.__setattr__(
            self,
            "private_target_instance_digest",
            _digest(
                self.private_target_instance_digest,
                "private_target_instance_digest",
            ),
        )
        object.__setattr__(
            self,
            "target_evidence_digest",
            _digest(self.target_evidence_digest, "target_evidence_digest"),
        )
        if not isinstance(self.target_execution_abi, ExecutionABIRecord):
            raise DevelopmentOracleAdmissionError(
                "target_execution_abi must be an ExecutionABIRecord"
            )
        object.__setattr__(
            self,
            "seed_contract_digest",
            _digest(self.seed_contract_digest, "seed_contract_digest"),
        )

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "opaque_query_id": self.opaque_query_id,
            "private_target_instance_digest": self.private_target_instance_digest,
            "target_evidence_digest": self.target_evidence_digest,
            "target_execution_abi": self.target_execution_abi.to_dict(),
            "seed_contract_digest": self.seed_contract_digest,
        }


@dataclass(frozen=True)
class FrozenDevelopmentOracleProtocol:
    """Exact development evaluation universe and all reviewed literals."""

    experiment_id: str
    stage: str
    protocol_family_id: str
    development_config_projection_digest: str
    policy_market_id: str
    market_ids: tuple[str, ...]
    deployment_registry_digest: str
    development_query_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    target_protocols: Mapping[str, DevelopmentTargetEvaluationProtocol]
    evaluation_protocol_id: str
    episodes_per_executable_policy: int
    failure_floor: float
    epsilon: float
    tie_atol: float
    candidate_paired_seeds: bool

    def __post_init__(self) -> None:
        for name in ("experiment_id", "stage", "protocol_family_id", "policy_market_id"):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        for name in (
            "development_config_projection_digest",
            "deployment_registry_digest",
            "evaluation_protocol_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        market_ids = _exact_ids(self.market_ids, "market_ids")
        query_ids = _exact_ids(self.development_query_ids, "development_query_ids")
        method_ids = _exact_ids(self.method_ids, "method_ids")
        if not isinstance(self.target_protocols, Mapping):
            raise DevelopmentOracleAdmissionError("target_protocols must be a mapping")
        targets = dict(self.target_protocols)
        if set(targets) != set(query_ids):
            raise DevelopmentOracleAdmissionError(
                "target protocols must exactly cover config-derived development query IDs"
            )
        for query_id, target in targets.items():
            if not isinstance(target, DevelopmentTargetEvaluationProtocol):
                raise DevelopmentOracleAdmissionError(
                    "target_protocols values have the wrong type"
                )
            if query_id != target.opaque_query_id:
                raise DevelopmentOracleAdmissionError(
                    "target protocol key differs from opaque_query_id"
                )
            if target.target_execution_abi.protocol_family_id != self.protocol_family_id:
                raise DevelopmentOracleAdmissionError(
                    "target minimum ABI uses another protocol family"
                )
        episode_count = _positive_int(
            self.episodes_per_executable_policy,
            "episodes_per_executable_policy",
        )
        floor = _unit_interval(self.failure_floor, "failure_floor")
        epsilon = _unit_interval(self.epsilon, "epsilon")
        tolerance = _finite(self.tie_atol, "tie_atol")
        if tolerance < 0.0:
            raise DevelopmentOracleAdmissionError("tie_atol cannot be negative")
        if type(self.candidate_paired_seeds) is not bool:
            raise DevelopmentOracleAdmissionError(
                "candidate_paired_seeds must be boolean"
            )
        object.__setattr__(self, "market_ids", market_ids)
        object.__setattr__(self, "development_query_ids", query_ids)
        object.__setattr__(self, "method_ids", method_ids)
        object.__setattr__(
            self,
            "target_protocols",
            MappingProxyType(dict(sorted(targets.items()))),
        )
        object.__setattr__(self, "episodes_per_executable_policy", episode_count)
        object.__setattr__(self, "failure_floor", floor)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "tie_atol", tolerance)

    @classmethod
    def from_config(
        cls,
        config: V02ExperimentConfig,
        market: V02PolicyMarket,
        target_protocols: Sequence[DevelopmentTargetEvaluationProtocol],
        *,
        evaluation_protocol_id: str,
        episodes_per_executable_policy: int,
        failure_floor: float,
        epsilon: float,
        tie_atol: float,
        candidate_paired_seeds: bool,
    ) -> "FrozenDevelopmentOracleProtocol":
        """Freeze development coverage without consulting any future split."""

        if not isinstance(config, V02ExperimentConfig):
            raise DevelopmentOracleAdmissionError(
                "config must be a V02ExperimentConfig"
            )
        if not isinstance(market, V02PolicyMarket):
            raise DevelopmentOracleAdmissionError("market must be a V02PolicyMarket")
        query_ids = tuple(target.target_id for target in config.development_targets)
        if not query_ids:
            raise DevelopmentOracleAdmissionError(
                "configuration has no development targets"
            )
        targets = tuple(target_protocols)
        if not targets or any(
            not isinstance(item, DevelopmentTargetEvaluationProtocol)
            for item in targets
        ):
            raise DevelopmentOracleAdmissionError(
                "target_protocols must be a non-empty typed sequence"
            )
        keyed = {item.opaque_query_id: item for item in targets}
        if len(keyed) != len(targets) or set(keyed) != set(query_ids):
            raise DevelopmentOracleAdmissionError(
                "target protocols must exactly match config-derived development target IDs"
            )
        projection = _development_config_projection(config)
        return cls(
            experiment_id=config.experiment_id,
            stage=config.stage,
            protocol_family_id=config.protocol_family_id,
            development_config_projection_digest=sha256_json(projection),
            policy_market_id=market.policy_market_id,
            market_ids=tuple(market.entries),
            deployment_registry_digest=_registry_digest(
                market.policy_market_id, market.deployment_private
            ),
            development_query_ids=query_ids,
            method_ids=config.method_ids,
            target_protocols=keyed,
            evaluation_protocol_id=evaluation_protocol_id,
            episodes_per_executable_policy=episodes_per_executable_policy,
            failure_floor=failure_floor,
            epsilon=epsilon,
            tie_atol=tie_atol,
            candidate_paired_seeds=candidate_paired_seeds,
        )

    @property
    def expected_selection_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                f"{method_id}::{query_id}"
                for query_id in self.development_query_ids
                for method_id in self.method_ids
            )
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema": DEVELOPMENT_ORACLE_PROTOCOL_SCHEMA,
            "scope": "v02-development-only",
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "protocol_family_id": self.protocol_family_id,
            "development_config_projection_digest": (
                self.development_config_projection_digest
            ),
            "policy_market_id": self.policy_market_id,
            "market_ids": list(self.market_ids),
            "deployment_registry_digest": self.deployment_registry_digest,
            "development_query_ids": list(self.development_query_ids),
            "method_ids": list(self.method_ids),
            "target_protocols": {
                query_id: target.to_private_dict()
                for query_id, target in self.target_protocols.items()
            },
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "episodes_per_executable_policy": self.episodes_per_executable_policy,
            "failure_floor": self.failure_floor,
            "epsilon": self.epsilon,
            "tie_atol": self.tie_atol,
            "candidate_paired_seeds": self.candidate_paired_seeds,
        }


@dataclass(frozen=True)
class PublishedSelectionRanking:
    """Immutable full-ranking evidence and its minimal published projection."""

    opaque_query_id: str
    published_selection: PublishedSelection
    full_ranking: tuple[RankingRow, ...]
    selection_record: SelectionRecord

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "opaque_query_id",
            _identifier(self.opaque_query_id, "opaque_query_id"),
        )
        if not isinstance(self.published_selection, PublishedSelection):
            raise DevelopmentOracleAdmissionError(
                "published_selection must be a PublishedSelection"
            )
        if not isinstance(self.selection_record, SelectionRecord):
            raise DevelopmentOracleAdmissionError(
                "selection_record must be a SelectionRecord"
            )
        ranking = tuple(self.full_ranking)
        if not ranking or any(not isinstance(row, RankingRow) for row in ranking):
            raise DevelopmentOracleAdmissionError(
                "full_ranking must be a non-empty tuple of RankingRow objects"
            )
        if ranking != self.selection_record.ranking:
            raise DevelopmentOracleAdmissionError(
                "full_ranking differs from its immutable selection record"
            )
        projected = PublishedSelection.from_selection_record(self.selection_record)
        if projected != self.published_selection:
            raise DevelopmentOracleAdmissionError(
                "PublishedSelection differs from the bound selection record"
            )
        object.__setattr__(self, "full_ranking", ranking)

    @classmethod
    def from_selection_record(
        cls, opaque_query_id: str, selection_record: SelectionRecord
    ) -> "PublishedSelectionRanking":
        if not isinstance(selection_record, SelectionRecord):
            raise DevelopmentOracleAdmissionError(
                "selection_record must be a SelectionRecord"
            )
        return cls(
            opaque_query_id=opaque_query_id,
            published_selection=PublishedSelection.from_selection_record(
                selection_record
            ),
            full_ranking=selection_record.ranking,
            selection_record=selection_record,
        )

    @property
    def method_id(self) -> str:
        return self.published_selection.method_id

    @property
    def unit_id(self) -> str:
        return f"{self.method_id}::{self.opaque_query_id}"

    @property
    def full_ranking_digest(self) -> str:
        return sha256_json([row.to_dict() for row in self.full_ranking])

    def to_binding_dict(self) -> dict[str, Any]:
        return {
            "opaque_query_id": self.opaque_query_id,
            "method_id": self.method_id,
            "selection_record_digest": (
                self.published_selection.selection_record_digest
            ),
            "selected_id": self.published_selection.selected_id,
            "full_ranking_digest": self.full_ranking_digest,
            "ranking_count": len(self.full_ranking),
        }


@dataclass(frozen=True)
class DevelopmentMethodMetrics:
    """Derived method metrics with all raw/private evidence bindings."""

    opaque_query_id: str
    method_id: str
    selected_id: str
    deployment_status: str
    protocol_digest: str
    full_pool_oracle_digest: str
    episode_rows_digest: str
    execution_abi_census_digest: str
    selection_record_digest: str
    full_ranking_digest: str
    selection_metrics: SelectionMetrics
    ranking_metrics: RankingMetrics

    def __post_init__(self) -> None:
        for name in (
            "opaque_query_id",
            "method_id",
            "selected_id",
            "deployment_status",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name)
            )
        for name in (
            "protocol_digest",
            "full_pool_oracle_digest",
            "episode_rows_digest",
            "execution_abi_census_digest",
            "selection_record_digest",
            "full_ranking_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.selection_metrics, SelectionMetrics):
            raise DevelopmentOracleAdmissionError(
                "selection_metrics has the wrong type"
            )
        if not isinstance(self.ranking_metrics, RankingMetrics):
            raise DevelopmentOracleAdmissionError("ranking_metrics has the wrong type")
        if self.selection_metrics.selected_policy_id != self.selected_id:
            raise DevelopmentOracleAdmissionError(
                "selection metrics selected ID differs from the publication"
            )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-development-method-metrics.v0",
            "opaque_query_id": self.opaque_query_id,
            "method_id": self.method_id,
            "selected_id": self.selected_id,
            "deployment_status": self.deployment_status,
            "protocol_digest": self.protocol_digest,
            "full_pool_oracle_digest": self.full_pool_oracle_digest,
            "episode_rows_digest": self.episode_rows_digest,
            "execution_abi_census_digest": self.execution_abi_census_digest,
            "selection_record_digest": self.selection_record_digest,
            "full_ranking_digest": self.full_ranking_digest,
            "selection_metrics": self.selection_metrics.to_dict(),
            "ranking_metrics": self.ranking_metrics.to_dict(),
            "ranking_scope": "abi-executable-pool-no-fallback",
        }


@dataclass(frozen=True)
class DevelopmentOracleAdmission:
    """Complete typed replay result for the frozen development universe."""

    protocol: FrozenDevelopmentOracleProtocol
    protocol_digest: str
    policy_market_id: str
    oracle_by_query: Mapping[str, FullPoolOracleResult]
    metrics_by_unit: Mapping[str, DevelopmentMethodMetrics]
    selection_bindings_digest: str
    raw_episode_rows_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, FrozenDevelopmentOracleProtocol):
            raise DevelopmentOracleAdmissionError(
                "protocol must be a FrozenDevelopmentOracleProtocol"
            )
        object.__setattr__(
            self, "protocol_digest", _digest(self.protocol_digest, "protocol_digest")
        )
        if self.protocol_digest != self.protocol.digest:
            raise DevelopmentOracleAdmissionError(
                "admission protocol digest differs from its frozen protocol"
            )
        object.__setattr__(
            self,
            "policy_market_id",
            _identifier(self.policy_market_id, "policy_market_id"),
        )
        for name in ("selection_bindings_digest", "raw_episode_rows_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        oracle = dict(self.oracle_by_query)
        metrics = dict(self.metrics_by_unit)
        if set(oracle) != set(self.protocol.development_query_ids) or any(
            not isinstance(result, FullPoolOracleResult)
            or query_id != result.opaque_query_id
            or set(result.market_ids) != set(self.protocol.market_ids)
            for query_id, result in oracle.items()
        ):
            raise DevelopmentOracleAdmissionError(
                "oracle_by_query differs from frozen query/market coverage"
            )
        if set(metrics) != set(self.protocol.expected_selection_unit_ids) or any(
            not isinstance(metric, DevelopmentMethodMetrics)
            or unit_id != f"{metric.method_id}::{metric.opaque_query_id}"
            or metric.protocol_digest != self.protocol_digest
            for unit_id, metric in metrics.items()
        ):
            raise DevelopmentOracleAdmissionError(
                "metrics_by_unit differs from frozen method/query coverage"
            )
        if self.policy_market_id != self.protocol.policy_market_id:
            raise DevelopmentOracleAdmissionError(
                "admission policy market differs from its frozen protocol"
            )
        object.__setattr__(
            self, "oracle_by_query", MappingProxyType(dict(sorted(oracle.items())))
        )
        object.__setattr__(
            self, "metrics_by_unit", MappingProxyType(dict(sorted(metrics.items())))
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema": DEVELOPMENT_ORACLE_RESULT_SCHEMA,
            "scope": "v02-development-only-private",
            "protocol_digest": self.protocol_digest,
            "policy_market_id": self.policy_market_id,
            "selection_bindings_digest": self.selection_bindings_digest,
            "raw_episode_rows_digest": self.raw_episode_rows_digest,
            "oracle_by_query": {
                query_id: result.to_private_dict()
                for query_id, result in self.oracle_by_query.items()
            },
            "metrics_by_unit": {
                unit_id: metric.to_private_dict()
                for unit_id, metric in self.metrics_by_unit.items()
            },
        }


def _validate_market_binding(
    protocol: FrozenDevelopmentOracleProtocol,
    market: V02PolicyMarket,
) -> None:
    if not isinstance(market, V02PolicyMarket):
        raise DevelopmentOracleAdmissionError("market must be a V02PolicyMarket")
    if market.policy_market_id != protocol.policy_market_id:
        raise DevelopmentOracleAdmissionError(
            "policy market differs from the frozen development protocol"
        )
    if set(market.entries) != set(protocol.market_ids):
        raise DevelopmentOracleAdmissionError(
            "submarket or expanded market differs from frozen exact coverage"
        )
    if set(market.deployment_private) != set(protocol.market_ids):
        raise DevelopmentOracleAdmissionError(
            "deployment-private registry does not cover the full market"
        )
    if any(
        not isinstance(entry, DeploymentPrivateEntry)
        for entry in market.deployment_private.values()
    ):
        raise DevelopmentOracleAdmissionError(
            "deployment-private registry entries have the wrong type"
        )
    actual = _registry_digest(market.policy_market_id, market.deployment_private)
    if actual != protocol.deployment_registry_digest:
        raise DevelopmentOracleAdmissionError(
            "deployment-private bundle/ABI registry differs from the frozen census"
        )


def _validate_selections(
    protocol: FrozenDevelopmentOracleProtocol,
    selections: Sequence[PublishedSelectionRanking],
) -> Mapping[str, PublishedSelectionRanking]:
    if isinstance(selections, (str, bytes, Mapping)):
        raise DevelopmentOracleAdmissionError(
            "selection evidence must be a typed sequence, not an uploaded map"
        )
    typed = tuple(selections)
    if not typed or any(
        not isinstance(item, PublishedSelectionRanking) for item in typed
    ):
        raise DevelopmentOracleAdmissionError(
            "selection evidence must contain PublishedSelectionRanking objects"
        )
    keyed = {item.unit_id: item for item in typed}
    if len(keyed) != len(typed) or set(keyed) != set(
        protocol.expected_selection_unit_ids
    ):
        raise DevelopmentOracleAdmissionError(
            "selection evidence omits or adds a frozen method/query work unit"
        )
    market_ids = set(protocol.market_ids)
    for unit_id, evidence in keyed.items():
        ranked_ids = tuple(
            row.opaque_learnware_id for row in evidence.full_ranking
        )
        if len(ranked_ids) != len(protocol.market_ids) or set(ranked_ids) != market_ids:
            raise DevelopmentOracleAdmissionError(
                f"selection {unit_id!r} does not rank the full frozen market"
            )
        if evidence.method_id not in protocol.method_ids:
            raise DevelopmentOracleAdmissionError(
                f"selection {unit_id!r} uses an unregistered method"
            )
        if evidence.opaque_query_id not in protocol.development_query_ids:
            raise DevelopmentOracleAdmissionError(
                f"selection {unit_id!r} uses a non-development query"
            )
        expected_evidence_digest = protocol.target_protocols[
            evidence.opaque_query_id
        ].target_evidence_digest
        if (
            evidence.selection_record.target_evidence_digest
            != expected_evidence_digest
        ):
            raise DevelopmentOracleAdmissionError(
                f"selection {unit_id!r} target evidence differs from the frozen raw query evidence"
            )
    return MappingProxyType(dict(sorted(keyed.items())))


def _validate_raw_rows(
    protocol: FrozenDevelopmentOracleProtocol,
    market: V02PolicyMarket,
    episode_rows: Sequence[OracleEpisodeRow],
) -> Mapping[str, tuple[OracleEpisodeRow, ...]]:
    if isinstance(episode_rows, (str, bytes, Mapping)):
        raise DevelopmentOracleAdmissionError(
            "oracle evidence must be typed raw episode rows; uploaded value maps are forbidden"
        )
    rows = tuple(episode_rows)
    if not rows or any(not isinstance(row, OracleEpisodeRow) for row in rows):
        raise DevelopmentOracleAdmissionError(
            "oracle evidence must be a non-empty sequence of OracleEpisodeRow objects"
        )
    if {row.opaque_query_id for row in rows} != set(
        protocol.development_query_ids
    ):
        raise DevelopmentOracleAdmissionError(
            "raw oracle query IDs differ from config-derived development coverage"
        )
    grouped: dict[str, tuple[OracleEpisodeRow, ...]] = {}
    for query_id in protocol.development_query_ids:
        target = protocol.target_protocols[query_id]
        query_rows = tuple(row for row in rows if row.opaque_query_id == query_id)
        executable = minimum_executable_set(
            protocol.market_ids,
            market.deployment_private,
            target.target_execution_abi,
        )
        if len(executable) < 2:
            raise DevelopmentOracleAdmissionError(
                "development ranking metrics require at least two executable policies"
            )
        by_policy = {
            opaque_id: tuple(
                row for row in query_rows if row.opaque_learnware_id == opaque_id
            )
            for opaque_id in protocol.market_ids
        }
        for opaque_id in protocol.market_ids:
            policy_rows = by_policy[opaque_id]
            if opaque_id not in executable:
                if policy_rows:
                    raise DevelopmentOracleAdmissionError(
                        "ABI-incompatible policies cannot be evaluated or used as fallback"
                    )
                continue
            if len(policy_rows) != protocol.episodes_per_executable_policy:
                raise DevelopmentOracleAdmissionError(
                    f"oracle episode count for {query_id!r}/{opaque_id!r} differs "
                    "from the frozen protocol"
                )
            indices = tuple(sorted(row.episode_index for row in policy_rows))
            if indices != tuple(range(protocol.episodes_per_executable_policy)):
                raise DevelopmentOracleAdmissionError(
                    "oracle episode indices must exactly cover the frozen range"
                )
            deployment = market.deployment_private[opaque_id]
            for row in policy_rows:
                _unit_interval(row.normalized_return, "oracle normalized_return")
                if row.private_target_instance_digest != (
                    target.private_target_instance_digest
                ):
                    raise DevelopmentOracleAdmissionError(
                        "oracle row target instance differs from its frozen target"
                    )
                if row.evaluation_protocol_id != protocol.evaluation_protocol_id:
                    raise DevelopmentOracleAdmissionError(
                        "oracle row evaluation protocol differs from the freeze"
                    )
                if row.seed_contract_digest != target.seed_contract_digest:
                    raise DevelopmentOracleAdmissionError(
                        "oracle row seed contract differs from the frozen target"
                    )
                if row.bundle_digest != deployment.bundle_digest:
                    raise DevelopmentOracleAdmissionError(
                        "oracle row bundle differs from the deployment-private registry"
                    )
        expected_count = len(executable) * protocol.episodes_per_executable_policy
        if len(query_rows) != expected_count:
            raise DevelopmentOracleAdmissionError(
                "oracle rows add an unknown policy or omit exact executable coverage"
            )
        grouped[query_id] = query_rows
    return MappingProxyType(grouped)


def recompute_development_oracle(
    protocol: FrozenDevelopmentOracleProtocol,
    *,
    market: V02PolicyMarket,
    episode_rows: Sequence[OracleEpisodeRow],
    selections: Sequence[PublishedSelectionRanking],
) -> DevelopmentOracleAdmission:
    """Rebuild full-pool oracle and metrics from typed development evidence.

    The API intentionally has no ``value_map``, fallback-selection, or expected
    coverage override.  Every universe is taken from ``protocol`` and compared
    against the immutable market/config bindings before aggregation.
    """

    if not isinstance(protocol, FrozenDevelopmentOracleProtocol):
        raise DevelopmentOracleAdmissionError(
            "protocol must be a FrozenDevelopmentOracleProtocol"
        )
    _validate_market_binding(protocol, market)
    selection_by_unit = _validate_selections(protocol, selections)
    rows_by_query = _validate_raw_rows(protocol, market, episode_rows)

    protocol_digest = protocol.digest
    oracle_by_query: dict[str, FullPoolOracleResult] = {}
    metrics_by_unit: dict[str, DevelopmentMethodMetrics] = {}
    selection_bindings = {
        unit_id: evidence.to_binding_dict()
        for unit_id, evidence in selection_by_unit.items()
    }

    for query_id in protocol.development_query_ids:
        target = protocol.target_protocols[query_id]
        query_evidence = tuple(
            selection_by_unit[f"{method_id}::{query_id}"]
            for method_id in protocol.method_ids
        )
        result = aggregate_full_pool_oracle(
            opaque_query_id=query_id,
            private_target_instance_digest=target.private_target_instance_digest,
            evaluation_protocol_id=protocol.evaluation_protocol_id,
            market_ids=protocol.market_ids,
            deployment_registry=market.deployment_private,
            target_execution_abi=target.target_execution_abi,
            episode_rows=rows_by_query[query_id],
            published_selections=tuple(
                evidence.published_selection for evidence in query_evidence
            ),
            failure_floor=protocol.failure_floor,
            tie_atol=protocol.tie_atol,
            candidate_paired_seeds=protocol.candidate_paired_seeds,
        )
        if any(
            result.episode_counts[opaque_id]
            != protocol.episodes_per_executable_policy
            for opaque_id in result.executable_ids
        ):
            raise DevelopmentOracleAdmissionError(
                "full-pool oracle episode counts differ from the frozen protocol"
            )
        oracle_by_query[query_id] = result
        executable_values = {
            opaque_id: float(result.normalized_value_vector[opaque_id])
            for opaque_id in result.executable_ids
        }
        executable_set = set(result.executable_ids)

        for evidence in query_evidence:
            selected_id = evidence.selection_record.selected_id
            selection_metrics = compute_selection_metrics(
                selected_policy_id=selected_id,
                normalized_returns_by_policy=executable_values,
                executable_policy_ids=result.executable_ids,
                incompatible_failure_value=protocol.failure_floor,
                epsilon=protocol.epsilon,
                tie_tolerance=protocol.tie_atol,
            )
            outcome = result.outcomes[evidence.method_id]
            if (
                not math.isclose(
                    selection_metrics.selected_normalized_return,
                    outcome.selected_value,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    selection_metrics.pool_regret,
                    outcome.regret,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or selection_metrics.top1_agreement
                != outcome.oracle_top1_agreement
            ):
                raise DevelopmentOracleAdmissionError(
                    "derived selection metrics disagree with the full-pool oracle"
                )
            executable_ranking = tuple(
                row.opaque_learnware_id
                for row in evidence.full_ranking
                if row.opaque_learnware_id in executable_set
            )
            ranking_metrics = compute_ranking_metrics(
                executable_ranking,
                executable_values,
                tie_tolerance=protocol.tie_atol,
            )
            unit_id = evidence.unit_id
            metrics_by_unit[unit_id] = DevelopmentMethodMetrics(
                opaque_query_id=query_id,
                method_id=evidence.method_id,
                selected_id=selected_id,
                deployment_status=outcome.deployment_status,
                protocol_digest=protocol_digest,
                full_pool_oracle_digest=result.digest,
                episode_rows_digest=result.episode_rows_digest,
                execution_abi_census_digest=result.execution_abi_census_digest,
                selection_record_digest=(
                    evidence.published_selection.selection_record_digest
                ),
                full_ranking_digest=evidence.full_ranking_digest,
                selection_metrics=selection_metrics,
                ranking_metrics=ranking_metrics,
            )

    sorted_rows = sorted(
        tuple(episode_rows),
        key=lambda row: (
            row.opaque_query_id,
            row.opaque_learnware_id,
            row.episode_index,
        ),
    )
    return DevelopmentOracleAdmission(
        protocol=protocol,
        protocol_digest=protocol_digest,
        policy_market_id=protocol.policy_market_id,
        oracle_by_query=oracle_by_query,
        metrics_by_unit=metrics_by_unit,
        selection_bindings_digest=sha256_json(
            {
                "schema": "policy-learnware.v02-development-selection-bindings.v0",
                "protocol_digest": protocol_digest,
                "units": selection_bindings,
            }
        ),
        raw_episode_rows_digest=sha256_json(
            {
                "schema": "policy-learnware.v02-development-oracle-raw-rows.v0",
                "protocol_digest": protocol_digest,
                "rows": [row.to_private_dict() for row in sorted_rows],
            }
        ),
    )


__all__ = [
    "DEVELOPMENT_ORACLE_PROTOCOL_SCHEMA",
    "DEVELOPMENT_ORACLE_RESULT_SCHEMA",
    "DevelopmentMethodMetrics",
    "DevelopmentOracleAdmission",
    "DevelopmentOracleAdmissionError",
    "DevelopmentTargetEvaluationProtocol",
    "FrozenDevelopmentOracleProtocol",
    "PublishedSelectionRanking",
    "recompute_development_oracle",
]
