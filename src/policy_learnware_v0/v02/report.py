"""Strictly separated v0.2 P/O/reference-E analysis table schemas.

The development P-table contains only opaque query/method outcomes.  The
private O-table is the sole table containing benchmark truth, full candidate
values, and ABI-census-derived diagnostics.  The reference E-table contains
representation evidence but no target identity or full-pool oracle vectors.
These are separate serialisation APIs on purpose; there is no combined public
report payload that could accidentally carry the private O-table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json
from .oracle import (
    DeploymentStatus,
    FullPoolOracleResult,
    OracleSelectionOutcome,
)


ReferenceKind = Literal["raw", "legacy_corro", "refit_corro"]


class ReportContractError(ValueError):
    """A report row is malformed or crosses an information boundary."""


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReportContractError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _identifier(value, where).lower()
    if len(result) != 64:
        raise ReportContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise ReportContractError(f"{where} must be a SHA-256 digest") from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ReportContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportContractError(f"{where} must be finite")
    return result


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ReportContractError(f"{where} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ReportContractError(f"{where} must be a non-negative integer")
    return result


def _positive_int(value: Any, where: str) -> int:
    result = _nonnegative_int(value, where)
    if result == 0:
        raise ReportContractError(f"{where} must be positive")
    return result


def _deep_freeze(value: Any, where: str) -> Any:
    try:
        canonical = canonicalize(value)
    except (TypeError, ValueError) as error:
        raise ReportContractError(f"{where} is not canonical JSON: {error}") from error
    if isinstance(canonical, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item, where) for key, item in canonical.items()}
        )
    if isinstance(canonical, list):
        return tuple(_deep_freeze(item, where) for item in canonical)
    return canonical


# These key families are legal only in the private O-table.  P/E rows are
# typed, but the recursive audit is retained as defense in depth against a
# future schema extension accidentally copying an oracle-private mapping.
_PRIVATE_ORACLE_KEY_FRAGMENTS = (
    "execution_abi",
    "value_vector",
    "target_instance",
    "episode_rows",
    "bundle_digest",
)
_PRIVATE_ORACLE_KEYS = frozenset(
    {
        "true_task_id",
        "true_axis_id",
        "true_factor",
        "physical_nearest_anchor_id",
        "true_distance_lmin_selected_id",
        "executable_ids",
        "incompatible_ids",
        "best_in_pool_ids",
        "best_in_pool_value",
        "source_global_champion_id",
    }
)


def assert_nonprivate_table_payload(payload: Mapping[str, Any], *, table: str) -> None:
    """Reject O-table-only fields from development P or reference E payloads."""

    if not isinstance(payload, Mapping):
        raise ReportContractError(f"{table} payload must be a mapping")

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                if not isinstance(raw_key, str):
                    raise ReportContractError(f"{table} has a non-string key at {path}")
                key = raw_key.lower()
                if key in _PRIVATE_ORACLE_KEYS or any(
                    fragment in key for fragment in _PRIVATE_ORACLE_KEY_FRAGMENTS
                ):
                    raise ReportContractError(
                        f"{table} leaks private oracle field {raw_key!r} at {path}"
                    )
                visit(item, f"{path}.{raw_key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(payload, "$")


@dataclass(frozen=True)
class DevelopmentPTableRow:
    """One method/query-bank/prefix development result with opaque identity."""

    opaque_query_id: str
    bank_index: int
    prefix: int
    method_id: str
    representation_id: str
    selection_record_digest: str
    deployment_status: DeploymentStatus
    selected_id: str | None
    selected_normalized_return: float
    pool_regret: float
    epsilon_optimal: bool
    top1_agreement: bool
    target_transition_count: int
    target_gradient_updates: int
    target_evidence_digest: str
    evidence_contract_digest: str
    cost_digest: str

    def __post_init__(self) -> None:
        for name in ("opaque_query_id", "method_id", "representation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "bank_index", _nonnegative_int(self.bank_index, "bank_index")
        )
        object.__setattr__(self, "prefix", _positive_int(self.prefix, "prefix"))
        for name in (
            "selection_record_digest",
            "target_evidence_digest",
            "evidence_contract_digest",
            "cost_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.deployment_status not in {
            "SELECTED_EXECUTABLE",
            "SELECTED_INCOMPATIBLE_ABI",
            "NO_SELECTION",
        }:
            raise ReportContractError(
                f"unknown deployment status: {self.deployment_status!r}"
            )
        if self.selected_id is not None:
            object.__setattr__(
                self, "selected_id", _identifier(self.selected_id, "selected_id")
            )
        if self.deployment_status == "NO_SELECTION" and self.selected_id is not None:
            raise ReportContractError("NO_SELECTION row cannot name a selected policy")
        if self.deployment_status != "NO_SELECTION" and self.selected_id is None:
            raise ReportContractError("selected deployment row must name a policy")
        for name in ("selected_normalized_return", "pool_regret"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.pool_regret < 0.0:
            raise ReportContractError("pool_regret cannot be negative")
        for name in ("epsilon_optimal", "top1_agreement"):
            if type(getattr(self, name)) is not bool:
                raise ReportContractError(f"{name} must be boolean")
        object.__setattr__(
            self,
            "target_transition_count",
            _nonnegative_int(self.target_transition_count, "target_transition_count"),
        )
        updates = _nonnegative_int(
            self.target_gradient_updates, "target_gradient_updates"
        )
        if updates != 0:
            raise ReportContractError("P-table requires zero target gradient updates")
        object.__setattr__(self, "target_gradient_updates", updates)

    @classmethod
    def from_oracle_outcome(
        cls,
        *,
        opaque_query_id: str,
        bank_index: int,
        prefix: int,
        representation_id: str,
        outcome: OracleSelectionOutcome,
        epsilon: float,
        target_transition_count: int,
        target_evidence_digest: str,
        evidence_contract_digest: str,
        cost_digest: str,
    ) -> "DevelopmentPTableRow":
        if not isinstance(outcome, OracleSelectionOutcome):
            raise ReportContractError("outcome must be an OracleSelectionOutcome")
        threshold = _finite(epsilon, "epsilon")
        if threshold < 0.0:
            raise ReportContractError("epsilon cannot be negative")
        return cls(
            opaque_query_id=opaque_query_id,
            bank_index=bank_index,
            prefix=prefix,
            method_id=outcome.method_id,
            representation_id=representation_id,
            selection_record_digest=outcome.selection_record_digest,
            deployment_status=outcome.deployment_status,
            selected_id=outcome.selected_id,
            selected_normalized_return=outcome.selected_value,
            pool_regret=outcome.regret,
            epsilon_optimal=outcome.regret <= threshold,
            top1_agreement=outcome.oracle_top1_agreement,
            target_transition_count=target_transition_count,
            target_gradient_updates=0,
            target_evidence_digest=target_evidence_digest,
            evidence_contract_digest=evidence_contract_digest,
            cost_digest=cost_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class DevelopmentPTable:
    development_split_digest: str
    policy_market_id: str
    evaluation_protocol_id: str
    rows: tuple[DevelopmentPTableRow, ...]
    stage: str = "development_discovery"

    def __post_init__(self) -> None:
        for name in ("development_split_digest", "evaluation_protocol_id"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self, "policy_market_id", _identifier(self.policy_market_id, "policy_market_id")
        )
        if self.stage != "development_discovery":
            raise ReportContractError("P-table schema is development-only")
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, DevelopmentPTableRow) for row in rows):
            raise ReportContractError("P-table requires typed development rows")
        keys = tuple(
            (
                row.opaque_query_id,
                row.bank_index,
                row.prefix,
                row.method_id,
                row.representation_id,
            )
            for row in rows
        )
        if len(keys) != len(set(keys)):
            raise ReportContractError("duplicate P-table method/query/bank/prefix row")
        object.__setattr__(self, "rows", tuple(sorted(rows, key=lambda row: (
            row.opaque_query_id,
            row.bank_index,
            row.prefix,
            row.method_id,
            row.representation_id,
        ))))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "policy-learnware.v02-development-p-table.v0",
            "visibility": "development-analysis-only",
            "stage": self.stage,
            "development_split_digest": self.development_split_digest,
            "policy_market_id": self.policy_market_id,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "rows": [row.to_dict() for row in self.rows],
        }
        assert_nonprivate_table_payload(payload, table="development P-table")
        return payload


def build_development_p_table(
    rows: Sequence[DevelopmentPTableRow],
    *,
    development_split_digest: str,
    policy_market_id: str,
    evaluation_protocol_id: str,
) -> DevelopmentPTable:
    return DevelopmentPTable(
        development_split_digest=development_split_digest,
        policy_market_id=policy_market_id,
        evaluation_protocol_id=evaluation_protocol_id,
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class PrivateOTableRow:
    """One benchmark-private target skyline and diagnostic decomposition."""

    opaque_query_id: str
    true_task_id: str
    true_axis_id: str
    true_factor: float
    regime: str
    physical_nearest_anchor_id: str
    true_distance_lmin_selected_id: str
    true_distance_lmin_value: float
    true_distance_lmin_regret: float
    source_global_champion_id: str
    source_global_champion_value: float
    source_global_champion_regret: float
    executable_ids: tuple[str, ...]
    incompatible_ids: tuple[str, ...]
    full_candidate_value_vector: Mapping[str, float | None]
    best_in_pool_ids: tuple[str, ...]
    best_in_pool_value: float
    pool_viability: float
    failure_floor: float
    selected_method_regret_decomposition: Mapping[str, Any]
    episode_rows_digest: str
    execution_abi_census_digest: str
    oracle_result_digest: str

    def __post_init__(self) -> None:
        for name in (
            "opaque_query_id",
            "true_task_id",
            "true_axis_id",
            "regime",
            "physical_nearest_anchor_id",
            "true_distance_lmin_selected_id",
            "source_global_champion_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.regime not in {
            "safety_exact",
            "heldout_interpolation",
            "heldout_extrapolation",
            "market_ood_boundary",
        }:
            raise ReportContractError(f"unsupported target regime: {self.regime!r}")
        for name in (
            "true_factor",
            "true_distance_lmin_value",
            "true_distance_lmin_regret",
            "source_global_champion_value",
            "source_global_champion_regret",
            "best_in_pool_value",
            "pool_viability",
            "failure_floor",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in (
            "true_distance_lmin_regret",
            "source_global_champion_regret",
        ):
            if getattr(self, name) < 0.0:
                raise ReportContractError(f"{name} cannot be negative")
        if not math.isclose(
            self.pool_viability,
            self.best_in_pool_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ReportContractError("pool viability Q* must equal best-in-pool value")
        executable = tuple(sorted(_identifier(item, "executable_ids[]") for item in self.executable_ids))
        incompatible = tuple(sorted(_identifier(item, "incompatible_ids[]") for item in self.incompatible_ids))
        if not executable or set(executable) & set(incompatible):
            raise ReportContractError("private O-table has an invalid ABI partition")
        values = dict(self.full_candidate_value_vector)
        if set(values) != set(executable) | set(incompatible):
            raise ReportContractError("O-table value vector must cover the full market")
        for opaque_id in executable:
            values[opaque_id] = _finite(values[opaque_id], f"value[{opaque_id!r}]")
        for opaque_id in incompatible:
            if values[opaque_id] is not None:
                raise ReportContractError("incompatible O-table entry must have null value")
        best_ids = tuple(sorted(_identifier(item, "best_in_pool_ids[]") for item in self.best_in_pool_ids))
        if not best_ids or not set(best_ids) <= set(executable):
            raise ReportContractError("best-in-pool ties must be executable policies")
        if self.true_distance_lmin_selected_id not in values:
            raise ReportContractError("true-distance L-min policy is absent from the pool")
        if self.source_global_champion_id not in values:
            raise ReportContractError("source/global champion is absent from the pool")
        decomposition = _deep_freeze(
            self.selected_method_regret_decomposition,
            "selected_method_regret_decomposition",
        )
        if not isinstance(decomposition, Mapping) or not decomposition:
            raise ReportContractError("O-table requires selected method decompositions")
        for name in (
            "episode_rows_digest",
            "execution_abi_census_digest",
            "oracle_result_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "executable_ids", executable)
        object.__setattr__(self, "incompatible_ids", incompatible)
        object.__setattr__(self, "full_candidate_value_vector", MappingProxyType(values))
        object.__setattr__(self, "best_in_pool_ids", best_ids)
        object.__setattr__(self, "selected_method_regret_decomposition", decomposition)

    @classmethod
    def from_oracle_result(
        cls,
        oracle_result: FullPoolOracleResult,
        *,
        true_task_id: str,
        true_axis_id: str,
        true_factor: float,
        regime: str,
        physical_nearest_anchor_id: str,
        true_distance_lmin_selected_id: str,
        source_global_champion_id: str,
    ) -> "PrivateOTableRow":
        if not isinstance(oracle_result, FullPoolOracleResult):
            raise ReportContractError("oracle_result must be a FullPoolOracleResult")
        decompositions = {
            method_id: outcome.to_private_dict()
            for method_id, outcome in sorted(oracle_result.outcomes.items())
        }
        return cls(
            opaque_query_id=oracle_result.opaque_query_id,
            true_task_id=true_task_id,
            true_axis_id=true_axis_id,
            true_factor=true_factor,
            regime=regime,
            physical_nearest_anchor_id=physical_nearest_anchor_id,
            true_distance_lmin_selected_id=true_distance_lmin_selected_id,
            true_distance_lmin_value=oracle_result.deployed_value(
                true_distance_lmin_selected_id
            ),
            true_distance_lmin_regret=oracle_result.regret_for(
                true_distance_lmin_selected_id
            ),
            source_global_champion_id=source_global_champion_id,
            source_global_champion_value=oracle_result.deployed_value(
                source_global_champion_id
            ),
            source_global_champion_regret=oracle_result.regret_for(
                source_global_champion_id
            ),
            executable_ids=oracle_result.executable_ids,
            incompatible_ids=oracle_result.incompatible_ids,
            full_candidate_value_vector=oracle_result.normalized_value_vector,
            best_in_pool_ids=oracle_result.best_in_pool_ids,
            best_in_pool_value=oracle_result.best_in_pool_value,
            pool_viability=oracle_result.pool_viability,
            failure_floor=oracle_result.failure_floor,
            selected_method_regret_decomposition=decompositions,
            episode_rows_digest=oracle_result.episode_rows_digest,
            execution_abi_census_digest=oracle_result.execution_abi_census_digest,
            oracle_result_digest=oracle_result.digest,
        )

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "opaque_query_id": self.opaque_query_id,
            "true_task_id": self.true_task_id,
            "true_axis_id": self.true_axis_id,
            "true_factor": self.true_factor,
            "regime": self.regime,
            "physical_nearest_anchor_id": self.physical_nearest_anchor_id,
            "true_distance_lmin": {
                "selected_id": self.true_distance_lmin_selected_id,
                "selected_value": self.true_distance_lmin_value,
                "regret": self.true_distance_lmin_regret,
            },
            "source_global_champion": {
                "selected_id": self.source_global_champion_id,
                "selected_value": self.source_global_champion_value,
                "regret_g0": self.source_global_champion_regret,
            },
            "executable_ids": list(self.executable_ids),
            "incompatible_ids": list(self.incompatible_ids),
            "full_candidate_value_vector": dict(self.full_candidate_value_vector),
            "best_in_pool_ids": list(self.best_in_pool_ids),
            "best_in_pool_value": self.best_in_pool_value,
            "pool_viability_q_star": self.pool_viability,
            "failure_floor": self.failure_floor,
            "selected_method_regret_decomposition": canonicalize(
                self.selected_method_regret_decomposition
            ),
            "episode_rows_digest": self.episode_rows_digest,
            "execution_abi_census_digest": self.execution_abi_census_digest,
            "oracle_result_digest": self.oracle_result_digest,
        }


@dataclass(frozen=True)
class PrivateOTable:
    policy_market_id: str
    evaluation_protocol_id: str
    rows: tuple[PrivateOTableRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_market_id", _identifier(self.policy_market_id, "policy_market_id")
        )
        object.__setattr__(
            self,
            "evaluation_protocol_id",
            _digest(self.evaluation_protocol_id, "evaluation_protocol_id"),
        )
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, PrivateOTableRow) for row in rows):
            raise ReportContractError("private O-table requires typed rows")
        query_ids = tuple(row.opaque_query_id for row in rows)
        if len(query_ids) != len(set(query_ids)):
            raise ReportContractError("private O-table has duplicate query IDs")
        object.__setattr__(self, "rows", tuple(sorted(rows, key=lambda row: row.opaque_query_id)))

    @property
    def digest(self) -> str:
        return sha256_json(self.to_private_dict())

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-private-o-table.v0",
            "visibility": "private-oracle-analysis-only",
            "policy_market_id": self.policy_market_id,
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "rows": [row.to_private_dict() for row in self.rows],
        }


def build_private_o_table(
    rows: Sequence[PrivateOTableRow],
    *,
    policy_market_id: str,
    evaluation_protocol_id: str,
) -> PrivateOTable:
    return PrivateOTable(
        policy_market_id=policy_market_id,
        evaluation_protocol_id=evaluation_protocol_id,
        rows=tuple(rows),
    )


@dataclass(frozen=True)
class ReferenceETableRow:
    """Complete v0.2 reference evidence for raw/legacy/refit representations."""

    reference_kind: ReferenceKind
    representation_id: str
    representation_version: str
    training_split_digest: str
    canonical_event_view_digest: str
    probe_protocol_id: str
    checkpoint_digest: str | None
    normalizer_digest: str | None
    latent_contract_digest: str
    kernel_digest: str
    reducer_digest: str
    heldout_neighborhood_score: float
    heldout_order_score: float
    repeated_bank_stability: float
    signal_to_noise_ratio: float
    prefix_budgets: tuple[int, ...]
    prefix_selected_returns: tuple[float, ...]
    prefix_regrets: tuple[float, ...]
    sample_efficiency_auc: float
    fixed_lmin_selected_return: float
    fixed_lmin_regret: float
    cold_encoding_seconds: float
    warm_encoding_seconds: float

    def __post_init__(self) -> None:
        if self.reference_kind not in {"raw", "legacy_corro", "refit_corro"}:
            raise ReportContractError(
                "v0.2 reference E-table permits only raw/legacy/refit CORRO rows"
            )
        for name in ("representation_id", "representation_version"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "training_split_digest",
            "canonical_event_view_digest",
            "probe_protocol_id",
            "latent_contract_digest",
            "kernel_digest",
            "reducer_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        checkpoint = self.checkpoint_digest
        normalizer = self.normalizer_digest
        if self.reference_kind == "raw":
            if checkpoint is not None or normalizer is not None:
                raise ReportContractError("raw reference cannot claim checkpoint/normalizer")
        else:
            if checkpoint is None or normalizer is None:
                raise ReportContractError(
                    "CORRO reference requires checkpoint and normalizer digests"
                )
            object.__setattr__(
                self, "checkpoint_digest", _digest(checkpoint, "checkpoint_digest")
            )
            object.__setattr__(
                self, "normalizer_digest", _digest(normalizer, "normalizer_digest")
            )
        for name in (
            "heldout_neighborhood_score",
            "heldout_order_score",
            "repeated_bank_stability",
            "signal_to_noise_ratio",
            "sample_efficiency_auc",
            "fixed_lmin_selected_return",
            "fixed_lmin_regret",
            "cold_encoding_seconds",
            "warm_encoding_seconds",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if min(
            self.signal_to_noise_ratio,
            self.fixed_lmin_regret,
            self.cold_encoding_seconds,
            self.warm_encoding_seconds,
        ) < 0.0:
            raise ReportContractError("SNR, regret, and latencies cannot be negative")
        budgets = tuple(_positive_int(value, "prefix_budgets[]") for value in self.prefix_budgets)
        if not budgets or tuple(sorted(set(budgets))) != budgets:
            raise ReportContractError("prefix budgets must be unique and strictly increasing")
        returns = tuple(
            _finite(value, "prefix_selected_returns[]")
            for value in self.prefix_selected_returns
        )
        regrets = tuple(_finite(value, "prefix_regrets[]") for value in self.prefix_regrets)
        if len(returns) != len(budgets) or len(regrets) != len(budgets):
            raise ReportContractError("prefix curves must align with prefix budgets")
        if any(value < 0.0 for value in regrets):
            raise ReportContractError("prefix regrets cannot be negative")
        object.__setattr__(self, "prefix_budgets", budgets)
        object.__setattr__(self, "prefix_selected_returns", returns)
        object.__setattr__(self, "prefix_regrets", regrets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_kind": self.reference_kind,
            "representation_id": self.representation_id,
            "representation_version": self.representation_version,
            "training_split_digest": self.training_split_digest,
            "canonical_event_view_digest": self.canonical_event_view_digest,
            "probe_protocol_id": self.probe_protocol_id,
            "component_digests": {
                "checkpoint": self.checkpoint_digest,
                "normalizer": self.normalizer_digest,
                "latent_contract": self.latent_contract_digest,
                "kernel": self.kernel_digest,
                "reducer": self.reducer_digest,
            },
            "heldout_metrics": {
                "neighborhood": self.heldout_neighborhood_score,
                "order": self.heldout_order_score,
            },
            "repeated_bank": {
                "stability": self.repeated_bank_stability,
                "signal_to_noise_ratio": self.signal_to_noise_ratio,
            },
            "prefix_curve": [
                {
                    "prefix": prefix,
                    "selected_return": selected_return,
                    "regret": regret,
                }
                for prefix, selected_return, regret in zip(
                    self.prefix_budgets,
                    self.prefix_selected_returns,
                    self.prefix_regrets,
                    strict=True,
                )
            ],
            "sample_efficiency_auc": self.sample_efficiency_auc,
            "fixed_lmin": {
                "selected_return": self.fixed_lmin_selected_return,
                "regret": self.fixed_lmin_regret,
            },
            "encoding_latency_seconds": {
                "cold": self.cold_encoding_seconds,
                "warm": self.warm_encoding_seconds,
            },
        }


@dataclass(frozen=True)
class ReferenceETable:
    evaluation_protocol_id: str
    rows: tuple[ReferenceETableRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_protocol_id",
            _digest(self.evaluation_protocol_id, "evaluation_protocol_id"),
        )
        rows = tuple(self.rows)
        if not rows or any(not isinstance(row, ReferenceETableRow) for row in rows):
            raise ReportContractError("reference E-table requires typed reference rows")
        ids = tuple(row.representation_id for row in rows)
        if len(ids) != len(set(ids)):
            raise ReportContractError("reference E-table has duplicate representation IDs")
        object.__setattr__(
            self,
            "rows",
            tuple(sorted(rows, key=lambda row: (row.reference_kind, row.representation_id))),
        )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "policy-learnware.v02-reference-e-table.v0",
            "visibility": "representation-reference-analysis",
            "evaluation_protocol_id": self.evaluation_protocol_id,
            "included_reference_kinds": sorted(
                {row.reference_kind for row in self.rows}
            ),
            "rows": [row.to_dict() for row in self.rows],
        }
        assert_nonprivate_table_payload(payload, table="reference E-table")
        return payload


def build_reference_e_table(
    rows: Sequence[ReferenceETableRow],
    *,
    evaluation_protocol_id: str,
) -> ReferenceETable:
    return ReferenceETable(
        evaluation_protocol_id=evaluation_protocol_id,
        rows=tuple(rows),
    )


__all__ = [
    "DevelopmentPTable",
    "DevelopmentPTableRow",
    "PrivateOTable",
    "PrivateOTableRow",
    "ReferenceETable",
    "ReferenceETableRow",
    "ReferenceKind",
    "ReportContractError",
    "assert_nonprivate_table_payload",
    "build_development_p_table",
    "build_private_o_table",
    "build_reference_e_table",
]
