"""Oracle-safe policy outcomes and the canonical P8 statistics bridge.

This module is the only v0.3 join between public anonymous rankings and
confirmatory policy outcomes.  It intentionally has no filesystem API and no
oracle reader: callers must supply an external, digest-bound release receipt
and a complete typed evidence manifest.  Formal construction is fail closed
on the exact 66-query by 30-policy oracle rectangle and the exact public
method-by-query ranking barrier.

The resulting bridge exposes policy-selection endpoints, prefix/linkage
inputs, and deterministic :class:`FrozenContrastInputRow` objects.  It does
not run bootstrap statistics and it cannot mint external oracle authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from ..hashing import sha256_json
from .baselines import PublishedFullRanking, REQUIRED_BASELINE_METHOD_IDS
from .preflight import (
    FORMAL_QUERY_REGIME_COUNTS,
    ORACLE_OWNER,
    OracleUnlockHandoff,
    PublicRankingBarrier,
    PublicRankingPublication,
)
from .schemas import checked_digest, checked_safe_id, strict_mapping
from .signal_prefix import FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
from .statistics import (
    FORMAL_CONTRAST_FAMILY_IDS,
    FormalStatisticsPlan,
    FrozenContrastInputRow,
    FrozenStatisticsInput,
    MultiplicityFamilyPlan,
    StatisticsContrast,
    StatisticsEndpoint,
)


ORACLE_EPISODE_EVIDENCE_SCHEMA = "policy-learnware.v03-oracle-episode-evidence.v0"
ORACLE_POLICY_EVIDENCE_SCHEMA = "policy-learnware.v03-oracle-policy-evidence.v0"
ORACLE_EVIDENCE_MANIFEST_SCHEMA = "policy-learnware.v03-oracle-evidence-manifest.v0"
ORACLE_RELEASE_RECEIPT_SCHEMA = "policy-learnware.v03-external-oracle-release.v0"
SIGNAL_OUTCOME_ROW_SCHEMA = "policy-learnware.v03-signal-outcome-row.v0"
SIGNAL_OUTCOME_MANIFEST_SCHEMA = "policy-learnware.v03-signal-outcome-manifest.v0"
PRIMARY_COMPARISON_PLAN_SCHEMA = "policy-learnware.v03-primary-comparison-plan.v0"
POLICY_OUTCOME_SCHEMA = "policy-learnware.v03-policy-outcome.v0"
PREFIX_EFFICIENCY_INPUT_SCHEMA = "policy-learnware.v03-prefix-efficiency-input.v0"
SIGNAL_REGRET_INPUT_SCHEMA = "policy-learnware.v03-signal-regret-input.v0"
POLICY_OUTCOME_BRIDGE_SCHEMA = "policy-learnware.v03-policy-outcome-bridge.v0"

FORMAL_QUERY_COUNT = sum(FORMAL_QUERY_REGIME_COUNTS.values())
FORMAL_MARKET_SIZE = 30
FORMAL_TOP_K = (1, 3, 5)
G03_POLICY_LINK_MIN_TASKS = 2
G03_POLICY_LINK_MIN_AXES = 2
PRIMARY_COMPARISON_IDS = (
    "PC01_M02_VS_B1",
    "PC02_M02_VS_A_ENV",
    "PC03_M02_VS_B2",
    "PC04_M02_VS_STRONGEST_B3_B4",
    "PC05_B3B_VS_A_ENV",
    "PC06_EXACT_RECURRENCE_VS_B2_NI",
    "PC07_SIGNAL_REGRET_LINKAGE",
)
STRONG_BASELINE_CANDIDATES = frozenset({"B3a", "B3b", "B4a", "B4b"})
LINKAGE_N_A_REASONS = (
    "constant-regret-within-axis",
    "constant-signal-within-axis",
    "insufficient-axis-support",
)

_QUERY_ID = re.compile(r"^v03q-[0-9a-f]{32}$")
_POLICY_ID = re.compile(r"^lw-[0-9a-f]{32}$")


class PolicyOutcomeError(ValueError):
    """An oracle release, public join, or canonical contrast is invalid."""


def _strict(value: Mapping[str, Any], fields: set[str], where: str) -> Mapping[str, Any]:
    try:
        return strict_mapping(value, fields, where)
    except ValueError as error:
        raise PolicyOutcomeError(str(error)) from error


def _digest(value: Any, where: str) -> str:
    try:
        return checked_digest(value, where)
    except ValueError as error:
        raise PolicyOutcomeError(str(error)) from error


def _id(value: Any, where: str) -> str:
    try:
        return checked_safe_id(value, where)
    except ValueError as error:
        raise PolicyOutcomeError(str(error)) from error


def _query_id(value: Any, where: str = "opaque_query_id") -> str:
    if not isinstance(value, str) or _QUERY_ID.fullmatch(value) is None:
        raise PolicyOutcomeError(f"{where} must be a canonical v03 query ID")
    return value


def _policy_id(value: Any, where: str = "opaque_policy_id") -> str:
    if not isinstance(value, str) or _POLICY_ID.fullmatch(value) is None:
        raise PolicyOutcomeError(f"{where} must be a canonical anonymous policy ID")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyOutcomeError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PolicyOutcomeError(f"{where} must be finite")
    return result


@dataclass(frozen=True)
class OracleEpisodeEvidence:
    episode_id: str
    episode_seed_digest: str
    status: Literal["EXECUTED", "ABI_INCOMPATIBLE"]
    return_value: float | None
    evidence_digest: str
    schema: str = ORACLE_EPISODE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_EPISODE_EVIDENCE_SCHEMA:
            raise PolicyOutcomeError("unsupported OracleEpisodeEvidence schema")
        object.__setattr__(self, "episode_id", _id(self.episode_id, "episode_id"))
        object.__setattr__(
            self, "episode_seed_digest", _digest(self.episode_seed_digest, "episode_seed_digest")
        )
        object.__setattr__(self, "evidence_digest", _digest(self.evidence_digest, "evidence_digest"))
        if self.status == "EXECUTED":
            object.__setattr__(self, "return_value", _finite(self.return_value, "return_value"))
        elif self.status == "ABI_INCOMPATIBLE":
            if self.return_value is not None:
                raise PolicyOutcomeError("ABI-incompatible episode cannot carry a return")
        else:
            raise PolicyOutcomeError("unknown oracle episode status")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleEpisodeEvidence":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "oracle episode evidence")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class OraclePolicyEvidence:
    opaque_query_id: str
    opaque_policy_id: str
    target_execution_abi_digest: str
    policy_execution_abi_digest: str
    executable: bool
    policy_value: float | None
    episodes: tuple[OracleEpisodeEvidence, ...]
    policy_evidence_digest: str | None = None
    schema: str = ORACLE_POLICY_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_POLICY_EVIDENCE_SCHEMA:
            raise PolicyOutcomeError("unsupported OraclePolicyEvidence schema")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        object.__setattr__(self, "opaque_policy_id", _policy_id(self.opaque_policy_id))
        for name in ("target_execution_abi_digest", "policy_execution_abi_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if type(self.executable) is not bool:
            raise PolicyOutcomeError("executable must be boolean")
        episodes = tuple(self.episodes)
        if not episodes or not all(isinstance(item, OracleEpisodeEvidence) for item in episodes):
            raise PolicyOutcomeError("policy evidence requires typed episode evidence")
        episode_ids = tuple(item.episode_id for item in episodes)
        if len(set(episode_ids)) != len(episode_ids):
            raise PolicyOutcomeError("episode evidence IDs must be unique")
        episodes = tuple(sorted(episodes, key=lambda item: item.episode_id))
        if self.executable:
            if any(item.status != "EXECUTED" for item in episodes):
                raise PolicyOutcomeError("executable policy requires all episodes executed")
            value = _finite(self.policy_value, "policy_value")
            mean = math.fsum(float(item.return_value) for item in episodes) / len(episodes)
            if not math.isclose(value, mean, rel_tol=0.0, abs_tol=1e-12):
                raise PolicyOutcomeError("policy_value must equal the episode-return mean")
            object.__setattr__(self, "policy_value", value)
        else:
            if self.policy_value is not None:
                raise PolicyOutcomeError("non-executable policy cannot carry a policy value")
            if any(item.status != "ABI_INCOMPATIBLE" for item in episodes):
                raise PolicyOutcomeError("non-executable policy requires ABI evidence per episode")
        object.__setattr__(self, "episodes", episodes)
        expected = sha256_json(self._payload_without_digest())
        if self.policy_evidence_digest is None:
            object.__setattr__(self, "policy_evidence_digest", expected)
        elif _digest(self.policy_evidence_digest, "policy_evidence_digest") != expected:
            raise PolicyOutcomeError("policy evidence digest mismatch")

    @property
    def evidence_key(self) -> tuple[str, str]:
        return self.opaque_query_id, self.opaque_policy_id

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "opaque_query_id": self.opaque_query_id,
            "opaque_policy_id": self.opaque_policy_id,
            "target_execution_abi_digest": self.target_execution_abi_digest,
            "policy_execution_abi_digest": self.policy_execution_abi_digest,
            "executable": self.executable,
            "policy_value": self.policy_value,
            "episodes": [item.to_dict() for item in self.episodes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "policy_evidence_digest": self.policy_evidence_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OraclePolicyEvidence":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "oracle policy evidence")
        return cls(
            **{
                field: tuple(OracleEpisodeEvidence.from_dict(item) for item in data[field])
                if field == "episodes"
                else data[field]
                for field in fields
            }
        )


OracleEvidenceScope = Literal["FORMAL", "DEVELOPMENT"]


@dataclass(frozen=True)
class ExternalOracleEvidenceManifest:
    scope: OracleEvidenceScope
    run_id: str
    freeze_manifest_digest: str
    public_ranking_barrier_digest: str
    public_query_plan_digest: str
    query_alias_manifest_digest: str
    signal_outcome_manifest_digest: str
    policy_market_id: str
    expected_opaque_query_ids: tuple[str, ...]
    expected_opaque_policy_ids: tuple[str, ...]
    episode_ids_by_query: Mapping[str, tuple[str, ...]]
    rows: tuple[OraclePolicyEvidence, ...]
    evidence_manifest_digest: str | None = None
    schema: str = ORACLE_EVIDENCE_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_EVIDENCE_MANIFEST_SCHEMA:
            raise PolicyOutcomeError("unsupported ExternalOracleEvidenceManifest schema")
        if self.scope not in {"FORMAL", "DEVELOPMENT"}:
            raise PolicyOutcomeError("unknown oracle evidence scope")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "freeze_manifest_digest",
            "public_ranking_barrier_digest",
            "public_query_plan_digest",
            "query_alias_manifest_digest",
            "signal_outcome_manifest_digest",
            "policy_market_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        queries = tuple(sorted(_query_id(item, "expected query ID") for item in self.expected_opaque_query_ids))
        policies = tuple(sorted(_policy_id(item, "expected policy ID") for item in self.expected_opaque_policy_ids))
        if not queries or len(set(queries)) != len(queries):
            raise PolicyOutcomeError("expected query IDs must be non-empty and unique")
        if not policies or len(set(policies)) != len(policies):
            raise PolicyOutcomeError("expected policy IDs must be non-empty and unique")
        if self.scope == "FORMAL" and (len(queries) != FORMAL_QUERY_COUNT or len(policies) != FORMAL_MARKET_SIZE):
            raise PolicyOutcomeError("formal oracle evidence requires exactly 66 queries x 30 policies")
        schedule: dict[str, tuple[str, ...]] = {}
        for query_id, episode_ids in sorted(self.episode_ids_by_query.items()):
            canonical_query = _query_id(query_id, "episode schedule query ID")
            values = tuple(sorted(_id(item, "episode schedule ID") for item in episode_ids))
            if not values or len(set(values)) != len(values):
                raise PolicyOutcomeError("each query needs unique non-empty episode evidence")
            schedule[canonical_query] = values
        if set(schedule) != set(queries):
            raise PolicyOutcomeError("episode schedule must cover the exact query set")
        rows = tuple(self.rows)
        if not all(isinstance(item, OraclePolicyEvidence) for item in rows):
            raise PolicyOutcomeError("oracle manifest rows must be typed policy evidence")
        expected_pairs = {(query_id, policy_id) for query_id in queries for policy_id in policies}
        observed_pairs = {item.evidence_key for item in rows}
        if len(rows) != len(observed_pairs) or observed_pairs != expected_pairs:
            raise PolicyOutcomeError("oracle evidence must cover the exact query x policy rectangle")
        for row in rows:
            if tuple(item.episode_id for item in row.episodes) != schedule[row.opaque_query_id]:
                raise PolicyOutcomeError("policy episode evidence differs from the frozen query schedule")
        for query_id in queries:
            query_rows = tuple(
                row for row in rows if row.opaque_query_id == query_id
            )
            if len({row.target_execution_abi_digest for row in query_rows}) != 1:
                raise PolicyOutcomeError(
                    "one oracle query cannot carry multiple target execution ABIs"
                )
            if not any(row.executable for row in query_rows):
                raise PolicyOutcomeError("every query requires at least one executable market policy")
        for policy_id in policies:
            if len(
                {
                    row.policy_execution_abi_digest
                    for row in rows
                    if row.opaque_policy_id == policy_id
                }
            ) != 1:
                raise PolicyOutcomeError(
                    "one anonymous policy cannot carry multiple execution ABIs"
                )
        rows = tuple(sorted(rows, key=lambda item: item.evidence_key))
        object.__setattr__(self, "expected_opaque_query_ids", queries)
        object.__setattr__(self, "expected_opaque_policy_ids", policies)
        object.__setattr__(self, "episode_ids_by_query", MappingProxyType(schedule))
        object.__setattr__(self, "rows", rows)
        expected = sha256_json(self._payload_without_digest())
        if self.evidence_manifest_digest is None:
            object.__setattr__(self, "evidence_manifest_digest", expected)
        elif _digest(self.evidence_manifest_digest, "evidence_manifest_digest") != expected:
            raise PolicyOutcomeError("oracle evidence manifest digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "scope": self.scope,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "public_ranking_barrier_digest": self.public_ranking_barrier_digest,
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "signal_outcome_manifest_digest": self.signal_outcome_manifest_digest,
            "policy_market_id": self.policy_market_id,
            "expected_opaque_query_ids": list(self.expected_opaque_query_ids),
            "expected_opaque_policy_ids": list(self.expected_opaque_policy_ids),
            "episode_ids_by_query": {key: list(value) for key, value in self.episode_ids_by_query.items()},
            "rows": [item.to_dict() for item in self.rows],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "evidence_manifest_digest": self.evidence_manifest_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalOracleEvidenceManifest":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "external oracle evidence manifest")
        return cls(
            **{
                field: tuple(OraclePolicyEvidence.from_dict(item) for item in data[field])
                if field == "rows"
                else tuple(data[field])
                if field in {"expected_opaque_query_ids", "expected_opaque_policy_ids"}
                else {key: tuple(items) for key, items in data[field].items()}
                if field == "episode_ids_by_query"
                else data[field]
                for field in fields
            }
        )


@dataclass(frozen=True)
class ExternalOracleReleaseReceipt:
    run_id: str
    freeze_manifest_digest: str
    public_ranking_barrier_digest: str
    oracle_unlock_handoff_digest: str
    oracle_evidence_manifest_digest: str
    external_authority_receipt_digest: str
    oracle_owner: str = ORACLE_OWNER
    v03_oracle_read_capability: bool = False
    v03_oracle_write_capability: bool = False
    release_receipt_digest: str | None = None
    schema: str = ORACLE_RELEASE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ORACLE_RELEASE_RECEIPT_SCHEMA:
            raise PolicyOutcomeError("unsupported ExternalOracleReleaseReceipt schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "freeze_manifest_digest",
            "public_ranking_barrier_digest",
            "oracle_unlock_handoff_digest",
            "oracle_evidence_manifest_digest",
            "external_authority_receipt_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.oracle_owner != ORACLE_OWNER:
            raise PolicyOutcomeError(f"oracle owner must be {ORACLE_OWNER}")
        if self.v03_oracle_read_capability is not False or self.v03_oracle_write_capability is not False:
            raise PolicyOutcomeError("external release receipt cannot grant v0.3 oracle capabilities")
        expected = sha256_json(self._payload_without_digest())
        if self.release_receipt_digest is None:
            object.__setattr__(self, "release_receipt_digest", expected)
        elif _digest(self.release_receipt_digest, "release_receipt_digest") != expected:
            raise PolicyOutcomeError("external oracle release receipt digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "public_ranking_barrier_digest": self.public_ranking_barrier_digest,
            "oracle_unlock_handoff_digest": self.oracle_unlock_handoff_digest,
            "oracle_evidence_manifest_digest": self.oracle_evidence_manifest_digest,
            "external_authority_receipt_digest": self.external_authority_receipt_digest,
            "oracle_owner": self.oracle_owner,
            "v03_oracle_read_capability": self.v03_oracle_read_capability,
            "v03_oracle_write_capability": self.v03_oracle_write_capability,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "release_receipt_digest": self.release_receipt_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExternalOracleReleaseReceipt":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "external oracle release receipt")
        return cls(**{field: data[field] for field in fields})


@dataclass(frozen=True)
class SignalOutcomeRow:
    opaque_query_id: str
    task_id: str
    axis_id: str
    context_id: str
    signal_metric_id: str
    signal_value: float
    prefix_signal_values: Mapping[int, float]
    signal_evidence_digest: str
    schema: str = SIGNAL_OUTCOME_ROW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_OUTCOME_ROW_SCHEMA:
            raise PolicyOutcomeError("unsupported SignalOutcomeRow schema")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        for name in ("task_id", "axis_id", "context_id", "signal_metric_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "signal_value", _finite(self.signal_value, "signal_value"))
        prefixes = {
            int(prefix): _finite(value, f"prefix_signal_values[{prefix}]")
            for prefix, value in sorted(self.prefix_signal_values.items())
        }
        if tuple(prefixes) != FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS:
            raise PolicyOutcomeError("signal outcome row requires exact 1/2/4/8/16/32/64 prefixes")
        if not math.isclose(prefixes[FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS[-1]], self.signal_value, rel_tol=0.0, abs_tol=1e-12):
            raise PolicyOutcomeError("max-prefix signal must equal the registered signal value")
        object.__setattr__(self, "prefix_signal_values", MappingProxyType(prefixes))
        object.__setattr__(self, "signal_evidence_digest", _digest(self.signal_evidence_digest, "signal_evidence_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "opaque_query_id": self.opaque_query_id,
            "task_id": self.task_id,
            "axis_id": self.axis_id,
            "context_id": self.context_id,
            "signal_metric_id": self.signal_metric_id,
            "signal_value": self.signal_value,
            "prefix_signal_values": {str(key): value for key, value in self.prefix_signal_values.items()},
            "signal_evidence_digest": self.signal_evidence_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalOutcomeRow":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "signal outcome row")
        prefixes = data["prefix_signal_values"]
        if not isinstance(prefixes, Mapping):
            raise PolicyOutcomeError("prefix_signal_values must be a mapping")
        return cls(
            **{
                field: (
                    {int(prefix): amount for prefix, amount in prefixes.items()}
                    if field == "prefix_signal_values"
                    else data[field]
                )
                for field in fields
            }
        )


@dataclass(frozen=True)
class SignalOutcomeManifest:
    run_id: str
    freeze_manifest_digest: str
    public_query_plan_digest: str
    query_alias_manifest_digest: str
    signal_atlas_digest: str
    signal_prefix_schedule_digest: str
    rows: tuple[SignalOutcomeRow, ...]
    manifest_digest: str | None = None
    schema: str = SIGNAL_OUTCOME_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_OUTCOME_MANIFEST_SCHEMA:
            raise PolicyOutcomeError("unsupported SignalOutcomeManifest schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "freeze_manifest_digest",
            "public_query_plan_digest",
            "query_alias_manifest_digest",
            "signal_atlas_digest",
            "signal_prefix_schedule_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        rows = tuple(self.rows)
        if len(rows) != FORMAL_QUERY_COUNT or not all(isinstance(item, SignalOutcomeRow) for item in rows):
            raise PolicyOutcomeError("formal signal outcome manifest requires exactly 66 typed query rows")
        query_ids = tuple(item.opaque_query_id for item in rows)
        if len(set(query_ids)) != len(query_ids):
            raise PolicyOutcomeError("signal outcome query IDs must be unique")
        task_ids = {item.task_id for item in rows}
        axis_ids = {item.axis_id for item in rows}
        if (
            len(task_ids) < G03_POLICY_LINK_MIN_TASKS
            or len(axis_ids) < G03_POLICY_LINK_MIN_AXES
        ):
            raise PolicyOutcomeError(
                "G03-PolicyLink requires signal/outcome coverage for at least "
                "two tasks and two axes"
            )
        coverage_pairs = {(item.task_id, item.axis_id) for item in rows}
        if not any(
            all((task_id, axis_id) in coverage_pairs for task_id in task_pair for axis_id in axis_pair)
            for task_pair in _pairs(sorted(task_ids))
            for axis_pair in _pairs(sorted(axis_ids))
        ):
            raise PolicyOutcomeError(
                "G03-PolicyLink requires a complete 2-task x 2-axis query panel"
            )
        rows = tuple(sorted(rows, key=lambda item: item.opaque_query_id))
        object.__setattr__(self, "rows", rows)
        expected = sha256_json(self._payload_without_digest())
        if self.manifest_digest is None:
            object.__setattr__(self, "manifest_digest", expected)
        elif _digest(self.manifest_digest, "signal outcome manifest_digest") != expected:
            raise PolicyOutcomeError("signal outcome manifest digest mismatch")

    @property
    def opaque_query_ids(self) -> tuple[str, ...]:
        return tuple(item.opaque_query_id for item in self.rows)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "signal_atlas_digest": self.signal_atlas_digest,
            "signal_prefix_schedule_digest": self.signal_prefix_schedule_digest,
            "rows": [item.to_dict() for item in self.rows],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SignalOutcomeManifest":
        fields = set(cls.__dataclass_fields__)
        data = _strict(value, fields, "signal outcome manifest")
        return cls(
            **{
                field: (
                    tuple(SignalOutcomeRow.from_dict(item) for item in data[field])
                    if field == "rows"
                    else data[field]
                )
                for field in fields
            }
        )


@dataclass(frozen=True)
class CanonicalPrimaryComparisonPlan:
    strongest_b3_b4_method_id: str
    strongest_method_selection_receipt_digest: str
    exact_recurrence_noninferiority_margin: float = 0.05
    bootstrap_resamples: int = 10_000
    schema: str = PRIMARY_COMPARISON_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRIMARY_COMPARISON_PLAN_SCHEMA:
            raise PolicyOutcomeError("unsupported CanonicalPrimaryComparisonPlan schema")
        if self.strongest_b3_b4_method_id not in STRONG_BASELINE_CANDIDATES:
            raise PolicyOutcomeError("strongest B3-B4 method must be development-frozen")
        object.__setattr__(
            self,
            "strongest_method_selection_receipt_digest",
            _digest(self.strongest_method_selection_receipt_digest, "strongest method selection receipt"),
        )
        margin = _finite(self.exact_recurrence_noninferiority_margin, "exact recurrence NI margin")
        if not 0.0 < margin < 1.0:
            raise PolicyOutcomeError("exact recurrence NI margin must lie in (0, 1)")
        object.__setattr__(self, "exact_recurrence_noninferiority_margin", margin)
        if type(self.bootstrap_resamples) is not int or self.bootstrap_resamples < 2:
            raise PolicyOutcomeError("bootstrap_resamples must be an integer >= 2")

    @property
    def comparison_definitions(self) -> tuple[tuple[str, str, str, str, str], ...]:
        return (
            (PRIMARY_COMPARISON_IDS[0], "SCHEMA", "normalized_pool_regret", "M02/B5", "B1"),
            (PRIMARY_COMPARISON_IDS[1], "REPRESENTATION_LADDER", "normalized_pool_regret", "M02/B5", "A-Env"),
            (PRIMARY_COMPARISON_IDS[2], "SCHEMA", "normalized_pool_regret", "M02/B5", "B2"),
            (PRIMARY_COMPARISON_IDS[3], "REWARD_GOAL", "normalized_pool_regret", "M02/B5", self.strongest_b3_b4_method_id),
            (PRIMARY_COMPARISON_IDS[4], "TEMPORAL_HISTORY", "normalized_pool_regret", "B3b", "A-Env"),
            (PRIMARY_COMPARISON_IDS[5], "TRANSITION_MECHANISM", "oracle_top1_coverage", "M02/B5", "B2"),
            (PRIMARY_COMPARISON_IDS[6], "POLICY_SELECTION_LINKAGE", "signal_regret_association", "registered-signal", "zero-association"),
        )

    @property
    def formal_statistics_plan(self) -> FormalStatisticsPlan:
        endpoints = (
            StatisticsEndpoint("normalized_pool_regret", "normalized_pool_regret", False),
            StatisticsEndpoint("oracle_top1_coverage", "oracle_top1_coverage", True),
            StatisticsEndpoint("signal_regret_association", "spearman_signal_negative_regret", True),
        )
        contrasts = tuple(
            StatisticsContrast(
                hypothesis_id=comparison_id,
                contrast_family_id=family_id,
                endpoint_id=endpoint_id,
                left_method_id=left_method,
                right_method_id=right_method,
                null_boundary=(
                    -self.exact_recurrence_noninferiority_margin
                    if comparison_id == PRIMARY_COMPARISON_IDS[5]
                    else 0.0
                ),
            )
            for comparison_id, family_id, endpoint_id, left_method, right_method in self.comparison_definitions
        )
        by_family = {
            family_id: tuple(
                comparison_id
                for comparison_id, candidate_family, *_ in self.comparison_definitions
                if candidate_family == family_id
            )
            for family_id in FORMAL_CONTRAST_FAMILY_IDS
        }
        if any(not hypotheses for hypotheses in by_family.values()):
            raise PolicyOutcomeError("canonical plan must cover all six formal contrast families")
        return FormalStatisticsPlan(
            endpoints=endpoints,
            contrasts=contrasts,
            multiplicity_families=tuple(
                MultiplicityFamilyPlan(family_id, hypotheses)
                for family_id, hypotheses in by_family.items()
            ),
            bootstrap_resamples=self.bootstrap_resamples,
            seed_namespace="v03-canonical-primary-comparisons",
            formal_confirmatory=True,
            registered_n_a_reasons=self.registered_n_a_reasons,
        )

    @property
    def registered_n_a_reasons(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                comparison_id: (LINKAGE_N_A_REASONS if comparison_id == PRIMARY_COMPARISON_IDS[6] else ())
                for comparison_id in PRIMARY_COMPARISON_IDS
            }
        )

    @property
    def plan_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "strongest_b3_b4_method_id": self.strongest_b3_b4_method_id,
            "strongest_method_selection_receipt_digest": self.strongest_method_selection_receipt_digest,
            "exact_recurrence_noninferiority_margin": self.exact_recurrence_noninferiority_margin,
            "bootstrap_resamples": self.bootstrap_resamples,
            "primary_comparison_ids": list(PRIMARY_COMPARISON_IDS),
            "comparison_definitions": [list(item) for item in self.comparison_definitions],
            "registered_n_a_reasons": {key: list(value) for key, value in self.registered_n_a_reasons.items()},
            "formal_statistics_plan": self.formal_statistics_plan.to_dict(),
            "formal_statistics_plan_digest": self.formal_statistics_plan.plan_digest,
        }


@dataclass(frozen=True)
class PolicyOutcomeRecord:
    method_id: str
    opaque_query_id: str
    query_regime: str
    ranking_digest: str
    selected_opaque_policy_id: str
    selected_return: float | None
    best_pool_return: float
    worst_pool_return: float
    normalized_pool_regret: float
    top_k_oracle_coverage: Mapping[int, float]
    abi_incompatible: bool
    oracle_policy_evidence_digest: str
    record_digest: str | None = None
    schema: str = POLICY_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POLICY_OUTCOME_SCHEMA:
            raise PolicyOutcomeError("unsupported PolicyOutcomeRecord schema")
        if self.method_id not in REQUIRED_BASELINE_METHOD_IDS:
            raise PolicyOutcomeError("unknown policy outcome method")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        if self.query_regime not in FORMAL_QUERY_REGIME_COUNTS:
            raise PolicyOutcomeError("unknown policy outcome query regime")
        object.__setattr__(self, "ranking_digest", _digest(self.ranking_digest, "ranking_digest"))
        object.__setattr__(self, "selected_opaque_policy_id", _policy_id(self.selected_opaque_policy_id))
        if self.selected_return is not None:
            object.__setattr__(self, "selected_return", _finite(self.selected_return, "selected_return"))
        best = _finite(self.best_pool_return, "best_pool_return")
        worst = _finite(self.worst_pool_return, "worst_pool_return")
        if best < worst:
            raise PolicyOutcomeError("best pool return cannot be below worst pool return")
        object.__setattr__(self, "best_pool_return", best)
        object.__setattr__(self, "worst_pool_return", worst)
        regret = _finite(self.normalized_pool_regret, "normalized_pool_regret")
        if regret < 0.0 or regret > 1.0:
            raise PolicyOutcomeError("normalized pool regret must lie in [0, 1]")
        object.__setattr__(self, "normalized_pool_regret", regret)
        coverage = {int(key): _finite(value, f"top_k_oracle_coverage[{key}]") for key, value in sorted(self.top_k_oracle_coverage.items())}
        if tuple(coverage) != FORMAL_TOP_K or any(value not in {0.0, 1.0} for value in coverage.values()):
            raise PolicyOutcomeError("top-k coverage must be binary at k=1,3,5")
        object.__setattr__(self, "top_k_oracle_coverage", MappingProxyType(coverage))
        if type(self.abi_incompatible) is not bool:
            raise PolicyOutcomeError("abi_incompatible must be boolean")
        if self.abi_incompatible != (self.selected_return is None):
            raise PolicyOutcomeError("ABI incompatibility must match absent selected return")
        if self.abi_incompatible and regret != 1.0:
            raise PolicyOutcomeError("ABI-incompatible rank one receives unit normalized regret")
        if not self.abi_incompatible:
            selected = float(self.selected_return)
            if not worst <= selected <= best:
                raise PolicyOutcomeError(
                    "selected return must lie inside the executable pool skyline"
                )
            expected_regret = (
                0.0 if best == worst else (best - selected) / (best - worst)
            )
            if not math.isclose(
                regret, expected_regret, rel_tol=0.0, abs_tol=1e-12
            ):
                raise PolicyOutcomeError(
                    "normalized pool regret disagrees with selected/best/worst returns"
                )
        if any(
            coverage[lower] > coverage[upper]
            for lower, upper in zip(FORMAL_TOP_K, FORMAL_TOP_K[1:])
        ):
            raise PolicyOutcomeError("top-k oracle coverage must be monotone in k")
        object.__setattr__(self, "oracle_policy_evidence_digest", _digest(self.oracle_policy_evidence_digest, "oracle policy evidence digest"))
        expected = sha256_json(self._payload_without_digest())
        if self.record_digest is None:
            object.__setattr__(self, "record_digest", expected)
        elif _digest(self.record_digest, "policy outcome record_digest") != expected:
            raise PolicyOutcomeError("policy outcome record digest mismatch")

    @property
    def outcome_key(self) -> tuple[str, str]:
        return self.method_id, self.opaque_query_id

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "opaque_query_id": self.opaque_query_id,
            "query_regime": self.query_regime,
            "ranking_digest": self.ranking_digest,
            "selected_opaque_policy_id": self.selected_opaque_policy_id,
            "selected_return": self.selected_return,
            "best_pool_return": self.best_pool_return,
            "worst_pool_return": self.worst_pool_return,
            "normalized_pool_regret": self.normalized_pool_regret,
            "top_k_oracle_coverage": {str(key): value for key, value in self.top_k_oracle_coverage.items()},
            "abi_incompatible": self.abi_incompatible,
            "oracle_policy_evidence_digest": self.oracle_policy_evidence_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "record_digest": self.record_digest}


@dataclass(frozen=True)
class PrefixSampleEfficiencyInput:
    method_id: str
    opaque_query_id: str
    prefix_episode_count: int
    signal_metric_id: str
    prefix_signal_value: float
    normalized_pool_regret: float
    signal_evidence_digest: str
    policy_outcome_digest: str
    schema: str = PREFIX_EFFICIENCY_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREFIX_EFFICIENCY_INPUT_SCHEMA:
            raise PolicyOutcomeError("unsupported PrefixSampleEfficiencyInput schema")
        if self.method_id not in REQUIRED_BASELINE_METHOD_IDS:
            raise PolicyOutcomeError("unknown prefix efficiency method")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        if self.prefix_episode_count not in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS:
            raise PolicyOutcomeError("prefix sample-efficiency input uses an unregistered prefix")
        object.__setattr__(self, "signal_metric_id", _id(self.signal_metric_id, "signal_metric_id"))
        object.__setattr__(self, "prefix_signal_value", _finite(self.prefix_signal_value, "prefix_signal_value"))
        regret = _finite(self.normalized_pool_regret, "normalized_pool_regret")
        if not 0.0 <= regret <= 1.0:
            raise PolicyOutcomeError("prefix input regret must lie in [0, 1]")
        object.__setattr__(self, "normalized_pool_regret", regret)
        object.__setattr__(self, "signal_evidence_digest", _digest(self.signal_evidence_digest, "signal_evidence_digest"))
        object.__setattr__(self, "policy_outcome_digest", _digest(self.policy_outcome_digest, "policy_outcome_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class SignalRegretLinkageInput:
    method_id: str
    opaque_query_id: str
    task_id: str
    axis_id: str
    context_id: str
    signal_metric_id: str
    signal_value: float
    normalized_pool_regret: float
    signal_evidence_digest: str
    policy_outcome_digest: str
    schema: str = SIGNAL_REGRET_INPUT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_REGRET_INPUT_SCHEMA:
            raise PolicyOutcomeError("unsupported SignalRegretLinkageInput schema")
        if self.method_id not in REQUIRED_BASELINE_METHOD_IDS:
            raise PolicyOutcomeError("unknown signal-regret method")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        for name in ("task_id", "axis_id", "context_id", "signal_metric_id"):
            object.__setattr__(self, name, _id(getattr(self, name), name))
        object.__setattr__(self, "signal_value", _finite(self.signal_value, "signal_value"))
        regret = _finite(self.normalized_pool_regret, "normalized_pool_regret")
        if not 0.0 <= regret <= 1.0:
            raise PolicyOutcomeError("linkage regret must lie in [0, 1]")
        object.__setattr__(self, "normalized_pool_regret", regret)
        object.__setattr__(self, "signal_evidence_digest", _digest(self.signal_evidence_digest, "signal_evidence_digest"))
        object.__setattr__(self, "policy_outcome_digest", _digest(self.policy_outcome_digest, "policy_outcome_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PolicyOutcomeStatisticsBridge:
    run_id: str
    freeze_manifest_digest: str
    public_ranking_barrier_digest: str
    oracle_unlock_handoff_digest: str
    oracle_release_receipt_digest: str
    oracle_evidence_manifest_digest: str
    signal_outcome_manifest_digest: str
    primary_comparison_plan_digest: str
    ranking_join_digest: str
    outcomes: tuple[PolicyOutcomeRecord, ...]
    prefix_inputs: tuple[PrefixSampleEfficiencyInput, ...]
    linkage_inputs: tuple[SignalRegretLinkageInput, ...]
    frozen_statistics_input: FrozenStatisticsInput
    bridge_digest: str | None = None
    schema: str = POLICY_OUTCOME_BRIDGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != POLICY_OUTCOME_BRIDGE_SCHEMA:
            raise PolicyOutcomeError("unsupported PolicyOutcomeStatisticsBridge schema")
        object.__setattr__(self, "run_id", _id(self.run_id, "run_id"))
        for name in (
            "freeze_manifest_digest",
            "public_ranking_barrier_digest",
            "oracle_unlock_handoff_digest",
            "oracle_release_receipt_digest",
            "oracle_evidence_manifest_digest",
            "signal_outcome_manifest_digest",
            "primary_comparison_plan_digest",
            "ranking_join_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        outcomes = tuple(self.outcomes)
        if len(outcomes) != len(REQUIRED_BASELINE_METHOD_IDS) * FORMAL_QUERY_COUNT:
            raise PolicyOutcomeError("formal bridge requires the exact method x query outcomes")
        if len({item.outcome_key for item in outcomes}) != len(outcomes):
            raise PolicyOutcomeError("policy outcome keys must be unique")
        prefixes = tuple(self.prefix_inputs)
        expected_prefix_count = FORMAL_QUERY_COUNT * len(FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS)
        if len(prefixes) != expected_prefix_count:
            raise PolicyOutcomeError("formal bridge requires every M02 query x signal prefix")
        linkage = tuple(self.linkage_inputs)
        if len(linkage) != FORMAL_QUERY_COUNT:
            raise PolicyOutcomeError("formal bridge requires every M02 signal-regret query")
        if not isinstance(self.frozen_statistics_input, FrozenStatisticsInput):
            raise PolicyOutcomeError("bridge requires a typed FrozenStatisticsInput")
        frozen = self.frozen_statistics_input
        if (
            frozen.run_id != self.run_id
            or frozen.preexperiment_freeze_manifest_digest != self.freeze_manifest_digest
            or frozen.public_ranking_barrier_digest != self.public_ranking_barrier_digest
            or frozen.oracle_unlock_handoff_digest != self.oracle_unlock_handoff_digest
            or frozen.oracle_release_receipt_digest != self.oracle_release_receipt_digest
            or frozen.oracle_evidence_manifest_digest != self.oracle_evidence_manifest_digest
        ):
            raise PolicyOutcomeError("frozen statistics input differs from bridge provenance")
        object.__setattr__(self, "outcomes", tuple(sorted(outcomes, key=lambda item: item.outcome_key)))
        object.__setattr__(self, "prefix_inputs", tuple(sorted(prefixes, key=lambda item: (item.opaque_query_id, item.prefix_episode_count))))
        object.__setattr__(self, "linkage_inputs", tuple(sorted(linkage, key=lambda item: item.opaque_query_id)))
        expected = sha256_json(self._payload_without_digest())
        if self.bridge_digest is None:
            object.__setattr__(self, "bridge_digest", expected)
        elif _digest(self.bridge_digest, "bridge_digest") != expected:
            raise PolicyOutcomeError("policy outcome bridge digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "public_ranking_barrier_digest": self.public_ranking_barrier_digest,
            "oracle_unlock_handoff_digest": self.oracle_unlock_handoff_digest,
            "oracle_release_receipt_digest": self.oracle_release_receipt_digest,
            "oracle_evidence_manifest_digest": self.oracle_evidence_manifest_digest,
            "signal_outcome_manifest_digest": self.signal_outcome_manifest_digest,
            "primary_comparison_plan_digest": self.primary_comparison_plan_digest,
            "ranking_join_digest": self.ranking_join_digest,
            "outcome_digests": [item.record_digest for item in self.outcomes],
            "prefix_inputs_digest": sha256_json([item.to_dict() for item in self.prefix_inputs]),
            "linkage_inputs_digest": sha256_json([item.to_dict() for item in self.linkage_inputs]),
            "frozen_statistics_input_digest": self.frozen_statistics_input.input_manifest_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "outcomes": [item.to_dict() for item in self.outcomes],
            "prefix_inputs": [item.to_dict() for item in self.prefix_inputs],
            "linkage_inputs": [item.to_dict() for item in self.linkage_inputs],
            "frozen_statistics_input": self.frozen_statistics_input.to_dict(),
            "bridge_digest": self.bridge_digest,
        }


def _rankdata(values: Sequence[float]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for offset in range(start, end):
            result[order[offset]] = average_rank
        start = end
    return tuple(result)


def _pairs(values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )


def _spearman(left: Sequence[float], right: Sequence[float]) -> tuple[float | None, str | None]:
    if len(left) < 3:
        return None, "insufficient-axis-support"
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = math.fsum(left_ranks) / len(left_ranks)
    right_mean = math.fsum(right_ranks) / len(right_ranks)
    left_ss = math.fsum((value - left_mean) ** 2 for value in left_ranks)
    right_ss = math.fsum((value - right_mean) ** 2 for value in right_ranks)
    if left_ss == 0.0:
        return None, "constant-signal-within-axis"
    if right_ss == 0.0:
        return None, "constant-regret-within-axis"
    covariance = math.fsum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks)
    )
    return covariance / math.sqrt(left_ss * right_ss), None


def _validate_external_join(
    *,
    barrier: PublicRankingBarrier,
    rankings: Sequence[PublishedFullRanking],
    handoff: OracleUnlockHandoff,
    release: ExternalOracleReleaseReceipt,
    oracle: ExternalOracleEvidenceManifest,
    signal: SignalOutcomeManifest,
) -> tuple[tuple[PublishedFullRanking, ...], str]:
    if not isinstance(barrier, PublicRankingBarrier) or not barrier.freeze_manifest.formal_run_authorized:
        raise PolicyOutcomeError("formal outcome join requires an authorized PublicRankingBarrier")
    if not isinstance(handoff, OracleUnlockHandoff):
        raise PolicyOutcomeError("formal outcome join requires a typed oracle handoff")
    if not isinstance(release, ExternalOracleReleaseReceipt):
        raise PolicyOutcomeError("formal outcome join only consumes an external release receipt")
    if not isinstance(oracle, ExternalOracleEvidenceManifest) or oracle.scope != "FORMAL":
        raise PolicyOutcomeError("formal outcome join requires formal typed oracle evidence")
    if not isinstance(signal, SignalOutcomeManifest):
        raise PolicyOutcomeError("formal outcome join requires typed pre-oracle signal evidence")
    expected_identity = (barrier.run_id, barrier.freeze_manifest_digest, barrier.barrier_digest)
    if (handoff.run_id, handoff.freeze_manifest_digest, handoff.public_ranking_barrier_digest) != expected_identity:
        raise PolicyOutcomeError("oracle handoff differs from the public ranking barrier")
    if (
        release.run_id != barrier.run_id
        or release.freeze_manifest_digest != barrier.freeze_manifest_digest
        or release.public_ranking_barrier_digest != barrier.barrier_digest
        or release.oracle_unlock_handoff_digest != handoff.handoff_digest
        or release.oracle_evidence_manifest_digest != oracle.evidence_manifest_digest
    ):
        raise PolicyOutcomeError("external oracle release provenance differs from barrier/evidence")
    if (
        oracle.run_id != barrier.run_id
        or oracle.freeze_manifest_digest != barrier.freeze_manifest_digest
        or oracle.public_ranking_barrier_digest != barrier.barrier_digest
        or oracle.public_query_plan_digest != barrier.query_plan.plan_digest
        or oracle.query_alias_manifest_digest != barrier.query_alias_manifest_digest
        or oracle.signal_outcome_manifest_digest != signal.manifest_digest
    ):
        raise PolicyOutcomeError("oracle evidence provenance differs from public/signal freeze")
    if (
        signal.run_id != barrier.run_id
        or signal.freeze_manifest_digest != barrier.freeze_manifest_digest
        or signal.manifest_digest
        != barrier.preoracle_signal_outcome_manifest_digest
        or signal.public_query_plan_digest != barrier.query_plan.plan_digest
        or signal.query_alias_manifest_digest != barrier.query_alias_manifest_digest
        or signal.signal_prefix_schedule_digest != barrier.freeze_manifest.formal_signal_prefix_schedule_digest
        or set(signal.opaque_query_ids) != set(barrier.expected_opaque_query_ids)
    ):
        raise PolicyOutcomeError("signal outcome evidence differs from the frozen public query plan")
    if tuple(oracle.expected_opaque_query_ids) != tuple(barrier.expected_opaque_query_ids):
        raise PolicyOutcomeError("oracle query coverage differs from public ranking barrier")
    rows = tuple(rankings)
    if not all(isinstance(item, PublishedFullRanking) for item in rows):
        raise PolicyOutcomeError("ranking join requires typed PublishedFullRanking objects")
    expected_pairs = {
        (method_id, query_id)
        for method_id in barrier.expected_method_ids
        for query_id in barrier.expected_opaque_query_ids
    }
    by_key = {(item.method_id, item.opaque_query_id): item for item in rows}
    if len(rows) != len(by_key) or set(by_key) != expected_pairs:
        raise PolicyOutcomeError("full rankings must cover the exact barrier method x query matrix")
    publication_by_key = {item.publication_key: item for item in barrier.publications}
    for key, ranking in by_key.items():
        publication = PublicRankingPublication.from_published_ranking(ranking)
        if publication.to_dict() != publication_by_key[key].to_dict():
            raise PolicyOutcomeError("full ranking bytes differ from their barrier publication")
        policy_ids = tuple(sorted(item.opaque_learnware_id for item in ranking.rows))
        if policy_ids != oracle.expected_opaque_policy_ids:
            raise PolicyOutcomeError("full ranking does not cover the external oracle market")
    market_ids = {item.policy_market_id for item in rows}
    if market_ids != {oracle.policy_market_id}:
        raise PolicyOutcomeError("ranking and oracle policy-market identities differ")
    method_order = {method: index for index, method in enumerate(barrier.expected_method_ids)}
    rows = tuple(sorted(rows, key=lambda item: (method_order[item.method_id], item.opaque_query_id)))
    join_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-ranking-oracle-join.v0",
            "public_ranking_barrier_digest": barrier.barrier_digest,
            "ranking_digests": [item.ranking_digest for item in rows],
            "oracle_release_receipt_digest": release.release_receipt_digest,
            "oracle_evidence_manifest_digest": oracle.evidence_manifest_digest,
            "signal_outcome_manifest_digest": signal.manifest_digest,
        }
    )
    return rows, join_digest


def _build_outcomes(
    *,
    barrier: PublicRankingBarrier,
    rankings: Sequence[PublishedFullRanking],
    oracle: ExternalOracleEvidenceManifest,
) -> tuple[PolicyOutcomeRecord, ...]:
    oracle_by_key = {item.evidence_key: item for item in oracle.rows}
    records: list[PolicyOutcomeRecord] = []
    for ranking in rankings:
        query_rows = {
            policy_id: oracle_by_key[(ranking.opaque_query_id, policy_id)]
            for policy_id in oracle.expected_opaque_policy_ids
        }
        executable_values = {
            policy_id: float(row.policy_value)
            for policy_id, row in query_rows.items()
            if row.executable
        }
        best = max(executable_values.values())
        worst = min(executable_values.values())
        best_ids = {
            policy_id for policy_id, value in executable_values.items() if value == best
        }
        selected = query_rows[ranking.selected_opaque_learnware_id]
        if selected.executable:
            selected_return = float(selected.policy_value)
            regret = 0.0 if best == worst else (best - selected_return) / (best - worst)
        else:
            selected_return = None
            regret = 1.0
        ranked_ids = tuple(item.opaque_learnware_id for item in ranking.rows)
        top_k = {
            k: float(any(policy_id in best_ids for policy_id in ranked_ids[:k]))
            for k in FORMAL_TOP_K
        }
        records.append(
            PolicyOutcomeRecord(
                method_id=ranking.method_id,
                opaque_query_id=ranking.opaque_query_id,
                query_regime=barrier.query_plan.regime_by_opaque_query_id[ranking.opaque_query_id],
                ranking_digest=str(ranking.ranking_digest),
                selected_opaque_policy_id=ranking.selected_opaque_learnware_id,
                selected_return=selected_return,
                best_pool_return=best,
                worst_pool_return=worst,
                normalized_pool_regret=regret,
                top_k_oracle_coverage=top_k,
                abi_incompatible=not selected.executable,
                oracle_policy_evidence_digest=str(selected.policy_evidence_digest),
            )
        )
    return tuple(records)


def _build_auxiliary_inputs(
    outcomes: Sequence[PolicyOutcomeRecord], signal: SignalOutcomeManifest
) -> tuple[tuple[PrefixSampleEfficiencyInput, ...], tuple[SignalRegretLinkageInput, ...]]:
    outcome_by_key = {item.outcome_key: item for item in outcomes}
    signal_by_query = {item.opaque_query_id: item for item in signal.rows}
    prefixes: list[PrefixSampleEfficiencyInput] = []
    linkage: list[SignalRegretLinkageInput] = []
    for query_id, signal_row in sorted(signal_by_query.items()):
        outcome = outcome_by_key[("M02/B5", query_id)]
        linkage.append(
            SignalRegretLinkageInput(
                method_id="M02/B5",
                opaque_query_id=query_id,
                task_id=signal_row.task_id,
                axis_id=signal_row.axis_id,
                context_id=signal_row.context_id,
                signal_metric_id=signal_row.signal_metric_id,
                signal_value=signal_row.signal_value,
                normalized_pool_regret=outcome.normalized_pool_regret,
                signal_evidence_digest=signal_row.signal_evidence_digest,
                policy_outcome_digest=str(outcome.record_digest),
            )
        )
        for prefix, value in signal_row.prefix_signal_values.items():
            prefixes.append(
                PrefixSampleEfficiencyInput(
                    method_id="M02/B5",
                    opaque_query_id=query_id,
                    prefix_episode_count=prefix,
                    signal_metric_id=signal_row.signal_metric_id,
                    prefix_signal_value=value,
                    normalized_pool_regret=outcome.normalized_pool_regret,
                    signal_evidence_digest=signal_row.signal_evidence_digest,
                    policy_outcome_digest=str(outcome.record_digest),
                )
            )
    return tuple(prefixes), tuple(linkage)


def _build_contrast_rows(
    *,
    barrier: PublicRankingBarrier,
    outcomes: Sequence[PolicyOutcomeRecord],
    signal: SignalOutcomeManifest,
    plan: CanonicalPrimaryComparisonPlan,
) -> tuple[FrozenContrastInputRow, ...]:
    outcome_by_key = {item.outcome_key: item for item in outcomes}
    signal_by_query = {item.opaque_query_id: item for item in signal.rows}
    rows: list[FrozenContrastInputRow] = []
    comparison_methods = {
        item[0]: (item[3], item[4]) for item in plan.comparison_definitions[:5]
    }
    for comparison_id, (left_method, right_method) in comparison_methods.items():
        for query_id in barrier.expected_opaque_query_ids:
            identity = signal_by_query[query_id]
            rows.append(
                FrozenContrastInputRow(
                    hypothesis_id=comparison_id,
                    task_id=identity.task_id,
                    axis_id=identity.axis_id,
                    context_id=identity.context_id,
                    observation_id=query_id,
                    status="OBSERVED",
                    left_value=outcome_by_key[(left_method, query_id)].normalized_pool_regret,
                    right_value=outcome_by_key[(right_method, query_id)].normalized_pool_regret,
                )
            )
    recurrence_id = PRIMARY_COMPARISON_IDS[5]
    for query_id, regime in barrier.query_plan.regime_by_opaque_query_id.items():
        if regime != "EXACT":
            continue
        identity = signal_by_query[query_id]
        rows.append(
            FrozenContrastInputRow(
                hypothesis_id=recurrence_id,
                task_id=identity.task_id,
                axis_id=identity.axis_id,
                context_id=identity.context_id,
                observation_id=query_id,
                status="OBSERVED",
                left_value=outcome_by_key[("M02/B5", query_id)].top_k_oracle_coverage[1],
                right_value=outcome_by_key[("B2", query_id)].top_k_oracle_coverage[1],
            )
        )
    linkage_id = PRIMARY_COMPARISON_IDS[6]
    by_axis: dict[tuple[str, str], list[SignalRegretLinkageInput]] = {}
    _, linkage = _build_auxiliary_inputs(outcomes, signal)
    for item in linkage:
        by_axis.setdefault((item.task_id, item.axis_id), []).append(item)
    for (task_id, axis_id), group in sorted(by_axis.items()):
        correlation, reason = _spearman(
            [item.signal_value for item in group],
            [-item.normalized_pool_regret for item in group],
        )
        if correlation is None:
            rows.append(
                FrozenContrastInputRow(
                    hypothesis_id=linkage_id,
                    task_id=task_id,
                    axis_id=axis_id,
                    context_id=f"axis-summary-{task_id}-{axis_id}",
                    observation_id="signal-regret-association",
                    status="N_A",
                    left_value=None,
                    right_value=None,
                    n_a_reason=reason,
                )
            )
        else:
            rows.append(
                FrozenContrastInputRow(
                    hypothesis_id=linkage_id,
                    task_id=task_id,
                    axis_id=axis_id,
                    context_id=f"axis-summary-{task_id}-{axis_id}",
                    observation_id="signal-regret-association",
                    status="OBSERVED",
                    left_value=correlation,
                    right_value=0.0,
                )
            )
    observed_by_hypothesis = {
        hypothesis_id: sum(
            item.hypothesis_id == hypothesis_id and item.status == "OBSERVED" for item in rows
        )
        for hypothesis_id in PRIMARY_COMPARISON_IDS
    }
    if any(count == 0 for count in observed_by_hypothesis.values()):
        raise PolicyOutcomeError(
            "formal primary comparisons cannot be self-filled as all N/A; "
            f"observed={observed_by_hypothesis}"
        )
    positive_linkage_pairs = {
        (row.task_id, row.axis_id)
        for row in rows
        if row.hypothesis_id == linkage_id
        and row.status == "OBSERVED"
        and float(row.left_value) > float(row.right_value)
    }
    positive_tasks = sorted({task_id for task_id, _ in positive_linkage_pairs})
    positive_axes = sorted({axis_id for _, axis_id in positive_linkage_pairs})
    if not any(
        all(
            (task_id, axis_id) in positive_linkage_pairs
            for task_id in task_pair
            for axis_id in axis_pair
        )
        for task_pair in _pairs(positive_tasks)
        for axis_pair in _pairs(positive_axes)
    ):
        raise PolicyOutcomeError(
            "G03-PolicyLink requires observed positive-direction linkage for a "
            "complete 2-task x 2-axis panel"
        )
    registered = plan.registered_n_a_reasons
    for row in rows:
        if row.status == "N_A" and row.n_a_reason not in registered[row.hypothesis_id]:
            raise PolicyOutcomeError("statistics row uses an unregistered N/A reason")
    return tuple(rows)


def build_policy_outcome_statistics_bridge(
    *,
    barrier: PublicRankingBarrier,
    rankings: Sequence[PublishedFullRanking],
    oracle_handoff: OracleUnlockHandoff,
    external_release_receipt: ExternalOracleReleaseReceipt,
    oracle_evidence: ExternalOracleEvidenceManifest,
    signal_outcomes: SignalOutcomeManifest,
    comparison_plan: CanonicalPrimaryComparisonPlan,
) -> PolicyOutcomeStatisticsBridge:
    """Join formal public rankings to an externally released oracle manifest.

    No path, oracle loader, release signer, or authority flag is accepted.  A
    caller can only supply already-typed immutable artifacts from each side of
    the public/private boundary.
    """

    if not isinstance(comparison_plan, CanonicalPrimaryComparisonPlan):
        raise PolicyOutcomeError("comparison_plan must be canonical and typed")
    full_rankings, ranking_join_digest = _validate_external_join(
        barrier=barrier,
        rankings=rankings,
        handoff=oracle_handoff,
        release=external_release_receipt,
        oracle=oracle_evidence,
        signal=signal_outcomes,
    )
    statistics_plan = comparison_plan.formal_statistics_plan
    if barrier.freeze_manifest.statistics_plan_digest != statistics_plan.plan_digest:
        raise PolicyOutcomeError("canonical primary statistics plan differs from pre-experiment freeze")
    if set(family.contrast_family_id for family in statistics_plan.multiplicity_families) != set(FORMAL_CONTRAST_FAMILY_IDS):
        raise PolicyOutcomeError("formal statistics must cover all six registered contrast families")
    outcomes = _build_outcomes(barrier=barrier, rankings=full_rankings, oracle=oracle_evidence)
    prefix_inputs, linkage_inputs = _build_auxiliary_inputs(outcomes, signal_outcomes)
    contrast_rows = _build_contrast_rows(
        barrier=barrier,
        outcomes=outcomes,
        signal=signal_outcomes,
        plan=comparison_plan,
    )
    frozen = FrozenStatisticsInput(
        run_id=barrier.run_id,
        preexperiment_freeze_manifest_digest=barrier.freeze_manifest_digest,
        statistics_plan_digest=statistics_plan.plan_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
        oracle_unlock_handoff_digest=oracle_handoff.handoff_digest,
        oracle_release_receipt_digest=str(external_release_receipt.release_receipt_digest),
        oracle_evidence_manifest_digest=str(oracle_evidence.evidence_manifest_digest),
        rows=contrast_rows,
    )
    return PolicyOutcomeStatisticsBridge(
        run_id=barrier.run_id,
        freeze_manifest_digest=barrier.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
        oracle_unlock_handoff_digest=oracle_handoff.handoff_digest,
        oracle_release_receipt_digest=str(external_release_receipt.release_receipt_digest),
        oracle_evidence_manifest_digest=str(oracle_evidence.evidence_manifest_digest),
        signal_outcome_manifest_digest=str(signal_outcomes.manifest_digest),
        primary_comparison_plan_digest=comparison_plan.plan_digest,
        ranking_join_digest=ranking_join_digest,
        outcomes=outcomes,
        prefix_inputs=prefix_inputs,
        linkage_inputs=linkage_inputs,
        frozen_statistics_input=frozen,
    )


__all__ = [
    "CanonicalPrimaryComparisonPlan",
    "ExternalOracleEvidenceManifest",
    "ExternalOracleReleaseReceipt",
    "FORMAL_MARKET_SIZE",
    "FORMAL_QUERY_COUNT",
    "FORMAL_TOP_K",
    "G03_POLICY_LINK_MIN_AXES",
    "G03_POLICY_LINK_MIN_TASKS",
    "LINKAGE_N_A_REASONS",
    "OracleEpisodeEvidence",
    "OraclePolicyEvidence",
    "PolicyOutcomeError",
    "PolicyOutcomeRecord",
    "PolicyOutcomeStatisticsBridge",
    "PrefixSampleEfficiencyInput",
    "PRIMARY_COMPARISON_IDS",
    "SignalOutcomeManifest",
    "SignalOutcomeRow",
    "SignalRegretLinkageInput",
    "build_policy_outcome_statistics_bridge",
]
