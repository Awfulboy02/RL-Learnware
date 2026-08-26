"""Episode-boundary prefix curves over one immutable max-prefix cache.

Prefix evaluation is deliberately separate from :class:`SignalCellRun`: the
main signal metric and exact-repeat max-prefix controls remain unchanged.  A
caller encodes each query bank once into a :class:`SemanticCacheRecord`; this
module only takes immutable episode-prefix slices and constructs empirical
QueryKMEs against the already reduced source index.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..hashing import sha256_json
from .compute import JointDistanceRequest, run_joint_distance_stage, tie_break_digest
from .contracts import (
    RankingKey,
    SemanticCacheRecord,
    SourceRepresentationIndex,
    build_empirical_query_spec,
)
from .signal_matrix import SignalCell
from .signal_metrics import SignalDistanceRow, SignalMetricRecord
from .signal_runtime import (
    DEVELOPMENT_SMOKE_MODE,
    FORMAL_MODE,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
)


FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
SIGNAL_PREFIX_SCHEDULE_SCHEMA = "policy-learnware.v03-signal-prefix-schedule.v0"
SIGNAL_PREFIX_CACHE_SET_SCHEMA = "policy-learnware.v03-signal-prefix-cache-set.v0"
SIGNAL_PREFIX_EXECUTION_PROTOCOL_SCHEMA = (
    "policy-learnware.v03-signal-prefix-execution-protocol.v0"
)
SIGNAL_PREFIX_POINT_SCHEMA = "policy-learnware.v03-signal-prefix-point.v0"
SIGNAL_PREFIX_RUN_SCHEMA = "policy-learnware.v03-signal-prefix-run.v0"


class SignalPrefixError(ValueError):
    """A prefix schedule, cache join, or numeric result is invalid."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SignalPrefixError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result != result.lower():
        raise SignalPrefixError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise SignalPrefixError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


@dataclass(frozen=True)
class SignalPrefixSchedule:
    prefix_episode_counts: tuple[int, ...]
    scope: str
    schedule_digest: str | None = None
    schema: str = SIGNAL_PREFIX_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_PREFIX_SCHEDULE_SCHEMA:
            raise SignalPrefixError("unsupported SignalPrefixSchedule schema")
        prefixes = tuple(self.prefix_episode_counts)
        if (
            not prefixes
            or any(type(item) is not int or item <= 0 for item in prefixes)
            or prefixes != tuple(sorted(set(prefixes)))
        ):
            raise SignalPrefixError(
                "prefix counts must be positive, unique, and strictly increasing"
            )
        if self.scope not in {"FORMAL", "DEVELOPMENT"}:
            raise SignalPrefixError("unknown signal-prefix schedule scope")
        if (
            self.scope == "FORMAL"
            and prefixes != FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
        ):
            raise SignalPrefixError(
                "formal signal prefixes must be exactly 1/2/4/8/16/32/64"
            )
        object.__setattr__(self, "prefix_episode_counts", prefixes)
        expected = sha256_json(self._payload_without_digest())
        if self.schedule_digest is None:
            object.__setattr__(self, "schedule_digest", expected)
        elif _digest(self.schedule_digest, "schedule_digest") != expected:
            raise SignalPrefixError("prefix schedule digest does not match contents")

    @classmethod
    def formal(cls) -> "SignalPrefixSchedule":
        return cls(FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS, "FORMAL")

    @classmethod
    def development(cls, prefixes: Sequence[int]) -> "SignalPrefixSchedule":
        return cls(tuple(prefixes), "DEVELOPMENT")

    @property
    def max_prefix(self) -> int:
        return self.prefix_episode_counts[-1]

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "scope": self.scope,
            "prefix_episode_counts": list(self.prefix_episode_counts),
            "slice_unit": "COMPLETE_EPISODE_PREFIX",
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "schedule_digest": self.schedule_digest}


@dataclass(frozen=True)
class SignalPrefixCacheSet:
    """Exactly one max-prefix semantic cache per query bank."""

    identity_registry_digest: str
    query_caches: Mapping[str, SemanticCacheRecord]
    query_receipt_digests: Mapping[str, str]
    query_raw_dataset_digests: Mapping[str, str]
    cache_set_digest: str | None = None
    schema: str = SIGNAL_PREFIX_CACHE_SET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_PREFIX_CACHE_SET_SCHEMA:
            raise SignalPrefixError("unsupported SignalPrefixCacheSet schema")
        object.__setattr__(
            self,
            "identity_registry_digest",
            _digest(self.identity_registry_digest, "identity_registry_digest"),
        )
        caches = dict(self.query_caches)
        if not caches:
            raise SignalPrefixError("prefix cache set cannot be empty")
        for query_id, cache in caches.items():
            _nonempty(query_id, "query bank ID")
            if not isinstance(cache, SemanticCacheRecord):
                raise SignalPrefixError(
                    "prefix cache set requires SemanticCacheRecord values"
                )
        if len({cache.semantic_cache_digest for cache in caches.values()}) != len(caches):
            raise SignalPrefixError("query banks cannot alias one semantic cache")
        if len(
            {cache.key.representation_protocol_digest for cache in caches.values()}
        ) != 1:
            raise SignalPrefixError("query prefix caches mix representation protocols")
        if len({cache.key.canonical_view_digest for cache in caches.values()}) != 1:
            raise SignalPrefixError("query prefix caches mix canonical views")
        caches = dict(sorted(caches.items()))
        receipt_digests = {
            _nonempty(query_id, "query bank ID"): _digest(
                value, "query receipt digest"
            )
            for query_id, value in sorted(self.query_receipt_digests.items())
        }
        raw_digests = {
            _nonempty(query_id, "query bank ID"): _digest(
                value, "query raw-dataset digest"
            )
            for query_id, value in sorted(self.query_raw_dataset_digests.items())
        }
        if set(receipt_digests) != set(caches) or set(raw_digests) != set(caches):
            raise SignalPrefixError(
                "query cache, receipt and raw-dataset memberships must match"
            )
        if any(
            raw_digests[query_id] != cache.key.raw_dataset_digest
            for query_id, cache in caches.items()
        ):
            raise SignalPrefixError(
                "query raw-dataset provenance differs from semantic-cache key"
            )
        object.__setattr__(self, "query_caches", MappingProxyType(caches))
        object.__setattr__(
            self, "query_receipt_digests", MappingProxyType(receipt_digests)
        )
        object.__setattr__(
            self, "query_raw_dataset_digests", MappingProxyType(raw_digests)
        )
        expected = sha256_json(self._payload_without_digest())
        if self.cache_set_digest is None:
            object.__setattr__(self, "cache_set_digest", expected)
        elif _digest(self.cache_set_digest, "cache_set_digest") != expected:
            raise SignalPrefixError("prefix cache-set digest does not match contents")

    @property
    def episode_count(self) -> int:
        counts = {cache.episode_count for cache in self.query_caches.values()}
        if len(counts) != 1:
            raise SignalPrefixError("max-prefix query caches have different episode counts")
        return next(iter(counts))

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "identity_registry_digest": self.identity_registry_digest,
            "query_caches": {
                query_id: cache.to_manifest_dict()
                for query_id, cache in self.query_caches.items()
            },
            "query_receipt_digests": dict(self.query_receipt_digests),
            "query_raw_dataset_digests": dict(self.query_raw_dataset_digests),
            "encoder_forward_count_per_query": 1,
            "downstream_operation": "EPISODE_BOUNDARY_SLICE_ONLY",
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "cache_set_digest": self.cache_set_digest}


@dataclass(frozen=True)
class SignalPrefixExecutionProtocol:
    signal_execution_protocol_digest: str
    signal_execution_mode: str
    plan_digest: str
    cell_id: str
    cell_digest: str
    identity_registry_digest: str
    measurement_protocol_digest: str
    source_index_digest: str
    query_cache_set_digest: str
    prefix_schedule: SignalPrefixSchedule
    block_size: int
    protocol_digest: str | None = None
    schema: str = SIGNAL_PREFIX_EXECUTION_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_PREFIX_EXECUTION_PROTOCOL_SCHEMA:
            raise SignalPrefixError(
                "unsupported SignalPrefixExecutionProtocol schema"
            )
        for name in (
            "signal_execution_protocol_digest",
            "plan_digest",
            "cell_digest",
            "identity_registry_digest",
            "measurement_protocol_digest",
            "source_index_digest",
            "query_cache_set_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "cell_id", _nonempty(self.cell_id, "cell_id"))
        if self.signal_execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise SignalPrefixError("unknown signal execution mode")
        if not isinstance(self.prefix_schedule, SignalPrefixSchedule):
            raise SignalPrefixError("prefix protocol requires a typed schedule")
        if (
            self.signal_execution_mode == FORMAL_MODE
            and self.prefix_schedule.scope != "FORMAL"
        ):
            raise SignalPrefixError(
                "formal signal execution requires the exact formal prefix schedule"
            )
        if type(self.block_size) is not int or self.block_size <= 0:
            raise SignalPrefixError("block_size must be a positive integer")
        expected = sha256_json(self._payload_without_digest())
        if self.protocol_digest is None:
            object.__setattr__(self, "protocol_digest", expected)
        elif _digest(self.protocol_digest, "protocol_digest") != expected:
            raise SignalPrefixError("prefix protocol digest does not match contents")

    @classmethod
    def create(
        cls,
        *,
        signal_execution_protocol: SignalExecutionProtocol,
        cell: SignalCell,
        source_index: SourceRepresentationIndex,
        query_cache_set: SignalPrefixCacheSet,
        prefix_schedule: SignalPrefixSchedule,
    ) -> "SignalPrefixExecutionProtocol":
        if not isinstance(signal_execution_protocol, SignalExecutionProtocol):
            raise SignalPrefixError("typed signal execution protocol is required")
        if not isinstance(cell, SignalCell) or cell.applicability != "NUMERIC":
            raise SignalPrefixError("prefix execution requires a numeric SignalCell")
        if not isinstance(source_index, SourceRepresentationIndex):
            raise SignalPrefixError("prefix execution requires a source index")
        if not isinstance(query_cache_set, SignalPrefixCacheSet):
            raise SignalPrefixError("prefix execution requires a max-prefix cache set")
        if (
            signal_execution_protocol.identity_registry_digest
            != query_cache_set.identity_registry_digest
        ):
            raise SignalPrefixError("query caches differ from identity freeze")
        reference = next(iter(source_index.entries.values()))
        if (
            source_index.representation_protocol_id
            != next(iter(query_cache_set.query_caches.values())).key.representation_protocol_digest
            or reference.canonical_view_digest
            != next(iter(query_cache_set.query_caches.values())).key.canonical_view_digest
            or reference.measurement_protocol_id
            != signal_execution_protocol.protocol_digest
        ):
            raise SignalPrefixError(
                "source index/query caches differ from the signal coordinate"
            )
        if query_cache_set.episode_count != prefix_schedule.max_prefix:
            raise SignalPrefixError(
                "each cache must be the exact maximum prefix, not a larger/smaller bank"
            )
        return cls(
            signal_execution_protocol_digest=str(
                signal_execution_protocol.protocol_digest
            ),
            signal_execution_mode=signal_execution_protocol.execution_mode,
            plan_digest=signal_execution_protocol.plan_digest,
            cell_id=cell.cell_id,
            cell_digest=str(cell.cell_digest),
            identity_registry_digest=signal_execution_protocol.identity_registry_digest,
            measurement_protocol_digest=signal_execution_protocol.measurement_protocol_digest,
            source_index_digest=str(source_index.representation_index_digest),
            query_cache_set_digest=str(query_cache_set.cache_set_digest),
            prefix_schedule=prefix_schedule,
            block_size=signal_execution_protocol.block_size,
        )

    def validate_inputs(
        self,
        *,
        signal_execution_protocol: SignalExecutionProtocol,
        cell: SignalCell,
        source_index: SourceRepresentationIndex,
        query_cache_set: SignalPrefixCacheSet,
    ) -> None:
        rebuilt = type(self).create(
            signal_execution_protocol=signal_execution_protocol,
            cell=cell,
            source_index=source_index,
            query_cache_set=query_cache_set,
            prefix_schedule=self.prefix_schedule,
        )
        if rebuilt.to_dict() != self.to_dict():
            raise SignalPrefixError("runtime inputs differ from frozen prefix protocol")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "signal_execution_protocol_digest": self.signal_execution_protocol_digest,
            "signal_execution_mode": self.signal_execution_mode,
            "plan_digest": self.plan_digest,
            "cell_id": self.cell_id,
            "cell_digest": self.cell_digest,
            "identity_registry_digest": self.identity_registry_digest,
            "measurement_protocol_digest": self.measurement_protocol_digest,
            "source_index_digest": self.source_index_digest,
            "query_cache_set_digest": self.query_cache_set_digest,
            "prefix_schedule_digest": self.prefix_schedule.schedule_digest,
            "block_size": self.block_size,
            "source_kme_mode": "REDUCED",
            "query_kme_mode": "EMPIRICAL",
            "query_encoding_rule": "ONE_MAX_PREFIX_FORWARD_THEN_EPISODE_SLICES",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "prefix_schedule": self.prefix_schedule.to_dict(),
            "protocol_digest": self.protocol_digest,
        }


@dataclass(frozen=True)
class SignalPrefixPoint:
    prefix_episode_count: int
    query_spec_digests: Mapping[str, str]
    query_run_digests: Mapping[str, str]
    metric_record: SignalMetricRecord
    point_digest: str | None = None
    schema: str = SIGNAL_PREFIX_POINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_PREFIX_POINT_SCHEMA:
            raise SignalPrefixError("unsupported SignalPrefixPoint schema")
        if type(self.prefix_episode_count) is not int or self.prefix_episode_count <= 0:
            raise SignalPrefixError("prefix_episode_count must be positive")
        specs = {
            _nonempty(key, "query ID"): _digest(value, "query spec digest")
            for key, value in sorted(self.query_spec_digests.items())
        }
        runs = {
            _nonempty(key, "query ID"): _digest(value, "query run digest")
            for key, value in sorted(self.query_run_digests.items())
        }
        if not specs or specs.keys() != runs.keys():
            raise SignalPrefixError("prefix query spec/run membership differs")
        if not isinstance(self.metric_record, SignalMetricRecord):
            raise SignalPrefixError("prefix point requires a SignalMetricRecord")
        if set(self.metric_record.expected_source_by_query) != set(specs):
            raise SignalPrefixError("prefix metric query membership differs from specs")
        object.__setattr__(self, "query_spec_digests", MappingProxyType(specs))
        object.__setattr__(self, "query_run_digests", MappingProxyType(runs))
        expected = sha256_json(self._payload_without_digest())
        if self.point_digest is None:
            object.__setattr__(self, "point_digest", expected)
        elif _digest(self.point_digest, "point_digest") != expected:
            raise SignalPrefixError("prefix point digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "prefix_episode_count": self.prefix_episode_count,
            "query_spec_digests": dict(self.query_spec_digests),
            "query_run_digests": dict(self.query_run_digests),
            "metric_record_digest": self.metric_record.record_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "metric_record": self.metric_record.to_dict(),
            "point_digest": self.point_digest,
        }


@dataclass(frozen=True)
class SignalPrefixRun:
    execution_protocol: SignalPrefixExecutionProtocol
    points: tuple[SignalPrefixPoint, ...]
    formal_authorization_digest: str | None = None
    run_digest: str | None = None
    schema: str = SIGNAL_PREFIX_RUN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIGNAL_PREFIX_RUN_SCHEMA:
            raise SignalPrefixError("unsupported SignalPrefixRun schema")
        if not isinstance(self.execution_protocol, SignalPrefixExecutionProtocol):
            raise SignalPrefixError("prefix run requires typed execution protocol")
        if self.execution_protocol.signal_execution_mode == FORMAL_MODE:
            if self.formal_authorization_digest is None:
                raise SignalPrefixError(
                    "formal prefix run requires an authorization digest"
                )
            object.__setattr__(
                self,
                "formal_authorization_digest",
                _digest(
                    self.formal_authorization_digest,
                    "formal_authorization_digest",
                ),
            )
        elif self.formal_authorization_digest is not None:
            raise SignalPrefixError(
                "development prefix run cannot carry formal authorization"
            )
        points = tuple(self.points)
        if not points or not all(isinstance(item, SignalPrefixPoint) for item in points):
            raise SignalPrefixError("prefix run requires typed points")
        if tuple(item.prefix_episode_count for item in points) != (
            self.execution_protocol.prefix_schedule.prefix_episode_counts
        ):
            raise SignalPrefixError("prefix points differ from frozen schedule")
        for point in points:
            record = point.metric_record
            if (
                record.cell_id != self.execution_protocol.cell_id
                or record.source_index_digest
                != self.execution_protocol.source_index_digest
            ):
                raise SignalPrefixError("prefix metric differs from frozen cell/index")
        object.__setattr__(self, "points", points)
        expected = sha256_json(self._payload_without_digest())
        if self.run_digest is None:
            object.__setattr__(self, "run_digest", expected)
        elif _digest(self.run_digest, "run_digest") != expected:
            raise SignalPrefixError("prefix run digest does not match contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "execution_protocol_digest": self.execution_protocol.protocol_digest,
            "formal_authorization_digest": self.formal_authorization_digest,
            "point_digests": [item.point_digest for item in self.points],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "execution_protocol": self.execution_protocol.to_dict(),
            "points": [item.to_dict() for item in self.points],
            "run_digest": self.run_digest,
        }

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "policy-learnware.v03-public-signal-prefix-run.v0",
            "execution_protocol_digest": self.execution_protocol.protocol_digest,
            "formal_authorization_digest": self.formal_authorization_digest,
            "schedule_digest": self.execution_protocol.prefix_schedule.schedule_digest,
            "points": [
                {
                    "prefix_episode_count": point.prefix_episode_count,
                    "metric_record": point.metric_record.to_public_dict(),
                }
                for point in self.points
            ],
            "private_query_specs_and_distance_rows_withheld": True,
            "run_digest": self.run_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


def run_signal_prefixes(
    *,
    protocol: SignalPrefixExecutionProtocol,
    signal_execution_protocol: SignalExecutionProtocol,
    cell: SignalCell,
    source_index: SourceRepresentationIndex,
    query_cache_set: SignalPrefixCacheSet,
    identity_registry: SignalIdentityRegistry,
    expected_source_by_query: Mapping[str, str],
    representation_coordinate_digest: str,
    representation_seed: int | None,
    formal_authorization: Any | None = None,
) -> SignalPrefixRun:
    """Build every prefix metric without invoking an encoder or a reducer."""

    if not isinstance(protocol, SignalPrefixExecutionProtocol):
        raise SignalPrefixError("run requires SignalPrefixExecutionProtocol")
    protocol.validate_inputs(
        signal_execution_protocol=signal_execution_protocol,
        cell=cell,
        source_index=source_index,
        query_cache_set=query_cache_set,
    )
    if protocol.signal_execution_mode == FORMAL_MODE:
        # Local import keeps the prefix contract independent of the atlas
        # module at import time while still making formal execution fail
        # closed on external authority.
        from .signal_atlas import FormalSignalAtlasAuthorization

        if not isinstance(formal_authorization, FormalSignalAtlasAuthorization):
            raise SignalPrefixError(
                "formal prefix execution requires signal-atlas authorization"
            )
        if (
            formal_authorization.plan_digest != protocol.plan_digest
            or formal_authorization.execution_protocol_digest
            != protocol.signal_execution_protocol_digest
            or formal_authorization.identity_registry_digest
            != protocol.identity_registry_digest
        ):
            raise SignalPrefixError(
                "formal prefix authorization belongs to another signal freeze"
            )
        try:
            formal_authorization.validate_signal_prefix_schedule(
                protocol.prefix_schedule
            )
        except Exception as error:
            raise SignalPrefixError(str(error)) from error
    elif formal_authorization is not None:
        raise SignalPrefixError(
            "development prefix execution must remain outside formal authorization"
        )
    if not isinstance(identity_registry, SignalIdentityRegistry):
        raise SignalPrefixError("run requires SignalIdentityRegistry")
    if identity_registry.registry_digest != protocol.identity_registry_digest:
        raise SignalPrefixError("runtime identity registry differs from prefix freeze")
    coordinate = _digest(
        representation_coordinate_digest, "representation_coordinate_digest"
    )
    if representation_seed is not None and (
        type(representation_seed) is not int or representation_seed < 0
    ):
        raise SignalPrefixError("representation_seed must be non-negative or null")
    expected = dict(sorted(expected_source_by_query.items()))
    if set(expected) != set(query_cache_set.query_caches):
        raise SignalPrefixError("expected-source map differs from query cache set")
    if any(source_id not in source_index.entries for source_id in expected.values()):
        raise SignalPrefixError("expected source is absent from source index")

    identities = {item.bank_id: item for item in identity_registry.identities}
    required_ids = set(source_index.entries) | set(query_cache_set.query_caches)
    if not required_ids <= set(identities):
        raise SignalPrefixError("source/query bank is absent from identity registry")
    for bank_id in required_ids:
        if identities[bank_id].measurement_protocol_digest != (
            protocol.measurement_protocol_digest
        ):
            raise SignalPrefixError("bank identity differs from measurement freeze")
    if any(
        identities[query_id].receipt_digest
        != query_cache_set.query_receipt_digests[query_id]
        for query_id in query_cache_set.query_caches
    ):
        raise SignalPrefixError(
            "query semantic cache differs from the frozen receipt identity"
        )

    source_reference = next(iter(source_index.entries.values()))
    measurement_protocol_id = source_reference.measurement_protocol_id
    tie_tokens = {
        source_id: sha256_json(
            {
                "schema": "policy-learnware.v03-signal-prefix-tie-token.v0",
                "cell_id": cell.cell_id,
                "source_bank_id": source_id,
            }
        )
        for source_id in source_index.entries
    }
    tie_digest = tie_break_digest(tie_tokens)
    selector_digest = sha256_json(
        {
            "schema": "policy-learnware.v03-signal-prefix-selector.v0",
            "formula": "ascending_empirical_to_reduced_mmd",
            "tie_break_digest": tie_digest,
            "prefix_protocol_digest": protocol.protocol_digest,
        }
    )
    points: list[SignalPrefixPoint] = []
    for prefix in protocol.prefix_schedule.prefix_episode_counts:
        query_specs: dict[str, str] = {}
        query_runs: dict[str, str] = {}
        distance_rows: list[SignalDistanceRow] = []
        for query_id, cache in query_cache_set.query_caches.items():
            query_identity = identities[query_id]
            query = build_empirical_query_spec(
                cache,
                kernel_bandwidth=source_reference.kernel_bandwidth,
                measurement_protocol_id=measurement_protocol_id,
                probe_dataset_digest=cache.key.raw_dataset_digest,
                episode_count=prefix,
                block_size=protocol.block_size,
            )
            query_specs[query_id] = str(query.query_spec_digest)
            ranking_key = RankingKey(
                query_spec_digest=str(query.query_spec_digest),
                representation_index_digest=str(
                    source_index.representation_index_digest
                ),
                selector_digest=selector_digest,
                tie_break_digest=tie_digest,
            )
            run = run_joint_distance_stage(
                JointDistanceRequest(
                    query_spec=query,
                    source_index=source_index,
                    ranking_key=ranking_key,
                    tie_break_tokens=tie_tokens,
                    block_size=protocol.block_size,
                )
            )
            query_runs[query_id] = str(run.run_digest)
            for result_row in run.rows:
                source_id = result_row.opaque_learnware_id
                source_identity = identities[source_id]
                source_spec = source_index.entries[source_id]
                distance_rows.append(
                    SignalDistanceRow(
                        query_bank_id=query_id,
                        source_bank_id=source_id,
                        query_receipt_digest=query_identity.receipt_digest,
                        source_receipt_digest=source_identity.receipt_digest,
                        query_raw_dataset_digest=cache.key.raw_dataset_digest,
                        source_raw_dataset_digest=source_spec.probe_dataset_digest,
                        query_task_id=query_identity.task_private_id,
                        source_task_id=source_identity.task_private_id,
                        query_context_id=query_identity.context_id,
                        source_context_id=source_identity.context_id,
                        query_embodiment_id=query_identity.embodiment_id,
                        source_embodiment_id=source_identity.embodiment_id,
                        query_abi_contract_id=query_identity.abi_contract_id,
                        source_abi_contract_id=source_identity.abi_contract_id,
                        query_goal_contract_id=query_identity.goal_contract_id,
                        source_goal_contract_id=source_identity.goal_contract_id,
                        query_dynamics_context_id=(
                            query_identity.dynamics_context_id
                        ),
                        source_dynamics_context_id=(
                            source_identity.dynamics_context_id
                        ),
                        query_equivalence_class_id=(
                            query_identity.equivalence_class_id
                        ),
                        source_equivalence_class_id=(
                            source_identity.equivalence_class_id
                        ),
                        distance=result_row.result.value,
                    )
                )
        metric = SignalMetricRecord(
            cell_id=cell.cell_id,
            view_or_condition_id=cell.condition_id,
            representation_id=cell.representation_id,
            representation_coordinate_digest=coordinate,
            representation_seed=representation_seed,
            source_index_digest=str(source_index.representation_index_digest),
            query_manifest_digest=sha256_json(
                {
                    "schema": "policy-learnware.v03-prefix-query-manifest.v0",
                    "prefix_episode_count": prefix,
                    "query_cache_set_digest": query_cache_set.cache_set_digest,
                    "query_spec_digests": dict(sorted(query_specs.items())),
                    "expected_source_by_query": expected,
                }
            ),
            rows=tuple(distance_rows),
            expected_source_by_query=expected,
        )
        points.append(
            SignalPrefixPoint(
                prefix_episode_count=prefix,
                query_spec_digests=query_specs,
                query_run_digests=query_runs,
                metric_record=metric,
            )
        )
    return SignalPrefixRun(
        execution_protocol=protocol,
        points=tuple(points),
        formal_authorization_digest=(
            None
            if formal_authorization is None
            else str(formal_authorization.authorization_digest)
        ),
    )


__all__ = [
    "FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS",
    "SIGNAL_PREFIX_CACHE_SET_SCHEMA",
    "SIGNAL_PREFIX_EXECUTION_PROTOCOL_SCHEMA",
    "SIGNAL_PREFIX_POINT_SCHEMA",
    "SIGNAL_PREFIX_RUN_SCHEMA",
    "SIGNAL_PREFIX_SCHEDULE_SCHEMA",
    "SignalPrefixCacheSet",
    "SignalPrefixError",
    "SignalPrefixExecutionProtocol",
    "SignalPrefixPoint",
    "SignalPrefixRun",
    "SignalPrefixSchedule",
    "run_signal_prefixes",
]
