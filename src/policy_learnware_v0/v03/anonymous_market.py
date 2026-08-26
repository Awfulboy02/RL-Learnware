"""Anonymous, full-pool selection boundary for the v0.3 engineering slice.

This module closes the *engineering* P0 contract only.  In particular, the
The :class:`V03SourcePolicyMarket` accepted here is explicitly tagged
``ENGINEERING_CONTRACT_ONLY``.  It is not P5M formal evidence, an authority
grant, or a conformance verdict.  Formal asset authority must be established
by a later, separately governed P5M acceptance path.

The public phase joins exactly thirty canonical anonymous market identities to
their source-representation digests and public competence/tie data.  It ranks
the complete pool without consulting deployment-private metadata.  Only after
that immutable ranking exists may the private phase inspect the rank-one
``ExecutionABIRecord``.  An incompatible rank one is a terminal selection
failure: rank two is never inspected as a fallback candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping

from ..hashing import sha256_json
from ..v02.schemas import ExecutionABIRecord
from .compute import (
    DistanceForm,
    JointDistanceRequest,
    JointDistanceRun,
    tie_break_digest,
)
from .contracts import (
    MarketBoundSourceRepresentationIndex,
    QuerySpec,
    RankingKey,
)
from .pool_intake import EXPECTED_ANCHOR_COUNT
from .schemas import (
    AnonymousSelectorViewEntry,
    SelectionFailureRecord,
    checked_digest,
    strict_mapping,
)
from .source_market import V03SourcePolicyMarket


ANONYMOUS_SELECTOR_VIEW_MANIFEST_SCHEMA = (
    "policy-learnware.v03-anonymous-selector-view-manifest.v0"
)
EXECUTION_ABI_AUDIT_SCHEMA = "policy-learnware.v03-execution-abi-audit.v0"
ANONYMOUS_SELECTION_RESULT_SCHEMA = "policy-learnware.v03-anonymous-selection-result.v0"

ENGINEERING_EVIDENCE_SCOPE = "ENGINEERING_CONTRACT_ONLY"
NO_FALLBACK_POLICY = "RANK_ONE_ONLY_NO_FALLBACK"

_OPAQUE_LEARNWARE_ID = re.compile(r"^lw-[0-9a-f]{32}$")
_OPAQUE_QUERY_ID = re.compile(r"^v03q-[0-9a-f]{32}$")


class AnonymousMarketError(ValueError):
    """The anonymous market boundary is incomplete or cross-bound."""


def _canonical_pool_ids(ids: Any, where: str) -> frozenset[str]:
    if isinstance(ids, (str, bytes)):
        raise AnonymousMarketError(f"{where} must be an ID collection")
    try:
        values = tuple(ids)
    except TypeError as error:
        raise AnonymousMarketError(f"{where} must be an ID collection") from error
    if len(values) != EXPECTED_ANCHOR_COUNT or len(set(values)) != EXPECTED_ANCHOR_COUNT:
        raise AnonymousMarketError(
            f"{where} must contain exactly {EXPECTED_ANCHOR_COUNT} distinct IDs"
        )
    invalid = sorted(
        str(value)
        for value in values
        if not isinstance(value, str) or _OPAQUE_LEARNWARE_ID.fullmatch(value) is None
    )
    if invalid:
        raise AnonymousMarketError(
            f"{where} contains non-canonical anonymous IDs: {invalid}"
        )
    return frozenset(values)


def _query_id(value: Any) -> str:
    if not isinstance(value, str) or _OPAQUE_QUERY_ID.fullmatch(value) is None:
        raise AnonymousMarketError("opaque_query_id has invalid canonical format")
    return value


@dataclass(frozen=True)
class AnonymousSelectorViewManifest:
    """Typed public selector projection formed by the exact market/index join.

    ``formal_authority_available`` is deliberately fixed to ``False``.  The
    record certifies only that the software boundary was joined and digested;
    it cannot self-authorize P5M or a formal experiment run.
    """

    policy_market_id: str
    representation_index_digest: str
    representation_protocol_id: str
    entries: Mapping[str, AnonymousSelectorViewEntry]
    evidence_scope: Literal["ENGINEERING_CONTRACT_ONLY"] = ENGINEERING_EVIDENCE_SCOPE
    formal_authority_available: Literal[False] = False
    selector_view_digest: str | None = None
    schema: str = ANONYMOUS_SELECTOR_VIEW_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ANONYMOUS_SELECTOR_VIEW_MANIFEST_SCHEMA:
            raise AnonymousMarketError("unsupported anonymous selector-view schema")
        try:
            market_id = checked_digest(self.policy_market_id, "policy_market_id")
            index_digest = checked_digest(
                self.representation_index_digest,
                "representation_index_digest",
            )
            protocol_id = checked_digest(
                self.representation_protocol_id,
                "representation_protocol_id",
            )
        except ValueError as error:
            raise AnonymousMarketError(str(error)) from error
        if self.evidence_scope != ENGINEERING_EVIDENCE_SCOPE:
            raise AnonymousMarketError(
                "anonymous selector view can only claim engineering-contract scope"
            )
        if self.formal_authority_available is not False:
            raise AnonymousMarketError(
                "anonymous selector view cannot self-assert formal P5M authority"
            )
        if not isinstance(self.entries, Mapping):
            raise AnonymousMarketError("anonymous selector entries must be a mapping")
        entries = dict(self.entries)
        ids = _canonical_pool_ids(entries, "anonymous selector view")
        for opaque_id, entry in entries.items():
            if not isinstance(entry, AnonymousSelectorViewEntry):
                raise AnonymousMarketError(
                    "anonymous selector view contains an untyped entry"
                )
            if opaque_id != entry.opaque_learnware_id:
                raise AnonymousMarketError(
                    "anonymous selector entry key and opaque identity differ"
                )
        if len({entry.tie_break_token for entry in entries.values()}) != len(ids):
            raise AnonymousMarketError("anonymous selector tie-break tokens must be unique")
        object.__setattr__(self, "policy_market_id", market_id)
        object.__setattr__(self, "representation_index_digest", index_digest)
        object.__setattr__(self, "representation_protocol_id", protocol_id)
        object.__setattr__(
            self,
            "entries",
            MappingProxyType(dict(sorted(entries.items()))),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.selector_view_digest is None:
            object.__setattr__(self, "selector_view_digest", expected)
        else:
            try:
                observed = checked_digest(
                    self.selector_view_digest,
                    "selector_view_digest",
                )
            except ValueError as error:
                raise AnonymousMarketError(str(error)) from error
            if observed != expected:
                raise AnonymousMarketError(
                    "selector_view_digest does not match anonymous selector view"
                )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_market_id": self.policy_market_id,
            "representation_index_digest": self.representation_index_digest,
            "representation_protocol_id": self.representation_protocol_id,
            "entries": {
                opaque_id: entry.to_dict()
                for opaque_id, entry in self.entries.items()
            },
            "evidence_scope": self.evidence_scope,
            "formal_authority_available": self.formal_authority_available,
        }

    @property
    def tie_break_tokens(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                opaque_id: entry.tie_break_token
                for opaque_id, entry in self.entries.items()
            }
        )

    @property
    def tie_break_map_digest(self) -> str:
        return tie_break_digest(self.tie_break_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "selector_view_digest": self.selector_view_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnonymousSelectorViewManifest":
        fields = {
            "schema",
            "policy_market_id",
            "representation_index_digest",
            "representation_protocol_id",
            "entries",
            "evidence_scope",
            "formal_authority_available",
            "selector_view_digest",
        }
        try:
            data = strict_mapping(value, fields, "anonymous selector-view manifest")
        except ValueError as error:
            raise AnonymousMarketError(str(error)) from error
        if not isinstance(data["entries"], Mapping):
            raise AnonymousMarketError("anonymous selector entries must be a mapping")
        try:
            entries = {
                opaque_id: AnonymousSelectorViewEntry.from_dict(entry)
                for opaque_id, entry in data["entries"].items()
            }
        except (TypeError, ValueError) as error:
            raise AnonymousMarketError(
                "anonymous selector-view entry is invalid"
            ) from error
        return cls(
            policy_market_id=data["policy_market_id"],
            representation_index_digest=data["representation_index_digest"],
            representation_protocol_id=data["representation_protocol_id"],
            entries=entries,
            evidence_scope=data["evidence_scope"],
            formal_authority_available=data["formal_authority_available"],
            selector_view_digest=data["selector_view_digest"],
            schema=data["schema"],
        )


def _validate_market_index_join(
    market: V03SourcePolicyMarket,
    source_index: MarketBoundSourceRepresentationIndex,
) -> frozenset[str]:
    if not isinstance(market, V03SourcePolicyMarket):
        raise AnonymousMarketError("market must be a typed V03SourcePolicyMarket")
    if not isinstance(source_index, MarketBoundSourceRepresentationIndex):
        raise AnonymousMarketError(
            "source_index must be an explicitly market-bound representation index"
        )
    if market.policy_market_id != source_index.policy_market_id:
        raise AnonymousMarketError("market/index policy_market_id mismatch")
    public_ids = _canonical_pool_ids(market.entries, "public policy market")
    private_ids = _canonical_pool_ids(
        market.deployment_private,
        "deployment-private policy market",
    )
    index_ids = _canonical_pool_ids(source_index.entries, "source representation index")
    if public_ids != private_ids or public_ids != index_ids:
        missing = sorted(public_ids - index_ids)
        extra = sorted(index_ids - public_ids)
        raise AnonymousMarketError(
            "anonymous market/index coverage differs; "
            f"missing={missing}, extra={extra}"
        )
    return public_ids


def build_anonymous_selector_view(
    market: V03SourcePolicyMarket,
    source_index: MarketBoundSourceRepresentationIndex,
) -> AnonymousSelectorViewManifest:
    """Join all and only the thirty public identities to representation specs."""

    ids = _validate_market_index_join(market, source_index)
    entries: dict[str, AnonymousSelectorViewEntry] = {}
    for opaque_id in sorted(ids):
        public = market.entries[opaque_id]
        source = source_index.entries[opaque_id]
        if public.opaque_learnware_id != opaque_id:
            raise AnonymousMarketError("public market entry identity mismatch")
        if not isinstance(source.source_spec_digest, str):
            raise AnonymousMarketError("source entry lacks a bound source_spec_digest")
        entries[opaque_id] = AnonymousSelectorViewEntry(
            opaque_learnware_id=opaque_id,
            environment_spec_digest=source.source_spec_digest,
            normalized_source_competence=public.normalized_source_competence,
            tie_break_token=public.tie_break_token,
        )
    return AnonymousSelectorViewManifest(
        policy_market_id=market.policy_market_id,
        representation_index_digest=str(source_index.representation_index_digest),
        representation_protocol_id=source_index.representation_protocol_id,
        entries=entries,
    )


def _validate_view_index_join(
    selector_view: AnonymousSelectorViewManifest,
    source_index: MarketBoundSourceRepresentationIndex,
) -> None:
    if not isinstance(selector_view, AnonymousSelectorViewManifest):
        raise AnonymousMarketError(
            "selector_view must be a typed AnonymousSelectorViewManifest"
        )
    if not isinstance(source_index, MarketBoundSourceRepresentationIndex):
        raise AnonymousMarketError(
            "source_index must be an explicitly market-bound representation index"
        )
    view_ids = _canonical_pool_ids(selector_view.entries, "anonymous selector view")
    index_ids = _canonical_pool_ids(source_index.entries, "source representation index")
    if view_ids != index_ids:
        raise AnonymousMarketError("selector-view/index coverage differs")
    if selector_view.policy_market_id != source_index.policy_market_id:
        raise AnonymousMarketError("selector-view/index policy_market_id mismatch")
    if (
        selector_view.representation_index_digest
        != source_index.representation_index_digest
    ):
        raise AnonymousMarketError("selector view is bound to another source index")
    if (
        selector_view.representation_protocol_id
        != source_index.representation_protocol_id
    ):
        raise AnonymousMarketError(
            "selector view is bound to another representation protocol"
        )
    for opaque_id in view_ids:
        if (
            selector_view.entries[opaque_id].environment_spec_digest
            != source_index.entries[opaque_id].source_spec_digest
        ):
            raise AnonymousMarketError(
                "selector-view/index source-spec digest mismatch"
            )


def build_anonymous_joint_distance_request(
    query_spec: QuerySpec,
    source_index: MarketBoundSourceRepresentationIndex,
    selector_view: AnonymousSelectorViewManifest,
    *,
    distance_form: DistanceForm = "mmd",
    block_size: int = 2048,
    negative_tolerance: float = 1.0e-8,
) -> JointDistanceRequest:
    """Build a full-pool distance request bound to the public selector view."""

    _validate_view_index_join(selector_view, source_index)
    if not hasattr(query_spec, "query_spec_digest"):
        raise AnonymousMarketError("query_spec has the wrong typed query contract")
    ranking_key = RankingKey(
        query_spec_digest=str(query_spec.query_spec_digest),
        representation_index_digest=str(source_index.representation_index_digest),
        selector_digest=str(selector_view.selector_view_digest),
        tie_break_digest=selector_view.tie_break_map_digest,
    )
    return JointDistanceRequest(
        query_spec=query_spec,
        source_index=source_index,
        ranking_key=ranking_key,
        tie_break_tokens=selector_view.tie_break_tokens,
        distance_form=distance_form,
        block_size=block_size,
        negative_tolerance=negative_tolerance,
    )


@dataclass(frozen=True)
class ExecutionABIAuditRecord:
    """Private audit of the already-published rank-one selection only."""

    opaque_query_id: str
    policy_market_id: str
    ranking_digest: str
    selected_opaque_learnware_id: str
    selected_execution_abi_digest: str
    target_execution_abi_digest: str
    compatible: bool
    audited_rank: Literal[1] = 1
    fallback_policy: Literal["RANK_ONE_ONLY_NO_FALLBACK"] = NO_FALLBACK_POLICY
    audit_digest: str | None = None
    schema: str = EXECUTION_ABI_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_ABI_AUDIT_SCHEMA:
            raise AnonymousMarketError("unsupported execution-ABI audit schema")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        if (
            not isinstance(self.selected_opaque_learnware_id, str)
            or _OPAQUE_LEARNWARE_ID.fullmatch(self.selected_opaque_learnware_id) is None
        ):
            raise AnonymousMarketError("selected opaque learnware ID is non-canonical")
        for name in (
            "policy_market_id",
            "ranking_digest",
            "selected_execution_abi_digest",
            "target_execution_abi_digest",
        ):
            try:
                object.__setattr__(self, name, checked_digest(getattr(self, name), name))
            except ValueError as error:
                raise AnonymousMarketError(str(error)) from error
        if type(self.compatible) is not bool:
            raise AnonymousMarketError("execution-ABI compatible flag must be boolean")
        expected_compatible = (
            self.selected_execution_abi_digest == self.target_execution_abi_digest
        )
        if self.compatible != expected_compatible:
            raise AnonymousMarketError(
                "execution-ABI compatible flag disagrees with ABI digests"
            )
        if self.audited_rank != 1 or self.fallback_policy != NO_FALLBACK_POLICY:
            raise AnonymousMarketError("only rank-one ABI audit without fallback is allowed")
        expected = sha256_json(self._payload_without_digest())
        if self.audit_digest is None:
            object.__setattr__(self, "audit_digest", expected)
        else:
            try:
                observed = checked_digest(self.audit_digest, "audit_digest")
            except ValueError as error:
                raise AnonymousMarketError(str(error)) from error
            if observed != expected:
                raise AnonymousMarketError("ABI audit digest does not match payload")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "opaque_query_id": self.opaque_query_id,
            "policy_market_id": self.policy_market_id,
            "ranking_digest": self.ranking_digest,
            "selected_opaque_learnware_id": self.selected_opaque_learnware_id,
            "selected_execution_abi_digest": self.selected_execution_abi_digest,
            "target_execution_abi_digest": self.target_execution_abi_digest,
            "compatible": self.compatible,
            "audited_rank": self.audited_rank,
            "fallback_policy": self.fallback_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "audit_digest": self.audit_digest}


@dataclass(frozen=True)
class AnonymousSelectionResult:
    """Outcome of the public selection followed by one private ABI audit."""

    opaque_query_id: str
    selected_opaque_learnware_id: str
    ranking_digest: str
    abi_audit: ExecutionABIAuditRecord
    failure_record: SelectionFailureRecord | None
    fallback_attempted: Literal[False] = False
    schema: str = ANONYMOUS_SELECTION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ANONYMOUS_SELECTION_RESULT_SCHEMA:
            raise AnonymousMarketError("unsupported anonymous selection result schema")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        if not isinstance(self.abi_audit, ExecutionABIAuditRecord):
            raise AnonymousMarketError("selection result requires a typed ABI audit")
        if self.fallback_attempted is not False:
            raise AnonymousMarketError("anonymous selection cannot attempt fallback")
        if (
            self.selected_opaque_learnware_id
            != self.abi_audit.selected_opaque_learnware_id
            or self.opaque_query_id != self.abi_audit.opaque_query_id
        ):
            raise AnonymousMarketError("selection result and ABI audit identities differ")
        try:
            ranking_digest = checked_digest(self.ranking_digest, "ranking_digest")
        except ValueError as error:
            raise AnonymousMarketError(str(error)) from error
        object.__setattr__(self, "ranking_digest", ranking_digest)
        if ranking_digest != self.abi_audit.ranking_digest:
            raise AnonymousMarketError("selection result and ABI audit rankings differ")
        if self.abi_audit.compatible:
            if self.failure_record is not None:
                raise AnonymousMarketError("compatible selection cannot carry a failure")
        else:
            failure = self.failure_record
            if not isinstance(failure, SelectionFailureRecord):
                raise AnonymousMarketError(
                    "incompatible rank one requires SelectionFailureRecord"
                )
            if (
                failure.status != "SELECTED_INCOMPATIBLE_ABI"
                or failure.opaque_query_id != self.opaque_query_id
                or failure.selected_opaque_learnware_id
                != self.selected_opaque_learnware_id
                or failure.ranking_digest != self.ranking_digest
                or failure.abi_audit_digest != self.abi_audit.audit_digest
            ):
                raise AnonymousMarketError(
                    "selection failure is not bound to the rank-one ABI audit"
                )

    @property
    def deployable(self) -> bool:
        return self.abi_audit.compatible

    @property
    def status(self) -> str:
        return (
            "SELECTED_ABI_COMPATIBLE"
            if self.deployable
            else "SELECTED_INCOMPATIBLE_ABI"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "opaque_query_id": self.opaque_query_id,
            "selected_opaque_learnware_id": self.selected_opaque_learnware_id,
            "ranking_digest": self.ranking_digest,
            "status": self.status,
            "deployable": self.deployable,
            "fallback_attempted": self.fallback_attempted,
            "abi_audit": self.abi_audit.to_dict(),
            "failure_record": (
                None if self.failure_record is None else self.failure_record.to_dict()
            ),
        }


def audit_rank1_execution_abi(
    *,
    opaque_query_id: str,
    request: JointDistanceRequest,
    distance_run: JointDistanceRun,
    selector_view: AnonymousSelectorViewManifest,
    market: V03SourcePolicyMarket,
    target_execution_abi: ExecutionABIRecord,
) -> AnonymousSelectionResult:
    """Audit only rank one after a complete, selector-bound public run.

    The implementation deliberately retrieves ``execution_abi`` from exactly
    one deployment-private entry.  Incompatibility emits a terminal failure;
    no later row is consulted for execution compatibility.
    """

    query_id = _query_id(opaque_query_id)
    if not isinstance(request, JointDistanceRequest):
        raise AnonymousMarketError("request must be a typed JointDistanceRequest")
    if not isinstance(distance_run, JointDistanceRun):
        raise AnonymousMarketError("distance_run must be a typed JointDistanceRun")
    if not isinstance(target_execution_abi, ExecutionABIRecord):
        raise AnonymousMarketError(
            "target_execution_abi must be a typed ExecutionABIRecord"
        )
    _validate_view_index_join(selector_view, request.source_index)
    ids = _validate_market_index_join(market, request.source_index)
    if selector_view.policy_market_id != market.policy_market_id:
        raise AnonymousMarketError("selector view is bound to another policy market")
    if set(selector_view.entries) != ids:
        raise AnonymousMarketError("selector view does not cover the exact market")
    for opaque_id in ids:
        public = market.entries[opaque_id]
        view = selector_view.entries[opaque_id]
        if (
            public.normalized_source_competence
            != view.normalized_source_competence
            or public.tie_break_token != view.tie_break_token
        ):
            raise AnonymousMarketError(
                "selector-view competence/tie data differ from public market"
            )
    if request.ranking_key.selector_digest != selector_view.selector_view_digest:
        raise AnonymousMarketError("ranking key is bound to another selector view")
    if request.ranking_key.tie_break_digest != selector_view.tie_break_map_digest:
        raise AnonymousMarketError("ranking key tie-break digest differs from selector view")
    if dict(request.tie_break_tokens) != dict(selector_view.tie_break_tokens):
        raise AnonymousMarketError("distance request tie map differs from selector view")
    if distance_run.request_digest != request.request_digest:
        raise AnonymousMarketError("distance run is bound to another request")
    if distance_run.query_spec_digest != request.query_spec.query_spec_digest:
        raise AnonymousMarketError("distance run is bound to another query")
    if (
        distance_run.representation_index_digest
        != request.source_index.representation_index_digest
    ):
        raise AnonymousMarketError("distance run is bound to another source index")
    if distance_run.ranking_key_digest != request.ranking_key.ranking_key_digest:
        raise AnonymousMarketError("distance run is bound to another ranking key")
    if (
        distance_run.query_mode != request.query_spec.query_mode
        or distance_run.distance_form != request.distance_form
        or any(
            row.result.query_mode != request.query_spec.query_mode
            or row.result.distance_form != request.distance_form
            for row in distance_run.rows
        )
    ):
        raise AnonymousMarketError("distance run mode differs from its bound request")
    run_ids = _canonical_pool_ids(
        (row.opaque_learnware_id for row in distance_run.rows),
        "joint distance run",
    )
    if run_ids != ids:
        raise AnonymousMarketError("joint distance run is not the complete market")

    rank_one = distance_run.rows[0]
    if rank_one.rank != 1:
        raise AnonymousMarketError("joint distance run does not begin at rank one")
    # This is the only deployment-private ABI lookup in the selection phase.
    selected_execution_abi = market.deployment_private[
        rank_one.opaque_learnware_id
    ].execution_abi
    compatible = selected_execution_abi.digest == target_execution_abi.digest
    audit = ExecutionABIAuditRecord(
        opaque_query_id=query_id,
        policy_market_id=market.policy_market_id,
        ranking_digest=str(distance_run.run_digest),
        selected_opaque_learnware_id=rank_one.opaque_learnware_id,
        selected_execution_abi_digest=selected_execution_abi.digest,
        target_execution_abi_digest=target_execution_abi.digest,
        compatible=compatible,
    )
    failure = None
    if not compatible:
        failure = SelectionFailureRecord(
            opaque_query_id=query_id,
            selected_opaque_learnware_id=rank_one.opaque_learnware_id,
            status="SELECTED_INCOMPATIBLE_ABI",
            ranking_digest=str(distance_run.run_digest),
            abi_audit_digest=str(audit.audit_digest),
        )
    return AnonymousSelectionResult(
        opaque_query_id=query_id,
        selected_opaque_learnware_id=rank_one.opaque_learnware_id,
        ranking_digest=str(distance_run.run_digest),
        abi_audit=audit,
        failure_record=failure,
    )


__all__ = [
    "ANONYMOUS_SELECTION_RESULT_SCHEMA",
    "ANONYMOUS_SELECTOR_VIEW_MANIFEST_SCHEMA",
    "ENGINEERING_EVIDENCE_SCOPE",
    "EXECUTION_ABI_AUDIT_SCHEMA",
    "NO_FALLBACK_POLICY",
    "AnonymousMarketError",
    "AnonymousSelectionResult",
    "AnonymousSelectorViewManifest",
    "ExecutionABIAuditRecord",
    "audit_rank1_execution_abi",
    "build_anonymous_joint_distance_request",
    "build_anonymous_selector_view",
]
