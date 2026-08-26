"""Pure pre-oracle derivation of the registered policy-link signal.

No caller supplies a signal value.  The only selectable input is one work key
already reviewed for *both* prefix and dynamics readouts.  Every prefix point
is replayed through the typed dynamics-axis evaluator, joined to the exact 66
opaque queries, and converted to the existing ``SignalOutcomeManifest``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Any, Mapping

from ..hashing import sha256_json
from .dynamics_axis import (
    DynamicsAxisDiagnostics,
    DynamicsPublicQueryJoin,
    build_dynamics_axis_diagnostics,
)
from .policy_outcomes import SignalOutcomeManifest, SignalOutcomeRow
from .preflight import PublicQueryPlan
from .signal_prefix import FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
from .signal_readout import FormalSignalReadoutBundle, FormalSignalReadoutPlan
from .signal_runtime import FORMAL_MODE


FORMAL_SIGNAL_EXTRACTION_PLAN_SCHEMA = (
    "policy-learnware.v03-formal-signal-extraction-plan.v0"
)
FORMAL_SIGNAL_QUERY_BANK_JOIN_SCHEMA = (
    "policy-learnware.v03-formal-signal-query-bank-join.v0"
)
PUBLIC_SIGNAL_QUERY_BANK_JOIN_SCHEMA = (
    "policy-learnware.v03-public-signal-query-bank-join.v0"
)
PREORACLE_SIGNAL_OUTCOME_SCHEMA = (
    "policy-learnware.v03-preoracle-signal-outcome.v0"
)
PREORACLE_SIGNAL_OUTCOME_PUBLICATION_SCHEMA = (
    "policy-learnware.v03-preoracle-signal-outcome-publication.v0"
)
REGISTERED_SIGNAL_METRIC_ID = "dynamics_neighborhood_top1"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_OPAQUE_QUERY_ID = re.compile(r"^v03q-[0-9a-f]{32}$")


class PreOracleSignalError(ValueError):
    """A reviewed signal extraction or its pure derivation is invalid."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PreOracleSignalError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PreOracleSignalError(f"{where} must be a canonical safe ID")
    return value


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PreOracleSignalError(f"{where} must be a non-empty canonical string")
    return value


def _opaque_query_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or _OPAQUE_QUERY_ID.fullmatch(value) is None:
        raise PreOracleSignalError(f"{where} must be a canonical v03 query ID")
    return value


@dataclass(frozen=True)
class FormalSignalQueryBankJoin:
    """Private bijection from public query aliases to sealed signal-bank IDs.

    The public query plan intentionally does not expose signal-bank identity.
    This typed join commits that missing edge without assuming that an opaque
    query ID is also a bank ID.  Its public projection withholds the mapping.
    """

    formal_signal_readout_plan_digest: str
    selected_work_key: str
    public_query_plan_digest: str
    query_alias_manifest_digest: str
    dynamics_public_query_join_digest: str
    dynamics_axis_registry_digest: str
    query_bank_id_by_opaque_query_id: Mapping[str, str]
    join_digest: str | None = None
    schema: str = FORMAL_SIGNAL_QUERY_BANK_JOIN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SIGNAL_QUERY_BANK_JOIN_SCHEMA:
            raise PreOracleSignalError(
                "unsupported FormalSignalQueryBankJoin schema"
            )
        for name in (
            "formal_signal_readout_plan_digest",
            "public_query_plan_digest",
            "query_alias_manifest_digest",
            "dynamics_public_query_join_digest",
            "dynamics_axis_registry_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "selected_work_key",
            _nonempty(self.selected_work_key, "selected_work_key"),
        )
        if not isinstance(self.query_bank_id_by_opaque_query_id, Mapping):
            raise PreOracleSignalError(
                "formal query-bank aliases must be a mapping"
            )
        mapping = {
            _opaque_query_id(query_id, "opaque query ID"): _nonempty(
                bank_id, "private query bank ID"
            )
            for query_id, bank_id in sorted(
                self.query_bank_id_by_opaque_query_id.items()
            )
        }
        if len(mapping) != 66:
            raise PreOracleSignalError(
                "formal query-bank join requires exactly 66 opaque queries"
            )
        if len(set(mapping.values())) != len(mapping):
            raise PreOracleSignalError(
                "formal query-bank join must be a one-to-one bank mapping"
            )
        object.__setattr__(
            self,
            "query_bank_id_by_opaque_query_id",
            MappingProxyType(mapping),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.join_digest is None:
            object.__setattr__(self, "join_digest", expected)
        elif _digest(self.join_digest, "join_digest") != expected:
            raise PreOracleSignalError("formal query-bank join digest mismatch")

    @classmethod
    def bind(
        cls,
        *,
        readout_plan: FormalSignalReadoutPlan,
        public_query_plan: PublicQueryPlan,
        dynamics_public_query_join: DynamicsPublicQueryJoin,
        selected_work_key: str,
        query_bank_id_by_opaque_query_id: Mapping[str, str],
    ) -> "FormalSignalQueryBankJoin":
        if not isinstance(readout_plan, FormalSignalReadoutPlan):
            raise PreOracleSignalError("query-bank join requires a typed readout plan")
        if not isinstance(public_query_plan, PublicQueryPlan):
            raise PreOracleSignalError("query-bank join requires a public query plan")
        if not isinstance(dynamics_public_query_join, DynamicsPublicQueryJoin):
            raise PreOracleSignalError("query-bank join requires a dynamics query join")
        if not isinstance(query_bank_id_by_opaque_query_id, Mapping):
            raise PreOracleSignalError("private query-bank aliases must be a mapping")
        if selected_work_key not in (
            set(readout_plan.prefix_work_keys)
            & set(readout_plan.dynamics_work_keys)
        ):
            raise PreOracleSignalError(
                "query-bank join work key must be in both reviewed readout sets"
            )
        if (
            public_query_plan.plan_digest != readout_plan.public_query_plan_digest
            or dynamics_public_query_join.public_query_plan_digest
            != public_query_plan.plan_digest
            or dynamics_public_query_join.query_alias_manifest_digest
            != public_query_plan.query_alias_manifest_digest
            or dynamics_public_query_join.dynamics_axis_registry_digest
            != readout_plan.dynamics_axis_registry_digest
            or set(query_bank_id_by_opaque_query_id)
            != set(public_query_plan.opaque_query_ids)
        ):
            raise PreOracleSignalError(
                "query-bank mapping differs from the reviewed query/readout plans"
            )
        return cls(
            formal_signal_readout_plan_digest=str(readout_plan.plan_digest),
            selected_work_key=selected_work_key,
            public_query_plan_digest=str(public_query_plan.plan_digest),
            query_alias_manifest_digest=(
                dynamics_public_query_join.query_alias_manifest_digest
            ),
            dynamics_public_query_join_digest=str(
                dynamics_public_query_join.join_digest
            ),
            dynamics_axis_registry_digest=(
                dynamics_public_query_join.dynamics_axis_registry_digest
            ),
            query_bank_id_by_opaque_query_id=query_bank_id_by_opaque_query_id,
        )

    def validate_against_bundle(self, bundle: FormalSignalReadoutBundle) -> None:
        if not isinstance(bundle, FormalSignalReadoutBundle):
            raise PreOracleSignalError("query-bank validation requires a readout bundle")
        if self.join_digest != sha256_json(self._payload_without_digest()):
            raise PreOracleSignalError("formal query-bank join drifted after review")
        dynamics_join = bundle.dynamics_public_query_join
        if (
            self.formal_signal_readout_plan_digest != bundle.plan.plan_digest
            or self.selected_work_key
            not in set(bundle.plan.prefix_work_keys)
            & set(bundle.plan.dynamics_work_keys)
            or self.public_query_plan_digest != bundle.public_query_plan.plan_digest
            or self.query_alias_manifest_digest
            != bundle.public_query_plan.query_alias_manifest_digest
            or self.dynamics_public_query_join_digest != dynamics_join.join_digest
            or self.dynamics_axis_registry_digest
            != bundle.dynamics_axis_registry.registry_digest
            or set(self.query_bank_id_by_opaque_query_id)
            != set(bundle.public_query_plan.opaque_query_ids)
        ):
            raise PreOracleSignalError(
                "formal query-bank join differs from the reviewed readout bundle"
            )
        maximum = bundle.dynamics_diagnostics[self.selected_work_key]
        diagnostic_by_bank = {
            row.query_bank_id: row for row in maximum.query_diagnostics
        }
        if set(diagnostic_by_bank) != set(
            self.query_bank_id_by_opaque_query_id.values()
        ):
            raise PreOracleSignalError(
                "private query-bank mapping differs from atlas diagnostics"
            )
        for opaque_id, bank_id in self.query_bank_id_by_opaque_query_id.items():
            if diagnostic_by_bank[bank_id].query_dynamics_context_id != (
                dynamics_join.dynamics_context_by_opaque_query_id[opaque_id]
            ):
                raise PreOracleSignalError(
                    "private query-bank mapping changes opaque-query dynamics identity"
                )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "formal_signal_readout_plan_digest": (
                self.formal_signal_readout_plan_digest
            ),
            "selected_work_key": self.selected_work_key,
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "dynamics_public_query_join_digest": (
                self.dynamics_public_query_join_digest
            ),
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "query_bank_id_by_opaque_query_id": dict(
                self.query_bank_id_by_opaque_query_id
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "join_digest": self.join_digest}

    def to_public_dict(self) -> dict[str, Any]:
        payload = {
            "schema": PUBLIC_SIGNAL_QUERY_BANK_JOIN_SCHEMA,
            "formal_signal_readout_plan_digest": (
                self.formal_signal_readout_plan_digest
            ),
            "selected_work_key": self.selected_work_key,
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "dynamics_public_query_join_digest": (
                self.dynamics_public_query_join_digest
            ),
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "query_count": len(self.query_bank_id_by_opaque_query_id),
            "private_query_bank_aliases_withheld": True,
            "private_join_digest": self.join_digest,
        }
        return {**payload, "public_projection_digest": sha256_json(payload)}


@dataclass(frozen=True)
class FormalSignalExtractionPlan:
    formal_signal_readout_plan_digest: str
    selected_work_key: str
    formal_query_bank_alias_join_digest: str
    selection_review_evidence_digest: str
    review_decisions_digest: str
    review_authority_receipt_digest: str
    signal_prefix_schedule_digest: str
    dynamics_axis_registry_digest: str
    public_query_plan_digest: str
    query_alias_manifest_digest: str
    signal_metric_id: str = REGISTERED_SIGNAL_METRIC_ID
    plan_digest: str | None = None
    schema: str = FORMAL_SIGNAL_EXTRACTION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FORMAL_SIGNAL_EXTRACTION_PLAN_SCHEMA:
            raise PreOracleSignalError("unsupported FormalSignalExtractionPlan schema")
        for name in (
            "formal_signal_readout_plan_digest",
            "formal_query_bank_alias_join_digest",
            "selection_review_evidence_digest",
            "review_decisions_digest",
            "review_authority_receipt_digest",
            "signal_prefix_schedule_digest",
            "dynamics_axis_registry_digest",
            "public_query_plan_digest",
            "query_alias_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.selected_work_key, str) or not self.selected_work_key:
            raise PreOracleSignalError("selected_work_key must be non-empty")
        if self.signal_metric_id != REGISTERED_SIGNAL_METRIC_ID:
            raise PreOracleSignalError(
                "formal pre-oracle metric is fixed as dynamics_neighborhood_top1"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.plan_digest is None:
            object.__setattr__(self, "plan_digest", expected)
        elif _digest(self.plan_digest, "plan_digest") != expected:
            raise PreOracleSignalError("signal extraction plan digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        readout_plan: FormalSignalReadoutPlan,
        selected_work_key: str,
        query_bank_alias_join: FormalSignalQueryBankJoin,
        selection_review_evidence_digest: str,
        review_authority_receipt_digest: str,
    ) -> "FormalSignalExtractionPlan":
        if not isinstance(readout_plan, FormalSignalReadoutPlan):
            raise PreOracleSignalError("extraction requires a typed readout plan")
        if selected_work_key not in (
            set(readout_plan.prefix_work_keys)
            & set(readout_plan.dynamics_work_keys)
        ):
            raise PreOracleSignalError(
                "selected work key must be in both reviewed prefix and dynamics sets"
            )
        if not isinstance(query_bank_alias_join, FormalSignalQueryBankJoin):
            raise PreOracleSignalError(
                "extraction requires a typed private query-bank join"
            )
        if (
            query_bank_alias_join.formal_signal_readout_plan_digest
            != readout_plan.plan_digest
            or query_bank_alias_join.selected_work_key != selected_work_key
            or query_bank_alias_join.public_query_plan_digest
            != readout_plan.public_query_plan_digest
            or query_bank_alias_join.dynamics_axis_registry_digest
            != readout_plan.dynamics_axis_registry_digest
        ):
            raise PreOracleSignalError(
                "private query-bank join differs from extraction selection"
            )
        return cls(
            formal_signal_readout_plan_digest=str(readout_plan.plan_digest),
            selected_work_key=selected_work_key,
            formal_query_bank_alias_join_digest=str(
                query_bank_alias_join.join_digest
            ),
            selection_review_evidence_digest=selection_review_evidence_digest,
            review_decisions_digest=readout_plan.review_decisions_digest,
            review_authority_receipt_digest=review_authority_receipt_digest,
            signal_prefix_schedule_digest=(
                readout_plan.formal_signal_prefix_schedule_digest
            ),
            dynamics_axis_registry_digest=(
                readout_plan.dynamics_axis_registry_digest
            ),
            public_query_plan_digest=readout_plan.public_query_plan_digest,
            query_alias_manifest_digest=(
                query_bank_alias_join.query_alias_manifest_digest
            ),
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "formal_signal_readout_plan_digest": (
                self.formal_signal_readout_plan_digest
            ),
            "selected_work_key": self.selected_work_key,
            "formal_query_bank_alias_join_digest": (
                self.formal_query_bank_alias_join_digest
            ),
            "selection_review_evidence_digest": (
                self.selection_review_evidence_digest
            ),
            "review_decisions_digest": self.review_decisions_digest,
            "review_authority_receipt_digest": (
                self.review_authority_receipt_digest
            ),
            "signal_prefix_schedule_digest": self.signal_prefix_schedule_digest,
            "dynamics_axis_registry_digest": self.dynamics_axis_registry_digest,
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "signal_metric_id": self.signal_metric_id,
            "metric_definition": (
                "per-query DynamicsQueryDiagnostic.neighborhood_top1 rebuilt "
                "from every formal prefix point"
            ),
            "caller_supplied_numeric_values": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "plan_digest": self.plan_digest}


def _validate_plan_bundle_join(
    plan: FormalSignalExtractionPlan,
    bundle: FormalSignalReadoutBundle,
    query_bank_alias_join: FormalSignalQueryBankJoin,
) -> None:
    if not isinstance(plan, FormalSignalExtractionPlan):
        raise PreOracleSignalError("builder requires a typed extraction plan")
    if not isinstance(bundle, FormalSignalReadoutBundle):
        raise PreOracleSignalError("builder requires a typed readout bundle")
    if not isinstance(query_bank_alias_join, FormalSignalQueryBankJoin):
        raise PreOracleSignalError("builder requires a typed private query-bank join")
    freeze = bundle.atlas_run.formal_authorization.freeze_manifest
    join = bundle.dynamics_public_query_join
    try:
        join.validate(
            public_query_plan=bundle.public_query_plan,
            registry=bundle.dynamics_axis_registry,
        )
    except Exception as error:
        raise PreOracleSignalError(
            "private query mapping differs from its reviewed typed join"
        ) from error
    query_bank_alias_join.validate_against_bundle(bundle)
    if plan.plan_digest != freeze.preoracle_signal_outcome_plan_digest:
        raise PreOracleSignalError(
            "signal extraction plan differs from the externally frozen plan"
        )
    if (
        plan.formal_signal_readout_plan_digest != bundle.plan.plan_digest
        or plan.formal_query_bank_alias_join_digest
        != query_bank_alias_join.join_digest
        or plan.selected_work_key != query_bank_alias_join.selected_work_key
        or plan.selected_work_key
        not in set(bundle.plan.prefix_work_keys) & set(bundle.plan.dynamics_work_keys)
        or plan.review_decisions_digest != bundle.plan.review_decisions_digest
        or plan.review_decisions_digest != freeze.review_decisions_digest
        or plan.review_authority_receipt_digest
        != freeze.review_authority_receipt_digest
        or plan.signal_prefix_schedule_digest
        != bundle.plan.formal_signal_prefix_schedule_digest
        or plan.dynamics_axis_registry_digest
        != bundle.plan.dynamics_axis_registry_digest
        or plan.public_query_plan_digest != bundle.plan.public_query_plan_digest
        or plan.public_query_plan_digest != join.public_query_plan_digest
        or plan.query_alias_manifest_digest != join.query_alias_manifest_digest
        or plan.query_alias_manifest_digest
        != bundle.public_query_plan.query_alias_manifest_digest
    ):
        raise PreOracleSignalError(
            "signal extraction plan differs from the reviewed readout bundle"
        )


def _rebuild_prefix_diagnostics(
    plan: FormalSignalExtractionPlan,
    bundle: FormalSignalReadoutBundle,
) -> Mapping[int, DynamicsAxisDiagnostics]:
    prefix_run = bundle.prefix_runs[plan.selected_work_key]
    authorization = bundle.atlas_run.formal_authorization
    diagnostics: dict[int, DynamicsAxisDiagnostics] = {}
    for point in prefix_run.points:
        rebuilt = build_dynamics_axis_diagnostics(
            metric_record=point.metric_record,
            registry=bundle.dynamics_axis_registry,
            execution_mode=FORMAL_MODE,
            signal_plan_digest=bundle.plan.signal_matrix_digest,
            signal_execution_protocol_digest=(
                bundle.plan.signal_execution_protocol_digest
            ),
            identity_registry_digest=bundle.plan.signal_identity_registry_digest,
            formal_authorization=authorization,
        )
        diagnostics[point.prefix_episode_count] = rebuilt
    if tuple(diagnostics) != FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS:
        raise PreOracleSignalError(
            "derived dynamics diagnostics require exact 1/2/4/8/16/32/64 prefixes"
        )
    maximum = diagnostics[FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS[-1]]
    frozen_maximum = bundle.dynamics_diagnostics[plan.selected_work_key]
    if maximum.to_dict() != frozen_maximum.to_dict():
        raise PreOracleSignalError(
            "maximum-prefix dynamics diagnostics differ from the atlas readout"
        )
    return MappingProxyType(diagnostics)


def _derive_manifest(
    *,
    run_id: str,
    plan: FormalSignalExtractionPlan,
    bundle: FormalSignalReadoutBundle,
    query_bank_alias_join: FormalSignalQueryBankJoin,
) -> tuple[SignalOutcomeManifest, Mapping[int, str]]:
    canonical_run_id = _safe_id(run_id, "run_id")
    _validate_plan_bundle_join(plan, bundle, query_bank_alias_join)
    diagnostics = _rebuild_prefix_diagnostics(plan, bundle)
    join = bundle.dynamics_public_query_join
    expected_queries = tuple(bundle.public_query_plan.opaque_query_ids)
    if len(expected_queries) != 66 or set(
        join.dynamics_context_by_opaque_query_id
    ) != set(expected_queries):
        raise PreOracleSignalError("pre-oracle derivation requires exact 66-query join")

    expected_banks = set(
        query_bank_alias_join.query_bank_id_by_opaque_query_id.values()
    )
    by_prefix: dict[int, dict[str, Any]] = {}
    for prefix, diagnostic in diagnostics.items():
        diagnostic_by_bank = {
            item.query_bank_id: item for item in diagnostic.query_diagnostics
        }
        if set(diagnostic_by_bank) != expected_banks:
            raise PreOracleSignalError(
                "prefix dynamics diagnostics differ from exact private query-bank set"
            )
        by_opaque_query = {}
        for query_id in expected_queries:
            bank_id = query_bank_alias_join.query_bank_id_by_opaque_query_id[
                query_id
            ]
            item = diagnostic_by_bank[bank_id]
            if (
                item.query_dynamics_context_id
                != join.dynamics_context_by_opaque_query_id[query_id]
            ):
                raise PreOracleSignalError(
                    "prefix query dynamics identity differs from private query join"
                )
            by_opaque_query[query_id] = item
        by_prefix[prefix] = by_opaque_query

    freeze = bundle.atlas_run.formal_authorization.freeze_manifest
    prefix_digests = {
        prefix: str(diagnostic.diagnostics_digest)
        for prefix, diagnostic in diagnostics.items()
    }
    rows = []
    for query_id in expected_queries:
        context_id = join.dynamics_context_by_opaque_query_id[query_id]
        entry = bundle.dynamics_axis_registry.entry(context_id)
        values = {
            prefix: float(by_prefix[prefix][query_id].neighborhood_top1)
            for prefix in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
        }
        evidence_digest = sha256_json(
            {
                "schema": "policy-learnware.v03-derived-signal-row-evidence.v0",
                "signal_extraction_plan_digest": plan.plan_digest,
                "selection_review_evidence_digest": (
                    plan.selection_review_evidence_digest
                ),
                "formal_signal_readout_bundle_digest": bundle.bundle_digest,
                "formal_signal_readout_plan_digest": bundle.plan.plan_digest,
                "freeze_manifest_digest": freeze.freeze_manifest_digest,
                "formal_signal_atlas_authorization_digest": (
                    bundle.formal_authorization_digest
                ),
                "atlas_run_digest": bundle.atlas_run.run_digest,
                "selected_work_key": plan.selected_work_key,
                "public_query_plan_digest": bundle.public_query_plan.plan_digest,
                "query_alias_manifest_digest": join.query_alias_manifest_digest,
                "dynamics_public_query_join_digest": join.join_digest,
                "formal_query_bank_alias_join_digest": (
                    query_bank_alias_join.join_digest
                ),
                "dynamics_axis_registry_digest": (
                    bundle.dynamics_axis_registry.registry_digest
                ),
                "signal_prefix_schedule_digest": (
                    plan.signal_prefix_schedule_digest
                ),
                "prefix_diagnostics_digests": {
                    str(prefix): prefix_digests[prefix]
                    for prefix in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
                },
                "prefix_query_diagnostic_digests": {
                    str(prefix): by_prefix[prefix][query_id].diagnostic_digest
                    for prefix in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
                },
                "opaque_query_id": query_id,
                "private_query_bank_id": (
                    query_bank_alias_join.query_bank_id_by_opaque_query_id[
                        query_id
                    ]
                ),
                "task_id": entry.task_id,
                "axis_id": entry.axis_id,
                "context_id": context_id,
                "signal_metric_id": REGISTERED_SIGNAL_METRIC_ID,
                "prefix_signal_values": {
                    str(prefix): values[prefix]
                    for prefix in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
                },
                "max_prefix_equality": True,
            }
        )
        rows.append(
            SignalOutcomeRow(
                opaque_query_id=query_id,
                task_id=entry.task_id,
                axis_id=entry.axis_id,
                context_id=context_id,
                signal_metric_id=REGISTERED_SIGNAL_METRIC_ID,
                signal_value=values[FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS[-1]],
                prefix_signal_values=values,
                signal_evidence_digest=evidence_digest,
            )
        )
    manifest = SignalOutcomeManifest(
        run_id=canonical_run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        public_query_plan_digest=str(bundle.public_query_plan.plan_digest),
        query_alias_manifest_digest=join.query_alias_manifest_digest,
        signal_atlas_digest=str(bundle.atlas_run.run_digest),
        signal_prefix_schedule_digest=plan.signal_prefix_schedule_digest,
        rows=tuple(rows),
    )
    return manifest, MappingProxyType(prefix_digests)


@dataclass(frozen=True)
class PreOracleSignalOutcome:
    run_id: str
    extraction_plan: FormalSignalExtractionPlan
    query_bank_alias_join: FormalSignalQueryBankJoin
    readout_bundle: FormalSignalReadoutBundle
    signal_outcome_manifest: SignalOutcomeManifest
    prefix_diagnostics_digests: Mapping[int, str]
    outcome_digest: str | None = None
    schema: str = PREORACLE_SIGNAL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREORACLE_SIGNAL_OUTCOME_SCHEMA:
            raise PreOracleSignalError("unsupported PreOracleSignalOutcome schema")
        if not isinstance(self.extraction_plan, FormalSignalExtractionPlan):
            raise PreOracleSignalError("pre-oracle outcome requires an extraction plan")
        if not isinstance(self.query_bank_alias_join, FormalSignalQueryBankJoin):
            raise PreOracleSignalError(
                "pre-oracle outcome requires a private query-bank join"
            )
        if not isinstance(self.readout_bundle, FormalSignalReadoutBundle):
            raise PreOracleSignalError("pre-oracle outcome requires a readout bundle")
        if not isinstance(self.signal_outcome_manifest, SignalOutcomeManifest):
            raise PreOracleSignalError("pre-oracle outcome requires a signal manifest")
        expected_manifest, expected_prefixes = _derive_manifest(
            run_id=self.run_id,
            plan=self.extraction_plan,
            bundle=self.readout_bundle,
            query_bank_alias_join=self.query_bank_alias_join,
        )
        supplied_prefixes = {
            int(prefix): _digest(value, f"prefix_diagnostics_digests[{prefix}]")
            for prefix, value in sorted(self.prefix_diagnostics_digests.items())
        }
        if (
            self.signal_outcome_manifest.to_dict() != expected_manifest.to_dict()
            or supplied_prefixes != dict(expected_prefixes)
        ):
            raise PreOracleSignalError(
                "signal outcome rows/values differ from pure bundle derivation"
            )
        object.__setattr__(self, "run_id", _safe_id(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "prefix_diagnostics_digests",
            MappingProxyType(supplied_prefixes),
        )
        expected = sha256_json(self._payload_without_digest())
        if self.outcome_digest is None:
            object.__setattr__(self, "outcome_digest", expected)
        elif _digest(self.outcome_digest, "outcome_digest") != expected:
            raise PreOracleSignalError("pre-oracle signal outcome digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "signal_extraction_plan_digest": self.extraction_plan.plan_digest,
            "formal_query_bank_alias_join_digest": (
                self.query_bank_alias_join.join_digest
            ),
            "formal_signal_readout_bundle_digest": self.readout_bundle.bundle_digest,
            "freeze_manifest_digest": (
                self.readout_bundle.atlas_run.formal_authorization.freeze_manifest.freeze_manifest_digest
            ),
            "signal_outcome_manifest_digest": (
                self.signal_outcome_manifest.manifest_digest
            ),
            "prefix_diagnostics_digests": {
                str(prefix): digest
                for prefix, digest in self.prefix_diagnostics_digests.items()
            },
            "oracle_data_accessed": False,
            "caller_supplied_numeric_values": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "outcome_digest": self.outcome_digest}


@dataclass(frozen=True)
class PreOracleSignalOutcomePublication:
    """Immutable full-row receipt for the pre-oracle signal derivation."""

    run_id: str
    signal_extraction_plan_digest: str
    formal_query_bank_alias_join_digest: str
    formal_signal_readout_bundle_digest: str
    preoracle_signal_outcome_digest: str
    freeze_manifest_digest: str
    public_query_plan_digest: str
    query_alias_manifest_digest: str
    signal_atlas_digest: str
    signal_prefix_schedule_digest: str
    signal_outcome_manifest_digest: str
    prefix_diagnostics_digests: Mapping[int, str]
    signal_outcome_manifest: SignalOutcomeManifest
    oracle_data_accessed: bool = False
    publication_digest: str | None = None
    schema: str = PREORACLE_SIGNAL_OUTCOME_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PREORACLE_SIGNAL_OUTCOME_PUBLICATION_SCHEMA:
            raise PreOracleSignalError(
                "unsupported PreOracleSignalOutcomePublication schema"
            )
        object.__setattr__(self, "run_id", _safe_id(self.run_id, "run_id"))
        for name in (
            "signal_extraction_plan_digest",
            "formal_query_bank_alias_join_digest",
            "formal_signal_readout_bundle_digest",
            "preoracle_signal_outcome_digest",
            "freeze_manifest_digest",
            "public_query_plan_digest",
            "query_alias_manifest_digest",
            "signal_atlas_digest",
            "signal_prefix_schedule_digest",
            "signal_outcome_manifest_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.oracle_data_accessed is not False:
            raise PreOracleSignalError(
                "pre-oracle publication cannot report oracle access"
            )
        if not isinstance(self.signal_outcome_manifest, SignalOutcomeManifest):
            raise PreOracleSignalError(
                "publication requires a typed full SignalOutcomeManifest"
            )
        prefixes = {
            int(prefix): _digest(value, f"prefix_diagnostics_digests[{prefix}]")
            for prefix, value in sorted(self.prefix_diagnostics_digests.items())
        }
        if tuple(prefixes) != FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS:
            raise PreOracleSignalError(
                "publication requires exact 1/2/4/8/16/32/64 diagnostics"
            )
        object.__setattr__(
            self, "prefix_diagnostics_digests", MappingProxyType(prefixes)
        )
        manifest = self.signal_outcome_manifest
        if (
            self.signal_outcome_manifest_digest != manifest.manifest_digest
            or self.run_id != manifest.run_id
            or self.freeze_manifest_digest != manifest.freeze_manifest_digest
            or self.public_query_plan_digest != manifest.public_query_plan_digest
            or self.query_alias_manifest_digest
            != manifest.query_alias_manifest_digest
            or self.signal_atlas_digest != manifest.signal_atlas_digest
            or self.signal_prefix_schedule_digest
            != manifest.signal_prefix_schedule_digest
        ):
            raise PreOracleSignalError(
                "publication provenance differs from its full signal manifest"
            )
        expected = sha256_json(self._payload_without_digest())
        if self.publication_digest is None:
            object.__setattr__(self, "publication_digest", expected)
        elif _digest(self.publication_digest, "publication_digest") != expected:
            raise PreOracleSignalError("pre-oracle publication digest mismatch")

    @classmethod
    def from_outcome(
        cls, outcome: PreOracleSignalOutcome
    ) -> "PreOracleSignalOutcomePublication":
        if not isinstance(outcome, PreOracleSignalOutcome):
            raise PreOracleSignalError(
                "publication requires a typed pre-oracle outcome"
            )
        manifest = outcome.signal_outcome_manifest
        return cls(
            run_id=outcome.run_id,
            signal_extraction_plan_digest=str(outcome.extraction_plan.plan_digest),
            formal_query_bank_alias_join_digest=str(
                outcome.query_bank_alias_join.join_digest
            ),
            formal_signal_readout_bundle_digest=str(
                outcome.readout_bundle.bundle_digest
            ),
            preoracle_signal_outcome_digest=str(outcome.outcome_digest),
            freeze_manifest_digest=manifest.freeze_manifest_digest,
            public_query_plan_digest=manifest.public_query_plan_digest,
            query_alias_manifest_digest=manifest.query_alias_manifest_digest,
            signal_atlas_digest=manifest.signal_atlas_digest,
            signal_prefix_schedule_digest=manifest.signal_prefix_schedule_digest,
            signal_outcome_manifest_digest=str(manifest.manifest_digest),
            prefix_diagnostics_digests=outcome.prefix_diagnostics_digests,
            signal_outcome_manifest=manifest,
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "signal_extraction_plan_digest": self.signal_extraction_plan_digest,
            "formal_query_bank_alias_join_digest": (
                self.formal_query_bank_alias_join_digest
            ),
            "formal_signal_readout_bundle_digest": (
                self.formal_signal_readout_bundle_digest
            ),
            "preoracle_signal_outcome_digest": (
                self.preoracle_signal_outcome_digest
            ),
            "freeze_manifest_digest": self.freeze_manifest_digest,
            "public_query_plan_digest": self.public_query_plan_digest,
            "query_alias_manifest_digest": self.query_alias_manifest_digest,
            "signal_atlas_digest": self.signal_atlas_digest,
            "signal_prefix_schedule_digest": self.signal_prefix_schedule_digest,
            "signal_outcome_manifest_digest": self.signal_outcome_manifest_digest,
            "prefix_diagnostics_digests": {
                str(prefix): digest
                for prefix, digest in self.prefix_diagnostics_digests.items()
            },
            "signal_outcome_manifest": self.signal_outcome_manifest.to_dict(),
            "oracle_data_accessed": self.oracle_data_accessed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "publication_digest": self.publication_digest,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "PreOracleSignalOutcomePublication":
        if not isinstance(value, Mapping):
            raise PreOracleSignalError("pre-oracle publication must be a mapping")
        fields = set(cls.__dataclass_fields__)
        if set(value) != fields:
            missing = sorted(fields - set(value))
            extra = sorted(set(value) - fields)
            raise PreOracleSignalError(
                f"pre-oracle publication fields differ; missing={missing}, extra={extra}"
            )
        prefixes = value["prefix_diagnostics_digests"]
        expected_prefix_keys = {
            str(prefix) for prefix in FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
        }
        if not isinstance(prefixes, Mapping) or set(prefixes) != expected_prefix_keys:
            raise PreOracleSignalError(
                "serialized publication requires exact string prefix keys"
            )
        manifest = value["signal_outcome_manifest"]
        if not isinstance(manifest, Mapping):
            raise PreOracleSignalError(
                "serialized publication requires a signal manifest mapping"
            )
        _digest(value["publication_digest"], "publication_digest")
        _digest(
            manifest.get("manifest_digest"),
            "signal_outcome_manifest.manifest_digest",
        )
        return cls(
            **{
                field: (
                    {int(prefix): digest for prefix, digest in prefixes.items()}
                    if field == "prefix_diagnostics_digests"
                    else SignalOutcomeManifest.from_dict(manifest)
                    if field == "signal_outcome_manifest"
                    else value[field]
                )
                for field in fields
            }
        )


def build_preoracle_signal_outcome(
    *,
    run_id: str,
    extraction_plan: FormalSignalExtractionPlan,
    query_bank_alias_join: FormalSignalQueryBankJoin,
    readout_bundle: FormalSignalReadoutBundle,
) -> PreOracleSignalOutcome:
    """Derive the exact 66-row manifest without accepting any numeric values."""

    manifest, prefix_digests = _derive_manifest(
        run_id=run_id,
        plan=extraction_plan,
        bundle=readout_bundle,
        query_bank_alias_join=query_bank_alias_join,
    )
    return PreOracleSignalOutcome(
        run_id=run_id,
        extraction_plan=extraction_plan,
        query_bank_alias_join=query_bank_alias_join,
        readout_bundle=readout_bundle,
        signal_outcome_manifest=manifest,
        prefix_diagnostics_digests=prefix_digests,
    )


__all__ = [
    "FORMAL_SIGNAL_EXTRACTION_PLAN_SCHEMA",
    "FORMAL_SIGNAL_QUERY_BANK_JOIN_SCHEMA",
    "PUBLIC_SIGNAL_QUERY_BANK_JOIN_SCHEMA",
    "PREORACLE_SIGNAL_OUTCOME_SCHEMA",
    "PREORACLE_SIGNAL_OUTCOME_PUBLICATION_SCHEMA",
    "REGISTERED_SIGNAL_METRIC_ID",
    "FormalSignalExtractionPlan",
    "FormalSignalQueryBankJoin",
    "PreOracleSignalError",
    "PreOracleSignalOutcome",
    "PreOracleSignalOutcomePublication",
    "build_preoracle_signal_outcome",
]
