"""Native v0.3 simple-baseline contracts and public full-pool ranking.

The v0.2 baseline implementations assume that both source and query are
reduced ``EnvironmentSpec`` objects.  v0.3 deliberately uses an empirical
query KME against reduced source KMEs, so its public ranking layer must consume
the already-audited :class:`~policy_learnware_v0.v03.compute.JointDistanceRun`
instead of silently reducing the query.

This module is dependency-light and has no artifact, CLI, environment, policy
execution, or oracle writer.  Development returns enter only the two explicitly
supervised B4 fits.  Public ranking accepts only the anonymous selector view,
frozen selector artifacts, query evidence, and (where needed) public distance
runs.  Deployment-private state is inspected by one separate rank-one ABI
audit, which never falls back to rank two.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json
from ..rkme.empirical import episode_balanced_weights
from ..rkme.gaussian import GaussianKernel
from ..v02.baselines import DevelopmentView
from ..v02.representation import TraceFeatureVector
from ..v02.schemas import ExecutionABIRecord
from ..v02.selectors import EvidenceContract
from .anonymous_market import AnonymousSelectorViewManifest
from .compute import JointDistanceRun
from .contracts import MarketBoundSourceRepresentationIndex
from .pool_intake import EXPECTED_ANCHOR_COUNT
from .source_market import V03SourcePolicyMarket


BASELINE_QUERY_SCHEMA = "policy-learnware.v03-baseline-query.v0"
BASELINE_ARTIFACT_SCHEMA = "policy-learnware.v03-baseline-selector-artifact.v0"
BASELINE_FREEZE_SCHEMA = "policy-learnware.v03-baseline-development-freeze.v0"
FULL_RANKING_SCHEMA = "policy-learnware.v03-published-full-ranking.v0"
RANK_ONE_ABI_AUDIT_SCHEMA = "policy-learnware.v03-baseline-rank-one-abi-audit.v0"
SOURCE_SIGMA_SCHEMA = "policy-learnware.v03-source-only-sigma.v0"
QUERY_ALIAS_SCHEMA = "policy-learnware.v03-v02-v03-query-alias-manifest.v0"
RAW_MOMENT_PROTOCOL_SCHEMA = "policy-learnware.v03-raw-moment-feature-protocol.v0"

REQUIRED_BASELINE_METHOD_IDS = (
    "B0",
    "B1",
    "B2",
    "B3a",
    "B3b",
    "B4a",
    "B4b",
    "A-Env",
    "M02/B5",
)
OPTIONAL_BASELINE_STATES = MappingProxyType({"B4c": "DISABLED", "B6": "DISABLED"})
BASELINE_METHOD_KINDS = MappingProxyType(
    {
        "B0": "random",
        "B1": "competence_only",
        "B2": "legacy_taskspec_mapping",
        "B3a": "raw_moments_nearest",
        "B3b": "nearest_joint_distance",
        "B4a": "knn_development",
        "B4b": "linear_development",
        "A-Env": "nearest_joint_distance",
        "M02/B5": "lmin_joint_distance",
    }
)
FORMAL_DEVELOPMENT_CONTEXT_COUNT = 24
NO_FALLBACK_POLICY = "RANK_ONE_ONLY_NO_FALLBACK"

BaselineExecutionMode = Literal["DEVELOPMENT_SMOKE", "FORMAL"]
DEVELOPMENT_SMOKE_MODE: BaselineExecutionMode = "DEVELOPMENT_SMOKE"
FORMAL_MODE: BaselineExecutionMode = "FORMAL"

_OPAQUE_LEARNWARE_ID = re.compile(r"^lw-[0-9a-f]{32}$")
_V02_QUERY_ID = re.compile(r"^v02q-[0-9a-f]{32}$")
_V03_QUERY_ID = re.compile(r"^v03q-[0-9a-f]{32}$")


class V03BaselineError(ValueError):
    """A baseline crossed an evidence boundary or broke a frozen binding."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V03BaselineError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result != result.lower():
        raise V03BaselineError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise V03BaselineError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V03BaselineError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V03BaselineError(f"{where} must be finite")
    return result


def _positive(value: Any, where: str) -> float:
    result = _finite(value, where)
    if result <= 0.0:
        raise V03BaselineError(f"{where} must be positive")
    return result


def _deep_freeze(value: Any) -> Any:
    try:
        normalized = canonicalize(value)
    except (TypeError, ValueError) as error:
        raise V03BaselineError(f"payload is not canonical JSON: {error}") from error
    if isinstance(normalized, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_deep_freeze(item) for item in normalized)
    return normalized


def _canonical_pool_ids(values: Sequence[str] | Mapping[str, Any], where: str) -> tuple[str, ...]:
    ids = tuple(values)
    if len(ids) != EXPECTED_ANCHOR_COUNT or len(set(ids)) != EXPECTED_ANCHOR_COUNT:
        raise V03BaselineError(
            f"{where} must contain exactly {EXPECTED_ANCHOR_COUNT} unique IDs"
        )
    if any(not isinstance(item, str) or _OPAQUE_LEARNWARE_ID.fullmatch(item) is None for item in ids):
        raise V03BaselineError(f"{where} contains a non-canonical learnware ID")
    return tuple(sorted(ids))


def _query_id(value: Any, *, version: str = "v03") -> str:
    pattern = _V03_QUERY_ID if version == "v03" else _V02_QUERY_ID
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise V03BaselineError(f"{version} query ID has invalid canonical format")
    return value


def evidence_contract_digest(contract: EvidenceContract) -> str:
    if not isinstance(contract, EvidenceContract):
        raise V03BaselineError("evidence contract has the wrong type")
    contract.require_public_selector_safe()
    return sha256_json(contract.to_dict())


PUBLIC_NO_QUERY_EVIDENCE = EvidenceContract(
    reads_source_raw_data=False,
    reads_development_policy_returns=False,
    reads_target_parameters=False,
    reads_target_transitions=False,
    reads_candidate_independent_probe_rewards=False,
    reads_candidate_target_rollouts=False,
    reads_candidate_policy_target_rewards=False,
    target_gradient_updates=0,
    reads_submit_side_profiles=False,
)


def public_probe_evidence_contract(
    *, development_supervised: bool = False, source_side_labels: bool = False
) -> EvidenceContract:
    """Return the exact public evidence card used by a v0.3 probe baseline."""

    if type(development_supervised) is not bool or type(source_side_labels) is not bool:
        raise V03BaselineError("evidence capability flags must be boolean")
    return EvidenceContract(
        reads_source_raw_data=False,
        reads_development_policy_returns=development_supervised,
        reads_target_parameters=False,
        reads_target_transitions=True,
        reads_candidate_independent_probe_rewards=False,
        reads_candidate_target_rollouts=False,
        reads_candidate_policy_target_rewards=False,
        target_gradient_updates=0,
        reads_submit_side_profiles=False,
        reads_source_side_labels=source_side_labels,
        reads_target_task_reward_schema_identity=False,
    )


RawMomentStatistic = Literal["mean", "std", "second_moment"]
RawMomentWeighting = Literal["transition_uniform", "episode_balanced"]


@dataclass(frozen=True)
class RawMomentFeatureProtocol:
    """Explicit B3a feature definition for one frozen transition view."""

    transition_view_spec_digest: str
    statistics: tuple[RawMomentStatistic, ...]
    weighting: RawMomentWeighting
    schema: str = RAW_MOMENT_PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RAW_MOMENT_PROTOCOL_SCHEMA:
            raise V03BaselineError("unsupported raw-moment protocol schema")
        object.__setattr__(
            self,
            "transition_view_spec_digest",
            _digest(self.transition_view_spec_digest, "transition_view_spec_digest"),
        )
        statistics = tuple(self.statistics)
        allowed = {"mean", "std", "second_moment"}
        if not statistics or len(set(statistics)) != len(statistics) or not set(statistics) <= allowed:
            raise V03BaselineError("raw-moment statistics are invalid")
        if self.weighting not in {"transition_uniform", "episode_balanced"}:
            raise V03BaselineError("raw-moment weighting is invalid")
        object.__setattr__(self, "statistics", statistics)

    @property
    def feature_protocol_id(self) -> str:
        return sha256_json(
            {
                "schema": self.schema,
                "transition_view_spec_digest": self.transition_view_spec_digest,
                "statistics": list(self.statistics),
                "weighting": self.weighting,
            }
        )


def raw_moment_feature_from_view(
    view: Any,
    protocol: RawMomentFeatureProtocol,
    *,
    probe_dataset_digest: str,
) -> TraceFeatureVector:
    """Build B3a moments directly from a v0.3 ``TransitionViewResult``."""

    # Local import keeps transition attribution out of dependency-light B0/B1
    # users while still providing an explicit v0.3 bridge for B3a.
    from .transition_views import TransitionViewResult

    if not isinstance(view, TransitionViewResult):
        raise V03BaselineError("raw moments require a TransitionViewResult")
    if not isinstance(protocol, RawMomentFeatureProtocol):
        raise V03BaselineError("raw-moment protocol has the wrong type")
    if view.spec.digest != protocol.transition_view_spec_digest:
        raise V03BaselineError("transition view differs from the raw-moment protocol")
    dataset_digest = _digest(probe_dataset_digest, "probe_dataset_digest")
    points = np.asarray(view.feature_matrix, dtype=np.float64)
    if protocol.weighting == "transition_uniform":
        weights = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    else:
        weights = episode_balanced_weights(view.episode_offsets)
    mean = np.sum(points * weights[:, None], axis=0)
    second = np.sum(np.square(points) * weights[:, None], axis=0)
    variance = np.maximum(second - np.square(mean), 0.0)
    values = {
        "mean": mean,
        "std": np.sqrt(variance),
        "second_moment": second,
    }
    return TraceFeatureVector(
        values=np.concatenate([values[name] for name in protocol.statistics]),
        feature_protocol_id=protocol.feature_protocol_id,
        probe_dataset_digest=dataset_digest,
    )


@dataclass(frozen=True)
class V03BaselineQuery:
    """Public query projection shared by every simple baseline."""

    opaque_query_id: str
    query_spec_digest: str
    probe_dataset_digest: str
    target_evidence_digest: str
    cost_digest: str
    execution_mode: BaselineExecutionMode
    query_mode: Literal["QUERY_EMPIRICAL", "QUERY_REDUCED"]
    trace_feature: TraceFeatureVector | None = None
    schema: str = BASELINE_QUERY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BASELINE_QUERY_SCHEMA:
            raise V03BaselineError("unsupported baseline query schema")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        for name in (
            "query_spec_digest",
            "probe_dataset_digest",
            "target_evidence_digest",
            "cost_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise V03BaselineError("baseline query execution_mode is invalid")
        if self.query_mode not in {"QUERY_EMPIRICAL", "QUERY_REDUCED"}:
            raise V03BaselineError("baseline query_mode is invalid")
        if self.execution_mode == FORMAL_MODE and self.query_mode != "QUERY_EMPIRICAL":
            raise V03BaselineError(
                "formal baseline queries require the frozen QUERY_EMPIRICAL protocol"
            )
        if self.trace_feature is not None and not isinstance(
            self.trace_feature, TraceFeatureVector
        ):
            raise V03BaselineError("trace_feature must be a TraceFeatureVector or None")
        if (
            self.trace_feature is not None
            and self.trace_feature.probe_dataset_digest != self.probe_dataset_digest
        ):
            raise V03BaselineError(
                "trace feature and query spec must use the same probe dataset"
            )

    @property
    def trace_feature_digest(self) -> str | None:
        return None if self.trace_feature is None else self.trace_feature.digest

    @property
    def query_input_digest(self) -> str:
        return sha256_json(
            {
                "schema": self.schema,
                "opaque_query_id": self.opaque_query_id,
                "query_spec_digest": self.query_spec_digest,
                "probe_dataset_digest": self.probe_dataset_digest,
                "target_evidence_digest": self.target_evidence_digest,
                "cost_digest": self.cost_digest,
                "execution_mode": self.execution_mode,
                "query_mode": self.query_mode,
                "trace_feature_digest": self.trace_feature_digest,
            }
        )


@dataclass(frozen=True)
class V02V03QueryAliasEntry:
    v02_query_id: str
    v03_query_id: str
    context_binding_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "v02_query_id", _query_id(self.v02_query_id, version="v02"))
        object.__setattr__(self, "v03_query_id", _query_id(self.v03_query_id))
        object.__setattr__(
            self,
            "context_binding_digest",
            _digest(self.context_binding_digest, "context_binding_digest"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "v02_query_id": self.v02_query_id,
            "v03_query_id": self.v03_query_id,
            "context_binding_digest": self.context_binding_digest,
        }


@dataclass(frozen=True)
class V02V03QueryAliasManifest:
    alias_nonce_digest: str
    entries: Mapping[str, V02V03QueryAliasEntry]
    manifest_digest: str | None = None
    schema: str = QUERY_ALIAS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUERY_ALIAS_SCHEMA:
            raise V03BaselineError("unsupported query alias schema")
        nonce = _digest(self.alias_nonce_digest, "alias_nonce_digest")
        if not isinstance(self.entries, Mapping) or not self.entries:
            raise V03BaselineError("query alias manifest cannot be empty")
        entries = dict(self.entries)
        for key, entry in entries.items():
            if not isinstance(entry, V02V03QueryAliasEntry) or key != entry.v02_query_id:
                raise V03BaselineError("query alias key differs from its typed entry")
            expected_alias = "v03q-" + sha256_json(
                {
                    "schema": "policy-learnware.v03-query-alias.v0",
                    "alias_nonce_digest": nonce,
                    "v02_query_id": entry.v02_query_id,
                    "context_binding_digest": entry.context_binding_digest,
                }
            )[:32]
            if entry.v03_query_id != expected_alias:
                raise V03BaselineError("v03 query alias is not derived from its binding")
        if len({entry.v03_query_id for entry in entries.values()}) != len(entries):
            raise V03BaselineError("v03 query aliases collide")
        object.__setattr__(self, "alias_nonce_digest", nonce)
        object.__setattr__(self, "entries", MappingProxyType(dict(sorted(entries.items()))))
        expected = sha256_json(self._payload_without_digest())
        if self.manifest_digest is None:
            object.__setattr__(self, "manifest_digest", expected)
        elif _digest(self.manifest_digest, "manifest_digest") != expected:
            raise V03BaselineError("query alias manifest digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alias_nonce_digest": self.alias_nonce_digest,
            "entries": {key: entry.to_dict() for key, entry in self.entries.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "manifest_digest": self.manifest_digest}


def build_v02_v03_query_alias_manifest(
    context_bindings: Mapping[str, str], *, alias_nonce_digest: str
) -> V02V03QueryAliasManifest:
    """Create an explicit cross-version alias; silent query renaming is forbidden."""

    nonce = _digest(alias_nonce_digest, "alias_nonce_digest")
    if not isinstance(context_bindings, Mapping) or not context_bindings:
        raise V03BaselineError("context_bindings cannot be empty")
    entries: dict[str, V02V03QueryAliasEntry] = {}
    for v02_query_id, context_digest in sorted(context_bindings.items()):
        source_id = _query_id(v02_query_id, version="v02")
        binding = _digest(context_digest, "context_binding_digest")
        alias = "v03q-" + sha256_json(
            {
                "schema": "policy-learnware.v03-query-alias.v0",
                "alias_nonce_digest": nonce,
                "v02_query_id": source_id,
                "context_binding_digest": binding,
            }
        )[:32]
        entries[source_id] = V02V03QueryAliasEntry(source_id, alias, binding)
    return V02V03QueryAliasManifest(nonce, entries)


@dataclass(frozen=True)
class DevelopmentBaselineFreeze:
    """Private development fit contract; it grants no confirmatory authority."""

    policy_market_id: str
    development_view_digest: str
    split_manifest_digest: str
    label_contract_digest: str
    training_context_ids: tuple[str, ...]
    validation_context_ids: tuple[str, ...]
    label_count: int
    selector_seed: int
    b4a_neighbor_count: int
    b4b_ridge: float
    execution_mode: BaselineExecutionMode
    method_ids: tuple[str, ...] = REQUIRED_BASELINE_METHOD_IDS
    optional_method_states: Mapping[str, str] = OPTIONAL_BASELINE_STATES
    freeze_digest: str | None = None
    schema: str = BASELINE_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BASELINE_FREEZE_SCHEMA:
            raise V03BaselineError("unsupported development baseline freeze schema")
        object.__setattr__(self, "policy_market_id", _digest(self.policy_market_id, "policy_market_id"))
        for name in (
            "development_view_digest",
            "split_manifest_digest",
            "label_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        methods = tuple(self.method_ids)
        if methods != REQUIRED_BASELINE_METHOD_IDS:
            raise V03BaselineError("development freeze must bind the exact B0-B5 method registry")
        optional = dict(self.optional_method_states)
        if optional != dict(OPTIONAL_BASELINE_STATES):
            raise V03BaselineError("B4c and B6 must be explicitly DISABLED in this contract")
        if self.execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise V03BaselineError("development freeze execution_mode is invalid")
        training = tuple(_nonempty(item, "training_context_ids[]") for item in self.training_context_ids)
        validation = tuple(
            _nonempty(item, "validation_context_ids[]") for item in self.validation_context_ids
        )
        if (
            not training
            or not validation
            or len(set(training)) != len(training)
            or len(set(validation)) != len(validation)
            or set(training) & set(validation)
        ):
            raise V03BaselineError("development train/validation contexts must be disjoint and unique")
        if isinstance(self.label_count, bool) or not isinstance(self.label_count, int) or self.label_count <= 0:
            raise V03BaselineError("label_count must be a positive integer")
        if self.label_count != (len(training) + len(validation)) * EXPECTED_ANCHOR_COUNT:
            raise V03BaselineError(
                "label_count must cover every development context-policy pair"
            )
        if (
            self.execution_mode == FORMAL_MODE
            and len(training) + len(validation) != FORMAL_DEVELOPMENT_CONTEXT_COUNT
        ):
            raise V03BaselineError(
                "formal baseline freeze requires exactly 24 development contexts"
            )
        if isinstance(self.selector_seed, bool) or not isinstance(self.selector_seed, int) or self.selector_seed < 0:
            raise V03BaselineError("selector_seed must be a non-negative integer")
        if (
            isinstance(self.b4a_neighbor_count, bool)
            or not isinstance(self.b4a_neighbor_count, int)
            or not 1 <= self.b4a_neighbor_count <= len(training)
        ):
            raise V03BaselineError("B4a neighbor count exceeds the frozen training contexts")
        ridge = _positive(self.b4b_ridge, "b4b_ridge")
        object.__setattr__(self, "training_context_ids", training)
        object.__setattr__(self, "validation_context_ids", validation)
        object.__setattr__(self, "b4b_ridge", ridge)
        object.__setattr__(self, "optional_method_states", MappingProxyType(optional))
        expected = sha256_json(self._payload_without_digest())
        if self.freeze_digest is None:
            object.__setattr__(self, "freeze_digest", expected)
        elif _digest(self.freeze_digest, "freeze_digest") != expected:
            raise V03BaselineError("development baseline freeze digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority": "DEVELOPMENT_FIT_ONLY",
            "confirmatory_oracle_access": False,
            "policy_market_id": self.policy_market_id,
            "development_view_digest": self.development_view_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "label_contract_digest": self.label_contract_digest,
            "training_context_ids": list(self.training_context_ids),
            "validation_context_ids": list(self.validation_context_ids),
            "label_count": self.label_count,
            "selector_seed": self.selector_seed,
            "b4a_neighbor_count": self.b4a_neighbor_count,
            "b4b_ridge": self.b4b_ridge,
            "execution_mode": self.execution_mode,
            "method_ids": list(self.method_ids),
            "optional_method_states": dict(self.optional_method_states),
        }

    def to_private_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "freeze_digest": self.freeze_digest}

    @property
    def development_context_count(self) -> int:
        return len(self.training_context_ids) + len(self.validation_context_ids)

    @property
    def formal_24_context_matrix_ready(self) -> bool:
        return self.development_context_count == FORMAL_DEVELOPMENT_CONTEXT_COUNT

    def require_formal_24_context_matrix(self) -> None:
        if not self.formal_24_context_matrix_ready:
            raise V03BaselineError(
                "formal baseline freeze requires exactly 24 development contexts"
            )


def freeze_development_baselines(
    view: DevelopmentView,
    *,
    selector_seed: int,
    b4a_neighbor_count: int,
    b4b_ridge: float,
    execution_mode: BaselineExecutionMode,
) -> DevelopmentBaselineFreeze:
    if not isinstance(view, DevelopmentView):
        raise V03BaselineError("development freeze requires a typed DevelopmentView")
    return DevelopmentBaselineFreeze(
        policy_market_id=view.policy_market_id,
        development_view_digest=view.digest,
        split_manifest_digest=view.split_manifest_digest,
        label_contract_digest=view.label_contract_digest,
        training_context_ids=view.training_context_ids,
        validation_context_ids=view.validation_context_ids,
        label_count=view.label_count,
        selector_seed=selector_seed,
        b4a_neighbor_count=b4a_neighbor_count,
        b4b_ridge=b4b_ridge,
        execution_mode=execution_mode,
    )


FitScope = Literal["source_only", "development_supervised"]


@dataclass(frozen=True)
class FrozenBaselineSelectorArtifact:
    method_id: str
    policy_market_id: str
    representation_index_digest: str
    evidence_contract: EvidenceContract
    fit_scope: FitScope
    training_data_digest: str
    development_freeze_digest: str
    execution_mode: BaselineExecutionMode
    development_context_count: int
    payload: Mapping[str, Any]
    selector_artifact_digest: str | None = None
    schema: str = BASELINE_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BASELINE_ARTIFACT_SCHEMA:
            raise V03BaselineError("unsupported baseline artifact schema")
        if self.method_id not in REQUIRED_BASELINE_METHOD_IDS:
            raise V03BaselineError(f"unknown required baseline method {self.method_id!r}")
        for name in (
            "policy_market_id",
            "representation_index_digest",
            "training_data_digest",
            "development_freeze_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.evidence_contract, EvidenceContract):
            raise V03BaselineError("artifact evidence_contract has the wrong type")
        self.evidence_contract.require_public_selector_safe()
        if self.fit_scope not in {"source_only", "development_supervised"}:
            raise V03BaselineError("unsupported baseline fit scope")
        if self.execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise V03BaselineError("baseline artifact execution_mode is invalid")
        if (
            isinstance(self.development_context_count, bool)
            or not isinstance(self.development_context_count, int)
            or self.development_context_count <= 0
        ):
            raise V03BaselineError(
                "development_context_count must be a positive integer"
            )
        if (
            self.execution_mode == FORMAL_MODE
            and self.development_context_count != FORMAL_DEVELOPMENT_CONTEXT_COUNT
        ):
            raise V03BaselineError(
                "formal baseline artifact requires exactly 24 development contexts"
            )
        if (
            self.fit_scope == "development_supervised"
        ) != self.evidence_contract.reads_development_policy_returns:
            raise V03BaselineError("fit scope disagrees with development-label permission")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))
        observed_kind = self.payload.get("kind")
        if observed_kind != BASELINE_METHOD_KINDS[self.method_id]:
            raise V03BaselineError("baseline method ID and implementation kind disagree")
        expected_fit_scope: FitScope = (
            "development_supervised"
            if self.method_id in {"B4a", "B4b"}
            else "source_only"
        )
        if self.fit_scope != expected_fit_scope:
            raise V03BaselineError("baseline method ID and fit scope disagree")
        expected_evidence = {
            "B0": PUBLIC_NO_QUERY_EVIDENCE,
            "B1": PUBLIC_NO_QUERY_EVIDENCE,
            "B2": public_probe_evidence_contract(source_side_labels=True),
            "B3a": public_probe_evidence_contract(),
            "B3b": public_probe_evidence_contract(),
            "B4a": public_probe_evidence_contract(development_supervised=True),
            "B4b": public_probe_evidence_contract(development_supervised=True),
            "A-Env": public_probe_evidence_contract(),
            "M02/B5": public_probe_evidence_contract(),
        }[self.method_id]
        if self.evidence_contract != expected_evidence:
            raise V03BaselineError("baseline method ID and evidence contract disagree")
        expected = sha256_json(self._payload_without_digest())
        if self.selector_artifact_digest is None:
            object.__setattr__(self, "selector_artifact_digest", expected)
        elif _digest(self.selector_artifact_digest, "selector_artifact_digest") != expected:
            raise V03BaselineError("selector artifact digest mismatch")

    @property
    def evidence_contract_digest(self) -> str:
        return evidence_contract_digest(self.evidence_contract)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "policy_market_id": self.policy_market_id,
            "representation_index_digest": self.representation_index_digest,
            "evidence_contract": self.evidence_contract.to_dict(),
            "fit_scope": self.fit_scope,
            "training_data_digest": self.training_data_digest,
            "development_freeze_digest": self.development_freeze_digest,
            "execution_mode": self.execution_mode,
            "development_context_count": self.development_context_count,
            "payload": canonicalize(self.payload),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "selector_artifact_digest": self.selector_artifact_digest,
        }


@dataclass(frozen=True)
class SourceOnlySigmaArtifact:
    policy_market_id: str
    representation_index_digest: str
    source_ids: tuple[str, ...]
    source_spec_digests: tuple[str, ...]
    distance_form: Literal["mmd", "mmd2"]
    sigma: float
    artifact_digest: str | None = None
    schema: str = SOURCE_SIGMA_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_SIGMA_SCHEMA:
            raise V03BaselineError("unsupported source-only sigma schema")
        object.__setattr__(self, "policy_market_id", _digest(self.policy_market_id, "policy_market_id"))
        object.__setattr__(
            self,
            "representation_index_digest",
            _digest(self.representation_index_digest, "representation_index_digest"),
        )
        ids = _canonical_pool_ids(self.source_ids, "source-only sigma pool")
        digests = tuple(_digest(item, "source_spec_digests[]") for item in self.source_spec_digests)
        if len(digests) != EXPECTED_ANCHOR_COUNT:
            raise V03BaselineError("source spec digests must cover the exact source pool")
        if self.distance_form not in {"mmd", "mmd2"}:
            raise V03BaselineError("source sigma distance_form must be mmd or mmd2")
        object.__setattr__(self, "source_ids", ids)
        object.__setattr__(self, "source_spec_digests", digests)
        object.__setattr__(self, "sigma", _positive(self.sigma, "sigma"))
        expected = sha256_json(self._payload_without_digest())
        if self.artifact_digest is None:
            object.__setattr__(self, "artifact_digest", expected)
        elif _digest(self.artifact_digest, "artifact_digest") != expected:
            raise V03BaselineError("source-only sigma artifact digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_market_id": self.policy_market_id,
            "representation_index_digest": self.representation_index_digest,
            "source_ids": list(self.source_ids),
            "source_spec_digests": list(self.source_spec_digests),
            "distance_form": self.distance_form,
            "sigma": self.sigma,
            "derivation": "median(nonzero_source_pair_distances)",
            "zero_distance_fallback": None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "artifact_digest": self.artifact_digest}


def derive_v03_source_only_sigma(
    index: MarketBoundSourceRepresentationIndex,
    *,
    distance_form: Literal["mmd", "mmd2"] = "mmd",
) -> SourceOnlySigmaArtifact:
    """Derive the global L-min scale from source specs only, with no fallback."""

    if not isinstance(index, MarketBoundSourceRepresentationIndex):
        raise V03BaselineError("source-only sigma requires a market-bound v0.3 index")
    if distance_form not in {"mmd", "mmd2"}:
        raise V03BaselineError("distance_form must be mmd or mmd2")
    ids = _canonical_pool_ids(index.entries, "source representation index")
    distances: list[float] = []
    for left_position, left_id in enumerate(ids):
        left = index.entries[left_id]
        for right_id in ids[left_position + 1 :]:
            right = index.entries[right_id]
            if left.kernel_bandwidth != right.kernel_bandwidth:
                raise V03BaselineError("source-only sigma cannot mix kernel bandwidths")
            kernel = GaussianKernel(left.kernel_bandwidth)
            cross = float(
                left.reduced_kme.beta
                @ kernel.gram(left.reduced_kme.supports, right.reduced_kme.supports)
                @ right.reduced_kme.beta
            )
            raw_squared = float(
                left.reduced_kme.rkme_norm2 + right.reduced_kme.rkme_norm2 - 2.0 * cross
            )
            scale = max(
                1.0,
                abs(left.reduced_kme.rkme_norm2),
                abs(right.reduced_kme.rkme_norm2),
            )
            if raw_squared < -1.0e-8 * scale:
                raise V03BaselineError("source pair MMD is materially negative")
            squared = max(raw_squared, 0.0)
            value = squared if distance_form == "mmd2" else math.sqrt(squared)
            if value > 0.0:
                distances.append(value)
    if not distances:
        raise V03BaselineError("source-only sigma has no nonzero source-pair distance")
    return SourceOnlySigmaArtifact(
        policy_market_id=index.policy_market_id,
        representation_index_digest=index.representation_index_digest,
        source_ids=ids,
        source_spec_digests=tuple(
            str(index.entries[opaque_id].source_spec_digest) for opaque_id in ids
        ),
        distance_form=distance_form,
        sigma=float(np.median(np.asarray(distances, dtype=np.float64))),
    )


def _trace_payload(feature: TraceFeatureVector) -> dict[str, Any]:
    return {
        "values": feature.values.tolist(),
        "feature_protocol_id": feature.feature_protocol_id,
        "probe_dataset_digest": feature.probe_dataset_digest,
        "feature_digest": feature.digest,
    }


def _validate_market_index(
    market: V03SourcePolicyMarket, index: MarketBoundSourceRepresentationIndex
) -> tuple[str, ...]:
    if not isinstance(market, V03SourcePolicyMarket):
        raise V03BaselineError("market must be a typed V03SourcePolicyMarket")
    if not isinstance(index, MarketBoundSourceRepresentationIndex):
        raise V03BaselineError("baseline index must be explicitly market-bound")
    if market.policy_market_id != index.policy_market_id:
        raise V03BaselineError("market/index policy_market_id mismatch")
    market_ids = _canonical_pool_ids(market.entries, "public market")
    if market_ids != _canonical_pool_ids(index.entries, "representation index"):
        raise V03BaselineError("market/index anonymous coverage differs")
    return market_ids


def _artifact(
    *,
    method_id: str,
    index: MarketBoundSourceRepresentationIndex,
    evidence: EvidenceContract,
    fit_scope: FitScope,
    training_data_digest: str,
    freeze: DevelopmentBaselineFreeze,
    payload: Mapping[str, Any],
) -> FrozenBaselineSelectorArtifact:
    if freeze.policy_market_id != index.policy_market_id:
        raise V03BaselineError("development freeze belongs to another policy market")
    return FrozenBaselineSelectorArtifact(
        method_id=method_id,
        policy_market_id=index.policy_market_id,
        representation_index_digest=index.representation_index_digest,
        evidence_contract=evidence,
        fit_scope=fit_scope,
        training_data_digest=training_data_digest,
        development_freeze_digest=str(freeze.freeze_digest),
        execution_mode=freeze.execution_mode,
        development_context_count=freeze.development_context_count,
        payload=payload,
    )


def fit_baseline_suite(
    *,
    market: V03SourcePolicyMarket,
    raw_index: MarketBoundSourceRepresentationIndex,
    corro_index: MarketBoundSourceRepresentationIndex,
    development_view: DevelopmentView,
    development_freeze: DevelopmentBaselineFreeze,
    legacy_task_specs: Mapping[str, TraceFeatureVector],
    nominal_champions: Mapping[str, str],
    raw_moment_features: Mapping[str, TraceFeatureVector],
    lmin_epsilon: float = 1.0e-12,
    distance_form: Literal["mmd", "mmd2"] = "mmd",
) -> Mapping[str, FrozenBaselineSelectorArtifact]:
    """Fit/freeze the exact required v0.3 simple-baseline registry.

    No query or confirmatory-oracle value is accepted by this function.
    """

    ids = _validate_market_index(market, raw_index)
    if _validate_market_index(market, corro_index) != ids:
        raise V03BaselineError("Raw/CORRO index coverage differs")
    if not isinstance(development_view, DevelopmentView):
        raise V03BaselineError("development_view has the wrong type")
    if not isinstance(development_freeze, DevelopmentBaselineFreeze):
        raise V03BaselineError("development_freeze has the wrong type")
    if development_freeze.execution_mode == FORMAL_MODE:
        development_freeze.require_formal_24_context_matrix()
    if development_view.policy_market_id != market.policy_market_id:
        raise V03BaselineError("development labels belong to another market")
    if development_view.digest != development_freeze.development_view_digest:
        raise V03BaselineError("development view differs from its freeze")
    if development_view.split_manifest_digest != development_freeze.split_manifest_digest:
        raise V03BaselineError("development split manifest differs from its freeze")
    if development_view.label_contract_digest != development_freeze.label_contract_digest:
        raise V03BaselineError("development label contract differs from its freeze")
    if development_view.label_count != development_freeze.label_count:
        raise V03BaselineError("development label count differs from its freeze")
    if tuple(development_view.training_context_ids) != development_freeze.training_context_ids:
        raise V03BaselineError("development training split differs from its freeze")
    if tuple(development_view.validation_context_ids) != development_freeze.validation_context_ids:
        raise V03BaselineError("development validation split differs from its freeze")
    if set(development_view.opaque_policy_ids) != set(ids):
        raise V03BaselineError("development labels must cover the exact anonymous market")

    features = dict(raw_moment_features)
    if set(features) != set(ids) or any(
        not isinstance(feature, TraceFeatureVector) for feature in features.values()
    ):
        raise V03BaselineError("B3a source features must cover the exact anonymous market")
    feature_protocols = {feature.feature_protocol_id for feature in features.values()}
    dimensions = {int(feature.values.size) for feature in features.values()}
    if len(feature_protocols) != 1 or len(dimensions) != 1:
        raise V03BaselineError("B3a source features must share protocol and dimension")

    task_specs = dict(legacy_task_specs)
    champions = dict(nominal_champions)
    if len(task_specs) != 6 or set(task_specs) != set(champions):
        raise V03BaselineError(
            "B2 requires the exact six legacy TaskSpecs and matching nominal champions"
        )
    if any(not isinstance(feature, TraceFeatureVector) for feature in task_specs.values()):
        raise V03BaselineError("B2 TaskSpecs must be TraceFeatureVector objects")
    if not set(champions.values()).issubset(ids) or len(set(champions.values())) != len(champions):
        raise V03BaselineError("B2 nominal champions must be distinct market entries")
    task_protocols = {feature.feature_protocol_id for feature in task_specs.values()}
    task_dimensions = {int(feature.values.size) for feature in task_specs.values()}
    if len(task_protocols) != 1 or len(task_dimensions) != 1:
        raise V03BaselineError("B2 TaskSpecs must share protocol and dimension")

    epsilon = _positive(lmin_epsilon, "lmin_epsilon")
    if epsilon >= 1.0:
        raise V03BaselineError("lmin_epsilon must be less than one")
    sigma = derive_v03_source_only_sigma(corro_index, distance_form=distance_form)
    source_training = sha256_json(
        {
            "schema": "policy-learnware.v03-source-only-baseline-input.v0",
            "policy_market_id": market.policy_market_id,
            "raw_index": raw_index.representation_index_digest,
            "corro_index": corro_index.representation_index_digest,
        }
    )
    no_query = PUBLIC_NO_QUERY_EVIDENCE
    probe = public_probe_evidence_contract()
    source_labels = public_probe_evidence_contract(source_side_labels=True)
    supervised = public_probe_evidence_contract(development_supervised=True)

    artifacts: dict[str, FrozenBaselineSelectorArtifact] = {}
    artifacts["B0"] = _artifact(
        method_id="B0",
        index=raw_index,
        evidence=no_query,
        fit_scope="source_only",
        training_data_digest=source_training,
        freeze=development_freeze,
        payload={"kind": "random", "selector_seed": development_freeze.selector_seed},
    )
    artifacts["B1"] = _artifact(
        method_id="B1",
        index=raw_index,
        evidence=no_query,
        fit_scope="source_only",
        training_data_digest=source_training,
        freeze=development_freeze,
        payload={"kind": "competence_only"},
    )
    artifacts["B2"] = _artifact(
        method_id="B2",
        index=raw_index,
        evidence=source_labels,
        fit_scope="source_only",
        training_data_digest=sha256_json(
            {
                "task_specs": {key: feature.digest for key, feature in sorted(task_specs.items())},
                "nominal_champions": dict(sorted(champions.items())),
            }
        ),
        freeze=development_freeze,
        payload={
            "kind": "legacy_taskspec_mapping",
            "feature_protocol_id": next(iter(task_protocols)),
            "source_task_specs": {
                key: _trace_payload(feature) for key, feature in sorted(task_specs.items())
            },
            "nominal_champions": dict(sorted(champions.items())),
        },
    )
    artifacts["B3a"] = _artifact(
        method_id="B3a",
        index=raw_index,
        evidence=probe,
        fit_scope="source_only",
        training_data_digest=sha256_json(
            {key: feature.digest for key, feature in sorted(features.items())}
        ),
        freeze=development_freeze,
        payload={
            "kind": "raw_moments_nearest",
            "feature_protocol_id": next(iter(feature_protocols)),
            "source_features": {
                key: _trace_payload(feature) for key, feature in sorted(features.items())
            },
        },
    )
    artifacts["B3b"] = _artifact(
        method_id="B3b",
        index=raw_index,
        evidence=probe,
        fit_scope="source_only",
        training_data_digest=str(raw_index.source_representation_index_digest),
        freeze=development_freeze,
        payload={"kind": "nearest_joint_distance", "distance_form": distance_form},
    )

    train_indices = np.asarray(development_view.training_indices, dtype=np.int64)
    train_features = development_view.context_features[train_indices]
    train_returns = development_view.normalized_returns[train_indices]
    artifacts["B4a"] = _artifact(
        method_id="B4a",
        index=raw_index,
        evidence=supervised,
        fit_scope="development_supervised",
        training_data_digest=development_view.digest,
        freeze=development_freeze,
        payload={
            "kind": "knn_development",
            "neighbor_count": development_freeze.b4a_neighbor_count,
            "feature_protocol_id": development_view.feature_protocol_id,
            "opaque_policy_ids": list(development_view.opaque_policy_ids),
            "training_context_ids": list(development_view.training_context_ids),
            "training_features": train_features.tolist(),
            "training_returns": train_returns.tolist(),
            "label_count": development_view.label_count,
        },
    )
    design = np.concatenate(
        [train_features, np.ones((train_features.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    system = design.T @ design + development_freeze.b4b_ridge * np.eye(
        design.shape[1], dtype=np.float64
    )
    coefficients = np.linalg.solve(system, design.T @ train_returns)
    if not np.all(np.isfinite(coefficients)):
        raise V03BaselineError("B4b fit produced non-finite coefficients")
    artifacts["B4b"] = _artifact(
        method_id="B4b",
        index=raw_index,
        evidence=supervised,
        fit_scope="development_supervised",
        training_data_digest=development_view.digest,
        freeze=development_freeze,
        payload={
            "kind": "linear_development",
            "ridge": development_freeze.b4b_ridge,
            "feature_protocol_id": development_view.feature_protocol_id,
            "opaque_policy_ids": list(development_view.opaque_policy_ids),
            "training_context_ids": list(development_view.training_context_ids),
            "coefficients": coefficients.tolist(),
            "intercept_column": True,
            "label_count": development_view.label_count,
        },
    )
    artifacts["A-Env"] = _artifact(
        method_id="A-Env",
        index=corro_index,
        evidence=probe,
        fit_scope="source_only",
        training_data_digest=str(corro_index.source_representation_index_digest),
        freeze=development_freeze,
        payload={"kind": "nearest_joint_distance", "distance_form": distance_form},
    )
    artifacts["M02/B5"] = _artifact(
        method_id="M02/B5",
        index=corro_index,
        evidence=probe,
        fit_scope="source_only",
        training_data_digest=str(sigma.artifact_digest),
        freeze=development_freeze,
        payload={
            "kind": "lmin_joint_distance",
            "distance_form": distance_form,
            "epsilon": epsilon,
            "source_only_sigma": sigma.to_dict(),
        },
    )
    if tuple(artifacts) != REQUIRED_BASELINE_METHOD_IDS:
        raise AssertionError("baseline registry construction order drifted")
    return MappingProxyType(artifacts)


@dataclass(frozen=True)
class PublishedRankingRow:
    opaque_learnware_id: str
    rank: int
    score: float
    tie_break_token: str
    distance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_learnware_id, str) or _OPAQUE_LEARNWARE_ID.fullmatch(
            self.opaque_learnware_id
        ) is None:
            raise V03BaselineError("ranking row has non-canonical learnware ID")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise V03BaselineError("ranking row rank must be positive")
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(
            self, "tie_break_token", _digest(self.tie_break_token, "tie_break_token")
        )
        if self.distance is not None:
            distance = _finite(self.distance, "distance")
            if distance < 0.0:
                raise V03BaselineError("ranking distance cannot be negative")
            object.__setattr__(self, "distance", distance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opaque_learnware_id": self.opaque_learnware_id,
            "rank": self.rank,
            "score": self.score,
            "distance": self.distance,
            "tie_break_token": self.tie_break_token,
        }


@dataclass(frozen=True)
class PublishedFullRanking:
    method_id: str
    opaque_query_id: str
    query_spec_digest: str
    probe_dataset_digest: str
    target_evidence_digest: str
    policy_market_id: str
    representation_index_digest: str
    selector_view_digest: str
    evidence_contract_digest: str
    cost_digest: str
    selector_artifact_digest: str
    development_freeze_digest: str
    query_input_digest: str
    execution_mode: BaselineExecutionMode
    query_mode: Literal["QUERY_EMPIRICAL", "QUERY_REDUCED"]
    development_context_count: int
    score_semantics: str
    selected_opaque_learnware_id: str
    rows: tuple[PublishedRankingRow, ...]
    ranking_digest: str | None = None
    schema: str = FULL_RANKING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FULL_RANKING_SCHEMA:
            raise V03BaselineError("unsupported full-ranking schema")
        if self.method_id not in REQUIRED_BASELINE_METHOD_IDS:
            raise V03BaselineError("full ranking has unknown method ID")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        for name in (
            "query_spec_digest",
            "probe_dataset_digest",
            "target_evidence_digest",
            "policy_market_id",
            "representation_index_digest",
            "selector_view_digest",
            "evidence_contract_digest",
            "cost_digest",
            "selector_artifact_digest",
            "development_freeze_digest",
            "query_input_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "score_semantics", _nonempty(self.score_semantics, "score_semantics"))
        if self.execution_mode not in {DEVELOPMENT_SMOKE_MODE, FORMAL_MODE}:
            raise V03BaselineError("full ranking execution_mode is invalid")
        if self.query_mode not in {"QUERY_EMPIRICAL", "QUERY_REDUCED"}:
            raise V03BaselineError("full ranking query_mode is invalid")
        if self.execution_mode == FORMAL_MODE and self.query_mode != "QUERY_EMPIRICAL":
            raise V03BaselineError(
                "formal full rankings require QUERY_EMPIRICAL"
            )
        if (
            isinstance(self.development_context_count, bool)
            or not isinstance(self.development_context_count, int)
            or self.development_context_count <= 0
        ):
            raise V03BaselineError(
                "full ranking development_context_count must be positive"
            )
        if (
            self.execution_mode == FORMAL_MODE
            and self.development_context_count != FORMAL_DEVELOPMENT_CONTEXT_COUNT
        ):
            raise V03BaselineError(
                "formal full ranking requires exactly 24 development contexts"
            )
        rows = tuple(self.rows)
        if len(rows) != EXPECTED_ANCHOR_COUNT or any(
            not isinstance(row, PublishedRankingRow) for row in rows
        ):
            raise V03BaselineError("full ranking must contain exactly 30 typed rows")
        _canonical_pool_ids(tuple(row.opaque_learnware_id for row in rows), "full ranking")
        if tuple(row.rank for row in rows) != tuple(range(1, EXPECTED_ANCHOR_COUNT + 1)):
            raise V03BaselineError("full ranking ranks must be contiguous")
        expected_order = tuple(
            sorted(rows, key=lambda row: (-row.score, row.tie_break_token))
        )
        if rows != expected_order:
            raise V03BaselineError("ranking is not ordered by score then frozen tie token")
        if self.selected_opaque_learnware_id != rows[0].opaque_learnware_id:
            raise V03BaselineError("selected ID must equal rank one")
        object.__setattr__(self, "rows", rows)
        expected = sha256_json(self._payload_without_digest())
        if self.ranking_digest is None:
            object.__setattr__(self, "ranking_digest", expected)
        elif _digest(self.ranking_digest, "ranking_digest") != expected:
            raise V03BaselineError("full-ranking digest mismatch")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "opaque_query_id": self.opaque_query_id,
            "query_spec_digest": self.query_spec_digest,
            "probe_dataset_digest": self.probe_dataset_digest,
            "target_evidence_digest": self.target_evidence_digest,
            "policy_market_id": self.policy_market_id,
            "representation_index_digest": self.representation_index_digest,
            "selector_view_digest": self.selector_view_digest,
            "evidence_contract_digest": self.evidence_contract_digest,
            "cost_digest": self.cost_digest,
            "selector_artifact_digest": self.selector_artifact_digest,
            "development_freeze_digest": self.development_freeze_digest,
            "query_input_digest": self.query_input_digest,
            "execution_mode": self.execution_mode,
            "query_mode": self.query_mode,
            "development_context_count": self.development_context_count,
            "score_semantics": self.score_semantics,
            "selected_opaque_learnware_id": self.selected_opaque_learnware_id,
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "ranking_digest": self.ranking_digest}


def _publish_scores(
    *,
    query: V03BaselineQuery,
    selector_view: AnonymousSelectorViewManifest,
    artifact: FrozenBaselineSelectorArtifact,
    scores: Mapping[str, float],
    score_semantics: str,
    distances: Mapping[str, float] | None = None,
) -> PublishedFullRanking:
    if artifact.policy_market_id != selector_view.policy_market_id:
        raise V03BaselineError("selector artifact belongs to another market")
    if artifact.representation_index_digest != selector_view.representation_index_digest:
        raise V03BaselineError("selector artifact belongs to another representation index")
    ids = _canonical_pool_ids(selector_view.entries, "anonymous selector view")
    if set(scores) != set(ids):
        raise V03BaselineError("baseline scores must cover the exact anonymous market")
    if distances is not None and set(distances) != set(ids):
        raise V03BaselineError("baseline distances must cover the exact anonymous market")
    rows = tuple(
        PublishedRankingRow(
            opaque_learnware_id=opaque_id,
            rank=rank,
            score=_finite(scores[opaque_id], f"score[{opaque_id}]"),
            distance=(None if distances is None else distances[opaque_id]),
            tie_break_token=selector_view.entries[opaque_id].tie_break_token,
        )
        for rank, opaque_id in enumerate(
            sorted(
                ids,
                key=lambda item: (
                    -_finite(scores[item], f"score[{item}]"),
                    selector_view.entries[item].tie_break_token,
                ),
            ),
            start=1,
        )
    )
    return PublishedFullRanking(
        method_id=artifact.method_id,
        opaque_query_id=query.opaque_query_id,
        query_spec_digest=query.query_spec_digest,
        probe_dataset_digest=query.probe_dataset_digest,
        target_evidence_digest=query.target_evidence_digest,
        policy_market_id=selector_view.policy_market_id,
        representation_index_digest=selector_view.representation_index_digest,
        selector_view_digest=str(selector_view.selector_view_digest),
        evidence_contract_digest=artifact.evidence_contract_digest,
        cost_digest=query.cost_digest,
        selector_artifact_digest=str(artifact.selector_artifact_digest),
        development_freeze_digest=artifact.development_freeze_digest,
        query_input_digest=query.query_input_digest,
        execution_mode=query.execution_mode,
        query_mode=query.query_mode,
        development_context_count=artifact.development_context_count,
        score_semantics=score_semantics,
        selected_opaque_learnware_id=rows[0].opaque_learnware_id,
        rows=rows,
    )


def _payload_feature(raw: Mapping[str, Any], where: str) -> tuple[np.ndarray, str]:
    if not isinstance(raw, Mapping):
        raise V03BaselineError(f"{where} feature payload is invalid")
    values = np.asarray(raw["values"], dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise V03BaselineError(f"{where} feature vector is invalid")
    protocol = _digest(raw["feature_protocol_id"], f"{where}.feature_protocol_id")
    feature = TraceFeatureVector(
        values=values,
        feature_protocol_id=protocol,
        probe_dataset_digest=raw["probe_dataset_digest"],
    )
    if feature.digest != raw["feature_digest"]:
        raise V03BaselineError(f"{where} feature digest mismatch")
    return values, protocol


def _joint_distances(
    run: JointDistanceRun,
    *,
    query: V03BaselineQuery,
    artifact: FrozenBaselineSelectorArtifact,
    selector_view: AnonymousSelectorViewManifest,
) -> Mapping[str, float]:
    if not isinstance(run, JointDistanceRun):
        raise V03BaselineError("distance baseline requires a JointDistanceRun")
    if run.query_spec_digest != query.query_spec_digest:
        raise V03BaselineError("distance run belongs to another query spec")
    if run.query_mode != query.query_mode:
        raise V03BaselineError("distance run query mode differs from the baseline query")
    if run.representation_index_digest != artifact.representation_index_digest:
        raise V03BaselineError("distance run belongs to another representation index")
    if run.representation_index_digest != selector_view.representation_index_digest:
        raise V03BaselineError("distance run differs from selector view")
    if run.distance_form != artifact.payload["distance_form"]:
        raise V03BaselineError("distance run differs from the frozen distance form")
    ids = _canonical_pool_ids(selector_view.entries, "anonymous selector view")
    by_id = {row.opaque_learnware_id: row.result.value for row in run.rows}
    if set(by_id) != set(ids):
        raise V03BaselineError("distance run does not cover the exact anonymous market")
    return MappingProxyType(by_id)


def run_baseline_ranking(
    *,
    query: V03BaselineQuery,
    selector_view: AnonymousSelectorViewManifest,
    artifact: FrozenBaselineSelectorArtifact,
    distance_run: JointDistanceRun | None = None,
) -> PublishedFullRanking:
    """Execute one frozen baseline without accepting any oracle argument."""

    if not isinstance(query, V03BaselineQuery):
        raise V03BaselineError("query has the wrong type")
    if not isinstance(selector_view, AnonymousSelectorViewManifest):
        raise V03BaselineError("selector_view has the wrong type")
    if not isinstance(artifact, FrozenBaselineSelectorArtifact):
        raise V03BaselineError("selector artifact has the wrong type")
    artifact.evidence_contract.require_public_selector_safe()
    if query.execution_mode != artifact.execution_mode:
        raise V03BaselineError(
            "query and selector artifact execution modes differ"
        )
    if (
        query.execution_mode == FORMAL_MODE
        and artifact.development_context_count != FORMAL_DEVELOPMENT_CONTEXT_COUNT
    ):
        raise V03BaselineError(
            "formal baseline run requires a 24-context selector artifact"
        )
    ids = _canonical_pool_ids(selector_view.entries, "anonymous selector view")
    payload = artifact.payload
    kind = payload["kind"]
    distances: Mapping[str, float] | None = None

    if kind == "random":
        seed = payload["selector_seed"]
        scores = {
            opaque_id: float(
                int(
                    sha256_json(
                        {
                            "schema": "policy-learnware.v03-random-baseline-key.v0",
                            "selector_seed": seed,
                            "query_spec_digest": query.query_spec_digest,
                            "tie_break_token": selector_view.entries[opaque_id].tie_break_token,
                        }
                    )[:13],
                    16,
                )
                / float(16**13)
            )
            for opaque_id in ids
        }
        semantics = "frozen_random_key"
    elif kind == "competence_only":
        if any(
            selector_view.entries[opaque_id].normalized_source_competence is None
            for opaque_id in ids
        ):
            raise V03BaselineError("B1 requires public source competence for every entry")
        scores = {
            opaque_id: float(
                selector_view.entries[opaque_id].normalized_source_competence
            )
            for opaque_id in ids
        }
        semantics = "normalized_source_competence"
    elif kind == "legacy_taskspec_mapping":
        feature = query.trace_feature
        if feature is None or feature.feature_protocol_id != payload["feature_protocol_id"]:
            raise V03BaselineError("B2 requires its frozen legacy TaskSpec query feature")
        source_specs = payload["source_task_specs"]
        champions = payload["nominal_champions"]
        task_distances: dict[str, float] = {}
        for source_key, raw in source_specs.items():
            values, protocol = _payload_feature(raw, f"B2[{source_key}]")
            if protocol != feature.feature_protocol_id or values.shape != feature.values.shape:
                raise V03BaselineError("B2 query/source TaskSpec feature mismatch")
            task_distances[source_key] = float(np.linalg.norm(feature.values - values))
        nearest = min(
            task_distances,
            key=lambda key: (
                task_distances[key],
                selector_view.entries[champions[key]].tie_break_token,
            ),
        )
        selected = champions[nearest]
        scores = {opaque_id: float(opaque_id == selected) for opaque_id in ids}
        semantics = "legacy_taskspec_nearest_nominal_champion"
    elif kind == "raw_moments_nearest":
        feature = query.trace_feature
        if feature is None or feature.feature_protocol_id != payload["feature_protocol_id"]:
            raise V03BaselineError("B3a requires its frozen raw-moment query feature")
        local_distances: dict[str, float] = {}
        for opaque_id, raw in payload["source_features"].items():
            values, protocol = _payload_feature(raw, f"B3a[{opaque_id}]")
            if protocol != feature.feature_protocol_id or values.shape != feature.values.shape:
                raise V03BaselineError("B3a query/source feature mismatch")
            local_distances[opaque_id] = float(np.linalg.norm(feature.values - values))
        distances = MappingProxyType(local_distances)
        scores = {opaque_id: -distances[opaque_id] for opaque_id in ids}
        semantics = "negative_raw_moment_euclidean_distance"
    elif kind == "knn_development":
        feature = query.trace_feature
        if feature is None or feature.feature_protocol_id != payload["feature_protocol_id"]:
            raise V03BaselineError("B4a requires its frozen development query feature")
        training = np.asarray(payload["training_features"], dtype=np.float64)
        returns = np.asarray(payload["training_returns"], dtype=np.float64)
        if training.ndim != 2 or feature.values.shape != (training.shape[1],):
            raise V03BaselineError("B4a query/development feature mismatch")
        context_ids = tuple(payload["training_context_ids"])
        local = np.linalg.norm(training - feature.values[None, :], axis=1)
        order = sorted(range(len(context_ids)), key=lambda i: (local[i], context_ids[i]))
        neighbors = np.asarray(order[: int(payload["neighbor_count"])], dtype=np.int64)
        predictions = np.mean(returns[neighbors], axis=0)
        policy_ids = tuple(payload["opaque_policy_ids"])
        if set(policy_ids) != set(ids):
            raise V03BaselineError("B4a fitted policy columns differ from the market")
        lookup = {opaque_id: position for position, opaque_id in enumerate(policy_ids)}
        scores = {opaque_id: float(predictions[lookup[opaque_id]]) for opaque_id in ids}
        semantics = "knn_predicted_normalized_return"
    elif kind == "linear_development":
        feature = query.trace_feature
        if feature is None or feature.feature_protocol_id != payload["feature_protocol_id"]:
            raise V03BaselineError("B4b requires its frozen development query feature")
        coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        design = np.concatenate([feature.values, np.ones(1, dtype=np.float64)])
        if coefficients.ndim != 2 or design.shape != (coefficients.shape[0],):
            raise V03BaselineError("B4b query/development feature mismatch")
        predictions = design @ coefficients
        if not np.all(np.isfinite(predictions)):
            raise V03BaselineError("B4b produced non-finite predictions")
        policy_ids = tuple(payload["opaque_policy_ids"])
        if set(policy_ids) != set(ids):
            raise V03BaselineError("B4b fitted policy columns differ from the market")
        lookup = {opaque_id: position for position, opaque_id in enumerate(policy_ids)}
        scores = {opaque_id: float(predictions[lookup[opaque_id]]) for opaque_id in ids}
        semantics = "ridge_linear_predicted_normalized_return"
    elif kind == "nearest_joint_distance":
        if distance_run is None:
            raise V03BaselineError("distance baseline requires a public distance run")
        distances = _joint_distances(
            distance_run,
            query=query,
            artifact=artifact,
            selector_view=selector_view,
        )
        scores = {opaque_id: -distances[opaque_id] for opaque_id in ids}
        semantics = "negative_empirical_to_reduced_distance"
    elif kind == "lmin_joint_distance":
        if distance_run is None:
            raise V03BaselineError("M02/B5 requires a public distance run")
        distances = _joint_distances(
            distance_run,
            query=query,
            artifact=artifact,
            selector_view=selector_view,
        )
        sigma_payload = payload["source_only_sigma"]
        if not isinstance(sigma_payload, Mapping):
            raise V03BaselineError("M02/B5 source-only sigma payload is invalid")
        try:
            sigma_artifact = SourceOnlySigmaArtifact(
                schema=sigma_payload["schema"],
                policy_market_id=sigma_payload["policy_market_id"],
                representation_index_digest=sigma_payload[
                    "representation_index_digest"
                ],
                source_ids=tuple(sigma_payload["source_ids"]),
                source_spec_digests=tuple(sigma_payload["source_spec_digests"]),
                distance_form=sigma_payload["distance_form"],
                sigma=sigma_payload["sigma"],
                artifact_digest=sigma_payload["artifact_digest"],
            )
        except (KeyError, TypeError, V03BaselineError) as error:
            raise V03BaselineError(
                "M02/B5 source-only sigma payload failed validation"
            ) from error
        if sigma_artifact.artifact_digest != artifact.training_data_digest:
            raise V03BaselineError("M02/B5 sigma differs from its training binding")
        if sigma_artifact.representation_index_digest != artifact.representation_index_digest:
            raise V03BaselineError("M02/B5 sigma belongs to another source index")
        if set(sigma_artifact.source_ids) != set(ids):
            raise V03BaselineError("M02/B5 sigma source pool differs from the market")
        if sigma_artifact.distance_form != distance_run.distance_form:
            raise V03BaselineError("M02/B5 sigma and query distance forms differ")
        sigma = sigma_artifact.sigma
        epsilon = _positive(payload["epsilon"], "M02/B5 epsilon")
        scores = {}
        for opaque_id in ids:
            competence = selector_view.entries[opaque_id].normalized_source_competence
            if competence is None or not 0.0 <= competence <= 1.0:
                raise V03BaselineError("M02/B5 requires valid public competence")
            scores[opaque_id] = math.log(max(float(competence), epsilon)) - (
                distances[opaque_id] / sigma
            )
        semantics = "log_competence_minus_distance_over_source_sigma"
    else:  # pragma: no cover - artifact construction owns the closed registry
        raise V03BaselineError(f"unsupported frozen baseline kind {kind!r}")

    return _publish_scores(
        query=query,
        selector_view=selector_view,
        artifact=artifact,
        scores=scores,
        score_semantics=semantics,
        distances=distances,
    )


@dataclass(frozen=True)
class BaselineRankOneABIAudit:
    method_id: str
    opaque_query_id: str
    policy_market_id: str
    ranking_digest: str
    selected_opaque_learnware_id: str
    selected_execution_abi_digest: str
    target_execution_abi_digest: str
    compatible: bool
    inspected_opaque_learnware_ids: tuple[str, ...]
    fallback_attempted: Literal[False] = False
    fallback_policy: Literal["RANK_ONE_ONLY_NO_FALLBACK"] = NO_FALLBACK_POLICY
    audit_digest: str | None = None
    schema: str = RANK_ONE_ABI_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RANK_ONE_ABI_AUDIT_SCHEMA:
            raise V03BaselineError("unsupported baseline ABI audit schema")
        if self.method_id not in REQUIRED_BASELINE_METHOD_IDS:
            raise V03BaselineError("ABI audit has unknown method ID")
        object.__setattr__(self, "opaque_query_id", _query_id(self.opaque_query_id))
        if not isinstance(self.selected_opaque_learnware_id, str) or _OPAQUE_LEARNWARE_ID.fullmatch(
            self.selected_opaque_learnware_id
        ) is None:
            raise V03BaselineError("ABI audit selected ID is invalid")
        for name in (
            "policy_market_id",
            "ranking_digest",
            "selected_execution_abi_digest",
            "target_execution_abi_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if type(self.compatible) is not bool:
            raise V03BaselineError("ABI compatible flag must be boolean")
        expected_compatible = self.selected_execution_abi_digest == self.target_execution_abi_digest
        if self.compatible != expected_compatible:
            raise V03BaselineError("ABI compatible flag disagrees with ABI digests")
        inspected = tuple(self.inspected_opaque_learnware_ids)
        if inspected != (self.selected_opaque_learnware_id,):
            raise V03BaselineError("ABI audit may inspect rank one only")
        if self.fallback_attempted is not False or self.fallback_policy != NO_FALLBACK_POLICY:
            raise V03BaselineError("baseline ABI audit cannot fall back")
        expected = sha256_json(self._payload_without_digest())
        if self.audit_digest is None:
            object.__setattr__(self, "audit_digest", expected)
        elif _digest(self.audit_digest, "audit_digest") != expected:
            raise V03BaselineError("baseline ABI audit digest mismatch")

    @property
    def status(self) -> str:
        return "SELECTED_ABI_COMPATIBLE" if self.compatible else "SELECTED_INCOMPATIBLE_ABI"

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "opaque_query_id": self.opaque_query_id,
            "policy_market_id": self.policy_market_id,
            "ranking_digest": self.ranking_digest,
            "selected_opaque_learnware_id": self.selected_opaque_learnware_id,
            "selected_execution_abi_digest": self.selected_execution_abi_digest,
            "target_execution_abi_digest": self.target_execution_abi_digest,
            "compatible": self.compatible,
            "status": self.status,
            "inspected_opaque_learnware_ids": list(self.inspected_opaque_learnware_ids),
            "fallback_attempted": self.fallback_attempted,
            "fallback_policy": self.fallback_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "audit_digest": self.audit_digest}


def audit_baseline_rank_one_execution_abi(
    ranking: PublishedFullRanking,
    market: V03SourcePolicyMarket,
    target_execution_abi: ExecutionABIRecord,
) -> BaselineRankOneABIAudit:
    """Inspect exactly the already-published rank one; never try another policy."""

    if not isinstance(ranking, PublishedFullRanking):
        raise V03BaselineError("ranking has the wrong type")
    if not isinstance(market, V03SourcePolicyMarket):
        raise V03BaselineError("market has the wrong type")
    if not isinstance(target_execution_abi, ExecutionABIRecord):
        raise V03BaselineError("target_execution_abi has the wrong type")
    if ranking.policy_market_id != market.policy_market_id:
        raise V03BaselineError("ranking belongs to another policy market")
    if set(row.opaque_learnware_id for row in ranking.rows) != set(market.entries):
        raise V03BaselineError("ranking does not cover the exact policy market")
    expected_tie_tokens = {
        opaque_id: entry.tie_break_token for opaque_id, entry in market.entries.items()
    }
    observed_tie_tokens = {
        row.opaque_learnware_id: row.tie_break_token for row in ranking.rows
    }
    if observed_tie_tokens != expected_tie_tokens:
        raise V03BaselineError("ranking tie-break tokens differ from the frozen market")
    selected_id = ranking.selected_opaque_learnware_id
    # This is deliberately the only deployment-private lookup in the audit.
    selected_abi = market.deployment_private[selected_id].execution_abi
    return BaselineRankOneABIAudit(
        method_id=ranking.method_id,
        opaque_query_id=ranking.opaque_query_id,
        policy_market_id=ranking.policy_market_id,
        ranking_digest=str(ranking.ranking_digest),
        selected_opaque_learnware_id=selected_id,
        selected_execution_abi_digest=selected_abi.digest,
        target_execution_abi_digest=target_execution_abi.digest,
        compatible=selected_abi.digest == target_execution_abi.digest,
        inspected_opaque_learnware_ids=(selected_id,),
    )


__all__ = [
    "BaselineExecutionMode",
    "BASELINE_METHOD_KINDS",
    "BASELINE_ARTIFACT_SCHEMA",
    "BASELINE_FREEZE_SCHEMA",
    "BASELINE_QUERY_SCHEMA",
    "FULL_RANKING_SCHEMA",
    "FORMAL_DEVELOPMENT_CONTEXT_COUNT",
    "DEVELOPMENT_SMOKE_MODE",
    "FORMAL_MODE",
    "NO_FALLBACK_POLICY",
    "OPTIONAL_BASELINE_STATES",
    "PUBLIC_NO_QUERY_EVIDENCE",
    "QUERY_ALIAS_SCHEMA",
    "RAW_MOMENT_PROTOCOL_SCHEMA",
    "RANK_ONE_ABI_AUDIT_SCHEMA",
    "REQUIRED_BASELINE_METHOD_IDS",
    "SOURCE_SIGMA_SCHEMA",
    "BaselineRankOneABIAudit",
    "DevelopmentBaselineFreeze",
    "FrozenBaselineSelectorArtifact",
    "PublishedFullRanking",
    "PublishedRankingRow",
    "RawMomentFeatureProtocol",
    "SourceOnlySigmaArtifact",
    "V02V03QueryAliasEntry",
    "V02V03QueryAliasManifest",
    "V03BaselineError",
    "V03BaselineQuery",
    "audit_baseline_rank_one_execution_abi",
    "build_v02_v03_query_alias_manifest",
    "derive_v03_source_only_sigma",
    "evidence_contract_digest",
    "fit_baseline_suite",
    "freeze_development_baselines",
    "public_probe_evidence_contract",
    "raw_moment_feature_from_view",
    "run_baseline_ranking",
]
