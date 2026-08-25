"""Dependency-light Paper-I baseline contracts and deterministic selectors.

The public selectors in this module never accept target-policy returns.  Any
selector that learns from return labels can fit only a typed
``DevelopmentView`` and its artifact must be bound to a development freeze
manifest before it can execute on a confirmatory query.  L-min scale derivation
accepts only the source ``RepresentationIndex`` and deliberately has no
development/confirmatory tuning API or zero-distance fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..hashing import canonicalize, sha256_json, sha256_ndarrays
from .environment_spec import (
    DistanceForm,
    RepresentationIndex,
    environment_spec_distance,
    source_only_median_scale,
)
from .representation import TraceFeatureVector
from .schemas import EnvironmentSpec
from .selectors import (
    EvidenceContract,
    LMinSelector,
    PublicMarketView,
    RankingRow,
    SelectionRecord,
)


ExecutionStage = Literal["development_discovery", "paper1_joint_confirmatory"]
FitCapability = Literal["source_only", "development_supervised"]
EXECUTION_STAGES = frozenset({"development_discovery", "paper1_joint_confirmatory"})
FIT_CAPABILITIES = frozenset({"source_only", "development_supervised"})


class BaselineContractError(ValueError):
    """A baseline crossed an evidence boundary or violated a frozen contract."""


class DuplicateBaselineError(BaselineContractError):
    """A method id was registered more than once."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BaselineContractError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise BaselineContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise BaselineContractError(f"{where} must be a SHA-256 digest") from error
    return result


def _deep_freeze(value: Any) -> Any:
    try:
        canonical = canonicalize(value)
    except (TypeError, ValueError) as error:
        raise BaselineContractError(
            f"selector payload is not canonical-JSON compatible: {error}"
        ) from error
    if isinstance(canonical, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in canonical.items()}
        )
    if isinstance(canonical, list):
        return tuple(_deep_freeze(item) for item in canonical)
    return canonical


def _readonly_matrix(value: Any, *, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or any(size <= 0 for size in array.shape)
        or not np.all(np.isfinite(array))
    ):
        raise BaselineContractError(f"{where} must be a finite non-empty matrix")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


PUBLIC_NO_TARGET_EVIDENCE = EvidenceContract(
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


def target_probe_evidence_contract(
    *,
    reads_development_policy_returns: bool,
    reads_probe_rewards: bool,
    reads_source_side_labels: bool = False,
) -> EvidenceContract:
    """Create an explicit public access card for probe-based methods."""

    if type(reads_development_policy_returns) is not bool:
        raise BaselineContractError(
            "reads_development_policy_returns must be boolean"
        )
    if type(reads_probe_rewards) is not bool:
        raise BaselineContractError("reads_probe_rewards must be boolean")
    if type(reads_source_side_labels) is not bool:
        raise BaselineContractError("reads_source_side_labels must be boolean")
    return EvidenceContract(
        reads_source_raw_data=False,
        reads_development_policy_returns=reads_development_policy_returns,
        reads_target_parameters=False,
        reads_target_transitions=True,
        reads_candidate_independent_probe_rewards=reads_probe_rewards,
        reads_candidate_target_rollouts=False,
        reads_candidate_policy_target_rewards=False,
        target_gradient_updates=0,
        reads_submit_side_profiles=False,
        reads_source_side_labels=reads_source_side_labels,
        reads_target_task_reward_schema_identity=False,
    )


@dataclass(frozen=True)
class DevelopmentView:
    """Rectangular development-only policy-by-context supervision table."""

    context_ids: tuple[str, ...]
    opaque_policy_ids: tuple[str, ...]
    context_features: np.ndarray
    normalized_returns: np.ndarray
    training_context_ids: tuple[str, ...]
    validation_context_ids: tuple[str, ...]
    evaluation_seed_digests: tuple[str, ...]
    policy_market_id: str
    feature_protocol_id: str
    split_manifest_digest: str
    label_contract_digest: str
    candidate_paired_seeds: bool
    stage: str = "development_discovery"

    def __post_init__(self) -> None:
        if self.stage != "development_discovery":
            raise BaselineContractError(
                "supervised selector fitting accepts development_discovery only"
            )
        contexts = tuple(_nonempty(item, "context_ids[]") for item in self.context_ids)
        policies = tuple(
            _nonempty(item, "opaque_policy_ids[]") for item in self.opaque_policy_ids
        )
        if not contexts or len(set(contexts)) != len(contexts):
            raise BaselineContractError("context_ids must be non-empty and unique")
        if not policies or len(set(policies)) != len(policies):
            raise BaselineContractError(
                "opaque_policy_ids must be non-empty and unique"
            )
        features = _readonly_matrix(self.context_features, where="context_features")
        returns = _readonly_matrix(self.normalized_returns, where="normalized_returns")
        if features.shape[0] != len(contexts):
            raise BaselineContractError("context feature rows do not match context_ids")
        if returns.shape != (len(contexts), len(policies)):
            raise BaselineContractError(
                "normalized_returns must cover every context-policy pair"
            )
        if np.any(returns < 0.0) or np.any(returns > 1.0):
            raise BaselineContractError("normalized_returns must lie in [0, 1]")
        training = tuple(
            _nonempty(item, "training_context_ids[]")
            for item in self.training_context_ids
        )
        validation = tuple(
            _nonempty(item, "validation_context_ids[]")
            for item in self.validation_context_ids
        )
        if (
            not training
            or not validation
            or len(set(training)) != len(training)
            or len(set(validation)) != len(validation)
            or set(training) & set(validation)
            or set(training) | set(validation) != set(contexts)
        ):
            raise BaselineContractError(
                "training/validation context IDs must be non-empty, disjoint, and exhaustive"
            )
        seeds = tuple(
            _digest(item, "evaluation_seed_digests[]")
            for item in self.evaluation_seed_digests
        )
        if len(seeds) != len(contexts):
            raise BaselineContractError(
                "evaluation_seed_digests must contain one contract per context"
            )
        if type(self.candidate_paired_seeds) is not bool:
            raise BaselineContractError("candidate_paired_seeds must be boolean")
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "opaque_policy_ids", policies)
        object.__setattr__(self, "context_features", features)
        object.__setattr__(self, "normalized_returns", returns)
        object.__setattr__(self, "training_context_ids", training)
        object.__setattr__(self, "validation_context_ids", validation)
        object.__setattr__(self, "evaluation_seed_digests", seeds)
        object.__setattr__(
            self,
            "policy_market_id",
            _nonempty(self.policy_market_id, "policy_market_id"),
        )
        for name in (
            "feature_protocol_id",
            "split_manifest_digest",
            "label_contract_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def training_indices(self) -> tuple[int, ...]:
        lookup = {context_id: index for index, context_id in enumerate(self.context_ids)}
        return tuple(lookup[context_id] for context_id in self.training_context_ids)

    @property
    def validation_indices(self) -> tuple[int, ...]:
        lookup = {context_id: index for index, context_id in enumerate(self.context_ids)}
        return tuple(lookup[context_id] for context_id in self.validation_context_ids)

    @property
    def label_count(self) -> int:
        return int(self.normalized_returns.size)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v02-development-view.v0",
                "stage": self.stage,
                "context_ids": list(self.context_ids),
                "opaque_policy_ids": list(self.opaque_policy_ids),
                "arrays_digest": sha256_ndarrays(
                    {
                        "context_features": self.context_features,
                        "normalized_returns": self.normalized_returns,
                    }
                ),
                "training_context_ids": list(self.training_context_ids),
                "validation_context_ids": list(self.validation_context_ids),
                "evaluation_seed_digests": list(self.evaluation_seed_digests),
                "policy_market_id": self.policy_market_id,
                "feature_protocol_id": self.feature_protocol_id,
                "split_manifest_digest": self.split_manifest_digest,
                "label_contract_digest": self.label_contract_digest,
                "candidate_paired_seeds": self.candidate_paired_seeds,
                "label_count": self.label_count,
            }
        )


@dataclass(frozen=True)
class TargetQueryView:
    """Public query view; target parameters and candidate returns are absent."""

    stage: ExecutionStage
    query_spec: EnvironmentSpec
    target_evidence_digest: str
    cost_digest: str
    probe_rewards_included: bool
    trace_feature: TraceFeatureVector | None = None

    def __post_init__(self) -> None:
        if self.stage not in EXECUTION_STAGES:
            raise BaselineContractError(f"unsupported query stage {self.stage!r}")
        if not isinstance(self.query_spec, EnvironmentSpec):
            raise BaselineContractError("query_spec must be an EnvironmentSpec")
        object.__setattr__(
            self,
            "target_evidence_digest",
            _digest(self.target_evidence_digest, "target_evidence_digest"),
        )
        object.__setattr__(self, "cost_digest", _digest(self.cost_digest, "cost_digest"))
        if type(self.probe_rewards_included) is not bool:
            raise BaselineContractError("probe_rewards_included must be boolean")
        if self.trace_feature is not None:
            if not isinstance(self.trace_feature, TraceFeatureVector):
                raise BaselineContractError(
                    "trace_feature must be a TraceFeatureVector or None"
                )
            if (
                self.trace_feature.probe_dataset_digest
                != self.query_spec.probe_dataset_digest
            ):
                raise BaselineContractError(
                    "trace feature and EnvironmentSpec use different probe evidence"
                )


@dataclass(frozen=True)
class FrozenSelectorArtifact:
    method_id: str
    evidence_contract: EvidenceContract
    fit_capability: FitCapability
    training_data_digest: str
    payload: Mapping[str, Any]
    development_freeze_ref: str | None = None
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _nonempty(self.method_id, "method_id"))
        if not isinstance(self.evidence_contract, EvidenceContract):
            raise BaselineContractError(
                "artifact evidence_contract has the wrong type"
            )
        self.evidence_contract.require_public_selector_safe()
        if self.fit_capability not in FIT_CAPABILITIES:
            raise BaselineContractError(
                f"unsupported fit capability {self.fit_capability!r}"
            )
        reads_development = self.evidence_contract.reads_development_policy_returns
        if (self.fit_capability == "development_supervised") != reads_development:
            raise BaselineContractError(
                "fit capability disagrees with development-return permission"
            )
        object.__setattr__(
            self,
            "training_data_digest",
            _digest(self.training_data_digest, "training_data_digest"),
        )
        object.__setattr__(self, "payload", _deep_freeze(self.payload))
        freeze_ref = self.development_freeze_ref
        if freeze_ref is not None:
            freeze_ref = _digest(freeze_ref, "development_freeze_ref")
        object.__setattr__(self, "development_freeze_ref", freeze_ref)
        expected = sha256_json(self._payload_without_digest())
        if self.artifact_digest is None:
            object.__setattr__(self, "artifact_digest", expected)
        else:
            actual = _digest(self.artifact_digest, "artifact_digest")
            if actual != expected:
                raise BaselineContractError(
                    "artifact_digest does not match selector artifact contents"
                )
            object.__setattr__(self, "artifact_digest", actual)

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-frozen-selector-artifact.v0",
            "method_id": self.method_id,
            "evidence_contract": self.evidence_contract.to_dict(),
            "fit_capability": self.fit_capability,
            "training_data_digest": self.training_data_digest,
            "payload": canonicalize(self.payload),
            "development_freeze_ref": self.development_freeze_ref,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "artifact_digest": self.artifact_digest,
        }

    def freeze_for_confirmatory(
        self, development_freeze_ref: str
    ) -> "FrozenSelectorArtifact":
        freeze_ref = _digest(development_freeze_ref, "development_freeze_ref")
        if self.development_freeze_ref is not None:
            if self.development_freeze_ref != freeze_ref:
                raise BaselineContractError(
                    "selector artifact is already bound to another development freeze"
                )
            return self
        return FrozenSelectorArtifact(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            fit_capability=self.fit_capability,
            training_data_digest=self.training_data_digest,
            payload=self.payload,
            development_freeze_ref=freeze_ref,
        )

    def require_execution_stage(self, stage: ExecutionStage) -> None:
        if stage not in EXECUTION_STAGES:
            raise BaselineContractError(f"unsupported execution stage {stage!r}")
        if stage == "paper1_joint_confirmatory" and self.development_freeze_ref is None:
            raise BaselineContractError(
                "confirmatory selection requires a development freeze reference"
            )


@dataclass(frozen=True)
class FrozenFeatureIndex:
    policy_market_id: str
    feature_protocol_id: str
    entries: Mapping[str, TraceFeatureVector]
    feature_index_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_market_id", _nonempty(self.policy_market_id, "policy_market_id")
        )
        object.__setattr__(
            self,
            "feature_protocol_id",
            _digest(self.feature_protocol_id, "feature_protocol_id"),
        )
        entries = dict(self.entries)
        if not entries:
            raise BaselineContractError("feature index cannot be empty")
        dimension: int | None = None
        for opaque_id, feature in entries.items():
            _nonempty(opaque_id, "feature index opaque_id")
            if not isinstance(feature, TraceFeatureVector):
                raise BaselineContractError(
                    "feature index entries must be TraceFeatureVector objects"
                )
            if feature.feature_protocol_id != self.feature_protocol_id:
                raise BaselineContractError("feature protocol mismatch in source index")
            if dimension is None:
                dimension = int(feature.values.size)
            elif feature.values.size != dimension:
                raise BaselineContractError("feature index dimensions differ")
        object.__setattr__(self, "entries", MappingProxyType(entries))
        expected = sha256_json(
            {
                "schema": "policy-learnware.v02-frozen-feature-index.v0",
                "policy_market_id": self.policy_market_id,
                "feature_protocol_id": self.feature_protocol_id,
                "entries": {
                    key: value.digest for key, value in sorted(entries.items())
                },
            }
        )
        if self.feature_index_id is None:
            object.__setattr__(self, "feature_index_id", expected)
        elif _digest(self.feature_index_id, "feature_index_id") != expected:
            raise BaselineContractError(
                "feature_index_id does not match source feature contents"
            )


@runtime_checkable
class Paper1SelectorProtocol(Protocol):
    method_id: str
    evidence_contract: EvidenceContract
    selector_binding: Mapping[str, Any]

    def fit(
        self, development_data: DevelopmentView | None
    ) -> FrozenSelectorArtifact: ...

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord: ...


class BaselineRegistry:
    def __init__(self) -> None:
        self._selectors: dict[str, Paper1SelectorProtocol] = {}

    def register(self, selector: Paper1SelectorProtocol) -> None:
        if not isinstance(selector, Paper1SelectorProtocol):
            raise BaselineContractError(
                "baseline does not implement Paper1SelectorProtocol"
            )
        method_id = _nonempty(selector.method_id, "method_id")
        if method_id in self._selectors:
            raise DuplicateBaselineError(f"baseline {method_id!r} is already registered")
        if not isinstance(selector.evidence_contract, EvidenceContract):
            raise BaselineContractError("baseline has no typed EvidenceContract")
        selector.evidence_contract.require_public_selector_safe()
        self._selectors[method_id] = selector

    def resolve(self, method_id: str) -> Paper1SelectorProtocol:
        key = _nonempty(method_id, "method_id")
        try:
            return self._selectors[key]
        except KeyError as error:
            raise BaselineContractError(f"unknown baseline {key!r}") from error

    @property
    def selectors(self) -> Mapping[str, Paper1SelectorProtocol]:
        return MappingProxyType(dict(self._selectors))


def _validate_evidence_for_method(
    evidence: EvidenceContract,
    *,
    development_supervised: bool,
    target_probe: bool,
    source_side_labels: bool | None = None,
) -> None:
    if not isinstance(evidence, EvidenceContract):
        raise BaselineContractError("selector evidence contract has the wrong type")
    evidence.require_public_selector_safe()
    if evidence.reads_development_policy_returns != development_supervised:
        raise BaselineContractError(
            "selector development permission differs from its implementation"
        )
    if evidence.reads_target_transitions != target_probe:
        raise BaselineContractError(
            "selector target-probe permission differs from its implementation"
        )
    if (
        source_side_labels is not None
        and evidence.reads_source_side_labels != source_side_labels
    ):
        raise BaselineContractError(
            "selector source-label permission differs from its implementation"
        )
    if not target_probe and evidence.reads_candidate_independent_probe_rewards:
        raise BaselineContractError(
            "selector cannot read probe rewards without reading target probe evidence"
        )


def _validate_query_access(query: TargetQueryView, evidence: EvidenceContract) -> None:
    if not isinstance(query, TargetQueryView):
        raise BaselineContractError("query must be a TargetQueryView")
    if evidence.reads_target_transitions and (
        evidence.reads_candidate_independent_probe_rewards
        != query.probe_rewards_included
    ):
        raise BaselineContractError(
            "selector probe-reward permission differs from the frozen query view"
        )


def _require_artifact(
    selector: Paper1SelectorProtocol,
    artifact: FrozenSelectorArtifact,
    query: TargetQueryView,
    market: PublicMarketView,
) -> None:
    if not isinstance(artifact, FrozenSelectorArtifact):
        raise BaselineContractError("selector requires a FrozenSelectorArtifact")
    if artifact.method_id != selector.method_id:
        raise BaselineContractError("selector artifact belongs to another method")
    if artifact.evidence_contract != selector.evidence_contract:
        raise BaselineContractError("selector artifact evidence contract differs")
    if not isinstance(selector.selector_binding, Mapping):
        raise BaselineContractError("selector has no typed runtime binding")
    actual_binding = artifact.payload.get("selector_binding")
    if not isinstance(actual_binding, Mapping) or canonicalize(
        actual_binding
    ) != canonicalize(selector.selector_binding):
        raise BaselineContractError(
            "selector artifact runtime binding differs from the executing instance"
        )
    expected_market_id = selector.selector_binding.get("policy_market_id")
    if expected_market_id != market.policy_market_id:
        raise BaselineContractError(
            "selector artifact belongs to another policy market"
        )
    expected_index_id = selector.selector_binding.get("representation_index_id")
    if (
        expected_index_id is not None
        and expected_index_id
        != market.representation_index.representation_index_id
    ):
        raise BaselineContractError(
            "selector artifact belongs to another representation index"
        )
    artifact.require_execution_stage(query.stage)
    _validate_query_access(query, selector.evidence_contract)


def _selector_binding(
    *,
    method_id: str,
    evidence_contract: EvidenceContract,
    policy_market_id: str,
    representation_index_id: str | None,
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Freeze every runtime input owned by a selector instance.

    The public market remains a separate immutable input, but an artifact is
    never portable to another market (or, where applicable, another source
    index).  Source assets are represented by their content-derived digests.
    """

    binding = {
        "schema": "policy-learnware.v02-selector-runtime-binding.v0",
        "method_id": _nonempty(method_id, "method_id"),
        "evidence_contract": evidence_contract.to_dict(),
        "policy_market_id": _nonempty(policy_market_id, "policy_market_id"),
        "representation_index_id": (
            None
            if representation_index_id is None
            else _digest(representation_index_id, "representation_index_id")
        ),
        "parameters": canonicalize(parameters),
    }
    frozen = _deep_freeze(binding)
    assert isinstance(frozen, Mapping)
    return frozen


def _with_selector_binding(
    selector: Paper1SelectorProtocol, payload: Mapping[str, Any]
) -> dict[str, Any]:
    if "selector_binding" in payload:
        raise BaselineContractError("selector payload reserves selector_binding")
    return {
        "selector_binding": canonicalize(selector.selector_binding),
        **dict(payload),
    }


def _source_only_artifact(
    *,
    selector: Paper1SelectorProtocol,
    payload: Mapping[str, Any],
) -> FrozenSelectorArtifact:
    bound_payload = _with_selector_binding(selector, payload)
    return FrozenSelectorArtifact(
        method_id=selector.method_id,
        evidence_contract=selector.evidence_contract,
        fit_capability="source_only",
        training_data_digest=sha256_json(
            {
                "schema": "policy-learnware.v02-source-only-selector-input.v0",
                "method_id": selector.method_id,
                "payload": canonicalize(bound_payload),
            }
        ),
        payload=bound_payload,
    )


def _anonymous_market_ids(
    query: TargetQueryView, market: PublicMarketView
) -> tuple[str, ...]:
    """Return every anonymous entry without a task/schema/ABI hard gate."""

    if not isinstance(query, TargetQueryView):
        raise BaselineContractError("query must be a TargetQueryView")
    if not isinstance(market, PublicMarketView):
        raise BaselineContractError("market must be a PublicMarketView")
    return tuple(sorted(market.entries))


def _selection_record(
    *,
    selector: Paper1SelectorProtocol,
    query: TargetQueryView,
    market: PublicMarketView,
    artifact: FrozenSelectorArtifact,
    scores: Mapping[str, float],
    distances: Mapping[str, float] | None = None,
) -> SelectionRecord:
    if not isinstance(scores, Mapping):
        raise BaselineContractError("selector scores must be a mapping")
    market_ids = _anonymous_market_ids(query, market)
    if set(scores) != set(market_ids):
        raise BaselineContractError(
            "selector scores do not exactly cover the anonymous full market"
        )
    for opaque_id, score in scores.items():
        if not math.isfinite(float(score)):
            raise BaselineContractError(f"non-finite score for {opaque_id!r}")
    if distances is not None:
        if not isinstance(distances, Mapping):
            raise BaselineContractError("selector distances must be a mapping")
        if set(distances) != set(market_ids):
            raise BaselineContractError(
                "selector distances do not exactly cover the anonymous full market"
            )
        for opaque_id, distance in distances.items():
            if not math.isfinite(float(distance)) or float(distance) < 0.0:
                raise BaselineContractError(
                    f"invalid environment distance for {opaque_id!r}"
                )
    # This is the only public policy-ranking tie path in the module.  In
    # particular, opaque IDs and source-side label keys never break a tie.
    ordered = tuple(
        sorted(
            market_ids,
            key=lambda item: (-scores[item], market.entries[item].tie_break_token),
        )
    )
    ranking = tuple(
        RankingRow(
            opaque_learnware_id=opaque_id,
            rank=rank,
            environment_distance=(
                None if distances is None else float(distances[opaque_id])
            ),
            normalized_source_competence=(
                market.entries[opaque_id].normalized_source_competence
            ),
            log_score=float(scores[opaque_id]),
        )
        for rank, opaque_id in enumerate(ordered, start=1)
    )
    return SelectionRecord(
        method_id=selector.method_id,
        selected_id=ordered[0],
        ranking=ranking,
        target_evidence_digest=query.target_evidence_digest,
        selector_artifact_digest=str(artifact.artifact_digest),
        cost_digest=query.cost_digest,
        evidence_contract=selector.evidence_contract,
    )


class RandomAnonymousMarketSelector:
    """B0: deterministic replay of a frozen anonymous-market random seed."""

    evidence_contract = PUBLIC_NO_TARGET_EVIDENCE

    def __init__(
        self, *, method_id: str, selector_seed: int, policy_market_id: str
    ) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        if type(selector_seed) is not int or selector_seed < 0:
            raise BaselineContractError(
                "selector_seed must be an explicit non-negative integer"
            )
        self.selector_seed = selector_seed
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=policy_market_id,
            representation_index_id=None,
            parameters={"selector_seed": self.selector_seed},
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if development_data is not None:
            raise BaselineContractError("random baseline cannot fit development data")
        return _source_only_artifact(
            selector=self,
            payload={"selector_seed": self.selector_seed},
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        market_ids = _anonymous_market_ids(query, market)
        scores = {
            opaque_id: float(
                int(
                    sha256_json(
                        {
                            "schema": "policy-learnware.v02-random-anonymous-market-key.v0",
                            "selector_seed": self.selector_seed,
                            "target_evidence_digest": query.target_evidence_digest,
                            "entry_random_token": market.entries[opaque_id].tie_break_token,
                        }
                    )[:13],
                    16,
                )
                / float(16**13)
            )
            for opaque_id in market_ids
        }
        return _selection_record(
            selector=self,
            query=query,
            market=market,
            artifact=artifact,
            scores=scores,
        )


def finite_pool_random_probabilities(
    query: TargetQueryView, market: PublicMarketView
) -> Mapping[str, float]:
    """Return the exact B0 selection distribution over the anonymous full pool."""

    market_ids = _anonymous_market_ids(query, market)
    probability = 1.0 / len(market_ids)
    return MappingProxyType({opaque_id: probability for opaque_id in market_ids})


# Backward import alias; behavior is the anonymous full-market B0 above.
RandomCompatibleSelector = RandomAnonymousMarketSelector


class CompetenceOnlySelector:
    """B1: global source-attestation champion over the anonymous full pool."""

    evidence_contract = PUBLIC_NO_TARGET_EVIDENCE

    def __init__(self, *, method_id: str, policy_market_id: str) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=policy_market_id,
            representation_index_id=None,
            parameters={
                "tie_break": "(-normalized_source_competence,tie_break_token)"
            },
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if development_data is not None:
            raise BaselineContractError("competence-only baseline cannot fit development data")
        return _source_only_artifact(
            selector=self,
            payload={"tie_break": "(-normalized_source_competence,tie_break_token)"},
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        scores = {
            opaque_id: market.entries[opaque_id].normalized_source_competence
            for opaque_id in _anonymous_market_ids(query, market)
        }
        return _selection_record(
            selector=self,
            query=query,
            market=market,
            artifact=artifact,
            scores=scores,
        )


class LegacyTaskSpecSelector:
    """B2: identify a nominal source TaskSpec from target probe evidence.

    No target task ID, task-contract digest, runtime schema, or ABI is present
    in the query.  Source-side nominal labels are frozen into this predecessor
    artifact and only the learned TaskSpec feature is used at selection time.
    """

    def __init__(
        self,
        *,
        method_id: str,
        source_task_specs: Mapping[str, TraceFeatureVector],
        nominal_champions: Mapping[str, str],
        policy_market_id: str,
        evidence_contract: EvidenceContract,
    ) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        _validate_evidence_for_method(
            evidence_contract,
            development_supervised=False,
            target_probe=True,
            source_side_labels=True,
        )
        specs = dict(source_task_specs)
        champions = dict(nominal_champions)
        if not specs or set(specs) != set(champions):
            raise BaselineContractError(
                "legacy source TaskSpecs and nominal champions must share non-empty opaque keys"
            )
        for source_key, spec in specs.items():
            _nonempty(source_key, "legacy source TaskSpec key")
            if not isinstance(spec, TraceFeatureVector):
                raise BaselineContractError(
                    "legacy source TaskSpecs must be TraceFeatureVector objects"
                )
        protocols = {spec.feature_protocol_id for spec in specs.values()}
        dimensions = {int(spec.values.size) for spec in specs.values()}
        if len(protocols) != 1 or len(dimensions) != 1:
            raise BaselineContractError("legacy source TaskSpecs must share one feature protocol")
        self.source_task_specs = MappingProxyType(specs)
        parsed_champions = {
            key: _nonempty(value, "nominal champion")
            for key, value in champions.items()
        }
        if len(set(parsed_champions.values())) != len(parsed_champions):
            raise BaselineContractError(
                "each legacy nominal TaskSpec must bind a distinct champion"
            )
        self.nominal_champions = MappingProxyType(parsed_champions)
        self.feature_protocol_id = next(iter(protocols))
        self.evidence_contract = evidence_contract
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=policy_market_id,
            representation_index_id=None,
            parameters={
                "feature_protocol_id": self.feature_protocol_id,
                "source_task_spec_digests": {
                    key: spec.digest
                    for key, spec in sorted(self.source_task_specs.items())
                },
                "nominal_champions": dict(self.nominal_champions),
                "distance": "euclidean",
                "tie_break": "(distance,nominal_champion.tie_break_token)",
            },
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if development_data is not None:
            raise BaselineContractError("legacy TaskSpec predecessor cannot fit development labels")
        return _source_only_artifact(
            selector=self,
            payload={
                "feature_protocol_id": self.feature_protocol_id,
                "source_task_spec_digests": {
                    key: spec.digest for key, spec in sorted(self.source_task_specs.items())
                },
                "nominal_champions": dict(self.nominal_champions),
                "distance": "euclidean",
                "tie_break": "(distance,nominal_champion.tie_break_token)",
            },
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        feature = query.trace_feature
        if feature is None or feature.feature_protocol_id != self.feature_protocol_id:
            raise BaselineContractError("legacy TaskSpec predecessor requires its frozen query view")
        expected_dimension = next(iter(self.source_task_specs.values())).values.shape
        if feature.values.shape != expected_dimension:
            raise BaselineContractError(
                "legacy query/source TaskSpec feature dimensions differ"
            )
        missing_champions = set(self.nominal_champions.values()) - set(market.entries)
        if missing_champions:
            raise BaselineContractError(
                "legacy nominal champions are absent from the frozen anonymous market"
            )
        distances = {
            key: float(np.linalg.norm(feature.values - spec.values))
            for key, spec in self.source_task_specs.items()
        }
        nearest_key = min(
            distances,
            key=lambda key: (
                distances[key],
                market.entries[self.nominal_champions[key]].tie_break_token,
            ),
        )
        selected = self.nominal_champions[nearest_key]
        market_ids = _anonymous_market_ids(query, market)
        scores = {opaque_id: float(opaque_id == selected) for opaque_id in market_ids}
        return _selection_record(
            selector=self,
            query=query,
            market=market,
            artifact=artifact,
            scores=scores,
        )


# Backward import alias with the unsafe task-ID routing semantics removed.
TaskNominalChampionSelector = LegacyTaskSpecSelector


class EnvironmentOnlySelector:
    """A-Env/B3b: nearest EnvironmentSpec with market-token ties only."""

    def __init__(
        self,
        *,
        method_id: str,
        distance_form: DistanceForm,
        policy_market_id: str,
        representation_index_id: str,
        evidence_contract: EvidenceContract,
    ) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        if distance_form not in {"mmd", "mmd2"}:
            raise BaselineContractError("distance_form must be 'mmd' or 'mmd2'")
        _validate_evidence_for_method(
            evidence_contract,
            development_supervised=False,
            target_probe=True,
        )
        self.distance_form = distance_form
        self.evidence_contract = evidence_contract
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=policy_market_id,
            representation_index_id=representation_index_id,
            parameters={
                "distance_form": self.distance_form,
                "tie_break": "(distance,tie_break_token)",
            },
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if development_data is not None:
            raise BaselineContractError(
                "environment-only baseline cannot fit development labels"
            )
        return _source_only_artifact(
            selector=self,
            payload={
                "distance_form": self.distance_form,
                "tie_break": "(distance,tie_break_token)",
            },
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        if (
            query.query_spec.representation_protocol_id
            != market.representation_index.representation_protocol_id
        ):
            raise BaselineContractError(
                "query representation protocol differs from source index"
            )
        distances = {
            opaque_id: environment_spec_distance(
                query.query_spec,
                market.representation_index.entries[opaque_id].environment_spec,
                distance_form=self.distance_form,
            ).value
            for opaque_id in _anonymous_market_ids(query, market)
        }
        scores = {opaque_id: -distance for opaque_id, distance in distances.items()}
        return _selection_record(
            selector=self,
            query=query,
            market=market,
            artifact=artifact,
            scores=scores,
            distances=distances,
        )


class VectorNearestSelector:
    """B3a adapter for frozen raw-transition moment vectors."""

    def __init__(
        self,
        *,
        method_id: str,
        feature_index: FrozenFeatureIndex,
        evidence_contract: EvidenceContract,
    ) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        if not isinstance(feature_index, FrozenFeatureIndex):
            raise BaselineContractError("feature_index has the wrong type")
        _validate_evidence_for_method(
            evidence_contract,
            development_supervised=False,
            target_probe=True,
            source_side_labels=False,
        )
        self.feature_index = feature_index
        self.evidence_contract = evidence_contract
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=self.feature_index.policy_market_id,
            representation_index_id=None,
            parameters={
                "feature_index_id": self.feature_index.feature_index_id,
                "feature_protocol_id": self.feature_index.feature_protocol_id,
                "source_feature_digests": {
                    key: value.digest
                    for key, value in sorted(self.feature_index.entries.items())
                },
                "distance": "euclidean",
                "tie_break": "(distance,tie_break_token)",
            },
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if development_data is not None:
            raise BaselineContractError("vector nearest cannot fit development labels")
        return _source_only_artifact(
            selector=self,
            payload={
                "feature_index_id": self.feature_index.feature_index_id,
                "policy_market_id": self.feature_index.policy_market_id,
                "feature_protocol_id": self.feature_index.feature_protocol_id,
                "distance": "euclidean",
                "tie_break": "(distance,tie_break_token)",
            },
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        if self.feature_index.policy_market_id != market.policy_market_id:
            raise BaselineContractError("feature index belongs to another policy market")
        feature = query.trace_feature
        if feature is None:
            raise BaselineContractError("vector nearest requires a query trace feature")
        if feature.feature_protocol_id != self.feature_index.feature_protocol_id:
            raise BaselineContractError("query/source feature protocols differ")
        market_ids = _anonymous_market_ids(query, market)
        if set(self.feature_index.entries) != set(market_ids):
            raise BaselineContractError(
                "source feature index must exactly cover the anonymous full market"
            )
        distances = {
            opaque_id: float(
                np.linalg.norm(
                    feature.values - self.feature_index.entries[opaque_id].values
                )
            )
            for opaque_id in market_ids
        }
        scores = {opaque_id: -distance for opaque_id, distance in distances.items()}
        return _selection_record(
            selector=self,
            query=query,
            market=market,
            artifact=artifact,
            scores=scores,
            distances=distances,
        )


class KnnDevelopmentSelector:
    """B4a: context-local return ranker with an explicitly frozen ``k``."""

    def __init__(
        self,
        *,
        method_id: str,
        neighbor_count: int,
        policy_market_id: str,
        evidence_contract: EvidenceContract,
    ) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        if type(neighbor_count) is not int or neighbor_count <= 0:
            raise BaselineContractError(
                "neighbor_count must be an explicit positive integer"
            )
        _validate_evidence_for_method(
            evidence_contract,
            development_supervised=True,
            target_probe=True,
        )
        self.neighbor_count = neighbor_count
        self.evidence_contract = evidence_contract
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=policy_market_id,
            representation_index_id=None,
            parameters={
                "neighbor_count": self.neighbor_count,
                "tie_break": "(-predicted_return,tie_break_token)",
            },
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if not isinstance(development_data, DevelopmentView):
            raise BaselineContractError("kNN ranker requires a DevelopmentView")
        if (
            development_data.policy_market_id
            != self.selector_binding["policy_market_id"]
        ):
            raise BaselineContractError(
                "development labels belong to another policy market"
            )
        indices = development_data.training_indices
        if self.neighbor_count > len(indices):
            raise BaselineContractError(
                "neighbor_count exceeds frozen training-context count"
            )
        features = development_data.context_features[np.asarray(indices)]
        returns = development_data.normalized_returns[np.asarray(indices)]
        context_ids = tuple(development_data.context_ids[index] for index in indices)
        payload = {
            "neighbor_count": self.neighbor_count,
            "feature_protocol_id": development_data.feature_protocol_id,
            "opaque_policy_ids": list(development_data.opaque_policy_ids),
            "training_context_ids": list(context_ids),
            "validation_context_ids": list(development_data.validation_context_ids),
            "features": features.tolist(),
            "returns": returns.tolist(),
            "label_count": development_data.label_count,
            "candidate_paired_seeds": development_data.candidate_paired_seeds,
            "tie_break": "(-predicted_return,tie_break_token)",
        }
        return FrozenSelectorArtifact(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            fit_capability="development_supervised",
            training_data_digest=development_data.digest,
            payload=_with_selector_binding(self, payload),
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        feature = query.trace_feature
        if feature is None:
            raise BaselineContractError("kNN ranker requires a query trace feature")
        payload = artifact.payload
        if feature.feature_protocol_id != payload["feature_protocol_id"]:
            raise BaselineContractError("query/development feature protocols differ")
        features = np.asarray(payload["features"], dtype=np.float64)
        returns = np.asarray(payload["returns"], dtype=np.float64)
        policy_ids = tuple(payload["opaque_policy_ids"])
        contexts = tuple(payload["training_context_ids"])
        if feature.values.shape != (features.shape[1],):
            raise BaselineContractError("query/development feature dimensions differ")
        distances = np.linalg.norm(features - feature.values[None, :], axis=1)
        order = sorted(range(len(contexts)), key=lambda i: (distances[i], contexts[i]))
        neighbors = np.asarray(order[: int(payload["neighbor_count"])], dtype=np.int64)
        predictions = np.mean(returns[neighbors], axis=0)
        by_policy = {opaque_id: index for index, opaque_id in enumerate(policy_ids)}
        market_ids = _anonymous_market_ids(query, market)
        if set(by_policy) != set(market_ids):
            raise BaselineContractError(
                "development labels must exactly cover the anonymous full market"
            )
        scores = {
            opaque_id: float(predictions[by_policy[opaque_id]])
            for opaque_id in market_ids
        }
        return _selection_record(
            selector=self,
            query=query,
            market=market,
            artifact=artifact,
            scores=scores,
        )


class LinearDevelopmentSelector:
    """B4b: per-policy ridge-linear return model with explicit regularization."""

    def __init__(
        self,
        *,
        method_id: str,
        ridge: float,
        policy_market_id: str,
        evidence_contract: EvidenceContract,
    ) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        ridge_value = float(ridge)
        if not math.isfinite(ridge_value) or ridge_value <= 0.0:
            raise BaselineContractError("ridge must be an explicit positive value")
        _validate_evidence_for_method(
            evidence_contract,
            development_supervised=True,
            target_probe=True,
        )
        self.ridge = ridge_value
        self.evidence_contract = evidence_contract
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=policy_market_id,
            representation_index_id=None,
            parameters={
                "ridge": self.ridge,
                "intercept_column": True,
                "tie_break": "(-predicted_return,tie_break_token)",
            },
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if not isinstance(development_data, DevelopmentView):
            raise BaselineContractError("linear ranker requires a DevelopmentView")
        if (
            development_data.policy_market_id
            != self.selector_binding["policy_market_id"]
        ):
            raise BaselineContractError(
                "development labels belong to another policy market"
            )
        indices = np.asarray(development_data.training_indices, dtype=np.int64)
        features = development_data.context_features[indices]
        targets = development_data.normalized_returns[indices]
        design = np.concatenate(
            [features, np.ones((features.shape[0], 1), dtype=np.float64)], axis=1
        )
        system = design.T @ design + self.ridge * np.eye(
            design.shape[1], dtype=np.float64
        )
        coefficients = np.linalg.solve(system, design.T @ targets)
        if not np.all(np.isfinite(coefficients)):
            raise BaselineContractError("linear fit produced non-finite coefficients")
        payload = {
            "ridge": self.ridge,
            "feature_protocol_id": development_data.feature_protocol_id,
            "opaque_policy_ids": list(development_data.opaque_policy_ids),
            "training_context_ids": list(development_data.training_context_ids),
            "validation_context_ids": list(development_data.validation_context_ids),
            "coefficients": coefficients.tolist(),
            "intercept_column": True,
            "label_count": development_data.label_count,
            "candidate_paired_seeds": development_data.candidate_paired_seeds,
            "tie_break": "(-predicted_return,tie_break_token)",
        }
        return FrozenSelectorArtifact(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            fit_capability="development_supervised",
            training_data_digest=development_data.digest,
            payload=_with_selector_binding(self, payload),
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        feature = query.trace_feature
        if feature is None:
            raise BaselineContractError("linear ranker requires a query trace feature")
        payload = artifact.payload
        if feature.feature_protocol_id != payload["feature_protocol_id"]:
            raise BaselineContractError("query/development feature protocols differ")
        coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        design = np.concatenate([feature.values, np.ones(1, dtype=np.float64)])
        if design.shape != (coefficients.shape[0],):
            raise BaselineContractError("query/development feature dimensions differ")
        predictions = design @ coefficients
        if not np.all(np.isfinite(predictions)):
            raise BaselineContractError("linear selector produced non-finite predictions")
        policy_ids = tuple(payload["opaque_policy_ids"])
        by_policy = {opaque_id: index for index, opaque_id in enumerate(policy_ids)}
        market_ids = _anonymous_market_ids(query, market)
        if set(by_policy) != set(market_ids):
            raise BaselineContractError(
                "development labels must exactly cover the anonymous full market"
            )
        scores = {
            opaque_id: float(predictions[by_policy[opaque_id]])
            for opaque_id in market_ids
        }
        return _selection_record(
            selector=self,
            query=query,
            market=market,
            artifact=artifact,
            scores=scores,
        )


@dataclass(frozen=True)
class SourceOnlySigmaArtifact:
    policy_market_id: str
    representation_index_id: str
    partition_id: str
    source_ids: tuple[str, ...]
    source_spec_digests: tuple[str, ...]
    distance_form: DistanceForm
    sigma: float
    artifact_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_market_id",
            _nonempty(self.policy_market_id, "policy_market_id"),
        )
        object.__setattr__(
            self,
            "representation_index_id",
            _digest(self.representation_index_id, "representation_index_id"),
        )
        object.__setattr__(self, "partition_id", _nonempty(self.partition_id, "partition_id"))
        source_ids = tuple(_nonempty(item, "source_ids[]") for item in self.source_ids)
        if len(source_ids) < 2 or len(set(source_ids)) != len(source_ids):
            raise BaselineContractError(
                "source-only sigma requires at least two unique source IDs"
            )
        digests = tuple(
            _digest(item, "source_spec_digests[]")
            for item in self.source_spec_digests
        )
        if len(digests) != len(source_ids):
            raise BaselineContractError(
                "source_spec_digests must align with source_ids"
            )
        if self.distance_form not in {"mmd", "mmd2"}:
            raise BaselineContractError("distance_form must be 'mmd' or 'mmd2'")
        sigma = float(self.sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise BaselineContractError("source-only sigma must be finite and positive")
        object.__setattr__(self, "source_ids", source_ids)
        object.__setattr__(self, "source_spec_digests", digests)
        object.__setattr__(self, "sigma", sigma)
        expected = sha256_json(self._payload_without_digest())
        if self.artifact_digest is None:
            object.__setattr__(self, "artifact_digest", expected)
        elif _digest(self.artifact_digest, "artifact_digest") != expected:
            raise BaselineContractError(
                "source-only sigma artifact digest mismatch"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-source-only-sigma.v0",
            "policy_market_id": self.policy_market_id,
            "representation_index_id": self.representation_index_id,
            "partition_id": self.partition_id,
            "source_ids": list(self.source_ids),
            "source_spec_digests": list(self.source_spec_digests),
            "distance_form": self.distance_form,
            "sigma": self.sigma,
            "derivation": "median(nonzero_source_pair_distances)",
            "zero_distance_fallback": None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "artifact_digest": self.artifact_digest}


def derive_source_only_sigma_artifacts(
    index: RepresentationIndex,
    *,
    partitions: Mapping[str, Sequence[str]],
    distance_form: DistanceForm,
) -> Mapping[str, SourceOnlySigmaArtifact]:
    """Freeze per-partition sigma from source specs only, with no fallback."""

    if not isinstance(index, RepresentationIndex):
        raise BaselineContractError("source sigma requires a RepresentationIndex")
    if not partitions:
        raise BaselineContractError("source sigma partitions cannot be empty")
    claimed: set[str] = set()
    normalized: dict[str, tuple[str, ...]] = {}
    for partition_id, values in partitions.items():
        key = _nonempty(partition_id, "partition_id")
        source_ids = tuple(values)
        overlap = claimed & set(source_ids)
        if overlap:
            raise BaselineContractError(
                f"source IDs occur in multiple sigma partitions: {sorted(overlap)}"
            )
        claimed.update(source_ids)
        normalized[key] = source_ids
    specs = {
        opaque_id: entry.environment_spec for opaque_id, entry in index.entries.items()
    }
    scales = source_only_median_scale(
        specs,
        partitions=normalized,
        distance_form=distance_form,
        zero_fallback=None,
    )
    assert index.representation_index_id is not None
    result = {
        partition_id: SourceOnlySigmaArtifact(
            policy_market_id=index.policy_market_id,
            representation_index_id=index.representation_index_id,
            partition_id=partition_id,
            source_ids=tuple(source_ids),
            source_spec_digests=tuple(
                str(index.entries[opaque_id].environment_spec.environment_spec_digest)
                for opaque_id in source_ids
            ),
            distance_form=distance_form,
            sigma=scales[partition_id],
        )
        for partition_id, source_ids in normalized.items()
    }
    return MappingProxyType(result)


class SourceOnlyLMinSelector:
    """M02/B5 adapter using an auditable source-only sigma artifact."""

    def __init__(
        self,
        *,
        method_id: str,
        sigma_artifact: SourceOnlySigmaArtifact,
        epsilon: float,
        evidence_contract: EvidenceContract,
    ) -> None:
        self.method_id = _nonempty(method_id, "method_id")
        if not isinstance(sigma_artifact, SourceOnlySigmaArtifact):
            raise BaselineContractError("sigma_artifact has the wrong type")
        _validate_evidence_for_method(
            evidence_contract,
            development_supervised=False,
            target_probe=True,
        )
        epsilon_value = float(epsilon)
        if not math.isfinite(epsilon_value) or not 0.0 < epsilon_value < 1.0:
            raise BaselineContractError("epsilon must be explicitly in (0, 1)")
        self.sigma_artifact = sigma_artifact
        self.epsilon = epsilon_value
        self.evidence_contract = evidence_contract
        self.selector_binding = _selector_binding(
            method_id=self.method_id,
            evidence_contract=self.evidence_contract,
            policy_market_id=self.sigma_artifact.policy_market_id,
            representation_index_id=self.sigma_artifact.representation_index_id,
            parameters={
                "source_only_sigma_artifact": self.sigma_artifact.to_dict(),
                "epsilon": self.epsilon,
                "tie_break": "(-log_score,tie_break_token)",
            },
        )

    def fit(self, development_data: DevelopmentView | None) -> FrozenSelectorArtifact:
        if development_data is not None:
            raise BaselineContractError(
                "formal source-only L-min cannot fit development labels"
            )
        return _source_only_artifact(
            selector=self,
            payload={
                "source_only_sigma_artifact_digest": self.sigma_artifact.artifact_digest,
                "policy_market_id": self.sigma_artifact.policy_market_id,
                "representation_index_id": self.sigma_artifact.representation_index_id,
                "partition_id": self.sigma_artifact.partition_id,
                "source_ids": list(self.sigma_artifact.source_ids),
                "source_spec_digests": list(
                    self.sigma_artifact.source_spec_digests
                ),
                "distance_form": self.sigma_artifact.distance_form,
                "sigma": self.sigma_artifact.sigma,
                "epsilon": self.epsilon,
                "tie_break": "(-log_score,tie_break_token)",
            },
        )

    def select(
        self,
        query: TargetQueryView,
        market: PublicMarketView,
        artifact: FrozenSelectorArtifact,
    ) -> SelectionRecord:
        _require_artifact(self, artifact, query, market)
        if (
            market.representation_index.representation_index_id
            != self.sigma_artifact.representation_index_id
        ):
            raise BaselineContractError(
                "source-only sigma belongs to another representation index"
            )
        market_ids = _anonymous_market_ids(query, market)
        if set(market_ids) != set(self.sigma_artifact.source_ids):
            raise BaselineContractError(
                "source-only sigma partition differs from the anonymous full market"
            )
        delegate = LMinSelector(
            method_id=self.method_id,
            sigma=self.sigma_artifact.sigma,
            epsilon=self.epsilon,
            distance_form=self.sigma_artifact.distance_form,
            evidence_contract=self.evidence_contract,
        )
        result = delegate.select(
            query_spec=query.query_spec,
            market=market,
            target_evidence_digest=query.target_evidence_digest,
            cost_digest=query.cost_digest,
        )
        return SelectionRecord(
            method_id=result.method_id,
            selected_id=result.selected_id,
            ranking=result.ranking,
            target_evidence_digest=result.target_evidence_digest,
            selector_artifact_digest=str(artifact.artifact_digest),
            cost_digest=result.cost_digest,
            evidence_contract=result.evidence_contract,
        )


__all__ = [
    "BaselineContractError",
    "BaselineRegistry",
    "CompetenceOnlySelector",
    "DevelopmentView",
    "DuplicateBaselineError",
    "EXECUTION_STAGES",
    "FIT_CAPABILITIES",
    "EnvironmentOnlySelector",
    "ExecutionStage",
    "FitCapability",
    "FrozenFeatureIndex",
    "FrozenSelectorArtifact",
    "KnnDevelopmentSelector",
    "LegacyTaskSpecSelector",
    "LinearDevelopmentSelector",
    "PUBLIC_NO_TARGET_EVIDENCE",
    "Paper1SelectorProtocol",
    "RandomAnonymousMarketSelector",
    "RandomCompatibleSelector",
    "SourceOnlyLMinSelector",
    "SourceOnlySigmaArtifact",
    "TargetQueryView",
    "TaskNominalChampionSelector",
    "VectorNearestSelector",
    "derive_source_only_sigma_artifacts",
    "finite_pool_random_probabilities",
    "target_probe_evidence_contract",
]
