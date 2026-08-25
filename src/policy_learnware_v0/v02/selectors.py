"""Anonymous full-market selector contracts and transparent L-min scoring.

Public selection intentionally has no execution-ABI or task/schema hard gate.
The selected opaque ID is published first; private deployment code performs
the minimum ABI check afterwards and records failure without falling back.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from ..hashing import sha256_json
from .environment_spec import DistanceForm, RepresentationIndex, environment_spec_distance
from .schemas import EnvironmentSpec, PublicMarketEntry


@dataclass(frozen=True)
class EvidenceContract:
    reads_source_raw_data: bool
    reads_development_policy_returns: bool
    reads_target_parameters: bool
    reads_target_transitions: bool
    reads_candidate_independent_probe_rewards: bool
    reads_candidate_target_rollouts: bool
    reads_candidate_policy_target_rewards: bool
    target_gradient_updates: int
    reads_submit_side_profiles: bool
    reads_source_side_labels: bool = False
    reads_target_task_reward_schema_identity: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "target_gradient_updates":
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("target_gradient_updates must be a non-negative integer")
            elif type(value) is not bool:
                raise ValueError(f"{name} must be boolean")

    @property
    def is_public_zero_update(self) -> bool:
        return bool(
            not self.reads_target_parameters
            and not self.reads_target_task_reward_schema_identity
            and not self.reads_candidate_target_rollouts
            and not self.reads_candidate_policy_target_rewards
            and self.target_gradient_updates == 0
            and not self.reads_submit_side_profiles
        )

    def require_public_selector_safe(self) -> None:
        if not self.is_public_zero_update:
            raise ValueError("public selector EvidenceContract grants private/oracle/update permissions")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceContract":
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                f"invalid EvidenceContract keys; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        return cls(**{key: value[key] for key in expected})


L_MIN_EVIDENCE = EvidenceContract(
    reads_source_raw_data=False,
    reads_development_policy_returns=False,
    reads_target_parameters=False,
    reads_target_transitions=True,
    reads_candidate_independent_probe_rewards=False,
    reads_candidate_target_rollouts=False,
    reads_candidate_policy_target_rewards=False,
    target_gradient_updates=0,
    reads_submit_side_profiles=False,
)


@dataclass(frozen=True)
class PublicMarketView:
    policy_market_id: str
    entries: Mapping[str, PublicMarketEntry]
    representation_index: RepresentationIndex
    protocol_family_id: str = "continuous-vector-mdp-v02"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_market_id, str) or not self.policy_market_id:
            raise ValueError("policy_market_id must be non-empty")
        if not isinstance(self.protocol_family_id, str) or not self.protocol_family_id:
            raise ValueError("protocol_family_id must be non-empty")
        entries = dict(self.entries)
        if not entries:
            raise ValueError("public market cannot be empty")
        for key, entry in entries.items():
            if key != entry.opaque_learnware_id:
                raise ValueError("public market key must match opaque_learnware_id")
        tokens = tuple(entry.tie_break_token for entry in entries.values())
        if len(tokens) != len(set(tokens)):
            raise ValueError("tie_break_token must be unique across the frozen market")
        if self.representation_index.policy_market_id != self.policy_market_id:
            raise ValueError("representation index is bound to another policy market")
        if set(entries) != set(self.representation_index.entries):
            raise ValueError("policy market and representation index IDs differ")
        object.__setattr__(self, "entries", MappingProxyType(entries))


@dataclass(frozen=True)
class RankingRow:
    opaque_learnware_id: str
    rank: int
    environment_distance: float | None
    normalized_source_competence: float
    log_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_learnware_id, str) or not self.opaque_learnware_id:
            raise ValueError("opaque_learnware_id must be non-empty")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        for name in ("normalized_source_competence", "log_score"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.environment_distance is not None:
            distance = float(self.environment_distance)
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError("environment_distance must be finite and non-negative")

    @property
    def opaque_id(self) -> str:
        return self.opaque_learnware_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "opaque_learnware_id": self.opaque_learnware_id,
            "rank": self.rank,
            "environment_distance": self.environment_distance,
            "normalized_source_competence": self.normalized_source_competence,
            "log_score": self.log_score,
        }


@dataclass(frozen=True)
class SelectionRecord:
    method_id: str
    selected_id: str
    ranking: tuple[RankingRow, ...]
    target_evidence_digest: str
    selector_artifact_digest: str
    cost_digest: str
    evidence_contract: EvidenceContract
    status: str = "SELECTED"

    def __post_init__(self) -> None:
        self.evidence_contract.require_public_selector_safe()
        if self.status != "SELECTED":
            raise ValueError("anonymous non-empty markets always publish a selected ID")
        if not self.selected_id or not self.ranking:
            raise ValueError("SelectionRecord requires selected_id and a full ranking")
        ranks = tuple(row.rank for row in self.ranking)
        ids = tuple(row.opaque_learnware_id for row in self.ranking)
        if ranks != tuple(range(1, len(self.ranking) + 1)) or len(ids) != len(set(ids)):
            raise ValueError("ranking must contain every entry exactly once with contiguous ranks")
        if self.ranking[0].opaque_learnware_id != self.selected_id:
            raise ValueError("selected_id must equal the first ranked entry")

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-selection-record.v0",
            "method_id": self.method_id,
            "status": self.status,
            "selected_id": self.selected_id,
            "ranking": [row.to_dict() for row in self.ranking],
            "target_evidence_digest": self.target_evidence_digest,
            "selector_artifact_digest": self.selector_artifact_digest,
            "cost_digest": self.cost_digest,
            "evidence_contract": self.evidence_contract.to_dict(),
        }


class LMinSelector:
    """Score competence × exp(-EnvironmentSpec distance / sigma) in log space."""

    def __init__(
        self,
        *,
        method_id: str,
        sigma: float,
        epsilon: float,
        distance_form: DistanceForm,
        evidence_contract: EvidenceContract = L_MIN_EVIDENCE,
    ) -> None:
        if not isinstance(method_id, str) or not method_id:
            raise ValueError("method_id must be non-empty")
        if not math.isfinite(float(sigma)) or sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if not math.isfinite(float(epsilon)) or not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must lie strictly between 0 and 1")
        if distance_form not in {"mmd", "mmd2"}:
            raise ValueError("distance_form must be 'mmd' or 'mmd2'")
        evidence_contract.require_public_selector_safe()
        self.method_id = method_id
        self.sigma = float(sigma)
        self.epsilon = float(epsilon)
        self.distance_form = distance_form
        self.evidence_contract = evidence_contract
        self.selector_artifact_digest = sha256_json(
            {
                "schema": "policy-learnware.v02-lmin-selector-artifact.v0",
                "method_id": method_id,
                "sigma": self.sigma,
                "epsilon": self.epsilon,
                "distance_form": distance_form,
                "evidence_contract": evidence_contract.to_dict(),
                "tie_break": "(-log_score,tie_break_token)",
            }
        )

    def select(
        self,
        *,
        query_spec: EnvironmentSpec,
        market: PublicMarketView,
        target_evidence_digest: str,
        cost_digest: str,
    ) -> SelectionRecord:
        if query_spec.representation_protocol_id != market.representation_index.representation_protocol_id:
            raise ValueError("query representation protocol differs from the market index")
        scored: list[tuple[float, str, str, float]] = []
        for opaque_id, entry in market.entries.items():
            source_spec = market.representation_index.entries[opaque_id].environment_spec
            distance = environment_spec_distance(
                query_spec, source_spec, distance_form=self.distance_form
            ).value
            competence = entry.normalized_source_competence
            if not math.isfinite(competence) or not 0.0 <= competence <= 1.0:
                raise ValueError("public source competence is invalid")
            log_score = math.log(max(competence, self.epsilon)) - distance / self.sigma
            scored.append((log_score, entry.tie_break_token, opaque_id, distance))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranking = tuple(
            RankingRow(
                opaque_learnware_id=opaque_id,
                rank=rank,
                environment_distance=distance,
                normalized_source_competence=market.entries[opaque_id].normalized_source_competence,
                log_score=log_score,
            )
            for rank, (log_score, _token, opaque_id, distance) in enumerate(scored, start=1)
        )
        return SelectionRecord(
            method_id=self.method_id,
            selected_id=ranking[0].opaque_learnware_id,
            ranking=ranking,
            target_evidence_digest=target_evidence_digest,
            selector_artifact_digest=self.selector_artifact_digest,
            cost_digest=cost_digest,
            evidence_contract=self.evidence_contract,
        )


__all__ = [
    "EvidenceContract",
    "L_MIN_EVIDENCE",
    "LMinSelector",
    "PublicMarketView",
    "RankingRow",
    "SelectionRecord",
]
