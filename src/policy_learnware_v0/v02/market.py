"""Source-anchor public market and physically separate deployment registry."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..hashing import sha256_json
from .competence import ChampionizationResult
from .schemas import ExecutionABIRecord, PublicMarketEntry
from .training import AdmittedTrainingRecord, admitted_training_records_digest


@dataclass(frozen=True)
class DeploymentPrivateEntry:
    opaque_learnware_id: str
    learnware_key: str
    bundle_digest: str
    bundle_path: str
    training_attestation_digest: str
    source_selection_digest: str
    source_attestation_digest: str
    formal_championization_admission_digest: str | None
    execution_abi: ExecutionABIRecord

    def __post_init__(self) -> None:
        if not isinstance(self.execution_abi, ExecutionABIRecord):
            raise ValueError("deployment entry requires a private ExecutionABIRecord")
        for name in (
            "bundle_digest",
            "training_attestation_digest",
            "source_selection_digest",
            "source_attestation_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a SHA-256 digest") from exc
        if self.formal_championization_admission_digest is not None:
            value = self.formal_championization_admission_digest
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(
                    "formal_championization_admission_digest must be a SHA-256 digest"
                )
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(
                    "formal_championization_admission_digest must be a SHA-256 digest"
                ) from exc

    @property
    def opaque_id(self) -> str:
        return self.opaque_learnware_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "opaque_learnware_id": self.opaque_learnware_id,
            "learnware_key": self.learnware_key,
            "bundle_digest": self.bundle_digest,
            "bundle_path": self.bundle_path,
            "training_attestation_digest": self.training_attestation_digest,
            "source_selection_digest": self.source_selection_digest,
            "source_attestation_digest": self.source_attestation_digest,
            "formal_championization_admission_digest": (
                self.formal_championization_admission_digest
            ),
            "execution_abi": self.execution_abi.to_dict(),
        }


@dataclass(frozen=True)
class V02PolicyMarket:
    policy_market_id: str
    entries: Mapping[str, PublicMarketEntry]
    deployment_private: Mapping[str, DeploymentPrivateEntry]
    anchor_to_opaque_id: Mapping[str, str]

    def __post_init__(self) -> None:
        entries = dict(self.entries)
        private = dict(self.deployment_private)
        anchor_map = dict(self.anchor_to_opaque_id)
        if not entries or set(entries) != set(private) or set(entries) != set(anchor_map.values()):
            raise ValueError("market public/private/anchor identity sets differ")
        tokens = tuple(entry.tie_break_token for entry in entries.values())
        if len(tokens) != len(set(tokens)):
            raise ValueError("public tie_break_token collision")
        expected = sha256_json(
            {
                "schema": "policy-learnware.v02-policy-market-id.v0",
                "entries": {
                    opaque_id: entry.to_dict() for opaque_id, entry in sorted(entries.items())
                },
                "deployment_binding_digest": sha256_json({
                    opaque_id: {
                        "bundle_digest": private[opaque_id].bundle_digest,
                        "training_attestation_digest": private[opaque_id].training_attestation_digest,
                        "source_selection_digest": private[opaque_id].source_selection_digest,
                        "source_attestation_digest": private[opaque_id].source_attestation_digest,
                        "formal_championization_admission_digest": (
                            private[opaque_id].formal_championization_admission_digest
                        ),
                    }
                    for opaque_id in sorted(private)
                }),
            }
        )
        if self.policy_market_id != expected:
            raise ValueError("policy_market_id does not match public/private bindings")
        object.__setattr__(self, "entries", MappingProxyType(entries))
        object.__setattr__(self, "deployment_private", MappingProxyType(private))
        object.__setattr__(self, "anchor_to_opaque_id", MappingProxyType(anchor_map))

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-public-policy-market.v0",
            "policy_market_id": self.policy_market_id,
            "entries": {
                opaque_id: entry.to_dict() for opaque_id, entry in sorted(self.entries.items())
            },
        }

    def deployment_manifest(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-deployment-private-registry.v0",
            "policy_market_id": self.policy_market_id,
            "entries": {
                opaque_id: entry.to_dict()
                for opaque_id, entry in sorted(self.deployment_private.items())
            },
        }


def build_policy_market(
    admitted_records: Mapping[str, AdmittedTrainingRecord],
    championization: ChampionizationResult,
    execution_abis: Mapping[str, ExecutionABIRecord],
    *,
    expected_anchor_ids: Iterable[str] | None = None,
    expected_anchor_count: int | None = None,
    market_alias_nonce: str,
    tie_break_nonce: str,
) -> V02PolicyMarket:
    """Publish exactly one competent, attested policy per source anchor.

    Formal publication obtains the exact anchor set from the strict
    championization admission (and may redundantly receive it from the caller).
    ``expected_anchor_count`` remains solely for CPU/audit-smoke compatibility;
    it is never sufficient to override a formal anchor-set binding.
    """

    if expected_anchor_count is not None and (
        isinstance(expected_anchor_count, bool)
        or not isinstance(expected_anchor_count, int)
        or expected_anchor_count <= 0
    ):
        raise ValueError("expected_anchor_count must be positive")
    for name, nonce in (
        ("market_alias_nonce", market_alias_nonce),
        ("tie_break_nonce", tie_break_nonce),
    ):
        if not isinstance(nonce, str) or len(nonce) != 64:
            raise ValueError(f"{name} must be an independently frozen 256-bit hex nonce")
        try:
            int(nonce, 16)
        except ValueError as error:
            raise ValueError(f"{name} must be an independently frozen 256-bit hex nonce") from error
    if market_alias_nonce == tie_break_nonce:
        raise ValueError("alias and tie-break random domains must use distinct nonces")
    selected = dict(championization.selected_by_anchor)
    formal = championization.formal_admission
    supplied_anchor_ids = None
    if expected_anchor_ids is not None:
        supplied = tuple(expected_anchor_ids)
        if not supplied or any(not isinstance(item, str) or not item for item in supplied):
            raise ValueError("expected_anchor_ids must be non-empty strings")
        if len(supplied) != len(set(supplied)):
            raise ValueError("expected_anchor_ids must be unique")
        supplied_anchor_ids = frozenset(supplied)
    if formal is not None:
        frozen_anchor_ids = frozenset(formal.expected_anchor_ids)
        if supplied_anchor_ids is not None and supplied_anchor_ids != frozen_anchor_ids:
            raise ValueError("caller anchor IDs differ from formal championization admission")
        expected = frozen_anchor_ids
        if formal.admitted_records_digest != admitted_training_records_digest(admitted_records):
            raise ValueError("admitted training records differ from formal championization")
        if set(admitted_records) != set(formal.expected_candidate_ids):
            raise ValueError("admitted candidate set differs from formal championization")
    elif supplied_anchor_ids is not None:
        expected = supplied_anchor_ids
    elif expected_anchor_count is not None:
        # Compatibility for explicitly non-formal CPU acceptance.  Production
        # callers must use the exact ID set or a formal admission binding.
        expected = frozenset(selected)
    else:
        raise ValueError("market construction requires exact expected anchor IDs")
    if expected_anchor_count is not None and len(expected) != expected_anchor_count:
        raise ValueError("expected anchor count disagrees with the exact anchor set")
    if set(selected) != set(expected):
        raise ValueError("championization source anchors differ from the exact expected set")
    if championization.rejected_anchors:
        raise ValueError("an anchor failed independent competence attestation")
    if set(championization.competence_records) != set(selected):
        raise ValueError("competence records do not cover every selected anchor")
    if set(championization.attested_bundle_digests) != set(selected):
        raise ValueError("championization lacks exact attested bundle bindings")
    if set(execution_abis) != set(selected.values()):
        raise ValueError("private execution ABIs must cover exactly the selected champions")
    public: dict[str, PublicMarketEntry] = {}
    private: dict[str, DeploymentPrivateEntry] = {}
    anchor_map: dict[str, str] = {}
    for anchor, candidate in sorted(selected.items()):
        if candidate not in admitted_records or candidate not in execution_abis:
            raise ValueError("selected champion lacks admitted training/private ABI record")
        admitted = admitted_records[candidate]
        if admitted.job.source_anchor_id != anchor:
            raise ValueError("selected candidate belongs to another source anchor")
        selected_bundle = championization.selected_bundle_digests[anchor]
        attested_bundle = championization.attested_bundle_digests[anchor]
        admitted_bundle = admitted.attestation.bundle_digest
        if len({selected_bundle, attested_bundle, admitted_bundle}) != 1:
            raise ValueError(
                "selection, source attestation, and admitted training bundle digests differ"
            )
        competence = championization.competence_records[anchor]
        if competence.opaque_source_anchor_id != anchor:
            raise ValueError("source competence record belongs to another anchor")
        opaque_id = "lw-" + sha256_json(
            {
                "schema": "policy-learnware.v02-market-alias.v0",
                "market_alias_nonce": market_alias_nonce,
                "learnware_key": candidate,
            }
        )[:32]
        if opaque_id in public:
            raise ValueError("duplicate opaque learnware ID")
        public[opaque_id] = PublicMarketEntry(
            opaque_learnware_id=opaque_id,
            normalized_source_competence=competence.normalized_competence,
            tie_break_token=sha256_json(
                {
                    "schema": "policy-learnware.v02-market-tie-break.v0",
                    "tie_break_nonce": tie_break_nonce,
                    "learnware_key": candidate,
                }
            ),
        )
        path = admitted.attestation.bundle_path
        if path is None:
            raise ValueError("deployment-private admitted record lacks bundle path")
        private[opaque_id] = DeploymentPrivateEntry(
            opaque_learnware_id=opaque_id,
            learnware_key=candidate,
            bundle_digest=admitted.attestation.bundle_digest,
            bundle_path=path,
            training_attestation_digest=admitted.attestation.digest,
            source_selection_digest=championization.selection_digest,
            source_attestation_digest=competence.private_attestation_digest,
            formal_championization_admission_digest=(
                None if formal is None else formal.digest
            ),
            execution_abi=execution_abis[candidate],
        )
        anchor_map[anchor] = opaque_id
    market_id = sha256_json(
        {
            "schema": "policy-learnware.v02-policy-market-id.v0",
            "entries": {
                opaque_id: entry.to_dict() for opaque_id, entry in sorted(public.items())
            },
            "deployment_binding_digest": sha256_json({
                opaque_id: {
                    "bundle_digest": private[opaque_id].bundle_digest,
                    "training_attestation_digest": private[opaque_id].training_attestation_digest,
                    "source_selection_digest": private[opaque_id].source_selection_digest,
                    "source_attestation_digest": private[opaque_id].source_attestation_digest,
                    "formal_championization_admission_digest": (
                        private[opaque_id].formal_championization_admission_digest
                    ),
                }
                for opaque_id in sorted(private)
            }),
        }
    )
    return V02PolicyMarket(market_id, public, private, anchor_map)


__all__ = ["DeploymentPrivateEntry", "V02PolicyMarket", "build_policy_market"]
