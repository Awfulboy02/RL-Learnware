"""Frozen source-certificate projection and policy-label resolution for v0.5.

This module deliberately has no target-return input.  A certificate manifest is
only a deterministic projection of the already-frozen source championization,
private deployment registry, and execution-ABI bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from ..hashing import canonical_json, sha256_json


CERTIFICATE_MANIFEST_SCHEMA = "policy-learnware.v05-certified-policy-manifest.v1"
_OPAQUE_POLICY_ID = re.compile(r"^lw-[0-9a-f]{20,64}$")


class V05LabelError(ValueError):
    """A certificate projection is malformed or changes a frozen binding."""


def _canonical_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V05LabelError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _canonical_string(value, where)
    if len(result) != 64 or result != result.lower():
        raise V05LabelError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise V05LabelError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _opaque_policy_id(value: Any, where: str) -> str:
    result = _canonical_string(value, where)
    if _OPAQUE_POLICY_ID.fullmatch(result) is None:
        raise V05LabelError(f"{where} must match lw-[0-9a-f]{{20,64}}")
    return result


def _exact_keys(payload: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(payload, Mapping):
        raise V05LabelError(f"{where} must be an object")
    actual = set(payload)
    if actual != expected:
        raise V05LabelError(
            f"{where} has invalid fields; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True, order=True)
class CertificateBinding:
    """One source anchor's selector-external certified-policy binding."""

    source_anchor_id: str
    task_id: str
    opaque_certified_policy_id: str
    policy_bundle_digest: str
    championization_admission_digest: str
    execution_abi_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_anchor_id",
            _canonical_string(self.source_anchor_id, "source_anchor_id"),
        )
        object.__setattr__(self, "task_id", _canonical_string(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "opaque_certified_policy_id",
            _opaque_policy_id(
                self.opaque_certified_policy_id, "opaque_certified_policy_id"
            ),
        )
        for field_name in (
            "policy_bundle_digest",
            "championization_admission_digest",
            "execution_abi_digest",
        ):
            object.__setattr__(
                self, field_name, _digest(getattr(self, field_name), field_name)
            )

    @property
    def certified_policy_id(self) -> str:
        """Compatibility alias for the manifest's opaque policy label."""

        return self.opaque_certified_policy_id

    def to_dict(self) -> dict[str, str]:
        return {
            "source_anchor_id": self.source_anchor_id,
            "task_id": self.task_id,
            "opaque_certified_policy_id": self.opaque_certified_policy_id,
            "policy_bundle_digest": self.policy_bundle_digest,
            "championization_admission_digest": self.championization_admission_digest,
            "execution_abi_digest": self.execution_abi_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertificateBinding":
        fields = {
            "source_anchor_id",
            "task_id",
            "opaque_certified_policy_id",
            "policy_bundle_digest",
            "championization_admission_digest",
            "execution_abi_digest",
        }
        _exact_keys(payload, fields, "certificate binding")
        return cls(**{field: payload[field] for field in fields})


@dataclass(frozen=True)
class CertifiedPolicyManifest:
    """Canonical, replayable certificate projection.

    Multiple anchors may bind to the same opaque policy.  Reusing one opaque
    policy ID with a different bundle or ABI is rejected because that would make
    the label resolver deployment-dependent.
    """

    bindings: tuple[CertificateBinding, ...]
    schema: str = CERTIFICATE_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CERTIFICATE_MANIFEST_SCHEMA:
            raise V05LabelError("unsupported certified-policy manifest schema")
        normalized = tuple(
            item
            if isinstance(item, CertificateBinding)
            else CertificateBinding.from_dict(item)
            for item in self.bindings
        )
        if not normalized:
            raise V05LabelError("certificate manifest must not be empty")
        normalized = tuple(sorted(normalized, key=lambda item: item.source_anchor_id))
        anchors = [item.source_anchor_id for item in normalized]
        if len(anchors) != len(set(anchors)):
            raise V05LabelError("certificate manifest has duplicate source anchors")

        policy_identity: dict[str, tuple[str, str]] = {}
        bundle_owner: dict[str, str] = {}
        for item in normalized:
            identity = (item.policy_bundle_digest, item.execution_abi_digest)
            previous = policy_identity.setdefault(
                item.opaque_certified_policy_id, identity
            )
            if previous != identity:
                raise V05LabelError(
                    "one opaque certified policy has inconsistent bundle/ABI bindings"
                )
            owner = bundle_owner.setdefault(
                item.policy_bundle_digest, item.opaque_certified_policy_id
            )
            if owner != item.opaque_certified_policy_id:
                raise V05LabelError(
                    "one policy bundle is exposed under multiple opaque policy IDs"
                )
        object.__setattr__(self, "bindings", normalized)

    @property
    def records(self) -> tuple[CertificateBinding, ...]:
        """Alias used by artifact-oriented callers."""

        return self.bindings

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bindings": [item.to_dict() for item in self.bindings],
        }

    @property
    def certificate_manifest_digest(self) -> str:
        return sha256_json(self._payload())

    @property
    def manifest_digest(self) -> str:
        return self.certificate_manifest_digest

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["certificate_manifest_digest"] = self.certificate_manifest_digest
        return payload

    @classmethod
    def from_records(
        cls,
        records: Iterable[CertificateBinding | Mapping[str, Any]],
        *,
        expected_anchor_ids: Iterable[str] | None = None,
    ) -> "CertifiedPolicyManifest":
        result = cls(tuple(records))
        if expected_anchor_ids is not None:
            _require_exact_anchor_set(result, expected_anchor_ids)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CertifiedPolicyManifest":
        fields = {"schema", "bindings", "certificate_manifest_digest"}
        _exact_keys(payload, fields, "certified-policy manifest")
        raw_bindings = payload["bindings"]
        if not isinstance(raw_bindings, list):
            raise V05LabelError("certificate manifest bindings must be a list")
        result = cls(
            bindings=tuple(CertificateBinding.from_dict(item) for item in raw_bindings),
            schema=payload["schema"],
        )
        claimed = _digest(
            payload["certificate_manifest_digest"], "certificate_manifest_digest"
        )
        if claimed != result.certificate_manifest_digest:
            raise V05LabelError("certificate manifest digest does not match payload")
        return result

    @classmethod
    def from_json(cls, payload: str | bytes) -> "CertifiedPolicyManifest":
        if not isinstance(payload, (str, bytes)):
            raise V05LabelError("certificate manifest JSON must be str or bytes")
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V05LabelError("certificate manifest is not valid JSON") from error
        if not isinstance(decoded, Mapping):
            raise V05LabelError("certificate manifest JSON must encode an object")
        return cls.from_dict(decoded)


def _require_exact_anchor_set(
    manifest: CertifiedPolicyManifest, expected_anchor_ids: Iterable[str]
) -> None:
    expected_items = tuple(
        _canonical_string(item, "expected_anchor_id") for item in expected_anchor_ids
    )
    if len(expected_items) != len(set(expected_items)):
        raise V05LabelError("expected_anchor_ids must be unique")
    actual = {item.source_anchor_id for item in manifest.bindings}
    expected = set(expected_items)
    if actual != expected:
        raise V05LabelError(
            "certificate anchor coverage mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _exact_mapping_keys(
    mapping: Mapping[str, Any], expected: set[str], where: str
) -> None:
    if not isinstance(mapping, Mapping):
        raise V05LabelError(f"{where} must be an object")
    if any(not isinstance(item, str) for item in mapping):
        raise V05LabelError(f"{where} keys must be strings")
    actual = set(mapping)
    if actual != expected:
        raise V05LabelError(
            f"{where} coverage mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def project_certificate_manifest(
    anchor_to_policy: Mapping[str, str],
    *,
    task_by_anchor: Mapping[str, str],
    policy_bundle_digest_by_policy: Mapping[str, str],
    championization_admission_digest_by_anchor: Mapping[str, str],
    execution_abi_digest_by_policy: Mapping[str, str],
    expected_anchor_ids: Iterable[str] | None = None,
) -> CertifiedPolicyManifest:
    """Project independently frozen source-side maps into one manifest.

    The deliberately narrow signature has no target return, oracle winner, or
    target evaluator argument.  Every upstream map must have exact coverage;
    absent provenance therefore fails closed rather than being synthesized.
    """

    if not isinstance(anchor_to_policy, Mapping) or not anchor_to_policy:
        raise V05LabelError("anchor_to_policy must be a non-empty object")
    anchors = set(anchor_to_policy)
    if any(not isinstance(item, str) for item in anchors):
        raise V05LabelError("anchor_to_policy keys must be strings")
    policy_items = tuple(anchor_to_policy.values())
    if any(not isinstance(item, str) for item in policy_items):
        raise V05LabelError("anchor_to_policy values must be strings")
    policies = set(policy_items)
    _exact_mapping_keys(task_by_anchor, anchors, "task_by_anchor")
    _exact_mapping_keys(
        championization_admission_digest_by_anchor,
        anchors,
        "championization_admission_digest_by_anchor",
    )
    _exact_mapping_keys(
        policy_bundle_digest_by_policy,
        policies,
        "policy_bundle_digest_by_policy",
    )
    _exact_mapping_keys(
        execution_abi_digest_by_policy,
        policies,
        "execution_abi_digest_by_policy",
    )
    bindings = tuple(
        CertificateBinding(
            source_anchor_id=anchor,
            task_id=task_by_anchor[anchor],
            opaque_certified_policy_id=anchor_to_policy[anchor],
            policy_bundle_digest=policy_bundle_digest_by_policy[
                anchor_to_policy[anchor]
            ],
            championization_admission_digest=(
                championization_admission_digest_by_anchor[anchor]
            ),
            execution_abi_digest=execution_abi_digest_by_policy[
                anchor_to_policy[anchor]
            ],
        )
        for anchor in sorted(anchors)
    )
    return CertifiedPolicyManifest.from_records(
        bindings, expected_anchor_ids=expected_anchor_ids
    )


class CertificateResolver:
    """Immutable anchor-to-certified-policy resolver and score aggregator."""

    def __init__(self, manifest: CertifiedPolicyManifest) -> None:
        if not isinstance(manifest, CertifiedPolicyManifest):
            raise V05LabelError("resolver requires a CertifiedPolicyManifest")
        by_anchor = {item.source_anchor_id: item for item in manifest.bindings}
        by_policy: dict[str, list[str]] = {}
        for item in manifest.bindings:
            by_policy.setdefault(item.opaque_certified_policy_id, []).append(
                item.source_anchor_id
            )
        self.manifest = manifest
        self._by_anchor = MappingProxyType(by_anchor)
        self._policy_to_anchors = MappingProxyType(
            {
                policy: tuple(sorted(anchors))
                for policy, anchors in sorted(by_policy.items())
            }
        )

    @property
    def anchor_ids(self) -> tuple[str, ...]:
        return tuple(self._by_anchor)

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return tuple(self._policy_to_anchors)

    @property
    def anchor_to_policy(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                anchor: binding.opaque_certified_policy_id
                for anchor, binding in self._by_anchor.items()
            }
        )

    @property
    def policy_to_anchors(self) -> Mapping[str, tuple[str, ...]]:
        return self._policy_to_anchors

    def record_for_anchor(self, source_anchor_id: str) -> CertificateBinding:
        anchor = _canonical_string(source_anchor_id, "source_anchor_id")
        try:
            return self._by_anchor[anchor]
        except KeyError as error:
            raise V05LabelError(f"unknown source anchor: {anchor}") from error

    def policy_for_anchor(self, source_anchor_id: str) -> str:
        return self.record_for_anchor(source_anchor_id).opaque_certified_policy_id

    def anchors_for_policy(self, opaque_policy_id: str) -> tuple[str, ...]:
        policy = _opaque_policy_id(opaque_policy_id, "opaque_policy_id")
        try:
            return self._policy_to_anchors[policy]
        except KeyError as error:
            raise V05LabelError(f"unknown certified policy: {policy}") from error

    def aggregate_anchor_scores(
        self,
        anchor_scores: Mapping[str, Any],
        *,
        candidate_anchor_ids: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """Sum complete anchor score mass into certified-policy score mass.

        Summation is the fixed many-to-one resolver.  Callers supplying
        distances or logits must first transform them into additive,
        higher-is-better score mass; the resolver never consults target truth.
        ``candidate_anchor_ids`` applies a public candidate mask to the same
        complete score vector rather than accepting a separately scored subset.
        """

        if not isinstance(anchor_scores, Mapping):
            raise V05LabelError("anchor_scores must be an object")
        actual = set(anchor_scores)
        expected = set(self._by_anchor)
        if actual != expected:
            raise V05LabelError(
                "anchor score coverage mismatch; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        normalized: dict[str, float] = {}
        for anchor in self.anchor_ids:
            value = anchor_scores[anchor]
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise V05LabelError("anchor scores must be numeric")
            score = float(value)
            if not math.isfinite(score):
                raise V05LabelError("anchor scores must be finite")
            normalized[anchor] = score

        if candidate_anchor_ids is None:
            candidates = set(self.anchor_ids)
        else:
            candidate_items = tuple(
                _canonical_string(item, "candidate_anchor_id")
                for item in candidate_anchor_ids
            )
            if not candidate_items or len(candidate_items) != len(set(candidate_items)):
                raise V05LabelError("candidate_anchor_ids must be non-empty and unique")
            candidates = set(candidate_items)
            if not candidates.issubset(self._by_anchor):
                raise V05LabelError("candidate_anchor_ids contain an unknown anchor")
        return {
            policy: math.fsum(
                normalized[anchor] for anchor in anchors if anchor in candidates
            )
            for policy, anchors in self._policy_to_anchors.items()
            if candidates.intersection(anchors)
        }

    def rank_policies(
        self,
        anchor_scores: Mapping[str, Any],
        *,
        candidate_anchor_ids: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        """Return a deterministic descending policy ranking from anchor scores."""

        scores = self.aggregate_anchor_scores(
            anchor_scores, candidate_anchor_ids=candidate_anchor_ids
        )
        return tuple(sorted(scores, key=lambda policy: (-scores[policy], policy)))


# Concise aliases for artifact callers and older draft names.
CertificateManifest = CertifiedPolicyManifest
CertificateRecord = CertificateBinding


__all__ = [
    "CERTIFICATE_MANIFEST_SCHEMA",
    "CertificateBinding",
    "CertificateManifest",
    "CertificateRecord",
    "CertificateResolver",
    "CertifiedPolicyManifest",
    "V05LabelError",
    "project_certificate_manifest",
]
