"""Gate-0-to-live environment instance bindings for collection and oracle use."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from ..hashing import canonical_json_bytes, canonicalize
from ..probe.dataset import EpisodeDataset
from .schemas import EnvironmentInstanceRecord


class LiveInstanceBindingError(RuntimeError):
    """A freshly created adapter differs from its Gate-0-audited instance."""


LIVE_INSTANCE_BINDING_SCHEMA = "policy-learnware.v01-live-instance-binding.v0"
LIVE_PROBE_FINITE_EVIDENCE_SCHEMA = (
    "policy-learnware.v01-live-probe-finite-evidence.v0"
)
COLLECTION_BINDING_ATTESTATION_SCHEMA = (
    "policy-learnware.v01-private-collection-attestation.v1"
)


_FINITE_FIELDS = {
    "schema",
    "episode_count",
    "steps_per_episode",
    "all_finite",
    "no_early_termination",
    "reward_minimum",
    "reward_maximum",
    "passed",
    "reason",
}


def _verified_finite_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _FINITE_FIELDS:
        raise LiveInstanceBindingError(
            "audited finite/termination summary has missing or unknown fields"
        )
    result = canonicalize(value)
    if result["schema"] != "policy-learnware.v01-finite-termination-audit.v0":
        raise LiveInstanceBindingError("unsupported finite/termination audit schema")
    if (
        type(result["episode_count"]) is not int
        or result["episode_count"] <= 0
        or type(result["steps_per_episode"]) is not int
        or result["steps_per_episode"] <= 0
    ):
        raise LiveInstanceBindingError("audited finite/termination counts are invalid")
    for field in ("all_finite", "no_early_termination", "passed"):
        if type(result[field]) is not bool:
            raise LiveInstanceBindingError(f"audited finite field {field} must be bool")
    for field in ("reward_minimum", "reward_maximum"):
        if (
            isinstance(result[field], bool)
            or not isinstance(result[field], (int, float))
            or not math.isfinite(float(result[field]))
        ):
            raise LiveInstanceBindingError(f"audited finite field {field} is non-finite")
    if (
        result["passed"] is not True
        or result["all_finite"] is not True
        or result["no_early_termination"] is not True
        or result["reason"] is not None
    ):
        raise LiveInstanceBindingError(
            "Gate-0 finite/termination evidence is not an unconditional pass"
        )
    return result


def _sha256(value: Any, where: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64:
        raise LiveInstanceBindingError(f"{where} is not SHA-256")
    try:
        int(digest, 16)
    except ValueError as error:
        raise LiveInstanceBindingError(f"{where} is not SHA-256") from error
    return digest


@dataclass(frozen=True)
class LiveInstanceBinding:
    """Exact equality proof between a fresh adapter and the Gate-0 record."""

    audited_record: EnvironmentInstanceRecord
    live_record: EnvironmentInstanceRecord
    audited_instance_record_sha256: str

    @property
    def verified_instance_digest(self) -> str:
        return self.live_record.digest

    @property
    def passed(self) -> bool:
        return self.audited_record.to_dict() == self.live_record.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIVE_INSTANCE_BINDING_SCHEMA,
            "variant_id": self.live_record.variant_id,
            "passed": self.passed,
            "exact_record_identity": self.passed,
            "audited_finite_gate_passed": True,
            "audited_instance_record_sha256": self.audited_instance_record_sha256,
            "audited_instance_digest": self.audited_record.digest,
            "live_instance_digest": self.live_record.digest,
            "verified_instance_digest": self.verified_instance_digest,
        }


def verify_live_instance_binding(
    adapter: Any,
    audited_payload: Mapping[str, Any],
    *,
    audited_instance_record_sha256: str,
) -> LiveInstanceBinding:
    """Recreate and compare the complete typed instance record fail-closed."""

    try:
        audited = EnvironmentInstanceRecord.from_dict(audited_payload)
    except (TypeError, ValueError) as error:
        raise LiveInstanceBindingError(f"invalid Gate-0 instance record: {error}") from error
    finite = _verified_finite_summary(audited.finite_termination_audit_summary)
    try:
        live_raw = adapter.create_instance_record(
            finite_termination_audit_summary=finite
        )
        live = (
            live_raw
            if isinstance(live_raw, EnvironmentInstanceRecord)
            else EnvironmentInstanceRecord.from_dict(live_raw)
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise LiveInstanceBindingError(
            f"cannot construct a typed live instance record: {error}"
        ) from error
    if live.to_dict() != audited.to_dict():
        left = audited.to_dict()
        right = live.to_dict()
        differing = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        raise LiveInstanceBindingError(
            "fresh environment differs from Gate-0 instance record in fields: "
            + ", ".join(differing)
        )
    digest = _sha256(
        audited_instance_record_sha256, "audited instance file digest"
    )
    return LiveInstanceBinding(audited, live, digest)


def probe_dataset_finite_evidence(
    dataset: EpisodeDataset,
    *,
    expected_episode_count: int,
    expected_horizon: int,
) -> dict[str, Any]:
    """Derive current live finite/termination evidence from collected rows."""

    lengths = np.diff(dataset.episode_offsets)
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            dataset.observation,
            dataset.action,
            dataset.reward,
            dataset.next_observation,
        )
    )
    done = np.logical_or(dataset.terminated, dataset.truncated)
    no_early = True
    final_done = True
    for start, stop in zip(
        dataset.episode_offsets[:-1], dataset.episode_offsets[1:], strict=True
    ):
        no_early &= not bool(np.any(done[int(start) : int(stop) - 1]))
        final_done &= bool(done[int(stop) - 1])
    passed = bool(
        dataset.episode_count == int(expected_episode_count)
        and np.all(lengths == int(expected_horizon))
        and finite
        and no_early
        and final_done
    )
    return {
        "schema": LIVE_PROBE_FINITE_EVIDENCE_SCHEMA,
        "passed": passed,
        "episode_count": dataset.episode_count,
        "expected_episode_count": int(expected_episode_count),
        "steps_per_episode": [int(value) for value in lengths],
        "expected_horizon": int(expected_horizon),
        "all_finite": bool(finite),
        "no_early_termination": bool(no_early),
        "every_episode_closed_at_horizon": bool(final_done),
    }


def build_collection_binding_attestation(
    binding: LiveInstanceBinding,
    dataset: EpisodeDataset,
    *,
    bank: int,
    expected_episode_count: int,
    expected_horizon: int,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    """Build the private attestation from live evidence without pass input."""

    if type(bank) is not int or bank < 0:
        raise LiveInstanceBindingError("collection bank must be a non-negative integer")
    run_digest = _sha256(run_manifest_sha256, "run manifest digest")
    finite = probe_dataset_finite_evidence(
        dataset,
        expected_episode_count=expected_episode_count,
        expected_horizon=expected_horizon,
    )
    if not binding.passed or not finite["passed"]:
        raise LiveInstanceBindingError(
            "cannot attest a collection without exact instance and live finite evidence"
        )
    return {
        "schema": COLLECTION_BINDING_ATTESTATION_SCHEMA,
        "variant_id": binding.live_record.variant_id,
        "bank": int(bank),
        "dataset_digest": dataset.digest,
        "verified_instance_digest": binding.verified_instance_digest,
        "audited_instance_record_sha256": binding.audited_instance_record_sha256,
        "run_manifest_sha256": run_digest,
        "live_instance_binding": binding.to_dict(),
        "live_collection_finite_evidence": finite,
        "passed": True,
    }


def verify_collection_binding_attestation(
    value: Mapping[str, Any],
    *,
    audited_record: EnvironmentInstanceRecord,
    audited_instance_record_sha256: str,
    dataset: EpisodeDataset,
    bank: int,
    expected_episode_count: int,
    expected_horizon: int,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    """Rebuild a private collection attestation from persisted source bytes.

    This verifier never accepts the attestation's ``passed`` members as
    evidence.  It derives the only valid payload from the audited instance,
    the raw dataset, and immutable digests, then requires canonical byte-level
    equality.  Boolean/integer aliases therefore cannot pass Python equality.
    """

    finite_summary = _verified_finite_summary(
        audited_record.finite_termination_audit_summary
    )
    file_digest = _sha256(
        audited_instance_record_sha256, "audited instance file digest"
    )
    run_digest = _sha256(run_manifest_sha256, "run manifest digest")
    if type(bank) is not int or bank < 0:
        raise LiveInstanceBindingError("collection bank must be a non-negative integer")
    # The expected live record is exactly the typed record persisted by Gate 0.
    # Only collect-probes can produce this statement: it reconstructs the
    # adapter and compares that fresh record before collecting any transition.
    expected_binding = {
        "schema": LIVE_INSTANCE_BINDING_SCHEMA,
        "variant_id": audited_record.variant_id,
        "passed": True,
        "exact_record_identity": True,
        "audited_finite_gate_passed": bool(
            finite_summary["passed"]
            and finite_summary["all_finite"]
            and finite_summary["no_early_termination"]
        ),
        "audited_instance_record_sha256": file_digest,
        "audited_instance_digest": audited_record.digest,
        "live_instance_digest": audited_record.digest,
        "verified_instance_digest": audited_record.digest,
    }
    expected_finite = probe_dataset_finite_evidence(
        dataset,
        expected_episode_count=expected_episode_count,
        expected_horizon=expected_horizon,
    )
    if expected_finite["passed"] is not True:
        raise LiveInstanceBindingError(
            "persisted probe dataset fails live finite/termination requirements"
        )
    expected = {
        "schema": COLLECTION_BINDING_ATTESTATION_SCHEMA,
        "variant_id": audited_record.variant_id,
        "bank": bank,
        "dataset_digest": dataset.digest,
        "verified_instance_digest": audited_record.digest,
        "audited_instance_record_sha256": file_digest,
        "run_manifest_sha256": run_digest,
        "live_instance_binding": expected_binding,
        "live_collection_finite_evidence": expected_finite,
        "passed": True,
    }
    if not isinstance(value, Mapping):
        raise LiveInstanceBindingError("collection binding attestation is not an object")
    try:
        observed_bytes = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise LiveInstanceBindingError(
            f"collection binding attestation is not canonical: {error}"
        ) from error
    if observed_bytes != canonical_json_bytes(expected):
        observed_keys = set(value)
        expected_keys = set(expected)
        differing = sorted(
            key
            for key in observed_keys | expected_keys
            if value.get(key) != expected.get(key)
            or type(value.get(key)) is not type(expected.get(key))
        )
        raise LiveInstanceBindingError(
            "private collection binding differs from executable evidence in fields: "
            + ", ".join(differing)
        )
    return expected


__all__ = [
    "COLLECTION_BINDING_ATTESTATION_SCHEMA",
    "LIVE_INSTANCE_BINDING_SCHEMA",
    "LIVE_PROBE_FINITE_EVIDENCE_SCHEMA",
    "LiveInstanceBinding",
    "LiveInstanceBindingError",
    "build_collection_binding_attestation",
    "probe_dataset_finite_evidence",
    "verify_collection_binding_attestation",
    "verify_live_instance_binding",
]
