"""Construction, validation, indexing, and RKME distance for EnvironmentSpec."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from ..rkme.gaussian import GaussianKernel
from ..rkme.reducer import ReducedRKME
from .schemas import EnvironmentSpec


DistanceForm = Literal["mmd", "mmd2"]


@dataclass(frozen=True)
class EnvironmentSpecDistance:
    distance: float
    squared_distance: float
    raw_squared_distance: float
    clamped: bool
    distance_form: DistanceForm

    @property
    def value(self) -> float:
        return self.distance if self.distance_form == "mmd" else self.squared_distance

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "squared_distance": self.squared_distance,
            "raw_squared_distance": self.raw_squared_distance,
            "clamped": self.clamped,
            "distance_form": self.distance_form,
            "value": self.value,
        }


def environment_spec_from_reduced(
    reduced: ReducedRKME,
    *,
    reducer_digest: str,
    representation_protocol_id: str,
    measurement_protocol_id: str,
    canonical_view_digest: str,
    probe_dataset_digest: str,
) -> EnvironmentSpec:
    """Adapt an existing reduced KME without weakening its numeric checks."""

    if reduced.supports.shape[0] != reduced.beta.shape[0]:
        raise ValueError("reduced RKME supports and beta are misaligned")
    total = float(np.sum(reduced.beta))
    if np.any(reduced.beta < 0.0) or not np.isfinite(total) or total <= 0.0:
        raise ValueError("EnvironmentSpec requires non-negative simplex RKME weights")
    beta = np.asarray(reduced.beta, dtype=np.float64) / total
    return EnvironmentSpec(
        supports=reduced.supports,
        beta=beta,
        empirical_norm2=reduced.empirical_norm2,
        rkme_norm2=reduced.rkme_norm2,
        reconstruction_error=reduced.reduction_error,
        reducer_digest=reducer_digest,
        support_budget=int(reduced.supports.shape[0]),
        latent_dim=int(reduced.supports.shape[1]),
        representation_protocol_id=representation_protocol_id,
        measurement_protocol_id=measurement_protocol_id,
        canonical_view_digest=canonical_view_digest,
        kernel_bandwidth=reduced.bandwidth,
        probe_dataset_digest=probe_dataset_digest,
    )


def environment_spec_distance(
    left: EnvironmentSpec,
    right: EnvironmentSpec,
    *,
    distance_form: DistanceForm,
    negative_tolerance: float = 1.0e-8,
) -> EnvironmentSpecDistance:
    """Compute reduced-to-reduced MMD under one frozen representation/kernel."""

    if distance_form not in {"mmd", "mmd2"}:
        raise ValueError("distance_form must be 'mmd' or 'mmd2'")
    for name in (
        "representation_protocol_id",
        "measurement_protocol_id",
        "canonical_view_digest",
    ):
        if getattr(left, name) != getattr(right, name):
            raise ValueError(f"EnvironmentSpecs have different {name}")
    if left.latent_dim != right.latent_dim:
        raise ValueError("EnvironmentSpecs have different latent dimensions")
    if not math.isclose(
        left.kernel_bandwidth, right.kernel_bandwidth, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("EnvironmentSpecs have different kernel bandwidths")
    if not math.isfinite(float(negative_tolerance)) or negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be finite and non-negative")

    kernel = GaussianKernel(left.kernel_bandwidth)
    cross = float(left.beta @ kernel.gram(left.supports, right.supports) @ right.beta)
    raw = float(left.rkme_norm2 - 2.0 * cross + right.rkme_norm2)
    scale = max(1.0, abs(left.rkme_norm2), abs(right.rkme_norm2), abs(2.0 * cross))
    if raw < -float(negative_tolerance) * scale:
        raise ArithmeticError(f"EnvironmentSpec MMD squared is materially negative ({raw})")
    squared = max(raw, 0.0)
    return EnvironmentSpecDistance(
        distance=float(math.sqrt(squared)),
        squared_distance=squared,
        raw_squared_distance=raw,
        clamped=raw < 0.0,
        distance_form=distance_form,
    )


@dataclass(frozen=True)
class RepresentationIndexEntry:
    opaque_id: str
    environment_spec: EnvironmentSpec

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_id, str) or not self.opaque_id:
            raise ValueError("opaque_id must be non-empty")
        if not isinstance(self.environment_spec, EnvironmentSpec):
            raise ValueError("environment_spec must be an EnvironmentSpec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "opaque_id": self.opaque_id,
            "environment_spec": self.environment_spec.to_dict(),
        }


@dataclass(frozen=True)
class RepresentationIndex:
    policy_market_id: str
    representation_protocol_id: str
    entries: Mapping[str, RepresentationIndexEntry]
    representation_index_id: str | None = None

    def __post_init__(self) -> None:
        entries = dict(self.entries)
        if not entries:
            raise ValueError("representation index cannot be empty")
        for opaque_id, entry in entries.items():
            if opaque_id != entry.opaque_id:
                raise ValueError("representation index key must match entry.opaque_id")
            if entry.environment_spec.representation_protocol_id != self.representation_protocol_id:
                raise ValueError("EnvironmentSpec representation protocol mismatch")
        object.__setattr__(self, "entries", MappingProxyType(entries))
        expected = sha256_json(
            {
                "schema": "policy-learnware.v02-representation-index.v0",
                "policy_market_id": self.policy_market_id,
                "representation_protocol_id": self.representation_protocol_id,
                "entries": {
                    key: entry.environment_spec.environment_spec_digest
                    for key, entry in sorted(entries.items())
                },
            }
        )
        if self.representation_index_id is None:
            object.__setattr__(self, "representation_index_id", expected)
        elif self.representation_index_id != expected:
            raise ValueError("representation_index_id does not match index contents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-representation-index.v0",
            "representation_index_id": self.representation_index_id,
            "policy_market_id": self.policy_market_id,
            "representation_protocol_id": self.representation_protocol_id,
            "entries": {key: value.to_dict() for key, value in sorted(self.entries.items())},
        }


def source_only_median_scale(
    specs: Mapping[str, EnvironmentSpec],
    *,
    partitions: Mapping[str, Sequence[str]],
    distance_form: DistanceForm,
    zero_fallback: float | None = None,
) -> Mapping[str, float]:
    """Derive per-partition sigma using only non-zero source-source distances."""

    if not partitions:
        raise ValueError("at least one source partition is required")
    result: dict[str, float] = {}
    for partition, opaque_ids in sorted(partitions.items()):
        ids = tuple(opaque_ids)
        if len(ids) != len(set(ids)) or any(item not in specs for item in ids):
            raise ValueError(f"invalid source IDs for partition {partition!r}")
        values: list[float] = []
        for left_index, left_id in enumerate(ids):
            for right_id in ids[left_index + 1 :]:
                value = environment_spec_distance(
                    specs[left_id], specs[right_id], distance_form=distance_form
                ).value
                if value > np.finfo(np.float64).eps:
                    values.append(value)
        if values:
            sigma = float(np.median(np.asarray(values, dtype=np.float64)))
        elif zero_fallback is not None and math.isfinite(zero_fallback) and zero_fallback > 0.0:
            sigma = float(zero_fallback)
        else:
            raise ValueError(f"partition {partition!r} has no non-zero source-pair distance")
        result[partition] = sigma
    return MappingProxyType(result)


__all__ = [
    "DistanceForm",
    "EnvironmentSpecDistance",
    "RepresentationIndex",
    "RepresentationIndexEntry",
    "environment_spec_distance",
    "environment_spec_from_reduced",
    "source_only_median_scale",
]
