"""Candidate-independent trace and representation assembly for v0.2.

This module contains no representation fitting or hyper-parameter search.  It
only validates frozen probe evidence, applies an already registered semantic
encoder, and assembles an :class:`EnvironmentSpec` through the stable RKME
primitives.  In particular, a confirmatory trace can be transformed but can
never be presented as representation-training or tuning evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..hashing import canonicalize, sha256_json, sha256_ndarrays
from ..probe.dataset import EpisodeDataset
from ..representation.canonicalizer import PackedEpisodeDataset
from ..rkme.empirical import build_empirical_kme, episode_balanced_weights
from ..rkme.gaussian import GaussianKernel
from ..rkme.reducer import ReducerConfig, reduce_kme
from .environment_spec import environment_spec_from_reduced
from .extensions.representation import (
    EncodedEpisodeDataset,
    SemanticEncoderMetadata,
    SemanticEncoderProtocol,
)
from .schemas import EnvironmentSpec


TraceRole = Literal[
    "representation_train",
    "representation_validation",
    "source_reference",
    "development_query",
    "confirmatory_query",
]
TracePairPurpose = Literal["paired_dynamics", "identity_audit", "shared_query_replay"]
MomentStatistic = Literal["mean", "std", "second_moment"]
MomentWeighting = Literal["transition_uniform", "episode_balanced"]

TRACE_ROLES = frozenset(
    {
        "representation_train",
        "representation_validation",
        "source_reference",
        "development_query",
        "confirmatory_query",
    }
)
TRACE_PAIR_PURPOSES = frozenset(
    {"paired_dynamics", "identity_audit", "shared_query_replay"}
)
MOMENT_STATISTICS = frozenset({"mean", "std", "second_moment"})
MOMENT_WEIGHTINGS = frozenset({"transition_uniform", "episode_balanced"})


class RepresentationContractError(ValueError):
    """Frozen representation evidence or configuration is inconsistent."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RepresentationContractError(
            f"{where} must be a non-empty canonical string"
        )
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise RepresentationContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise RepresentationContractError(
            f"{where} must be a SHA-256 digest"
        ) from error
    return result


def _readonly_vector(value: Any, *, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise RepresentationContractError(f"{where} must be a finite non-empty vector")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _packed_arrays_digest(dataset: PackedEpisodeDataset) -> str:
    return sha256_ndarrays(
        {
            "packed": dataset.packed,
            "episode_offsets": dataset.episode_offsets,
            "reset_seeds": dataset.reset_seeds,
            "probe_seeds": dataset.probe_seeds,
        }
    )


@dataclass(frozen=True)
class ProbeTraceView:
    """Immutable, role-bound view of one candidate-independent probe bank.

    ``probe_dataset_digest`` includes the role and seed namespace.  Therefore a
    recurrence query measured with independent seeds does not collapse onto a
    source-reference identity merely because its numeric transitions happen to
    match.
    """

    dataset: PackedEpisodeDataset
    role: TraceRole
    context_id: str
    bank_id: str
    seed_namespace: str
    probe_protocol_id: str
    measurement_protocol_id: str
    canonical_view_digest: str
    probe_rewards_included: bool
    probe_dataset_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, PackedEpisodeDataset):
            raise RepresentationContractError(
                "ProbeTraceView.dataset must be a PackedEpisodeDataset"
            )
        if self.role not in TRACE_ROLES:
            raise RepresentationContractError(f"unsupported trace role {self.role!r}")
        for name in ("context_id", "bank_id", "seed_namespace"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "probe_protocol_id",
            "measurement_protocol_id",
            "canonical_view_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if type(self.probe_rewards_included) is not bool:
            raise RepresentationContractError(
                "probe_rewards_included must be boolean"
            )
        payload = {
            "schema": "policy-learnware.v02-probe-trace-view.v0",
            "arrays_digest": _packed_arrays_digest(self.dataset),
            "role": self.role,
            "context_id": self.context_id,
            "bank_id": self.bank_id,
            "seed_namespace": self.seed_namespace,
            "probe_protocol_id": self.probe_protocol_id,
            "measurement_protocol_id": self.measurement_protocol_id,
            "canonical_view_digest": self.canonical_view_digest,
            "probe_rewards_included": self.probe_rewards_included,
        }
        expected = sha256_json(payload)
        if self.probe_dataset_digest is None:
            object.__setattr__(self, "probe_dataset_digest", expected)
        else:
            actual = _digest(self.probe_dataset_digest, "probe_dataset_digest")
            if actual != expected:
                raise RepresentationContractError(
                    "probe_dataset_digest does not match the trace payload"
                )
            object.__setattr__(self, "probe_dataset_digest", actual)

    @property
    def stage(self) -> str:
        if self.role == "confirmatory_query":
            return "confirmatory"
        if self.role in {"representation_validation", "development_query"}:
            return "development_discovery"
        return "source"

    @property
    def episode_count(self) -> int:
        return self.dataset.episode_count

    def require_encoder_fit_evidence(self) -> None:
        if self.role != "representation_train":
            raise RepresentationContractError(
                f"trace role {self.role!r} cannot fit an encoder"
            )

    def require_representation_validation_evidence(self) -> None:
        if self.role != "representation_validation":
            raise RepresentationContractError(
                f"trace role {self.role!r} cannot tune a representation"
            )

    def require_query_evidence(self) -> None:
        if self.role not in {"development_query", "confirmatory_query"}:
            raise RepresentationContractError(
                f"trace role {self.role!r} is not target query evidence"
            )

    def prefix(self, episode_count: int) -> "ProbeTraceView":
        if type(episode_count) is not int or not 1 <= episode_count <= self.episode_count:
            raise RepresentationContractError(
                "trace prefix episode_count lies outside the bank"
            )
        stop = int(self.dataset.episode_offsets[episode_count])
        prefix = PackedEpisodeDataset(
            packed=self.dataset.packed[:stop],
            episode_offsets=self.dataset.episode_offsets[: episode_count + 1],
            reset_seeds=self.dataset.reset_seeds[:episode_count],
            probe_seeds=self.dataset.probe_seeds[:episode_count],
            task=self.dataset.task,
            schema_fingerprint=self.dataset.schema_fingerprint,
        )
        return ProbeTraceView(
            dataset=prefix,
            role=self.role,
            context_id=self.context_id,
            bank_id=f"{self.bank_id}:prefix-{episode_count}",
            seed_namespace=self.seed_namespace,
            probe_protocol_id=self.probe_protocol_id,
            measurement_protocol_id=self.measurement_protocol_id,
            canonical_view_digest=self.canonical_view_digest,
            probe_rewards_included=self.probe_rewards_included,
        )


def validate_trace_partition_disjointness(
    *,
    representation_train: Sequence[ProbeTraceView],
    representation_validation: Sequence[ProbeTraceView],
    development_signal: Sequence[ProbeTraceView],
    confirmatory: Sequence[ProbeTraceView] = (),
) -> Mapping[str, tuple[str, ...]]:
    """Reject context/data reuse across representation evidence partitions."""

    groups = {
        "representation_train": tuple(representation_train),
        "representation_validation": tuple(representation_validation),
        "development_query": tuple(development_signal),
        "confirmatory_query": tuple(confirmatory),
    }
    seen_contexts: dict[str, str] = {}
    seen_arrays: dict[str, str] = {}
    seen_namespaces: dict[str, str] = {}
    result: dict[str, tuple[str, ...]] = {}
    for expected_role, traces in groups.items():
        digests: list[str] = []
        local_digests: set[str] = set()
        for trace in traces:
            if not isinstance(trace, ProbeTraceView) or trace.role != expected_role:
                raise RepresentationContractError(
                    f"{expected_role} partition contains a trace with another role"
                )
            assert trace.probe_dataset_digest is not None
            if trace.probe_dataset_digest in local_digests:
                raise RepresentationContractError(
                    f"duplicate trace in {expected_role} partition"
                )
            local_digests.add(trace.probe_dataset_digest)
            arrays_digest = _packed_arrays_digest(trace.dataset)
            for value, seen, label in (
                (trace.context_id, seen_contexts, "context_id"),
                (arrays_digest, seen_arrays, "probe arrays"),
                (trace.seed_namespace, seen_namespaces, "seed namespace"),
            ):
                previous = seen.get(value)
                if previous is not None and previous != expected_role:
                    raise RepresentationContractError(
                        f"{label} overlaps {previous} and {expected_role} partitions"
                    )
                seen[value] = expected_role
            digests.append(trace.probe_dataset_digest)
        result[expected_role] = tuple(digests)
    return MappingProxyType(result)


@dataclass(frozen=True)
class PairedTraceAudit:
    purpose: TracePairPurpose
    left_digest: str
    right_digest: str
    offsets_paired: bool
    reset_seeds_paired: bool
    probe_seeds_paired: bool
    actions_paired: bool
    full_trajectory_identity: bool
    passed: bool

    def __post_init__(self) -> None:
        if self.purpose not in TRACE_PAIR_PURPOSES:
            raise RepresentationContractError(
                f"unsupported trace-pair purpose {self.purpose!r}"
            )
        object.__setattr__(self, "left_digest", _digest(self.left_digest, "left_digest"))
        object.__setattr__(self, "right_digest", _digest(self.right_digest, "right_digest"))
        for name in (
            "offsets_paired",
            "reset_seeds_paired",
            "probe_seeds_paired",
            "actions_paired",
            "full_trajectory_identity",
            "passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise RepresentationContractError(f"{name} must be boolean")

    def require(self) -> "PairedTraceAudit":
        if not self.passed:
            raise RepresentationContractError(
                f"paired trace audit failed for purpose {self.purpose!r}"
            )
        return self


def audit_paired_episode_traces(
    left: EpisodeDataset,
    right: EpisodeDataset,
    *,
    purpose: TracePairPurpose,
) -> PairedTraceAudit:
    """Audit exact seed/action pairing without confusing it with dynamics identity."""

    if not isinstance(left, EpisodeDataset) or not isinstance(right, EpisodeDataset):
        raise RepresentationContractError(
            "paired trace audit requires two EpisodeDataset objects"
        )
    if purpose not in TRACE_PAIR_PURPOSES:
        raise RepresentationContractError(f"unsupported trace-pair purpose {purpose!r}")
    offsets = np.array_equal(left.episode_offsets, right.episode_offsets)
    reset = np.array_equal(left.reset_seeds, right.reset_seeds)
    probe = np.array_equal(left.probe_seeds, right.probe_seeds)
    actions = np.array_equal(left.action, right.action)
    full_identity = all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in (
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
            "episode_offsets",
            "reset_seeds",
            "probe_seeds",
        )
    )
    paired = offsets and reset and probe and actions
    passed = {
        "paired_dynamics": paired,
        "identity_audit": full_identity,
        "shared_query_replay": left.digest == right.digest,
    }[purpose]
    return PairedTraceAudit(
        purpose=purpose,
        left_digest=left.digest,
        right_digest=right.digest,
        offsets_paired=offsets,
        reset_seeds_paired=reset,
        probe_seeds_paired=probe,
        actions_paired=actions,
        full_trajectory_identity=full_identity,
        passed=bool(passed),
    )


@dataclass(frozen=True)
class RawMomentContract:
    """Explicit B3a feature definition; no scientific default is supplied."""

    canonical_view_digest: str
    statistics: tuple[MomentStatistic, ...]
    weighting: MomentWeighting

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canonical_view_digest",
            _digest(self.canonical_view_digest, "canonical_view_digest"),
        )
        statistics = tuple(self.statistics)
        if (
            not statistics
            or len(set(statistics)) != len(statistics)
            or any(item not in MOMENT_STATISTICS for item in statistics)
        ):
            raise RepresentationContractError(
                "statistics must be a non-empty duplicate-free supported tuple"
            )
        if self.weighting not in MOMENT_WEIGHTINGS:
            raise RepresentationContractError(
                f"unsupported raw-moment weighting {self.weighting!r}"
            )
        object.__setattr__(self, "statistics", statistics)

    @property
    def feature_protocol_id(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v02-raw-moment-feature.v0",
                "canonical_view_digest": self.canonical_view_digest,
                "statistics": list(self.statistics),
                "weighting": self.weighting,
            }
        )


@dataclass(frozen=True)
class TraceFeatureVector:
    values: np.ndarray
    feature_protocol_id: str
    probe_dataset_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly_vector(self.values, where="values"))
        object.__setattr__(
            self,
            "feature_protocol_id",
            _digest(self.feature_protocol_id, "feature_protocol_id"),
        )
        object.__setattr__(
            self,
            "probe_dataset_digest",
            _digest(self.probe_dataset_digest, "probe_dataset_digest"),
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v02-trace-feature-vector.v0",
                "values_digest": sha256_ndarrays({"values": self.values}),
                "feature_protocol_id": self.feature_protocol_id,
                "probe_dataset_digest": self.probe_dataset_digest,
            }
        )


def raw_moment_feature(
    trace: ProbeTraceView,
    contract: RawMomentContract,
) -> TraceFeatureVector:
    if trace.canonical_view_digest != contract.canonical_view_digest:
        raise RepresentationContractError(
            "raw-moment contract and trace canonical views differ"
        )
    points = np.asarray(trace.dataset.packed, dtype=np.float64)
    if contract.weighting == "transition_uniform":
        weights = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    else:
        weights = episode_balanced_weights(trace.dataset.episode_offsets)
    mean = np.sum(points * weights[:, None], axis=0)
    second = np.sum(np.square(points) * weights[:, None], axis=0)
    variance = np.maximum(second - np.square(mean), 0.0)
    values = {
        "mean": mean,
        "std": np.sqrt(variance),
        "second_moment": second,
    }
    return TraceFeatureVector(
        values=np.concatenate([values[name] for name in contract.statistics]),
        feature_protocol_id=contract.feature_protocol_id,
        probe_dataset_digest=str(trace.probe_dataset_digest),
    )


@dataclass(frozen=True)
class RepresentationBuildContract:
    """All numeric choices required to build one frozen EnvironmentSpec."""

    representation_protocol_id: str
    measurement_protocol_id: str
    canonical_view_digest: str
    probe_rewards_included: bool
    kernel_bandwidth: float
    batch_size: int
    block_size: int
    computation_backend: Literal["numpy", "jax"]
    reducer_config: ReducerConfig

    def __post_init__(self) -> None:
        for name in (
            "representation_protocol_id",
            "measurement_protocol_id",
            "canonical_view_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if type(self.probe_rewards_included) is not bool:
            raise RepresentationContractError(
                "probe_rewards_included must be boolean"
            )
        bandwidth = float(self.kernel_bandwidth)
        if not math.isfinite(bandwidth) or bandwidth <= 0.0:
            raise RepresentationContractError(
                "kernel_bandwidth must be finite and positive"
            )
        object.__setattr__(self, "kernel_bandwidth", bandwidth)
        for name in ("batch_size", "block_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise RepresentationContractError(f"{name} must be a positive integer")
        if self.computation_backend not in {"numpy", "jax"}:
            raise RepresentationContractError(
                "computation_backend must be 'numpy' or 'jax'"
            )
        if not isinstance(self.reducer_config, ReducerConfig):
            raise RepresentationContractError(
                "reducer_config must be an explicit ReducerConfig"
            )

    @property
    def reducer_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v02-reducer-contract.v0",
                "config": canonicalize(asdict(self.reducer_config)),
            }
        )

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v02-representation-build-contract.v0",
                "representation_protocol_id": self.representation_protocol_id,
                "measurement_protocol_id": self.measurement_protocol_id,
                "canonical_view_digest": self.canonical_view_digest,
                "probe_rewards_included": self.probe_rewards_included,
                "kernel_bandwidth": self.kernel_bandwidth,
                "batch_size": self.batch_size,
                "block_size": self.block_size,
                "computation_backend": self.computation_backend,
                "reducer_digest": self.reducer_digest,
            }
        )


def build_environment_spec(
    trace: ProbeTraceView,
    encoder: SemanticEncoderProtocol,
    contract: RepresentationBuildContract,
) -> EnvironmentSpec:
    """Encode a frozen trace and construct its digest-bound EnvironmentSpec."""

    if not isinstance(trace, ProbeTraceView):
        raise RepresentationContractError("trace must be a ProbeTraceView")
    if not isinstance(encoder, SemanticEncoderProtocol):
        raise RepresentationContractError(
            "encoder does not implement SemanticEncoderProtocol"
        )
    metadata = encoder.metadata
    if not isinstance(metadata, SemanticEncoderMetadata):
        raise RepresentationContractError("encoder metadata has the wrong type")
    checks = {
        "representation_protocol_id": (
            metadata.representation_protocol_id,
            contract.representation_protocol_id,
        ),
        "canonical_view_digest": (
            metadata.canonical_event_view_digest,
            contract.canonical_view_digest,
        ),
        "trace canonical_view_digest": (
            trace.canonical_view_digest,
            contract.canonical_view_digest,
        ),
        "measurement_protocol_id": (
            trace.measurement_protocol_id,
            contract.measurement_protocol_id,
        ),
        "probe_rewards_included": (
            trace.probe_rewards_included,
            contract.probe_rewards_included,
        ),
    }
    mismatches = {
        name: values for name, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        raise RepresentationContractError(
            f"representation build bindings differ: {mismatches}"
        )
    encoded = encoder.encode(trace.dataset, batch_size=contract.batch_size)
    if not isinstance(encoded, EncodedEpisodeDataset):
        raise RepresentationContractError(
            "encoder returned a non-EncodedEpisodeDataset"
        )
    if encoded.representation_protocol_id != contract.representation_protocol_id:
        raise RepresentationContractError(
            "encoded events are bound to another representation protocol"
        )
    kernel = GaussianKernel(contract.kernel_bandwidth)
    empirical = build_empirical_kme(
        encoded,
        kernel,
        protocol_id=contract.representation_protocol_id,
        dataset_digest=str(trace.probe_dataset_digest),
        source_task=trace.dataset.task,
        block_size=contract.block_size,
        computation_backend=contract.computation_backend,
    )
    reduced = reduce_kme(empirical, contract.reducer_config)
    if reduced.supports.shape[0] != contract.reducer_config.support_budget:
        raise RepresentationContractError(
            "trace has fewer usable events than the frozen support budget"
        )
    return environment_spec_from_reduced(
        reduced,
        reducer_digest=contract.reducer_digest,
        representation_protocol_id=contract.representation_protocol_id,
        measurement_protocol_id=contract.measurement_protocol_id,
        canonical_view_digest=contract.canonical_view_digest,
        probe_dataset_digest=str(trace.probe_dataset_digest),
    )


__all__ = [
    "MOMENT_STATISTICS",
    "MOMENT_WEIGHTINGS",
    "PairedTraceAudit",
    "ProbeTraceView",
    "RawMomentContract",
    "RepresentationBuildContract",
    "RepresentationContractError",
    "TRACE_PAIR_PURPOSES",
    "TRACE_ROLES",
    "TraceFeatureVector",
    "TracePairPurpose",
    "TraceRole",
    "audit_paired_episode_traces",
    "build_environment_spec",
    "raw_moment_feature",
    "validate_trace_partition_disjointness",
]
