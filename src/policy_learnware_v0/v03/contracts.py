"""Strict v0.3 representation artifacts and cache-key boundaries.

The historical v0.2 :class:`EnvironmentSpec` remains the reduced source
payload.  v0.3 wraps it with a source role and introduces a distinct empirical
query artifact.  Query reduction is available only through an explicit,
separately identified contract; no builder silently falls back between modes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np

from ..hashing import canonicalize, sha256_json, sha256_ndarrays
from ..rkme.empirical import (
    EmpiricalKME,
    _has_exact_norm2_attestation,
    blockwise_weighted_kernel_sum,
    blockwise_weighted_self_kernel_sum,
    build_empirical_kme,
    episode_balanced_weights,
)
from ..rkme.gaussian import GaussianKernel
from ..rkme.reducer import ReducedRKME, ReducerConfig, reduce_kme
from ..v02.schemas import EnvironmentSpec


class V03ContractError(ValueError):
    """A v0.3 artifact is malformed or crosses a frozen role boundary."""


SpecRole = Literal["SOURCE_REDUCED", "QUERY_EMPIRICAL", "QUERY_REDUCED"]
SPEC_ROLES = frozenset({"SOURCE_REDUCED", "QUERY_EMPIRICAL", "QUERY_REDUCED"})

SEMANTIC_CACHE_KEY_SCHEMA = "policy-learnware.v03-semantic-cache-key.v0"
SEMANTIC_CACHE_RECORD_SCHEMA = "policy-learnware.v03-semantic-cache-record.v0"
SEMANTIC_CACHE_SLICE_SCHEMA = "policy-learnware.v03-semantic-cache-slice.v0"
SPEC_KEY_SCHEMA = "policy-learnware.v03-spec-key.v0"
RANKING_KEY_SCHEMA = "policy-learnware.v03-ranking-key.v0"
SOURCE_REDUCED_SPEC_SCHEMA = "policy-learnware.v03-source-reduced-spec.v0"
EMPIRICAL_QUERY_SPEC_SCHEMA = "policy-learnware.v03-empirical-query-spec.v0"
REDUCED_QUERY_SPEC_SCHEMA = "policy-learnware.v03-reduced-query-spec.v0"
SOURCE_INDEX_SCHEMA = "policy-learnware.v03-source-representation-index.v0"

EMPIRICAL_QUERY_MARKER = sha256_json(
    {
        "schema": "policy-learnware.v03-query-spec-marker.v0",
        "role": "QUERY_EMPIRICAL",
        "reducer": None,
    }
)
EPISODE_BALANCED_WEIGHTING_DIGEST = sha256_json(
    {
        "schema": "policy-learnware.v03-sample-weighting-protocol.v0",
        "family": "episode_balanced",
        "formula": "alpha[n,h]=1/(N*H_n)",
        "implementation": "policy_learnware_v0.rkme.empirical:episode_balanced_weights",
    }
)
GAUSSIAN_KERNEL_EVALUATOR_DIGEST = sha256_json(
    {
        "schema": "policy-learnware.v03-kernel-evaluator-protocol.v0",
        "family": "gaussian_rbf",
        "implementation": "policy_learnware_v0.rkme.gaussian:GaussianKernel/v0",
    }
)
FLOAT64_MATHEMATICAL_DTYPE_DIGEST = sha256_json(
    {
        "schema": "policy-learnware.v03-mathematical-dtype-protocol.v0",
        "numpy_dtype": np.dtype(np.float64).str,
        "kernel_accumulation": "float64",
    }
)
QUERY_EMPIRICAL_PROTOCOL_ID = sha256_json(
    {
        "schema": "policy-learnware.v03-query-mode-protocol.v0",
        "mode": "QUERY_EMPIRICAL",
        "version": 1,
    }
)
QUERY_REDUCED_PROTOCOL_ID = sha256_json(
    {
        "schema": "policy-learnware.v03-query-mode-protocol.v0",
        "mode": "QUERY_REDUCED",
        "version": 1,
    }
)


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V03ContractError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise V03ContractError(f"{where} must be a SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise V03ContractError(f"{where} must be a SHA-256 digest") from error
    if result != value:
        raise V03ContractError(f"{where} must use lowercase hexadecimal")
    return result


def _positive_finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V03ContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise V03ContractError(f"{where} must be finite and positive")
    return result


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise V03ContractError(f"{where} must be a mapping")
    observed = set(value)
    if observed != expected:
        raise V03ContractError(
            f"{where} fields differ: missing={sorted(expected-observed)}, "
            f"unknown={sorted(observed-expected)}"
        )


def _validate_reduced_norm(reduced: ReducedRKME, where: str) -> None:
    kernel = GaussianKernel(reduced.bandwidth)
    computed = float(reduced.beta @ kernel.gram(reduced.supports) @ reduced.beta)
    scale = max(1.0, abs(computed), abs(reduced.rkme_norm2))
    if abs(computed - reduced.rkme_norm2) > 1.0e-8 * scale:
        raise V03ContractError(
            f"{where} rkme_norm2 disagrees with supports and beta"
        )


def derive_reducer_digest(config: ReducerConfig) -> str:
    if not isinstance(config, ReducerConfig):
        raise V03ContractError("reducer config must be an explicit ReducerConfig")
    return sha256_json(
        {
            "schema": "policy-learnware.v03-reducer-contract.v0",
            "config": canonicalize(asdict(config)),
        }
    )


@dataclass(frozen=True)
class SemanticCacheKey:
    raw_dataset_digest: str
    ordered_episode_window_digest: str
    canonical_view_digest: str
    window_protocol_digest: str
    normalizer_digest: str
    encoder_implementation_digest: str
    checkpoint_digest: str
    semantic_output_protocol_digest: str
    mathematical_dtype_digest: str
    semantic_cache_key_digest: str | None = None
    schema: str = SEMANTIC_CACHE_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEMANTIC_CACHE_KEY_SCHEMA:
            raise V03ContractError("unsupported SemanticCacheKey schema")
        for name in (
            "raw_dataset_digest",
            "ordered_episode_window_digest",
            "canonical_view_digest",
            "window_protocol_digest",
            "normalizer_digest",
            "encoder_implementation_digest",
            "checkpoint_digest",
            "semantic_output_protocol_digest",
            "mathematical_dtype_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        expected = sha256_json(self._payload_without_digest())
        if self.semantic_cache_key_digest is None:
            object.__setattr__(self, "semantic_cache_key_digest", expected)
        elif _digest(
            self.semantic_cache_key_digest, "semantic_cache_key_digest"
        ) != expected:
            raise V03ContractError(
                "semantic_cache_key_digest does not match key contents"
            )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "raw_dataset_digest": self.raw_dataset_digest,
            "ordered_episode_window_digest": self.ordered_episode_window_digest,
            "canonical_view_digest": self.canonical_view_digest,
            "window_protocol_digest": self.window_protocol_digest,
            "normalizer_digest": self.normalizer_digest,
            "encoder_implementation_digest": self.encoder_implementation_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "semantic_output_protocol_digest": self.semantic_output_protocol_digest,
            "mathematical_dtype_digest": self.mathematical_dtype_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "semantic_cache_key_digest": self.semantic_cache_key_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticCacheKey":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SemanticCacheKey")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class SemanticCacheSlice:
    """One exact episode-boundary slice of an immutable semantic cache."""

    semantic_cache_digest: str
    points: np.ndarray
    episode_offsets: np.ndarray
    prefix_episode_count: int
    exact_slice_or_prefix_digest: str | None = None
    schema: str = SEMANTIC_CACHE_SLICE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEMANTIC_CACHE_SLICE_SCHEMA:
            raise V03ContractError("unsupported SemanticCacheSlice schema")
        object.__setattr__(
            self,
            "semantic_cache_digest",
            _digest(self.semantic_cache_digest, "semantic_cache_digest"),
        )
        raw_points = np.asarray(self.points)
        if raw_points.dtype != np.dtype(np.float64):
            raise V03ContractError("semantic slice points must use frozen float64 dtype")
        points = np.array(raw_points, copy=True)
        offsets = np.array(self.episode_offsets, dtype=np.int64, copy=True)
        if points.ndim != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise V03ContractError("semantic slice points must have finite shape [T,Q]")
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != points.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise V03ContractError("semantic slice episode_offsets are invalid")
        if (
            isinstance(self.prefix_episode_count, bool)
            or not isinstance(self.prefix_episode_count, int)
            or self.prefix_episode_count != offsets.size - 1
        ):
            raise V03ContractError("prefix_episode_count must match episode_offsets")
        points.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "episode_offsets", offsets)
        expected = sha256_json(
            {
                "schema": self.schema,
                "semantic_cache_digest": self.semantic_cache_digest,
                "prefix_episode_count": self.prefix_episode_count,
                "arrays_digest": sha256_ndarrays(
                    {"points": points, "episode_offsets": offsets}
                ),
            }
        )
        if self.exact_slice_or_prefix_digest is None:
            object.__setattr__(self, "exact_slice_or_prefix_digest", expected)
        elif _digest(
            self.exact_slice_or_prefix_digest, "exact_slice_or_prefix_digest"
        ) != expected:
            raise V03ContractError("slice digest does not match semantic slice")


@dataclass(frozen=True)
class SemanticCacheRecord:
    """Content-bound result of one frozen encoder forward over a max-prefix bank."""

    key: SemanticCacheKey
    points: np.ndarray
    episode_offsets: np.ndarray
    semantic_cache_digest: str | None = None
    schema: str = SEMANTIC_CACHE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEMANTIC_CACHE_RECORD_SCHEMA:
            raise V03ContractError("unsupported SemanticCacheRecord schema")
        if not isinstance(self.key, SemanticCacheKey):
            raise V03ContractError("semantic cache key has the wrong type")
        raw_points = np.asarray(self.points)
        if raw_points.dtype != np.dtype(np.float64):
            raise V03ContractError("semantic cache points must use frozen float64 dtype")
        if self.key.mathematical_dtype_digest != FLOAT64_MATHEMATICAL_DTYPE_DIGEST:
            raise V03ContractError(
                "semantic cache dtype digest does not bind the implemented float64 math"
            )
        points = np.array(raw_points, copy=True)
        offsets = np.array(self.episode_offsets, dtype=np.int64, copy=True)
        if points.ndim != 2 or points.shape[0] == 0 or not np.all(np.isfinite(points)):
            raise V03ContractError("semantic cache points must have finite shape [T,Q]")
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != points.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise V03ContractError("semantic cache episode_offsets are invalid")
        points.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "episode_offsets", offsets)
        expected = sha256_json(
            {
                "schema": self.schema,
                "semantic_cache_key_digest": self.key.semantic_cache_key_digest,
                "arrays_digest": sha256_ndarrays(
                    {"points": points, "episode_offsets": offsets}
                ),
            }
        )
        if self.semantic_cache_digest is None:
            object.__setattr__(self, "semantic_cache_digest", expected)
        elif _digest(self.semantic_cache_digest, "semantic_cache_digest") != expected:
            raise V03ContractError("semantic_cache_digest does not match cache contents")

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    def episode_prefix(self, episode_count: int | None = None) -> SemanticCacheSlice:
        count = self.episode_count if episode_count is None else episode_count
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= self.episode_count:
            raise V03ContractError("episode prefix is outside the semantic cache")
        stop = int(self.episode_offsets[count])
        return SemanticCacheSlice(
            semantic_cache_digest=str(self.semantic_cache_digest),
            points=self.points[:stop],
            episode_offsets=self.episode_offsets[: count + 1],
            prefix_episode_count=count,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "semantic_cache_key_digest": self.key.semantic_cache_key_digest,
            "semantic_cache_digest": self.semantic_cache_digest,
            "arrays_digest": sha256_ndarrays(
                {"points": self.points, "episode_offsets": self.episode_offsets}
            ),
            "episode_count": self.episode_count,
            "transition_count": int(self.points.shape[0]),
            "latent_dim": int(self.points.shape[1]),
        }


@dataclass(frozen=True)
class SpecKey:
    semantic_cache_digest: str
    exact_slice_or_prefix_digest: str
    sample_weighting_digest: str
    spec_role: SpecRole
    kernel_evaluator_digest: str
    kernel_bandwidth: float
    reducer_digest_or_empirical_query_marker: str
    spec_key_digest: str | None = None
    schema: str = SPEC_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SPEC_KEY_SCHEMA:
            raise V03ContractError("unsupported SpecKey schema")
        for name in (
            "semantic_cache_digest",
            "exact_slice_or_prefix_digest",
            "sample_weighting_digest",
            "kernel_evaluator_digest",
            "reducer_digest_or_empirical_query_marker",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.spec_role not in SPEC_ROLES:
            raise V03ContractError(f"unsupported spec_role: {self.spec_role!r}")
        bandwidth = _positive_finite(self.kernel_bandwidth, "kernel_bandwidth")
        object.__setattr__(self, "kernel_bandwidth", bandwidth)
        marker = self.reducer_digest_or_empirical_query_marker
        if self.spec_role == "QUERY_EMPIRICAL" and marker != EMPIRICAL_QUERY_MARKER:
            raise V03ContractError("QUERY_EMPIRICAL must use the empirical-query marker")
        if self.spec_role != "QUERY_EMPIRICAL" and marker == EMPIRICAL_QUERY_MARKER:
            raise V03ContractError("reduced specs must bind a reducer digest")
        expected = sha256_json(self._payload_without_digest())
        if self.spec_key_digest is None:
            object.__setattr__(self, "spec_key_digest", expected)
        elif _digest(self.spec_key_digest, "spec_key_digest") != expected:
            raise V03ContractError("spec_key_digest does not match key contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "semantic_cache_digest": self.semantic_cache_digest,
            "exact_slice_or_prefix_digest": self.exact_slice_or_prefix_digest,
            "sample_weighting_digest": self.sample_weighting_digest,
            "spec_role": self.spec_role,
            "kernel_evaluator_digest": self.kernel_evaluator_digest,
            "kernel_bandwidth": self.kernel_bandwidth,
            "reducer_digest_or_empirical_query_marker": (
                self.reducer_digest_or_empirical_query_marker
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "spec_key_digest": self.spec_key_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpecKey":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "SpecKey")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class RankingKey:
    query_spec_digest: str
    representation_index_digest: str
    selector_digest: str
    tie_break_digest: str
    ranking_key_digest: str | None = None
    schema: str = RANKING_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RANKING_KEY_SCHEMA:
            raise V03ContractError("unsupported RankingKey schema")
        for name in (
            "query_spec_digest",
            "representation_index_digest",
            "selector_digest",
            "tie_break_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        expected = sha256_json(self._payload_without_digest())
        if self.ranking_key_digest is None:
            object.__setattr__(self, "ranking_key_digest", expected)
        elif _digest(self.ranking_key_digest, "ranking_key_digest") != expected:
            raise V03ContractError("ranking_key_digest does not match key contents")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "query_spec_digest": self.query_spec_digest,
            "representation_index_digest": self.representation_index_digest,
            "selector_digest": self.selector_digest,
            "tie_break_digest": self.tie_break_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "ranking_key_digest": self.ranking_key_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RankingKey":
        fields = set(cls.__dataclass_fields__)
        _strict(value, fields, "RankingKey")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True)
class SourceReducedSpec:
    reduced_kme: ReducedRKME
    semantic_cache_key: SemanticCacheKey
    semantic_cache_digest: str
    spec_key: SpecKey
    measurement_protocol_id: str
    canonical_view_digest: str
    probe_dataset_digest: str
    legacy_environment_spec_digest: str | None = None
    source_spec_digest: str | None = None
    schema: str = SOURCE_REDUCED_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_REDUCED_SPEC_SCHEMA:
            raise V03ContractError("unsupported SourceReducedSpec schema")
        if not isinstance(self.reduced_kme, ReducedRKME):
            raise V03ContractError("source payload must be a ReducedRKME")
        if not isinstance(self.semantic_cache_key, SemanticCacheKey):
            raise V03ContractError("semantic_cache_key has the wrong type")
        object.__setattr__(
            self,
            "semantic_cache_digest",
            _digest(self.semantic_cache_digest, "semantic_cache_digest"),
        )
        if not isinstance(self.spec_key, SpecKey) or self.spec_key.spec_role != "SOURCE_REDUCED":
            raise V03ContractError("source spec_key must have SOURCE_REDUCED role")
        if self.spec_key.semantic_cache_digest != self.semantic_cache_digest:
            raise V03ContractError("source SpecKey is bound to another semantic cache")
        if (
            self.spec_key.sample_weighting_digest
            != EPISODE_BALANCED_WEIGHTING_DIGEST
            or self.spec_key.kernel_evaluator_digest
            != GAUSSIAN_KERNEL_EVALUATOR_DIGEST
        ):
            raise V03ContractError(
                "source SpecKey does not bind the implemented weighting/kernel"
            )
        for name in (
            "measurement_protocol_id",
            "canonical_view_digest",
            "probe_dataset_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "legacy_environment_spec_digest",
            (
                None
                if self.legacy_environment_spec_digest is None
                else _digest(
                    self.legacy_environment_spec_digest,
                    "legacy_environment_spec_digest",
                )
            ),
        )
        _digest(self.reduced_kme.protocol_id, "reduced_kme.protocol_id")
        if (
            self.reduced_kme.protocol_id
            != self.semantic_cache_key.semantic_output_protocol_digest
        ):
            raise V03ContractError(
                "source ReducedRKME and semantic-output protocol differ"
            )
        if self.spec_key.kernel_bandwidth != self.reduced_kme.bandwidth:
            raise V03ContractError("source SpecKey and ReducedRKME bandwidth differ")
        if self.semantic_cache_key.canonical_view_digest != self.canonical_view_digest:
            raise V03ContractError("source semantic cache and canonical view differ")
        if self.reduced_kme.source_dataset_digest != self.probe_dataset_digest:
            raise V03ContractError("source ReducedRKME and probe dataset digest differ")
        _validate_reduced_norm(self.reduced_kme, "source ReducedRKME")
        expected = sha256_json(self._payload_without_digest())
        if self.source_spec_digest is None:
            object.__setattr__(self, "source_spec_digest", expected)
        elif _digest(self.source_spec_digest, "source_spec_digest") != expected:
            raise V03ContractError("source_spec_digest does not match source contents")

    @property
    def representation_protocol_id(self) -> str:
        return self.reduced_kme.protocol_id

    @property
    def latent_dim(self) -> int:
        return int(self.reduced_kme.supports.shape[1])

    @property
    def kernel_bandwidth(self) -> float:
        return self.reduced_kme.bandwidth

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "reduced_arrays_digest": sha256_ndarrays(
                {"supports": self.reduced_kme.supports, "beta": self.reduced_kme.beta}
            ),
            "rkme_norm2": self.reduced_kme.rkme_norm2,
            "empirical_norm2": self.reduced_kme.empirical_norm2,
            "reconstruction_error": self.reduced_kme.reduction_error,
            "representation_protocol_id": self.representation_protocol_id,
            "measurement_protocol_id": self.measurement_protocol_id,
            "canonical_view_digest": self.canonical_view_digest,
            "probe_dataset_digest": self.probe_dataset_digest,
            "legacy_environment_spec_digest": self.legacy_environment_spec_digest,
            "semantic_cache_key_digest": self.semantic_cache_key.semantic_cache_key_digest,
            "semantic_cache_digest": self.semantic_cache_digest,
            "spec_key_digest": self.spec_key.spec_key_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "semantic_cache_key": self.semantic_cache_key.to_dict(),
            "spec_key": self.spec_key.to_dict(),
            "source_spec_digest": self.source_spec_digest,
        }


@dataclass(frozen=True)
class EmpiricalQuerySpec:
    empirical_kme: EmpiricalKME
    semantic_cache_key: SemanticCacheKey
    semantic_cache_digest: str
    spec_key: SpecKey
    representation_protocol_id: str
    measurement_protocol_id: str
    canonical_view_digest: str
    probe_dataset_digest: str
    query_protocol_id: str = QUERY_EMPIRICAL_PROTOCOL_ID
    query_spec_digest: str | None = None
    schema: str = EMPIRICAL_QUERY_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EMPIRICAL_QUERY_SPEC_SCHEMA:
            raise V03ContractError("unsupported EmpiricalQuerySpec schema")
        if not isinstance(self.empirical_kme, EmpiricalKME):
            raise V03ContractError("empirical_kme has the wrong type")
        if not isinstance(self.semantic_cache_key, SemanticCacheKey):
            raise V03ContractError("semantic_cache_key has the wrong type")
        object.__setattr__(
            self,
            "semantic_cache_digest",
            _digest(self.semantic_cache_digest, "semantic_cache_digest"),
        )
        if not isinstance(self.spec_key, SpecKey) or self.spec_key.spec_role != "QUERY_EMPIRICAL":
            raise V03ContractError("empirical query requires QUERY_EMPIRICAL SpecKey")
        for name in (
            "representation_protocol_id",
            "measurement_protocol_id",
            "canonical_view_digest",
            "probe_dataset_digest",
            "query_protocol_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.query_protocol_id != QUERY_EMPIRICAL_PROTOCOL_ID:
            raise V03ContractError("empirical query uses an unsupported query protocol")
        if self.spec_key.semantic_cache_digest != self.semantic_cache_digest:
            raise V03ContractError("query SpecKey is bound to another semantic cache")
        if (
            self.spec_key.sample_weighting_digest
            != EPISODE_BALANCED_WEIGHTING_DIGEST
            or self.spec_key.kernel_evaluator_digest
            != GAUSSIAN_KERNEL_EVALUATOR_DIGEST
        ):
            raise V03ContractError(
                "query SpecKey does not bind the implemented weighting/kernel"
            )
        if self.spec_key.kernel_bandwidth != self.empirical_kme.bandwidth:
            raise V03ContractError("query SpecKey and empirical bandwidth differ")
        if self.empirical_kme.protocol_id != self.representation_protocol_id:
            raise V03ContractError("empirical KME and representation protocol differ")
        if (
            self.representation_protocol_id
            != self.semantic_cache_key.semantic_output_protocol_digest
        ):
            raise V03ContractError(
                "query representation and semantic-output protocol differ"
            )
        if self.empirical_kme.dataset_digest != self.probe_dataset_digest:
            raise V03ContractError("empirical KME and probe dataset digest differ")
        if self.canonical_view_digest != self.semantic_cache_key.canonical_view_digest:
            raise V03ContractError("query semantic cache and canonical view differ")
        expected_weights = episode_balanced_weights(self.empirical_kme.episode_offsets)
        if not np.allclose(
            self.empirical_kme.weights, expected_weights, rtol=1.0e-10, atol=1.0e-12
        ):
            raise V03ContractError("query empirical weights are not episode-balanced")
        if not _has_exact_norm2_attestation(self.empirical_kme):
            computed = blockwise_weighted_self_kernel_sum(
                self.empirical_kme.points,
                self.empirical_kme.weights,
                GaussianKernel(self.empirical_kme.bandwidth),
            )
            scale = max(1.0, abs(computed), abs(self.empirical_kme.norm2))
            if abs(computed - self.empirical_kme.norm2) > 1.0e-8 * scale:
                raise V03ContractError("query empirical norm disagrees with its payload")
        expected = sha256_json(self._payload_without_digest())
        if self.query_spec_digest is None:
            object.__setattr__(self, "query_spec_digest", expected)
        elif _digest(self.query_spec_digest, "query_spec_digest") != expected:
            raise V03ContractError("query_spec_digest does not match query contents")

    @property
    def spec_role(self) -> SpecRole:
        return "QUERY_EMPIRICAL"

    @property
    def query_mode(self) -> str:
        return "QUERY_EMPIRICAL"

    @property
    def latent_dim(self) -> int:
        return int(self.empirical_kme.points.shape[1])

    @property
    def kernel_bandwidth(self) -> float:
        return self.empirical_kme.bandwidth

    def _payload_without_digest(self) -> dict[str, Any]:
        arrays_digest = sha256_ndarrays(
            {
                "points": self.empirical_kme.points,
                "weights": self.empirical_kme.weights,
                "episode_offsets": self.empirical_kme.episode_offsets,
            }
        )
        return {
            "schema": self.schema,
            "query_protocol_id": self.query_protocol_id,
            "representation_protocol_id": self.representation_protocol_id,
            "measurement_protocol_id": self.measurement_protocol_id,
            "canonical_view_digest": self.canonical_view_digest,
            "probe_dataset_digest": self.probe_dataset_digest,
            "semantic_cache_key_digest": self.semantic_cache_key.semantic_cache_key_digest,
            "semantic_cache_digest": self.semantic_cache_digest,
            "spec_key_digest": self.spec_key.spec_key_digest,
            "arrays_digest": arrays_digest,
            "transition_count": self.empirical_kme.transition_count,
            "episode_count": self.empirical_kme.episode_count,
            "latent_dim": self.latent_dim,
            "kernel_bandwidth": self.kernel_bandwidth,
            "empirical_norm2": self.empirical_kme.norm2,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "query_spec_digest": self.query_spec_digest}


@dataclass(frozen=True)
class ReducedQuerySpec:
    reduced_kme: ReducedRKME
    semantic_cache_key: SemanticCacheKey
    semantic_cache_digest: str
    spec_key: SpecKey
    measurement_protocol_id: str
    canonical_view_digest: str
    probe_dataset_digest: str
    query_protocol_id: str = QUERY_REDUCED_PROTOCOL_ID
    query_spec_digest: str | None = None
    schema: str = REDUCED_QUERY_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REDUCED_QUERY_SPEC_SCHEMA:
            raise V03ContractError("unsupported ReducedQuerySpec schema")
        if not isinstance(self.reduced_kme, ReducedRKME):
            raise V03ContractError("reduced query payload must be a ReducedRKME")
        if not isinstance(self.semantic_cache_key, SemanticCacheKey):
            raise V03ContractError("semantic_cache_key has the wrong type")
        object.__setattr__(
            self,
            "semantic_cache_digest",
            _digest(self.semantic_cache_digest, "semantic_cache_digest"),
        )
        if not isinstance(self.spec_key, SpecKey) or self.spec_key.spec_role != "QUERY_REDUCED":
            raise V03ContractError("reduced query requires QUERY_REDUCED SpecKey")
        object.__setattr__(
            self, "query_protocol_id", _digest(self.query_protocol_id, "query_protocol_id")
        )
        if self.query_protocol_id != QUERY_REDUCED_PROTOCOL_ID:
            raise V03ContractError("reduced query uses an unsupported query protocol")
        for name in (
            "measurement_protocol_id",
            "canonical_view_digest",
            "probe_dataset_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _digest(self.reduced_kme.protocol_id, "reduced_kme.protocol_id")
        if (
            self.reduced_kme.protocol_id
            != self.semantic_cache_key.semantic_output_protocol_digest
        ):
            raise V03ContractError(
                "reduced query and semantic-output protocol differ"
            )
        if self.spec_key.semantic_cache_digest != self.semantic_cache_digest:
            raise V03ContractError("reduced-query SpecKey is bound to another semantic cache")
        if (
            self.spec_key.sample_weighting_digest
            != EPISODE_BALANCED_WEIGHTING_DIGEST
            or self.spec_key.kernel_evaluator_digest
            != GAUSSIAN_KERNEL_EVALUATOR_DIGEST
        ):
            raise V03ContractError(
                "reduced-query SpecKey does not bind the implemented weighting/kernel"
            )
        if self.spec_key.kernel_bandwidth != self.reduced_kme.bandwidth:
            raise V03ContractError("reduced-query SpecKey and bandwidth differ")
        if self.semantic_cache_key.canonical_view_digest != self.canonical_view_digest:
            raise V03ContractError("reduced-query cache and canonical view differ")
        if self.reduced_kme.source_dataset_digest != self.probe_dataset_digest:
            raise V03ContractError("reduced query and probe dataset digest differ")
        _validate_reduced_norm(self.reduced_kme, "query ReducedRKME")
        expected = sha256_json(self._payload_without_digest())
        if self.query_spec_digest is None:
            object.__setattr__(self, "query_spec_digest", expected)
        elif _digest(self.query_spec_digest, "query_spec_digest") != expected:
            raise V03ContractError("query_spec_digest does not match reduced query")

    @property
    def spec_role(self) -> SpecRole:
        return "QUERY_REDUCED"

    @property
    def query_mode(self) -> str:
        return "QUERY_REDUCED"

    @property
    def representation_protocol_id(self) -> str:
        return self.reduced_kme.protocol_id

    @property
    def latent_dim(self) -> int:
        return int(self.reduced_kme.supports.shape[1])

    @property
    def kernel_bandwidth(self) -> float:
        return self.reduced_kme.bandwidth

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "query_protocol_id": self.query_protocol_id,
            "reduced_arrays_digest": sha256_ndarrays(
                {"supports": self.reduced_kme.supports, "beta": self.reduced_kme.beta}
            ),
            "rkme_norm2": self.reduced_kme.rkme_norm2,
            "empirical_norm2": self.reduced_kme.empirical_norm2,
            "reconstruction_error": self.reduced_kme.reduction_error,
            "representation_protocol_id": self.representation_protocol_id,
            "measurement_protocol_id": self.measurement_protocol_id,
            "canonical_view_digest": self.canonical_view_digest,
            "probe_dataset_digest": self.probe_dataset_digest,
            "semantic_cache_key_digest": self.semantic_cache_key.semantic_cache_key_digest,
            "semantic_cache_digest": self.semantic_cache_digest,
            "spec_key_digest": self.spec_key.spec_key_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "semantic_cache_key": self.semantic_cache_key.to_dict(),
            "spec_key": self.spec_key.to_dict(),
            "query_spec_digest": self.query_spec_digest,
        }


QuerySpec: TypeAlias = EmpiricalQuerySpec | ReducedQuerySpec


@dataclass(frozen=True)
class SourceRepresentationIndex:
    policy_market_id: str
    representation_protocol_id: str
    entries: Mapping[str, SourceReducedSpec]
    representation_index_digest: str | None = None
    schema: str = SOURCE_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_INDEX_SCHEMA:
            raise V03ContractError("unsupported SourceRepresentationIndex schema")
        market_id = _nonempty(self.policy_market_id, "policy_market_id")
        protocol = _digest(self.representation_protocol_id, "representation_protocol_id")
        entries = dict(self.entries)
        if not entries:
            raise V03ContractError("source representation index cannot be empty")
        for opaque_id, source in entries.items():
            _nonempty(opaque_id, "source opaque_id")
            if not isinstance(source, SourceReducedSpec):
                raise V03ContractError("source index entries must be SourceReducedSpec")
            if source.representation_protocol_id != protocol:
                raise V03ContractError("source index contains another representation protocol")
        reference = next(iter(entries.values()))
        for opaque_id, source in entries.items():
            shared = {
                "measurement_protocol_id": (
                    source.measurement_protocol_id,
                    reference.measurement_protocol_id,
                ),
                "canonical_view_digest": (
                    source.canonical_view_digest,
                    reference.canonical_view_digest,
                ),
                "kernel_evaluator_digest": (
                    source.spec_key.kernel_evaluator_digest,
                    reference.spec_key.kernel_evaluator_digest,
                ),
                "sample_weighting_digest": (
                    source.spec_key.sample_weighting_digest,
                    reference.spec_key.sample_weighting_digest,
                ),
                "reducer_digest": (
                    source.spec_key.reducer_digest_or_empirical_query_marker,
                    reference.spec_key.reducer_digest_or_empirical_query_marker,
                ),
                "kernel_bandwidth": (
                    source.kernel_bandwidth,
                    reference.kernel_bandwidth,
                ),
                "latent_dim": (source.latent_dim, reference.latent_dim),
            }
            mismatches = {
                name: values for name, values in shared.items() if values[0] != values[1]
            }
            if mismatches:
                raise V03ContractError(
                    f"source index entry {opaque_id!r} has incompatible bindings: "
                    f"{mismatches}"
                )
        object.__setattr__(self, "policy_market_id", market_id)
        object.__setattr__(self, "representation_protocol_id", protocol)
        object.__setattr__(self, "entries", MappingProxyType(dict(sorted(entries.items()))))
        expected = sha256_json(self._payload_without_digest())
        if self.representation_index_digest is None:
            object.__setattr__(self, "representation_index_digest", expected)
        elif _digest(
            self.representation_index_digest, "representation_index_digest"
        ) != expected:
            raise V03ContractError("representation_index_digest does not match index")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_market_id": self.policy_market_id,
            "representation_protocol_id": self.representation_protocol_id,
            "entries": {
                opaque_id: source.source_spec_digest
                for opaque_id, source in self.entries.items()
            },
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "representation_index_digest": self.representation_index_digest,
        }


def source_from_v02_environment_spec(
    environment_spec: EnvironmentSpec,
    *,
    semantic_cache: SemanticCacheRecord,
) -> SourceReducedSpec:
    """Role-bind a mathematically self-consistent historical source spec.

    v0.2 normalized some unconstrained reducer weights without recomputing the
    stored RKME norm.  Such artifacts fail closed here and must be rebuilt from
    their semantic cache instead of silently changing the distance geometry.
    """

    if not isinstance(environment_spec, EnvironmentSpec):
        raise V03ContractError("legacy source must be an EnvironmentSpec")
    if not isinstance(semantic_cache, SemanticCacheRecord):
        raise V03ContractError("semantic_cache must be a SemanticCacheRecord")
    if environment_spec.probe_dataset_digest != semantic_cache.key.raw_dataset_digest:
        raise V03ContractError("legacy source and semantic-cache raw dataset differ")
    if (
        environment_spec.representation_protocol_id
        != semantic_cache.key.semantic_output_protocol_digest
    ):
        raise V03ContractError(
            "legacy source and semantic-cache output protocol differ"
        )
    semantic_slice = semantic_cache.episode_prefix()
    kernel = GaussianKernel(environment_spec.kernel_bandwidth)
    computed_norm = float(
        environment_spec.beta
        @ kernel.gram(environment_spec.supports)
        @ environment_spec.beta
    )
    scale = max(1.0, abs(computed_norm), abs(environment_spec.rkme_norm2))
    if abs(computed_norm - environment_spec.rkme_norm2) > 1.0e-8 * scale:
        raise V03ContractError(
            "legacy EnvironmentSpec is norm-inconsistent; rebuild from semantic cache"
        )
    empirical = build_empirical_kme(
        semantic_slice.points,
        kernel,
        episode_offsets=semantic_slice.episode_offsets,
        protocol_id=environment_spec.representation_protocol_id,
        dataset_digest=environment_spec.probe_dataset_digest,
    )
    empirical_scale = max(
        1.0, abs(empirical.norm2), abs(environment_spec.empirical_norm2)
    )
    if (
        abs(empirical.norm2 - environment_spec.empirical_norm2)
        > 1.0e-8 * empirical_scale
    ):
        raise V03ContractError(
            "legacy EnvironmentSpec empirical norm is not derived from semantic cache"
        )
    cross = blockwise_weighted_kernel_sum(
        empirical.points,
        empirical.weights,
        environment_spec.supports,
        environment_spec.beta,
        kernel,
    )
    residual_squared = float(
        empirical.norm2 - 2.0 * cross + environment_spec.rkme_norm2
    )
    recorded_residual = float(environment_spec.reconstruction_error) ** 2
    residual_scale = max(1.0, abs(residual_squared), abs(recorded_residual))
    if abs(residual_squared - recorded_residual) > 1.0e-8 * residual_scale:
        raise V03ContractError(
            "legacy EnvironmentSpec reconstruction error is not derived from semantic cache"
        )
    reduced = ReducedRKME(
        supports=environment_spec.supports,
        beta=environment_spec.beta,
        bandwidth=environment_spec.kernel_bandwidth,
        rkme_norm2=environment_spec.rkme_norm2,
        empirical_norm2=environment_spec.empirical_norm2,
        reduction_error=environment_spec.reconstruction_error,
        protocol_id=environment_spec.representation_protocol_id,
        source_dataset_digest=environment_spec.probe_dataset_digest,
        raw_reduction_residual_squared=environment_spec.reconstruction_error**2,
    )
    key = SpecKey(
        semantic_cache_digest=str(semantic_cache.semantic_cache_digest),
        exact_slice_or_prefix_digest=str(
            semantic_slice.exact_slice_or_prefix_digest
        ),
        sample_weighting_digest=EPISODE_BALANCED_WEIGHTING_DIGEST,
        spec_role="SOURCE_REDUCED",
        kernel_evaluator_digest=GAUSSIAN_KERNEL_EVALUATOR_DIGEST,
        kernel_bandwidth=environment_spec.kernel_bandwidth,
        reducer_digest_or_empirical_query_marker=environment_spec.reducer_digest,
    )
    return SourceReducedSpec(
        reduced_kme=reduced,
        semantic_cache_key=semantic_cache.key,
        semantic_cache_digest=str(semantic_cache.semantic_cache_digest),
        spec_key=key,
        measurement_protocol_id=environment_spec.measurement_protocol_id,
        canonical_view_digest=environment_spec.canonical_view_digest,
        probe_dataset_digest=environment_spec.probe_dataset_digest,
        legacy_environment_spec_digest=environment_spec.environment_spec_digest,
    )


def build_empirical_query_spec(
    semantic_cache: SemanticCacheRecord,
    *,
    kernel_bandwidth: float,
    measurement_protocol_id: str,
    probe_dataset_digest: str,
    episode_count: int | None = None,
    block_size: int = 2048,
    computation_backend: str = "numpy",
) -> EmpiricalQuerySpec:
    """Build the primary query artifact; this function never calls ``reduce_kme``."""

    if not isinstance(semantic_cache, SemanticCacheRecord):
        raise V03ContractError("semantic_cache must be a SemanticCacheRecord")
    if probe_dataset_digest != semantic_cache.key.raw_dataset_digest:
        raise V03ContractError("probe dataset and semantic-cache raw dataset differ")
    semantic_slice = semantic_cache.episode_prefix(episode_count)
    key = SpecKey(
        semantic_cache_digest=str(semantic_cache.semantic_cache_digest),
        exact_slice_or_prefix_digest=str(
            semantic_slice.exact_slice_or_prefix_digest
        ),
        sample_weighting_digest=EPISODE_BALANCED_WEIGHTING_DIGEST,
        spec_role="QUERY_EMPIRICAL",
        kernel_evaluator_digest=GAUSSIAN_KERNEL_EVALUATOR_DIGEST,
        kernel_bandwidth=kernel_bandwidth,
        reducer_digest_or_empirical_query_marker=EMPIRICAL_QUERY_MARKER,
    )
    empirical = build_empirical_kme(
        semantic_slice.points,
        GaussianKernel(kernel_bandwidth),
        episode_offsets=semantic_slice.episode_offsets,
        protocol_id=semantic_cache.key.semantic_output_protocol_digest,
        dataset_digest=probe_dataset_digest,
        block_size=block_size,
        computation_backend=computation_backend,
    )
    return EmpiricalQuerySpec(
        empirical_kme=empirical,
        semantic_cache_key=semantic_cache.key,
        semantic_cache_digest=str(semantic_cache.semantic_cache_digest),
        spec_key=key,
        representation_protocol_id=semantic_cache.key.semantic_output_protocol_digest,
        measurement_protocol_id=measurement_protocol_id,
        canonical_view_digest=semantic_cache.key.canonical_view_digest,
        probe_dataset_digest=probe_dataset_digest,
    )


def build_source_reduced_spec(
    semantic_cache: SemanticCacheRecord,
    *,
    kernel_bandwidth: float,
    measurement_protocol_id: str,
    probe_dataset_digest: str,
    reducer_config: ReducerConfig,
    episode_count: int | None = None,
    block_size: int = 2048,
    computation_backend: str = "numpy",
) -> SourceReducedSpec:
    """Build one role-bound source RKME; source reduction occurs exactly here."""

    if not isinstance(semantic_cache, SemanticCacheRecord):
        raise V03ContractError("semantic_cache must be a SemanticCacheRecord")
    if probe_dataset_digest != semantic_cache.key.raw_dataset_digest:
        raise V03ContractError("probe dataset and semantic-cache raw dataset differ")
    semantic_slice = semantic_cache.episode_prefix(episode_count)
    reducer_digest = derive_reducer_digest(reducer_config)
    empirical = build_empirical_kme(
        semantic_slice.points,
        GaussianKernel(kernel_bandwidth),
        episode_offsets=semantic_slice.episode_offsets,
        protocol_id=semantic_cache.key.semantic_output_protocol_digest,
        dataset_digest=probe_dataset_digest,
        block_size=block_size,
        computation_backend=computation_backend,
    )
    reduced = reduce_kme(empirical, reducer_config)
    key = SpecKey(
        semantic_cache_digest=str(semantic_cache.semantic_cache_digest),
        exact_slice_or_prefix_digest=str(
            semantic_slice.exact_slice_or_prefix_digest
        ),
        sample_weighting_digest=EPISODE_BALANCED_WEIGHTING_DIGEST,
        spec_role="SOURCE_REDUCED",
        kernel_evaluator_digest=GAUSSIAN_KERNEL_EVALUATOR_DIGEST,
        kernel_bandwidth=kernel_bandwidth,
        reducer_digest_or_empirical_query_marker=reducer_digest,
    )
    return SourceReducedSpec(
        reduced_kme=reduced,
        semantic_cache_key=semantic_cache.key,
        semantic_cache_digest=str(semantic_cache.semantic_cache_digest),
        spec_key=key,
        measurement_protocol_id=measurement_protocol_id,
        canonical_view_digest=semantic_cache.key.canonical_view_digest,
        probe_dataset_digest=probe_dataset_digest,
    )


def reduce_query_spec(
    query: EmpiricalQuerySpec,
    *,
    reducer_config: ReducerConfig,
) -> ReducedQuerySpec:
    """Explicitly construct the separately versioned reduced-query candidate.

    Calling this function is the only supported mode transition.  The primary
    builder never catches errors and never invokes this function automatically.
    """

    if not isinstance(query, EmpiricalQuerySpec):
        raise V03ContractError("reduce_query_spec requires an empirical query")
    reducer_digest = derive_reducer_digest(reducer_config)
    reduced = reduce_kme(query.empirical_kme, reducer_config)
    key = SpecKey(
        semantic_cache_digest=query.semantic_cache_digest,
        exact_slice_or_prefix_digest=query.spec_key.exact_slice_or_prefix_digest,
        sample_weighting_digest=query.spec_key.sample_weighting_digest,
        spec_role="QUERY_REDUCED",
        kernel_evaluator_digest=query.spec_key.kernel_evaluator_digest,
        kernel_bandwidth=query.kernel_bandwidth,
        reducer_digest_or_empirical_query_marker=reducer_digest,
    )
    return ReducedQuerySpec(
        reduced_kme=reduced,
        semantic_cache_key=query.semantic_cache_key,
        semantic_cache_digest=query.semantic_cache_digest,
        spec_key=key,
        measurement_protocol_id=query.measurement_protocol_id,
        canonical_view_digest=query.canonical_view_digest,
        probe_dataset_digest=query.probe_dataset_digest,
    )


__all__ = [
    "EMPIRICAL_QUERY_MARKER",
    "EPISODE_BALANCED_WEIGHTING_DIGEST",
    "FLOAT64_MATHEMATICAL_DTYPE_DIGEST",
    "GAUSSIAN_KERNEL_EVALUATOR_DIGEST",
    "QUERY_EMPIRICAL_PROTOCOL_ID",
    "QUERY_REDUCED_PROTOCOL_ID",
    "EmpiricalQuerySpec",
    "QuerySpec",
    "RankingKey",
    "ReducedQuerySpec",
    "SemanticCacheRecord",
    "SemanticCacheSlice",
    "SemanticCacheKey",
    "SourceReducedSpec",
    "SourceRepresentationIndex",
    "SpecKey",
    "SpecRole",
    "V03ContractError",
    "build_empirical_query_spec",
    "build_source_reduced_spec",
    "derive_reducer_digest",
    "reduce_query_spec",
    "source_from_v02_environment_spec",
]
