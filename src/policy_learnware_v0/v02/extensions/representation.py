"""Replaceable semantic-encoder contracts for v0.2 EnvironmentSpecs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...hashing import canonicalize, sha256_json
from .environment import CONTINUOUS_VECTOR_MDP_V02, ProtocolFamilyMismatch


class RepresentationPluginError(ValueError):
    """A semantic encoder or its metadata violated the extension contract."""


class DuplicateSemanticEncoderError(RepresentationPluginError):
    """A representation id was registered more than once."""


class RepresentationBindingError(RepresentationPluginError):
    """Dataset, cache or protocol metadata does not bind to the encoder."""


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RepresentationPluginError(
            f"{where} must be a non-empty canonical string"
        )
    return value


def _digest(value: Any, where: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    candidate = _identifier(value, where).lower()
    if len(candidate) != 64:
        raise RepresentationPluginError(f"{where} must be a SHA-256 digest")
    try:
        int(candidate, 16)
    except ValueError as error:
        raise RepresentationPluginError(f"{where} must be a SHA-256 digest") from error
    return candidate


def _frozen_mapping(value: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise RepresentationPluginError(f"{where} must be a string-keyed mapping")
    try:
        canonical = canonicalize(dict(value))
    except (TypeError, ValueError) as error:
        raise RepresentationPluginError(
            f"{where} is not canonical-JSON compatible: {error}"
        ) from error

    def deep_freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType(
                {str(key): deep_freeze(nested) for key, nested in item.items()}
            )
        if isinstance(item, (list, tuple)):
            return tuple(deep_freeze(nested) for nested in item)
        return item

    return deep_freeze(canonical)


@dataclass(frozen=True)
class SemanticEncoderMetadata:
    """Hash-addressed access card for one frozen representation protocol."""

    representation_id: str
    family: str
    version: str
    protocol_family_id: str
    canonical_event_view_digest: str
    input_dim: int
    output_dim: int
    code_digest: str
    runtime_digest: str
    dependency_digest: str
    checkpoint_digest: str | None = None
    normalizer_digest: str | None = None
    training_split_digest: str | None = None
    source_permissions: tuple[str, ...] = ()
    latent_config: Mapping[str, Any] = field(default_factory=dict)
    kernel_config: Mapping[str, Any] = field(default_factory=dict)
    reducer_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("representation_id", "family", "version", "protocol_family_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in (
            "canonical_event_view_digest",
            "code_digest",
            "runtime_digest",
            "dependency_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("checkpoint_digest", "normalizer_digest", "training_split_digest"):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name, nullable=True),
            )
        for name in ("input_dim", "output_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise RepresentationPluginError(f"{name} must be a positive integer")
        permissions = tuple(
            _identifier(item, "source_permissions[]")
            for item in self.source_permissions
        )
        if len(set(permissions)) != len(permissions):
            raise RepresentationPluginError("source_permissions contains duplicates")
        object.__setattr__(self, "source_permissions", permissions)
        for name in ("latent_config", "kernel_config", "reducer_config"):
            object.__setattr__(
                self, name, _frozen_mapping(getattr(self, name), name)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation_id": self.representation_id,
            "family": self.family,
            "version": self.version,
            "protocol_family_id": self.protocol_family_id,
            "canonical_event_view_digest": self.canonical_event_view_digest,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "code_digest": self.code_digest,
            "runtime_digest": self.runtime_digest,
            "dependency_digest": self.dependency_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "normalizer_digest": self.normalizer_digest,
            "training_split_digest": self.training_split_digest,
            "source_permissions": list(self.source_permissions),
            "latent_config": canonicalize(self.latent_config),
            "kernel_config": canonicalize(self.kernel_config),
            "reducer_config": canonicalize(self.reducer_config),
        }

    @property
    def representation_protocol_id(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class EncodedEpisodeDataset:
    """Immutable encoded events with the original episode partition."""

    points: np.ndarray
    episode_offsets: np.ndarray
    representation_protocol_id: str

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        offsets = np.asarray(self.episode_offsets, dtype=np.int64)
        if points.ndim != 2 or not np.all(np.isfinite(points)):
            raise RepresentationPluginError("encoded points must be finite [T,D]")
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != points.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise RepresentationPluginError(
                "episode_offsets must strictly partition all encoded points"
            )
        points = np.array(points, copy=True)
        offsets = np.array(offsets, copy=True)
        points.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "episode_offsets", offsets)
        object.__setattr__(
            self,
            "representation_protocol_id",
            _digest(self.representation_protocol_id, "representation_protocol_id"),
        )

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)


@runtime_checkable
class SemanticEncoderProtocol(Protocol):
    @property
    def metadata(self) -> SemanticEncoderMetadata: ...

    def encode(
        self,
        dataset: Any,
        *,
        batch_size: int,
    ) -> EncodedEpisodeDataset: ...


def _packed_dataset(
    dataset: Any, metadata: SemanticEncoderMetadata, *, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    if type(batch_size) is not int or batch_size <= 0:
        raise RepresentationPluginError("batch_size must be a positive integer")
    if not hasattr(dataset, "packed") or not hasattr(dataset, "episode_offsets"):
        raise RepresentationPluginError(
            "encoder input must expose packed and episode_offsets"
        )
    packed = np.asarray(dataset.packed, dtype=np.float32)
    offsets = np.asarray(dataset.episode_offsets, dtype=np.int64)
    if (
        packed.ndim != 2
        or packed.shape[1] != metadata.input_dim
        or not np.all(np.isfinite(packed))
    ):
        raise RepresentationBindingError(
            f"packed events must have finite shape [T,{metadata.input_dim}]"
        )
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != packed.shape[0]
        or np.any(np.diff(offsets) <= 0)
    ):
        raise RepresentationBindingError("invalid packed episode offsets")
    return packed, offsets


def _encoded(
    points: Any,
    offsets: np.ndarray,
    metadata: SemanticEncoderMetadata,
) -> EncodedEpisodeDataset:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != metadata.output_dim:
        raise RepresentationBindingError(
            f"encoder output must have shape [T,{metadata.output_dim}]"
        )
    return EncodedEpisodeDataset(
        points=array,
        episode_offsets=offsets,
        representation_protocol_id=metadata.representation_protocol_id,
    )


class RawTransitionEncoder:
    """Identity event encoder used as the transparent raw control."""

    def __init__(self, metadata: SemanticEncoderMetadata) -> None:
        if metadata.family != "raw":
            raise RepresentationPluginError("raw encoder metadata family must be 'raw'")
        if metadata.input_dim != metadata.output_dim:
            raise RepresentationPluginError("raw encoder input/output dimensions must match")
        if metadata.checkpoint_digest is not None:
            raise RepresentationPluginError("raw encoder must not claim a checkpoint")
        self._metadata = metadata

    @property
    def metadata(self) -> SemanticEncoderMetadata:
        return self._metadata

    def encode(self, dataset: Any, *, batch_size: int) -> EncodedEpisodeDataset:
        packed, offsets = _packed_dataset(dataset, self.metadata, batch_size=batch_size)
        return _encoded(np.array(packed, copy=True), offsets, self.metadata)


class FrozenCorroEncoderAdapter:
    """Adapter for the existing frozen ``TransitionSemanticEncoder`` API."""

    def __init__(self, encoder: Any, metadata: SemanticEncoderMetadata) -> None:
        if metadata.family != "corro":
            raise RepresentationPluginError("CORRO metadata family must be 'corro'")
        if metadata.checkpoint_digest is None:
            raise RepresentationPluginError("frozen CORRO metadata needs a checkpoint digest")
        if not callable(getattr(encoder, "encode", None)):
            raise RepresentationPluginError("wrapped CORRO encoder lacks encode()")
        self._encoder = encoder
        self._metadata = metadata

    @property
    def metadata(self) -> SemanticEncoderMetadata:
        return self._metadata

    def encode(self, dataset: Any, *, batch_size: int) -> EncodedEpisodeDataset:
        packed, offsets = _packed_dataset(dataset, self.metadata, batch_size=batch_size)
        points = self._encoder.encode(packed, batch_size=batch_size)
        if np.asarray(points).shape[0] != packed.shape[0]:
            raise RepresentationBindingError(
                "CORRO encoder changed the transition count"
            )
        return _encoded(points, offsets, self.metadata)


class LegacyFrozenCorroEncoderAdapter(FrozenCorroEncoderAdapter):
    """Named adapter for the ``corro_v0_frozen`` regression representation."""


class TaskSupConCorroAdapter(FrozenCorroEncoderAdapter):
    """Named adapter for task-labelled v0.2 SupCon checkpoints."""


class AnchorSupConCorroAdapter(FrozenCorroEncoderAdapter):
    """Named adapter for source-anchor-labelled v0.2 SupCon checkpoints."""


class SyntheticEncoderAdapter:
    """Callback-backed future encoder used only by conformance tests."""

    def __init__(
        self,
        metadata: SemanticEncoderMetadata,
        transform: Callable[[np.ndarray], Any],
    ) -> None:
        if metadata.family != "synthetic-test-only":
            raise RepresentationPluginError(
                "synthetic encoder family must be 'synthetic-test-only'"
            )
        if not callable(transform):
            raise RepresentationPluginError("synthetic transform must be callable")
        self._metadata = metadata
        self._transform = transform

    @property
    def metadata(self) -> SemanticEncoderMetadata:
        return self._metadata

    def encode(self, dataset: Any, *, batch_size: int) -> EncodedEpisodeDataset:
        packed, offsets = _packed_dataset(dataset, self.metadata, batch_size=batch_size)
        before = np.array(packed, copy=True)
        points = self._transform(np.array(packed, copy=True))
        if not np.array_equal(packed, before):
            raise RepresentationPluginError("synthetic encoder mutated its input")
        if np.asarray(points).shape[0] != packed.shape[0]:
            raise RepresentationBindingError(
                "synthetic encoder changed the transition count"
            )
        return _encoded(points, offsets, self.metadata)


class SemanticEncoderRegistry:
    """Closed representation registry with explicit family/view checks."""

    def __init__(self) -> None:
        self._encoders: dict[str, SemanticEncoderProtocol] = {}

    def register(self, encoder: SemanticEncoderProtocol) -> None:
        if not isinstance(encoder, SemanticEncoderProtocol):
            raise RepresentationPluginError(
                "semantic encoder does not implement the required protocol"
            )
        if not isinstance(encoder.metadata, SemanticEncoderMetadata):
            raise RepresentationPluginError("encoder metadata has the wrong type")
        identifier = encoder.metadata.representation_id
        if identifier in self._encoders:
            raise DuplicateSemanticEncoderError(
                f"semantic encoder {identifier!r} is already registered"
            )
        self._encoders[identifier] = encoder

    def resolve(
        self,
        representation_id: str,
        *,
        protocol_family_id: str | None = None,
        canonical_event_view_digest: str | None = None,
    ) -> SemanticEncoderProtocol:
        key = _identifier(representation_id, "representation_id")
        try:
            encoder = self._encoders[key]
        except KeyError as error:
            raise RepresentationPluginError(
                f"unknown semantic encoder {key!r}"
            ) from error
        metadata = encoder.metadata
        if (
            protocol_family_id is not None
            and metadata.protocol_family_id != protocol_family_id
        ):
            raise ProtocolFamilyMismatch(
                f"encoder family {metadata.protocol_family_id!r} != "
                f"required {protocol_family_id!r}"
            )
        if canonical_event_view_digest is not None:
            expected = _digest(
                canonical_event_view_digest, "canonical_event_view_digest"
            )
            if metadata.canonical_event_view_digest != expected:
                raise RepresentationBindingError(
                    "encoder canonical event view digest mismatch"
                )
        return encoder

    @property
    def encoders(self) -> Mapping[str, SemanticEncoderProtocol]:
        return MappingProxyType(dict(self._encoders))


def default_metadata(
    *,
    representation_id: str,
    family: str,
    input_dim: int,
    output_dim: int,
    canonical_event_view_digest: str,
    checkpoint_digest: str | None = None,
    normalizer_digest: str | None = None,
    training_split_digest: str | None = None,
    source_permissions: tuple[str, ...] = (),
) -> SemanticEncoderMetadata:
    """Build deterministic metadata for examples/tests, never formal freezing."""

    implementation = sha256_json(
        {
            "module": __name__,
            "adapter": family,
            "version": "v0",
        }
    )
    return SemanticEncoderMetadata(
        representation_id=representation_id,
        family=family,
        version="v0",
        protocol_family_id=CONTINUOUS_VECTOR_MDP_V02,
        canonical_event_view_digest=canonical_event_view_digest,
        input_dim=input_dim,
        output_dim=output_dim,
        code_digest=implementation,
        runtime_digest=sha256_json({"runtime": "extension-conformance"}),
        dependency_digest=sha256_json({"dependencies": ["numpy"]}),
        checkpoint_digest=checkpoint_digest,
        normalizer_digest=normalizer_digest,
        training_split_digest=training_split_digest,
        source_permissions=source_permissions,
        latent_config={"output_dim": output_dim},
        kernel_config={},
        reducer_config={},
    )


__all__ = [
    "AnchorSupConCorroAdapter",
    "DuplicateSemanticEncoderError",
    "EncodedEpisodeDataset",
    "FrozenCorroEncoderAdapter",
    "LegacyFrozenCorroEncoderAdapter",
    "RawTransitionEncoder",
    "RepresentationBindingError",
    "RepresentationPluginError",
    "SemanticEncoderMetadata",
    "SemanticEncoderProtocol",
    "SemanticEncoderRegistry",
    "SyntheticEncoderAdapter",
    "TaskSupConCorroAdapter",
    "default_metadata",
]
