"""Immutable encoder checkpoint publication and protocol-bound loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from ..hashing import sha256_bytes, sha256_json
from .artifacts import V03ArtifactReader, V03ArtifactWriter
from .encoder_protocol import EncoderFitResult
from .schemas import (
    EncoderProtocolRecord,
    V03SchemaError,
    checked_digest,
    checked_safe_id,
    strict_mapping,
)


ENCODER_CHECKPOINT_MANIFEST_SCHEMA = (
    "policy-learnware.v03-encoder-checkpoint-manifest.v0"
)


class EncoderCheckpointError(V03SchemaError):
    """Checkpoint bytes, manifest, or protocol binding is inconsistent."""


@dataclass(frozen=True)
class EncoderCheckpointManifest:
    encoder_id: str
    fold_id: str
    seed: int
    checkpoint_digest: str
    checkpoint_size_bytes: int
    protocol_record_digest: str
    training_manifest_digest: str
    training_contract_digest: str
    access_card_digest: str
    runtime_digest: str
    input_view_digest: str
    window_protocol_digest: str
    semantic_output_protocol_digest: str
    latent_dim: int
    frozen: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_id", checked_safe_id(self.encoder_id, "encoder_id"))
        object.__setattr__(self, "fold_id", checked_safe_id(self.fold_id, "fold_id"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EncoderCheckpointError("checkpoint seed must be a non-negative integer")
        if (
            isinstance(self.checkpoint_size_bytes, bool)
            or not isinstance(self.checkpoint_size_bytes, int)
            or self.checkpoint_size_bytes <= 0
        ):
            raise EncoderCheckpointError("checkpoint_size_bytes must be positive")
        for name in (
            "checkpoint_digest",
            "protocol_record_digest",
            "training_manifest_digest",
            "training_contract_digest",
            "access_card_digest",
            "runtime_digest",
            "input_view_digest",
            "window_protocol_digest",
            "semantic_output_protocol_digest",
        ):
            object.__setattr__(self, name, checked_digest(getattr(self, name), name))
        if isinstance(self.latent_dim, bool) or not isinstance(self.latent_dim, int) or self.latent_dim <= 0:
            raise EncoderCheckpointError("latent_dim must be a positive integer")
        if self.frozen is not True:
            raise EncoderCheckpointError("published encoder checkpoint must be frozen=true")

    def material_dict(self) -> dict[str, Any]:
        return {
            "schema": ENCODER_CHECKPOINT_MANIFEST_SCHEMA,
            "encoder_id": self.encoder_id,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "checkpoint_digest": self.checkpoint_digest,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "protocol_record_digest": self.protocol_record_digest,
            "training_manifest_digest": self.training_manifest_digest,
            "training_contract_digest": self.training_contract_digest,
            "access_card_digest": self.access_card_digest,
            "runtime_digest": self.runtime_digest,
            "input_view_digest": self.input_view_digest,
            "window_protocol_digest": self.window_protocol_digest,
            "semantic_output_protocol_digest": self.semantic_output_protocol_digest,
            "latent_dim": self.latent_dim,
            "frozen": self.frozen,
        }

    @property
    def checkpoint_manifest_digest(self) -> str:
        return sha256_json(self.material_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.material_dict(),
            "checkpoint_manifest_digest": self.checkpoint_manifest_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EncoderCheckpointManifest":
        fields = {
            "schema",
            "encoder_id",
            "fold_id",
            "seed",
            "checkpoint_digest",
            "checkpoint_size_bytes",
            "protocol_record_digest",
            "training_manifest_digest",
            "training_contract_digest",
            "access_card_digest",
            "runtime_digest",
            "input_view_digest",
            "window_protocol_digest",
            "semantic_output_protocol_digest",
            "latent_dim",
            "frozen",
            "checkpoint_manifest_digest",
        }
        data = strict_mapping(value, fields, "encoder checkpoint manifest")
        if data["schema"] != ENCODER_CHECKPOINT_MANIFEST_SCHEMA:
            raise EncoderCheckpointError("unknown encoder checkpoint manifest schema")
        manifest = cls(
            **{
                name: data[name]
                for name in fields - {"schema", "checkpoint_manifest_digest"}
            }
        )
        expected = checked_digest(
            data["checkpoint_manifest_digest"], "checkpoint_manifest_digest"
        )
        if expected != manifest.checkpoint_manifest_digest:
            raise EncoderCheckpointError("checkpoint manifest digest does not match payload")
        return manifest


@runtime_checkable
class ImmutableEncoderCheckpointAdapter(Protocol):
    """Minimal adapter surface for byte-exact checkpoint persistence.

    The artifact layer owns publication and digest verification.  An adapter
    owns only the deterministic conversion between its typed fit result and
    canonical checkpoint bytes, plus reconstruction on a fresh instance.
    """

    def export_frozen_checkpoint_bytes(self, fit: EncoderFitResult) -> bytes: ...

    def load_frozen_checkpoint_bytes(
        self,
        *,
        manifest: EncoderCheckpointManifest,
        checkpoint_bytes: bytes,
    ) -> EncoderFitResult: ...


@dataclass(frozen=True)
class FrozenCheckpointPublication:
    manifest: EncoderCheckpointManifest
    checkpoint_artifact_digest: str
    manifest_artifact_digest: str


def _assert_encoder_private_domain(
    store: V03ArtifactWriter | V03ArtifactReader,
) -> None:
    expected = (
        "encoder_checkpoints"
        if store.layout.namespace == "development"
        else "encoder_training_private"
    )
    if store.domain != expected:
        raise EncoderCheckpointError(
            f"encoder checkpoints require the {expected!r} private artifact domain"
        )


def freeze_encoder_checkpoint(
    *,
    writer: V03ArtifactWriter,
    checkpoint_path: str,
    manifest_path: str,
    checkpoint_bytes: bytes,
    fit: EncoderFitResult,
    protocol_record: EncoderProtocolRecord,
    fold_id: str,
    seed: int,
    runtime_digest: str,
    semantic_output_protocol_digest: str,
    resume: bool = False,
) -> FrozenCheckpointPublication:
    _assert_encoder_private_domain(writer)
    payload = bytes(checkpoint_bytes)
    if not payload:
        raise EncoderCheckpointError("checkpoint payload cannot be empty")
    content_digest = sha256_bytes(payload)
    if content_digest != fit.checkpoint_digest:
        raise EncoderCheckpointError(
            "fit checkpoint_digest must be the exact checkpoint content digest"
        )
    if fit.encoder_id != protocol_record.encoder_id:
        raise EncoderCheckpointError("fit/protocol encoder IDs disagree")
    if fit.protocol_record_digest != protocol_record.protocol_record_digest:
        raise EncoderCheckpointError("fit/protocol record digests disagree")
    if fit.access_card_digest != protocol_record.access_card_digest:
        raise EncoderCheckpointError("fit/protocol access-card digests disagree")
    if fit.input_view_digest != protocol_record.input_view_digest:
        raise EncoderCheckpointError("fit/protocol input-view digests disagree")
    if fit.latent_dim != protocol_record.latent_dim:
        raise EncoderCheckpointError("fit/protocol latent dimensions disagree")
    if fit.fold_id != fold_id or fit.seed != seed:
        raise EncoderCheckpointError("fit fold/seed differs from checkpoint publication")
    expected_runtime = checked_digest(runtime_digest, "runtime_digest")
    if fit.runtime_digest != expected_runtime:
        raise EncoderCheckpointError("fit runtime differs from checkpoint publication")
    expected_semantic = checked_digest(
        semantic_output_protocol_digest, "semantic_output_protocol_digest"
    )
    if fit.semantic_output_protocol_digest != expected_semantic:
        raise EncoderCheckpointError(
            "fit semantic-output protocol differs from checkpoint publication"
        )
    manifest = EncoderCheckpointManifest(
        encoder_id=fit.encoder_id,
        fold_id=fold_id,
        seed=seed,
        checkpoint_digest=content_digest,
        checkpoint_size_bytes=len(payload),
        protocol_record_digest=protocol_record.protocol_record_digest,
        training_manifest_digest=fit.training_manifest_digest,
        training_contract_digest=fit.training_contract_digest,
        access_card_digest=fit.access_card_digest,
        runtime_digest=expected_runtime,
        input_view_digest=fit.input_view_digest,
        window_protocol_digest=protocol_record.window_protocol_digest,
        semantic_output_protocol_digest=expected_semantic,
        latent_dim=fit.latent_dim,
    )
    checkpoint_artifact_digest = writer.publish_bytes(
        checkpoint_path, payload, resume=resume
    )
    manifest_artifact_digest = writer.publish_json(
        manifest_path, manifest.to_dict(), resume=resume
    )
    return FrozenCheckpointPublication(
        manifest=manifest,
        checkpoint_artifact_digest=checkpoint_artifact_digest,
        manifest_artifact_digest=manifest_artifact_digest,
    )


def load_frozen_encoder_checkpoint(
    *,
    reader: V03ArtifactReader,
    checkpoint_path: str,
    manifest_path: str,
    expected_manifest_artifact_digest: str,
    expected_protocol_record: EncoderProtocolRecord,
    expected_fold_id: str,
    expected_seed: int,
    expected_training_manifest_digest: str,
    expected_training_contract_digest: str,
    expected_runtime_digest: str,
    expected_semantic_output_protocol_digest: str,
) -> tuple[EncoderCheckpointManifest, bytes]:
    _assert_encoder_private_domain(reader)
    raw_manifest = reader.load_json(
        manifest_path, expected_sha256=expected_manifest_artifact_digest
    )
    if not isinstance(raw_manifest, Mapping):
        raise EncoderCheckpointError("checkpoint manifest artifact must be a mapping")
    manifest = EncoderCheckpointManifest.from_dict(raw_manifest)
    if manifest.protocol_record_digest != expected_protocol_record.protocol_record_digest:
        raise EncoderCheckpointError("checkpoint references a different encoder protocol")
    if manifest.encoder_id != expected_protocol_record.encoder_id:
        raise EncoderCheckpointError("checkpoint encoder ID differs from protocol")
    if manifest.access_card_digest != expected_protocol_record.access_card_digest:
        raise EncoderCheckpointError("checkpoint access-card digest differs from protocol")
    if manifest.input_view_digest != expected_protocol_record.input_view_digest:
        raise EncoderCheckpointError("checkpoint input-view digest differs from protocol")
    if manifest.window_protocol_digest != expected_protocol_record.window_protocol_digest:
        raise EncoderCheckpointError("checkpoint window protocol differs from protocol")
    if manifest.latent_dim != expected_protocol_record.latent_dim:
        raise EncoderCheckpointError("checkpoint latent dimension differs from protocol")
    if manifest.fold_id != checked_safe_id(expected_fold_id, "expected_fold_id"):
        raise EncoderCheckpointError("checkpoint fold differs from expected fold")
    if (
        isinstance(expected_seed, bool)
        or not isinstance(expected_seed, int)
        or expected_seed < 0
    ):
        raise EncoderCheckpointError("expected_seed must be a non-negative integer")
    if manifest.seed != expected_seed:
        raise EncoderCheckpointError("checkpoint seed differs from expected seed")
    expected_bindings = {
        "training manifest": (
            manifest.training_manifest_digest,
            checked_digest(
                expected_training_manifest_digest,
                "expected_training_manifest_digest",
            ),
        ),
        "training contract": (
            manifest.training_contract_digest,
            checked_digest(
                expected_training_contract_digest,
                "expected_training_contract_digest",
            ),
        ),
        "runtime": (
            manifest.runtime_digest,
            checked_digest(expected_runtime_digest, "expected_runtime_digest"),
        ),
        "semantic output": (
            manifest.semantic_output_protocol_digest,
            checked_digest(
                expected_semantic_output_protocol_digest,
                "expected_semantic_output_protocol_digest",
            ),
        ),
    }
    for name, (actual, expected) in expected_bindings.items():
        if actual != expected:
            raise EncoderCheckpointError(f"checkpoint {name} digest differs from expected")
    payload = reader.load_bytes(
        checkpoint_path, expected_sha256=manifest.checkpoint_digest
    )
    if len(payload) != manifest.checkpoint_size_bytes:
        raise EncoderCheckpointError("checkpoint byte size differs from manifest")
    return manifest, payload


__all__ = [
    "ENCODER_CHECKPOINT_MANIFEST_SCHEMA",
    "EncoderCheckpointError",
    "EncoderCheckpointManifest",
    "FrozenCheckpointPublication",
    "ImmutableEncoderCheckpointAdapter",
    "freeze_encoder_checkpoint",
    "load_frozen_encoder_checkpoint",
]
