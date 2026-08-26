from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from policy_learnware_v0.hashing import sha256_bytes, sha256_json
from policy_learnware_v0.v03.artifacts import V03ArtifactLayout
from policy_learnware_v0.v03.checkpoints import (
    EncoderCheckpointError,
    freeze_encoder_checkpoint,
    load_frozen_encoder_checkpoint,
)
from policy_learnware_v0.v03.encoder_protocol import CostRecord, EncoderFitResult
from policy_learnware_v0.v03.schemas import EncoderProtocolRecord


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _protocol() -> EncoderProtocolRecord:
    return EncoderProtocolRecord(
        encoder_id="fake-e0",
        family="fake",
        implementation_digest=_d("implementation"),
        input_view_digest=_d("view"),
        window_protocol_digest=_d("window"),
        access_card_digest=_d("access"),
        architecture_digest=_d("architecture"),
        objective_digest=_d("objective"),
        training_protocol_digest=_d("training"),
        latent_dim=4,
    )


def _fit(protocol: EncoderProtocolRecord, payload: bytes) -> EncoderFitResult:
    return EncoderFitResult(
        encoder_id=protocol.encoder_id,
        checkpoint_digest=sha256_bytes(payload),
        training_manifest_digest=_d("training-manifest"),
        protocol_record_digest=protocol.protocol_record_digest,
        training_contract_digest=_d("training-contract"),
        access_card_digest=protocol.access_card_digest,
        input_view_digest=protocol.input_view_digest,
        semantic_output_protocol_digest=_d("semantic-output"),
        runtime_digest=_d("runtime"),
        fold_id="fold-a",
        seed=7,
        latent_dim=protocol.latent_dim,
        training_cost=CostRecord(1.0, 128, "cpu", 1),
    )


def _expected_load(fit: EncoderFitResult) -> dict[str, object]:
    return {
        "expected_fold_id": fit.fold_id,
        "expected_seed": fit.seed,
        "expected_training_manifest_digest": fit.training_manifest_digest,
        "expected_training_contract_digest": fit.training_contract_digest,
        "expected_runtime_digest": fit.runtime_digest,
        "expected_semantic_output_protocol_digest": (
            fit.semantic_output_protocol_digest
        ),
    }


def test_checkpoint_freeze_is_content_bound_immutable_and_loadable(
    tmp_path: Path,
) -> None:
    payload = b"deterministic-frozen-checkpoint"
    protocol = _protocol()
    fit = _fit(protocol, payload)
    layout = V03ArtifactLayout.development(tmp_path, "dev-r0")
    checkpoint_path = layout.encoder_checkpoint_artifact(
        "fold-a", protocol.encoder_id, "checkpoint.bin"
    )
    manifest_path = layout.encoder_checkpoint_artifact(
        "fold-a", protocol.encoder_id, "checkpoint_manifest.json"
    )
    publication = freeze_encoder_checkpoint(
        writer=layout.writer("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        checkpoint_bytes=payload,
        fit=fit,
        protocol_record=protocol,
        fold_id="fold-a",
        seed=7,
        runtime_digest=_d("runtime"),
        semantic_output_protocol_digest=_d("semantic-output"),
    )
    resumed = freeze_encoder_checkpoint(
        writer=layout.writer("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        checkpoint_bytes=payload,
        fit=fit,
        protocol_record=protocol,
        fold_id="fold-a",
        seed=7,
        runtime_digest=_d("runtime"),
        semantic_output_protocol_digest=_d("semantic-output"),
        resume=True,
    )
    assert resumed.manifest_artifact_digest == publication.manifest_artifact_digest

    manifest, loaded = load_frozen_encoder_checkpoint(
        reader=layout.reader("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        expected_manifest_artifact_digest=publication.manifest_artifact_digest,
        expected_protocol_record=protocol,
        **_expected_load(fit),
    )
    assert loaded == payload
    assert manifest.checkpoint_digest == fit.checkpoint_digest


def test_checkpoint_rejects_abstract_digest_and_protocol_drift(tmp_path: Path) -> None:
    payload = b"checkpoint"
    protocol = _protocol()
    bad_fit = replace(_fit(protocol, payload), checkpoint_digest=_d("not-content"))
    layout = V03ArtifactLayout.development(tmp_path, "dev-r0")
    checkpoint_path = layout.encoder_checkpoint_artifact(
        "fold-a", protocol.encoder_id, "checkpoint.bin"
    )
    manifest_path = layout.encoder_checkpoint_artifact(
        "fold-a", protocol.encoder_id, "checkpoint_manifest.json"
    )
    with pytest.raises(EncoderCheckpointError, match="exact checkpoint content"):
        freeze_encoder_checkpoint(
            writer=layout.writer("encoder_checkpoints"),
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            manifest_path=manifest_path,  # type: ignore[arg-type]
            checkpoint_bytes=payload,
            fit=bad_fit,
            protocol_record=protocol,
            fold_id="fold-a",
            seed=7,
            runtime_digest=_d("runtime"),
            semantic_output_protocol_digest=_d("semantic-output"),
        )

    fit = replace(bad_fit, checkpoint_digest=sha256_bytes(payload))
    publication = freeze_encoder_checkpoint(
        writer=layout.writer("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        checkpoint_bytes=payload,
        fit=fit,
        protocol_record=protocol,
        fold_id="fold-a",
        seed=7,
        runtime_digest=_d("runtime"),
        semantic_output_protocol_digest=_d("semantic-output"),
    )
    drifted = replace(protocol, architecture_digest=_d("changed-architecture"))
    with pytest.raises(EncoderCheckpointError, match="different encoder protocol"):
        load_frozen_encoder_checkpoint(
            reader=layout.reader("encoder_checkpoints"),
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            manifest_path=manifest_path,  # type: ignore[arg-type]
            expected_manifest_artifact_digest=publication.manifest_artifact_digest,
            expected_protocol_record=drifted,
            **_expected_load(fit),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("expected_fold_id", "fold-b", "fold"),
        ("expected_seed", 8, "seed"),
        (
            "expected_training_manifest_digest",
            _d("other-training-manifest"),
            "training manifest",
        ),
        (
            "expected_training_contract_digest",
            _d("other-training-contract"),
            "training contract",
        ),
        ("expected_runtime_digest", _d("other-runtime"), "runtime"),
        (
            "expected_semantic_output_protocol_digest",
            _d("other-semantic-output"),
            "semantic output",
        ),
    ),
)
def test_checkpoint_load_requires_every_private_binding(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    payload = b"bound-checkpoint"
    protocol = _protocol()
    fit = _fit(protocol, payload)
    layout = V03ArtifactLayout.development(tmp_path, "dev-r0")
    checkpoint_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, protocol.encoder_id, "checkpoint.bin"
    )
    manifest_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, protocol.encoder_id, "checkpoint_manifest.json"
    )
    publication = freeze_encoder_checkpoint(
        writer=layout.writer("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
        manifest_path=manifest_path,  # type: ignore[arg-type]
        checkpoint_bytes=payload,
        fit=fit,
        protocol_record=protocol,
        fold_id=fit.fold_id,
        seed=fit.seed,
        runtime_digest=fit.runtime_digest,
        semantic_output_protocol_digest=fit.semantic_output_protocol_digest,
    )
    expected = _expected_load(fit)
    expected[field] = value
    with pytest.raises(EncoderCheckpointError, match=match):
        load_frozen_encoder_checkpoint(
            reader=layout.reader("encoder_checkpoints"),
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            manifest_path=manifest_path,  # type: ignore[arg-type]
            expected_manifest_artifact_digest=publication.manifest_artifact_digest,
            expected_protocol_record=protocol,
            **expected,  # type: ignore[arg-type]
        )


def test_checkpoint_publication_rejects_non_encoder_private_domain(
    tmp_path: Path,
) -> None:
    payload = b"private-checkpoint"
    protocol = _protocol()
    fit = _fit(protocol, payload)
    layout = V03ArtifactLayout.development(tmp_path, "dev-r0")
    checkpoint_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, protocol.encoder_id, "checkpoint.bin"
    )
    manifest_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, protocol.encoder_id, "checkpoint_manifest.json"
    )
    with pytest.raises(EncoderCheckpointError, match="private artifact domain"):
        freeze_encoder_checkpoint(
            writer=layout.writer("source_market_private"),
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            manifest_path=manifest_path,  # type: ignore[arg-type]
            checkpoint_bytes=payload,
            fit=fit,
            protocol_record=protocol,
            fold_id=fit.fold_id,
            seed=fit.seed,
            runtime_digest=fit.runtime_digest,
            semantic_output_protocol_digest=fit.semantic_output_protocol_digest,
        )
