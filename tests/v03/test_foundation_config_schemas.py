from __future__ import annotations

import copy

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.config import (
    EncoderExtensionGateConfig,
    V03ConfigError,
    V03FoundationConfig,
    load_v03_foundation_config,
)
from policy_learnware_v0.v03.schemas import (
    ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST,
    AnonymousSelectorViewEntry,
    EncoderProtocolRecord,
    LOTOFoldRecord,
    V03SchemaError,
    validate_public_mapping,
)
from policy_learnware_v0.v03.windowing import WindowingProtocol


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _payload() -> dict:
    return {
        "schema": "policy-learnware.v03-foundation-config.v0",
        "development_id": "v03-foundation-test",
        "stage": "foundation_development",
        "protocol_id": _d("protocol"),
        "task_private_ids": ["task-a", "task-b", "task-c"],
        "artifact_root": "/tmp/policy-learnware-v03-test",
        "anonymous_public_allowlist": sorted(ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST),
        "window_protocol": {
            "window_length": 4,
            "stride": 2,
            "pooling": "mean",
            "pad_final_window": True,
            "protocol_id": WindowingProtocol(
                window_length=4,
                stride=2,
                pooling="mean",
                pad_final_window=True,
            ).window_protocol_digest,
        },
        "primary_freeze": {
            "query_mode": "QUERY_EMPIRICAL",
            "selector_mode": "distance_only",
            "pool_scope": "anonymous_global",
            "opaque_learnware_field": "opaque_learnware_id",
            "opaque_query_field": "opaque_query_id",
            "oracle_owner": "policy-learnware-paper1",
        },
        "review_decisions_digest": None,
    }


def test_foundation_config_is_strict_and_digest_bound() -> None:
    config = V03FoundationConfig.from_dict(_payload())
    assert config.config_digest == V03FoundationConfig.from_dict(
        config.to_dict()
    ).config_digest
    changed = _payload()
    changed["window_protocol"]["stride"] = 1
    changed["window_protocol"]["protocol_id"] = WindowingProtocol(
        window_length=4,
        stride=1,
        pooling="mean",
        pad_final_window=True,
    ).window_protocol_digest
    assert V03FoundationConfig.from_dict(changed).config_digest != config.config_digest

    drifted_id = _payload()
    drifted_id["window_protocol"]["stride"] = 1
    with pytest.raises(V03ConfigError, match="derived"):
        V03FoundationConfig.from_dict(drifted_id)

    unknown = _payload()
    unknown["surprise"] = True
    with pytest.raises(V03ConfigError, match="unknown"):
        V03FoundationConfig.from_dict(unknown)
    nested = _payload()
    nested["primary_freeze"]["task_id"] = "leak"
    with pytest.raises(V03ConfigError, match="unknown"):
        V03FoundationConfig.from_dict(nested)


def test_encoder_extension_gate_defaults_disabled_without_asset_requirements() -> None:
    omitted = V03FoundationConfig.from_dict(_payload())
    assert omitted.encoder_extension_gate == EncoderExtensionGateConfig.disabled()
    assert omitted.encoder_extension_gate.to_dict() == {"enabled": False}

    explicit = _payload()
    explicit["encoder_extension_gate"] = {"enabled": False}
    configured = V03FoundationConfig.from_dict(explicit)
    assert configured.encoder_extension_gate == omitted.encoder_extension_gate
    assert configured.config_digest == omitted.config_digest
    assert V03FoundationConfig.from_dict(configured.to_dict()) == configured


def test_encoder_extension_gate_is_opt_in_v04_and_fail_closed() -> None:
    enabled = _payload()
    enabled["encoder_extension_gate"] = {
        "enabled": True,
        "migration_target": "v0.4",
        "authority_digest": _d("v04-migration-authority"),
    }
    config = V03FoundationConfig.from_dict(enabled)
    assert config.encoder_extension_gate.enabled is True
    assert config.encoder_extension_gate.migration_target == "v0.4"
    assert config.encoder_extension_gate.authority_digest == _d(
        "v04-migration-authority"
    )
    assert config.stage == "foundation_development"

    formal = copy.deepcopy(enabled)
    formal["stage"] = "formal_freeze"
    formal["review_decisions_digest"] = _d("declared-review")
    with pytest.raises(
        V03ConfigError,
        match=r"formal_freeze requires encoder_extension_gate\.enabled=false",
    ):
        V03FoundationConfig.from_dict(formal)

    malformed_cases = [
        {"enabled": 1},
        {"enabled": True},
        {
            "enabled": True,
            "migration_target": "v0.3",
            "authority_digest": _d("authority"),
        },
        {
            "enabled": True,
            "migration_target": "v0.4",
            "authority_digest": "not-a-digest",
        },
        {
            "enabled": False,
            "migration_target": "v0.4",
            "authority_digest": _d("authority"),
        },
        {"enabled": False, "checkpoint": "/tmp/not-allowed"},
    ]
    for gate in malformed_cases:
        payload = _payload()
        payload["encoder_extension_gate"] = gate
        with pytest.raises(V03ConfigError):
            V03FoundationConfig.from_dict(payload)


def test_formal_stage_requires_declared_review_digest_and_no_markers() -> None:
    formal = _payload()
    formal["stage"] = "formal_freeze"
    with pytest.raises(V03ConfigError, match="review_decisions_digest"):
        V03FoundationConfig.from_dict(formal)
    formal["review_decisions_digest"] = _d("reviewed")
    assert V03FoundationConfig.from_dict(formal).stage == "formal_freeze"

    unresolved = _payload()
    unresolved["primary_freeze"]["query_mode"] = "TBD"
    with pytest.raises(V03ConfigError, match="unresolved"):
        V03FoundationConfig.from_dict(unresolved)


def test_anonymous_public_projection_has_closed_allowlist() -> None:
    entry = AnonymousSelectorViewEntry(
        opaque_learnware_id="lw-00000000000000000000000000000001",
        environment_spec_digest=_d("spec"),
        normalized_source_competence=0.7,
        tie_break_token=_d("token-1"),
    )
    assert AnonymousSelectorViewEntry.from_dict(entry.to_dict()) == entry
    leaked = entry.to_dict()
    leaked["task_id"] = "WalkerWalk"
    with pytest.raises(V03SchemaError, match="unknown|forbidden"):
        AnonymousSelectorViewEntry.from_dict(leaked)
    with pytest.raises(V03SchemaError, match="forbidden"):
        validate_public_mapping(
            {"entries": [{"task_id": "WalkerWalk"}]},
            allowlist={"entries"},
            required={"entries"},
        )
    with pytest.raises(V03SchemaError, match=r"\[0, 1\]"):
        AnonymousSelectorViewEntry(
            opaque_learnware_id="lw-00000000000000000000000000000001",
            environment_spec_digest=_d("spec"),
            normalized_source_competence=1.1,
            tie_break_token=_d("token-1"),
        )
    with pytest.raises(V03SchemaError, match="lowercase"):
        AnonymousSelectorViewEntry(
            opaque_learnware_id="lw-00000000000000000000000000000001",
            environment_spec_digest=_d("spec").upper(),
            normalized_source_competence=0.7,
            tie_break_token=_d("token-1"),
        )
    with pytest.raises(V03SchemaError, match="canonical format"):
        AnonymousSelectorViewEntry(
            opaque_learnware_id="v03lw-00000000000000000000000000000001",
            environment_spec_digest=_d("spec"),
            normalized_source_competence=0.7,
            tie_break_token=_d("token-1"),
        )


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


def test_protocol_and_loto_records_roundtrip_and_reject_tamper() -> None:
    protocol = _protocol()
    assert EncoderProtocolRecord.from_dict(protocol.to_dict()) == protocol
    tampered = copy.deepcopy(protocol.to_dict())
    tampered["latent_dim"] = 8
    with pytest.raises(V03SchemaError, match="does not match"):
        EncoderProtocolRecord.from_dict(tampered)

    fold = LOTOFoldRecord(
        fold_id="fold-c",
        held_out_task_private_id="task-c",
        train_task_private_ids=("task-a", "task-b"),
        train_dataset_digests=(_d("train"),),
        validation_dataset_digests=(_d("validation"),),
        source_reference_role_digest=_d("source-ref"),
        target_query_role_digest=_d("query"),
        split_nonce_digest=_d("nonce"),
    )
    assert LOTOFoldRecord.from_dict(fold.to_dict()) == fold
    with pytest.raises(V03SchemaError, match="held-out"):
        LOTOFoldRecord(
            fold_id="bad",
            held_out_task_private_id="task-c",
            train_task_private_ids=("task-a", "task-c"),
            train_dataset_digests=(_d("train"),),
            validation_dataset_digests=(_d("validation"),),
            source_reference_role_digest=_d("source-ref"),
            target_query_role_digest=_d("query"),
            split_nonce_digest=_d("nonce"),
        )


def test_config_loader_rejects_duplicate_yaml_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema: one\nschema: two\n", encoding="utf-8")
    with pytest.raises(V03ConfigError, match="duplicate YAML key"):
        load_v03_foundation_config(path)
