from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.access import (
    CANDIDATE_ROLLOUTS_Q_POLICY_IDENTITY,
    DEVELOPMENT_ORACLE_RETURNS_RANK_LABELS,
    PROBE_STYLE_LABELS,
    SOURCE_AXIS_CATEGORICAL_LABELS,
    SOURCE_NUMERIC_FACTOR_PARAMETERS,
    SOURCE_RAW_TRANSITIONS,
    SOURCE_TASK_CATEGORICAL_LABELS,
    EncoderAccessCard,
    EncoderAccessError,
)
from policy_learnware_v0.v03.data_roles import (
    DataRoleError,
    DataRoleManifest,
    DataRoleRecord,
    assert_confirmatory_queries_unseen,
    assert_process_can_read,
    validate_loto_isolation,
)
from policy_learnware_v0.v03.schemas import LOTOFoldRecord


def _d(label: str) -> str:
    return sha256_json({"test": label})


def test_access_tiers_and_undeclared_reads_fail_closed() -> None:
    card = EncoderAccessCard(
        encoder_id="corro-e1",
        access_tier="E1_CATEGORICAL_SOURCE",
        declared_capabilities=(
            SOURCE_RAW_TRANSITIONS,
            SOURCE_TASK_CATEGORICAL_LABELS,
        ),
        external_pretrained_weights_digest=None,
        max_hyperparameter_trials=3,
        total_train_compute_hours=2.5,
        formal_eligible=True,
    )
    assert EncoderAccessCard.from_dict(card.to_dict()) == card
    card.assert_can_read(SOURCE_RAW_TRANSITIONS)
    with pytest.raises(EncoderAccessError, match="undeclared"):
        card.assert_can_read("reward_channel")
    with pytest.raises(EncoderAccessError, match="exceed"):
        EncoderAccessCard(
            encoder_id="bad-e1",
            access_tier="E1_CATEGORICAL_SOURCE",
            declared_capabilities=(
                SOURCE_RAW_TRANSITIONS,
                SOURCE_NUMERIC_FACTOR_PARAMETERS,
            ),
            external_pretrained_weights_digest=None,
            max_hyperparameter_trials=1,
            total_train_compute_hours=1.0,
            formal_eligible=True,
        )

    with pytest.raises(EncoderAccessError, match="exceed"):
        EncoderAccessCard(
            encoder_id="bad-e1-axis",
            access_tier="E1_CATEGORICAL_SOURCE",
            declared_capabilities=(
                SOURCE_RAW_TRANSITIONS,
                SOURCE_AXIS_CATEGORICAL_LABELS,
            ),
            external_pretrained_weights_digest=None,
            max_hyperparameter_trials=1,
            total_train_compute_hours=1.0,
            formal_eligible=True,
        )


def test_probe_style_supervision_is_development_only() -> None:
    with pytest.raises(EncoderAccessError, match="formal E-table"):
        EncoderAccessCard(
            encoder_id="bad-formal-probe-style",
            access_tier="E1_CATEGORICAL_SOURCE",
            declared_capabilities=(SOURCE_RAW_TRANSITIONS, PROBE_STYLE_LABELS),
            external_pretrained_weights_digest=None,
            max_hyperparameter_trials=1,
            total_train_compute_hours=1.0,
            formal_eligible=True,
        )

    diagnostic = EncoderAccessCard(
        encoder_id="development-probe-style-diagnostic",
        access_tier="E1_CATEGORICAL_SOURCE",
        declared_capabilities=(SOURCE_RAW_TRANSITIONS, PROBE_STYLE_LABELS),
        external_pretrained_weights_digest=None,
        max_hyperparameter_trials=1,
        total_train_compute_hours=1.0,
        formal_eligible=False,
    )
    diagnostic.assert_can_read(PROBE_STYLE_LABELS)


def test_formal_card_cannot_read_oracle_or_candidate_evidence() -> None:
    with pytest.raises(EncoderAccessError, match="formal E-table"):
        EncoderAccessCard(
            encoder_id="bad-formal",
            access_tier="E1_CATEGORICAL_SOURCE",
            declared_capabilities=(
                SOURCE_RAW_TRANSITIONS,
                DEVELOPMENT_ORACLE_RETURNS_RANK_LABELS,
            ),
            external_pretrained_weights_digest=None,
            max_hyperparameter_trials=1,
            total_train_compute_hours=1.0,
            formal_eligible=True,
        )
    diagnostic = EncoderAccessCard(
        encoder_id="dev-diagnostic",
        access_tier="E1_CATEGORICAL_SOURCE",
        declared_capabilities=(
            SOURCE_RAW_TRANSITIONS,
            CANDIDATE_ROLLOUTS_Q_POLICY_IDENTITY,
        ),
        external_pretrained_weights_digest=None,
        max_hyperparameter_trials=1,
        total_train_compute_hours=1.0,
        formal_eligible=False,
    )
    with pytest.raises(EncoderAccessError, match="E-table"):
        diagnostic.validate_e_table()


def _role(
    role: str,
    name: str,
    tasks: tuple[str, ...],
    seeds: tuple[str, ...],
    *,
    nonce: str = "fold-c",
) -> DataRoleRecord:
    return DataRoleRecord(
        role=role,  # type: ignore[arg-type]
        dataset_id=name,
        dataset_digest=_d(f"dataset:{name}"),
        task_private_ids=tasks,
        seed_tokens=seeds,
        split_nonce_digest=_d(f"nonce:{nonce}"),
    )


def _manifest() -> DataRoleManifest:
    return DataRoleManifest(
        manifest_id="fold-c-roles",
        records=(
            _role("source_encoder_train", "train", ("task-a", "task-b"), ("s-train",)),
            _role(
                "source_encoder_validation",
                "validation",
                ("task-a", "task-b"),
                ("s-validation",),
            ),
            _role(
                "source_reference_spec",
                "reference",
                ("task-a", "task-b", "task-c"),
                ("s-reference",),
                nonce="reference",
            ),
            _role(
                "confirmatory_query",
                "query-c",
                ("task-c",),
                ("s-query",),
            ),
        ),
    )


def _fold(manifest: DataRoleManifest) -> LOTOFoldRecord:
    return LOTOFoldRecord(
        fold_id="fold-c",
        held_out_task_private_id="task-c",
        train_task_private_ids=("task-a", "task-b"),
        train_dataset_digests=tuple(
            record.dataset_digest
            for record in manifest.records_for("source_encoder_train")
        ),
        validation_dataset_digests=tuple(
            record.dataset_digest
            for record in manifest.records_for("source_encoder_validation")
        ),
        source_reference_role_digest=manifest.role_digest("source_reference_spec"),
        target_query_role_digest=manifest.role_digest("confirmatory_query"),
        split_nonce_digest=_d("nonce:fold-c"),
    )


def test_data_roles_roundtrip_process_access_and_loto_isolation() -> None:
    manifest = _manifest()
    assert DataRoleManifest.from_dict(manifest.to_dict()) == manifest
    validate_loto_isolation(
        _fold(manifest), manifest, target_query_role="confirmatory_query"
    )
    assert_process_can_read("encoder_trainer", "source_encoder_train")
    with pytest.raises(DataRoleError, match="cannot read"):
        assert_process_can_read("encoder_trainer", "confirmatory_query")
    assert_confirmatory_queries_unseen(manifest, [_d("some-development-dataset")])
    with pytest.raises(DataRoleError, match="downgraded"):
        assert_confirmatory_queries_unseen(
            manifest,
            [manifest.records_for("confirmatory_query")[0].dataset_digest],
        )


def test_loto_heldout_leak_and_seed_overlap_are_rejected() -> None:
    manifest = _manifest()
    bad_records = list(manifest.records)
    train_index = next(
        index
        for index, record in enumerate(bad_records)
        if record.role == "source_encoder_train"
    )
    bad_records[train_index] = replace(
        bad_records[train_index], task_private_ids=("task-a", "task-b", "task-c")
    )
    leaked = DataRoleManifest("leaked", tuple(bad_records))
    with pytest.raises(DataRoleError, match="held-out"):
        validate_loto_isolation(
            _fold(leaked), leaked, target_query_role="confirmatory_query"
        )

    query = manifest.records_for("confirmatory_query")[0]
    overlapping = replace(query, seed_tokens=("s-reference",))
    with pytest.raises(DataRoleError, match="seed tokens overlap"):
        DataRoleManifest(
            "overlap",
            tuple(
                overlapping if item.role == "confirmatory_query" else item
                for item in manifest.records
            ),
        )
