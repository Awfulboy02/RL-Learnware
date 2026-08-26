from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.data_roles import (
    DataRoleError,
    DataRoleManifest,
    DataRoleRecord,
    assert_process_can_read,
    validate_representation_isolation,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _record(role: str, name: str, tasks: tuple[str, ...], seed: str) -> DataRoleRecord:
    return DataRoleRecord(
        role=role,  # type: ignore[arg-type]
        dataset_id=name,
        dataset_digest=_d(f"dataset:{name}"),
        task_private_ids=tasks,
        seed_tokens=(seed,),
        split_nonce_digest=_d("split"),
    )


def _manifest() -> DataRoleManifest:
    return DataRoleManifest(
        manifest_id="v03-representation-roles",
        records=(
            _record("archived_legacy_replay", "archive", ("task-a",), "archive-seed"),
            _record(
                "source_representation_train",
                "fit-train",
                ("task-a", "task-b"),
                "fit-train-seed",
            ),
            _record(
                "source_representation_validation",
                "fit-validation",
                ("task-a", "task-b"),
                "fit-validation-seed",
            ),
            _record(
                "source_reference_spec",
                "reference",
                ("task-a", "task-b"),
                "reference-seed",
            ),
            _record("development_query", "query", ("task-a",), "query-seed"),
        ),
    )


def test_active_v03_representation_roles_are_non_loto_and_isolated() -> None:
    manifest = _manifest()
    validate_representation_isolation(manifest)
    assert_process_can_read("legacy_replay", "archived_legacy_replay")
    assert_process_can_read("representation_trainer", "source_representation_train")
    assert_process_can_read("canonicalizer_fitter", "source_representation_validation")


def test_representation_fit_seed_overlap_and_reference_coverage_fail() -> None:
    manifest = _manifest()
    validation = manifest.records_for("source_representation_validation")[0]
    with pytest.raises(DataRoleError, match="representation train and validation"):
        DataRoleManifest(
            "overlap",
            tuple(
                replace(item, seed_tokens=("fit-train-seed",))
                if item is validation
                else item
                for item in manifest.records
            ),
        )

    reference = manifest.records_for("source_reference_spec")[0]
    missing = DataRoleManifest(
        "missing-reference-task",
        tuple(
            replace(item, task_private_ids=("task-a",)) if item is reference else item
            for item in manifest.records
        ),
    )
    with pytest.raises(DataRoleError, match="cover every"):
        validate_representation_isolation(missing)
