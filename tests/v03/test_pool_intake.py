from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from policy_learnware_v0.hashing import canonical_json_bytes, sha256_file, sha256_json
from policy_learnware_v0.v03.pool_intake import (
    FROZEN_V02_EXACT90_TRUST_ANCHOR,
    FROZEN_V02_VALIDATOR_FILE_SHA256,
    FROZEN_V02_VALIDATOR_REPO_PATH,
    PoolIntakeError,
    V03PoolIntakeRecord,
    _load_verified_v02_intake,
    _intake_v02_policy_pool,
    assert_frozen_v02_intake_authority,
)
sys.path.insert(0, str(Path(__file__).parent))

from p5_asset_fixtures import (  # noqa: E402
    exact90_handoff,
    refresh_handoff_trust,
    self_digest,
    write_json,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_replayer(_root: Path, handoff: Path, _promotions):
    return _load(handoff / "policy_pool_acceptance.json")


def _intake_fixture(root: Path, handoff: Path, trust):
    return _intake_v02_policy_pool(
        handoff,
        trusted_experiment_root=root,
        trust_anchor=trust,
        _acceptance_replayer=_fixture_replayer,
    )


def _rebind_queue_sha(root: Path, handoff: Path):
    queue_path = root / "training_private" / "server_runs" / "queue_status.json"
    queue_sha = sha256_file(queue_path)
    promotion_path = handoff / "compiled_parity_promotions.json"
    promotion = _load(promotion_path)
    promotion["queue_status_sha256"] = queue_sha
    promotion = self_digest(
        {key: value for key, value in promotion.items() if key != "manifest_digest"},
        "manifest_digest",
    )
    write_json(promotion_path, promotion)
    acceptance_path = handoff / "policy_pool_acceptance.json"
    acceptance = _load(acceptance_path)
    acceptance["queue_status_sha256"] = queue_sha
    acceptance["promotion_manifest_digest"] = promotion["manifest_digest"]
    acceptance = self_digest(
        {key: value for key, value in acceptance.items() if key != "report_digest"},
        "report_digest",
    )
    write_json(acceptance_path, acceptance)
    return refresh_handoff_trust(
        root,
        handoff,
        plan_digest=acceptance["server_plan_digest"],
        queue_digest=queue_sha,
    )


def test_exact90_overlay_becomes_distinct_v03_pool_ready_record(tmp_path: Path) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    record = _intake_fixture(root, handoff, trust)
    assert record.pool_state == "POOL_READY"
    assert len(record.cells) == 90
    assert len(record.candidates_by_anchor) == 30
    assert all([cell.seed for cell in cells] == [0, 1, 2] for cells in record.candidates_by_anchor.values())
    assert sum(cell.resolution == "direct_terminal_record" for cell in record.cells.values()) == 84
    assert sum(
        cell.resolution == "compiled_parity_fallback_promotion"
        for cell in record.cells.values()
    ) == 6
    assert record.schema == "policy-learnware.v03-v02-pool-intake.v0"
    assert "TrainingRunRecord" not in json.dumps(record.to_dict())
    assert V03PoolIntakeRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()


def test_production_entrypoint_uses_non_overridable_reviewed_digests() -> None:
    anchor = FROZEN_V02_EXACT90_TRUST_ANCHOR
    assert anchor.server_plan_digest == "aa6a969ef70b5ca0d73b2cb3efb119860d3ca796d336c741fc2e9895ce05278c"
    assert anchor.queue_status_sha256 == "ff44038bb69bf3dea710acd3c20757ddcaefb1a33eb852365f5158a10d42f6f8"
    assert anchor.pool_digest == "e478ef1d38b7eea1a38691d4ea2bd25dc0356cd7264f5a5bd6df5e6de5e0d15f"
    repository_root = Path(__file__).resolve().parents[2]
    validator = repository_root / FROZEN_V02_VALIDATOR_REPO_PATH
    assert sha256_file(validator) == FROZEN_V02_VALIDATOR_FILE_SHA256


def test_identity_replayer_cannot_bypass_direct_training_record_mutation(
    tmp_path: Path,
) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    acceptance_path = handoff / "policy_pool_acceptance.json"
    acceptance = _load(acceptance_path)
    job_id = next(
        job_id
        for job_id, cell in acceptance["cells"].items()
        if cell["resolution"] == "direct_terminal_record"
    )
    cell = acceptance["cells"][job_id]
    record_path = (
        root
        / "training_private"
        / "server_runs"
        / "jobs"
        / job_id
        / f"attempt_{cell['attempt_number']:03d}"
        / "training_record.json"
    )
    record = _load(record_path)
    record["seed"] = (record["seed"] + 1) % 3
    record = self_digest(
        {key: value for key, value in record.items() if key != "record_digest"},
        "record_digest",
    )
    write_json(record_path, record)
    cell["training_record_digest"] = record["record_digest"]
    acceptance = self_digest(
        {key: value for key, value in acceptance.items() if key != "report_digest"},
        "report_digest",
    )
    write_json(acceptance_path, acceptance)
    trust = refresh_handoff_trust(
        root,
        handoff,
        plan_digest=trust.server_plan_digest,
        queue_digest=trust.queue_status_sha256,
    )

    with pytest.raises(PoolIntakeError, match="direct training record differs"):
        _intake_v02_policy_pool(
            handoff,
            trusted_experiment_root=root,
            trust_anchor=trust,
            _acceptance_replayer=_fixture_replayer,
        )


def test_identity_replayer_cannot_bypass_final_queue_mutation(tmp_path: Path) -> None:
    root, handoff, _trust = exact90_handoff(tmp_path)
    queue_path = root / "training_private" / "server_runs" / "queue_status.json"
    queue = _load(queue_path)
    recovered = next(
        job_id
        for job_id, state in queue["jobs"].items()
        if state.get("terminal_record_state") == "recovered"
    )
    succeeded = next(
        job_id
        for job_id, state in queue["jobs"].items()
        if state.get("terminal_record_state") == "succeeded"
    )
    queue["jobs"][recovered]["terminal_record_state"] = "succeeded"
    queue["jobs"][succeeded]["terminal_record_state"] = "recovered"
    write_json(queue_path, queue)
    trust = _rebind_queue_sha(root, handoff)

    with pytest.raises(PoolIntakeError, match="direct acceptance cell differs"):
        _intake_v02_policy_pool(
            handoff,
            trusted_experiment_root=root,
            trust_anchor=trust,
            _acceptance_replayer=_fixture_replayer,
        )


def test_persisted_intake_requires_explicit_authority_and_rechecks_bundle_bytes(
    tmp_path: Path,
) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    record = _intake_fixture(root, handoff, trust)
    artifact = tmp_path / "intake_record.json"
    artifact.write_bytes(canonical_json_bytes(record.to_dict()) + b"\n")
    loaded = _load_verified_v02_intake(
        artifact,
        expected_artifact_sha256=sha256_file(artifact),
        trusted_experiment_root=root,
        trust_anchor=trust,
    )
    assert loaded.intake_record_digest == record.intake_record_digest

    # A structurally valid fixture record is not the reviewed production authority.
    with pytest.raises(PoolIntakeError, match="frozen production trust anchor"):
        assert_frozen_v02_intake_authority(record)

    first = next(iter(record.cells.values()))
    (Path(first.bundle_path) / "actor.npz").write_bytes(b"tampered after intake")
    with pytest.raises(PoolIntakeError, match="payload .* differs"):
        _load_verified_v02_intake(
            artifact,
            expected_artifact_sha256=sha256_file(artifact),
            trusted_experiment_root=root,
            trust_anchor=trust,
        )


def test_verified_intake_loader_rejects_noncanonical_json_even_with_matching_sha(
    tmp_path: Path,
) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    record = _intake_fixture(root, handoff, trust)
    artifact = tmp_path / "intake_record.json"
    artifact.write_text(json.dumps(record.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PoolIntakeError, match="not canonical JSON"):
        _load_verified_v02_intake(
            artifact,
            expected_artifact_sha256=sha256_file(artifact),
            trusted_experiment_root=root,
            trust_anchor=trust,
        )


def test_referenced_bundle_payload_tamper_fails_even_when_overlay_is_unchanged(
    tmp_path: Path,
) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    acceptance = _load(handoff / "policy_pool_acceptance.json")
    first = acceptance["cells"][sorted(acceptance["cells"])[0]]
    (Path(first["bundle_path"]) / "actor.npz").write_bytes(b"tampered")
    with pytest.raises(PoolIntakeError, match="payload .* differs"):
        _intake_fixture(root, handoff, trust)


def test_bundle_path_escape_fails_after_all_overlay_digests_are_recomputed(
    tmp_path: Path,
) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    acceptance_path = handoff / "policy_pool_acceptance.json"
    acceptance = _load(acceptance_path)
    job_id = sorted(acceptance["cells"])[0]
    external = tmp_path / "outside" / "bundle"
    external.mkdir(parents=True)
    original = Path(acceptance["cells"][job_id]["bundle_path"])
    for child in original.iterdir():
        if child.is_file():
            (external / child.name).write_bytes(child.read_bytes())
    acceptance["cells"][job_id]["bundle_path"] = str(external.resolve())
    acceptance = self_digest(
        {key: value for key, value in acceptance.items() if key != "report_digest"},
        "report_digest",
    )
    write_json(acceptance_path, acceptance)
    trust = refresh_handoff_trust(
        root,
        handoff,
        plan_digest=trust.server_plan_digest,
        queue_digest=trust.queue_status_sha256,
    )
    with pytest.raises(PoolIntakeError, match="escapes the trusted experiment root"):
        _intake_fixture(root, handoff, trust)


def test_promotion_lineage_event_tamper_is_detected(tmp_path: Path) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    promotion = _load(handoff / "compiled_parity_promotions.json")
    entry = promotion["entries"][sorted(promotion["entries"])[0]]
    events = Path(entry["events_path"])
    events.write_text(events.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(PoolIntakeError, match="event bytes changed"):
        _intake_fixture(root, handoff, trust)


def test_unknown_overlay_field_fails_closed_even_with_new_file_and_self_digests(
    tmp_path: Path,
) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)
    acceptance_path = handoff / "policy_pool_acceptance.json"
    acceptance = _load(acceptance_path)
    acceptance["training_summary"] = {"return": 1.0}
    acceptance = self_digest(
        {key: value for key, value in acceptance.items() if key != "report_digest"},
        "report_digest",
    )
    write_json(acceptance_path, acceptance)
    trust = refresh_handoff_trust(
        root,
        handoff,
        plan_digest=trust.server_plan_digest,
        queue_digest=trust.queue_status_sha256,
    )
    with pytest.raises(PoolIntakeError, match="unknown=.*training_summary"):
        _intake_fixture(root, handoff, trust)


def test_frozen_replay_mismatch_blocks_pool_ready(tmp_path: Path) -> None:
    root, handoff, trust = exact90_handoff(tmp_path)

    def mismatched(_root: Path, handoff_path: Path, _promotions):
        value = _load(handoff_path / "policy_pool_acceptance.json")
        value["accepted_at"] = "2026-08-26T00:00:01+00:00"
        value = self_digest(
            {key: item for key, item in value.items() if key != "report_digest"},
            "report_digest",
        )
        value["job_count"] = 89
        value = self_digest(
            {key: item for key, item in value.items() if key != "report_digest"},
            "report_digest",
        )
        return value

    with pytest.raises(PoolIntakeError, match="differs from a fresh frozen-validator replay"):
        _intake_v02_policy_pool(
            handoff,
            trusted_experiment_root=root,
            trust_anchor=trust,
            _acceptance_replayer=mismatched,
        )
