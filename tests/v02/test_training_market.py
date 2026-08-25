from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.v02.competence import SourceEpisodeRow, championize_by_anchor
from policy_learnware_v0.v02.market import build_policy_market
from policy_learnware_v0.v02.schemas import ExecutionABIRecord
from policy_learnware_v0.v02.training import (
    AdmittedTrainingRecord,
    PolicyTrainingAttestation,
    PolicyTrainingJob,
    plan_training_jobs,
)


ANCHOR = "a" * 64
ENVIRONMENT = "b" * 64
ANCHOR_MANIFEST = "c" * 64
TRAINER_COMMIT = "d" * 40


def _job(seed: int = 0, candidate: str = "candidate-a") -> PolicyTrainingJob:
    return PolicyTrainingJob(
        job_id=candidate,
        config_digest="0" * 64,
        execution_purpose="audit_smoke",
        source_anchor_id=ANCHOR,
        environment_instance_digest=ENVIRONMENT,
        anchor_manifest_digest=ANCHOR_MANIFEST,
        algorithm="ppo",
        trainer_config={"learning_rate": 3.0e-4, "batch_size": 1024},
        seed=seed,
        environment_steps=100,
        checkpoint_rule="final-fixed-budget",
        trainer_commit=TRAINER_COMMIT,
        dependency_digest="e" * 64,
        runtime_digest="f" * 64,
        training_protocol_id="1" * 64,
    )


def _attestation(job: PolicyTrainingJob) -> PolicyTrainingAttestation:
    return PolicyTrainingAttestation(
        job_id=job.job_id,
        job_digest=job.digest,
        attempt_id="attempt-01",
        attempt_number=1,
        source_anchor_id=job.source_anchor_id,
        anchor_manifest_digest=job.anchor_manifest_digest,
        declared_environment_instance_digest=job.environment_instance_digest,
        actual_train_environment_instance_digest=job.environment_instance_digest,
        actual_eval_environment_instance_digest=job.environment_instance_digest,
        operator_digest="2" * 64,
        model_diff_digest="3" * 64,
        algorithm=job.algorithm,
        seed=job.seed,
        environment_steps=job.environment_steps,
        checkpoint_rule=job.checkpoint_rule,
        checkpoint_digests={"final": "4" * 64},
        bundle_digest="5" * 64,
        bundle_manifest_digest="6" * 64,
        golden_parity_digest="7" * 64,
        compiled_parity_digest="8" * 64,
        finiteness_audit_digest="9" * 64,
        all_arrays_finite=True,
        golden_parity_passed=True,
        compiled_parity_passed=True,
        trainer_commit=job.trainer_commit,
        dependency_digest=job.dependency_digest,
        runtime_digest=job.runtime_digest,
        hardware_digest="0" * 64,
        started_at="2026-08-24T00:00:00Z",
        finished_at="2026-08-24T00:01:00Z",
        elapsed_seconds=60.0,
        status="succeeded",
        bundle_path="/private/bundle",
    )


def _runtime() -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest="2" * 64,
        action_tensor_abi_digest="3" * 64,
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v02",
        state_abi_id="stateless",
    )


def test_training_plan_is_anchor_seed_matrix_and_digest_stable() -> None:
    jobs = plan_training_jobs(
        {ANCHOR: {"environment_instance_digest": ENVIRONMENT, "anchor_manifest_digest": ANCHOR_MANIFEST}},
        config_digest="0" * 64,
        execution_purpose="audit_smoke",
        algorithm="ppo",
        seeds=(0, 1, 2),
        environment_steps=100,
        checkpoint_rule="final-fixed-budget",
        trainer_config={"learning_rate": 3.0e-4},
        trainer_commit=TRAINER_COMMIT,
        dependency_digest="e" * 64,
        runtime_digest="f" * 64,
        training_protocol_id="1" * 64,
    )
    assert len(jobs) == 3
    assert {job.seed for job in jobs} == {0, 1, 2}
    assert len({job.job_id for job in jobs}) == 3


def test_attestation_rejects_nominal_poison_and_nonfinite_success() -> None:
    job = _job()
    valid = _attestation(job)
    AdmittedTrainingRecord(job, valid)
    with pytest.raises(ValueError, match="environment digest"):
        replace(valid, actual_train_environment_instance_digest="a" * 64)
    with pytest.raises(ValueError, match="non-finite"):
        replace(valid, all_arrays_finite=False)
    with pytest.raises(ValueError, match="frozen job"):
        AdmittedTrainingRecord(job, replace(valid, job_id="different"))


def _rows(candidate: str, block: str, seeds: tuple[int, ...], values: tuple[float, ...]) -> list[SourceEpisodeRow]:
    return [
        SourceEpisodeRow(
            source_anchor_id=ANCHOR,
            candidate_id=candidate,
            bundle_digest=("5" if candidate == "candidate-a" else "6") * 64,
            block=block,  # type: ignore[arg-type]
            reset_seed=seed,
            normalized_return=value,
        )
        for seed, value in zip(seeds, values, strict=True)
    ]


def test_championization_uses_disjoint_attestation_and_never_falls_back() -> None:
    selection = _rows("candidate-a", "source_selection", (1, 2), (0.9, 0.9))
    selection += _rows("candidate-b", "source_selection", (1, 2), (0.8, 0.8))
    attestation = _rows("candidate-a", "source_attestation", (101, 102), (0.2, 0.2))
    result = championize_by_anchor(
        selection,
        attestation,
        competence_floors={ANCHOR: 0.5},
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id="7" * 64,
        competence_mode="ENFORCE",
    )
    assert result.selected_by_anchor[ANCHOR] == "candidate-a"
    assert ANCHOR in result.rejected_anchors
    assert ANCHOR not in result.competence_records
    with pytest.raises(ValueError, match="overlap"):
        championize_by_anchor(
            selection,
            _rows("candidate-a", "source_attestation", (1, 2), (0.9, 0.9)),
            competence_floors={ANCHOR: 0.5},
            mean_tolerance=0.0,
            lcb_z=None,
            return_contract_id="7" * 64,
            competence_mode="ENFORCE",
        )


def test_observe_keeps_low_competence_as_attested_market_metadata() -> None:
    job = _job()
    admitted = AdmittedTrainingRecord(job, _attestation(job))
    selection = _rows("candidate-a", "source_selection", (1, 2), (0.9, 0.9))
    attestation = _rows("candidate-a", "source_attestation", (101, 102), (0.2, 0.2))
    result = championize_by_anchor(
        selection,
        attestation,
        competence_floors={ANCHOR: 0.5},
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id="7" * 64,
        competence_mode="OBSERVE",
    )

    record = result.competence_records[ANCHOR]
    assert result.competence_mode == "OBSERVE"
    assert not result.rejected_anchors
    assert record.normalized_competence == pytest.approx(0.2)
    assert record.passed is False

    enforced = championize_by_anchor(
        selection,
        attestation,
        competence_floors={ANCHOR: 0.5},
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id="7" * 64,
        competence_mode="ENFORCE",
    )
    assert ANCHOR in enforced.rejected_anchors
    assert result.selection_digest != enforced.selection_digest

    market = build_policy_market(
        {"candidate-a": admitted},
        result,
        {"candidate-a": _runtime()},
        expected_anchor_count=1,
        market_alias_nonce="a" * 64,
        tie_break_nonce="b" * 64,
    )
    entry = next(iter(market.entries.values()))
    assert entry.normalized_source_competence == pytest.approx(0.2)


def test_market_has_exactly_one_public_entry_without_private_bundle_metadata() -> None:
    job = _job()
    admitted = AdmittedTrainingRecord(job, _attestation(job))
    selection = _rows("candidate-a", "source_selection", (1, 2), (0.9, 0.9))
    attestation = _rows("candidate-a", "source_attestation", (101, 102), (0.8, 0.8))
    champions = championize_by_anchor(
        selection,
        attestation,
        competence_floors={ANCHOR: 0.5},
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id="7" * 64,
    )
    market = build_policy_market(
        {"candidate-a": admitted},
        champions,
        {"candidate-a": _runtime()},
        expected_anchor_count=1,
        market_alias_nonce="a" * 64,
        tie_break_nonce="b" * 64,
    )
    manifest_text = str(market.public_manifest())
    assert len(market.entries) == 1
    assert "bundle_path" not in manifest_text
    assert "bundle_digest" not in manifest_text
    assert "seed" not in manifest_text
    assert "execution_abi" not in manifest_text
    assert "/private/bundle" in str(market.deployment_manifest())
