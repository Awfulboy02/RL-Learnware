from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.v02.training import (
    AdmittedTrainingRecord,
    PolicyTrainingAttestation,
    PolicyTrainingJob,
)


def _job() -> PolicyTrainingJob:
    return PolicyTrainingJob(
        job_id="recovered-candidate",
        config_digest="0" * 64,
        execution_purpose="v02_freeze_ready",
        source_anchor_id="a" * 64,
        environment_instance_digest="b" * 64,
        anchor_manifest_digest="c" * 64,
        algorithm="fpo",
        trainer_config={"num_timesteps": 300},
        seed=0,
        environment_steps=300,
        checkpoint_rule="fixed_ladder",
        trainer_commit="d" * 40,
        dependency_digest="e" * 64,
        runtime_digest="f" * 64,
        training_protocol_id="1" * 64,
    )


def _recovered(job: PolicyTrainingJob) -> PolicyTrainingAttestation:
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
        environment_steps=100,
        checkpoint_rule=job.checkpoint_rule,
        checkpoint_digests={"outer_000001": "4" * 64},
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
        status="recovered",
        failure_reason="training_step contains non-finite values",
        bundle_path="/private/recovered-bundle",
        planned_outer_iterations=3,
        completed_outer_iterations=2,
        promoted_outer_iteration=1,
        planned_environment_steps=300,
        completed_environment_steps=200,
        promoted_environment_steps=100,
        failure_type="NumericalIntegrityError",
        failure_trace_digest="a" * 64,
    )


def test_recovered_attestation_is_admissible_but_retains_actual_budget() -> None:
    job = _job()
    attestation = _recovered(job)
    admitted = AdmittedTrainingRecord(job, attestation)
    assert admitted.attestation.status == "recovered"
    assert admitted.attestation.environment_steps == 100
    assert admitted.attestation.planned_environment_steps == job.environment_steps


def test_recovered_attestation_rejects_non_numerical_or_inconsistent_fallback() -> None:
    attestation = _recovered(_job())
    with pytest.raises(ValueError, match="only NumericalIntegrityError"):
        replace(attestation, failure_type="ContractError")
    with pytest.raises(ValueError, match="promoted bundle budget"):
        replace(attestation, environment_steps=200)
    with pytest.raises(ValueError, match="geometry"):
        replace(attestation, completed_environment_steps=201)
