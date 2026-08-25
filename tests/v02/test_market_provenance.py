from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from policy_learnware_v0.v02.competence import (
    SourceEpisodeRow,
    admit_formal_championization,
)
from policy_learnware_v0.v02.config import V02ExperimentConfig
from policy_learnware_v0.v02.market import build_policy_market
from policy_learnware_v0.v02.schemas import ExecutionABIRecord
from policy_learnware_v0.v02.training import (
    AdmittedTrainingRecord,
    PolicyTrainingAttestation,
    PolicyTrainingJob,
    admitted_training_records_digest,
)
from tests.v02.test_config import _formal_payload


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record(anchor: str, seed: int, config: V02ExperimentConfig) -> AdmittedTrainingRecord:
    candidate = f"candidate-{anchor}-{seed}"
    job = PolicyTrainingJob(
        job_id=candidate,
        config_digest=config.config_digest,
        execution_purpose=config.stage,
        source_anchor_id=anchor,
        environment_instance_digest=_d(f"{anchor}:environment"),
        anchor_manifest_digest=_d(f"{anchor}:manifest"),
        algorithm=config.primary_algorithm.lower(),
        trainer_config={"reviewed_config": _d("trainer-config")},
        seed=seed,
        environment_steps=config.training_steps,
        checkpoint_rule=config.checkpoint_rule,
        trainer_commit="d" * 40,
        dependency_digest="e" * 64,
        runtime_digest="f" * 64,
        training_protocol_id="1" * 64,
    )
    attestation = PolicyTrainingAttestation(
        job_id=job.job_id,
        job_digest=job.digest,
        attempt_id=f"attempt-{seed}",
        attempt_number=1,
        source_anchor_id=anchor,
        anchor_manifest_digest=job.anchor_manifest_digest,
        declared_environment_instance_digest=job.environment_instance_digest,
        actual_train_environment_instance_digest=job.environment_instance_digest,
        actual_eval_environment_instance_digest=job.environment_instance_digest,
        operator_digest=_d(f"{anchor}:operator"),
        model_diff_digest=_d(f"{anchor}:model-diff"),
        algorithm=job.algorithm,
        seed=seed,
        environment_steps=job.environment_steps,
        checkpoint_rule=job.checkpoint_rule,
        checkpoint_digests={"final": _d(f"{candidate}:checkpoint")},
        bundle_digest=_d(f"{candidate}:bundle"),
        bundle_manifest_digest=_d(f"{candidate}:bundle-manifest"),
        golden_parity_digest=_d(f"{candidate}:golden"),
        compiled_parity_digest=_d(f"{candidate}:compiled"),
        finiteness_audit_digest=_d(f"{candidate}:finite"),
        all_arrays_finite=True,
        golden_parity_passed=True,
        compiled_parity_passed=True,
        trainer_commit=job.trainer_commit,
        dependency_digest=job.dependency_digest,
        runtime_digest=job.runtime_digest,
        hardware_digest=_d("hardware"),
        started_at="2026-08-24T00:00:00Z",
        finished_at="2026-08-24T00:01:00Z",
        elapsed_seconds=60.0,
        status="succeeded",
        bundle_path=f"/private/{candidate}.zip",
        server_plan_binding_digest=_d(f"{candidate}:server-plan-binding"),
        server_training_plan_digest=_d(f"{candidate}:server-training-plan"),
        server_job_digest=_d(f"{candidate}:server-job"),
        server_attempt_digest=_d(f"{candidate}:server-attempt"),
        server_run_manifest_digest=_d(f"{candidate}:server-run-manifest"),
        server_training_record_digest=_d(f"{candidate}:server-training-record"),
    )
    return AdmittedTrainingRecord(job, attestation)


def _formal_case():
    config = V02ExperimentConfig.from_dict(_formal_payload())
    admitted = {
        record.job.job_id: record
        for anchor in config.source_anchor_ids
        for seed in config.training_seeds
        for record in (_record(anchor, seed, config),)
    }
    selection: list[SourceEpisodeRow] = []
    for candidate, record in admitted.items():
        score = 0.9 if record.job.seed == config.training_seeds[0] else 0.8
        for reset_seed in (101, 102):
            selection.append(
                SourceEpisodeRow(
                    source_anchor_id=record.job.source_anchor_id,
                    candidate_id=candidate,
                    bundle_digest=record.attestation.bundle_digest,
                    block="source_selection",
                    reset_seed=reset_seed,
                    normalized_return=score,
                )
            )
    winner_by_anchor = {
        anchor: next(
            candidate
            for candidate, record in admitted.items()
            if record.job.source_anchor_id == anchor
            and record.job.seed == config.training_seeds[0]
        )
        for anchor in config.source_anchor_ids
    }
    attestation = [
        SourceEpisodeRow(
            source_anchor_id=anchor,
            candidate_id=candidate,
            bundle_digest=admitted[candidate].attestation.bundle_digest,
            block="source_attestation",
            reset_seed=reset_seed,
            normalized_return=0.8,
        )
        for anchor, candidate in winner_by_anchor.items()
        for reset_seed in (201, 202, 203)
    ]
    return config, admitted, selection, attestation


def _runtime() -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest="2" * 64,
        action_tensor_abi_digest="3" * 64,
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v02",
        state_abi_id="stateless",
    )


def test_formal_championization_binds_config_grid_episodes_floors_and_bundles() -> None:
    config, admitted, selection, attestation = _formal_case()
    result = admit_formal_championization(
        config,
        admitted,
        selection,
        attestation,
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id=_d("return-contract"),
    )
    binding = result.formal_admission
    assert binding is not None
    assert set(binding.expected_anchor_ids) == set(config.source_anchor_ids)
    assert set(binding.expected_candidate_ids) == set(admitted)
    assert binding.admitted_records_digest == admitted_training_records_digest(admitted)
    assert binding.selection_episodes_per_candidate == 2
    assert binding.attestation_episodes_per_champion == 3
    assert set(result.attested_bundle_digests) == set(config.source_anchor_ids)
    assert all(
        record.competence_floor
        == config.competence_floor[config.source_anchor_to_task[anchor]]
        for anchor, record in result.competence_records.items()
    )

    market = build_policy_market(
        admitted,
        result,
        {candidate: _runtime() for candidate in result.selected_by_anchor.values()},
        expected_anchor_ids=config.source_anchor_ids,
        market_alias_nonce="a" * 64,
        tie_break_nonce="b" * 64,
    )
    assert set(market.anchor_to_opaque_id) == set(config.source_anchor_ids)


@pytest.mark.parametrize("poison", ["unadmitted_candidate", "selection_bundle", "episode_count"])
def test_formal_championization_rejects_candidate_bundle_and_count_poison(poison: str) -> None:
    config, admitted, selection, attestation = _formal_case()
    if poison == "unadmitted_candidate":
        selection[0] = replace(selection[0], candidate_id="not-an-admitted-job")
        match = "unadmitted candidate"
    elif poison == "selection_bundle":
        selection[0] = replace(selection[0], bundle_digest=_d("poison-bundle"))
        match = "bundle digest differs"
    else:
        selection.pop()
        match = "episode count"
    with pytest.raises(ValueError, match=match):
        admit_formal_championization(
            config,
            admitted,
            selection,
            attestation,
            mean_tolerance=0.0,
            lcb_z=None,
            return_contract_id=_d("return-contract"),
        )


@pytest.mark.parametrize(
    ("poison", "match"),
    [
        ("config", "another config"),
        ("purpose", "non-formal execution purpose"),
        ("server_provenance", "lacks raw server provenance"),
    ],
)
def test_formal_championization_rejects_nonformal_training_provenance(
    poison: str, match: str
) -> None:
    config, admitted, selection, attestation = _formal_case()
    candidate = next(iter(admitted))
    original = admitted[candidate]
    if poison == "server_provenance":
        poisoned_attestation = replace(
            original.attestation,
            server_plan_binding_digest=None,
            server_training_plan_digest=None,
            server_job_digest=None,
            server_attempt_digest=None,
            server_run_manifest_digest=None,
            server_training_record_digest=None,
        )
        poisoned = AdmittedTrainingRecord(original.job, poisoned_attestation)
    else:
        poisoned_job = replace(
            original.job,
            **(
                {"config_digest": _d("another-config")}
                if poison == "config"
                else {"execution_purpose": "audit_smoke"}
            ),
        )
        poisoned_attestation = replace(
            original.attestation,
            job_digest=poisoned_job.digest,
        )
        poisoned = AdmittedTrainingRecord(poisoned_job, poisoned_attestation)
    admitted = dict(admitted)
    admitted[candidate] = poisoned

    with pytest.raises(ValueError, match=match):
        admit_formal_championization(
            config,
            admitted,
            selection,
            attestation,
            mean_tolerance=0.0,
            lcb_z=None,
            return_contract_id=_d("return-contract"),
        )


def test_market_rejects_post_championization_training_and_anchor_poison() -> None:
    config, admitted, selection, attestation = _formal_case()
    result = admit_formal_championization(
        config,
        admitted,
        selection,
        attestation,
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id=_d("return-contract"),
    )
    abis = {candidate: _runtime() for candidate in result.selected_by_anchor.values()}
    first_candidate = next(iter(admitted))
    poisoned = dict(admitted)
    original = poisoned[first_candidate]
    poisoned[first_candidate] = AdmittedTrainingRecord(
        original.job,
        replace(original.attestation, bundle_digest=_d("post-selection-swap")),
    )
    with pytest.raises(ValueError, match="differ from formal championization"):
        build_policy_market(
            poisoned,
            result,
            abis,
            expected_anchor_ids=config.source_anchor_ids,
            market_alias_nonce="a" * 64,
            tie_break_nonce="b" * 64,
        )
    with pytest.raises(ValueError, match="caller anchor IDs differ"):
        build_policy_market(
            admitted,
            result,
            abis,
            expected_anchor_ids=config.source_anchor_ids[1:],
            market_alias_nonce="a" * 64,
            tie_break_nonce="b" * 64,
        )


def test_attestation_bundle_poison_is_rejected_before_market_publication() -> None:
    config, admitted, selection, attestation = _formal_case()
    attestation[0] = replace(attestation[0], bundle_digest=_d("attestation-poison"))
    with pytest.raises(ValueError, match="bundle digest differs"):
        admit_formal_championization(
            config,
            admitted,
            selection,
            attestation,
            mean_tolerance=0.0,
            lcb_z=None,
            return_contract_id=_d("return-contract"),
        )
