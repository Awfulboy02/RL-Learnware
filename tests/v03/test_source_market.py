from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v02.schemas import ExecutionABIRecord
from policy_learnware_v0.v03.pool_intake import _intake_v02_policy_pool
from policy_learnware_v0.v03.source_evaluator import (
    SourceEpisodeAttempt,
    SourceEvaluationAttemptRecord,
)
from policy_learnware_v0.v03.source_market import (
    EvaluatorSourceReceipt,
    PUBLIC_ENTRY_ALLOWLIST,
    PUBLIC_MANIFEST_ALLOWLIST,
    SourceEvaluationProtocol,
    SourceChampionizationRecord,
    SourceEvaluationWorkUnit,
    RawSourceEpisodeShard,
    ProvisionalSourceSelection,
    SourceAttestationPlan,
    SourceMarketError,
    V03SourcePolicyMarket,
    build_source_evaluation_work_unit,
    build_source_policy_market,
    championize_source_pool,
    finalize_source_championization,
    freeze_source_attestation_plan,
    provisionally_select_source_pool,
    receipt_from_source_episode_shard,
)

sys.path.insert(0, str(Path(__file__).parent))
from p5_asset_fixtures import (  # noqa: E402
    digest,
    exact90_handoff,
    fixture_source_anchor,
    self_digest,
)


def _protocol(intake) -> SourceEvaluationProtocol:
    anchors = tuple(intake.candidates_by_anchor)
    return SourceEvaluationProtocol(
        intake_record_digest=intake.intake_record_digest,
        evaluator_implementation_digest=digest("source-evaluator-implementation"),
        return_contract_digest=digest("dmc-normalized-return-contract"),
        selection_seed_namespace_digest=digest("selection-seed-namespace"),
        attestation_seed_namespace_digest=digest("attestation-seed-namespace"),
        selection_reset_seeds=(101, 102),
        attestation_reset_seeds=(201, 202, 203),
        selection_episodes_per_candidate=2,
        attestation_episodes_per_champion=3,
        source_environment_digests={
            fixture_source_anchor(index).anchor_id:
            fixture_source_anchor(index).environment_instance_digest
            for index in range(30)
        },
        competence_floors={
            anchor: (0.95 if index == 0 else 0.5)
            for index, anchor in enumerate(anchors)
        },
        mean_tolerance=0.01,
        lcb_z=1.645,
    )


def _receipt(
    intake,
    protocol: SourceEvaluationProtocol,
    candidate: str,
    *,
    block: str,
    returns: tuple[float, ...],
    work_unit: SourceEvaluationWorkUnit | None = None,
) -> EvaluatorSourceReceipt:
    cell = intake.cells[candidate]
    work_unit_digest = (
        work_unit.work_unit_digest
        if work_unit is not None
        else digest(f"work-unit:{block}:{candidate}")
    )
    attempt_record_digest = digest(f"attempt-record:{block}:{candidate}")
    validated_binding_digest = digest(f"validated-binding:{block}:{candidate}")
    evaluation_attempt_number = 1
    runtime_digest = (
        work_unit.anchor_runtime_digest
        if work_unit is not None
        else digest(f"runtime:{block}:{candidate}")
    )
    raw_episode_shard_digest = digest(f"raw-shard:{block}:{candidate}")
    return EvaluatorSourceReceipt(
        source_evaluation_protocol_digest=protocol.source_evaluation_protocol_digest,
        intake_record_digest=intake.intake_record_digest,
        intake_cell_digest=cell.intake_cell_digest,
        block=block,
        seed_namespace_digest=(
            protocol.selection_seed_namespace_digest
            if block == "source_selection"
            else protocol.attestation_seed_namespace_digest
        ),
        candidate_id=candidate,
        source_anchor_id=cell.source_anchor_id,
        bundle_digest=cell.bundle_digest,
        source_environment_digest=protocol.source_environment_digests[cell.source_anchor_id],
        evaluator_implementation_digest=protocol.evaluator_implementation_digest,
        return_contract_digest=protocol.return_contract_digest,
        work_unit_digest=work_unit_digest,
        attempt_record_digest=attempt_record_digest,
        validated_binding_digest=validated_binding_digest,
        evaluation_attempt_number=evaluation_attempt_number,
        runtime_digest=runtime_digest,
        raw_episode_shard_digest=raw_episode_shard_digest,
        dataset_digest=sha256_json(
            {
                "schema": "policy-learnware.v03-source-rollout-dataset.v0",
                "work_unit_digest": work_unit_digest,
                "attempt_record_digest": attempt_record_digest,
                "validated_binding_digest": validated_binding_digest,
                "evaluation_attempt_number": evaluation_attempt_number,
                "raw_episode_shard_digest": raw_episode_shard_digest,
            }
        ),
        reset_seeds=(101, 102) if block == "source_selection" else (201, 202, 203),
        normalized_returns=returns,
    )


def _synthetic_work_unit(
    intake,
    protocol: SourceEvaluationProtocol,
    candidate: str,
    block: str,
) -> SourceEvaluationWorkUnit:
    cell = intake.cells[candidate]
    experiment_root = Path(cell.bundle_path).parents[6]
    anchor_manifest_path = (
        experiment_root / "source_anchor_manifests" / f"{cell.source_anchor_id}.json"
    )
    anchor_manifest = json.loads(anchor_manifest_path.read_text(encoding="utf-8"))
    return SourceEvaluationWorkUnit(
        source_evaluation_protocol_digest=protocol.source_evaluation_protocol_digest,
        intake_record_digest=intake.intake_record_digest,
        intake_cell_digest=cell.intake_cell_digest,
        block=block,
        seed_namespace_digest=(
            protocol.selection_seed_namespace_digest
            if block == "source_selection"
            else protocol.attestation_seed_namespace_digest
        ),
        candidate_id=candidate,
        source_anchor_id=cell.source_anchor_id,
        attempt_number=cell.attempt_number,
        attempt_digest=cell.attempt_digest,
        bundle_digest=cell.bundle_digest,
        bundle_path=cell.bundle_path,
        outer_iteration=cell.outer_iteration,
        environment_steps=cell.environment_steps,
        anchor_manifest_path=str(anchor_manifest_path.resolve()),
        anchor_manifest_digest=anchor_manifest["manifest_digest"],
        anchor_runtime_digest=anchor_manifest["runtime_digest"],
        source_environment_digest=protocol.source_environment_digests[cell.source_anchor_id],
        evaluator_implementation_digest=protocol.evaluator_implementation_digest,
        return_contract_digest=protocol.return_contract_digest,
        execution_abi=_abi(),
        reset_seeds=(
            protocol.selection_reset_seeds
            if block == "source_selection"
            else protocol.attestation_reset_seeds
        ),
    )


def _case(tmp_path: Path):
    root, handoff, trust = exact90_handoff(tmp_path)
    intake = _intake_v02_policy_pool(
        handoff,
        trusted_experiment_root=root,
        trust_anchor=trust,
        _acceptance_replayer=lambda _root, handoff_path, _promotions: json.loads(
            (handoff_path / "policy_pool_acceptance.json").read_text(encoding="utf-8")
        ),
    )
    protocol = _protocol(intake)
    first_anchor = next(iter(intake.candidates_by_anchor))
    selections = []
    expected = {}
    for anchor, cells in intake.candidates_by_anchor.items():
        for cell in cells:
            score = 0.8 if anchor == first_anchor else {0: 0.9, 1: 0.8, 2: 0.7}[cell.seed]
            selections.append(
                _receipt(
                    intake,
                    protocol,
                    cell.job_id,
                    block="source_selection",
                    returns=(score, score),
                )
            )
        expected[anchor] = (
            min(cells, key=lambda cell: (cell.bundle_digest, cell.job_id))
            if anchor == first_anchor
            else next(cell for cell in cells if cell.seed == 0)
        )
    attestations = [
        _receipt(
            intake,
            protocol,
            cell.job_id,
            block="source_attestation",
            returns=((0.1, 0.1, 0.1) if anchor == first_anchor else (0.8, 0.8, 0.8)),
        )
        for anchor, cell in expected.items()
    ]
    return intake, protocol, selections, attestations, expected, first_anchor


def _abi() -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=digest("observation-abi"),
        action_tensor_abi_digest=digest("action-abi"),
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v02",
        state_abi_id="stateless",
    )


def _formal_case(tmp_path: Path):
    intake, protocol, old_selections, old_attestations, expected, first_anchor = _case(
        tmp_path
    )
    selection_units = {
        candidate: _synthetic_work_unit(intake, protocol, candidate, "source_selection")
        for candidate in intake.cells
    }
    selections = [
        _receipt(
            intake,
            protocol,
            receipt.candidate_id,
            block="source_selection",
            returns=receipt.normalized_returns,
            work_unit=selection_units[receipt.candidate_id],
        )
        for receipt in old_selections
    ]
    provisional = provisionally_select_source_pool(
        intake, protocol, selection_units, selections
    )
    attestation_units = {
        candidate: _synthetic_work_unit(intake, protocol, candidate, "source_attestation")
        for candidate in provisional.selected_candidate_ids.values()
    }
    plan = freeze_source_attestation_plan(
        intake, protocol, provisional, attestation_units
    )
    old_by_candidate = {receipt.candidate_id: receipt for receipt in old_attestations}
    attestations = [
        _receipt(
            intake,
            protocol,
            candidate,
            block="source_attestation",
            returns=old_by_candidate[candidate].normalized_returns,
            work_unit=unit,
        )
        for candidate, unit in attestation_units.items()
    ]
    result = finalize_source_championization(
        intake, protocol, provisional, plan, attestations
    )
    return (
        intake,
        protocol,
        provisional,
        plan,
        attestations,
        result,
        expected,
        first_anchor,
    )


def test_source_only_three_candidate_championization_observes_but_does_not_block(
    tmp_path: Path,
) -> None:
    intake, protocol, selections, attestations, expected, first_anchor = _case(tmp_path)
    result = championize_source_pool(intake, protocol, selections, attestations)
    assert len(result.champions) == 30
    assert result.competence_mode == "OBSERVE"
    assert {
        anchor: champion.candidate_id for anchor, champion in result.champions.items()
    } == {anchor: cell.job_id for anchor, cell in expected.items()}
    assert result.champions[first_anchor].competence.passed is False
    assert first_anchor in result.champions

    assert (
        SourceChampionizationRecord.from_dict(result.to_dict()).to_dict()
        == result.to_dict()
    )
    with pytest.raises(SourceMarketError, match="formal provisional-selection"):
        build_source_policy_market(
            result,
            {champion.candidate_id: _abi() for champion in result.champions.values()},
            market_alias_nonce="a" * 64,
            tie_break_nonce="b" * 64,
        )


def test_formal_championization_freezes_winners_before_attestation(
    tmp_path: Path,
) -> None:
    intake, protocol, old_selections, old_attestations, expected, first_anchor = _case(
        tmp_path
    )
    selection_units = {
        candidate: _synthetic_work_unit(
            intake, protocol, candidate, "source_selection"
        )
        for candidate in intake.cells
    }
    selections = [
        _receipt(
            intake,
            protocol,
            receipt.candidate_id,
            block="source_selection",
            returns=receipt.normalized_returns,
            work_unit=selection_units[receipt.candidate_id],
        )
        for receipt in old_selections
    ]
    provisional = provisionally_select_source_pool(
        intake, protocol, selection_units, selections
    )
    assert ProvisionalSourceSelection.from_dict(provisional.to_dict()).to_dict() == provisional.to_dict()
    assert provisional.selected_candidate_ids == {
        anchor: cell.job_id for anchor, cell in expected.items()
    }

    attestation_units = {
        candidate: _synthetic_work_unit(
            intake, protocol, candidate, "source_attestation"
        )
        for candidate in provisional.selected_candidate_ids.values()
    }
    plan = freeze_source_attestation_plan(
        intake, protocol, provisional, attestation_units
    )
    assert SourceAttestationPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
    old_by_candidate = {receipt.candidate_id: receipt for receipt in old_attestations}
    attestations = [
        _receipt(
            intake,
            protocol,
            candidate,
            block="source_attestation",
            returns=old_by_candidate[candidate].normalized_returns,
            work_unit=unit,
        )
        for candidate, unit in attestation_units.items()
    ]
    result = finalize_source_championization(
        intake, protocol, provisional, plan, attestations
    )
    assert len(result.champions) == 30
    assert result.champions[first_anchor].competence.passed is False
    assert result.provisional_selection_digest == provisional.provisional_selection_digest
    assert result.attestation_plan_digest == plan.attestation_plan_digest

    abis = {champion.candidate_id: _abi() for champion in result.champions.values()}
    first = build_source_policy_market(
        result,
        abis,
        market_alias_nonce="a" * 64,
        tie_break_nonce="b" * 64,
    )
    second = build_source_policy_market(
        result,
        abis,
        market_alias_nonce="a" * 64,
        tie_break_nonce="b" * 64,
    )
    assert first.asset_state == "ENGINEERING_CONTRACT_ONLY"
    assert len(first.entries) == len(first.deployment_private) == 30
    assert first.policy_market_id == second.policy_market_id
    assert first.public_manifest() == second.public_manifest()
    restored = V03SourcePolicyMarket.from_manifests(
        first.public_manifest(), first.deployment_manifest()
    )
    assert restored.public_manifest() == first.public_manifest()
    assert restored.deployment_manifest() == first.deployment_manifest()
    tampered_private = json.loads(json.dumps(first.deployment_manifest()))
    tampered_id = next(iter(tampered_private["entries"]))
    tampered_private["entries"][tampered_id]["bundle_path"] += ".tampered"
    with pytest.raises(SourceMarketError, match="policy_market_id"):
        V03SourcePolicyMarket.from_manifests(
            first.public_manifest(), tampered_private
        )
    assert set(first.public_manifest()) == PUBLIC_MANIFEST_ALLOWLIST
    assert all(
        set(entry) == PUBLIC_ENTRY_ALLOWLIST
        for entry in first.public_manifest()["entries"].values()
    )
    public_text = str(first.public_manifest())
    for forbidden in (
        "source_anchor_id",
        "candidate_id",
        "bundle_path",
        "task",
        "target",
        "oracle",
    ):
        assert forbidden not in public_text

    drifted = replace(
        attestations[0],
        work_unit_digest=digest("post-outcome-work-unit-drift"),
        dataset_digest=sha256_json(
            {
                "schema": "policy-learnware.v03-source-rollout-dataset.v0",
                "work_unit_digest": digest("post-outcome-work-unit-drift"),
                "attempt_record_digest": attestations[0].attempt_record_digest,
                "validated_binding_digest": attestations[0].validated_binding_digest,
                "evaluation_attempt_number": attestations[0].evaluation_attempt_number,
                "raw_episode_shard_digest": attestations[0].raw_episode_shard_digest,
            }
        ),
        receipt_digest=None,
    )
    with pytest.raises(SourceMarketError, match="another frozen work unit"):
        finalize_source_championization(
            intake, protocol, provisional, plan, [drifted, *attestations[1:]]
        )


@pytest.mark.parametrize("poison", ["training_summary", "target_evidence", "oracle_return"])
def test_receipt_contract_rejects_training_or_target_evidence(
    tmp_path: Path, poison: str
) -> None:
    intake, protocol, selections, _attestations, _expected, _anchor = _case(tmp_path)
    payload = selections[0].to_dict()
    payload[poison] = {"value": 1.0}
    with pytest.raises(SourceMarketError, match="unknown=.*" + poison):
        EvaluatorSourceReceipt.from_dict(payload)


def test_receipt_coverage_and_disjoint_evaluator_seed_blocks_are_fail_closed(
    tmp_path: Path,
) -> None:
    intake, protocol, selections, attestations, _expected, _anchor = _case(tmp_path)
    with pytest.raises(SourceMarketError, match="cover all exact-90"):
        championize_source_pool(intake, protocol, selections[:-1], attestations)

    overlapped = [
        replace(
            receipt,
            reset_seeds=(101, 102, 103),
            receipt_digest=None,
        )
        for receipt in attestations
    ]
    with pytest.raises(SourceMarketError, match="differs from intake/protocol binding"):
        championize_source_pool(intake, protocol, selections, overlapped)

    with pytest.raises(SourceMarketError, match="reset-seed blocks overlap"):
        replace(
            protocol,
            attestation_reset_seeds=(102, 202, 203),
            source_evaluation_protocol_digest=None,
        )


def test_protocol_literal_seed_vectors_are_digest_bound_and_round_trip(
    tmp_path: Path,
) -> None:
    intake, protocol, selections, attestations, _expected, _anchor = _case(tmp_path)
    assert SourceEvaluationProtocol.from_dict(protocol.to_dict()).to_dict() == protocol.to_dict()
    drifted = replace(
        selections[0],
        reset_seeds=(103, 104),
        receipt_digest=None,
    )
    with pytest.raises(SourceMarketError, match="differs from intake/protocol binding"):
        championize_source_pool(
            intake,
            protocol,
            [drifted, *selections[1:]],
            attestations,
        )


def test_evaluator_work_unit_and_raw_return_receipt_are_strictly_bound(
    tmp_path: Path,
) -> None:
    intake, protocol, _selections, _attestations, _expected, _anchor = _case(tmp_path)
    cell = next(iter(intake.cells.values()))
    anchor_index = next(
        index
        for index in range(30)
        if fixture_source_anchor(index).anchor_id == cell.source_anchor_id
    )
    anchor = fixture_source_anchor(anchor_index)
    manifest = self_digest(
        {
            "schema": "policy-learnware.v02-anchor-manifest.v0",
            "anchor_id": anchor.anchor_id,
            "environment_instance_digest": anchor.environment_instance_digest,
            "axis_binding_digest": anchor.axis_binding_digest,
            "runtime": {"fixture": "v0.2"},
            "runtime_digest": sha256_json({"fixture": "v0.2"}),
        },
        "manifest_digest",
    )
    manifest_path = tmp_path / "anchor_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    unit = build_source_evaluation_work_unit(
        intake,
        protocol,
        cell.job_id,
        block="source_selection",
        anchor_manifest_path=manifest_path,
        execution_abi=_abi(),
    )
    assert unit.reset_seeds == protocol.selection_reset_seeds
    assert SourceEvaluationWorkUnit.from_dict(unit.to_dict()).to_dict() == unit.to_dict()
    episodes = (
        SourceEpisodeAttempt(
            reset_seed=unit.reset_seeds[0],
            runtime_digest=unit.anchor_runtime_digest,
            state="SUCCEEDED",
            raw_return=250.0,
            normalized_return=0.25,
            steps=1000,
            terminated=False,
            truncated=True,
        ),
        SourceEpisodeAttempt(
            reset_seed=unit.reset_seeds[1],
            runtime_digest=unit.anchor_runtime_digest,
            state="SUCCEEDED",
            raw_return=750.0,
            normalized_return=0.75,
            steps=1000,
            terminated=False,
            truncated=True,
        ),
    )
    attempt = SourceEvaluationAttemptRecord(
        work_unit_digest=unit.work_unit_digest,
        candidate_id=unit.candidate_id,
        block=unit.block,
        evaluator_implementation_digest=unit.evaluator_implementation_digest,
        validated_binding_digest=digest("validated-binding"),
        runtime_digest=unit.anchor_runtime_digest,
        return_contract_digest=unit.return_contract_digest,
        evaluation_attempt_number=1,
        expected_reset_seeds=unit.reset_seeds,
        episodes=episodes,
        state="SUCCEEDED",
    )
    shard = RawSourceEpisodeShard(
        work_unit_digest=unit.work_unit_digest,
        attempt_record_digest=attempt.attempt_record_digest,
        validated_binding_digest=attempt.validated_binding_digest,
        evaluation_attempt_number=attempt.evaluation_attempt_number,
        block=unit.block,
        candidate_id=unit.candidate_id,
        runtime_digest=unit.anchor_runtime_digest,
        reset_seeds=unit.reset_seeds,
        raw_episode_returns=(250.0, 750.0),
        normalized_returns=(0.25, 0.75),
        return_contract_digest=unit.return_contract_digest,
    )
    assert RawSourceEpisodeShard.from_dict(shard.to_dict()).to_dict() == shard.to_dict()
    receipt = receipt_from_source_episode_shard(unit, shard, attempt)
    assert receipt.reset_seeds == protocol.selection_reset_seeds
    assert receipt.normalized_returns == (0.25, 0.75)
    assert receipt.attempt_record_digest == attempt.attempt_record_digest
    assert receipt.validated_binding_digest == attempt.validated_binding_digest
    assert receipt.evaluation_attempt_number == attempt.evaluation_attempt_number

    tampered_shard = replace(
        shard,
        normalized_returns=(1.0, 1.0),
        episode_shard_digest=None,
    )
    with pytest.raises(SourceMarketError, match="successful attempt/raw episode shard"):
        receipt_from_source_episode_shard(unit, tampered_shard, attempt)

    poisoned = {**manifest, "target_evidence": {"return": 1.0}}
    poisoned = self_digest(
        {key: value for key, value in poisoned.items() if key != "manifest_digest"},
        "manifest_digest",
    )
    poisoned_path = tmp_path / "poisoned_anchor_manifest.json"
    poisoned_path.write_text(json.dumps(poisoned, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SourceMarketError, match="forbidden non-source fields"):
        build_source_evaluation_work_unit(
            intake,
            protocol,
            cell.job_id,
            block="source_selection",
            anchor_manifest_path=poisoned_path,
            execution_abi=_abi(),
        )
    with pytest.raises(SourceMarketError, match="typed raw episode shard"):
        receipt_from_source_episode_shard(
            unit, {"training_summary": 1.0}, attempt
        )


def test_market_nonce_domains_are_separate_and_alias_changes_are_explicit(
    tmp_path: Path,
) -> None:
    _intake, _protocol, _provisional, _plan, _receipts, result, _expected, _anchor = (
        _formal_case(tmp_path)
    )
    abis = {champion.candidate_id: _abi() for champion in result.champions.values()}
    with pytest.raises(SourceMarketError, match="distinct nonces"):
        build_source_policy_market(
            result,
            abis,
            market_alias_nonce="a" * 64,
            tie_break_nonce="a" * 64,
        )
    first = build_source_policy_market(
        result,
        abis,
        market_alias_nonce="a" * 64,
        tie_break_nonce="b" * 64,
    )
    changed = build_source_policy_market(
        result,
        abis,
        market_alias_nonce="c" * 64,
        tie_break_nonce="b" * 64,
    )
    assert set(first.entries) != set(changed.entries)
    assert first.policy_market_id != changed.policy_market_id
