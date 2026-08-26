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
    BackendEpisodeResult,
    CanonicalSourceAnchor,
    DmcFixedHorizonReturnContract,
    FrozenPlanJobBinding,
    FrozenServerPlanBinding,
    SourceEvaluationAttemptFailed,
    SourceEvaluatorError,
    ValidatedSourceBinding,
    plan_source_selection_from_server_plan,
    plan_source_selection_work_units,
    rebuild_raw_source_episode_shard,
    run_source_evaluation_work_unit,
    source_work_unit_manifest,
)
from policy_learnware_v0.v03.source_market import SourceEvaluationProtocol

sys.path.insert(0, str(Path(__file__).parent))
from p5_asset_fixtures import digest, exact90_handoff  # noqa: E402


def _abi(label: str = "a") -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=digest(f"observation-abi:{label}"),
        action_tensor_abi_digest=digest(f"action-abi:{label}"),
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v0",
        state_abi_id="stateless-v0",
    )


class FakeSourceBackend:
    def __init__(
        self,
        implementation_digest: str,
        *,
        abi: ExecutionABIRecord | None = None,
    ) -> None:
        self.evaluator_implementation_digest = implementation_digest
        self.abi = abi or _abi()
        self.bundle_drift = False
        self.fail_seed: int | None = None
        self.runtime_drift_seed: int | None = None
        self.raw_returns = {101: 250.0, 102: 750.0}
        self.calls: list[int] = []

    def validate_candidate(self, request):
        return ValidatedSourceBinding(
            request_digest=request.request_digest,
            candidate_id=request.candidate_id,
            evaluator_implementation_digest=request.evaluator_implementation_digest,
            bundle_path=request.bundle_path,
            bundle_digest=(digest("bundle-drift") if self.bundle_drift else request.bundle_digest),
            anchor_manifest_path=request.anchor.manifest_path,
            anchor_manifest_digest=request.anchor.manifest_digest,
            anchor_runtime_digest=request.anchor.runtime_digest,
            source_environment_digest=request.source_environment_digest,
            execution_abi=self.abi,
        )

    def evaluate_episode(self, binding, *, reset_seed: int):
        self.calls.append(reset_seed)
        runtime = (
            digest("runtime-drift")
            if reset_seed == self.runtime_drift_seed
            else binding.anchor_runtime_digest
        )
        if reset_seed == self.fail_seed:
            return BackendEpisodeResult.failed(
                reset_seed=reset_seed,
                runtime_digest=runtime,
                failure_code="FAKE_ROLLOUT_FAILURE",
                failure_message="fixture backend failed",
            )
        return BackendEpisodeResult.succeeded(
            reset_seed=reset_seed,
            runtime_digest=runtime,
            raw_return=self.raw_returns[reset_seed],
            steps=1000,
            terminated=False,
            truncated=True,
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
    plan_path = root / "training_private" / "plans" / "server_training_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    anchors = {
        anchor_id: CanonicalSourceAnchor.from_path(
            root / "source_anchor_manifests" / f"{anchor_id}.json"
        )
        for anchor_id in intake.candidates_by_anchor
    }
    jobs = {
        candidate_id: FrozenPlanJobBinding(
            candidate_id=candidate_id,
            job_digest=cell.job_digest,
            seed=cell.seed,
            training_protocol_digest=digest("frozen-training-protocol"),
            anchor=anchors[cell.source_anchor_id],
        )
        for candidate_id, cell in intake.cells.items()
    }
    plan_payload = {
        "schema": "policy-learnware.fixture-server-plan.v0",
        "jobs": [
            {
                "job_id": job.candidate_id,
                "job_digest": job.job_digest,
                "seed": job.seed,
                "training_protocol_digest": job.training_protocol_digest,
                "anchor_manifest_path": job.anchor.manifest_path,
                "anchor_manifest_digest": job.anchor.manifest_digest,
            }
            for job in jobs.values()
        ],
    }
    plan_digest = sha256_json(plan_payload)
    plan_path.write_text(
        json.dumps({**plan_payload, "plan_digest": plan_digest}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    intake = replace(
        intake,
        server_plan_digest=plan_digest,
        intake_record_digest=None,
    )
    plan = FrozenServerPlanBinding(
        plan_path=str(plan_path.resolve()),
        plan_digest=plan_digest,
        jobs=jobs,
    )
    return_contract = DmcFixedHorizonReturnContract(horizon=1000)
    protocol = SourceEvaluationProtocol(
        intake_record_digest=intake.intake_record_digest,
        evaluator_implementation_digest=digest("source-evaluator-implementation"),
        return_contract_digest=return_contract.return_contract_digest,
        selection_seed_namespace_digest=digest("selection-seed-namespace"),
        attestation_seed_namespace_digest=digest("attestation-seed-namespace"),
        selection_reset_seeds=(101, 102),
        attestation_reset_seeds=(201, 202, 203),
        selection_episodes_per_candidate=2,
        attestation_episodes_per_champion=3,
        source_environment_digests={
            anchor_id: anchor.environment_instance_digest
            for anchor_id, anchor in anchors.items()
        },
        competence_floors={anchor_id: 0.5 for anchor_id in anchors},
        mean_tolerance=0.01,
        lcb_z=1.645,
    )
    backend = FakeSourceBackend(protocol.evaluator_implementation_digest)
    return intake, plan, protocol, return_contract, backend


def test_planner_automatically_joins_exact90_plan_anchor_bundle_and_abi(
    tmp_path: Path,
) -> None:
    intake, plan, protocol, _return_contract, backend = _case(tmp_path)
    units = plan_source_selection_work_units(intake, protocol, plan, backend)
    assert len(units) == 90
    assert set(units) == set(intake.cells)
    for candidate_id, unit in units.items():
        cell = intake.cells[candidate_id]
        planned = plan.jobs[candidate_id]
        assert unit.source_anchor_id == cell.source_anchor_id
        assert unit.anchor_manifest_path == planned.anchor.manifest_path
        assert unit.anchor_manifest_digest == planned.anchor.manifest_digest
        assert unit.anchor_runtime_digest == planned.anchor.runtime_digest
        assert unit.bundle_digest == cell.bundle_digest
        assert unit.execution_abi == backend.abi
        assert unit.reset_seeds == (101, 102)

    manifest = source_work_unit_manifest(units)
    assert manifest["work_unit_count"] == 90
    assert manifest["block"] == "source_selection"
    assert manifest["manifest_digest"] == sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_digest"}
    )


def test_production_planner_rejects_self_consistent_nonproduction_intake(
    tmp_path: Path,
) -> None:
    intake, plan, protocol, _return_contract, backend = _case(tmp_path)
    with pytest.raises(SourceEvaluatorError, match="frozen production trust anchor"):
        plan_source_selection_from_server_plan(
            intake,
            protocol,
            server_plan_path=plan.plan_path,
            backend=backend,
        )


def test_planner_fails_closed_on_plan_anchor_and_bundle_drift(tmp_path: Path) -> None:
    intake, plan, protocol, _return_contract, backend = _case(tmp_path)
    with pytest.raises(SourceEvaluatorError, match="path/self digest drifted"):
        plan_source_selection_work_units(
            intake,
            protocol,
            replace(plan, plan_digest=digest("another-plan")),
            backend,
        )

    first_anchor_id = next(iter(intake.candidates_by_anchor))
    original = next(
        job.anchor for job in plan.jobs.values() if job.anchor.source_anchor_id == first_anchor_id
    )
    poisoned_content = dict(original.manifest_content)
    poisoned_content["environment_instance_digest"] = digest("poisoned-environment")
    poisoned_content["manifest_digest"] = sha256_json(
        {key: value for key, value in poisoned_content.items() if key != "manifest_digest"}
    )
    poisoned_path = tmp_path / "poisoned_anchor.json"
    poisoned_path.write_text(json.dumps(poisoned_content, sort_keys=True) + "\n", encoding="utf-8")
    poisoned_anchor = CanonicalSourceAnchor.from_path(poisoned_path)
    poisoned_jobs = {
        candidate_id: (
            replace(job, anchor=poisoned_anchor)
            if job.anchor.source_anchor_id == first_anchor_id
            else job
        )
        for candidate_id, job in plan.jobs.items()
    }
    with pytest.raises(SourceEvaluatorError, match="differ from immutable plan bytes"):
        replace(plan, jobs=poisoned_jobs)

    backend.bundle_drift = True
    with pytest.raises(SourceEvaluatorError, match="validated binding drifted"):
        plan_source_selection_work_units(intake, protocol, plan, backend)


def test_runner_revalidates_abi_and_recomputes_raw_episode_shard(tmp_path: Path) -> None:
    intake, plan, protocol, return_contract, backend = _case(tmp_path)
    units = plan_source_selection_work_units(intake, protocol, plan, backend)
    unit = next(iter(units.values()))
    run = run_source_evaluation_work_unit(
        unit,
        backend=backend,
        return_contract=return_contract,
    )
    assert backend.calls == [101, 102]
    assert run.attempt.state == "SUCCEEDED"
    assert run.raw_episode_shard.raw_episode_returns == (250.0, 750.0)
    assert run.raw_episode_shard.normalized_returns == (0.25, 0.75)
    assert run.receipt.normalized_returns == (0.25, 0.75)
    restored = type(run.attempt).from_dict(run.attempt.to_dict())
    assert restored.to_dict() == run.attempt.to_dict()

    tampered_receipt = replace(
        run.receipt,
        normalized_returns=(1.0, 1.0),
        receipt_digest=None,
    )
    with pytest.raises(SourceEvaluatorError, match="inconsistent evidence"):
        replace(run, receipt=tampered_receipt, run_digest=None)

    first = replace(
        run.attempt.episodes[0],
        normalized_return=0.5,
        episode_record_digest=None,
    )
    poisoned_attempt = replace(
        run.attempt,
        episodes=(first, *run.attempt.episodes[1:]),
        attempt_record_digest=None,
    )
    with pytest.raises(SourceEvaluatorError, match="differs from raw-return recomputation"):
        rebuild_raw_source_episode_shard(unit, poisoned_attempt, return_contract)

    backend.abi = _abi("drifted")
    with pytest.raises(SourceEvaluatorError, match="execution ABI drifted"):
        run_source_evaluation_work_unit(
            unit,
            backend=backend,
            return_contract=return_contract,
        )


def test_episode_failure_and_runtime_drift_never_issue_receipt(tmp_path: Path) -> None:
    intake, plan, protocol, return_contract, backend = _case(tmp_path)
    unit = next(iter(plan_source_selection_work_units(intake, protocol, plan, backend).values()))
    backend.fail_seed = 102
    with pytest.raises(SourceEvaluationAttemptFailed) as captured:
        run_source_evaluation_work_unit(
            unit,
            backend=backend,
            return_contract=return_contract,
        )
    failed = captured.value.attempt_record
    assert failed.state == "FAILED"
    assert failed.failure_code == "FAKE_ROLLOUT_FAILURE"
    assert [row.reset_seed for row in failed.episodes] == [101, 102]
    assert failed.episodes[-1].normalized_return is None

    backend.fail_seed = None
    backend.runtime_drift_seed = 101
    with pytest.raises(SourceEvaluationAttemptFailed) as captured:
        run_source_evaluation_work_unit(
            unit,
            backend=backend,
            return_contract=return_contract,
        )
    assert captured.value.attempt_record.failure_code == "BACKEND_PROTOCOL_VIOLATION"
    assert len(captured.value.attempt_record.episodes) == 1


def test_return_contract_violation_preserves_raw_private_failure(tmp_path: Path) -> None:
    intake, plan, protocol, return_contract, backend = _case(tmp_path)
    unit = next(iter(plan_source_selection_work_units(intake, protocol, plan, backend).values()))
    backend.raw_returns[101] = 1000.01
    with pytest.raises(SourceEvaluationAttemptFailed) as captured:
        run_source_evaluation_work_unit(
            unit,
            backend=backend,
            return_contract=return_contract,
        )
    failed = captured.value.attempt_record
    assert failed.failure_code == "RETURN_CONTRACT_VIOLATION"
    assert failed.episodes[-1].raw_return == 1000.01
    assert failed.episodes[-1].normalized_return is None
