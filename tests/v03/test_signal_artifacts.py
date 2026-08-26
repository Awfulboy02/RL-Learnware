from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from policy_learnware_v0.hashing import sha256_file, sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.artifacts import V03ArtifactError, V03ArtifactLayout
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.dynamics_axis import (
    DynamicsAxisEntry,
    DynamicsAxisRegistry,
)
from policy_learnware_v0.v03.preflight import (
    HARD_TODO_IDS,
    FORMAL_PRODUCTION_STAGE_IDS,
    ExecutionCheckpoint,
    HardTodoEvidence,
    PreExperimentFreezeManifest,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.representation_plan import RepresentationExecutionPlan
from policy_learnware_v0.v03.signal_artifacts import (
    PUBLIC_SIGNAL_ATLAS_ARTIFACT_SCHEMA,
    SignalArtifactError,
    SignalAtlasArtifactRunner,
    SignalAtlasExecutionInterrupted,
    deserialize_signal_cell_run,
    serialize_signal_cell_run,
    transition_execution_checkpoint,
)
from policy_learnware_v0.v03.signal_atlas import (
    FormalSignalAtlasAuthorization,
    SignalAtlasError,
    SignalAtlasRun,
    expected_signal_work_keys,
    initialize_signal_execution_checkpoint,
    signal_asymmetric_kme_protocol_digest,
    signal_work_item_graph_digest,
    signal_work_key,
)
from policy_learnware_v0.v03.signal_controls import HistoricalRandomTanhSpec
from policy_learnware_v0.v03.signal_contrasts import build_signal_contrast_plan
from policy_learnware_v0.v03.signal_prefix import SignalPrefixSchedule
from policy_learnware_v0.v03.signal_matrix import (
    SignalCellRecord,
    SignalMatrixLedger,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.signal_metrics import (
    SignalDistanceRow,
    SignalMetricRecord,
)
from policy_learnware_v0.v03.signal_diagnostics import (
    AxisConfusionRecord,
    BankGeometryDiagnostic,
    SignalCellDiagnostics,
    axis_confusion_records,
)
from policy_learnware_v0.v03.signal_runtime import (
    DEVELOPMENT_SMOKE_MODE,
    FORMAL_MODE,
    SignalBankIdentity,
    SignalCellRun,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
    SourceKernelProtocol,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _dynamics_registry() -> DynamicsAxisRegistry:
    return DynamicsAxisRegistry(
        (
            DynamicsAxisEntry(
                "dynamics-low", "test-axis", "private-task", "embodiment",
                "abi", "goal", 0.0, "ANCHOR"
            ),
            DynamicsAxisEntry(
                "dynamics-high", "test-axis", "private-task", "embodiment",
                "abi", "goal", 1.0, "ANCHOR"
            ),
        )
    )


def _diagnostics(metric: SignalMetricRecord) -> SignalCellDiagnostics:
    bank_ids = sorted(
        {row.query_bank_id for row in metric.rows}
        | {row.source_bank_id for row in metric.rows}
    )
    return SignalCellDiagnostics(
        metric_record_digest=metric.record_digest,
        representation_coordinate_digest=metric.representation_coordinate_digest,
        bank_geometries=tuple(
            BankGeometryDiagnostic(
                bank_id=bank_id,
                represented_bank_digest=_d(f"diagnostic:{bank_id}"),
                data_role=(
                    "development_query"
                    if any(row.query_bank_id == bank_id for row in metric.rows)
                    else "source_reference_spec"
                ),
                sample_count=2,
                output_dim=1,
                numerical_rank=1,
                effective_rank=1.0,
                total_centered_variance=1.0,
                zero_variance_fraction=0.0,
                collapsed=False,
            )
            for bank_id in bank_ids
        ),
        confusion_records=axis_confusion_records(metric),
    )


def _run(
    cell_id: str,
    *,
    plan_digest: str,
    protocol_digest: str,
    work_item_digest: str,
    execution_mode: str = DEVELOPMENT_SMOKE_MODE,
) -> SignalCellRun:
    source_index_digest = _d(f"{cell_id}:source-index")
    coordinate_digest = _d(f"{cell_id}:coordinate")
    kernel = SourceKernelProtocol(
        representation_coordinate_digest=coordinate_digest,
        source_represented_bank_digests=(
            _d(f"{cell_id}:represented-a"),
            _d(f"{cell_id}:represented-b"),
        ),
        measurement_protocol_digest=_d("measurement"),
        execution_protocol_digest=protocol_digest,
        bandwidth=0.75,
        pair_budget=128,
        seed=11,
    )
    common = {
        "query_bank_id": "query-bank",
        "query_receipt_digest": _d(f"{cell_id}:query-receipt"),
        "query_raw_dataset_digest": _d(f"{cell_id}:query-raw"),
        "query_task_id": "private-query-task",
        "query_context_id": "context-a",
        "query_embodiment_id": "embodiment-a",
        "query_abi_contract_id": "abi-a",
        "query_goal_contract_id": "goal-a",
        "query_dynamics_context_id": "dynamics-a",
        "query_equivalence_class_id": "equivalence-a",
    }
    rows = (
        SignalDistanceRow(
            **common,
            source_bank_id="source-a",
            source_receipt_digest=_d(f"{cell_id}:source-a-receipt"),
            source_raw_dataset_digest=_d(f"{cell_id}:source-a-raw"),
            source_task_id="private-query-task",
            source_context_id="context-a",
            source_embodiment_id="embodiment-a",
            source_abi_contract_id="abi-a",
            source_goal_contract_id="goal-a",
            source_dynamics_context_id="dynamics-a",
            source_equivalence_class_id="equivalence-a",
            distance=0.1,
        ),
        SignalDistanceRow(
            **common,
            source_bank_id="source-b",
            source_receipt_digest=_d(f"{cell_id}:source-b-receipt"),
            source_raw_dataset_digest=_d(f"{cell_id}:source-b-raw"),
            source_task_id="private-source-task-b",
            source_context_id="context-b",
            source_embodiment_id="embodiment-b",
            source_abi_contract_id="abi-b",
            source_goal_contract_id="goal-b",
            source_dynamics_context_id="dynamics-b",
            source_equivalence_class_id="equivalence-b",
            distance=0.9,
        ),
    )
    metric = SignalMetricRecord(
        cell_id=cell_id,
        view_or_condition_id="V_FULL",
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=coordinate_digest,
        representation_seed=None,
        source_index_digest=source_index_digest,
        query_manifest_digest=_d(f"{cell_id}:queries"),
        rows=rows,
        expected_source_by_query={"query-bank": "source-a"},
    )
    return SignalCellRun(
        plan_digest=plan_digest,
        cell_id=cell_id,
        cell_digest=_d(f"{cell_id}:cell"),
        execution_protocol_digest=protocol_digest,
        execution_mode=execution_mode,
        source_fit_provenance_digest=None,
        work_item_digest=work_item_digest,
        evaluation_seed=None,
        kernel_protocol=kernel,
        source_index_digest=source_index_digest,
        query_run_digests={"query-bank": _d(f"{cell_id}:query-run")},
        metric_record=metric,
        diagnostics=_diagnostics(metric),
    )


def _pending_checkpoint(
    execution_plan_digest: str, work_ids: tuple[str, ...]
) -> ExecutionCheckpoint:
    return ExecutionCheckpoint(
        execution_plan_digest=execution_plan_digest,
        work_item_states={work_id: "PENDING" for work_id in work_ids},
        completed_artifact_digests={},
        attempt=0,
    )


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:synthetic"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _formal_contract():
    plan = build_signal_matrix_plan()
    historical = HistoricalRandomTanhSpec.create(
        seed=0, input_dim=3, output_dim=5
    )
    representation_plan = RepresentationExecutionPlan.create(
        signal_plan=plan, historical_spec=historical
    )
    condition_plan = ConditionExecutionPlan.create(historical_spec=historical)
    identity = SignalBankIdentity(
        receipt_digest=_d("receipt"),
        bank_id="source-bank",
        task_private_id="private-task",
        embodiment_id="embodiment",
        abi_contract_id="abi",
        goal_contract_id="goal",
        dynamics_context_id="dynamics",
        context_id="context",
        measurement_protocol_digest=_d("measurement"),
        probe_seed_digest=_d("probe-seed"),
        equivalence_class_id="equivalence",
    )
    registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=_d("taxonomy"), identities=(identity,)
    )
    protocol = SignalExecutionProtocol(
        plan_digest=str(plan.plan_digest),
        identity_registry_digest=str(registry.registry_digest),
        measurement_protocol_digest=registry.measurement_protocol_digest,
        representation_plan=representation_plan,
        condition_plan=condition_plan,
        execution_mode=FORMAL_MODE,
        reducer_config=ReducerConfig(
            support_budget=4,
            support_steps=0,
            kmeans_steps=0,
            ridge=0.0,
            pinv_rcond=1.0e-12,
        ),
        historical_seed=historical.seed,
    )
    work_digests = {
        work_id: _d(f"work:{work_id}")
        for work_id in expected_signal_work_keys(plan, protocol)
    }
    dynamics_registry = _dynamics_registry()
    freeze = PreExperimentFreezeManifest(
        freeze_id="formal-signal-artifact-test",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=_d("external-authority"),
        review_authority_verified=True,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=str(plan.plan_digest),
        signal_contrast_plan_digest=str(build_signal_contrast_plan().plan_digest),
        signal_materiality_threshold_digest=_d("signal-materiality-thresholds"),
        formal_signal_readout_plan_digest=_d("formal-signal-readout-plan"),
        preoracle_signal_outcome_plan_digest=_d("preoracle-signal-outcome-plan"),
        signal_identity_registry_digest=str(registry.registry_digest),
        signal_execution_protocol_digest=str(protocol.protocol_digest),
        representation_plan_digest=str(representation_plan.plan_digest),
        condition_plan_digest=str(condition_plan.plan_digest),
        formal_source_fit_schedule_digest=_d("source-fit-schedule"),
        formal_source_membership_digest=_d("source-membership"),
        signal_work_item_graph_digest=signal_work_item_graph_digest(
            plan, protocol, work_digests
        ),
        formal_signal_prefix_schedule_digest=str(
            SignalPrefixSchedule.formal().schedule_digest
        ),
        dynamics_axis_registry_digest=str(dynamics_registry.registry_digest),
        public_query_plan_digest=_d("public-query-plan"),
        baseline_plan_digest=_d("baselines"),
        statistics_plan_digest=_d("statistics"),
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=(
            signal_asymmetric_kme_protocol_digest(protocol)
        ),
        formal_gate_plan_digests={
            "G03-Attribution": _d("formal-attribution-plan"),
            "G03-Probe": _d("formal-probe-plan"),
            "G03-Market": _d("formal-market-plan"),
        },
        formal_stage_request_template_digests={
            stage_id: _d(f"formal-stage-request:{stage_id}")
            for stage_id in FORMAL_PRODUCTION_STAGE_IDS
        },
        hard_todo_evidence=tuple(_todo(item) for item in HARD_TODO_IDS),
        formal_stage_adapter_binding_digests={
            stage_id: formal_stage_adapter_binding_digest(
                stage_id,
                f"signal-artifact-adapter-{index}",
                _d(f"signal-adapter-contract-{index}"),
            )
            for index, stage_id in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
        },
    )
    authorization = FormalSignalAtlasAuthorization.bind(
        freeze,
        plan=plan,
        execution_protocol=protocol,
        identity_registry=registry,
        dynamics_axis_registry=dynamics_registry,
        work_item_digests=work_digests,
    )
    authorization.validate_signal_prefix_schedule(SignalPrefixSchedule.formal())
    authorization.validate_dynamics_axis_registry(dynamics_registry)
    return plan, protocol, authorization, work_digests


def test_checkpoint_state_machine_requires_bytes_digest() -> None:
    work_id = signal_work_key("cell-a", None)
    checkpoint = _pending_checkpoint(_d("execution-plan"), (work_id,))
    running = transition_execution_checkpoint(checkpoint, work_id, "RUNNING")
    failed = transition_execution_checkpoint(running, work_id, "FAILED")
    assert failed.work_item_states[work_id] == "FAILED"
    resumed = failed.resume()
    assert resumed.work_item_states[work_id] == "PENDING"
    assert resumed.attempt == 1

    running = transition_execution_checkpoint(resumed, work_id, "RUNNING")
    with pytest.raises(SignalArtifactError, match="requires the actual"):
        transition_execution_checkpoint(running, work_id, "COMPLETE")
    complete = transition_execution_checkpoint(
        running,
        work_id,
        "COMPLETE",
        completed_artifact_digest=_d("actual-file-bytes"),
    )
    assert complete.completed_artifact_digests[work_id] == _d(
        "actual-file-bytes"
    )
    with pytest.raises(SignalArtifactError, match="illegal"):
        transition_execution_checkpoint(complete, work_id, "RUNNING")


def test_formal_authorization_freezes_prefix_and_dynamics_readouts() -> None:
    _, _, authorization, _ = _formal_contract()
    with pytest.raises(SignalAtlasError, match="prefix schedule differs"):
        authorization.validate_signal_prefix_schedule(
            SignalPrefixSchedule.development((1, 2, 4, 8, 16, 32, 64))
        )
    drifted = DynamicsAxisRegistry(
        (
            DynamicsAxisEntry(
                "dynamics-low", "test-axis", "private-task", "embodiment",
                "abi", "goal", 0.0, "ANCHOR"
            ),
            DynamicsAxisEntry(
                "dynamics-high", "test-axis", "private-task", "embodiment",
                "abi", "goal", 2.0, "ANCHOR"
            ),
        )
    )
    with pytest.raises(SignalAtlasError, match="dynamics-axis registry differs"):
        authorization.validate_dynamics_axis_registry(drifted)


def test_private_signal_cell_round_trip_reconstructs_typed_rows() -> None:
    plan_digest = _d("plan")
    protocol_digest = _d("protocol")
    work_digest = _d("work")
    run = _run(
        "cell-a",
        plan_digest=plan_digest,
        protocol_digest=protocol_digest,
        work_item_digest=work_digest,
    )
    work_id = signal_work_key(run.cell_id, run.evaluation_seed)
    payload = serialize_signal_cell_run(
        work_id=work_id, work_item_digest=work_digest, run=run
    )
    restored = deserialize_signal_cell_run(
        payload,
        expected_work_id=work_id,
        expected_work_item_digest=work_digest,
        expected_plan_digest=plan_digest,
        expected_execution_protocol_digest=protocol_digest,
        expected_execution_mode=DEVELOPMENT_SMOKE_MODE,
    )
    assert restored == run
    assert restored.metric_record.rows == run.metric_record.rows
    assert restored.diagnostics.to_private_dict() == run.diagnostics.to_private_dict()
    assert restored.run_digest == run.run_digest


def test_fresh_process_resume_verifies_completed_bytes_and_continues(
    tmp_path: Path,
) -> None:
    execution_plan_digest = _d("execution-plan")
    plan_digest = _d("plan")
    protocol_digest = _d("protocol")
    work_ids = (
        signal_work_key("cell-a", None),
        signal_work_key("cell-b", None),
    )
    work_digests = {work_id: _d(f"work:{work_id}") for work_id in work_ids}
    layout = V03ArtifactLayout.development(tmp_path, "resume-test")
    runner = SignalAtlasArtifactRunner(
        layout=layout,
        execution_plan_digest=execution_plan_digest,
        plan_digest=plan_digest,
        execution_protocol_digest=protocol_digest,
        expected_work_item_digests=work_digests,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
    )
    progress = runner.start(_pending_checkpoint(execution_plan_digest, work_ids))
    calls: list[str] = []

    def interrupted_executor(work_id: str) -> SignalCellRun:
        calls.append(work_id)
        if work_id == work_ids[1]:
            raise RuntimeError("synthetic interruption")
        cell_id = work_id.removesuffix("--seed-NONE")
        return _run(
            cell_id,
            plan_digest=plan_digest,
            protocol_digest=protocol_digest,
            work_item_digest=work_digests[work_id],
        )

    with pytest.raises(SignalAtlasExecutionInterrupted) as raised:
        runner.run_remaining(progress, interrupted_executor)
    interrupted = raised.value
    assert interrupted.checkpoint_publication.checkpoint.work_item_states == {
        work_ids[0]: "COMPLETE",
        work_ids[1]: "FAILED",
    }
    completed_sha = interrupted.checkpoint_publication.checkpoint.completed_artifact_digests[
        work_ids[0]
    ]
    completed_path = layout.artifact(
        "signal_atlas_private", "work_items", f"{work_ids[0]}.json"
    )
    assert sha256_file(completed_path) == completed_sha

    # A new object simulates a fresh process: it has no in-memory completed run.
    restarted = SignalAtlasArtifactRunner(
        layout=V03ArtifactLayout.development(tmp_path, "resume-test"),
        execution_plan_digest=execution_plan_digest,
        plan_digest=plan_digest,
        execution_protocol_digest=protocol_digest,
        expected_work_item_digests=work_digests,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
    )
    recovered = restarted.resume(
        interrupted.checkpoint_publication.path,
        expected_checkpoint_sha256=interrupted.checkpoint_publication.artifact_sha256,
    )
    assert set(recovered.completed_runs) == {work_ids[0]}
    assert recovered.checkpoint_publication.checkpoint.attempt == 1
    assert recovered.checkpoint_publication.checkpoint.work_item_states[
        work_ids[1]
    ] == "PENDING"

    def completing_executor(work_id: str) -> SignalCellRun:
        cell_id = work_id.removesuffix("--seed-NONE")
        return _run(
            cell_id,
            plan_digest=plan_digest,
            protocol_digest=protocol_digest,
            work_item_digest=work_digests[work_id],
        )

    finished = restarted.run_remaining(recovered, completing_executor)
    assert set(finished.checkpoint_publication.checkpoint.work_item_states.values()) == {
        "COMPLETE"
    }
    assert set(finished.completed_runs) == set(work_ids)
    assert calls == [work_ids[0], work_ids[1]]

    # A COMPLETE item is trusted only through its exact checkpoint-bound bytes.
    completed_path.write_bytes(b"tampered\n")
    another_process = SignalAtlasArtifactRunner(
        layout=V03ArtifactLayout.development(tmp_path, "resume-test"),
        execution_plan_digest=execution_plan_digest,
        plan_digest=plan_digest,
        execution_protocol_digest=protocol_digest,
        expected_work_item_digests=work_digests,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
    )
    with pytest.raises(SignalArtifactError, match="failed byte verification"):
        another_process.resume(
            finished.checkpoint_publication.path,
            expected_checkpoint_sha256=finished.checkpoint_publication.artifact_sha256,
        )
    completed_path.unlink()
    with pytest.raises(SignalArtifactError, match="failed byte verification"):
        another_process.resume(
            finished.checkpoint_publication.path,
            expected_checkpoint_sha256=finished.checkpoint_publication.artifact_sha256,
        )


@dataclass(frozen=True)
class _PublicAtlasStub:
    work_item_digests: Mapping[str, str]
    leak_private_rows: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": PUBLIC_SIGNAL_ATLAS_ARTIFACT_SCHEMA,
            "plan_digest": _d("plan"),
            "logical_cell_records": [{"cell_id": "opaque-cell", "metrics": {"mrr": 1.0}}],
            "private_distance_rows_withheld": True,
        }
        if self.leak_private_rows:
            body["rows"] = [{"query_task_id": "secret-task"}]
        return {**body, "public_projection_digest": sha256_json(body)}


def test_formal_runner_requires_reviewed_exact_79_graph_and_seals_private_rows(
    tmp_path: Path,
) -> None:
    plan, protocol, authorization, work_digests = _formal_contract()
    assert len(work_digests) == 79
    assert (
        authorization.formal_source_fit_schedule_digest
        == authorization.freeze_manifest.formal_source_fit_schedule_digest
    )
    assert (
        authorization.formal_source_membership_digest
        == authorization.freeze_manifest.formal_source_membership_digest
    )
    with pytest.raises(SignalArtifactError, match="external atlas authorization"):
        SignalAtlasArtifactRunner(
            layout=V03ArtifactLayout.joint(tmp_path, "missing-authority"),
            execution_plan_digest=str(authorization.execution_plan_digest),
            plan_digest=str(plan.plan_digest),
            execution_protocol_digest=str(protocol.protocol_digest),
            expected_work_item_digests=work_digests,
            execution_mode=FORMAL_MODE,
        )
    one_key = next(iter(work_digests))
    with pytest.raises(SignalArtifactError, match="reviewed work graph"):
        SignalAtlasArtifactRunner(
            layout=V03ArtifactLayout.joint(tmp_path, "partial-graph"),
            execution_plan_digest=str(authorization.execution_plan_digest),
            plan_digest=str(plan.plan_digest),
            execution_protocol_digest=str(protocol.protocol_digest),
            expected_work_item_digests={one_key: work_digests[one_key]},
            execution_mode=FORMAL_MODE,
            formal_authorization=authorization,
        )
    substituted_work = dict(work_digests)
    substituted_work[one_key] = _d("substituted-bank-or-source-fit")
    with pytest.raises(SignalArtifactError, match="contents differ"):
        SignalAtlasArtifactRunner(
            layout=V03ArtifactLayout.joint(tmp_path, "substituted-work"),
            execution_plan_digest=str(authorization.execution_plan_digest),
            plan_digest=str(plan.plan_digest),
            execution_protocol_digest=str(protocol.protocol_digest),
            expected_work_item_digests=substituted_work,
            execution_mode=FORMAL_MODE,
            formal_authorization=authorization,
        )

    runner = SignalAtlasArtifactRunner(
        layout=V03ArtifactLayout.joint(tmp_path, "formal-test"),
        execution_plan_digest=str(authorization.execution_plan_digest),
        plan_digest=str(plan.plan_digest),
        execution_protocol_digest=str(protocol.protocol_digest),
        expected_work_item_digests=work_digests,
        execution_mode=FORMAL_MODE,
        formal_authorization=authorization,
    )
    checkpoint = initialize_signal_execution_checkpoint(plan, protocol)
    progress = runner.start(checkpoint)
    assert len(progress.checkpoint_publication.checkpoint.work_item_states) == 79
    with pytest.raises(SignalArtifactError, match="before every work item"):
        runner.publish_public_atlas(
            atlas_run=_PublicAtlasStub(work_digests), checkpoint=checkpoint
        )

    with pytest.raises(SignalArtifactError, match="leaks private fields"):
        SignalAtlasArtifactRunner._reject_private_public_payload(
            _PublicAtlasStub(work_digests, leak_private_rows=True).to_public_dict()
        )
    # Even a caller-supplied payload that happens to have the full public
    # shape cannot bypass the typed runner through the generic capability.
    forged_body = {
        "schema": PUBLIC_SIGNAL_ATLAS_ARTIFACT_SCHEMA,
        "plan_digest": str(plan.plan_digest),
        "execution_protocol_digest": str(protocol.protocol_digest),
        "identity_registry_digest": str(protocol.identity_registry_digest),
        "formal_authorization_digest": str(authorization.authorization_digest),
        "freeze_manifest_digest": str(
            authorization.freeze_manifest.freeze_manifest_digest
        ),
        "logical_cell_records": [],
        "seed_metric_records": {},
        "private_distance_rows_withheld": True,
        "private_run_digest": _d("forged-private-run"),
    }
    forged_payload = {
        **forged_body,
        "public_projection_digest": sha256_json(forged_body),
    }
    with pytest.raises(V03ArtifactError, match="authorized atlas publisher"):
        runner.layout.writer("signal_atlas").publish_json(
            runner.layout.artifact("signal_atlas", "public", "bypass.json"),
            forged_payload,
        )


def test_joint_public_atlas_writer_accepts_aggregate_diagnostics_and_rejects_rows(
    tmp_path: Path,
) -> None:
    """Keep the final joint whitelist aligned with SignalAtlasRun projection."""

    layout = V03ArtifactLayout.joint(tmp_path, "diagnostic-projection")
    body = {
        "schema": PUBLIC_SIGNAL_ATLAS_ARTIFACT_SCHEMA,
        "plan_digest": _d("plan"),
        "execution_protocol_digest": _d("protocol"),
        "identity_registry_digest": _d("identities"),
        "formal_authorization_digest": _d("authorization"),
        "freeze_manifest_digest": _d("freeze"),
        "logical_cell_records": [],
        "seed_metric_records": {},
        "seed_diagnostic_records": {
            "opaque-work": {
                "schema": "policy-learnware.v03-public-signal-cell-diagnostics.v0",
                "metric_record_digest": _d("metric"),
                "representation_coordinate_digest": _d("coordinate"),
                "bank_count": 4,
                "effective_rank_mean": 2.0,
                "effective_rank_min": 1.0,
                "effective_rank_max": 3.0,
                "collapsed_bank_count": 0,
                "axis_summaries": {},
                "private_bank_and_taxonomy_rows_withheld": True,
                "private_diagnostics_digest": _d("private-diagnostics"),
                "public_projection_digest": _d("cell-public-projection"),
            }
        },
        "control_audit_records": {},
        "private_distance_rows_withheld": True,
        "private_run_digest": _d("private-run"),
    }
    payload = {**body, "public_projection_digest": sha256_json(body)}
    destination = layout.artifact("signal_atlas", "public", "signal_atlas.json")
    digest = layout._authorized_signal_atlas_writer().publish_json(
        destination, payload
    )
    assert len(digest) == 64

    leaked_body = dict(body)
    leaked_body["seed_diagnostic_records"] = {
        "opaque-work": {"bank_geometries": [{"bank_id": "private-bank"}]}
    }
    leaked = {
        **leaked_body,
        "public_projection_digest": sha256_json(leaked_body),
    }
    with pytest.raises(V03ArtifactError, match="leaks private fields"):
        layout._authorized_signal_atlas_writer().publish_json(
            layout.artifact("signal_atlas", "public", "leaked.json"), leaked
        )


def test_typed_atlas_constructor_rejects_partial_runs_even_with_full_ledger() -> None:
    plan, protocol, authorization, _work_digests = _formal_contract()
    registry = protocol.identity_registry_digest
    # Recover the exact registry object carried by the authorization fixture by
    # rebuilding its one frozen identity.  The digest join is the invariant.
    identity = SignalBankIdentity(
        receipt_digest=_d("receipt"),
        bank_id="source-bank",
        task_private_id="private-task",
        embodiment_id="embodiment",
        abi_contract_id="abi",
        goal_contract_id="goal",
        dynamics_context_id="dynamics",
        context_id="context",
        measurement_protocol_digest=_d("measurement"),
        probe_seed_digest=_d("probe-seed"),
        equivalence_class_id="equivalence",
    )
    identity_registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=_d("taxonomy"), identities=(identity,)
    )
    assert identity_registry.registry_digest == registry
    records = tuple(
        SignalCellRecord(
            plan_digest=str(plan.plan_digest),
            cell_id=cell.cell_id,
            cell_digest=str(cell.cell_digest),
            status=(
                "STRUCTURAL_NA"
                if cell.applicability == "STRUCTURAL_NA"
                else "COMPUTED"
            ),
            metrics=(None if cell.applicability == "STRUCTURAL_NA" else {"mrr": 1.0}),
            numeric_artifact_digest=(
                None if cell.applicability == "STRUCTURAL_NA" else _d(cell.cell_id)
            ),
        )
        for cell in plan.cells
    )
    ledger = SignalMatrixLedger(plan=plan, records=records)
    with pytest.raises(SignalAtlasError, match="complete frozen schedule"):
        SignalAtlasRun(
            plan=plan,
            execution_protocol=protocol,
            identity_registry=identity_registry,
            formal_authorization=authorization,
            work_items={},
            cell_runs={},
            ledger=ledger,
        )
