from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from policy_learnware_v0.hashing import sha256_json
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
    HardTodoEvidence,
    PreExperimentFreezeManifest,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.representation_plan import RepresentationExecutionPlan
from policy_learnware_v0.v03.signal_artifacts import (
    FormalPairControlAuthorization,
    PairControlArtifactRunner,
    SignalArtifactError,
)
from policy_learnware_v0.v03.signal_atlas import (
    FormalSignalAtlasAuthorization,
    expected_signal_work_keys,
    signal_asymmetric_kme_protocol_digest,
    signal_work_item_graph_digest,
)
from policy_learnware_v0.v03.signal_controls import (
    BankControlReference,
    ExactRepeatDistanceResult,
    ExactRepeatPairContract,
    HistoricalRandomTanhSpec,
    PairControlEvaluation,
    PairControlMembershipEvidence,
    PairControlPlan,
    SchemaCollisionPairContract,
)
from policy_learnware_v0.v03.signal_contrasts import build_signal_contrast_plan
from policy_learnware_v0.v03.signal_matrix import build_signal_matrix_plan
from policy_learnware_v0.v03.signal_prefix import SignalPrefixSchedule
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
    FORMAL_MODE,
    SignalBankIdentity,
    SignalCellRun,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
    SourceKernelProtocol,
)
from policy_learnware_v0.v03.transition_views import V_REWARD_FREE_TRANSITION


def _d(label: str) -> str:
    return sha256_json({"schema": "pair-control-artifact-test.v0", "label": label})


def _dynamics_registry() -> DynamicsAxisRegistry:
    return DynamicsAxisRegistry(
        (
            DynamicsAxisEntry(
                "test-dynamics-low", "test-axis", "test-task", "test-emb",
                "test-abi", "test-goal", 0.0, "ANCHOR"
            ),
            DynamicsAxisEntry(
                "test-dynamics-high", "test-axis", "test-task", "test-emb",
                "test-abi", "test-goal", 1.0, "ANCHOR"
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


def _reference(
    bank_id: str,
    *,
    task: str,
    goal: str,
    context: str,
    probe: str,
) -> BankControlReference:
    return BankControlReference(
        bank_id=bank_id,
        registered_task_id=task,
        embodiment_id=task,
        abi_contract_id="abi-3-2",
        goal_contract_id=goal,
        dynamics_context_id=context,
        context_id=context,
        observation_dim=3,
        action_dim=2,
        bank_digest=_d(f"native:{bank_id}"),
        measurement_protocol_digest=_d("measurement"),
        probe_seed_digest=_d(f"probe:{probe}"),
    )


def _pair_plan() -> PairControlPlan:
    left = _reference(
        "walker-repeat-a",
        task="walker",
        goal="walk",
        context="nominal",
        probe="walker-a",
    )
    schema_right = _reference(
        "finger-schema",
        task="finger",
        goal="turn",
        context="nominal",
        probe="finger",
    )
    repeat_right = _reference(
        "walker-repeat-b",
        task="walker",
        goal="walk",
        context="nominal",
        probe="walker-b",
    )
    return PairControlPlan(
        (
            SchemaCollisionPairContract(
                pair_id="opaque-schema-pair",
                left=left,
                right=schema_right,
                metric_ids=("task_top1",),
                statistical_identity="schema_collision_primary",
                preregistration_digest=_d("schema-preregistration"),
            ),
            ExactRepeatPairContract(
                pair_id="opaque-repeat-pair",
                left=left,
                right=repeat_right,
                metric_ids=("direct_repeat_mmd",),
                statistical_identity="exact_repeat_noise_floor",
                preregistration_digest=_d("repeat-preregistration"),
            ),
        )
    )


def _identity(reference: BankControlReference) -> SignalBankIdentity:
    return SignalBankIdentity(
        receipt_digest=_d(f"receipt:{reference.bank_id}"),
        bank_id=reference.bank_id,
        task_private_id=reference.registered_task_id,
        embodiment_id=reference.embodiment_id,
        abi_contract_id=reference.abi_contract_id,
        goal_contract_id=reference.goal_contract_id,
        dynamics_context_id=reference.dynamics_context_id,
        context_id=reference.context_id,
        measurement_protocol_digest=reference.measurement_protocol_digest,
        probe_seed_digest=reference.probe_seed_digest,
        equivalence_class_id=f"eq-{reference.registered_task_id}-{reference.context_id}",
    )


def _todo(todo_id: str, *, pair_plan_digest: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=(
            pair_plan_digest if todo_id == "T-P4-03" else _d(f"{todo_id}:contract")
        ),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:synthetic"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _formal_pair_contract():
    pair_plan = _pair_plan()
    references = {
        reference.bank_id: reference
        for contract in pair_plan.contracts
        for reference in (contract.left, contract.right)
    }
    registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=_d("taxonomy"),
        identities=tuple(_identity(item) for item in references.values()),
    )
    signal_plan = build_signal_matrix_plan()
    historical = HistoricalRandomTanhSpec.create(
        seed=0, input_dim=3, output_dim=5
    )
    representation_plan = RepresentationExecutionPlan.create(
        signal_plan=signal_plan, historical_spec=historical
    )
    condition_plan = ConditionExecutionPlan.create(historical_spec=historical)
    protocol = SignalExecutionProtocol(
        plan_digest=str(signal_plan.plan_digest),
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
        key: _d(f"work:{key}")
        for key in expected_signal_work_keys(signal_plan, protocol)
    }
    dynamics_registry = _dynamics_registry()
    freeze = PreExperimentFreezeManifest(
        freeze_id="pair-control-artifact-test",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=_d("external-authority"),
        review_authority_verified=True,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=str(signal_plan.plan_digest),
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
            signal_plan, protocol, work_digests
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
        hard_todo_evidence=tuple(
            _todo(item, pair_plan_digest=pair_plan.plan_digest)
            for item in HARD_TODO_IDS
        ),
        formal_stage_adapter_binding_digests={
            stage_id: formal_stage_adapter_binding_digest(
                stage_id,
                f"pair-control-adapter-{index}",
                _d(f"pair-adapter-contract-{index}"),
            )
            for index, stage_id in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
        },
    )
    atlas_authorization = FormalSignalAtlasAuthorization.bind(
        freeze,
        plan=signal_plan,
        execution_protocol=protocol,
        identity_registry=registry,
        dynamics_axis_registry=dynamics_registry,
        work_item_digests=work_digests,
    )
    pair_authorization = FormalPairControlAuthorization.bind(
        atlas_authorization,
        plan=pair_plan,
        identity_registry=registry,
    )
    return pair_plan, registry, pair_authorization


def _evaluations(
    plan: PairControlPlan, registry: SignalIdentityRegistry
) -> tuple[PairControlEvaluation | ExactRepeatDistanceResult, ...]:
    identities = {item.bank_id: item for item in registry.identities}
    references = {
        reference.bank_id: reference
        for contract in plan.contracts
        for reference in (contract.left, contract.right)
    }
    source_ids = {bank_id: bank_id for bank_id in references}
    rows = []
    for query_id, reference in sorted(references.items()):
        for source_bank_id, source_reference in sorted(references.items()):
            rows.append(
                SignalDistanceRow(
                    query_bank_id=query_id,
                    source_bank_id=source_ids[source_bank_id],
                    query_receipt_digest=identities[query_id].receipt_digest,
                    source_receipt_digest=identities[source_bank_id].receipt_digest,
                    query_raw_dataset_digest=reference.bank_digest,
                    source_raw_dataset_digest=source_reference.bank_digest,
                    query_task_id=reference.registered_task_id,
                    source_task_id=source_reference.registered_task_id,
                    query_context_id=reference.context_id,
                    source_context_id=source_reference.context_id,
                    query_embodiment_id=reference.registered_task_id,
                    source_embodiment_id=source_reference.registered_task_id,
                    query_abi_contract_id="abi-3-2",
                    source_abi_contract_id="abi-3-2",
                    query_goal_contract_id=reference.goal_contract_id,
                    source_goal_contract_id=source_reference.goal_contract_id,
                    query_dynamics_context_id=reference.context_id,
                    source_dynamics_context_id=source_reference.context_id,
                    query_equivalence_class_id=(
                        f"eq-{reference.registered_task_id}-{reference.context_id}"
                    ),
                    source_equivalence_class_id=(
                        f"eq-{source_reference.registered_task_id}-{source_reference.context_id}"
                    ),
                    distance=0.1 if source_bank_id == query_id else 2.0,
                )
            )
    record = SignalMetricRecord(
        cell_id="CORE_PAIRED::V_REWARD_FREE_TRANSITION::R0_PADDED_RAW",
        view_or_condition_id=V_REWARD_FREE_TRANSITION,
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=_d("coordinate"),
        representation_seed=None,
        source_index_digest=_d("source-index"),
        query_manifest_digest=_d("query-manifest"),
        rows=tuple(rows),
        expected_source_by_query={
            bank_id: source_ids[bank_id] for bank_id in references
        },
    )
    run_protocol_digest = _d("pair-panel-run-protocol")
    kernel = SourceKernelProtocol(
        representation_coordinate_digest=record.representation_coordinate_digest,
        source_represented_bank_digests=(
            _d("pair-panel-represented-a"),
            _d("pair-panel-represented-b"),
        ),
        measurement_protocol_digest=registry.measurement_protocol_digest,
        execution_protocol_digest=run_protocol_digest,
        bandwidth=1.0,
        pair_budget=64,
        seed=0,
    )
    run = SignalCellRun(
        plan_digest=_d("pair-panel-run-plan"),
        cell_id=record.cell_id,
        cell_digest=_d("pair-panel-run-cell"),
        execution_protocol_digest=run_protocol_digest,
        execution_mode=FORMAL_MODE,
        source_fit_provenance_digest=None,
        work_item_digest=_d("pair-panel-work-item"),
        evaluation_seed=None,
        kernel_protocol=kernel,
        source_index_digest=record.source_index_digest,
        query_run_digests={
            bank_id: _d(f"pair-panel-query-run:{bank_id}")
            for bank_id in references
        },
        metric_record=record,
        diagnostics=_diagnostics(record),
    )
    results = []
    for contract in plan.contracts:
        membership = PairControlMembershipEvidence.create(
            contract,
            left_receipt_digest=identities[contract.left.bank_id].receipt_digest,
            right_receipt_digest=identities[contract.right.bank_id].receipt_digest,
            left_feature_bank_digest=_d(f"feature:{contract.left.bank_id}"),
            right_feature_bank_digest=_d(f"feature:{contract.right.bank_id}"),
        )
        if isinstance(contract, SchemaCollisionPairContract):
            results.append(
                PairControlEvaluation.evaluate(contract, record, membership)
            )
        else:
            results.append(
                ExactRepeatDistanceResult.evaluate(
                    contract,
                    membership,
                    run,
                )
            )
    return tuple(results)


def test_formal_pair_authorization_requires_exact_reviewed_t_p4_03_plan() -> None:
    plan, registry, authorization = _formal_pair_contract()
    authorization.validate(plan=plan, identity_registry=registry)
    changed_plan = replace(
        plan,
        contracts=tuple(
            replace(
                item,
                statistical_identity=f"{item.statistical_identity}_changed",
                pair_digest=None,
            )
            for item in plan.contracts
        ),
    )
    with pytest.raises(SignalArtifactError, match="reviewed T-P4-03"):
        FormalPairControlAuthorization.bind(
            authorization.atlas_authorization,
            plan=changed_plan,
            identity_registry=registry,
        )


def test_pair_panel_is_separate_private_byte_bound_and_public_aggregate_only(
    tmp_path: Path,
) -> None:
    plan, registry, authorization = _formal_pair_contract()
    evaluations = _evaluations(plan, registry)
    runner = PairControlArtifactRunner(
        layout=V03ArtifactLayout.joint(tmp_path, "formal-pair-panel"),
        plan=plan,
        identity_registry=registry,
        formal_authorization=authorization,
    )
    publications = tuple(
        runner.publish_private_evaluation(item) for item in evaluations
    )
    with pytest.raises(SignalArtifactError, match="exact frozen pair coverage"):
        runner.publish_public_panel(
            atlas_run=None,  # type: ignore[arg-type]
            evaluations=evaluations[:1], publications=publications[:1]
        )

    with pytest.raises(SignalArtifactError, match="authorized complete SignalAtlasRun"):
        runner.publish_public_panel(
            atlas_run=None,  # type: ignore[arg-type]
            evaluations=evaluations,
            publications=publications,
        )
    public_results = [runner._public_result(item) for item in evaluations]
    rendered = str(public_results)
    assert all(
        item["private_pair_membership_withheld"] is True
        for item in public_results
    )
    for forbidden in (
        "pair_id",
        "left_bank_id",
        "right_bank_id",
        "query_bank_id",
        "rows",
        "walker-repeat-a",
        "finger-schema",
    ):
        assert forbidden not in rendered
    assert "logical_cell_records" not in rendered

    with pytest.raises(V03ArtifactError, match="authorized pair-control publisher"):
        runner.layout.writer("pair_controls")


def test_public_pair_panel_rejects_swapped_or_tampered_private_evidence(
    tmp_path: Path,
) -> None:
    plan, registry, authorization = _formal_pair_contract()
    evaluations = _evaluations(plan, registry)
    runner = PairControlArtifactRunner(
        layout=V03ArtifactLayout.joint(tmp_path, "tampered-pair-panel"),
        plan=plan,
        identity_registry=registry,
        formal_authorization=authorization,
    )
    publications = tuple(
        runner.publish_private_evaluation(item) for item in evaluations
    )
    mismatched_publications = (
        replace(
            publications[0],
            evaluation_digest=publications[1].evaluation_digest,
        ),
        publications[1],
    )
    with pytest.raises(SignalArtifactError, match="another evaluation"):
        runner._verify_private_publication(
            evaluations[0], mismatched_publications[0]
        )

    publications[0].path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SignalArtifactError, match="byte verification"):
        runner._verify_private_publication(evaluations[0], publications[0])
