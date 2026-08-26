from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json, sha256_ndarrays
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.representation_ladder import (
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    FormalTrainedRepresentationReceipt,
    RepresentationBatch,
    TrainedCallableArtifact,
    TrainingRequest,
    bind_historical_random_tanh,
    fit_r0_identity,
    fit_r1_random_linear,
    fit_r3_matched_random_mlp,
    fit_r5_corro_style,
)
from policy_learnware_v0.v03.corro_trainers import (
    CorroOptimizationConfig,
    TASK_SUPCON_OBJECTIVE_DIGEST,
)
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.representation_plan import (
    RepresentationExecutionPlan,
    RepresentationPlanError,
)
from policy_learnware_v0.v03.signal_runtime import (
    FormalFeatureBank,
    SignalBankIdentity,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
    SignalRuntimeError,
    bank_control_reference_from_feature_bank,
    feature_bank_from_rf_shuffled_next,
    feature_bank_from_transition_view,
    fit_source_kernel_protocol,
    represented_bank_from_historical_random_tanh,
    run_signal_cell,
    transform_feature_banks,
    validate_pair_control_feature_banks,
)
from policy_learnware_v0.v03.signal_atlas import (
    SignalAtlasError,
    SignalCellWorkItem,
    build_rf_control_audit_summary,
    expected_signal_work_keys,
    initialize_signal_execution_checkpoint,
    run_signal_atlas,
    validate_formal_atlas_fit_schedule_bindings,
)
from policy_learnware_v0.v03.signal_controls import (
    ExactRepeatPairContract,
    HistoricalRandomTanhSpec,
    RewardFreeShuffledNextSpec,
)
from policy_learnware_v0.v03.signal_matrix import (
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.transition_views import (
    V_FULL_LEGACY,
    V_REWARD_FREE_TRANSITION,
    V_SHUFFLED_NEXT,
    TransitionBank,
    apply_transition_view,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _native(
    bank_id: str,
    task: str,
    role: str,
    *,
    obs_dim: int,
    act_dim: int,
    center: float,
) -> NativeTransitionBank:
    observation = center + np.arange(4 * obs_dim, dtype=np.float64).reshape(4, obs_dim) / 100.0
    action = center / 10.0 + np.arange(4 * act_dim, dtype=np.float64).reshape(4, act_dim) / 100.0
    return NativeTransitionBank(
        bank_id=bank_id,
        task_private_id=task,
        data_role=role,  # type: ignore[arg-type]
        native_schema_digest=_d(f"schema:{task}"),
        raw_dataset_digest=_d(f"raw:{bank_id}"),
        observation=observation,
        action=action,
        reward=np.asarray([0.0, 0.1, 0.2, 0.3]) + center / 100.0,
        next_observation=observation + 0.05,
        terminated=np.asarray([False, True, False, True]),
        truncated=np.asarray([False, False, False, False]),
        episode_id=np.asarray([0, 0, 1, 1]),
        timestep=np.asarray([0, 1, 0, 1]),
    )


def _fixture():
    fit_a = _native(
        "fit-a", "task-a", "source_representation_train", obs_dim=2, act_dim=1, center=0.0
    )
    fit_b = _native(
        "fit-b", "task-b", "source_representation_train", obs_dim=3, act_dim=2, center=10.0
    )
    registry = NativeShapeRegistry.from_source_banks((fit_a, fit_b))
    normalizer = fit_global_normalizer((fit_a, fit_b), registry=registry)
    canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
    native = (
        _native("source-a", "task-a", "source_reference_spec", obs_dim=2, act_dim=1, center=0.1),
        _native("source-b", "task-b", "source_reference_spec", obs_dim=3, act_dim=2, center=10.1),
        _native("query-a", "task-a", "development_query", obs_dim=2, act_dim=1, center=0.12),
        _native("query-b", "task-b", "development_query", obs_dim=3, act_dim=2, center=10.12),
    )
    receipts = tuple(canonicalizer.transform(item) for item in native)
    measurement = _d("measurement-protocol")
    identities = tuple(
        SignalBankIdentity.from_receipt(
            receipt,
            embodiment_id=("walker" if receipt.task_private_id == "task-a" else "finger"),
            abi_contract_id=f"abi-{receipt.task_private_id}",
            goal_contract_id=f"goal-{receipt.task_private_id}",
            dynamics_context_id=f"dynamics-{receipt.task_private_id}",
            context_id=f"context-{receipt.task_private_id}",
            measurement_protocol_digest=measurement,
            probe_seed_digest=_d(f"probe:{receipt.bank_id}"),
            equivalence_class_id=f"equivalence-{receipt.task_private_id}",
        )
        for receipt in receipts
    )
    features = []
    for receipt, identity in zip(receipts, identities, strict=True):
        bank = TransitionBank.from_canonical_batch(receipt.batch)
        view = apply_transition_view(bank, V_REWARD_FREE_TRANSITION)
        features.append(feature_bank_from_transition_view(receipt, identity, view))
    source_fit = RepresentationBatch(
        values=np.concatenate([item.values for item in features[:2]], axis=0),
        dataset_digest=_d("source-fit-features"),
        role="SOURCE_FIT",
    )
    fitted = fit_r0_identity(source_fit)
    represented = transform_feature_banks(fitted, features)
    return canonicalizer, receipts, identities, features, represented


def _execution(
    identities,
    plan,
    *,
    historical_spec: HistoricalRandomTanhSpec | None = None,
    execution_mode: str = "DEVELOPMENT_SMOKE",
) -> tuple[SignalIdentityRegistry, SignalExecutionProtocol]:
    if historical_spec is None:
        historical_spec = HistoricalRandomTanhSpec.create(
            seed=0, input_dim=1, output_dim=5
        )
    registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=_d("task-taxonomy"), identities=tuple(identities)
    )
    representation_plan = RepresentationExecutionPlan.create(
        signal_plan=plan, historical_spec=historical_spec
    )
    condition_plan = ConditionExecutionPlan.create(historical_spec=historical_spec)
    protocol = SignalExecutionProtocol(
        plan_digest=str(plan.plan_digest),
        identity_registry_digest=str(registry.registry_digest),
        measurement_protocol_digest=registry.measurement_protocol_digest,
        representation_plan=representation_plan,
        condition_plan=condition_plan,
        execution_mode=execution_mode,
        reducer_config=ReducerConfig(
            support_budget=4,
            support_steps=0,
            kmeans_steps=0,
            ridge=0.0,
            pinv_rcond=1.0e-12,
        ),
        pair_budget=64,
        block_size=2,
        historical_seed=historical_spec.seed,
    )
    return registry, protocol


def test_canonical_receipt_to_raw_kme_empirical_query_cpu_cell() -> None:
    _canonicalizer, _receipts, identities, _features, represented = _fixture()
    plan = build_signal_matrix_plan()
    registry, protocol = _execution(identities, plan)
    cell = plan.cell("CORE_PAIRED::V_REWARD_FREE_TRANSITION::R0_PADDED_RAW")
    run = run_signal_cell(
        plan=plan,
        cell=cell,
        source_banks=represented[:2],
        query_banks=represented[2:],
        expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
        identity_registry=registry,
        execution_protocol=protocol,
    )
    assert run.metric_record.metric_values["context_top1"] == 1.0
    assert run.metric_record.metric_values["task_top1"] == 1.0
    assert run.metric_record.metric_values["between_within_ratio"] > 1.0
    assert len(run.query_run_digests) == 2
    assert run.execution_mode == "DEVELOPMENT_SMOKE"
    assert run.source_fit_provenance_digest is None
    assert run.work_item_digest is None

    work_item = SignalCellWorkItem(
        plan=plan,
        cell=cell,
        source_banks=represented[:2],
        query_banks=represented[2:],
        expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
        identity_registry=registry,
        execution_protocol=protocol,
        evaluation_seed=None,
    )
    work_run = work_item.execute()
    assert work_run.metric_record.record_digest == run.metric_record.record_digest
    assert work_run.work_item_digest == work_item.work_item_digest
    assert work_run.run_digest != run.run_digest
    work_item.validate_run(work_run)
    mismatched_query_schedule = replace(
        work_run,
        query_run_digests={"query-not-in-work-item": _d("forged-query-run")},
        run_digest=None,
    )
    with pytest.raises(SignalAtlasError, match="metric/query schedule"):
        work_item.validate_run(mismatched_query_schedule)
    # 37 numeric logical cells expand to 79 deterministic/stochastic run keys;
    # a single-row smoke can never impersonate formal three-seed coverage.
    assert len(expected_signal_work_keys(plan, protocol)) == 79
    checkpoint = initialize_signal_execution_checkpoint(plan, protocol)
    assert len(checkpoint.work_item_states) == 79
    assert set(checkpoint.work_item_states.values()) == {"PENDING"}
    with pytest.raises(SignalAtlasError, match="FORMAL execution protocol"):
        run_signal_atlas(plan, protocol, registry, (work_item,))


def test_historical_fourteenth_cell_is_bridged_without_double_encoding() -> None:
    _canonicalizer, receipts, identities, _features, _represented = _fixture()
    banks = tuple(TransitionBank.from_canonical_batch(item.batch) for item in receipts)
    full_views = tuple(apply_transition_view(bank, V_FULL_LEGACY) for bank in banks)
    input_dim = full_views[0].feature_matrix.shape[1]
    spec = HistoricalRandomTanhSpec.create(
        seed=17, input_dim=input_dim, output_dim=5
    )
    source_fit = RepresentationBatch(
        values=np.concatenate(
            [view.feature_matrix for view in full_views[:2]], axis=0
        ),
        dataset_digest=_d("historical-source-fit"),
        role="SOURCE_FIT",
    )
    fitted = bind_historical_random_tanh(source_fit, spec=spec)
    represented = tuple(
        represented_bank_from_historical_random_tanh(
            receipt,
            identity,
            full,
            spec.apply(bank),
            fitted,
        )
        for receipt, identity, full, bank in zip(
            receipts, identities, full_views, banks, strict=True
        )
    )
    assert all(
        item.feature_bank.values.shape[1] == input_dim for item in represented
    )
    assert all(item.values.shape[1] == spec.output_dim for item in represented)
    plan = build_signal_matrix_plan()
    registry, protocol = _execution(identities, plan, historical_spec=spec)
    cell = plan.cell(
        "HISTORICAL_CONTROL::V_RANDOM_ENCODER::R_HIST_RANDOM_TANH"
    )
    run = run_signal_cell(
        plan=plan,
        cell=cell,
        source_banks=represented[:2],
        query_banks=represented[2:],
        expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
        identity_registry=registry,
        execution_protocol=protocol,
    )
    assert run.cell_id == cell.cell_id

    wrong_result = spec.apply(banks[0])
    with pytest.raises(SignalRuntimeError, match="not built from"):
        represented_bank_from_historical_random_tanh(
            receipts[1], identities[1], full_views[1], wrong_result, fitted
        )


def test_runtime_rejects_bare_batch_target_bandwidth_fit_and_mixed_coordinate() -> None:
    _canonicalizer, receipts, identities, features, represented = _fixture()
    plan = build_signal_matrix_plan()
    _registry, protocol = _execution(identities, plan)
    with pytest.raises(SignalRuntimeError, match="CanonicalizedBankReceipt"):
        FormalFeatureBank(  # type: ignore[arg-type]
            receipt=receipts[0].batch,
            identity=identities[0],
            condition_id=V_REWARD_FREE_TRANSITION,
            condition_transform_digest=_d("view"),
            values=features[0].values,
        )
    with pytest.raises(SignalRuntimeError, match="source_reference_spec"):
        fit_source_kernel_protocol(
            (represented[0], represented[2]), execution_protocol=protocol
        )
    alternate_source = RepresentationBatch(
        values=np.concatenate([item.values for item in features[:2]], axis=0),
        dataset_digest=_d("alternate-source-fit-features"),
        role="SOURCE_FIT",
    )
    alternate = transform_feature_banks(
        fit_r0_identity(alternate_source), (features[1],)
    )[0]
    with pytest.raises(SignalRuntimeError, match="mix representation"):
        fit_source_kernel_protocol(
            (represented[0], alternate), execution_protocol=protocol
        )


def test_feature_bridges_reject_cross_bank_view_and_control_provenance() -> None:
    _canonicalizer, receipts, identities, _features, _represented = _fixture()
    left_bank = TransitionBank.from_canonical_batch(receipts[0].batch)
    left_view = apply_transition_view(left_bank, V_REWARD_FREE_TRANSITION)
    forged_view = replace(
        left_view,
        archived_dataset_digest=receipts[1].canonical_transition_digest,
    )
    with pytest.raises(SignalRuntimeError, match="arrays were not built"):
        feature_bank_from_transition_view(receipts[1], identities[1], forged_view)

    left_control = RewardFreeShuffledNextSpec(seed=7).apply(left_bank)
    with pytest.raises(SignalRuntimeError, match="control was not built"):
        feature_bank_from_rf_shuffled_next(
            receipts[1], identities[1], left_control
        )


def test_rf_feature_bank_retains_typed_result_and_audit_provenance() -> None:
    _canonicalizer, receipts, identities, _features, _represented = _fixture()
    bank = TransitionBank.from_canonical_batch(receipts[0].batch)
    control = RewardFreeShuffledNextSpec(seed=109).apply(bank)
    feature = feature_bank_from_rf_shuffled_next(
        receipts[0], identities[0], control
    )
    assert feature.rf_shuffled_next_result is control
    assert feature.condition_result_digest == control.dataset_digest
    assert feature.condition_audit_digest == control.marginal_audit.audit_digest
    assert feature.condition_audit_passed is True
    assert feature.to_dict()["condition_audit_passed"] is True

    with pytest.raises(SignalRuntimeError, match="typed transform-result"):
        FormalFeatureBank(
            receipt=receipts[0],
            identity=identities[0],
            condition_id=control.control_id,
            condition_transform_digest=str(control.transform_digest),
            values=control.feature_matrix,
        )
    with pytest.raises(SignalRuntimeError, match="feature values differ"):
        FormalFeatureBank(
            receipt=receipts[0],
            identity=identities[0],
            condition_id=control.control_id,
            condition_transform_digest=str(control.transform_digest),
            values=np.asarray(control.feature_matrix) + 0.25,
            rf_shuffled_next_result=control,
        )

    other_bank = TransitionBank.from_canonical_batch(receipts[1].batch)
    other_control = RewardFreeShuffledNextSpec(seed=109).apply(other_bank)
    forged_source = replace(
        other_control,
        source_bank_digest=bank.canonical_bank_digest,
        dataset_digest=None,
    )
    with pytest.raises(SignalRuntimeError, match="receipt/base pairing"):
        FormalFeatureBank(
            receipt=receipts[0],
            identity=identities[0],
            condition_id=forged_source.control_id,
            condition_transform_digest=str(forged_source.transform_digest),
            values=forged_source.feature_matrix,
            rf_shuffled_next_result=forged_source,
        )


def test_formal_rf_work_item_requires_frozen_typed_audits_for_every_bank() -> None:
    _canonicalizer, receipts, identities, _features, _represented = _fixture()
    controls = tuple(
        RewardFreeShuffledNextSpec(seed=109).apply(
            TransitionBank.from_canonical_batch(receipt.batch)
        )
        for receipt in receipts
    )
    features = tuple(
        feature_bank_from_rf_shuffled_next(receipt, identity, control)
        for receipt, identity, control in zip(
            receipts, identities, controls, strict=True
        )
    )
    fitted = fit_r1_random_linear(
        RepresentationBatch(
            values=np.concatenate([item.values for item in features[:2]], axis=0),
            dataset_digest=_d("rf-source-fit"),
            role="SOURCE_FIT",
        ),
        output_dim=32,
        seed=0,
    )
    plan = build_signal_matrix_plan()
    registry, protocol = _execution(
        identities, plan, execution_mode="FORMAL"
    )
    represented = transform_feature_banks(fitted, features)
    cell = plan.cell("MECHANISM_STAIRCASE::C_RF_SHUFFLED_NEXT::R1_FIXED_RANDOM_LINEAR")
    work_item = SignalCellWorkItem(
        plan=plan,
        cell=cell,
        source_banks=represented[:2],
        query_banks=represented[2:],
        expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
        identity_registry=registry,
        execution_protocol=protocol,
        evaluation_seed=0,
        execution_mode="FORMAL",
    )
    assert len(str(work_item.work_item_digest)) == 64
    assert all(
        item.feature_bank.condition_audit_passed is True for item in represented
    )
    audit_summary = build_rf_control_audit_summary(
        {work_item.work_key: work_item}
    )
    assert set(audit_summary) == {work_item.work_key}
    public_audit = audit_summary[work_item.work_key]
    assert public_audit["all_marginal_audits_passed"] is True
    assert public_audit["audited_bank_count"] == 4
    assert public_audit["private_bank_membership_withheld"] is True
    assert "bank_id" not in str(dict(public_audit))
    wrong_control = RewardFreeShuffledNextSpec(seed=7).apply(
        TransitionBank.from_canonical_batch(receipts[0].batch)
    )
    wrong_feature = feature_bank_from_rf_shuffled_next(
        receipts[0], identities[0], wrong_control
    )
    wrong_represented = transform_feature_banks(fitted, (wrong_feature,))[0]
    with pytest.raises(SignalAtlasError, match="executable transform"):
        SignalCellWorkItem(
            plan=plan,
            cell=cell,
            source_banks=(wrong_represented, represented[1]),
            query_banks=represented[2:],
            expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
            identity_registry=registry,
            execution_protocol=protocol,
            evaluation_seed=0,
            execution_mode="FORMAL",
        )


def test_pair_control_contract_is_joined_to_exact_feature_banks() -> None:
    _canonicalizer, _receipts, _identities, features, _represented = _fixture()
    left = bank_control_reference_from_feature_bank(features[0])
    right = bank_control_reference_from_feature_bank(features[2])
    contract = ExactRepeatPairContract(
        pair_id="exact-repeat-source-query-a",
        left=left,
        right=right,
        metric_ids=("between_within_margin",),
        statistical_identity="repeat-noise-floor",
        preregistration_digest=_d("repeat-preregistration"),
    )
    evidence = validate_pair_control_feature_banks(
        contract, features[0], features[2]
    )
    assert len(str(evidence.evidence_digest)) == 64
    assert evidence.left_feature_bank_digest == features[0].feature_bank_digest
    assert evidence.right_feature_bank_digest == features[2].feature_bank_digest
    with pytest.raises(SignalRuntimeError, match="preregistered membership"):
        validate_pair_control_feature_banks(contract, features[0], features[3])


def test_represented_bank_rejects_value_tamper_with_stale_output_digest() -> None:
    _canonicalizer, _receipts, _identities, _features, represented = _fixture()
    with pytest.raises(SignalRuntimeError, match="not bound"):
        replace(
            represented[0],
            values=np.asarray(represented[0].values) + 0.25,
            represented_bank_digest=None,
        )


def test_execution_plans_fail_closed_on_representation_schedule_drift() -> None:
    _canonicalizer, _receipts, identities, features, _represented = _fixture()
    plan = build_signal_matrix_plan()
    registry, protocol = _execution(identities, plan)
    source = RepresentationBatch(
        values=np.concatenate([item.values for item in features[:2]], axis=0),
        dataset_digest=_d("plan-drift-source"),
        role="SOURCE_FIT",
    )
    cell_r1 = plan.cell(
        "MECHANISM_STAIRCASE::V_REWARD_FREE_TRANSITION::R1_FIXED_RANDOM_LINEAR"
    )
    wrong_output = transform_feature_banks(
        fit_r1_random_linear(source, output_dim=4, seed=0), features
    )
    with pytest.raises(SignalRuntimeError, match="manifest drifted"):
        run_signal_cell(
            plan=plan,
            cell=cell_r1,
            source_banks=wrong_output[:2],
            query_banks=wrong_output[2:],
            expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
            identity_registry=registry,
            execution_protocol=protocol,
        )

    wrong_hidden = transform_feature_banks(
        fit_r3_matched_random_mlp(
            source, output_dim=32, hidden_dims=(8, 8), seed=0
        ),
        features,
    )
    cell_r3 = plan.cell(
        "MECHANISM_STAIRCASE::V_REWARD_FREE_TRANSITION::R3_MATCHED_RANDOM_MLP"
    )
    with pytest.raises(SignalRuntimeError, match="manifest drifted"):
        run_signal_cell(
            plan=plan,
            cell=cell_r3,
            source_banks=wrong_hidden[:2],
            query_banks=wrong_hidden[2:],
            expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
            identity_registry=registry,
            execution_protocol=protocol,
        )

    request = TrainingRequest(
        representation_id=R5_VIEW_SPECIFIC_CORRO_REFIT,
        input_dim=features[0].values.shape[1],
        output_dim=32,
        hidden_dims=(256, 256),
        activation="relu",
        l2_normalize_output=True,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=0,
    )
    protocol.representation_plan.validate_training_request(request)
    for drifted in (
        replace(request, output_dim=64),
        replace(request, hidden_dims=(128, 128)),
        replace(request, objective_digest=_d("wrong-objective")),
        replace(request, seed=3),
    ):
        with pytest.raises(RepresentationPlanError):
            protocol.representation_plan.validate_training_request(drifted)
    protocol.representation_plan.validate_optimization_config(
        CorroOptimizationConfig()
    )
    with pytest.raises(RepresentationPlanError, match="optimization config drifted"):
        protocol.representation_plan.validate_optimization_config(
            CorroOptimizationConfig(train_steps=1)
        )


def test_condition_plan_rejects_unfrozen_shuffle_seed_before_numeric_work() -> None:
    _canonicalizer, receipts, identities, _features, _represented = _fixture()
    plan = build_signal_matrix_plan()
    registry, protocol = _execution(identities, plan)

    def represented_for_seed(seed: int):
        features = tuple(
            feature_bank_from_transition_view(
                receipt,
                identity,
                apply_transition_view(
                    TransitionBank.from_canonical_batch(receipt.batch),
                    V_SHUFFLED_NEXT,
                    shuffle_seed=seed,
                ),
            )
            for receipt, identity in zip(receipts, identities, strict=True)
        )
        source = RepresentationBatch(
            values=np.concatenate([item.values for item in features[:2]], axis=0),
            dataset_digest=_d(f"shuffle-source:{seed}"),
            role="SOURCE_FIT",
        )
        return transform_feature_banks(fit_r0_identity(source), features)

    cell = plan.cell("CORE_PAIRED::V_SHUFFLED_NEXT::R0_PADDED_RAW")
    wrong = represented_for_seed(0)
    with pytest.raises(SignalRuntimeError, match="condition freeze"):
        run_signal_cell(
            plan=plan,
            cell=cell,
            source_banks=wrong[:2],
            query_banks=wrong[2:],
            expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
            identity_registry=registry,
            execution_protocol=protocol,
        )

    frozen = represented_for_seed(
        protocol.condition_plan.transition_view_seeds[V_SHUFFLED_NEXT]
    )
    run = run_signal_cell(
        plan=plan,
        cell=cell,
        source_banks=frozen[:2],
        query_banks=frozen[2:],
        expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
        identity_registry=registry,
        execution_protocol=protocol,
    )
    assert run.execution_mode == "DEVELOPMENT_SMOKE"


def test_formal_data_fitted_runtime_records_source_fit_and_work_identity() -> None:
    _canonicalizer, _receipts, identities, features, _represented = _fixture()
    source = RepresentationBatch(
        values=np.concatenate([item.values for item in features[:2]], axis=0),
        dataset_digest=_d("formal-r5-source"),
        role="SOURCE_FIT",
    )

    def trainer(values, labels, request):
        del labels
        weight = np.random.default_rng(request.seed).normal(
            size=(request.input_dim, request.output_dim)
        )

        def transform(input_values):
            projected = np.asarray(input_values, dtype=np.float64) @ weight
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            return projected / np.maximum(norms, np.finfo(np.float64).eps)

        return TrainedCallableArtifact(
            checkpoint_bytes=b"formal-r5-fixture",
            parameter_digest=sha256_ndarrays({"weight": weight}),
            trainer_implementation_digest=_d("formal-r5-trainer"),
            transform=transform,
        )

    fitted = fit_r5_corro_style(
        source,
        labels=np.asarray([0] * 4 + [1] * 4, dtype=np.int64),
        trainer=trainer,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=0,
        output_dim=32,
        hidden_dims=(256, 256),
    )
    plan = build_signal_matrix_plan()
    registry, protocol = _execution(
        identities, plan, execution_mode="FORMAL"
    )
    source_fit_digest = _d("formal-r5-source-fit-provenance")
    represented_without_receipt = transform_feature_banks(fitted, features)
    fit_job = next(
        item
        for item in build_optimization_fit_jobs(plan)
        if item.condition_id == V_REWARD_FREE_TRANSITION
        and item.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
        and item.seed == 0
    )
    receipt = FormalTrainedRepresentationReceipt(
        representation_id=fitted.manifest.representation_id,
        representation_coordinate_digest=str(fitted.manifest.coordinate_digest),
        checkpoint_artifact_digest=str(fitted.manifest.checkpoint_digest),
        checkpoint_manifest_digest=_d("formal-r5-checkpoint-manifest"),
        training_request_digest=fitted.manifest.protocol_digest,
        representation_execution_plan_digest=str(
            protocol.representation_plan.plan_digest
        ),
        formal_source_fit_batch_digest=source_fit_digest,
        formal_trainer_contract_digest=_d("formal-r5-trainer-contract"),
        formal_fit_job_digest=str(fit_job.job_digest),
        formal_source_fit_schedule_digest=_d("formal-r5-source-fit-schedule"),
    )
    represented = transform_feature_banks(
        fitted, features, formal_fit_receipt=receipt
    )
    cell = plan.cell(
        "CORE_PAIRED::V_REWARD_FREE_TRANSITION::R5_VIEW_SPECIFIC_CORRO_REFIT"
    )
    common = dict(
        plan=plan,
        cell=cell,
        source_banks=represented[:2],
        query_banks=represented[2:],
        expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
        identity_registry=registry,
        execution_protocol=protocol,
        work_item_digest=_d("formal-r5-work-item"),
    )
    with pytest.raises(SignalRuntimeError, match="source-fit provenance"):
        run_signal_cell(**common)
    without_receipt = {
        **common,
        "source_banks": represented_without_receipt[:2],
        "query_banks": represented_without_receipt[2:],
    }
    with pytest.raises(SignalRuntimeError, match="checkpoint receipt"):
        run_signal_cell(
            **without_receipt,
            source_fit_provenance_digest=source_fit_digest,
        )
    run = run_signal_cell(
        **common,
        source_fit_provenance_digest=source_fit_digest,
    )
    assert run.execution_mode == "FORMAL"
    assert run.work_item_digest == _d("formal-r5-work-item")
    assert run.source_fit_provenance_digest == _d(
        "formal-r5-source-fit-provenance"
    )


def test_formal_atlas_requires_one_complete_fit_schedule_and_source_membership() -> None:
    plan = build_signal_matrix_plan()
    jobs = build_optimization_fit_jobs(plan)
    schedule_digest = _d("atlas-fit-schedule")
    receipts = {
        job.job_id: FormalTrainedRepresentationReceipt(
            representation_id=job.representation_id,
            representation_coordinate_digest=_d(f"coordinate:{job.job_id}"),
            checkpoint_artifact_digest=_d(f"checkpoint:{job.job_id}"),
            checkpoint_manifest_digest=_d(f"manifest:{job.job_id}"),
            training_request_digest=_d(f"request:{job.job_id}"),
            representation_execution_plan_digest=_d("representation-plan"),
            formal_source_fit_batch_digest=_d(f"source-fit:{job.condition_id}"),
            formal_trainer_contract_digest=_d(f"trainer:{job.job_id}"),
            formal_fit_job_digest=str(job.job_digest),
            formal_source_fit_schedule_digest=schedule_digest,
        )
        for job in jobs
    }
    assert validate_formal_atlas_fit_schedule_bindings(
        plan=plan,
        source_membership_digests=(_d("source-membership"),) * len(jobs),
        fit_job_receipts=receipts,
    ) == (schedule_digest, _d("source-membership"))

    with pytest.raises(SignalAtlasError, match="source-row membership"):
        validate_formal_atlas_fit_schedule_bindings(
            plan=plan,
            source_membership_digests=(
                _d("source-membership"),
                _d("different-membership"),
            ),
            fit_job_receipts=receipts,
        )

    drifted = dict(receipts)
    first_job = jobs[0]
    drifted[first_job.job_id] = replace(
        drifted[first_job.job_id],
        formal_source_fit_schedule_digest=_d("different-fit-schedule"),
        receipt_digest=None,
    )
    with pytest.raises(SignalAtlasError, match="one source-fit schedule"):
        validate_formal_atlas_fit_schedule_bindings(
            plan=plan,
            source_membership_digests=(_d("source-membership"),),
            fit_job_receipts=drifted,
        )

    incomplete = dict(receipts)
    incomplete.pop(first_job.job_id)
    with pytest.raises(SignalAtlasError, match="exact 45-fit schedule"):
        validate_formal_atlas_fit_schedule_bindings(
            plan=plan,
            source_membership_digests=(_d("source-membership"),),
            fit_job_receipts=incomplete,
        )
