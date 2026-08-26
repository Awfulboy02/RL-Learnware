from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.dynamics_axis import (
    DynamicsAxisEntry,
    DynamicsAxisRegistry,
)
from policy_learnware_v0.v03.preflight import (
    FORMAL_PRODUCTION_STAGE_IDS,
    HARD_TODO_IDS,
    HardTodoEvidence,
    PreExperimentFreezeManifest,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.representation_plan import RepresentationExecutionPlan
from policy_learnware_v0.v03.signal_atlas import (
    FormalSignalAtlasAuthorization,
    expected_signal_work_keys,
    signal_asymmetric_kme_protocol_digest,
    signal_work_item_graph_digest,
)
from policy_learnware_v0.v03.signal_contrasts import (
    INTERPRETABLE_REPRESENTATION_GAIN_METRIC_IDS,
    SIGNAL_CONTRAST_FAMILIES,
    SignalContrastError,
    SignalMaterialityThresholds,
    build_signal_contrast_plan,
    evaluate_formal_signal_contrasts,
)
from policy_learnware_v0.v03.signal_controls import (
    EXACT_REPEAT_CONTROL_ID,
    SCHEMA_COLLISION_CONTROL_ID,
    HistoricalRandomTanhSpec,
)
from policy_learnware_v0.v03.signal_matrix import C_RF_SHUFFLED_NEXT, build_signal_matrix_plan
from policy_learnware_v0.v03.signal_metrics import SignalDistanceRow, SignalMetricRecord
from policy_learnware_v0.v03.signal_prefix import SignalPrefixSchedule
from policy_learnware_v0.v03.signal_runtime import (
    FORMAL_MODE,
    SignalBankIdentity,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
)
from policy_learnware_v0.v03.transition_views import (
    V_REWARD_FREE_TRANSITION,
    V_SHUFFLED_REWARD,
    V_TEMPORAL_SHUFFLE,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:synthetic"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _dynamics_registry() -> DynamicsAxisRegistry:
    return DynamicsAxisRegistry(
        (
            DynamicsAxisEntry(
                "dynamics-low",
                "mass",
                "walker-run",
                "walker",
                "walker-abi",
                "run",
                0.0,
                "ANCHOR",
            ),
            DynamicsAxisEntry(
                "dynamics-high",
                "mass",
                "walker-run",
                "walker",
                "walker-abi",
                "run",
                1.0,
                "ANCHOR",
            ),
        )
    )


def _formal_authorization(
    thresholds: SignalMaterialityThresholds,
    *,
    freeze_review_digest: str | None = None,
    historical_seed: int = 0,
    dynamics_registry: DynamicsAxisRegistry | None = None,
    public_query_plan_digest: str | None = None,
    formal_signal_readout_plan_digest: str | None = None,
    preoracle_signal_outcome_plan_digest: str | None = None,
    review_authority_receipt_digest: str | None = None,
) -> FormalSignalAtlasAuthorization:
    signal_plan = build_signal_matrix_plan()
    contrast_plan = build_signal_contrast_plan(historical_seed=historical_seed)
    historical = HistoricalRandomTanhSpec.create(
        seed=historical_seed, input_dim=3, output_dim=5
    )
    representation_plan = RepresentationExecutionPlan.create(
        signal_plan=signal_plan, historical_spec=historical
    )
    condition_plan = ConditionExecutionPlan.create(historical_spec=historical)
    identity = SignalBankIdentity(
        receipt_digest=_d("identity-receipt"),
        bank_id="source-bank",
        task_private_id="walker-run",
        embodiment_id="walker",
        abi_contract_id="walker-abi",
        goal_contract_id="run",
        dynamics_context_id="nominal",
        context_id="walker-run-nominal",
        measurement_protocol_digest=_d("measurement"),
        probe_seed_digest=_d("probe-seed"),
        equivalence_class_id="eq-run-nominal",
    )
    registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=_d("taxonomy"), identities=(identity,)
    )
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
        key: _d(f"formal-work:{key}")
        for key in expected_signal_work_keys(signal_plan, protocol)
    }
    dynamics_registry = (
        _dynamics_registry() if dynamics_registry is None else dynamics_registry
    )
    review_digest = (
        thresholds.review_decision_digest
        if freeze_review_digest is None
        else freeze_review_digest
    )
    freeze = PreExperimentFreezeManifest(
        freeze_id="signal-contrast-test",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=review_digest,
        review_authority_receipt_digest=(
            _d("external-authority")
            if review_authority_receipt_digest is None
            else review_authority_receipt_digest
        ),
        review_authority_verified=True,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=str(signal_plan.plan_digest),
        signal_contrast_plan_digest=str(contrast_plan.plan_digest),
        signal_materiality_threshold_digest=str(thresholds.threshold_digest),
        formal_signal_readout_plan_digest=(
            _d("formal-signal-readout-plan")
            if formal_signal_readout_plan_digest is None
            else formal_signal_readout_plan_digest
        ),
        preoracle_signal_outcome_plan_digest=(
            _d("preoracle-signal-outcome-plan")
            if preoracle_signal_outcome_plan_digest is None
            else preoracle_signal_outcome_plan_digest
        ),
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
        public_query_plan_digest=(
            _d("public-query-plan")
            if public_query_plan_digest is None
            else public_query_plan_digest
        ),
        baseline_plan_digest=_d("baseline-plan"),
        statistics_plan_digest=_d("statistics-plan"),
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
                f"contrast-adapter-{index}",
                _d(f"contrast-adapter-contract-{index}"),
            )
            for index, stage_id in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
        },
    )
    return FormalSignalAtlasAuthorization.bind(
        freeze,
        plan=signal_plan,
        execution_protocol=protocol,
        identity_registry=registry,
        dynamics_axis_registry=dynamics_registry,
        work_item_digests=work_digests,
    )


_SOURCES = (
    ("source-run-nominal", "walker-run", "run", "nominal"),
    ("source-run-heavy", "walker-run", "run", "heavy"),
    ("source-walk-nominal", "walker-walk", "walk", "nominal"),
    ("source-walk-heavy", "walker-walk", "walk", "heavy"),
)


def _record(
    *,
    cell_id: str,
    condition_id: str,
    representation_id: str,
    seed: int | None,
) -> SignalMetricRecord:
    if condition_id == C_RF_SHUFFLED_NEXT:
        distances = (
            (0.20, 0.05, 1.00, 1.10),
            (1.00, 1.10, 0.20, 0.05),
        )
    elif condition_id == V_SHUFFLED_REWARD:
        distances = (
            (0.30, 0.40, 0.05, 0.10),
            (0.05, 0.10, 0.30, 0.40),
        )
    else:
        distances = (
            (0.05, 0.20, 1.00, 1.10),
            (1.00, 1.10, 0.05, 0.20),
        )
    rows = []
    query_specs = (
        ("query-run", "walker-run", "run", "nominal"),
        ("query-walk", "walker-walk", "walk", "nominal"),
    )
    for query_index, (query_id, task_id, goal_id, dynamics_id) in enumerate(
        query_specs
    ):
        for source_index, (source_id, source_task, source_goal, source_dynamics) in enumerate(
            _SOURCES
        ):
            rows.append(
                SignalDistanceRow(
                    query_bank_id=query_id,
                    source_bank_id=source_id,
                    query_receipt_digest=_d(f"{query_id}:receipt"),
                    source_receipt_digest=_d(f"{source_id}:receipt"),
                    query_raw_dataset_digest=_d(f"{query_id}:raw"),
                    source_raw_dataset_digest=_d(f"{source_id}:raw"),
                    query_task_id=task_id,
                    source_task_id=source_task,
                    query_context_id=f"{task_id}/{dynamics_id}",
                    source_context_id=f"{source_task}/{source_dynamics}",
                    query_embodiment_id="walker",
                    source_embodiment_id="walker",
                    query_abi_contract_id="walker-abi",
                    source_abi_contract_id="walker-abi",
                    query_goal_contract_id=goal_id,
                    source_goal_contract_id=source_goal,
                    query_dynamics_context_id=dynamics_id,
                    source_dynamics_context_id=source_dynamics,
                    query_equivalence_class_id=f"{task_id}/{dynamics_id}",
                    source_equivalence_class_id=f"{source_task}/{source_dynamics}",
                    distance=distances[query_index][source_index],
                )
            )
    return SignalMetricRecord(
        cell_id=cell_id,
        view_or_condition_id=condition_id,
        representation_id=representation_id,
        representation_coordinate_digest=_d(f"coordinate:{cell_id}:{seed}"),
        representation_seed=seed,
        source_index_digest=_d("shared-source-index"),
        query_manifest_digest=_d("shared-query-manifest"),
        rows=tuple(rows),
        expected_source_by_query={
            "query-run": "source-run-nominal",
            "query-walk": "source-walk-nominal",
        },
    )


def _records(*, historical_seed: int = 0) -> dict[str, SignalMetricRecord]:
    plan = build_signal_contrast_plan(historical_seed=historical_seed)
    signal_plan = build_signal_matrix_plan()
    records = {}
    for work_key in plan.expected_numeric_work_keys:
        cell = next(
            item
            for item in signal_plan.numeric_cells
            if work_key.startswith(item.cell_id.replace("::", "--") + "--seed-")
        )
        seed_token = work_key.rsplit("--seed-", 1)[1]
        seed = None if seed_token == "NONE" else int(seed_token)
        records[work_key] = _record(
            cell_id=cell.cell_id,
            condition_id=cell.condition_id,
            representation_id=cell.representation_id,
            seed=seed,
        )
    return records


def _thresholds(
    *,
    top1: float = 0.75,
    mrr: float = 0.40,
    review: str | None = None,
    historical_seed: int = 0,
) -> SignalMaterialityThresholds:
    plan = build_signal_contrast_plan(historical_seed=historical_seed)
    return SignalMaterialityThresholds(
        contrast_plan_digest=str(plan.plan_digest),
        minimum_transition_degradation_by_metric={
            "dynamics_top1": top1,
            "dynamics_mrr": mrr,
        },
        review_decision_digest=_d("review") if review is None else review,
    )


def _pair_evidence() -> dict[str, tuple[str, ...]]:
    return {
        SCHEMA_COLLISION_CONTROL_ID: (_d("schema-collision-evidence"),),
        EXACT_REPEAT_CONTROL_ID: (_d("exact-repeat-evidence-a"), _d("exact-repeat-evidence-b")),
    }


def test_canonical_plan_covers_five_families_exact_atlas_and_na() -> None:
    plan = build_signal_contrast_plan()
    assert len(plan.expected_numeric_work_keys) == 79
    assert plan.numeric_contrast_count == 74
    assert plan.structural_na_count == 2
    assert {item.family for item in plan.specs} == set(SIGNAL_CONTRAST_FAMILIES)
    assert plan.required_pair_control_ids == (
        SCHEMA_COLLISION_CONTROL_ID,
        EXACT_REPEAT_CONTROL_ID,
    )
    temporal = tuple(
        item for item in plan.specs if item.family == "TEMPORAL_HISTORY"
    )
    assert len(temporal) == 2
    assert all(item.kind == "STRUCTURAL_NA" and not item.metric_ids for item in temporal)
    assert all(V_TEMPORAL_SHUFFLE not in key for key in plan.expected_numeric_work_keys)
    gains = tuple(
        item for item in plan.specs if item.family == "REPRESENTATION_LADDER"
    )
    assert len(gains) == 36
    assert all(item.metric_ids == INTERPRETABLE_REPRESENTATION_GAIN_METRIC_IDS for item in gains)
    assert all(not any("distance" in metric for metric in item.metric_ids) for item in gains)


def test_nonzero_historical_seed_is_bound_to_exact_79_work_schedule() -> None:
    plan = build_signal_contrast_plan(historical_seed=11)
    historical_keys = tuple(
        key for key in plan.expected_numeric_work_keys if "R_HIST_RANDOM_TANH" in key
    )
    assert historical_keys == (
        "HISTORICAL_CONTROL--V_RANDOM_ENCODER--R_HIST_RANDOM_TANH--seed-11",
    )
    assert all(not key.endswith("R_HIST_RANDOM_TANH--seed-0") for key in plan.expected_numeric_work_keys)
    thresholds = _thresholds(historical_seed=11)
    result = evaluate_formal_signal_contrasts(
        plan=plan,
        thresholds=thresholds,
        metric_records=_records(historical_seed=11),
        pair_control_evidence_digests=_pair_evidence(),
        formal_atlas_authorization=_formal_authorization(
            thresholds, historical_seed=11
        ),
    )
    assert result.signal_historical_seed == 11
    assert result.gate_status == "PASS"


def test_plan_and_threshold_tamper_fail_closed() -> None:
    plan = build_signal_contrast_plan()
    with pytest.raises(SignalContrastError, match="canonical plan"):
        replace(plan, specs=plan.specs[:-1], plan_digest=None)
    with pytest.raises(SignalContrastError, match="exactly cover"):
        SignalMaterialityThresholds(
            contrast_plan_digest=str(plan.plan_digest),
            minimum_transition_degradation_by_metric={"dynamics_top1": 0.5},
            review_decision_digest=_d("review"),
        )
    with pytest.raises(SignalContrastError, match=r"\(0, 1\]"):
        _thresholds(top1=0.0)
    thresholds = _thresholds()
    with pytest.raises(SignalContrastError, match="digest mismatch"):
        replace(
            thresholds,
            minimum_transition_degradation_by_metric={
                "dynamics_top1": 0.80,
                "dynamics_mrr": 0.40,
            },
        )


def test_formal_gate_passes_and_excludes_structural_na_from_denominators() -> None:
    thresholds = _thresholds()
    authorization = _formal_authorization(thresholds)
    result = evaluate_formal_signal_contrasts(
        plan=build_signal_contrast_plan(),
        thresholds=thresholds,
        metric_records=_records(),
        pair_control_evidence_digests=_pair_evidence(),
        formal_atlas_authorization=authorization,
    )
    assert result.gate_status == "PASS"
    assert result.transition_mean_degradation_by_metric == {
        "dynamics_mrr": 0.5,
        "dynamics_top1": 1.0,
    }
    assert result.family_numeric_denominators == {
        "REPRESENTATION_LADDER": 36,
        "REWARD_GOAL": 16,
        "SCHEMA": 12,
        "TEMPORAL_HISTORY": 0,
        "TRANSITION_MECHANISM": 10,
    }
    assert result.family_structural_na_counts["TEMPORAL_HISTORY"] == 2
    assert len(result.results) == 76


def test_transition_threshold_failure_is_explicit_no_go() -> None:
    thresholds = _thresholds(mrr=0.75)
    result = evaluate_formal_signal_contrasts(
        plan=build_signal_contrast_plan(),
        thresholds=thresholds,
        metric_records=_records(),
        pair_control_evidence_digests=_pair_evidence(),
        formal_atlas_authorization=_formal_authorization(thresholds),
    )
    assert result.gate_status == "NO_GO_TRANSITION_SIGNAL"
    assert result.transition_threshold_pass_by_metric == {
        "dynamics_mrr": False,
        "dynamics_top1": True,
    }


def test_record_coverage_seed_and_raw_membership_tamper_fail_closed() -> None:
    thresholds = _thresholds()
    authorization = _formal_authorization(thresholds)
    records = _records()
    missing = dict(records)
    missing.pop(next(iter(missing)))
    with pytest.raises(SignalContrastError, match="exactly cover"):
        evaluate_formal_signal_contrasts(
            plan=build_signal_contrast_plan(),
            thresholds=thresholds,
            metric_records=missing,
            pair_control_evidence_digests=_pair_evidence(),
            formal_atlas_authorization=authorization,
        )

    wrong_seed = dict(records)
    seeded_key = next(
        key
        for key, record in wrong_seed.items()
        if record.representation_seed == 0
    )
    wrong_seed[seeded_key] = replace(
        wrong_seed[seeded_key], representation_seed=1
    )
    with pytest.raises(SignalContrastError, match="identity differs"):
        evaluate_formal_signal_contrasts(
            plan=build_signal_contrast_plan(),
            thresholds=thresholds,
            metric_records=wrong_seed,
            pair_control_evidence_digests=_pair_evidence(),
            formal_atlas_authorization=authorization,
        )

    forged_membership = dict(records)
    shuffled_key = next(
        key
        for key, record in forged_membership.items()
        if record.view_or_condition_id == C_RF_SHUFFLED_NEXT
    )
    shuffled = forged_membership[shuffled_key]
    rows = list(shuffled.rows)
    rows[0] = replace(rows[0], source_raw_dataset_digest=_d("forged-raw"))
    forged_membership[shuffled_key] = replace(shuffled, rows=tuple(rows))
    with pytest.raises(SignalContrastError, match="raw bank membership"):
        evaluate_formal_signal_contrasts(
            plan=build_signal_contrast_plan(),
            thresholds=thresholds,
            metric_records=forged_membership,
            pair_control_evidence_digests=_pair_evidence(),
            formal_atlas_authorization=authorization,
        )


def test_threshold_and_review_must_match_external_freeze() -> None:
    frozen = _thresholds()
    authorization = _formal_authorization(frozen)
    changed = _thresholds(mrr=0.45)
    with pytest.raises(SignalContrastError, match="externally reviewed freeze"):
        evaluate_formal_signal_contrasts(
            plan=build_signal_contrast_plan(),
            thresholds=changed,
            metric_records=_records(),
            pair_control_evidence_digests=_pair_evidence(),
            formal_atlas_authorization=authorization,
        )

    wrong_review = _thresholds(review=_d("different-review"))
    wrong_review_authorization = _formal_authorization(
        wrong_review, freeze_review_digest=_d("review")
    )
    with pytest.raises(SignalContrastError, match="externally reviewed freeze"):
        evaluate_formal_signal_contrasts(
            plan=build_signal_contrast_plan(),
            thresholds=wrong_review,
            metric_records=_records(),
            pair_control_evidence_digests=_pair_evidence(),
            formal_atlas_authorization=wrong_review_authorization,
        )
