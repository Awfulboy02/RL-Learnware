from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v03.signal_controls import (
    EXACT_REPEAT_CONTROL_ID,
    HISTORICAL_RANDOM_TANH_ID,
    MATCHED_RANDOM_MLP_ID,
    RF_SHUFFLED_NEXT_CONTROL_ID,
    SCHEMA_COLLISION_CONTROL_ID,
    BankControlReference,
    ExactRepeatDistanceResult,
    ExactRepeatNoiseRatio,
    ExactRepeatPairContract,
    HistoricalRandomTanhSpec,
    PairControlEvaluation,
    PairControlMembershipEvidence,
    PairControlPlan,
    RewardFreeShuffledNextSpec,
    SchemaCollisionPairContract,
    SignalControlError,
    evaluate_pair_control,
    exact_repeat_noise_ratio,
    filter_pair_control_metric_record,
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
from policy_learnware_v0.v03.signal_runtime import SignalCellRun, SourceKernelProtocol
from policy_learnware_v0.v03.transition_views import (
    REGISTERED_VIEW_IDS,
    V_FULL_LEGACY,
    V_RANDOM_ENCODER,
    V_REWARD_FREE_TRANSITION,
    TransitionBank,
    apply_transition_view,
)


def _d(label: str) -> str:
    return sha256_json(
        {"schema": "policy-learnware.v03-signal-control-test.v0", "label": label}
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


def _bank() -> TransitionBank:
    observation = np.asarray(
        [
            [0.0, 0.1, 0.0],
            [0.2, 0.3, 0.0],
            [0.4, 0.5, 0.0],
            [1.0, 1.1, 0.0],
            [1.2, 1.3, 0.0],
            [1.4, 1.5, 0.0],
        ],
        dtype=np.float32,
    )
    dataset = EpisodeDataset(
        observation=observation,
        action=np.asarray(
            [
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
                [-0.1, 0.0],
                [-0.2, 0.0],
                [-0.3, 0.0],
            ],
            dtype=np.float32,
        ),
        reward=np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], dtype=np.float32),
        next_observation=observation
        + np.asarray([0.05, -0.02, 0.0], dtype=np.float32),
        terminated=np.asarray([False, False, False, False, False, False]),
        truncated=np.asarray([False, False, True, False, False, True]),
        episode_offsets=np.asarray([0, 3, 6]),
        reset_seeds=np.asarray([1, 2]),
        probe_seeds=np.asarray([3, 4]),
    )
    return TransitionBank.from_episode_dataset(
        dataset,
        observation_mask=np.asarray([True, True, False]),
        action_mask=np.asarray([True, False]),
    )


def _reference(
    bank_id: str,
    *,
    task: str,
    goal: str,
    context: str,
    embodiment: str | None = None,
    abi: str | None = None,
    dynamics: str | None = None,
    observation_dim: int = 2,
    action_dim: int = 1,
    bank_label: str | None = None,
    seed_label: str | None = None,
) -> BankControlReference:
    label = bank_label or bank_id
    return BankControlReference(
        bank_id=bank_id,
        registered_task_id=task,
        embodiment_id=embodiment or task,
        abi_contract_id=abi or f"abi-{observation_dim}-{action_dim}",
        goal_contract_id=goal,
        dynamics_context_id=dynamics or context,
        context_id=context,
        observation_dim=observation_dim,
        action_dim=action_dim,
        bank_digest=_d(f"bank:{label}"),
        measurement_protocol_digest=_d("measurement"),
        probe_seed_digest=_d(f"probe:{seed_label or bank_id}"),
    )


def _metric_record(
    left: BankControlReference,
    right: BankControlReference,
    *,
    left_receipt_digest: str,
    right_receipt_digest: str,
) -> SignalMetricRecord:
    queries = (
        (left, left_receipt_digest, "source-left"),
        (right, right_receipt_digest, "source-right"),
    )
    sources = (
        ("source-left", left, _d("source-left-receipt")),
        ("source-right", right, _d("source-right-receipt")),
    )
    rows = []
    for query_index, (query, query_receipt, _expected) in enumerate(queries):
        for source_index, (source_id, source, source_receipt) in enumerate(sources):
            rows.append(
                SignalDistanceRow(
                    query_bank_id=query.bank_id,
                    source_bank_id=source_id,
                    query_receipt_digest=query_receipt,
                    source_receipt_digest=source_receipt,
                    query_raw_dataset_digest=query.bank_digest,
                    source_raw_dataset_digest=source.bank_digest,
                    query_task_id=query.registered_task_id,
                    source_task_id=source.registered_task_id,
                    query_context_id=query.context_id,
                    source_context_id=source.context_id,
                    query_embodiment_id=query.embodiment_id,
                    source_embodiment_id=source.embodiment_id,
                    query_abi_contract_id=query.abi_contract_id,
                    source_abi_contract_id=source.abi_contract_id,
                    query_goal_contract_id=query.goal_contract_id,
                    source_goal_contract_id=source.goal_contract_id,
                    query_dynamics_context_id=query.dynamics_context_id,
                    source_dynamics_context_id=source.dynamics_context_id,
                    query_equivalence_class_id=(
                        f"equivalence-{query.registered_task_id}-{query.context_id}"
                    ),
                    source_equivalence_class_id=(
                        f"equivalence-{source.registered_task_id}-{source.context_id}"
                    ),
                    distance=(0.1 if query_index == source_index else 1.5),
                )
            )
    return SignalMetricRecord(
        cell_id="pair-control-r0",
        view_or_condition_id=V_REWARD_FREE_TRANSITION,
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=_d("pair-coordinate"),
        representation_seed=None,
        source_index_digest=_d("pair-source-index"),
        query_manifest_digest=_d("pair-query-manifest"),
        rows=tuple(rows),
        expected_source_by_query={
            left.bank_id: "source-left",
            right.bank_id: "source-right",
        },
    )


def test_historical_random_tanh_has_frozen_identity_and_is_not_r3() -> None:
    bank = _bank()
    input_dim = apply_transition_view(bank, V_FULL_LEGACY).feature_matrix.shape[1]
    first = HistoricalRandomTanhSpec.create(
        seed=29, input_dim=input_dim, output_dim=16
    )
    second = HistoricalRandomTanhSpec.create(
        seed=29, input_dim=input_dim, output_dim=16
    )
    changed = HistoricalRandomTanhSpec.create(
        seed=31, input_dim=input_dim, output_dim=16
    )

    assert first.representation_id == HISTORICAL_RANDOM_TANH_ID
    assert first.representation_id != MATCHED_RANDOM_MLP_ID
    assert not first.is_matched_random_mlp
    assert first.parameter_digest == second.parameter_digest
    assert first.checkpoint_digest == second.checkpoint_digest
    assert first.checkpoint_digest != changed.checkpoint_digest
    assert first.representation_protocol_digest == second.representation_protocol_digest
    assert not first.matrix.flags.writeable and not first.bias.flags.writeable
    first_result = first.apply(bank)
    second_result = second.apply(bank)
    np.testing.assert_array_equal(first_result.values, second_result.values)
    canonical_view = apply_transition_view(
        bank,
        V_RANDOM_ENCODER,
        shuffle_seed=first.seed,
        random_output_dim=first.output_dim,
    )
    assert canonical_view.random_projection_digest == first.parameter_digest
    np.testing.assert_array_equal(
        first_result.values, canonical_view.channels["random_embedding"]
    )
    assert first_result.result_digest == second_result.result_digest
    with pytest.raises(SignalControlError, match="seed recipe"):
        replace(first, matrix=np.zeros_like(first.matrix))


def test_reward_free_shuffled_next_has_same_channels_and_strict_marginals() -> None:
    bank = _bank()
    base = apply_transition_view(bank, V_REWARD_FREE_TRANSITION)
    spec = RewardFreeShuffledNextSpec(seed=7)
    result = spec.apply(bank)

    assert result.control_id == RF_SHUFFLED_NEXT_CONTROL_ID
    assert result.base_view_id == V_REWARD_FREE_TRANSITION
    assert set(result.channels) == set(base.channels) == {
        "observation",
        "action",
        "next_observation",
    }
    assert "reward" not in result.channels
    assert not any("mask" in name for name in result.channels)
    np.testing.assert_array_equal(
        result.channels["observation"], base.channels["observation"]
    )
    np.testing.assert_array_equal(result.channels["action"], base.channels["action"])
    np.testing.assert_array_equal(
        result.channels["next_observation"],
        base.channels["next_observation"][result.next_source_indices],
    )
    assert not np.array_equal(result.next_source_indices, np.arange(6))
    assert result.marginal_audit.passed
    assert (
        result.marginal_audit.base_next_marginal_digest
        == result.marginal_audit.control_next_marginal_digest
    )
    assert result.dataset_digest != spec.transform_digest

    forbidden = dict(result.channels)
    forbidden["reward"] = bank.reward[:, None]
    with pytest.raises(SignalControlError, match="forbidden"):
        replace(result, channels=forbidden)
    with pytest.raises(SignalControlError, match="digest"):
        replace(
            result,
            channels={
                **dict(result.channels),
                "next_observation": np.roll(
                    result.channels["next_observation"], 1, axis=0
                ),
            },
        )


def test_reward_free_shuffled_next_rejects_observationally_null_control() -> None:
    bank = _bank()
    repeated = TransitionBank(
        observation=bank.observation,
        action=bank.action,
        reward=bank.reward,
        next_observation=np.zeros_like(bank.next_observation),
        terminated=bank.terminated,
        truncated=bank.truncated,
        episode_offsets=bank.episode_offsets,
        observation_mask=bank.observation_mask,
        action_mask=bank.action_mask,
    )
    with pytest.raises(SignalControlError, match="cannot destroy pairing"):
        RewardFreeShuffledNextSpec(seed=7).apply(repeated)


def test_schema_collision_and_exact_repeat_are_digest_bound_pair_controls() -> None:
    view_count_before = len(REGISTERED_VIEW_IDS)
    left = _reference(
        "bank-walker-walk", task="walker", goal="walk", context="nominal"
    )
    right = _reference(
        "bank-finger-turn", task="finger", goal="turn", context="nominal"
    )
    collision = SchemaCollisionPairContract(
        pair_id="pair-schema-001",
        left=left,
        right=right,
        metric_ids=("task_top1", "between_within"),
        statistical_identity="schema_collision_primary",
        preregistration_digest=_d("preregister:schema"),
    )
    assert collision.control_id == SCHEMA_COLLISION_CONTROL_ID
    assert not collision.adds_input_view
    assert collision.left.bank_reference_digest != collision.right.bank_reference_digest
    assert collision.metric_protocol_digest != collision.pair_digest
    with pytest.raises(SignalControlError, match="measurement protocol"):
        SchemaCollisionPairContract(
            pair_id="pair-schema-cross-measurement",
            left=left,
            right=replace(
                right, measurement_protocol_digest=_d("other-measurement")
            ),
            metric_ids=("task_top1",),
            statistical_identity="schema_collision_primary",
            preregistration_digest=_d("preregister:schema-other-measurement"),
        )

    repeat_left = _reference(
        "bank-repeat-a",
        task="walker",
        goal="walk",
        context="damping-100",
        bank_label="repeat-a",
        seed_label="seed-a",
    )
    repeat_right = _reference(
        "bank-repeat-b",
        task="walker",
        goal="walk",
        context="damping-100",
        bank_label="repeat-b",
        seed_label="seed-b",
    )
    repeat = ExactRepeatPairContract(
        pair_id="pair-repeat-001",
        left=repeat_left,
        right=repeat_right,
        metric_ids=("mmd_noise_floor",),
        statistical_identity="exact_repeat_noise_floor",
        preregistration_digest=_d("preregister:repeat"),
    )
    assert repeat.control_id == EXACT_REPEAT_CONTROL_ID
    assert not repeat.adds_input_view
    assert repeat.pair_digest != collision.pair_digest
    assert len(REGISTERED_VIEW_IDS) == view_count_before == 14
    assert RF_SHUFFLED_NEXT_CONTROL_ID not in REGISTERED_VIEW_IDS
    assert SCHEMA_COLLISION_CONTROL_ID not in REGISTERED_VIEW_IDS
    assert EXACT_REPEAT_CONTROL_ID not in REGISTERED_VIEW_IDS


def test_pair_controls_fail_closed_on_wrong_eligibility_or_tamper() -> None:
    left = _reference("left", task="walker", goal="walk", context="ctx")
    wrong_width = _reference(
        "right-wide",
        task="finger",
        goal="turn",
        context="ctx",
        observation_dim=3,
    )
    with pytest.raises(SignalControlError, match="native dimensions"):
        SchemaCollisionPairContract(
            pair_id="bad-schema",
            left=left,
            right=wrong_width,
            metric_ids=("top1",),
            statistical_identity="schema_primary",
            preregistration_digest=_d("pre"),
        )

    same_seed = _reference(
        "repeat-b",
        task="walker",
        goal="walk",
        context="ctx",
        bank_label="repeat-b",
        seed_label="left",
    )
    left_same_seed = replace(left, probe_seed_digest=_d("probe:left"))
    with pytest.raises(SignalControlError, match="independent probe seeds"):
        ExactRepeatPairContract(
            pair_id="bad-repeat",
            left=left_same_seed,
            right=same_seed,
            metric_ids=("noise",),
            statistical_identity="repeat_primary",
            preregistration_digest=_d("pre-repeat"),
        )

    valid_right = replace(same_seed, probe_seed_digest=_d("probe:right"))
    valid = ExactRepeatPairContract(
        pair_id="valid-repeat",
        left=left_same_seed,
        right=valid_right,
        metric_ids=("noise",),
        statistical_identity="repeat_primary",
        preregistration_digest=_d("pre-repeat"),
    )
    for changed in (
        replace(valid_right, embodiment_id="other-embodiment"),
        replace(valid_right, abi_contract_id="other-abi"),
        replace(valid_right, dynamics_context_id="other-dynamics"),
    ):
        with pytest.raises(SignalControlError, match="must share"):
            ExactRepeatPairContract(
                pair_id="wrong-structural-repeat",
                left=left_same_seed,
                right=changed,
                metric_ids=("noise",),
                statistical_identity="repeat_primary",
                preregistration_digest=_d("pre-repeat-structural"),
            )
    with pytest.raises(SignalControlError, match="pair digest"):
        replace(valid, pair_digest=_d("tampered-pair"))


def test_pair_control_evaluator_binds_contract_membership_and_metrics() -> None:
    left = _reference(
        "schema-left", task="walker", goal="walk", context="nominal"
    )
    right = _reference(
        "schema-right", task="finger", goal="turn", context="nominal"
    )
    contract = SchemaCollisionPairContract(
        pair_id="schema-result",
        left=left,
        right=right,
        metric_ids=("task_top1", "between_within_ratio"),
        statistical_identity="schema_collision_primary",
        preregistration_digest=_d("schema-result-preregistration"),
    )
    left_receipt = _d("schema-left-receipt")
    right_receipt = _d("schema-right-receipt")
    record = _metric_record(
        left,
        right,
        left_receipt_digest=left_receipt,
        right_receipt_digest=right_receipt,
    )
    membership = PairControlMembershipEvidence.create(
        contract,
        left_receipt_digest=left_receipt,
        right_receipt_digest=right_receipt,
        left_feature_bank_digest=_d("schema-left-feature"),
        right_feature_bank_digest=_d("schema-right-feature"),
    )

    first = evaluate_pair_control(contract, record, membership)
    second = evaluate_pair_control(contract, record, membership)
    assert first.result_digest == second.result_digest
    assert first.pair_digest == contract.pair_digest
    assert first.preregistration_digest == contract.preregistration_digest
    assert first.metric_protocol_digest == contract.metric_protocol_digest
    assert first.source_metric_record_digest == record.record_digest
    assert first.metric_record_digest == first.pair_metric_record_digest
    assert first.pair_metric_record_digest != record.record_digest
    assert first.membership_evidence_digest == membership.evidence_digest
    assert tuple(first.metric_values) == contract.metric_ids
    assert first.pair_query_count == 2
    assert first.pair_source_count == 2
    assert not first.adds_input_view
    public = first.to_public_dict()
    assert "pair_id" not in public
    assert "rows" not in public
    assert public["private_pair_membership_withheld"] is True
    assert json.dumps(first.to_dict(), sort_keys=True)
    with pytest.raises(SignalControlError, match="result digest"):
        replace(first, result_digest=_d("tampered-result"))

    pair_record, pair_filter_digest = filter_pair_control_metric_record(
        contract, record, membership
    )
    evaluation = PairControlEvaluation.evaluate(contract, record, membership)
    assert evaluation.pair_metric_record.record_digest == pair_record.record_digest
    assert evaluation.result.pair_filter_digest == pair_filter_digest
    assert evaluation.result.to_dict() == first.to_dict()


def test_exact_repeat_uses_direct_mmd_and_typed_between_noise_ratio() -> None:
    left = _reference(
        "repeat-query",
        task="walker",
        goal="walk",
        context="damping-100",
        seed_label="repeat-query",
    )
    right = _reference(
        "repeat-source",
        task="walker",
        goal="walk",
        context="damping-100",
        seed_label="repeat-source",
    )
    far = _reference(
        "far-source",
        task="walker",
        goal="walk",
        context="damping-150",
        seed_label="far-source",
    )
    contract = ExactRepeatPairContract(
        pair_id="repeat-noise-floor",
        left=left,
        right=right,
        metric_ids=("direct_repeat_mmd",),
        statistical_identity="exact_repeat_noise_floor",
        preregistration_digest=_d("repeat-noise-preregistration"),
    )
    left_receipt = _d("repeat-query-receipt")
    right_receipt = _d("repeat-source-receipt")
    other_receipt = _d("other-query-receipt")
    far_receipt = _d("far-source-receipt")

    def row(
        query: BankControlReference,
        source: BankControlReference,
        *,
        query_receipt: str,
        source_receipt: str,
        distance: float,
    ) -> SignalDistanceRow:
        return SignalDistanceRow(
            query_bank_id=query.bank_id,
            source_bank_id=source.bank_id,
            query_receipt_digest=query_receipt,
            source_receipt_digest=source_receipt,
            query_raw_dataset_digest=query.bank_digest,
            source_raw_dataset_digest=source.bank_digest,
            query_task_id=query.registered_task_id,
            source_task_id=source.registered_task_id,
            query_context_id=query.context_id,
            source_context_id=source.context_id,
            query_embodiment_id=query.embodiment_id,
            source_embodiment_id=source.embodiment_id,
            query_abi_contract_id=query.abi_contract_id,
            source_abi_contract_id=source.abi_contract_id,
            query_goal_contract_id=query.goal_contract_id,
            source_goal_contract_id=source.goal_contract_id,
            query_dynamics_context_id=query.dynamics_context_id,
            source_dynamics_context_id=source.dynamics_context_id,
            query_equivalence_class_id=f"eq-{query.context_id}",
            source_equivalence_class_id=f"eq-{source.context_id}",
            distance=distance,
        )

    other_query = replace(
        far,
        bank_id="far-query",
        bank_digest=_d("far-query-bank"),
        probe_seed_digest=_d("far-query-probe"),
    )
    rows = (
        row(
            left,
            right,
            query_receipt=left_receipt,
            source_receipt=right_receipt,
            distance=0.2,
        ),
        row(
            left,
            far,
            query_receipt=left_receipt,
            source_receipt=far_receipt,
            distance=2.0,
        ),
        row(
            other_query,
            right,
            query_receipt=other_receipt,
            source_receipt=right_receipt,
            distance=2.0,
        ),
        row(
            other_query,
            far,
            query_receipt=other_receipt,
            source_receipt=far_receipt,
            distance=0.25,
        ),
    )
    record = SignalMetricRecord(
        cell_id="repeat-r0",
        view_or_condition_id=V_REWARD_FREE_TRANSITION,
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=_d("repeat-coordinate"),
        representation_seed=None,
        source_index_digest=_d("repeat-source-index"),
        query_manifest_digest=_d("repeat-query-manifest"),
        rows=rows,
        expected_source_by_query={
            left.bank_id: right.bank_id,
            other_query.bank_id: far.bank_id,
        },
    )
    membership = PairControlMembershipEvidence.create(
        contract,
        left_receipt_digest=left_receipt,
        right_receipt_digest=right_receipt,
        left_feature_bank_digest=_d("repeat-query-feature"),
        right_feature_bank_digest=_d("repeat-source-feature"),
    )
    execution_protocol_digest = _d("repeat-execution-protocol")
    kernel = SourceKernelProtocol(
        representation_coordinate_digest=record.representation_coordinate_digest,
        source_represented_bank_digests=(
            _d("repeat-represented-source"),
            _d("far-represented-source"),
        ),
        measurement_protocol_digest=_d("measurement"),
        execution_protocol_digest=execution_protocol_digest,
        bandwidth=1.0,
        pair_budget=64,
        seed=0,
    )
    run = SignalCellRun(
        plan_digest=_d("repeat-plan"),
        cell_id=record.cell_id,
        cell_digest=_d("repeat-cell"),
        execution_protocol_digest=execution_protocol_digest,
        execution_mode="DEVELOPMENT_SMOKE",
        source_fit_provenance_digest=None,
        work_item_digest=None,
        evaluation_seed=None,
        kernel_protocol=kernel,
        source_index_digest=record.source_index_digest,
        query_run_digests={
            left.bank_id: _d("repeat-query-run"),
            other_query.bank_id: _d("other-query-run"),
        },
        metric_record=record,
        diagnostics=_diagnostics(record),
    )
    direct = ExactRepeatDistanceResult.evaluate(
        contract,
        membership,
        run,
    )
    assert direct.metric_id == "direct_repeat_mmd"
    assert direct.distance == 0.2
    assert direct.direct_row.query_bank_id == left.bank_id
    assert direct.direct_row.source_bank_id == right.bank_id
    public = direct.to_public_dict()
    assert public["distance"] == 0.2
    assert "pair_id" not in public and "direct_row" not in public

    ratio = exact_repeat_noise_ratio(direct)
    assert isinstance(ratio, ExactRepeatNoiseRatio)
    assert ratio.between_distance == 2.0
    assert ratio.repeat_distance == 0.2
    assert ratio.ratio == 10.0

    # A changed collection/context label with the *same* dynamics identity is
    # not a dynamics-axis numerator.  Retag every row for that source so the
    # matrix remains structurally valid, then require a typed N/A/failure.
    context_only_rows = tuple(
        replace(row_item, source_dynamics_context_id=left.dynamics_context_id)
        if row_item.source_bank_id == far.bank_id
        else row_item
        for row_item in record.rows
    )
    context_only_record = replace(
        record,
        rows=context_only_rows,
        metric_values=None,
    )
    context_only_run = replace(
        run,
        metric_record=context_only_record,
        diagnostics=_diagnostics(context_only_record),
        run_digest=None,
    )
    context_only_direct = ExactRepeatDistanceResult.evaluate(
        contract, membership, context_only_run
    )
    with pytest.raises(SignalControlError, match="different-dynamics rows"):
        exact_repeat_noise_ratio(context_only_direct)

    # The direct-row join checks the explicit dynamics identity, not merely
    # bank IDs/receipts.  Retagging the repeated source therefore fails closed.
    wrong_direct_rows = tuple(
        replace(row_item, source_dynamics_context_id="forged-dynamics")
        if row_item.source_bank_id == right.bank_id
        else row_item
        for row_item in record.rows
    )
    wrong_direct_record = replace(
        record,
        rows=wrong_direct_rows,
        metric_values=None,
    )
    wrong_direct_run = replace(
        run,
        metric_record=wrong_direct_record,
        diagnostics=_diagnostics(wrong_direct_record),
        run_digest=None,
    )
    with pytest.raises(SignalControlError, match="row identity"):
        ExactRepeatDistanceResult.evaluate(contract, membership, wrong_direct_run)
    with pytest.raises(SignalControlError, match="direct repeated-bank"):
        evaluate_pair_control(contract, record, membership)


def test_pair_control_evaluator_fails_closed_on_missing_metric_or_membership() -> None:
    left = _reference(
        "schema-left", task="walker", goal="walk", context="nominal"
    )
    right = _reference(
        "schema-right", task="finger", goal="turn", context="nominal"
    )
    left_receipt = _d("strict-left-receipt")
    right_receipt = _d("strict-right-receipt")
    record = _metric_record(
        left,
        right,
        left_receipt_digest=left_receipt,
        right_receipt_digest=right_receipt,
    )
    missing_metric = SchemaCollisionPairContract(
        pair_id="schema-missing-metric",
        left=left,
        right=right,
        metric_ids=("schema_pair_accuracy",),
        statistical_identity="schema_collision_primary",
        preregistration_digest=_d("strict-preregistration"),
    )
    membership = PairControlMembershipEvidence.create(
        missing_metric,
        left_receipt_digest=left_receipt,
        right_receipt_digest=right_receipt,
        left_feature_bank_digest=_d("strict-left-feature"),
        right_feature_bank_digest=_d("strict-right-feature"),
    )
    with pytest.raises(SignalControlError, match="missing preregistered metrics"):
        evaluate_pair_control(missing_metric, record, membership)

    valid = replace(
        missing_metric,
        metric_ids=("task_top1",),
        pair_digest=None,
    )
    valid_membership = PairControlMembershipEvidence.create(
        valid,
        left_receipt_digest=left_receipt,
        right_receipt_digest=right_receipt,
        left_feature_bank_digest=_d("strict-left-feature"),
        right_feature_bank_digest=_d("strict-right-feature"),
    )
    wrong_bank = replace(
        valid_membership,
        left_bank_id="other-bank",
        evidence_digest=None,
    )
    with pytest.raises(SignalControlError, match="membership"):
        evaluate_pair_control(valid, record, wrong_bank)

    wrong_receipt = replace(
        valid_membership,
        left_receipt_digest=_d("wrong-left-receipt"),
        evidence_digest=None,
    )
    with pytest.raises(SignalControlError, match="canonical receipt"):
        evaluate_pair_control(valid, record, wrong_receipt)

    wrong_protocol = replace(
        valid_membership,
        measurement_protocol_digest=_d("wrong-measurement-protocol"),
        evidence_digest=None,
    )
    with pytest.raises(SignalControlError, match="measurement protocol"):
        evaluate_pair_control(valid, record, wrong_protocol)

    absent_record = _metric_record(
        _reference("other-left", task="walker", goal="walk", context="nominal"),
        _reference("other-right", task="finger", goal="turn", context="nominal"),
        left_receipt_digest=_d("other-left-receipt"),
        right_receipt_digest=_d("other-right-receipt"),
    )
    with pytest.raises(SignalControlError, match="absent from metric record"):
        evaluate_pair_control(valid, absent_record, valid_membership)


def test_pair_control_metrics_are_recomputed_from_only_the_preregistered_queries() -> None:
    pair_a = (
        _reference("pair-a-left", task="walker", goal="walk", context="nominal"),
        _reference("pair-a-right", task="finger", goal="turn", context="nominal"),
    )
    pair_b = (
        _reference("pair-b-left", task="cheetah", goal="run", context="nominal"),
        _reference("pair-b-right", task="hopper", goal="hop", context="nominal"),
    )
    sources = (
        ("source-walker", pair_a[0]),
        ("source-finger", pair_a[1]),
        ("source-cheetah", pair_b[0]),
        ("source-hopper", pair_b[1]),
    )
    query_receipts = {
        item.bank_id: _d(f"receipt:{item.bank_id}")
        for item in (*pair_a, *pair_b)
    }
    expected_sources = {
        pair_a[0].bank_id: "source-walker",
        pair_a[1].bank_id: "source-finger",
        pair_b[0].bank_id: "source-cheetah",
        pair_b[1].bank_id: "source-hopper",
    }
    rows = []
    for query in (*pair_a, *pair_b):
        for source_id, source in sources:
            # Pair A retrieves the correct task; pair B deliberately retrieves
            # the first (wrong) task so the full-matrix mean is 0.5.
            if query in pair_a:
                distance = 0.05 if source_id == expected_sources[query.bank_id] else 2.0
            else:
                distance = 0.05 if source_id == "source-walker" else 2.0
            rows.append(
                SignalDistanceRow(
                    query_bank_id=query.bank_id,
                    source_bank_id=source_id,
                    query_receipt_digest=query_receipts[query.bank_id],
                    source_receipt_digest=_d(f"source-receipt:{source_id}"),
                    query_raw_dataset_digest=query.bank_digest,
                    source_raw_dataset_digest=source.bank_digest,
                    query_task_id=query.registered_task_id,
                    source_task_id=source.registered_task_id,
                    query_context_id=query.context_id,
                    source_context_id=source.context_id,
                    query_embodiment_id=query.registered_task_id,
                    source_embodiment_id=source.registered_task_id,
                    query_abi_contract_id="abi-2-1",
                    source_abi_contract_id="abi-2-1",
                    query_goal_contract_id=query.goal_contract_id,
                    source_goal_contract_id=source.goal_contract_id,
                    query_dynamics_context_id=query.context_id,
                    source_dynamics_context_id=source.context_id,
                    query_equivalence_class_id=f"eq-{query.registered_task_id}",
                    source_equivalence_class_id=f"eq-{source.registered_task_id}",
                    distance=distance,
                )
            )
    full = SignalMetricRecord(
        cell_id="full-r0",
        view_or_condition_id=V_REWARD_FREE_TRANSITION,
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=_d("four-query-coordinate"),
        representation_seed=None,
        source_index_digest=_d("four-query-source-index"),
        query_manifest_digest=_d("four-query-manifest"),
        rows=tuple(rows),
        expected_source_by_query=expected_sources,
    )
    contract_a = SchemaCollisionPairContract(
        pair_id="pair-a",
        left=pair_a[0],
        right=pair_a[1],
        metric_ids=("task_top1",),
        statistical_identity="schema_pair_a",
        preregistration_digest=_d("pair-a-preregistration"),
    )
    membership_a = PairControlMembershipEvidence.create(
        contract_a,
        left_receipt_digest=query_receipts[pair_a[0].bank_id],
        right_receipt_digest=query_receipts[pair_a[1].bank_id],
        left_feature_bank_digest=_d("pair-a-left-feature"),
        right_feature_bank_digest=_d("pair-a-right-feature"),
    )
    contract_b = SchemaCollisionPairContract(
        pair_id="pair-b",
        left=pair_b[0],
        right=pair_b[1],
        metric_ids=("task_top1",),
        statistical_identity="schema_pair_b",
        preregistration_digest=_d("pair-b-preregistration"),
    )
    membership_b = PairControlMembershipEvidence.create(
        contract_b,
        left_receipt_digest=query_receipts[pair_b[0].bank_id],
        right_receipt_digest=query_receipts[pair_b[1].bank_id],
        left_feature_bank_digest=_d("pair-b-left-feature"),
        right_feature_bank_digest=_d("pair-b-right-feature"),
    )

    assert full.metric_values["task_top1"] == 0.5
    result_a = evaluate_pair_control(contract_a, full, membership_a)
    result_b = evaluate_pair_control(contract_b, full, membership_b)
    assert result_a.metric_values["task_top1"] == 1.0
    assert result_b.metric_values["task_top1"] == 0.0
    assert result_a.pair_metric_record_digest != result_b.pair_metric_record_digest
    assert result_a.result_digest != result_b.result_digest
    assert result_a.source_metric_record_digest == result_b.source_metric_record_digest


def test_pair_control_plan_is_private_digest_bound_and_requires_both_controls() -> None:
    schema_left = _reference("schema-left", task="walker", goal="walk", context="ctx")
    schema_right = _reference("schema-right", task="finger", goal="turn", context="ctx")
    schema = SchemaCollisionPairContract(
        pair_id="schema-plan",
        left=schema_left,
        right=schema_right,
        metric_ids=("task_top1",),
        statistical_identity="schema_primary",
        preregistration_digest=_d("schema-plan-prereg"),
    )
    repeat_right = _reference(
        "repeat-right",
        task="walker",
        goal="walk",
        context="ctx",
        seed_label="repeat-independent",
    )
    repeat = ExactRepeatPairContract(
        pair_id="repeat-plan",
        left=schema_left,
        right=repeat_right,
        metric_ids=("exact_source_top1",),
        statistical_identity="repeat_primary",
        preregistration_digest=_d("repeat-plan-prereg"),
    )
    with pytest.raises(SignalControlError, match="requires schema-collision"):
        PairControlPlan((schema,))
    plan = PairControlPlan((repeat, schema))
    assert plan.contract(str(schema.pair_digest)).pair_id == schema.pair_id
    assert plan.measurement_protocol_digest == schema_left.measurement_protocol_digest
    assert plan.plan_digest == PairControlPlan((schema, repeat)).plan_digest
    private = plan.to_dict()
    assert private["contracts"][0]["left_bank_reference"]["bank_id"]
