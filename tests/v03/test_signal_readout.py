from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.dynamics_axis import (
    DynamicsAxisEntry,
    DynamicsAxisRegistry,
    DynamicsPublicQueryJoin,
    build_dynamics_axis_diagnostics,
    dynamics_query_alias_manifest_digest,
)
from policy_learnware_v0.v03.preflight import PublicQueryPlan
from policy_learnware_v0.v03.preoracle_signal import (
    FormalSignalExtractionPlan,
    FormalSignalQueryBankJoin,
)
from policy_learnware_v0.v03.representation_plan import RepresentationExecutionPlan
from policy_learnware_v0.v03.signal_atlas import SignalAtlasRun
from policy_learnware_v0.v03.signal_contrasts import (
    build_signal_contrast_plan,
    evaluate_formal_signal_contrasts,
)
from policy_learnware_v0.v03.signal_controls import HistoricalRandomTanhSpec
from policy_learnware_v0.v03.signal_matrix import build_signal_matrix_plan
from policy_learnware_v0.v03.signal_metrics import SignalDistanceRow, SignalMetricRecord
from policy_learnware_v0.v03.signal_prefix import (
    SignalPrefixExecutionProtocol,
    SignalPrefixPoint,
    SignalPrefixRun,
    SignalPrefixSchedule,
)
from policy_learnware_v0.v03.signal_readout import (
    FormalSignalReadoutBundle,
    FormalSignalReadoutPlan,
    SignalReadoutError,
)
from policy_learnware_v0.v03.signal_runtime import (
    FORMAL_MODE,
    SignalBankIdentity,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
)
from tests.v03.test_signal_contrasts import (
    _formal_authorization,
    _pair_evidence,
    _records,
    _thresholds,
)


def _d(label: str) -> str:
    return sha256_json({"readout-test": label})


def _protocol() -> tuple[
    object,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
]:
    signal_plan = build_signal_matrix_plan()
    historical = HistoricalRandomTanhSpec.create(seed=0, input_dim=3, output_dim=5)
    representation_plan = RepresentationExecutionPlan.create(
        signal_plan=signal_plan, historical_spec=historical
    )
    condition_plan = ConditionExecutionPlan.create(historical_spec=historical)
    identity = SignalBankIdentity(
        receipt_digest=sha256_json({"test": "identity-receipt"}),
        bank_id="source-bank",
        task_private_id="walker-run",
        embodiment_id="walker",
        abi_contract_id="walker-abi",
        goal_contract_id="run",
        dynamics_context_id="nominal",
        context_id="walker-run-nominal",
        measurement_protocol_digest=sha256_json({"test": "measurement"}),
        probe_seed_digest=sha256_json({"test": "probe-seed"}),
        equivalence_class_id="eq-run-nominal",
    )
    registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=sha256_json({"test": "taxonomy"}),
        identities=(identity,),
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
        historical_seed=0,
    )
    return signal_plan, protocol, registry


def _dynamics_and_queries() -> tuple[
    DynamicsAxisRegistry,
    PublicQueryPlan,
    DynamicsPublicQueryJoin,
]:
    entries = []
    for task_index in range(2):
        for axis_index in range(2):
            prefix = f"t{task_index}-a{axis_index}"
            common = (
                f"axis-{axis_index}",
                f"task-{task_index}",
                f"embodiment-{task_index}",
                f"abi-{task_index}",
                f"goal-{task_index}",
            )
            entries.extend(
                (
                    DynamicsAxisEntry(
                        f"{prefix}-anchor-0", *common, 0.0, "ANCHOR"
                    ),
                    DynamicsAxisEntry(
                        f"{prefix}-anchor-1", *common, 1.0, "ANCHOR"
                    ),
                    DynamicsAxisEntry(
                        f"{prefix}-interp", *common, 0.5, "INTERPOLATION"
                    ),
                    DynamicsAxisEntry(
                        f"{prefix}-extra", *common, 2.0, "EXTRAPOLATION"
                    ),
                )
            )
    registry = DynamicsAxisRegistry(tuple(entries))
    regimes = {}
    aliases = {}
    for index in range(66):
        query_id = f"v03q-{index:032x}"
        scope = f"t{index % 2}-a{(index // 2) % 2}"
        if index < 30:
            regimes[query_id] = "EXACT"
            aliases[query_id] = f"{scope}-anchor-{index % 2}"
        elif index < 54:
            regimes[query_id] = "INTERPOLATION"
            aliases[query_id] = f"{scope}-interp"
        else:
            regimes[query_id] = "EXTRAPOLATION"
            aliases[query_id] = f"{scope}-extra"
    query_plan = PublicQueryPlan(
        regime_by_opaque_query_id=regimes,
        query_alias_manifest_digest=dynamics_query_alias_manifest_digest(
            aliases, registry
        ),
    )
    join = DynamicsPublicQueryJoin.bind(
        public_query_plan=query_plan,
        registry=registry,
        dynamics_context_by_opaque_query_id=aliases,
    )
    return registry, query_plan, join


def _dynamics_record(
    *,
    cell_id: str,
    representation_id: str,
    seed: int | None,
    registry: DynamicsAxisRegistry,
    query_join: DynamicsPublicQueryJoin,
    query_bank_id_by_opaque_query_id: dict[str, str],
) -> SignalMetricRecord:
    anchors = tuple(item for item in registry.entries if item.role == "ANCHOR")
    rows = []
    expected = {}
    for query_id, context_id in query_join.dynamics_context_by_opaque_query_id.items():
        query_bank_id = query_bank_id_by_opaque_query_id[query_id]
        query_entry = registry.entry(context_id)
        scope_anchors = tuple(
            item for item in anchors if item.scope_key == query_entry.scope_key
        )
        nearest = min(
            scope_anchors,
            key=lambda item: (
                abs(item.factor_value - query_entry.factor_value),
                item.dynamics_context_id,
            ),
        )
        expected[query_bank_id] = f"source-{nearest.dynamics_context_id}"
        for source_entry in anchors:
            in_scope = source_entry.scope_key == query_entry.scope_key
            distance = (
                abs(source_entry.factor_value - query_entry.factor_value)
                if in_scope
                else 10.0
            )
            rows.append(
                SignalDistanceRow(
                    query_bank_id=query_bank_id,
                    source_bank_id=f"source-{source_entry.dynamics_context_id}",
                    query_receipt_digest=_d(f"{query_id}:receipt"),
                    source_receipt_digest=_d(
                        f"{source_entry.dynamics_context_id}:receipt"
                    ),
                    query_raw_dataset_digest=_d(f"{query_id}:raw"),
                    source_raw_dataset_digest=_d(
                        f"{source_entry.dynamics_context_id}:raw"
                    ),
                    query_task_id=query_entry.task_id,
                    source_task_id=source_entry.task_id,
                    query_context_id=query_entry.dynamics_context_id,
                    source_context_id=source_entry.dynamics_context_id,
                    query_embodiment_id=query_entry.embodiment_id,
                    source_embodiment_id=source_entry.embodiment_id,
                    query_abi_contract_id=query_entry.abi_contract_id,
                    source_abi_contract_id=source_entry.abi_contract_id,
                    query_goal_contract_id=query_entry.goal_contract_id,
                    source_goal_contract_id=source_entry.goal_contract_id,
                    query_dynamics_context_id=query_entry.dynamics_context_id,
                    source_dynamics_context_id=source_entry.dynamics_context_id,
                    query_equivalence_class_id=query_entry.dynamics_context_id,
                    source_equivalence_class_id=source_entry.dynamics_context_id,
                    distance=distance,
                )
            )
    return SignalMetricRecord(
        cell_id=cell_id,
        view_or_condition_id="V_RANDOM_ENCODER",
        representation_id=representation_id,
        representation_coordinate_digest=_d("historical-coordinate"),
        representation_seed=seed,
        source_index_digest=_d("prefix-source-index"),
        query_manifest_digest=_d("prefix-query-manifest"),
        rows=tuple(rows),
        expected_source_by_query=expected,
    )


def _fixture() -> dict[str, object]:
    signal_plan, protocol, identity_registry = _protocol()
    contrast_plan = build_signal_contrast_plan()
    thresholds = _thresholds()
    dynamics_registry, query_plan, query_join = _dynamics_and_queries()
    historical_key = next(
        key
        for key in contrast_plan.expected_numeric_work_keys
        if "R_HIST_RANDOM_TANH" in key
    )
    readout_plan = FormalSignalReadoutPlan.create(
        signal_plan=signal_plan,
        signal_execution_protocol=protocol,
        prefix_work_keys=(historical_key,),
        dynamics_work_keys=(historical_key,),
        review_decisions_digest=thresholds.review_decision_digest,
        prefix_schedule=SignalPrefixSchedule.formal(),
        dynamics_axis_registry=dynamics_registry,
        public_query_plan=query_plan,
        contrast_plan=contrast_plan,
        materiality_thresholds=thresholds,
        attribution_gate_evidence_digest=_d("external-attribution-gate"),
    )
    query_bank_id_by_opaque_query_id = {
        query_id: f"private-query-bank-{index:02d}"
        for index, query_id in enumerate(query_plan.opaque_query_ids)
    }
    query_bank_alias_join = FormalSignalQueryBankJoin.bind(
        readout_plan=readout_plan,
        public_query_plan=query_plan,
        dynamics_public_query_join=query_join,
        selected_work_key=historical_key,
        query_bank_id_by_opaque_query_id=query_bank_id_by_opaque_query_id,
    )
    extraction_plan = FormalSignalExtractionPlan.create(
        readout_plan=readout_plan,
        selected_work_key=historical_key,
        query_bank_alias_join=query_bank_alias_join,
        selection_review_evidence_digest=_d(
            "external-preoracle-signal-selection-review"
        ),
        review_authority_receipt_digest=_d("external-authority"),
    )
    authorization = _formal_authorization(
        thresholds,
        dynamics_registry=dynamics_registry,
        public_query_plan_digest=str(query_plan.plan_digest),
        formal_signal_readout_plan_digest=str(readout_plan.plan_digest),
        preoracle_signal_outcome_plan_digest=str(extraction_plan.plan_digest),
        review_authority_receipt_digest=_d("external-authority"),
    )
    records = _records()
    historical_cell = signal_plan.cell(
        "HISTORICAL_CONTROL::V_RANDOM_ENCODER::R_HIST_RANDOM_TANH"
    )
    records[historical_key] = _dynamics_record(
        cell_id=historical_cell.cell_id,
        representation_id=historical_cell.representation_id,
        seed=0,
        registry=dynamics_registry,
        query_join=query_join,
        query_bank_id_by_opaque_query_id=query_bank_id_by_opaque_query_id,
    )
    contrast_gate = evaluate_formal_signal_contrasts(
        plan=contrast_plan,
        thresholds=thresholds,
        metric_records=records,
        pair_control_evidence_digests=_pair_evidence(),
        formal_atlas_authorization=authorization,
    )

    atlas = object.__new__(SignalAtlasRun)
    object.__setattr__(atlas, "plan", signal_plan)
    object.__setattr__(atlas, "execution_protocol", protocol)
    object.__setattr__(atlas, "identity_registry", identity_registry)
    object.__setattr__(atlas, "formal_authorization", authorization)
    object.__setattr__(
        atlas,
        "cell_runs",
        {key: SimpleNamespace(metric_record=record) for key, record in records.items()},
    )
    object.__setattr__(atlas, "run_digest", _d("atlas-run"))
    atlas_public_body = {
        "schema": "policy-learnware.v03-public-signal-atlas.v0",
        "formal_authorization_digest": authorization.authorization_digest,
        "private_distance_rows_withheld": True,
    }
    object.__setattr__(
        atlas,
        "to_public_dict",
        lambda: {
            **atlas_public_body,
            "public_projection_digest": sha256_json(atlas_public_body),
        },
    )

    prefix_protocol = SignalPrefixExecutionProtocol(
        signal_execution_protocol_digest=str(protocol.protocol_digest),
        signal_execution_mode=FORMAL_MODE,
        plan_digest=str(signal_plan.plan_digest),
        cell_id=historical_cell.cell_id,
        cell_digest=str(historical_cell.cell_digest),
        identity_registry_digest=str(identity_registry.registry_digest),
        measurement_protocol_digest=identity_registry.measurement_protocol_digest,
        source_index_digest=records[historical_key].source_index_digest,
        query_cache_set_digest=_d("query-cache-set"),
        prefix_schedule=SignalPrefixSchedule.formal(),
        block_size=protocol.block_size,
    )
    points = tuple(
        SignalPrefixPoint(
            prefix_episode_count=count,
            query_spec_digests={
                bank_id: _d(f"query-spec:{bank_id}:{count}")
                for bank_id in query_bank_id_by_opaque_query_id.values()
            },
            query_run_digests={
                bank_id: _d(f"query-run:{bank_id}:{count}")
                for bank_id in query_bank_id_by_opaque_query_id.values()
            },
            metric_record=records[historical_key],
        )
        for count in SignalPrefixSchedule.formal().prefix_episode_counts
    )
    prefix_run = SignalPrefixRun(
        execution_protocol=prefix_protocol,
        points=points,
        formal_authorization_digest=str(authorization.authorization_digest),
    )
    dynamics = build_dynamics_axis_diagnostics(
        metric_record=records[historical_key],
        registry=dynamics_registry,
        execution_mode=FORMAL_MODE,
        signal_plan_digest=str(signal_plan.plan_digest),
        signal_execution_protocol_digest=str(protocol.protocol_digest),
        identity_registry_digest=str(identity_registry.registry_digest),
        formal_authorization=authorization,
    )
    bundle = FormalSignalReadoutBundle(
        plan=readout_plan,
        atlas_run=atlas,
        prefix_runs={historical_key: prefix_run},
        dynamics_diagnostics={historical_key: dynamics},
        dynamics_axis_registry=dynamics_registry,
        public_query_plan=query_plan,
        dynamics_public_query_join=query_join,
        contrast_gate_evaluation=contrast_gate,
        pair_control_evidence_digests=_pair_evidence(),
        attribution_gate_evidence_digest=_d("external-attribution-gate"),
    )
    return {
        "plan": readout_plan,
        "bundle": bundle,
        "atlas": atlas,
        "authorization": authorization,
        "historical_key": historical_key,
        "prefix_run": prefix_run,
        "dynamics": dynamics,
        "dynamics_registry": dynamics_registry,
        "query_plan": query_plan,
        "query_join": query_join,
        "query_bank_id_by_opaque_query_id": query_bank_id_by_opaque_query_id,
        "query_bank_alias_join": query_bank_alias_join,
        "extraction_plan": extraction_plan,
        "contrast_gate": contrast_gate,
    }


def test_reviewed_plan_is_nonempty_exact_numeric_79_subset() -> None:
    fixture = _fixture()
    plan = fixture["plan"]
    assert isinstance(plan, FormalSignalReadoutPlan)
    assert plan.prefix_work_keys == (fixture["historical_key"],)
    assert plan.dynamics_work_keys == (fixture["historical_key"],)
    assert plan.plan_digest == sha256_json(plan._payload_without_digest())

    with pytest.raises(SignalReadoutError, match="non-empty"):
        replace(plan, prefix_work_keys=(), plan_digest=None)
    temporal_na = (
        "CORE_PAIRED--V_TEMPORAL_SHUFFLE--R0_PADDED_RAW--seed-NONE"
    )
    with pytest.raises(SignalReadoutError, match="numeric members"):
        replace(plan, dynamics_work_keys=(temporal_na,), plan_digest=None)


def test_bundle_exact_coverage_and_cross_freeze_fail_closed() -> None:
    fixture = _fixture()
    bundle = fixture["bundle"]
    assert isinstance(bundle, FormalSignalReadoutBundle)
    assert bundle.bundle_digest

    with pytest.raises(SignalReadoutError, match="exact reviewed coverage"):
        replace(bundle, prefix_runs={}, bundle_digest=None)
    with pytest.raises(SignalReadoutError, match="exact reviewed coverage"):
        replace(bundle, dynamics_diagnostics={}, bundle_digest=None)

    other_authorization = _formal_authorization(
        _thresholds(),
        dynamics_registry=fixture["dynamics_registry"],
        public_query_plan_digest=str(fixture["query_plan"].plan_digest),
        formal_signal_readout_plan_digest=_d("another-readout-plan"),
    )
    atlas = fixture["atlas"]
    object.__setattr__(atlas, "formal_authorization", other_authorization)
    with pytest.raises(SignalReadoutError, match="readout plan differs"):
        replace(bundle, atlas_run=atlas, bundle_digest=None)


def test_external_attribution_and_pair_evidence_cannot_be_replaced() -> None:
    fixture = _fixture()
    bundle = fixture["bundle"]
    with pytest.raises(SignalReadoutError, match="attribution gate"):
        replace(
            bundle,
            attribution_gate_evidence_digest=_d("forged-attribution"),
            bundle_digest=None,
        )
    forged_pair = dict(_pair_evidence())
    forged_pair["C_EXACT_REPEAT"] = (_d("forged-pair"),)
    with pytest.raises(SignalReadoutError, match="contrast gate"):
        replace(
            bundle,
            pair_control_evidence_digests=forged_pair,
            bundle_digest=None,
        )


def test_prefix_max_point_cannot_switch_raw_bank_membership() -> None:
    fixture = _fixture()
    bundle = fixture["bundle"]
    prefix_run = fixture["prefix_run"]
    points = list(prefix_run.points)
    maximum = points[-1]
    metric = maximum.metric_record
    rows = list(metric.rows)
    rows[0] = replace(
        rows[0], query_raw_dataset_digest=_d("forged-prefix-query-raw")
    )
    forged_metric = replace(metric, rows=tuple(rows))
    points[-1] = replace(
        maximum, metric_record=forged_metric, point_digest=None
    )
    forged_run = replace(prefix_run, points=tuple(points), run_digest=None)
    with pytest.raises(SignalReadoutError, match="prefix run differs"):
        replace(
            bundle,
            prefix_runs={fixture["historical_key"]: forged_run},
            bundle_digest=None,
        )


def test_public_bundle_is_aggregate_only_and_rejects_private_nested_fields() -> None:
    fixture = _fixture()
    bundle = fixture["bundle"]
    public = bundle.to_public_dict()
    serialized = json.dumps(public, sort_keys=True)
    for private_value in (
        "task-0",
        "source-t0-a0-anchor-0",
        "t0-a0-anchor-0",
    ):
        assert private_value not in serialized
    assert public["private_bank_task_context_and_alias_rows_withheld"] is True

    leaked_body = {"rows": [{"query_bank_id": "v03q-" + "0" * 32}]}
    object.__setattr__(
        fixture["atlas"],
        "to_public_dict",
        lambda: {
            **leaked_body,
            "public_projection_digest": sha256_json(leaked_body),
        },
    )
    with pytest.raises(SignalReadoutError, match="leaks private fields"):
        bundle.to_public_dict()
