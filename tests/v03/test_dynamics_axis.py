from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.dynamics_axis import (
    DynamicsAxisEntry,
    DynamicsAxisError,
    DynamicsAxisRegistry,
    DynamicsPublicQueryJoin,
    build_dynamics_axis_diagnostics,
    dynamics_query_alias_manifest_digest,
)
from policy_learnware_v0.v03.preflight import PublicQueryPlan
from policy_learnware_v0.v03.signal_metrics import (
    SignalDistanceRow,
    SignalMetricRecord,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _entry(
    context: str,
    factor: float,
    role: str,
    *,
    task: str = "walker-walk",
) -> DynamicsAxisEntry:
    return DynamicsAxisEntry(
        dynamics_context_id=context,
        axis_id=f"mass-axis:{task}",
        task_id=task,
        embodiment_id="walker" if task.startswith("walker") else "finger",
        abi_contract_id=f"abi:{task}",
        goal_contract_id=f"goal:{task}",
        factor_value=factor,
        role=role,  # type: ignore[arg-type]
    )


def _registry() -> DynamicsAxisRegistry:
    return DynamicsAxisRegistry(
        entries=(
            _entry("walker/mass/0", 0.0, "ANCHOR"),
            _entry("walker/mass/10", 10.0, "ANCHOR"),
            _entry("walker/mass/4", 4.0, "INTERPOLATION"),
            _entry("walker/mass/15", 15.0, "EXTRAPOLATION"),
            _entry("finger/mass/0", 0.0, "ANCHOR", task="finger-spin"),
            _entry("finger/mass/10", 10.0, "ANCHOR", task="finger-spin"),
        )
    )


def _row(
    *,
    query_id: str,
    query_context: str,
    query_role_task: str,
    source_id: str,
    source_context: str,
    source_task: str,
    distance: float,
) -> SignalDistanceRow:
    query_emb = "walker" if query_role_task.startswith("walker") else "finger"
    source_emb = "walker" if source_task.startswith("walker") else "finger"
    return SignalDistanceRow(
        query_bank_id=query_id,
        source_bank_id=source_id,
        query_receipt_digest=_d(f"receipt:{query_id}"),
        source_receipt_digest=_d(f"receipt:{source_id}"),
        query_raw_dataset_digest=_d(f"raw:{query_id}"),
        source_raw_dataset_digest=_d(f"raw:{source_id}"),
        query_task_id=query_role_task,
        source_task_id=source_task,
        query_context_id=query_context,
        source_context_id=source_context,
        query_embodiment_id=query_emb,
        source_embodiment_id=source_emb,
        query_abi_contract_id=f"abi:{query_role_task}",
        source_abi_contract_id=f"abi:{source_task}",
        query_goal_contract_id=f"goal:{query_role_task}",
        source_goal_contract_id=f"goal:{source_task}",
        query_dynamics_context_id=query_context,
        source_dynamics_context_id=source_context,
        query_equivalence_class_id=None,
        source_equivalence_class_id=None,
        distance=distance,
    )


def _metric() -> SignalMetricRecord:
    sources = (
        ("walker-0", "walker/mass/0", "walker-walk"),
        ("walker-10", "walker/mass/10", "walker-walk"),
        ("finger-0", "finger/mass/0", "finger-spin"),
        ("finger-10", "finger/mass/10", "finger-spin"),
    )
    query_specs = (
        ("query-interp", "walker/mass/4", {"walker-0": 0.4, "walker-10": 0.6, "finger-0": 0.01, "finger-10": 1.0}),
        ("query-extra", "walker/mass/15", {"walker-0": 0.7, "walker-10": 0.3, "finger-0": 0.02, "finger-10": 1.1}),
    )
    rows = tuple(
        _row(
            query_id=query_id,
            query_context=query_context,
            query_role_task="walker-walk",
            source_id=source_id,
            source_context=source_context,
            source_task=source_task,
            distance=distances[source_id],
        )
        for query_id, query_context, distances in query_specs
        for source_id, source_context, source_task in sources
    )
    return SignalMetricRecord(
        cell_id="CORE_PAIRED::V_REWARD_FREE_TRANSITION::R0_PADDED_RAW",
        view_or_condition_id="V_REWARD_FREE_TRANSITION",
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=_d("coordinate"),
        representation_seed=None,
        source_index_digest=_d("source-index"),
        query_manifest_digest=_d("queries"),
        rows=rows,
        # The globally closest source is deliberately cross-task.  Numeric
        # dynamics diagnostics must ignore that schema/task shortcut.
        expected_source_by_query={
            "query-interp": "finger-0",
            "query-extra": "finger-0",
        },
    )


def test_registry_freezes_roles_against_anchor_range() -> None:
    registry = _registry()
    assert registry.entry("walker/mass/4").factor_value == 4.0
    assert len(registry.registry_digest or "") == 64
    bad = replace(
        registry.entry("walker/mass/4"), factor_value=12.0
    )
    with pytest.raises(DynamicsAxisError, match="strictly inside"):
        DynamicsAxisRegistry(
            tuple(
                bad if item.dynamics_context_id == bad.dynamics_context_id else item
                for item in registry.entries
            )
        )


def test_dynamics_diagnostics_are_scope_conditional_and_numeric() -> None:
    metric = _metric()
    diagnostics = build_dynamics_axis_diagnostics(
        metric_record=metric,
        registry=_registry(),
        execution_mode="DEVELOPMENT_SMOKE",
        signal_plan_digest=_d("plan"),
        signal_execution_protocol_digest=_d("execution"),
        identity_registry_digest=_d("identity-registry"),
    )
    values = diagnostics.metric_values
    assert metric.metric_values["task_top1"] == 0.0
    assert values["all.neighborhood_top1"] == 1.0
    assert values["all.factor_mae"] == 4.5
    assert values["all.order_accuracy"] == 1.0
    assert values["interpolation.query_count"] == 1.0
    assert values["extrapolation.query_count"] == 1.0
    assert values["anchor.query_count"] == 0.0
    assert diagnostics.query_diagnostics[0].nearest_source_bank_ids == ("walker-10",)
    assert diagnostics.query_diagnostics[1].nearest_source_bank_ids == ("walker-0",)
    public = diagnostics.to_public_dict()
    assert public["private_query_and_axis_rows_withheld"] is True
    assert "query-interp" not in str(public)
    assert "walker/mass/4" not in str(public)


def test_dynamics_diagnostics_reject_identity_drift() -> None:
    metric = _metric()
    rows = [
        (
            replace(row, query_goal_contract_id="goal:tampered")
            if row.query_bank_id == "query-extra"
            else row
        )
        for row in metric.rows
    ]
    drifted = SignalMetricRecord(
        cell_id=metric.cell_id,
        view_or_condition_id=metric.view_or_condition_id,
        representation_id=metric.representation_id,
        representation_coordinate_digest=metric.representation_coordinate_digest,
        representation_seed=metric.representation_seed,
        source_index_digest=metric.source_index_digest,
        query_manifest_digest=metric.query_manifest_digest,
        rows=tuple(rows),
        expected_source_by_query=metric.expected_source_by_query,
    )
    with pytest.raises(DynamicsAxisError, match="query row identity"):
        build_dynamics_axis_diagnostics(
            metric_record=drifted,
            registry=_registry(),
            execution_mode="DEVELOPMENT_SMOKE",
            signal_plan_digest=_d("plan"),
            signal_execution_protocol_digest=_d("execution"),
            identity_registry_digest=_d("identity-registry"),
        )


def test_formal_dynamics_diagnostics_fail_closed_without_authority() -> None:
    with pytest.raises(DynamicsAxisError, match="require signal-atlas authorization"):
        build_dynamics_axis_diagnostics(
            metric_record=_metric(),
            registry=_registry(),
            execution_mode="FORMAL",
            signal_plan_digest=_d("plan"),
            signal_execution_protocol_digest=_d("execution"),
            identity_registry_digest=_d("identity-registry"),
        )


def test_public_query_regimes_join_exactly_to_dynamics_axis_roles() -> None:
    registry = _registry()
    regimes = {}
    aliases = {}
    index = 0
    for regime, count, contexts in (
        ("EXACT", 30, ("walker/mass/0", "walker/mass/10")),
        ("INTERPOLATION", 24, ("walker/mass/4",)),
        ("EXTRAPOLATION", 12, ("walker/mass/15",)),
    ):
        for offset in range(count):
            query_id = f"v03q-{index:032x}"
            regimes[query_id] = regime
            aliases[query_id] = contexts[offset % len(contexts)]
            index += 1
    alias_digest = dynamics_query_alias_manifest_digest(aliases, registry)
    plan = PublicQueryPlan(
        regime_by_opaque_query_id=regimes,
        query_alias_manifest_digest=alias_digest,
    )
    join = DynamicsPublicQueryJoin.bind(
        public_query_plan=plan,
        registry=registry,
        dynamics_context_by_opaque_query_id=aliases,
    )
    assert join.to_public_dict()["query_count"] == 66
    assert join.to_public_dict()[
        "private_query_to_context_aliases_withheld"
    ] is True
    interpolation_id = next(
        query_id for query_id, regime in regimes.items() if regime == "INTERPOLATION"
    )
    drifted = dict(aliases)
    drifted[interpolation_id] = "walker/mass/0"
    with pytest.raises(DynamicsAxisError, match="disagrees with dynamics-axis role"):
        DynamicsPublicQueryJoin.bind(
            public_query_plan=plan,
            registry=registry,
            dynamics_context_by_opaque_query_id=drifted,
        )
