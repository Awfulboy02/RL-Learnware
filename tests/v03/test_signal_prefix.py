from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.contracts import (
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    SemanticCacheKey,
    SemanticCacheRecord,
    SemanticTransform,
    SourceRepresentationIndex,
    build_source_reduced_spec,
)
from policy_learnware_v0.v03.representation_plan import RepresentationExecutionPlan
from policy_learnware_v0.v03.signal_controls import HistoricalRandomTanhSpec
from policy_learnware_v0.v03.signal_matrix import build_signal_matrix_plan
from policy_learnware_v0.v03.signal_prefix import (
    FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS,
    SignalPrefixCacheSet,
    SignalPrefixError,
    SignalPrefixExecutionProtocol,
    SignalPrefixSchedule,
    run_signal_prefixes,
)
from policy_learnware_v0.v03.signal_runtime import (
    SignalBankIdentity,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
)
from policy_learnware_v0.v03.transition_views import V_REWARD_FREE_TRANSITION
from policy_learnware_v0.v03.representation_ladder import R0_PADDED_RAW


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _cache(label: str, values: np.ndarray) -> SemanticCacheRecord:
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    episode_offsets = np.arange(values.shape[0] + 1, dtype=np.int64)
    return SemanticCacheRecord(
        key=SemanticCacheKey(
            raw_dataset_digest=_d(f"raw:{label}"),
            ordered_episode_window_digest=_d(f"windows:{label}"),
            canonical_view_digest=_d("canonical-view"),
            window_protocol_digest=_d("one-step-window"),
            normalizer_digest=_d("normalizer"),
            semantic_transform=SemanticTransform.raw_identity(),
            mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
        ),
        points=values,
        episode_offsets=episode_offsets,
    )


def _identity(
    bank_id: str,
    task: str,
    dynamics: str,
    measurement: str,
) -> SignalBankIdentity:
    return SignalBankIdentity(
        receipt_digest=_d(f"receipt:{bank_id}"),
        bank_id=bank_id,
        task_private_id=task,
        embodiment_id=f"emb-{task}",
        abi_contract_id=f"abi-{task}",
        goal_contract_id=f"goal-{task}",
        dynamics_context_id=dynamics,
        context_id=dynamics,
        measurement_protocol_digest=measurement,
        probe_seed_digest=_d(f"probe:{bank_id}"),
        equivalence_class_id=task,
    )


def _fixture(*, formal: bool = True, query_episode_count: int = 64):
    plan = build_signal_matrix_plan()
    cell = next(
        item
        for item in plan.numeric_cells
        if item.condition_id == V_REWARD_FREE_TRANSITION
        and item.representation_id == R0_PADDED_RAW
    )
    measurement = _d("measurement")
    identities = (
        _identity("source-a", "task-a", "dyn-a", measurement),
        _identity("source-b", "task-b", "dyn-b", measurement),
        _identity("query-a", "task-a", "dyn-a", measurement),
    )
    registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=_d("taxonomy"), identities=identities
    )
    historical = HistoricalRandomTanhSpec.create(
        seed=0, input_dim=1, output_dim=5
    )
    base = SignalExecutionProtocol(
        plan_digest=str(plan.plan_digest),
        identity_registry_digest=str(registry.registry_digest),
        measurement_protocol_digest=measurement,
        representation_plan=RepresentationExecutionPlan.create(
            signal_plan=plan, historical_spec=historical
        ),
        condition_plan=ConditionExecutionPlan.create(historical_spec=historical),
        execution_mode="FORMAL" if formal else "DEVELOPMENT_SMOKE",
        reducer_config=ReducerConfig(
            support_budget=2, support_steps=0, kmeans_steps=1
        ),
        block_size=16,
    )
    source_caches = {
        "source-a": _cache("source-a", np.asarray([-1.0, -0.9])),
        "source-b": _cache("source-b", np.asarray([1.0, 0.9])),
    }
    source_index = SourceRepresentationIndex(
        representation_protocol_id=next(
            iter(source_caches.values())
        ).key.representation_protocol_digest,
        entries={
            source_id: build_source_reduced_spec(
                cache,
                kernel_bandwidth=1.0,
                measurement_protocol_id=str(base.protocol_digest),
                probe_dataset_digest=cache.key.raw_dataset_digest,
                reducer_config=base.reducer_config,
            )
            for source_id, cache in source_caches.items()
        },
    )
    query = _cache(
        "query-a", np.linspace(-1.0, -0.75, query_episode_count)
    )
    cache_set = SignalPrefixCacheSet(
        identity_registry_digest=str(registry.registry_digest),
        query_caches={"query-a": query},
        query_receipt_digests={
            "query-a": next(
                item.receipt_digest
                for item in identities
                if item.bank_id == "query-a"
            )
        },
        query_raw_dataset_digests={
            "query-a": query.key.raw_dataset_digest
        },
    )
    schedule = (
        SignalPrefixSchedule.formal()
        if formal
        else SignalPrefixSchedule.development((1, 2, 4))
    )
    prefix_protocol = SignalPrefixExecutionProtocol.create(
        signal_execution_protocol=base,
        cell=cell,
        source_index=source_index,
        query_cache_set=cache_set,
        prefix_schedule=schedule,
    )
    return (
        plan,
        cell,
        registry,
        base,
        source_index,
        cache_set,
        prefix_protocol,
    )


def test_formal_schedule_and_max_prefix_cache_are_exact() -> None:
    schedule = SignalPrefixSchedule.formal()
    assert schedule.prefix_episode_counts == FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
    with pytest.raises(SignalPrefixError, match="exactly"):
        SignalPrefixSchedule((1, 2, 4, 8, 16, 32), "FORMAL")
    with pytest.raises(SignalPrefixError, match="exact maximum prefix"):
        _fixture(query_episode_count=65)


def test_formal_mode_rejects_development_prefix_schedule() -> None:
    _, cell, registry, base, source_index, cache_set, _ = _fixture()
    with pytest.raises(SignalPrefixError, match="formal prefix schedule"):
        SignalPrefixExecutionProtocol.create(
            signal_execution_protocol=base,
            cell=cell,
            source_index=source_index,
            query_cache_set=cache_set,
            prefix_schedule=SignalPrefixSchedule.development((1, 2, 4, 8, 16, 32, 64)),
        )
    assert registry.registry_digest == cache_set.identity_registry_digest


def test_prefix_run_slices_one_cache_and_builds_complete_metric_per_prefix() -> None:
    _, cell, registry, base, source_index, cache_set, protocol = _fixture(
        formal=False, query_episode_count=4
    )
    run = run_signal_prefixes(
        protocol=protocol,
        signal_execution_protocol=base,
        cell=cell,
        source_index=source_index,
        query_cache_set=cache_set,
        identity_registry=registry,
        expected_source_by_query={"query-a": "source-a"},
        representation_coordinate_digest=_d("r0-coordinate"),
        representation_seed=None,
    )
    assert tuple(point.prefix_episode_count for point in run.points) == (
        (1, 2, 4)
    )
    assert len({point.point_digest for point in run.points}) == 3
    assert len({point.query_spec_digests["query-a"] for point in run.points}) == 3
    assert all(point.metric_record.metric_values["source_count"] == 2.0 for point in run.points)
    assert all(len(point.metric_record.rows) == 2 for point in run.points)
    assert cache_set.to_dict()["encoder_forward_count_per_query"] == 1
    assert run.to_public_dict()["schedule_digest"] == (
        protocol.prefix_schedule.schedule_digest
    )
    assert run.to_public_dict()[
        "private_query_specs_and_distance_rows_withheld"
    ] is True
    assert "query-a" not in str(run.to_public_dict())


def test_formal_prefix_run_fails_closed_without_authority() -> None:
    _, cell, registry, base, source_index, cache_set, protocol = _fixture()
    with pytest.raises(SignalPrefixError, match="requires signal-atlas authorization"):
        run_signal_prefixes(
            protocol=protocol,
            signal_execution_protocol=base,
            cell=cell,
            source_index=source_index,
            query_cache_set=cache_set,
            identity_registry=registry,
            expected_source_by_query={"query-a": "source-a"},
            representation_coordinate_digest=_d("r0-coordinate"),
            representation_seed=None,
        )


def test_prefix_protocol_detects_cache_or_source_drift() -> None:
    _, cell, _, base, source_index, cache_set, protocol = _fixture()
    drifted = replace(
        cache_set,
        query_caches={
            "query-a": _cache("query-drift", np.linspace(-1.0, -0.5, 64))
        },
        query_raw_dataset_digests={
            "query-a": _d("raw:query-drift")
        },
        cache_set_digest=None,
    )
    with pytest.raises(SignalPrefixError, match="runtime inputs differ"):
        protocol.validate_inputs(
            signal_execution_protocol=base,
            cell=cell,
            source_index=source_index,
            query_cache_set=drifted,
        )
