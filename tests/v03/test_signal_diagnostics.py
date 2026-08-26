from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.signal_diagnostics import (
    SignalDiagnosticError,
    bank_geometry_diagnostic,
    build_signal_cell_diagnostics,
)
from policy_learnware_v0.v03.signal_metrics import SignalDistanceRow, SignalMetricRecord
from policy_learnware_v0.v03.signal_runtime import RepresentedBank
from policy_learnware_v0.v03.representation_ladder import (
    RepresentationBatch,
    RepresentationOutput,
    fit_r0_identity,
)
from policy_learnware_v0.v03.signal_runtime import (
    SignalBankIdentity,
    feature_bank_from_transition_view,
    transform_feature_banks,
)
from policy_learnware_v0.v03.transition_views import (
    V_REWARD_FREE_TRANSITION,
    TransitionBank,
    apply_transition_view,
)


def _d(label: str) -> str:
    return sha256_json({"signal-diagnostic-test": label})


def _native(bank_id: str, task: str, role: str, center: float) -> NativeTransitionBank:
    observation = center + np.arange(8, dtype=np.float64).reshape(4, 2) / 100.0
    return NativeTransitionBank(
        bank_id=bank_id,
        task_private_id=task,
        data_role=role,  # type: ignore[arg-type]
        native_schema_digest=_d(f"schema:{task}"),
        raw_dataset_digest=_d(f"raw:{bank_id}"),
        observation=observation,
        action=np.asarray([[0.1], [0.2], [0.3], [0.4]], dtype=np.float64),
        reward=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        next_observation=observation + 0.05,
        terminated=np.asarray([False, True, False, True]),
        truncated=np.zeros(4, dtype=np.bool_),
        episode_id=np.asarray([0, 0, 1, 1]),
        timestep=np.asarray([0, 1, 0, 1]),
    )


def _represented_fixture():
    fit = (
        _native("fit-a", "task-a", "source_representation_train", 0.0),
        _native("fit-b", "task-b", "source_representation_train", 10.0),
    )
    shape_registry = NativeShapeRegistry.from_source_banks(fit)
    normalizer = fit_global_normalizer(fit, registry=shape_registry)
    canonicalizer = GlobalCanonicalizerSpec(shape_registry, normalizer)
    native = (
        _native("source-a", "task-a", "source_reference_spec", 0.1),
        _native("source-b", "task-b", "source_reference_spec", 10.1),
        _native("query-a", "task-a", "development_query", 0.12),
        _native("query-b", "task-b", "development_query", 10.12),
    )
    receipts = tuple(canonicalizer.transform(item) for item in native)
    identities = tuple(
        SignalBankIdentity.from_receipt(
            receipt,
            embodiment_id=f"embodiment-{receipt.task_private_id}",
            abi_contract_id=f"abi-{receipt.task_private_id}",
            goal_contract_id=f"goal-{receipt.task_private_id}",
            dynamics_context_id=f"dynamics-{receipt.task_private_id}",
            context_id=f"context-{receipt.task_private_id}",
            measurement_protocol_digest=_d("measurement"),
            probe_seed_digest=_d(f"probe:{receipt.bank_id}"),
            equivalence_class_id=f"eq-{receipt.task_private_id}",
        )
        for receipt in receipts
    )
    features = tuple(
        feature_bank_from_transition_view(
            receipt,
            identity,
            apply_transition_view(
                TransitionBank.from_canonical_batch(receipt.batch),
                V_REWARD_FREE_TRANSITION,
            ),
        )
        for receipt, identity in zip(receipts, identities, strict=True)
    )
    source_fit = RepresentationBatch(
        values=np.concatenate([item.values for item in features[:2]], axis=0),
        dataset_digest=_d("source-fit"),
        role="SOURCE_FIT",
    )
    represented = transform_feature_banks(fit_r0_identity(source_fit), features)
    rows = []
    for query_index, query in enumerate(represented[2:]):
        for source_index, source in enumerate(represented[:2]):
            query_identity = query.feature_bank.identity
            source_identity = source.feature_bank.identity
            rows.append(
                SignalDistanceRow(
                    query_bank_id=query.feature_bank.receipt.bank_id,
                    source_bank_id=source.feature_bank.receipt.bank_id,
                    query_receipt_digest=str(query.feature_bank.receipt.receipt_digest),
                    source_receipt_digest=str(source.feature_bank.receipt.receipt_digest),
                    query_raw_dataset_digest=query.feature_bank.receipt.raw_dataset_digest,
                    source_raw_dataset_digest=source.feature_bank.receipt.raw_dataset_digest,
                    query_task_id=query.feature_bank.receipt.task_private_id,
                    source_task_id=source.feature_bank.receipt.task_private_id,
                    query_context_id=query_identity.context_id,
                    source_context_id=source_identity.context_id,
                    query_embodiment_id=query_identity.embodiment_id,
                    source_embodiment_id=source_identity.embodiment_id,
                    query_abi_contract_id=query_identity.abi_contract_id,
                    source_abi_contract_id=source_identity.abi_contract_id,
                    query_goal_contract_id=query_identity.goal_contract_id,
                    source_goal_contract_id=source_identity.goal_contract_id,
                    query_dynamics_context_id=query_identity.dynamics_context_id,
                    source_dynamics_context_id=source_identity.dynamics_context_id,
                    query_equivalence_class_id=query_identity.equivalence_class_id,
                    source_equivalence_class_id=source_identity.equivalence_class_id,
                    distance=0.1 if query_index == source_index else 2.0,
                )
            )
    metric = SignalMetricRecord(
        cell_id="diagnostic-cell",
        view_or_condition_id=V_REWARD_FREE_TRANSITION,
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=str(
            represented[0].representation_manifest.coordinate_digest
        ),
        representation_seed=None,
        source_index_digest=_d("source-index"),
        query_manifest_digest=_d("query-manifest"),
        rows=tuple(rows),
        expected_source_by_query={"query-a": "source-a", "query-b": "source-b"},
    )
    return represented, metric


def test_collapsed_geometry_is_explicit_not_nan() -> None:
    represented, _metric = _represented_fixture()
    base = represented[0]
    zero_values = np.zeros_like(base.values)
    input_batch = RepresentationBatch(
        values=base.feature_bank.values,
        dataset_digest=str(base.feature_bank.feature_bank_digest),
        role="QUERY_TRANSFORM",
    )
    output = RepresentationOutput(
        values=zero_values,
        input_batch_digest=input_batch.batch_digest,
        coordinate_digest=str(base.representation_manifest.coordinate_digest),
    )
    collapsed = RepresentedBank(
        feature_bank=base.feature_bank,
        representation_manifest=base.representation_manifest,
        values=zero_values,
        representation_output_digest=str(output.output_digest),
    )
    diagnostic = bank_geometry_diagnostic(collapsed)
    assert diagnostic.collapsed
    assert diagnostic.effective_rank == 0.0
    assert diagnostic.zero_variance_fraction == 1.0


def test_signal_diagnostics_publish_geometry_and_hide_taxonomy() -> None:
    represented, metric = _represented_fixture()
    diagnostics = build_signal_cell_diagnostics(
        source_banks=represented[:2],
        query_banks=represented[2:],
        metric_record=metric,
    )
    public = diagnostics.to_public_dict()
    assert public["bank_count"] == 4
    assert public["axis_summaries"]["TASK_GLOBAL"]["accuracy"] == 1.0
    assert public["private_bank_and_taxonomy_rows_withheld"] is True
    assert "task-a" not in str(public)
    assert len(diagnostics.confusion_records) == 2

    with pytest.raises(SignalDiagnosticError, match="membership"):
        build_signal_cell_diagnostics(
            source_banks=represented[:2],
            query_banks=represented[2:3],
            metric_record=metric,
        )
