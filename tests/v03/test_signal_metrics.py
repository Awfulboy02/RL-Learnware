from __future__ import annotations

from dataclasses import replace
import json

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.signal_metrics import (
    SignalDistanceRow,
    SignalMetricError,
    SignalMetricRecord,
    paired_signal_contrast,
    representation_gain_contrast,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _record(cell: str, distances: tuple[tuple[float, float], ...]) -> SignalMetricRecord:
    rows = []
    for query_index, pair in enumerate(distances):
        for source_index, distance in enumerate(pair):
            rows.append(
                SignalDistanceRow(
                    query_bank_id=f"query-{query_index}",
                    source_bank_id=f"source-{source_index}",
                    query_receipt_digest=_d(f"query-receipt-{query_index}"),
                    source_receipt_digest=_d(f"source-receipt-{source_index}"),
                    query_raw_dataset_digest=_d(f"query-raw-{query_index}"),
                    source_raw_dataset_digest=_d(f"source-raw-{source_index}"),
                    query_task_id=f"task-{query_index}",
                    source_task_id=f"task-{source_index}",
                    query_context_id=f"context-{query_index}",
                    source_context_id=f"context-{source_index}",
                    query_embodiment_id=f"embodiment-{query_index}",
                    source_embodiment_id=f"embodiment-{source_index}",
                    query_abi_contract_id=f"abi-{query_index}",
                    source_abi_contract_id=f"abi-{source_index}",
                    query_goal_contract_id=f"goal-{query_index}",
                    source_goal_contract_id=f"goal-{source_index}",
                    query_dynamics_context_id=f"dynamics-{query_index}",
                    source_dynamics_context_id=f"dynamics-{source_index}",
                    query_equivalence_class_id=f"equivalence-{query_index}",
                    source_equivalence_class_id=f"equivalence-{source_index}",
                    distance=distance,
                )
            )
    return SignalMetricRecord(
        cell_id=cell,
        view_or_condition_id="V_REWARD_FREE_TRANSITION",
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=_d("coordinate"),
        representation_seed=None,
        source_index_digest=_d("source-index"),
        query_manifest_digest=_d("queries"),
        rows=tuple(rows),
        expected_source_by_query={"query-0": "source-0", "query-1": "source-1"},
    )


def test_signal_metrics_recover_known_nearest_contexts_and_separation() -> None:
    record = _record("rf-r0", ((0.1, 2.0), (1.5, 0.2)))
    assert record.metric_values["context_top1"] == 1.0
    assert record.metric_values["task_top1"] == 1.0
    assert record.metric_values["task_mrr"] == 1.0
    assert record.metric_values["context_mrr"] == 1.0
    assert record.metric_values["between_within_margin"] > 0.0
    assert record.metric_values["between_within_ratio"] > 1.0
    assert record.metric_values["exact_source_top1"] == 1.0


def test_expected_source_mapping_cannot_redefine_structural_truth() -> None:
    record = _record("rf-r0", ((0.1, 2.0), (1.5, 0.2)))
    swapped = replace(
        record,
        expected_source_by_query={"query-0": "source-1", "query-1": "source-0"},
        metric_values=None,
    )
    assert swapped.metric_values["exact_source_top1"] == 0.0
    assert swapped.metric_values["context_top1"] == 1.0
    assert swapped.metric_values["task_top1"] == 1.0


def test_goal_and_dynamics_metrics_use_axis_scoped_candidate_sets() -> None:
    """A global schema shortcut must not decide goal or dynamics readout."""

    identities = (
        # Globally nearest, but outside the query embodiment/ABI.
        ("schema-shortcut", "finger-turn", "finger", "finger-abi", "turn", "nominal", 0.01),
        # Two same-schema goals establish a genuine inter-goal contrast.
        ("walker-run-nominal", "walker-run", "walker", "walker-abi", "run", "nominal", 0.20),
        ("walker-walk-nominal", "walker-walk", "walker", "walker-abi", "walk", "nominal", 0.30),
        # A second dynamics context makes the within-task axis eligible.  It is
        # closer than nominal, so conditional dynamics top-1 is intentionally 0.
        ("walker-run-heavy", "walker-run", "walker", "walker-abi", "run", "heavy", 0.10),
    )
    rows = tuple(
        SignalDistanceRow(
            query_bank_id="query-walker-run",
            source_bank_id=source_id,
            query_receipt_digest=_d("conditional-query-receipt"),
            source_receipt_digest=_d(f"conditional-{source_id}-receipt"),
            query_raw_dataset_digest=_d("conditional-query-raw"),
            source_raw_dataset_digest=_d(f"conditional-{source_id}-raw"),
            query_task_id="walker-run",
            source_task_id=task_id,
            query_context_id="walker-run/nominal",
            source_context_id=f"{task_id}/{dynamics_id}",
            query_embodiment_id="walker",
            source_embodiment_id=embodiment_id,
            query_abi_contract_id="walker-abi",
            source_abi_contract_id=abi_id,
            query_goal_contract_id="run",
            source_goal_contract_id=goal_id,
            query_dynamics_context_id="nominal",
            source_dynamics_context_id=dynamics_id,
            query_equivalence_class_id=None,
            source_equivalence_class_id=None,
            distance=distance,
        )
        for (
            source_id,
            task_id,
            embodiment_id,
            abi_id,
            goal_id,
            dynamics_id,
            distance,
        ) in identities
    )
    record = SignalMetricRecord(
        cell_id="conditional-axis-test",
        view_or_condition_id="V_REWARD_FREE_TRANSITION",
        representation_id="R0_PADDED_RAW",
        representation_coordinate_digest=_d("conditional-coordinate"),
        representation_seed=None,
        source_index_digest=_d("conditional-source-index"),
        query_manifest_digest=_d("conditional-query-manifest"),
        rows=rows,
        expected_source_by_query={
            "query-walker-run": "walker-run-nominal"
        },
    )
    # Global ranking is won by an incompatible schema and is diagnostic only.
    assert record.metric_values["task_top1"] == 0.0
    assert record.metric_values["embodiment_top1"] == 0.0
    # Goal is evaluated only within walker/walker-ABI, while dynamics is
    # evaluated only within walker-run/run/walker-ABI.
    assert record.metric_values["goal_top1"] == 1.0
    assert record.metric_values["goal_mrr"] == 1.0
    assert record.metric_values["dynamics_top1"] == 0.0
    assert record.metric_values["dynamics_mrr"] == 0.5
    assert record.metric_values["dynamics_between_mean_distance"] == 0.10
    assert record.metric_values["dynamics_between_pair_count"] == 1.0
    assert record.metric_values["goal_query_coverage"] == 1.0
    assert record.metric_values["dynamics_query_coverage"] == 1.0


def test_missing_axis_contrast_is_explicit_na_not_zero_accuracy() -> None:
    record = _record("rf-r0", ((0.1, 2.0), (1.5, 0.2)))
    assert record.metric_values["goal_query_count"] == 0.0
    assert record.metric_values["goal_query_coverage"] == 0.0
    assert "goal_top1" not in record.metric_values
    assert "goal_mrr" not in record.metric_values
    assert record.metric_values["dynamics_query_count"] == 0.0
    assert record.metric_values["dynamics_query_coverage"] == 0.0
    assert "dynamics_top1" not in record.metric_values
    assert "dynamics_mrr" not in record.metric_values


def test_paired_contrast_pairs_family_seed_and_banks_not_fitted_coordinate() -> None:
    base = _record("rf-r0", ((0.1, 2.0), (1.5, 0.2)))
    control = _record("rf-shuffled-r0", ((1.1, 1.0), (0.9, 1.0)))
    contrast = paired_signal_contrast(base, control)
    assert contrast.metric_deltas["context_top1"] == 1.0
    assert contrast.metric_deltas["context_mrr"] == 0.5
    # A view-specific R5 refit necessarily has a different fitted coordinate;
    # aggregate retrieval metrics remain paired by family, seed and raw banks.
    changed_coordinate = replace(
        control, representation_coordinate_digest=_d("other-coordinate")
    )
    assert paired_signal_contrast(base, changed_coordinate).metric_deltas[
        "context_top1"
    ] == 1.0
    with pytest.raises(SignalMetricError, match="same representation family"):
        paired_signal_contrast(base, replace(control, representation_id="R5"))
    with pytest.raises(SignalMetricError, match="same representation seed"):
        paired_signal_contrast(base, replace(control, representation_seed=1))

    forged_rows = list(control.rows)
    forged_rows[0] = replace(
        forged_rows[0], source_raw_dataset_digest=_d("forged-source-raw")
    )
    forged = replace(control, rows=tuple(forged_rows))
    with pytest.raises(SignalMetricError, match="raw bank membership"):
        paired_signal_contrast(base, forged)


def test_representation_gain_is_r5_minus_r0_on_exact_raw_banks() -> None:
    raw = _record("rf-r0", ((1.1, 1.0), (0.9, 1.0)))
    learned = replace(
        _record("rf-r5", ((0.1, 2.0), (1.5, 0.2))),
        representation_id="R5_VIEW_SPECIFIC_CORRO_REFIT",
        representation_seed=0,
        representation_coordinate_digest=_d("learned-coordinate"),
    )
    gain = representation_gain_contrast(raw, learned)
    assert gain.metric_gains["context_top1"] == 1.0
    with pytest.raises(SignalMetricError, match="same input view"):
        representation_gain_contrast(
            raw, replace(learned, view_or_condition_id="V_FULL_LEGACY")
        )


def test_incomplete_or_duplicate_distance_matrix_fails_closed() -> None:
    record = _record("rf-r0", ((0.1, 2.0), (1.5, 0.2)))
    with pytest.raises(SignalMetricError, match="same complete source set"):
        replace(record, rows=record.rows[:-1])
    with pytest.raises(SignalMetricError, match="duplicate"):
        replace(record, rows=(*record.rows, record.rows[0]))


def test_public_signal_projection_withholds_private_rows_and_taxonomy() -> None:
    record = _record("rf-r0", ((0.1, 2.0), (1.5, 0.2)))
    public = record.to_public_dict()
    serialized = json.dumps(public, sort_keys=True)
    assert public["private_distance_rows_withheld"] is True
    assert "rows" not in public
    assert "task-0" not in serialized
    assert "query-0" not in serialized
