from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.representation_ladder import (
    R0_PADDED_RAW,
    R3_MATCHED_RANDOM_MLP,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    R_HIST_RANDOM_TANH,
)
from policy_learnware_v0.v03.signal_matrix import (
    C_RF_SHUFFLED_NEXT,
    SignalCellRecord,
    SignalFitJob,
    SignalMatrixError,
    SignalMatrixLedger,
    SignalMatrixPlan,
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.transition_views import (
    REGISTERED_VIEW_IDS,
    V_FULL_LEGACY,
    V_RANDOM_ENCODER,
    V_TEMPORAL_SHUFFLE,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def test_canonical_signal_plan_has_39_logical_37_numeric_and_two_temporal_na() -> None:
    plan = build_signal_matrix_plan()
    assert len(REGISTERED_VIEW_IDS) == 14
    assert plan.logical_cell_count == 39
    assert plan.numeric_cell_count == 37
    assert plan.structural_na_count == 2
    structural = tuple(
        (cell.condition_id, cell.representation_id)
        for cell in plan.cells
        if cell.applicability == "STRUCTURAL_NA"
    )
    assert structural == (
        (V_TEMPORAL_SHUFFLE, R0_PADDED_RAW),
        (V_TEMPORAL_SHUFFLE, R5_VIEW_SPECIFIC_CORRO_REFIT),
    )
    assert all(not cell.optimization_fit_required for cell in plan.cells if cell.applicability == "STRUCTURAL_NA")

    historical = next(
        cell
        for cell in plan.cells
        if cell.condition_id == V_RANDOM_ENCODER
        and cell.representation_id == R_HIST_RANDOM_TANH
    )
    matched = next(
        cell
        for cell in plan.cells
        if cell.condition_id == V_FULL_LEGACY
        and cell.representation_id == R3_MATCHED_RANDOM_MLP
    )
    assert historical.cell_id != matched.cell_id
    assert historical.cell_digest != matched.cell_digest


def test_plan_roundtrip_is_strict_and_rejects_cell_or_count_tamper() -> None:
    plan = build_signal_matrix_plan()
    assert SignalMatrixPlan.from_dict(plan.to_dict()) == plan

    tampered = plan.to_dict()
    tampered["cells"][0]["condition_id"] = "V_POSTHOC"
    with pytest.raises(SignalMatrixError):
        SignalMatrixPlan.from_dict(tampered)

    wrong_count = plan.to_dict()
    wrong_count["numeric_cell_count"] = 38
    with pytest.raises(SignalMatrixError, match="count|digest"):
        SignalMatrixPlan.from_dict(wrong_count)

    unknown = plan.to_dict()
    unknown["optional_cell"] = {}
    with pytest.raises(SignalMatrixError, match="unknown"):
        SignalMatrixPlan.from_dict(unknown)


def test_fit_plan_is_exactly_36_r5_plus_9_r5l_and_never_schedules_na() -> None:
    plan = build_signal_matrix_plan()
    jobs = build_optimization_fit_jobs(plan)
    assert len(jobs) == 45
    assert sum(job.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT for job in jobs) == 36
    assert sum(job.representation_id == R5L_SUPERVISED_LINEAR for job in jobs) == 9
    assert {job.seed for job in jobs} == {0, 1, 2}
    assert all(job.condition_id != V_TEMPORAL_SHUFFLE for job in jobs)
    assert {
        job.condition_id
        for job in jobs
        if job.representation_id == R5L_SUPERVISED_LINEAR
    } == {V_FULL_LEGACY, "V_REWARD_FREE_TRANSITION", C_RF_SHUFFLED_NEXT}
    assert SignalFitJob.from_dict(jobs[0].to_dict()) == jobs[0]


def _records(plan: SignalMatrixPlan) -> tuple[SignalCellRecord, ...]:
    result = []
    for index, cell in enumerate(plan.cells):
        if cell.applicability == "STRUCTURAL_NA":
            result.append(
                SignalCellRecord(
                    plan_digest=str(plan.plan_digest),
                    cell_id=cell.cell_id,
                    cell_digest=str(cell.cell_digest),
                    status="STRUCTURAL_NA",
                    metrics=None,
                    numeric_artifact_digest=None,
                )
            )
        else:
            result.append(
                SignalCellRecord(
                    plan_digest=str(plan.plan_digest),
                    cell_id=cell.cell_id,
                    cell_digest=str(cell.cell_digest),
                    status="COMPUTED",
                    metrics={"retrieval": float(index) / 100.0},
                    numeric_artifact_digest=_d(f"artifact-{index}"),
                )
            )
    return tuple(result)


def test_ledger_excludes_na_from_denominator_and_roundtrips() -> None:
    plan = build_signal_matrix_plan()
    ledger = SignalMatrixLedger(plan=plan, records=_records(plan))
    assert len(ledger.records) == 39
    assert len(ledger.numeric_records) == 37
    assert ledger.metric_denominator("retrieval") == 37
    assert all(record.status == "COMPUTED" for record in ledger.numeric_records)
    restored = SignalMatrixLedger.from_dict(ledger.to_dict(), plan=plan)
    assert restored == ledger


def test_structural_na_cannot_be_zero_nan_computed_or_fit() -> None:
    plan = build_signal_matrix_plan()
    cell = next(cell for cell in plan.cells if cell.applicability == "STRUCTURAL_NA")
    with pytest.raises(SignalMatrixError, match="cannot carry"):
        SignalCellRecord(
            plan_digest=str(plan.plan_digest),
            cell_id=cell.cell_id,
            cell_digest=str(cell.cell_digest),
            status="STRUCTURAL_NA",
            metrics={"retrieval": 0.0},
            numeric_artifact_digest=None,
        )
    with pytest.raises(SignalMatrixError, match="finite"):
        SignalCellRecord(
            plan_digest=str(plan.plan_digest),
            cell_id=cell.cell_id,
            cell_digest=str(cell.cell_digest),
            status="COMPUTED",
            metrics={"retrieval": float("nan")},
            numeric_artifact_digest=_d("na-artifact"),
        )

    records = list(_records(plan))
    index = next(i for i, item in enumerate(records) if item.cell_id == cell.cell_id)
    records[index] = SignalCellRecord(
        plan_digest=str(plan.plan_digest),
        cell_id=cell.cell_id,
        cell_digest=str(cell.cell_digest),
        status="COMPUTED",
        metrics={"retrieval": 0.0},
        numeric_artifact_digest=_d("forged-numeric"),
    )
    with pytest.raises(SignalMatrixError, match="applicability"):
        SignalMatrixLedger(plan=plan, records=tuple(records))


def test_ledger_rejects_missing_duplicate_and_cross_plan_records() -> None:
    plan = build_signal_matrix_plan()
    records = list(_records(plan))
    with pytest.raises(SignalMatrixError, match="exactly one"):
        SignalMatrixLedger(plan=plan, records=tuple(records[:-1]))

    duplicate = records[:-1] + [records[0]]
    with pytest.raises(SignalMatrixError, match="coverage|exactly"):
        SignalMatrixLedger(plan=plan, records=tuple(duplicate))

    records[0] = replace(records[0], plan_digest=_d("another-plan"), record_digest=None)
    with pytest.raises(SignalMatrixError, match="another plan"):
        SignalMatrixLedger(plan=plan, records=tuple(records))

