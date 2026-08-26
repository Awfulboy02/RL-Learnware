from __future__ import annotations

from policy_learnware_v0.v03.prelarge_acceptance import run_prelarge_acceptance


def test_prelarge_acceptance_is_deterministic_and_stops_before_formal_work() -> None:
    first = run_prelarge_acceptance()
    second = run_prelarge_acceptance()
    assert first.passed
    assert first.status == "ENGINEERING_COMPONENTS_PASS_FORMAL_FREEZE_PENDING"
    assert first.report_digest == second.report_digest
    assert (first.signal_matrix_logical_cells, first.signal_matrix_numeric_cells) == (
        39,
        37,
    )
    assert first.optimization_fit_jobs == 45
    assert first.formal_run_authorized is False
    assert first.large_experiment_executed is False
    assert first.v04_assets_required is False
    assert first.checks["signal_diagnostics_bound_before_array_discard"]
    assert "signal_cell_diagnostics" in first.evidence_digests
