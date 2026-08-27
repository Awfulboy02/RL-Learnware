from policy_learnware_v0.v03.signal_matrix import (
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)


def test_signal_plan_retains_the_14_view_experiment() -> None:
    plan = build_signal_matrix_plan()
    jobs = build_optimization_fit_jobs(plan)
    assert plan.logical_cell_count == 39
    assert plan.numeric_cell_count == 37
    assert plan.structural_na_count == 2
    assert len(jobs) == 45
    assert {cell.condition_id for cell in plan.cells} >= {
        "V_FULL_LEGACY",
        "V_SHUFFLED_NEXT",
        "V_RANDOM_ENCODER",
    }
