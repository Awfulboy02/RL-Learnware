from policy_learnware_v0.v03.acceptance import run_minimal_compute_acceptance


def test_minimal_compute_acceptance_passes_and_is_deterministic() -> None:
    first = run_minimal_compute_acceptance()
    second = run_minimal_compute_acceptance()
    assert first.passed
    assert first.to_dict() == second.to_dict()
    assert all(first.checks.values())
