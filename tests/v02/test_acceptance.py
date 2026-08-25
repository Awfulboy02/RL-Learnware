from __future__ import annotations

import copy

import numpy as np
import pytest

from policy_learnware_v0.v02.acceptance import (
    CORRO_FAIL_METHOD_IDS,
    FULL_METHOD_IDS,
    AcceptanceContractError,
    AcceptanceReport,
    run_cpu_acceptance_fixture,
)


@pytest.fixture(scope="module")
def scientific_run():
    return run_cpu_acceptance_fixture("scientific_pass")


@pytest.fixture(scope="module")
def market_fail_run():
    return run_cpu_acceptance_fixture("no_go_market")


@pytest.fixture(scope="module")
def corro_fail_run():
    return run_cpu_acceptance_fixture("no_go_corro")


def test_scientific_pass_runs_the_real_twenty_job_chain(scientific_run) -> None:
    report = scientific_run.report
    assert report.status == "SCIENTIFIC_PASS"
    assert report.formal_completion_claimed is False
    assert report.stage == "development_discovery"
    assert report.formal_task_requirement == 6
    assert (
        report.fixture_task_count,
        report.axes_per_task,
        report.anchors_per_task,
        report.source_anchor_count,
        report.training_seed_count,
        report.planned_training_run_count,
    ) == (2, 2, 5, 10, 2, 20)
    assert len(scientific_run.training_jobs) == 20
    jobs_by_anchor: dict[str, list[int]] = {}
    for job in scientific_run.training_jobs:
        jobs_by_anchor.setdefault(job.source_anchor_id, []).append(job.seed)
    assert len(jobs_by_anchor) == 10
    assert all(sorted(seeds) == [101, 202] for seeds in jobs_by_anchor.values())
    assert len(scientific_run.admitted_records) == 20
    assert scientific_run.championization is not None
    assert len(scientific_run.championization.competence_records) == 10
    assert not scientific_run.championization.rejected_anchors
    assert scientific_run.market is not None
    assert len(scientific_run.market.entries) == 10
    assert all(gate.passed for gate in report.gates.values())


def test_scientific_pass_runs_common_baselines_oracle_metrics_cost_and_report(
    scientific_run,
) -> None:
    assert scientific_run.report.method_ids == FULL_METHOD_IDS
    assert len(scientific_run.selection_records) == 4
    for by_method in scientific_run.selection_records.values():
        assert tuple(sorted(by_method)) == FULL_METHOD_IDS
        assert all(len(record.ranking) == 10 for record in by_method.values())
        assert all(record.selected_id == record.ranking[0].opaque_learnware_id for record in by_method.values())
        assert all(record.evidence_contract.target_gradient_updates == 0 for record in by_method.values())
    assert len(scientific_run.oracle_results) == 4
    assert all(tuple(sorted(result.outcomes)) == FULL_METHOD_IDS for result in scientific_run.oracle_results.values())
    assert all(len(result.normalized_value_vector) == 10 for result in scientific_run.oracle_results.values())
    assert set(scientific_run.metrics) == set(FULL_METHOD_IDS)
    assert all(metric.task_count == 2 for metric in scientific_run.metrics.values())
    assert all(metric.axis_count == 4 for metric in scientific_run.metrics.values())
    assert scientific_run.metrics["M02/B5"].macro_mean < scientific_run.metrics["B1"].macro_mean
    assert scientific_run.cost_reconciliation is not None
    assert scientific_run.cost_reconciliation.cold.query_count == 4
    assert scientific_run.cost_reconciliation.warm.query_count == 4
    assert scientific_run.cost_reconciliation.cold_to_warm_ratio == pytest.approx(2.0)
    assert scientific_run.development_p_table is not None
    assert len(scientific_run.development_p_table.rows) == 4 * len(FULL_METHOD_IDS)
    public_payload = scientific_run.development_p_table.to_dict()
    assert "normalized_value_vector" not in repr(public_payload)
    assert "private_target_instance_digest" not in repr(public_payload)


def test_no_go_market_is_a_scientific_outcome_not_an_engineering_failure(
    market_fail_run,
) -> None:
    report = market_fail_run.report
    assert report.status == "NO_GO_MARKET"
    assert report.gates["CPU-Engineering"].passed
    assert not report.gates["CPU-Market"].passed
    assert report.gates["CPU-CORRO"].passed
    assert not report.gates["CPU-Scientific"].passed
    assert market_fail_run.metrics["B1"].macro_mean == pytest.approx(0.0)
    assert market_fail_run.metrics["B3b"].macro_mean > 0.0
    assert len(market_fail_run.oracle_results) == 4
    assert market_fail_run.development_p_table is not None


def test_no_go_corro_fails_closed_without_a_sigma_fallback(corro_fail_run) -> None:
    report = corro_fail_run.report
    assert report.status == "NO_GO_CORRO"
    assert report.gates["CPU-Engineering"].passed
    assert report.gates["CPU-Market"].passed
    assert not report.gates["CPU-CORRO"].passed
    assert report.gates["CPU-CORRO"].checks["raw_signal"]
    assert not report.gates["CPU-CORRO"].checks["corro_signal"]
    assert report.gates["CPU-CORRO"].checks["no_zero_distance_sigma_fallback"]
    assert report.source_sigma_artifact_digest is None
    assert report.method_ids == CORRO_FAIL_METHOD_IDS
    assert "M02/B5" not in corro_fail_run.metrics
    assert corro_fail_run.raw_representation_index is not None
    assert corro_fail_run.corro_representation_index is not None
    raw_centers = {
        float(entry.environment_spec.supports[0, 0])
        for entry in corro_fail_run.raw_representation_index.entries.values()
    }
    corro_centers = {
        float(entry.environment_spec.supports[0, 0])
        for entry in corro_fail_run.corro_representation_index.entries.values()
    }
    assert len(raw_centers) > 1
    assert corro_centers == {0.0}


def test_engineering_poison_blocks_before_market_or_oracle() -> None:
    run = run_cpu_acceptance_fixture("engineering_blocked")
    report = run.report
    assert report.status == "BLOCKED_ENGINEERING"
    assert not report.gates["CPU-Engineering"].passed
    assert not report.gates["CPU-Engineering"].checks["attestation_environment_identity"]
    assert len(run.training_jobs) == 20
    assert not run.admitted_records
    assert run.championization is None
    assert run.market is None
    assert not run.selection_records
    assert not run.oracle_results
    assert run.development_p_table is None
    assert report.blocked_reason_digest is not None
    assert run.blocked_reason is not None
    assert "environment digest differs" in run.blocked_reason


def test_shuffled_shards_have_identical_digest_bound_outputs(scientific_run) -> None:
    shuffled = run_cpu_acceptance_fixture("scientific_pass", shuffle_inputs=True)
    assert shuffled.report.digest == scientific_run.report.digest
    assert shuffled.report.to_dict() == scientific_run.report.to_dict()
    assert shuffled.report.training_plan_digest == scientific_run.report.training_plan_digest
    assert shuffled.report.selection_records_digest == scientific_run.report.selection_records_digest
    assert shuffled.report.private_oracle_results_digest == scientific_run.report.private_oracle_results_digest
    assert shuffled.report.development_p_table_digest == scientific_run.report.development_p_table_digest
    assert shuffled.development_p_table is not None
    assert scientific_run.development_p_table is not None
    assert shuffled.development_p_table.to_dict() == scientific_run.development_p_table.to_dict()


def test_acceptance_report_round_trip_and_digest_tamper_rejection(scientific_run) -> None:
    payload = scientific_run.report.to_dict()
    restored = AcceptanceReport.from_dict(payload)
    assert restored == scientific_run.report
    assert restored.digest == scientific_run.report.digest

    tampered = copy.deepcopy(payload)
    tampered["scope"]["planned_training_run_count"] = 19
    with pytest.raises(AcceptanceContractError):
        AcceptanceReport.from_dict(tampered)

    unknown = copy.deepcopy(payload)
    unknown["artifact_digests"]["private_bundle_path"] = "/tmp/leak"
    with pytest.raises(AcceptanceContractError, match="unknown"):
        AcceptanceReport.from_dict(unknown)


def test_acceptance_inputs_are_immutable_copies(scientific_run) -> None:
    assert scientific_run.raw_representation_index is not None
    entry = next(iter(scientific_run.raw_representation_index.entries.values()))
    assert not entry.environment_spec.supports.flags.writeable
    with pytest.raises(ValueError):
        entry.environment_spec.supports[0, 0] = np.nan
