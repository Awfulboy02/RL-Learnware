from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.cli import _parser, main
from policy_learnware_v0.v03.preflight import (
    HARD_TODO_IDS,
    FORMAL_PRODUCTION_STAGE_IDS,
    HardTodoEvidence,
    OracleUnlockHandoff,
    PreExperimentFreezeManifest,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.statistics import (
    FormalStatisticsPlan,
    FrozenContrastInputRow,
    FrozenStatisticsInput,
    MultiplicityFamilyPlan,
    StatisticsContrast,
    StatisticsEndpoint,
    V03StatisticsError,
    compute_formal_statistics,
)


def _d(label: str) -> str:
    return sha256_json({"v03-statistics-test": label})


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:synthetic"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _formal_adapter_bindings() -> dict[str, str]:
    return {
        stage_id: formal_stage_adapter_binding_digest(
            stage_id, f"statistics-adapter-{index}", _d(f"adapter-{index}")
        )
        for index, stage_id in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
    }


def _plan() -> FormalStatisticsPlan:
    return FormalStatisticsPlan(
        endpoints=(
            StatisticsEndpoint("selection_return", "normalized_return", True),
            StatisticsEndpoint("selection_cost", "probe_cost", False),
        ),
        contrasts=(
            StatisticsContrast(
                "h_return",
                "POLICY_SELECTION_LINKAGE",
                "selection_return",
                "B0",
                "B1",
            ),
            StatisticsContrast(
                "h_cost",
                "POLICY_SELECTION_LINKAGE",
                "selection_cost",
                "B0",
                "B1",
            ),
            StatisticsContrast(
                "h_temporal_na",
                "POLICY_SELECTION_LINKAGE",
                "selection_return",
                "B0",
                "B1",
            ),
        ),
        multiplicity_families=(
            MultiplicityFamilyPlan(
                "POLICY_SELECTION_LINKAGE",
                ("h_return", "h_cost", "h_temporal_na"),
            ),
        ),
        bootstrap_resamples=128,
        seed_namespace="v03-statistics-synthetic",
    )


def _freeze(plan: FormalStatisticsPlan, *, authority: bool = True) -> PreExperimentFreezeManifest:
    return PreExperimentFreezeManifest(
        freeze_id="v03-statistics-freeze",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=_d("authority") if authority else None,
        review_authority_verified=authority,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=_d("matrix"),
        signal_contrast_plan_digest=_d("signal-contrast-plan"),
        signal_materiality_threshold_digest=_d("signal-materiality-thresholds"),
        formal_signal_readout_plan_digest=_d("formal-signal-readout-plan"),
        preoracle_signal_outcome_plan_digest=_d("preoracle-signal-outcome-plan"),
        signal_identity_registry_digest=_d("identities"),
        signal_execution_protocol_digest=_d("execution"),
        representation_plan_digest=_d("representations"),
        condition_plan_digest=_d("conditions"),
        formal_source_fit_schedule_digest=_d("source-fit"),
        formal_source_membership_digest=_d("source-membership"),
        signal_work_item_graph_digest=_d("work-items"),
        formal_signal_prefix_schedule_digest=_d("signal-prefix-schedule"),
        dynamics_axis_registry_digest=_d("dynamics-axis-registry"),
        public_query_plan_digest=_d("queries"),
        baseline_plan_digest=_d("baselines"),
        statistics_plan_digest=plan.plan_digest,
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=_d("asymmetric-kme"),
        formal_gate_plan_digests=(
            {
                "G03-Attribution": _d("formal-attribution-plan"),
                "G03-Probe": _d("formal-probe-plan"),
                "G03-Market": _d("formal-market-plan"),
            }
            if authority
            else {}
        ),
        formal_stage_request_template_digests=(
            {
                stage_id: _d(f"formal-stage-request:{stage_id}")
                for stage_id in FORMAL_PRODUCTION_STAGE_IDS
            }
            if authority
            else {}
        ),
        hard_todo_evidence=tuple(_todo(item) for item in HARD_TODO_IDS),
        formal_stage_adapter_binding_digests=(
            _formal_adapter_bindings() if authority else {}
        ),
    )


def _observed(hypothesis_id: str, index: int, left: float, right: float) -> FrozenContrastInputRow:
    return FrozenContrastInputRow(
        hypothesis_id=hypothesis_id,
        task_id=f"task-{index // 2}",
        axis_id="axis-a",
        context_id=f"context-{index // 2}",
        observation_id=f"episode-{index}",
        status="OBSERVED",
        left_value=left,
        right_value=right,
    )


def _input(plan: FormalStatisticsPlan, freeze: PreExperimentFreezeManifest) -> FrozenStatisticsInput:
    barrier_digest = _d("public-ranking-barrier")
    run_id = "v03-statistics-run"
    handoff = OracleUnlockHandoff(
        run_id=run_id,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier_digest,
    )
    rows = []
    for index, (left, right) in enumerate(((0.8, 0.4), (0.7, 0.5), (0.9, 0.6), (0.6, 0.3))):
        rows.append(_observed("h_return", index, left, right))
    # Lower cost is better; the registered left method should still have a
    # positive oriented effect after the endpoint direction is applied.
    for index, (left, right) in enumerate(((1.0, 2.0), (1.5, 2.2), (0.8, 1.7), (1.2, 2.5))):
        rows.append(_observed("h_cost", index, left, right))
    for index in range(4):
        rows.append(
            FrozenContrastInputRow(
                hypothesis_id="h_temporal_na",
                task_id=f"task-{index // 2}",
                axis_id="axis-a",
                context_id=f"context-{index // 2}",
                observation_id=f"episode-{index}",
                status="N_A",
                left_value=None,
                right_value=None,
                n_a_reason="one-step-structural-null",
            )
        )
    return FrozenStatisticsInput(
        run_id=run_id,
        preexperiment_freeze_manifest_digest=freeze.freeze_manifest_digest,
        statistics_plan_digest=plan.plan_digest,
        public_ranking_barrier_digest=barrier_digest,
        oracle_unlock_handoff_digest=handoff.handoff_digest,
        oracle_release_receipt_digest=_d("external-oracle-release"),
        oracle_evidence_manifest_digest=_d("oracle-evidence"),
        rows=tuple(rows),
    )


def test_formal_statistics_reuses_paired_bootstrap_max_t_and_holm() -> None:
    plan = _plan()
    freeze = _freeze(plan)
    frozen_input = _input(plan, freeze)
    assert FormalStatisticsPlan.from_dict(plan.to_dict()) == plan
    assert FrozenStatisticsInput.from_dict(frozen_input.to_dict()) == frozen_input

    result = compute_formal_statistics(
        plan=plan,
        freeze_manifest=freeze,
        frozen_input=frozen_input,
    )
    repeated = compute_formal_statistics(
        plan=plan,
        freeze_manifest=freeze,
        frozen_input=frozen_input,
    )
    assert result.result_digest == repeated.result_digest
    assert result.contrast_results["h_return"]["observed_effect"] > 0.0
    assert result.contrast_results["h_cost"]["observed_effect"] > 0.0
    assert result.contrast_results["h_return"]["simultaneous_interval"] is not None
    assert result.contrast_results["h_return"]["holm_adjusted_p_value"] is not None
    assert result.contrast_results["h_return"]["positive_task_count"] == 2
    assert set(result.contrast_results["h_return"]["leave_one_task_out_effects"]) == {
        "task-0",
        "task-1",
    }

    n_a = result.contrast_results["h_temporal_na"]
    assert n_a["status"] == "N_A"
    assert n_a["eligible_row_count"] == 0
    assert n_a["n_a_row_count"] == 4
    assert n_a["coverage"] == 0.0
    assert n_a["raw_p_value"] is None
    family = result.family_results["POLICY_SELECTION_LINKAGE"]
    assert family["registered_family_size"] == 3
    assert family["multiplicity_denominator"] == 2
    assert family["excluded_n_a_hypothesis_ids"] == ["h_temporal_na"]
    assert family["n_a_entered_denominator"] is False


def test_formal_statistics_fails_closed_before_external_authority() -> None:
    plan = _plan()
    unverified = _freeze(plan, authority=False)
    frozen_input = _input(plan, unverified)
    with pytest.raises(V03StatisticsError, match="external pre-experiment authority"):
        compute_formal_statistics(
            plan=plan,
            freeze_manifest=unverified,
            frozen_input=frozen_input,
        )


def test_max_t_family_rejects_different_post_na_leaf_denominators() -> None:
    plan = _plan()
    freeze = _freeze(plan)
    frozen_input = _input(plan, freeze)
    rows = tuple(
        row
        for row in frozen_input.rows
        if not (row.hypothesis_id == "h_cost" and row.observation_id == "episode-3")
    )
    mismatched = replace(frozen_input, rows=rows)
    with pytest.raises(V03StatisticsError, match="identical eligible paired leaves"):
        compute_formal_statistics(
            plan=plan,
            freeze_manifest=freeze,
            frozen_input=mismatched,
        )


def test_na_rows_cannot_smuggle_numeric_values_or_release_without_handoff() -> None:
    with pytest.raises(V03StatisticsError, match="cannot carry numeric"):
        FrozenContrastInputRow(
            "h",
            "task",
            "axis",
            "context",
            "episode",
            "N_A",
            1.0,
            None,
            "not-applicable",
        )

    plan = _plan()
    freeze = _freeze(plan)
    frozen_input = _input(plan, freeze)
    with pytest.raises(V03StatisticsError, match="handoff does not bind"):
        replace(frozen_input, oracle_unlock_handoff_digest=_d("forged-handoff"))


def test_compute_statistics_cli_consumes_only_released_typed_manifests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "compute-statistics" in set(
        _parser()._subparsers._group_actions[0].choices
    )
    plan = _plan()
    freeze = _freeze(plan)
    frozen_input = _input(plan, freeze)
    paths = {
        "--plan": (tmp_path / "plan.json", plan.to_dict()),
        "--freeze-manifest": (tmp_path / "freeze.json", freeze.to_dict()),
        "--frozen-input": (tmp_path / "input.json", frozen_input.to_dict()),
    }
    argv = ["compute-statistics"]
    for option, (path, value) in paths.items():
        path.write_text(json.dumps(value), encoding="utf-8")
        argv.extend((option, str(path)))
    assert main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "FORMAL_STATISTICS_COMPUTED"
    assert output["payload"]["oracle_read_by_cli"] is False
    assert output["payload"]["statistics_result"]["statistics_plan_digest"] == plan.plan_digest

    unverified = _freeze(plan, authority=False)
    paths["--freeze-manifest"][0].write_text(
        json.dumps(unverified.to_dict()), encoding="utf-8"
    )
    unverified_input = _input(plan, unverified)
    paths["--frozen-input"][0].write_text(
        json.dumps(unverified_input.to_dict()), encoding="utf-8"
    )
    assert main(argv) == 1
    blocked = json.loads(capsys.readouterr().err)
    assert blocked["status"] == "BLOCKED"
    assert "external pre-experiment authority" in blocked["payload"]["error"]
