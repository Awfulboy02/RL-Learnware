from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.baselines import (
    FORMAL_MODE,
    REQUIRED_BASELINE_METHOD_IDS,
)
from policy_learnware_v0.v03.cli import CLI_VERSION, _parser, main
from policy_learnware_v0.v03.preflight import (
    HARD_TODO_IDS,
    FORMAL_PRODUCTION_STAGE_IDS,
    HardTodoEvidence,
    IndependentRecomputeAttestation,
    PreExperimentFreezeManifest,
    PublicRankingBarrier,
    PublicRankingPublication,
    PublicQueryPlan,
    formal_baseline_input_plan_digest,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.schemas import ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST
from policy_learnware_v0.v03.windowing import WindowingProtocol


def _config(path: Path) -> None:
    protocol = WindowingProtocol(4, 2, "mean", True)
    payload = {
        "schema": "policy-learnware.v03-foundation-config.v0",
        "development_id": "v03-cli-test",
        "stage": "foundation_development",
        "protocol_id": sha256_json({"protocol": "test"}),
        "task_private_ids": ["task-a", "task-b"],
        "artifact_root": str(path.parent / "artifacts"),
        "anonymous_public_allowlist": sorted(ANONYMOUS_SELECTOR_ENTRY_ALLOWLIST),
        "window_protocol": {
            "window_length": 4,
            "stride": 2,
            "pooling": "mean",
            "pad_final_window": True,
            "protocol_id": protocol.window_protocol_digest,
        },
        "primary_freeze": {
            "query_mode": "QUERY_EMPIRICAL",
            "selector_mode": "distance_only",
            "pool_scope": "anonymous_global",
            "opaque_learnware_field": "opaque_learnware_id",
            "opaque_query_field": "opaque_query_id",
            "oracle_owner": "policy-learnware-paper1",
        },
        "review_decisions_digest": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_help_is_a_real_command_surface_and_does_not_run_acceptance(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "accept-numeric" in output
    assert "accept-prelarge" in output
    assert "intake-v02-policy-pool" in output
    assert '"checks"' not in output
    assert _parser().parse_args(["accept-numeric"]).command == "accept-numeric"
    commands = set(_parser()._subparsers._group_actions[0].choices)
    assert {
        "accept-numeric",
        "accept-prelarge",
        "build-canonical-banks",
        "build-market",
        "build-query-specs",
        "build-signal-atlas",
        "build-source-specs",
        "build-transition-views",
        "collect-source-receipts",
        "complete",
        "compute-statistics",
        "fit-representation-controls",
        "fit-baselines",
        "freeze-preexperiment",
        "intake-v02-policy-pool",
        "recompute",
        "replay-legacy-attribution",
        "run-public-rankings",
        "unlock-oracle",
        "validate-config",
    }.issubset(commands)


def test_validate_config_and_numeric_acceptance_emit_canonical_json(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03.json"
    _config(config)
    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "VALID"
    assert validated["payload"]["formal_scope_ready"] is False
    assert validated["payload"]["review_digest_status"] == "NOT_DECLARED"
    gate = validated["payload"]["encoder_extension_gate"]
    assert gate == {
        "enabled": False,
        "migration_target": None,
        "optional_asset_requirements_active": False,
        "completion_eligible": False,
        "confirmatory_artifact_access": False,
        "formal_authority": False,
    }

    assert main(["accept-numeric"]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["status"] == "ENGINEERING_PASS"
    assert accepted["payload"]["passed"] is True

    assert main(["accept-prelarge"]) == 0
    prelarge = json.loads(capsys.readouterr().out)
    assert (
        prelarge["status"]
        == "ENGINEERING_COMPONENTS_PASS_FORMAL_FREEZE_PENDING"
    )
    assert prelarge["payload"]["signal_matrix_logical_cells"] == 39
    assert prelarge["payload"]["signal_matrix_numeric_cells"] == 37
    assert prelarge["payload"]["formal_run_authorized"] is False


def test_validate_config_does_not_self_sign_declared_review_digest(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-formal.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["stage"] = "formal_freeze"
    payload["review_decisions_digest"] = sha256_json({"review": "declared-only"})
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "VALID"
    assert validated["payload"]["review_digest_status"] == "DECLARED_UNVERIFIED"
    assert validated["payload"]["formal_scope_ready"] is False


def test_validate_config_reports_opted_in_v04_gate_without_granting_authority(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-v04-gate.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["encoder_extension_gate"] = {
        "enabled": True,
        "migration_target": "v0.4",
        "authority_digest": sha256_json({"v04": "migration-decision"}),
    }
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["payload"]["encoder_extension_gate"] == {
        "enabled": True,
        "migration_target": "v0.4",
        "optional_asset_requirements_active": False,
        "completion_eligible": False,
        "confirmatory_artifact_access": False,
        "formal_authority": False,
    }
    assert validated["payload"]["formal_scope_ready"] is False


def test_validate_config_rejects_enabled_extension_gate_at_formal_freeze(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-formal-enabled-gate.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload.update(
        {
            "stage": "formal_freeze",
            "review_decisions_digest": sha256_json({"review": "declared-only"}),
            "encoder_extension_gate": {
                "enabled": True,
                "migration_target": "v0.4",
                "authority_digest": sha256_json({"v04": "migration-decision"}),
            },
        }
    )
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 1
    blocked = json.loads(capsys.readouterr().err)
    assert blocked["status"] == "BLOCKED"
    assert "encoder_extension_gate.enabled=false" in blocked["payload"]["error"]


def test_validate_config_explicit_disabled_gate_is_asset_free(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / "v03-disabled-gate.json"
    _config(config)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["encoder_extension_gate"] = {"enabled": False}
    config.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["validate-config", str(config)]) == 0
    validated = json.loads(capsys.readouterr().out)
    gate = validated["payload"]["encoder_extension_gate"]
    assert gate["enabled"] is False
    assert gate["optional_asset_requirements_active"] is False
    assert gate["confirmatory_artifact_access"] is False
    assert gate["formal_authority"] is False


def test_production_intake_parser_has_no_trust_anchor_override() -> None:
    options = {
        option
        for action in _parser()._subparsers._group_actions[0].choices[
            "intake-v02-policy-pool"
        ]._actions
        for option in action.option_strings
    }
    assert "--trust-anchor" not in options
    assert CLI_VERSION == "0.3.0"


def test_prelarge_dry_run_plans_publish_immutably_without_starting_work(
    tmp_path: Path, capsys
) -> None:
    common = [
        "--dry-run",
        "--artifacts-root",
        str(tmp_path / "artifacts"),
        "--development-id",
        "v03-prelarge-cli",
    ]
    assert main(["build-signal-atlas", *common]) == 0
    atlas = json.loads(capsys.readouterr().out)
    assert atlas["status"] == "DRY_RUN_READY"
    assert atlas["payload"]["plan"]["logical_cell_count"] == 39
    assert atlas["payload"]["plan"]["numeric_cell_count"] == 37
    assert atlas["payload"]["large_experiment_executed"] is False

    assert main(["build-signal-atlas", *common, "--resume"]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["payload"]["artifact_sha256"] == atlas["payload"]["artifact_sha256"]

    assert main(["fit-representation-controls", *common]) == 0
    representations = json.loads(capsys.readouterr().out)
    assert representations["payload"]["fit_job_count"] == 45
    assert representations["payload"]["r5_fit_job_count"] == 36
    assert representations["payload"]["r5l_fit_job_count"] == 9

    assert main(["fit-baselines", *common]) == 0
    baselines = json.loads(capsys.readouterr().out)
    assert baselines["payload"]["optional_method_states"] == {
        "B4c": "DISABLED",
        "B6": "DISABLED",
    }
    assert baselines["payload"]["confirmatory_oracle_access"] is False


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=sha256_json({"todo": todo_id, "kind": "contract"}),
        implementation_digest=sha256_json({"todo": todo_id, "kind": "implementation"}),
        unit_test_evidence_digest=sha256_json({"todo": todo_id, "kind": "unit"}),
        synthetic_fixture_evidence_digest=sha256_json({"todo": todo_id, "kind": "fixture"}),
        cpu_integration_evidence_digest=sha256_json({"todo": todo_id, "kind": "cpu"}),
    )


def _formal_adapter_bindings() -> dict[str, str]:
    return {
        stage_id: formal_stage_adapter_binding_digest(
            stage_id, f"cli-adapter-{index}", sha256_json({"cli-adapter": index})
        )
        for index, stage_id in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
    }


def _preexperiment_manifest() -> PreExperimentFreezeManifest:
    digest = lambda label: sha256_json({"cli-preflight": label})
    return PreExperimentFreezeManifest(
        freeze_id="v03-cli-preexperiment",
        config_bytes_digest=digest("config"),
        implementation_tree_digest=digest("tree"),
        clean_commit_digest=digest("commit"),
        review_decisions_digest=digest("review"),
        review_authority_receipt_digest=None,
        review_authority_verified=False,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=digest("roles"),
        canonicalizer_registry_digest=digest("canonicalizer"),
        signal_matrix_digest=digest("matrix"),
        signal_contrast_plan_digest=digest("signal-contrast-plan"),
        signal_materiality_threshold_digest=digest("signal-materiality-thresholds"),
        formal_signal_readout_plan_digest=digest("formal-signal-readout-plan"),
        preoracle_signal_outcome_plan_digest=digest("preoracle-signal-outcome-plan"),
        signal_identity_registry_digest=digest("signal-identities"),
        signal_execution_protocol_digest=digest("signal-execution"),
        representation_plan_digest=digest("representations"),
        condition_plan_digest=digest("conditions"),
        formal_source_fit_schedule_digest=digest("source-fit-schedule"),
        formal_source_membership_digest=digest("source-membership"),
        signal_work_item_graph_digest=digest("signal-work-items"),
        formal_signal_prefix_schedule_digest=digest("signal-prefix-schedule"),
        dynamics_axis_registry_digest=digest("dynamics-axis-registry"),
        public_query_plan_digest=digest("public-query-plan"),
        baseline_plan_digest=digest("baselines"),
        statistics_plan_digest=digest("statistics"),
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=digest("kme"),
        formal_gate_plan_digests={},
        formal_stage_request_template_digests={},
        hard_todo_evidence=tuple(_todo(item) for item in HARD_TODO_IDS),
    )


def _cli_publication(method_id: str, query_id: str) -> PublicRankingPublication:
    digest = lambda label: sha256_json({"cli-ranking": label})
    return PublicRankingPublication(
        method_id=method_id,
        opaque_query_id=query_id,
        ranking_digest=digest(f"{method_id}:{query_id}:ranking"),
        query_spec_digest=digest(f"{query_id}:query-spec"),
        probe_dataset_digest=digest(f"{query_id}:probe"),
        target_evidence_digest=digest(f"{query_id}:target-evidence"),
        cost_digest=digest(f"{method_id}:{query_id}:cost"),
        policy_market_id=digest("policy-market"),
        representation_index_digest=digest(f"{method_id}:representation-index"),
        selector_view_digest=digest(f"{method_id}:selector-view"),
        evidence_contract_digest=digest(f"{method_id}:evidence-contract"),
        selector_artifact_digest=digest(f"{method_id}:selector-artifact"),
        development_freeze_digest=digest("development-freeze"),
        query_input_digest=digest(f"{method_id}:{query_id}:query-input"),
        query_mode="QUERY_EMPIRICAL",
        execution_mode=FORMAL_MODE,
        development_context_count=24,
    )


def test_freeze_oracle_handoff_and_recompute_commands_remain_fail_closed(
    tmp_path: Path, capsys
) -> None:
    manifest = _preexperiment_manifest()
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    assert (
        main(
            [
                "freeze-preexperiment",
                "--manifest",
                str(manifest_path),
                "--artifacts-root",
                str(tmp_path / "artifacts"),
                "--development-id",
                "v03-preflight-cli",
            ]
        )
        == 0
    )
    frozen = json.loads(capsys.readouterr().out)
    assert frozen["status"] == "ENGINEERING_READY_REVIEW_UNVERIFIED"
    assert frozen["payload"]["formal_run_authorized"] is False

    authorized_manifest = replace(
        manifest,
        review_authority_receipt_digest=sha256_json({"external": "receipt"}),
        review_authority_verified=True,
        formal_gate_plan_digests={
            "G03-Attribution": sha256_json({"formal-gate": "attribution"}),
            "G03-Probe": sha256_json({"formal-gate": "probe"}),
            "G03-Market": sha256_json({"formal-gate": "market"}),
        },
        formal_stage_request_template_digests={
            stage_id: sha256_json({"formal-stage-request": stage_id})
            for stage_id in FORMAL_PRODUCTION_STAGE_IDS
        },
        formal_stage_adapter_binding_digests=_formal_adapter_bindings(),
    )
    authorized_path = tmp_path / "freeze-authorized.json"
    authorized_path.write_text(
        json.dumps(authorized_manifest.to_dict()), encoding="utf-8"
    )
    assert (
        main(
            [
                "freeze-preexperiment",
                "--manifest",
                str(authorized_path),
                "--artifacts-root",
                str(tmp_path / "artifacts-authorized"),
                "--development-id",
                "v03-cannot-self-authorize",
            ]
        )
        == 1
    )
    blocked = json.loads(capsys.readouterr().err)
    assert blocked["status"] == "BLOCKED"
    assert "cannot accept or mint" in blocked["payload"]["error"]

    regimes = {}
    for index in range(66):
        regime = (
            "EXACT"
            if index < 30
            else "INTERPOLATION"
            if index < 54
            else "EXTRAPOLATION"
        )
        regimes[f"v03q-{index:032x}"] = regime
    alias_digest = sha256_json({"aliases": 1})
    query_plan = PublicQueryPlan(
        regime_by_opaque_query_id=regimes,
        query_alias_manifest_digest=alias_digest,
    )
    query_ids = query_plan.opaque_query_ids
    publications = tuple(
        _cli_publication(method_id, query_id)
        for method_id in REQUIRED_BASELINE_METHOD_IDS
        for query_id in query_ids
    )
    barrier_manifest = replace(
        authorized_manifest,
        public_query_plan_digest=query_plan.plan_digest,
        baseline_plan_digest=formal_baseline_input_plan_digest(
            publications,
            expected_opaque_query_ids=query_ids,
            query_alias_manifest_digest=alias_digest,
        ),
    )
    barrier = PublicRankingBarrier(
        run_id="v03-cli-formal-run",
        freeze_manifest=barrier_manifest,
        query_plan=query_plan,
        expected_opaque_query_ids=query_ids,
        expected_method_ids=REQUIRED_BASELINE_METHOD_IDS,
        publications=publications,
        query_alias_manifest_digest=alias_digest,
            preoracle_signal_outcome_manifest_digest=sha256_json(
                {"cli-preflight": "preoracle-signal-manifest"}
            ),
    )
    barrier_path = tmp_path / "barrier.json"
    barrier_path.write_text(json.dumps(barrier.to_dict()), encoding="utf-8")
    assert main(["unlock-oracle", "--public-ranking-barrier", str(barrier_path)]) == 0
    unlock = json.loads(capsys.readouterr().out)
    assert unlock["status"] == "HANDOFF_REQUIRED"
    assert unlock["payload"]["oracle_unlocked_by_v03"] is False
    assert unlock["payload"]["handoff"]["requested_owner"] == "policy-learnware-paper1"

    recompute = IndependentRecomputeAttestation(
        run_id="v03-cli-recompute-run",
        freeze_manifest_digest=sha256_json({"freeze": 1}),
        public_ranking_barrier_digest=sha256_json({"barrier": 1}),
        formal_statistics_result_digest=sha256_json({"result": 1}),
        raw_input_manifest_digest=sha256_json({"raw": 1}),
        primary_artifact_root_digest=sha256_json({"root": 1}),
        recompute_artifact_root_digest=sha256_json({"root": 2}),
        primary_result_digest=sha256_json({"result": 1}),
        recompute_result_digest=sha256_json({"result": 1}),
        primary_process_nonce_digest=sha256_json({"process": 1}),
        recompute_process_nonce_digest=sha256_json({"process": 2}),
    )
    recompute_path = tmp_path / "recompute.json"
    recompute_path.write_text(json.dumps(recompute.to_dict()), encoding="utf-8")
    assert main(["recompute", "--attestation", str(recompute_path)]) == 0
    recomputed = json.loads(capsys.readouterr().out)
    assert recomputed["status"] == "INDEPENDENT_RECOMPUTE_MATCH"
