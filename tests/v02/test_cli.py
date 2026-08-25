from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from policy_learnware_v0.hashing import sha256_file, sha256_json
from policy_learnware_v0.v02 import cli, completion
from policy_learnware_v0.v02.audit import artifact_tree_digest
from policy_learnware_v0.v02.axis_catalog import build_candidate_axis_catalog
from policy_learnware_v0.v02.axis_integration import axis_registry_from_config
from policy_learnware_v0.v02.config import load_v02_experiment_config


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _smoke_payload(tmp_path: Path) -> tuple[dict, object]:
    catalog, _ = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    candidate = catalog.entries["cheetah_actuator_gain"]
    payload = {
        "schema": "policy-learnware.v02-experiment-config.v0",
        "experiment_id": "v02-cli-audit-smoke-r0",
        "stage": "audit_smoke",
        "protocol_family_id": "continuous-vector-mdp-v02",
        "tasks": [candidate.task_id],
        "dynamics_axes": {
            candidate.task_id: [
                {
                    "axis_id": candidate.axis_id,
                    "operator_id": candidate.operator_id,
                    "operator_digest": candidate.operator_digest,
                    "leaf_allowlist": [
                        f"_mjx_model.{selection.leaf}"
                        for selection in candidate.selections
                    ],
                    "static_within_episode": True,
                }
            ]
        },
        "source_factors": {
            candidate.task_id: {
                candidate.axis_id: [
                    {
                        "factor_id": "source_low",
                        "value": 0.9,
                        "roles": ["source"],
                        "source_anchor_id": _d("source-low-anchor"),
                        "axis_binding_digest": _d("source-low-binding"),
                    },
                    {
                        "factor_id": "source_nominal",
                        "value": 1.0,
                        "roles": ["source"],
                        "source_anchor_id": _d("source-nominal-anchor"),
                        "axis_binding_digest": None,
                    },
                    {
                        "factor_id": "source_high",
                        "value": 1.1,
                        "roles": ["source"],
                        "source_anchor_id": _d("source-high-anchor"),
                        "axis_binding_digest": _d("source-high-binding"),
                    },
                ]
            }
        },
        "development_targets": [],
        "confirmatory_targets": [],
        "safety_exact_targets": [],
        "primary_algorithm": "PPO",
        "training_steps": 10,
        "training_seeds": [11],
        "checkpoint_rule": "fixed_final_checkpoint",
        "source_eval_episodes": {"selection": 2, "attestation": 3},
        "competence_floor": {candidate.task_id: 0.5},
        "probe_protocol_id": _d("probe-protocol"),
        "probe_prefixes": [1, 2],
        "encoder_eval_prefixes": [1, 2, 4],
        "representation_ids": ["raw_transition_v02", "corro_anchor_supcon_v02"],
        "method_ids": ["B0", "B1", "M02/B5"],
        "primary_endpoint": "pool_regret",
        "noninferiority_margin": 0.01,
        "minimum_effect": 0.05,
        "bootstrap_plan": {
            "resamples": 100,
            "confidence": 0.95,
            "hierarchy": ["task", "axis", "context", "episode_bank"],
            "method": "deterministic_hierarchical_bootstrap",
        },
        "multiple_testing_plan": {
            "simultaneous_interval": "bootstrap_max-T",
            "p_value_adjustment": "holm_bonferroni",
            "alpha": 0.05,
            "families": ["primary_superiority", "nominal_noninferiority"],
        },
        "artifact_root": str((tmp_path / "artifacts").resolve()),
    }
    return payload, candidate


def _write_smoke_config(tmp_path: Path) -> tuple[Path, object]:
    payload, candidate = _smoke_payload(tmp_path)
    path = tmp_path / "v02_smoke.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path, candidate


def test_verify_server_anchor_semantics_preserves_qualified_manifest_leaves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    anchor_id = _d("qualified-leaf-anchor")
    environment_digest = _d("qualified-leaf-environment")
    manifest_digest = _d("qualified-leaf-manifest")
    registry_digest = _d("qualified-leaf-registry")
    binding_digest = _d("qualified-leaf-binding")
    axis = SimpleNamespace(
        axis_id="finger_joint_damping",
        operator_id="scale_selected_model_leaf_v1",
        leaf_allowlist=("_mjx_model.dof_damping",),
    )
    factor = SimpleNamespace(
        source_anchor_id=anchor_id,
        value=0.75,
        is_nominal=False,
        axis_binding_digest=binding_digest,
    )
    config = SimpleNamespace(
        tasks=("FingerTurnEasy",),
        dynamics_axes={"FingerTurnEasy": (axis,)},
        source_factors={"FingerTurnEasy": {axis.axis_id: (factor,)}},
    )
    manifest = SimpleNamespace(
        anchor_id=anchor_id,
        task="FingerTurnEasy",
        nominal=False,
        factor=0.75,
        environment_instance_digest=environment_digest,
        manifest_digest=manifest_digest,
        axis_binding_digest=binding_digest,
        operator=SimpleNamespace(
            axis_id=axis.axis_id,
            operator_id=axis.operator_id,
            axis_registry_digest=registry_digest,
            mutations=(SimpleNamespace(leaf="_mjx_model.dof_damping"),),
        ),
    )

    monkeypatch.setattr(cli, "_server_training_bridge", lambda: (None, None))
    monkeypatch.setattr(cli, "_candidate_catalog_for_config", lambda _config: object())
    monkeypatch.setattr(
        cli,
        "axis_registry_from_config",
        lambda _config, _catalog: SimpleNamespace(digest=registry_digest),
    )
    from server.repro_fpo_ppo_v02.anchor_binding import AnchorManifest

    monkeypatch.setattr(AnchorManifest, "from_path", lambda _path: manifest)

    cli._verify_server_anchor_semantics(
        config,
        anchor_plan={
            anchor_id: {
                "environment_instance_digest": environment_digest,
                "anchor_manifest_digest": manifest_digest,
            }
        },
        anchor_paths={anchor_id: tmp_path / "anchor_manifest.json"},
    )


def _axis_manifest(config_path: Path) -> dict:
    config = load_v02_experiment_config(config_path)
    catalog, _ = build_candidate_axis_catalog((0.9, 1.0, 1.1))
    registry = axis_registry_from_config(config, catalog)
    entry = next(iter(registry.entries.values()))
    records = []
    for factor in entry.factors:
        nominal = factor.value == 1.0
        operator_audit = {
            "schema": "policy-learnware.v02-dynamics-operator-audit.v0",
            "axis_id": entry.axis_id,
            "operator_id": entry.operator_id,
            "operator_version": entry.operator_version,
            "task_id": entry.task_id,
            "factor": factor.value,
            "base_model_digest": _d(f"base:{factor.factor_id}"),
            "shifted_model_digest": (
                _d(f"base:{factor.factor_id}")
                if nominal
                else _d(f"shifted:{factor.factor_id}")
            ),
            "changed_leaves": (
                [] if nominal else sorted(selection.leaf for selection in entry.selections)
            ),
            "unchanged_leaves": ["body_mass", "body_inertia"],
            "selected_element_count": 6,
            "changed_element_count": 0 if nominal else 6,
            "source_object_unchanged": True,
            "exact_allowlist": True,
            "coupling_check": True,
            "finite": True,
            "passed": True,
            "reason": None,
        }
        records.append(
            {
                "task_id": entry.task_id,
                "axis_id": entry.axis_id,
                "factor_id": factor.factor_id,
                "factor_value": factor.value,
                "operator_audit": operator_audit,
                "operator_audit_digest": sha256_json(operator_audit),
                "runtime_checks": {
                    "fresh_instance_isolation": True,
                    "reset_finite": True,
                    "step_finite": True,
                    "jit_reset": True,
                    "jit_step": True,
                },
            }
        )
    return {
        "schema": "policy-learnware.v02-axis-audit-manifest.v0",
        "config_digest": config.config_digest,
        "axis_registry_digest": registry.digest,
        "records": records,
    }


def test_validate_config_is_strict_resumable_and_has_no_joint_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = _write_smoke_config(tmp_path)
    output = tmp_path / "validation.json"
    assert cli.main(
        [
            "validate-config",
            "--config",
            str(config_path),
            "--expect-stage",
            "audit_smoke",
            "--output",
            str(output),
        ]
    ) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["passed"] is True
    assert emitted["stage"] == "audit_smoke"
    assert json.loads(output.read_text(encoding="utf-8")) == emitted

    assert cli.main(
        [
            "validate-config",
            "--config",
            str(config_path),
            "--expect-stage",
            "audit_smoke",
            "--output",
            str(output),
            "--resume",
        ]
    ) == 0
    capsys.readouterr()

    payload, _ = _smoke_payload(tmp_path)
    payload["primary_algorithm"] = "REVIEW_REQUIRED"
    draft = tmp_path / "draft.yaml"
    draft.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    assert cli.main(["validate-config", "--config", str(draft)]) == 2
    capsys.readouterr()
    assert cli.main(["validate-config", "--config", str(draft), "--draft"]) == 0
    draft_result = json.loads(capsys.readouterr().out)
    assert draft_result["passed"] is True
    assert draft_result["executable"] is False

    restricted = tmp_path / "sealed_target_fixture"
    restricted.mkdir()
    restricted_config = restricted / "config.yaml"
    restricted_config.write_bytes(config_path.read_bytes())
    assert cli.main(["validate-config", "--config", str(restricted_config)]) == 2
    assert "joint/sealed" in capsys.readouterr().err

    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {
        "validate-config",
        "audit-environment-abi",
        "audit-axes",
        "freeze-run",
        "plan-training",
        "admit-training-records",
        "evaluate-source-competence",
        "championize-anchors",
        "build-market",
        "collect-probes",
        "build-environment-specs",
        "fit-baselines",
        "run-selectors",
        "compute-metrics",
        "evaluate-gates",
        "audit-information",
        "audit-public",
        "build-report",
        "recompute",
        "audit-recompute",
        "complete",
    }
    assert set(cli.HANDLERS) == set(subparsers.choices)


def test_file_driven_training_to_championization_pipeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = _write_smoke_config(tmp_path)
    config = load_v02_experiment_config(config_path)
    anchor_id = _d("source-nominal-anchor")
    anchors = {
        "schema": "policy-learnware.v02-training-anchor-plan.v0",
        "config_digest": config.config_digest,
        "anchors": {
            anchor_id: {
                "environment_instance_digest": _d("pipeline-environment"),
                "anchor_manifest_digest": _d("pipeline-manifest"),
            }
        },
    }
    trainer = {
        "schema": "policy-learnware.v02-trainer-contract.v0",
        "config_digest": config.config_digest,
        "trainer_config": {"learning_rate": 0.001, "batch_size": 8},
        "trainer_commit": "a" * 40,
        "dependency_digest": _d("pipeline-dependencies"),
        "runtime_digest": _d("pipeline-runtime"),
        "training_protocol_id": _d("pipeline-training-protocol"),
    }
    anchors_path = tmp_path / "anchors.json"
    trainer_path = tmp_path / "trainer.json"
    jobs_path = tmp_path / "jobs.json"
    _write_json(anchors_path, anchors)
    _write_json(trainer_path, trainer)
    plan_args = [
        "plan-training",
        "--config",
        str(config_path),
        "--anchors",
        str(anchors_path),
        "--trainer-contract",
        str(trainer_path),
        "--output",
        str(jobs_path),
    ]
    assert cli.main(plan_args) == 0
    jobs = json.loads(capsys.readouterr().out)
    assert jobs["job_count"] == 1
    assert jobs["jobs"][0]["source_anchor_id"] == anchor_id
    assert cli.main([*plan_args, "--resume"]) == 0
    capsys.readouterr()

    candidate_id = jobs["jobs"][0]["job_id"]
    job = jobs["jobs"][0]
    bundle_path = tmp_path / "bundles" / candidate_id
    bundle_path.mkdir(parents=True)
    _write_json(bundle_path / "bundle_manifest.json", {"job_id": candidate_id})
    attestation = {
        "schema": "policy-learnware.v02-policy-training-attestation.v0",
        "job_id": candidate_id,
        "job_digest": sha256_json(job),
        "attempt_id": "attempt-0001",
        "attempt_number": 1,
        "source_anchor_id": job["source_anchor_id"],
        "anchor_manifest_digest": job["anchor_manifest_digest"],
        "declared_environment_instance_digest": job["environment_instance_digest"],
        "actual_train_environment_instance_digest": job["environment_instance_digest"],
        "actual_eval_environment_instance_digest": job["environment_instance_digest"],
        "operator_digest": _d("pipeline-operator"),
        "model_diff_digest": _d("pipeline-model-diff"),
        "algorithm": job["algorithm"],
        "seed": job["seed"],
        "environment_steps": job["environment_steps"],
        "checkpoint_rule": job["checkpoint_rule"],
        "checkpoint_digests": {"final": _d("pipeline-final-checkpoint")},
        "bundle_digest": _d("pipeline-bundle"),
        "bundle_manifest_digest": _d("pipeline-bundle-manifest"),
        "golden_parity_digest": _d("pipeline-golden-parity"),
        "compiled_parity_digest": _d("pipeline-compiled-parity"),
        "finiteness_audit_digest": _d("pipeline-finiteness"),
        "all_arrays_finite": True,
        "golden_parity_passed": True,
        "compiled_parity_passed": True,
        "trainer_commit": job["trainer_commit"],
        "dependency_digest": job["dependency_digest"],
        "runtime_digest": job["runtime_digest"],
        "hardware_digest": _d("pipeline-hardware"),
        "started_at": "2026-08-24T00:00:00Z",
        "finished_at": "2026-08-24T00:01:00Z",
        "elapsed_seconds": 60.0,
        "status": "succeeded",
        "failure_reason": None,
        "bundle_path": str(bundle_path),
        "server_plan_binding_digest": None,
        "server_training_plan_digest": None,
        "server_job_digest": None,
        "server_attempt_digest": None,
        "server_run_manifest_digest": None,
        "server_training_record_digest": None,
    }
    attestations_path = tmp_path / "attestations.json"
    _write_json(
        attestations_path,
        {
            "schema": "policy-learnware.v02-training-attestations.v0",
            "job_plan_digest": sha256_json(jobs),
            "records": [attestation],
        },
    )
    assert cli.main(
        [
            "admit-training-records",
            "--config",
            str(config_path),
            "--jobs",
            str(jobs_path),
            "--attestations",
            str(attestations_path),
            "--output",
            str(tmp_path / "admitted_records.json"),
        ]
    ) == 0
    admitted = json.loads(capsys.readouterr().out)
    assert admitted["record_count"] == 1
    assert admitted["records"][candidate_id]["attestation"]["status"] == "succeeded"

    bundle_digest = _d("pipeline-bundle")
    rows = [
        {
            "source_anchor_id": anchor_id,
            "candidate_id": candidate_id,
            "bundle_digest": bundle_digest,
            "block": "source_selection",
            "reset_seed": seed,
            "normalized_return": value,
        }
        for seed, value in ((101, 0.8), (102, 0.9))
    ] + [
        {
            "source_anchor_id": anchor_id,
            "candidate_id": candidate_id,
            "bundle_digest": bundle_digest,
            "block": "source_attestation",
            "reset_seed": seed,
            "normalized_return": value,
        }
        for seed, value in ((201, 0.7), (202, 0.8))
    ]
    rows_path = tmp_path / "source_rows.json"
    _write_json(
        rows_path,
        {
            "schema": "policy-learnware.v02-source-episode-rows.v0",
            "config_digest": config.config_digest,
            "rows": rows,
        },
    )
    assert cli.main(
        [
            "evaluate-source-competence",
            "--config",
            str(config_path),
            "--rows",
            str(rows_path),
            "--output",
            str(tmp_path / "source_evidence_audit.json"),
        ]
    ) == 0
    source_audit = json.loads(capsys.readouterr().out)
    assert source_audit["passed"] is True
    assert source_audit["seed_blocks_disjoint"] is True

    champion_inputs = {
        "schema": "policy-learnware.v02-championization-inputs.v0",
        "config_digest": config.config_digest,
        "selection_rows": rows[:2],
        "attestation_rows": rows[2:],
        "competence_floors": {anchor_id: 0.9},
        "competence_mode": "OBSERVE",
        "mean_tolerance": 0.0,
        "lcb_z": None,
        "return_contract_id": _d("normalized-return-contract"),
    }
    champion_inputs_path = tmp_path / "champion_inputs.json"
    _write_json(champion_inputs_path, champion_inputs)
    assert cli.main(
        [
            "championize-anchors",
            "--config",
            str(config_path),
            "--manifest",
            str(champion_inputs_path),
            "--output",
            str(tmp_path / "championization.json"),
        ]
    ) == 0
    champion = json.loads(capsys.readouterr().out)
    assert champion["passed"] is True
    assert champion["competence_mode"] == "OBSERVE"
    assert champion["selected_by_anchor"] == {anchor_id: candidate_id}
    # OBSERVE publishes the independently attested low-q record while keeping
    # the threshold outcome visible and distinct from admission.
    assert champion["competence_records"][anchor_id]["passed"] is False
    assert champion["rejected_anchors"] == {}


def test_audit_axes_recomputes_primitives_and_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = _write_smoke_config(tmp_path)
    manifest = _axis_manifest(config_path)
    manifest_path = tmp_path / "axis_manifest.json"
    _write_json(manifest_path, manifest)
    output = tmp_path / "axis_audit.json"
    assert cli.main(
        [
            "audit-axes",
            "--config",
            str(config_path),
            "--audit-manifest",
            str(manifest_path),
            "--output",
            str(output),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert result["expected_work_units"] == 3
    assert result["violations"] == []

    broken = copy.deepcopy(manifest)
    broken["records"][0]["runtime_checks"]["jit_step"] = False
    broken_path = tmp_path / "axis_manifest_broken.json"
    _write_json(broken_path, broken)
    assert cli.main(
        [
            "audit-axes",
            "--config",
            str(config_path),
            "--audit-manifest",
            str(broken_path),
            "--output",
            str(tmp_path / "axis_audit_broken.json"),
        ]
    ) == 2
    failed = json.loads(capsys.readouterr().out)
    assert failed["passed"] is False
    assert any(
        item["reason"] == "runtime_finite_jit_or_isolation_check_failed"
        for item in failed["violations"]
    )


def test_audit_public_enforces_exact_surface_and_rejects_private_abi(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    public_root = tmp_path / "market_public"
    public_root.mkdir()
    first = "lw-" + "1" * 20
    second = "lw-" + "2" * 20
    market = {
        "schema": "policy-learnware.v02-public-policy-market.v0",
        "policy_market_id": _d("policy-market"),
        "entries": {
            first: {
                "schema": "policy-learnware.v02-public-market-entry.v0",
                "opaque_learnware_id": first,
                "normalized_source_competence": 0.75,
                "tie_break_token": _d("tie-one"),
            },
            second: {
                "schema": "policy-learnware.v02-public-market-entry.v0",
                "opaque_learnware_id": second,
                "normalized_source_competence": 0.5,
                "tie_break_token": _d("tie-two"),
            },
        },
    }
    market_path = public_root / "market.json"
    _write_json(market_path, market)
    rules = {
        "schema": "policy-learnware.v02-public-artifact-rules.v0",
        "rules": [
            {
                "pattern": "market.json",
                "kind": "json",
                "json_keys": ["schema", "policy_market_id", "entries"],
                "npz_members": [],
                "permitted_forbidden_keys": [],
                "permitted_forbidden_string_tokens": [],
                "permitted_forbidden_npz_members": [],
            }
        ],
    }
    rules_path = tmp_path / "public_rules.json"
    _write_json(rules_path, rules)
    assert cli.main(
        [
            "audit-public",
            "--public-root",
            str(public_root),
            "--market-manifest",
            str(market_path),
            "--rules",
            str(rules_path),
            "--output",
            str(tmp_path / "public_audit.json"),
        ]
    ) == 0
    passed = json.loads(capsys.readouterr().out)
    assert passed["passed"] is True
    assert passed["market"]["allowed_entry_fields"] == [
        "normalized_source_competence",
        "opaque_learnware_id",
        "tie_break_token",
    ]

    leaked = copy.deepcopy(market)
    leaked["entries"][first]["runtime_contract"] = {"device": "cpu"}
    _write_json(market_path, leaked)
    assert cli.main(
        [
            "audit-public",
            "--public-root",
            str(public_root),
            "--market-manifest",
            str(market_path),
            "--rules",
            str(rules_path),
            "--output",
            str(tmp_path / "public_audit_leaked.json"),
        ]
    ) == 2
    failed = json.loads(capsys.readouterr().out)
    assert failed["passed"] is False
    assert any(
        item["reason"] == "entry_field_allowlist_mismatch"
        for item in failed["market"]["violations"]
    )


def test_recompute_binds_raw_tree_and_emits_completion_compatible_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path, _ = _write_smoke_config(tmp_path)
    config = load_v02_experiment_config(config_path)
    raw_root = tmp_path / "raw"
    recompute_root = tmp_path / "independent"
    _write_json(raw_root / "episode_rows.json", {"rows": [1, 2, 3]})
    _write_json(raw_root / "timings.json", {"seconds": [0.1, 0.2]})
    source_files = {
        "episode_rows.json": sha256_file(raw_root / "episode_rows.json"),
        "timings.json": sha256_file(raw_root / "timings.json"),
    }
    sections = {}
    for index, name in enumerate(cli.RECOMPUTE_SECTIONS):
        recomputed = recompute_root / f"{index:02d}-{name}.json"
        _write_json(recomputed, {"check": name, "recomputed": True})
        sections[name] = {
            "source_files": source_files,
            "recomputed_file": recomputed.name,
            "expected_output_digest": sha256_file(recomputed),
        }
    plan = {
        "schema": "policy-learnware.v02-independent-recompute-plan.v0",
        "config_digest": config.config_digest,
        "coverage_contract_digest": _d("coverage-contract"),
        "raw_tree_digest": artifact_tree_digest(raw_root),
        "sections": sections,
    }
    plan_path = tmp_path / "recompute_plan.json"
    _write_json(plan_path, plan)
    output = tmp_path / "recompute_report.json"
    arguments = [
        "audit-recompute",
        "--config",
        str(config_path),
        "--manifest",
        str(plan_path),
        "--raw-root",
        str(raw_root),
        "--recompute-root",
        str(recompute_root),
        "--output",
        str(output),
    ]
    assert cli.main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "policy-learnware.v02-independent-recompute-report.v0"
    assert report["passed"] is True
    assert report["precomputed_aggregates_or_gates_consumed"] is False
    assert all(report[name] is True for name in cli.RECOMPUTE_CHECKS)
    assert json.loads(output.read_text(encoding="utf-8")) == report

    assert cli.main([*arguments, "--resume"]) == 0
    capsys.readouterr()

    changed = recompute_root / f"00-{cli.RECOMPUTE_SECTIONS[0]}.json"
    _write_json(changed, {"check": "tampered"})
    broken_arguments = arguments[:-1] + [str(tmp_path / "recompute_failed.json")]
    assert cli.main(broken_arguments) == 2
    failed = json.loads(capsys.readouterr().out)
    assert failed["passed"] is False
    assert failed["full_digest_coverage"] is False
    assert any("output digest mismatch" in item for item in failed["errors"])


def test_complete_delegates_to_strict_completion_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "freeze.yaml"
    artifact_root = tmp_path / "artifacts"
    experiment_id = "v02-cli-freeze-ready-r0"
    experiment_root = artifact_root / experiment_id
    gate_state = experiment_root / "analysis" / "gates" / "v02_gate_state.json"
    recompute = experiment_root / "analysis" / "recompute_audit.json"
    config.write_text("placeholder", encoding="utf-8")
    _write_json(gate_state, {})
    _write_json(recompute, {})
    output = experiment_root / "completion_manifest.json"
    calls = []

    monkeypatch.setattr(
        cli,
        "_load_config_for_command",
        lambda path: SimpleNamespace(
            artifact_root=str(artifact_root), experiment_id=experiment_id
        ),
    )

    def fake_complete(**kwargs):
        calls.append(kwargs)
        return {
            "schema": "policy-learnware.v02-completion-manifest.v0",
            "passed": True,
            "status": "READY_FOR_V03_JOINT_CONFIRMATORY",
            "paper1_empirical_claim_authorized": False,
        }

    monkeypatch.setattr(completion, "complete_v02_from_files", fake_complete)
    assert cli.main(
        [
            "complete",
            "--config",
            str(config),
            "--gate-state",
            str(gate_state),
            "--recompute-audit",
            str(recompute),
            "--theory-status",
            "MINIMAL_FINITE_POOL_CLOSED",
            "--literature-novelty-audit-status",
            "PENDING",
            "--output",
            str(output),
        ]
    ) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["paper1_empirical_claim_authorized"] is False
    assert calls == [
        {
            "config_path": config.resolve(),
            "gate_state_path": gate_state.resolve(),
            "recompute_audit_path": recompute.resolve(),
            "output_path": output.resolve(),
            "theory_status": "MINIMAL_FINITE_POOL_CLOSED",
            "literature_novelty_audit_status": "PENDING",
            "resume": False,
        }
    ]
