from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.rkme.reducer import ReducedRKME
from policy_learnware_v0.v02.artifacts import V02AssetError
from policy_learnware_v0.v03.transition_views import (
    TransitionBank,
    V_DELTA_ONLY,
    apply_transition_view,
)
from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    BudgetLedger,
    RewardFreeProbe,
    seal_rankings,
    tie_break_key,
)
from server.repro_fpo_ppo_v03.development_baseline_runner import _distance, _empirical
from server.repro_fpo_ppo_v04a.bpr_runner import (
    ALL_FP_METHODS,
    CROSS_BACKEND_PARITY,
    GateFailure,
    SCHEMA,
    V04ARunnerError,
    _load_config,
    _deployment_action_audit,
    _method_cards,
    _origin_pool_acceptance,
    _origin_parity_receipt,
    _parser,
    _publish,
    _publish_jsonl,
    _read_jsonl,
    _utility_matrix,
    fit_source,
    main,
    oracle_evaluate,
    prepare,
    raw_delta_task5_scores,
    score_fp,
    seal_stage,
    smoke_fp,
)
from server.repro_fpo_ppo_v04a import bpr_runner as runner_module


REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY / "configs" / "v04a_bayesian_reuse.yaml"


def _prepared_run(
    run_dir: Path,
    scoring_manifest: dict | None = None,
    run_fields: dict | None = None,
) -> None:
    config, digest = _load_config(CONFIG)
    run = {
        "schema": SCHEMA,
        "stage": "prepare",
        "status": "PREPARED",
        "config": config,
        "config_digest": digest,
    }
    if scoring_manifest is not None:
        run["scoring_manifest_payload_digest"] = sha256_json(scoring_manifest)
        _publish(run_dir / "scoring_manifest.json", scoring_manifest)
    if run_fields is not None:
        run.update(run_fields)
    _publish(
        run_dir / "run.json",
        run,
    )


def _seal_fixture() -> tuple[dict, list[dict]]:
    _, config_digest = _load_config(CONFIG)
    contexts: list[dict] = []
    tasks: dict[str, dict] = {}
    rankings: list[dict] = []
    for task_index in range(6):
        task_id = f"Task-{task_index}"
        candidates = [f"{task_id}-policy-{index}" for index in range(5)]
        tasks[task_id] = {
            "task_id": task_id,
            "source_type_ids": [f"{task_id}-source-{index}" for index in range(5)],
            "paired_policy_by_type": {
                f"{task_id}-source-{index}": candidate
                for index, candidate in enumerate(candidates)
            },
            "candidate_ids": candidates,
            "candidate_bundle_digests": {
                candidate: sha256_json({"bundle": candidate})
                for candidate in candidates
            },
            "raw_tie_break_tokens": {
                candidate: sha256_json({"raw-tie": candidate})
                for candidate in candidates
            },
        }
        for source_index in range(5):
            context_id = f"{task_id}-source-{source_index}"
            contexts.append(
                {
                    "context_id": context_id,
                    "role": "source",
                    "task_id": task_id,
                    "reward_free_npz": f"reward_free_banks/{context_id}.npz",
                    "reward_free_npz_sha256": sha256_json({"probe": context_id}),
                    "probe_membership_digest": sha256_json({"membership": context_id}),
                    "episode_count": 32,
                    "visible_transitions_per_episode": 64,
                }
            )
        for context_index in range(4):
            context_id = f"{task_id}-development-{context_index}"
            membership_digest = sha256_json({"membership": context_id})
            contexts.append(
                {
                    "context_id": context_id,
                    "role": "development",
                    "task_id": task_id,
                    "reward_free_npz": f"reward_free_banks/{context_id}.npz",
                    "reward_free_npz_sha256": sha256_json({"probe": context_id}),
                    "probe_membership_digest": membership_digest,
                    "episode_count": 32,
                    "visible_transitions_per_episode": 64,
                }
            )
            for budget in BUDGET_EPISODES:
                budget_ledger = BudgetLedger.for_budget(budget).to_dict()
                flat_ledger = {
                    key: value
                    for key, value in budget_ledger.items()
                    if key != "schema"
                }
                for method_id in ALL_FP_METHODS:
                    expected_ties = (
                        tasks[task_id]["raw_tie_break_tokens"]
                        if method_id == "RAW_DELTA_TASK5"
                        else {
                            candidate: tie_break_key(config_digest, candidate)
                            for candidate in candidates
                        }
                    )
                    ranking = [
                        {
                            "rank": rank,
                            "opaque_candidate_id": candidate,
                            "score": float(6 - rank),
                            "tie_break_token": expected_ties[candidate],
                        }
                        for rank, candidate in enumerate(candidates, 1)
                    ]
                    rankings.append(
                        {
                            "schema": SCHEMA,
                            "stage": "PUBLIC_RANKING_PRE_ORACLE",
                            "context_id": context_id,
                            "context_role": "development",
                            "task_id": task_id,
                            "method_id": method_id,
                            "method_version": "0.4a.0",
                            "access_track": "BI0-FP-RF",
                            "candidate_scope": "TASK_5",
                            "faithfulness": "test-fixture",
                            "source_evidence_privileges": "source-only test fixture",
                            "budget_episodes": budget,
                            "probe_membership_digest": membership_digest,
                            **flat_ledger,
                            "budget_ledger": budget_ledger,
                            "selected_opaque_candidate_id": candidates[0],
                            "score_semantics": "test-score",
                            "ranking": ranking,
                            "runtime_seconds": 0.001,
                            "peak_memory_ru_maxrss": 1,
                            "status": "OK",
                        }
                    )
    scoring_manifest = {
        "schema": "policy-learnware.v04a-sanitized-scoring-manifest.v1",
        "access_track": "BI0-FP-RF",
        "contains_reward_or_done": False,
        "contains_target_construction_metadata": False,
        "contexts": contexts,
        "tasks": tasks,
    }
    return scoring_manifest, rankings


def _write_utility_evidence(
    root: Path,
    *,
    task_id: str,
    context_id: str,
    candidates: tuple[str, ...],
    bundle_digests: dict[str, str],
    omit_episode_returns_for: str | None = None,
) -> None:
    for candidate_index, candidate in enumerate(candidates):
        episode_returns = [500.0 + candidate_index] * 50
        record = {
            "schema": "policy-learnware.v03-minimal-development-baseline.v0",
            "stage": "PRIVATE_ORACLE",
            "context_id": context_id,
            "task_id": task_id,
            "opaque_learnware_id": candidate,
            "bundle_digest": bundle_digests[candidate],
            "status": "OK",
            "executed": True,
            "horizon": 1000,
            "episode_returns": episode_returns,
            "mean_return": float(np.mean(episode_returns)),
            "normalized_mean_return": float(np.mean(episode_returns)) / 1000.0,
            "reset_seeds": list(range(730000, 730050)),
            "policy_seeds": list(range(1_730_003, 1_730_053)),
        }
        if candidate == omit_episode_returns_for:
            record.pop("episode_returns")
        _publish(root / context_id / f"{candidate}.json", record)


def test_prepare_missing_real_assets_fails_closed_with_metadata(tmp_path: Path) -> None:
    raw_root = tmp_path / "local-v031"
    _publish(
        raw_root / "views" / "delta_action" / "config.json",
        {
            "view_id": "V_DELTA_ONLY",
            "source_count": 30,
            "query_count": 24,
            "protocol_id": "frozen-protocol",
        },
    )
    run_dir = tmp_path / "new-v04a-run"
    result = prepare(
        argparse.Namespace(
            config=CONFIG,
            run_dir=run_dir,
            context_index=tmp_path / "missing-context-index.json",
            public_policy_market=tmp_path / "missing-public-market.json",
            deployment_private_registry=tmp_path / "missing-private-registry.json",
            origin_pool_acceptance=tmp_path / "missing-pool-acceptance.json",
            raw_delta_root=raw_root,
            fpo_root=tmp_path / "missing-fpo-runtime",
            source_utility_root=tmp_path / "missing-source-utility",
        )
    )

    assert result["status"] == "NO_GO_REQUIRED_ASSETS_ABSENT"
    assert result["formal"] is False
    assert (
        result["available_input_metadata"]["raw_delta_root"]["view_id"]
        == "V_DELTA_ONLY"
    )
    assert (
        result["available_input_metadata"]["raw_delta_root"][
            "source_rkme_count_present"
        ]
        == 0
    )
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "asset_census.json").is_file()


def test_no_go_prepare_returns_nonzero_process_status(tmp_path: Path) -> None:
    result = main(
        [
            "prepare",
            "--config",
            str(CONFIG),
            "--run-dir",
            str(tmp_path / "failed-run"),
            "--context-index",
            str(tmp_path / "missing-context.json"),
            "--public-policy-market",
            str(tmp_path / "missing-public.json"),
            "--deployment-private-registry",
            str(tmp_path / "missing-private.json"),
            "--origin-pool-acceptance",
            str(tmp_path / "missing-pool-acceptance.json"),
            "--raw-delta-root",
            str(tmp_path / "missing-raw"),
            "--fpo-root",
            str(tmp_path / "missing-fpo-runtime"),
            "--source-utility-root",
            str(tmp_path / "missing-source-utility"),
        ]
    )
    assert result == 2


def test_origin_parity_receipt_remains_digest_bound_at_one_e_minus_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id = "v02j-fixture"
    attempt = tmp_path / "jobs" / job_id / "attempt_001"
    bundle = attempt / "checkpoints" / "outer_000001"
    bundle.mkdir(parents=True)
    golden = {
        "passed": True,
        "raw_checked": True,
        "sample_count": 8,
        "atol": 1.0e-6,
        "rtol": 1.0e-6,
    }
    golden["report_digest"] = sha256_json(golden)
    compiled = {
        "passed": True,
        "next_keys_equal": True,
        "sample_count": 2,
        "atol": 1.0e-6,
        "rtol": 1.0e-6,
    }
    compiled["report_digest"] = sha256_json(compiled)
    record = {
        "schema": "policy-learnware.v02-training-record.v1",
        "state": "recovered",
        "promoted_outer_iteration": 1,
        "promoted_environment_steps": 100,
        "checkpoint_bundles": [
            {
                "path": str(bundle),
                "bundle_digest": "a" * 64,
                "golden_parity": golden,
                "compiled_parity": compiled,
            }
        ],
    }
    record["record_digest"] = sha256_json(record)
    _publish(attempt / "training_record.json", record)
    _publish(
        attempt / "status.json",
        {
            "state": "recovered",
            "training_record_digest": record["record_digest"],
            "promoted_outer_iteration": 1,
            "promoted_environment_steps": 100,
        },
    )
    metadata = SimpleNamespace(
        bundle_dir=bundle,
        bundle_digest="a" * 64,
        outer_iteration=1,
        environment_steps=100,
        training_seed=2,
        provenance={
            "job_digest": "b" * 64,
            "attempt_digest": "c" * 64,
            "anchor_manifest_digest": "d" * 64,
            "environment_instance_digest": "e" * 64,
            "config_digest": "f" * 64,
            "execution_purpose": "test",
        },
    )
    accepted_cell = {
        "resolution": "direct_terminal_record",
        "job_id": job_id,
        "job_digest": "b" * 64,
        "attempt_digest": "c" * 64,
        "bundle_path": str(bundle),
        "bundle_digest": "a" * 64,
        "outer_iteration": 1,
        "environment_steps": 100,
        "seed": 2,
        "training_record_digest": record["record_digest"],
        "terminal_record_state": "recovered",
        "golden_parity_digest": golden["report_digest"],
        "compiled_parity_digest": compiled["report_digest"],
    }
    monkeypatch.setattr(
        runner_module, "validate_success_record", lambda *a, **k: record
    )
    assert _origin_parity_receipt(metadata, accepted_cell)["status"] == "PASS"
    historical_bundle = (
        Path("/historical") / tmp_path.name / bundle.relative_to(tmp_path)
    )
    assert (
        _origin_parity_receipt(
            metadata,
            {**accepted_cell, "bundle_path": str(historical_bundle)},
            resolver=SimpleNamespace(resolve=lambda _value: bundle.resolve()),
        )["status"]
        == "PASS"
    )
    with pytest.raises(GateFailure, match="no verified v02 relocation"):
        _origin_parity_receipt(
            metadata,
            {**accepted_cell, "bundle_path": str(historical_bundle)},
            resolver=SimpleNamespace(
                resolve=lambda _value: (_ for _ in ()).throw(
                    V02AssetError("unknown recorded path")
                )
            ),
        )

    (attempt / "status.json").unlink()
    _publish(
        attempt / "status.json",
        {
            "state": "completed",
            "training_record_digest": record["record_digest"],
            "promoted_outer_iteration": 1,
            "promoted_environment_steps": 100,
        },
    )
    with pytest.raises(GateFailure, match="training status"):
        _origin_parity_receipt(metadata, accepted_cell)
    (attempt / "status.json").unlink()
    _publish(
        attempt / "status.json",
        {
            "state": "recovered",
            "training_record_digest": record["record_digest"],
            "promoted_outer_iteration": 1,
            "promoted_environment_steps": 100,
        },
    )

    tampered = dict(record)
    tampered["checkpoint_bundles"] = [
        {
            **record["checkpoint_bundles"][0],
            "golden_parity": {**golden, "passed": False},
        }
    ]
    (attempt / "training_record.json").unlink()
    _publish(attempt / "training_record.json", tampered)
    monkeypatch.setattr(
        runner_module, "validate_success_record", lambda *a, **k: tampered
    )
    with pytest.raises(GateFailure, match="origin parity"):
        _origin_parity_receipt(metadata, accepted_cell)


def test_origin_pool_acceptance_replays_canonical_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff = tmp_path / "experiment" / "policy_pool_handoff_fixture"
    acceptance_path = handoff / "policy_pool_acceptance.json"
    promotion_path = handoff / "compiled_parity_promotions.json"
    plan_path = (
        tmp_path
        / "experiment"
        / "training_private"
        / "plans"
        / "server_training_plan.json"
    )
    (tmp_path / "experiment" / "training_private" / "server_runs").mkdir(parents=True)
    stored = {
        "schema": "policy-learnware.v02-policy-pool-acceptance.v0",
        "decision": "PASS",
        "accepted_at": "frozen",
        "direct_terminal_record_count": 84,
        "compiled_parity_fallback_promotion_count": 6,
        "cells": {"v02j-fixture": {"resolution": "direct_terminal_record"}},
    }
    stored["report_digest"] = sha256_json(stored)
    _publish(acceptance_path, stored)
    _publish(promotion_path, {"manifest_digest": "a" * 64})
    _publish(plan_path, {"plan": "fixture"})
    acceptance_path.chmod(0o444)
    promotion_path.chmod(0o444)
    replayed = {**stored, "accepted_at": "replayed", "report_digest": "b" * 64}
    resolver = SimpleNamespace(
        layout=SimpleNamespace(
            root=tmp_path / "artifacts",
            frozen_acceptance=acceptance_path,
            promotions=promotion_path,
            server_plan=plan_path,
            runs_root=tmp_path / "experiment" / "training_private" / "server_runs",
        )
    )
    monkeypatch.setattr(
        runner_module,
        "replay_relocated_policy_pool_acceptance",
        lambda **kwargs: replayed,
    )

    cells, receipt = _origin_pool_acceptance(acceptance_path, resolver=resolver)
    assert cells == stored["cells"]
    assert receipt["canonical_replay"] == "PASS"

    monkeypatch.setattr(
        runner_module,
        "replay_relocated_policy_pool_acceptance",
        lambda **kwargs: (_ for _ in ()).throw(
            runner_module.V02ContractError("canonical replay differs")
        ),
    )
    with pytest.raises(GateFailure, match="cannot revalidate"):
        _origin_pool_acceptance(acceptance_path, resolver=resolver)


def test_origin_parity_receipt_requires_canonical_fallback_promotion(
    tmp_path: Path,
) -> None:
    job_id = "v02j-promoted"
    attempt = tmp_path / "jobs" / job_id / "attempt_003"
    bundle = attempt / "checkpoints" / "outer_000060"
    bundle.mkdir(parents=True)
    _publish(attempt / "status.json", {"state": "failed"})
    metadata = SimpleNamespace(
        bundle_dir=bundle,
        bundle_digest="a" * 64,
        outer_iteration=60,
        environment_steps=6000,
        training_seed=1,
        provenance={"job_digest": "b" * 64, "attempt_digest": "c" * 64},
    )
    accepted_cell = {
        "resolution": "compiled_parity_fallback_promotion",
        "job_id": job_id,
        "job_digest": "b" * 64,
        "attempt_digest": "c" * 64,
        "bundle_path": str(bundle),
        "bundle_digest": "a" * 64,
        "outer_iteration": 60,
        "environment_steps": 6000,
        "seed": 1,
        "golden_parity_digest": "d" * 64,
        "compiled_parity_digest": "e" * 64,
        "promotion_entry_digest": "f" * 64,
        "failure_trace_digest": "1" * 64,
    }
    receipt = _origin_parity_receipt(metadata, accepted_cell)
    assert receipt["resolution"] == "compiled_parity_fallback_promotion"
    assert receipt["training_record_digest"] is None

    with pytest.raises(GateFailure, match="origin parity receipt is absent"):
        _origin_parity_receipt(
            metadata, {**accepted_cell, "resolution": "direct_terminal_record"}
        )


def test_deployment_audit_separates_determinism_from_cross_backend_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    observation = np.asarray([[0.1], [0.2]], dtype=np.float32)
    actual_raw = np.asarray([[0.25], [-0.5]], dtype=np.float32)
    expected_raw = actual_raw + np.float32(1.0e-3)
    expected_environment = np.tanh(expected_raw).astype(np.float32)
    np.savez_compressed(
        bundle / "golden_io.npz",
        observation=observation,
        prng_key_data=np.asarray([1, 2], dtype=np.uint32),
        raw_action=expected_raw,
        environment_action=expected_environment,
    )

    class FakePolicy:
        def act_raw(self, observation, key, *, deterministic=True):
            assert deterministic
            return actual_raw.copy(), np.asarray(key) + 1

        def act(self, observation, key, *, deterministic=True):
            assert deterministic
            return np.tanh(actual_raw).astype(np.float32), np.asarray(key) + 1

    compiled = SimpleNamespace(
        passed=True,
        next_keys_equal=True,
        max_abs_error=1.0e-7,
        sample_count=2,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    monkeypatch.setattr(runner_module, "_restore_policy_key", lambda key: key)
    monkeypatch.setattr(runner_module, "_policy_key_data", np.asarray)
    monkeypatch.setattr(
        runner_module, "verify_compiled_policy_parity", lambda *args, **kwargs: compiled
    )
    result = _deployment_action_audit(FakePolicy(), SimpleNamespace(bundle_dir=bundle))
    assert result["status"] == "WARNING_CROSS_BACKEND_COMPATIBLE"
    assert result["compiled_parity"]["status"] == "PASS"
    assert result["cross_backend_golden_diagnostic"]["compatibility_envelope_passed"]
    assert CROSS_BACKEND_PARITY["raw_atol"] < 0.1
    assert CROSS_BACKEND_PARITY["environment_atol"] < 0.05
    assert (
        CROSS_BACKEND_PARITY["observed_raw_max_abs_error"]
        < CROSS_BACKEND_PARITY["raw_atol"]
    )
    assert (
        CROSS_BACKEND_PARITY["observed_environment_max_abs_error"]
        < CROSS_BACKEND_PARITY["environment_atol"]
    )
    assert (
        CROSS_BACKEND_PARITY["m2_cpu_evidence_sha256"]
        == "50ac5e13b021a415ab251f51672fabb61a334e79ae25f94e95d63c35a8f9fc46"
    )

    incompatible_raw = actual_raw + np.float32(0.2)
    np.savez_compressed(
        bundle / "golden_io.npz",
        observation=observation,
        prng_key_data=np.asarray([1, 2], dtype=np.uint32),
        raw_action=incompatible_raw,
        environment_action=np.tanh(incompatible_raw).astype(np.float32),
    )
    with pytest.raises(GateFailure, match="cross-backend action drift"):
        _deployment_action_audit(FakePolicy(), SimpleNamespace(bundle_dir=bundle))

    class WrongTransformPolicy(FakePolicy):
        def act(self, observation, key, *, deterministic=True):
            return np.zeros_like(actual_raw), np.asarray(key) + 1

    with pytest.raises(GateFailure, match="deployment action invariants"):
        _deployment_action_audit(
            WrongTransformPolicy(), SimpleNamespace(bundle_dir=bundle)
        )


def test_score_parser_has_no_oracle_capability() -> None:
    parser = _parser()
    fit = parser.parse_args(["fit-source", "--run-dir", "/tmp/v04a-run"])
    assert not hasattr(fit, "source_utility_root")
    parsed = parser.parse_args(["score-fp", "--run-dir", "/tmp/v04a-run"])
    assert parsed.stage == "score-fp"
    assert not hasattr(parsed, "oracle_root")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "score-fp",
                "--run-dir",
                "/tmp/v04a-run",
                "--oracle-root",
                "/tmp/private-oracle",
            ]
        )

    smoke = parser.parse_args(
        [
            "smoke-fp",
            "--run-dir",
            "/tmp/v04a-run",
            "--context-id",
            "development-context",
            "--attempt-id",
            "attempt-1",
        ]
    )
    assert smoke.stage == "smoke-fp"
    assert smoke.context_id == "development-context"
    assert not hasattr(smoke, "oracle_root")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "smoke-fp",
                "--run-dir",
                "/tmp/v04a-run",
                "--context-id",
                "development-context",
                "--attempt-id",
                "attempt-1",
                "--oracle-root",
                "/tmp/private-oracle",
            ]
        )


def test_artifact_paths_use_one_root_with_explicit_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_root = tmp_path / "environment-artifacts"
    explicit_root = tmp_path / "explicit-artifacts"
    monkeypatch.setenv(runner_module.ARTIFACTS_ROOT_ENV, str(environment_root))
    assert runner_module._artifacts_root() == environment_root.resolve()
    assert runner_module._artifacts_root(explicit_root) == explicit_root.resolve()

    args = runner_module._resolve_cli_paths(
        argparse.Namespace(
            artifacts_root=explicit_root,
            run_dir=Path("v04a/runs/new-run"),
        )
    )
    assert args.run_dir == (explicit_root / "v04a/runs/new-run").resolve()
    with pytest.raises(V04ARunnerError, match="escapes"):
        runner_module._artifact_path(Path("../outside"), explicit_root.resolve())

    monkeypatch.delenv(runner_module.ARTIFACTS_ROOT_ENV)
    expected_default = REPOSITORY.parent / "artifacts"
    assert runner_module._artifacts_root() == expected_default.resolve()


def test_r4_relocation_manifest_requires_byte_identical_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "historical" / runner_module.R4_RUN_ID
    artifacts_root = tmp_path / "artifacts"
    destination = artifacts_root / runner_module.R4_RUN_RELATIVE
    ranking_digest = "a" * 64
    payloads = {
        "run.json": {"status": "PREPARED", "formal": False},
        "asset_census.json": {
            "context_index_sha256": "b" * 64,
            "bank_digests": {"context": "c" * 64},
            "policy_bundle_and_abi": {"status": "PASS"},
            "raw_delta": {"status": "PASS"},
            "source_utility": {"status": "PASS_SOURCE_ONLY_PROJECTION"},
        },
        "fit_source_status.json": {
            "status": "COMPLETE",
            "source_only": True,
            "target_contexts_read": 0,
        },
        "score_fp_status.json": {
            "status": "COMPLETE",
            "ranking_record_count": 672,
        },
        "seal_status.json": {
            "status": "SEALED_PRE_ORACLE",
            "rankings_digest": ranking_digest,
        },
        "rankings.seal.json": {"rankings_digest": ranking_digest},
        "oracle_evaluate_status.json": {
            "status": "COMPLETE",
            "metric_record_count": 672,
            "rankings_digest": ranking_digest,
        },
        "oracle_binding.json": {"rankings_digest": ranking_digest},
        "summary.json": {
            "status": "COMPLETE_DEVELOPMENT",
            "formal": False,
            "scope": "24 frozen development contexts; not confirmatory",
        },
    }
    for name, payload in payloads.items():
        _publish(source / name, payload)
    _publish_jsonl(source / "rankings.jsonl", [{"fixture": "ranking"}])
    _publish_jsonl(source / "metrics.jsonl", [{"fixture": "metric"}])
    core_sha256 = {
        name: sha256_file(source / name) for name in runner_module.R4_CORE_SHA256
    }
    records, total_bytes, tree_digest = runner_module._tree_inventory(source)
    monkeypatch.setattr(runner_module, "R4_CORE_SHA256", core_sha256)
    monkeypatch.setattr(runner_module, "R4_RUN_FILE_COUNT", len(records))
    monkeypatch.setattr(runner_module, "R4_RUN_TOTAL_BYTES", total_bytes)
    monkeypatch.setattr(runner_module, "R4_RUN_TREE_SHA256", tree_digest)
    shutil.copytree(source, destination)
    source_logs = tmp_path / "historical-logs"
    source_diagnostics = tmp_path / "historical-diagnostics"
    source_logs.mkdir()
    source_diagnostics.mkdir()
    (source_logs / "fixture.log").write_bytes(b"frozen log\n")
    (source_diagnostics / "fixture.json").write_bytes(b'{"frozen":true}\n')
    log_sha256 = {"fixture.log": sha256_file(source_logs / "fixture.log")}
    diagnostic_sha256 = {
        "fixture.json": sha256_file(source_diagnostics / "fixture.json")
    }
    monkeypatch.setattr(runner_module, "R4_LOG_SHA256", log_sha256)
    monkeypatch.setattr(runner_module, "R4_DIAGNOSTIC_SHA256", diagnostic_sha256)
    target_logs = artifacts_root / "v04a" / "logs" / runner_module.R4_RUN_ID
    target_diagnostics = artifacts_root / runner_module.R4_DIAGNOSTIC_RELATIVE
    shutil.copytree(source_logs, target_logs)
    shutil.copytree(source_diagnostics, target_diagnostics)

    result = runner_module.relocation_manifest(
        argparse.Namespace(
            source_run=source,
            source_log_root=source_logs,
            source_diagnostic_root=source_diagnostics,
            artifacts_root=artifacts_root,
            manifest_output=None,
            resume=False,
        )
    )
    assert result["status"] == "VERIFIED_RELOCATION"
    assert result["destination"] == {
        "canonical_relative_path": runner_module.R4_RUN_RELATIVE.as_posix(),
        "role": "single canonical relocated R4 tree",
        "access_class": "frozen-read-only",
    }
    assert result["immutable_receipts_modified"] is False
    assert {
        row["canonical_relative_path"] for row in result["shared_dependencies"]
    } >= {
        "v02/exact90/v02-reacher-formal-2r-20260825-r2",
        "v02/formal_inputs/v02-reacher-formal-2r-20260825-r2",
        "v03/runs/v03-main-20260827-r0/source-market",
        "shared/runtime/fpo-418c2554",
    }
    manifest_path = artifacts_root / "v04a" / "relocation_manifest.json"
    before = manifest_path.read_bytes()
    runner_module.relocation_manifest(
        argparse.Namespace(
            source_run=source,
            source_log_root=source_logs,
            source_diagnostic_root=source_diagnostics,
            artifacts_root=artifacts_root,
            manifest_output=None,
            resume=True,
        )
    )
    assert manifest_path.read_bytes() == before

    (destination / "metrics.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(V04ARunnerError, match="not byte-identical"):
        runner_module.relocation_manifest(
            argparse.Namespace(
                source_run=source,
                source_log_root=source_logs,
                source_diagnostic_root=source_diagnostics,
                artifacts_root=artifacts_root,
                manifest_output=tmp_path / "rejected.json",
                resume=False,
            )
        )


def test_ranking_seal_is_required_and_detects_tampering(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    scoring_manifest, rankings = _seal_fixture()
    _prepared_run(run_dir, scoring_manifest)

    _publish_jsonl(run_dir / "rankings.jsonl", rankings[:-1])
    with pytest.raises(V04ARunnerError, match="grid|ranking|record"):
        seal_stage(argparse.Namespace(run_dir=run_dir, resume=False))

    (run_dir / "rankings.jsonl").unlink()
    rankings_sha256 = _publish_jsonl(run_dir / "rankings.jsonl", rankings)
    _publish(
        run_dir / "score_fp_status.json",
        {
            "schema": SCHEMA,
            "stage": "score-fp",
            "status": "COMPLETE",
            "oracle_access": False,
            "development_context_count": 24,
            "method_count": 4,
            "budget_count": 7,
            "ranking_record_count": len(rankings),
            "rankings_sha256": rankings_sha256,
        },
    )
    sealed = seal_stage(argparse.Namespace(run_dir=run_dir, resume=False))
    assert sealed["status"] == "SEALED_PRE_ORACLE"
    assert sealed["ranking_record_count"] == 24 * 7 * 4
    assert not (run_dir / "oracle_binding.json").exists()

    tampered = [
        {**rankings[0], "selected_opaque_candidate_id": "tampered"},
        *rankings[1:],
    ]
    (run_dir / "rankings.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in tampered)
    )
    with pytest.raises(ValueError, match="sealed|rankings"):
        oracle_evaluate(
            argparse.Namespace(
                run_dir=run_dir,
                oracle_root=tmp_path / "oracle-that-must-not-be-read",
                resume=False,
            )
        )


def test_oracle_operational_failure_is_recorded_after_valid_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "oracle-fail-closed"
    _prepared_run(run_dir, run_fields={"policy_market_id": "market-test"})
    candidates = [f"candidate-{index}" for index in range(5)]
    ranking = {
        "context_id": "development-0",
        "task_id": "Task-0",
        "method_id": "RAW_DELTA_TASK5",
        "budget_episodes": 1,
        "ranking": [{"opaque_candidate_id": candidate} for candidate in candidates],
    }
    _publish_jsonl(run_dir / "rankings.jsonl", [ranking])
    seal = seal_rankings([ranking])
    _publish(run_dir / "rankings.seal.json", seal.to_dict())

    monkeypatch.setattr(
        runner_module, "_validate_ranking_records", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner_module,
        "_sanitized_layout",
        lambda *args, **kwargs: (
            (
                {
                    "context_id": "development-0",
                    "role": "development",
                    "task_id": "Task-0",
                },
            ),
            {
                "Task-0": {
                    "candidate_bundle_digests": {
                        candidate: sha256_json({"bundle": candidate})
                        for candidate in candidates
                    }
                }
            },
        ),
    )
    monkeypatch.setattr(
        runner_module, "_posterior_trace_map", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        runner_module,
        "_validated_evidence_namespace",
        lambda root, **kwargs: (root, {"test": True}),
    )

    def malformed_oracle(*args, **kwargs):
        raise ValueError("malformed oracle payload")

    monkeypatch.setattr(runner_module, "_oracle_returns", malformed_oracle)
    result = oracle_evaluate(
        argparse.Namespace(
            run_dir=run_dir,
            oracle_root=tmp_path / "private-oracle",
            resume=False,
        )
    )

    assert result["status"] == "NO_GO_ORACLE_EVALUATION"
    assert result["rankings_digest"] == seal.rankings_digest
    assert runner_module._json(run_dir / "oracle_evaluate_status.json") == result
    assert not (run_dir / "oracle_binding.json").exists()


def test_raw_delta_adapter_matches_frozen_v03_numeric_helpers(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260828)
    observation = rng.normal(size=(64, 1))
    action = rng.normal(size=(64, 1))
    next_observation = observation + 0.2 * action + rng.normal(scale=0.01, size=(64, 1))
    probe = RewardFreeProbe(
        observation=observation,
        action=action,
        next_observation=next_observation,
        episode_offsets=np.asarray([0, 64], dtype=np.int64),
        probe_membership_digest="0" * 64,
    )
    raw_root = tmp_path / "delta_action"
    bandwidth = 1.25
    protocol_id = "frozen-v031-protocol"
    _publish(
        raw_root / "config.json",
        {
            "bandwidth": bandwidth,
            "protocol_id": protocol_id,
            "canonicalizer_digest": "identity-canonicalizer",
            "feature_width": 2,
        },
    )
    bank = TransitionBank(
        observation=observation,
        action=action,
        reward=np.zeros(64, dtype=np.float64),
        next_observation=next_observation,
        terminated=np.zeros(64, dtype=np.bool_),
        truncated=np.zeros(64, dtype=np.bool_),
        episode_offsets=probe.episode_offsets,
    )
    points = apply_transition_view(bank, V_DELTA_ONLY).feature_matrix.astype(np.float64)
    digest = sha256_ndarrays(
        {"points": points, "episode_offsets": probe.episode_offsets}
    )
    empirical = _empirical(
        points,
        probe.episode_offsets,
        bandwidth=bandwidth,
        protocol_id=protocol_id,
        dataset_digest=digest,
        task="Task",
        backend="numpy",
        block_size=16,
    )
    source = ReducedRKME(
        supports=empirical.points,
        beta=empirical.weights,
        bandwidth=bandwidth,
        rkme_norm2=empirical.norm2,
        empirical_norm2=empirical.norm2,
        reduction_error=0.0,
        protocol_id=protocol_id,
        source_dataset_digest=digest,
        source_task="Task",
    )
    candidate = "lw-frozen-source"
    source.save_npz(raw_root / "source" / f"{candidate}.npz")

    observed = raw_delta_task5_scores(
        probe=probe,
        task_id="Task",
        candidate_ids=(candidate,),
        raw_view_root=raw_root,
        raw_adapter={
            "canonicalizer_digest": "identity-canonicalizer",
            "max_observation_dim": 1,
            "max_action_dim": 1,
            "observation_mean": [0.0],
            "observation_std": [1.0],
            "action_mean": [0.0],
            "action_std": [1.0],
            "tasks": {"Task": {"observation_dim": 1, "action_dim": 1}},
        },
        block_size=16,
    )
    expected = -_distance(empirical, source, backend="numpy", block_size=16)
    assert observed[candidate] == pytest.approx(expected, abs=1e-14)


def test_score_fp_uses_only_sanitized_reward_free_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "sanitized-only-run"
    scoring_manifest, _ = _seal_fixture()
    _, config_digest = _load_config(CONFIG)

    # Materialize all 54 score-visible banks.  These files deliberately expose
    # only (s, a, s', offsets, membership), never reward/done/private metadata.
    for context_index, row in enumerate(scoring_manifest["contexts"]):
        transition_count = 32 * 64
        observation = np.full(
            (transition_count, 1), context_index / 100.0, dtype=np.float32
        )
        action = np.zeros((transition_count, 1), dtype=np.float32)
        next_observation = observation + np.float32(0.01)
        projection = run_dir / row["reward_free_npz"]
        projection.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            projection,
            observation=observation,
            action=action,
            next_observation=next_observation,
            episode_offsets=np.arange(0, transition_count + 1, 64, dtype=np.int64),
            probe_membership_digest=np.asarray(row["probe_membership_digest"]),
        )
        row["reward_free_npz_sha256"] = sha256_file(projection)

    method_cards = _method_cards(config_digest)
    _publish(run_dir / "method_cards.json", method_cards)
    raw_adapter = {
        "canonicalizer_digest": "source-only-identity",
        "max_observation_dim": 1,
        "max_action_dim": 1,
        "observation_mean": [0.0],
        "observation_std": [1.0],
        "action_mean": [0.0],
        "action_std": [1.0],
        "tasks": {
            task_id: {"observation_dim": 1, "action_dim": 1}
            for task_id in scoring_manifest["tasks"]
        },
    }
    _publish(run_dir / "raw_delta_adapter.json", raw_adapter)

    raw_root = run_dir / "source_only_raw"
    _publish(
        raw_root / "config.json",
        {
            "protocol_id": "source-only-test",
            "canonicalizer_digest": "source-only-identity",
            "bandwidth": 1.0,
            "feature_width": 2,
        },
    )
    source_rkme_digests: dict[str, str] = {}
    for task in scoring_manifest["tasks"].values():
        for candidate in task["candidate_ids"]:
            source_path = raw_root / "source" / f"{candidate}.npz"
            _publish(source_path, {"source_only_fixture": candidate})
            source_rkme_digests[candidate] = sha256_file(source_path)

    utility = {
        "schema": SCHEMA,
        "source_only": True,
        "tasks": {
            task_id: {
                type_id: {
                    candidate: 0.1 * (candidate_index + 1)
                    for candidate_index, candidate in enumerate(task["candidate_ids"])
                }
                for type_id in task["source_type_ids"]
            }
            for task_id, task in scoring_manifest["tasks"].items()
        },
    }
    _publish(run_dir / "source_utility.json", utility)
    source_fit_manifest = {
        "schema": "policy-learnware.v04a-source-fit-manifest.v1",
        "contains_reward_or_done": False,
        "contains_target_contexts": False,
        "contexts": [
            {
                **row,
                "reward_free_npz": f"source_fit_banks/{row['context_id']}.npz",
            }
            for row in scoring_manifest["contexts"]
            if row["role"] == "source"
        ],
        "tasks": {
            task_id: {
                key: value
                for key, value in task.items()
                if key != "raw_tie_break_tokens"
            }
            for task_id, task in scoring_manifest["tasks"].items()
        },
    }
    _publish(run_dir / "source_fit_manifest.json", source_fit_manifest)
    fixed_probe_protocol_id = sha256_json({"protocol": "sanitized-only-test"})
    model_manifest = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "source_only": True,
        "source_fit_manifest_sha256": sha256_file(run_dir / "source_fit_manifest.json"),
        "config_digest": config_digest,
        "fixed_probe_protocol_id": fixed_probe_protocol_id,
        "source_utility_sha256": sha256_file(run_dir / "source_utility.json"),
        "models": {task_id: {} for task_id in scoring_manifest["tasks"]},
        "target_contexts_read": 0,
    }
    _publish(run_dir / "source_observation_model_manifest.json", model_manifest)
    _prepared_run(
        run_dir,
        scoring_manifest,
        {
            "fixed_probe_protocol_id": fixed_probe_protocol_id,
            "source_fit_manifest_payload_digest": sha256_json(source_fit_manifest),
            "method_cards_payload_digest": sha256_json(method_cards),
            "raw_delta_adapter_payload_digest": sha256_json(raw_adapter),
            "raw_delta": {
                "root": "source_only_raw",
                "root_scope": "run_relative_source_only_copy",
                "contains_query_artifacts": False,
                "config_sha256": sha256_file(raw_root / "config.json"),
                "source_rkme_sha256": source_rkme_digests,
            },
        },
    )

    class FakeBPR:
        def __init__(self, task: dict):
            self.type_ids = tuple(task["source_type_ids"])
            self.candidate_ids = tuple(task["candidate_ids"])

        def posterior_dict(self, summaries: np.ndarray) -> dict[str, float]:
            assert summaries.shape[0] in BUDGET_EPISODES
            return dict(
                zip(
                    self.type_ids,
                    (0.4, 0.25, 0.15, 0.1, 0.1),
                    strict=True,
                )
            )

        def utility_scores(self, summaries: np.ndarray) -> dict[str, float]:
            return {
                candidate: float(5 - index)
                for index, candidate in enumerate(self.candidate_ids)
            }

        def target_predictive_nll(self, summaries: np.ndarray) -> float:
            return float(summaries.shape[0]) / 100.0

    class FakeEBPR:
        def __init__(self, task: dict):
            self.type_ids = tuple(task["source_type_ids"])
            self.candidate_ids = tuple(task["candidate_ids"])

        def _posterior(self) -> dict[str, float]:
            return dict(
                zip(
                    self.type_ids,
                    (0.35, 0.25, 0.2, 0.1, 0.1),
                    strict=True,
                )
            )

        def select_map(self, episodes: tuple) -> SimpleNamespace:
            assert len(episodes) in BUDGET_EPISODES
            return SimpleNamespace(
                posterior=self._posterior(),
                target_predictive_nll=float(len(episodes)) / 50.0,
            )

        def select_hybrid(self, episodes: tuple, utility: dict) -> SimpleNamespace:
            assert utility
            return SimpleNamespace(
                posterior=self._posterior(),
                target_predictive_nll=float(len(episodes)) / 50.0,
                expected_utility={
                    candidate: float(5 - index)
                    for index, candidate in enumerate(self.candidate_ids)
                },
            )

    def fake_load_source_models(
        run_root: Path, task_id: str, manifest: dict, **bindings
    ):
        assert run_root == run_dir
        assert manifest["source_only"] is True
        task = scoring_manifest["tasks"][task_id]
        assert bindings["task_layout"] == task
        assert bindings["utility"] == utility["tasks"][task_id]
        return FakeBPR(task), FakeEBPR(task)

    raw_calls: list[tuple[str, int, str]] = []

    def fake_raw_scores(*, probe, task_id, candidate_ids, **kwargs):
        raw_calls.append((task_id, probe.episode_count, probe.probe_membership_digest))
        return {
            candidate: float(5 - index) for index, candidate in enumerate(candidate_ids)
        }

    def forbidden_private_access(*args, **kwargs):
        raise AssertionError("score_fp attempted private/context/oracle access")

    monkeypatch.setattr(runner_module, "_load_source_models", fake_load_source_models)
    monkeypatch.setattr(runner_module, "raw_delta_task5_scores", fake_raw_scores)
    monkeypatch.setattr(runner_module.ReducedRKME, "load_npz", lambda path: object())
    monkeypatch.setattr(runner_module, "_context_rows", forbidden_private_access)
    monkeypatch.setattr(runner_module, "_market", forbidden_private_access)
    monkeypatch.setattr(runner_module, "_evidence_root", forbidden_private_access)

    smoke_context = next(
        row["context_id"]
        for row in scoring_manifest["contexts"]
        if row["role"] == "development"
    )
    smoke = smoke_fp(
        argparse.Namespace(
            run_dir=run_dir,
            block_size=16,
            context_id=smoke_context,
            attempt_id="smoke-1",
        )
    )
    assert smoke["status"] == "COMPLETE_SMOKE"
    assert smoke["budget_episodes"] == 32
    assert smoke["ranking_record_count"] == 4
    assert smoke["posterior_trace_count"] == 3
    assert smoke["seal_eligible"] is False
    assert len(raw_calls) == 1
    smoke_root = run_dir / "smoke_fp" / smoke_context / "smoke-1"
    assert len(_read_jsonl(smoke_root / "rankings.jsonl")) == 4
    assert all(
        row["stage"] == "DEVELOPMENT_SMOKE_PRE_ORACLE"
        for row in _read_jsonl(smoke_root / "rankings.jsonl")
    )
    assert not (run_dir / "rankings.jsonl").exists()
    assert not (run_dir / "score_fp_status.json").exists()

    calls_before_full = len(raw_calls)
    result = score_fp(argparse.Namespace(run_dir=run_dir, block_size=16))
    assert result["status"] == "COMPLETE"
    assert result["ranking_record_count"] == 24 * 7 * 4 == 672
    assert result["posterior_trace_count"] == 24 * 7 * 3 == 504
    assert len(raw_calls) - calls_before_full == 24 * 7
    assert {budget for _, budget, _ in raw_calls} == set(BUDGET_EPISODES)
    assert not (run_dir / "score_cells").exists()
    assert not (run_dir / "score_attempts").exists()

    rankings = _read_jsonl(run_dir / "rankings.jsonl")
    traces = _read_jsonl(run_dir / "posterior_traces.jsonl")
    assert len(rankings) == 672
    assert len(traces) == 504
    assert all(row["schema"] == SCHEMA for row in rankings)
    assert all(
        row["budget_ledger"]
        == BudgetLedger.for_budget(row["budget_episodes"]).to_dict()
        for row in rankings
    )
    assert all(row["fit_on_target"] is False for row in traces)
    run_bytes = (run_dir / "run.json").read_bytes()
    assert b"context_index" not in run_bytes
    assert b"private_registry" not in run_bytes
    assert b"oracle" not in run_bytes

    seal_stage(argparse.Namespace(run_dir=run_dir, resume=False))
    with pytest.raises(V04ARunnerError, match="after ranking seal or oracle"):
        score_fp(argparse.Namespace(run_dir=run_dir, block_size=16, resume=True))
    _publish(run_dir / "oracle_binding.json", {"test": "post-seal oracle binding"})
    with pytest.raises(V04ARunnerError, match="after ranking seal or oracle"):
        score_fp(argparse.Namespace(run_dir=run_dir, block_size=16, resume=True))


def _source_projection_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict, dict, dict]:
    run_dir = tmp_path / "run"
    mixed_root = tmp_path / "mixed-authority"
    oracle_root = mixed_root / "oracle"
    scoring_manifest, _ = _seal_fixture()
    contexts = [
        {
            **row,
            "reward_free_npz": f"source_fit_banks/{row['context_id']}.npz",
        }
        for row in scoring_manifest["contexts"]
        if row["role"] == "source"
    ]
    tasks = {
        task_id: {
            key: value for key, value in task.items() if key != "raw_tie_break_tokens"
        }
        for task_id, task in scoring_manifest["tasks"].items()
    }
    layout = {
        task_id: {
            **task,
            "source_rows": [row for row in contexts if row["task_id"] == task_id],
        }
        for task_id, task in tasks.items()
    }
    for task_id, task in layout.items():
        for row in task["source_rows"]:
            _write_utility_evidence(
                oracle_root,
                task_id=task_id,
                context_id=row["context_id"],
                candidates=tuple(task["candidate_ids"]),
                bundle_digests=dict(task["candidate_bundle_digests"]),
            )
    _publish(
        oracle_root / "private-target" / "secret.json",
        {"private": "must-not-be-projected"},
    )
    config, _ = _load_config(CONFIG)
    projection = runner_module._materialize_source_utility_projection(
        run_dir=run_dir,
        source_utility_root=mixed_root,
        layout=layout,
        config=config,
    )
    source_manifest = {
        "schema": "policy-learnware.v04a-source-fit-manifest.v1",
        "contains_reward_or_done": False,
        "contains_target_contexts": False,
        "contexts": contexts,
        "tasks": tasks,
    }
    _publish(run_dir / "source_fit_manifest.json", source_manifest)
    _publish(run_dir / "source_utility_projection_manifest.json", projection)
    _prepared_run(
        run_dir,
        run_fields={
            "fixed_probe_protocol_id": sha256_json({"source-only": True}),
            "source_fit_manifest_payload_digest": sha256_json(source_manifest),
            "source_utility_projection_manifest_payload_digest": sha256_json(
                projection
            ),
        },
    )
    return (
        run_dir,
        mixed_root,
        layout,
        projection,
        runner_module._json(run_dir / "run.json"),
    )


def test_source_utility_projection_exact_and_no_clobber(tmp_path: Path) -> None:
    run_dir, mixed_root, layout, projection, run = _source_projection_fixture(tmp_path)
    root = run_dir / projection["root"]
    assert projection["cell_count"] == len(projection["cells"]) == 150
    assert projection["contains_target_contexts"] is False
    assert projection["aggregate_digest"] == sha256_json(projection["cells"])
    assert len(tuple(root.glob("*/*.json"))) == 150
    assert not (root / "private-target").exists()
    assert runner_module._validated_source_utility_projection(run_dir, run) == (
        root,
        projection["cells"],
    )
    config, _ = _load_config(CONFIG)
    with pytest.raises(FileExistsError):
        runner_module._materialize_source_utility_projection(
            run_dir=run_dir,
            source_utility_root=mixed_root,
            layout=layout,
            config=config,
        )


def test_source_utility_projection_rejects_tamper_and_extra(tmp_path: Path) -> None:
    run_dir, _, _, projection, run = _source_projection_fixture(tmp_path)
    root = run_dir / projection["root"]
    cell = root / sorted(projection["cells"])[0]
    original = cell.read_bytes()
    cell.write_bytes(original + b"tampered")
    with pytest.raises(GateFailure, match="files or digests differ"):
        runner_module._validated_source_utility_projection(run_dir, run)
    cell.write_bytes(original)
    _publish(root / "unexpected.json", {"unexpected": True})
    with pytest.raises(GateFailure, match="files or digests differ"):
        runner_module._validated_source_utility_projection(run_dir, run)


def test_fit_source_can_only_open_source_utility_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, mixed_root, _, projection, _ = _source_projection_fixture(tmp_path)
    projection_root = (run_dir / projection["root"]).resolve()
    original_evidence_root = runner_module._evidence_root
    opened: list[Path] = []

    def projection_only(root: Path) -> Path:
        resolved = Path(root).resolve()
        assert resolved == projection_root
        opened.append(resolved)
        return original_evidence_root(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("fit-source opened the target scoring closure")

    monkeypatch.setattr(runner_module, "_evidence_root", projection_only)
    monkeypatch.setattr(runner_module, "_scoring_manifest", forbidden)
    monkeypatch.setattr(runner_module, "_sanitized_layout", forbidden)
    result = fit_source(argparse.Namespace(run_dir=run_dir, resume=False))
    assert result["status"] == "NO_GO_SOURCE_MODEL_FIT"
    assert opened == [projection_root] * 6
    assert (mixed_root / "oracle" / "private-target" / "secret.json").is_file()
    assert not (run_dir / "checkpoints").exists()


def test_utility_builder_rejects_non_source_roles(tmp_path: Path) -> None:
    evidence = tmp_path / "utility"
    task_id = "Task"
    context_id = "development-context"
    candidates = tuple(f"policy-{index}" for index in range(5))
    bundle_digests = {
        candidate: sha256_json({"bundle": candidate}) for candidate in candidates
    }
    _write_utility_evidence(
        evidence,
        task_id=task_id,
        context_id=context_id,
        candidates=candidates,
        bundle_digests=bundle_digests,
    )
    with pytest.raises(GateFailure, match="source-role"):
        _utility_matrix(
            evidence,
            {
                "task_id": task_id,
                "source_rows": [{"context_id": context_id, "role": "development"}],
                "candidate_ids": candidates,
                "candidate_bundle_digests": bundle_digests,
            },
        )


def test_utility_builder_requires_complete_per_episode_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "utility"
    task_id = "Task"
    context_id = "source-context"
    candidates = tuple(f"policy-{index}" for index in range(5))
    bundle_digests = {
        candidate: sha256_json({"bundle": candidate}) for candidate in candidates
    }
    _write_utility_evidence(
        evidence,
        task_id=task_id,
        context_id=context_id,
        candidates=candidates,
        bundle_digests=bundle_digests,
        omit_episode_returns_for=candidates[-1],
    )
    with pytest.raises(GateFailure, match="per-episode evidence"):
        _utility_matrix(
            evidence,
            {
                "task_id": task_id,
                "source_rows": [{"context_id": context_id, "role": "source"}],
                "candidate_ids": candidates,
                "candidate_bundle_digests": bundle_digests,
            },
        )


def test_utility_builder_uses_only_frozen_source_train_seed_prefix(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "utility"
    task_id = "Task"
    context_id = "source-context"
    candidates = tuple(f"policy-{index}" for index in range(5))
    bundle_digests = {
        candidate: sha256_json({"bundle": candidate}) for candidate in candidates
    }
    _write_utility_evidence(
        evidence,
        task_id=task_id,
        context_id=context_id,
        candidates=candidates,
        bundle_digests=bundle_digests,
    )
    path = evidence / context_id / f"{candidates[0]}.json"
    record = dict(runner_module._json(path))
    path.unlink()
    returns = [100.0] * 30 + [900.0] * 20
    record["episode_returns"] = returns
    record["mean_return"] = float(np.mean(returns))
    record["normalized_mean_return"] = float(np.mean(returns)) / 1000.0
    _publish(path, record)

    utility, _ = _utility_matrix(
        evidence,
        {
            "task_id": task_id,
            "source_rows": [{"context_id": context_id, "role": "source"}],
            "candidate_ids": candidates,
            "candidate_bundle_digests": bundle_digests,
        },
        train_episode_count=30,
    )

    assert utility[context_id][candidates[0]] == pytest.approx(0.1)
    assert utility[context_id][candidates[0]] != record["normalized_mean_return"]


def test_utility_builder_rejects_synchronous_seed_reordering(tmp_path: Path) -> None:
    evidence = tmp_path / "utility"
    task_id = "Task"
    context_id = "source-context"
    candidates = tuple(f"policy-{index}" for index in range(5))
    bundle_digests = {
        candidate: sha256_json({"bundle": candidate}) for candidate in candidates
    }
    _write_utility_evidence(
        evidence,
        task_id=task_id,
        context_id=context_id,
        candidates=candidates,
        bundle_digests=bundle_digests,
    )
    for candidate in candidates:
        path = evidence / context_id / f"{candidate}.json"
        record = dict(runner_module._json(path))
        path.unlink()
        record["episode_returns"] = list(reversed(record["episode_returns"]))
        record["reset_seeds"] = list(reversed(record["reset_seeds"]))
        record["policy_seeds"] = list(reversed(record["policy_seeds"]))
        _publish(path, record)

    with pytest.raises(GateFailure, match="seed identity/order"):
        _utility_matrix(
            evidence,
            {
                "task_id": task_id,
                "source_rows": [{"context_id": context_id, "role": "source"}],
                "candidate_ids": candidates,
                "candidate_bundle_digests": bundle_digests,
            },
        )


@pytest.mark.parametrize("tamper", ["schema", "seed_bank", "mean"])
def test_utility_builder_requires_v03_provenance_and_common_seeds(
    tmp_path: Path, tamper: str
) -> None:
    evidence = tmp_path / "utility"
    task_id = "Task"
    context_id = "source-context"
    candidates = tuple(f"policy-{index}" for index in range(5))
    bundle_digests = {
        candidate: sha256_json({"bundle": candidate}) for candidate in candidates
    }
    _write_utility_evidence(
        evidence,
        task_id=task_id,
        context_id=context_id,
        candidates=candidates,
        bundle_digests=bundle_digests,
    )
    path = evidence / context_id / f"{candidates[1]}.json"
    record = dict(runner_module._json(path))
    path.unlink()
    if tamper == "schema":
        record.pop("schema")
    elif tamper == "seed_bank":
        record["reset_seeds"] = list(range(730100, 730150))
        record["policy_seeds"] = list(range(1_730_103, 1_730_153))
    else:
        record["mean_return"] = "not-a-number"
    _publish(path, record)

    with pytest.raises(GateFailure, match="identity|common seed bank|mean evidence"):
        _utility_matrix(
            evidence,
            {
                "task_id": task_id,
                "source_rows": [{"context_id": context_id, "role": "source"}],
                "candidate_ids": candidates,
                "candidate_bundle_digests": bundle_digests,
            },
        )
