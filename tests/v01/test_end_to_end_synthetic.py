from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import random

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v01.analysis import assemble_scientific_analysis
from policy_learnware_v0.v01.config import (
    TasksConfig,
    load_v01_experiment_config,
)
from policy_learnware_v0.v01.gates import (
    GATE_0_REQUIRED_CHECKS,
    GATE_D_REQUIRED_CHECKS,
    evaluate_gate_0,
    evaluate_gate_d,
)
from policy_learnware_v0.v01.oracle import CandidateRecord
from policy_learnware_v0.v01.report import decide_v01, render_summary
from policy_learnware_v0.v01.schemas import (
    PrivateContextRecord,
    ShiftManifest,
    derive_variant_id,
)


PROJECT = Path(__file__).resolve().parents[2]
TASKS = ("WalkerWalk", "FingerTurnEasy")
FACTORS = (0.75, 1.0, 2.0)
EPISODES = 4


def _synthetic_config():
    """Small CPU contract: two shifted variants plus nominal per task."""

    base = load_v01_experiment_config(PROJECT / "configs" / "v01_smoke.yaml")
    return replace(
        base,
        experiment_id="v01-synthetic-e2e",
        base=replace(base.base, candidates_per_task=2),
        tasks=TasksConfig(infrastructure=(TASKS[0],), confirmatory=(TASKS[1],)),
        shift=replace(base.shift, diagnostic_grid=FACTORS),
        oracle=replace(base.oracle, episodes_per_candidate_variant=EPISODES),
        statistics=replace(base.statistics, bootstrap_resamples=256),
    )


def _synthetic_raw(*, scientific_pass: bool) -> dict[str, object]:
    config = _synthetic_config()
    protocol_id = "1" * 64
    registry_digest = "2" * 64
    evaluator_digest = "4" * 64

    context_entries: list[dict[str, object]] = []
    bindings: list[tuple[str, float, float, str]] = []
    token_index = 1
    for task in TASKS:
        for factor in FACTORS:
            context = PrivateContextRecord.new(
                task=task,
                shift_id=config.shift.shift_id,
                factor=factor,
                context_token=token_index.to_bytes(16, "big"),
                nonce_token=(1000 + token_index).to_bytes(32, "big"),
            )
            manifest = ShiftManifest.create(
                shift_id=config.shift.shift_id,
                factor=factor,
                registry_digest=registry_digest,
                base_protocol_id=config.base.expected_protocol_id,
                task=task,
                private_context_id=context.private_context_id,
            )
            variant_id = derive_variant_id(
                measurement_protocol_id=protocol_id,
                private_nonce=context.private_nonce,
                shift_manifest_digest=manifest.digest,
            )
            context_entries.append(
                {
                    "context": context.to_dict(),
                    "shift_manifest": manifest.to_dict(),
                    "shift_manifest_digest": manifest.digest,
                    "variant_id": variant_id,
                }
            )
            bindings.append((task, factor, context.d_theta, variant_id))
            token_index += 1

    candidates: list[CandidateRecord] = []
    for task_index, task in enumerate(TASKS):
        for candidate_index, algorithm in enumerate(("fpo", "ppo")):
            candidate_id = f"{task.lower()}-{algorithm}-0"
            candidates.append(
                CandidateRecord(
                    candidate_id=candidate_id,
                    task_private=task,
                    algorithm=algorithm,
                    training_seed=0,
                    job_id=candidate_id,
                    bundle_path=f"/provenance/{candidate_id}",
                    bundle_digest=sha256_json({"bundle": candidate_id}),
                    observation_dim=24 if task_index == 0 else 12,
                    action_dim=6 if task_index == 0 else 2,
                    outer_iteration=config.base.checkpoint_outer,
                    environment_steps=config.base.actual_environment_steps,
                )
            )

    shards: list[dict[str, object]] = []
    for task_index, task in enumerate(TASKS):
        task_candidates = [item for item in candidates if item.task_private == task]
        for _, factor, _, variant_id in [item for item in bindings if item[0] == task]:
            for candidate_index, candidate in enumerate(task_candidates):
                nominal = (1.0, 0.95)[candidate_index]
                if scientific_pass:
                    effect = {
                        0.75: (-0.35, -0.02),
                        1.0: (0.0, 0.0),
                        2.0: (-0.55, -0.04),
                    }[factor][candidate_index]
                else:
                    effect = 0.0
                mean_return = nominal + effect
                episodes = []
                for episode_index in range(EPISODES):
                    # Seeds are paired across variants and disjoint across both
                    # task and policy streams. Constant returns make the
                    # expected pass/fail fixture exact rather than stochastic.
                    stream = task_index * 10_000 + candidate_index * 100
                    episodes.append(
                        {
                            "task_private": task,
                            "variant_id": variant_id,
                            "candidate_id": candidate.candidate_id,
                            "episode_index": episode_index,
                            "reset_seed": stream + episode_index,
                            "policy_seed": 1_000_000 + stream + episode_index,
                            "raw_episodic_sum": mean_return * 1000.0,
                            "mean_step_return": mean_return,
                            "instance_digest": sha256_json({"instance": variant_id}),
                            "bundle_digest": candidate.bundle_digest,
                            "evaluator_contract_digest": evaluator_digest,
                        }
                    )
                shards.append(
                    {
                        "schema": "policy-learnware.v01-oracle-shard.v0",
                        "task_private": task,
                        "variant_id": variant_id,
                        "candidate_id": candidate.candidate_id,
                        "instance_digest": sha256_json({"instance": variant_id}),
                        "bundle_digest": candidate.bundle_digest,
                        "evaluator_contract_digest": evaluator_digest,
                        "episodes": episodes,
                    }
                )

    source_map = {
        "source-finger-easy": "FingerTurnEasy",
        "source-walker-walk": "WalkerWalk",
    }
    pair_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    within_index = 0
    between_index = 0
    routing_index = 0
    for task in TASKS:
        task_bindings = [item for item in bindings if item[0] == task]
        nominal_id = next(item[3] for item in task_bindings if item[1] == 1.0)
        selected = (
            "source-walker-walk" if task == "WalkerWalk" else "source-finger-easy"
        )
        other = next(source for source in source_map if source != selected)
        for _, factor, d_theta, variant_id in task_bindings:
            within_distance = 0.01
            pair_rows.append(
                {
                    "family": "within",
                    "pair_index": within_index,
                    "left_variant_id": variant_id,
                    "left_bank": 0,
                    "right_variant_id": variant_id,
                    "right_bank": 1,
                    "prefix": config.probe.gate_b_unreduced_prefix,
                    "raw_mmd2": within_distance**2,
                    "mmd2": within_distance**2,
                    "d_phi": within_distance,
                    "roundoff_clamped": False,
                    "cross_term": 0.5,
                }
            )
            within_index += 1
            if factor != 1.0:
                between_distance = (
                    (0.30 if factor == 0.75 else 0.70)
                    if scientific_pass
                    else 0.01
                )
                for bank in range(config.probe.banks):
                    pair_rows.append(
                        {
                            "family": "between",
                            "pair_index": between_index,
                            "left_variant_id": nominal_id,
                            "left_bank": bank,
                            "right_variant_id": variant_id,
                            "right_bank": bank,
                            "prefix": config.probe.gate_b_unreduced_prefix,
                            "raw_mmd2": between_distance**2,
                            "mmd2": between_distance**2,
                            "d_phi": between_distance,
                            "roundoff_clamped": False,
                            "cross_term": 0.4,
                        }
                    )
                    between_index += 1
            for bank in range(config.probe.banks):
                routing_rows.append(
                    {
                        "routing_index": routing_index,
                        "variant_id": variant_id,
                        "bank": bank,
                        "prefix": config.probe.max_episodes_per_bank,
                        "selected_source_id": selected,
                        "ranking": [
                            {"source_id": selected, "routing_score": 0.0},
                            {"source_id": other, "routing_score": 1.0},
                        ],
                    }
                )
                routing_index += 1

    return {
        "config": config,
        "contexts": {
            "schema": "policy-learnware.v01-private-context-map.v0",
            "experiment_id": config.experiment_id,
            "entries": context_entries,
        },
        "candidates": {
            "schema": "policy-learnware.v01-candidates.v0",
            "oracle_protocol_id": "3" * 64,
            "candidates": [item.to_dict() for item in candidates],
        },
        "shards": shards,
        "taskspec": {
            "schema": "policy-learnware.v01-taskspec-matrix.v0",
            "plan_digest": "5" * 64,
            "pair_rows": pair_rows,
            "routing_rows": routing_rows,
            "self_norm_rows": [],
            "clamp_count": 0,
        },
        "source_map": source_map,
        "schema_digests": {variant_id: "6" * 64 for *_, variant_id in bindings},
    }


def _assemble(raw: dict[str, object], *, gate_d_pass: bool = True):
    gate_d_checks = {
        name: gate_d_pass or name != "oracle_poison_does_not_change_taskspec_digest"
        for name in GATE_D_REQUIRED_CHECKS
    }
    return assemble_scientific_analysis(
        raw["contexts"],
        raw["candidates"],
        raw["shards"],
        raw["taskspec"],
        config=raw["config"],
        analysis_seed_namespace="synthetic-e2e",
        source_task_by_id=raw["source_map"],
        schema_view_digest_by_variant=raw["schema_digests"],
        gate_d_checks=gate_d_checks,
    )


def _decision(payload: dict[str, object], *, gate_0_pass: bool = True):
    gate_0_checks = {
        name: gate_0_pass or name != "identity_trajectory_and_policy_returns"
        for name in GATE_0_REQUIRED_CHECKS
    }
    return decide_v01(
        gate_0=evaluate_gate_0(gate_0_checks).to_dict(),
        gate_a=payload["gate_a"],
        gate_b=payload["gate_b"],
        gate_d=payload["gate_d_dependency"],
        recompute_audit={"passed": True},
    )


def test_raw_to_aggregate_gates_and_go_decision_is_deterministic() -> None:
    payload = _assemble(_synthetic_raw(scientific_pass=True)).to_dict()

    assert payload["oracle_episodes"]["episode_count"] == 2 * 3 * 2 * EPISODES
    assert payload["oracle_aggregates"]["aggregate_count"] == 2 * 3 * 2
    assert payload["gate_a"]["passed"] is True
    assert payload["gate_a"]["replicated_across_both_tasks"] is True
    assert payload["gate_b"]["passed"] is True
    assert len(payload["gate_c_diagnostics"]) == 2
    assert all("passed" not in row and "strong" not in row for row in payload["gate_c_diagnostics"])

    decision = _decision(payload)
    assert decision.code == "GO_V02_TRANSFERSPEC"
    assert decision.formal_complete is True
    summary = render_summary(
        experiment_id="v01-synthetic-e2e",
        measurement_run_id="7" * 64,
        oracle_protocol_id="3" * 64,
        decision=decision,
        gate_0=evaluate_gate_0({name: True for name in GATE_0_REQUIRED_CHECKS}).to_dict(),
        gate_a=payload["gate_a"],
        gate_b=payload["gate_b"],
        gate_c={"diagnostics": payload["gate_c_diagnostics"]},
        gate_d=payload["gate_d_dependency"],
        recompute_audit={"passed": True},
    )
    assert "GO_V02_TRANSFERSPEC" in summary


def test_scientific_fail_is_a_complete_no_go_not_an_engineering_error() -> None:
    payload = _assemble(_synthetic_raw(scientific_pass=False)).to_dict()
    assert payload["gate_a"]["passed"] is False
    assert payload["gate_b"]["passed"] is False

    decision = _decision(payload)
    assert decision.code == "NO_GO_CURRENT_POOL_SHIFT"
    assert decision.formal_complete is True


@pytest.mark.parametrize("failed_gate", ["gate_0", "gate_d"])
def test_hard_gate_failure_blocks_formal_report_decision(failed_gate: str) -> None:
    raw = _synthetic_raw(scientific_pass=True)
    payload = _assemble(raw, gate_d_pass=failed_gate != "gate_d").to_dict()
    decision = _decision(payload, gate_0_pass=failed_gate != "gate_0")

    assert decision.code == "BLOCKED_ENGINEERING"
    assert decision.formal_complete is False


def test_oracle_shard_permutation_has_identical_matrix_json_and_digest() -> None:
    raw = _synthetic_raw(scientific_pass=True)
    canonical = _assemble(raw).to_dict()
    shuffled_raw = copy.deepcopy(raw)
    random.Random(20260819).shuffle(shuffled_raw["shards"])
    shuffled = _assemble(shuffled_raw).to_dict()

    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    shuffled_json = json.dumps(shuffled, sort_keys=True, separators=(",", ":"))
    assert shuffled_json == canonical_json
    assert sha256_json(shuffled) == sha256_json(canonical)
