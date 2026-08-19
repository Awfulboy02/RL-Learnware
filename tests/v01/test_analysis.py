from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v01.analysis import (
    AnalysisContractError,
    assemble_scientific_analysis,
    build_oracle_matrices,
    required_join_evidence,
)
from policy_learnware_v0.v01.config import load_v01_experiment_config
from policy_learnware_v0.v01.gates import GATE_D_REQUIRED_CHECKS
from policy_learnware_v0.v01.oracle import CandidateRecord
from policy_learnware_v0.v01.schemas import (
    PrivateContextRecord,
    ShiftManifest,
    derive_variant_id,
)


def _config():
    path = Path(__file__).parents[2] / "configs" / "v01_smoke.yaml"
    config = load_v01_experiment_config(path)
    return replace(
        config,
        # At least 4,500 replicates are needed for a minimum attainable
        # centered-bootstrap p-value that can survive the 225-way Holm family.
        statistics=replace(config.statistics, bootstrap_resamples=5_000),
    )


def _fixture():
    config = _config()
    protocol_id = "1" * 64
    registry_digest = "2" * 64
    private_entries = []
    bindings = []
    for index, factor in enumerate(config.shift.diagnostic_grid):
        context = PrivateContextRecord.new(
            task="WalkerWalk",
            shift_id=config.shift.shift_id,
            factor=factor,
            context_token=index.to_bytes(16, "big"),
            nonce_token=(100 + index).to_bytes(32, "big"),
        )
        manifest = ShiftManifest.create(
            shift_id=config.shift.shift_id,
            factor=factor,
            registry_digest=registry_digest,
            base_protocol_id=config.base.expected_protocol_id,
            task="WalkerWalk",
            private_context_id=context.private_context_id,
        )
        variant_id = derive_variant_id(
            measurement_protocol_id=protocol_id,
            private_nonce=context.private_nonce,
            shift_manifest_digest=manifest.digest,
        )
        private_entries.append(
            {
                "context": context.to_dict(),
                "shift_manifest": manifest.to_dict(),
                "shift_manifest_digest": manifest.digest,
                "variant_id": variant_id,
            }
        )
        bindings.append((factor, context.d_theta, variant_id))
    contexts = {
        "schema": "policy-learnware.v01-private-context-map.v0",
        "experiment_id": config.experiment_id,
        "entries": private_entries,
    }

    candidates = []
    for index in range(10):
        candidate_id = f"candidate-{index:02d}"
        candidates.append(
            CandidateRecord(
                candidate_id=candidate_id,
                task_private="WalkerWalk",
                algorithm="fpo" if index < 5 else "ppo",
                training_seed=index % 5,
                job_id=candidate_id,
                bundle_path=f"/provenance/{candidate_id}",
                bundle_digest=sha256_json({"bundle": candidate_id}),
                observation_dim=24,
                action_dim=6,
                outer_iteration=config.base.checkpoint_outer,
                environment_steps=config.base.actual_environment_steps,
            )
        )
    candidate_manifest = {
        "schema": "policy-learnware.v01-candidates.v0",
        "oracle_protocol_id": "3" * 64,
        "candidates": [record.to_dict() for record in candidates],
    }

    nominal_means = [1.0, 0.95] + [0.90 - 0.005 * index for index in range(8)]
    maximum_severity = abs(__import__("math").log(0.5))
    oracle_shards = []
    for factor, d_theta, variant_id in bindings:
        severity_scale = d_theta / maximum_severity
        for candidate_index, candidate in enumerate(candidates):
            if factor == 1.0:
                effect = 0.0
            elif candidate_index == 0:
                effect = -0.30 * severity_scale
            elif candidate_index == 1:
                effect = -0.02 * severity_scale
            else:
                effect = -(0.04 + 0.005 * candidate_index) * severity_scale
            episodes = []
            for episode_index in range(config.oracle.episodes_per_candidate_variant):
                # Small common pattern keeps effects exactly paired while
                # retaining a non-degenerate episode distribution.
                noise = (-1.5, -0.5, 0.5, 1.5)[episode_index] * 0.001
                mean_return = nominal_means[candidate_index] + effect + noise
                episodes.append(
                    {
                        "task_private": "WalkerWalk",
                        "variant_id": variant_id,
                        "candidate_id": candidate.candidate_id,
                        "episode_index": episode_index,
                        "reset_seed": 1000 * candidate_index + episode_index,
                        "policy_seed": 100000 + 1000 * candidate_index + episode_index,
                        "raw_episodic_sum": mean_return * 1000.0,
                        "mean_step_return": mean_return,
                        "instance_digest": sha256_json({"instance": variant_id}),
                        "bundle_digest": candidate.bundle_digest,
                        "evaluator_contract_digest": "4" * 64,
                    }
                )
            oracle_shards.append(
                {
                    "schema": "policy-learnware.v01-oracle-shard.v0",
                    "task_private": "WalkerWalk",
                    "variant_id": variant_id,
                    "candidate_id": candidate.candidate_id,
                    "instance_digest": sha256_json({"instance": variant_id}),
                    "bundle_digest": candidate.bundle_digest,
                    "evaluator_contract_digest": "4" * 64,
                    "episodes": episodes,
                }
            )

    pair_rows = []
    within_index = 0
    between_index = 0
    nominal_variant = next(variant for factor, _, variant in bindings if factor == 1.0)
    for factor, d_theta, variant_id in bindings:
        d_phi = 0.01
        pair_rows.append(
            {
                "family": "within",
                "pair_index": within_index,
                "left_variant_id": variant_id,
                "left_bank": 0,
                "right_variant_id": variant_id,
                "right_bank": 1,
                "prefix": config.probe.gate_b_unreduced_prefix,
                "raw_mmd2": d_phi**2,
                "mmd2": d_phi**2,
                "d_phi": d_phi,
                "roundoff_clamped": False,
                "cross_term": 0.5,
            }
        )
        within_index += 1
        if factor != 1.0:
            for bank in range(config.probe.banks):
                d_phi = d_theta * (1.0 + bank * 0.01)
                pair_rows.append(
                    {
                        "family": "between",
                        "pair_index": between_index,
                        "left_variant_id": nominal_variant,
                        "left_bank": bank,
                        "right_variant_id": variant_id,
                        "right_bank": bank,
                        "prefix": config.probe.gate_b_unreduced_prefix,
                        "raw_mmd2": d_phi**2,
                        "mmd2": d_phi**2,
                        "d_phi": d_phi,
                        "roundoff_clamped": False,
                        "cross_term": 0.4,
                    }
                )
                between_index += 1

    source_task_by_id = {
        "source-finger-spin": "FingerSpin",
        "source-finger-easy": "FingerTurnEasy",
        "source-finger-hard": "FingerTurnHard",
        "source-walker-run": "WalkerRun",
        "source-walker-stand": "WalkerStand",
        "source-walker-walk": "WalkerWalk",
    }
    routing_rows = []
    for routing_index, (_, _, variant_id) in enumerate(
        (binding for binding in bindings for _ in range(config.probe.banks))
    ):
        bank = routing_index % config.probe.banks
        selected = "source-walker-walk"
        ordered_sources = [selected] + sorted(set(source_task_by_id) - {selected})
        routing_rows.append(
            {
                "routing_index": routing_index,
                "variant_id": variant_id,
                "bank": bank,
                "prefix": config.probe.max_episodes_per_bank,
                "selected_source_id": selected,
                "ranking": [
                    {"source_id": source_id, "routing_score": float(index)}
                    for index, source_id in enumerate(ordered_sources)
                ],
            }
        )
    taskspec = {
        "schema": "policy-learnware.v01-taskspec-matrix.v0",
        "plan_digest": "5" * 64,
        "pair_rows": pair_rows,
        "routing_rows": routing_rows,
        "self_norm_rows": [],
        "clamp_count": 0,
    }
    schema_digests = {variant_id: "6" * 64 for _, _, variant_id in bindings}
    gate_d_checks = {name: True for name in GATE_D_REQUIRED_CHECKS}
    return {
        "config": config,
        "contexts": contexts,
        "candidates": candidate_manifest,
        "shards": oracle_shards,
        "taskspec": taskspec,
        "source_map": source_task_by_id,
        "schema_digests": schema_digests,
        "gate_d_checks": gate_d_checks,
        "bindings": bindings,
    }


def _assemble(fixture):
    return assemble_scientific_analysis(
        fixture["contexts"],
        fixture["candidates"],
        fixture["shards"],
        fixture["taskspec"],
        config=fixture["config"],
        analysis_seed_namespace="test-protocol",
        source_task_by_id=fixture["source_map"],
        schema_view_digest_by_variant=fixture["schema_digests"],
        gate_d_checks=fixture["gate_d_checks"],
    )


def test_full_analysis_recomputes_complete_families_and_is_json_serializable() -> None:
    fixture = _fixture()
    result = _assemble(fixture)
    payload = result.to_dict()

    assert payload["oracle_episodes"]["episode_count"] == 5 * 10 * 4
    assert payload["oracle_aggregates"]["aggregate_count"] == 5 * 10
    assert payload["gate_a"]["passed"]
    task = payload["gate_a"]["tasks"][0]
    assert len(task["corrected_families"]["material"]) == 40
    assert {row["family_size"] for row in task["corrected_families"]["material"]} == {40}
    assert len(task["corrected_families"]["heterogeneity"]) == 180
    assert {row["family_size"] for row in task["corrected_families"]["heterogeneity"]} == {180}
    ranking = task["corrected_families"]["ranking_reversal_views"]
    assert len(ranking) == 180
    assert {
        view[side]["family_size"] for view in ranking for side in ("nominal_gap", "shifted_gap")
    } == {225}
    assert payload["gate_b"]["passed"]
    assert payload["gate_b"]["routing_accuracy"] == 1.0
    assert payload["gate_c_diagnostics"]
    assert all("passed" not in item and "strong" not in item for item in payload["gate_c_diagnostics"])
    assert payload["join_audit"]["precomputed_scientific_pass_fields_consumed"] is False
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_missing_private_join_evidence_fails_closed() -> None:
    fixture = _fixture()
    assert "source_task_by_id" in required_join_evidence()[0]
    with pytest.raises(AnalysisContractError, match="source_task_by_id"):
        assemble_scientific_analysis(
            fixture["contexts"],
            fixture["candidates"],
            fixture["shards"],
            fixture["taskspec"],
            config=fixture["config"],
            analysis_seed_namespace="test-protocol",
            source_task_by_id={},
            schema_view_digest_by_variant=fixture["schema_digests"],
            gate_d_checks=fixture["gate_d_checks"],
        )

    bad_schema = dict(fixture["schema_digests"])
    bad_schema[next(iter(bad_schema))] = "7" * 64
    with pytest.raises(AnalysisContractError, match="schema views differ"):
        assemble_scientific_analysis(
            fixture["contexts"],
            fixture["candidates"],
            fixture["shards"],
            fixture["taskspec"],
            config=fixture["config"],
            analysis_seed_namespace="test-protocol",
            source_task_by_id=fixture["source_map"],
            schema_view_digest_by_variant=bad_schema,
            gate_d_checks=fixture["gate_d_checks"],
        )


def test_precomputed_pass_or_aggregate_injection_is_rejected() -> None:
    fixture = _fixture()
    poisoned_taskspec = copy.deepcopy(fixture["taskspec"])
    poisoned_taskspec["passed"] = True
    with pytest.raises(AnalysisContractError, match="TaskSpecMatrix keys"):
        assemble_scientific_analysis(
            fixture["contexts"],
            fixture["candidates"],
            fixture["shards"],
            poisoned_taskspec,
            config=fixture["config"],
            analysis_seed_namespace="test-protocol",
            source_task_by_id=fixture["source_map"],
            schema_view_digest_by_variant=fixture["schema_digests"],
            gate_d_checks=fixture["gate_d_checks"],
        )

    poisoned_shards = copy.deepcopy(fixture["shards"])
    poisoned_shards[0]["passed"] = True
    with pytest.raises(AnalysisContractError, match="oracle shard 0 keys"):
        assemble_scientific_analysis(
            fixture["contexts"],
            fixture["candidates"],
            poisoned_shards,
            fixture["taskspec"],
            config=fixture["config"],
            analysis_seed_namespace="test-protocol",
            source_task_by_id=fixture["source_map"],
            schema_view_digest_by_variant=fixture["schema_digests"],
            gate_d_checks=fixture["gate_d_checks"],
        )


def test_oracle_seed_pairing_and_abs_of_mean_are_recomputed_from_rows() -> None:
    fixture = _fixture()
    # Give one non-nominal candidate alternating deltas.  J must be
    # abs(mean(D)) == 0, not mean(abs(D)) == 0.1.
    factor, _, variant_id = next(item for item in fixture["bindings"] if item[0] == 0.75)
    del factor
    shard = next(
        item
        for item in fixture["shards"]
        if item["variant_id"] == variant_id and item["candidate_id"] == "candidate-09"
    )
    nominal_id = next(item[2] for item in fixture["bindings"] if item[0] == 1.0)
    nominal = next(
        item
        for item in fixture["shards"]
        if item["variant_id"] == nominal_id and item["candidate_id"] == "candidate-09"
    )
    for index, (shifted_row, nominal_row) in enumerate(zip(shard["episodes"], nominal["episodes"], strict=True)):
        delta = 0.1 if index % 2 == 0 else -0.1
        shifted_row["mean_step_return"] = nominal_row["mean_step_return"] + delta
        shifted_row["raw_episodic_sum"] = shifted_row["mean_step_return"] * 1000.0

    _, _, _, matrices = build_oracle_matrices(
        fixture["contexts"],
        fixture["candidates"],
        fixture["shards"],
        config=fixture["config"],
        analysis_seed_namespace="test-protocol",
    )
    aggregate = next(
        item
        for item in matrices.aggregates
        if item.variant_id == variant_id and item.candidate_id == "candidate-09"
    )
    assert aggregate.delta_return == pytest.approx(0.0, abs=1e-12)
    assert aggregate.abs_transfer_gap == pytest.approx(0.0, abs=1e-12)

    broken = copy.deepcopy(fixture["shards"])
    target = next(item for item in broken if item["variant_id"] == variant_id)
    target["episodes"][0]["reset_seed"] += 1
    with pytest.raises(AnalysisContractError, match="not paired across variants"):
        build_oracle_matrices(
            fixture["contexts"],
            fixture["candidates"],
            broken,
            config=fixture["config"],
            analysis_seed_namespace="test-protocol",
        )


def test_gate_d_is_recomputed_from_primitive_checks_not_a_passed_field() -> None:
    fixture = _fixture()
    fixture["gate_d_checks"] = {
        name: name != "oracle_poison_does_not_change_taskspec_digest"
        for name in GATE_D_REQUIRED_CHECKS
    }
    result = _assemble(fixture).to_dict()
    assert result["gate_d_dependency"]["passed"] is False
    assert result["gate_b"]["gate_d_passed"] is False
    assert result["gate_b"]["passed"] is False
