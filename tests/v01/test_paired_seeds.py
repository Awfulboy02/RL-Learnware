from __future__ import annotations

import pytest

from policy_learnware_v0.v01.seeds import (
    ProbeEpisodeSeeds,
    V01SeedPlan,
    assert_no_base_seed_pair_collision,
    assert_v01_seed_records_disjoint,
    collect_known_base_seed_pairs,
)


def test_probe_and_oracle_pair_across_variants_but_not_candidates() -> None:
    plan = V01SeedPlan(20260819)
    # No variant argument exists: both environments receive exactly this record.
    probe = plan.probe_episode("WalkerWalk", 3, 7)
    assert probe == plan.probe_episode("WalkerWalk", 3, 7)
    assert probe != plan.probe_episode("WalkerWalk", 4, 7)
    assert probe != plan.probe_episode("FingerTurnEasy", 3, 7)

    candidate_a = plan.oracle_episode("WalkerWalk", "candidate-a", 7)
    assert candidate_a == plan.oracle_episode("WalkerWalk", "candidate-a", 7)
    assert candidate_a != plan.oracle_episode("WalkerWalk", "candidate-b", 7)
    assert probe.collision_key != candidate_a.collision_key


def test_gate_bootstrap_and_report_namespaces_are_deterministic_and_separate() -> None:
    plan = V01SeedPlan(9)
    gate = plan.gate0_episode("WalkerWalk", 0)
    assert gate == plan.gate0_episode("WalkerWalk", 0)
    assert plan.bootstrap_seed("r" * 64, "WalkerWalk", "material") != plan.report_seed("r" * 64, "summary")


def test_observed_collision_audits_fail_closed() -> None:
    plan = V01SeedPlan(10)
    probe = plan.probe_episode("WalkerWalk", 0, 0)
    oracle = plan.oracle_episode("WalkerWalk", "candidate", 0)
    assert_v01_seed_records_disjoint({"probe": [probe], "oracle": [oracle]})
    fake = ProbeEpisodeSeeds(
        task="FingerTurnEasy", bank=0, episode_index=1,
        reset_seed=probe.reset_seed, probe_seed=probe.probe_seed,
    )
    with pytest.raises(ValueError):
        assert_v01_seed_records_disjoint({"probe": [probe], "other": [fake]})
    with pytest.raises(ValueError):
        assert_v01_seed_records_disjoint({"probe": [probe, fake]})
    with pytest.raises(ValueError):
        assert_no_base_seed_pair_collision([probe], [probe.collision_key])


def test_known_base_seed_pairs_are_collected_recursively(tmp_path) -> None:
    datasets = tmp_path / "datasets"
    policy = tmp_path / "policy"
    datasets.mkdir()
    policy.mkdir()
    (datasets / "probe.json").write_text(
        '{"nested":{"reset_seeds":[1,2],"probe_seeds":[3,4]}}',
        encoding="utf-8",
    )
    (policy / "returns.json").write_text(
        '{"evaluation_reset_seeds":[5],"evaluation_policy_seeds":[6]}',
        encoding="utf-8",
    )
    assert collect_known_base_seed_pairs(tmp_path) == frozenset({(1, 3), (2, 4), (5, 6)})
