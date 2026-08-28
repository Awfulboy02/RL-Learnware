from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    BudgetLedger,
    RankingSeal,
    RewardFreeProbe,
    V04AProtocolError,
    break_tie,
    canonical_tie_token,
    derive_probe_membership,
    seal_rankings,
    tie_break_key,
    verify_ranking_seal,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_membership_is_deterministic_nested_and_round_trips() -> None:
    membership = derive_probe_membership("context-a", 713)
    repeated = derive_probe_membership("context-a", 713)
    changed = derive_probe_membership("context-a", 714)

    assert BUDGET_EPISODES == (1, 2, 4, 8, 16, 24, 32)
    assert membership.to_dict() == repeated.to_dict()
    assert membership.probe_membership_digest != changed.probe_membership_digest
    assert sorted(membership.episode_order) == list(range(32))
    assert len(membership.entries) == 32 * 64
    for episode_index in range(32):
        entries = membership.indices_for_episode(episode_index)
        assert len(entries) == 64
        assert entries[0].native_timestep == 0
        assert entries[-1].native_timestep == 999
        assert [entry.native_timestep for entry in entries] == sorted(
            entry.native_timestep for entry in entries
        )

    previous = ()
    for budget in BUDGET_EPISODES:
        current = membership.for_budget(budget)
        assert current[: len(previous)] == previous
        assert len(current) == budget * 64
        previous = current

    restored = type(membership).from_dict(membership.to_dict())
    assert restored.to_dict() == membership.to_dict()
    tampered = membership.to_dict()
    tampered["episode_order"] = list(reversed(tampered["episode_order"]))
    with pytest.raises(V04AProtocolError, match="digest"):
        type(membership).from_dict(tampered)


def test_reward_free_projection_and_dual_cost_ledger() -> None:
    membership = derive_probe_membership("context-projection", 22)
    rows = 32 * 1_000
    observation = np.arange(rows, dtype=np.float64)[:, None]
    action = np.column_stack((observation[:, 0] * 0.01, -observation[:, 0]))
    next_observation = observation + 3.0

    smaller = RewardFreeProbe.from_full_episodes(
        observation,
        action,
        next_observation,
        membership=membership,
        budget_episodes=2,
    )
    larger = RewardFreeProbe.from_full_episodes(
        observation,
        action,
        next_observation,
        membership=membership,
        budget_episodes=4,
    )
    expected_indices = [entry.flat_index for entry in membership.for_budget(2)]
    np.testing.assert_array_equal(smaller.observation[:, 0], expected_indices)
    np.testing.assert_array_equal(
        smaller.observation, larger.observation[: smaller.transition_count]
    )
    assert smaller.probe_membership_digest == membership.probe_membership_digest
    assert set(field.name for field in dataclasses.fields(RewardFreeProbe)) == {
        "observation",
        "action",
        "next_observation",
        "episode_offsets",
        "probe_membership_digest",
    }

    for budget in BUDGET_EPISODES:
        ledger = BudgetLedger.for_budget(budget)
        assert ledger.visible_transition_count == budget * 64
        assert ledger.interaction_cost_steps == budget * 1_000
        assert ledger.candidate_conditioned_steps == 0
        assert ledger.reward_queries == 0
        assert ledger.total_target_steps == budget * 1_000
    with pytest.raises(V04AProtocolError, match="visible_transition_count"):
        BudgetLedger(
            budget_episodes=2,
            visible_transition_count=2_000,
            interaction_cost_steps=2_000,
        )


def test_canonical_tie_is_order_independent_and_config_bound() -> None:
    config = _d("config")
    ids = ("opaque-c", "opaque-a", "opaque-b")
    expected_token = hashlib.sha256(
        (config + "v04a-bpr-tie-v1").encode("utf-8")
    ).hexdigest()
    assert canonical_tie_token(config) == expected_token
    winner = break_tie(config, ids)
    assert winner == break_tie(config, tuple(reversed(ids)))
    assert winner == min(ids, key=lambda item: (tie_break_key(config, item), item))
    assert {tie_break_key(config, item) for item in ids} != {
        tie_break_key(_d("other-config"), item) for item in ids
    }


def test_ranking_seal_detects_object_and_byte_tampering() -> None:
    rankings = [
        {
            "context_id": "c0",
            "method": "BPR_FP",
            "budget": 2,
            "ranking": ["opaque-a", "opaque-b"],
        }
    ]
    seal = seal_rankings(rankings)
    assert verify_ranking_seal(seal, rankings)
    assert RankingSeal.from_dict(seal.to_dict()).rankings == rankings

    changed = [dict(rankings[0], ranking=["opaque-b", "opaque-a"])]
    with pytest.raises(V04AProtocolError, match="sealed bytes"):
        verify_ranking_seal(seal, changed)

    payload = seal.to_dict()
    payload["canonical_rankings_json"] = payload["canonical_rankings_json"].replace(
        "opaque-a", "opaque-z"
    )
    with pytest.raises(V04AProtocolError, match="digest"):
        RankingSeal.from_dict(payload)
