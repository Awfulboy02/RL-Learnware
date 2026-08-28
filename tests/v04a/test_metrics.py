from __future__ import annotations

import pytest

from policy_learnware_v0.v04a.metrics import (
    V04MetricError,
    aggregate_metrics,
    hierarchical_bootstrap_intervals,
)


def _bootstrap_fixture():
    rows = []
    episodes = {}
    for task_index in range(6):
        task_id = f"task-{task_index}"
        for context_index in range(4):
            context_id = f"{task_id}-context-{context_index}"
            candidates = [f"{task_id}-policy-{index}" for index in range(5)]
            rows.append(
                {
                    "context_id": context_id,
                    "task_id": task_id,
                    "method_id": "BPR_FP",
                    "budget_episodes": 8,
                    "ranking": [
                        {"opaque_candidate_id": candidate} for candidate in candidates
                    ],
                }
            )
            episodes[context_id] = {
                candidate: [1.0 - 0.1 * index] * 50
                for index, candidate in enumerate(candidates)
            }
    return rows, episodes


def test_hierarchical_bootstrap_is_deterministic_and_nested() -> None:
    rows, episodes = _bootstrap_fixture()
    first = hierarchical_bootstrap_intervals(
        rows, episodes, epsilon=0.01, seed=17, replicates=200
    )
    repeated = hierarchical_bootstrap_intervals(
        list(reversed(rows)), episodes, epsilon=0.01, seed=17, replicates=200
    )

    assert first == repeated
    assert first["hierarchy"] == ["task", "context", "paired_oracle_episode"]
    assert first["intervals"]["hit_at_1"] == {"lower": 1.0, "upper": 1.0}
    assert first["intervals"]["regret_at_1"] == {"lower": 0.0, "upper": 0.0}


def test_hierarchical_bootstrap_rejects_missing_episode_evidence() -> None:
    rows, episodes = _bootstrap_fixture()
    episodes.pop(rows[0]["context_id"])
    with pytest.raises(V04MetricError, match="incomplete"):
        hierarchical_bootstrap_intervals(
            rows, episodes, epsilon=0.01, seed=17, replicates=100
        )


def test_aggregate_metrics_returns_point_estimates() -> None:
    result = aggregate_metrics(
        [
            {
                "hit_at_1": True,
                "epsilon_optimal": True,
                "regret_at_1": 0.0,
                "regret_at_3": 0.0,
                "selected_j_norm": 0.8,
                "spearman": 1.0,
            },
            {
                "hit_at_1": False,
                "epsilon_optimal": False,
                "regret_at_1": 0.2,
                "regret_at_3": 0.1,
                "selected_j_norm": 0.6,
                "spearman": None,
            },
        ]
    )
    assert result["row_count"] == 2
    assert result["hit_at_1"] == 0.5
    assert result["regret_at_1"] == pytest.approx(0.1)
    assert result["selected_j_norm"] == pytest.approx(0.7)
