"""Small, dependency-light metrics for v0.4a policy-library selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class V04MetricError(ValueError):
    pass


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float | None:
    left = np.asarray(values, dtype=np.float64)
    right = np.asarray(targets, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or left.size < 2:
        raise V04MetricError("spearman inputs must be same-length vectors")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise V04MetricError("spearman inputs must be finite")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def evaluate_ranking(
    *,
    context_id: str,
    task_id: str,
    method_id: str,
    budget_episodes: int,
    ranked_candidate_ids: Sequence[str],
    scores: Mapping[str, float],
    oracle_returns: Mapping[str, float],
    epsilon: float,
) -> dict[str, Any]:
    ranked = tuple(str(value) for value in ranked_candidate_ids)
    if len(ranked) != 5 or len(set(ranked)) != 5:
        raise V04MetricError("TASK_5 ranking must contain five unique candidates")
    if set(ranked) != set(scores) or set(ranked) != set(oracle_returns):
        raise V04MetricError("ranking, scores and oracle must cover the same TASK_5")
    score_values = np.asarray([scores[key] for key in ranked], dtype=np.float64)
    returns = np.asarray([oracle_returns[key] for key in ranked], dtype=np.float64)
    if not np.all(np.isfinite(score_values)) or not np.all(np.isfinite(returns)):
        raise V04MetricError("scores and oracle returns must be finite")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise V04MetricError("epsilon must be finite and non-negative")

    best = float(np.max(returns))
    selected = float(returns[0])
    top3 = float(np.max(returns[:3]))
    winner_set = tuple(
        key
        for key, value in oracle_returns.items()
        if np.isclose(float(value), best, rtol=0.0, atol=1.0e-12)
    )
    return {
        "context_id": context_id,
        "task_id": task_id,
        "method_id": method_id,
        "budget_episodes": int(budget_episodes),
        "selected_candidate_id": ranked[0],
        "winner_set": sorted(winner_set),
        "hit_at_1": ranked[0] in winner_set,
        "epsilon_optimal": best - selected <= float(epsilon) + 1.0e-12,
        "regret_at_1": best - selected,
        "regret_at_3": best - top3,
        "selected_j_norm": selected,
        "oracle_best_j_norm": best,
        "spearman": spearman(score_values, returns),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise V04MetricError("cannot aggregate an empty metric table")
    required = {
        "hit_at_1",
        "epsilon_optimal",
        "regret_at_1",
        "regret_at_3",
        "selected_j_norm",
    }
    if any(not required.issubset(row) for row in rows):
        raise V04MetricError("metric row is incomplete")
    correlations = [
        float(row["spearman"]) for row in rows if row.get("spearman") is not None
    ]
    return {
        "row_count": len(rows),
        "hit_at_1": float(np.mean([bool(row["hit_at_1"]) for row in rows])),
        "epsilon_optimal": float(
            np.mean([bool(row["epsilon_optimal"]) for row in rows])
        ),
        "regret_at_1": float(np.mean([float(row["regret_at_1"]) for row in rows])),
        "regret_at_3": float(np.mean([float(row["regret_at_3"]) for row in rows])),
        "selected_j_norm": float(
            np.mean([float(row["selected_j_norm"]) for row in rows])
        ),
        "spearman": None if not correlations else float(np.mean(correlations)),
    }


def hierarchical_bootstrap_intervals(
    ranking_rows: Sequence[Mapping[str, Any]],
    oracle_episode_returns: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    epsilon: float,
    seed: int,
    replicates: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Task→context→paired-oracle-episode bootstrap for one result cell.

    ``ranking_rows`` must be the 24 development rankings for exactly one
    method/budget.  Candidate episode vectors are sampled by a common index
    within each context, preserving the common-reset pairing across TASK_5.
    A repeated context occurrence receives an independent lower-level episode
    resample.
    """

    if len(ranking_rows) != 24:
        raise V04MetricError("hierarchical bootstrap requires 24 context rows")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise V04MetricError("bootstrap seed must be a non-negative integer")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 100
    ):
        raise V04MetricError("bootstrap requires at least 100 replicates")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise V04MetricError("epsilon must be finite and non-negative")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise V04MetricError("confidence_level must lie strictly inside (0,1)")

    rows = sorted(ranking_rows, key=lambda row: str(row.get("context_id", "")))
    context_ids = [str(row.get("context_id", "")) for row in rows]
    if not all(context_ids) or len(set(context_ids)) != 24:
        raise V04MetricError("bootstrap context identities must be unique")
    methods = {str(row.get("method_id", "")) for row in rows}
    budgets = {int(row.get("budget_episodes", -1)) for row in rows}
    if len(methods) != 1 or len(budgets) != 1:
        raise V04MetricError("bootstrap rows must share one method and budget")

    tasks = sorted({str(row.get("task_id", "")) for row in rows})
    if len(tasks) != 6 or not all(tasks):
        raise V04MetricError("bootstrap requires six task clusters")
    task_context_indices: list[list[int]] = []
    episode_matrices: list[np.ndarray] = []
    for task_id in tasks:
        indices = [
            index
            for index, row in enumerate(rows)
            if str(row.get("task_id")) == task_id
        ]
        if len(indices) != 4:
            raise V04MetricError(
                "each bootstrap task cluster must contain four contexts"
            )
        task_context_indices.append(indices)
    for row in rows:
        context_id = str(row["context_id"])
        ranking = row.get("ranking")
        evidence = oracle_episode_returns.get(context_id)
        if (
            not isinstance(ranking, list)
            or len(ranking) != 5
            or not isinstance(evidence, Mapping)
        ):
            raise V04MetricError("bootstrap ranking/oracle evidence is incomplete")
        candidates = [str(item.get("opaque_candidate_id", "")) for item in ranking]
        if len(set(candidates)) != 5 or set(candidates) != set(evidence):
            raise V04MetricError("bootstrap TASK_5 identities differ")
        matrix = np.asarray(
            [evidence[candidate] for candidate in candidates], dtype=np.float64
        )
        if matrix.shape != (5, 50) or not np.all(np.isfinite(matrix)):
            raise V04MetricError("bootstrap oracle evidence must be finite TASK_5 x 50")
        episode_matrices.append(matrix)
    rng = np.random.default_rng(seed)
    task_matrix = np.asarray(task_context_indices, dtype=np.int64)
    sampled_tasks = rng.integers(0, 6, size=(replicates, 6))
    sampled_context_offsets = rng.integers(0, 4, size=(replicates, 6, 4))
    sampled_contexts = task_matrix[
        sampled_tasks[:, :, None], sampled_context_offsets
    ].reshape(replicates, 24)
    evidence_array = np.asarray(episode_matrices, dtype=np.float64)
    resampled_means = np.empty((replicates, 24, 5), dtype=np.float64)
    for slot in range(24):
        selected = evidence_array[sampled_contexts[:, slot]]
        episode_indices = rng.integers(0, 50, size=(replicates, 50))
        sampled = np.take_along_axis(selected, episode_indices[:, None, :], axis=2)
        resampled_means[:, slot, :] = np.mean(sampled, axis=2)

    best = np.max(resampled_means, axis=2)
    selected = resampled_means[:, :, 0]
    top3 = np.max(resampled_means[:, :, :3], axis=2)
    replicate_metrics = {
        "hit_at_1": np.mean(np.isclose(selected, best, rtol=0.0, atol=1.0e-12), axis=1),
        "epsilon_optimal": np.mean(best - selected <= float(epsilon) + 1.0e-12, axis=1),
        "regret_at_1": np.mean(best - selected, axis=1),
        "regret_at_3": np.mean(best - top3, axis=1),
        "selected_j_norm": np.mean(selected, axis=1),
    }
    alpha = (1.0 - float(confidence_level)) / 2.0
    intervals = {
        name: {
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
        }
        for name, values in replicate_metrics.items()
    }
    return {
        "method_id": next(iter(methods)),
        "budget_episodes": next(iter(budgets)),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": float(confidence_level),
        "hierarchy": ["task", "context", "paired_oracle_episode"],
        "task_count": 6,
        "contexts_per_task": 4,
        "oracle_episodes_per_candidate": 50,
        "intervals": intervals,
    }


__all__ = [
    "V04MetricError",
    "aggregate_metrics",
    "evaluate_ranking",
    "hierarchical_bootstrap_intervals",
    "spearman",
]
