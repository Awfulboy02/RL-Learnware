"""Strict metric primitives for the v0.2 source-anchor market.

The functions in this module operate only on already-authorized public
selection records and private oracle aggregates.  They do not load artifacts
or infer missing candidates.  Every aggregation level is explicit so a large
task, axis, context, or query-bank count cannot silently change the registered
macro weighting contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np


PrefixScale = Literal["linear", "log2"]


class MetricContractError(ValueError):
    """An input does not satisfy the frozen v0.2 metric contract."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MetricContractError(f"{where} must be a non-empty canonical string")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise MetricContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MetricContractError(f"{where} must be finite")
    return result


def _unit_interval(value: Any, where: str) -> float:
    result = _finite(value, where)
    if result < 0.0 or result > 1.0:
        raise MetricContractError(f"{where} must lie in [0, 1]")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise MetricContractError(f"{where} must be an integer")
    result = int(value)
    if result <= 0:
        raise MetricContractError(f"{where} must be positive")
    return result


def _readonly_vector(value: Any, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise MetricContractError(f"{where} must be a finite non-empty vector")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReturnSummary:
    """Summary of normalized episode returns on the registered [0, 1] scale."""

    episode_count: int
    mean: float
    std: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        count = _positive_int(self.episode_count, "episode_count")
        mean = _unit_interval(self.mean, "mean")
        std = _finite(self.std, "std")
        minimum = _unit_interval(self.minimum, "minimum")
        maximum = _unit_interval(self.maximum, "maximum")
        if std < 0.0:
            raise MetricContractError("std must be non-negative")
        if minimum > mean or mean > maximum:
            raise MetricContractError("return summary endpoints must contain the mean")
        object.__setattr__(self, "episode_count", count)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def normalize_episode_returns(
    episode_returns: Sequence[float] | np.ndarray,
    *,
    horizon: int,
    per_step_lower: float = 0.0,
    per_step_upper: float = 1.0,
) -> np.ndarray:
    """Normalize fixed-horizon returns using a frozen per-step reward range.

    Values outside the declared attainable range are rejected rather than
    clipped.  This catches a reward-contract or horizon mismatch before it can
    enter regret or cross-task aggregation.
    """

    values = _readonly_vector(episode_returns, "episode_returns")
    steps = _positive_int(horizon, "horizon")
    lower = _finite(per_step_lower, "per_step_lower")
    upper = _finite(per_step_upper, "per_step_upper")
    if upper <= lower:
        raise MetricContractError("per_step_upper must exceed per_step_lower")
    total_lower = steps * lower
    total_upper = steps * upper
    tolerance = 1e-12 * max(1.0, abs(total_lower), abs(total_upper))
    if np.any(values < total_lower - tolerance) or np.any(values > total_upper + tolerance):
        raise MetricContractError("episode return violates the frozen reward/horizon range")
    normalized = (np.clip(values, total_lower, total_upper) - total_lower) / (
        total_upper - total_lower
    )
    result = np.array(normalized, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def summarize_normalized_returns(
    normalized_returns: Sequence[float] | np.ndarray,
) -> ReturnSummary:
    values = _readonly_vector(normalized_returns, "normalized_returns")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise MetricContractError("normalized_returns must lie in [0, 1]")
    return ReturnSummary(
        episode_count=int(values.size),
        mean=float(values.mean()),
        std=float(values.std(ddof=0)),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


@dataclass(frozen=True)
class SelectionMetrics:
    """Selected value and finite-pool regret for one target query bank."""

    selected_policy_id: str
    selected_executable: bool
    selected_normalized_return: float
    oracle_best_normalized_return: float
    oracle_best_policy_ids: tuple[str, ...]
    pool_regret: float
    epsilon: float
    epsilon_optimal: bool
    top1_agreement: bool

    def __post_init__(self) -> None:
        selected = _nonempty(self.selected_policy_id, "selected_policy_id")
        if type(self.selected_executable) is not bool:
            raise MetricContractError("selected_executable must be boolean")
        selected_return = _unit_interval(
            self.selected_normalized_return, "selected_normalized_return"
        )
        best_return = _unit_interval(
            self.oracle_best_normalized_return, "oracle_best_normalized_return"
        )
        best_ids = tuple(
            _nonempty(item, "oracle_best_policy_ids[]")
            for item in self.oracle_best_policy_ids
        )
        if not best_ids or len(best_ids) != len(set(best_ids)) or best_ids != tuple(sorted(best_ids)):
            raise MetricContractError(
                "oracle_best_policy_ids must be non-empty, unique, and sorted"
            )
        regret = _finite(self.pool_regret, "pool_regret")
        if regret < 0.0 or not math.isclose(
            regret, best_return - selected_return, rel_tol=0.0, abs_tol=1e-12
        ):
            raise MetricContractError("pool_regret does not reconcile with oracle-selected value")
        epsilon = _unit_interval(self.epsilon, "epsilon")
        expected_epsilon = regret <= epsilon + 1e-12
        expected_top1 = self.selected_executable and selected in best_ids
        if type(self.epsilon_optimal) is not bool or self.epsilon_optimal != expected_epsilon:
            raise MetricContractError("epsilon_optimal does not match pool_regret")
        if type(self.top1_agreement) is not bool or self.top1_agreement != expected_top1:
            raise MetricContractError("top1_agreement does not match the oracle tie set")
        object.__setattr__(self, "selected_policy_id", selected)
        object.__setattr__(self, "selected_normalized_return", selected_return)
        object.__setattr__(self, "oracle_best_normalized_return", best_return)
        object.__setattr__(self, "oracle_best_policy_ids", best_ids)
        object.__setattr__(self, "pool_regret", regret)
        object.__setattr__(self, "epsilon", epsilon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_policy_id": self.selected_policy_id,
            "selected_executable": self.selected_executable,
            "selected_normalized_return": self.selected_normalized_return,
            "oracle_best_normalized_return": self.oracle_best_normalized_return,
            "oracle_best_policy_ids": list(self.oracle_best_policy_ids),
            "pool_regret": self.pool_regret,
            "epsilon": self.epsilon,
            "epsilon_optimal": self.epsilon_optimal,
            "top1_agreement": self.top1_agreement,
        }


def compute_selection_metrics(
    *,
    selected_policy_id: str,
    normalized_returns_by_policy: Mapping[str, float],
    executable_policy_ids: Sequence[str] | None = None,
    incompatible_failure_value: float | None = None,
    epsilon: float = 0.0,
    tie_tolerance: float = 0.0,
) -> SelectionMetrics:
    """Compute selected return and regret against the private executable pool.

    ``normalized_returns_by_policy`` must contain values for every executable
    policy and may contain no unregistered extras when an explicit executable
    set is supplied.  A selected incompatible policy requires an explicit
    preregistered failure value; no next-ranked fallback is performed.
    """

    selected = _nonempty(selected_policy_id, "selected_policy_id")
    if not isinstance(normalized_returns_by_policy, Mapping) or not normalized_returns_by_policy:
        raise MetricContractError("normalized_returns_by_policy must be a non-empty mapping")
    parsed: dict[str, float] = {}
    for raw_id, raw_value in normalized_returns_by_policy.items():
        policy_id = _nonempty(raw_id, "normalized_returns_by_policy key")
        if policy_id in parsed:
            raise MetricContractError("policy IDs must be unique")
        parsed[policy_id] = _unit_interval(
            raw_value, f"normalized_returns_by_policy[{policy_id!r}]"
        )
    if executable_policy_ids is None:
        executable = frozenset(parsed)
    else:
        executable_tuple = tuple(
            _nonempty(item, "executable_policy_ids[]") for item in executable_policy_ids
        )
        if not executable_tuple or len(executable_tuple) != len(set(executable_tuple)):
            raise MetricContractError("executable_policy_ids must be non-empty and unique")
        executable = frozenset(executable_tuple)
        if executable != set(parsed):
            raise MetricContractError(
                "oracle values must exactly cover the declared executable policy set"
            )
    tolerance = _finite(tie_tolerance, "tie_tolerance")
    if tolerance < 0.0:
        raise MetricContractError("tie_tolerance must be non-negative")
    epsilon_value = _unit_interval(epsilon, "epsilon")
    best = max(parsed[policy_id] for policy_id in executable)
    best_ids = tuple(
        sorted(policy_id for policy_id in executable if best - parsed[policy_id] <= tolerance)
    )
    selected_executable = selected in executable
    if selected_executable:
        selected_value = parsed[selected]
        if incompatible_failure_value is not None:
            _unit_interval(incompatible_failure_value, "incompatible_failure_value")
    else:
        if incompatible_failure_value is None:
            raise MetricContractError(
                "an incompatible selection requires incompatible_failure_value"
            )
        selected_value = _unit_interval(
            incompatible_failure_value, "incompatible_failure_value"
        )
        if selected_value > best:
            raise MetricContractError(
                "incompatible_failure_value cannot exceed the executable-pool oracle"
            )
    regret = best - selected_value
    return SelectionMetrics(
        selected_policy_id=selected,
        selected_executable=selected_executable,
        selected_normalized_return=selected_value,
        oracle_best_normalized_return=best,
        oracle_best_policy_ids=best_ids,
        pool_regret=regret,
        epsilon=epsilon_value,
        epsilon_optimal=regret <= epsilon_value + 1e-12,
        top1_agreement=selected_executable and selected in best_ids,
    )


@dataclass(frozen=True)
class RankingMetrics:
    policy_count: int
    pair_count: int
    concordant_pairs: int
    discordant_pairs: int
    oracle_tie_pairs: int
    pairwise_accuracy: float
    kendall_tau_b: float | None
    spearman_rho: float | None
    top1_agreement: bool

    def __post_init__(self) -> None:
        count = _positive_int(self.policy_count, "policy_count")
        if count < 2:
            raise MetricContractError("ranking metrics require at least two policies")
        expected_pairs = count * (count - 1) // 2
        if self.pair_count != expected_pairs:
            raise MetricContractError("pair_count does not match policy_count")
        components = (
            self.concordant_pairs,
            self.discordant_pairs,
            self.oracle_tie_pairs,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in components
        ) or sum(components) != expected_pairs:
            raise MetricContractError("ranking pair counts are invalid")
        accuracy = _unit_interval(self.pairwise_accuracy, "pairwise_accuracy")
        for name in ("kendall_tau_b", "spearman_rho"):
            value = getattr(self, name)
            if value is not None:
                parsed = _finite(value, name)
                if parsed < -1.0 or parsed > 1.0:
                    raise MetricContractError(f"{name} must lie in [-1, 1]")
                object.__setattr__(self, name, parsed)
        if type(self.top1_agreement) is not bool:
            raise MetricContractError("top1_agreement must be boolean")
        object.__setattr__(self, "policy_count", count)
        object.__setattr__(self, "pairwise_accuracy", accuracy)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _average_descending_ranks(values: np.ndarray, tolerance: float) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and abs(values[order[stop]] - values[order[start]]) <= tolerance:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def compute_ranking_metrics(
    predicted_ranking: Sequence[str],
    normalized_returns_by_policy: Mapping[str, float],
    *,
    tie_tolerance: float = 0.0,
) -> RankingMetrics:
    """Compare one full deterministic ranking with oracle return ordering.

    Oracle ties receive half credit in pairwise accuracy and average ranks for
    Spearman.  Kendall uses tau-b's oracle-tie correction.  Undefined
    correlations are represented by ``None`` rather than NaN.
    """

    predicted = tuple(_nonempty(item, "predicted_ranking[]") for item in predicted_ranking)
    if len(predicted) < 2 or len(predicted) != len(set(predicted)):
        raise MetricContractError("predicted_ranking must contain at least two unique IDs")
    if not isinstance(normalized_returns_by_policy, Mapping):
        raise MetricContractError("normalized_returns_by_policy must be a mapping")
    parsed = {
        _nonempty(policy_id, "normalized_returns_by_policy key"): _unit_interval(
            value, f"normalized_returns_by_policy[{policy_id!r}]"
        )
        for policy_id, value in normalized_returns_by_policy.items()
    }
    if set(predicted) != set(parsed) or len(parsed) != len(predicted):
        raise MetricContractError("predicted ranking and oracle policy IDs must match exactly")
    tolerance = _finite(tie_tolerance, "tie_tolerance")
    if tolerance < 0.0:
        raise MetricContractError("tie_tolerance must be non-negative")

    concordant = 0
    discordant = 0
    ties = 0
    for left_index, left_id in enumerate(predicted[:-1]):
        for right_id in predicted[left_index + 1 :]:
            difference = parsed[left_id] - parsed[right_id]
            if abs(difference) <= tolerance:
                ties += 1
            elif difference > 0.0:
                concordant += 1
            else:
                discordant += 1
    pair_count = len(predicted) * (len(predicted) - 1) // 2
    pairwise_accuracy = (concordant + 0.5 * ties) / pair_count
    non_ties = concordant + discordant
    if non_ties == 0:
        kendall = None
    else:
        kendall = (concordant - discordant) / math.sqrt(pair_count * non_ties)

    oracle_values = np.asarray([parsed[item] for item in predicted], dtype=np.float64)
    predicted_ranks = np.arange(1, len(predicted) + 1, dtype=np.float64)
    oracle_ranks = _average_descending_ranks(oracle_values, tolerance)
    predicted_centered = predicted_ranks - predicted_ranks.mean()
    oracle_centered = oracle_ranks - oracle_ranks.mean()
    denominator = math.sqrt(
        float(np.dot(predicted_centered, predicted_centered))
        * float(np.dot(oracle_centered, oracle_centered))
    )
    if denominator == 0.0:
        spearman = None
    else:
        spearman = float(np.dot(predicted_centered, oracle_centered) / denominator)
        spearman = min(1.0, max(-1.0, spearman))
    best = max(parsed.values())
    oracle_best = {
        policy_id for policy_id, value in parsed.items() if best - value <= tolerance
    }
    return RankingMetrics(
        policy_count=len(predicted),
        pair_count=pair_count,
        concordant_pairs=concordant,
        discordant_pairs=discordant,
        oracle_tie_pairs=ties,
        pairwise_accuracy=pairwise_accuracy,
        kendall_tau_b=kendall,
        spearman_rho=spearman,
        top1_agreement=predicted[0] in oracle_best,
    )


@dataclass(frozen=True)
class PrefixAUC:
    prefixes: tuple[int, ...]
    values: tuple[float, ...]
    x_scale: PrefixScale
    normalized_auc: float

    def __post_init__(self) -> None:
        prefixes = tuple(_positive_int(item, "prefixes[]") for item in self.prefixes)
        if len(prefixes) < 2 or any(
            right <= left for left, right in zip(prefixes, prefixes[1:])
        ):
            raise MetricContractError("prefixes must be strictly increasing with length >= 2")
        values = tuple(_finite(item, "values[]") for item in self.values)
        if len(values) != len(prefixes):
            raise MetricContractError("values must align one-to-one with prefixes")
        if self.x_scale not in {"linear", "log2"}:
            raise MetricContractError("x_scale must be 'linear' or 'log2'")
        auc = _finite(self.normalized_auc, "normalized_auc")
        object.__setattr__(self, "prefixes", prefixes)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "normalized_auc", auc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefixes": list(self.prefixes),
            "values": list(self.values),
            "x_scale": self.x_scale,
            "normalized_auc": self.normalized_auc,
        }


def compute_prefix_auc(
    prefixes: Sequence[int],
    values: Sequence[float] | np.ndarray,
    *,
    x_scale: PrefixScale = "log2",
) -> PrefixAUC:
    """Return a trapezoidal AUC normalized by the registered prefix span."""

    parsed_prefixes = tuple(_positive_int(item, "prefixes[]") for item in prefixes)
    parsed_values = tuple(_finite(item, "values[]") for item in values)
    if len(parsed_prefixes) < 2 or len(parsed_prefixes) != len(parsed_values):
        raise MetricContractError("prefix curve requires at least two aligned points")
    if any(right <= left for left, right in zip(parsed_prefixes, parsed_prefixes[1:])):
        raise MetricContractError("prefixes must be strictly increasing")
    if x_scale == "linear":
        x_values = np.asarray(parsed_prefixes, dtype=np.float64)
    elif x_scale == "log2":
        x_values = np.log2(np.asarray(parsed_prefixes, dtype=np.float64))
    else:
        raise MetricContractError("x_scale must be 'linear' or 'log2'")
    y_values = np.asarray(parsed_values, dtype=np.float64)
    span = float(x_values[-1] - x_values[0])
    if not math.isfinite(span) or span <= 0.0:
        raise MetricContractError("prefix x-axis has no positive finite span")
    area = float(np.sum(np.diff(x_values) * (y_values[:-1] + y_values[1:]) * 0.5))
    return PrefixAUC(
        prefixes=parsed_prefixes,
        values=parsed_values,
        x_scale=x_scale,
        normalized_auc=area / span,
    )


@dataclass(frozen=True)
class HierarchicalValue:
    """One episode/bank leaf under task -> axis -> context."""

    task_id: str
    axis_id: str
    context_id: str
    observation_id: str
    value: float

    def __post_init__(self) -> None:
        for name in ("task_id", "axis_id", "context_id", "observation_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(self, "value", _finite(self.value, "value"))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.task_id, self.axis_id, self.context_id, self.observation_id)


@dataclass(frozen=True)
class ContextAggregate:
    task_id: str
    axis_id: str
    context_id: str
    observation_count: int
    mean: float

    def __post_init__(self) -> None:
        for name in ("task_id", "axis_id", "context_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self, "observation_count", _positive_int(self.observation_count, "observation_count")
        )
        object.__setattr__(self, "mean", _finite(self.mean, "mean"))


@dataclass(frozen=True)
class AxisAggregate:
    task_id: str
    axis_id: str
    context_count: int
    mean: float

    def __post_init__(self) -> None:
        for name in ("task_id", "axis_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self, "context_count", _positive_int(self.context_count, "context_count")
        )
        object.__setattr__(self, "mean", _finite(self.mean, "mean"))


@dataclass(frozen=True)
class TaskAggregate:
    task_id: str
    axis_count: int
    mean: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _nonempty(self.task_id, "task_id"))
        object.__setattr__(self, "axis_count", _positive_int(self.axis_count, "axis_count"))
        object.__setattr__(self, "mean", _finite(self.mean, "mean"))


@dataclass(frozen=True)
class HierarchicalAggregate:
    macro_mean: float
    task_count: int
    axis_count: int
    context_count: int
    observation_count: int
    task_aggregates: tuple[TaskAggregate, ...]
    axis_aggregates: tuple[AxisAggregate, ...]
    context_aggregates: tuple[ContextAggregate, ...]

    def __post_init__(self) -> None:
        macro = _finite(self.macro_mean, "macro_mean")
        task_count = _positive_int(self.task_count, "task_count")
        axis_count = _positive_int(self.axis_count, "axis_count")
        context_count = _positive_int(self.context_count, "context_count")
        observation_count = _positive_int(self.observation_count, "observation_count")
        tasks = tuple(self.task_aggregates)
        axes = tuple(self.axis_aggregates)
        contexts = tuple(self.context_aggregates)
        if (
            len(tasks) != task_count
            or len(axes) != axis_count
            or len(contexts) != context_count
            or any(not isinstance(item, TaskAggregate) for item in tasks)
            or any(not isinstance(item, AxisAggregate) for item in axes)
            or any(not isinstance(item, ContextAggregate) for item in contexts)
            or sum(item.observation_count for item in contexts) != observation_count
        ):
            raise MetricContractError("hierarchical aggregate counts do not reconcile")
        if not math.isclose(
            macro,
            float(np.mean([item.mean for item in tasks], dtype=np.float64)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MetricContractError("macro_mean does not reconcile with task means")
        object.__setattr__(self, "macro_mean", macro)
        object.__setattr__(self, "task_count", task_count)
        object.__setattr__(self, "axis_count", axis_count)
        object.__setattr__(self, "context_count", context_count)
        object.__setattr__(self, "observation_count", observation_count)
        object.__setattr__(self, "task_aggregates", tasks)
        object.__setattr__(self, "axis_aggregates", axes)
        object.__setattr__(self, "context_aggregates", contexts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_mean": self.macro_mean,
            "task_count": self.task_count,
            "axis_count": self.axis_count,
            "context_count": self.context_count,
            "observation_count": self.observation_count,
            "task_aggregates": [item.__dict__ for item in self.task_aggregates],
            "axis_aggregates": [item.__dict__ for item in self.axis_aggregates],
            "context_aggregates": [item.__dict__ for item in self.context_aggregates],
        }


def aggregate_hierarchy(values: Sequence[HierarchicalValue]) -> HierarchicalAggregate:
    """Aggregate with equal weight at every registered hierarchy level.

    Leaves are averaged within context, context means within axis, axis means
    within task, and task means into the macro mean.  Duplicate leaves and a
    context ID appearing under multiple owners fail closed.
    """

    rows = tuple(values)
    if not rows or any(not isinstance(row, HierarchicalValue) for row in rows):
        raise MetricContractError("values must be a non-empty HierarchicalValue sequence")
    keys = [row.key for row in rows]
    if len(keys) != len(set(keys)):
        raise MetricContractError("hierarchical observation keys must be unique")
    context_owners: dict[str, tuple[str, str]] = {}
    nested: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        owner = (row.task_id, row.axis_id)
        previous = context_owners.setdefault(row.context_id, owner)
        if previous != owner:
            raise MetricContractError("a context_id cannot belong to multiple task/axis owners")
        nested.setdefault(row.task_id, {}).setdefault(row.axis_id, {}).setdefault(
            row.context_id, []
        ).append(row.value)

    contexts: list[ContextAggregate] = []
    axes: list[AxisAggregate] = []
    tasks: list[TaskAggregate] = []
    for task_id in sorted(nested):
        axis_means: list[float] = []
        for axis_id in sorted(nested[task_id]):
            context_means: list[float] = []
            for context_id in sorted(nested[task_id][axis_id]):
                leaves = nested[task_id][axis_id][context_id]
                context_mean = float(np.mean(leaves, dtype=np.float64))
                contexts.append(
                    ContextAggregate(
                        task_id=task_id,
                        axis_id=axis_id,
                        context_id=context_id,
                        observation_count=len(leaves),
                        mean=context_mean,
                    )
                )
                context_means.append(context_mean)
            axis_mean = float(np.mean(context_means, dtype=np.float64))
            axes.append(
                AxisAggregate(
                    task_id=task_id,
                    axis_id=axis_id,
                    context_count=len(context_means),
                    mean=axis_mean,
                )
            )
            axis_means.append(axis_mean)
        task_mean = float(np.mean(axis_means, dtype=np.float64))
        tasks.append(TaskAggregate(task_id=task_id, axis_count=len(axis_means), mean=task_mean))
    macro = float(np.mean([item.mean for item in tasks], dtype=np.float64))
    return HierarchicalAggregate(
        macro_mean=macro,
        task_count=len(tasks),
        axis_count=len(axes),
        context_count=len(contexts),
        observation_count=len(rows),
        task_aggregates=tuple(tasks),
        axis_aggregates=tuple(axes),
        context_aggregates=tuple(contexts),
    )


__all__ = [
    "AxisAggregate",
    "ContextAggregate",
    "HierarchicalAggregate",
    "HierarchicalValue",
    "MetricContractError",
    "PrefixAUC",
    "PrefixScale",
    "RankingMetrics",
    "ReturnSummary",
    "SelectionMetrics",
    "TaskAggregate",
    "aggregate_hierarchy",
    "compute_prefix_auc",
    "compute_ranking_metrics",
    "compute_selection_metrics",
    "normalize_episode_returns",
    "summarize_normalized_returns",
]
