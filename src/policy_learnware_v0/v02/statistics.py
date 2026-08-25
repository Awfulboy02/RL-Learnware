"""Deterministic hierarchical statistics for Policy Learnware v0.2.

The registered macro endpoint is task-equal, axis-equal, context-equal, with
episode/query-bank leaves below each context.  Bootstrap resampling mirrors
that hierarchy exactly.  The implementation intentionally depends only on
NumPy and the standard library so independent recompute does not depend on a
server's optional scientific stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .metrics import (
    HierarchicalValue,
    MetricContractError,
    aggregate_hierarchy,
)


class StatisticalContractError(ValueError):
    """A statistical input violates the frozen resampling contract."""


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise StatisticalContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StatisticalContractError(f"{where} must be finite")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise StatisticalContractError(f"{where} must be an integer")
    result = int(value)
    if result <= 0:
        raise StatisticalContractError(f"{where} must be positive")
    return result


def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise StatisticalContractError("seed must be an integer")
    result = int(value)
    if result < 0 or result >= 2**64:
        raise StatisticalContractError("seed must lie in [0, 2**64)")
    return result


def _confidence(value: Any) -> float:
    result = _finite(value, "confidence_level")
    if not 0.0 < result < 1.0:
        raise StatisticalContractError("confidence_level must lie strictly in (0, 1)")
    return result


def _readonly_vector(value: Any, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise StatisticalContractError(f"{where} must be a finite non-empty vector")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def derive_bootstrap_seed(*parts: object) -> int:
    """Derive a stable 64-bit seed from an unambiguous namespace tuple."""

    if not parts:
        raise StatisticalContractError("at least one bootstrap seed namespace is required")
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float
    confidence_level: float
    sides: str = "two-sided"

    def __post_init__(self) -> None:
        low = _finite(self.low, "low")
        high = _finite(self.high, "high")
        level = _confidence(self.confidence_level)
        if low > high:
            raise StatisticalContractError("confidence interval endpoints are reversed")
        if self.sides != "two-sided":
            raise StatisticalContractError("ConfidenceInterval must be two-sided")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "confidence_level", level)

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "confidence_level": self.confidence_level,
            "sides": self.sides,
        }


def percentile_interval(
    replicates: Sequence[float] | np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    values = _readonly_vector(replicates, "replicates")
    level = _confidence(confidence_level)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail], method="linear")
    return ConfidenceInterval(float(low), float(high), level)


def one_sided_lower_bound(
    replicates: Sequence[float] | np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> float:
    """Return the percentile lower confidence bound for a one-sided test."""

    values = _readonly_vector(replicates, "replicates")
    level = _confidence(confidence_level)
    return float(np.quantile(values, 1.0 - level, method="linear"))


@dataclass(frozen=True)
class HierarchicalBootstrapResult:
    observed: float
    replicates: np.ndarray
    confidence_level: float
    seed: int
    resampling_plan_digest: str
    resampling_contract: str = "task_axis_context_episode_bank_hierarchical_equal_weight"

    def __post_init__(self) -> None:
        observed = _finite(self.observed, "observed")
        replicates = _readonly_vector(self.replicates, "replicates")
        level = _confidence(self.confidence_level)
        seed = _seed(self.seed)
        if self.resampling_contract != "task_axis_context_episode_bank_hierarchical_equal_weight":
            raise StatisticalContractError("unexpected hierarchical resampling contract")
        if (
            not isinstance(self.resampling_plan_digest, str)
            or len(self.resampling_plan_digest) != 64
            or self.resampling_plan_digest.lower() != self.resampling_plan_digest
        ):
            raise StatisticalContractError(
                "resampling_plan_digest must be a lowercase SHA-256 digest"
            )
        try:
            int(self.resampling_plan_digest, 16)
        except ValueError as error:
            raise StatisticalContractError(
                "resampling_plan_digest must be a lowercase SHA-256 digest"
            ) from error
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(self, "seed", seed)

    @property
    def interval(self) -> ConfidenceInterval:
        return percentile_interval(
            self.replicates, confidence_level=self.confidence_level
        )

    @property
    def one_sided_lower(self) -> float:
        return one_sided_lower_bound(
            self.replicates, confidence_level=self.confidence_level
        )

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "ci": self.interval.to_dict(),
            "one_sided_lower": self.one_sided_lower,
            "bootstrap_resamples": int(self.replicates.size),
            "bootstrap_seed": self.seed,
            "bootstrap_method": "deterministic_hierarchical_percentile",
            "resampling_contract": self.resampling_contract,
            "resampling_plan_digest": self.resampling_plan_digest,
        }


def _nested_leaves(
    values: Sequence[HierarchicalValue],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    rows = tuple(values)
    try:
        aggregate_hierarchy(rows)
    except MetricContractError as error:
        raise StatisticalContractError(str(error)) from error
    nested_rows: dict[str, dict[str, dict[str, list[tuple[str, float]]]]] = {}
    for row in rows:
        nested_rows.setdefault(row.task_id, {}).setdefault(row.axis_id, {}).setdefault(
            row.context_id, []
        ).append((row.observation_id, row.value))
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for task_id in sorted(nested_rows):
        result[task_id] = {}
        for axis_id in sorted(nested_rows[task_id]):
            result[task_id][axis_id] = {}
            for context_id in sorted(nested_rows[task_id][axis_id]):
                ordered = sorted(nested_rows[task_id][axis_id][context_id])
                array = np.asarray([value for _, value in ordered], dtype=np.float64)
                array.setflags(write=False)
                result[task_id][axis_id][context_id] = array
    return result


def _bootstrap_once(
    nested: dict[str, dict[str, dict[str, np.ndarray]]],
    rng: np.random.Generator,
) -> float:
    tasks = tuple(sorted(nested))
    sampled_task_indices = rng.integers(0, len(tasks), size=len(tasks))
    sampled_task_means: list[float] = []
    for task_index in sampled_task_indices:
        task_id = tasks[int(task_index)]
        axes = tuple(sorted(nested[task_id]))
        sampled_axis_indices = rng.integers(0, len(axes), size=len(axes))
        sampled_axis_means: list[float] = []
        for axis_index in sampled_axis_indices:
            axis_id = axes[int(axis_index)]
            contexts = tuple(sorted(nested[task_id][axis_id]))
            sampled_context_indices = rng.integers(0, len(contexts), size=len(contexts))
            sampled_context_means: list[float] = []
            for context_index in sampled_context_indices:
                context_id = contexts[int(context_index)]
                leaves = nested[task_id][axis_id][context_id]
                sampled_leaf_indices = rng.integers(0, leaves.size, size=leaves.size)
                sampled_context_means.append(float(leaves[sampled_leaf_indices].mean()))
            sampled_axis_means.append(float(np.mean(sampled_context_means)))
        sampled_task_means.append(float(np.mean(sampled_axis_means)))
    return float(np.mean(sampled_task_means))


def hierarchical_bootstrap(
    values: Sequence[HierarchicalValue],
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> HierarchicalBootstrapResult:
    """Bootstrap task, then axis, then context, then episode/query-bank leaves."""

    rows = tuple(values)
    nested = _nested_leaves(rows)
    total = _positive_int(resamples, "resamples")
    parsed_seed = _seed(seed)
    level = _confidence(confidence_level)
    observed = aggregate_hierarchy(rows).macro_mean
    rng = np.random.default_rng(parsed_seed)
    replicates = np.empty(total, dtype=np.float64)
    for index in range(total):
        replicates[index] = _bootstrap_once(nested, rng)
    resampling_contract = "task_axis_context_episode_bank_hierarchical_equal_weight"
    resampling_plan_digest = sha256_json(
        {
            "schema": "policy-learnware.v02-hierarchical-resampling-plan.v0",
            "resampling_contract": resampling_contract,
            "ordered_leaf_keys": [
                list(row.key) for row in sorted(rows, key=lambda row: row.key)
            ],
            "resamples": total,
            "seed": parsed_seed,
        }
    )
    return HierarchicalBootstrapResult(
        observed=observed,
        replicates=replicates,
        confidence_level=level,
        seed=parsed_seed,
        resampling_plan_digest=resampling_plan_digest,
        resampling_contract=resampling_contract,
    )


def hierarchical_paired_difference_bootstrap(
    left: Sequence[HierarchicalValue],
    right: Sequence[HierarchicalValue],
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> HierarchicalBootstrapResult:
    """Bootstrap ``left - right`` after an exact leaf-key pairing audit."""

    left_rows = tuple(left)
    right_rows = tuple(right)
    if not left_rows or not right_rows:
        raise StatisticalContractError("paired hierarchical samples cannot be empty")
    if any(not isinstance(row, HierarchicalValue) for row in left_rows + right_rows):
        raise StatisticalContractError("paired samples must contain HierarchicalValue rows")
    left_map = {row.key: row.value for row in left_rows}
    right_map = {row.key: row.value for row in right_rows}
    if len(left_map) != len(left_rows) or len(right_map) != len(right_rows):
        raise StatisticalContractError("paired sample keys must be unique")
    if set(left_map) != set(right_map):
        raise StatisticalContractError("paired hierarchical samples must have identical keys")
    differences = tuple(
        HierarchicalValue(
            task_id=key[0],
            axis_id=key[1],
            context_id=key[2],
            observation_id=key[3],
            value=left_map[key] - right_map[key],
        )
        for key in sorted(left_map)
    )
    return hierarchical_bootstrap(
        differences,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


@dataclass(frozen=True)
class SimultaneousMaxTInterval:
    hypothesis_id: str
    observed: float
    bootstrap_standard_error: float
    low: float
    high: float
    critical_value: float
    confidence_level: float

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id:
            raise StatisticalContractError("hypothesis_id must be non-empty")
        for name in (
            "observed",
            "bootstrap_standard_error",
            "low",
            "high",
            "critical_value",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.bootstrap_standard_error < 0.0 or self.critical_value < 0.0:
            raise StatisticalContractError("max-T scale and critical value must be non-negative")
        if self.low > self.high:
            raise StatisticalContractError("simultaneous interval endpoints are reversed")
        expected_delta = self.critical_value * self.bootstrap_standard_error
        if not (
            math.isclose(self.low, self.observed - expected_delta, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(self.high, self.observed + expected_delta, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise StatisticalContractError("simultaneous interval does not reconcile")
        object.__setattr__(self, "confidence_level", _confidence(self.confidence_level))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class SimultaneousMaxTResult:
    intervals: Mapping[str, SimultaneousMaxTInterval]
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    resampling_plan_digest: str
    method: str = "paired_studentized_bootstrap_max-T"

    def __post_init__(self) -> None:
        values = dict(self.intervals)
        if not values or any(
            key != interval.hypothesis_id
            or not isinstance(interval, SimultaneousMaxTInterval)
            for key, interval in values.items()
        ):
            raise StatisticalContractError("max-T intervals require a keyed typed family")
        level = _confidence(self.confidence_level)
        if any(interval.confidence_level != level for interval in values.values()):
            raise StatisticalContractError("max-T interval confidence levels differ")
        if self.method != "paired_studentized_bootstrap_max-T":
            raise StatisticalContractError("unsupported simultaneous interval method")
        object.__setattr__(self, "intervals", MappingProxyType(dict(sorted(values.items()))))
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(
            self, "bootstrap_resamples", _positive_int(self.bootstrap_resamples, "bootstrap_resamples")
        )
        object.__setattr__(self, "bootstrap_seed", _seed(self.bootstrap_seed))
        if (
            not isinstance(self.resampling_plan_digest, str)
            or len(self.resampling_plan_digest) != 64
            or self.resampling_plan_digest.lower() != self.resampling_plan_digest
        ):
            raise StatisticalContractError(
                "max-T resampling_plan_digest must be a lowercase SHA-256 digest"
            )
        try:
            int(self.resampling_plan_digest, 16)
        except ValueError as error:
            raise StatisticalContractError(
                "max-T resampling_plan_digest must be a lowercase SHA-256 digest"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-simultaneous-max-t.v0",
            "method": self.method,
            "confidence_level": self.confidence_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "resampling_plan_digest": self.resampling_plan_digest,
            "intervals": {
                key: value.to_dict() for key, value in self.intervals.items()
            },
        }


def bootstrap_max_t_intervals(
    family: Mapping[str, HierarchicalBootstrapResult],
) -> SimultaneousMaxTResult:
    """Construct paired studentized max-T intervals for one frozen family.

    All members must come from the same exact hierarchy/leaf resampling plan,
    bootstrap seed, replicate count, confidence level, and resampling
    algorithm.  The plan digest prevents unrelated marginal bootstraps with
    coincidentally equal seeds and lengths from masquerading as joint draws.
    """

    if not isinstance(family, Mapping) or not family:
        raise StatisticalContractError("max-T requires a non-empty hypothesis family")
    parsed: dict[str, HierarchicalBootstrapResult] = {}
    for name, result in family.items():
        if not isinstance(name, str) or not name or name.strip() != name:
            raise StatisticalContractError("max-T hypothesis IDs must be canonical strings")
        if not isinstance(result, HierarchicalBootstrapResult):
            raise StatisticalContractError("max-T family values must be bootstrap results")
        parsed[name] = result
    first = next(iter(parsed.values()))
    contract = (
        int(first.replicates.size),
        first.seed,
        first.confidence_level,
        first.resampling_contract,
        first.resampling_plan_digest,
    )
    if contract[0] < 2:
        raise StatisticalContractError("studentized max-T requires at least two replicates")
    if any(
        (
            int(result.replicates.size),
            result.seed,
            result.confidence_level,
            result.resampling_contract,
            result.resampling_plan_digest,
        )
        != contract
        for result in parsed.values()
    ):
        raise StatisticalContractError(
            "max-T family must share paired exact hierarchy/leaf resampling plan"
        )
    ordered = tuple(sorted(parsed))
    centered = np.stack(
        [parsed[name].replicates - parsed[name].observed for name in ordered], axis=1
    )
    scales = np.std(centered, axis=0, ddof=1)
    zero = scales == 0.0
    if np.any(zero & np.any(centered != 0.0, axis=0)):
        raise StatisticalContractError("zero-variance max-T member has nonzero deviations")
    standardized = np.zeros_like(centered)
    nonzero = ~zero
    standardized[:, nonzero] = np.abs(centered[:, nonzero] / scales[nonzero])
    maxima = np.max(standardized, axis=1)
    critical = float(
        np.quantile(maxima, first.confidence_level, method="higher")
    )
    intervals = {
        name: SimultaneousMaxTInterval(
            hypothesis_id=name,
            observed=parsed[name].observed,
            bootstrap_standard_error=float(scales[index]),
            low=parsed[name].observed - critical * float(scales[index]),
            high=parsed[name].observed + critical * float(scales[index]),
            critical_value=critical,
            confidence_level=first.confidence_level,
        )
        for index, name in enumerate(ordered)
    }
    return SimultaneousMaxTResult(
        intervals=intervals,
        confidence_level=first.confidence_level,
        bootstrap_resamples=contract[0],
        bootstrap_seed=first.seed,
        resampling_plan_digest=first.resampling_plan_digest,
    )


@dataclass(frozen=True)
class HolmResult:
    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    correction_order: int
    family_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id:
            raise StatisticalContractError("hypothesis_id must be non-empty")
        raw = _finite(self.raw_p_value, "raw_p_value")
        adjusted = _finite(self.adjusted_p_value, "adjusted_p_value")
        if not 0.0 <= raw <= 1.0 or not raw <= adjusted <= 1.0:
            raise StatisticalContractError("Holm p-values are invalid")
        order = _positive_int(self.correction_order, "correction_order")
        size = _positive_int(self.family_size, "family_size")
        if order > size:
            raise StatisticalContractError("correction_order exceeds family_size")
        object.__setattr__(self, "raw_p_value", raw)
        object.__setattr__(self, "adjusted_p_value", adjusted)
        object.__setattr__(self, "correction_order", order)
        object.__setattr__(self, "family_size", size)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, HolmResult]:
    """Adjust one complete, preregistered family with deterministic tie order."""

    if not isinstance(p_values, Mapping) or not p_values:
        raise StatisticalContractError("Holm correction requires a non-empty mapping")
    parsed: dict[str, float] = {}
    for hypothesis_id, raw_value in p_values.items():
        if not isinstance(hypothesis_id, str) or not hypothesis_id or hypothesis_id.strip() != hypothesis_id:
            raise StatisticalContractError("hypothesis IDs must be canonical non-empty strings")
        value = _finite(raw_value, f"p_values[{hypothesis_id!r}]")
        if not 0.0 <= value <= 1.0:
            raise StatisticalContractError("p-values must lie in [0, 1]")
        parsed[hypothesis_id] = value
    ordered = sorted(parsed.items(), key=lambda item: (item[1], item[0]))
    size = len(ordered)
    running = 0.0
    results: dict[str, HolmResult] = {}
    for index, (hypothesis_id, raw_value) in enumerate(ordered):
        running = max(running, min(1.0, (size - index) * raw_value))
        results[hypothesis_id] = HolmResult(
            hypothesis_id=hypothesis_id,
            raw_p_value=raw_value,
            adjusted_p_value=running,
            correction_order=index + 1,
            family_size=size,
        )
    return results


def centered_one_sided_p_value(
    observed: float,
    replicates: Sequence[float] | np.ndarray,
    *,
    null_boundary: float,
) -> float:
    """Centered-bootstrap upper-tail p-value for ``H0: effect <= boundary``."""

    point = _finite(observed, "observed")
    boundary = _finite(null_boundary, "null_boundary")
    values = _readonly_vector(replicates, "replicates")
    distance_from_null = point - boundary
    centered_null = values - point
    exceedances = int(np.count_nonzero(centered_null >= distance_from_null))
    return float((1 + exceedances) / (values.size + 1))


@dataclass(frozen=True)
class NonInferiorityResult:
    observed_difference: float
    margin: float
    null_boundary: float
    one_sided_lower_bound: float
    confidence_level: float
    raw_p_value: float
    passed: bool
    bootstrap_resamples: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        observed = _finite(self.observed_difference, "observed_difference")
        margin = _finite(self.margin, "margin")
        if margin < 0.0:
            raise StatisticalContractError("non-inferiority margin must be non-negative")
        boundary = _finite(self.null_boundary, "null_boundary")
        if not math.isclose(boundary, -margin, rel_tol=0.0, abs_tol=1e-15):
            raise StatisticalContractError("non-inferiority null boundary must equal -margin")
        lower = _finite(self.one_sided_lower_bound, "one_sided_lower_bound")
        level = _confidence(self.confidence_level)
        p_value = _finite(self.raw_p_value, "raw_p_value")
        if not 0.0 <= p_value <= 1.0:
            raise StatisticalContractError("raw_p_value must lie in [0, 1]")
        expected_pass = lower >= boundary
        if type(self.passed) is not bool or self.passed != expected_pass:
            raise StatisticalContractError("passed does not match the one-sided bound")
        object.__setattr__(self, "observed_difference", observed)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "null_boundary", boundary)
        object.__setattr__(self, "one_sided_lower_bound", lower)
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(self, "raw_p_value", p_value)
        object.__setattr__(
            self, "bootstrap_resamples", _positive_int(self.bootstrap_resamples, "bootstrap_resamples")
        )
        object.__setattr__(self, "bootstrap_seed", _seed(self.bootstrap_seed))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def evaluate_noninferiority(
    bootstrap: HierarchicalBootstrapResult,
    *,
    margin: float,
) -> NonInferiorityResult:
    """Evaluate the one-sided condition ``LCB(method - comparator) >= -margin``."""

    if not isinstance(bootstrap, HierarchicalBootstrapResult):
        raise StatisticalContractError("bootstrap has the wrong result type")
    parsed_margin = _finite(margin, "margin")
    if parsed_margin < 0.0:
        raise StatisticalContractError("non-inferiority margin must be non-negative")
    boundary = -parsed_margin
    lower = bootstrap.one_sided_lower
    p_value = centered_one_sided_p_value(
        bootstrap.observed,
        bootstrap.replicates,
        null_boundary=boundary,
    )
    return NonInferiorityResult(
        observed_difference=bootstrap.observed,
        margin=parsed_margin,
        null_boundary=boundary,
        one_sided_lower_bound=lower,
        confidence_level=bootstrap.confidence_level,
        raw_p_value=p_value,
        passed=lower >= boundary,
        bootstrap_resamples=int(bootstrap.replicates.size),
        bootstrap_seed=bootstrap.seed,
    )


__all__ = [
    "ConfidenceInterval",
    "HierarchicalBootstrapResult",
    "HolmResult",
    "NonInferiorityResult",
    "SimultaneousMaxTInterval",
    "SimultaneousMaxTResult",
    "StatisticalContractError",
    "bootstrap_max_t_intervals",
    "centered_one_sided_p_value",
    "derive_bootstrap_seed",
    "evaluate_noninferiority",
    "hierarchical_bootstrap",
    "hierarchical_paired_difference_bootstrap",
    "holm_bonferroni",
    "one_sided_lower_bound",
    "percentile_interval",
]
