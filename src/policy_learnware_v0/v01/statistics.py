"""Deterministic statistical primitives for the v0.1 dynamics-shift study.

The functions in this module deliberately encode the experiment's resampling
contracts.  A candidate's shifted and nominal episodes are paired, while two
different candidates are always resampled independently.  The module has no
SciPy dependency so that the preregistered computations can be unit-tested in
the lightweight CPU environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


UNDEFINED_SPEARMAN_REASON = "undefined_nonfinite_or_constant_input"


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _confidence_level(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    return value


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be an integer")
    value = int(value)
    if value < 0 or value >= 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    return value


def derive_bootstrap_seed(*parts: object) -> int:
    """Derive a stable 64-bit seed from an explicit statistical namespace.

    Length-prefixing prevents ambiguous tuples such as ``("ab", "c")`` and
    ``("a", "bc")`` from sharing a seed.  Formal callers should include the
    measurement/full run id, task and statistic family in ``parts``.
    """

    if not parts:
        raise ValueError("at least one seed namespace part is required")
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

    def __post_init__(self) -> None:
        low = float(self.low)
        high = float(self.high)
        level = _confidence_level(self.confidence_level)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("confidence interval endpoints must be finite and ordered")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "confidence_level", level)

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "low": self.low,
            "high": self.high,
            "confidence_level": self.confidence_level,
        }


def percentile_interval(
    replicates: Sequence[float] | np.ndarray, *, confidence_level: float = 0.95
) -> ConfidenceInterval:
    values = _finite_vector(replicates, name="replicates")
    level = _confidence_level(confidence_level)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail], method="linear")
    return ConfidenceInterval(float(low), float(high), level)


def centered_bootstrap_p_value(
    observed: float, replicates: Sequence[float] | np.ndarray
) -> float:
    """Return the preregistered two-sided centered-bootstrap p-value."""

    observed = float(observed)
    if not math.isfinite(observed):
        raise ValueError("observed statistic must be finite")
    values = _finite_vector(replicates, name="replicates")
    exceedances = np.count_nonzero(
        np.abs(values - observed) >= abs(observed)
    )
    return float((1 + int(exceedances)) / (values.size + 1))


@dataclass(frozen=True)
class BootstrapResult:
    """An observed scalar and deterministic bootstrap distribution."""

    observed: float
    replicates: np.ndarray
    confidence_level: float
    resampling_contract: str
    seed: int

    def __post_init__(self) -> None:
        observed = float(self.observed)
        if not math.isfinite(observed):
            raise ValueError("observed statistic must be finite")
        values = _finite_vector(self.replicates, name="replicates").copy()
        values.setflags(write=False)
        if not self.resampling_contract:
            raise ValueError("resampling_contract cannot be empty")
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "replicates", values)
        object.__setattr__(self, "confidence_level", _confidence_level(self.confidence_level))
        object.__setattr__(self, "seed", _seed(self.seed))

    @property
    def interval(self) -> ConfidenceInterval:
        return percentile_interval(
            self.replicates, confidence_level=self.confidence_level
        )

    @property
    def centered_p_value(self) -> float:
        return centered_bootstrap_p_value(self.observed, self.replicates)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "ci": self.interval.to_dict(),
            "raw_p_value": self.centered_p_value,
            "bootstrap_resamples": int(self.replicates.size),
            "bootstrap_method": "contract_aware_percentile",
            "p_value_method": "centered_bootstrap",
            "resampling_contract": self.resampling_contract,
            "bootstrap_seed": self.seed,
        }


@dataclass(frozen=True)
class TransferBootstrapResult:
    """Paired signed transfer effect and derived absolute-gap interval."""

    delta: BootstrapResult

    @property
    def abs_gap(self) -> float:
        return abs(self.delta.observed)

    @property
    def abs_gap_interval(self) -> ConfidenceInterval:
        return percentile_interval(
            np.abs(self.delta.replicates),
            confidence_level=self.delta.confidence_level,
        )

    def to_summary_dict(self) -> dict[str, Any]:
        result = self.delta.to_summary_dict()
        result.update(
            {
                "delta_return": self.delta.observed,
                "abs_transfer_gap": self.abs_gap,
                "abs_transfer_gap_ci": self.abs_gap_interval.to_dict(),
            }
        )
        return result


def mean_bootstrap(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap for one candidate-variant mean return."""

    samples = _finite_vector(values, name="values")
    total = _positive_int(resamples, name="resamples")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, samples.size, size=(total, samples.size))
    return BootstrapResult(
        observed=float(samples.mean()),
        replicates=samples[indices].mean(axis=1),
        confidence_level=confidence_level,
        resampling_contract="episode_indices_within_candidate_variant",
        seed=seed,
    )


def paired_transfer_bootstrap(
    shifted_returns: Sequence[float] | np.ndarray,
    nominal_returns: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> TransferBootstrapResult:
    """Bootstrap ``mean(shifted - nominal)`` using paired episode indices."""

    shifted = _finite_vector(shifted_returns, name="shifted_returns")
    nominal = _finite_vector(nominal_returns, name="nominal_returns")
    if shifted.shape != nominal.shape:
        raise ValueError("paired shifted and nominal returns must have identical shape")
    count = shifted.size
    total = _positive_int(resamples, name="resamples")
    seed = _seed(seed)
    differences = shifted - nominal
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(total, count))
    replicates = differences[indices].mean(axis=1)
    return TransferBootstrapResult(
        BootstrapResult(
            observed=float(differences.mean()),
            replicates=replicates,
            confidence_level=confidence_level,
            resampling_contract="paired_episode_indices_within_candidate",
            seed=seed,
        )
    )


def independent_mean_difference_bootstrap(
    left_values: Sequence[float] | np.ndarray,
    right_values: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Bootstrap a mean difference with independent candidate resampling."""

    left = _finite_vector(left_values, name="left_values")
    right = _finite_vector(right_values, name="right_values")
    total = _positive_int(resamples, name="resamples")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    left_indices = rng.integers(0, left.size, size=(total, left.size))
    right_indices = rng.integers(0, right.size, size=(total, right.size))
    replicates = left[left_indices].mean(axis=1) - right[right_indices].mean(axis=1)
    return BootstrapResult(
        observed=float(left.mean() - right.mean()),
        replicates=replicates,
        confidence_level=confidence_level,
        resampling_contract="independent_episode_indices_across_candidates",
        seed=seed,
    )


def independent_sensitivity_difference_bootstrap(
    left_paired_differences: Sequence[float] | np.ndarray,
    right_paired_differences: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Bootstrap ``abs(mean(D_i)) - abs(mean(D_j))`` independently.

    Each input already represents episode-aligned shifted-minus-nominal
    differences for one candidate.  Independent index matrices ensure that a
    shared episode number is never mistaken for cross-candidate pairing.
    """

    left = _finite_vector(left_paired_differences, name="left_paired_differences")
    right = _finite_vector(right_paired_differences, name="right_paired_differences")
    total = _positive_int(resamples, name="resamples")
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    left_indices = rng.integers(0, left.size, size=(total, left.size))
    right_indices = rng.integers(0, right.size, size=(total, right.size))
    replicates = np.abs(left[left_indices].mean(axis=1)) - np.abs(
        right[right_indices].mean(axis=1)
    )
    return BootstrapResult(
        observed=float(abs(left.mean()) - abs(right.mean())),
        replicates=replicates,
        confidence_level=confidence_level,
        resampling_contract="independent_candidates_with_paired_differences_within_each",
        seed=seed,
    )


@dataclass(frozen=True)
class HolmResult:
    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    correction_order: int
    family_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "raw_p_value": self.raw_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "correction_order": self.correction_order,
            "family_size": self.family_size,
        }


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, HolmResult]:
    """Adjust a complete, preregistered hypothesis family using Holm's method."""

    if not p_values:
        raise ValueError("Holm correction requires a non-empty hypothesis family")
    parsed: dict[str, float] = {}
    for raw_id, raw_p in p_values.items():
        hypothesis_id = str(raw_id)
        p_value = float(raw_p)
        if not hypothesis_id or hypothesis_id in parsed:
            raise ValueError("hypothesis ids must be unique and non-empty")
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError("p-values must be finite and in [0, 1]")
        parsed[hypothesis_id] = p_value

    ordered = sorted(parsed.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_adjusted = 0.0
    results: dict[str, HolmResult] = {}
    for index, (hypothesis_id, p_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - index) * p_value)
        running_adjusted = max(running_adjusted, candidate)
        results[hypothesis_id] = HolmResult(
            hypothesis_id=hypothesis_id,
            raw_p_value=p_value,
            adjusted_p_value=running_adjusted,
            correction_order=index + 1,
            family_size=family_size,
        )
    return results


def significant_after_holm(
    interval: ConfidenceInterval, adjusted_p_value: float, *, alpha: float = 0.05
) -> bool:
    adjusted = float(adjusted_p_value)
    alpha = float(alpha)
    if not math.isfinite(adjusted) or not 0.0 <= adjusted <= 1.0:
        raise ValueError("adjusted_p_value must be finite and in [0, 1]")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    return interval.excludes_zero and adjusted < alpha


@dataclass(frozen=True)
class CompetenceSet:
    candidate_ids: tuple[str, ...]
    threshold: float
    best_nominal_return: float
    alpha: float

    @property
    def sufficient(self) -> bool:
        return len(self.candidate_ids) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "threshold": self.threshold,
            "best_nominal_return": self.best_nominal_return,
            "alpha": self.alpha,
            "sufficient": self.sufficient,
        }


def empirical_competence_set(
    nominal_returns: Mapping[str, float], *, alpha: float = 0.8
) -> CompetenceSet:
    """Build the point-estimate competence set before inspecting shifts."""

    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("competence alpha must be finite and in (0, 1]")
    if not nominal_returns:
        raise ValueError("nominal_returns cannot be empty")
    parsed: dict[str, float] = {}
    for raw_id, raw_return in nominal_returns.items():
        candidate_id = str(raw_id)
        value = float(raw_return)
        if not candidate_id or candidate_id in parsed:
            raise ValueError("candidate ids must be unique and non-empty")
        if not math.isfinite(value):
            raise ValueError("nominal returns must be finite")
        parsed[candidate_id] = value
    best = max(parsed.values())
    threshold = alpha * best
    members = tuple(sorted(key for key, value in parsed.items() if value >= threshold))
    return CompetenceSet(members, threshold, best, alpha)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        # Rank convention is 1..n.  All values in a tie receive the average.
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


@dataclass(frozen=True)
class SpearmanResult:
    rho: float | None
    reason: str | None
    sample_size: int

    @property
    def defined(self) -> bool:
        return self.rho is not None

    def to_dict(self) -> dict[str, Any]:
        return {"rho": self.rho, "reason": self.reason, "sample_size": self.sample_size}


def spearman_correlation(
    left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray
) -> SpearmanResult:
    """Compute tie-aware Spearman correlation without serializing NaN."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1:
        raise ValueError("Spearman inputs must be one-dimensional")
    if left_array.size != right_array.size or left_array.size < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        return SpearmanResult(None, UNDEFINED_SPEARMAN_REASON, int(left_array.size))
    left_ranks = _average_ranks(left_array)
    right_ranks = _average_ranks(right_array)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if denominator == 0.0:
        return SpearmanResult(None, UNDEFINED_SPEARMAN_REASON, int(left_array.size))
    rho = float(np.dot(left_centered, right_centered) / denominator)
    # Protect artifacts from a one-ulp excursion outside the mathematical range.
    rho = min(1.0, max(-1.0, rho))
    return SpearmanResult(rho, None, int(left_array.size))


@dataclass(frozen=True)
class Top1BootstrapResult:
    empirical_winner: str
    probabilities: Mapping[str, float]
    resamples: int
    seed: int

    def __post_init__(self) -> None:
        winner = str(self.empirical_winner)
        if not winner:
            raise ValueError("empirical_winner cannot be empty")
        parsed: dict[str, float] = {}
        for raw_id, raw_probability in self.probabilities.items():
            candidate_id = str(raw_id)
            probability = float(raw_probability)
            if not candidate_id or candidate_id in parsed:
                raise ValueError("top-1 candidate ids must be unique and non-empty")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("top-1 probabilities must be finite and in [0, 1]")
            parsed[candidate_id] = probability
        if winner not in parsed:
            raise ValueError("empirical_winner must appear in probabilities")
        if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("top-1 probabilities must sum to one")
        object.__setattr__(self, "empirical_winner", winner)
        object.__setattr__(self, "probabilities", dict(sorted(parsed.items())))
        object.__setattr__(self, "resamples", _positive_int(self.resamples, name="resamples"))
        object.__setattr__(self, "seed", _seed(self.seed))

    def probability(self, candidate_id: str) -> float:
        if candidate_id not in self.probabilities:
            raise KeyError(candidate_id)
        return float(self.probabilities[candidate_id])

    def to_dict(self) -> dict[str, Any]:
        return {
            "empirical_winner": self.empirical_winner,
            "top1_probabilities": dict(self.probabilities),
            "bootstrap_resamples": self.resamples,
            "bootstrap_seed": self.seed,
            "resampling_contract": "independent_episode_indices_across_candidates",
            "tie_break": "lexical_candidate_id",
        }


def top1_bootstrap_probabilities(
    returns_by_candidate: Mapping[str, Sequence[float] | np.ndarray],
    *,
    resamples: int,
    seed: int,
) -> Top1BootstrapResult:
    """Estimate winner probabilities under independent candidate bootstrap."""

    if not returns_by_candidate:
        raise ValueError("returns_by_candidate cannot be empty")
    arrays: dict[str, np.ndarray] = {}
    for raw_id, raw_values in returns_by_candidate.items():
        candidate = str(raw_id)
        if not candidate or candidate in arrays:
            raise ValueError("candidate ids must be unique and non-empty")
        arrays[candidate] = _finite_vector(raw_values, name=f"returns[{candidate}]")
    candidates = tuple(sorted(arrays))
    total = _positive_int(resamples, name="resamples")
    seed = _seed(seed)
    observed_means = {candidate: float(arrays[candidate].mean()) for candidate in candidates}
    empirical_winner = min(candidates, key=lambda key: (-observed_means[key], key))
    rng = np.random.default_rng(seed)
    replicate_means = np.empty((len(candidates), total), dtype=np.float64)
    for candidate_index, candidate in enumerate(candidates):
        values = arrays[candidate]
        indices = rng.integers(0, values.size, size=(total, values.size))
        replicate_means[candidate_index] = values[indices].mean(axis=1)
    # np.argmax returns the first index; sorted candidate ids therefore provide
    # the declared lexical tie-break.
    winners = np.argmax(replicate_means, axis=0)
    probabilities = {
        candidate: float(np.count_nonzero(winners == index) / total)
        for index, candidate in enumerate(candidates)
    }
    return Top1BootstrapResult(empirical_winner, probabilities, total, seed)


@dataclass(frozen=True)
class NestedSpearmanResult:
    point: SpearmanResult
    interval: ConfidenceInterval | None
    finite_bootstrap_count: int
    bootstrap_resamples: int
    finite_bootstrap_fraction: float
    reason: str | None
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho": self.point.rho,
            "point_reason": self.point.reason,
            "ci": None if self.interval is None else self.interval.to_dict(),
            "finite_bootstrap_count": self.finite_bootstrap_count,
            "bootstrap_resamples": self.bootstrap_resamples,
            "finite_bootstrap_fraction": self.finite_bootstrap_fraction,
            "reason": self.reason,
            "bootstrap_seed": self.seed,
            "resampling_contract": "nested_probe_banks_and_oracle_episodes_fixed_severity_grid",
        }


def nested_gate_c_spearman(
    probe_distances: Sequence[Sequence[float]] | np.ndarray,
    paired_transfer_differences: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
    minimum_finite_fraction: float = 0.95,
) -> NestedSpearmanResult:
    """Nested-bootstrap Gate C while keeping severity points fixed.

    ``probe_distances`` has shape ``(severity, bank)``.
    ``paired_transfer_differences`` has shape
    ``(severity, competent_candidate, episode)``.  One bank-index sample is
    shared across all severities; for each candidate, one episode-index sample
    is likewise shared across all severities, preserving the variant pairing.
    Different candidates receive independent index samples.
    """

    probe = np.asarray(probe_distances, dtype=np.float64)
    transfer = np.asarray(paired_transfer_differences, dtype=np.float64)
    if probe.ndim != 2 or transfer.ndim != 3:
        raise ValueError("probe and transfer arrays must have rank 2 and 3 respectively")
    if probe.shape[0] != transfer.shape[0] or probe.shape[0] < 2:
        raise ValueError("probe and transfer must share at least two severity points")
    if probe.shape[1] == 0 or transfer.shape[1] == 0 or transfer.shape[2] == 0:
        raise ValueError("nested bootstrap axes cannot be empty")
    if not np.all(np.isfinite(probe)) or not np.all(np.isfinite(transfer)):
        point = SpearmanResult(None, UNDEFINED_SPEARMAN_REASON, int(probe.shape[0]))
        return NestedSpearmanResult(
            point=point,
            interval=None,
            finite_bootstrap_count=0,
            bootstrap_resamples=_positive_int(resamples, name="resamples"),
            finite_bootstrap_fraction=0.0,
            reason=UNDEFINED_SPEARMAN_REASON,
            seed=_seed(seed),
        )
    minimum_finite_fraction = float(minimum_finite_fraction)
    if not math.isfinite(minimum_finite_fraction) or not 0.0 <= minimum_finite_fraction <= 1.0:
        raise ValueError("minimum_finite_fraction must be in [0, 1]")
    total = _positive_int(resamples, name="resamples")
    seed = _seed(seed)
    level = _confidence_level(confidence_level)

    point_probe = np.median(probe, axis=1)
    point_effect = np.median(np.abs(transfer.mean(axis=2)), axis=1)
    point = spearman_correlation(point_probe, point_effect)
    if not point.defined:
        return NestedSpearmanResult(
            point=point,
            interval=None,
            finite_bootstrap_count=0,
            bootstrap_resamples=total,
            finite_bootstrap_fraction=0.0,
            reason=point.reason,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    finite_rhos: list[float] = []
    severity_count, bank_count = probe.shape
    candidate_count, episode_count = transfer.shape[1:]
    for _ in range(total):
        bank_indices = rng.integers(0, bank_count, size=bank_count)
        probe_stat = np.median(probe[:, bank_indices], axis=1)
        candidate_effects = np.empty((severity_count, candidate_count), dtype=np.float64)
        for candidate_index in range(candidate_count):
            episode_indices = rng.integers(0, episode_count, size=episode_count)
            candidate_effects[:, candidate_index] = np.abs(
                transfer[:, candidate_index, :][:, episode_indices].mean(axis=1)
            )
        effect_stat = np.median(candidate_effects, axis=1)
        result = spearman_correlation(probe_stat, effect_stat)
        if result.rho is not None:
            finite_rhos.append(result.rho)

    finite_count = len(finite_rhos)
    finite_fraction = finite_count / total
    if finite_fraction < minimum_finite_fraction:
        interval = None
        reason = "insufficient_finite_bootstrap"
    else:
        interval = percentile_interval(finite_rhos, confidence_level=level)
        reason = None
    return NestedSpearmanResult(
        point=point,
        interval=interval,
        finite_bootstrap_count=finite_count,
        bootstrap_resamples=total,
        finite_bootstrap_fraction=finite_fraction,
        reason=reason,
        seed=seed,
    )


__all__ = [
    "BootstrapResult",
    "CompetenceSet",
    "ConfidenceInterval",
    "HolmResult",
    "NestedSpearmanResult",
    "SpearmanResult",
    "Top1BootstrapResult",
    "TransferBootstrapResult",
    "UNDEFINED_SPEARMAN_REASON",
    "centered_bootstrap_p_value",
    "derive_bootstrap_seed",
    "empirical_competence_set",
    "holm_bonferroni",
    "independent_mean_difference_bootstrap",
    "independent_sensitivity_difference_bootstrap",
    "nested_gate_c_spearman",
    "mean_bootstrap",
    "paired_transfer_bootstrap",
    "percentile_interval",
    "significant_after_holm",
    "spearman_correlation",
    "top1_bootstrap_probabilities",
]
