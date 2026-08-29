"""Source-fitted Gaussian task belief and utility decision for ``BPR_FP``.

This is the fixed-probe adapter from the v0.4a plan: reward-free episode
summaries form a diagonal-Gaussian task observation model, while the decision
layer uses a separately supplied source-only task-policy utility matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json
from .protocol import RewardFreeProbe, canonical_tie_token, stable_argmax


BPR_MODEL_SCHEMA = "policy-learnware.v04a-bpr-gaussian-model.v1"
SUMMARY_DEFINITION = ("mean_delta", "std_delta", "mean_action", "std_action")
DEFAULT_LAMBDA_GRID = (0.0, 0.5, 1.0)
DEFAULT_VARIANCE_FLOOR_GRID = (1e-6, 1e-4, 1e-2)
DEFAULT_TEMPERATURE_GRID = (0.25, 0.5, 1.0, 2.0)
_LOG_2PI = math.log(2.0 * math.pi)
_MAX_SAFE_RESIDUAL = 1.0e150


class BPRModelError(ValueError):
    """Source data or BPR scoring violates the frozen model contract."""


def _nonempty(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BPRModelError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where)
    if len(result) != 64 or result != result.lower():
        raise BPRModelError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise BPRModelError(f"{where} must be a lowercase SHA-256 digest") from error
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise BPRModelError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BPRModelError(f"{where} must be finite")
    return result


def _positive(value: Any, where: str) -> float:
    result = _finite(value, where)
    if result <= 0.0:
        raise BPRModelError(f"{where} must be positive")
    return result


def _readonly_matrix(
    value: Any,
    where: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[0] <= 0:
        raise BPRModelError(f"{where} must be a non-empty numeric matrix")
    result = np.ascontiguousarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise BPRModelError(f"{where} must be finite")
    if rows is not None and result.shape[0] != rows:
        raise BPRModelError(f"{where} has the wrong row count")
    if columns is not None and result.shape[1] != columns:
        raise BPRModelError(f"{where} has the wrong feature count")
    result.setflags(write=False)
    return result


def _readonly_vector(
    value: Any,
    where: str,
    *,
    length: int | None = None,
    positive: bool = False,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size <= 0:
        raise BPRModelError(f"{where} must be a non-empty numeric vector")
    result = np.ascontiguousarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise BPRModelError(f"{where} must be finite")
    if length is not None and result.shape != (length,):
        raise BPRModelError(f"{where} has the wrong length")
    if positive and np.any(result <= 0.0):
        raise BPRModelError(f"{where} must be strictly positive")
    result.setflags(write=False)
    return result


def _summary_matrix(
    value: Any, where: str, *, dimension: int | None = None
) -> np.ndarray:
    result = _readonly_matrix(value, where, columns=dimension)
    if result.shape[1] <= 0:
        raise BPRModelError(f"{where} must have at least one feature")
    return result


def _logsumexp(value: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    values = np.asarray(value, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    if axis is None:
        return float(result.squeeze())
    return np.squeeze(result, axis=axis)


def summarize_episode(
    observation: Any,
    action: Any,
    next_observation: Any,
) -> np.ndarray:
    """Return ``[mean/std delta, mean/std action]`` for one episode view."""

    observations = _readonly_matrix(observation, "observation")
    actions = _readonly_matrix(action, "action")
    next_observations = _readonly_matrix(next_observation, "next_observation")
    if observations.shape != next_observations.shape:
        raise BPRModelError("observation and next_observation must have the same shape")
    if actions.shape[0] != observations.shape[0]:
        raise BPRModelError("observation and action row counts disagree")
    delta = next_observations - observations
    summary = np.concatenate(
        (
            np.mean(delta, axis=0),
            np.std(delta, axis=0, ddof=0),
            np.mean(actions, axis=0),
            np.std(actions, axis=0, ddof=0),
        )
    ).astype(np.float64, copy=False)
    if not np.all(np.isfinite(summary)):
        raise BPRModelError("episode summary is non-finite")
    summary.setflags(write=False)
    return summary


def summarize_probe(probe: RewardFreeProbe) -> np.ndarray:
    """Build one summary row per frozen probe episode."""

    if not isinstance(probe, RewardFreeProbe):
        raise BPRModelError("probe must be a RewardFreeProbe")
    summaries = np.stack(
        [
            summarize_episode(
                probe.observation[probe.episode_slice(index)],
                probe.action[probe.episode_slice(index)],
                probe.next_observation[probe.episode_slice(index)],
            )
            for index in range(probe.episode_count)
        ],
        axis=0,
    )
    summaries.setflags(write=False)
    return summaries


def _canonical_ids(values: Sequence[Any], where: str) -> tuple[str, ...]:
    result = tuple(_nonempty(item, f"{where} item") for item in values)
    if not result or len(set(result)) != len(result):
        raise BPRModelError(f"{where} must contain unique IDs")
    return result


def _normalize_prior(
    prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None,
    type_ids: tuple[str, ...],
) -> np.ndarray:
    if prior is None:
        result = np.full(len(type_ids), 1.0 / len(type_ids), dtype=np.float64)
    elif isinstance(prior, Mapping):
        if set(prior) != set(type_ids):
            raise BPRModelError("prior keys must exactly equal source type IDs")
        result = np.asarray([prior[type_id] for type_id in type_ids], dtype=np.float64)
    else:
        result = np.asarray(prior, dtype=np.float64)
    result = _readonly_vector(
        result, "prior", length=len(type_ids), positive=True
    ).copy()
    total = float(np.sum(result))
    if not math.isfinite(total) or total <= 0.0:
        raise BPRModelError("prior must have positive finite mass")
    result /= total
    result.setflags(write=False)
    return result


def _utility_array(
    utility: Mapping[str, Mapping[str, Any]] | Any,
    type_ids: tuple[str, ...],
    candidate_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(utility, Mapping):
        if set(utility) != set(type_ids):
            raise BPRModelError("utility type keys must exactly equal source type IDs")
        first = utility[type_ids[0]]
        if not isinstance(first, Mapping):
            raise BPRModelError("utility rows must map candidate IDs to source utility")
        inferred = tuple(sorted(_nonempty(item, "candidate ID") for item in first))
        candidates = (
            inferred
            if candidate_ids is None
            else _canonical_ids(candidate_ids, "candidate_ids")
        )
        if set(candidates) != set(inferred):
            raise BPRModelError("candidate_ids disagree with utility mapping columns")
        for type_id in type_ids:
            row = utility[type_id]
            if not isinstance(row, Mapping) or set(row) != set(candidates):
                raise BPRModelError(
                    "every utility row must contain the same candidate IDs"
                )
        matrix = np.asarray(
            [
                [
                    _finite(utility[type_id][candidate], "source utility")
                    for candidate in candidates
                ]
                for type_id in type_ids
            ],
            dtype=np.float64,
        )
    else:
        if candidate_ids is None:
            raise BPRModelError(
                "candidate_ids are required for an array utility matrix"
            )
        candidates = _canonical_ids(candidate_ids, "candidate_ids")
        matrix = np.asarray(utility)
    result = _readonly_matrix(
        matrix,
        "utility_matrix",
        rows=len(type_ids),
        columns=len(candidates),
    )
    return candidates, result


def _grid(
    values: Sequence[Any],
    where: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
    positive: bool = False,
) -> tuple[float, ...]:
    normalized = tuple(sorted({_finite(value, f"{where} item") for value in values}))
    if not normalized:
        raise BPRModelError(f"{where} cannot be empty")
    if positive and any(item <= 0.0 for item in normalized):
        raise BPRModelError(f"{where} items must be positive")
    if lower is not None and any(item < lower for item in normalized):
        raise BPRModelError(f"{where} items must be >= {lower}")
    if upper is not None and any(item > upper for item in normalized):
        raise BPRModelError(f"{where} items must be <= {upper}")
    return normalized


def _gaussian_log_likelihood_rows(
    normalized_summaries: np.ndarray,
    type_means: np.ndarray,
    type_variances: np.ndarray,
) -> np.ndarray:
    # Shape: [episode, type, feature].
    difference = normalized_summaries[:, None, :] - type_means[None, :, :]
    safe_difference = np.clip(difference, -_MAX_SAFE_RESIDUAL, _MAX_SAFE_RESIDUAL)
    max_term = np.finfo(np.float64).max / (4.0 * max(1, difference.shape[2]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized_square = np.square(safe_difference) / type_variances[None, :, :]
    standardized_square = np.minimum(
        np.nan_to_num(
            standardized_square,
            nan=max_term,
            posinf=max_term,
            neginf=max_term,
        ),
        max_term,
    )
    return -0.5 * np.sum(
        _LOG_2PI + np.log(type_variances)[None, :, :] + standardized_square,
        axis=2,
    )


def _validation_posterior_nll(
    validation_by_type: Mapping[str, np.ndarray],
    type_ids: tuple[str, ...],
    pooled_mean: np.ndarray,
    pooled_scale: np.ndarray,
    type_means: np.ndarray,
    type_variances: np.ndarray,
    prior: np.ndarray,
    temperature: float,
) -> float:
    """Type-balanced correct-type posterior NLL, one validation episode at a time."""

    log_prior = np.log(prior)
    losses: list[float] = []
    for true_index, type_id in enumerate(type_ids):
        normalized = (validation_by_type[type_id] - pooled_mean) / pooled_scale
        row_likelihood = _gaussian_log_likelihood_rows(
            normalized, type_means, type_variances
        )
        logits = log_prior[None, :] + temperature * row_likelihood
        log_normalizer = np.asarray(_logsumexp(logits, axis=1), dtype=np.float64)
        true_log_posterior = logits[:, true_index] - log_normalizer
        losses.append(float(-np.mean(true_log_posterior)))
    return float(np.mean(losses))


@dataclass(frozen=True)
class BPRGaussianModel:
    """Frozen source-only diagonal-Gaussian BPR observation/utility model."""

    type_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    pooled_mean: np.ndarray
    pooled_scale: np.ndarray
    type_means: np.ndarray
    type_variances: np.ndarray
    prior: np.ndarray
    utility_matrix: np.ndarray
    shrinkage: float
    variance_floor: float
    temperature: float
    validation_nll: float
    config_digest: str
    protocol_id: str
    schema: str = BPR_MODEL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BPR_MODEL_SCHEMA:
            raise BPRModelError("unsupported BPR model schema")
        type_ids = _canonical_ids(self.type_ids, "type_ids")
        candidate_ids = _canonical_ids(self.candidate_ids, "candidate_ids")
        object.__setattr__(self, "type_ids", type_ids)
        object.__setattr__(self, "candidate_ids", candidate_ids)

        pooled_mean = _readonly_vector(self.pooled_mean, "pooled_mean")
        feature_count = int(pooled_mean.size)
        pooled_scale = _readonly_vector(
            self.pooled_scale,
            "pooled_scale",
            length=feature_count,
            positive=True,
        )
        type_means = _readonly_matrix(
            self.type_means,
            "type_means",
            rows=len(type_ids),
            columns=feature_count,
        )
        type_variances = _readonly_matrix(
            self.type_variances,
            "type_variances",
            rows=len(type_ids),
            columns=feature_count,
        )
        if np.any(type_variances <= 0.0):
            raise BPRModelError("type_variances must be strictly positive")
        prior = _readonly_vector(
            self.prior, "prior", length=len(type_ids), positive=True
        )
        if not np.isclose(float(np.sum(prior)), 1.0, rtol=0.0, atol=1e-12):
            raise BPRModelError("prior must sum to one")
        utility_matrix = _readonly_matrix(
            self.utility_matrix,
            "utility_matrix",
            rows=len(type_ids),
            columns=len(candidate_ids),
        )

        shrinkage = _finite(self.shrinkage, "shrinkage")
        if not 0.0 <= shrinkage <= 1.0:
            raise BPRModelError("shrinkage must be in [0, 1]")
        variance_floor = _positive(self.variance_floor, "variance_floor")
        temperature = _positive(self.temperature, "temperature")
        validation_nll = _finite(self.validation_nll, "validation_nll")

        object.__setattr__(self, "pooled_mean", pooled_mean)
        object.__setattr__(self, "pooled_scale", pooled_scale)
        object.__setattr__(self, "type_means", type_means)
        object.__setattr__(self, "type_variances", type_variances)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "utility_matrix", utility_matrix)
        object.__setattr__(self, "shrinkage", shrinkage)
        object.__setattr__(self, "variance_floor", variance_floor)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "validation_nll", validation_nll)
        object.__setattr__(
            self, "config_digest", _digest(self.config_digest, "config_digest")
        )
        object.__setattr__(
            self, "protocol_id", _nonempty(self.protocol_id, "protocol_id")
        )

    @property
    def feature_count(self) -> int:
        return int(self.pooled_mean.size)

    @property
    def model_digest(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def tie_token(self) -> str:
        return canonical_tie_token(self.config_digest)

    @classmethod
    def fit(
        cls,
        train_by_type: Mapping[str, Any],
        validation_by_type: Mapping[str, Any],
        utility_by_type: Mapping[str, Mapping[str, Any]] | Any,
        *,
        config_digest: str,
        protocol_id: str,
        candidate_ids: Sequence[str] | None = None,
        lambda_grid: Sequence[float] = DEFAULT_LAMBDA_GRID,
        variance_floor_grid: Sequence[float] = DEFAULT_VARIANCE_FLOOR_GRID,
        temperature_grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None = None,
        normalizer_floor: float = 1e-8,
    ) -> "BPRGaussianModel":
        """Fit from source-train and tune only on source-validation summaries."""

        if not isinstance(train_by_type, Mapping) or not train_by_type:
            raise BPRModelError("train_by_type must be a non-empty mapping")
        if not isinstance(validation_by_type, Mapping) or not validation_by_type:
            raise BPRModelError("validation_by_type must be a non-empty mapping")
        type_ids = tuple(
            sorted(_nonempty(item, "source type ID") for item in train_by_type)
        )
        if len(set(type_ids)) != len(type_ids) or set(validation_by_type) != set(
            type_ids
        ):
            raise BPRModelError(
                "source train and validation must contain the same unique type IDs"
            )

        train: dict[str, np.ndarray] = {}
        validation: dict[str, np.ndarray] = {}
        dimension: int | None = None
        for type_id in type_ids:
            train_matrix = _summary_matrix(
                train_by_type[type_id],
                f"train summaries for {type_id}",
                dimension=dimension,
            )
            if dimension is None:
                dimension = int(train_matrix.shape[1])
            validation_matrix = _summary_matrix(
                validation_by_type[type_id],
                f"validation summaries for {type_id}",
                dimension=dimension,
            )
            train[type_id] = train_matrix
            validation[type_id] = validation_matrix
        assert dimension is not None

        candidates, utility_matrix = _utility_array(
            utility_by_type, type_ids, candidate_ids
        )
        normalized_prior = _normalize_prior(prior, type_ids)
        normalizer_floor_value = _positive(normalizer_floor, "normalizer_floor")
        pooled_train = np.concatenate([train[type_id] for type_id in type_ids], axis=0)
        pooled_mean = np.mean(pooled_train, axis=0)
        pooled_scale = np.maximum(
            np.std(pooled_train, axis=0, ddof=0), normalizer_floor_value
        )
        normalized_train = {
            type_id: (train[type_id] - pooled_mean) / pooled_scale
            for type_id in type_ids
        }
        type_means = np.stack(
            [np.mean(normalized_train[type_id], axis=0) for type_id in type_ids], axis=0
        )
        type_raw_variances = np.stack(
            [np.var(normalized_train[type_id], axis=0, ddof=0) for type_id in type_ids],
            axis=0,
        )
        pooled_variance = np.mean(
            np.concatenate(
                [
                    np.square(normalized_train[type_id] - type_means[index])
                    for index, type_id in enumerate(type_ids)
                ],
                axis=0,
            ),
            axis=0,
        )

        lambdas = _grid(lambda_grid, "lambda_grid", lower=0.0, upper=1.0)
        floors = _grid(variance_floor_grid, "variance_floor_grid", positive=True)
        temperatures = _grid(temperature_grid, "temperature_grid", positive=True)
        best: tuple[float, float, float, float, np.ndarray] | None = None
        # Grids are sorted and exact NLL ties therefore use the lexicographically
        # smallest registered hyperparameter triple, independent of input order.
        for shrinkage in lambdas:
            shrunk = (
                shrinkage * type_raw_variances
                + (1.0 - shrinkage) * pooled_variance[None, :]
            )
            for variance_floor in floors:
                variances = np.maximum(shrunk, variance_floor)
                for temperature in temperatures:
                    validation_nll = _validation_posterior_nll(
                        validation,
                        type_ids,
                        pooled_mean,
                        pooled_scale,
                        type_means,
                        variances,
                        normalized_prior,
                        temperature,
                    )
                    candidate = (
                        validation_nll,
                        shrinkage,
                        variance_floor,
                        temperature,
                        variances,
                    )
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
        assert best is not None
        validation_nll, shrinkage, variance_floor, temperature, type_variances = best
        return cls(
            type_ids=type_ids,
            candidate_ids=candidates,
            pooled_mean=pooled_mean,
            pooled_scale=pooled_scale,
            type_means=type_means,
            type_variances=type_variances,
            prior=normalized_prior,
            utility_matrix=utility_matrix,
            shrinkage=shrinkage,
            variance_floor=variance_floor,
            temperature=temperature,
            validation_nll=validation_nll,
            config_digest=config_digest,
            protocol_id=protocol_id,
        )

    def normalize(self, summaries: Any) -> np.ndarray:
        matrix = _summary_matrix(summaries, "summaries", dimension=self.feature_count)
        normalized = (matrix - self.pooled_mean) / self.pooled_scale
        normalized.setflags(write=False)
        return normalized

    def log_likelihood_rows(self, summaries: Any) -> np.ndarray:
        normalized = self.normalize(summaries)
        result = _gaussian_log_likelihood_rows(
            normalized, self.type_means, self.type_variances
        )
        result.setflags(write=False)
        return result

    def log_likelihood(self, summaries: Any) -> np.ndarray:
        """Return accumulated, untempered log likelihood for every source type."""

        with np.errstate(over="ignore", invalid="ignore"):
            result = np.sum(self.log_likelihood_rows(summaries), axis=0)
        limit = np.finfo(np.float64).max / 4.0
        result = np.clip(
            np.nan_to_num(result, nan=-limit, posinf=limit, neginf=-limit),
            -limit,
            limit,
        )
        result.setflags(write=False)
        return result

    def _prior_array(
        self,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None,
    ) -> np.ndarray:
        return self.prior if prior is None else _normalize_prior(prior, self.type_ids)

    def posterior(
        self,
        summaries: Any,
        *,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None = None,
    ) -> np.ndarray:
        prior_array = self._prior_array(prior)
        limit = np.finfo(np.float64).max / 4.0
        with np.errstate(over="ignore", invalid="ignore"):
            logits = np.log(prior_array) + self.temperature * self.log_likelihood(
                summaries
            )
        logits = np.clip(
            np.nan_to_num(logits, nan=-limit, posinf=limit, neginf=-limit),
            -limit,
            limit,
        )
        shifted = logits - np.max(logits)
        weights = np.exp(shifted)
        result = weights / np.sum(weights)
        result.setflags(write=False)
        return result

    def posterior_dict(
        self,
        summaries: Any,
        *,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None = None,
    ) -> dict[str, float]:
        posterior = self.posterior(summaries, prior=prior)
        return {
            type_id: float(posterior[index])
            for index, type_id in enumerate(self.type_ids)
        }

    def map_type(
        self,
        summaries: Any,
        *,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None = None,
    ) -> str:
        posterior = self.posterior(summaries, prior=prior)
        return stable_argmax(
            {
                type_id: float(posterior[index])
                for index, type_id in enumerate(self.type_ids)
            },
            self.config_digest,
        )

    def expected_utility(
        self,
        posterior: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    ) -> np.ndarray:
        if isinstance(posterior, Mapping):
            if set(posterior) != set(self.type_ids):
                raise BPRModelError("posterior keys must exactly equal source type IDs")
            values = np.asarray(
                [posterior[type_id] for type_id in self.type_ids], dtype=np.float64
            )
        else:
            values = np.asarray(posterior)
        values = _readonly_vector(values, "posterior", length=len(self.type_ids))
        if np.any(values < 0.0) or not np.isclose(
            float(np.sum(values)), 1.0, rtol=0.0, atol=1e-10
        ):
            raise BPRModelError("posterior must be non-negative and sum to one")
        result = values @ self.utility_matrix
        result.setflags(write=False)
        return result

    def utility_scores(
        self,
        summaries: Any,
        *,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None = None,
    ) -> dict[str, float]:
        expected = self.expected_utility(self.posterior(summaries, prior=prior))
        return {
            candidate_id: float(expected[index])
            for index, candidate_id in enumerate(self.candidate_ids)
        }

    def select_candidate(
        self,
        summaries: Any,
        *,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None = None,
    ) -> str:
        """Choose posterior-weighted source utility with the canonical exact tie."""

        return stable_argmax(
            self.utility_scores(summaries, prior=prior), self.config_digest
        )

    def target_predictive_nll(
        self,
        summaries: Any,
        *,
        prior: Mapping[str, Any] | Sequence[Any] | np.ndarray | None = None,
    ) -> float:
        """Unlabelled source-mixture predictive NLL, normalized per episode."""

        matrix = _summary_matrix(summaries, "summaries", dimension=self.feature_count)
        prior_array = self._prior_array(prior)
        log_mixture_evidence = float(
            _logsumexp(np.log(prior_array) + self.log_likelihood(matrix))
        )
        return float(-log_mixture_evidence / matrix.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "type_ids": list(self.type_ids),
            "candidate_ids": list(self.candidate_ids),
            "pooled_mean": self.pooled_mean.tolist(),
            "pooled_scale": self.pooled_scale.tolist(),
            "type_means": self.type_means.tolist(),
            "type_variances": self.type_variances.tolist(),
            "prior": self.prior.tolist(),
            "utility_matrix": self.utility_matrix.tolist(),
            "lambda_shrinkage": self.shrinkage,
            "variance_floor": self.variance_floor,
            "temperature": self.temperature,
            "validation_correct_type_posterior_nll": self.validation_nll,
            "config_digest": self.config_digest,
            "tie_token": self.tie_token,
            "protocol_id": self.protocol_id,
            "summary_definition": list(SUMMARY_DEFINITION),
            "likelihood_family": "diagonal_gaussian",
            "normalizer_fit_role": "source_train_pooled",
            "hyperparameter_selection_role": "source_validation",
            "variance_shrinkage_target": "source_train_pooled_within_type_residual_variance",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BPRGaussianModel":
        required = {
            "schema",
            "type_ids",
            "candidate_ids",
            "pooled_mean",
            "pooled_scale",
            "type_means",
            "type_variances",
            "prior",
            "utility_matrix",
            "lambda_shrinkage",
            "variance_floor",
            "temperature",
            "validation_correct_type_posterior_nll",
            "config_digest",
            "tie_token",
            "protocol_id",
            "summary_definition",
            "likelihood_family",
            "normalizer_fit_role",
            "hyperparameter_selection_role",
            "variance_shrinkage_target",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            raise BPRModelError(
                f"BPR checkpoint keys mismatch; missing={missing}, extra={extra}"
            )
        if tuple(payload["summary_definition"]) != SUMMARY_DEFINITION:
            raise BPRModelError("serialized model has a different summary definition")
        if payload["likelihood_family"] != "diagonal_gaussian":
            raise BPRModelError("serialized model has a different likelihood family")
        if (
            payload["normalizer_fit_role"] != "source_train_pooled"
            or payload["hyperparameter_selection_role"] != "source_validation"
            or payload["variance_shrinkage_target"]
            != "source_train_pooled_within_type_residual_variance"
        ):
            raise BPRModelError("serialized model has different source-fit roles")
        result = cls(
            schema=payload["schema"],
            type_ids=tuple(payload["type_ids"]),
            candidate_ids=tuple(payload["candidate_ids"]),
            pooled_mean=payload["pooled_mean"],
            pooled_scale=payload["pooled_scale"],
            type_means=payload["type_means"],
            type_variances=payload["type_variances"],
            prior=payload["prior"],
            utility_matrix=payload["utility_matrix"],
            shrinkage=payload["lambda_shrinkage"],
            variance_floor=payload["variance_floor"],
            temperature=payload["temperature"],
            validation_nll=payload["validation_correct_type_posterior_nll"],
            config_digest=payload["config_digest"],
            protocol_id=payload["protocol_id"],
        )
        claimed_tie_token = payload["tie_token"]
        if claimed_tie_token != result.tie_token:
            raise BPRModelError("serialized tie token does not match config_digest")
        return result


__all__ = [
    "BPRGaussianModel",
    "BPRModelError",
    "BPR_MODEL_SCHEMA",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_TEMPERATURE_GRID",
    "DEFAULT_VARIANCE_FLOOR_GRID",
    "SUMMARY_DEFINITION",
    "summarize_episode",
    "summarize_probe",
]
