"""Source-only Efficient Bayesian Policy Reuse for the v0.4a fixed probe.

This module implements the observation and decision layer used by
``EBPR_FP``.  It deliberately has no dependency on the v0.4a runner or oracle:
the only target-side input is a reward-free sequence of ``(s, a, s')``
transitions collected by the candidate-independent fixed probe.

For each source type, a deterministic random-feature network models
``delta = s' - s`` conditional on ``(s, a)``.  The hidden tanh features are
fixed and the type-specific output head is fitted by ridge regression.  Input
and output normalization are pooled across *all source-training types* in one
task.  Diagonal residual variance is fitted from source-training residuals;
labelled source-validation episodes only choose its floor and the posterior
temperature.

``EBPR_FP`` selects the paired policy of the posterior-MAP source type.  The
separate ``EBPR_FP_BPR_U`` entry point is explicitly an inspired hybrid: it
uses the same EBPR posterior but chooses the policy with maximum BPR expected
utility.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EBPR_SCHEMA = "policy-learnware.v04a-ebpr-fixed-probe.v1"
EBPR_METHOD_ID = "EBPR_FP"
EBPR_HYBRID_METHOD_ID = "EBPR_FP_BPR_U"
_LOG_2PI = math.log(2.0 * math.pi)
_MAX_SAFE_RESIDUAL = 1.0e150

DEFAULT_VARIANCE_FLOOR_CANDIDATES = (
    1.0e-8,
    1.0e-6,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    5.0e-2,
    1.0e-1,
)
DEFAULT_TEMPERATURE_CANDIDATES = (
    1.0e-3,
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
)


class EBPRError(ValueError):
    """Raised when evidence or a frozen EBPR model violates its contract."""


def _readonly_float_array(value: Any, *, name: str, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise EBPRError(f"{name} must be a {ndim}-D array")
    if not np.all(np.isfinite(array)):
        raise EBPRError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_bool_array(value: Any, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.bool_, copy=True)
    if array.ndim != 1:
        raise EBPRError(f"{name} must be a 1-D mask")
    array.setflags(write=False)
    return array


def _positive(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise EBPRError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise EBPRError(f"{name} must be finite and positive")
    return result


def _nonempty_id(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EBPRError(f"{name} must be a non-empty canonical string")
    return value


def _logsumexp(values: np.ndarray) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise EBPRError("log-sum-exp input must be a non-empty finite vector")
    maximum = float(np.max(vector))
    return maximum + math.log(float(np.sum(np.exp(vector - maximum))))


def _safe_squared(value: np.ndarray) -> np.ndarray:
    """Square residuals without turning extreme finite evidence into NaN."""

    clipped = np.clip(value, -_MAX_SAFE_RESIDUAL, _MAX_SAFE_RESIDUAL)
    return clipped * clipped


def _stable_total(value: np.ndarray) -> float:
    total = float(np.sum(value, dtype=np.float64))
    if math.isfinite(total):
        return total
    # This can only be reached for an astronomically large finite evidence
    # bank.  Preserve ordering information as a finite, very poor likelihood.
    return -float(np.finfo(np.float64).max / 4.0)


@dataclass(frozen=True)
class TransitionEpisode:
    """One reward-free episode represented as aligned transition arrays.

    ``state[t]``, ``action[t]`` and ``next_state[t]`` form one transition.
    Rewards, task construction metadata and candidate returns are intentionally
    not fields of this type.
    """

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray

    def __post_init__(self) -> None:
        state = _readonly_float_array(self.state, name="state", ndim=2)
        action = _readonly_float_array(self.action, name="action", ndim=2)
        next_state = _readonly_float_array(self.next_state, name="next_state", ndim=2)
        if state.shape[0] == 0:
            raise EBPRError("an episode must contain at least one transition")
        if next_state.shape != state.shape:
            raise EBPRError("state and next_state must have identical shapes")
        if action.shape[0] != state.shape[0] or action.shape[1] == 0:
            raise EBPRError("action must align with state and have positive width")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "next_state", next_state)

    @classmethod
    def from_state_sequence(cls, states: Any, actions: Any) -> "TransitionEpisode":
        """Build an episode from ``T+1`` states and ``T`` actions."""

        state_sequence = np.asarray(states, dtype=np.float64)
        action_array = np.asarray(actions, dtype=np.float64)
        if state_sequence.ndim != 2 or state_sequence.shape[0] < 2:
            raise EBPRError("states must be a 2-D array with at least two rows")
        if (
            action_array.ndim != 2
            or action_array.shape[0] + 1 != state_sequence.shape[0]
        ):
            raise EBPRError("actions must be 2-D and contain one row per transition")
        return cls(state_sequence[:-1], action_array, state_sequence[1:])

    @classmethod
    def from_delta(cls, state: Any, action: Any, delta: Any) -> "TransitionEpisode":
        """Build an episode when the reward-free log already stores deltas."""

        state_array = np.asarray(state, dtype=np.float64)
        delta_array = np.asarray(delta, dtype=np.float64)
        if state_array.shape != delta_array.shape:
            raise EBPRError("state and delta must have identical shapes")
        return cls(state_array, action, state_array + delta_array)

    @property
    def transition_count(self) -> int:
        return int(self.state.shape[0])

    @property
    def delta(self) -> np.ndarray:
        return self.next_state - self.state

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate((self.state, self.action), axis=1), self.delta


@dataclass(frozen=True)
class EBPRSelection:
    """Auditable result of either the primary MAP or hybrid decision layer."""

    method_id: str
    selected_type_id: str | None
    selected_policy_id: str
    posterior: Mapping[str, float]
    log_likelihoods: Mapping[str, float]
    target_predictive_nll: float
    expected_utility: Mapping[str, float] | None = None

    @property
    def is_hybrid(self) -> bool:
        return self.method_id == EBPR_HYBRID_METHOD_ID

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method_id": self.method_id,
            "selected_type_id": self.selected_type_id,
            "selected_policy_id": self.selected_policy_id,
            "posterior": dict(self.posterior),
            "log_likelihoods": dict(self.log_likelihoods),
            "target_predictive_nll": float(self.target_predictive_nll),
            "decision_semantics": (
                "INSPIRED_HYBRID_EXPECTED_UTILITY"
                if self.is_hybrid
                else "POSTERIOR_MAP_PAIRED_POLICY"
            ),
        }
        if self.expected_utility is not None:
            payload["expected_utility"] = dict(self.expected_utility)
        return payload


@dataclass(frozen=True)
class EBPRFixedProbe:
    """Frozen source-only conditional models for one TASK_5 task family."""

    type_ids: tuple[str, ...]
    paired_policy_by_type: Mapping[str, str]
    x_mean: np.ndarray
    x_scale: np.ndarray
    x_valid_mask: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    y_valid_mask: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_head_by_type: Mapping[str, np.ndarray]
    residual_variance_by_type: Mapping[str, np.ndarray]
    variance_floor: float
    posterior_temperature: float
    ridge: float
    feature_seed: int
    tie_token: str
    calibration: Mapping[str, Any]
    schema: str = EBPR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EBPR_SCHEMA:
            raise EBPRError("unsupported EBPR model schema")
        type_ids = tuple(_nonempty_id(value, name="type_id") for value in self.type_ids)
        if not type_ids or len(type_ids) != len(set(type_ids)):
            raise EBPRError("type_ids must be non-empty and unique")
        if type_ids != tuple(sorted(type_ids)):
            raise EBPRError("type_ids must be in canonical sorted order")

        paired = {
            _nonempty_id(key, name="paired type_id"): _nonempty_id(
                value, name="paired policy_id"
            )
            for key, value in self.paired_policy_by_type.items()
        }
        if set(paired) != set(type_ids):
            raise EBPRError("paired policy mapping must cover every type exactly")
        if len(set(paired.values())) != len(paired):
            raise EBPRError("paired policy mapping must be one-to-one")

        x_mean = _readonly_float_array(self.x_mean, name="x_mean", ndim=1)
        x_scale = _readonly_float_array(self.x_scale, name="x_scale", ndim=1)
        x_mask = _readonly_bool_array(self.x_valid_mask, name="x_valid_mask")
        y_mean = _readonly_float_array(self.y_mean, name="y_mean", ndim=1)
        y_scale = _readonly_float_array(self.y_scale, name="y_scale", ndim=1)
        y_mask = _readonly_bool_array(self.y_valid_mask, name="y_valid_mask")
        if (
            x_mean.size == 0
            or x_scale.shape != x_mean.shape
            or x_mask.shape != x_mean.shape
        ):
            raise EBPRError("x normalizer arrays must be same-shape non-empty vectors")
        if (
            y_mean.size == 0
            or y_scale.shape != y_mean.shape
            or y_mask.shape != y_mean.shape
        ):
            raise EBPRError("y normalizer arrays must be same-shape non-empty vectors")
        if not np.any(x_mask) or not np.any(y_mask):
            raise EBPRError("at least one input and output dimension must be valid")
        if np.any(x_scale <= 0.0) or np.any(y_scale <= 0.0):
            raise EBPRError("normalizer scales must be positive")

        hidden_weight = _readonly_float_array(
            self.hidden_weight, name="hidden_weight", ndim=2
        )
        hidden_bias = _readonly_float_array(
            self.hidden_bias, name="hidden_bias", ndim=1
        )
        if hidden_weight.shape[0] != int(np.sum(x_mask)):
            raise EBPRError("hidden_weight input width disagrees with x_valid_mask")
        if hidden_weight.shape[1] == 0 or hidden_bias.shape != (
            hidden_weight.shape[1],
        ):
            raise EBPRError("hidden feature arrays have incompatible shapes")

        feature_width = 1 + int(np.sum(x_mask)) + hidden_weight.shape[1]
        output_width = int(np.sum(y_mask))
        if set(self.output_head_by_type) != set(type_ids):
            raise EBPRError("output heads must cover every type exactly")
        if set(self.residual_variance_by_type) != set(type_ids):
            raise EBPRError("residual variances must cover every type exactly")
        heads: dict[str, np.ndarray] = {}
        variances: dict[str, np.ndarray] = {}
        for type_id in type_ids:
            if type_id not in self.output_head_by_type:
                raise EBPRError(f"missing output head for {type_id!r}")
            if type_id not in self.residual_variance_by_type:
                raise EBPRError(f"missing residual variance for {type_id!r}")
            head = _readonly_float_array(
                self.output_head_by_type[type_id],
                name=f"output_head_by_type[{type_id!r}]",
                ndim=2,
            )
            variance = _readonly_float_array(
                self.residual_variance_by_type[type_id],
                name=f"residual_variance_by_type[{type_id!r}]",
                ndim=1,
            )
            if head.shape != (feature_width, output_width):
                raise EBPRError(f"output head for {type_id!r} has the wrong shape")
            if variance.shape != (output_width,) or np.any(variance <= 0.0):
                raise EBPRError(
                    f"residual variance for {type_id!r} must be positive and match output"
                )
            heads[type_id] = head
            variances[type_id] = variance

        if isinstance(self.feature_seed, (bool, np.bool_)):
            raise EBPRError("feature_seed must be an integer")
        feature_seed = int(self.feature_seed)
        if feature_seed != self.feature_seed or feature_seed < 0:
            raise EBPRError("feature_seed must be a non-negative integer")

        calibration = dict(self.calibration)
        # Enforce JSON-safe, finite calibration metadata at the model boundary.
        try:
            json.dumps(calibration, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise EBPRError("calibration metadata must be finite JSON") from error

        object.__setattr__(self, "type_ids", type_ids)
        object.__setattr__(self, "paired_policy_by_type", paired)
        object.__setattr__(self, "x_mean", x_mean)
        object.__setattr__(self, "x_scale", x_scale)
        object.__setattr__(self, "x_valid_mask", x_mask)
        object.__setattr__(self, "y_mean", y_mean)
        object.__setattr__(self, "y_scale", y_scale)
        object.__setattr__(self, "y_valid_mask", y_mask)
        object.__setattr__(self, "hidden_weight", hidden_weight)
        object.__setattr__(self, "hidden_bias", hidden_bias)
        object.__setattr__(self, "output_head_by_type", heads)
        object.__setattr__(self, "residual_variance_by_type", variances)
        object.__setattr__(
            self,
            "variance_floor",
            _positive(self.variance_floor, name="variance_floor"),
        )
        object.__setattr__(
            self,
            "posterior_temperature",
            _positive(self.posterior_temperature, name="posterior_temperature"),
        )
        object.__setattr__(self, "ridge", _positive(self.ridge, name="ridge"))
        object.__setattr__(self, "feature_seed", feature_seed)
        object.__setattr__(
            self, "tie_token", _nonempty_id(self.tie_token, name="tie_token")
        )
        object.__setattr__(self, "calibration", calibration)

    @classmethod
    def fit(
        cls,
        source_train: Mapping[str, Sequence[TransitionEpisode]],
        source_validation: Mapping[str, Sequence[TransitionEpisode]],
        paired_policy_by_type: Mapping[str, str],
        *,
        hidden_dim: int = 64,
        ridge: float = 1.0e-4,
        feature_seed: int = 0,
        variance_floor_candidates: Sequence[float] = DEFAULT_VARIANCE_FLOOR_CANDIDATES,
        temperature_candidates: Sequence[float] = DEFAULT_TEMPERATURE_CANDIDATES,
        tie_token: str = "v04a-bpr-tie-v1",
        dimension_scale_floor: float = 1.0e-10,
    ) -> "EBPRFixedProbe":
        """Fit and calibrate all source hypotheses without target evidence."""

        if isinstance(hidden_dim, (bool, np.bool_)) or int(hidden_dim) != hidden_dim:
            raise EBPRError("hidden_dim must be a positive integer")
        hidden_dim = int(hidden_dim)
        if hidden_dim <= 0:
            raise EBPRError("hidden_dim must be a positive integer")
        ridge = _positive(ridge, name="ridge")
        dimension_scale_floor = _positive(
            dimension_scale_floor, name="dimension_scale_floor"
        )
        if (
            isinstance(feature_seed, (bool, np.bool_))
            or int(feature_seed) != feature_seed
        ):
            raise EBPRError("feature_seed must be a non-negative integer")
        feature_seed = int(feature_seed)
        if feature_seed < 0:
            raise EBPRError("feature_seed must be a non-negative integer")

        type_ids = tuple(
            sorted(_nonempty_id(key, name="source type_id") for key in source_train)
        )
        if not type_ids or set(type_ids) != set(source_validation):
            raise EBPRError(
                "source_train and source_validation must cover identical types"
            )
        if len(type_ids) != len(set(type_ids)):
            raise EBPRError("source type IDs must be unique")
        if set(paired_policy_by_type) != set(type_ids):
            raise EBPRError("paired_policy_by_type must cover identical source types")

        train = {
            type_id: _episode_tuple(
                source_train[type_id], where=f"source_train[{type_id!r}]"
            )
            for type_id in type_ids
        }
        validation = {
            type_id: _episode_tuple(
                source_validation[type_id], where=f"source_validation[{type_id!r}]"
            )
            for type_id in type_ids
        }
        _require_shared_dimensions((*train.values(), *validation.values()))

        pooled_x_parts: list[np.ndarray] = []
        pooled_y_parts: list[np.ndarray] = []
        train_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for type_id in type_ids:
            x, y = _join_episodes(train[type_id])
            train_xy[type_id] = (x, y)
            pooled_x_parts.append(x)
            pooled_y_parts.append(y)
        pooled_x = np.concatenate(pooled_x_parts, axis=0)
        pooled_y = np.concatenate(pooled_y_parts, axis=0)

        x_mean = np.mean(pooled_x, axis=0)
        x_std = np.std(pooled_x, axis=0)
        x_valid_mask = x_std > dimension_scale_floor
        y_mean = np.mean(pooled_y, axis=0)
        y_std = np.std(pooled_y, axis=0)
        y_valid_mask = y_std > dimension_scale_floor
        if not np.any(x_valid_mask):
            raise EBPRError("all pooled source input dimensions are constant")
        if not np.any(y_valid_mask):
            raise EBPRError("all pooled source delta dimensions are constant")
        x_scale = np.where(x_valid_mask, x_std, 1.0)
        y_scale = np.where(y_valid_mask, y_std, 1.0)

        rng = np.random.default_rng(feature_seed)
        hidden_weight = rng.standard_normal((int(np.sum(x_valid_mask)), hidden_dim))
        hidden_weight /= math.sqrt(float(np.sum(x_valid_mask)))
        hidden_bias = rng.standard_normal(hidden_dim)

        heads: dict[str, np.ndarray] = {}
        for type_id, (x, y) in train_xy.items():
            features = _features(
                x,
                x_mean=x_mean,
                x_scale=x_scale,
                x_valid_mask=x_valid_mask,
                hidden_weight=hidden_weight,
                hidden_bias=hidden_bias,
            )
            normalized_y = (y[:, y_valid_mask] - y_mean[y_valid_mask]) / y_scale[
                y_valid_mask
            ]
            gram = features.T @ features
            regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge
            regularizer[0, 0] = 0.0  # never shrink the intercept
            rhs = features.T @ normalized_y
            try:
                heads[type_id] = np.linalg.solve(gram + regularizer, rhs)
            except np.linalg.LinAlgError:  # pragma: no cover - defensive fallback
                heads[type_id] = np.linalg.lstsq(gram + regularizer, rhs, rcond=None)[0]

        # Residual variance is a fitted model parameter, so it belongs solely
        # to source_train.  Source-validation may choose the floor applied to
        # this estimate, but must never be used to re-estimate the variance.
        raw_variances: dict[str, np.ndarray] = {}
        for type_id in type_ids:
            squared_residual_parts: list[np.ndarray] = []
            for episode in train[type_id]:
                x, y = episode.xy()
                prediction = _predict_normalized(
                    x,
                    head=heads[type_id],
                    x_mean=x_mean,
                    x_scale=x_scale,
                    x_valid_mask=x_valid_mask,
                    hidden_weight=hidden_weight,
                    hidden_bias=hidden_bias,
                )
                observed = (y[:, y_valid_mask] - y_mean[y_valid_mask]) / y_scale[
                    y_valid_mask
                ]
                squared_residual_parts.append(_safe_squared(observed - prediction))
            raw_variances[type_id] = np.mean(
                np.concatenate(squared_residual_parts, axis=0), axis=0
            )

        floors = _candidate_grid(
            variance_floor_candidates, name="variance_floor_candidates"
        )
        temperatures = _candidate_grid(
            temperature_candidates, name="temperature_candidates"
        )
        best_key: tuple[float, float, float, float] | None = None
        best_floor = floors[0]
        best_temperature = temperatures[0]
        best_variances: dict[str, np.ndarray] = {}
        best_classification_nll = math.inf
        best_predictive_nll = math.inf
        validation_episode_count = sum(len(value) for value in validation.values())

        for floor in floors:
            variances = {
                type_id: np.maximum(raw_variances[type_id], floor)
                for type_id in type_ids
            }
            examples: list[tuple[str, np.ndarray, int]] = []
            predictive_terms: list[float] = []
            for true_type in type_ids:
                for episode in validation[true_type]:
                    scores = np.asarray(
                        [
                            _episode_log_likelihood(
                                episode,
                                head=heads[hypothesis],
                                variance=variances[hypothesis],
                                x_mean=x_mean,
                                x_scale=x_scale,
                                x_valid_mask=x_valid_mask,
                                y_mean=y_mean,
                                y_scale=y_scale,
                                y_valid_mask=y_valid_mask,
                                hidden_weight=hidden_weight,
                                hidden_bias=hidden_bias,
                            )
                            for hypothesis in type_ids
                        ],
                        dtype=np.float64,
                    )
                    examples.append((true_type, scores, episode.transition_count))
                    true_index = type_ids.index(true_type)
                    predictive_terms.append(
                        -float(scores[true_index]) / episode.transition_count
                    )
            predictive_nll = float(np.mean(predictive_terms))
            uniform_log_prior = -math.log(len(type_ids))
            for temperature in temperatures:
                losses: list[float] = []
                for true_type, scores, _ in examples:
                    logits = uniform_log_prior + temperature * scores
                    log_normalizer = _logsumexp(logits)
                    losses.append(
                        log_normalizer - float(logits[type_ids.index(true_type)])
                    )
                classification_nll = float(np.mean(losses))
                key = (classification_nll, predictive_nll, floor, temperature)
                if best_key is None or key < best_key:
                    best_key = key
                    best_floor = floor
                    best_temperature = temperature
                    best_variances = variances
                    best_classification_nll = classification_nll
                    best_predictive_nll = predictive_nll

        calibration = {
            "residual_variance_fit_role": "source_train_residual_only",
            "hyperparameter_calibration_role": "source_validation_only",
            "train_residual_transition_count": int(
                sum(
                    episode.transition_count
                    for episodes in train.values()
                    for episode in episodes
                )
            ),
            "validation_episode_count": validation_episode_count,
            "validation_transition_count": int(
                sum(
                    episode.transition_count
                    for episodes in validation.values()
                    for episode in episodes
                )
            ),
            "classification_nll": best_classification_nll,
            "true_type_predictive_nll_per_transition_per_valid_dim": best_predictive_nll,
            "variance_floor_candidates": list(floors),
            "temperature_candidates": list(temperatures),
            "valid_input_dimension_count": int(np.sum(x_valid_mask)),
            "valid_output_dimension_count": int(np.sum(y_valid_mask)),
        }
        return cls(
            type_ids=type_ids,
            paired_policy_by_type=dict(paired_policy_by_type),
            x_mean=x_mean,
            x_scale=x_scale,
            x_valid_mask=x_valid_mask,
            y_mean=y_mean,
            y_scale=y_scale,
            y_valid_mask=y_valid_mask,
            hidden_weight=hidden_weight,
            hidden_bias=hidden_bias,
            output_head_by_type=heads,
            residual_variance_by_type=best_variances,
            variance_floor=best_floor,
            posterior_temperature=best_temperature,
            ridge=ridge,
            feature_seed=feature_seed,
            tie_token=tie_token,
            calibration=calibration,
        )

    @property
    def state_dim(self) -> int:
        return int(self.y_mean.size)

    @property
    def action_dim(self) -> int:
        return int(self.x_mean.size - self.y_mean.size)

    def _validate_episode_dimensions(
        self, episodes: tuple[TransitionEpisode, ...]
    ) -> None:
        for episode in episodes:
            if episode.state.shape[1] != self.state_dim:
                raise EBPRError("target state width disagrees with frozen model")
            if episode.action.shape[1] != self.action_dim:
                raise EBPRError("target action width disagrees with frozen model")

    def predict_delta(self, type_id: str, state: Any, action: Any) -> np.ndarray:
        """Predict conditional delta means in the original state units."""

        if type_id not in self.output_head_by_type:
            raise EBPRError(f"unknown source type {type_id!r}")
        state_array = np.asarray(state, dtype=np.float64)
        action_array = np.asarray(action, dtype=np.float64)
        if state_array.ndim != 2 or state_array.shape[1] != self.state_dim:
            raise EBPRError("state has the wrong shape")
        if action_array.ndim != 2 or action_array.shape != (
            state_array.shape[0],
            self.action_dim,
        ):
            raise EBPRError("action has the wrong shape")
        if not np.all(np.isfinite(state_array)) or not np.all(
            np.isfinite(action_array)
        ):
            raise EBPRError("prediction inputs must be finite")
        x = np.concatenate((state_array, action_array), axis=1)
        normalized = _predict_normalized(
            x,
            head=self.output_head_by_type[type_id],
            x_mean=self.x_mean,
            x_scale=self.x_scale,
            x_valid_mask=self.x_valid_mask,
            hidden_weight=self.hidden_weight,
            hidden_bias=self.hidden_bias,
        )
        prediction = np.broadcast_to(self.y_mean, (x.shape[0], self.state_dim)).copy()
        prediction[:, self.y_valid_mask] += normalized * self.y_scale[self.y_valid_mask]
        return prediction

    def log_likelihoods(
        self, episodes: Sequence[TransitionEpisode] | TransitionEpisode
    ) -> dict[str, float]:
        """Return per-valid-dimension log likelihood summed over transitions."""

        evidence = _episode_tuple(episodes, where="target episodes")
        self._validate_episode_dimensions(evidence)
        result: dict[str, float] = {}
        for type_id in self.type_ids:
            terms = np.asarray(
                [
                    _episode_log_likelihood(
                        episode,
                        head=self.output_head_by_type[type_id],
                        variance=self.residual_variance_by_type[type_id],
                        x_mean=self.x_mean,
                        x_scale=self.x_scale,
                        x_valid_mask=self.x_valid_mask,
                        y_mean=self.y_mean,
                        y_scale=self.y_scale,
                        y_valid_mask=self.y_valid_mask,
                        hidden_weight=self.hidden_weight,
                        hidden_bias=self.hidden_bias,
                    )
                    for episode in evidence
                ],
                dtype=np.float64,
            )
            result[type_id] = _stable_total(terms)
        return result

    def posterior(
        self,
        episodes: Sequence[TransitionEpisode] | TransitionEpisode,
        prior: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Compute the calibrated task belief in log space."""

        scores = self.log_likelihoods(episodes)
        return self._posterior_from_scores(scores, prior)

    def target_predictive_nll(
        self,
        episodes: Sequence[TransitionEpisode] | TransitionEpisode,
        prior: Mapping[str, float] | None = None,
    ) -> float:
        """Frozen source-mixture predictive NLL per visible transition.

        This uses the raw source-mixture likelihood, without the posterior
        temperature.  It is an unlabeled target fit diagnostic, not validation
        accuracy and not a target-side fitting objective.
        """

        evidence = _episode_tuple(episodes, where="target episodes")
        self._validate_episode_dimensions(evidence)
        scores = self.log_likelihoods(evidence)
        transition_count = sum(episode.transition_count for episode in evidence)
        return self._predictive_nll_from_scores(scores, transition_count, prior)

    def select_map(
        self,
        episodes: Sequence[TransitionEpisode] | TransitionEpisode,
        prior: Mapping[str, float] | None = None,
    ) -> EBPRSelection:
        """Primary F12-style MAP type to paired-policy decision."""

        evidence = _episode_tuple(episodes, where="target episodes")
        self._validate_episode_dimensions(evidence)
        scores = self.log_likelihoods(evidence)
        belief = self._posterior_from_scores(scores, prior)
        # Source hypotheses are private context IDs; the frozen tie protocol is
        # defined on their paired opaque policy IDs, just like every other
        # TASK_5 decision layer.
        selected_type = self._argmax_with_tie(
            belief, tie_identity_by_key=self.paired_policy_by_type
        )
        transition_count = sum(episode.transition_count for episode in evidence)
        return EBPRSelection(
            method_id=EBPR_METHOD_ID,
            selected_type_id=selected_type,
            selected_policy_id=self.paired_policy_by_type[selected_type],
            posterior=belief,
            log_likelihoods=scores,
            target_predictive_nll=self._predictive_nll_from_scores(
                scores, transition_count, prior
            ),
        )

    def select_hybrid(
        self,
        episodes: Sequence[TransitionEpisode] | TransitionEpisode,
        utility_by_type: Mapping[str, Mapping[str, float]],
        prior: Mapping[str, float] | None = None,
    ) -> EBPRSelection:
        """Inspired hybrid: EBPR posterior followed by BPR expected utility."""

        evidence = _episode_tuple(episodes, where="target episodes")
        self._validate_episode_dimensions(evidence)
        scores = self.log_likelihoods(evidence)
        belief = self._posterior_from_scores(scores, prior)
        if set(utility_by_type) != set(self.type_ids):
            raise EBPRError("hybrid utility rows must cover every source type")
        candidate_ids: tuple[str, ...] | None = None
        normalized_rows: dict[str, dict[str, float]] = {}
        for type_id in self.type_ids:
            row = {
                _nonempty_id(key, name="hybrid candidate policy_id"): float(value)
                for key, value in utility_by_type[type_id].items()
            }
            if not row or not all(math.isfinite(value) for value in row.values()):
                raise EBPRError("hybrid utility values must be non-empty and finite")
            row_candidates = tuple(sorted(row))
            if candidate_ids is None:
                candidate_ids = row_candidates
            elif candidate_ids != row_candidates:
                raise EBPRError(
                    "all hybrid utility rows must cover identical candidates"
                )
            normalized_rows[type_id] = row
        assert candidate_ids is not None
        expected_utility = {
            policy_id: float(
                sum(
                    belief[type_id] * normalized_rows[type_id][policy_id]
                    for type_id in self.type_ids
                )
            )
            for policy_id in candidate_ids
        }
        selected_policy = self._argmax_with_tie(expected_utility)
        transition_count = sum(episode.transition_count for episode in evidence)
        return EBPRSelection(
            method_id=EBPR_HYBRID_METHOD_ID,
            selected_type_id=None,
            selected_policy_id=selected_policy,
            posterior=belief,
            log_likelihoods=scores,
            target_predictive_nll=self._predictive_nll_from_scores(
                scores, transition_count, prior
            ),
            expected_utility=expected_utility,
        )

    def _posterior_from_scores(
        self,
        scores: Mapping[str, float],
        prior: Mapping[str, float] | None,
    ) -> dict[str, float]:
        prior_vector = self._prior_vector(prior)
        logits = np.log(prior_vector) + self.posterior_temperature * np.asarray(
            [scores[type_id] for type_id in self.type_ids], dtype=np.float64
        )
        log_normalizer = _logsumexp(logits)
        probabilities = np.exp(logits - log_normalizer)
        return {
            type_id: float(probabilities[index])
            for index, type_id in enumerate(self.type_ids)
        }

    def _predictive_nll_from_scores(
        self,
        scores: Mapping[str, float],
        transition_count: int,
        prior: Mapping[str, float] | None,
    ) -> float:
        prior_vector = self._prior_vector(prior)
        logits = np.log(prior_vector) + np.asarray(
            [scores[type_id] for type_id in self.type_ids], dtype=np.float64
        )
        return float(-_logsumexp(logits) / transition_count)

    def _prior_vector(self, prior: Mapping[str, float] | None) -> np.ndarray:
        if prior is None:
            return np.full(
                len(self.type_ids), 1.0 / len(self.type_ids), dtype=np.float64
            )
        if set(prior) != set(self.type_ids):
            raise EBPRError("prior must cover every source type exactly")
        values = np.asarray(
            [prior[type_id] for type_id in self.type_ids], dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise EBPRError("prior probabilities must be finite and strictly positive")
        total = float(np.sum(values))
        if not math.isfinite(total) or total <= 0.0:
            raise EBPRError("prior probabilities have an invalid sum")
        return values / total

    def _argmax_with_tie(
        self,
        values: Mapping[str, float],
        *,
        tie_identity_by_key: Mapping[str, str] | None = None,
    ) -> str:
        if not values:
            raise EBPRError("cannot select from an empty score mapping")
        identities = (
            {key: key for key in values}
            if tie_identity_by_key is None
            else tie_identity_by_key
        )
        if set(identities) != set(values):
            raise EBPRError("tie identities must cover every score key")
        best = max(values.values())
        tied = [key for key, value in values.items() if value == best]
        return min(
            tied,
            key=lambda key: hashlib.sha256(
                (self.tie_token + identities[key]).encode("utf-8")
            ).hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-safe source checkpoint."""

        return {
            "schema": self.schema,
            "type_ids": list(self.type_ids),
            "paired_policy_by_type": dict(self.paired_policy_by_type),
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "x_valid_mask": self.x_valid_mask.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_scale": self.y_scale.tolist(),
            "y_valid_mask": self.y_valid_mask.tolist(),
            "hidden_weight": self.hidden_weight.tolist(),
            "hidden_bias": self.hidden_bias.tolist(),
            "output_head_by_type": {
                key: value.tolist() for key, value in self.output_head_by_type.items()
            },
            "residual_variance_by_type": {
                key: value.tolist()
                for key, value in self.residual_variance_by_type.items()
            },
            "variance_floor": self.variance_floor,
            "posterior_temperature": self.posterior_temperature,
            "ridge": self.ridge,
            "feature_seed": self.feature_seed,
            "tie_token": self.tie_token,
            "calibration": dict(self.calibration),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EBPRFixedProbe":
        required = {
            "schema",
            "type_ids",
            "paired_policy_by_type",
            "x_mean",
            "x_scale",
            "x_valid_mask",
            "y_mean",
            "y_scale",
            "y_valid_mask",
            "hidden_weight",
            "hidden_bias",
            "output_head_by_type",
            "residual_variance_by_type",
            "variance_floor",
            "posterior_temperature",
            "ridge",
            "feature_seed",
            "tie_token",
            "calibration",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            extra = sorted(set(payload) - required)
            raise EBPRError(
                f"EBPR checkpoint keys mismatch; missing={missing}, extra={extra}"
            )
        try:
            return cls(
                schema=payload["schema"],
                type_ids=tuple(payload["type_ids"]),
                paired_policy_by_type=dict(payload["paired_policy_by_type"]),
                x_mean=np.asarray(payload["x_mean"], dtype=np.float64),
                x_scale=np.asarray(payload["x_scale"], dtype=np.float64),
                x_valid_mask=np.asarray(payload["x_valid_mask"], dtype=np.bool_),
                y_mean=np.asarray(payload["y_mean"], dtype=np.float64),
                y_scale=np.asarray(payload["y_scale"], dtype=np.float64),
                y_valid_mask=np.asarray(payload["y_valid_mask"], dtype=np.bool_),
                hidden_weight=np.asarray(payload["hidden_weight"], dtype=np.float64),
                hidden_bias=np.asarray(payload["hidden_bias"], dtype=np.float64),
                output_head_by_type={
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in dict(payload["output_head_by_type"]).items()
                },
                residual_variance_by_type={
                    key: np.asarray(value, dtype=np.float64)
                    for key, value in dict(payload["residual_variance_by_type"]).items()
                },
                variance_floor=payload["variance_floor"],
                posterior_temperature=payload["posterior_temperature"],
                ridge=payload["ridge"],
                feature_seed=payload["feature_seed"],
                tie_token=payload["tie_token"],
                calibration=dict(payload["calibration"]),
            )
        except (KeyError, TypeError) as error:
            raise EBPRError("malformed EBPR checkpoint payload") from error

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EBPRFixedProbe":
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EBPRError(f"cannot load EBPR checkpoint {source}") from error
        if not isinstance(payload, Mapping):
            raise EBPRError("EBPR checkpoint root must be an object")
        return cls.from_dict(payload)


def _episode_tuple(
    value: Sequence[TransitionEpisode] | TransitionEpisode,
    *,
    where: str,
) -> tuple[TransitionEpisode, ...]:
    if isinstance(value, TransitionEpisode):
        result = (value,)
    else:
        try:
            result = tuple(value)
        except TypeError as error:
            raise EBPRError(
                f"{where} must be a sequence of TransitionEpisode"
            ) from error
    if not result or any(not isinstance(item, TransitionEpisode) for item in result):
        raise EBPRError(f"{where} must contain at least one TransitionEpisode")
    return result


def _require_shared_dimensions(
    banks: Sequence[tuple[TransitionEpisode, ...]],
) -> None:
    state_dim: int | None = None
    action_dim: int | None = None
    for episodes in banks:
        for episode in episodes:
            if state_dim is None:
                state_dim = episode.state.shape[1]
                action_dim = episode.action.shape[1]
            elif (
                episode.state.shape[1] != state_dim
                or episode.action.shape[1] != action_dim
            ):
                raise EBPRError(
                    "all source episodes must share state/action dimensions"
                )


def _join_episodes(
    episodes: tuple[TransitionEpisode, ...],
) -> tuple[np.ndarray, np.ndarray]:
    xy = [episode.xy() for episode in episodes]
    return (
        np.concatenate([item[0] for item in xy], axis=0),
        np.concatenate([item[1] for item in xy], axis=0),
    )


def _candidate_grid(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    candidates = tuple(sorted({_positive(value, name=name) for value in values}))
    if not candidates:
        raise EBPRError(f"{name} must be non-empty")
    return candidates


def _features(
    x: np.ndarray,
    *,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    x_valid_mask: np.ndarray,
    hidden_weight: np.ndarray,
    hidden_bias: np.ndarray,
) -> np.ndarray:
    standardized = (x[:, x_valid_mask] - x_mean[x_valid_mask]) / x_scale[x_valid_mask]
    hidden = np.tanh(standardized @ hidden_weight + hidden_bias)
    return np.concatenate(
        (np.ones((x.shape[0], 1), dtype=np.float64), standardized, hidden), axis=1
    )


def _predict_normalized(
    x: np.ndarray,
    *,
    head: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    x_valid_mask: np.ndarray,
    hidden_weight: np.ndarray,
    hidden_bias: np.ndarray,
) -> np.ndarray:
    return (
        _features(
            x,
            x_mean=x_mean,
            x_scale=x_scale,
            x_valid_mask=x_valid_mask,
            hidden_weight=hidden_weight,
            hidden_bias=hidden_bias,
        )
        @ head
    )


def _episode_log_likelihood(
    episode: TransitionEpisode,
    *,
    head: np.ndarray,
    variance: np.ndarray,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    x_valid_mask: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    y_valid_mask: np.ndarray,
    hidden_weight: np.ndarray,
    hidden_bias: np.ndarray,
) -> float:
    x, y = episode.xy()
    prediction = _predict_normalized(
        x,
        head=head,
        x_mean=x_mean,
        x_scale=x_scale,
        x_valid_mask=x_valid_mask,
        hidden_weight=hidden_weight,
        hidden_bias=hidden_bias,
    )
    observed = (y[:, y_valid_mask] - y_mean[y_valid_mask]) / y_scale[y_valid_mask]
    residual_squared = _safe_squared(observed - prediction)
    log_density = -0.5 * (
        _LOG_2PI + np.log(variance)[None, :] + residual_squared / variance[None, :]
    )
    return _stable_total(log_density) / int(np.sum(y_valid_mask))


__all__ = [
    "DEFAULT_TEMPERATURE_CANDIDATES",
    "DEFAULT_VARIANCE_FLOOR_CANDIDATES",
    "EBPRError",
    "EBPRFixedProbe",
    "EBPRSelection",
    "EBPR_HYBRID_METHOD_ID",
    "EBPR_METHOD_ID",
    "EBPR_SCHEMA",
    "TransitionEpisode",
]
