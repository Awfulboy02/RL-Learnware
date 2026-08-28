"""Minimal, real P0 classifiers for reward-free v0.5 retrieval.

All inputs are already-canonicalized, reward-masked ``(delta_o, action)``
points grouped by complete episodes.  This module has no environment, reward,
target-label, or candidate-policy interface.  Supervised labels are accepted
only by the two source-fit classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from ..io import atomic_write_npz
from ..rkme.distance import empirical_to_reduced_distance
from ..rkme.empirical import EmpiricalKME, build_empirical_kme, empirical_mmd2
from ..rkme.gaussian import GaussianKernel
from ..rkme.reducer import ReducedRKME, ReducerConfig, reduce_kme
from .specifications import (
    RFFMap,
    RFFSpecification,
    SWEMap,
    SWESpecification,
    squared_vector_distance,
    validate_episode_offsets,
)


RAW_DELTA_RKME = "RAW_DELTA_RKME"
EMPIRICAL_MMD_NN = "EMPIRICAL_MMD_NN"
SUMMARY_LOGREG = "SUMMARY_LOGREG"
KME_KRR = "KME_KRR"
RFF_KME_NN = "RFF_KME_NN"
SWE_NN = "SWE_NN"
P0_METHOD_IDS = (
    RAW_DELTA_RKME,
    EMPIRICAL_MMD_NN,
    SUMMARY_LOGREG,
    KME_KRR,
    RFF_KME_NN,
    SWE_NN,
)


class V05ClassifierError(ValueError):
    """A classifier input, fit, or persisted model violates the P0 contract."""


def _canonical_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V05ClassifierError(f"{where} must be a non-empty canonical string")
    return value


def _positive_float(value: Any, where: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V05ClassifierError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise V05ClassifierError(f"{where} must be finite and {qualifier}")
    return result


def _readonly_numeric(value: Any, *, ndim: int, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim or raw.size == 0:
        raise V05ClassifierError(f"{where} must be a non-empty numeric {ndim}D array")
    result = np.ascontiguousarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise V05ClassifierError(f"{where} contains non-finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EpisodeBank:
    """A complete-episode view over one reward-free canonical point bank."""

    points: np.ndarray
    episode_offsets: np.ndarray

    def __post_init__(self) -> None:
        points = _readonly_numeric(self.points, ndim=2, where="points")
        offsets = validate_episode_offsets(
            self.episode_offsets, transition_count=points.shape[0]
        ).copy()
        offsets.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "episode_offsets", offsets)

    @property
    def input_dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    @property
    def bank_digest(self) -> str:
        return sha256_ndarrays(
            {"points": self.points, "episode_offsets": self.episode_offsets}
        )

    def episode(self, index: int) -> np.ndarray:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise V05ClassifierError("episode index must be an integer")
        resolved = int(index)
        if resolved < 0 or resolved >= self.episode_count:
            raise V05ClassifierError("episode index is outside the bank")
        start = int(self.episode_offsets[resolved])
        stop = int(self.episode_offsets[resolved + 1])
        return self.points[start:stop]

    def prefix(self, episode_count: int) -> "EpisodeBank":
        if isinstance(episode_count, (bool, np.bool_)) or not isinstance(
            episode_count, (int, np.integer)
        ):
            raise V05ClassifierError("episode_count must be an integer")
        count = int(episode_count)
        if count <= 0 or count > self.episode_count:
            raise V05ClassifierError("episode prefix is outside the bank")
        stop = int(self.episode_offsets[count])
        return EpisodeBank(self.points[:stop], self.episode_offsets[: count + 1])


def _bank(value: EpisodeBank | tuple[Any, Any]) -> EpisodeBank:
    if isinstance(value, EpisodeBank):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise V05ClassifierError("a bank must be EpisodeBank or (points, offsets)")
    return EpisodeBank(value[0], value[1])


def _source_banks(
    values: Mapping[str, EpisodeBank | tuple[Any, Any]],
) -> dict[str, EpisodeBank]:
    if not isinstance(values, Mapping) or not values:
        raise V05ClassifierError("source_banks must be a non-empty mapping")
    result: dict[str, EpisodeBank] = {}
    for source_id, value in values.items():
        identifier = _canonical_id(source_id, "source ID")
        if identifier in result:
            raise V05ClassifierError("source IDs must be unique")
        result[identifier] = _bank(value)
    ordered = {identifier: result[identifier] for identifier in sorted(result)}
    widths = {bank.input_dim for bank in ordered.values()}
    if len(widths) != 1:
        raise V05ClassifierError("all source banks must have the same point width")
    return ordered


def _labels_for_sources(
    source_ids: Iterable[str], labels: Mapping[str, str]
) -> dict[str, str]:
    if not isinstance(labels, Mapping):
        raise V05ClassifierError("source_labels must be a mapping")
    expected = set(source_ids)
    if set(labels) != expected:
        raise V05ClassifierError("source_labels must exactly cover source banks")
    return {
        source_id: _canonical_id(labels[source_id], "source supervised label")
        for source_id in sorted(expected)
    }


def _finite_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for identifier, value in scores.items():
        key = _canonical_id(identifier, "score ID")
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise V05ClassifierError("scores must be numeric")
        score = float(value)
        if not math.isfinite(score):
            raise V05ClassifierError("scores must be finite")
        result[key] = score
    if not result:
        raise V05ClassifierError("a score vector must not be empty")
    return result


@dataclass(frozen=True)
class RawDeltaRKMENN:
    """v0.31 Raw-Delta formula rebuilt on the matched source-train bank."""

    sources: Mapping[str, ReducedRKME]
    bandwidth: float
    protocol_id: str
    method_id: str = RAW_DELTA_RKME

    def __post_init__(self) -> None:
        bandwidth = _positive_float(self.bandwidth, "bandwidth")
        protocol_id = _canonical_id(self.protocol_id, "protocol_id")
        if not isinstance(self.sources, Mapping) or not self.sources:
            raise V05ClassifierError("Raw sources must be a non-empty mapping")
        sources: dict[str, ReducedRKME] = {}
        widths: set[int] = set()
        for source_id in sorted(self.sources):
            identifier = _canonical_id(source_id, "source ID")
            source = self.sources[source_id]
            if not isinstance(source, ReducedRKME):
                raise V05ClassifierError("Raw sources must be ReducedRKME objects")
            if source.bandwidth != bandwidth or (
                source.protocol_id and source.protocol_id != protocol_id
            ):
                raise V05ClassifierError("Raw source kernel/protocol mismatch")
            widths.add(int(source.supports.shape[1]))
            sources[identifier] = source
        if len(widths) != 1:
            raise V05ClassifierError("Raw source support dimensions differ")
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "sources", MappingProxyType(sources))

    @classmethod
    def fit(
        cls,
        source_banks: Mapping[str, EpisodeBank | tuple[Any, Any]],
        *,
        bandwidth: float,
        protocol_id: str,
        reducer_config: ReducerConfig | Mapping[str, Any] = ReducerConfig(),
    ) -> "RawDeltaRKMENN":
        banks = _source_banks(source_banks)
        kernel = GaussianKernel(bandwidth)
        sources = {
            source_id: reduce_kme(
                build_empirical_kme(
                    bank.points,
                    kernel,
                    episode_offsets=bank.episode_offsets,
                    protocol_id=protocol_id,
                    dataset_digest=bank.bank_digest,
                    source_task=source_id,
                ),
                reducer_config,
            )
            for source_id, bank in banks.items()
        }
        return cls(sources=sources, bandwidth=kernel.bandwidth, protocol_id=protocol_id)

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        query = _bank(query_bank)
        target = build_empirical_kme(
            query.points,
            GaussianKernel(self.bandwidth),
            episode_offsets=query.episode_offsets,
            protocol_id=self.protocol_id,
            dataset_digest=query.bank_digest,
        )
        return _finite_scores(
            {
                source_id: -empirical_to_reduced_distance(target, source).distance
                for source_id, source in self.sources.items()
            }
        )


@dataclass(frozen=True)
class EmpiricalMMDNN:
    """Unreduced, episode-balanced biased Gaussian-MMD nearest neighbour."""

    sources: Mapping[str, EmpiricalKME]
    bandwidth: float
    protocol_id: str
    method_id: str = EMPIRICAL_MMD_NN

    def __post_init__(self) -> None:
        bandwidth = _positive_float(self.bandwidth, "bandwidth")
        protocol_id = _canonical_id(self.protocol_id, "protocol_id")
        if not isinstance(self.sources, Mapping) or not self.sources:
            raise V05ClassifierError("MMD sources must be a non-empty mapping")
        sources: dict[str, EmpiricalKME] = {}
        widths: set[int] = set()
        for source_id in sorted(self.sources):
            identifier = _canonical_id(source_id, "source ID")
            source = self.sources[source_id]
            if not isinstance(source, EmpiricalKME):
                raise V05ClassifierError("MMD sources must be EmpiricalKME objects")
            if source.bandwidth != bandwidth or (
                source.protocol_id and source.protocol_id != protocol_id
            ):
                raise V05ClassifierError("MMD source kernel/protocol mismatch")
            widths.add(int(source.points.shape[1]))
            sources[identifier] = source
        if len(widths) != 1:
            raise V05ClassifierError("MMD source point dimensions differ")
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "sources", MappingProxyType(sources))

    @classmethod
    def fit(
        cls,
        source_banks: Mapping[str, EpisodeBank | tuple[Any, Any]],
        *,
        bandwidth: float,
        protocol_id: str,
    ) -> "EmpiricalMMDNN":
        banks = _source_banks(source_banks)
        kernel = GaussianKernel(bandwidth)
        sources = {
            source_id: build_empirical_kme(
                bank.points,
                kernel,
                episode_offsets=bank.episode_offsets,
                protocol_id=protocol_id,
                dataset_digest=bank.bank_digest,
                source_task=source_id,
            )
            for source_id, bank in banks.items()
        }
        return cls(sources=sources, bandwidth=kernel.bandwidth, protocol_id=protocol_id)

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        query = _bank(query_bank)
        target = build_empirical_kme(
            query.points,
            GaussianKernel(self.bandwidth),
            episode_offsets=query.episode_offsets,
            protocol_id=self.protocol_id,
            dataset_digest=query.bank_digest,
        )
        return _finite_scores(
            {
                source_id: -math.sqrt(empirical_mmd2(target, source))
                for source_id, source in self.sources.items()
            }
        )


def _episode_summaries(bank: EpisodeBank) -> np.ndarray:
    rows = []
    for episode_index in range(bank.episode_count):
        episode = bank.episode(episode_index)
        rows.append(np.concatenate((np.mean(episode, axis=0), np.std(episode, axis=0))))
    return np.asarray(rows, dtype=np.float64)


def _supervised_summary_rows(
    banks: Mapping[str, EpisodeBank], labels: Mapping[str, str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    features: list[np.ndarray] = []
    targets: list[str] = []
    for source_id, bank in banks.items():
        rows = _episode_summaries(bank)
        features.append(rows)
        targets.extend([labels[source_id]] * rows.shape[0])
    return np.concatenate(features, axis=0), tuple(targets)


def _softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _fit_softmax(
    features: np.ndarray,
    target_indices: np.ndarray,
    class_count: int,
    *,
    l2: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    sample_count, feature_count = features.shape
    weights = np.zeros((feature_count, class_count), dtype=np.float64)
    intercept = np.zeros(class_count, dtype=np.float64)
    augmented = np.concatenate(
        (features, np.ones((sample_count, 1), dtype=np.float64)), axis=1
    )
    spectral_squared = float(np.linalg.norm(augmented, ord=2) ** 2)
    step_size = 1.0 / (0.5 * spectral_squared / sample_count + l2 + 1.0e-12)
    one_hot = np.eye(class_count, dtype=np.float64)[target_indices]
    final_iteration = max_iter
    for iteration in range(1, max_iter + 1):
        probabilities = _softmax_probabilities(features @ weights + intercept)
        residual = (probabilities - one_hot) / sample_count
        gradient_weights = features.T @ residual + l2 * weights
        gradient_intercept = np.sum(residual, axis=0)
        gradient_norm = math.sqrt(
            float(np.sum(np.square(gradient_weights)))
            + float(np.sum(np.square(gradient_intercept)))
        )
        weights -= step_size * gradient_weights
        intercept -= step_size * gradient_intercept
        if gradient_norm <= tolerance:
            final_iteration = iteration
            break
    return weights, intercept, final_iteration


def _cross_entropy(logits: np.ndarray, target_indices: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    losses = log_normalizer - shifted[np.arange(shifted.shape[0]), target_indices]
    return float(np.mean(losses))


@dataclass(frozen=True)
class SummaryLogReg:
    """Episode mean/std followed by true multinomial logistic regression."""

    class_ids: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    selected_l2: float
    training_iterations: int
    method_id: str = SUMMARY_LOGREG

    def __post_init__(self) -> None:
        classes = tuple(_canonical_id(item, "class ID") for item in self.class_ids)
        if (
            len(classes) < 2
            or len(classes) != len(set(classes))
            or classes != tuple(sorted(classes))
        ):
            raise V05ClassifierError("class_ids must contain >=2 sorted unique IDs")
        mean = _readonly_numeric(self.feature_mean, ndim=1, where="feature_mean")
        scale = _readonly_numeric(self.feature_scale, ndim=1, where="feature_scale")
        weights = _readonly_numeric(self.weights, ndim=2, where="weights")
        intercept = _readonly_numeric(self.intercept, ndim=1, where="intercept")
        if (
            scale.shape != mean.shape
            or weights.shape != (mean.size, len(classes))
            or intercept.shape != (len(classes),)
            or np.any(scale <= 0.0)
            or mean.size % 2
        ):
            raise V05ClassifierError("Summary LogReg model arrays are incompatible")
        selected_l2 = _positive_float(self.selected_l2, "selected_l2", allow_zero=True)
        if (
            isinstance(self.training_iterations, (bool, np.bool_))
            or int(self.training_iterations) <= 0
        ):
            raise V05ClassifierError("training_iterations must be positive")
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "selected_l2", selected_l2)
        object.__setattr__(self, "training_iterations", int(self.training_iterations))

    @property
    def input_dim(self) -> int:
        return int(self.feature_mean.size // 2)

    @property
    def model_digest(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "class_ids": list(self.class_ids),
                "selected_l2": self.selected_l2,
                "training_iterations": self.training_iterations,
                "arrays_sha256": sha256_ndarrays(
                    {
                        "feature_mean": self.feature_mean,
                        "feature_scale": self.feature_scale,
                        "weights": self.weights,
                        "intercept": self.intercept,
                    }
                ),
            }
        )

    @classmethod
    def fit(
        cls,
        source_train: Mapping[str, EpisodeBank | tuple[Any, Any]],
        source_labels: Mapping[str, str],
        source_validation: Mapping[str, EpisodeBank | tuple[Any, Any]],
        *,
        l2_grid: Sequence[float] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0),
        max_iter: int = 3_000,
        tolerance: float = 1.0e-9,
    ) -> "SummaryLogReg":
        train = _source_banks(source_train)
        validation = _source_banks(source_validation)
        if set(train) != set(validation):
            raise V05ClassifierError("train/validation sources must match exactly")
        labels = _labels_for_sources(train, source_labels)
        classes = tuple(sorted(set(labels.values())))
        if len(classes) < 2:
            raise V05ClassifierError("Summary LogReg needs at least two labels")
        class_index = {label: index for index, label in enumerate(classes)}
        x_train, y_train_ids = _supervised_summary_rows(train, labels)
        x_validation, y_validation_ids = _supervised_summary_rows(validation, labels)
        feature_mean = np.mean(x_train, axis=0)
        raw_scale = np.std(x_train, axis=0)
        feature_scale = np.where(raw_scale > 1.0e-12, raw_scale, 1.0)
        train_z = (x_train - feature_mean) / feature_scale
        validation_z = (x_validation - feature_mean) / feature_scale
        y_train = np.asarray(
            [class_index[item] for item in y_train_ids], dtype=np.int64
        )
        y_validation = np.asarray(
            [class_index[item] for item in y_validation_ids], dtype=np.int64
        )
        if isinstance(max_iter, (bool, np.bool_)) or int(max_iter) <= 0:
            raise V05ClassifierError("max_iter must be positive")
        tolerance = _positive_float(tolerance, "tolerance")
        grid = tuple(
            sorted(
                {
                    _positive_float(value, "l2_grid item", allow_zero=True)
                    for value in l2_grid
                }
            )
        )
        if not grid:
            raise V05ClassifierError("l2_grid must not be empty")
        candidates = []
        for l2 in grid:
            weights, intercept, iterations = _fit_softmax(
                train_z,
                y_train,
                len(classes),
                l2=l2,
                max_iter=int(max_iter),
                tolerance=tolerance,
            )
            validation_loss = _cross_entropy(
                validation_z @ weights + intercept, y_validation
            )
            candidates.append((validation_loss, l2, weights, intercept, iterations))
        _, selected_l2, weights, intercept, iterations = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        return cls(
            class_ids=classes,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            weights=weights,
            intercept=intercept,
            selected_l2=selected_l2,
            training_iterations=iterations,
        )

    def episode_logits(self, query_bank: EpisodeBank | tuple[Any, Any]) -> np.ndarray:
        query = _bank(query_bank)
        if query.input_dim != self.input_dim:
            raise V05ClassifierError("Summary query point width differs from fit")
        summaries = _episode_summaries(query)
        standardized = (summaries - self.feature_mean) / self.feature_scale
        return standardized @ self.weights + self.intercept

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        logits = np.mean(self.episode_logits(query_bank), axis=0)
        return _finite_scores(dict(zip(self.class_ids, logits, strict=True)))

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        return atomic_write_npz(
            path,
            {
                "class_ids": np.asarray(self.class_ids),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "weights": self.weights,
                "intercept": self.intercept,
                "selected_l2": np.asarray(self.selected_l2, dtype=np.float64),
                "training_iterations": np.asarray(
                    self.training_iterations, dtype=np.int64
                ),
                "method_id": np.asarray(self.method_id),
                "model_digest": np.asarray(self.model_digest),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SummaryLogReg":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["method_id"]) != SUMMARY_LOGREG:
                raise V05ClassifierError("persisted model is not SUMMARY_LOGREG")
            result = cls(
                class_ids=tuple(str(item) for item in data["class_ids"]),
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                weights=data["weights"],
                intercept=data["intercept"],
                selected_l2=float(data["selected_l2"]),
                training_iterations=int(data["training_iterations"]),
            )
            if str(data["model_digest"]) != result.model_digest:
                raise V05ClassifierError("Summary LogReg model digest mismatch")
            return result


def _stack_bank_episodes(banks: Mapping[str, EpisodeBank]) -> EpisodeBank:
    episodes = [
        bank.episode(index)
        for bank in banks.values()
        for index in range(bank.episode_count)
    ]
    points = np.concatenate(episodes, axis=0)
    offsets = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum([episode.shape[0] for episode in episodes], dtype=np.int64),
        )
    )
    return EpisodeBank(points, offsets)


def _expected_kernel_cross(
    left: EpisodeBank, right: EpisodeBank, kernel: GaussianKernel
) -> np.ndarray:
    if left.input_dim != right.input_dim:
        raise V05ClassifierError("expected-kernel banks have different point widths")
    result = np.empty((left.episode_count, right.episode_count), dtype=np.float64)
    for left_index in range(left.episode_count):
        left_episode = left.episode(left_index)
        for right_index in range(right.episode_count):
            result[left_index, right_index] = float(
                np.mean(kernel.gram(left_episode, right.episode(right_index)))
            )
    return result


@dataclass(frozen=True)
class KMEKRR:
    """Multiclass ridge on an exact episode-level expected-kernel Gram."""

    class_ids: tuple[str, ...]
    training_bank: EpisodeBank
    alpha: np.ndarray
    bandwidth: float
    selected_ridge: float
    method_id: str = KME_KRR

    def __post_init__(self) -> None:
        classes = tuple(_canonical_id(item, "class ID") for item in self.class_ids)
        if (
            len(classes) < 2
            or len(classes) != len(set(classes))
            or classes != tuple(sorted(classes))
        ):
            raise V05ClassifierError("class_ids must contain >=2 sorted unique IDs")
        training_bank = _bank(self.training_bank)
        alpha = _readonly_numeric(self.alpha, ndim=2, where="alpha")
        if alpha.shape != (training_bank.episode_count, len(classes)):
            raise V05ClassifierError(
                "KRR alpha shape differs from training episodes/classes"
            )
        bandwidth = _positive_float(self.bandwidth, "bandwidth")
        ridge = _positive_float(self.selected_ridge, "selected_ridge")
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "training_bank", training_bank)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(self, "selected_ridge", ridge)

    @property
    def model_digest(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "class_ids": list(self.class_ids),
                "bandwidth": self.bandwidth,
                "selected_ridge": self.selected_ridge,
                "arrays_sha256": sha256_ndarrays(
                    {
                        "training_points": self.training_bank.points,
                        "training_episode_offsets": self.training_bank.episode_offsets,
                        "alpha": self.alpha,
                    }
                ),
            }
        )

    @classmethod
    def fit(
        cls,
        source_train: Mapping[str, EpisodeBank | tuple[Any, Any]],
        source_labels: Mapping[str, str],
        source_validation: Mapping[str, EpisodeBank | tuple[Any, Any]],
        *,
        bandwidth: float,
        ridge_grid: Sequence[float] = (1.0e-6, 1.0e-4, 1.0e-2, 1.0),
    ) -> "KMEKRR":
        train = _source_banks(source_train)
        validation = _source_banks(source_validation)
        if set(train) != set(validation):
            raise V05ClassifierError("train/validation sources must match exactly")
        labels = _labels_for_sources(train, source_labels)
        classes = tuple(sorted(set(labels.values())))
        if len(classes) < 2:
            raise V05ClassifierError("KRR needs at least two labels")
        class_index = {label: index for index, label in enumerate(classes)}
        training_bank = _stack_bank_episodes(train)
        validation_bank = _stack_bank_episodes(validation)
        train_targets = np.asarray(
            [
                class_index[labels[source_id]]
                for source_id, bank in train.items()
                for _ in range(bank.episode_count)
            ],
            dtype=np.int64,
        )
        validation_targets = np.asarray(
            [
                class_index[labels[source_id]]
                for source_id, bank in validation.items()
                for _ in range(bank.episode_count)
            ],
            dtype=np.int64,
        )
        target_matrix = np.eye(len(classes), dtype=np.float64)[train_targets]
        validation_matrix = np.eye(len(classes), dtype=np.float64)[validation_targets]
        kernel = GaussianKernel(bandwidth)
        gram = _expected_kernel_cross(training_bank, training_bank, kernel)
        gram = 0.5 * (gram + gram.T)
        validation_gram = _expected_kernel_cross(validation_bank, training_bank, kernel)
        grid = tuple(
            sorted({_positive_float(item, "ridge_grid item") for item in ridge_grid})
        )
        if not grid:
            raise V05ClassifierError("ridge_grid must not be empty")
        identity = np.eye(gram.shape[0], dtype=np.float64)
        candidates = []
        for ridge in grid:
            try:
                alpha = np.linalg.solve(gram + ridge * identity, target_matrix)
            except np.linalg.LinAlgError:
                continue
            validation_scores = validation_gram @ alpha
            validation_mse = float(
                np.mean(np.square(validation_scores - validation_matrix))
            )
            candidates.append((validation_mse, ridge, alpha))
        if not candidates:
            raise V05ClassifierError("all KRR ridge solves failed")
        _, selected_ridge, alpha = min(candidates, key=lambda item: (item[0], item[1]))
        return cls(
            class_ids=classes,
            training_bank=training_bank,
            alpha=alpha,
            bandwidth=kernel.bandwidth,
            selected_ridge=selected_ridge,
        )

    def episode_scores(self, query_bank: EpisodeBank | tuple[Any, Any]) -> np.ndarray:
        query = _bank(query_bank)
        gram = _expected_kernel_cross(
            query, self.training_bank, GaussianKernel(self.bandwidth)
        )
        return gram @ self.alpha

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        scores = np.mean(self.episode_scores(query_bank), axis=0)
        return _finite_scores(dict(zip(self.class_ids, scores, strict=True)))

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        return atomic_write_npz(
            path,
            {
                "class_ids": np.asarray(self.class_ids),
                "training_points": self.training_bank.points,
                "training_episode_offsets": self.training_bank.episode_offsets,
                "alpha": self.alpha,
                "bandwidth": np.asarray(self.bandwidth, dtype=np.float64),
                "selected_ridge": np.asarray(self.selected_ridge, dtype=np.float64),
                "method_id": np.asarray(self.method_id),
                "model_digest": np.asarray(self.model_digest),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "KMEKRR":
        with np.load(Path(path), allow_pickle=False) as data:
            if str(data["method_id"]) != KME_KRR:
                raise V05ClassifierError("persisted model is not KME_KRR")
            result = cls(
                class_ids=tuple(str(item) for item in data["class_ids"]),
                training_bank=EpisodeBank(
                    data["training_points"], data["training_episode_offsets"]
                ),
                alpha=data["alpha"],
                bandwidth=float(data["bandwidth"]),
                selected_ridge=float(data["selected_ridge"]),
            )
            if str(data["model_digest"]) != result.model_digest:
                raise V05ClassifierError("KRR model digest mismatch")
            return result


def _prototype_vectors(
    prototypes: Mapping[str, Any], *, output_dim: int
) -> Mapping[str, np.ndarray]:
    if not isinstance(prototypes, Mapping) or not prototypes:
        raise V05ClassifierError("fixed-vector prototypes must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    for source_id in sorted(prototypes):
        identifier = _canonical_id(source_id, "source ID")
        vector = _readonly_numeric(prototypes[source_id], ndim=1, where="prototype")
        if vector.shape != (output_dim,):
            raise V05ClassifierError("fixed-vector prototype dimension mismatch")
        result[identifier] = vector
    return MappingProxyType(result)


@dataclass(frozen=True)
class RFFKMENN:
    rff_map: RFFMap
    prototypes: Mapping[str, np.ndarray]
    method_id: str = RFF_KME_NN

    def __post_init__(self) -> None:
        if not isinstance(self.rff_map, RFFMap):
            raise V05ClassifierError("RFFKMENN requires an RFFMap")
        object.__setattr__(
            self,
            "prototypes",
            _prototype_vectors(self.prototypes, output_dim=self.rff_map.output_dim),
        )

    @classmethod
    def fit(
        cls,
        source_banks: Mapping[str, EpisodeBank | tuple[Any, Any]],
        *,
        rff_map: RFFMap,
    ) -> "RFFKMENN":
        banks = _source_banks(source_banks)
        prototypes = {
            source_id: rff_map.embed(bank.points, bank.episode_offsets).vector
            for source_id, bank in banks.items()
        }
        return cls(rff_map=rff_map, prototypes=prototypes)

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        query = _bank(query_bank)
        specification = self.rff_map.embed(query.points, query.episode_offsets)
        return self.score_specification(specification)

    def score_specification(self, query: RFFSpecification) -> dict[str, float]:
        if not isinstance(query, RFFSpecification):
            raise V05ClassifierError("RFF scorer accepts only an RFFSpecification")
        if query.map_digest != self.rff_map.map_digest:
            raise V05ClassifierError("RFF query/source map digests differ")
        return _finite_scores(
            {
                source_id: -math.sqrt(squared_vector_distance(query.vector, prototype))
                for source_id, prototype in self.prototypes.items()
            }
        )


@dataclass(frozen=True)
class SWENN:
    swe_map: SWEMap
    prototypes: Mapping[str, np.ndarray]
    method_id: str = SWE_NN

    def __post_init__(self) -> None:
        if not isinstance(self.swe_map, SWEMap):
            raise V05ClassifierError("SWENN requires an SWEMap")
        object.__setattr__(
            self,
            "prototypes",
            _prototype_vectors(self.prototypes, output_dim=self.swe_map.output_dim),
        )

    @classmethod
    def fit(
        cls,
        source_banks: Mapping[str, EpisodeBank | tuple[Any, Any]],
        *,
        swe_map: SWEMap,
    ) -> "SWENN":
        banks = _source_banks(source_banks)
        prototypes = {
            source_id: swe_map.embed(bank.points, bank.episode_offsets).vector
            for source_id, bank in banks.items()
        }
        return cls(swe_map=swe_map, prototypes=prototypes)

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        query = _bank(query_bank)
        specification = self.swe_map.embed(query.points, query.episode_offsets)
        return self.score_specification(specification)

    def score_specification(self, query: SWESpecification) -> dict[str, float]:
        if not isinstance(query, SWESpecification):
            raise V05ClassifierError("SWE scorer accepts only an SWESpecification")
        if query.map_digest != self.swe_map.map_digest:
            raise V05ClassifierError("SWE query/source map digests differ")
        return _finite_scores(
            {
                source_id: -math.sqrt(squared_vector_distance(query.vector, prototype))
                for source_id, prototype in self.prototypes.items()
            }
        )


def p0_method_cards() -> tuple[dict[str, Any], ...]:
    """Return the six frozen P0 identities without registering conditional P1 rows."""

    common = {
        "priority": "P0",
        "probe_protocol": "Q0_COMMON_GAUSSIAN_OPEN_LOOP",
        "input": "reward-masked canonical (delta_o, action)",
        "candidate_policy_access": False,
        "candidate_conditioned_steps": 0,
        "formal_privacy": "NONE",
    }
    rows = (
        {
            **common,
            "method_id": RAW_DELTA_RKME,
            "access_tier": "STRUCTURED_SPEC",
            "source_fit": "per-anchor ReducedRKME on equal-episode source train",
            "target_aggregation": "one equal-episode empirical KME over the B-prefix",
            "score": "negative Gaussian-RKHS distance to ReducedRKME",
            "adaptation_status": "frozen v0.31 numerical path, matched-source rebuild",
        },
        {
            **common,
            "method_id": EMPIRICAL_MMD_NN,
            "access_tier": "FULL_SUPPORT_CONTROL",
            "source_fit": "per-anchor unreduced equal-episode empirical KME",
            "target_aggregation": "one equal-episode empirical KME over the B-prefix",
            "score": "negative biased weighted Gaussian MMD",
            "adaptation_status": "unreduced compression control",
        },
        {
            **common,
            "method_id": SUMMARY_LOGREG,
            "access_tier": "STRUCTURED_SPEC",
            "source_fit": "source-only multinomial logistic regression; validation L2",
            "target_aggregation": "arithmetic mean of per-episode mean/std logits",
            "score": "multinomial decision logit before softmax",
            "adaptation_status": "simple supervised sanity baseline",
        },
        {
            **common,
            "method_id": KME_KRR,
            "access_tier": "JOINT_SOURCE_FIT",
            "source_fit": "episode-level exact expected-kernel Gram; validation ridge",
            "target_aggregation": "arithmetic mean of per-episode class scores",
            "score": "multiclass kernel-ridge decision score",
            "adaptation_status": "classical distribution-regression control; not SMM",
        },
        {
            **common,
            "method_id": RFF_KME_NN,
            "access_tier": "FIXED_VECTOR_SPEC",
            "source_fit": "per-anchor equal mean of bounded public cos/sin episode means",
            "target_aggregation": "equal mean of the B per-episode RFF means",
            "score": "negative L2 to source RFF mean",
            "adaptation_status": "finite-dimensional Gaussian-MMD interface control",
        },
        {
            **common,
            "method_id": SWE_NN,
            "access_tier": "FIXED_VECTOR_SPEC",
            "source_fit": "per-anchor fixed public-direction quantile sketch",
            "target_aggregation": "mix B episodes with equal mass, then sketch once",
            "score": "negative L2 to source SWE vector",
            "adaptation_status": (
                "reference-free fixed-projection retrieval adaptation with fixed "
                "interior quantile grid and linear interpolation; not a faithful "
                "PSWE/SLoSH training reproduction"
            ),
        },
    )
    return tuple(dict(row) for row in rows)


__all__ = [
    "EMPIRICAL_MMD_NN",
    "KME_KRR",
    "P0_METHOD_IDS",
    "RAW_DELTA_RKME",
    "RFF_KME_NN",
    "SUMMARY_LOGREG",
    "SWE_NN",
    "EmpiricalMMDNN",
    "EpisodeBank",
    "KMEKRR",
    "RFFKMENN",
    "RawDeltaRKMENN",
    "SWENN",
    "SummaryLogReg",
    "V05ClassifierError",
    "p0_method_cards",
]
