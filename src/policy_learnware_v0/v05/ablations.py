"""Small, source-only ablation heads for the v0.5 representation matrix.

The production six-method panel remains frozen.  This module adds only the
missing cells needed to compare representation and decision-head choices on
the same canonical, reward-free episode banks.  It deliberately reuses the
numeric helpers in :mod:`classifiers` instead of introducing another model
registry or training framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from ..rkme.empirical import episode_balanced_weights
from .classifiers import (
    EpisodeBank,
    SWENN,
    V05ClassifierError,
    _bank,
    _cross_entropy,
    _episode_summaries,
    _finite_scores,
    _fit_softmax,
    _labels_for_sources,
    _positive_float,
    _readonly_numeric,
    _source_banks,
)
from .specifications import RFFMap


B0_RANDOM = "B0_RANDOM"
B3A_RAW_MOMENT_NN = "B3A_RAW_MOMENT_NN"
SUMMARY_NN = "SUMMARY_NN"
RFF_LOGREG = "RFF_LOGREG"
RFF_RIDGE = "RFF_RIDGE"
SWE_1024_NN = "SWE_1024_NN"
ABLATION_METHOD_IDS = (
    B0_RANDOM,
    B3A_RAW_MOMENT_NN,
    SUMMARY_NN,
    RFF_LOGREG,
    RFF_RIDGE,
    SWE_1024_NN,
)


def nested_row_order(
    parent_membership_digest: str,
    physical_episode: int,
    public_seed: int,
    *,
    row_count: int = 64,
) -> tuple[int, ...]:
    """Freeze one label-independent nested row order inside an episode."""

    parent = _sha256(parent_membership_digest, "parent membership digest")
    for value, where, minimum in (
        (physical_episode, "physical episode", 0),
        (public_seed, "public seed", 0),
        (row_count, "row count", 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
        ):
            raise V05ClassifierError(f"{where} is outside its allowed range")
    return tuple(
        sorted(
            range(int(row_count)),
            key=lambda row: (
                sha256_json(
                    {
                        "schema": "policy-learnware.v05-ablation-row-key.v1",
                        "public_seed": int(public_seed),
                        "parent_membership_digest": parent,
                        "physical_episode_position": int(physical_episode),
                        "row_position": row,
                    }
                ),
                row,
            ),
        )
    )


def _class_prototypes(
    source_vectors: Mapping[str, Any],
    source_labels: Mapping[str, str],
) -> Mapping[str, np.ndarray]:
    if not isinstance(source_vectors, Mapping) or not source_vectors:
        raise V05ClassifierError("source vectors must be a non-empty mapping")
    labels = _labels_for_sources(source_vectors, source_labels)
    if len(set(labels.values())) != len(labels):
        raise V05ClassifierError("prototype baselines require one class per source")
    result: dict[str, np.ndarray] = {}
    width: int | None = None
    for source_id in sorted(source_vectors):
        vector = _readonly_numeric(
            source_vectors[source_id], ndim=1, where="source prototype"
        )
        if width is None:
            width = int(vector.size)
        elif vector.size != width:
            raise V05ClassifierError("source prototype widths differ")
        result[labels[source_id]] = vector
    return MappingProxyType({key: result[key] for key in sorted(result)})


def summary_episode_features(bank: EpisodeBank | tuple[Any, Any]) -> np.ndarray:
    """Return the production Summary-LogReg per-episode mean/std features."""

    return _episode_summaries(_bank(bank))


def episode_balanced_moment_vector(
    bank: EpisodeBank | tuple[Any, Any],
) -> np.ndarray:
    """Return frozen v03-B3a equal-episode mean/std/second moments."""

    value = _bank(bank)
    weights = episode_balanced_weights(value.episode_offsets)
    mean = weights @ value.points
    second = weights @ np.square(value.points)
    standard_deviation = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    return np.concatenate((mean, standard_deviation, second))


@dataclass(frozen=True)
class SummaryPrototypeNN:
    """Nearest class prototype in the effective Summary-LogReg feature space."""

    prototypes: Mapping[str, np.ndarray]
    method_id: str = SUMMARY_NN

    def __post_init__(self) -> None:
        labels = {key: key for key in self.prototypes}
        object.__setattr__(
            self, "prototypes", _class_prototypes(self.prototypes, labels)
        )

    @classmethod
    def fit(
        cls,
        source_banks: Mapping[str, EpisodeBank | tuple[Any, Any]],
        source_labels: Mapping[str, str],
    ) -> "SummaryPrototypeNN":
        banks = _source_banks(source_banks)
        vectors = {
            source_id: np.mean(summary_episode_features(bank), axis=0)
            for source_id, bank in banks.items()
        }
        return cls(_class_prototypes(vectors, source_labels))

    @property
    def model_digest(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "arrays_sha256": sha256_ndarrays(dict(self.prototypes)),
                "aggregation": "equal mean of per-episode mean/std vectors",
            }
        )

    @property
    def model_nbytes(self) -> int:
        return sum(vector.nbytes for vector in self.prototypes.values())

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        query = np.mean(summary_episode_features(query_bank), axis=0)
        return self.score_vector(query)

    def score_vector(self, query_vector: Any) -> dict[str, float]:
        query = _readonly_numeric(query_vector, ndim=1, where="query summary")
        if any(
            prototype.shape != query.shape for prototype in self.prototypes.values()
        ):
            raise V05ClassifierError("query summary width differs from prototypes")
        return _finite_scores(
            {
                class_id: -float(np.linalg.norm(query - prototype))
                for class_id, prototype in self.prototypes.items()
            }
        )


@dataclass(frozen=True)
class RawMomentNN:
    """A frozen v03-B3a adaptation on the v0.5 delta/action view."""

    prototypes: Mapping[str, np.ndarray]
    method_id: str = B3A_RAW_MOMENT_NN

    def __post_init__(self) -> None:
        labels = {key: key for key in self.prototypes}
        object.__setattr__(
            self, "prototypes", _class_prototypes(self.prototypes, labels)
        )

    @classmethod
    def fit(
        cls,
        source_banks: Mapping[str, EpisodeBank | tuple[Any, Any]],
        source_labels: Mapping[str, str],
    ) -> "RawMomentNN":
        banks = _source_banks(source_banks)
        vectors = {
            source_id: episode_balanced_moment_vector(bank)
            for source_id, bank in banks.items()
        }
        return cls(_class_prototypes(vectors, source_labels))

    @property
    def model_digest(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "arrays_sha256": sha256_ndarrays(dict(self.prototypes)),
                "weighting": "1/(episode_count*rows_in_episode)",
                "statistics": ["mean", "std", "second_moment"],
            }
        )

    @property
    def model_nbytes(self) -> int:
        return sum(vector.nbytes for vector in self.prototypes.values())

    def score(self, query_bank: EpisodeBank | tuple[Any, Any]) -> dict[str, float]:
        query = episode_balanced_moment_vector(query_bank)
        return self.score_vector(query)

    def score_vector(self, query_vector: Any) -> dict[str, float]:
        query = _readonly_numeric(query_vector, ndim=1, where="query moments")
        if any(
            prototype.shape != query.shape for prototype in self.prototypes.values()
        ):
            raise V05ClassifierError("query moment width differs from prototypes")
        return _finite_scores(
            {
                class_id: -float(np.linalg.norm(query - prototype))
                for class_id, prototype in self.prototypes.items()
            }
        )


@dataclass(frozen=True)
class DeterministicRandomRanker:
    """Frozen v03-B0 hash ranking; a deterministic chance lower bound."""

    class_ids: tuple[str, ...]
    public_seed: int
    method_id: str = B0_RANDOM

    def __post_init__(self) -> None:
        classes = tuple(sorted(self.class_ids))
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or any(not isinstance(item, str) or not item for item in classes)
        ):
            raise V05ClassifierError("random baseline needs sorted unique class IDs")
        if (
            isinstance(self.public_seed, (bool, np.bool_))
            or not isinstance(self.public_seed, (int, np.integer))
            or int(self.public_seed) < 0
        ):
            raise V05ClassifierError(
                "random baseline seed must be a nonnegative integer"
            )
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "public_seed", int(self.public_seed))

    @property
    def model_digest(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "class_ids": list(self.class_ids),
                "public_seed": self.public_seed,
            }
        )

    @property
    def model_nbytes(self) -> int:
        return 0

    def score(self, *, public_query_token: str) -> dict[str, float]:
        token = public_query_token
        if not isinstance(token, str) or not token:
            raise V05ClassifierError("public query token must be non-empty")
        denominator = float(16**13)
        return {
            class_id: float(
                int(
                    sha256_json(
                        {
                            "schema": "policy-learnware.v05-b0-random-key.v1",
                            "public_seed": self.public_seed,
                            "public_query_token": token,
                            "candidate_token": class_id,
                        }
                    )[:13],
                    16,
                )
                / denominator
            )
            for class_id in self.class_ids
        }


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise V05ClassifierError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise V05ClassifierError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


@dataclass(frozen=True)
class RFFEpisodeFeatures:
    """Typed per-episode RFF means bound to one replayable public map."""

    rows: np.ndarray
    map_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rows",
            _readonly_numeric(self.rows, ndim=2, where="RFF episode features"),
        )
        object.__setattr__(
            self, "map_digest", _sha256(self.map_digest, "RFF map digest")
        )

    @classmethod
    def from_bank(
        cls, rff_map: RFFMap, bank: EpisodeBank | tuple[Any, Any]
    ) -> "RFFEpisodeFeatures":
        if not isinstance(rff_map, RFFMap):
            raise V05ClassifierError("RFF episode features require an RFFMap")
        value = _bank(bank)
        return cls(
            rows=rff_map.episode_means(value.points, value.episode_offsets),
            map_digest=rff_map.map_digest,
        )

    @property
    def feature_digest(self) -> str:
        return sha256_json(
            {
                "map_digest": self.map_digest,
                "rows_sha256": sha256_ndarrays({"rows": self.rows}),
            }
        )


def _feature_rows(
    values: Mapping[str, Any], *, where: str
) -> tuple[dict[str, np.ndarray], str]:
    if not isinstance(values, Mapping) or not values:
        raise V05ClassifierError(f"{where} must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    width: int | None = None
    map_digest: str | None = None
    for source_id in sorted(values):
        if not isinstance(source_id, str) or not source_id:
            raise V05ClassifierError(f"{where} source IDs must be non-empty")
        feature = values[source_id]
        if not isinstance(feature, RFFEpisodeFeatures):
            raise V05ClassifierError(f"{where} must contain RFFEpisodeFeatures")
        rows = feature.rows
        if width is None:
            width = int(rows.shape[1])
        elif rows.shape[1] != width:
            raise V05ClassifierError(f"{where} feature widths differ")
        if map_digest is None:
            map_digest = feature.map_digest
        elif feature.map_digest != map_digest:
            raise V05ClassifierError(f"{where} RFF map digests differ")
        result[source_id] = rows
    assert map_digest is not None
    return result, map_digest


def _supervised_feature_matrix(
    rows: Mapping[str, np.ndarray], labels: Mapping[str, str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    features: list[np.ndarray] = []
    targets: list[str] = []
    for source_id, source_rows in rows.items():
        features.append(source_rows)
        targets.extend([labels[source_id]] * source_rows.shape[0])
    return np.concatenate(features, axis=0), tuple(targets)


@dataclass(frozen=True)
class FixedFeatureLogReg:
    """Multinomial logistic head on frozen per-episode feature rows."""

    class_ids: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    selected_l2: float
    training_iterations: int
    feature_map_digest: str
    method_id: str = RFF_LOGREG

    def __post_init__(self) -> None:
        classes = tuple(sorted(self.class_ids))
        mean = _readonly_numeric(self.feature_mean, ndim=1, where="feature_mean")
        scale = _readonly_numeric(self.feature_scale, ndim=1, where="feature_scale")
        weights = _readonly_numeric(self.weights, ndim=2, where="weights")
        intercept = _readonly_numeric(self.intercept, ndim=1, where="intercept")
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or any(not isinstance(item, str) or not item for item in classes)
            or mean.shape != scale.shape
            or weights.shape != (mean.size, len(classes))
            or intercept.shape != (len(classes),)
            or np.any(scale <= 0.0)
        ):
            raise V05ClassifierError("fixed-feature LogReg arrays are incompatible")
        if (
            isinstance(self.training_iterations, (bool, np.bool_))
            or not isinstance(self.training_iterations, (int, np.integer))
            or int(self.training_iterations) <= 0
        ):
            raise V05ClassifierError("training_iterations must be positive")
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(
            self,
            "feature_map_digest",
            _sha256(self.feature_map_digest, "feature_map_digest"),
        )
        object.__setattr__(
            self,
            "selected_l2",
            _positive_float(self.selected_l2, "selected_l2", allow_zero=True),
        )
        object.__setattr__(self, "training_iterations", int(self.training_iterations))

    @classmethod
    def fit(
        cls,
        source_train_features: Mapping[str, Any],
        source_labels: Mapping[str, str],
        source_validation_features: Mapping[str, Any],
        *,
        l2_grid: Sequence[float] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0),
        max_iter: int = 3_000,
        tolerance: float = 1.0e-9,
    ) -> "FixedFeatureLogReg":
        train, train_map_digest = _feature_rows(
            source_train_features, where="train features"
        )
        validation, validation_map_digest = _feature_rows(
            source_validation_features, where="validation features"
        )
        if set(train) != set(validation) or train_map_digest != validation_map_digest:
            raise V05ClassifierError("train/validation feature sources or maps differ")
        if (
            isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer))
            or int(max_iter) <= 0
        ):
            raise V05ClassifierError("max_iter must be positive")
        tolerance = _positive_float(tolerance, "tolerance")
        labels = _labels_for_sources(train, source_labels)
        classes = tuple(sorted(set(labels.values())))
        class_index = {label: index for index, label in enumerate(classes)}
        x_train, train_ids = _supervised_feature_matrix(train, labels)
        x_validation, validation_ids = _supervised_feature_matrix(validation, labels)
        feature_mean = np.mean(x_train, axis=0)
        raw_scale = np.std(x_train, axis=0)
        feature_scale = np.where(raw_scale > 1.0e-12, raw_scale, 1.0)
        train_z = (x_train - feature_mean) / feature_scale
        validation_z = (x_validation - feature_mean) / feature_scale
        y_train = np.asarray([class_index[item] for item in train_ids], dtype=np.int64)
        y_validation = np.asarray(
            [class_index[item] for item in validation_ids], dtype=np.int64
        )
        grid = tuple(
            sorted(
                {
                    _positive_float(item, "l2_grid item", allow_zero=True)
                    for item in l2_grid
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
            loss = _cross_entropy(validation_z @ weights + intercept, y_validation)
            candidates.append((loss, l2, weights, intercept, iterations))
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
            feature_map_digest=train_map_digest,
        )

    @property
    def model_digest(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "class_ids": list(self.class_ids),
                "feature_map_digest": self.feature_map_digest,
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

    @property
    def model_nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.feature_mean,
                self.feature_scale,
                self.weights,
                self.intercept,
            )
        )

    def score_features(self, query_features: RFFEpisodeFeatures) -> dict[str, float]:
        if not isinstance(query_features, RFFEpisodeFeatures):
            raise V05ClassifierError("RFF LogReg requires typed query features")
        if query_features.map_digest != self.feature_map_digest:
            raise V05ClassifierError("query/LogReg RFF map digests differ")
        features = query_features.rows
        if features.shape[1] != self.feature_mean.size:
            raise V05ClassifierError("query feature width differs from LogReg fit")
        standardized = (features - self.feature_mean) / self.feature_scale
        logits = np.mean(standardized @ self.weights + self.intercept, axis=0)
        return _finite_scores(dict(zip(self.class_ids, logits, strict=True)))


@dataclass(frozen=True)
class FixedFeatureRidge:
    """Dual multiclass ridge on fixed per-episode feature vectors."""

    class_ids: tuple[str, ...]
    training_features: np.ndarray
    alpha: np.ndarray
    selected_ridge: float
    feature_map_digest: str
    method_id: str = RFF_RIDGE

    def __post_init__(self) -> None:
        classes = tuple(sorted(self.class_ids))
        features = _readonly_numeric(
            self.training_features, ndim=2, where="training_features"
        )
        alpha = _readonly_numeric(self.alpha, ndim=2, where="alpha")
        if (
            len(classes) < 2
            or len(set(classes)) != len(classes)
            or any(not isinstance(item, str) or not item for item in classes)
            or alpha.shape != (features.shape[0], len(classes))
        ):
            raise V05ClassifierError("fixed-feature Ridge arrays are incompatible")
        object.__setattr__(self, "class_ids", classes)
        object.__setattr__(self, "training_features", features)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(
            self,
            "feature_map_digest",
            _sha256(self.feature_map_digest, "feature_map_digest"),
        )
        object.__setattr__(
            self,
            "selected_ridge",
            _positive_float(self.selected_ridge, "selected_ridge"),
        )

    @classmethod
    def fit(
        cls,
        source_train_features: Mapping[str, Any],
        source_labels: Mapping[str, str],
        source_validation_features: Mapping[str, Any],
        *,
        ridge_grid: Sequence[float] = (1.0e-6, 1.0e-4, 1.0e-2, 1.0),
    ) -> "FixedFeatureRidge":
        train, train_map_digest = _feature_rows(
            source_train_features, where="train features"
        )
        validation, validation_map_digest = _feature_rows(
            source_validation_features, where="validation features"
        )
        if set(train) != set(validation) or train_map_digest != validation_map_digest:
            raise V05ClassifierError("train/validation feature sources or maps differ")
        labels = _labels_for_sources(train, source_labels)
        classes = tuple(sorted(set(labels.values())))
        class_index = {label: index for index, label in enumerate(classes)}
        x_train, train_ids = _supervised_feature_matrix(train, labels)
        x_validation, validation_ids = _supervised_feature_matrix(validation, labels)
        y_train = np.asarray([class_index[item] for item in train_ids], dtype=np.int64)
        y_validation = np.asarray(
            [class_index[item] for item in validation_ids], dtype=np.int64
        )
        target = np.eye(len(classes), dtype=np.float64)[y_train]
        validation_target = np.eye(len(classes), dtype=np.float64)[y_validation]
        gram = x_train @ x_train.T
        gram = 0.5 * (gram + gram.T)
        validation_gram = x_validation @ x_train.T
        identity = np.eye(gram.shape[0], dtype=np.float64)
        grid = tuple(
            sorted({_positive_float(item, "ridge_grid item") for item in ridge_grid})
        )
        if not grid:
            raise V05ClassifierError("ridge_grid must not be empty")
        candidates = []
        for ridge in grid:
            try:
                alpha = np.linalg.solve(gram + ridge * identity, target)
            except np.linalg.LinAlgError:
                continue
            loss = float(
                np.mean(np.square(validation_gram @ alpha - validation_target))
            )
            candidates.append((loss, ridge, alpha))
        if not candidates:
            raise V05ClassifierError("all fixed-feature Ridge solves failed")
        _, selected_ridge, alpha = min(candidates, key=lambda item: (item[0], item[1]))
        return cls(
            class_ids=classes,
            training_features=x_train,
            alpha=alpha,
            selected_ridge=selected_ridge,
            feature_map_digest=train_map_digest,
        )

    @property
    def model_digest(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "class_ids": list(self.class_ids),
                "feature_map_digest": self.feature_map_digest,
                "selected_ridge": self.selected_ridge,
                "arrays_sha256": sha256_ndarrays(
                    {
                        "training_features": self.training_features,
                        "alpha": self.alpha,
                    }
                ),
            }
        )

    @property
    def model_nbytes(self) -> int:
        return self.training_features.nbytes + self.alpha.nbytes

    def score_features(self, query_features: RFFEpisodeFeatures) -> dict[str, float]:
        if not isinstance(query_features, RFFEpisodeFeatures):
            raise V05ClassifierError("RFF Ridge requires typed query features")
        if query_features.map_digest != self.feature_map_digest:
            raise V05ClassifierError("query/Ridge RFF map digests differ")
        features = query_features.rows
        if features.shape[1] != self.training_features.shape[1]:
            raise V05ClassifierError("query feature width differs from Ridge fit")
        scores = np.mean(features @ self.training_features.T @ self.alpha, axis=0)
        return _finite_scores(dict(zip(self.class_ids, scores, strict=True)))


@dataclass(frozen=True)
class SWE1024NN(SWENN):
    """The frozen 32-direction x 32-quantile SWE dimension control."""

    method_id: str = SWE_1024_NN

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.swe_map.direction_count != 32
            or self.swe_map.quantile_count != 32
            or self.swe_map.output_dim != 1024
        ):
            raise V05ClassifierError("SWE_1024_NN requires the frozen 32x32 map")


def ablation_method_cards() -> tuple[dict[str, Any], ...]:
    """Describe the six additions without changing the frozen P0 registry."""

    return (
        {
            "method_id": B0_RANDOM,
            "role": "chance_lower_bound",
            "source_fit": "none",
            "score": "v03-style frozen SHA256 random key",
        },
        {
            "method_id": B3A_RAW_MOMENT_NN,
            "role": "v03_trivial_baseline",
            "source_fit": "per-anchor episode-balanced mean/std/second moment",
            "score": "negative Euclidean distance",
        },
        {
            "method_id": SUMMARY_NN,
            "role": "decision_rule_and_preprocessing_ablation",
            "source_fit": "mean of per-episode mean/std prototypes",
            "score": "negative Euclidean distance",
        },
        {
            "method_id": RFF_LOGREG,
            "role": "decision_rule_and_preprocessing_ablation",
            "source_fit": "multinomial LogReg on frozen RFF episode means",
            "score": "mean episode logit",
        },
        {
            "method_id": RFF_RIDGE,
            "role": "decision_rule_and_preprocessing_ablation",
            "source_fit": "dual ridge on frozen RFF episode means",
            "score": "mean episode class score",
        },
        {
            "method_id": SWE_1024_NN,
            "role": "dimension_matched_representation_ablation",
            "source_fit": "32 directions x 32 quantiles",
            "score": "negative Euclidean distance",
        },
    )


__all__ = [
    "ABLATION_METHOD_IDS",
    "B0_RANDOM",
    "B3A_RAW_MOMENT_NN",
    "RFF_LOGREG",
    "RFF_RIDGE",
    "SUMMARY_NN",
    "SWE_1024_NN",
    "DeterministicRandomRanker",
    "FixedFeatureLogReg",
    "FixedFeatureRidge",
    "RawMomentNN",
    "RFFEpisodeFeatures",
    "SWE1024NN",
    "SummaryPrototypeNN",
    "ablation_method_cards",
    "episode_balanced_moment_vector",
    "nested_row_order",
    "summary_episode_features",
]
