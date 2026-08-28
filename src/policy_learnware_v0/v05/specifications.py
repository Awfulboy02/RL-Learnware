"""Fixed-vector environment specifications for the v0.5 matched panel.

The module deliberately starts from already canonicalized, reward-free
``(delta observation, action)`` points.  Environment collection, policy
execution, labels, rewards, and target paths are outside this interface.

Two public, data-independent maps are provided:

* a cosine/sine Random Fourier Feature mean for the frozen Gaussian kernel;
* a fixed-direction Sliced-Wasserstein quantile sketch.

Both maps preserve equal episode mass even if episode lengths differ.  The
current fixed probe happens to expose 64 transitions per episode, but making
the weighting explicit prevents a later loader from silently switching the
estimand to transition-weighted pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from ..io import atomic_write_npz
from ..rkme.empirical import episode_balanced_weights


RFF_METHOD_ID = "RFF_KME_NN"
SWE_METHOD_ID = "SWE_NN"

RFF_MAP_SCHEMA = "policy-learnware.v05-rff-map.v1"
RFF_SPEC_SCHEMA = "policy-learnware.v05-rff-spec.v1"
SWE_MAP_SCHEMA = "policy-learnware.v05-swe-map.v1"
SWE_SPEC_SCHEMA = "policy-learnware.v05-swe-spec.v1"


class V05SpecificationError(ValueError):
    """A fixed-vector map or specification violates its frozen contract."""


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise V05SpecificationError(f"{where} must be an integer")
    result = int(value)
    if result <= 0:
        raise V05SpecificationError(f"{where} must be positive")
    return result


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise V05SpecificationError(f"{where} must be an integer")
    result = int(value)
    if result < 0:
        raise V05SpecificationError(f"{where} must be nonnegative")
    return result


def _positive_float(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V05SpecificationError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise V05SpecificationError(f"{where} must be finite and positive")
    return result


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise V05SpecificationError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise V05SpecificationError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return value


def _points(value: Any, *, input_dim: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[0] == 0:
        raise V05SpecificationError("points must be a non-empty numeric [T,Q] matrix")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if input_dim is not None and result.shape[1] != input_dim:
        raise V05SpecificationError(
            f"point width {result.shape[1]} differs from frozen input_dim {input_dim}"
        )
    if not np.all(np.isfinite(result)):
        raise V05SpecificationError("points contain non-finite values")
    return result


def validate_episode_offsets(
    episode_offsets: Any, *, transition_count: int
) -> np.ndarray:
    raw = np.asarray(episode_offsets)
    if raw.dtype.kind not in "iu" or raw.ndim != 1:
        raise V05SpecificationError(
            "episode_offsets must be a one-dimensional integer array"
        )
    offsets = np.ascontiguousarray(raw, dtype=np.int64)
    if (
        offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != transition_count
        or np.any(np.diff(offsets) <= 0)
    ):
        raise V05SpecificationError(
            "episode_offsets must be a non-empty partition of the point rows"
        )
    return offsets


def episode_slices(episode_offsets: Any, *, transition_count: int) -> tuple[slice, ...]:
    offsets = validate_episode_offsets(
        episode_offsets, transition_count=transition_count
    )
    return tuple(
        slice(int(offsets[index]), int(offsets[index + 1]))
        for index in range(offsets.size - 1)
    )


def _pcg64_standard_normal(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """Use the explicitly named frozen NumPy bit generator, not a global RNG."""

    generator = np.random.Generator(np.random.PCG64(seed))
    return np.asarray(generator.standard_normal(shape), dtype=np.float64)


@dataclass(frozen=True)
class RFFMap:
    """Public cosine/sine features for ``exp(-||x-y||²/(2 sigma²))``.

    ``frequency_count`` frequencies produce a ``2 * frequency_count`` vector.
    Frequencies are sampled from ``N(0, sigma^-2 I)``.  Consequently each
    point feature has Euclidean norm exactly one up to floating-point error,
    and the inner product is the usual unbiased Monte-Carlo kernel estimate.
    """

    input_dim: int
    bandwidth: float
    normalization_digest: str
    frequency_count: int = 512
    public_seed: int = 50_501
    dtype: str = "float64"
    frequencies: np.ndarray | None = None
    schema: str = RFF_MAP_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RFF_MAP_SCHEMA:
            raise V05SpecificationError("unsupported RFF map schema")
        input_dim = _positive_int(self.input_dim, "input_dim")
        bandwidth = _positive_float(self.bandwidth, "bandwidth")
        frequency_count = _positive_int(self.frequency_count, "frequency_count")
        seed = _nonnegative_int(self.public_seed, "public_seed")
        normalization_digest = _sha256(
            self.normalization_digest, "normalization_digest"
        )
        if self.dtype != "float64":
            raise V05SpecificationError("the v0.5 RFF map freezes dtype=float64")
        expected = (
            _pcg64_standard_normal(seed, (frequency_count, input_dim)) / bandwidth
        )
        if self.frequencies is None:
            frequencies = expected
        else:
            frequencies = np.asarray(self.frequencies, dtype=np.float64)
            if frequencies.shape != expected.shape or not np.array_equal(
                frequencies, expected
            ):
                raise V05SpecificationError(
                    "persisted RFF frequencies do not replay from the public seed"
                )
        frequencies = np.ascontiguousarray(frequencies).copy()
        frequencies.setflags(write=False)
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "bandwidth", bandwidth)
        object.__setattr__(self, "frequency_count", frequency_count)
        object.__setattr__(self, "public_seed", seed)
        object.__setattr__(self, "normalization_digest", normalization_digest)
        object.__setattr__(self, "frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        return 2 * self.frequency_count

    @property
    def map_digest(self) -> str:
        return sha256_json(
            {
                **self.to_dict(include_digest=False),
                "frequencies_sha256": sha256_ndarrays(
                    {"frequencies": self.frequencies}
                ),
            }
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "method_id": RFF_METHOD_ID,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "bandwidth": self.bandwidth,
            "normalization_digest": self.normalization_digest,
            "frequency_count": self.frequency_count,
            "public_seed": self.public_seed,
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "dtype": self.dtype,
            "feature_formula": "sqrt(1/M)*concat(cos(Wx),sin(Wx))",
            "frequency_law": "iid N(0, bandwidth^-2 I)",
            "frequency_count_semantics": "M frequencies produce 2M coordinates",
            "point_feature_norm": "one",
            "aggregation": "per-episode feature mean; equal episode mean",
        }
        if include_digest:
            payload["map_digest"] = self.map_digest
        return payload

    def transform_points(self, points: Any) -> np.ndarray:
        x = _points(points, input_dim=self.input_dim)
        phase = x @ self.frequencies.T
        scale = 1.0 / math.sqrt(self.frequency_count)
        result = scale * np.concatenate((np.cos(phase), np.sin(phase)), axis=1)
        return np.asarray(result, dtype=np.float64)

    def episode_means(self, points: Any, episode_offsets: Any) -> np.ndarray:
        x = _points(points, input_dim=self.input_dim)
        slices = episode_slices(episode_offsets, transition_count=x.shape[0])
        features = self.transform_points(x)
        return np.stack(
            [np.mean(features[episode_slice], axis=0) for episode_slice in slices]
        )

    def embed(self, points: Any, episode_offsets: Any) -> "RFFSpecification":
        means = self.episode_means(points, episode_offsets)
        return RFFSpecification(
            vector=np.mean(means, axis=0), map_digest=self.map_digest
        )

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        return atomic_write_npz(
            path,
            {
                "input_dim": np.asarray(self.input_dim, dtype=np.int64),
                "bandwidth": np.asarray(self.bandwidth, dtype=np.float64),
                "normalization_digest": np.asarray(self.normalization_digest),
                "frequency_count": np.asarray(self.frequency_count, dtype=np.int64),
                "public_seed": np.asarray(self.public_seed, dtype=np.int64),
                "dtype": np.asarray(self.dtype),
                "frequencies": self.frequencies,
                "schema": np.asarray(self.schema),
                "map_digest": np.asarray(self.map_digest),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "RFFMap":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                input_dim=int(data["input_dim"]),
                bandwidth=float(data["bandwidth"]),
                normalization_digest=str(data["normalization_digest"]),
                frequency_count=int(data["frequency_count"]),
                public_seed=int(data["public_seed"]),
                dtype=str(data["dtype"]),
                frequencies=data["frequencies"],
                schema=str(data["schema"]),
            )
            if str(data["map_digest"]) != result.map_digest:
                raise V05SpecificationError("persisted RFF map digest does not match")
            return result


@dataclass(frozen=True)
class RFFSpecification:
    vector: np.ndarray
    map_digest: str
    schema: str = RFF_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RFF_SPEC_SCHEMA:
            raise V05SpecificationError("unsupported RFF specification schema")
        raw = np.asarray(self.vector)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
            raise V05SpecificationError("RFF specification must be a numeric vector")
        vector = np.ascontiguousarray(raw, dtype=np.float64).copy()
        if not np.all(np.isfinite(vector)):
            raise V05SpecificationError("RFF specification contains non-finite values")
        tolerance = 64.0 * np.finfo(vector.dtype).eps
        if np.linalg.norm(vector) > 1.0 + tolerance:
            raise V05SpecificationError("RFF mean exceeds its unit norm bound")
        object.__setattr__(self, "map_digest", _sha256(self.map_digest, "map_digest"))
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)

    @property
    def specification_digest(self) -> str:
        return sha256_json(
            {
                "schema": self.schema,
                "map_digest": self.map_digest,
                "vector_sha256": sha256_ndarrays({"vector": self.vector}),
            }
        )

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        return atomic_write_npz(
            path,
            {
                "vector": self.vector,
                "map_digest": np.asarray(self.map_digest),
                "schema": np.asarray(self.schema),
                "specification_digest": np.asarray(self.specification_digest),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "RFFSpecification":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                vector=data["vector"],
                map_digest=str(data["map_digest"]),
                schema=str(data["schema"]),
            )
            if str(data["specification_digest"]) != result.specification_digest:
                raise V05SpecificationError(
                    "persisted RFF specification digest differs"
                )
            return result


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantile_grid: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a weighted quantile curve at fixed public points.

    Repeated projected values are first coalesced into one atom.  Without this
    step, unequal atom weights could make the interpolation knots depend on the
    input order inside an exact-value tie, violating measure/permutation
    invariance.
    """

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    if np.any(sorted_weights < 0.0) or not np.isclose(
        np.sum(sorted_weights), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise V05SpecificationError(
            "quantile weights must be nonnegative and sum to one"
        )
    positive = sorted_weights > 0.0
    sorted_values = sorted_values[positive]
    sorted_weights = sorted_weights[positive]
    unique_values, first_indices = np.unique(sorted_values, return_index=True)
    if unique_values.size != sorted_values.size:
        sorted_weights = np.add.reduceat(sorted_weights, first_indices)
        sorted_values = unique_values
    midpoints = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return np.interp(
        quantile_grid,
        midpoints,
        sorted_values,
        left=float(sorted_values[0]),
        right=float(sorted_values[-1]),
    )


@dataclass(frozen=True)
class SWEMap:
    """Fixed random slices with a public inverse-CDF interpolation grid.

    The vector is scaled by ``1/sqrt(L*Q)``.  Squared Euclidean distance is a
    rectangular quadrature over frozen, linearly interpolated projected
    quantile curves.  This is a PSWE/SLoSH-inspired retrieval adaptation that
    approximates SW2 geometry; it is not the exact step-CDF empirical SW2
    integral from Meunier et al.'s Appendix D.
    """

    input_dim: int
    normalization_digest: str
    direction_count: int = 64
    quantile_count: int = 64
    public_seed: int = 50_502
    dtype: str = "float64"
    directions: np.ndarray | None = None
    quantile_grid: np.ndarray | None = None
    schema: str = SWE_MAP_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SWE_MAP_SCHEMA:
            raise V05SpecificationError("unsupported SWE map schema")
        input_dim = _positive_int(self.input_dim, "input_dim")
        direction_count = _positive_int(self.direction_count, "direction_count")
        quantile_count = _positive_int(self.quantile_count, "quantile_count")
        seed = _nonnegative_int(self.public_seed, "public_seed")
        normalization_digest = _sha256(
            self.normalization_digest, "normalization_digest"
        )
        if self.dtype != "float64":
            raise V05SpecificationError("the v0.5 SWE map freezes dtype=float64")
        raw_directions = _pcg64_standard_normal(seed, (direction_count, input_dim))
        norms = np.linalg.norm(raw_directions, axis=1)
        if np.any(norms == 0.0):  # practically unreachable, but fail closed
            raise V05SpecificationError("public SWE direction draw contains a zero row")
        expected_directions = raw_directions / norms[:, None]
        expected_grid = (
            np.arange(quantile_count, dtype=np.float64) + 0.5
        ) / quantile_count
        if self.directions is None:
            directions = expected_directions
        else:
            directions = np.asarray(self.directions, dtype=np.float64)
            if directions.shape != expected_directions.shape or not np.array_equal(
                directions, expected_directions
            ):
                raise V05SpecificationError(
                    "persisted SWE directions do not replay from the public seed"
                )
        if self.quantile_grid is None:
            grid = expected_grid
        else:
            grid = np.asarray(self.quantile_grid, dtype=np.float64)
            if grid.shape != expected_grid.shape or not np.array_equal(
                grid, expected_grid
            ):
                raise V05SpecificationError(
                    "persisted SWE quantile grid differs from the frozen midpoint grid"
                )
        directions = np.ascontiguousarray(directions).copy()
        grid = np.ascontiguousarray(grid).copy()
        directions.setflags(write=False)
        grid.setflags(write=False)
        object.__setattr__(self, "input_dim", input_dim)
        object.__setattr__(self, "direction_count", direction_count)
        object.__setattr__(self, "quantile_count", quantile_count)
        object.__setattr__(self, "public_seed", seed)
        object.__setattr__(self, "normalization_digest", normalization_digest)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "quantile_grid", grid)

    @property
    def output_dim(self) -> int:
        return self.direction_count * self.quantile_count

    @property
    def map_digest(self) -> str:
        return sha256_json(
            {
                **self.to_dict(include_digest=False),
                "public_arrays_sha256": sha256_ndarrays(
                    {
                        "directions": self.directions,
                        "quantile_grid": self.quantile_grid,
                    }
                ),
            }
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "method_id": SWE_METHOD_ID,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "normalization_digest": self.normalization_digest,
            "direction_count": self.direction_count,
            "quantile_count": self.quantile_count,
            "public_seed": self.public_seed,
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "dtype": self.dtype,
            "direction_law": "iid Gaussian rows normalized to the unit sphere",
            "quantile_grid": "midpoints (j+0.5)/Q",
            "interpolation": (
                "coalesce duplicate atoms; cumulative-mass midpoint knots; "
                "linear interpolation with constant endpoints"
            ),
            "normalization": "1/sqrt(direction_count*quantile_count)",
            "aggregation": "equal-episode-mass empirical distribution",
            "flatten_order": "direction-major then quantile-grid",
            "adaptation_status": (
                "reference-free fixed-projection retrieval adaptation; "
                "linearly interpolated quantile curves"
            ),
        }
        if include_digest:
            payload["map_digest"] = self.map_digest
        return payload

    def embed(self, points: Any, episode_offsets: Any) -> "SWESpecification":
        x = _points(points, input_dim=self.input_dim)
        offsets = validate_episode_offsets(episode_offsets, transition_count=x.shape[0])
        weights = episode_balanced_weights(offsets)
        projected = x @ self.directions.T
        quantiles = np.empty(
            (self.direction_count, self.quantile_count), dtype=np.float64
        )
        for direction_index in range(self.direction_count):
            quantiles[direction_index] = _weighted_quantiles(
                projected[:, direction_index], weights, self.quantile_grid
            )
        vector = quantiles.reshape(-1) / math.sqrt(self.output_dim)
        return SWESpecification(vector=vector, map_digest=self.map_digest)

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        return atomic_write_npz(
            path,
            {
                "input_dim": np.asarray(self.input_dim, dtype=np.int64),
                "normalization_digest": np.asarray(self.normalization_digest),
                "direction_count": np.asarray(self.direction_count, dtype=np.int64),
                "quantile_count": np.asarray(self.quantile_count, dtype=np.int64),
                "public_seed": np.asarray(self.public_seed, dtype=np.int64),
                "dtype": np.asarray(self.dtype),
                "directions": self.directions,
                "quantile_grid": self.quantile_grid,
                "schema": np.asarray(self.schema),
                "map_digest": np.asarray(self.map_digest),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SWEMap":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                input_dim=int(data["input_dim"]),
                normalization_digest=str(data["normalization_digest"]),
                direction_count=int(data["direction_count"]),
                quantile_count=int(data["quantile_count"]),
                public_seed=int(data["public_seed"]),
                dtype=str(data["dtype"]),
                directions=data["directions"],
                quantile_grid=data["quantile_grid"],
                schema=str(data["schema"]),
            )
            if str(data["map_digest"]) != result.map_digest:
                raise V05SpecificationError("persisted SWE map digest does not match")
            return result


@dataclass(frozen=True)
class SWESpecification:
    vector: np.ndarray
    map_digest: str
    schema: str = SWE_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SWE_SPEC_SCHEMA:
            raise V05SpecificationError("unsupported SWE specification schema")
        raw = np.asarray(self.vector)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
            raise V05SpecificationError("SWE specification must be a numeric vector")
        vector = np.ascontiguousarray(raw, dtype=np.float64).copy()
        if not np.all(np.isfinite(vector)):
            raise V05SpecificationError("SWE specification contains non-finite values")
        object.__setattr__(self, "map_digest", _sha256(self.map_digest, "map_digest"))
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)

    @property
    def specification_digest(self) -> str:
        return sha256_json(
            {
                "schema": self.schema,
                "map_digest": self.map_digest,
                "vector_sha256": sha256_ndarrays({"vector": self.vector}),
            }
        )

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> str:
        return atomic_write_npz(
            path,
            {
                "vector": self.vector,
                "map_digest": np.asarray(self.map_digest),
                "schema": np.asarray(self.schema),
                "specification_digest": np.asarray(self.specification_digest),
            },
            overwrite=overwrite,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "SWESpecification":
        with np.load(Path(path), allow_pickle=False) as data:
            result = cls(
                vector=data["vector"],
                map_digest=str(data["map_digest"]),
                schema=str(data["schema"]),
            )
            if str(data["specification_digest"]) != result.specification_digest:
                raise V05SpecificationError(
                    "persisted SWE specification digest differs"
                )
            return result


def squared_vector_distance(left: Any, right: Any) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if (
        x.ndim != 1
        or y.shape != x.shape
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
    ):
        raise V05SpecificationError("fixed-vector specifications are incompatible")
    return float(np.dot(x - y, x - y))


__all__ = [
    "RFF_METHOD_ID",
    "SWE_METHOD_ID",
    "RFFMap",
    "RFFSpecification",
    "SWEMap",
    "SWESpecification",
    "V05SpecificationError",
    "episode_slices",
    "squared_vector_distance",
    "validate_episode_offsets",
]
