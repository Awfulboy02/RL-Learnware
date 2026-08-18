"""The single global Gaussian kernel used by the v0 protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def pairwise_squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("kernel inputs must have shapes [N,Q] and [M,Q]")
    distance = (
        np.sum(np.square(x), axis=1)[:, None]
        + np.sum(np.square(y), axis=1)[None, :]
        - 2.0 * (x @ y.T)
    )
    # Round-off can produce tiny negative values even for identical inputs.
    return np.maximum(distance, 0.0)


@dataclass(frozen=True)
class GaussianKernel:
    bandwidth: float

    def __post_init__(self) -> None:
        bandwidth = float(self.bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Gaussian bandwidth must be finite and positive")
        object.__setattr__(self, "bandwidth", bandwidth)

    def gram(self, left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(left)
        y = x if right is None else np.asarray(right)
        squared = pairwise_squared_distance(x, y)
        gram = np.exp(-squared / (2.0 * self.bandwidth * self.bandwidth))
        if right is None:
            # Preserve exact kernel invariants for diagnostics and tests.
            gram = 0.5 * (gram + gram.T)
            np.fill_diagonal(gram, 1.0)
        return gram

    __call__ = gram

    def save_json(self, path: str | Path, *, overwrite: bool = False) -> str:
        from ..io import atomic_write_json

        return atomic_write_json(
            path,
            {"type": "gaussian", "bandwidth": self.bandwidth},
            overwrite=overwrite,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "GaussianKernel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "gaussian":
            raise ValueError("kernel artifact is not Gaussian")
        return cls(float(payload["bandwidth"]))


def _semantic_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    for attribute in ("points", "semantic_events", "embeddings", "z"):
        if hasattr(dataset, attribute):
            points = np.asarray(getattr(dataset, attribute), dtype=np.float64)
            break
    else:
        if isinstance(dataset, np.ndarray):
            points = np.asarray(dataset, dtype=np.float64)
        else:
            raise TypeError("semantic dataset must expose points/embeddings and offsets")
    offsets = getattr(dataset, "episode_offsets", None)
    if offsets is None:
        offsets = np.asarray([0, points.shape[0]], dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("semantic points must have non-empty shape [T,Q]")
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != points.shape[0]
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("invalid semantic episode_offsets")
    return points, offsets


def calibrate_bandwidth(
    semantic_events: Mapping[str, Any],
    config: Any | None = None,
    *,
    calibration_pairs: int | None = None,
    seed: int | None = None,
) -> float:
    """Estimate the one global bandwidth by source-balanced median distance.

    Each endpoint is sampled through ``task -> episode -> transition`` with a
    uniform choice at every level.  This prevents long episodes or tasks with
    more transitions from dominating calibration.
    """

    if not semantic_events:
        raise ValueError("at least one source task is required")
    if calibration_pairs is None:
        raw_pairs = (
            config.get("calibration_pairs", 10_000)
            if isinstance(config, Mapping)
            else getattr(config, "calibration_pairs", 10_000)
        )
        calibration_pairs = int(raw_pairs)
    if seed is None:
        raw_seed = (
            config.get("seed", 0)
            if isinstance(config, Mapping)
            else getattr(config, "seed", 0)
        )
        seed = int(raw_seed)
    if calibration_pairs <= 0:
        raise ValueError("calibration_pairs must be positive")

    tasks = tuple(sorted(semantic_events))
    prepared = {task: _semantic_arrays(semantic_events[task]) for task in tasks}
    rng = np.random.default_rng(seed)

    def sample_point() -> np.ndarray:
        task = tasks[int(rng.integers(0, len(tasks)))]
        points, offsets = prepared[task]
        episode = int(rng.integers(0, offsets.size - 1))
        transition = int(rng.integers(offsets[episode], offsets[episode + 1]))
        return points[transition]

    distances = np.empty(calibration_pairs, dtype=np.float64)
    for pair_index in range(calibration_pairs):
        left = sample_point()
        right = sample_point()
        distances[pair_index] = np.linalg.norm(left - right)
    positive = distances[distances > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise ValueError("bandwidth calibration has no positive pairwise distances")
    bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("calibrated Gaussian bandwidth is not positive")
    return bandwidth
