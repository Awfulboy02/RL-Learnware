"""Nearest-RKME selection over the public pool only."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from ..pool.learnware import LearnwarePool, SelectorTaskSpec
from ..rkme.empirical import (
    _has_exact_norm2_attestation,
    blockwise_weighted_self_kernel_sum_auto,
    episode_balanced_weights,
)
from ..rkme.gaussian import GaussianKernel


class SelectorError(ValueError):
    """The target KME is incompatible with the frozen pool protocol."""


def _field(value: Any, name: str, *aliases: str) -> Any:
    names = (name, *aliases)
    if isinstance(value, Mapping):
        for candidate in names:
            if candidate in value:
                return value[candidate]
    else:
        for candidate in names:
            if hasattr(value, candidate):
                return getattr(value, candidate)
    raise SelectorError(f"target empirical KME has no field among {names}")


def _optional_field(value: Any, *names: str) -> Any | None:
    try:
        return _field(value, names[0], *names[1:])
    except SelectorError:
        return None


def _gaussian(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    squared = np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)
    return np.exp(-squared / (2.0 * bandwidth * bandwidth))


def target_source_cross_terms(
    semantic_events: np.ndarray,
    weights: np.ndarray,
    pool: LearnwarePool,
    *,
    block_size: int = 2048,
) -> dict[str, float]:
    """Compute target-to-public-support kernel terms without a full Gram.

    This is linear in the number of target transitions and is reusable across
    nested target prefixes.  It contains no source labels or policy metadata.
    """

    events = np.asarray(semantic_events, dtype=np.float64)
    target_weights = np.asarray(weights, dtype=np.float64)
    if (
        events.ndim != 2
        or target_weights.shape != (events.shape[0],)
        or events.shape[0] == 0
        or block_size <= 0
    ):
        raise SelectorError("target cross-term arrays/block_size are invalid")
    terms = {entry.opaque_id: 0.0 for entry in pool.entries}
    for start in range(0, events.shape[0], block_size):
        stop = min(start + block_size, events.shape[0])
        chunk = events[start:stop]
        chunk_weights = target_weights[start:stop]
        for entry in pool.entries:
            terms[entry.opaque_id] += float(
                chunk_weights
                @ _gaussian(
                    chunk,
                    entry.task_spec.supports,
                    pool.kernel_bandwidth,
                )
                @ entry.task_spec.beta
            )
    return terms


_TARGET_NORM2_ATTESTATION = object()


def _target_norm2_binding_digest(target: "TargetSpecView") -> str:
    return sha256_json(
        {
            "arrays_sha256": sha256_ndarrays(
                {
                    "semantic_events": target.semantic_events,
                    "weights": target.weights,
                    "episode_offsets": target.episode_offsets,
                }
            ),
            "empirical_norm2": target.empirical_norm2,
            "protocol_id": target.protocol_id,
            "kernel_bandwidth": target.kernel_bandwidth,
            "dataset_digest": target.dataset_digest,
        }
    )


def _attest_target_norm2(target: "TargetSpecView") -> "TargetSpecView":
    object.__setattr__(
        target,
        "_exact_norm2_attestation",
        (_TARGET_NORM2_ATTESTATION, _target_norm2_binding_digest(target)),
    )
    return target


def _has_target_norm2_attestation(target: "TargetSpecView") -> bool:
    attestation = getattr(target, "_exact_norm2_attestation", None)
    return (
        isinstance(attestation, tuple)
        and len(attestation) == 2
        and attestation[0] is _TARGET_NORM2_ATTESTATION
        and attestation[1] == _target_norm2_binding_digest(target)
    )


@dataclass(frozen=True, eq=False)
class TargetSpecView:
    semantic_events: np.ndarray
    weights: np.ndarray
    episode_offsets: np.ndarray
    empirical_norm2: float
    protocol_id: str
    kernel_bandwidth: float
    dataset_digest: str

    _NORM_RTOL = 1.0e-8

    def __post_init__(self) -> None:
        events = np.array(self.semantic_events, dtype=np.float64, copy=True)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        offsets = np.array(self.episode_offsets, dtype=np.int64, copy=True)
        if events.ndim != 2 or weights.shape != (events.shape[0],) or events.shape[0] == 0:
            raise SelectorError("target events/weights have inconsistent shapes")
        if not np.all(np.isfinite(events)) or not np.all(np.isfinite(weights)):
            raise SelectorError("target empirical KME contains non-finite values")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1.0e-8):
            raise SelectorError("target KME weights must be non-negative and sum to one")
        if (
            offsets.ndim != 1
            or offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != events.shape[0]
            or np.any(np.diff(offsets) <= 0)
        ):
            raise SelectorError("target empirical episode_offsets are invalid")
        expected_weights = episode_balanced_weights(offsets)
        if not np.allclose(weights, expected_weights, rtol=1.0e-10, atol=1.0e-12):
            raise SelectorError("target KME weights are not episode-balanced")
        if not self.protocol_id:
            raise SelectorError("target protocol_id is required")
        if not np.isfinite(self.empirical_norm2) or self.empirical_norm2 < -1.0e-10:
            raise SelectorError("target empirical norm must be finite and non-negative")
        if not np.isfinite(self.kernel_bandwidth) or self.kernel_bandwidth <= 0:
            raise SelectorError("target kernel bandwidth must be positive")
        if not self.dataset_digest:
            raise SelectorError("target dataset_digest is required")
        events.setflags(write=False)
        weights.setflags(write=False)
        offsets.setflags(write=False)
        object.__setattr__(self, "semantic_events", events)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "episode_offsets", offsets)
        object.__setattr__(
            self, "empirical_norm2", max(float(self.empirical_norm2), 0.0)
        )
        object.__setattr__(self, "kernel_bandwidth", float(self.kernel_bandwidth))

    def verify_empirical_norm(self, *, block_size: int = 2048) -> float:
        """Audit and internally attest the cached norm against the query points.

        Selection invokes this automatically for public/unattested views.  The
        normal production path avoids it only when ``build_empirical_kme`` has
        supplied an intact producer attestation.
        """

        computed = blockwise_weighted_self_kernel_sum_auto(
            self.semantic_events,
            self.weights,
            GaussianKernel(self.kernel_bandwidth),
            block_size=block_size,
        )
        scale = max(1.0, abs(self.empirical_norm2), abs(computed))
        if abs(self.empirical_norm2 - computed) > self._NORM_RTOL * scale:
            raise SelectorError(
                "target empirical norm disagrees with its events and weights"
            )
        object.__setattr__(self, "empirical_norm2", max(float(computed), 0.0))
        _attest_target_norm2(self)
        return max(float(computed), 0.0)

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    @property
    def transition_count(self) -> int:
        return int(self.semantic_events.shape[0])

    @classmethod
    def from_empirical(
        cls,
        empirical: Any,
        *,
        bandwidth: float,
        protocol_id: str | None = None,
        verify_cached_norm2: bool = False,
    ) -> "TargetSpecView":
        events = np.array(
            _field(empirical, "semantic_events", "points", "events", "z"),
            dtype=np.float64,
            copy=True,
        )
        weights = np.array(
            _field(empirical, "weights", "alpha"), dtype=np.float64, copy=True
        )
        if events.ndim != 2 or weights.ndim != 1 or events.shape[0] != weights.shape[0]:
            raise SelectorError("target events/weights have inconsistent shapes")
        if events.shape[0] == 0 or not np.all(np.isfinite(events)):
            raise SelectorError("target semantic events must be non-empty and finite")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise SelectorError("target KME weights must be finite and non-negative")
        if not np.isclose(float(weights.sum()), 1.0, rtol=1.0e-6, atol=1.0e-8):
            raise SelectorError("target KME weights must sum to one")
        embedded_protocol = _optional_field(empirical, "protocol_id")
        if embedded_protocol is None or not str(embedded_protocol):
            raise SelectorError("target empirical artifact must carry protocol_id")
        resolved_protocol = str(embedded_protocol)
        if protocol_id is not None and resolved_protocol != protocol_id:
            raise SelectorError("target protocol mismatch")
        embedded_bandwidth = _optional_field(empirical, "kernel_bandwidth", "bandwidth")
        if embedded_bandwidth is None:
            raise SelectorError("target empirical artifact must carry kernel bandwidth")
        if not np.isclose(
            float(embedded_bandwidth), float(bandwidth), rtol=1.0e-12, atol=0.0
        ):
            raise SelectorError("target empirical KME kernel bandwidth differs from pool")
        offsets = _optional_field(empirical, "episode_offsets")
        if offsets is None:
            raise SelectorError("target empirical artifact must carry episode_offsets")
        dataset_digest = _optional_field(empirical, "dataset_digest")
        if dataset_digest is None or not str(dataset_digest):
            raise SelectorError("target empirical artifact must carry dataset_digest")
        norm_value = _optional_field(empirical, "empirical_norm2", "norm2", "self_norm2")
        # A public EmpiricalKME constructor and load_npz are intentionally
        # untrusted.  Only build_empirical_kme can attach the private, content-
        # bound attestation after completing the exact self-kernel calculation.
        must_recompute = (
            norm_value is None
            or verify_cached_norm2
            or not _has_exact_norm2_attestation(empirical)
        )
        if must_recompute:
            computed_norm2 = blockwise_weighted_self_kernel_sum_auto(
                events,
                weights,
                GaussianKernel(float(bandwidth)),
                block_size=2048,
            )
            if norm_value is not None:
                cached_norm2 = float(norm_value)
                scale = max(1.0, abs(cached_norm2), abs(computed_norm2))
                if abs(cached_norm2 - computed_norm2) > cls._NORM_RTOL * scale:
                    raise SelectorError(
                        "target empirical norm disagrees with its events and weights"
                    )
            norm2 = computed_norm2
        else:
            norm2 = float(norm_value)
        if not np.isfinite(norm2) or norm2 < -1.0e-10:
            raise SelectorError("target empirical norm must be finite and non-negative")
        return _attest_target_norm2(
            cls(
                events,
                weights,
                np.asarray(offsets, dtype=np.int64),
                max(norm2, 0.0),
                resolved_protocol,
                float(bandwidth),
                str(dataset_digest),
            )
        )


@dataclass(frozen=True)
class DistanceRecord:
    opaque_id: str
    distance_squared: float
    distance: float
    numerical_clamped: bool


@dataclass(frozen=True)
class SelectionResult:
    selection_id: str
    protocol_id: str
    target_dataset_digest: str
    selected_opaque_id: str
    sorted_distances: tuple[DistanceRecord, ...]
    probe_episode_count: int
    probe_steps: int
    selector_runtime_seconds: float
    clamp_count: int
    pool_id: str = ""
    pool_digest: str = ""

    @property
    def selected_learnware_id(self) -> str:
        """Compatibility name used by the Coding Plan; value remains opaque."""

        return self.selected_opaque_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.selection-result.v0",
            "selection_id": self.selection_id,
            "protocol_id": self.protocol_id,
            "pool_id": self.pool_id,
            "pool_digest": self.pool_digest,
            "target_dataset_digest": self.target_dataset_digest,
            "selected_opaque_id": self.selected_opaque_id,
            "sorted_distances": [
                {
                    "opaque_id": item.opaque_id,
                    "distance_squared": item.distance_squared,
                    "distance": item.distance,
                    "numerical_clamped": item.numerical_clamped,
                }
                for item in self.sorted_distances
            ],
            "probe_episode_count": self.probe_episode_count,
            "probe_steps": self.probe_steps,
            "selector_runtime_seconds": self.selector_runtime_seconds,
            "clamp_count": self.clamp_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionResult":
        expected = {
            "schema",
            "selection_id",
            "protocol_id",
            "pool_id",
            "pool_digest",
            "target_dataset_digest",
            "selected_opaque_id",
            "sorted_distances",
            "probe_episode_count",
            "probe_steps",
            "selector_runtime_seconds",
            "clamp_count",
        }
        if (
            set(value) != expected
            or value.get("schema") != "policy-learnware.selection-result.v0"
        ):
            raise SelectorError("unsupported or malformed selection result artifact")
        raw_distances = value.get("sorted_distances")
        if not isinstance(raw_distances, list) or not raw_distances:
            raise SelectorError("selection result has no distance ranking")
        distances_list: list[DistanceRecord] = []
        for item in raw_distances:
            if not isinstance(item, Mapping) or set(item) != {
                "opaque_id",
                "distance_squared",
                "distance",
                "numerical_clamped",
            }:
                raise SelectorError("selection distance record is malformed")
            squared = float(item["distance_squared"])
            distance = float(item["distance"])
            if (
                not math.isfinite(squared)
                or not math.isfinite(distance)
                or squared < 0.0
                or distance < 0.0
                or not math.isclose(distance * distance, squared, rel_tol=1.0e-10, abs_tol=1.0e-12)
                or not isinstance(item["numerical_clamped"], bool)
            ):
                raise SelectorError("selection distance values are invalid")
            distances_list.append(
                DistanceRecord(
                    opaque_id=str(item["opaque_id"]),
                    distance_squared=squared,
                    distance=distance,
                    numerical_clamped=item["numerical_clamped"],
                )
            )
        distances = tuple(distances_list)
        if len({item.opaque_id for item in distances}) != len(distances) or any(
            not item.opaque_id for item in distances
        ):
            raise SelectorError("selection distance ids are empty or duplicated")
        if distances != tuple(
            sorted(distances, key=lambda item: (item.distance_squared, item.opaque_id))
        ):
            raise SelectorError("selection distance ranking is not canonically sorted")
        selected = str(value["selected_opaque_id"])
        if distances[0].opaque_id != selected:
            raise SelectorError("selection result winner differs from ranking")
        selection_id = str(value["selection_id"])
        protocol_id = str(value["protocol_id"])
        target_dataset_digest = str(value["target_dataset_digest"])
        pool_id = str(value["pool_id"])
        pool_digest = str(value["pool_digest"])
        episode_count = int(value["probe_episode_count"])
        probe_steps = int(value["probe_steps"])
        runtime = float(value["selector_runtime_seconds"])
        clamp_count = int(value["clamp_count"])
        if (
            not protocol_id
            or not target_dataset_digest
            or not pool_id
            or len(pool_digest) != 64
            or episode_count <= 0
            or probe_steps < episode_count
            or not math.isfinite(runtime)
            or runtime < 0.0
            or clamp_count != sum(item.numerical_clamped for item in distances)
        ):
            raise SelectorError("selection result metadata is invalid")
        selection_payload = "\0".join(
            (
                pool_id,
                protocol_id,
                pool_digest,
                target_dataset_digest,
                str(episode_count),
                selected,
            )
        )
        expected_selection_id = (
            "sel-"
            + hashlib.sha256(selection_payload.encode("utf-8")).hexdigest()[:20]
        )
        if selection_id != expected_selection_id:
            raise SelectorError("selection id does not match its bound fields")
        return cls(
            selection_id=selection_id,
            protocol_id=protocol_id,
            target_dataset_digest=target_dataset_digest,
            selected_opaque_id=selected,
            sorted_distances=distances,
            probe_episode_count=episode_count,
            probe_steps=probe_steps,
            selector_runtime_seconds=runtime,
            clamp_count=clamp_count,
            pool_id=pool_id,
            pool_digest=pool_digest,
        )


def _distance_squared(
    target: TargetSpecView,
    source: SelectorTaskSpec,
    bandwidth: float,
    *,
    negative_tolerance: float,
) -> tuple[float, bool]:
    cross_kernel = _gaussian(target.semantic_events, source.supports, bandwidth)
    cross = float(target.weights @ cross_kernel @ source.beta)
    value = float(target.empirical_norm2 - 2.0 * cross + source.rkme_norm2)
    scale = max(1.0, abs(target.empirical_norm2), abs(source.rkme_norm2), abs(2.0 * cross))
    if value < -negative_tolerance * scale:
        raise SelectorError(
            f"MMD squared is materially negative ({value}); RKME norm/kernel metadata disagree"
        )
    clamped = value < 0.0
    return max(value, 0.0), clamped


class NearestSpecSelector:
    """A selector that has no reference to policies or the private registry."""

    def __init__(self, pool: LearnwarePool, *, negative_tolerance: float = 1.0e-8) -> None:
        self._pool = pool
        self._negative_tolerance = float(negative_tolerance)
        self._pool_digest = sha256_json(pool.public_manifest())

    @property
    def pool_id(self) -> str:
        return self._pool.pool_id

    def select(
        self,
        target_empirical: Any,
        *,
        target_dataset_digest: str | None = None,
        probe_episode_count: int | None = None,
        probe_steps: int | None = None,
        verify_target_norm2: bool = False,
    ) -> SelectionResult:
        start = time.perf_counter()
        target = (
            target_empirical
            if isinstance(target_empirical, TargetSpecView)
            else TargetSpecView.from_empirical(
                target_empirical,
                bandwidth=self._pool.kernel_bandwidth,
                protocol_id=self._pool.protocol_id,
                verify_cached_norm2=verify_target_norm2,
            )
        )
        if isinstance(target_empirical, TargetSpecView) and (
            verify_target_norm2
            or not _has_target_norm2_attestation(target_empirical)
        ):
            target.verify_empirical_norm()
        if target.protocol_id != self._pool.protocol_id:
            raise SelectorError("target and pool protocol ids differ")
        if not np.isclose(
            target.kernel_bandwidth,
            self._pool.kernel_bandwidth,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise SelectorError("target and pool kernel bandwidths differ")
        if target.semantic_events.shape[1] != self._pool.latent_dim:
            raise SelectorError("target and pool latent dimensions differ")
        if target_dataset_digest is not None and target_dataset_digest != target.dataset_digest:
            raise SelectorError("declared dataset digest disagrees with target artifact")
        if probe_episode_count is not None and int(probe_episode_count) != target.episode_count:
            raise SelectorError("declared episode count disagrees with target artifact")
        if probe_steps is not None and int(probe_steps) != target.transition_count:
            raise SelectorError("declared probe steps disagree with target artifact")
        target_dataset_digest = target.dataset_digest
        probe_episode_count = target.episode_count
        probe_steps = target.transition_count

        distances: list[DistanceRecord] = []
        for entry in self._pool.entries:
            squared, clamped = _distance_squared(
                target,
                entry.task_spec,
                self._pool.kernel_bandwidth,
                negative_tolerance=self._negative_tolerance,
            )
            distances.append(
                DistanceRecord(
                    opaque_id=entry.opaque_id,
                    distance_squared=squared,
                    distance=math.sqrt(squared),
                    numerical_clamped=clamped,
                )
            )
        ranking = tuple(sorted(distances, key=lambda item: (item.distance_squared, item.opaque_id)))
        selection_payload = "\0".join(
            (
                self._pool.pool_id,
                self._pool.protocol_id,
                self._pool_digest,
                target_dataset_digest,
                str(probe_episode_count),
                ranking[0].opaque_id,
            )
        )
        selection_id = "sel-" + hashlib.sha256(selection_payload.encode("utf-8")).hexdigest()[:20]
        runtime = time.perf_counter() - start
        return SelectionResult(
            selection_id=selection_id,
            protocol_id=self._pool.protocol_id,
            target_dataset_digest=str(target_dataset_digest),
            selected_opaque_id=ranking[0].opaque_id,
            sorted_distances=ranking,
            probe_episode_count=int(probe_episode_count),
            probe_steps=int(probe_steps),
            selector_runtime_seconds=float(runtime),
            clamp_count=sum(item.numerical_clamped for item in ranking),
            pool_id=self._pool.pool_id,
            pool_digest=self._pool_digest,
        )

    def select_from_precomputed_terms(
        self,
        *,
        target_empirical_norm2: float,
        target_source_cross: Mapping[str, float],
        target_dataset_digest: str,
        probe_episode_count: int,
        probe_steps: int,
    ) -> SelectionResult:
        """Select from exact norm/cross terms computed outside the selector.

        The inputs are precisely the sufficient statistics used by the usual
        RKHS distance formula.  This path exists so nested query prefixes can
        reuse exact block sums without rebuilding an EmpiricalKME seven times.
        """

        start = time.perf_counter()
        norm2 = float(target_empirical_norm2)
        if not math.isfinite(norm2) or norm2 < -self._negative_tolerance:
            raise SelectorError("target empirical norm is invalid")
        if not target_dataset_digest or probe_episode_count <= 0 or probe_steps < probe_episode_count:
            raise SelectorError("precomputed target metadata is invalid")
        expected_ids = {entry.opaque_id for entry in self._pool.entries}
        if set(target_source_cross) != expected_ids:
            raise SelectorError("precomputed target/source coverage differs from pool")

        distances: list[DistanceRecord] = []
        for entry in self._pool.entries:
            cross = float(target_source_cross[entry.opaque_id])
            if not math.isfinite(cross):
                raise SelectorError("precomputed target/source term is non-finite")
            value = float(norm2 - 2.0 * cross + entry.task_spec.rkme_norm2)
            scale = max(1.0, abs(norm2), abs(entry.task_spec.rkme_norm2), abs(2.0 * cross))
            if value < -self._negative_tolerance * scale:
                raise SelectorError(
                    f"MMD squared is materially negative ({value}); "
                    "RKME norm/kernel metadata disagree"
                )
            clamped = value < 0.0
            squared = max(value, 0.0)
            distances.append(
                DistanceRecord(
                    opaque_id=entry.opaque_id,
                    distance_squared=squared,
                    distance=math.sqrt(squared),
                    numerical_clamped=clamped,
                )
            )
        ranking = tuple(
            sorted(distances, key=lambda item: (item.distance_squared, item.opaque_id))
        )
        selection_payload = "\0".join(
            (
                self._pool.pool_id,
                self._pool.protocol_id,
                self._pool_digest,
                target_dataset_digest,
                str(probe_episode_count),
                ranking[0].opaque_id,
            )
        )
        return SelectionResult(
            selection_id=(
                "sel-"
                + hashlib.sha256(selection_payload.encode("utf-8")).hexdigest()[:20]
            ),
            protocol_id=self._pool.protocol_id,
            target_dataset_digest=target_dataset_digest,
            selected_opaque_id=ranking[0].opaque_id,
            sorted_distances=ranking,
            probe_episode_count=int(probe_episode_count),
            probe_steps=int(probe_steps),
            selector_runtime_seconds=float(time.perf_counter() - start),
            clamp_count=sum(item.numerical_clamped for item in ranking),
            pool_id=self._pool.pool_id,
            pool_digest=self._pool_digest,
        )
