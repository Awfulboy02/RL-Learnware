"""Canonical transition batches and deterministic episode-bound windows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays


class WindowingError(ValueError):
    """A transition batch or window partition violates the frozen contract."""


_WINDOW_ID = re.compile(r"^v03w-[0-9a-f]{32}$")


def _readonly_numeric(
    value: Any,
    *,
    where: str,
    ndim_min: int,
    dtype: np.dtype[Any] | type = np.float64,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim < ndim_min or array.shape[0] <= 0:
        raise WindowingError(f"{where} must be a non-empty array with ndim >= {ndim_min}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise WindowingError(f"{where} must be finite")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _readonly_optional_mask(
    value: Any | None, *, expected: tuple[int, ...], where: str
) -> np.ndarray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise WindowingError(f"{where} must be boolean")
    mask = np.asarray(raw, dtype=np.bool_)
    allowed = {expected, expected[1:]}
    if mask.shape not in allowed:
        raise WindowingError(
            f"{where} shape must equal full channel shape or static feature shape"
        )
    result = np.array(mask, copy=True)
    result.setflags(write=False)
    return result


def _readonly_boolean(value: Any, *, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.ndim != 1 or raw.shape[0] <= 0:
        raise WindowingError(f"{where} must be a non-empty boolean vector")
    result = np.array(raw, dtype=np.bool_, copy=True)
    result.setflags(write=False)
    return result


def _readonly_integer(value: Any, *, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.shape[0] <= 0:
        raise WindowingError(f"{where} must be a non-empty integer vector")
    result = np.array(raw, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CanonicalTransitionBatch:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray | None
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    observation_mask: np.ndarray | None
    action_mask: np.ndarray | None
    episode_id: np.ndarray
    timestep: np.ndarray

    def __post_init__(self) -> None:
        observation = _readonly_numeric(
            self.observation, where="observation", ndim_min=2
        )
        action = _readonly_numeric(self.action, where="action", ndim_min=2)
        next_observation = _readonly_numeric(
            self.next_observation, where="next_observation", ndim_min=2
        )
        count = observation.shape[0]
        if action.shape[0] != count or next_observation.shape != observation.shape:
            raise WindowingError(
                "observation/action/next_observation batch dimensions are inconsistent"
            )
        reward: np.ndarray | None
        if self.reward is None:
            reward = None
        else:
            reward = _readonly_numeric(self.reward, where="reward", ndim_min=1)
            if reward.shape[0] != count or reward.ndim not in {1, 2}:
                raise WindowingError("reward must have shape [N] or [N, channels]")

        terminated = _readonly_boolean(self.terminated, where="terminated")
        truncated = _readonly_boolean(self.truncated, where="truncated")
        episode_id = _readonly_integer(self.episode_id, where="episode_id")
        timestep = _readonly_integer(self.timestep, where="timestep")
        for name, array in (
            ("terminated", terminated),
            ("truncated", truncated),
            ("episode_id", episode_id),
            ("timestep", timestep),
        ):
            if array.shape != (count,):
                raise WindowingError(f"{name} must have shape [N]")
        if np.any(episode_id < 0) or np.any(timestep < 0):
            raise WindowingError("episode_id and timestep must be non-negative")

        seen: set[int] = set()
        start = 0
        for index in range(1, count + 1):
            if index == count or episode_id[index] != episode_id[start]:
                identifier = int(episode_id[start])
                if identifier in seen:
                    raise WindowingError("each episode_id must occupy one contiguous block")
                seen.add(identifier)
                local_timestep = timestep[start:index]
                if local_timestep.size > 1 and not np.all(np.diff(local_timestep) == 1):
                    raise WindowingError("timesteps must increase contiguously within an episode")
                done = np.logical_or(terminated[start:index], truncated[start:index])
                if np.any(done[:-1]):
                    raise WindowingError("an episode continues after terminated/truncated")
                start = index

        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "terminated", terminated)
        object.__setattr__(self, "truncated", truncated)
        object.__setattr__(
            self,
            "observation_mask",
            _readonly_optional_mask(
                self.observation_mask,
                expected=observation.shape,
                where="observation_mask",
            ),
        )
        object.__setattr__(
            self,
            "action_mask",
            _readonly_optional_mask(
                self.action_mask, expected=action.shape, where="action_mask"
            ),
        )
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "timestep", timestep)

    @property
    def transition_count(self) -> int:
        return int(self.observation.shape[0])

    @property
    def transition_digest(self) -> str:
        arrays: dict[str, np.ndarray] = {
            "observation": self.observation,
            "action": self.action,
            "next_observation": self.next_observation,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_id": self.episode_id,
            "timestep": self.timestep,
        }
        optional_presence = {
            "reward": self.reward is not None,
            "observation_mask": self.observation_mask is not None,
            "action_mask": self.action_mask is not None,
        }
        if self.reward is not None:
            arrays["reward"] = self.reward
        if self.observation_mask is not None:
            arrays["observation_mask"] = self.observation_mask
        if self.action_mask is not None:
            arrays["action_mask"] = self.action_mask
        return sha256_json(
            {
                "schema": "policy-learnware.v03-canonical-transition-batch.v0",
                "arrays_digest": sha256_ndarrays(arrays),
                "optional_presence": optional_presence,
            }
        )


@dataclass(frozen=True)
class WindowingProtocol:
    window_length: int
    stride: int
    pooling: str
    pad_final_window: bool

    def __post_init__(self) -> None:
        for name in ("window_length", "stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WindowingError(f"{name} must be a positive integer")
        if self.pooling not in {"mean", "last", "attention"}:
            raise WindowingError("unknown window pooling")
        if not isinstance(self.pad_final_window, bool):
            raise WindowingError("pad_final_window must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v03-window-protocol.v0",
            "window_length": self.window_length,
            "stride": self.stride,
            "pooling": self.pooling,
            "pad_final_window": self.pad_final_window,
        }

    @property
    def window_protocol_digest(self) -> str:
        return sha256_json(self.to_dict())


def _derive_window_partition(
    transitions: CanonicalTransitionBatch,
    protocol: WindowingProtocol,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Derive the one canonical partition for a transition batch/protocol pair.

    ``TransitionWindowBatch`` is a validated value object, not an alternate entry
    point for callers to invent partitions or opaque IDs.  Keeping derivation in
    one helper lets construction and validation use byte-identical semantics.
    """

    rows: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    window_ids: list[str] = []
    count = transitions.transition_count
    transition_digest = transitions.transition_digest
    episode_start = 0
    while episode_start < count:
        episode = int(transitions.episode_id[episode_start])
        episode_end = episode_start + 1
        while (
            episode_end < count
            and int(transitions.episode_id[episode_end]) == episode
        ):
            episode_end += 1
        episode_length = episode_end - episode_start
        if protocol.pad_final_window:
            starts = range(0, episode_length, protocol.stride)
        else:
            starts = range(
                0,
                max(0, episode_length - protocol.window_length + 1),
                protocol.stride,
            )
        for local_start in starts:
            valid_count = min(protocol.window_length, episode_length - local_start)
            if valid_count < protocol.window_length and not protocol.pad_final_window:
                continue
            absolute = np.arange(
                episode_start + local_start,
                episode_start + local_start + valid_count,
                dtype=np.int64,
            )
            row = np.full(protocol.window_length, -1, dtype=np.int64)
            mask = np.zeros(protocol.window_length, dtype=np.bool_)
            row[:valid_count] = absolute
            mask[:valid_count] = True
            material = {
                "transition_digest": transition_digest,
                "window_protocol_digest": protocol.window_protocol_digest,
                "episode_id": episode,
                "start_timestep": int(transitions.timestep[absolute[0]]),
                "valid_count": valid_count,
            }
            rows.append(row)
            masks.append(mask)
            window_ids.append(f"v03w-{sha256_json(material)[:32]}")
        episode_start = episode_end

    if not rows:
        raise WindowingError("window protocol produced no non-empty windows")
    return np.stack(rows), np.stack(masks), tuple(window_ids)


@dataclass(frozen=True)
class TransitionWindowBatch:
    transitions: CanonicalTransitionBatch
    window_indices: np.ndarray
    window_mask: np.ndarray
    window_ids: tuple[str, ...]
    protocol: WindowingProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.transitions, CanonicalTransitionBatch):
            raise WindowingError("transitions must be a CanonicalTransitionBatch")
        if not isinstance(self.protocol, WindowingProtocol):
            raise WindowingError("protocol must be WindowingProtocol")
        indices = np.asarray(self.window_indices, dtype=np.int64)
        mask = np.asarray(self.window_mask, dtype=np.bool_)
        expected_width = self.protocol.window_length
        if (
            indices.ndim != 2
            or indices.shape[0] <= 0
            or indices.shape[1] != expected_width
            or mask.shape != indices.shape
        ):
            raise WindowingError("window_indices/window_mask have an invalid shape")
        if len(self.window_ids) != indices.shape[0] or len(set(self.window_ids)) != len(
            self.window_ids
        ):
            raise WindowingError("window_ids must be unique and align with windows")
        if any(
            not isinstance(item, str) or not _WINDOW_ID.fullmatch(item)
            for item in self.window_ids
        ):
            raise WindowingError("window_ids must use the v03w- opaque format")

        for row, row_mask in zip(indices, mask, strict=True):
            valid_count = int(np.sum(row_mask))
            if valid_count <= 0 or not np.all(row_mask[:valid_count]) or np.any(
                row_mask[valid_count:]
            ):
                raise WindowingError("window padding mask must be a non-empty true prefix")
            if np.any(row[:valid_count] < 0) or np.any(
                row[:valid_count] >= self.transitions.transition_count
            ):
                raise WindowingError("window index is out of bounds")
            if np.any(row[valid_count:] != -1):
                raise WindowingError("padded window indices must be -1")
            selected = row[:valid_count]
            if selected.size > 1 and not np.all(np.diff(selected) == 1):
                raise WindowingError("window indices must be contiguous")
            episodes = self.transitions.episode_id[selected]
            if not np.all(episodes == episodes[0]):
                raise WindowingError("a transition window crosses an episode boundary")
            timesteps = self.transitions.timestep[selected]
            if timesteps.size > 1 and not np.all(np.diff(timesteps) == 1):
                raise WindowingError("a transition window skips timesteps")

        expected_indices, expected_mask, expected_ids = _derive_window_partition(
            self.transitions, self.protocol
        )
        if not np.array_equal(indices, expected_indices) or not np.array_equal(
            mask, expected_mask
        ):
            raise WindowingError(
                "window partition differs from the frozen stride/padding derivation"
            )
        if tuple(self.window_ids) != expected_ids:
            raise WindowingError(
                "window_ids differ from the canonical per-window derivation"
            )

        frozen_indices = np.array(indices, copy=True)
        frozen_mask = np.array(mask, copy=True)
        frozen_indices.setflags(write=False)
        frozen_mask.setflags(write=False)
        object.__setattr__(self, "window_indices", frozen_indices)
        object.__setattr__(self, "window_mask", frozen_mask)
        object.__setattr__(self, "window_ids", tuple(self.window_ids))

    @property
    def window_length(self) -> int:
        return self.protocol.window_length

    @property
    def window_protocol_digest(self) -> str:
        return self.protocol.window_protocol_digest

    @property
    def ordered_episode_window_digest(self) -> str:
        return sha256_json(
            {
                "schema": "policy-learnware.v03-ordered-episode-windows.v0",
                "transition_digest": self.transitions.transition_digest,
                "window_protocol_digest": self.window_protocol_digest,
                "partition_arrays_digest": sha256_ndarrays(
                    {"window_indices": self.window_indices, "window_mask": self.window_mask}
                ),
                "window_ids": list(self.window_ids),
            }
        )

    def materialize_channel(self, name: str) -> np.ndarray:
        if name not in {
            "observation",
            "action",
            "reward",
            "next_observation",
            "terminated",
            "truncated",
            "observation_mask",
            "action_mask",
        }:
            raise WindowingError(f"unknown transition channel: {name!r}")
        channel = getattr(self.transitions, name)
        if channel is None:
            raise WindowingError(f"transition channel {name!r} is absent")
        # Feature masks may either vary per transition or be a static schema mask.
        # Materialize both forms to the same [window, time, ...] representation so
        # adapters never need access to the underlying transition object.
        static_shape = (
            self.transitions.observation.shape[1:]
            if name == "observation_mask"
            else self.transitions.action.shape[1:]
        )
        if name in {"observation_mask", "action_mask"} and channel.shape == static_shape:
            channel = np.broadcast_to(
                channel,
                (self.transitions.transition_count,) + channel.shape,
            )
        trailing = channel.shape[1:]
        result = np.zeros(self.window_indices.shape + trailing, dtype=channel.dtype)
        for row_index, (indices, mask) in enumerate(
            zip(self.window_indices, self.window_mask, strict=True)
        ):
            result[row_index, mask] = channel[indices[mask]]
        result.setflags(write=False)
        return result


def build_transition_windows(
    transitions: CanonicalTransitionBatch, protocol: WindowingProtocol
) -> TransitionWindowBatch:
    if not isinstance(transitions, CanonicalTransitionBatch):
        raise WindowingError("transitions must be a CanonicalTransitionBatch")
    if not isinstance(protocol, WindowingProtocol):
        raise WindowingError("protocol must be WindowingProtocol")

    indices, mask, window_ids = _derive_window_partition(transitions, protocol)
    return TransitionWindowBatch(
        transitions=transitions,
        window_indices=indices,
        window_mask=mask,
        window_ids=window_ids,
        protocol=protocol,
    )


__all__ = [
    "CanonicalTransitionBatch",
    "TransitionWindowBatch",
    "WindowingError",
    "WindowingProtocol",
    "build_transition_windows",
]
