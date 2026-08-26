"""Fail-closed transition views for v0/v0.1 attribution replay.

The historical datasets are never rewritten.  They are adapted into an
immutable :class:`TransitionBank`, and every registered view exposes only its
declared channels.  Destructive controls additionally retain source-row
indices so an independent audit can prove which marginal was preserved and
which pairing was destroyed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from ..hashing import sha256_json, sha256_ndarrays
from ..probe.dataset import EpisodeDataset


TRANSITION_VIEW_PROTOCOL_ID = sha256_json(
    {
        "schema": "policy-learnware.v03-transition-view-protocol.v0",
        "registry_version": 1,
        "shuffle_scope": "within-bank",
        "shuffle_marginal_rule": "exact-array-permutation",
    }
)

V_FULL_LEGACY = "V_FULL_LEGACY"
V_MASK_ONLY = "V_MASK_ONLY"
V_DIMS_ONLY = "V_DIMS_ONLY"
V_REWARD_ONLY = "V_REWARD_ONLY"
V_STATE_ONLY = "V_STATE_ONLY"
V_ACTION_ONLY = "V_ACTION_ONLY"
V_STATE_ACTION = "V_STATE_ACTION"
V_DELTA_ONLY = "V_DELTA_ONLY"
V_REWARD_FREE_TRANSITION = "V_REWARD_FREE_TRANSITION"
V_NO_MASK = "V_NO_MASK"
V_SHUFFLED_NEXT = "V_SHUFFLED_NEXT"
V_SHUFFLED_REWARD = "V_SHUFFLED_REWARD"
V_TEMPORAL_SHUFFLE = "V_TEMPORAL_SHUFFLE"
V_RANDOM_ENCODER = "V_RANDOM_ENCODER"

REGISTERED_VIEW_IDS = (
    V_FULL_LEGACY,
    V_MASK_ONLY,
    V_DIMS_ONLY,
    V_REWARD_ONLY,
    V_STATE_ONLY,
    V_ACTION_ONLY,
    V_STATE_ACTION,
    V_DELTA_ONLY,
    V_REWARD_FREE_TRANSITION,
    V_NO_MASK,
    V_SHUFFLED_NEXT,
    V_SHUFFLED_REWARD,
    V_TEMPORAL_SHUFFLE,
    V_RANDOM_ENCODER,
)


class TransitionViewError(ValueError):
    """A transition bank or requested attribution view is invalid."""


def _readonly(
    value: Any,
    *,
    name: str,
    ndim: int,
    dtype: np.dtype[Any] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise TransitionViewError(f"{name} must be {ndim}D, got {array.shape}")
    if array.dtype.hasobject:
        raise TransitionViewError(f"{name} cannot use object dtype")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise TransitionViewError(f"{name} contains non-finite values")
    result = np.ascontiguousarray(array).copy()
    result.setflags(write=False)
    return result


def _sha256_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise TransitionViewError(f"{name} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise TransitionViewError(f"{name} must be a lowercase SHA-256 digest") from error
    return value


def _mask(value: Any | None, *, rows: int, width: int, name: str) -> np.ndarray:
    if value is None:
        array = np.ones((rows, width), dtype=np.float32)
    else:
        raw = np.asarray(value)
        if raw.shape == (width,):
            raw = np.broadcast_to(raw, (rows, width))
        if raw.shape != (rows, width):
            raise TransitionViewError(
                f"{name} must have shape ({width},) or ({rows}, {width})"
            )
        if not np.all(np.logical_or(raw == 0, raw == 1)):
            raise TransitionViewError(f"{name} must be binary")
        array = np.asarray(raw, dtype=np.float32)
    return _readonly(array, name=name, ndim=2, dtype=np.dtype(np.float32))


@dataclass(frozen=True)
class TransitionBank:
    """Canonical, immutable adapter around archived flat transition arrays."""

    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_offsets: np.ndarray
    observation_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    archived_dataset_digest: str | None = None
    padding_value: float = 0.0

    def __post_init__(self) -> None:
        observation = _readonly(
            self.observation, name="observation", ndim=2, dtype=np.dtype(np.float32)
        )
        action = _readonly(
            self.action, name="action", ndim=2, dtype=np.dtype(np.float32)
        )
        reward = _readonly(
            self.reward, name="reward", ndim=1, dtype=np.dtype(np.float32)
        )
        next_observation = _readonly(
            self.next_observation,
            name="next_observation",
            ndim=2,
            dtype=np.dtype(np.float32),
        )
        terminated = _readonly(
            self.terminated, name="terminated", ndim=1, dtype=np.dtype(np.bool_)
        )
        truncated = _readonly(
            self.truncated, name="truncated", ndim=1, dtype=np.dtype(np.bool_)
        )
        offsets = _readonly(
            self.episode_offsets,
            name="episode_offsets",
            ndim=1,
            dtype=np.dtype(np.int64),
        )
        rows = observation.shape[0]
        if rows == 0:
            raise TransitionViewError("transition bank cannot be empty")
        if next_observation.shape != observation.shape:
            raise TransitionViewError("observation shapes disagree")
        if action.shape[0] != rows:
            raise TransitionViewError("action transition count disagrees")
        for name, array in (
            ("reward", reward),
            ("terminated", terminated),
            ("truncated", truncated),
        ):
            if array.shape != (rows,):
                raise TransitionViewError(f"{name} transition count disagrees")
        if (
            offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != rows
            or np.any(np.diff(offsets) <= 0)
        ):
            raise TransitionViewError("episode_offsets are invalid")
        if not np.isfinite(self.padding_value):
            raise TransitionViewError("padding_value must be finite")
        observation_mask = _mask(
            self.observation_mask,
            rows=rows,
            width=observation.shape[1],
            name="observation_mask",
        )
        action_mask = _mask(
            self.action_mask,
            rows=rows,
            width=action.shape[1],
            name="action_mask",
        )
        if np.any(observation[observation_mask == 0] != self.padding_value):
            raise TransitionViewError(
                "padded observation slots must equal the declared padding_value"
            )
        if np.any(next_observation[observation_mask == 0] != self.padding_value):
            raise TransitionViewError(
                "padded next-observation slots must equal padding_value"
            )
        if np.any(action[action_mask == 0] != self.padding_value):
            raise TransitionViewError(
                "padded action slots must equal the declared padding_value"
            )
        for name, array in (
            ("observation", observation),
            ("action", action),
            ("reward", reward),
            ("next_observation", next_observation),
            ("terminated", terminated),
            ("truncated", truncated),
            ("episode_offsets", offsets),
            ("observation_mask", observation_mask),
            ("action_mask", action_mask),
        ):
            object.__setattr__(self, name, array)
        content_digest = sha256_ndarrays(self.to_arrays(copy=False))
        if self.archived_dataset_digest is None:
            object.__setattr__(self, "archived_dataset_digest", content_digest)
        else:
            object.__setattr__(
                self,
                "archived_dataset_digest",
                _sha256_digest(
                    self.archived_dataset_digest, "archived_dataset_digest"
                ),
            )

    @classmethod
    def from_episode_dataset(
        cls,
        dataset: EpisodeDataset,
        *,
        observation_mask: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
        padding_value: float = 0.0,
    ) -> "TransitionBank":
        if not isinstance(dataset, EpisodeDataset):
            raise TransitionViewError("dataset must be an EpisodeDataset")
        return cls(
            observation=dataset.observation,
            action=dataset.action,
            reward=dataset.reward,
            next_observation=dataset.next_observation,
            terminated=dataset.terminated,
            truncated=dataset.truncated,
            episode_offsets=dataset.episode_offsets,
            observation_mask=observation_mask,
            action_mask=action_mask,
            archived_dataset_digest=dataset.digest,
            padding_value=padding_value,
        )

    @classmethod
    def from_canonical_batch(
        cls,
        batch: Any,
        *,
        padding_value: float = 0.0,
    ) -> "TransitionBank":
        """Bridge the P3 canonical/window contract into attribution views."""

        # Local import avoids making the historical-attribution adapter a
        # prerequisite for dependency-light users of the windowing module.
        from .windowing import CanonicalTransitionBatch

        if not isinstance(batch, CanonicalTransitionBatch):
            raise TransitionViewError("batch must be a CanonicalTransitionBatch")
        if batch.reward is None:
            raise TransitionViewError(
                "attribution registry requires reward-present archived transitions"
            )
        episode_id = np.asarray(batch.episode_id, dtype=np.int64)
        boundaries = np.flatnonzero(episode_id[1:] != episode_id[:-1]) + 1
        offsets = np.concatenate(
            [np.asarray([0], dtype=np.int64), boundaries, [batch.transition_count]]
        )
        return cls(
            observation=batch.observation,
            action=batch.action,
            reward=np.asarray(batch.reward).reshape(batch.transition_count, -1)[:, 0],
            next_observation=batch.next_observation,
            terminated=batch.terminated,
            truncated=batch.truncated,
            episode_offsets=offsets,
            observation_mask=batch.observation_mask,
            action_mask=batch.action_mask,
            archived_dataset_digest=batch.transition_digest,
            padding_value=padding_value,
        )

    @property
    def transition_count(self) -> int:
        return int(self.observation.shape[0])

    @property
    def canonical_bank_digest(self) -> str:
        """Digest of the adapted arrays, distinct from the archived file digest."""

        return sha256_ndarrays(self.to_arrays(copy=False))

    def to_arrays(self, *, copy: bool = True) -> dict[str, np.ndarray]:
        arrays = {
            "observation": self.observation,
            "action": self.action,
            "reward": self.reward,
            "next_observation": self.next_observation,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_offsets": self.episode_offsets,
            "observation_mask": self.observation_mask,
            "action_mask": self.action_mask,
        }
        return (
            {name: np.array(array, copy=True) for name, array in arrays.items()}
            if copy
            else arrays
        )


@dataclass(frozen=True)
class TransitionViewSpec:
    view_id: str
    input_channel_allowlist: tuple[str, ...]
    destroys_pairing: str | None = None
    deployable: bool = False

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "protocol_id": TRANSITION_VIEW_PROTOCOL_ID,
                "view_id": self.view_id,
                "input_channel_allowlist": list(self.input_channel_allowlist),
                "destroys_pairing": self.destroys_pairing,
                "deployable": self.deployable,
            }
        )


VIEW_REGISTRY: Mapping[str, TransitionViewSpec] = MappingProxyType(
    {
        spec.view_id: spec
        for spec in (
            TransitionViewSpec(
                V_FULL_LEGACY,
                (
                    "observation",
                    "observation_mask",
                    "action",
                    "action_mask",
                    "reward",
                    "next_observation",
                    "next_observation_mask",
                ),
            ),
            TransitionViewSpec(
                V_MASK_ONLY,
                ("observation_mask", "action_mask", "next_observation_mask"),
            ),
            TransitionViewSpec(
                V_DIMS_ONLY, ("observation_native_dim", "action_native_dim")
            ),
            # P1 replays the frozen legacy packed input.  That historical input
            # never contained done flags, so the attribution registry must not
            # claim that the archived checkpoint consumed them.
            TransitionViewSpec(V_REWARD_ONLY, ("reward",)),
            TransitionViewSpec(V_STATE_ONLY, ("observation",)),
            TransitionViewSpec(V_ACTION_ONLY, ("action", "action_mask")),
            TransitionViewSpec(V_STATE_ACTION, ("observation", "action")),
            TransitionViewSpec(V_DELTA_ONLY, ("observation_delta", "action")),
            TransitionViewSpec(
                V_REWARD_FREE_TRANSITION,
                (
                    "observation",
                    "action",
                    "next_observation",
                ),
                deployable=True,
            ),
            TransitionViewSpec(
                V_NO_MASK,
                ("observation", "action", "reward", "next_observation"),
            ),
            TransitionViewSpec(
                V_SHUFFLED_NEXT,
                (
                    "observation",
                    "observation_mask",
                    "action",
                    "action_mask",
                    "reward",
                    "next_observation",
                    "next_observation_mask",
                ),
                destroys_pairing="next_observation",
            ),
            TransitionViewSpec(
                V_SHUFFLED_REWARD,
                (
                    "observation",
                    "observation_mask",
                    "action",
                    "action_mask",
                    "reward",
                    "next_observation",
                    "next_observation_mask",
                ),
                destroys_pairing="reward",
            ),
            TransitionViewSpec(
                V_TEMPORAL_SHUFFLE,
                (
                    "observation",
                    "observation_mask",
                    "action",
                    "action_mask",
                    "reward",
                    "next_observation",
                    "next_observation_mask",
                ),
                destroys_pairing="temporal_order",
            ),
            TransitionViewSpec(
                V_RANDOM_ENCODER,
                (
                    "observation",
                    "observation_mask",
                    "action",
                    "action_mask",
                    "reward",
                    "next_observation",
                    "next_observation_mask",
                ),
            ),
        )
    }
)


@dataclass(frozen=True)
class TransitionViewResult:
    spec: TransitionViewSpec
    channels: Mapping[str, np.ndarray]
    episode_offsets: np.ndarray
    row_source_indices: np.ndarray
    next_source_indices: np.ndarray
    reward_source_indices: np.ndarray
    archived_dataset_digest: str
    padding_value: float
    observation_width: int
    action_width: int
    random_projection_digest: str | None = None

    def __post_init__(self) -> None:
        if self.spec.view_id not in VIEW_REGISTRY or VIEW_REGISTRY[self.spec.view_id] != self.spec:
            raise TransitionViewError("view spec is not the frozen registry entry")
        offsets = _readonly(
            self.episode_offsets,
            name="episode_offsets",
            ndim=1,
            dtype=np.dtype(np.int64),
        )
        rows = int(offsets[-1]) if offsets.size else 0
        if rows <= 0:
            raise TransitionViewError("view result cannot be empty")
        for name in ("observation_width", "action_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TransitionViewError(f"{name} must be a positive integer")
        frozen_channels: dict[str, np.ndarray] = {}
        for name, value in self.channels.items():
            if name not in self.spec.input_channel_allowlist and name != "random_embedding":
                raise TransitionViewError(f"undeclared channel exposed: {name}")
            array = np.asarray(value)
            if array.ndim == 1:
                array = array[:, None]
            frozen = _readonly(array, name=name, ndim=2)
            if frozen.shape[0] != rows:
                raise TransitionViewError(f"channel {name} row count disagrees")
            frozen_channels[name] = frozen
        expected_channels = (
            {"random_embedding"}
            if self.spec.view_id == V_RANDOM_ENCODER
            else set(self.spec.input_channel_allowlist)
        )
        if set(frozen_channels) != expected_channels:
            raise TransitionViewError(
                f"view channels differ: missing={sorted(expected_channels-set(frozen_channels))}, "
                f"unknown={sorted(set(frozen_channels)-expected_channels)}"
            )
        for name in (
            "row_source_indices",
            "next_source_indices",
            "reward_source_indices",
        ):
            indices = _readonly(
                getattr(self, name), name=name, ndim=1, dtype=np.dtype(np.int64)
            )
            if indices.shape != (rows,) or set(indices.tolist()) != set(range(rows)):
                raise TransitionViewError(f"{name} must be a permutation of bank rows")
            object.__setattr__(self, name, indices)
        object.__setattr__(self, "channels", MappingProxyType(frozen_channels))
        object.__setattr__(self, "episode_offsets", offsets)
        object.__setattr__(
            self,
            "archived_dataset_digest",
            _sha256_digest(self.archived_dataset_digest, "archived_dataset_digest"),
        )
        if self.view_id == V_RANDOM_ENCODER:
            if self.random_projection_digest is None:
                raise TransitionViewError(
                    "random encoder control requires a projection digest"
                )
            object.__setattr__(
                self,
                "random_projection_digest",
                _sha256_digest(
                    self.random_projection_digest, "random_projection_digest"
                ),
            )
        elif self.random_projection_digest is not None:
            raise TransitionViewError(
                "only V_RANDOM_ENCODER may carry a projection digest"
            )

    @property
    def view_id(self) -> str:
        return self.spec.view_id

    @property
    def feature_matrix(self) -> np.ndarray:
        order = (
            ("random_embedding",)
            if self.view_id == V_RANDOM_ENCODER
            else self.spec.input_channel_allowlist
        )
        result = np.concatenate([self.channels[name] for name in order], axis=1)
        result.setflags(write=False)
        return result

    @property
    def legacy_packed_matrix(self) -> np.ndarray:
        """Pack controls at the exact historical fixed input width.

        This is the bridge used when replaying the existing CORRO-style
        checkpoint, whose first layer cannot accept the variable-width
        scientific view matrix.  Absent channels are zeroed.  ``V_DIMS_ONLY``
        places its two explicit dimension tokens in the first observation and
        action value slots; ``V_DELTA_ONLY`` places delta in the historical
        next-observation slot.  The random-network control is already an
        output-space control and therefore has no legacy input packing.
        """

        if self.view_id == V_RANDOM_ENCODER:
            raise TransitionViewError(
                "V_RANDOM_ENCODER is an output-space control, not a legacy input"
            )
        rows = int(self.episode_offsets[-1])
        observation_width = self.observation_width
        action_width = self.action_width
        # Historical layout: (o, o_mask, a, a_mask, r, o', o'_mask).
        widths = (
            observation_width,
            observation_width,
            action_width,
            action_width,
            1,
            observation_width,
            observation_width,
        )
        boundaries = np.cumsum((0, *widths))
        packed = np.zeros((rows, int(boundaries[-1])), dtype=np.float32)
        slots = {
            "observation": slice(boundaries[0], boundaries[1]),
            "observation_mask": slice(boundaries[1], boundaries[2]),
            "action": slice(boundaries[2], boundaries[3]),
            "action_mask": slice(boundaries[3], boundaries[4]),
            "reward": slice(boundaries[4], boundaries[5]),
            "next_observation": slice(boundaries[5], boundaries[6]),
            "next_observation_mask": slice(boundaries[6], boundaries[7]),
        }
        for name, destination in slots.items():
            if name in self.channels:
                packed[:, destination] = self.channels[name]
        if self.view_id == V_DIMS_ONLY:
            packed[:, slots["observation"].start] = self.channels[
                "observation_native_dim"
            ][:, 0]
            packed[:, slots["action"].start] = self.channels["action_native_dim"][:, 0]
            packed[:, slots["next_observation"].start] = self.channels[
                "observation_native_dim"
            ][:, 0]
        elif self.view_id == V_DELTA_ONLY:
            packed[:, slots["next_observation"]] = self.channels[
                "observation_delta"
            ]
        packed.setflags(write=False)
        return packed

    @property
    def padding_identity_audit(self) -> Mapping[str, Any]:
        """Expose whether a no-mask view still leaks fixed padding positions."""

        def constant_padding_columns(name: str) -> tuple[int, ...]:
            if name not in self.channels:
                return ()
            values = self.channels[name]
            return tuple(
                int(index)
                for index in np.flatnonzero(
                    np.all(values == self.padding_value, axis=0)
                )
            )

        observation_slots = constant_padding_columns("observation")
        action_slots = constant_padding_columns("action")
        next_slots = constant_padding_columns("next_observation")
        return MappingProxyType(
            {
                "padding_value": float(self.padding_value),
                "stable_observation_padding_slots": observation_slots,
                "stable_action_padding_slots": action_slots,
                "stable_next_observation_padding_slots": next_slots,
                "dimension_identity_inferable": bool(
                    self.view_id == V_NO_MASK
                    and (observation_slots or action_slots or next_slots)
                ),
            }
        )

    @property
    def view_digest(self) -> str:
        return sha256_json(
            {
                "spec_digest": self.spec.digest,
                "archived_dataset_digest": self.archived_dataset_digest,
                "arrays_digest": sha256_ndarrays(
                    {
                        **dict(self.channels),
                        "episode_offsets": self.episode_offsets,
                        "row_source_indices": self.row_source_indices,
                        "next_source_indices": self.next_source_indices,
                        "reward_source_indices": self.reward_source_indices,
                    }
                ),
                "padding_value": self.padding_value,
                "observation_width": self.observation_width,
                "action_width": self.action_width,
                "random_projection_digest": self.random_projection_digest,
            }
        )


@runtime_checkable
class TransitionViewProtocol(Protocol):
    view_id: str
    input_channel_allowlist: tuple[str, ...]

    def apply(
        self, bank: TransitionBank, *, shuffle_seed: int = 0
    ) -> TransitionViewResult: ...


def _nontrivial_permutation(size: int, seed: int) -> np.ndarray:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TransitionViewError("shuffle_seed must be an integer")
    if size < 2:
        raise TransitionViewError("destructive controls require at least two rows")
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(size)
    identity = np.arange(size, dtype=np.int64)
    if np.array_equal(permutation, identity):
        permutation = np.roll(identity, 1)
    return np.asarray(permutation, dtype=np.int64)


def _temporal_permutation(offsets: np.ndarray, seed: int) -> np.ndarray:
    result = np.arange(int(offsets[-1]), dtype=np.int64)
    changed = False
    for episode_index, (start, stop) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        start_i, stop_i = int(start), int(stop)
        if stop_i - start_i < 2:
            continue
        local = _nontrivial_permutation(stop_i - start_i, seed + episode_index)
        result[start_i:stop_i] = start_i + local
        changed = True
    if not changed:
        raise TransitionViewError(
            "temporal shuffle requires at least one multi-transition episode"
        )
    return result


def apply_transition_view(
    bank: TransitionBank,
    view_id: str,
    *,
    shuffle_seed: int = 0,
    random_output_dim: int = 16,
) -> TransitionViewResult:
    """Apply one frozen view without mutating the archived transition bank."""

    if not isinstance(bank, TransitionBank):
        raise TransitionViewError("bank must be a TransitionBank")
    try:
        spec = VIEW_REGISTRY[view_id]
    except KeyError as error:
        raise TransitionViewError(f"unregistered transition view: {view_id!r}") from error
    rows = bank.transition_count
    identity = np.arange(rows, dtype=np.int64)
    row_indices = identity
    next_indices = identity
    reward_indices = identity
    if view_id == V_SHUFFLED_NEXT:
        next_indices = _nontrivial_permutation(rows, shuffle_seed)
    elif view_id == V_SHUFFLED_REWARD:
        reward_indices = _nontrivial_permutation(rows, shuffle_seed)
    elif view_id == V_TEMPORAL_SHUFFLE:
        row_indices = _temporal_permutation(bank.episode_offsets, shuffle_seed)
        next_indices = row_indices
        reward_indices = row_indices

    observation = bank.observation[row_indices]
    observation_mask = bank.observation_mask[row_indices]
    action = bank.action[row_indices]
    action_mask = bank.action_mask[row_indices]
    reward = bank.reward[reward_indices, None]
    next_observation = bank.next_observation[next_indices]
    next_mask = bank.observation_mask[next_indices]

    channels: dict[str, np.ndarray]
    projection_digest: str | None = None
    if view_id in {V_FULL_LEGACY, V_SHUFFLED_NEXT, V_SHUFFLED_REWARD, V_TEMPORAL_SHUFFLE}:
        channels = {
            "observation": observation,
            "observation_mask": observation_mask,
            "action": action,
            "action_mask": action_mask,
            "reward": reward,
            "next_observation": next_observation,
            "next_observation_mask": next_mask,
        }
    elif view_id == V_MASK_ONLY:
        channels = {
            "observation_mask": observation_mask,
            "action_mask": action_mask,
            "next_observation_mask": next_mask,
        }
    elif view_id == V_DIMS_ONLY:
        channels = {
            "observation_native_dim": np.sum(observation_mask, axis=1, keepdims=True),
            "action_native_dim": np.sum(action_mask, axis=1, keepdims=True),
        }
    elif view_id == V_REWARD_ONLY:
        channels = {"reward": reward}
    elif view_id == V_STATE_ONLY:
        channels = {"observation": observation}
    elif view_id == V_ACTION_ONLY:
        channels = {"action": action, "action_mask": action_mask}
    elif view_id == V_STATE_ACTION:
        channels = {"observation": observation, "action": action}
    elif view_id == V_DELTA_ONLY:
        channels = {
            "observation_delta": next_observation - observation,
            "action": action,
        }
    elif view_id == V_REWARD_FREE_TRANSITION:
        channels = {
            "observation": observation,
            "action": action,
            "next_observation": next_observation,
        }
    elif view_id == V_NO_MASK:
        channels = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "next_observation": next_observation,
        }
    elif view_id == V_RANDOM_ENCODER:
        if (
            isinstance(random_output_dim, bool)
            or not isinstance(random_output_dim, int)
            or random_output_dim <= 0
        ):
            raise TransitionViewError("random_output_dim must be a positive integer")
        packed = np.concatenate(
            [
                observation,
                observation_mask,
                action,
                action_mask,
                reward,
                next_observation,
                next_mask,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        rng = np.random.default_rng(int(shuffle_seed))
        scale = 1.0 / np.sqrt(max(1, packed.shape[1]))
        matrix = rng.normal(
            0.0, scale, size=(packed.shape[1], random_output_dim)
        ).astype(np.float32)
        bias = rng.normal(0.0, scale, size=(random_output_dim,)).astype(np.float32)
        projection_digest = sha256_ndarrays({"matrix": matrix, "bias": bias})
        channels = {"random_embedding": np.tanh(packed @ matrix + bias)}
    else:  # pragma: no cover - registry and implementation are co-located
        raise AssertionError(view_id)

    return TransitionViewResult(
        spec=spec,
        channels=channels,
        episode_offsets=bank.episode_offsets,
        row_source_indices=row_indices,
        next_source_indices=next_indices,
        reward_source_indices=reward_indices,
        archived_dataset_digest=str(bank.archived_dataset_digest),
        padding_value=float(bank.padding_value),
        observation_width=int(bank.observation.shape[1]),
        action_width=int(bank.action.shape[1]),
        random_projection_digest=projection_digest,
    )


@dataclass(frozen=True)
class RegisteredTransitionView:
    """Small object adapter satisfying :class:`TransitionViewProtocol`."""

    view_id: str

    def __post_init__(self) -> None:
        if self.view_id not in VIEW_REGISTRY:
            raise TransitionViewError(f"unregistered transition view: {self.view_id!r}")

    @property
    def input_channel_allowlist(self) -> tuple[str, ...]:
        return VIEW_REGISTRY[self.view_id].input_channel_allowlist

    def apply(
        self, bank: TransitionBank, *, shuffle_seed: int = 0
    ) -> TransitionViewResult:
        return apply_transition_view(bank, self.view_id, shuffle_seed=shuffle_seed)
