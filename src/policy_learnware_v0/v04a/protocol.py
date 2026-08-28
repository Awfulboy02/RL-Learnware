"""Frozen fixed-probe protocol primitives for the v0.4a BI0 panel.

The objects in this module deliberately contain no environment or oracle
logic.  They make the evidence membership, target-interaction accounting,
tie-breaking, and pre-oracle ranking seal explicit and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ..hashing import canonical_json_bytes, sha256_bytes, sha256_json


BUDGET_EPISODES = (1, 2, 4, 8, 16, 24, 32)
EPISODES_PER_CONTEXT = 32
TRANSITIONS_PER_EPISODE = 1_000
VISIBLE_TRANSITIONS_PER_EPISODE = 64

MEMBERSHIP_SCHEMA = "policy-learnware.v04a-probe-membership.v1"
LEDGER_SCHEMA = "policy-learnware.v04a-budget-ledger.v1"
RANKING_SEAL_SCHEMA = "policy-learnware.v04a-ranking-seal.v1"
_TIE_DOMAIN = "v04a-bpr-tie-v1"


class V04AProtocolError(ValueError):
    """A v0.4a protocol value is malformed or breaks a frozen invariant."""


def _canonical_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V04AProtocolError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    result = _canonical_string(value, where)
    if len(result) != 64 or result != result.lower():
        raise V04AProtocolError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise V04AProtocolError(
            f"{where} must be a lowercase SHA-256 digest"
        ) from error
    return result


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise V04AProtocolError(f"{where} must be an integer")
    result = int(value)
    if result < minimum:
        raise V04AProtocolError(f"{where} must be >= {minimum}")
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V04AProtocolError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V04AProtocolError(f"{where} must be finite")
    return result


def _hash_rank(payload: Mapping[str, Any]) -> str:
    return sha256_json(payload)


@dataclass(frozen=True, order=True)
class TransitionIndex:
    """One frozen visible transition in an original full-episode bank."""

    episode_index: int
    native_timestep: int
    flat_index: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_index",
            _integer(self.episode_index, "episode_index"),
        )
        object.__setattr__(
            self,
            "native_timestep",
            _integer(self.native_timestep, "native_timestep"),
        )
        object.__setattr__(self, "flat_index", _integer(self.flat_index, "flat_index"))

    def to_dict(self) -> dict[str, int]:
        return {
            "episode_index": self.episode_index,
            "native_timestep": self.native_timestep,
            "flat_index": self.flat_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransitionIndex":
        return cls(
            episode_index=payload["episode_index"],
            native_timestep=payload["native_timestep"],
            flat_index=payload["flat_index"],
        )


@dataclass(frozen=True)
class ProbeMembership:
    """The complete 32-episode permutation and per-episode visible indices."""

    context_id: str
    split_seed: int
    episode_order: tuple[int, ...]
    episode_entries: tuple[tuple[TransitionIndex, ...], ...]
    episode_length: int = TRANSITIONS_PER_EPISODE
    samples_per_episode: int = VISIBLE_TRANSITIONS_PER_EPISODE
    schema: str = MEMBERSHIP_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MEMBERSHIP_SCHEMA:
            raise V04AProtocolError("unsupported probe membership schema")
        object.__setattr__(
            self, "context_id", _canonical_string(self.context_id, "context_id")
        )
        object.__setattr__(self, "split_seed", _integer(self.split_seed, "split_seed"))
        episode_length = _integer(self.episode_length, "episode_length", minimum=2)
        samples = _integer(self.samples_per_episode, "samples_per_episode", minimum=2)
        if (
            episode_length != TRANSITIONS_PER_EPISODE
            or samples != VISIBLE_TRANSITIONS_PER_EPISODE
        ):
            raise V04AProtocolError(
                "the v0.4a primary membership requires 1000 native steps and 64 visible transitions per episode"
            )
        object.__setattr__(self, "episode_length", episode_length)
        object.__setattr__(self, "samples_per_episode", samples)

        order = tuple(
            _integer(item, "episode_order item") for item in self.episode_order
        )
        if len(order) != EPISODES_PER_CONTEXT or set(order) != set(
            range(EPISODES_PER_CONTEXT)
        ):
            raise V04AProtocolError(
                "episode_order must be a permutation of the 32 original episodes"
            )
        object.__setattr__(self, "episode_order", order)

        entries = tuple(tuple(group) for group in self.episode_entries)
        if len(entries) != EPISODES_PER_CONTEXT:
            raise V04AProtocolError(
                "episode_entries must have one group per original episode"
            )
        normalized: list[tuple[TransitionIndex, ...]] = []
        for episode_index, raw_group in enumerate(entries):
            group = tuple(
                item
                if isinstance(item, TransitionIndex)
                else TransitionIndex.from_dict(item)
                for item in raw_group
            )
            if len(group) != samples:
                raise V04AProtocolError(
                    "each original episode must have samples_per_episode entries"
                )
            timesteps = tuple(item.native_timestep for item in group)
            if timesteps != tuple(sorted(set(timesteps))):
                raise V04AProtocolError(
                    "native timesteps must be unique and sorted within an episode"
                )
            if timesteps[0] != 0 or timesteps[-1] != episode_length - 1:
                raise V04AProtocolError(
                    "each episode membership must include its first and last timestep"
                )
            for item in group:
                expected_flat = episode_index * episode_length + item.native_timestep
                if (
                    item.episode_index != episode_index
                    or item.flat_index != expected_flat
                ):
                    raise V04AProtocolError(
                        "transition entry does not match its original episode/flat index"
                    )
            normalized.append(group)
        object.__setattr__(self, "episode_entries", tuple(normalized))

    @property
    def episode_permutation(self) -> tuple[int, ...]:
        """Compatibility alias for the frozen episode order."""

        return self.episode_order

    @property
    def entries(self) -> tuple[TransitionIndex, ...]:
        """All visible indices, ordered by frozen episode order then timestep."""

        return tuple(
            item
            for episode_index in self.episode_order
            for item in self.episode_entries[episode_index]
        )

    def indices_for_episode(
        self, original_episode_index: int
    ) -> tuple[TransitionIndex, ...]:
        episode_index = _integer(original_episode_index, "original_episode_index")
        if episode_index >= EPISODES_PER_CONTEXT:
            raise V04AProtocolError("original_episode_index is outside [0, 32)")
        return self.episode_entries[episode_index]

    def episode_indices_for_budget(self, budget_episodes: int) -> tuple[int, ...]:
        budget = require_budget(budget_episodes)
        return self.episode_order[:budget]

    def for_budget(self, budget_episodes: int) -> tuple[TransitionIndex, ...]:
        """Return the strict prefix membership for one registered budget."""

        return tuple(
            item
            for episode_index in self.episode_indices_for_budget(budget_episodes)
            for item in self.episode_entries[episode_index]
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "context_id": self.context_id,
            "split_seed": self.split_seed,
            "episode_length": self.episode_length,
            "samples_per_episode": self.samples_per_episode,
            "episode_order": list(self.episode_order),
            "episode_entries": [
                [entry.to_dict() for entry in group] for group in self.episode_entries
            ],
        }

    @property
    def probe_membership_digest(self) -> str:
        return sha256_json(self._payload())

    @property
    def membership_digest(self) -> str:
        return self.probe_membership_digest

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["probe_membership_digest"] = self.probe_membership_digest
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProbeMembership":
        result = cls(
            schema=payload.get("schema", MEMBERSHIP_SCHEMA),
            context_id=payload["context_id"],
            split_seed=payload["split_seed"],
            episode_length=payload["episode_length"],
            samples_per_episode=payload["samples_per_episode"],
            episode_order=tuple(payload["episode_order"]),
            episode_entries=tuple(
                tuple(TransitionIndex.from_dict(item) for item in group)
                for group in payload["episode_entries"]
            ),
        )
        claimed = payload.get("probe_membership_digest")
        if (
            claimed is not None
            and _digest(claimed, "probe_membership_digest")
            != result.probe_membership_digest
        ):
            raise V04AProtocolError("probe membership digest does not match payload")
        return result


def derive_probe_membership(
    context_id: str,
    split_seed: int,
    *,
    episode_count: int = EPISODES_PER_CONTEXT,
    episode_length: int = TRANSITIONS_PER_EPISODE,
    samples_per_episode: int = VISIBLE_TRANSITIONS_PER_EPISODE,
) -> ProbeMembership:
    """Derive the frozen membership using domain-separated SHA-256 ordering.

    Hash sorting, rather than a library PRNG, keeps membership byte-identical
    across NumPy versions.  The first and final native timestep are inserted
    explicitly; the remaining timesteps are selected by a separate hash rank.
    """

    context = _canonical_string(context_id, "context_id")
    seed = _integer(split_seed, "split_seed")
    count = _integer(episode_count, "episode_count", minimum=1)
    length = _integer(episode_length, "episode_length", minimum=2)
    samples = _integer(samples_per_episode, "samples_per_episode", minimum=2)
    if count != EPISODES_PER_CONTEXT:
        raise V04AProtocolError("the v0.4a primary protocol requires 32 episodes")
    if length != TRANSITIONS_PER_EPISODE or samples != VISIBLE_TRANSITIONS_PER_EPISODE:
        raise V04AProtocolError(
            "the v0.4a primary protocol requires 1000 native steps and 64 visible transitions per episode"
        )

    common = {
        "context_id": context,
        "split_seed": seed,
        "schema": MEMBERSHIP_SCHEMA,
    }
    episode_order = tuple(
        sorted(
            range(count),
            key=lambda episode_index: (
                _hash_rank(
                    {
                        **common,
                        "domain": "episode-order",
                        "episode_index": episode_index,
                    }
                ),
                episode_index,
            ),
        )
    )
    episode_entries: list[tuple[TransitionIndex, ...]] = []
    for episode_index in range(count):
        interior = sorted(
            range(1, length - 1),
            key=lambda timestep: (
                _hash_rank(
                    {
                        **common,
                        "domain": "native-timestep",
                        "episode_index": episode_index,
                        "native_timestep": timestep,
                    }
                ),
                timestep,
            ),
        )[: samples - 2]
        selected = tuple(sorted((0, *interior, length - 1)))
        episode_entries.append(
            tuple(
                TransitionIndex(
                    episode_index=episode_index,
                    native_timestep=timestep,
                    flat_index=episode_index * length + timestep,
                )
                for timestep in selected
            )
        )
    return ProbeMembership(
        context_id=context,
        split_seed=seed,
        episode_order=episode_order,
        episode_entries=tuple(episode_entries),
        episode_length=length,
        samples_per_episode=samples,
    )


def require_budget(value: Any) -> int:
    budget = _integer(value, "budget_episodes", minimum=1)
    if budget not in BUDGET_EPISODES:
        raise V04AProtocolError(f"budget_episodes must be one of {BUDGET_EPISODES}")
    return budget


def _readonly_matrix(value: Any, where: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[0] <= 0:
        raise V04AProtocolError(f"{where} must be a non-empty numeric matrix")
    result = np.ascontiguousarray(raw, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise V04AProtocolError(f"{where} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RewardFreeProbe:
    """The only target evidence view visible to BI0 selectors.

    Reward and termination arrays cannot be attached to this value object.
    Episode offsets refer to the concatenated visible 64-transition episodes.
    """

    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    episode_offsets: np.ndarray
    probe_membership_digest: str

    def __post_init__(self) -> None:
        observation = _readonly_matrix(self.observation, "observation")
        action = _readonly_matrix(self.action, "action")
        next_observation = _readonly_matrix(self.next_observation, "next_observation")
        if next_observation.shape != observation.shape:
            raise V04AProtocolError(
                "next_observation must have the same shape as observation"
            )
        if action.shape[0] != observation.shape[0]:
            raise V04AProtocolError("action and observation row counts disagree")

        raw_offsets = np.asarray(self.episode_offsets)
        if raw_offsets.dtype.kind not in "iu" or raw_offsets.ndim != 1:
            raise V04AProtocolError(
                "episode_offsets must be a one-dimensional integer array"
            )
        offsets = np.ascontiguousarray(raw_offsets, dtype=np.int64).copy()
        if (
            offsets.size < 2
            or offsets[0] != 0
            or offsets[-1] != observation.shape[0]
            or np.any(np.diff(offsets) != VISIBLE_TRANSITIONS_PER_EPISODE)
        ):
            raise V04AProtocolError(
                "episode_offsets must partition the view into 64-transition episodes"
            )
        episode_count = int(offsets.size - 1)
        if episode_count not in BUDGET_EPISODES:
            raise V04AProtocolError("probe episode count is not a registered budget")
        offsets.setflags(write=False)

        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "next_observation", next_observation)
        object.__setattr__(self, "episode_offsets", offsets)
        object.__setattr__(
            self,
            "probe_membership_digest",
            _digest(self.probe_membership_digest, "probe_membership_digest"),
        )

    @property
    def membership_digest(self) -> str:
        return self.probe_membership_digest

    @property
    def transition_count(self) -> int:
        return int(self.observation.shape[0])

    @property
    def episode_count(self) -> int:
        return int(self.episode_offsets.size - 1)

    @property
    def budget_episodes(self) -> int:
        return self.episode_count

    def episode_slice(self, index: int) -> slice:
        episode_index = _integer(index, "episode index")
        if episode_index >= self.episode_count:
            raise V04AProtocolError("episode index is outside this probe view")
        return slice(
            int(self.episode_offsets[episode_index]),
            int(self.episode_offsets[episode_index + 1]),
        )

    @classmethod
    def from_full_episodes(
        cls,
        observation: Any,
        action: Any,
        next_observation: Any,
        *,
        membership: ProbeMembership,
        budget_episodes: int,
    ) -> "RewardFreeProbe":
        """Select a reward-free nested prefix from flat or episode-major arrays."""

        if not isinstance(membership, ProbeMembership):
            raise V04AProtocolError("membership has the wrong type")
        budget = require_budget(budget_episodes)

        def flatten(value: Any, where: str) -> np.ndarray:
            raw = np.asarray(value)
            if raw.dtype.kind not in "iuf" or raw.ndim not in {2, 3}:
                raise V04AProtocolError(
                    f"{where} must be [episode,timestep,feature] or [row,feature]"
                )
            if raw.ndim == 3:
                expected = (EPISODES_PER_CONTEXT, membership.episode_length)
                if raw.shape[:2] != expected:
                    raise V04AProtocolError(
                        f"{where} episode/timestep shape must equal {expected}"
                    )
                raw = raw.reshape((-1, raw.shape[-1]))
            expected_rows = EPISODES_PER_CONTEXT * membership.episode_length
            if raw.shape[0] != expected_rows:
                raise V04AProtocolError(
                    f"{where} must contain exactly {expected_rows} full-bank rows"
                )
            return raw

        observation_flat = flatten(observation, "observation")
        action_flat = flatten(action, "action")
        next_flat = flatten(next_observation, "next_observation")
        selected = membership.for_budget(budget)
        flat_indices = np.asarray(
            [entry.flat_index for entry in selected], dtype=np.int64
        )
        offsets = np.arange(
            0,
            (budget + 1) * membership.samples_per_episode,
            membership.samples_per_episode,
            dtype=np.int64,
        )
        return cls(
            observation=observation_flat[flat_indices],
            action=action_flat[flat_indices],
            next_observation=next_flat[flat_indices],
            episode_offsets=offsets,
            probe_membership_digest=membership.probe_membership_digest,
        )


@dataclass(frozen=True)
class BudgetLedger:
    """The non-interchangeable visible-evidence and interaction costs."""

    budget_episodes: int
    visible_transition_count: int
    interaction_cost_steps: int
    candidate_conditioned_steps: int = 0
    reward_queries: int = 0
    schema: str = LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != LEDGER_SCHEMA:
            raise V04AProtocolError("unsupported budget ledger schema")
        budget = require_budget(self.budget_episodes)
        object.__setattr__(self, "budget_episodes", budget)
        visible = _integer(self.visible_transition_count, "visible_transition_count")
        interaction = _integer(self.interaction_cost_steps, "interaction_cost_steps")
        candidate = _integer(
            self.candidate_conditioned_steps, "candidate_conditioned_steps"
        )
        rewards = _integer(self.reward_queries, "reward_queries")
        if visible != budget * VISIBLE_TRANSITIONS_PER_EPISODE:
            raise V04AProtocolError(
                "visible_transition_count must equal budget_episodes * 64"
            )
        if interaction != budget * TRANSITIONS_PER_EPISODE:
            raise V04AProtocolError(
                "interaction_cost_steps must equal budget_episodes * 1000"
            )
        if candidate != 0 or rewards != 0:
            raise V04AProtocolError(
                "BI0 fixed-probe ledgers forbid candidate-conditioned steps and rewards"
            )
        object.__setattr__(self, "visible_transition_count", visible)
        object.__setattr__(self, "interaction_cost_steps", interaction)
        object.__setattr__(self, "candidate_conditioned_steps", candidate)
        object.__setattr__(self, "reward_queries", rewards)

    @classmethod
    def for_budget(cls, budget_episodes: int) -> "BudgetLedger":
        budget = require_budget(budget_episodes)
        return cls(
            budget_episodes=budget,
            visible_transition_count=budget * VISIBLE_TRANSITIONS_PER_EPISODE,
            interaction_cost_steps=budget * TRANSITIONS_PER_EPISODE,
        )

    @property
    def total_target_steps(self) -> int:
        return self.interaction_cost_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "budget_episodes": self.budget_episodes,
            "visible_transition_count": self.visible_transition_count,
            "interaction_cost_steps": self.interaction_cost_steps,
            "candidate_conditioned_steps": self.candidate_conditioned_steps,
            "reward_queries": self.reward_queries,
            "total_target_steps": self.total_target_steps,
        }


def canonical_tie_token(config_digest: str) -> str:
    """Return ``sha256(config_digest || 'v04a-bpr-tie-v1')``."""

    digest = _digest(config_digest, "config_digest")
    return hashlib.sha256((digest + _TIE_DOMAIN).encode("utf-8")).hexdigest()


def tie_break_key(config_digest: str, opaque_id: str) -> str:
    token = canonical_tie_token(config_digest)
    identifier = _canonical_string(opaque_id, "opaque_id")
    return hashlib.sha256((token + identifier).encode("utf-8")).hexdigest()


def break_tie(config_digest: str, opaque_ids: Sequence[str]) -> str:
    identifiers = tuple(
        _canonical_string(identifier, "opaque_id") for identifier in opaque_ids
    )
    if not identifiers:
        raise V04AProtocolError("cannot break an empty tie")
    if len(set(identifiers)) != len(identifiers):
        raise V04AProtocolError("tie IDs must be unique")
    return min(identifiers, key=lambda item: (tie_break_key(config_digest, item), item))


def stable_argmax(scores: Mapping[str, Any], config_digest: str) -> str:
    if not isinstance(scores, Mapping) or not scores:
        raise V04AProtocolError("scores must be a non-empty mapping")
    normalized = {
        _canonical_string(identifier, "score ID"): _finite(value, "score")
        for identifier, value in scores.items()
    }
    maximum = max(normalized.values())
    tied = tuple(
        identifier for identifier, value in normalized.items() if value == maximum
    )
    return tied[0] if len(tied) == 1 else break_tie(config_digest, tied)


@dataclass(frozen=True)
class RankingSeal:
    """Canonical ranking bytes and their digest, frozen before oracle access."""

    canonical_rankings_bytes: bytes
    rankings_digest: str
    schema: str = RANKING_SEAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RANKING_SEAL_SCHEMA:
            raise V04AProtocolError("unsupported ranking seal schema")
        if not isinstance(self.canonical_rankings_bytes, bytes):
            raise V04AProtocolError("canonical_rankings_bytes must be bytes")
        claimed = _digest(self.rankings_digest, "rankings_digest")
        actual = sha256_bytes(self.canonical_rankings_bytes)
        if claimed != actual:
            raise V04AProtocolError("ranking seal digest does not match sealed bytes")
        try:
            decoded = json.loads(self.canonical_rankings_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise V04AProtocolError("sealed rankings are not canonical JSON") from error
        if canonical_json_bytes(decoded) != self.canonical_rankings_bytes:
            raise V04AProtocolError("sealed ranking bytes are not canonical JSON bytes")

    @property
    def rankings(self) -> Any:
        return json.loads(self.canonical_rankings_bytes.decode("utf-8"))

    def verify(self, rankings: Any) -> bool:
        verify_ranking_seal(self, rankings)
        return True

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "canonical_rankings_json": self.canonical_rankings_bytes.decode("utf-8"),
            "rankings_digest": self.rankings_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RankingSeal":
        raw = payload.get("canonical_rankings_json")
        if not isinstance(raw, str):
            raise V04AProtocolError("canonical_rankings_json must be a string")
        return cls(
            schema=payload.get("schema", RANKING_SEAL_SCHEMA),
            canonical_rankings_bytes=raw.encode("utf-8"),
            rankings_digest=payload["rankings_digest"],
        )


def seal_rankings(rankings: Any) -> RankingSeal:
    canonical = canonical_json_bytes(rankings)
    return RankingSeal(
        canonical_rankings_bytes=canonical,
        rankings_digest=sha256_bytes(canonical),
    )


def verify_ranking_seal(seal: RankingSeal, rankings: Any) -> bool:
    if not isinstance(seal, RankingSeal):
        raise V04AProtocolError("seal has the wrong type")
    canonical = canonical_json_bytes(rankings)
    if canonical != seal.canonical_rankings_bytes:
        raise V04AProtocolError("rankings do not match the pre-oracle sealed bytes")
    if sha256_bytes(canonical) != seal.rankings_digest:
        raise V04AProtocolError("rankings do not match the pre-oracle digest")
    return True


__all__ = [
    "BUDGET_EPISODES",
    "EPISODES_PER_CONTEXT",
    "TRANSITIONS_PER_EPISODE",
    "VISIBLE_TRANSITIONS_PER_EPISODE",
    "BudgetLedger",
    "ProbeMembership",
    "RankingSeal",
    "RewardFreeProbe",
    "TransitionIndex",
    "V04AProtocolError",
    "break_tie",
    "canonical_tie_token",
    "derive_probe_membership",
    "require_budget",
    "seal_rankings",
    "stable_argmax",
    "tie_break_key",
    "verify_ranking_seal",
]
