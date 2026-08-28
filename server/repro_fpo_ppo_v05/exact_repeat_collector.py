#!/usr/bin/env python3
"""Thin v0.5 collector for one exact-repeat common-probe context.

There is deliberately no policy, label, candidate, reward, or oracle input.
The collector uses the audited v03 full-episode Gaussian rollout when it is
available, and otherwise supports an injected ``EnvAdapter`` for tests.  Only
``(observation, action, next_observation)`` is ever published.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npz,
    read_json,
)
from policy_learnware_v0.probe.gaussian import sample_clipped_gaussian_episode_jax
from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    EPISODES_PER_CONTEXT,
    TRANSITIONS_PER_EPISODE,
    VISIBLE_TRANSITIONS_PER_EPISODE,
    BudgetLedger,
    ProbeMembership,
    RewardFreeProbe,
    derive_probe_membership,
    require_budget,
)


Q0_COMMON_GAUSSIAN_OPEN_LOOP = "Q0_COMMON_GAUSSIAN_OPEN_LOOP"
RNG_BACKEND = "jax_threefry_full_episode_v0"
PROTOCOL_SCHEMA = "policy-learnware.v05-common-probe-protocol.v1"
COLLECTION_SCHEMA = "policy-learnware.v05-exact-repeat-collection.v1"
BANK_SCHEMA = "policy-learnware.v05-reward-free-bank.v1"
SEED_SCHEMA = "policy-learnware.v05-exact-repeat-seed-plan.v1"
LEDGERS_SCHEMA = "policy-learnware.v05-budget-ledgers.v1"
EVENT_SCHEMA = "policy-learnware.v05-collection-event.v1"

BANK_FILE = "reward_free_bank.npz"
MEMBERSHIP_FILE = "probe_membership.json"
LEDGERS_FILE = "budget_ledgers.json"
EVENTS_FILE = "execution_events.jsonl"
INDEX_FILE = "index.json"
OUTPUT_FILES = frozenset(
    {BANK_FILE, MEMBERSHIP_FILE, LEDGERS_FILE, EVENTS_FILE, INDEX_FILE}
)
BANK_ARRAYS = frozenset(
    {
        "observation",
        "action",
        "next_observation",
        "episode_offsets",
        "probe_membership_digest",
        "probe_protocol_digest",
        "reward_free_bank_digest",
    }
)


class V05CollectorError(RuntimeError):
    """The requested collection would break the Q0 evidence contract."""


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise V05CollectorError(f"{where} must be a non-empty canonical string")
    return value


def _uint(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise V05CollectorError(f"{where} must be an integer")
    result = int(value)
    if result < 0:
        raise V05CollectorError(f"{where} must be nonnegative")
    return result


def _positive(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise V05CollectorError(f"{where} must be numeric")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise V05CollectorError(f"{where} must be finite and positive")
    return result


def _sha(value: Any, where: str) -> str:
    result = _text(value, where)
    if len(result) != 64 or result != result.lower():
        raise V05CollectorError(f"{where} must be a lowercase SHA-256 digest")
    try:
        int(result, 16)
    except ValueError as error:
        raise V05CollectorError(f"{where} must be a SHA-256 digest") from error
    return result


def q0_probe_protocol(*, sigma: float) -> dict[str, Any]:
    """Canonical sampler card shared by every v0.5 method."""

    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": Q0_COMMON_GAUSSIAN_OPEN_LOOP,
        "action_rule": "clip(sigma * Normal(0, I), action_low, action_high)",
        "action_bounds_source": "frozen_environment_schema",
        "action_tensor_shape": [TRANSITIONS_PER_EPISODE, "action_dim"],
        "probe_type": "clipped_gaussian",
        "probe_sigma": _positive(sigma, "sigma"),
        "probe_rng_backend": RNG_BACKEND,
        "probe_implementation": (
            "policy_learnware_v0.probe.gaussian:" "sample_clipped_gaussian_episode_jax"
        ),
        "feedback_mode": "open_loop",
        "full_episode_draw": True,
        "state_access_for_action": False,
        "reward_access": False,
        "label_access": False,
        "candidate_policy_access": False,
        "candidate_conditioned_steps": 0,
        "reward_queries": 0,
        "episode_count": EPISODES_PER_CONTEXT,
        "native_steps_per_episode": TRANSITIONS_PER_EPISODE,
        "visible_transitions_per_episode": VISIBLE_TRANSITIONS_PER_EPISODE,
        "budget_episodes": list(BUDGET_EPISODES),
    }


def derive_episode_seed_plan(
    *, context_id: str, seed_namespace: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Derive independent reset/action streams by domain-separated SHA-256."""

    context = _text(context_id, "context_id")
    namespace = _text(seed_namespace, "seed_namespace")

    def one(index: int, stream: str) -> int:
        digest = sha256_json(
            {
                "schema": SEED_SCHEMA,
                "seed_namespace": namespace,
                "context_id": context,
                "episode_index": index,
                "stream": stream,
            }
        )
        return int.from_bytes(bytes.fromhex(digest)[:4], "big")

    reset = np.asarray(
        [one(index, "environment-reset") for index in range(32)], dtype=np.int64
    )
    action = np.asarray(
        [one(index, "open-loop-action-tensor") for index in range(32)],
        dtype=np.int64,
    )
    if np.unique(np.concatenate((reset, action))).size != 64:
        raise V05CollectorError("derived reset/action seed streams collide")
    digest = sha256_json(
        {
            "schema": SEED_SCHEMA,
            "seed_namespace": namespace,
            "context_id": context,
            "reset_seeds": reset.tolist(),
            "action_seeds": action.tolist(),
        }
    )
    return reset, action, digest


def _array_digest(probe: RewardFreeProbe) -> str:
    return sha256_ndarrays(
        {
            "observation": probe.observation,
            "action": probe.action,
            "next_observation": probe.next_observation,
            "episode_offsets": probe.episode_offsets,
        }
    )


def _bank_digest(probe: RewardFreeProbe, protocol_digest: str) -> str:
    return sha256_json(
        {
            "schema": BANK_SCHEMA,
            "probe_protocol_digest": protocol_digest,
            "probe_membership_digest": probe.probe_membership_digest,
            "array_digest": _array_digest(probe),
        }
    )


@dataclass(frozen=True)
class CommonProbeCollection:
    """One reward-free 32-episode bank plus its immutable accounting."""

    context_id: str
    seed_namespace: str
    membership_seed: int
    sigma: float
    probe: RewardFreeProbe
    membership: ProbeMembership
    ledgers: tuple[BudgetLedger, ...]
    reset_seeds: np.ndarray
    action_seeds: np.ndarray
    seed_plan_digest: str
    probe_protocol_digest: str
    environment_schema_digest: str
    environment_instance_digest: str | None = None

    def __post_init__(self) -> None:
        context = _text(self.context_id, "context_id")
        namespace = _text(self.seed_namespace, "seed_namespace")
        split_seed = _uint(self.membership_seed, "membership_seed")
        sigma = _positive(self.sigma, "sigma")
        if self.membership.context_id != context or (
            self.membership.split_seed != split_seed
        ):
            raise V05CollectorError("membership identity differs from collection")
        if self.probe.episode_count != 32 or (
            self.probe.membership_digest != self.membership.membership_digest
        ):
            raise V05CollectorError("probe is not the full membership projection")
        ledgers = tuple(self.ledgers)
        if tuple(item.budget_episodes for item in ledgers) != BUDGET_EPISODES:
            raise V05CollectorError("collection does not bind every frozen budget")
        (
            expected_reset,
            expected_action,
            expected_seed_digest,
        ) = derive_episode_seed_plan(context_id=context, seed_namespace=namespace)
        reset = np.asarray(self.reset_seeds, dtype=np.int64)
        action = np.asarray(self.action_seeds, dtype=np.int64)
        if not np.array_equal(reset, expected_reset) or not np.array_equal(
            action, expected_action
        ):
            raise V05CollectorError("episode seed plan changed")
        if _sha(self.seed_plan_digest, "seed_plan_digest") != expected_seed_digest:
            raise V05CollectorError("seed plan digest changed")
        expected_protocol = sha256_json(q0_probe_protocol(sigma=sigma))
        if _sha(self.probe_protocol_digest, "probe_protocol_digest") != (
            expected_protocol
        ):
            raise V05CollectorError("probe protocol digest changed")
        _sha(self.environment_schema_digest, "environment_schema_digest")
        if self.environment_instance_digest is not None:
            _sha(self.environment_instance_digest, "environment_instance_digest")
        reset = np.array(reset, copy=True)
        action = np.array(action, copy=True)
        reset.setflags(write=False)
        action.setflags(write=False)
        object.__setattr__(self, "context_id", context)
        object.__setattr__(self, "seed_namespace", namespace)
        object.__setattr__(self, "membership_seed", split_seed)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "ledgers", ledgers)
        object.__setattr__(self, "reset_seeds", reset)
        object.__setattr__(self, "action_seeds", action)

    @property
    def reward_free_bank_digest(self) -> str:
        return _bank_digest(self.probe, self.probe_protocol_digest)

    def probe_for_budget(self, budget_episodes: int) -> RewardFreeProbe:
        budget = require_budget(budget_episodes)
        stop = budget * VISIBLE_TRANSITIONS_PER_EPISODE
        return RewardFreeProbe(
            observation=self.probe.observation[:stop],
            action=self.probe.action[:stop],
            next_observation=self.probe.next_observation[:stop],
            episode_offsets=self.probe.episode_offsets[: budget + 1],
            probe_membership_digest=self.probe.membership_digest,
        )


def _sample_actions(
    *, seed: int, sigma: float, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Create the complete [H, action_dim] tensor before reset/step."""

    try:
        import jax
    except ImportError as error:  # pragma: no cover - dependency gate
        raise V05CollectorError("Q0 collection requires JAX") from error
    key = (
        jax.random.key(int(seed))
        if hasattr(jax.random, "key")
        else jax.random.PRNGKey(int(seed))  # pragma: no cover
    )
    value = sample_clipped_gaussian_episode_jax(
        key,
        steps=TRANSITIONS_PER_EPISODE,
        action_dim=low.size,
        sigma=sigma,
        action_low=low,
        action_high=high,
    )
    result = np.asarray(jax.device_get(value), dtype=np.float32)
    if result.shape != (TRANSITIONS_PER_EPISODE, low.size) or not np.all(
        np.isfinite(result)
    ):
        raise V05CollectorError("Q0 action tensor is malformed")
    return result


def _generic_rollout(
    adapter: Any,
    *,
    reset_seeds: np.ndarray,
    action_seeds: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real step-by-step fallback for an injected EnvAdapter."""

    schema = adapter.schema
    shape_o = (32, 1000, int(schema.observation_dim))
    shape_a = (32, 1000, int(schema.action_dim))
    observation = np.empty(shape_o, dtype=np.float32)
    action = np.empty(shape_a, dtype=np.float32)
    next_observation = np.empty(shape_o, dtype=np.float32)
    for episode in range(32):
        # Deliberately before reset: neither current nor future state exists.
        episode_actions = _sample_actions(
            seed=int(action_seeds[episode]),
            sigma=sigma,
            low=schema.action_low,
            high=schema.action_high,
        )
        state, current = adapter.reset(int(reset_seeds[episode]))
        current = np.asarray(current, dtype=np.float32)
        for timestep, chosen_action in enumerate(episode_actions):
            next_state, result = adapter.step(state, chosen_action)
            following = np.asarray(result.observation, dtype=np.float32)
            if current.shape != shape_o[2:] or following.shape != shape_o[2:]:
                raise V05CollectorError("environment observation shape changed")
            if timestep < 999 and (result.terminated or result.truncated):
                raise V05CollectorError("environment ended before step 1000")
            observation[episode, timestep] = current
            action[episode, timestep] = chosen_action
            next_observation[episode, timestep] = following
            state, current = next_state, following
            # result.reward is intentionally never read.
    if not all(
        np.all(np.isfinite(array)) for array in (observation, action, next_observation)
    ):
        raise V05CollectorError("environment emitted non-finite values")
    return observation, action, next_observation


def _rollout(
    adapter: Any,
    *,
    reset_seeds: np.ndarray,
    action_seeds: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefer the exact v03 JAX full-episode implementation."""

    collect = getattr(adapter, "collect_clipped_gaussian_batch", None)
    if not callable(collect):
        return _generic_rollout(
            adapter,
            reset_seeds=reset_seeds,
            action_seeds=action_seeds,
            sigma=sigma,
        )
    arrays = collect(
        reset_seeds=reset_seeds,
        probe_seeds=action_seeds,
        sigma=sigma,
        steps=1000,
    )
    if not isinstance(arrays, Mapping) or not {
        "observation",
        "action",
        "next_observation",
    }.issubset(arrays):
        raise V05CollectorError("v03 batch collector omitted Q0 arrays")
    # v03 may return reward/done diagnostics. They are never copied from this
    # local mapping and cannot enter the publication whitelist.
    schema = adapter.schema
    obs = np.asarray(arrays["observation"], dtype=np.float32)
    act = np.asarray(arrays["action"], dtype=np.float32)
    nxt = np.asarray(arrays["next_observation"], dtype=np.float32)
    try:
        obs = obs.reshape(32, 1000, int(schema.observation_dim))
        act = act.reshape(32, 1000, int(schema.action_dim))
        nxt = nxt.reshape(32, 1000, int(schema.observation_dim))
    except ValueError as error:
        raise V05CollectorError("v03 Q0 arrays have the wrong shape") from error
    if not all(np.all(np.isfinite(item)) for item in (obs, act, nxt)):
        raise V05CollectorError("v03 Q0 arrays contain non-finite values")
    return obs, act, nxt


def collect_common_probe(
    *,
    context_id: str,
    env_factory: Callable[[], Any],
    seed_namespace: str,
    membership_seed: int,
    sigma: float,
) -> CommonProbeCollection:
    """Collect Q0 from one fresh environment; no policy API is accepted."""

    context = _text(context_id, "context_id")
    namespace = _text(seed_namespace, "seed_namespace")
    split_seed = _uint(membership_seed, "membership_seed")
    sigma = _positive(sigma, "sigma")
    if not callable(env_factory):
        raise V05CollectorError("env_factory must be callable")
    built = env_factory()
    adapter = getattr(built, "adapter", built)
    for name in ("schema", "reset", "step"):
        if not hasattr(adapter, name):
            raise V05CollectorError(f"environment lacks {name}")
    schema = adapter.schema
    if int(schema.horizon) != 1000:
        raise V05CollectorError("Q0 requires an exact 1000-step horizon")
    low = np.asarray(schema.action_low, dtype=np.float32)
    high = np.asarray(schema.action_high, dtype=np.float32)
    if (
        low.shape != (int(schema.action_dim),)
        or high.shape != low.shape
        or not np.all(np.isfinite(low))
        or not np.all(np.isfinite(high))
        or np.any(low >= high)
    ):
        raise V05CollectorError("environment action bounds are malformed")
    reset, action_seed, seed_digest = derive_episode_seed_plan(
        context_id=context, seed_namespace=namespace
    )
    membership = derive_probe_membership(context, split_seed)
    observation, action, next_observation = _rollout(
        adapter,
        reset_seeds=reset,
        action_seeds=action_seed,
        sigma=sigma,
    )
    probe = RewardFreeProbe.from_full_episodes(
        observation,
        action,
        next_observation,
        membership=membership,
        budget_episodes=32,
    )
    instance_digest = getattr(built, "environment_instance_digest", None)
    return CommonProbeCollection(
        context_id=context,
        seed_namespace=namespace,
        membership_seed=split_seed,
        sigma=sigma,
        probe=probe,
        membership=membership,
        ledgers=tuple(BudgetLedger.for_budget(item) for item in BUDGET_EPISODES),
        reset_seeds=reset,
        action_seeds=action_seed,
        seed_plan_digest=seed_digest,
        probe_protocol_digest=sha256_json(q0_probe_protocol(sigma=sigma)),
        environment_schema_digest=_sha(schema.digest, "environment schema digest"),
        environment_instance_digest=(
            None
            if instance_digest is None
            else _sha(instance_digest, "environment instance digest")
        ),
    )


def _bank_arrays(collection: CommonProbeCollection) -> dict[str, np.ndarray]:
    probe = collection.probe
    return {
        "observation": probe.observation,
        "action": probe.action,
        "next_observation": probe.next_observation,
        "episode_offsets": probe.episode_offsets,
        "probe_membership_digest": np.asarray(probe.membership_digest),
        "probe_protocol_digest": np.asarray(collection.probe_protocol_digest),
        "reward_free_bank_digest": np.asarray(collection.reward_free_bank_digest),
    }


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _events(collection: CommonProbeCollection) -> list[dict[str, Any]]:
    return [
        {
            "schema": EVENT_SCHEMA,
            "sequence": 1,
            "event": "Q0_FULL_EPISODES_COLLECTED",
            "context_id": collection.context_id,
            "probe_protocol_digest": collection.probe_protocol_digest,
            "seed_plan_digest": collection.seed_plan_digest,
            "interaction_cost_steps": 32_000,
            "candidate_conditioned_steps": 0,
            "reward_queries": 0,
        },
        {
            "schema": EVENT_SCHEMA,
            "sequence": 2,
            "event": "REWARD_FREE_MEMBERSHIP_PROJECTED",
            "context_id": collection.context_id,
            "probe_membership_digest": collection.membership.membership_digest,
            "reward_free_bank_digest": collection.reward_free_bank_digest,
            "visible_transition_count": 2_048,
        },
    ]


def _index(
    collection: CommonProbeCollection, files: Mapping[str, str]
) -> dict[str, Any]:
    payload = {
        "schema": COLLECTION_SCHEMA,
        "status": "COMPLETE",
        "context_id": collection.context_id,
        "seed_namespace": collection.seed_namespace,
        "membership_seed": collection.membership_seed,
        "seed_plan_digest": collection.seed_plan_digest,
        "reset_seeds": collection.reset_seeds.tolist(),
        "action_seeds": collection.action_seeds.tolist(),
        "probe_protocol_id": Q0_COMMON_GAUSSIAN_OPEN_LOOP,
        "probe_protocol_digest": collection.probe_protocol_digest,
        "probe_sigma": collection.sigma,
        "probe_rng_backend": RNG_BACKEND,
        "probe_membership_digest": collection.membership.membership_digest,
        "reward_free_bank_digest": collection.reward_free_bank_digest,
        "environment_schema_digest": collection.environment_schema_digest,
        "environment_instance_digest": collection.environment_instance_digest,
        "budgets": list(BUDGET_EPISODES),
        "episode_count": 32,
        "native_steps_per_episode": 1000,
        "visible_transitions_per_episode": 64,
        "visible_transition_count": 2_048,
        "interaction_cost_steps": 32_000,
        "candidate_conditioned_steps": 0,
        "reward_queries": 0,
        "files": dict(files),
    }
    payload["index_digest"] = sha256_json(payload)
    return payload


def publish_collection(
    collection: CommonProbeCollection,
    *,
    output_dir: str | Path,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Atomically publish one immutable context directory."""

    destination = Path(output_dir).expanduser().absolute()
    if destination.exists():
        if not resume:
            raise V05CollectorError(f"refusing to overwrite: {destination}")
        restored = load_published_collection(destination)
        requested = (
            collection.context_id,
            collection.seed_namespace,
            collection.membership_seed,
            collection.probe_protocol_digest,
            collection.environment_schema_digest,
            collection.environment_instance_digest,
            collection.reward_free_bank_digest,
        )
        observed = (
            restored.context_id,
            restored.seed_namespace,
            restored.membership_seed,
            restored.probe_protocol_digest,
            restored.environment_schema_digest,
            restored.environment_instance_digest,
            restored.reward_free_bank_digest,
        )
        if observed != requested:
            raise V05CollectorError("resume bank differs from requested collection")
        value = read_json(destination / INDEX_FILE)
        if not isinstance(value, Mapping):
            raise V05CollectorError("resume index is malformed")
        return value
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise V05CollectorError("output parent cannot be a symlink")
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        files = {
            BANK_FILE: atomic_write_npz(stage / BANK_FILE, _bank_arrays(collection)),
            MEMBERSHIP_FILE: atomic_write_json(
                stage / MEMBERSHIP_FILE, collection.membership.to_dict()
            ),
        }
        files[LEDGERS_FILE] = atomic_write_json(
            stage / LEDGERS_FILE,
            {
                "schema": LEDGERS_SCHEMA,
                "probe_protocol_digest": collection.probe_protocol_digest,
                "probe_membership_digest": collection.membership.membership_digest,
                "ledgers": [item.to_dict() for item in collection.ledgers],
            },
        )
        files[EVENTS_FILE] = atomic_write_bytes(
            stage / EVENTS_FILE, _jsonl(_events(collection))
        )
        index = _index(collection, files)
        atomic_write_json(stage / INDEX_FILE, index)
        if destination.exists():
            raise V05CollectorError("another writer published this collection")
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return index


def _safe_existing(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise V05CollectorError("collection directory is absent or unsafe")
    names = {item.name for item in path.iterdir()}
    if names != OUTPUT_FILES:
        raise V05CollectorError("collection directory has unexpected files")
    for name in names:
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise V05CollectorError(f"unsafe collection artifact: {item}")
    value = read_json(path / INDEX_FILE)
    if not isinstance(value, Mapping) or value.get("schema") != COLLECTION_SCHEMA:
        raise V05CollectorError("collection index has the wrong schema")
    unsigned = {key: item for key, item in value.items() if key != "index_digest"}
    if value.get("index_digest") != sha256_json(unsigned):
        raise V05CollectorError("collection index digest changed")
    files = value.get("files")
    if not isinstance(files, Mapping) or set(files) != (OUTPUT_FILES - {INDEX_FILE}):
        raise V05CollectorError("collection file bindings are incomplete")
    for name, digest in files.items():
        if sha256_file(path / name) != _sha(digest, f"files[{name}]"):
            raise V05CollectorError(f"collection artifact changed: {name}")
    return value


def load_published_collection(output_dir: str | Path) -> CommonProbeCollection:
    """Verify hashes, whitelists, nesting, costs, and semantic bank digest."""

    root = Path(output_dir).expanduser().absolute()
    index = _safe_existing(root)
    membership_value = read_json(root / MEMBERSHIP_FILE)
    if not isinstance(membership_value, Mapping):
        raise V05CollectorError("probe membership is malformed")
    membership = ProbeMembership.from_dict(membership_value)
    with np.load(root / BANK_FILE, allow_pickle=False) as arrays:
        if set(arrays.files) != BANK_ARRAYS:
            raise V05CollectorError("reward-free bank exposes unexpected channels")
        membership_digest = str(arrays["probe_membership_digest"].item())
        protocol_digest = str(arrays["probe_protocol_digest"].item())
        claimed_bank_digest = str(arrays["reward_free_bank_digest"].item())
        probe = RewardFreeProbe(
            observation=arrays["observation"],
            action=arrays["action"],
            next_observation=arrays["next_observation"],
            episode_offsets=arrays["episode_offsets"],
            probe_membership_digest=membership_digest,
        )
    ledger_value = read_json(root / LEDGERS_FILE)
    if not isinstance(ledger_value, Mapping) or not isinstance(
        ledger_value.get("ledgers"), list
    ):
        raise V05CollectorError("budget ledger artifact is malformed")
    if (
        ledger_value.get("schema") != LEDGERS_SCHEMA
        or ledger_value.get("probe_protocol_digest") != protocol_digest
        or ledger_value.get("probe_membership_digest") != membership_digest
    ):
        raise V05CollectorError("budget ledger binding changed")
    ledgers = tuple(
        BudgetLedger(
            schema=row["schema"],
            budget_episodes=row["budget_episodes"],
            visible_transition_count=row["visible_transition_count"],
            interaction_cost_steps=row["interaction_cost_steps"],
            candidate_conditioned_steps=row["candidate_conditioned_steps"],
            reward_queries=row["reward_queries"],
        )
        for row in ledger_value["ledgers"]
    )
    collection = CommonProbeCollection(
        context_id=index["context_id"],
        seed_namespace=index["seed_namespace"],
        membership_seed=index["membership_seed"],
        sigma=index["probe_sigma"],
        probe=probe,
        membership=membership,
        ledgers=ledgers,
        reset_seeds=np.asarray(index["reset_seeds"], dtype=np.int64),
        action_seeds=np.asarray(index["action_seeds"], dtype=np.int64),
        seed_plan_digest=index["seed_plan_digest"],
        probe_protocol_digest=protocol_digest,
        environment_schema_digest=index["environment_schema_digest"],
        environment_instance_digest=index.get("environment_instance_digest"),
    )
    checks = {
        "status": "COMPLETE",
        "probe_protocol_id": Q0_COMMON_GAUSSIAN_OPEN_LOOP,
        "probe_protocol_digest": collection.probe_protocol_digest,
        "probe_membership_digest": collection.membership.membership_digest,
        "reward_free_bank_digest": collection.reward_free_bank_digest,
        "probe_rng_backend": RNG_BACKEND,
        "budgets": list(BUDGET_EPISODES),
        "episode_count": EPISODES_PER_CONTEXT,
        "native_steps_per_episode": TRANSITIONS_PER_EPISODE,
        "visible_transitions_per_episode": VISIBLE_TRANSITIONS_PER_EPISODE,
        "visible_transition_count": (
            EPISODES_PER_CONTEXT * VISIBLE_TRANSITIONS_PER_EPISODE
        ),
        "interaction_cost_steps": (EPISODES_PER_CONTEXT * TRANSITIONS_PER_EPISODE),
        "candidate_conditioned_steps": 0,
        "reward_queries": 0,
    }
    if any(index.get(key) != expected for key, expected in checks.items()):
        raise V05CollectorError("collection index semantic binding changed")
    if claimed_bank_digest != collection.reward_free_bank_digest:
        raise V05CollectorError("bank digest differs from its arrays")
    return collection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v02-config", type=Path, default=Path("configs/v02_freeze_ready.yaml")
    )
    parser.add_argument(
        "--cp0-config", type=Path, default=Path("configs/dmc6_outer006_v0.yaml")
    )
    parser.add_argument("--context-id", required=True)
    parser.add_argument("--seed-namespace", required=True)
    parser.add_argument("--membership-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = _text(args.context_id, "context_id")
    namespace = _text(args.seed_namespace, "seed_namespace")
    split_seed = _uint(args.membership_seed, "membership_seed")

    # Lazy import keeps the injectable collector free of MuJoCo dependencies.
    from policy_learnware_v0.v02.axes import SOURCE_ROLE
    from server.repro_fpo_ppo_v03.dynamics_probe_collector import (
        build_environment,
        build_variant_factory,
        load_collection_plan,
    )

    plan = load_collection_plan(args.v02_config, args.cp0_config)
    matches = tuple(item for item in plan.contexts if item.context_id == context)
    if len(matches) != 1 or matches[0].role != SOURCE_ROLE:
        raise V05CollectorError("context must be one frozen source anchor")
    if plan.cp0_config.probe.type != "clipped_gaussian" or (
        plan.cp0_config.probe.rng_backend != RNG_BACKEND
    ):
        raise V05CollectorError("CP0 sampler differs from Q0")
    sigma = _positive(plan.cp0_config.probe.sigma, "CP0 sigma")
    if args.output_dir.exists():
        if not args.resume:
            raise V05CollectorError(f"refusing to overwrite: {args.output_dir}")
        restored = load_published_collection(args.output_dir)
        if (
            restored.context_id,
            restored.seed_namespace,
            restored.membership_seed,
            restored.sigma,
        ) != (context, namespace, split_seed, sigma):
            raise V05CollectorError("resume inputs differ from existing collection")
        status = "RESUMED"
        collection = restored
    else:
        variant_factory = build_variant_factory(plan)

        def env_factory() -> Any:
            return build_environment(
                plan, matches[0], factory=variant_factory, jit=False
            )

        collection = collect_common_probe(
            context_id=context,
            env_factory=env_factory,
            seed_namespace=namespace,
            membership_seed=split_seed,
            sigma=sigma,
        )
        publish_collection(collection, output_dir=args.output_dir)
        status = "COMPLETE"
    print(
        json.dumps(
            {
                "status": status,
                "context_id": context,
                "output_dir": str(args.output_dir.expanduser().absolute()),
                "probe_protocol_digest": collection.probe_protocol_digest,
                "probe_membership_digest": collection.membership.membership_digest,
                "reward_free_bank_digest": collection.reward_free_bank_digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Q0_COMMON_GAUSSIAN_OPEN_LOOP",
    "CommonProbeCollection",
    "V05CollectorError",
    "collect_common_probe",
    "derive_episode_seed_plan",
    "load_published_collection",
    "publish_collection",
    "q0_probe_protocol",
]
