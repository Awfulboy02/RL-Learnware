"""Frozen native-policy loader with lazy upstream imports."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .bundle import PolicyBundleMetadata, validate_bundle


class RuntimeAdapterUnavailable(RuntimeError):
    """The locked upstream FPO/PPO runtime could not be reconstructed."""


class FrozenPolicy(Protocol):
    """Native frozen policy interface; PRNG ownership is always explicit."""

    observation_dim: int
    action_dim: int

    def act(
        self,
        observation: Any,
        key: Any,
        *,
        deterministic: bool = True,
    ) -> tuple[Any, Any]: ...


RuntimeFactory = Callable[
    [PolicyBundleMetadata, Mapping[str, np.ndarray], Mapping[str, np.ndarray], Path],
    Any,
]


@dataclass(frozen=True)
class _RestoredRuntime:
    state: Any
    runtime_receipt: Mapping[str, Any]


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _ordered_actor(actor: Mapping[str, np.ndarray], jnp: Any) -> tuple[tuple[Any, Any], ...]:
    kernel_names = sorted(name for name in actor if name.endswith("_kernel"))
    return tuple(
        (
            jnp.asarray(actor[kernel_name]),
            jnp.asarray(actor[kernel_name.replace("_kernel", "_bias")]),
        )
        for kernel_name in kernel_names
    )


def _default_runtime_factory(
    metadata: PolicyBundleMetadata,
    actor: Mapping[str, np.ndarray],
    obs_stats: Mapping[str, np.ndarray],
    fpo_root: Path,
) -> Any:
    """Initialize upstream state and replace only actor and observation stats.

    Imports occur here, not at module import time.  Thus inventory and selector
    tooling remain usable without JAX or MuJoCo installed.
    """

    try:
        runtime_module = importlib.import_module("policy_learnware_v0.v02.runtime")
    except ImportError as error:
        raise RuntimeAdapterUnavailable(
            "v0.2 reconstructed-runtime verifier is unavailable"
        ) from error
    try:
        upstream = runtime_module.load_verified_fpo_upstream(
            fpo_root,
            allow_reconstructed=True,
        )
    except runtime_module.RuntimeVerificationError as error:
        raise RuntimeAdapterUnavailable(
            "verified reconstructed FPO runtime is unavailable; run in the "
            "reviewed GoRL environment with bytecode writes disabled"
        ) from error
    if upstream.source_attestation.get("fpo_commit") != metadata.provenance.get(
        "fpo_commit"
    ):
        raise RuntimeAdapterUnavailable(
            "attested FPO commit differs from bundle provenance"
        )

    jax = upstream.jax
    jnp = upstream.jax_numpy
    jdc = upstream.jax_dataclasses
    registry = upstream.registry
    policy_module = upstream.fpo if metadata.algorithm == "fpo" else upstream.ppo

    config_name = "FpoConfig" if metadata.algorithm == "fpo" else "PpoConfig"
    state_name = "FpoState" if metadata.algorithm == "fpo" else "PpoState"
    try:
        config_class = getattr(policy_module, config_name)
        state_class = getattr(policy_module, state_name)
        training_config = dict(metadata.policy_spec["training_config"])
        config = config_class(**training_config)
        env_config = registry.get_default_config(metadata.task)
        env = registry.load(metadata.task, config=env_config)
        state = state_class.init(prng=jax.random.key(0), env=env, config=config)
        stats_changes = {
            name: jnp.asarray(obs_stats[name])
            for name in ("count", "mean", "var_sum", "std")
            if name in obs_stats
        }
        # Upstream states are jax_dataclasses and intentionally immutable.
        # copy_and_mutate is their supported restoration path and also permits
        # replacement of the nested policy tuple without rebuilding a critic.
        with jdc.copy_and_mutate(state) as restored:
            restored.params.policy = _ordered_actor(actor, jnp)
            for name, value in stats_changes.items():
                setattr(restored.obs_stats, name, value)
        return _RestoredRuntime(restored, upstream.runtime_receipt)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeAdapterUnavailable(
            f"failed to reconstruct upstream {metadata.algorithm.upper()} state: {error}"
        ) from error


class _UpstreamFrozenPolicy:
    def __init__(
        self,
        metadata: PolicyBundleMetadata,
        state: Any,
        *,
        runtime_receipt: Mapping[str, Any] | None = None,
    ) -> None:
        self.observation_dim = metadata.observation_dim
        self.action_dim = metadata.action_dim
        self.algorithm = metadata.algorithm
        self.bundle_digest = metadata.bundle_digest
        self.runtime_receipt = runtime_receipt
        self.runtime_warning = (
            None
            if runtime_receipt is None
            else "policy loaded under explicitly reconstructed runtime"
        )
        self._state = state

    @staticmethod
    def _next_key(key: Any) -> Any:
        try:
            jax = importlib.import_module("jax")
            return jax.random.split(key, 2)[1]
        except (ImportError, TypeError, ValueError) as error:
            raise RuntimeAdapterUnavailable("JAX PRNG key is required for native inference") from error

    def act_raw(
        self,
        observation: Any,
        key: Any,
        *,
        deterministic: bool = True,
    ) -> tuple[Any, Any]:
        # Use the incoming key itself for the current sample.  This exactly
        # matches exporter golden_io generation.  The returned key is reserved
        # for the caller's next action.
        raw_action, _ = self._state.sample_action(
            observation, key, deterministic=deterministic
        )
        return raw_action, self._next_key(key)

    def act(
        self,
        observation: Any,
        key: Any,
        *,
        deterministic: bool = True,
    ) -> tuple[Any, Any]:
        raw_action, next_key = self.act_raw(observation, key, deterministic=deterministic)
        jnp = importlib.import_module("jax.numpy")
        return jnp.tanh(raw_action), next_key

    @property
    def native_state(self) -> Any:
        """Locked upstream state for device-resident, parity-preserving rollout."""

        return self._state


def load_policy(
    bundle: str | Path | PolicyBundleMetadata,
    *,
    fpo_root: str | Path,
    runtime_factory: RuntimeFactory | None = None,
    expected_fpo_commit: str | None = None,
    expected_runtime_digest: str | None = None,
) -> FrozenPolicy:
    """Validate a bundle and reconstruct its native upstream inference state."""

    if isinstance(bundle, PolicyBundleMetadata) and (
        expected_fpo_commit is None and expected_runtime_digest is None
    ):
        metadata = bundle
    else:
        source = bundle.bundle_dir if isinstance(bundle, PolicyBundleMetadata) else bundle
        metadata = validate_bundle(
            source,
            expected_fpo_commit=expected_fpo_commit,
            expected_runtime_digest=expected_runtime_digest,
        )
    actor = _arrays(metadata.bundle_dir / "actor.npz")
    obs_stats = _arrays(metadata.bundle_dir / "obs_stats.npz")
    factory = runtime_factory or _default_runtime_factory
    state_or_policy = factory(metadata, actor, obs_stats, Path(fpo_root))
    if all(
        hasattr(state_or_policy, attribute)
        for attribute in ("observation_dim", "action_dim", "act")
    ):
        return state_or_policy
    if isinstance(state_or_policy, _RestoredRuntime):
        return _UpstreamFrozenPolicy(
            metadata,
            state_or_policy.state,
            runtime_receipt=state_or_policy.runtime_receipt,
        )
    return _UpstreamFrozenPolicy(metadata, state_or_policy)
