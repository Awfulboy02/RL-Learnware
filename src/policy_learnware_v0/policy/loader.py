"""Frozen native-policy loader with lazy upstream imports."""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import sys
import types
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
    runtime_warning: str | None


def _ensure_wandb_importable() -> str | None:
    """Install the smallest import-only shim needed by upstream rollouts.

    The locked GoRL environment does not contain wandb, while ``flow_policy``
    imports it at package import time even for frozen inference.  The shim is
    installed only after a real import fails and is never used for logging.
    """

    try:
        importlib.import_module("wandb")
        return None
    except ImportError:
        wandb = types.ModuleType("wandb")
        sdk = types.ModuleType("wandb.sdk")
        wandb_run = types.ModuleType("wandb.sdk.wandb_run")

        class Histogram:  # pragma: no cover - import compatibility only
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.args = args
                self.kwargs = kwargs

        class Run:  # pragma: no cover - import compatibility only
            pass

        wandb.Histogram = Histogram  # type: ignore[attr-defined]
        wandb.sdk = sdk  # type: ignore[attr-defined]
        sdk.wandb_run = wandb_run  # type: ignore[attr-defined]
        wandb_run.Run = Run  # type: ignore[attr-defined]
        sys.modules["wandb"] = wandb
        sys.modules["wandb.sdk"] = sdk
        sys.modules["wandb.sdk.wandb_run"] = wandb_run
        return "wandb unavailable; installed an import-only shim for frozen inference"


def _replace(instance: Any, **changes: Any) -> Any:
    try:
        return dataclasses.replace(instance, **changes)
    except (TypeError, ValueError):
        try:
            jdc = importlib.import_module("jax_dataclasses")
            return jdc.replace(instance, **changes)
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise RuntimeAdapterUnavailable(
                f"cannot replace fields on upstream {type(instance).__name__}"
            ) from error


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _verify_upstream_checkout(metadata: PolicyBundleMetadata, fpo_root: Path) -> Path:
    root = fpo_root.expanduser().resolve()
    source_dir = (root / "playground" / "src").resolve()
    if not (source_dir / "flow_policy").is_dir():
        raise RuntimeAdapterUnavailable(f"not an upstream FPO checkout: {root}")
    expected_commit = str(metadata.provenance.get("fpo_commit", ""))

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeAdapterUnavailable(f"cannot verify FPO checkout: {detail}")
        return completed.stdout.strip()

    if git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeAdapterUnavailable("FPO runtime commit differs from bundle provenance")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeAdapterUnavailable("FPO runtime has tracked modifications")
    for name, module in tuple(sys.modules.items()):
        if name != "flow_policy" and not name.startswith("flow_policy."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        try:
            Path(origin).resolve().relative_to(source_dir)
        except ValueError as error:
            raise RuntimeAdapterUnavailable(
                f"cached module {name!r} comes from another checkout: {origin}"
            ) from error
    return source_dir


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

    source_dir = _verify_upstream_checkout(metadata, fpo_root)
    source_string = str(source_dir.resolve())
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    runtime_warning = _ensure_wandb_importable()
    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        # Playground 0.0.5 re-exports ``registry`` from the package but does
        # not install it as an importable ``mujoco_playground.registry`` module.
        playground = importlib.import_module("mujoco_playground")
        registry = getattr(playground, "registry")
        jdc = importlib.import_module("jax_dataclasses")
        policy_module = importlib.import_module(f"flow_policy.{metadata.algorithm}")
    except ImportError as error:
        raise RuntimeAdapterUnavailable(
            "locked upstream runtime dependencies are unavailable; run in the GoRL environment"
        ) from error
    module_path = getattr(policy_module, "__file__", None)
    if module_path is None:
        raise RuntimeAdapterUnavailable("upstream policy module has no filesystem origin")
    try:
        Path(module_path).resolve().relative_to(source_dir)
    except ValueError as error:
        raise RuntimeAdapterUnavailable(
            f"upstream policy module was imported from {module_path}, not {source_dir}"
        ) from error

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
        return _RestoredRuntime(restored, runtime_warning)
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
        runtime_warning: str | None = None,
    ) -> None:
        self.observation_dim = metadata.observation_dim
        self.action_dim = metadata.action_dim
        self.algorithm = metadata.algorithm
        self.bundle_digest = metadata.bundle_digest
        self.runtime_warning = runtime_warning
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

    metadata = (
        bundle
        if isinstance(bundle, PolicyBundleMetadata)
        else validate_bundle(
            bundle,
            expected_fpo_commit=expected_fpo_commit,
            expected_runtime_digest=expected_runtime_digest,
        )
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
            runtime_warning=state_or_policy.runtime_warning,
        )
    return _UpstreamFrozenPolicy(metadata, state_or_policy)
