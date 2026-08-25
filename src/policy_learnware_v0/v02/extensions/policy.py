"""Frozen-policy runtime extension contracts and legacy PPO/FPO adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ...hashing import sha256_json
from ...policy.bundle import BUNDLE_SCHEMA, PolicyBundleMetadata, validate_bundle
from ...policy.evaluate import evaluate_frozen_policy_returns_batched
from ...policy.loader import FrozenPolicy, RuntimeFactory, load_policy
from ...policy.parity import ParityReport, verify_golden_parity
from ..schemas import RuntimeContract
from .environment import (
    CONTINUOUS_VECTOR_MDP_V02,
    EnvironmentHandle,
    ProtocolFamilyMismatch,
)


class PolicyPluginError(ValueError):
    """A policy runtime plugin violated the extension contract."""


class DuplicatePolicyRuntimeError(PolicyPluginError):
    """A runtime id was registered more than once."""


class PolicyCapabilityError(PolicyPluginError):
    """A runtime cannot provide a requested evaluation capability."""


class RuntimeCompatibilityError(PolicyPluginError):
    """A frozen policy and environment are not executable-compatible."""


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PolicyPluginError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    candidate = _identifier(value, where).lower()
    if len(candidate) != 64:
        raise PolicyPluginError(f"{where} must be a SHA-256 digest")
    try:
        int(candidate, 16)
    except ValueError as error:
        raise PolicyPluginError(f"{where} must be a SHA-256 digest") from error
    return candidate


def assert_runtime_environment_compatible(
    runtime_contract: RuntimeContract,
    environment: EnvironmentHandle,
) -> None:
    """Fail closed unless the task-anonymous calling ABI matches a live env.

    The source task/reward contract remains private provenance.  It is not an
    executable-set filter: a policy from another task is intentionally allowed
    when its tensor/action/runtime ABI can be invoked on this environment.
    """

    if not isinstance(runtime_contract, RuntimeContract):
        raise RuntimeCompatibilityError("policy lacks the shared v0.2 RuntimeContract")
    schema = environment.adapter.schema
    checks = {
        "protocol_family_id": (
            runtime_contract.protocol_family_id,
            environment.protocol_family_id,
        ),
        "observation_schema_digest": (
            runtime_contract.observation_schema_digest,
            environment.observation_schema_digest,
        ),
        "action_schema_digest": (
            runtime_contract.action_schema_digest,
            environment.action_schema_digest,
        ),
        "observation_dim": (
            runtime_contract.observation_dim,
            int(schema.observation_dim),
        ),
        "action_dim": (runtime_contract.action_dim, int(schema.action_dim)),
    }
    failed = {
        name: {"policy": left, "environment": right}
        for name, (left, right) in checks.items()
        if left != right
    }
    if failed:
        if "protocol_family_id" in failed:
            raise ProtocolFamilyMismatch(
                f"policy/environment protocol families differ: {failed['protocol_family_id']}"
            )
        raise RuntimeCompatibilityError(
            f"policy/environment runtime contract mismatch: {failed}"
        )


@dataclass(frozen=True)
class PolicyRuntimeCapabilities:
    protocol_family_ids: tuple[str, ...]
    supports_scalar_evaluation: bool = True
    supports_batched_evaluation: bool = False
    supports_stateful_policy: bool = False

    def __post_init__(self) -> None:
        families = tuple(
            _identifier(item, "protocol_family_ids[]")
            for item in self.protocol_family_ids
        )
        if not families or len(set(families)) != len(families):
            raise PolicyPluginError(
                "protocol_family_ids must be non-empty and duplicate-free"
            )
        object.__setattr__(self, "protocol_family_ids", families)
        for name in (
            "supports_scalar_evaluation",
            "supports_batched_evaluation",
            "supports_stateful_policy",
        ):
            if type(getattr(self, name)) is not bool:
                raise PolicyPluginError(f"{name} must be boolean")

    def require_family(self, protocol_family_id: str) -> None:
        if protocol_family_id not in self.protocol_family_ids:
            raise ProtocolFamilyMismatch(
                f"runtime does not support protocol family {protocol_family_id!r}"
            )

    def require_evaluation(self, *, batched: bool = False) -> None:
        available = (
            self.supports_batched_evaluation
            if batched
            else self.supports_scalar_evaluation
            or self.supports_batched_evaluation
        )
        if not available:
            qualifier = "batched " if batched else ""
            raise PolicyCapabilityError(
                f"runtime does not support {qualifier}policy evaluation"
            )


@dataclass(frozen=True)
class PolicyStep:
    action: Any
    state: Any
    next_key: Any


@runtime_checkable
class ExecutablePolicyProtocol(Protocol):
    runtime_contract: RuntimeContract

    def initial_state(self, seed: int) -> Any: ...

    def act(
        self,
        observation: Any,
        state: Any,
        key: Any,
        *,
        deterministic: bool,
    ) -> PolicyStep: ...


@dataclass(frozen=True)
class ValidatedPolicyBundle:
    """Runtime-owned validated bundle view; never selector-visible."""

    path: Path
    bundle_schema: str
    bundle_digest: str
    runtime_contract: RuntimeContract
    payload: Any

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        if not path.is_dir():
            raise PolicyPluginError(f"validated bundle path is not a directory: {path}")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "bundle_schema", _identifier(self.bundle_schema, "bundle_schema"))
        object.__setattr__(self, "bundle_digest", _digest(self.bundle_digest, "bundle_digest"))
        if not isinstance(self.runtime_contract, RuntimeContract):
            raise PolicyPluginError("validated bundle lacks a RuntimeContract")


@dataclass(frozen=True)
class EvaluationContract:
    horizon: int
    observation_dim: int
    action_dim: int
    deterministic: bool = True
    require_fixed_horizon: bool = True
    policy_seed_offset: int = 1_000_003

    def __post_init__(self) -> None:
        for name in ("horizon", "observation_dim", "action_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise PolicyPluginError(f"EvaluationContract.{name} must be positive")
        if type(self.deterministic) is not bool or type(self.require_fixed_horizon) is not bool:
            raise PolicyPluginError("evaluation boolean flags must be bool")
        if type(self.policy_seed_offset) is not int or self.policy_seed_offset < 0:
            raise PolicyPluginError("policy_seed_offset must be non-negative")


@dataclass(frozen=True)
class EpisodeRow:
    episode_index: int
    reset_seed: int
    policy_seed: int
    return_sum: float
    steps: int
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if self.episode_index < 0 or self.reset_seed < 0 or self.policy_seed < 0:
            raise PolicyPluginError("episode indices/seeds must be non-negative")
        if self.steps <= 0 or not np.isfinite(self.return_sum):
            raise PolicyPluginError("episode row has invalid steps or return")


class LegacyExecutablePolicyAdapter:
    """Adapt the v0 ``FrozenPolicy.act(obs, key)`` API to the v0.2 contract."""

    def __init__(self, policy: FrozenPolicy, runtime_contract: RuntimeContract) -> None:
        if int(getattr(policy, "observation_dim", -1)) != runtime_contract.observation_dim:
            raise RuntimeCompatibilityError("legacy policy observation dimension mismatch")
        if int(getattr(policy, "action_dim", -1)) != runtime_contract.action_dim:
            raise RuntimeCompatibilityError("legacy policy action dimension mismatch")
        self._policy = policy
        self.runtime_contract = runtime_contract

    def initial_state(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise PolicyPluginError("policy state seed must be a non-negative integer")
        return None

    def act(
        self,
        observation: Any,
        state: Any,
        key: Any,
        *,
        deterministic: bool,
    ) -> PolicyStep:
        if state is not None:
            raise RuntimeCompatibilityError(
                "legacy PPO/FPO actors require state=None"
            )
        action, next_key = self._policy.act(
            observation, key, deterministic=deterministic
        )
        return PolicyStep(action=action, state=None, next_key=next_key)

    @property
    def native_policy(self) -> FrozenPolicy:
        return self._policy


KeyFactory = Callable[[int], Any]


def evaluate_scalar_policy(
    policy: ExecutablePolicyProtocol,
    environment: EnvironmentHandle,
    seeds: Sequence[int],
    contract: EvaluationContract,
    *,
    key_factory: KeyFactory = lambda seed: seed,
) -> tuple[EpisodeRow, ...]:
    """Dependency-light reference evaluator for third-party scalar plugins."""

    if not isinstance(policy, ExecutablePolicyProtocol):
        raise PolicyPluginError("policy does not implement ExecutablePolicyProtocol")
    assert_runtime_environment_compatible(policy.runtime_contract, environment)
    if not environment.capabilities.supports_scalar_oracle:
        raise PolicyCapabilityError(
            "environment does not support scalar oracle evaluation"
        )
    schema = environment.adapter.schema
    if (
        contract.horizon != int(schema.horizon)
        or contract.observation_dim != int(schema.observation_dim)
        or contract.action_dim != int(schema.action_dim)
    ):
        raise RuntimeCompatibilityError(
            "evaluation contract differs from the environment schema"
        )
    seed_values = tuple(seeds)
    if not seed_values:
        raise PolicyPluginError("at least one evaluation seed is required")
    if any(type(seed) is not int or seed < 0 for seed in seed_values):
        raise PolicyPluginError("evaluation seeds must be non-negative integers")

    rows: list[EpisodeRow] = []
    for episode_index, reset_seed in enumerate(seed_values):
        policy_seed = reset_seed + contract.policy_seed_offset
        state, observation = environment.adapter.reset(reset_seed)
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (contract.observation_dim,) or not np.all(
            np.isfinite(observation)
        ):
            raise PolicyPluginError(
                "environment reset emitted an invalid observation vector"
            )
        policy_state = policy.initial_state(policy_seed)
        key = key_factory(policy_seed)
        total = 0.0
        terminated = False
        truncated = False
        steps = 0
        for step_index in range(contract.horizon):
            result = policy.act(
                observation,
                policy_state,
                key,
                deterministic=contract.deterministic,
            )
            if not isinstance(result, PolicyStep):
                raise PolicyPluginError("policy.act returned a non-PolicyStep")
            action = np.asarray(result.action, dtype=np.float32)
            if action.shape != (contract.action_dim,) or not np.all(np.isfinite(action)):
                raise PolicyPluginError("policy emitted an invalid action vector")
            if np.any(action < schema.action_low) or np.any(action > schema.action_high):
                raise PolicyPluginError("policy action lies outside environment bounds")
            state, transition = environment.adapter.step(state, action)
            observation = np.asarray(transition.observation, dtype=np.float32)
            if observation.shape != (contract.observation_dim,) or not np.all(
                np.isfinite(observation)
            ):
                raise PolicyPluginError(
                    "environment step emitted an invalid observation vector"
                )
            reward = float(transition.reward)
            if not np.isfinite(reward):
                raise PolicyPluginError("environment emitted a non-finite reward")
            policy_state = result.state
            key = result.next_key
            total += reward
            steps = step_index + 1
            terminated = bool(transition.terminated)
            truncated = bool(transition.truncated)
            if terminated or truncated:
                if contract.require_fixed_horizon and steps != contract.horizon:
                    raise PolicyPluginError(
                        "environment ended before the registered fixed horizon"
                    )
                break
        rows.append(
            EpisodeRow(
                episode_index=episode_index,
                reset_seed=reset_seed,
                policy_seed=policy_seed,
                return_sum=total,
                steps=steps,
                terminated=terminated,
                truncated=truncated,
            )
        )
    return tuple(rows)


@runtime_checkable
class PolicyRuntimePlugin(Protocol):
    runtime_id: str
    supported_bundle_schemas: tuple[str, ...]
    capabilities: PolicyRuntimeCapabilities

    def validate(self, bundle: Path) -> ValidatedPolicyBundle: ...

    def load(self, bundle: ValidatedPolicyBundle) -> ExecutablePolicyProtocol: ...

    def parity_check(
        self,
        bundle: ValidatedPolicyBundle,
        policy: ExecutablePolicyProtocol,
    ) -> Any: ...

    def evaluate_batched(
        self,
        policy: ExecutablePolicyProtocol,
        env: EnvironmentHandle,
        seeds: Sequence[int],
        contract: EvaluationContract,
    ) -> tuple[EpisodeRow, ...]: ...


class PolicyRuntimeRegistry:
    """Runtime registry that rejects duplicate ids and ambiguous schema routing."""

    def __init__(self) -> None:
        self._plugins: dict[str, PolicyRuntimePlugin] = {}

    def register(self, plugin: PolicyRuntimePlugin) -> None:
        if not isinstance(plugin, PolicyRuntimePlugin):
            raise PolicyPluginError("policy runtime does not implement the protocol")
        runtime_id = _identifier(plugin.runtime_id, "runtime_id")
        if runtime_id in self._plugins:
            raise DuplicatePolicyRuntimeError(
                f"policy runtime {runtime_id!r} is already registered"
            )
        schemas = tuple(
            _identifier(item, "supported_bundle_schemas[]")
            for item in plugin.supported_bundle_schemas
        )
        if not schemas or len(set(schemas)) != len(schemas):
            raise PolicyPluginError(
                "supported_bundle_schemas must be non-empty and duplicate-free"
            )
        if not isinstance(plugin.capabilities, PolicyRuntimeCapabilities):
            raise PolicyPluginError("runtime capabilities have the wrong type")
        self._plugins[runtime_id] = plugin

    def resolve(
        self, runtime_id: str, *, protocol_family_id: str | None = None
    ) -> PolicyRuntimePlugin:
        key = _identifier(runtime_id, "runtime_id")
        try:
            plugin = self._plugins[key]
        except KeyError as error:
            raise PolicyPluginError(f"unknown policy runtime {key!r}") from error
        if protocol_family_id is not None:
            plugin.capabilities.require_family(protocol_family_id)
        return plugin

    def resolve_bundle_schema(
        self, bundle_schema: str, *, protocol_family_id: str
    ) -> PolicyRuntimePlugin:
        schema = _identifier(bundle_schema, "bundle_schema")
        candidates = [
            plugin
            for plugin in self._plugins.values()
            if schema in plugin.supported_bundle_schemas
            and protocol_family_id in plugin.capabilities.protocol_family_ids
        ]
        if not candidates:
            raise PolicyPluginError(
                f"no runtime accepts bundle schema {schema!r} in family "
                f"{protocol_family_id!r}"
            )
        if len(candidates) != 1:
            raise PolicyPluginError(
                f"ambiguous runtime routing for schema {schema!r}: "
                f"{sorted(item.runtime_id for item in candidates)}"
            )
        return candidates[0]

    @property
    def plugins(self) -> Mapping[str, PolicyRuntimePlugin]:
        return MappingProxyType(dict(self._plugins))


RuntimeContractFactory = Callable[[PolicyBundleMetadata], RuntimeContract]


def _legacy_runtime_contract(
    metadata: PolicyBundleMetadata, *, runtime_id: str
) -> RuntimeContract:
    training = metadata.policy_spec.get("training_config", {})
    horizon = int(training.get("episode_length", 1000))
    action_repeat = int(training.get("action_repeat", 1))
    backend = "mujoco_playground.registry"
    task_digest = sha256_json(
        {
            "schema": "policy-learnware.v02-task-contract-projection.v0",
            "backend": backend,
            "task": metadata.task,
            "horizon": horizon,
            "action_repeat": action_repeat,
        }
    )
    observation_digest = sha256_json(
        {
            "schema": "policy-learnware.v02-observation-compatibility.v0",
            "task": metadata.task,
            "dimension": metadata.observation_dim,
            "dtype": "float32",
        }
    )
    action_digest = sha256_json(
        {
            "schema": "policy-learnware.v02-action-compatibility.v0",
            "dimension": metadata.action_dim,
            "dtype": "float32",
            "low": [-1.0] * metadata.action_dim,
            "high": [1.0] * metadata.action_dim,
        }
    )
    return RuntimeContract(
        protocol_family_id=CONTINUOUS_VECTOR_MDP_V02,
        task_contract_digest=task_digest,
        observation_schema_digest=observation_digest,
        action_schema_digest=action_digest,
        observation_dim=metadata.observation_dim,
        action_dim=metadata.action_dim,
        action_transform_id="tanh(raw_action)",
        policy_runtime_id=runtime_id,
        state_schema_id="stateless-v0",
    )


class LegacyPpoFpoRuntimePlugin:
    """Official adapter for the immutable v0 PPO/FPO policy bundle schema."""

    runtime_id = "legacy-ppo-fpo-v0"
    supported_bundle_schemas = (BUNDLE_SCHEMA,)
    capabilities = PolicyRuntimeCapabilities(
        protocol_family_ids=(CONTINUOUS_VECTOR_MDP_V02,),
        supports_scalar_evaluation=True,
        supports_batched_evaluation=True,
        supports_stateful_policy=False,
    )

    def __init__(
        self,
        *,
        fpo_root: str | Path,
        runtime_factory: RuntimeFactory | None = None,
        runtime_contract_factory: RuntimeContractFactory | None = None,
        expected_fpo_commit: str | None = None,
        expected_runtime_digest: str | None = None,
    ) -> None:
        self._fpo_root = Path(fpo_root).expanduser().resolve()
        self._runtime_factory = runtime_factory
        self._contract_factory = runtime_contract_factory
        self._expected_fpo_commit = expected_fpo_commit
        self._expected_runtime_digest = expected_runtime_digest

    def validate(self, bundle: Path) -> ValidatedPolicyBundle:
        metadata = validate_bundle(
            bundle,
            expected_fpo_commit=self._expected_fpo_commit,
            expected_runtime_digest=self._expected_runtime_digest,
        )
        contract = (
            self._contract_factory(metadata)
            if self._contract_factory is not None
            else _legacy_runtime_contract(metadata, runtime_id=self.runtime_id)
        )
        if contract.policy_runtime_id != self.runtime_id:
            raise RuntimeCompatibilityError(
                "runtime contract is bound to another policy runtime"
            )
        self.capabilities.require_family(contract.protocol_family_id)
        if (
            contract.observation_dim != metadata.observation_dim
            or contract.action_dim != metadata.action_dim
        ):
            raise RuntimeCompatibilityError(
                "runtime contract dimensions differ from validated bundle metadata"
            )
        return ValidatedPolicyBundle(
            path=metadata.bundle_dir,
            bundle_schema=BUNDLE_SCHEMA,
            bundle_digest=metadata.bundle_digest,
            runtime_contract=contract,
            payload=metadata,
        )

    def load(self, bundle: ValidatedPolicyBundle) -> LegacyExecutablePolicyAdapter:
        self._require_bundle(bundle)
        policy = load_policy(
            bundle.payload,
            fpo_root=self._fpo_root,
            runtime_factory=self._runtime_factory,
        )
        return LegacyExecutablePolicyAdapter(policy, bundle.runtime_contract)

    def parity_check(
        self,
        bundle: ValidatedPolicyBundle,
        policy: ExecutablePolicyProtocol,
    ) -> ParityReport:
        self._require_bundle(bundle)
        if not isinstance(policy, LegacyExecutablePolicyAdapter):
            raise PolicyPluginError("legacy parity requires LegacyExecutablePolicyAdapter")
        return verify_golden_parity(policy.native_policy, bundle.payload)

    def evaluate_batched(
        self,
        policy: ExecutablePolicyProtocol,
        env: EnvironmentHandle,
        seeds: Sequence[int],
        contract: EvaluationContract,
    ) -> tuple[EpisodeRow, ...]:
        if not isinstance(policy, LegacyExecutablePolicyAdapter):
            raise PolicyPluginError("legacy evaluator requires its policy adapter")
        assert_runtime_environment_compatible(policy.runtime_contract, env)
        if not contract.deterministic:
            raise PolicyCapabilityError(
                "legacy compiled evaluator is certified only for deterministic actors"
            )
        seed_values = tuple(seeds)
        if not seed_values or any(type(seed) is not int or seed < 0 for seed in seed_values):
            raise PolicyPluginError("evaluation seeds must be non-negative integers")
        if (
            contract.horizon != env.adapter.schema.horizon
            or contract.observation_dim != policy.runtime_contract.observation_dim
            or contract.action_dim != policy.runtime_contract.action_dim
        ):
            raise RuntimeCompatibilityError("evaluation contract dimensions differ")
        if not env.capabilities.supports_compiled_oracle:
            if not env.capabilities.supports_scalar_oracle:
                raise PolicyCapabilityError(
                    "environment supports neither scalar nor compiled oracle evaluation"
                )

            def key_factory(seed: int) -> Any:
                try:
                    import jax
                except ImportError as error:  # pragma: no cover - legacy dependency gate
                    raise PolicyCapabilityError(
                        "legacy scalar evaluation requires JAX keys"
                    ) from error
                return jax.random.key(seed)

            return evaluate_scalar_policy(
                policy, env, seed_values, contract, key_factory=key_factory
            )
        if env.native_env is None:
            raise PolicyCapabilityError(
                "compiled oracle capability was declared without a native environment"
            )
        policy_seeds = tuple(seed + contract.policy_seed_offset for seed in seed_values)
        returns = evaluate_frozen_policy_returns_batched(
            policy.native_policy,
            env.native_env,
            reset_seeds=seed_values,
            policy_seeds=policy_seeds,
            horizon=contract.horizon,
            observation_dim=contract.observation_dim,
            action_dim=contract.action_dim,
        )
        return tuple(
            EpisodeRow(
                episode_index=index,
                reset_seed=seed,
                policy_seed=policy_seeds[index],
                return_sum=value,
                steps=contract.horizon,
                terminated=False,
                truncated=True,
            )
            for index, (seed, value) in enumerate(zip(seed_values, returns, strict=True))
        )

    def _require_bundle(self, bundle: ValidatedPolicyBundle) -> None:
        if not isinstance(bundle, ValidatedPolicyBundle):
            raise PolicyPluginError("runtime received a non-validated bundle")
        if bundle.bundle_schema not in self.supported_bundle_schemas:
            raise PolicyPluginError("bundle schema is unsupported by legacy runtime")
        if not isinstance(bundle.payload, PolicyBundleMetadata):
            raise PolicyPluginError("legacy bundle payload has the wrong type")
        if bundle.runtime_contract.policy_runtime_id != self.runtime_id:
            raise RuntimeCompatibilityError("bundle belongs to another runtime")


class ScalarPolicyRuntimePlugin:
    """Small callback-backed runtime useful for third-party conformance tests."""

    def __init__(
        self,
        *,
        runtime_id: str,
        supported_bundle_schemas: Sequence[str],
        validator: Callable[[Path], ValidatedPolicyBundle],
        loader: Callable[[ValidatedPolicyBundle], ExecutablePolicyProtocol],
        parity_checker: Callable[[ValidatedPolicyBundle, ExecutablePolicyProtocol], Any],
        protocol_family_ids: Sequence[str] = (CONTINUOUS_VECTOR_MDP_V02,),
        key_factory: KeyFactory = lambda seed: seed,
    ) -> None:
        self.runtime_id = _identifier(runtime_id, "runtime_id")
        self.supported_bundle_schemas = tuple(supported_bundle_schemas)
        self.capabilities = PolicyRuntimeCapabilities(
            protocol_family_ids=tuple(protocol_family_ids),
            supports_scalar_evaluation=True,
            supports_batched_evaluation=False,
            supports_stateful_policy=True,
        )
        self._validator = validator
        self._loader = loader
        self._parity_checker = parity_checker
        self._key_factory = key_factory

    def validate(self, bundle: Path) -> ValidatedPolicyBundle:
        validated = self._validator(Path(bundle))
        self._require_bundle(validated)
        return validated

    def load(self, bundle: ValidatedPolicyBundle) -> ExecutablePolicyProtocol:
        self._require_bundle(bundle)
        policy = self._loader(bundle)
        if not isinstance(policy, ExecutablePolicyProtocol):
            raise PolicyPluginError("scalar loader returned an incompatible policy")
        if policy.runtime_contract != bundle.runtime_contract:
            raise RuntimeCompatibilityError(
                "loaded policy contract differs from validated bundle"
            )
        return policy

    def parity_check(
        self,
        bundle: ValidatedPolicyBundle,
        policy: ExecutablePolicyProtocol,
    ) -> Any:
        self._require_bundle(bundle)
        return self._parity_checker(bundle, policy)

    def evaluate_batched(
        self,
        policy: ExecutablePolicyProtocol,
        env: EnvironmentHandle,
        seeds: Sequence[int],
        contract: EvaluationContract,
    ) -> tuple[EpisodeRow, ...]:
        return evaluate_scalar_policy(
            policy, env, seeds, contract, key_factory=self._key_factory
        )

    def _require_bundle(self, bundle: ValidatedPolicyBundle) -> None:
        if not isinstance(bundle, ValidatedPolicyBundle):
            raise PolicyPluginError("runtime received a non-validated bundle")
        if bundle.bundle_schema not in self.supported_bundle_schemas:
            raise PolicyPluginError("bundle schema is unsupported")
        if bundle.runtime_contract.policy_runtime_id != self.runtime_id:
            raise RuntimeCompatibilityError("bundle belongs to another runtime")
        self.capabilities.require_family(bundle.runtime_contract.protocol_family_id)


__all__ = [
    "DuplicatePolicyRuntimeError",
    "EpisodeRow",
    "EvaluationContract",
    "ExecutablePolicyProtocol",
    "LegacyExecutablePolicyAdapter",
    "LegacyPpoFpoRuntimePlugin",
    "PolicyCapabilityError",
    "PolicyPluginError",
    "PolicyRuntimeCapabilities",
    "PolicyRuntimePlugin",
    "PolicyRuntimeRegistry",
    "PolicyStep",
    "RuntimeCompatibilityError",
    "RuntimeContract",
    "ScalarPolicyRuntimePlugin",
    "ValidatedPolicyBundle",
    "evaluate_scalar_policy",
]
