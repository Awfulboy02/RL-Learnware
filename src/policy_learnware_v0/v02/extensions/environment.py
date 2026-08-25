"""Fail-closed environment backend extension contracts for v0.2.

The extension layer deliberately depends only on the stable v0 ``EnvAdapter``
and ``EnvSchema`` APIs.  Axis records and v0.2 manifests are accepted as
opaque mappings so this module does not create a dependency on schemas that
are still being frozen by the v0.2 orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

from ...envs.base import EnvAdapter
from ...envs.mujoco_playground import MujocoPlaygroundEnvAdapter
from ...hashing import sha256_json
from ...schemas import EnvSchema


CONTINUOUS_VECTOR_MDP_V02 = "continuous-vector-mdp-v02"
EnvironmentPurpose = Literal["train", "probe", "oracle"]
_PURPOSES = frozenset({"train", "probe", "oracle"})


class EnvironmentPluginError(ValueError):
    """An environment plugin or handle violated the extension contract."""


class DuplicateEnvironmentBackendError(EnvironmentPluginError):
    """A backend id was registered more than once."""


class EnvironmentCapabilityError(EnvironmentPluginError):
    """A backend cannot satisfy the requested execution capability."""


class ProtocolFamilyMismatch(EnvironmentPluginError):
    """Components from different protocol families were combined."""


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EnvironmentPluginError(f"{where} must be a non-empty canonical string")
    return value


def _digest(value: Any, where: str) -> str:
    candidate = _identifier(value, where).lower()
    if len(candidate) != 64:
        raise EnvironmentPluginError(f"{where} must be a SHA-256 digest")
    try:
        int(candidate, 16)
    except ValueError as error:
        raise EnvironmentPluginError(f"{where} must be a SHA-256 digest") from error
    return candidate


def _manifest_alias(
    manifest: Mapping[str, Any], aliases: tuple[str, ...], where: str
) -> Any:
    present = [(name, manifest[name]) for name in aliases if name in manifest]
    if not present:
        raise EnvironmentPluginError(
            f"instance manifest is missing {where}; expected one of {aliases}"
        )
    canonical = present[0][1]
    if any(value != canonical for _, value in present[1:]):
        raise EnvironmentPluginError(f"conflicting aliases for {where}: {present}")
    return canonical


def task_contract_digest(schema: EnvSchema) -> str:
    """Return the minimum exact task/runtime compatibility projection.

    Dynamics parameters are intentionally absent.  The task id, fixed horizon,
    action repeat and backend bind the reward/task execution contract that a
    frozen actor expects while allowing source and target dynamics to differ.
    A formal run may supply a stronger precomputed digest in its manifest; the
    backend verifies it against this projection instead of silently trusting it.
    """

    return sha256_json(
        {
            "schema": "policy-learnware.v02-task-contract-projection.v0",
            "backend": schema.backend,
            "task": schema.task,
            "horizon": int(schema.horizon),
            "action_repeat": int(schema.action_repeat),
        }
    )


def observation_compatibility_digest(schema: EnvSchema) -> str:
    """Digest only the task-anonymous observation tensor ABI."""

    return sha256_json(
        {
            "schema": "policy-learnware.v02-observation-compatibility.v0",
            "dimension": int(schema.observation_dim),
            "dtype": str(schema.observation_dtype),
        }
    )


def action_compatibility_digest(schema: EnvSchema) -> str:
    """Digest action dimensions, dtype and native bounds."""

    return sha256_json(
        {
            "schema": "policy-learnware.v02-action-compatibility.v0",
            "dimension": int(schema.action_dim),
            "dtype": str(schema.action_dtype),
            "low": np.asarray(schema.action_low, dtype=np.float32).tolist(),
            "high": np.asarray(schema.action_high, dtype=np.float32).tolist(),
        }
    )


@dataclass(frozen=True)
class EnvironmentCapabilities:
    """Declared execution capabilities used before a handle is constructed."""

    protocol_family_id: str = CONTINUOUS_VECTOR_MDP_V02
    supports_training: bool = False
    supports_probe: bool = True
    supports_scalar_oracle: bool = True
    supports_compiled_oracle: bool = False
    provides_native_env: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_family_id",
            _identifier(self.protocol_family_id, "protocol_family_id"),
        )
        for name in (
            "supports_training",
            "supports_probe",
            "supports_scalar_oracle",
            "supports_compiled_oracle",
            "provides_native_env",
        ):
            if type(getattr(self, name)) is not bool:
                raise EnvironmentPluginError(f"{name} must be boolean")
        if (self.supports_training or self.supports_compiled_oracle) and not self.provides_native_env:
            raise EnvironmentPluginError(
                "training/compiled-oracle capability requires a native environment"
            )

    def require(
        self, purpose: EnvironmentPurpose, *, compiled_oracle: bool = False
    ) -> None:
        if purpose not in _PURPOSES:
            raise EnvironmentCapabilityError(f"unsupported environment purpose {purpose!r}")
        allowed = {
            "train": self.supports_training,
            "probe": self.supports_probe,
            "oracle": (
                self.supports_compiled_oracle
                if compiled_oracle
                else self.supports_scalar_oracle or self.supports_compiled_oracle
            ),
        }[purpose]
        if not allowed:
            qualifier = "compiled " if purpose == "oracle" and compiled_oracle else ""
            raise EnvironmentCapabilityError(
                f"backend does not support {qualifier}{purpose} execution"
            )


def _default_scalar_capabilities() -> EnvironmentCapabilities:
    return EnvironmentCapabilities()


@dataclass(frozen=True)
class EnvironmentHandle:
    """Digest-bound adapter/native pair returned by a backend plugin."""

    adapter: EnvAdapter
    native_env: Any | None
    instance_digest: str
    audit_digest: str
    task_contract_digest: str = ""
    capabilities: EnvironmentCapabilities = field(
        default_factory=_default_scalar_capabilities
    )

    def __post_init__(self) -> None:
        schema = getattr(self.adapter, "schema", None)
        if not isinstance(schema, EnvSchema):
            raise EnvironmentPluginError("EnvironmentHandle.adapter has no valid EnvSchema")
        if not callable(getattr(self.adapter, "reset", None)) or not callable(
            getattr(self.adapter, "step", None)
        ):
            raise EnvironmentPluginError("EnvironmentHandle.adapter lacks reset/step")
        object.__setattr__(
            self, "instance_digest", _digest(self.instance_digest, "instance_digest")
        )
        object.__setattr__(
            self, "audit_digest", _digest(self.audit_digest, "audit_digest")
        )
        expected_task_digest = task_contract_digest(schema)
        supplied = self.task_contract_digest or expected_task_digest
        supplied = _digest(supplied, "task_contract_digest")
        if supplied != expected_task_digest:
            raise EnvironmentPluginError(
                "task_contract_digest differs from the adapter compatibility projection"
            )
        object.__setattr__(self, "task_contract_digest", supplied)
        if self.capabilities.provides_native_env != (self.native_env is not None):
            raise EnvironmentPluginError(
                "native_env presence differs from declared environment capability"
            )

    @property
    def protocol_family_id(self) -> str:
        return self.capabilities.protocol_family_id

    @property
    def observation_schema_digest(self) -> str:
        return observation_compatibility_digest(self.adapter.schema)

    @property
    def action_schema_digest(self) -> str:
        return action_compatibility_digest(self.adapter.schema)


@runtime_checkable
class EnvironmentBackendPlugin(Protocol):
    backend_id: str
    capabilities: EnvironmentCapabilities

    def inspect(self, task_ref: str) -> EnvSchema: ...

    def make_handle(
        self,
        instance_manifest: Mapping[str, Any],
        *,
        purpose: EnvironmentPurpose,
    ) -> EnvironmentHandle: ...

    def axis_operators(self) -> Mapping[str, Any]: ...


class EnvironmentBackendRegistry:
    """Closed registry with explicit family and capability resolution."""

    def __init__(self) -> None:
        self._plugins: dict[str, EnvironmentBackendPlugin] = {}

    def register(self, plugin: EnvironmentBackendPlugin) -> None:
        if not isinstance(plugin, EnvironmentBackendPlugin):
            raise EnvironmentPluginError(
                "environment backend does not implement the required protocol"
            )
        backend_id = _identifier(plugin.backend_id, "backend_id")
        if backend_id in self._plugins:
            raise DuplicateEnvironmentBackendError(
                f"environment backend {backend_id!r} is already registered"
            )
        if not isinstance(plugin.capabilities, EnvironmentCapabilities):
            raise EnvironmentPluginError("backend capabilities have the wrong type")
        self._plugins[backend_id] = plugin

    def resolve(
        self,
        backend_id: str,
        *,
        protocol_family_id: str | None = None,
        purpose: EnvironmentPurpose | None = None,
        compiled_oracle: bool = False,
    ) -> EnvironmentBackendPlugin:
        key = _identifier(backend_id, "backend_id")
        try:
            plugin = self._plugins[key]
        except KeyError as error:
            raise EnvironmentPluginError(f"unknown environment backend {key!r}") from error
        if (
            protocol_family_id is not None
            and plugin.capabilities.protocol_family_id != protocol_family_id
        ):
            raise ProtocolFamilyMismatch(
                f"backend family {plugin.capabilities.protocol_family_id!r} != "
                f"required {protocol_family_id!r}"
            )
        if purpose is not None:
            plugin.capabilities.require(purpose, compiled_oracle=compiled_oracle)
        return plugin

    @property
    def plugins(self) -> Mapping[str, EnvironmentBackendPlugin]:
        return MappingProxyType(dict(self._plugins))


class HandleFactory(Protocol):
    def __call__(
        self,
        instance_manifest: Mapping[str, Any],
        *,
        purpose: EnvironmentPurpose,
    ) -> EnvironmentHandle: ...


class MujocoPlaygroundBackendPlugin:
    """Official real backend adapter around the existing Playground adapter.

    The default path intentionally accepts only canonical nominal instances.
    Shifted training must inject an audited ``handle_factory`` which consumes
    the complete frozen axis manifest.  This prevents a directory labelled as
    shifted from silently training on ``registry.load(task)`` nominal dynamics.
    """

    backend_id = MujocoPlaygroundEnvAdapter.backend
    capabilities = EnvironmentCapabilities(
        supports_training=True,
        supports_probe=True,
        supports_scalar_oracle=True,
        supports_compiled_oracle=True,
        provides_native_env=True,
    )

    def __init__(
        self,
        *,
        operators: Mapping[str, Any] | None = None,
        adapter_factory: Callable[..., EnvAdapter] = MujocoPlaygroundEnvAdapter,
        handle_factory: HandleFactory | None = None,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._handle_factory = handle_factory
        resolved: dict[str, Any] = {}
        for operator_id, operator in (operators or {}).items():
            key = _identifier(operator_id, "axis operator id")
            declared = getattr(operator, "operator_id", key)
            if declared != key:
                raise EnvironmentPluginError(
                    f"axis operator key {key!r} differs from operator_id {declared!r}"
                )
            if key in resolved:
                raise EnvironmentPluginError(f"duplicate axis operator {key!r}")
            resolved[key] = operator
        self._operators = MappingProxyType(resolved)

    def inspect(self, task_ref: str) -> EnvSchema:
        task = _identifier(task_ref, "task_ref")
        adapter = self._adapter_factory(task, jit=False)
        schema = getattr(adapter, "schema", None)
        if not isinstance(schema, EnvSchema) or schema.backend != self.backend_id:
            raise EnvironmentPluginError(
                "MuJoCo adapter inspection returned an incompatible EnvSchema"
            )
        return schema

    def make_handle(
        self,
        instance_manifest: Mapping[str, Any],
        *,
        purpose: EnvironmentPurpose,
    ) -> EnvironmentHandle:
        if not isinstance(instance_manifest, Mapping):
            raise EnvironmentPluginError("instance_manifest must be a mapping")
        self.capabilities.require(purpose)
        family = instance_manifest.get(
            "protocol_family_id", self.capabilities.protocol_family_id
        )
        if family != self.capabilities.protocol_family_id:
            raise ProtocolFamilyMismatch(
                f"instance family {family!r} cannot use {self.backend_id!r}"
            )
        if self._handle_factory is not None:
            handle = self._handle_factory(instance_manifest, purpose=purpose)
            self._validate_handle(handle, purpose=purpose)
            return handle

        axis_binding = instance_manifest.get("axis_binding_digest")
        if axis_binding not in (None, ""):
            raise EnvironmentCapabilityError(
                "default MuJoCo wrapper is nominal-only; shifted manifests require "
                "an audited anchor-aware handle_factory"
            )
        task = _identifier(
            _manifest_alias(
                instance_manifest, ("task_ref", "task_id", "task"), "task"
            ),
            "task",
        )
        instance_digest = _digest(
            _manifest_alias(
                instance_manifest,
                ("environment_instance_digest", "instance_digest"),
                "environment instance digest",
            ),
            "environment_instance_digest",
        )
        audit_digest = _digest(
            _manifest_alias(
                instance_manifest,
                ("environment_audit_digest", "audit_digest"),
                "environment audit digest",
            ),
            "environment_audit_digest",
        )
        kwargs: dict[str, Any] = {"jit": True}
        if "horizon" in instance_manifest:
            kwargs["expected_horizon"] = int(instance_manifest["horizon"])
        if "action_repeat" in instance_manifest:
            kwargs["expected_action_repeat"] = int(
                instance_manifest["action_repeat"]
            )
        adapter = self._adapter_factory(task, **kwargs)
        native_env = getattr(adapter, "environment", None)
        if native_env is None:
            raise EnvironmentCapabilityError(
                "official MuJoCo backend did not expose its native environment"
            )
        expected_schema_digest = instance_manifest.get("env_schema_digest")
        if (
            expected_schema_digest is not None
            and _digest(expected_schema_digest, "env_schema_digest")
            != adapter.schema.digest
        ):
            raise EnvironmentPluginError("live environment schema digest mismatch")
        handle = EnvironmentHandle(
            adapter=adapter,
            native_env=native_env,
            instance_digest=instance_digest,
            audit_digest=audit_digest,
            task_contract_digest=str(
                instance_manifest.get("task_contract_digest", "")
            ),
            capabilities=self.capabilities,
        )
        self._validate_handle(handle, purpose=purpose)
        return handle

    def _validate_handle(
        self, handle: EnvironmentHandle, *, purpose: EnvironmentPurpose
    ) -> None:
        if not isinstance(handle, EnvironmentHandle):
            raise EnvironmentPluginError("handle_factory returned a non-EnvironmentHandle")
        if handle.protocol_family_id != self.capabilities.protocol_family_id:
            raise ProtocolFamilyMismatch("handle and backend protocol families differ")
        if handle.capabilities != self.capabilities:
            raise EnvironmentCapabilityError(
                "official MuJoCo handle capabilities differ from its backend declaration"
            )
        if handle.adapter.schema.backend != self.backend_id:
            raise EnvironmentPluginError("handle adapter backend differs from plugin id")
        if handle.native_env is None:
            raise EnvironmentCapabilityError(
                "official MuJoCo handle must expose the native Playground environment"
            )
        handle.capabilities.require(purpose)

    def axis_operators(self) -> Mapping[str, Any]:
        return self._operators


__all__ = [
    "CONTINUOUS_VECTOR_MDP_V02",
    "DuplicateEnvironmentBackendError",
    "EnvironmentBackendPlugin",
    "EnvironmentBackendRegistry",
    "EnvironmentCapabilities",
    "EnvironmentCapabilityError",
    "EnvironmentHandle",
    "EnvironmentPluginError",
    "EnvironmentPurpose",
    "MujocoPlaygroundBackendPlugin",
    "ProtocolFamilyMismatch",
    "action_compatibility_digest",
    "observation_compatibility_digest",
    "task_contract_digest",
]
