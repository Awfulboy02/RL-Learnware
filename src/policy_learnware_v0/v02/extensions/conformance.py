"""Executable conformance checks for v0.2 third-party extensions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ...hashing import sha256_json, sha256_ndarrays
from .environment import (
    EnvironmentBackendPlugin,
    EnvironmentBackendRegistry,
    EnvironmentHandle,
    EnvironmentPurpose,
    ProtocolFamilyMismatch,
)
from .policy import (
    EpisodeRow,
    EvaluationContract,
    PolicyRuntimePlugin,
    PolicyRuntimeRegistry,
    RuntimeContract,
    assert_runtime_environment_compatible,
)
from .representation import (
    SemanticEncoderMetadata,
    SemanticEncoderProtocol,
    SemanticEncoderRegistry,
)


class ConformanceError(RuntimeError):
    """One or more executable extension checks failed."""


@dataclass(frozen=True)
class ConformanceReport:
    component_kind: str
    component_id: str
    checks: Mapping[str, bool]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_kind or not self.component_id:
            raise ValueError("conformance report needs component kind/id")
        checks = {str(name): bool(value) for name, value in self.checks.items()}
        if not checks:
            raise ValueError("conformance report must contain executable checks")
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "errors", tuple(str(item) for item in self.errors))

    @property
    def passed(self) -> bool:
        return not self.errors and all(self.checks.values())

    def require(self) -> "ConformanceReport":
        if not self.passed:
            failed = sorted(name for name, passed in self.checks.items() if not passed)
            raise ConformanceError(
                f"{self.component_kind} {self.component_id!r} failed conformance; "
                f"checks={failed}, errors={list(self.errors)}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "policy-learnware.v02-extension-conformance.v0",
            "component_kind": self.component_kind,
            "component_id": self.component_id,
            "checks": dict(self.checks),
            "errors": list(self.errors),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class CompatibilityPartitionKey:
    """Private full-pool partition using only the minimum execution ABI."""

    protocol_family_id: str
    execution_abi_digest: str

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "protocol_family_id": self.protocol_family_id,
                "execution_abi_digest": self.execution_abi_digest,
            }
        )


def compatibility_partition(
    environment: EnvironmentHandle,
    runtime_contract: RuntimeContract,
    *,
    encoder_metadata: SemanticEncoderMetadata | None = None,
) -> CompatibilityPartitionKey:
    """Validate cross-plugin compatibility before producing a partition key."""

    assert_runtime_environment_compatible(runtime_contract, environment)
    if (
        encoder_metadata is not None
        and encoder_metadata.protocol_family_id != environment.protocol_family_id
    ):
        raise ProtocolFamilyMismatch(
            "environment and semantic encoder protocol families differ"
        )
    return CompatibilityPartitionKey(
        protocol_family_id=environment.protocol_family_id,
        execution_abi_digest=runtime_contract.compatibility_digest,
    )


def _parity_passed(value: Any) -> bool:
    if hasattr(value, "passed"):
        return bool(value.passed)
    if isinstance(value, Mapping) and "passed" in value:
        return bool(value["passed"])
    if type(value) is bool:
        return value
    return False


def check_environment_backend(
    plugin: EnvironmentBackendPlugin,
    *,
    task_ref: str,
    instance_manifest: Mapping[str, Any],
    purpose: EnvironmentPurpose = "probe",
) -> ConformanceReport:
    checks = {
        "protocol_shape": isinstance(plugin, EnvironmentBackendPlugin),
        "inspect_schema": False,
        "handle_schema": False,
        "digest_bound": False,
        "capability_declared": False,
        "axis_registry_mapping": False,
    }
    errors: list[str] = []
    component_id = str(getattr(plugin, "backend_id", "<missing>"))
    try:
        schema = plugin.inspect(task_ref)
        checks["inspect_schema"] = bool(schema.task == task_ref)
        handle = plugin.make_handle(instance_manifest, purpose=purpose)
        checks["handle_schema"] = bool(
            handle.adapter.schema.task == schema.task
            and handle.adapter.schema.observation_dim == schema.observation_dim
            and handle.adapter.schema.action_dim == schema.action_dim
        )
        checks["digest_bound"] = bool(
            len(handle.instance_digest) == 64
            and len(handle.audit_digest) == 64
            and len(handle.task_contract_digest) == 64
        )
        handle.capabilities.require(purpose)
        checks["capability_declared"] = bool(
            handle.capabilities == plugin.capabilities
        )
        operators = plugin.axis_operators()
        checks["axis_registry_mapping"] = isinstance(operators, Mapping) and all(
            isinstance(key, str) and key for key in operators
        )
    except Exception as error:  # Conformance persists a diagnostic by design.
        errors.append(f"{type(error).__name__}: {error}")
    return ConformanceReport(
        component_kind="environment_backend",
        component_id=component_id,
        checks=checks,
        errors=tuple(errors),
    )


def check_semantic_encoder(
    encoder: SemanticEncoderProtocol,
    dataset: Any,
    *,
    batch_size: int = 8,
) -> ConformanceReport:
    checks = {
        "protocol_shape": isinstance(encoder, SemanticEncoderProtocol),
        "metadata": False,
        "input_immutable": False,
        "deterministic": False,
        "episode_partition": False,
        "protocol_binding": False,
    }
    errors: list[str] = []
    metadata = getattr(encoder, "metadata", None)
    component_id = str(getattr(metadata, "representation_id", "<missing>"))
    try:
        if not isinstance(metadata, SemanticEncoderMetadata):
            raise TypeError("encoder metadata has the wrong type")
        checks["metadata"] = True
        packed_before = np.array(dataset.packed, copy=True)
        offsets_before = np.array(dataset.episode_offsets, copy=True)
        left = encoder.encode(dataset, batch_size=batch_size)
        right = encoder.encode(dataset, batch_size=batch_size)
        checks["input_immutable"] = bool(
            np.array_equal(dataset.packed, packed_before)
            and np.array_equal(dataset.episode_offsets, offsets_before)
        )
        checks["deterministic"] = bool(
            sha256_ndarrays({"points": left.points})
            == sha256_ndarrays({"points": right.points})
        )
        checks["episode_partition"] = bool(
            np.array_equal(left.episode_offsets, offsets_before)
            and np.array_equal(right.episode_offsets, offsets_before)
        )
        checks["protocol_binding"] = bool(
            left.representation_protocol_id
            == right.representation_protocol_id
            == metadata.representation_protocol_id
        )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return ConformanceReport(
        component_kind="semantic_encoder",
        component_id=component_id,
        checks=checks,
        errors=tuple(errors),
    )


def check_policy_runtime(
    plugin: PolicyRuntimePlugin,
    *,
    bundle: Any,
    environment: EnvironmentHandle,
    seeds: Sequence[int],
    evaluation_contract: EvaluationContract,
) -> ConformanceReport:
    checks = {
        "protocol_shape": isinstance(plugin, PolicyRuntimePlugin),
        "bundle_validated": False,
        "runtime_binding": False,
        "parity": False,
        "family_partition": False,
        "episode_rows": False,
    }
    errors: list[str] = []
    component_id = str(getattr(plugin, "runtime_id", "<missing>"))
    try:
        seed_values = tuple(seeds)
        plugin.capabilities.require_family(environment.protocol_family_id)
        plugin.capabilities.require_evaluation()
        validated = plugin.validate(bundle)
        checks["bundle_validated"] = bool(
            validated.runtime_contract.policy_runtime_id == component_id
        )
        policy = plugin.load(validated)
        checks["runtime_binding"] = bool(
            policy.runtime_contract == validated.runtime_contract
        )
        checks["parity"] = _parity_passed(
            plugin.parity_check(validated, policy)
        )
        compatibility_partition(environment, policy.runtime_contract)
        checks["family_partition"] = True
        rows = plugin.evaluate_batched(
            policy, environment, seed_values, evaluation_contract
        )
        checks["episode_rows"] = bool(
            len(rows) == len(seed_values)
            and all(
                isinstance(row, EpisodeRow)
                and row.episode_index == index
                and row.reset_seed == seed_values[index]
                and np.isfinite(row.return_sum)
                for index, row in enumerate(rows)
            )
        )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return ConformanceReport(
        component_kind="policy_runtime",
        component_id=component_id,
        checks=checks,
        errors=tuple(errors),
    )


@dataclass
class V02ExtensionRegistries:
    """Three separate namespaces; cross-family joins remain explicit."""

    environments: EnvironmentBackendRegistry = field(
        default_factory=EnvironmentBackendRegistry
    )
    policies: PolicyRuntimeRegistry = field(default_factory=PolicyRuntimeRegistry)
    representations: SemanticEncoderRegistry = field(
        default_factory=SemanticEncoderRegistry
    )

    def resolve_partition(
        self,
        *,
        backend_id: str,
        runtime_id: str,
        representation_id: str,
        protocol_family_id: str,
        purpose: EnvironmentPurpose,
        compiled_oracle: bool = False,
    ) -> tuple[EnvironmentBackendPlugin, PolicyRuntimePlugin, SemanticEncoderProtocol]:
        environment = self.environments.resolve(
            backend_id,
            protocol_family_id=protocol_family_id,
            purpose=purpose,
            compiled_oracle=compiled_oracle,
        )
        runtime = self.policies.resolve(
            runtime_id, protocol_family_id=protocol_family_id
        )
        encoder = self.representations.resolve(
            representation_id, protocol_family_id=protocol_family_id
        )
        if purpose == "oracle":
            runtime.capabilities.require_evaluation(batched=compiled_oracle)
        return environment, runtime, encoder


__all__ = [
    "CompatibilityPartitionKey",
    "ConformanceError",
    "ConformanceReport",
    "V02ExtensionRegistries",
    "check_environment_backend",
    "check_policy_runtime",
    "check_semantic_encoder",
    "compatibility_partition",
]
