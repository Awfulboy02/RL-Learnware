"""Frozen v0.2 records retained as the minimal v0.3 intake boundary.

v0.2's scientific construction and evaluation schemas belong to historical
Git objects and external reports. The final branch keeps only the four
immutable records that v0.3 consumes directly. Their field sets, canonical
JSON projections, and digest domains are unchanged from ``v0.2.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

import numpy as np

from ..hashing import canonicalize, sha256_json


SOURCE_ANCHOR_SCHEMA = "policy-learnware.v02-source-anchor.v0"
EXECUTION_ABI_SCHEMA = "policy-learnware.v02-execution-abi.v0"
ENVIRONMENT_SPEC_SCHEMA = "policy-learnware.v02-environment-spec.v0"
PUBLIC_MARKET_ENTRY_SCHEMA = "policy-learnware.v02-public-market-entry.v0"

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:/-]+$")


def _strict(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"invalid {where} keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _nonempty(value: Any, where: str, *, safe: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    if safe and not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{where} is not a safe identifier")
    return value


def _digest(value: Any, where: str) -> str:
    result = _nonempty(value, where).lower()
    if len(result) != 64:
        raise ValueError(f"{where} must be a SHA-256 hex digest")
    try:
        int(result, 16)
    except ValueError as exc:
        raise ValueError(f"{where} must be a SHA-256 hex digest") from exc
    return result


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def _readonly_array(value: Any, *, ndim: int, where: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or any(size <= 0 for size in array.shape):
        raise ValueError(f"{where} must be a non-empty {ndim}-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{where} must be finite")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SourceAnchorRecord:
    """Digest-bound identity of one source environment anchor."""

    anchor_id: str
    environment_instance_digest: str
    axis_binding_digest: str | None
    split_role: str = "source"
    schema: str = SOURCE_ANCHOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SOURCE_ANCHOR_SCHEMA:
            raise ValueError(f"unsupported SourceAnchorRecord schema: {self.schema!r}")
        if self.split_role != "source":
            raise ValueError("SourceAnchorRecord.split_role must be 'source'")
        environment = _digest(
            self.environment_instance_digest, "environment_instance_digest"
        )
        binding = (
            None
            if self.axis_binding_digest is None
            else _digest(self.axis_binding_digest, "axis_binding_digest")
        )
        payload = {
            "schema": self.schema,
            "environment_instance_digest": environment,
            "axis_binding_digest": binding,
            "split_role": "source",
        }
        anchor_id = _digest(self.anchor_id, "anchor_id")
        if anchor_id != sha256_json(payload):
            raise ValueError("anchor_id does not match canonical source-anchor payload")
        object.__setattr__(self, "anchor_id", anchor_id)
        object.__setattr__(self, "environment_instance_digest", environment)
        object.__setattr__(self, "axis_binding_digest", binding)

    @classmethod
    def create(
        cls,
        *,
        environment_instance_digest: str,
        axis_binding_digest: str | None,
    ) -> "SourceAnchorRecord":
        environment = _digest(
            environment_instance_digest, "environment_instance_digest"
        )
        binding = (
            None
            if axis_binding_digest is None
            else _digest(axis_binding_digest, "axis_binding_digest")
        )
        payload = {
            "schema": SOURCE_ANCHOR_SCHEMA,
            "environment_instance_digest": environment,
            "axis_binding_digest": binding,
            "split_role": "source",
        }
        return cls(
            anchor_id=sha256_json(payload),
            environment_instance_digest=environment,
            axis_binding_digest=binding,
        )

    @property
    def is_nominal(self) -> bool:
        return self.axis_binding_digest is None

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceAnchorRecord":
        names = set(cls.__dataclass_fields__)
        _strict(value, names, "SourceAnchorRecord")
        return cls(**{name: value[name] for name in names})


@dataclass(frozen=True)
class ExecutionABIRecord:
    """Private, task-anonymous policy calling convention."""

    protocol_family_id: str
    observation_tensor_abi_digest: str
    action_tensor_abi_digest: str
    action_transform_id: str
    policy_runtime_id: str
    state_abi_id: str
    schema: str = EXECUTION_ABI_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXECUTION_ABI_SCHEMA:
            raise ValueError(f"unsupported ExecutionABIRecord schema: {self.schema!r}")
        object.__setattr__(
            self,
            "protocol_family_id",
            _nonempty(self.protocol_family_id, "protocol_family_id", safe=True),
        )
        for name in ("observation_tensor_abi_digest", "action_tensor_abi_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("action_transform_id", "policy_runtime_id", "state_abi_id"):
            object.__setattr__(
                self, name, _nonempty(getattr(self, name), name, safe=True)
            )

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionABIRecord":
        names = set(cls.__dataclass_fields__)
        _strict(value, names, "ExecutionABIRecord")
        return cls(**{name: value[name] for name in names})


@dataclass(frozen=True)
class EnvironmentSpec:
    """Digest-bound reduced environment representation used by v0.3 intake."""

    supports: np.ndarray
    beta: np.ndarray
    empirical_norm2: float
    rkme_norm2: float
    reconstruction_error: float
    reducer_digest: str
    support_budget: int
    latent_dim: int
    representation_protocol_id: str
    measurement_protocol_id: str
    canonical_view_digest: str
    kernel_bandwidth: float
    probe_dataset_digest: str
    environment_spec_digest: str | None = None
    schema: str = ENVIRONMENT_SPEC_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_SPEC_SCHEMA:
            raise ValueError(f"unsupported EnvironmentSpec schema: {self.schema!r}")
        supports = _readonly_array(self.supports, ndim=2, where="supports")
        beta = _readonly_array(self.beta, ndim=1, where="beta")
        if beta.shape != (supports.shape[0],):
            raise ValueError("beta must have one weight per support")
        if np.any(beta < 0.0) or not np.isclose(
            np.sum(beta), 1.0, rtol=0.0, atol=1e-8
        ):
            raise ValueError("beta must be a probability simplex")
        support_budget = _positive_int(self.support_budget, "support_budget")
        latent_dim = _positive_int(self.latent_dim, "latent_dim")
        if supports.shape != (support_budget, latent_dim):
            raise ValueError("supports shape must equal (support_budget, latent_dim)")
        for name in ("empirical_norm2", "rkme_norm2", "reconstruction_error"):
            number = _finite(getattr(self, name), name)
            if number < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, number)
        bandwidth = _finite(self.kernel_bandwidth, "kernel_bandwidth")
        if bandwidth <= 0.0:
            raise ValueError("kernel_bandwidth must be positive")
        for name in (
            "reducer_digest",
            "representation_protocol_id",
            "measurement_protocol_id",
            "canonical_view_digest",
            "probe_dataset_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "supports", supports)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "support_budget", support_budget)
        object.__setattr__(self, "latent_dim", latent_dim)
        object.__setattr__(self, "kernel_bandwidth", bandwidth)
        expected = sha256_json(self._payload_without_digest())
        if self.environment_spec_digest is None:
            object.__setattr__(self, "environment_spec_digest", expected)
        else:
            actual = _digest(self.environment_spec_digest, "environment_spec_digest")
            if actual != expected:
                raise ValueError(
                    "environment_spec_digest does not match canonical payload"
                )
            object.__setattr__(self, "environment_spec_digest", actual)

    def _payload_without_digest(self) -> dict[str, Any]:
        return canonicalize(
            {
                "schema": self.schema,
                "supports": self.supports,
                "beta": self.beta,
                "empirical_norm2": self.empirical_norm2,
                "rkme_norm2": self.rkme_norm2,
                "reconstruction_error": self.reconstruction_error,
                "reducer_digest": self.reducer_digest,
                "support_budget": self.support_budget,
                "latent_dim": self.latent_dim,
                "representation_protocol_id": self.representation_protocol_id,
                "measurement_protocol_id": self.measurement_protocol_id,
                "canonical_view_digest": self.canonical_view_digest,
                "kernel_bandwidth": self.kernel_bandwidth,
                "probe_dataset_digest": self.probe_dataset_digest,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload_without_digest(),
            "environment_spec_digest": self.environment_spec_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentSpec":
        names = set(cls.__dataclass_fields__)
        _strict(value, names, "EnvironmentSpec")
        return cls(**{name: value[name] for name in names})


@dataclass(frozen=True)
class PublicMarketEntry:
    """Anonymous public projection of one source policy."""

    opaque_learnware_id: str
    normalized_source_competence: float
    tie_break_token: str
    schema: str = PUBLIC_MARKET_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PUBLIC_MARKET_ENTRY_SCHEMA:
            raise ValueError(f"unsupported PublicMarketEntry schema: {self.schema!r}")
        object.__setattr__(
            self,
            "opaque_learnware_id",
            _nonempty(
                self.opaque_learnware_id, "opaque_learnware_id", safe=True
            ),
        )
        competence = _finite(
            self.normalized_source_competence, "normalized_source_competence"
        )
        if not 0.0 <= competence <= 1.0:
            raise ValueError("normalized_source_competence must lie in [0, 1]")
        object.__setattr__(
            self, "tie_break_token", _digest(self.tie_break_token, "tie_break_token")
        )
        object.__setattr__(self, "normalized_source_competence", competence)

    @property
    def opaque_id(self) -> str:
        """Internal compatibility alias; never serialized."""

        return self.opaque_learnware_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "opaque_learnware_id": self.opaque_learnware_id,
            "normalized_source_competence": self.normalized_source_competence,
            "tie_break_token": self.tie_break_token,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicMarketEntry":
        names = set(cls.__dataclass_fields__)
        _strict(value, names, "PublicMarketEntry")
        return cls(**{name: value[name] for name in names})


__all__ = [
    "ENVIRONMENT_SPEC_SCHEMA",
    "EXECUTION_ABI_SCHEMA",
    "EnvironmentSpec",
    "ExecutionABIRecord",
    "PUBLIC_MARKET_ENTRY_SCHEMA",
    "PublicMarketEntry",
    "SOURCE_ANCHOR_SCHEMA",
    "SourceAnchorRecord",
]
