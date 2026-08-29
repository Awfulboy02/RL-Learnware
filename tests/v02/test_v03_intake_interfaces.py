"""Compatibility checks for the minimal v0.2 -> v0.3 typed boundary."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v02.schemas import (
    EnvironmentSpec,
    ExecutionABIRecord,
    PublicMarketEntry,
    SourceAnchorRecord,
)


D0 = "0" * 64
D1 = "1" * 64
D2 = "2" * 64
D3 = "3" * 64
D4 = "4" * 64


def _environment_spec() -> tuple[EnvironmentSpec, np.ndarray, np.ndarray]:
    supports = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    beta = np.asarray([0.25, 0.75], dtype=np.float32)
    return (
        EnvironmentSpec(
            supports=supports,
            beta=beta,
            empirical_norm2=1.0,
            rkme_norm2=0.8,
            reconstruction_error=0.2,
            reducer_digest=D0,
            support_budget=2,
            latent_dim=2,
            representation_protocol_id=D1,
            measurement_protocol_id=D2,
            canonical_view_digest=D3,
            kernel_bandwidth=1.5,
            probe_dataset_digest=D4,
        ),
        supports,
        beta,
    )


def _execution_abi() -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=D0,
        action_tensor_abi_digest=D1,
        action_transform_id="clip-v1",
        policy_runtime_id="legacy-ppo-fpo-v02",
        state_abi_id="stateless-v1",
    )


def test_environment_spec_round_trip_tamper_and_deep_immutability() -> None:
    spec, supports, beta = _environment_spec()
    assert spec.environment_spec_digest == (
        "c07a3cb2843ce0c74b4c62a41ad9f79374aa1569a7afea561b57488c71753ba6"
    )
    restored = EnvironmentSpec.from_dict(spec.to_dict())
    assert restored.environment_spec_digest == spec.environment_spec_digest
    np.testing.assert_array_equal(restored.supports, spec.supports)
    np.testing.assert_array_equal(restored.beta, spec.beta)

    supports[0, 0] = 99.0
    beta[0] = 1.0
    assert spec.supports[0, 0] == 1.0
    assert spec.beta.tolist() == [0.25, 0.75]
    with pytest.raises(ValueError):
        spec.supports[0, 0] = -1.0
    with pytest.raises(FrozenInstanceError):
        spec.kernel_bandwidth = 2.0  # type: ignore[misc]

    tampered = copy.deepcopy(spec.to_dict())
    tampered["supports"][0][0] = -1.0
    with pytest.raises(ValueError, match="does not match canonical payload"):
        EnvironmentSpec.from_dict(tampered)


def test_source_anchor_round_trip_digest_tamper_and_immutability() -> None:
    anchor = SourceAnchorRecord.create(
        environment_instance_digest=D0,
        axis_binding_digest=None,
    )
    assert anchor.anchor_id == (
        "b68d463fdd21a4ca909f02fa4ec792596faa8926a21a19a7579d45e59f1e2c9a"
    )
    assert anchor.is_nominal
    assert SourceAnchorRecord.from_dict(anchor.to_dict()) == anchor

    tampered = anchor.to_dict()
    tampered["environment_instance_digest"] = D1
    with pytest.raises(ValueError, match="anchor_id does not match"):
        SourceAnchorRecord.from_dict(tampered)
    with pytest.raises(FrozenInstanceError):
        anchor.anchor_id = D2  # type: ignore[misc]


def test_execution_abi_round_trip_digest_domain_and_immutability() -> None:
    abi = _execution_abi()
    assert abi.digest == (
        "ac124ea1fe1ba54f35af57845416363785fa7d264c43b43f073667d81bbaf201"
    )
    assert ExecutionABIRecord.from_dict(abi.to_dict()) == abi

    changed = abi.to_dict()
    changed["observation_tensor_abi_digest"] = D2
    assert ExecutionABIRecord.from_dict(changed).digest != abi.digest
    invalid_schema = abi.to_dict()
    invalid_schema["schema"] = "tampered"
    with pytest.raises(ValueError, match="unsupported ExecutionABIRecord schema"):
        ExecutionABIRecord.from_dict(invalid_schema)
    with pytest.raises(FrozenInstanceError):
        abi.state_abi_id = "stateful-v1"  # type: ignore[misc]


def test_public_market_entry_round_trip_projection_and_immutability() -> None:
    entry = PublicMarketEntry(
        opaque_learnware_id="source-a",
        normalized_source_competence=0.5,
        tie_break_token=D2,
    )
    assert entry.opaque_id == entry.opaque_learnware_id
    assert sha256_json(entry.to_dict()) == (
        "a14cdf72217b23cc7717d2bbc9083fd9b84738a71a7e71d7e6ae67ba3532ee93"
    )
    assert PublicMarketEntry.from_dict(entry.to_dict()) == entry

    unknown = entry.to_dict()
    unknown["private_source_anchor"] = D0
    with pytest.raises(ValueError, match="unknown"):
        PublicMarketEntry.from_dict(unknown)
    with pytest.raises(ValueError, match="SHA-256"):
        PublicMarketEntry("source-a", 0.5, "not-a-digest")
    with pytest.raises(FrozenInstanceError):
        entry.normalized_source_competence = 0.9  # type: ignore[misc]


def test_interface_field_sets_are_frozen_for_v03_intake() -> None:
    assert set(SourceAnchorRecord.__dataclass_fields__) == {
        "anchor_id",
        "environment_instance_digest",
        "axis_binding_digest",
        "split_role",
        "schema",
    }
    assert set(ExecutionABIRecord.__dataclass_fields__) == {
        "protocol_family_id",
        "observation_tensor_abi_digest",
        "action_tensor_abi_digest",
        "action_transform_id",
        "policy_runtime_id",
        "state_abi_id",
        "schema",
    }
    assert set(PublicMarketEntry.__dataclass_fields__) == {
        "opaque_learnware_id",
        "normalized_source_competence",
        "tie_break_token",
        "schema",
    }
    assert set(EnvironmentSpec.__dataclass_fields__) == {
        "supports",
        "beta",
        "empirical_norm2",
        "rkme_norm2",
        "reconstruction_error",
        "reducer_digest",
        "support_budget",
        "latent_dim",
        "representation_protocol_id",
        "measurement_protocol_id",
        "canonical_view_digest",
        "kernel_bandwidth",
        "probe_dataset_digest",
        "environment_spec_digest",
        "schema",
    }
