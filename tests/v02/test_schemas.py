from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from policy_learnware_v0.v02.schemas import (
    AxisAnchorBinding,
    EnvironmentInstanceRecord,
    EnvironmentSpec,
    ProtocolIdentifiers,
    RuntimeContract,
    SourceAnchorRecord,
    TargetContextRecord,
    canonical_model_diff_projection,
    derive_protocol_id,
)
from policy_learnware_v0.hashing import sha256_json


D0 = "0" * 64
D1 = "1" * 64
D2 = "2" * 64
D3 = "3" * 64
D4 = "4" * 64
D5 = "5" * 64
D6 = "6" * 64
D7 = "7" * 64
D8 = "8" * 64
D9 = "9" * 64
DA = "a" * 64
DB = "b" * 64
DC = "c" * 64


def test_environment_axis_and_anchor_records_self_check_digests() -> None:
    model_diff_digest = sha256_json(
        canonical_model_diff_projection(
            nominal_model_digest=D1,
            bound_model_digest=D1,
            changes=(),
        )
    )
    instance = EnvironmentInstanceRecord.create(
        task="SyntheticTask",
        backend="mujoco_playground.registry",
        nominal=True,
        factor=1.0,
        environment_class="synthetic.FakeEnv",
        registry_config_digest=D0,
        runtime_digest=D2,
        expected_nominal_model_digest=D1,
        expected_bound_model_digest=D1,
        operator_digest=None,
        axis_binding_digest=None,
        model_diff_digest=model_diff_digest,
    )
    assert EnvironmentInstanceRecord.from_dict(instance.to_dict()) == instance
    tampered = instance.to_dict()
    tampered["runtime_digest"] = D3
    with pytest.raises(ValueError, match="does not match"):
        EnvironmentInstanceRecord.from_dict(tampered)
    bad_nominal_diff = instance.to_dict()
    bad_nominal_diff["model_diff_digest"] = D9
    with pytest.raises(ValueError, match="model_diff_digest mismatch"):
        EnvironmentInstanceRecord.create(
            **{
                key: value
                for key, value in bad_nominal_diff.items()
                if key not in {"schema", "environment_instance_digest"}
            }
        )

    binding = AxisAnchorBinding.create(
        axis_id="mass_inertia",
        factor_id="low",
        operator_digest=D3,
        model_diff_digest=D4,
    )
    assert AxisAnchorBinding.from_dict(binding.to_dict()) == binding
    nominal_a = SourceAnchorRecord.create(
        environment_instance_digest=instance.environment_instance_digest,
        axis_binding_digest=None,
    )
    nominal_b = SourceAnchorRecord.create(
        environment_instance_digest=instance.environment_instance_digest,
        axis_binding_digest=None,
    )
    shifted = SourceAnchorRecord.create(
        environment_instance_digest=instance.environment_instance_digest,
        axis_binding_digest=binding.axis_binding_digest,
    )
    assert nominal_a.anchor_id == nominal_b.anchor_id
    assert nominal_a.is_nominal
    assert shifted.anchor_id != nominal_a.anchor_id
    with pytest.raises(FrozenInstanceError):
        shifted.anchor_id = D0  # type: ignore[misc]


def test_target_context_enforces_explicit_safety_reference_and_opaque_id() -> None:
    exact = TargetContextRecord(
        opaque_target_id="v02q-" + "a" * 32,
        task_contract_id=D0,
        regime="safety_exact",
        source_anchor_ref=D1,
        private_environment_instance_digest=D2,
        split_manifest_digest=D3,
    )
    assert TargetContextRecord.from_dict(exact.to_dict()) == exact
    with pytest.raises(ValueError, match="require"):
        TargetContextRecord(
            opaque_target_id="v02q-" + "b" * 32,
            task_contract_id=D0,
            regime="safety_exact",
            source_anchor_ref=None,
            private_environment_instance_digest=D2,
            split_manifest_digest=D3,
        )
    with pytest.raises(ValueError, match="cannot reference"):
        TargetContextRecord(
            opaque_target_id="v02q-" + "c" * 32,
            task_contract_id=D0,
            regime="heldout_interpolation",
            source_anchor_ref=D1,
            private_environment_instance_digest=D2,
            split_manifest_digest=D3,
        )


def test_environment_spec_copies_arrays_is_readonly_and_self_authenticates() -> None:
    supports = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    beta = np.asarray([0.25, 0.75], dtype=np.float32)
    spec = EnvironmentSpec(
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
    )
    supports[0, 0] = 99.0
    beta[0] = 1.0
    assert spec.supports[0, 0] == 1.0
    assert spec.beta.tolist() == [0.25, 0.75]
    with pytest.raises(ValueError):
        spec.supports[0, 0] = 8.0
    assert len(spec.environment_spec_digest or "") == 64
    assert EnvironmentSpec.from_dict(spec.to_dict()).environment_spec_digest == spec.environment_spec_digest

    tampered = copy.deepcopy(spec.to_dict())
    tampered["supports"][0][0] = -1.0
    with pytest.raises(ValueError, match="does not match"):
        EnvironmentSpec.from_dict(tampered)
    unknown = spec.to_dict()
    unknown["factor"] = 2.0
    with pytest.raises(ValueError, match="unknown"):
        EnvironmentSpec.from_dict(unknown)


def test_environment_spec_rejects_non_simplex_or_shape_drift() -> None:
    kwargs = {
        "supports": np.ones((2, 3)),
        "beta": np.asarray([0.5, 0.5]),
        "empirical_norm2": 1.0,
        "rkme_norm2": 1.0,
        "reconstruction_error": 0.0,
        "reducer_digest": D0,
        "support_budget": 2,
        "latent_dim": 3,
        "representation_protocol_id": D1,
        "measurement_protocol_id": D2,
        "canonical_view_digest": D3,
        "kernel_bandwidth": 1.0,
        "probe_dataset_digest": D4,
    }
    with pytest.raises(ValueError, match="simplex"):
        EnvironmentSpec(**{**kwargs, "beta": np.asarray([-0.1, 1.1])})
    with pytest.raises(ValueError, match="supports shape"):
        EnvironmentSpec(**{**kwargs, "latent_dim": 4})


def test_protocol_id_domains_are_separate_and_composition_self_checks() -> None:
    dependencies = {"code": D0}
    benchmark = derive_protocol_id(
        "benchmark", config_projection={"x": 1}, dependency_digests=dependencies
    )
    training = derive_protocol_id(
        "training", config_projection={"x": 1}, dependency_digests=dependencies
    )
    assert benchmark != training
    assert benchmark == derive_protocol_id(
        "benchmark", config_projection={"x": 1}, dependency_digests=dependencies
    )

    components = {
        "benchmark_protocol_id": benchmark,
        "training_protocol_id": training,
        "policy_market_id": D1,
        "probe_protocol_id": D2,
        "representation_protocol_ids": {"raw": D3, "corro": D4},
        "representation_index_ids": {"raw": D5, "corro": D6},
        "selector_protocol_ids": {"B1": D7, "M02/B5": D8},
        "evaluation_protocol_id": D9,
        "statistics_protocol_id": DA,
        "cost_contract_digest": DB,
    }
    identifiers = ProtocolIdentifiers.create(**components)
    restored = ProtocolIdentifiers.from_dict(identifiers.to_dict())
    assert restored.experiment_protocol_id == identifiers.experiment_protocol_id
    with pytest.raises(TypeError):
        identifiers.selector_protocol_ids["poison"] = DC  # type: ignore[index]
    tampered = identifiers.to_dict()
    tampered["selector_protocol_ids"]["B1"] = DC
    with pytest.raises(ValueError, match="does not match"):
        ProtocolIdentifiers.from_dict(tampered)


def test_runtime_contract_is_immutable_and_digest_stable() -> None:
    runtime = RuntimeContract(
        protocol_family_id="continuous-vector-mdp-v02",
        task_contract_digest=D0,
        observation_schema_digest=D1,
        action_schema_digest=D2,
        observation_dim=24,
        action_dim=6,
        action_transform_id="identity",
        policy_runtime_id="legacy-ppo-fpo",
        state_schema_id="stateless",
    )
    assert RuntimeContract.from_dict(runtime.to_dict()).digest == runtime.digest
    changed_task = replace(runtime, task_contract_digest=D3)
    assert changed_task.digest != runtime.digest
    assert changed_task.compatibility_digest == runtime.compatibility_digest
    assert runtime.execution_abi.state_abi_id == "stateless"
    with pytest.raises(FrozenInstanceError):
        runtime.action_dim = 7  # type: ignore[misc]
