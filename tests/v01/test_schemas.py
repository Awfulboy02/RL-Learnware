from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from policy_learnware_v0.schemas import EnvSchema
from policy_learnware_v0.v01.schemas import (
    EnvironmentInstanceRecord,
    MeasurementSchemaView,
    OracleAggregateRecord,
    OracleEpisodeRecord,
    PrivateContextRecord,
    ShiftManifest,
    VariantDatasetManifest,
    derive_experiment_protocol_id,
    derive_measurement_protocol_id,
    derive_measurement_run_id,
    derive_oracle_protocol_id,
    derive_variant_id,
)


D0 = "0" * 64
D1 = "1" * 64
D2 = "2" * 64
D3 = "3" * 64


def _env(task: str = "WalkerWalk") -> EnvSchema:
    return EnvSchema(
        backend="mujoco_playground.registry",
        task=task,
        observation_dim=24,
        action_dim=6,
        action_low=np.full(6, -1.0, dtype=np.float32),
        action_high=np.full(6, 1.0, dtype=np.float32),
        horizon=1000,
        action_repeat=1,
        control_dt=0.02,
        flatten_fingerprint="same-interface-fingerprint",
        implementation_digest="implementation",
    )


def test_context_manifest_and_variant_identity_have_no_hash_loop_or_plaintext_context() -> None:
    context = PrivateContextRecord.new(
        task="WalkerWalk",
        shift_id="global_nonzero_dof_damping_scale",
        factor=2.0,
        context_token=b"c" * 16,
        nonce_token=b"n" * 32,
    )
    assert context.d_theta == pytest.approx(abs(math.log(2.0)))
    manifest = ShiftManifest.create(
        shift_id=context.shift_id,
        factor=context.factor,
        registry_digest=D1,
        base_protocol_id=D2,
        task=context.task,
        private_context_id=context.private_context_id,
    )
    variant_id = derive_variant_id(
        measurement_protocol_id=D3,
        private_nonce=context.private_nonce,
        shift_manifest_digest=manifest.digest,
    )
    assert variant_id == derive_variant_id(
        measurement_protocol_id=D3,
        private_nonce=context.private_nonce,
        shift_manifest_digest=manifest.digest,
    )
    assert variant_id.startswith("v01v-")
    assert "Walker" not in variant_id and "2.0" not in variant_id
    changed_nonce = PrivateContextRecord.new(
        task=context.task,
        shift_id=context.shift_id,
        factor=context.factor,
        context_token=b"c" * 16,
        nonce_token=b"m" * 32,
    )
    assert derive_variant_id(
        measurement_protocol_id=D3,
        private_nonce=changed_nonce.private_nonce,
        shift_manifest_digest=manifest.digest,
    ) != variant_id


def test_measurement_schema_projection_omits_task_and_is_immutable() -> None:
    left = MeasurementSchemaView.from_env_schema(_env("WalkerWalk"))
    right = MeasurementSchemaView.from_env_schema(_env("PrivateOtherName"))
    assert left.digest == right.digest
    assert "task" not in left.to_dict()
    with pytest.raises(ValueError):
        left.action_low[0] = 4
    restored = MeasurementSchemaView.from_dict(left.to_dict())
    assert restored.digest == left.digest
    tampered = left.to_dict()
    tampered["task"] = "leak"
    with pytest.raises(ValueError):
        MeasurementSchemaView.from_dict(tampered)


def test_instance_dataset_and_oracle_schemas_round_trip_strictly() -> None:
    context = PrivateContextRecord.new(
        task="WalkerWalk", shift_id="global_nonzero_dof_damping_scale", factor=1.0,
        context_token=b"a" * 16, nonce_token=b"b" * 32,
    )
    manifest = ShiftManifest.create(
        shift_id=context.shift_id, factor=1.0, registry_digest=D1,
        base_protocol_id=D2, task=context.task, private_context_id=context.private_context_id,
    )
    variant_id = derive_variant_id(
        measurement_protocol_id=D3, private_nonce=context.private_nonce,
        shift_manifest_digest=manifest.digest,
    )
    instance = EnvironmentInstanceRecord.create(
        variant_id=variant_id,
        env_schema_digest=D0,
        measurement_schema_view_digest=D1,
        shift_manifest_digest=manifest.digest,
        base_model_digest=D1,
        shifted_model_digest=D1,
        changed_leaf="_mjx_model.dof_damping",
        changed_index_count=0,
        before_leaf_digest=D2,
        after_leaf_digest=D2,
        operator_digest=D3,
        runtime_versions={"python": "3.12"},
        finite_termination_audit_summary={"finite": True, "episodes": 4},
    )
    assert EnvironmentInstanceRecord.from_dict(instance.to_dict()).digest == instance.digest

    dataset = VariantDatasetManifest(
        variant_id=variant_id, bank=0, episode_count=2, transition_count=2000,
        reset_seeds=(1, 2), probe_seeds=(3, 4), dataset_digest=D0,
        base_protocol_id=D1, measurement_contract_digest=D2,
        measurement_schema_view_digest=D3,
    )
    assert VariantDatasetManifest.from_dict(dataset.to_dict()).digest == dataset.digest
    assert "environment_instance_digest" not in dataset.to_dict()

    episode = OracleEpisodeRecord(
        task_private="WalkerWalk", variant_id=variant_id, candidate_id="candidate-0",
        episode_index=0, reset_seed=1, policy_seed=2, raw_episodic_sum=250.0,
        mean_step_return=0.25, instance_digest=D0, bundle_digest=D1,
        evaluator_contract_digest=D2,
    )
    assert OracleEpisodeRecord.from_dict(episode.to_dict()) == episode
    aggregate = OracleAggregateRecord(
        task_private="WalkerWalk", variant_id=variant_id, nominal_variant_id=variant_id,
        candidate_id="candidate-0", episode_count=50, mean_step_return=0.25,
        mean_return_ci_low=0.2, mean_return_ci_high=0.3, delta_return=-0.1,
        delta_ci_low=-0.2, delta_ci_high=-0.01, abs_transfer_gap=0.1,
        abs_gap_ci_low=0.01, abs_gap_ci_high=0.2,
    )
    assert OracleAggregateRecord.from_dict(aggregate.to_dict()) == aggregate
    bad = copy.deepcopy(aggregate.to_dict())
    bad["abs_transfer_gap"] = 0.2
    with pytest.raises(ValueError):
        OracleAggregateRecord.from_dict(bad)


def test_typed_protocol_ids_bind_only_their_inputs() -> None:
    measurement = derive_measurement_protocol_id(
        config_projection={"probe": 64}, registry_digest=D0,
        component_digests={"encoder": D1},
    )
    oracle_a = derive_oracle_protocol_id(
        config_projection={"candidate_ids": ["a"]}, registry_digest=D0,
        component_digests={"bundles": D2},
    )
    oracle_b = derive_oracle_protocol_id(
        config_projection={"candidate_ids": ["b"]}, registry_digest=D0,
        component_digests={"bundles": D2},
    )
    assert oracle_a != oracle_b
    assert measurement == derive_measurement_protocol_id(
        config_projection={"probe": 64}, registry_digest=D0,
        component_digests={"encoder": D1},
    )
    variant = "v01v-" + "a" * 20
    run_id = derive_measurement_run_id(
        measurement_protocol_id=measurement,
        variant_schema_view_digests={variant: D3},
        pair_plan_digest=D2,
    )
    full = derive_experiment_protocol_id(
        measurement_run_id=run_id, oracle_protocol_id=oracle_a,
        analysis_projection={"bootstrap": 10000}, component_digests={"gates": D1},
    )
    assert all(len(item) == 64 for item in (measurement, oracle_a, run_id, full))
