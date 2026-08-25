from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from policy_learnware_v0.envs.base import SyntheticEnvAdapter
from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v02.axes import (
    JOINT_DAMPING_OPERATOR,
    SOURCE_ROLE,
    AxisRegistry,
    AxisRegistryEntry,
    FactorDefinition,
    LeafSelection,
    operator_source_digest,
)
from policy_learnware_v0.v02.schemas import (
    EnvironmentInstanceRecord,
    SourceAnchorRecord,
    canonical_environment_instance_projection as package_environment_projection,
    canonical_model_snapshot as package_model_snapshot,
    derive_live_model_diff as package_live_model_diff,
    canonical_model_diff_projection as package_model_diff_projection,
)
from policy_learnware_v0.v02.variant_env import VariantEnvironmentFactory
from server.repro_fpo_ppo_v02.anchor_binding import (
    canonical_model_diff_projection as server_model_diff_projection,
    canonical_environment_instance_projection as server_environment_projection,
    derive_environment_instance_digest as server_environment_instance_digest,
    derive_live_model_diff as server_live_model_diff,
    derive_source_anchor_id as server_source_anchor_id,
    snapshot_model as server_model_snapshot,
)
from server.repro_fpo_ppo_v02.generate_anchor_manifest import (
    REVIEWED_ANCHOR_SPEC_SCHEMA,
    materialize_anchor_manifest,
)


CONFIG = {"episode_length": 64, "action_repeat": 1}
RUNTIME = {
    "fpo_commit": "b" * 40,
    "python_major_minor": "3.11",
    "jax": "synthetic",
    "jaxlib": "synthetic",
    "mujoco": "synthetic",
    "playground": "synthetic",
}


@dataclass(frozen=True)
class FakeModel:
    body_mass: np.ndarray
    dof_damping: np.ndarray

    def tree_replace(self, replacements: dict[str, Any]) -> "FakeModel":
        return replace(self, **replacements)


class FakeEnv:
    def __init__(self) -> None:
        self._mjx_model = FakeModel(
            body_mass=np.asarray([1.0, 2.0], dtype=np.float32),
            dof_damping=np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
        )

    @property
    def mjx_model(self) -> FakeModel:
        return self._mjx_model


class FakeAdapter(SyntheticEnvAdapter):
    def __init__(self, environment: FakeEnv, task: str) -> None:
        super().__init__(task=task, observation_dim=3, action_dim=2, horizon=4)
        self._environment = environment

    @property
    def environment(self) -> FakeEnv:
        return self._environment


class FakeServerRegistry:
    def get_default_config(self, task: str) -> dict[str, Any]:
        assert task == "SyntheticTask"
        return dict(CONFIG)

    def load(self, task: str, *, config: dict[str, Any]) -> FakeEnv:
        assert task == "SyntheticTask"
        assert config == CONFIG
        return FakeEnv()


def _factory() -> VariantEnvironmentFactory:
    factors = (
        FactorDefinition("nominal", 1.0, frozenset({SOURCE_ROLE})),
        FactorDefinition("high", 2.0, frozenset({SOURCE_ROLE})),
    )
    entry = AxisRegistryEntry(
        axis_id="synthetic-damping",
        task_id="SyntheticTask",
        backend_id="mujoco_playground.registry",
        operator_id=JOINT_DAMPING_OPERATOR,
        operator_version="1",
        operator_digest=operator_source_digest(),
        selections=(LeafSelection("dof_damping", (0, 2), require_nonzero=True),),
        factors=factors,
    )
    return VariantEnvironmentFactory(
        registry=AxisRegistry({"damping": entry}),
        nominal_loader=lambda _: FakeEnv(),
        adapter_factory=lambda env, task, jit: FakeAdapter(env, task),
        registry_config_digests={"SyntheticTask": sha256_json(CONFIG)},
        runtime_digest=sha256_json(RUNTIME),
    )


def test_variant_build_and_server_manifest_share_exact_anchor_identity() -> None:
    factory = _factory()
    build = factory.create(
        task_id="SyntheticTask",
        axis_id="synthetic-damping",
        factor_id="high",
        role=SOURCE_ROLE,
        jit=False,
    )
    assert build.axis_binding is not None
    assert build.anchor_operator is not None
    manifest = materialize_anchor_manifest(
        {
            "schema": REVIEWED_ANCHOR_SPEC_SCHEMA,
            "kind": "shifted",
            "task": build.task_id,
            "backend": build.backend_id,
            "registry_config": CONFIG,
            "runtime": RUNTIME,
            "axis": {
                "axis_id": build.axis_id,
                "axis_registry_digest": factory.registry.digest,
            },
            "operator": {
                "operator_id": build.anchor_operator["operator_id"],
                "operator_source_digest": build.axis_binding.operator_digest,
                "factor_id": build.factor_id,
                "factor": build.factor_value,
                "mutations": [
                    {
                        "leaf": item["leaf"],
                        "flat_indices": list(item["flat_indices"]),
                    }
                    for item in build.anchor_operator["mutations"]
                ],
            },
            "axis_binding": build.axis_binding.to_dict(),
        },
        registry=FakeServerRegistry(),
        resolve_commit=lambda: RUNTIME["fpo_commit"],
        resolve_runtime=lambda _: dict(RUNTIME),
    )
    assert manifest["model_diff_digest"] == build.model_diff_digest
    assert manifest["environment_instance_digest"] == build.environment_instance_digest
    assert manifest["anchor_id"] == build.source_anchor_id
    with pytest.raises(TypeError):
        build.anchor_operator["factor"] = 3.0  # type: ignore[index]
    with pytest.raises(TypeError):
        build.anchor_operator["mutations"][0]["leaf"] = "poison"  # type: ignore[index]

    nominal_build = factory.create(
        task_id="SyntheticTask",
        axis_id="synthetic-damping",
        factor_id="nominal",
        role=SOURCE_ROLE,
        jit=False,
    )
    nominal_manifest = materialize_anchor_manifest(
        {
            "schema": REVIEWED_ANCHOR_SPEC_SCHEMA,
            "kind": "nominal",
            "task": nominal_build.task_id,
            "backend": nominal_build.backend_id,
            "registry_config": CONFIG,
            "runtime": RUNTIME,
        },
        registry=FakeServerRegistry(),
        resolve_commit=lambda: RUNTIME["fpo_commit"],
        resolve_runtime=lambda _: dict(RUNTIME),
    )
    assert nominal_manifest["model_diff_digest"] == nominal_build.model_diff_digest
    assert (
        nominal_manifest["environment_instance_digest"]
        == nominal_build.environment_instance_digest
    )
    assert nominal_manifest["anchor_id"] == nominal_build.source_anchor_id


def test_package_and_server_golden_projection_vectors_are_identical() -> None:
    nominal = FakeEnv()._mjx_model
    damping = nominal.dof_damping.copy()
    damping[[0, 2]] *= np.float32(2.0)
    bound = nominal.tree_replace({"dof_damping": damping})
    package_snapshot = package_model_snapshot(nominal)
    server_snapshot = server_model_snapshot(nominal)
    assert package_snapshot["digest"] == server_snapshot.digest
    assert package_snapshot["leaves"] == [
        item.to_dict() for item in server_snapshot.leaves
    ]
    assert package_live_model_diff(nominal, bound) == server_live_model_diff(
        nominal, bound
    )

    changes = (
        {
            "leaf": "_mjx_model.dof_damping",
            "before_digest": "1" * 64,
            "after_digest": "2" * 64,
            "changed_flat_indices": [0, 2],
        },
    )
    package_diff = package_model_diff_projection(
        nominal_model_digest="3" * 64,
        bound_model_digest="4" * 64,
        changes=changes,
    )
    server_diff = server_model_diff_projection(
        nominal_model_digest="3" * 64,
        bound_model_digest="4" * 64,
        changes=changes,
    )
    assert package_diff == server_diff
    model_diff_digest = sha256_json(package_diff)
    assert model_diff_digest == "8c8f0f72a406ce406ff20f84b52b8e4069b9cfb894e50515c8f99e184cae552d"
    material = {
        "task": "SyntheticTask",
        "backend": "mujoco_playground.registry",
        "nominal": False,
        "factor": 2.0,
        "environment_class": "synthetic.FakeEnv",
        "registry_config_digest": "5" * 64,
        "runtime_digest": "6" * 64,
        "expected_nominal_model_digest": "3" * 64,
        "expected_bound_model_digest": "4" * 64,
        "operator_digest": "7" * 64,
        "axis_binding_digest": "8" * 64,
        "model_diff_digest": model_diff_digest,
    }
    package_instance = EnvironmentInstanceRecord.create(**material)
    assert package_environment_projection(material) == server_environment_projection(material)
    server_instance = server_environment_instance_digest(material)
    assert package_instance.environment_instance_digest == server_instance
    assert server_instance == "19486b83d21f9ebe7649bd053ffb5796cb0e73c51b169a3aabc14024f86a1e78"
    package_anchor = SourceAnchorRecord.create(
        environment_instance_digest=server_instance,
        axis_binding_digest=material["axis_binding_digest"],
    )
    assert package_anchor.anchor_id == server_source_anchor_id(
        environment_instance_digest=server_instance,
        axis_binding_digest=material["axis_binding_digest"],
    )
    assert package_anchor.anchor_id == (
        "29d7a081362e2223de9e45662ee118d96532abe972bdf32f4d8883c4c717778f"
    )
