"""Focused tests for the frozen v0.2 source-anchor binding contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from server.repro_fpo_ppo_v02.anchor_binding import (
    ANCHOR_MANIFEST_SCHEMA,
    ANCHOR_OPERATOR_SCHEMA,
    AnchorBindingError,
    AnchorManifest,
    array_digest,
    derive_environment_instance_digest,
    derive_manifest_model_diff_digest,
    derive_source_anchor_id,
    load_and_bind_anchor,
    snapshot_model,
)
from server.repro_fpo_ppo_v02.provenance import (
    ContractError,
    sha256_json,
    with_self_digest,
)


@dataclass(frozen=True)
class _Model:
    body_mass: np.ndarray
    dof_damping: np.ndarray

    def tree_replace(self, replacements: dict[str, Any]) -> "_Model":
        return replace(self, **replacements)


class _Environment:
    def __init__(self, model: _Model) -> None:
        self._mjx_model = model

    @property
    def mjx_model(self) -> _Model:
        return self._mjx_model


class _Registry:
    def __init__(self, model: _Model, config: dict[str, Any]) -> None:
        self._model = model
        self._config = deepcopy(config)

    def get_default_config(self, task: str) -> dict[str, Any]:
        assert task == "SyntheticTask"
        return deepcopy(self._config)

    def load(self, task: str, *, config: dict[str, Any]) -> _Environment:
        assert task == "SyntheticTask"
        assert config == self._config
        return _Environment(
            _Model(
                body_mass=self._model.body_mass.copy(),
                dof_damping=self._model.dof_damping.copy(),
            )
        )


def _model() -> _Model:
    return _Model(
        body_mass=np.asarray([1.0, 2.0], dtype=np.float32),
        dof_damping=np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
    )


def _shifted_manifest() -> dict[str, Any]:
    nominal = _model()
    factor = 2.0
    shifted_damping = nominal.dof_damping.copy()
    shifted_damping[[0, 2]] *= factor
    shifted = nominal.tree_replace({"dof_damping": shifted_damping})
    config = {"action_repeat": 1, "nested": {"episode_length": 64}}
    runtime = {
        "fpo_commit": "b" * 40,
        "python_major_minor": "3.11",
        "jax": "synthetic",
        "jaxlib": "synthetic",
        "mujoco": "synthetic",
        "playground": "synthetic",
    }
    operator = {
        "schema": ANCHOR_OPERATOR_SCHEMA,
        "operator_id": "synthetic-damping-x2",
        "axis_id": "synthetic-damping",
        "axis_registry_digest": "a" * 64,
        "factor": factor,
        "mutations": [
            {
                "leaf": "_mjx_model.dof_damping",
                "flat_indices": [0, 2],
                "multiplier": factor,
                "expected_before_digest": array_digest(nominal.dof_damping),
                "expected_after_digest": array_digest(shifted_damping),
            }
        ],
    }
    value: dict[str, Any] = {
        "schema": ANCHOR_MANIFEST_SCHEMA,
        "anchor_id": "0" * 64,
        "task": "SyntheticTask",
        "backend": "mujoco_playground.registry",
        "nominal": False,
        "factor": factor,
        "environment_class": f"{_Environment.__module__}.{_Environment.__qualname__}",
        "registry_config": config,
        "registry_config_digest": sha256_json(config),
        "runtime": runtime,
        "runtime_digest": sha256_json(runtime),
        "expected_nominal_model_digest": snapshot_model(nominal).digest,
        "expected_bound_model_digest": snapshot_model(shifted).digest,
        "operator": operator,
        "operator_digest": sha256_json(operator),
        "axis_binding_digest": "c" * 64,
        "model_diff_digest": "0" * 64,
        "environment_instance_digest": "0" * 64,
    }
    value["model_diff_digest"] = derive_manifest_model_diff_digest(value)
    value["environment_instance_digest"] = derive_environment_instance_digest(value)
    value["anchor_id"] = derive_source_anchor_id(
        environment_instance_digest=value["environment_instance_digest"],
        axis_binding_digest=value["axis_binding_digest"],
    )
    return with_self_digest(value, key="manifest_digest")


def test_manifest_strict_json_and_nested_deep_immutability(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"first","schema":"second"}', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON key"):
        AnchorManifest.from_path(duplicate)

    raw = _shifted_manifest()
    manifest = AnchorManifest.from_dict(raw)
    original_digest = manifest.manifest_digest

    raw["registry_config"]["nested"]["episode_length"] = 999
    exported = manifest.to_dict()
    exported["registry_config"]["nested"]["episode_length"] = 888
    attribute_copy = manifest.registry_config
    attribute_copy["nested"]["episode_length"] = 777

    assert manifest.registry_config["nested"]["episode_length"] == 64
    assert manifest.manifest_digest == original_digest


def test_real_synthetic_bind_is_immutable_and_tamper_fails_closed() -> None:
    raw = _shifted_manifest()
    manifest = AnchorManifest.from_dict(raw)
    source = _model()
    source_digest = snapshot_model(source).digest

    bound = load_and_bind_anchor(
        registry=_Registry(source, raw["registry_config"]),
        manifest=manifest,
    )
    assert bound.env._mjx_model is not source
    assert bound.verify().digest == raw["expected_bound_model_digest"]
    assert bound.audit.changed_leaves == ("_mjx_model.dof_damping",)
    assert bound.audit.source_unchanged is True
    assert snapshot_model(source).digest == source_digest

    tampered_manifest = deepcopy(raw)
    tampered_manifest["operator"]["mutations"][0]["flat_indices"] = [1]
    with pytest.raises(ContractError):
        AnchorManifest.from_dict(tampered_manifest)

    poisoned_source = _model().tree_replace(
        {"dof_damping": np.asarray([0.3, 0.5, 1.0], dtype=np.float32)}
    )
    with pytest.raises(AnchorBindingError, match="nominal model digest"):
        load_and_bind_anchor(
            registry=_Registry(poisoned_source, raw["registry_config"]),
            manifest=manifest,
        )
