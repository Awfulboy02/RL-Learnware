from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from typing import Any

import numpy as np

from repro_fpo_ppo_v02.anchor_binding import (
    AnchorBindingError,
    AnchorManifest,
    derive_live_model_diff,
)
from repro_fpo_ppo_v02.generate_anchor_manifest import (
    AXIS_ANCHOR_BINDING_SCHEMA,
    REVIEWED_ANCHOR_SPEC_SCHEMA,
    materialize_anchor_manifest,
    materialize_anchor_manifest_file,
    validate_reviewed_anchor_spec,
    verify_pinned_playground_dependency,
)
from repro_fpo_ppo_v02.provenance import ContractError, sha256_json


COMMIT = "b" * 40
RUNTIME = {
    "fpo_commit": COMMIT,
    "python_major_minor": "3.10",
    "jax": "synthetic-jax",
    "jaxlib": "synthetic-jaxlib",
    "mujoco": "synthetic-mujoco",
    "playground": "synthetic-playground",
}
CONFIG = {"episode_length": 64, "action_repeat": 1}


@dataclass(frozen=True)
class CoupledModel:
    body_mass: np.ndarray
    body_inertia: np.ndarray
    dof_damping: np.ndarray

    def tree_replace(self, replacements: dict[str, Any]) -> "CoupledModel":
        return replace(self, **replacements)


@dataclass(frozen=True)
class PoisonModel(CoupledModel):
    def tree_replace(self, replacements: dict[str, Any]) -> "PoisonModel":
        # Simulates an implementation that silently mutates an unreviewed leaf.
        return replace(
            self,
            **replacements,
            dof_damping=self.dof_damping * np.float32(3.0),
        )


@dataclass(frozen=True)
class SameLeafPoisonModel(CoupledModel):
    def tree_replace(self, replacements: dict[str, Any]) -> "SameLeafPoisonModel":
        poisoned = dict(replacements)
        if "body_mass" in poisoned:
            body_mass = np.asarray(poisoned["body_mass"]).copy()
            body_mass[0] *= np.float32(3.0)
            poisoned["body_mass"] = body_mass
        return replace(self, **poisoned)


class FakeEnv:
    def __init__(self, model: CoupledModel) -> None:
        self._mjx_model = model

    @property
    def mjx_model(self) -> CoupledModel:
        return self._mjx_model


class FakeRegistry:
    def __init__(self, model: CoupledModel, events: list[str] | None = None) -> None:
        self.model = model
        self.events = events if events is not None else []

    def get_default_config(self, task: str) -> dict[str, Any]:
        self.events.append("config")
        if task != "SyntheticTask":
            raise AssertionError("unexpected task")
        return dict(CONFIG)

    def load(self, task: str, *, config: dict[str, Any]) -> FakeEnv:
        self.events.append("load")
        if task != "SyntheticTask" or config != CONFIG:
            raise AssertionError("unexpected registry load")
        model_type = type(self.model)
        return FakeEnv(
            model_type(
                body_mass=self.model.body_mass.copy(),
                body_inertia=self.model.body_inertia.copy(),
                dof_damping=self.model.dof_damping.copy(),
            )
        )


class NativeRegistryDependencyTests(unittest.TestCase):
    def test_exact_installed_playground_pin_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "playground" / "pyproject.toml"
            project.parent.mkdir()
            project.write_text(
                '[project]\nname = "fixture"\ndependencies = ["playground==0.0.5"]\n',
                encoding="utf-8",
            )
            with patch(
                "repro_fpo_ppo_v02.generate_anchor_manifest.importlib.metadata.version",
                return_value="0.0.5",
            ):
                self.assertEqual(verify_pinned_playground_dependency(root), "0.0.5")

            project.write_text(
                '[project]\nname = "fixture"\ndependencies = ["playground>=0.0.5"]\n',
                encoding="utf-8",
            )
            with patch(
                "repro_fpo_ppo_v02.generate_anchor_manifest.importlib.metadata.version",
                return_value="0.0.5",
            ), self.assertRaisesRegex(ContractError, "does not pin the exact"):
                verify_pinned_playground_dependency(root)

    def test_playground_pin_file_must_not_be_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            external = root / "external.toml"
            external.write_text(
                '[project]\nname = "fixture"\ndependencies = ["playground==0.0.5"]\n',
                encoding="utf-8",
            )
            project = root / "playground" / "pyproject.toml"
            project.parent.mkdir()
            project.symlink_to(external)
            with self.assertRaisesRegex(ContractError, "regular playground"):
                verify_pinned_playground_dependency(root)


def model(*, poison: bool = False) -> CoupledModel:
    model_type = PoisonModel if poison else CoupledModel
    return model_type(
        body_mass=np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        body_inertia=np.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=np.float32,
        ),
        dof_damping=np.asarray([0.25, 0.5], dtype=np.float32),
    )


def common(kind: str) -> dict[str, Any]:
    return {
        "schema": REVIEWED_ANCHOR_SPEC_SCHEMA,
        "kind": kind,
        "task": "SyntheticTask",
        "backend": "mujoco_playground.registry",
        "registry_config": dict(CONFIG),
        "runtime": dict(RUNTIME),
    }


def binding(
    *, axis_id: str, factor_id: str, operator_digest: str, model_diff_digest: str
) -> dict[str, Any]:
    value = {
        "schema": AXIS_ANCHOR_BINDING_SCHEMA,
        "axis_id": axis_id,
        "factor_id": factor_id,
        "operator_digest": operator_digest,
        "model_diff_digest": model_diff_digest,
    }
    return {"axis_binding_digest": sha256_json(value), **value}


def mass_inertia_spec() -> dict[str, Any]:
    value = common("shifted")
    operator_source_digest = "d" * 64
    nominal = model()
    shifted_mass = nominal.body_mass.copy()
    shifted_mass[[1, 2]] *= np.float32(2.0)
    shifted_inertia = nominal.body_inertia.copy()
    shifted_inertia.reshape(-1)[[3, 4, 5, 6, 7, 8]] *= np.float32(2.0)
    shifted = nominal.tree_replace(
        {"body_mass": shifted_mass, "body_inertia": shifted_inertia}
    )
    _, model_diff_digest = derive_live_model_diff(nominal, shifted)
    value.update(
        {
            "axis": {
                "axis_id": "synthetic-mass-inertia",
                "axis_registry_digest": "a" * 64,
            },
            "operator": {
                "operator_id": "mass_inertia_scale_v02",
                "operator_source_digest": operator_source_digest,
                "factor_id": "source-high",
                "factor": 2.0,
                "mutations": [
                    {
                        "leaf": "_mjx_model.body_inertia",
                        "flat_indices": [3, 4, 5, 6, 7, 8],
                    },
                    {"leaf": "_mjx_model.body_mass", "flat_indices": [1, 2]},
                ],
            },
            "axis_binding": binding(
                axis_id="synthetic-mass-inertia",
                factor_id="source-high",
                operator_digest=operator_source_digest,
                model_diff_digest=model_diff_digest,
            ),
        }
    )
    return value


def materialize(spec: dict[str, Any], registry: FakeRegistry) -> dict[str, Any]:
    return materialize_anchor_manifest(
        spec,
        registry=registry,
        resolve_commit=lambda: COMMIT,
        resolve_runtime=lambda _: dict(RUNTIME),
    )


class GenerateAnchorManifestTests(unittest.TestCase):
    def test_shifted_materialization_is_deterministic_and_runtime_precedes_registry(self) -> None:
        events: list[str] = []

        def commit() -> str:
            events.append("commit")
            return COMMIT

        def runtime(_: str) -> dict[str, Any]:
            events.append("runtime")
            return dict(RUNTIME)

        first = materialize_anchor_manifest(
            mass_inertia_spec(),
            registry=FakeRegistry(model(), events),
            resolve_commit=commit,
            resolve_runtime=runtime,
        )
        second = materialize(mass_inertia_spec(), FakeRegistry(model()))
        self.assertEqual(first, second)
        self.assertEqual(events, ["commit", "runtime", "config", "load"])

        manifest = AnchorManifest.from_dict(first)
        self.assertFalse(manifest.nominal)
        self.assertEqual(manifest.factor, 2.0)
        self.assertNotEqual(
            manifest.expected_nominal_model_digest,
            manifest.expected_bound_model_digest,
        )
        self.assertEqual(
            tuple(item.leaf for item in manifest.operator.mutations),
            ("_mjx_model.body_inertia", "_mjx_model.body_mass"),
        )
        self.assertTrue(
            all(
                row.expected_before_digest != row.expected_after_digest
                for row in manifest.operator.mutations
            )
        )

    def test_nominal_union_has_one_canonical_unbound_anchor(self) -> None:
        manifest = AnchorManifest.from_dict(materialize(common("nominal"), FakeRegistry(model())))
        self.assertTrue(manifest.nominal)
        self.assertEqual(manifest.factor, 1.0)
        self.assertIsNone(manifest.operator)
        self.assertIsNone(manifest.operator_digest)
        self.assertIsNone(manifest.axis_binding_digest)
        self.assertEqual(
            manifest.expected_nominal_model_digest,
            manifest.expected_bound_model_digest,
        )

    def test_reviewed_model_diff_digest_must_match_live_changed_indices(self) -> None:
        poisoned = mass_inertia_spec()
        poisoned["axis_binding"] = binding(
            axis_id="synthetic-mass-inertia",
            factor_id="source-high",
            operator_digest="d" * 64,
            model_diff_digest="e" * 64,
        )
        with self.assertRaisesRegex(ContractError, "live model diff"):
            materialize(poisoned, FakeRegistry(model()))

    def test_strict_union_unknown_leaf_extra_and_bad_coupling_fail_closed(self) -> None:
        nominal_extra = common("nominal")
        nominal_extra["axis"] = {}
        with self.assertRaisesRegex(ContractError, "unknown=.*axis"):
            validate_reviewed_anchor_spec(nominal_extra)

        unknown_leaf = mass_inertia_spec()
        unknown_leaf["operator"]["mutations"][0]["leaf"] = "_mjx_model.secret"
        with self.assertRaisesRegex(ContractError, "unallowlisted"):
            validate_reviewed_anchor_spec(unknown_leaf)

        extra = mass_inertia_spec()
        extra["operator"]["mutations"][0]["unreviewed"] = True
        with self.assertRaisesRegex(ContractError, "unknown=.*unreviewed"):
            validate_reviewed_anchor_spec(extra)

        uncoupled = mass_inertia_spec()
        uncoupled["operator"]["mutations"][0]["flat_indices"] = [3, 4]
        with self.assertRaisesRegex(ContractError, "couple every principal component"):
            materialize(uncoupled, FakeRegistry(model()))

    def test_poisoned_runtime_and_out_of_allowlist_tree_replace_are_rejected(self) -> None:
        touched: list[str] = []
        with self.assertRaisesRegex(ContractError, "commit mismatch"):
            materialize_anchor_manifest(
                mass_inertia_spec(),
                registry=FakeRegistry(model(), touched),
                resolve_commit=lambda: "c" * 40,
                resolve_runtime=lambda _: dict(RUNTIME),
            )
        self.assertEqual(touched, [])

        with self.assertRaisesRegex(AnchorBindingError, "escaped allowlist"):
            materialize(mass_inertia_spec(), FakeRegistry(model(poison=True)))

        nominal = model()
        same_leaf_poison = SameLeafPoisonModel(
            body_mass=nominal.body_mass,
            body_inertia=nominal.body_inertia,
            dof_damping=nominal.dof_damping,
        )
        with self.assertRaisesRegex(AnchorBindingError, "indices"):
            materialize(mass_inertia_spec(), FakeRegistry(same_leaf_poison))

    def test_file_materialization_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reviewed.json"
            output = root / "anchor.json"
            source.write_text(json.dumps(common("nominal")), encoding="utf-8")
            with (
                patch(
                    "repro_fpo_ppo_v02.generate_anchor_manifest._checkout_resolver",
                    return_value=lambda: COMMIT,
                ),
                patch(
                    "repro_fpo_ppo_v02.generate_anchor_manifest.runtime_contract_projection",
                    return_value=dict(RUNTIME),
                ),
            ):
                first = materialize_anchor_manifest_file(
                    spec_path=source,
                    output_path=output,
                    fpo_root=root,
                    registry_loader=lambda _: FakeRegistry(model()),
                )
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    materialize_anchor_manifest_file(
                        spec_path=source,
                        output_path=output,
                        fpo_root=root,
                        registry_loader=lambda _: FakeRegistry(model()),
                    )

    def test_axis_binding_fields_cannot_be_relabelled(self) -> None:
        drifted = deepcopy(mass_inertia_spec())
        drifted["axis_binding"]["factor_id"] = "source-low"
        with self.assertRaises(ContractError):
            validate_reviewed_anchor_spec(drifted)


if __name__ == "__main__":
    unittest.main()
