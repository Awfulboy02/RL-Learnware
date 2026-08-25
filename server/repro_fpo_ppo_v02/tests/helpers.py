from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ..anchor_binding import (
        ANCHOR_MANIFEST_SCHEMA,
        ANCHOR_OPERATOR_SCHEMA,
        array_digest,
        finalize_anchor_manifest,
        snapshot_model,
    )
    from ..provenance import (
        FORMAL_FREEZE_BINDING_SCHEMA,
        FORMAL_TRAINING_CONTRACT_SCHEMA,
        FORMAL_EXECUTION_PURPOSE,
        FORMAL_GPU_EXECUTION_MODE,
        TRAINING_PROTOCOL_SCHEMA,
        atomic_write_json,
        finalize_training_protocol,
        sha256_file,
        sha256_json,
        with_self_digest,
    )
except ImportError:  # pragma: no cover - top-level server test execution
    from repro_fpo_ppo_v02.anchor_binding import (
        ANCHOR_MANIFEST_SCHEMA,
        ANCHOR_OPERATOR_SCHEMA,
        array_digest,
        finalize_anchor_manifest,
        snapshot_model,
    )
    from repro_fpo_ppo_v02.provenance import (
        FORMAL_FREEZE_BINDING_SCHEMA,
        FORMAL_TRAINING_CONTRACT_SCHEMA,
        FORMAL_EXECUTION_PURPOSE,
        FORMAL_GPU_EXECUTION_MODE,
        TRAINING_PROTOCOL_SCHEMA,
        atomic_write_json,
        finalize_training_protocol,
        sha256_file,
        sha256_json,
        with_self_digest,
    )


@dataclass(frozen=True)
class FakeModel:
    body_mass: np.ndarray
    dof_damping: np.ndarray

    def tree_replace(self, replacements: dict[str, Any]) -> "FakeModel":
        return replace(self, **replacements)


class FakeEnv:
    def __init__(self, model: FakeModel) -> None:
        self._mjx_model = model

    @property
    def mjx_model(self) -> FakeModel:
        return self._mjx_model


class FakeRegistry:
    def __init__(self, model: FakeModel, config: dict[str, Any]) -> None:
        self._model = model
        self._config = config

    def get_default_config(self, task: str) -> dict[str, Any]:
        assert task == "SyntheticTask"
        return dict(self._config)

    def load(self, task: str, *, config: dict[str, Any]) -> FakeEnv:
        assert task == "SyntheticTask"
        assert config == self._config
        return FakeEnv(
            FakeModel(
                body_mass=self._model.body_mass.copy(),
                dof_damping=self._model.dof_damping.copy(),
            )
        )


def fake_model() -> FakeModel:
    return FakeModel(
        body_mass=np.asarray([1.0, 2.0], dtype=np.float32),
        dof_damping=np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
    )


def make_shifted_anchor(path: Path, *, marker: str | None = None) -> dict[str, Any]:
    nominal = fake_model()
    factor = 2.0
    shifted_damping = nominal.dof_damping.copy()
    shifted_damping[[0, 2]] *= factor
    shifted = nominal.tree_replace({"dof_damping": shifted_damping})
    config = {"episode_length": 64, "action_repeat": 1}
    if marker is not None:
        config["formal_anchor_marker"] = marker
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
    value = finalize_anchor_manifest(
        {
            "schema": ANCHOR_MANIFEST_SCHEMA,
            "task": "SyntheticTask",
            "backend": "mujoco_playground.registry",
            "nominal": False,
            "factor": factor,
            "environment_class": f"{FakeEnv.__module__}.{FakeEnv.__qualname__}",
            "registry_config": config,
            "runtime": {
                "fpo_commit": "b" * 40,
                "python_major_minor": "3.10",
                "jax": "synthetic",
                "jaxlib": "synthetic",
                "mujoco": "synthetic",
                "playground": "synthetic",
            },
            "expected_nominal_model_digest": snapshot_model(nominal).digest,
            "expected_bound_model_digest": snapshot_model(shifted).digest,
            "operator": operator,
            "axis_binding_digest": "c" * 64,
        }
    )
    atomic_write_json(path, value, overwrite=False)
    return value


def make_formal_freeze_binding(
    root: Path,
    *,
    anchor_ids: list[str],
    seeds: list[int],
    config_digest: str,
    algorithm: str,
    training_steps: int,
    checkpoint_rule: str,
    anchor_semantics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct a structurally valid, non-live formal binding for unit tests."""

    training_projection_digest = sha256_json(
        {
            "primary_algorithm": algorithm.upper(),
            "training_steps": training_steps,
            "training_seeds": seeds,
            "checkpoint_rule": checkpoint_rule,
        }
    )
    freeze_record = {
        "schema": "policy-learnware.v02-formal-protocol-freeze.v0",
        "experiment_id": "synthetic-formal-fixture",
        "stage": "v02_freeze_ready",
        "config_digest": config_digest,
        "config_file_sha256": "1" * 64,
        "benchmark_projection_digest": "2" * 64,
        "training_projection_digest": training_projection_digest,
        "probe_projection_digest": "3" * 64,
        "analysis_projection_digest": "4" * 64,
        "implementation_tree_digest": "5" * 64,
        "software_commit": "6" * 40,
        "worktree_clean_at_freeze": True,
        "sealed_target_state": "NOT_INSTANTIATED_OR_READ",
        "confirmatory_oracle_state": "NOT_READ",
        "maximum_authorized_status": "READY_FOR_V03_JOINT_CONFIRMATORY",
    }
    sorted_anchor_ids = sorted(anchor_ids)
    if anchor_semantics is None:
        anchor_semantics = [
            {
                "source_anchor_id": anchor_id,
                "task": "SyntheticTask",
                "nominal": True,
                "factor": 1.0,
                "factor_id": "nominal",
                "axis_id": None,
                "operator_id": None,
                "axis_binding_digest": None,
                "leaf_allowlist": [],
            }
            for anchor_id in sorted_anchor_ids
        ]
    anchor_semantics = sorted(
        anchor_semantics, key=lambda row: row["source_anchor_id"]
    )
    contract = with_self_digest(
        {
            "schema": FORMAL_TRAINING_CONTRACT_SCHEMA,
            "source_anchor_ids": sorted_anchor_ids,
            "source_anchors": anchor_semantics,
            "training_seeds": sorted(seeds),
            "primary_algorithm": algorithm,
            "training_steps": training_steps,
            "checkpoint_rule": checkpoint_rule,
            "training_projection_digest": training_projection_digest,
        },
        key="contract_digest",
    )
    return with_self_digest(
        {
            "schema": FORMAL_FREEZE_BINDING_SCHEMA,
            "config_path": str((root / "formal.yaml").resolve()),
            "freeze_manifest_path": str((root / "v02_freeze_manifest.json").resolve()),
            "config_digest": config_digest,
            "freeze_record": freeze_record,
            "freeze_digest": sha256_json(freeze_record),
            "training_contract": contract,
        },
        key="binding_digest",
    )


def make_protocol(*, algorithm: str = "ppo") -> dict[str, Any]:
    return finalize_training_protocol(
        {
            "schema": TRAINING_PROTOCOL_SCHEMA,
            "algorithm": algorithm,
            "trainer_config": {
                "num_timesteps": 128,
                "num_envs": 8,
                "num_minibatches": 2,
                "batch_size": 8,
                "unroll_length": 8,
            },
            "max_outer_iterations": 1,
            "export_outer_iterations": [1],
            "evaluation": {"enabled": False, "num_envs": 1, "base_seed": 0},
            "parity": {
                "atol": 1.0e-6,
                "rtol": 1.0e-6,
                "golden_sample_count": 8,
                "compiled_sample_count": 2,
            },
            "checkpoint_rule": "fixed_final",
        }
    )


def make_bundle(
    root: Path,
    *,
    finite: bool = True,
    algorithm: str = "ppo",
    task: str = "SyntheticTask",
    seed: int = 0,
    config_digest: str = "0" * 64,
    execution_purpose: str = FORMAL_EXECUTION_PURPOSE,
    execution_mode: str = FORMAL_GPU_EXECUTION_MODE,
    formal_eligible: bool = True,
    execution_evidence_digest: str = "f" * 64,
    attempt_root: str = "/synthetic/jobs/synthetic/attempt_001",
    provenance_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    actor = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    if not finite:
        actor[1, 0] = np.nan
    np.savez(
        root / "actor.npz",
        layer_00_kernel=actor,
        layer_00_bias=np.zeros(2, dtype=np.float32),
    )
    np.savez(
        root / "obs_stats.npz",
        count=np.asarray(1.0),
        mean=np.zeros(2, dtype=np.float32),
        var_sum=np.ones(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )
    np.savez(
        root / "golden_io.npz",
        observation=np.zeros((8, 2), dtype=np.float32),
        prng_key_data=np.asarray([0, 1], dtype=np.uint32),
        raw_action=np.zeros((8, 1), dtype=np.float32),
        environment_action=np.zeros((8, 1), dtype=np.float32),
    )
    atomic_write_json(
        root / "policy_spec.json",
        {
            "schema": "policy-learnware.policy-bundle.v0",
            "task": task,
            "algorithm": algorithm,
            "actor_weights_file": "actor.npz",
            "golden_parity_file": "golden_io.npz",
            "environment_action_transform": "tanh(raw_action)",
            "observation_preprocessing": {
                "statistics_file": "obs_stats.npz",
                "normalize": True,
            },
            "observation_size": 2,
            "action_size": 1,
            "training_config": {
                "num_timesteps": 128,
                "episode_length": 1000,
                "normalize_observations": True,
            },
        },
        overwrite=False,
    )
    atomic_write_json(
        root / "provenance.json",
        {
            "schema": "policy-learnware.policy-bundle.v0",
            "task": task,
            "algorithm": algorithm,
            "training_seed": seed,
            "outer_iteration": 1,
            "environment_steps": 128,
            "evaluation": None,
            "config_digest": config_digest,
            "execution_purpose": execution_purpose,
            "execution_mode": execution_mode,
            "formal_eligible": formal_eligible,
            "execution_evidence_digest": execution_evidence_digest,
            "attempt_root": attempt_root,
            **(provenance_extra or {}),
        },
        overwrite=False,
    )
    names = [
        "actor.npz",
        "golden_io.npz",
        "obs_stats.npz",
        "policy_spec.json",
        "provenance.json",
    ]
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
        for name in names
    }
    manifest = {
        "schema": "policy-learnware.policy-bundle.v0",
        "complete": True,
        "created_at": "synthetic",
        "algorithm": algorithm,
        "task": task,
        "seed": seed,
        "outer_iteration": 1,
        "environment_steps": 128,
        "files": files,
    }
    atomic_write_json(root / "bundle_manifest.json", manifest, overwrite=False)
    bundle_manifest_sha256 = sha256_file(root / "bundle_manifest.json")
    finiteness = with_self_digest(
        {
            "passed": True,
            "all_arrays_finite": True,
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "validated_file_digests": {
                name: files[name]["sha256"] for name in sorted(files)
            },
        },
        key="report_digest",
    )
    golden_parity = with_self_digest(
        {
            "passed": True,
            "raw_checked": True,
            "raw_max_abs_error": 0.0,
            "environment_max_abs_error": 0.0,
            "atol": 1.0e-6,
            "rtol": 1.0e-6,
            "sample_count": 8,
        },
        key="report_digest",
    )
    compiled_parity = with_self_digest(
        {
            "passed": True,
            "max_abs_error": 0.0,
            "atol": 1.0e-6,
            "rtol": 1.0e-6,
            "sample_count": 2,
            "next_keys_equal": True,
        },
        key="report_digest",
    )
    return {
        "outer_iteration": 1,
        "environment_steps": 128,
        "path": str(root.resolve()),
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "bundle_manifest_digest": sha256_json(manifest),
        "files": {name: files[name]["sha256"] for name in sorted(files)},
        "bundle_digest": bundle_manifest_sha256,
        "config_digest": config_digest,
        "execution_purpose": execution_purpose,
        "execution_mode": execution_mode,
        "formal_eligible": formal_eligible,
        "execution_evidence_digest": execution_evidence_digest,
        "finiteness_audit": finiteness,
        "golden_parity": golden_parity,
        "compiled_parity": compiled_parity,
    }
