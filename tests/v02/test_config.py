from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest

from policy_learnware_v0.v02.config import (
    V02ConfigError,
    V02ExperimentConfig,
    load_v02_config_draft,
    load_v02_formal_config,
)
from policy_learnware_v0.hashing import sha256_json


PROJECT = Path(__file__).resolve().parents[2]
TASKS = (
    "CartpoleSwingup",
    "CheetahRun",
    "FingerTurnEasy",
    "FishSwim",
    "ReacherEasy",
    "WalkerWalk",
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _qid(index: int) -> str:
    return f"v02q-{index:032x}"


def _formal_payload() -> dict:
    axes: dict[str, list[dict]] = {}
    factors: dict[str, dict[str, list[dict]]] = {}
    development: list[dict] = []
    confirmatory: list[dict] = []
    safety: list[dict] = []
    target_index = 1
    for task in TASKS:
        axes[task] = [
            {
                "axis_id": "axis_a",
                "operator_id": "mass_inertia_scale",
                "operator_digest": _d(f"{task}:axis_a:operator"),
                "leaf_allowlist": ["_mjx_model.body_mass", "_mjx_model.body_inertia"],
                "static_within_episode": True,
            },
            {
                "axis_id": "axis_b",
                "operator_id": "actuator_gain_scale",
                "operator_digest": _d(f"{task}:axis_b:operator"),
                "leaf_allowlist": ["_mjx_model.actuator_gainprm"],
                "static_within_episode": True,
            },
        ]
        nominal_anchor = _d(f"{task}:nominal-anchor")
        factors[task] = {}
        for axis_id in ("axis_a", "axis_b"):
            factors[task][axis_id] = [
                {
                    "factor_id": "low",
                    "value": 0.5,
                    "roles": ["source"],
                    "source_anchor_id": _d(f"{task}:{axis_id}:low-anchor"),
                    "axis_binding_digest": _d(f"{task}:{axis_id}:low-binding"),
                },
                {
                    "factor_id": "nominal",
                    "value": 1.0,
                    "roles": ["source"],
                    "source_anchor_id": nominal_anchor,
                    "axis_binding_digest": None,
                },
                {
                    "factor_id": "high",
                    "value": 2.0,
                    "roles": ["source"],
                    "source_anchor_id": _d(f"{task}:{axis_id}:high-anchor"),
                    "axis_binding_digest": _d(f"{task}:{axis_id}:high-binding"),
                },
            ]
            development.append(
                {
                    "target_id": _qid(target_index),
                    "task_id": task,
                    "axis_id": axis_id,
                    "factor_id": "development_mid",
                    "factor_value": 0.75,
                    "roles": ["development"],
                    "regime": "heldout_interpolation",
                    "source_anchor_ref": None,
                }
            )
            target_index += 1
            confirmatory.append(
                {
                    "target_id": _qid(target_index),
                    "task_id": task,
                    "axis_id": axis_id,
                    "factor_id": "confirmatory_mid",
                    "factor_value": 1.5,
                    "roles": ["confirmatory_heldout"],
                    "regime": "heldout_interpolation",
                    "source_anchor_ref": None,
                }
            )
            target_index += 1
        safety.append(
            {
                "target_id": _qid(target_index),
                "task_id": task,
                "axis_id": None,
                "factor_id": "nominal",
                "factor_value": 1.0,
                "roles": ["safety_exact_reference"],
                "regime": "safety_exact",
                "source_anchor_ref": nominal_anchor,
            }
        )
        target_index += 1

    return {
        "schema": "policy-learnware.v02-experiment-config.v0",
        "experiment_id": "v02-test-freeze-ready-r0",
        "stage": "v02_freeze_ready",
        "protocol_family_id": "continuous-vector-mdp-v02",
        "tasks": list(TASKS),
        "dynamics_axes": axes,
        "source_factors": factors,
        "development_targets": development,
        "confirmatory_targets": [],
        "safety_exact_targets": [],
        "primary_algorithm": "PPO",
        "training_steps": 10,
        "training_seeds": [11, 22, 33],
        "checkpoint_rule": "fixed_final_checkpoint",
        "source_eval_episodes": {"selection": 2, "attestation": 3},
        "competence_floor": {task: 0.5 for task in TASKS},
        "source_championization": {
            "mean_tolerance": 0.01,
            "lcb_z": 1.645,
            "competence_mode": "OBSERVE",
        },
        "probe_protocol_id": _d("probe"),
        "probe_prefixes": [1, 2, 4, 8, 16, 32],
        "encoder_eval_prefixes": [1, 2, 4, 8, 16, 32, 64],
        "representation_ids": ["raw_transition_v02", "corro_anchor_supcon_v02"],
        "method_ids": [
            "B0",
            "B1",
            "B2",
            "B3a",
            "B3b",
            "B4a",
            "B4b",
            "A-Env",
            "M02/B5",
        ],
        "primary_endpoint": "pool_regret",
        "noninferiority_margin": 0.01,
        "minimum_effect": 0.05,
        "bootstrap_plan": {
            "resamples": 100,
            "confidence": 0.95,
            "hierarchy": ["task", "axis", "context", "episode_bank"],
            "method": "deterministic_hierarchical_bootstrap",
        },
        "multiple_testing_plan": {
            "simultaneous_interval": "bootstrap_max-T",
            "p_value_adjustment": "holm_bonferroni",
            "alpha": 0.05,
            "families": ["primary_superiority", "nominal_noninferiority"],
        },
        "artifact_root": "/tmp/policy-learnware-v02-test",
    }


def test_complete_config_is_strict_immutable_and_has_separate_projections() -> None:
    config = V02ExperimentConfig.from_dict(_formal_payload())
    assert len(config.tasks) == 6
    assert all(len(config.dynamics_axes[task]) == 2 for task in config.tasks)
    assert all(
        len({
            factor.source_anchor_id
            for axis in config.dynamics_axes[task]
            for factor in config.source_factors[task][axis.axis_id]
        }) == 5
        for task in config.tasks
    )
    assert len(config.config_digest) == 64
    assert "training_steps" not in config.benchmark_projection
    assert "development_targets" not in config.training_projection
    with pytest.raises(FrozenInstanceError):
        config.stage = "audit_smoke"  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.competence_floor[config.tasks[0]] = 0.0  # type: ignore[index]


def test_formal_source_championization_is_required_and_digest_bound() -> None:
    payload = _formal_payload()
    payload.pop("source_championization")
    with pytest.raises(V02ConfigError, match="requires reviewed source_championization"):
        V02ExperimentConfig.from_dict(payload)

    baseline = V02ExperimentConfig.from_dict(_formal_payload())
    baseline_training_digest = sha256_json(baseline.training_projection)
    mutations = (
        lambda value: value["source_championization"].update(mean_tolerance=0.02),
        lambda value: value["source_championization"].update(lcb_z=1.96),
        lambda value: value["source_championization"].update(competence_mode="ENFORCE"),
        lambda value: value["source_eval_episodes"].update(selection=25),
        lambda value: value["competence_floor"].update({TASKS[0]: 0.51}),
    )
    for mutate in mutations:
        changed_payload = _formal_payload()
        mutate(changed_payload)
        changed = V02ExperimentConfig.from_dict(changed_payload)
        assert changed.config_digest != baseline.config_digest
        assert sha256_json(changed.training_projection) != baseline_training_digest


@pytest.mark.parametrize(
    "source_championization",
    [
        {"mean_tolerance": -0.01, "lcb_z": 1.645, "competence_mode": "OBSERVE"},
        {"mean_tolerance": 0.01, "lcb_z": float("inf"), "competence_mode": "OBSERVE"},
        {"mean_tolerance": 0.01, "lcb_z": 1.645},
        {"mean_tolerance": 0.01, "lcb_z": 1.645, "competence_mode": "OFF"},
        {
            "mean_tolerance": 0.01,
            "lcb_z": 1.645,
            "competence_mode": "OBSERVE",
            "unknown": 1,
        },
    ],
)
def test_source_championization_literals_fail_closed(source_championization) -> None:
    payload = _formal_payload()
    payload["source_championization"] = source_championization
    with pytest.raises(V02ConfigError):
        V02ExperimentConfig.from_dict(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(unknown=True),
        lambda payload: payload["dynamics_axes"][TASKS[0]][0].update(unknown=True),
        lambda payload: payload.update(primary_algorithm=None),
        lambda payload: payload.update(primary_endpoint="TBD"),
        lambda payload: payload.update(training_steps="REVIEW_REQUIRED"),
        lambda payload: payload.update(training_seeds=[1, 2]),
    ],
)
def test_unknown_null_tbd_and_incomplete_formal_values_fail_closed(mutation) -> None:
    payload = _formal_payload()
    mutation(payload)
    with pytest.raises(V02ConfigError):
        V02ExperimentConfig.from_dict(payload)


def test_source_development_confirmation_overlap_fails_closed() -> None:
    payload = _formal_payload()
    target = payload["development_targets"][0]
    target["factor_id"] = "low"
    target["factor_value"] = 0.5
    with pytest.raises(V02ConfigError, match="overlap source"):
        V02ExperimentConfig.from_dict(payload)

    payload = _formal_payload()
    dev = payload["development_targets"][0]
    target = copy.deepcopy(dev)
    target["target_id"] = _qid(999)
    target["roles"] = ["confirmatory_heldout"]
    payload["confirmatory_targets"] = [target]
    target["factor_id"] = dev["factor_id"]
    target["factor_value"] = dev["factor_value"]
    with pytest.raises(V02ConfigError, match="development and confirmatory"):
        V02ExperimentConfig.from_dict(payload)


def test_factor_roles_and_safety_exact_exception_are_explicit() -> None:
    payload = _formal_payload()
    payload["development_targets"][0]["roles"] = ["source"]
    with pytest.raises(V02ConfigError, match="invalid role"):
        V02ExperimentConfig.from_dict(payload)

    payload = _formal_payload()
    payload["safety_exact_targets"] = [
        {
            "target_id": _qid(998),
            "task_id": TASKS[0],
            "axis_id": None,
            "factor_id": "nominal",
            "factor_value": 1.0,
            "roles": ["safety_exact_reference"],
            "regime": "safety_exact",
            "source_anchor_ref": _d("not-source"),
        }
    ]
    with pytest.raises(V02ConfigError, match="not a source anchor"):
        V02ExperimentConfig.from_dict(payload)


def test_two_axes_must_share_exactly_one_nominal_anchor() -> None:
    payload = _formal_payload()
    payload["source_factors"][TASKS[0]]["axis_b"][1]["source_anchor_id"] = _d(
        "duplicate-nominal-bug"
    )
    with pytest.raises(V02ConfigError, match="share one canonical nominal anchor"):
        V02ExperimentConfig.from_dict(payload)


def test_rfc_discovery_is_recordable_and_reviewed_freeze_is_executable() -> None:
    discovery = load_v02_config_draft(PROJECT / "configs" / "v02_discovery.yaml")
    reviewed = load_v02_config_draft(PROJECT / "configs" / "v02_freeze_ready.yaml")
    assert discovery.unresolved_fields
    assert reviewed.unresolved_fields == ()
    assert discovery.config_digest != reviewed.config_digest

    formal = load_v02_formal_config(PROJECT / "configs" / "v02_freeze_ready.yaml")
    assert formal.config_digest == reviewed.config_digest
    assert formal.stage == "v02_freeze_ready"
    assert len(formal.source_anchor_ids) == 30


def test_config_round_trip_is_canonical() -> None:
    original = V02ExperimentConfig.from_dict(_formal_payload())
    restored = V02ExperimentConfig.from_dict(copy.deepcopy(original.to_dict()))
    assert restored.config_digest == original.config_digest
    assert restored.benchmark_projection == original.benchmark_projection
