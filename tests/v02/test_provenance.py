"""Focused tests for v0.2's retained read-only provenance primitives."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from server.repro_fpo_ppo_v02.provenance import (
    ContractError,
    NumericalIntegrityError,
    TRAINING_JOB_SCHEMA,
    TRAINING_PLAN_SCHEMA,
    TRAINING_PROTOCOL_SCHEMA,
    json_ready,
    load_strict_json,
    sha256_json,
    validate_self_digest,
    validate_training_plan,
    with_self_digest,
)


def _training_plan() -> dict[str, object]:
    protocol = with_self_digest(
        {
            "schema": TRAINING_PROTOCOL_SCHEMA,
            "algorithm": "fpo",
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
        },
        key="protocol_digest",
    )
    job = with_self_digest(
        {
            "schema": TRAINING_JOB_SCHEMA,
            "job_id": "synthetic-anchor-seed0",
            "config_digest": "0" * 64,
            "execution_purpose": "development_discovery",
            "formal_protocol_freeze_digest": None,
            "anchor_manifest_path": "/frozen/anchors/synthetic.json",
            "anchor_manifest_digest": "1" * 64,
            "training_protocol": protocol,
            "training_protocol_digest": protocol["protocol_digest"],
            "seed": 0,
        },
        key="job_digest",
    )
    return with_self_digest(
        {
            "schema": TRAINING_PLAN_SCHEMA,
            "config_digest": "0" * 64,
            "execution_purpose": "development_discovery",
            "formal_protocol_freeze": None,
            "formal_protocol_freeze_digest": None,
            "jobs": [job],
            "expected_job_count": 1,
        },
        key="plan_digest",
    )


def test_strict_json_duplicate_nonfinite_and_self_digest(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON key"):
        load_strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ContractError, match="non-finite JSON constant"):
        load_strict_json(nonfinite)
    with pytest.raises(NumericalIntegrityError, match="non-finite"):
        json_ready({"nested": [float("inf")]})

    sealed = with_self_digest({"schema": "synthetic", "nested": {"value": 1}}, key="digest")
    assert validate_self_digest(sealed, key="digest", where="synthetic") == sealed["digest"]
    forged = deepcopy(sealed)
    forged["nested"]["value"] = 2
    with pytest.raises(ContractError, match="digest mismatch"):
        validate_self_digest(forged, key="digest", where="synthetic")


def test_training_plan_deep_tamper_fails_despite_fresh_outer_digest() -> None:
    plan = _training_plan()
    assert validate_training_plan(plan) == plan

    tampered = deepcopy(plan)
    tampered["jobs"][0]["training_protocol"]["trainer_config"]["num_timesteps"] = 256
    material = {key: value for key, value in tampered.items() if key != "plan_digest"}
    tampered["plan_digest"] = sha256_json(material)

    with pytest.raises(ContractError, match="protocol_digest mismatch"):
        validate_training_plan(tampered)
