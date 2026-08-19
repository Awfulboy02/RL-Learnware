from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from policy_learnware_v0.v01.config import (
    APPROVED_FORMAL_CONFIG_DIGEST,
    V01ConfigError,
    V01ExperimentConfig,
    load_v01_experiment_config,
)


PROJECT = Path(__file__).resolve().parents[2]


def _formal_payload() -> dict:
    return yaml.safe_load((PROJECT / "configs" / "dmc2_damping_v01.yaml").read_text())


def test_bundled_formal_and_smoke_configs_are_strict_and_immutable() -> None:
    formal = load_v01_experiment_config(PROJECT / "configs" / "dmc2_damping_v01.yaml")
    smoke = load_v01_experiment_config(PROJECT / "configs" / "v01_smoke.yaml")
    assert formal.tasks.all == ("WalkerWalk", "FingerTurnEasy")
    assert formal.shift.diagnostic_grid == (0.5, 0.75, 1.0, 1.5, 2.0)
    assert formal.probe.sparse_within_bank_pairs[-1] == (8, 9)
    assert formal.oracle.episodes_per_candidate_variant == 50
    assert smoke.tasks.all == ("WalkerWalk",)
    assert smoke.probe.banks == 2
    assert len(formal.config_digest) == 64
    assert formal.config_digest == APPROVED_FORMAL_CONFIG_DIGEST
    assert smoke.config_digest != APPROVED_FORMAL_CONFIG_DIGEST
    with pytest.raises(FrozenInstanceError):
        formal.project_seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(unexpected=True),
        lambda p: p["base"].pop("pool_id"),
        lambda p: p["shift"].update(diagnostic_grid=[0.5, 1.0, 1.0, 2.0]),
        lambda p: p["tasks"].update(infrastructure=["WalkerRun"]),
        lambda p: p["shift"].update(factor=1.5),
        lambda p: p["oracle"].update(paired_across_candidates=True),
    ],
)
def test_missing_unknown_or_unapproved_protocol_drift_fails_closed(mutation) -> None:
    payload = _formal_payload()
    mutation(payload)
    with pytest.raises(V01ConfigError):
        V01ExperimentConfig.from_dict(payload)


def test_typed_identity_projections_separate_measurement_oracle_and_analysis() -> None:
    payload = _formal_payload()
    original = V01ExperimentConfig.from_dict(copy.deepcopy(payload))

    payload["oracle"]["episodes_per_candidate_variant"] = 25
    oracle_changed = V01ExperimentConfig.from_dict(payload)
    assert original.measurement_config_digest == oracle_changed.measurement_config_digest
    assert original.oracle_config_digest != oracle_changed.oracle_config_digest
    assert original.analysis_config_digest == oracle_changed.analysis_config_digest

    payload = _formal_payload()
    payload["statistics"]["bootstrap_resamples"] = 5000
    analysis_changed = V01ExperimentConfig.from_dict(payload)
    assert original.measurement_config_digest == analysis_changed.measurement_config_digest
    assert original.oracle_config_digest == analysis_changed.oracle_config_digest
    assert original.analysis_config_digest != analysis_changed.analysis_config_digest


def test_yaml_parse_error_is_wrapped(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("x: [", encoding="utf-8")
    with pytest.raises(V01ConfigError):
        load_v01_experiment_config(path)
