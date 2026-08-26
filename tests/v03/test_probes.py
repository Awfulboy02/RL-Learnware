from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_v0.probe.gaussian import GaussianRandomProbe
from policy_learnware_v0.v03.probes import (
    CP0_STYLE_ID,
    CP1_OU_STYLE_ID,
    CP1_SWEEP_STYLE_ID,
    CP2_STYLE_ID,
    FROZEN_PROBE_STYLES,
    ActionABI,
    FrozenProbePolicy,
    ProbeContractError,
    ProbePolicyProtocol,
    ProbeSeedBinding,
    ProbeStyle,
    ProbeTrainingManifest,
    registered_probe,
)


def _abi() -> ActionABI:
    return ActionABI(
        low=np.asarray([-2.0, 1.0], dtype=np.float32),
        high=np.asarray([2.0, 5.0], dtype=np.float32),
    )


def test_normalized_action_mapping_respects_asymmetric_native_bounds() -> None:
    abi = _abi()
    np.testing.assert_array_equal(abi.map_normalized([-1.0, -1.0]), [-2.0, 1.0])
    np.testing.assert_array_equal(abi.map_normalized([1.0, 1.0]), [2.0, 5.0])
    np.testing.assert_array_equal(abi.map_normalized([0.0, 0.0]), [0.0, 3.0])
    with pytest.raises(ProbeContractError, match="outside"):
        abi.map_normalized([1.2, 0.0])


@pytest.mark.parametrize(
    "style_id", [CP0_STYLE_ID, CP1_OU_STYLE_ID, CP1_SWEEP_STYLE_ID, CP2_STYLE_ID]
)
def test_registered_probe_is_protocol_conformant_deterministic_and_bounded(
    style_id: str,
) -> None:
    probe = registered_probe(style_id)
    assert isinstance(probe, ProbePolicyProtocol)
    trajectories = []
    for _ in range(2):
        state = probe.reset(17, _abi())
        values = []
        for step in range(20):
            action, state = probe.act(
                np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
                state,
                step=step,
            )
            assert action.shape == (2,)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)
            assert not action.flags.writeable
            native = _abi().map_normalized(action)
            assert np.all(native >= _abi().low) and np.all(native <= _abi().high)
            values.append(action)
        trajectories.append(np.stack(values))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_probe_step_state_and_candidate_independence_fail_closed() -> None:
    probe = registered_probe(CP0_STYLE_ID)
    state = probe.reset(3, _abi())
    with pytest.raises(ProbeContractError, match="contiguous"):
        probe.act(np.asarray([0.0]), state, step=1)
    with pytest.raises(ProbeContractError, match="forbidden"):
        ProbeStyle(
            probe_family_id="bad",
            probe_style_id="bad",
            regime="CP1_FAMILY_SHIFT",
            implementation="gaussian_white",
            parameters={"candidate": 1},
            freeze_authority="test",
            eligible_for_encoder_training=True,
        )


def test_source_and_target_seed_namespaces_are_cryptographically_separate() -> None:
    common = {
        "style_id": CP0_STYLE_ID,
        "namespace": "paper1",
        "nonce": "frozen-nonce",
        "episode_id": 4,
    }
    source = ProbeSeedBinding(role="source_reference", **common)
    target = ProbeSeedBinding(role="target_query", **common)
    assert source.seed != target.seed
    assert source.digest != target.digest


def test_cp0_backend_neutral_reference_reuses_existing_gaussian_sequence() -> None:
    probe = registered_probe(CP0_STYLE_ID)
    state = probe.reset(37, _abi())
    actual = []
    for step in range(8):
        action, state = probe.act(np.asarray([0.0]), state, step=step)
        actual.append(action)
    expected = GaussianRandomProbe(sigma=1.0).sample_sequence_numpy(
        seed=37,
        steps=8,
        action_low=-np.ones(2, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32),
    )
    np.testing.assert_array_equal(np.stack(actual), expected)


def test_cp2_holdout_cannot_enter_encoder_training_manifest() -> None:
    manifest = ProbeTrainingManifest(
        training_style_ids=(CP0_STYLE_ID, CP1_OU_STYLE_ID, CP1_SWEEP_STYLE_ID),
        confirmatory_style_id=CP2_STYLE_ID,
        fold_ids=("fold-a", "fold-b"),
        freeze_authority="development-contract-test",
    )
    assert manifest.confirmatory_style_id not in manifest.training_style_ids
    assert not FROZEN_PROBE_STYLES[CP2_STYLE_ID].eligible_for_encoder_training
    with pytest.raises(ProbeContractError, match="training style IDs|CP2|non-training"):
        ProbeTrainingManifest(
            training_style_ids=(CP0_STYLE_ID, CP2_STYLE_ID),
            confirmatory_style_id=CP2_STYLE_ID,
            fold_ids=("fold-a",),
            freeze_authority="invalid",
        )
