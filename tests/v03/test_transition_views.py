from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v03.windowing import CanonicalTransitionBatch
from policy_learnware_v0.v03.transition_views import (
    REGISTERED_VIEW_IDS,
    V_DIMS_ONLY,
    V_MASK_ONLY,
    V_NO_MASK,
    V_RANDOM_ENCODER,
    V_REWARD_FREE_TRANSITION,
    V_STATE_ACTION,
    V_STATE_ONLY,
    V_SHUFFLED_NEXT,
    V_SHUFFLED_REWARD,
    V_TEMPORAL_SHUFFLE,
    TransitionBank,
    TransitionViewError,
    apply_transition_view,
)


def _dataset() -> EpisodeDataset:
    observation = np.asarray(
        [
            [0.0, 0.1, 0.0],
            [0.2, 0.3, 0.0],
            [0.4, 0.5, 0.0],
            [1.0, 1.1, 0.0],
            [1.2, 1.3, 0.0],
            [1.4, 1.5, 0.0],
        ],
        dtype=np.float32,
    )
    return EpisodeDataset(
        observation=observation,
        action=np.asarray(
            [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [-0.1, 0.0], [-0.2, 0.0], [-0.3, 0.0]],
            dtype=np.float32,
        ),
        reward=np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0], dtype=np.float32),
        next_observation=observation + np.asarray([0.05, -0.02, 0.0], dtype=np.float32),
        terminated=np.asarray([False, False, False, False, False, False]),
        truncated=np.asarray([False, False, True, False, False, True]),
        episode_offsets=np.asarray([0, 3, 6]),
        reset_seeds=np.asarray([1, 2]),
        probe_seeds=np.asarray([3, 4]),
    )


def _bank() -> TransitionBank:
    return TransitionBank.from_episode_dataset(
        _dataset(),
        observation_mask=np.asarray([True, True, False]),
        action_mask=np.asarray([True, False]),
    )


def test_every_registered_view_is_finite_immutable_and_channel_exact() -> None:
    bank = _bank()
    assert bank.archived_dataset_digest == _dataset().digest
    assert bank.canonical_bank_digest != bank.archived_dataset_digest
    for view_id in REGISTERED_VIEW_IDS:
        result = apply_transition_view(bank, view_id, shuffle_seed=19)
        assert result.feature_matrix.shape[0] == bank.transition_count
        assert np.all(np.isfinite(result.feature_matrix))
        assert not result.feature_matrix.flags.writeable
        expected = (
            {"random_embedding"}
            if view_id == V_RANDOM_ENCODER
            else set(result.spec.input_channel_allowlist)
        )
        assert set(result.channels) == expected
        assert all(not value.flags.writeable for value in result.channels.values())
        assert result.archived_dataset_digest == bank.archived_dataset_digest
        if view_id != V_RANDOM_ENCODER:
            # o/o-mask/a/a-mask/r/o'/o'-mask frozen legacy layout.
            assert result.legacy_packed_matrix.shape == (6, 4 * 3 + 2 * 2 + 1)
            assert not result.legacy_packed_matrix.flags.writeable


def test_shortcut_and_reward_free_views_do_not_expose_forbidden_values() -> None:
    bank = _bank()
    mask = apply_transition_view(bank, V_MASK_ONLY)
    assert set(mask.channels) == {
        "observation_mask",
        "action_mask",
        "next_observation_mask",
    }
    dims = apply_transition_view(bank, V_DIMS_ONLY)
    np.testing.assert_array_equal(dims.channels["observation_native_dim"], 2.0)
    np.testing.assert_array_equal(dims.channels["action_native_dim"], 1.0)

    reward_free = apply_transition_view(bank, V_REWARD_FREE_TRANSITION)
    assert "reward" not in reward_free.channels
    no_mask = apply_transition_view(bank, V_NO_MASK)
    assert all("mask" not in name for name in no_mask.channels)
    assert no_mask.padding_value == 0.0
    assert no_mask.padding_identity_audit["dimension_identity_inferable"]
    assert no_mask.padding_identity_audit["stable_action_padding_slots"] == (1,)
    state_only = apply_transition_view(bank, V_STATE_ONLY)
    assert state_only.padding_identity_audit["dimension_identity_inferable"]
    assert state_only.padding_identity_audit["unmasked_padding_channels"] == (
        "observation",
    )
    state_action = apply_transition_view(bank, V_STATE_ACTION)
    assert set(state_action.padding_identity_audit["unmasked_padding_channels"]) == {
        "action",
        "observation",
    }
    action_with_mask = apply_transition_view(bank, "V_ACTION_ONLY")
    assert not action_with_mask.padding_identity_audit[
        "dimension_identity_inferable"
    ]


def test_shuffle_controls_preserve_marginals_and_destroy_only_pairing() -> None:
    bank = _bank()
    next_control = apply_transition_view(bank, V_SHUFFLED_NEXT, shuffle_seed=7)
    np.testing.assert_array_equal(next_control.channels["observation"], bank.observation)
    np.testing.assert_array_equal(next_control.channels["reward"][:, 0], bank.reward)
    np.testing.assert_array_equal(
        np.sort(next_control.channels["next_observation"], axis=0),
        np.sort(bank.next_observation, axis=0),
    )
    assert not np.array_equal(next_control.next_source_indices, np.arange(6))
    np.testing.assert_array_equal(next_control.row_source_indices, np.arange(6))

    reward_control = apply_transition_view(bank, V_SHUFFLED_REWARD, shuffle_seed=11)
    np.testing.assert_array_equal(
        np.sort(reward_control.channels["reward"][:, 0]), np.sort(bank.reward)
    )
    np.testing.assert_array_equal(
        reward_control.channels["next_observation"], bank.next_observation
    )
    assert not np.array_equal(reward_control.reward_source_indices, np.arange(6))


def test_temporal_shuffle_stays_inside_episode_and_random_control_is_frozen() -> None:
    bank = _bank()
    temporal = apply_transition_view(bank, V_TEMPORAL_SHUFFLE, shuffle_seed=23)
    assert set(temporal.row_source_indices[:3]) == {0, 1, 2}
    assert set(temporal.row_source_indices[3:]) == {3, 4, 5}
    assert not np.array_equal(temporal.row_source_indices, np.arange(6))

    first = apply_transition_view(bank, V_RANDOM_ENCODER, shuffle_seed=29)
    second = apply_transition_view(bank, V_RANDOM_ENCODER, shuffle_seed=29)
    third = apply_transition_view(bank, V_RANDOM_ENCODER, shuffle_seed=31)
    np.testing.assert_array_equal(first.feature_matrix, second.feature_matrix)
    assert first.random_projection_digest == second.random_projection_digest
    assert first.random_projection_digest != third.random_projection_digest


def test_padding_contract_and_unknown_view_fail_closed() -> None:
    dataset = _dataset()
    broken = np.array(dataset.observation, copy=True)
    broken[0, -1] = 1.0
    with pytest.raises(TransitionViewError, match="padded observation"):
        TransitionBank(
            observation=broken,
            action=dataset.action,
            reward=dataset.reward,
            next_observation=dataset.next_observation,
            terminated=dataset.terminated,
            truncated=dataset.truncated,
            episode_offsets=dataset.episode_offsets,
            observation_mask=np.asarray([1, 1, 0]),
            action_mask=np.asarray([1, 0]),
        )
    with pytest.raises(TransitionViewError, match="unregistered"):
        apply_transition_view(_bank(), "V_POSTHOC")


def test_p3_canonical_batch_bridge_preserves_masks_and_episode_boundaries() -> None:
    dataset = _dataset()
    canonical = CanonicalTransitionBatch(
        observation=dataset.observation,
        action=dataset.action,
        reward=dataset.reward,
        next_observation=dataset.next_observation,
        terminated=dataset.terminated,
        truncated=dataset.truncated,
        observation_mask=np.asarray([True, True, False]),
        action_mask=np.asarray([True, False]),
        episode_id=np.asarray([7, 7, 7, 9, 9, 9]),
        timestep=np.asarray([0, 1, 2, 0, 1, 2]),
    )
    bank = TransitionBank.from_canonical_batch(canonical)
    np.testing.assert_array_equal(bank.episode_offsets, [0, 3, 6])
    np.testing.assert_array_equal(bank.observation_mask[0], [1, 1, 0])
    assert bank.archived_dataset_digest == canonical.transition_digest
