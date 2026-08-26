from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v03.attribution import (
    ArchivedLegacyReference,
    AttributionError,
    AttributionGateEvidence,
    AttributionMeasurement,
    CallableLegacyReplayAdapter,
    run_attribution_replay,
)
from policy_learnware_v0.v03.transition_views import (
    V_FULL_LEGACY,
    TransitionBank,
    TransitionViewResult,
    apply_transition_view,
)


def _dataset() -> EpisodeDataset:
    observation = np.asarray(
        [[0.0, 0.1], [0.2, 0.4], [0.5, 0.8], [1.0, -0.1], [0.7, -0.3], [0.2, -0.5]],
        dtype=np.float32,
    )
    return EpisodeDataset(
        observation=observation,
        action=np.asarray([[0.1], [0.3], [-0.2], [0.4], [-0.5], [0.2]], dtype=np.float32),
        reward=np.asarray([0.0, 0.2, 0.4, 1.0, 0.8, 0.6], dtype=np.float32),
        next_observation=np.roll(observation, -1, axis=0) + 0.03,
        terminated=np.zeros(6, dtype=np.bool_),
        truncated=np.asarray([False, False, True, False, False, True]),
        episode_offsets=np.asarray([0, 3, 6]),
        reset_seeds=np.asarray([1, 2]),
        probe_seeds=np.asarray([3, 4]),
    )


class _LegacyAdapter:
    encoder_checkpoint_digest = sha256_json({"checkpoint": "legacy"})
    implementation_digest = sha256_json({"implementation": "legacy-test"})

    def __init__(self, *, insensitive: bool = False) -> None:
        self.insensitive = insensitive

    def replay(
        self,
        view: TransitionViewResult,
        *,
        prefix_episode_counts: tuple[int, ...],
    ) -> AttributionMeasurement:
        matrix = (
            view.feature_matrix
            if view.view_id == "V_RANDOM_ENCODER"
            else view.legacy_packed_matrix
        )
        if self.insensitive:
            paired_score = 0.75
            spread = 0.25
        else:
            if "observation" in view.channels and "next_observation" in view.channels:
                paired_score = float(
                    np.mean(
                        np.sum(
                            view.channels["observation"]
                            * view.channels["next_observation"],
                            axis=1,
                        )
                    )
                )
            else:
                paired_score = float(np.mean(np.abs(matrix)))
            spread = float(np.std(matrix))
        return AttributionMeasurement(
            view_id=view.view_id,
            task_group="cross-registered-task",
            shared_schema_group="walker-stand-walk-run",
            retrieval_metrics={"task_top1": paired_score, "mrr": 0.5 + 0.01 * spread},
            between_within_mmd_summaries={
                "between_bank": spread + 0.1,
                "within_bank": spread,
                "separation": 0.1,
            },
            prefix_curves={
                "task_top1": {
                    prefix: paired_score + 0.001 * prefix
                    for prefix in prefix_episode_counts
                }
            },
            failure_identifiability_notes=(
                "Shared-schema reward semantics require reward-included comparison.",
            ),
        )


def _bank() -> TransitionBank:
    return TransitionBank.from_episode_dataset(_dataset())


def _reference(adapter: _LegacyAdapter, bank: TransitionBank, *, delta: float = 0.0):
    full = adapter.replay(
        apply_transition_view(bank, V_FULL_LEGACY),
        prefix_episode_counts=(1, 2),
    )
    metrics = dict(full.flat_metrics)
    metrics["retrieval.task_top1"] += delta
    return ArchivedLegacyReference(
        archive_protocol_id="v0-frozen-exact-recurrence",
        archive_manifest_digest=sha256_json({"archive": "manifest"}),
        archived_dataset_digest=str(bank.archived_dataset_digest),
        canonical_bank_digest=bank.canonical_bank_digest,
        encoder_checkpoint_digest=adapter.encoder_checkpoint_digest,
        encoder_implementation_digest=adapter.implementation_digest,
        reference_metrics=metrics,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )


def test_all_views_replay_with_paired_reports_and_no_raw_mutation() -> None:
    bank = _bank()
    before = {name: np.array(value, copy=True) for name, value in bank.to_arrays(copy=False).items()}
    adapter = _LegacyAdapter()
    suite = run_attribution_replay(
        bank,
        adapter,
        _reference(adapter, bank),
        prefix_episode_counts=(1, 2),
        shuffle_seed=17,
    )
    assert suite.gate_evidence.gate_status == "DEVELOPMENT_PASS"
    assert suite.gate_evidence.full_legacy_replay_pass
    assert suite.gate_evidence.controls_fail_closed_pass
    assert not suite.gate_evidence.independently_recomputable_pass
    assert suite.gate_evidence.dynamics_interpretation == "LEGACY_ENCODER_DYNAMICS_SENSITIVE"
    assert len(suite.reports) > 10
    assert all(report.paired_deltas_vs_full_legacy for report in suite.reports)
    assert all(report.shuffled_control_deltas for report in suite.reports)
    for name, value in bank.to_arrays(copy=False).items():
        np.testing.assert_array_equal(value, before[name])


def test_development_runner_cannot_self_attest_archived_pass_and_mismatch_fails() -> None:
    bank = _bank()
    adapter = _LegacyAdapter()
    passed = run_attribution_replay(
        bank,
        adapter,
        _reference(adapter, bank),
        prefix_episode_counts=(1, 2),
    )
    assert passed.gate_evidence.gate_status == "DEVELOPMENT_PASS"
    assert passed.gate_evidence.evidence_scope == "SYNTHETIC"

    failed = run_attribution_replay(
        bank,
        adapter,
        _reference(adapter, bank, delta=0.25),
        prefix_episode_counts=(1, 2),
    )
    assert failed.gate_evidence.gate_status == "FAIL"
    assert any(
        reason.startswith("LEGACY_REPLAY_MISMATCH")
        for reason in failed.gate_evidence.failure_reasons
    )


def test_foundation_record_cannot_be_used_to_self_sign_formal_attribution_pass() -> None:
    from policy_learnware_v0.v03.attribution import AttributionGateEvidence

    with pytest.raises(AttributionError, match="formal attribution PASS is unavailable"):
        AttributionGateEvidence(
            gate_status="PASS",  # type: ignore[arg-type]
            evidence_scope="LEGACY_ARCHIVED",
            full_legacy_replay_pass=True,
            controls_fail_closed_pass=True,
            contribution_quantified_pass=True,
            shared_schema_explanation_pass=True,
            independently_recomputable_pass=True,
            dynamics_interpretation="LEGACY_ENCODER_DYNAMICS_SENSITIVE",
            maximum_legacy_replay_error=0.0,
            failure_reasons=(),
        )


def test_pairing_insensitive_legacy_encoder_is_evidence_not_engineering_failure() -> None:
    bank = _bank()
    adapter = _LegacyAdapter(insensitive=True)
    suite = run_attribution_replay(
        bank,
        adapter,
        _reference(adapter, bank),
        prefix_episode_counts=(1, 2),
    )
    assert suite.gate_evidence.gate_status == "DEVELOPMENT_PASS"
    assert (
        suite.gate_evidence.dynamics_interpretation
        == "LEGACY_ENCODER_NOT_DYNAMICS_SENSITIVE"
    )


def test_callable_bridge_binds_existing_legacy_replay_to_frozen_digests() -> None:
    bank = _bank()
    implementation = _LegacyAdapter()
    bridge = CallableLegacyReplayAdapter(
        encoder_checkpoint_digest=implementation.encoder_checkpoint_digest,
        implementation_digest=implementation.implementation_digest,
        replay_callable=lambda view, prefixes: implementation.replay(
            view, prefix_episode_counts=prefixes
        ),
    )
    suite = run_attribution_replay(
        bank,
        bridge,
        _reference(implementation, bank),
        prefix_episode_counts=(1, 2),
    )
    assert suite.gate_evidence.gate_status == "DEVELOPMENT_PASS"


def test_archived_label_cannot_hide_changed_adapted_arrays() -> None:
    bank = _bank()
    adapter = _LegacyAdapter()
    reference = _reference(adapter, bank)
    changed_observation = np.array(bank.observation, copy=True)
    changed_observation[0, 0] += 0.5
    changed = TransitionBank(
        observation=changed_observation,
        action=bank.action,
        reward=bank.reward,
        next_observation=bank.next_observation,
        terminated=bank.terminated,
        truncated=bank.truncated,
        episode_offsets=bank.episode_offsets,
        observation_mask=bank.observation_mask,
        action_mask=bank.action_mask,
        archived_dataset_digest=bank.archived_dataset_digest,
    )
    with pytest.raises(AttributionError, match="adapted transition arrays"):
        run_attribution_replay(
            changed,
            adapter,
            reference,
            prefix_episode_counts=(1, 2),
        )


def test_gate_record_cannot_claim_pass_with_failed_checks() -> None:
    with pytest.raises(AttributionError, match="formal attribution PASS"):
        AttributionGateEvidence(
            gate_status="PASS",
            evidence_scope="LEGACY_ARCHIVED",
            full_legacy_replay_pass=True,
            controls_fail_closed_pass=False,
            contribution_quantified_pass=True,
            shared_schema_explanation_pass=True,
            independently_recomputable_pass=True,
            dynamics_interpretation="UNASSESSED",
            maximum_legacy_replay_error=0.0,
            failure_reasons=(),
        )
