from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v03.probe_audit import (
    ProbeAuditError,
    ProbeDistanceEvidence,
    ProbeGateFreezeDecision,
    ProbeGateThresholds,
    evaluate_probe_gate,
    summarize_probe_bank,
)
from policy_learnware_v0.v03.probes import (
    CP0_STYLE_ID,
    CP1_OU_STYLE_ID,
    CP2_STYLE_ID,
    ActionABI,
    ProbeSeedBinding,
    ProbeTrainingManifest,
    registered_probe,
)
from policy_learnware_v0.v03.representation_ladder import R0_PADDED_RAW


COLLECTOR_IMPLEMENTATION_DIGEST = sha256_json(
    {"implementation": "typed-development-probe-collector-test-v0"}
)


def _abi() -> ActionABI:
    return ActionABI(
        low=np.asarray([-2.0, 1.0], dtype=np.float32),
        high=np.asarray([2.0, 5.0], dtype=np.float32),
    )


def _bindings(
    style_id: str,
    *,
    nonce: str = "bank-nonce",
    role: str = "target_query",
) -> tuple[ProbeSeedBinding, ...]:
    return tuple(
        ProbeSeedBinding(
            role=role,
            style_id=style_id,
            namespace="paper1-development",
            nonce=nonce,
            episode_id=episode_id,
        )
        for episode_id in range(2)
    )


def _dataset(
    style_id: str,
) -> tuple[EpisodeDataset, ActionABI, tuple[ProbeSeedBinding, ...]]:
    abi = _abi()
    bindings = _bindings(style_id)
    observation = np.asarray(
        [[0.1 * i, np.sin(i)] for i in range(8)], dtype=np.float32
    )
    probe = registered_probe(style_id)
    native_actions: list[np.ndarray] = []
    for episode_index, binding in enumerate(bindings):
        state = probe.reset(int(binding.seed), abi)
        for step in range(4):
            row = episode_index * 4 + step
            normalized, state = probe.act(observation[row], state, step=step)
            native_actions.append(abi.map_normalized(normalized))
    dataset = EpisodeDataset(
        observation=observation,
        action=np.stack(native_actions),
        reward=np.linspace(0.0, 1.0, 8, dtype=np.float32),
        next_observation=observation
        + np.asarray([0.08, -0.04], dtype=np.float32),
        terminated=np.zeros(8, dtype=np.bool_),
        truncated=np.asarray(
            [False, False, False, True, False, False, False, True]
        ),
        episode_offsets=np.asarray([0, 4, 8]),
        reset_seeds=np.asarray([1, 2]),
        probe_seeds=np.asarray([binding.seed for binding in bindings]),
    )
    return dataset, abi, bindings


def _replace_dataset(dataset: EpisodeDataset, **changes: np.ndarray) -> EpisodeDataset:
    arrays = dataset.to_arrays()
    arrays.update(changes)
    return EpisodeDataset(**arrays)


def _summary(task: str, style_id: str):
    dataset, abi, bindings = _dataset(style_id)
    return summarize_probe_bank(
        dataset,
        action_abi=abi,
        seed_bindings=bindings,
        collection_implementation_digest=COLLECTOR_IMPLEMENTATION_DIGEST,
        task_id=task,
        context_id=f"{task}-nominal-{style_id}",
        probe_style_id=style_id,
        collection_wall_seconds=2.0,
        stored_bytes=4096,
    )


def _manifest() -> ProbeTrainingManifest:
    return ProbeTrainingManifest(
        training_style_ids=(CP0_STYLE_ID, CP1_OU_STYLE_ID),
        confirmatory_style_id=CP2_STYLE_ID,
        fold_ids=("fold-a", "fold-b"),
        freeze_authority="synthetic-test",
    )


def _thresholds(**changes: float) -> ProbeGateThresholds:
    values = {
        "min_action_energy": 1.0e-4,
        "min_state_coverage": 1.0e-5,
        "min_raw_transition_signal": 1.0e-3,
        "min_different_dynamics_distance": 0.10,
        "minimum_signal_to_noise_ratio": 2.0,
        "maximum_invariance_ratio": 0.30,
        "maximum_probe_style_classifier_accuracy": 0.60,
        "max_saturation_rate": 0.95,
        "max_termination_rate": 0.50,
        "max_failure_rate": 0.0,
    }
    values.update(changes)
    return ProbeGateThresholds(**values)


def _freeze(
    tasks: tuple[str, ...],
    thresholds: ProbeGateThresholds,
) -> ProbeGateFreezeDecision:
    return ProbeGateFreezeDecision(
        required_task_ids=tasks,
        required_task_axis_pairs=tuple(
            (task_id, axis_id)
            for task_id in tasks
            for axis_id in ("mass", "damping")
        ),
        training_manifest_digest=_manifest().digest,
        thresholds_digest=thresholds.digest,
        decision_authority="p2-development-review",
    )


def _evidence(
    task: str,
    target_bank_digest: str,
    *,
    axis_id: str = "mass",
    signal: float = 0.40,
    noise: float = 0.05,
    cross_probe: float = 0.08,
    classifier_accuracy: float = 0.55,
) -> ProbeDistanceEvidence:
    return ProbeDistanceEvidence(
        task_id=task,
        axis_id=axis_id,
        representation_id=R0_PADDED_RAW,
        representation_protocol_digest=sha256_json(
            {"representation": R0_PADDED_RAW, "protocol": "probe-gate"}
        ),
        semantic_bank_digests={
            "target_query": target_bank_digest,
            "source_nominal": sha256_json(
                {"task": task, "axis": axis_id, "bank": "nominal"}
            ),
            "source_shifted": sha256_json(
                {"task": task, "axis": axis_id, "bank": "shifted"}
            ),
        },
        encoder_checkpoint_digest=sha256_json(
            {"task": task, "axis": axis_id, "checkpoint": "raw-identity"}
        ),
        distance_matrix_digest=sha256_json(
            {"task": task, "axis": axis_id, "matrix": "primary"}
        ),
        independent_recompute_digest=sha256_json(
            {
                "task": task,
                "axis": axis_id,
                "matrix": "independent-recompute-receipt",
            }
        ),
        same_environment_cross_probe_distances=(cross_probe, cross_probe * 1.1),
        different_dynamics_same_probe_distances=(signal, signal * 1.1),
        repeated_bank_noise_distances=(noise, noise * 1.1),
        probe_style_classifier_accuracy=classifier_accuracy,
    )


def _inputs():
    tasks = ("task-a", "task-b")
    summaries = tuple(
        _summary(task, style)
        for task in tasks
        for style in (CP0_STYLE_ID, CP2_STYLE_ID)
    )
    shared_target = {
        "task-a": sha256_json({"bank": "a"}),
        "task-b": sha256_json({"bank": "b"}),
    }
    evidence = tuple(
        _evidence(task, shared_target[task], axis_id=axis_id)
        for task in tasks
        for axis_id in ("mass", "damping")
    )
    bindings = {
        "encoder-a": dict(shared_target),
        "encoder-b": dict(shared_target),
    }
    thresholds = _thresholds()
    freeze = _freeze(tasks, thresholds)
    return tasks, summaries, evidence, bindings, thresholds, freeze


def _evaluate():
    _, summaries, evidence, bindings, thresholds, freeze = _inputs()
    return evaluate_probe_gate(
        summaries=summaries,
        distance_evidence=evidence,
        training_manifest=_manifest(),
        thresholds=thresholds,
        freeze_decision=freeze,
        target_bank_bindings_by_encoder=bindings,
    )


def test_bank_summary_replays_actions_and_emits_typed_collection_receipt() -> None:
    summary = _summary("task-a", CP0_STYLE_ID)
    assert summary.finite
    assert summary.action_energy > 0
    assert summary.state_coverage > 0
    assert summary.raw_transition_signal > 0
    assert summary.termination_rate == 0
    assert summary.failure_rate == 0
    assert [record.episode_count for record in summary.prefix_costs] == [1, 2]
    assert summary.prefix_costs[-1].wall_seconds == 2.0
    assert summary.prefix_costs[-1].stored_bytes == 4096
    assert summary.collection_receipt.dataset_digest == summary.dataset_digest
    assert (
        summary.collection_receipt.collection_implementation_digest
        == COLLECTOR_IMPLEMENTATION_DIGEST
    )
    assert summary.candidate_independence_pass


def test_collection_rejects_replaced_actions_seeds_and_style() -> None:
    dataset, abi, bindings = _dataset(CP0_STYLE_ID)
    changed_actions = np.array(dataset.action, copy=True)
    changed_actions[0, 0] += np.float32(0.25)
    with pytest.raises(ProbeAuditError, match="native actions differ"):
        summarize_probe_bank(
            _replace_dataset(dataset, action=changed_actions),
            action_abi=abi,
            seed_bindings=bindings,
            collection_implementation_digest=COLLECTOR_IMPLEMENTATION_DIGEST,
            task_id="task-a",
            context_id="tampered-action",
            probe_style_id=CP0_STYLE_ID,
            collection_wall_seconds=1.0,
            stored_bytes=1,
        )

    changed_seeds = np.asarray(dataset.probe_seeds, dtype=np.int64).copy()
    changed_seeds[0] += 1
    with pytest.raises(ProbeAuditError, match="probe seeds differ"):
        summarize_probe_bank(
            _replace_dataset(dataset, probe_seeds=changed_seeds),
            action_abi=abi,
            seed_bindings=bindings,
            collection_implementation_digest=COLLECTOR_IMPLEMENTATION_DIGEST,
            task_id="task-a",
            context_id="tampered-seed",
            probe_style_id=CP0_STYLE_ID,
            collection_wall_seconds=1.0,
            stored_bytes=1,
        )

    with pytest.raises(ProbeAuditError, match="binding style differs"):
        summarize_probe_bank(
            dataset,
            action_abi=abi,
            seed_bindings=bindings,
            collection_implementation_digest=COLLECTOR_IMPLEMENTATION_DIGEST,
            task_id="task-a",
            context_id="tampered-style",
            probe_style_id=CP2_STYLE_ID,
            collection_wall_seconds=1.0,
            stored_bytes=1,
        )


def test_distance_evidence_binds_banks_checkpoint_matrix_and_recompute() -> None:
    evidence = _evidence("task-a", sha256_json({"bank": "target"}))
    assert len(evidence.semantic_bank_digests) == 3
    assert evidence.encoder_checkpoint_digest
    assert evidence.distance_matrix_digest != evidence.independent_recompute_digest
    assert evidence.representation_id == R0_PADDED_RAW
    assert evidence.axis_id == "mass"
    assert evidence.digest == sha256_json(evidence.to_dict())
    with pytest.raises(ProbeAuditError, match="independent recompute"):
        replace(
            evidence,
            independent_recompute_digest=evidence.distance_matrix_digest,
        )
    with pytest.raises(ProbeAuditError, match="R0 padded-Raw"):
        replace(evidence, representation_id="R5_VIEW_SPECIFIC_CORRO_REFIT")


def test_good_evidence_can_only_receive_development_pass() -> None:
    report = _evaluate()
    assert report.gate_status == "DEVELOPMENT_PASS"
    assert not report.failure_reasons
    assert report.evidence_scope == "DEVELOPMENT"
    assert not report.formal_authority_available
    assert not report.formal_pass_eligible
    assert report.cp2_holdout_pass
    assert report.shared_target_banks_pass


def test_caller_cannot_self_attest_formal_scope_or_direct_fake_pass() -> None:
    _, summaries, evidence, bindings, thresholds, freeze = _inputs()
    with pytest.raises(TypeError):
        evaluate_probe_gate(
            summaries=summaries,
            distance_evidence=evidence,
            training_manifest=_manifest(),
            thresholds=thresholds,
            freeze_decision=freeze,
            target_bank_bindings_by_encoder=bindings,
            evidence_scope="FORMAL_REAL",  # type: ignore[call-arg]
            formal_freeze_attested=True,  # type: ignore[call-arg]
        )
    report = _evaluate()
    with pytest.raises(ProbeAuditError, match="formal PASS is unavailable"):
        replace(report, gate_status="PASS")
    with pytest.raises(ProbeAuditError, match="inconsistent with failed"):
        replace(report, candidate_independence_pass=False)


def test_thresholds_are_typed_frozen_and_bound_to_decision() -> None:
    with pytest.raises(ProbeAuditError, match="strictly positive"):
        _thresholds(min_raw_transition_signal=0.0)
    with pytest.raises(ProbeAuditError, match=r"\[0, 1\]"):
        _thresholds(maximum_probe_style_classifier_accuracy=1.1)
    _, summaries, evidence, bindings, _, freeze = _inputs()
    changed = _thresholds(maximum_invariance_ratio=0.01)
    with pytest.raises(ProbeAuditError, match="another threshold"):
        evaluate_probe_gate(
            summaries=summaries,
            distance_evidence=evidence,
            training_manifest=_manifest(),
            thresholds=changed,
            freeze_decision=freeze,
            target_bank_bindings_by_encoder=bindings,
        )
    with pytest.raises(ProbeAuditError, match="formal G03-Probe"):
        replace(freeze, decision_status="FORMAL_FROZEN")
    with pytest.raises(ProbeAuditError, match="cannot be self-attested"):
        replace(
            freeze,
            formal_authority_receipt_digest=sha256_json(
                {"fake": "formal-authority"}
            ),
        )


def test_low_signal_collapse_invariance_style_or_bank_mismatch_is_no_go() -> None:
    tasks, summaries, evidence, bindings, thresholds, freeze = _inputs()
    weak = (
        _evidence(
            "task-a",
            bindings["encoder-a"]["task-a"],
            signal=0.02,
            noise=0.05,
            cross_probe=0.03,
            classifier_accuracy=0.90,
        ),
        evidence[1],
        evidence[2],
        evidence[3],
    )
    bindings["encoder-b"] = {
        "task-a": sha256_json({"bank": "different"}),
        "task-b": sha256_json({"bank": "b"}),
    }
    report = evaluate_probe_gate(
        summaries=summaries,
        distance_evidence=weak,
        training_manifest=_manifest(),
        thresholds=thresholds,
        freeze_decision=freeze,
        target_bank_bindings_by_encoder=bindings,
    )
    assert tasks == freeze.required_task_ids
    assert report.gate_status == "NO_GO_PROBE_COVERAGE"
    assert "SEMANTIC_BANK_COLLAPSE:task-a:mass" in report.failure_reasons
    assert "RAW_DYNAMICS_BELOW_BANK_NOISE:task-a:mass" in report.failure_reasons
    assert "PROBE_INVARIANCE_FAILURE:task-a:mass" in report.failure_reasons
    assert "PROBE_STYLE_CLASSIFIER_TOO_ACCURATE:task-a:mass" in report.failure_reasons
    assert "TARGET_PROBE_BANK_MISMATCH" in report.failure_reasons
    with pytest.raises(ProbeAuditError, match="inconsistent with failure reason"):
        replace(report, shared_target_banks_pass=True)
