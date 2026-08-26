from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.v03.attribution import (
    ATTRIBUTION_REPLAY_PROTOCOL_ID,
    ArchivedLegacyReference,
    AttributionGateEvidence,
    AttributionMeasurement,
    AttributionPrefixSchedule,
    AttributionReport,
    AttributionSuite,
    REQUIRED_ATTRIBUTION_VIEW_IDS,
)
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.formal_gates import (
    FORMAL_ATTRIBUTION_INPUT_VIEW_IDS,
    FormalAttributionAdmission,
    FormalAttributionPlan,
    FormalAttributionRecomputeEvidence,
    FormalGateAuthorityReceipt,
    FormalGateError,
    FormalMarketAdmission,
    FormalMarketEvidence,
    FormalMarketPlan,
    FormalProbeAdmission,
    FormalProbePlan,
    admit_formal_attribution,
    admit_formal_market,
    admit_formal_probe,
    build_formal_attribution_recompute_evidence,
)
from policy_learnware_v0.v03.preflight import (
    FORMAL_PRODUCTION_STAGE_IDS,
    HARD_TODO_IDS,
    HardTodoEvidence,
    PreExperimentFreezeManifest,
    PreflightError,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.probe_audit import (
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
from policy_learnware_v0.v03.representation_ladder import (
    R0_PADDED_RAW,
    R_HIST_RANDOM_TANH,
)
from policy_learnware_v0.v03.transition_views import (
    TRANSITION_VIEW_PROTOCOL_ID,
    VIEW_REGISTRY,
    V_RANDOM_ENCODER,
)


def _d(label: str) -> str:
    return sha256_json({"formal-gate-test": label})


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:fixture"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _freeze(gate_id: str, plan_digest: str) -> PreExperimentFreezeManifest:
    gate_plans = {
        "G03-Attribution": _d("other-attribution-plan"),
        "G03-Probe": _d("other-probe-plan"),
        "G03-Market": _d("other-market-plan"),
    }
    gate_plans[gate_id] = plan_digest
    return PreExperimentFreezeManifest(
        freeze_id="formal-gate-test-freeze",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=_d("freeze-authority"),
        review_authority_verified=True,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=_d("matrix"),
        signal_contrast_plan_digest=_d("contrast-plan"),
        signal_materiality_threshold_digest=_d("materiality"),
        formal_signal_readout_plan_digest=_d("readout"),
        preoracle_signal_outcome_plan_digest=_d("preoracle-signal-outcome-plan"),
        signal_identity_registry_digest=_d("identities"),
        signal_execution_protocol_digest=_d("signal-execution"),
        representation_plan_digest=_d("representations"),
        condition_plan_digest=_d("conditions"),
        formal_source_fit_schedule_digest=_d("source-fit"),
        formal_source_membership_digest=_d("source-membership"),
        signal_work_item_graph_digest=_d("work-graph"),
        formal_signal_prefix_schedule_digest=_d("signal-prefix"),
        dynamics_axis_registry_digest=_d("dynamics-axis"),
        public_query_plan_digest=_d("queries"),
        baseline_plan_digest=_d("baselines"),
        statistics_plan_digest=_d("statistics"),
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=_d("asymmetric-kme"),
        formal_gate_plan_digests=gate_plans,
        formal_stage_request_template_digests={
            stage_id: _d(f"formal-stage-request:{stage_id}")
            for stage_id in FORMAL_PRODUCTION_STAGE_IDS
        },
        hard_todo_evidence=tuple(_todo(item) for item in HARD_TODO_IDS),
        formal_stage_adapter_binding_digests={
            stage_id: formal_stage_adapter_binding_digest(
                stage_id,
                f"formal-gate-adapter-{index}",
                _d(f"adapter-contract-{index}"),
            )
            for index, stage_id in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
        },
    )


def _authority(
    gate_id: str,
    plan_digest: str,
    evidence_digest: str,
    freeze: PreExperimentFreezeManifest,
) -> FormalGateAuthorityReceipt:
    return FormalGateAuthorityReceipt(
        gate_id=gate_id,
        plan_digest=plan_digest,
        evidence_digest=evidence_digest,
        freeze_manifest_digest=freeze.freeze_manifest_digest,
        authority_id="paper1-external-review",
        external_review_record_digest=_d(f"review:{gate_id}:{plan_digest}"),
    )


def _archive_case():
    schedule = AttributionPrefixSchedule.formal()
    reference = ArchivedLegacyReference(
        archive_protocol_id="legacy-archive-v0",
        archive_manifest_digest=_d("archive-manifest"),
        archived_dataset_digest=_d("archived-dataset"),
        canonical_bank_digest=_d("canonical-bank"),
        encoder_checkpoint_digest=_d("legacy-checkpoint"),
        encoder_implementation_digest=_d("legacy-implementation"),
        reference_metrics={"retrieval.top1": 0.75},
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-6,
    )
    reports = []
    for index, view_id in enumerate(REQUIRED_ATTRIBUTION_VIEW_IDS):
        measurement = AttributionMeasurement(
            view_id=view_id,
            task_group="cross-embodiment",
            shared_schema_group="shared-schema-a",
            retrieval_metrics={"top1": 0.75 - index * 0.001},
            between_within_mmd_summaries={"ratio": 2.0},
            prefix_curves={
                "top1": {prefix: 0.5 + prefix * 0.001 for prefix in schedule.prefix_episode_counts}
            },
            failure_identifiability_notes=("bounded intervention interpretation",),
        )
        reports.append(
            AttributionReport(
                view_protocol_id=TRANSITION_VIEW_PROTOCOL_ID,
                view_id=view_id,
                input_channel_allowlist=VIEW_REGISTRY[view_id].input_channel_allowlist,
                encoder_checkpoint_digest=reference.encoder_checkpoint_digest,
                encoder_implementation_digest=reference.encoder_implementation_digest,
                archived_dataset_digest=reference.archived_dataset_digest,
                canonical_bank_digest=reference.canonical_bank_digest,
                transition_view_digest=_d(f"view:{view_id}"),
                prefix_schedule_digest=schedule.schedule_digest,
                prefix_schedule_scope="FORMAL",
                task_group=measurement.task_group,
                shared_schema_group=measurement.shared_schema_group,
                retrieval_metrics=measurement.retrieval_metrics,
                between_within_mmd_summaries=measurement.between_within_mmd_summaries,
                prefix_curves=measurement.prefix_curves,
                paired_deltas_vs_full_legacy={"retrieval.top1": index * -0.001},
                shuffled_control_deltas={"V_SHUFFLED_NEXT:retrieval.top1": -0.1},
                failure_identifiability_notes=measurement.failure_identifiability_notes,
            )
        )
    suite = AttributionSuite(
        reports=tuple(reports),
        archived_reference_digest=reference.digest,
        prefix_schedule=schedule,
        # The engineering runner still says SYNTHETIC/DEVELOPMENT_PASS.  This
        # deliberately demonstrates that it cannot authorize the formal gate.
        gate_evidence=AttributionGateEvidence(
            gate_status="DEVELOPMENT_PASS",
            evidence_scope="SYNTHETIC",
            full_legacy_replay_pass=True,
            controls_fail_closed_pass=True,
            contribution_quantified_pass=True,
            shared_schema_explanation_pass=True,
            independently_recomputable_pass=False,
            dynamics_interpretation="LEGACY_ENCODER_DYNAMICS_SENSITIVE",
            maximum_legacy_replay_error=0.0,
            failure_reasons=(),
        ),
    )
    plan = FormalAttributionPlan(
        required_input_view_ids=FORMAL_ATTRIBUTION_INPUT_VIEW_IDS,
        historical_random_view_id=V_RANDOM_ENCODER,
        historical_random_representation_id=R_HIST_RANDOM_TANH,
        prefix_episode_counts=schedule.prefix_episode_counts,
        prefix_schedule_digest=schedule.schedule_digest,
        attribution_replay_protocol_id=ATTRIBUTION_REPLAY_PROTOCOL_ID,
        archive_protocol_id=reference.archive_protocol_id,
        archive_manifest_digest=reference.archive_manifest_digest,
        archived_reference_digest=reference.digest,
        archived_dataset_digest=reference.archived_dataset_digest,
        canonical_bank_digest=reference.canonical_bank_digest,
        encoder_checkpoint_digest=reference.encoder_checkpoint_digest,
        encoder_implementation_digest=reference.encoder_implementation_digest,
        legacy_normalizer_digest=_d("legacy-normalizer"),
        independent_recompute_protocol_digest=_d("attribution-independent-recompute"),
    )
    report_index = {report.view_id: report.digest for report in reports}
    evidence = build_formal_attribution_recompute_evidence(
        suite,
        independent_report_digests_by_view=report_index,
        independent_recompute_protocol_digest=plan.independent_recompute_protocol_digest,
        independent_recompute_receipt_digest=_d("attribution-recompute-receipt"),
        independent_execution_digest=_d("attribution-recompute-process"),
        test_time_ood_claim_bounded_pass=True,
    )
    return reference, suite, plan, evidence


def test_formal_attribution_requires_archive_recompute_and_external_authority() -> None:
    reference, suite, plan, evidence = _archive_case()
    freeze = _freeze("G03-Attribution", str(plan.plan_digest))
    authority = _authority(
        "G03-Attribution", str(plan.plan_digest), str(evidence.evidence_digest), freeze
    )
    result = admit_formal_attribution(
        plan=plan,
        archived_reference=reference,
        suite=suite,
        evidence=evidence,
        authority=authority,
        freeze=freeze,
    )
    assert result.status == "PASS"
    assert FormalAttributionAdmission.from_dict(result.to_dict()) == result
    assert FormalAttributionPlan.from_dict(plan.to_dict()) == plan
    assert FormalAttributionRecomputeEvidence.from_dict(evidence.to_dict()) == evidence
    public = result.public_projection()
    assert "archive" not in str(public).lower()
    assert "checkpoint" not in str(public).lower()

    with pytest.raises(FormalGateError, match="authority receipt binding"):
        admit_formal_attribution(
            plan=plan,
            archived_reference=reference,
            suite=suite,
            evidence=evidence,
                authority=replace(
                    authority,
                    evidence_digest=_d("forged-evidence"),
                    receipt_digest=None,
                ),
            freeze=freeze,
        )


def test_formal_attribution_rejects_synthetic_scope_short_prefix_and_fake_recompute() -> None:
    _, _, plan, evidence = _archive_case()
    with pytest.raises(FormalGateError, match="rejects synthetic"):
        replace(evidence, evidence_scope="SYNTHETIC")
    with pytest.raises(FormalGateError, match="prefix schedule"):
        replace(plan, prefix_episode_counts=(1, 2, 4))
    independent = dict(evidence.independent_report_digests_by_view)
    independent[V_RANDOM_ENCODER] = _d("forged-independent-report")
    with pytest.raises(FormalGateError, match="reproduce all"):
        replace(evidence, independent_report_digests_by_view=independent)
    payload = evidence.to_dict()
    payload["independent_recompute_receipt_digest"] = _d("tampered-receipt")
    with pytest.raises(FormalGateError, match="evidence digest"):
        FormalAttributionRecomputeEvidence.from_dict(payload)


def _probe_dataset(style_id: str):
    abi = ActionABI(
        low=np.asarray([-2.0, 1.0], dtype=np.float32),
        high=np.asarray([2.0, 5.0], dtype=np.float32),
    )
    bindings = tuple(
        ProbeSeedBinding(
            role="target_query",
            style_id=style_id,
            namespace="formal-probe-test",
            nonce=f"nonce-{style_id}",
            episode_id=episode_id,
        )
        for episode_id in range(2)
    )
    observations = np.asarray(
        [[0.1 * index, np.sin(index)] for index in range(8)], dtype=np.float32
    )
    probe = registered_probe(style_id)
    actions = []
    for episode_id, binding in enumerate(bindings):
        state = probe.reset(int(binding.seed), abi)
        for step in range(4):
            normalized, state = probe.act(
                observations[episode_id * 4 + step], state, step=step
            )
            actions.append(abi.map_normalized(normalized))
    dataset = EpisodeDataset(
        observation=observations,
        action=np.stack(actions),
        reward=np.linspace(0.0, 1.0, 8, dtype=np.float32),
        next_observation=observations + np.asarray([0.08, -0.04], dtype=np.float32),
        terminated=np.zeros(8, dtype=np.bool_),
        truncated=np.asarray([False, False, False, True] * 2),
        episode_offsets=np.asarray([0, 4, 8]),
        reset_seeds=np.asarray([1, 2]),
        probe_seeds=np.asarray([binding.seed for binding in bindings]),
    )
    return dataset, abi, bindings


def _probe_case():
    tasks = ("task-a", "task-b")
    axes = ("mass", "damping")
    summaries = []
    for task in tasks:
        for style in (CP0_STYLE_ID, CP2_STYLE_ID):
            dataset, abi, bindings = _probe_dataset(style)
            summaries.append(
                summarize_probe_bank(
                    dataset,
                    action_abi=abi,
                    seed_bindings=bindings,
                    collection_implementation_digest=_d("probe-collector"),
                    task_id=task,
                    context_id=f"{task}-{style}",
                    probe_style_id=style,
                    collection_wall_seconds=1.0,
                    stored_bytes=512,
                )
            )
    raw_protocol = _d("r0-probe-protocol")
    shared_target = {task: _d(f"target:{task}") for task in tasks}
    distances = []
    for task in tasks:
        for axis in axes:
            distances.append(
                ProbeDistanceEvidence(
                    task_id=task,
                    axis_id=axis,
                    representation_id=R0_PADDED_RAW,
                    representation_protocol_digest=raw_protocol,
                    semantic_bank_digests={
                        "target_query": shared_target[task],
                        "source_nominal": _d(f"nominal:{task}:{axis}"),
                        "source_shifted": _d(f"shifted:{task}:{axis}"),
                    },
                    encoder_checkpoint_digest=_d(f"raw:{task}:{axis}"),
                    distance_matrix_digest=_d(f"distance:{task}:{axis}"),
                    independent_recompute_digest=_d(f"recompute:{task}:{axis}"),
                    same_environment_cross_probe_distances=(0.05, 0.06),
                    different_dynamics_same_probe_distances=(0.4, 0.45),
                    repeated_bank_noise_distances=(0.04, 0.05),
                    probe_style_classifier_accuracy=0.5,
                )
            )
    manifest = ProbeTrainingManifest(
        training_style_ids=(CP0_STYLE_ID, CP1_OU_STYLE_ID),
        confirmatory_style_id=CP2_STYLE_ID,
        fold_ids=("fold-a", "fold-b"),
        freeze_authority="development-probe-freeze",
    )
    thresholds = ProbeGateThresholds(
        min_action_energy=1.0e-4,
        min_state_coverage=1.0e-5,
        min_raw_transition_signal=1.0e-3,
        min_different_dynamics_distance=0.1,
        minimum_signal_to_noise_ratio=2.0,
        maximum_invariance_ratio=0.3,
        maximum_probe_style_classifier_accuracy=0.6,
        max_saturation_rate=0.95,
        max_termination_rate=0.5,
        max_failure_rate=0.0,
    )
    decision = ProbeGateFreezeDecision(
        required_task_ids=tasks,
        required_task_axis_pairs=tuple((task, axis) for task in tasks for axis in axes),
        training_manifest_digest=manifest.digest,
        thresholds_digest=thresholds.digest,
        decision_authority="development-probe-review",
    )
    report = evaluate_probe_gate(
        summaries=summaries,
        distance_evidence=distances,
        training_manifest=manifest,
        thresholds=thresholds,
        freeze_decision=decision,
        target_bank_bindings_by_encoder={
            "encoder-a": shared_target,
            "encoder-b": shared_target,
        },
    )
    plan = FormalProbePlan(
        required_task_ids=tasks,
        required_task_axis_pairs=decision.required_task_axis_pairs,
        cp0_style_id=CP0_STYLE_ID,
        cp2_style_id=CP2_STYLE_ID,
        raw_representation_id=R0_PADDED_RAW,
        raw_representation_protocol_digest=raw_protocol,
        thresholds_digest=thresholds.digest,
        training_manifest_digest=manifest.digest,
        target_bank_digests_by_task={
            task: {
                style: next(
                    row.dataset_digest
                    for row in summaries
                    if row.task_id == task and row.probe_style_id == style
                )
                for style in (CP0_STYLE_ID, CP2_STYLE_ID)
            }
            for task in tasks
        },
        distance_semantic_bank_digests_by_pair={
            f"{row.task_id}::{row.axis_id}": dict(row.semantic_bank_digests)
            for row in distances
        },
    )
    return manifest, report, plan


def test_formal_probe_binds_raw_task_axis_banks_thresholds_and_external_freeze() -> None:
    manifest, report, plan = _probe_case()
    assert report.gate_status == "DEVELOPMENT_PASS"
    freeze = _freeze("G03-Probe", str(plan.plan_digest))
    authority = _authority(
        "G03-Probe", str(plan.plan_digest), report.digest, freeze
    )
    result = admit_formal_probe(
        plan=plan,
        report=report,
        training_manifest=manifest,
        authority=authority,
        freeze=freeze,
    )
    assert result.status == "PASS"
    assert FormalProbePlan.from_dict(plan.to_dict()) == plan
    assert FormalProbeAdmission.from_dict(result.to_dict()) == result
    assert set(result.public_projection()) == {
        "schema",
        "status",
        "task_count",
        "axis_count",
        "plan_digest",
        "evidence_digest",
        "admission_digest",
    }


def test_formal_probe_rejects_threshold_or_bank_digest_drift() -> None:
    manifest, report, plan = _probe_case()
    changed = replace(plan, thresholds_digest=_d("changed-threshold-values"), plan_digest=None)
    freeze = _freeze("G03-Probe", str(changed.plan_digest))
    authority = _authority("G03-Probe", str(changed.plan_digest), report.digest, freeze)
    with pytest.raises(FormalGateError, match="threshold/training/freeze"):
        admit_formal_probe(
            plan=changed,
            report=report,
            training_manifest=manifest,
            authority=authority,
            freeze=freeze,
        )
    targets = {
        task: dict(rows) for task, rows in plan.target_bank_digests_by_task.items()
    }
    targets["task-a"][CP2_STYLE_ID] = _d("forged-cp2-bank")
    changed_bank = replace(plan, target_bank_digests_by_task=targets, plan_digest=None)
    freeze = _freeze("G03-Probe", str(changed_bank.plan_digest))
    authority = _authority("G03-Probe", str(changed_bank.plan_digest), report.digest, freeze)
    with pytest.raises(FormalGateError, match="target-bank digest"):
        admit_formal_probe(
            plan=changed_bank,
            report=report,
            training_manifest=manifest,
            authority=authority,
            freeze=freeze,
        )


def _market_case():
    anchors = [_d(f"anchor:{index}") for index in range(30)]
    candidates = [f"candidate-{index:03d}" for index in range(90)]
    candidate_anchor = {
        candidate: anchors[index // 3] for index, candidate in enumerate(candidates)
    }
    abis = {candidate: _d(f"abi:{candidate}") for candidate in candidates}
    plan = FormalMarketPlan(
        intake_record_digest=_d("exact-90-intake"),
        source_pool_digest=_d("exact-90-source-pool"),
        source_evaluation_protocol_digest=_d("source-evaluation-protocol"),
        intake_cell_digests_by_candidate={
            candidate: _d(f"cell:{candidate}") for candidate in candidates
        },
        source_anchor_id_by_candidate=candidate_anchor,
        deployment_abi_digests_by_candidate=abis,
        market_alias_protocol_digest=_d("market-alias-protocol"),
        market_alias_commitment_digest=_d("market-alias-secret-commitment"),
        tie_break_commitment_digest=_d("tie-break-secret-commitment"),
    )
    champions = {anchor: candidates[index * 3] for index, anchor in enumerate(anchors)}
    opaque_ids = [f"lw-{index:032x}" for index in range(30)]
    deployment_candidates = {
        opaque_id: champions[anchors[index]] for index, opaque_id in enumerate(opaque_ids)
    }
    evidence = FormalMarketEvidence(
        intake_record_digest=plan.intake_record_digest,
        source_pool_digest=plan.source_pool_digest,
        source_evaluation_protocol_digest=plan.source_evaluation_protocol_digest,
        selection_receipt_digests_by_candidate={
            candidate: _d(f"selection:{candidate}") for candidate in candidates
        },
        attestation_receipt_digests_by_candidate={
            candidate: _d(f"attestation:{candidate}") for candidate in champions.values()
        },
        champion_candidate_ids_by_anchor=champions,
        champion_digests_by_anchor={
            anchor: _d(f"champion:{anchor}") for anchor in anchors
        },
        competence_observation_digests_by_anchor={
            anchor: _d(f"observe:{anchor}") for anchor in anchors
        },
        championization_digest=_d("championization"),
        policy_market_id=_d("policy-market"),
        public_entry_digests_by_opaque_id={
            opaque_id: _d(f"public:{opaque_id}") for opaque_id in opaque_ids
        },
        deployment_entry_digests_by_opaque_id={
            opaque_id: _d(f"private:{opaque_id}") for opaque_id in opaque_ids
        },
        deployment_candidate_ids_by_opaque_id=deployment_candidates,
        deployment_abi_digests_by_candidate={
            candidate: abis[candidate] for candidate in champions.values()
        },
        receipts_binding_pass=True,
        market_binding_pass=True,
        observe_mode_pass=True,
        failure_reasons=(),
    )
    return plan, evidence


def test_formal_market_exact_90_to_30_observe_and_abi_admission() -> None:
    plan, evidence = _market_case()
    freeze = _freeze("G03-Market", str(plan.plan_digest))
    authority = _authority(
        "G03-Market", str(plan.plan_digest), str(evidence.evidence_digest), freeze
    )
    result = admit_formal_market(
        plan=plan, evidence=evidence, authority=authority, freeze=freeze
    )
    assert result.status == "ASSET_READY"
    assert FormalMarketPlan.from_dict(plan.to_dict()) == plan
    assert FormalMarketEvidence.from_dict(evidence.to_dict()) == evidence
    assert FormalMarketAdmission.from_dict(result.to_dict()) == result
    projection = result.public_projection()
    assert projection["candidate_count"] == 90
    assert projection["market_entry_count"] == 30
    assert not any("candidate-" in str(value) for value in projection.values())


def test_formal_market_29_entries_and_forged_receipt_fail_closed() -> None:
    plan, evidence = _market_case()
    public = dict(evidence.public_entry_digests_by_opaque_id)
    public.pop(next(iter(public)))
    incomplete = replace(
        evidence,
        public_entry_digests_by_opaque_id=public,
        failure_reasons=("EXACT_30_MARKET_ENTRIES_MISSING",),
        market_binding_pass=False,
        evidence_digest=None,
    )
    freeze = _freeze("G03-Market", str(plan.plan_digest))
    authority = _authority(
        "G03-Market", str(plan.plan_digest), str(incomplete.evidence_digest), freeze
    )
    result = admit_formal_market(
        plan=plan, evidence=incomplete, authority=authority, freeze=freeze
    )
    assert result.status == "NO_GO"
    assert result.market_entry_count == 29

    payload = authority.to_dict()
    payload["evidence_digest"] = evidence.evidence_digest
    with pytest.raises(FormalGateError, match="receipt digest"):
        FormalGateAuthorityReceipt.from_dict(payload)


def test_preflight_requires_exact_formal_gate_plans_but_development_has_none() -> None:
    plan, _ = _market_case()
    formal = _freeze("G03-Market", str(plan.plan_digest))
    with pytest.raises(PreflightError, match="exact G03"):
        replace(
            formal,
            formal_gate_plan_digests={
                "G03-Market": str(plan.plan_digest),
                "G03-Probe": _d("probe"),
            },
        )
    development = replace(
        formal,
        review_authority_receipt_digest=None,
        review_authority_verified=False,
        formal_stage_adapter_binding_digests={},
        formal_gate_plan_digests={},
        formal_stage_request_template_digests={},
    )
    assert not development.formal_run_authorized
    with pytest.raises(PreflightError, match="development freeze cannot carry"):
        replace(
            development,
            formal_gate_plan_digests={
                "G03-Attribution": _d("a"),
                "G03-Probe": _d("p"),
                "G03-Market": _d("m"),
            },
        )
