from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.baselines import (
    FORMAL_MODE,
    REQUIRED_BASELINE_METHOD_IDS,
)
from policy_learnware_v0.v03.preflight import (
    HARD_TODO_IDS,
    ExecutionCheckpoint,
    HardTodoEvidence,
    IndependentRecomputeAttestation,
    OracleUnlockHandoff,
    PreExperimentFreezeManifest,
    PreflightError,
    PublicRankingBarrier,
    PublicRankingPublication,
    PublicQueryPlan,
    FORMAL_PRODUCTION_STAGE_IDS,
    formal_baseline_input_plan_digest,
    formal_stage_adapter_binding_digest,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:synthetic"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _formal_adapter_bindings() -> dict[str, str]:
    return {
        stage_id: formal_stage_adapter_binding_digest(
            stage_id, f"test-adapter-{index}", _d(f"adapter-contract-{index}")
        )
        for index, stage_id in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
    }


def _freeze(
    *,
    authority: bool = False,
    baseline_plan_digest: str | None = None,
    public_query_plan_digest: str | None = None,
) -> PreExperimentFreezeManifest:
    return PreExperimentFreezeManifest(
        freeze_id="v03-pre-large-test",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=_d("authority") if authority else None,
        review_authority_verified=authority,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=_d("matrix"),
        signal_contrast_plan_digest=_d("signal-contrast-plan"),
        signal_materiality_threshold_digest=_d("signal-materiality-thresholds"),
        formal_signal_readout_plan_digest=_d("formal-signal-readout-plan"),
        preoracle_signal_outcome_plan_digest=_d("preoracle-signal-outcome-plan"),
        signal_identity_registry_digest=_d("signal-identities"),
        signal_execution_protocol_digest=_d("signal-execution"),
        representation_plan_digest=_d("representations"),
        condition_plan_digest=_d("conditions"),
        formal_source_fit_schedule_digest=_d("source-fit-schedule"),
        formal_source_membership_digest=_d("source-membership"),
        signal_work_item_graph_digest=_d("signal-work-items"),
        formal_signal_prefix_schedule_digest=_d("signal-prefix-schedule"),
        dynamics_axis_registry_digest=_d("dynamics-axis-registry"),
        public_query_plan_digest=(
            _d("public-query-plan")
            if public_query_plan_digest is None
            else public_query_plan_digest
        ),
        baseline_plan_digest=(
            _d("baselines")
            if baseline_plan_digest is None
            else baseline_plan_digest
        ),
        statistics_plan_digest=_d("statistics"),
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=_d("asymmetric-kme"),
        formal_gate_plan_digests=(
            {
                "G03-Attribution": _d("formal-attribution-plan"),
                "G03-Probe": _d("formal-probe-plan"),
                "G03-Market": _d("formal-market-plan"),
            }
            if authority
            else {}
        ),
        formal_stage_request_template_digests=(
            {
                stage_id: _d(f"formal-stage-request:{stage_id}")
                for stage_id in FORMAL_PRODUCTION_STAGE_IDS
            }
            if authority
            else {}
        ),
        hard_todo_evidence=tuple(_todo(item) for item in HARD_TODO_IDS),
        formal_stage_adapter_binding_digests=(
            _formal_adapter_bindings() if authority else {}
        ),
    )


def _publication(method_id: str, query_id: str) -> PublicRankingPublication:
    return PublicRankingPublication(
        method_id=method_id,
        opaque_query_id=query_id,
        ranking_digest=_d(f"{method_id}:{query_id}:ranking"),
        query_spec_digest=_d(f"{query_id}:query-spec"),
        probe_dataset_digest=_d(f"{query_id}:probe"),
        target_evidence_digest=_d(f"{query_id}:target-evidence"),
        cost_digest=_d(f"{method_id}:{query_id}:cost"),
        policy_market_id=_d("policy-market"),
        representation_index_digest=_d(f"{method_id}:representation-index"),
        selector_view_digest=_d(f"{method_id}:selector-view"),
        evidence_contract_digest=_d(f"{method_id}:evidence-contract"),
        selector_artifact_digest=_d(f"{method_id}:selector-artifact"),
        development_freeze_digest=_d("development-freeze"),
        query_input_digest=_d(f"{method_id}:{query_id}:query-input"),
        query_mode="QUERY_EMPIRICAL",
        execution_mode=FORMAL_MODE,
        development_context_count=24,
    )


def _formal_query_plan() -> PublicQueryPlan:
    regimes = {}
    for index in range(66):
        regime = (
            "EXACT"
            if index < 30
            else "INTERPOLATION"
            if index < 54
            else "EXTRAPOLATION"
        )
        regimes[f"v03q-{index:032x}"] = regime
    return PublicQueryPlan(
        regime_by_opaque_query_id=regimes,
        query_alias_manifest_digest=_d("aliases"),
    )


def test_pre_experiment_freeze_binds_all_five_todos_without_self_signing() -> None:
    manifest = _freeze()
    assert manifest.engineering_ready
    assert not manifest.formal_run_authorized
    assert manifest.to_dict()["representation_plan_digest"] == _d("representations")
    assert manifest.to_dict()["condition_plan_digest"] == _d("conditions")
    assert manifest.to_dict()["statistics_plan_digest"] == _d("statistics")
    tampered = manifest.to_dict()
    tampered["condition_plan_digest"] = _d("tampered-conditions")
    with pytest.raises(PreflightError, match="freeze manifest digest"):
        PreExperimentFreezeManifest.from_dict(tampered)
    assert PreExperimentFreezeManifest.from_dict(manifest.to_dict()) == manifest

    with pytest.raises(PreflightError, match="exactly one"):
        replace(manifest, hard_todo_evidence=manifest.hard_todo_evidence[:-1])
    with pytest.raises(PreflightError, match="extension gate disabled"):
        replace(manifest, encoder_extension_gate_enabled=True)
    with pytest.raises(PreflightError, match="external receipt"):
        replace(manifest, review_authority_verified=True)
    with pytest.raises(PreflightError, match="development freeze cannot carry"):
        replace(
            manifest,
            formal_stage_adapter_binding_digests=_formal_adapter_bindings(),
        )

    authorized = _freeze(authority=True)
    assert authorized.formal_run_authorized
    assert (
        authorized.to_dict()["formal_stage_adapter_registry_digest"]
        == authorized.formal_stage_adapter_registry_digest
    )
    with pytest.raises(PreflightError, match="every production stage"):
        replace(
            authorized,
            formal_stage_adapter_binding_digests=dict(
                tuple(authorized.formal_stage_adapter_binding_digests.items())[:-1]
            ),
        )


def test_execution_checkpoint_resumes_without_overwriting_complete_outputs() -> None:
    checkpoint = ExecutionCheckpoint(
        execution_plan_digest=_d("plan"),
        work_item_states={
            "canonicalize-a": "COMPLETE",
            "build-view-a": "RUNNING",
            "rank-a": "FAILED",
        },
        completed_artifact_digests={"canonicalize-a": _d("canonical-a")},
        attempt=0,
    )
    resumed = checkpoint.resume()
    assert resumed.work_item_states == {
        "build-view-a": "PENDING",
        "canonicalize-a": "COMPLETE",
        "rank-a": "PENDING",
    }
    assert resumed.completed_artifact_digests == checkpoint.completed_artifact_digests
    assert resumed.attempt == 1

    with pytest.raises(PreflightError, match="coverage"):
        ExecutionCheckpoint(
            execution_plan_digest=_d("plan"),
            work_item_states={"work-a": "COMPLETE"},
            completed_artifact_digests={},
            attempt=0,
        )


def test_public_ranking_barrier_precedes_external_oracle_handoff() -> None:
    query_plan = _formal_query_plan()
    queries = query_plan.opaque_query_ids
    alias_digest = query_plan.query_alias_manifest_digest
    publications = tuple(
        _publication(method_id, query_id)
        for method_id in REQUIRED_BASELINE_METHOD_IDS
        for query_id in queries
    )
    baseline_plan_digest = formal_baseline_input_plan_digest(
        publications,
        expected_opaque_query_ids=queries,
        query_alias_manifest_digest=alias_digest,
    )
    barrier = PublicRankingBarrier(
        run_id="v03-preflight-run",
        freeze_manifest=_freeze(
            authority=True,
            baseline_plan_digest=baseline_plan_digest,
            public_query_plan_digest=str(query_plan.plan_digest),
        ),
        query_plan=query_plan,
        expected_opaque_query_ids=queries,
        expected_method_ids=REQUIRED_BASELINE_METHOD_IDS,
        publications=publications,
        query_alias_manifest_digest=alias_digest,
        preoracle_signal_outcome_manifest_digest=_d("preoracle-signal-manifest"),
    )
    assert barrier.publication_count == len(REQUIRED_BASELINE_METHOD_IDS) * len(queries)
    assert PublicRankingBarrier.from_dict(barrier.to_dict()) == barrier
    handoff = OracleUnlockHandoff(
        run_id=barrier.run_id,
        freeze_manifest_digest=barrier.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
    )
    assert handoff.requested_owner == "policy-learnware-paper1"
    assert handoff.v03_oracle_write_capability is False

    with pytest.raises(PreflightError, match="full method×query matrix"):
        replace(barrier, publications=barrier.publications[:-1])
    with pytest.raises(PreflightError, match="full method×query matrix"):
        replace(barrier, publications=barrier.publications + (barrier.publications[0],))
    conflicting_probe = replace(
        barrier.publications[0],
        probe_dataset_digest=_d("conflicting-probe"),
        publication_digest=None,
    )
    with pytest.raises(PreflightError, match="multiple probe datasets"):
        replace(
            barrier,
            publications=(conflicting_probe,) + barrier.publications[1:],
        )
    with pytest.raises(PreflightError, match="before any oracle read"):
        replace(barrier, oracle_read_count=1)
    changed_input = replace(
        barrier.publications[0],
        query_input_digest=_d("post-freeze-query-input"),
        publication_digest=None,
    )
    with pytest.raises(PreflightError, match="reviewed baseline plan"):
        replace(
            barrier,
            publications=(changed_input,) + barrier.publications[1:],
        )
    with pytest.raises(PreflightError, match="cannot acquire"):
        replace(handoff, v03_oracle_write_capability=True)


def test_publication_is_formal_24_context_and_content_digest_bound() -> None:
    query_id = "v03q-" + "a" * 32
    publication = _publication(REQUIRED_BASELINE_METHOD_IDS[0], query_id)
    assert PublicRankingPublication.from_dict(publication.to_dict()) == publication

    with pytest.raises(PreflightError, match="formal rankings only"):
        replace(
            publication,
            execution_mode="DEVELOPMENT_SMOKE",
            publication_digest=None,
        )
    with pytest.raises(PreflightError, match="exactly 24"):
        replace(publication, development_context_count=23, publication_digest=None)
    with pytest.raises(PreflightError, match="QUERY_EMPIRICAL"):
        replace(publication, query_mode="QUERY_REDUCED", publication_digest=None)
    with pytest.raises(PreflightError, match="does not match contents"):
        replace(publication, cost_digest=_d("changed-cost"))


def test_independent_recompute_requires_distinct_root_nonce_and_equal_result() -> None:
    record = IndependentRecomputeAttestation(
        run_id="v03-recompute-run",
        freeze_manifest_digest=_d("freeze"),
        public_ranking_barrier_digest=_d("barrier"),
        formal_statistics_result_digest=_d("result"),
        raw_input_manifest_digest=_d("raw"),
        primary_artifact_root_digest=_d("root-primary"),
        recompute_artifact_root_digest=_d("root-recompute"),
        primary_result_digest=_d("result"),
        recompute_result_digest=_d("result"),
        primary_process_nonce_digest=_d("process-primary"),
        recompute_process_nonce_digest=_d("process-recompute"),
    )
    assert record.attestation_digest == sha256_json(record.to_dict())
    with pytest.raises(PreflightError, match="distinct artifact root"):
        replace(
            record,
            recompute_artifact_root_digest=record.primary_artifact_root_digest,
        )
    with pytest.raises(PreflightError, match="does not match"):
        replace(record, recompute_result_digest=_d("different"))
