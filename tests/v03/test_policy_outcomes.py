from __future__ import annotations

from dataclasses import replace

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.baselines import (
    FORMAL_MODE,
    REQUIRED_BASELINE_METHOD_IDS,
    PublishedFullRanking,
    PublishedRankingRow,
)
from policy_learnware_v0.v03.costs import frozen_cost_protocol_digest
from policy_learnware_v0.v03.policy_outcomes import (
    CanonicalPrimaryComparisonPlan,
    ExternalOracleEvidenceManifest,
    ExternalOracleReleaseReceipt,
    OracleEpisodeEvidence,
    OraclePolicyEvidence,
    PolicyOutcomeError,
    PRIMARY_COMPARISON_IDS,
    SignalOutcomeManifest,
    SignalOutcomeRow,
    build_policy_outcome_statistics_bridge,
)
from policy_learnware_v0.v03.preflight import (
    FORMAL_PRODUCTION_STAGE_IDS,
    HARD_TODO_IDS,
    HardTodoEvidence,
    OracleUnlockHandoff,
    PreExperimentFreezeManifest,
    PublicQueryPlan,
    PublicRankingBarrier,
    PublicRankingPublication,
    formal_baseline_input_plan_digest,
    formal_stage_adapter_binding_digest,
)
from policy_learnware_v0.v03.signal_prefix import SignalPrefixSchedule
from policy_learnware_v0.v03.statistics import (
    FORMAL_CONTRAST_FAMILY_IDS,
    V03StatisticsError,
    compute_formal_statistics,
)


def _d(label: str) -> str:
    return sha256_json({"policy-outcome-test": label})


def _todo(todo_id: str) -> HardTodoEvidence:
    return HardTodoEvidence(
        todo_id=todo_id,
        contract_digest=_d(f"{todo_id}:contract"),
        implementation_digest=_d(f"{todo_id}:implementation"),
        unit_test_evidence_digest=_d(f"{todo_id}:unit"),
        synthetic_fixture_evidence_digest=_d(f"{todo_id}:synthetic"),
        cpu_integration_evidence_digest=_d(f"{todo_id}:cpu"),
    )


def _adapter_bindings() -> dict[str, str]:
    return {
        stage: formal_stage_adapter_binding_digest(
            stage, f"policy-outcome-adapter-{index}", _d(f"adapter:{index}")
        )
        for index, stage in enumerate(FORMAL_PRODUCTION_STAGE_IDS)
    }


def _query_plan() -> PublicQueryPlan:
    regimes = {
        f"v03q-{index:032x}": (
            "EXACT" if index < 30 else "INTERPOLATION" if index < 54 else "EXTRAPOLATION"
        )
        for index in range(66)
    }
    return PublicQueryPlan(regimes, _d("query-aliases"))


def _ranking(method_id: str, query_id: str, query_index: int) -> PublishedFullRanking:
    policies = tuple(f"lw-{index:032x}" for index in range(30))
    method_index = REQUIRED_BASELINE_METHOD_IDS.index(method_id)
    selected_index = (query_index + 3 * method_index) % len(policies)
    ordered = (policies[selected_index],) + tuple(
        item for item in policies if item != policies[selected_index]
    )
    rows = tuple(
        PublishedRankingRow(
            opaque_learnware_id=policy_id,
            rank=rank,
            score=float(31 - rank),
            tie_break_token=_d(f"tie:{policy_id}"),
        )
        for rank, policy_id in enumerate(ordered, start=1)
    )
    return PublishedFullRanking(
        method_id=method_id,
        opaque_query_id=query_id,
        query_spec_digest=_d(f"query-spec:{query_id}"),
        probe_dataset_digest=_d(f"probe:{query_id}"),
        target_evidence_digest=_d(f"target:{query_id}"),
        policy_market_id=_d("policy-market"),
        representation_index_digest=_d(f"index:{method_id}"),
        selector_view_digest=_d(f"selector:{method_id}"),
        evidence_contract_digest=_d(f"evidence:{method_id}"),
        cost_digest=_d(f"cost:{method_id}:{query_id}"),
        selector_artifact_digest=_d(f"artifact:{method_id}"),
        development_freeze_digest=_d("development-freeze"),
        query_input_digest=_d(f"query-input:{method_id}:{query_id}"),
        execution_mode=FORMAL_MODE,
        query_mode="QUERY_EMPIRICAL",
        development_context_count=24,
        score_semantics="synthetic-fixed-order",
        selected_opaque_learnware_id=ordered[0],
        rows=rows,
    )


def _freeze(
    *,
    query_plan: PublicQueryPlan,
    baseline_plan_digest: str,
    statistics_plan_digest: str,
) -> PreExperimentFreezeManifest:
    return PreExperimentFreezeManifest(
        freeze_id="v03-policy-outcome-freeze",
        config_bytes_digest=_d("config"),
        implementation_tree_digest=_d("tree"),
        clean_commit_digest=_d("commit"),
        review_decisions_digest=_d("review"),
        review_authority_receipt_digest=_d("external-authority"),
        review_authority_verified=True,
        encoder_extension_gate_enabled=False,
        data_role_manifest_digest=_d("roles"),
        canonicalizer_registry_digest=_d("canonicalizer"),
        signal_matrix_digest=_d("signal-matrix"),
        signal_contrast_plan_digest=_d("signal-contrast-plan"),
        signal_materiality_threshold_digest=_d("signal-materiality-thresholds"),
        formal_signal_readout_plan_digest=_d("formal-signal-readout-plan"),
        preoracle_signal_outcome_plan_digest=_d("preoracle-signal-outcome-plan"),
        signal_identity_registry_digest=_d("signal-identities"),
        signal_execution_protocol_digest=_d("signal-execution"),
        representation_plan_digest=_d("representation-plan"),
        condition_plan_digest=_d("condition-plan"),
        formal_source_fit_schedule_digest=_d("source-fit-schedule"),
        formal_source_membership_digest=_d("source-membership"),
        signal_work_item_graph_digest=_d("signal-work-graph"),
        formal_signal_prefix_schedule_digest=str(SignalPrefixSchedule.formal().schedule_digest),
        dynamics_axis_registry_digest=_d("dynamics-axis"),
        public_query_plan_digest=str(query_plan.plan_digest),
        baseline_plan_digest=baseline_plan_digest,
        statistics_plan_digest=statistics_plan_digest,
        cost_protocol_digest=frozen_cost_protocol_digest(),
        source_reduced_query_empirical_protocol_digest=_d("asymmetric-kme"),
        formal_gate_plan_digests={
            "G03-Attribution": _d("formal-attribution-plan"),
            "G03-Probe": _d("formal-probe-plan"),
            "G03-Market": _d("formal-market-plan"),
        },
        formal_stage_request_template_digests={
            stage_id: _d(f"formal-stage-request:{stage_id}")
            for stage_id in FORMAL_PRODUCTION_STAGE_IDS
        },
        hard_todo_evidence=tuple(_todo(item) for item in HARD_TODO_IDS),
        formal_stage_adapter_binding_digests=_adapter_bindings(),
    )


@pytest.fixture(scope="module")
def formal_case():
    plan = CanonicalPrimaryComparisonPlan(
        strongest_b3_b4_method_id="B4b",
        strongest_method_selection_receipt_digest=_d("strongest-selection"),
        bootstrap_resamples=64,
    )
    query_plan = _query_plan()
    rankings = tuple(
        _ranking(method, query_id, query_index)
        for method in REQUIRED_BASELINE_METHOD_IDS
        for query_index, query_id in enumerate(query_plan.opaque_query_ids)
    )
    publications = tuple(PublicRankingPublication.from_published_ranking(item) for item in rankings)
    baseline_digest = formal_baseline_input_plan_digest(
        publications,
        expected_opaque_query_ids=query_plan.opaque_query_ids,
        query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
    )
    freeze = _freeze(
        query_plan=query_plan,
        baseline_plan_digest=baseline_digest,
        statistics_plan_digest=plan.formal_statistics_plan.plan_digest,
    )
    barrier = PublicRankingBarrier(
        run_id="v03-policy-outcome-run",
        freeze_manifest=freeze,
        query_plan=query_plan,
        expected_opaque_query_ids=query_plan.opaque_query_ids,
        expected_method_ids=REQUIRED_BASELINE_METHOD_IDS,
        publications=publications,
        query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
        preoracle_signal_outcome_manifest_digest=_d("preoracle-signal-manifest"),
    )
    handoff = OracleUnlockHandoff(
        run_id=barrier.run_id,
        freeze_manifest_digest=barrier.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
    )
    signal_rows = []
    for query_index, query_id in enumerate(query_plan.opaque_query_ids):
        # M02/B5 uses method index 8 in the frozen synthetic ranking.  The
        # registered signal is deliberately monotone with its selected
        # policy value so every task×axis cell carries positive direction
        # evidence for the G03 linkage acceptance test.
        signal_value = ((query_index + 24) % 30) / 29.0
        signal_rows.append(
            SignalOutcomeRow(
                opaque_query_id=query_id,
                task_id=f"task-{query_index % 2}",
                axis_id=f"axis-{(query_index // 2) % 2}",
                context_id=f"context-{query_index}",
                signal_metric_id="dynamics-order-score",
                signal_value=signal_value,
                prefix_signal_values={
                    prefix: signal_value * prefix / 64.0
                    for prefix in SignalPrefixSchedule.formal().prefix_episode_counts
                },
                signal_evidence_digest=_d(f"signal:{query_id}"),
            )
        )
    signal = SignalOutcomeManifest(
        run_id=barrier.run_id,
        freeze_manifest_digest=barrier.freeze_manifest_digest,
        public_query_plan_digest=str(query_plan.plan_digest),
        query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
        signal_atlas_digest=_d("signal-atlas"),
        signal_prefix_schedule_digest=freeze.formal_signal_prefix_schedule_digest,
        rows=tuple(signal_rows),
    )
    barrier = replace(
        barrier,
        preoracle_signal_outcome_manifest_digest=str(signal.manifest_digest),
    )
    handoff = OracleUnlockHandoff(
        run_id=barrier.run_id,
        freeze_manifest_digest=barrier.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
    )
    policy_ids = tuple(f"lw-{index:032x}" for index in range(30))
    episode_ids = {
        query_id: ("episode-0", "episode-1") for query_id in query_plan.opaque_query_ids
    }
    oracle_rows = []
    for query_index, query_id in enumerate(query_plan.opaque_query_ids):
        for policy_index, policy_id in enumerate(policy_ids):
            executable = policy_index != 0
            episodes = tuple(
                OracleEpisodeEvidence(
                    episode_id=episode_id,
                    episode_seed_digest=_d(f"seed:{query_id}:{episode_id}"),
                    status="EXECUTED" if executable else "ABI_INCOMPATIBLE",
                    return_value=(
                        policy_index + query_index / 100.0 + episode_index / 10.0
                        if executable
                        else None
                    ),
                    evidence_digest=_d(f"episode-evidence:{query_id}:{policy_id}:{episode_id}"),
                )
                for episode_index, episode_id in enumerate(episode_ids[query_id])
            )
            oracle_rows.append(
                OraclePolicyEvidence(
                    opaque_query_id=query_id,
                    opaque_policy_id=policy_id,
                    target_execution_abi_digest=_d(f"target-abi:{query_id}"),
                    policy_execution_abi_digest=_d(f"policy-abi:{policy_id}"),
                    executable=executable,
                    policy_value=(
                        sum(float(item.return_value) for item in episodes) / len(episodes)
                        if executable
                        else None
                    ),
                    episodes=episodes,
                )
            )
    oracle = ExternalOracleEvidenceManifest(
        scope="FORMAL",
        run_id=barrier.run_id,
        freeze_manifest_digest=barrier.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
        public_query_plan_digest=str(query_plan.plan_digest),
        query_alias_manifest_digest=query_plan.query_alias_manifest_digest,
        signal_outcome_manifest_digest=str(signal.manifest_digest),
        policy_market_id=_d("policy-market"),
        expected_opaque_query_ids=query_plan.opaque_query_ids,
        expected_opaque_policy_ids=policy_ids,
        episode_ids_by_query=episode_ids,
        rows=tuple(oracle_rows),
    )
    release = ExternalOracleReleaseReceipt(
        run_id=barrier.run_id,
        freeze_manifest_digest=barrier.freeze_manifest_digest,
        public_ranking_barrier_digest=barrier.barrier_digest,
        oracle_unlock_handoff_digest=handoff.handoff_digest,
        oracle_evidence_manifest_digest=str(oracle.evidence_manifest_digest),
        external_authority_receipt_digest=_d("oracle-owner-attestation"),
    )
    return {
        "plan": plan,
        "query_plan": query_plan,
        "rankings": rankings,
        "barrier": barrier,
        "handoff": handoff,
        "signal": signal,
        "oracle": oracle,
        "release": release,
    }


def _build(case, **overrides):
    arguments = {
        "barrier": case["barrier"],
        "rankings": case["rankings"],
        "oracle_handoff": case["handoff"],
        "external_release_receipt": case["release"],
        "oracle_evidence": case["oracle"],
        "signal_outcomes": case["signal"],
        "comparison_plan": case["plan"],
    }
    arguments.update(overrides)
    return build_policy_outcome_statistics_bridge(**arguments)


def _rebind_preoracle_signal(case, signal: SignalOutcomeManifest) -> dict[str, object]:
    """Keep negative linkage fixtures behind the immutable pre-oracle barrier."""

    barrier = replace(
        case["barrier"],
        preoracle_signal_outcome_manifest_digest=str(signal.manifest_digest),
    )
    handoff = replace(
        case["handoff"], public_ranking_barrier_digest=barrier.barrier_digest
    )
    oracle = replace(
        case["oracle"],
        public_ranking_barrier_digest=barrier.barrier_digest,
        signal_outcome_manifest_digest=str(signal.manifest_digest),
        evidence_manifest_digest=None,
    )
    release = replace(
        case["release"],
        public_ranking_barrier_digest=barrier.barrier_digest,
        oracle_unlock_handoff_digest=handoff.handoff_digest,
        oracle_evidence_manifest_digest=str(oracle.evidence_manifest_digest),
        release_receipt_digest=None,
    )
    return {
        "barrier": barrier,
        "oracle_handoff": handoff,
        "oracle_evidence": oracle,
        "external_release_receipt": release,
        "signal_outcomes": signal,
    }


def test_canonical_bridge_joins_rankings_oracle_and_all_primary_inputs(formal_case) -> None:
    bridge = _build(formal_case)
    assert len(bridge.outcomes) == 9 * 66
    assert len(bridge.prefix_inputs) == 66 * 7
    assert len(bridge.linkage_inputs) == 66
    assert set(row.hypothesis_id for row in bridge.frozen_statistics_input.rows) == set(
        PRIMARY_COMPARISON_IDS
    )
    assert set(
        family.contrast_family_id
        for family in formal_case["plan"].formal_statistics_plan.multiplicity_families
    ) == set(FORMAL_CONTRAST_FAMILY_IDS)
    assert all(
        any(
            row.hypothesis_id == hypothesis_id and row.status == "OBSERVED"
            for row in bridge.frozen_statistics_input.rows
        )
        for hypothesis_id in PRIMARY_COMPARISON_IDS
    )
    incompatible = next(
        item for item in bridge.outcomes if item.selected_opaque_policy_id.endswith("0" * 32)
    )
    assert incompatible.selected_return is None
    assert incompatible.abi_incompatible
    assert incompatible.normalized_pool_regret == 1.0
    assert bridge.oracle_release_receipt_digest == formal_case["release"].release_receipt_digest
    compatible = next(item for item in bridge.outcomes if not item.abi_incompatible)
    with pytest.raises(PolicyOutcomeError, match="normalized pool regret disagrees"):
        replace(compatible, normalized_pool_regret=1.0, record_digest=None)
    result = compute_formal_statistics(
        plan=formal_case["plan"].formal_statistics_plan,
        freeze_manifest=formal_case["barrier"].freeze_manifest,
        frozen_input=bridge.frozen_statistics_input,
    )
    assert set(result.contrast_results) == set(PRIMARY_COMPARISON_IDS)


def test_development_oracle_fixture_can_be_small_but_formal_cardinality_fails_closed() -> None:
    query_id = "v03q-" + "a" * 32
    policy_id = "lw-" + "b" * 32
    episode = OracleEpisodeEvidence(
        "episode-0", _d("small-seed"), "EXECUTED", 1.0, _d("small-evidence")
    )
    row = OraclePolicyEvidence(
        query_id,
        policy_id,
        _d("target-abi"),
        _d("policy-abi"),
        True,
        1.0,
        (episode,),
    )
    common = dict(
        run_id="development-oracle-fixture",
        freeze_manifest_digest=_d("freeze"),
        public_ranking_barrier_digest=_d("barrier"),
        public_query_plan_digest=_d("query-plan"),
        query_alias_manifest_digest=_d("aliases"),
        signal_outcome_manifest_digest=_d("signal"),
        policy_market_id=_d("market"),
        expected_opaque_query_ids=(query_id,),
        expected_opaque_policy_ids=(policy_id,),
        episode_ids_by_query={query_id: ("episode-0",)},
        rows=(row,),
    )
    fixture = ExternalOracleEvidenceManifest(scope="DEVELOPMENT", **common)
    assert ExternalOracleEvidenceManifest.from_dict(fixture.to_dict()) == fixture
    with pytest.raises(PolicyOutcomeError, match="66 queries x 30 policies"):
        ExternalOracleEvidenceManifest(scope="FORMAL", **common)


def test_oracle_aggregate_and_rectangle_tampering_are_rejected(formal_case) -> None:
    original = next(item for item in formal_case["oracle"].rows if item.executable)
    with pytest.raises(PolicyOutcomeError, match="episode-return mean"):
        replace(original, policy_value=999.0, policy_evidence_digest=None)
    with pytest.raises(PolicyOutcomeError, match="exact query x policy rectangle"):
        replace(
            formal_case["oracle"],
            rows=formal_case["oracle"].rows[:-1],
            evidence_manifest_digest=None,
        )
    payload = formal_case["oracle"].to_dict()
    payload["rows"][1]["policy_value"] = -123.0
    with pytest.raises(PolicyOutcomeError):
        ExternalOracleEvidenceManifest.from_dict(payload)


def test_ranking_digest_and_exact_method_query_join_are_fail_closed(formal_case) -> None:
    with pytest.raises(PolicyOutcomeError, match="exact barrier method x query matrix"):
        _build(formal_case, rankings=formal_case["rankings"][:-1])
    changed = replace(
        formal_case["rankings"][0],
        cost_digest=_d("changed-cost"),
        ranking_digest=None,
    )
    with pytest.raises(PolicyOutcomeError, match="barrier publication"):
        _build(formal_case, rankings=(changed,) + formal_case["rankings"][1:])


def test_external_receipt_grants_no_oracle_capability_and_plan_must_match_freeze(formal_case) -> None:
    with pytest.raises(PolicyOutcomeError, match="cannot grant"):
        replace(formal_case["release"], v03_oracle_read_capability=True, release_receipt_digest=None)
    changed_plan = CanonicalPrimaryComparisonPlan(
        strongest_b3_b4_method_id="B3b",
        strongest_method_selection_receipt_digest=_d("another-selection"),
        bootstrap_resamples=64,
    )
    with pytest.raises(PolicyOutcomeError, match="pre-experiment freeze"):
        _build(formal_case, comparison_plan=changed_plan)


def test_formal_linkage_cannot_be_self_filled_as_all_na(formal_case) -> None:
    constant_rows = tuple(
        replace(
            item,
            signal_value=0.5,
            prefix_signal_values={prefix: 0.5 for prefix in SignalPrefixSchedule.formal().prefix_episode_counts},
        )
        for item in formal_case["signal"].rows
    )
    constant_signal = replace(formal_case["signal"], rows=constant_rows, manifest_digest=None)
    with pytest.raises(PolicyOutcomeError, match="all N/A"):
        _build(formal_case, **_rebind_preoracle_signal(formal_case, constant_signal))


def test_g03_policy_link_requires_two_tasks_by_two_axes_and_positive_direction(formal_case) -> None:
    one_axis_rows = tuple(
        replace(item, axis_id="only-axis") for item in formal_case["signal"].rows
    )
    with pytest.raises(PolicyOutcomeError, match="at least two tasks and two axes"):
        replace(formal_case["signal"], rows=one_axis_rows, manifest_digest=None)

    reversed_rows = []
    for item in formal_case["signal"].rows:
        value = 1.0 - item.signal_value
        reversed_rows.append(
            replace(
                item,
                signal_value=value,
                prefix_signal_values={
                    prefix: value * prefix / 64.0
                    for prefix in SignalPrefixSchedule.formal().prefix_episode_counts
                },
            )
        )
    reversed_signal = replace(
        formal_case["signal"], rows=tuple(reversed_rows), manifest_digest=None
    )
    with pytest.raises(PolicyOutcomeError, match="positive-direction linkage"):
        _build(formal_case, **_rebind_preoracle_signal(formal_case, reversed_signal))


def test_formal_statistics_plan_requires_six_families_and_frozen_na_reasons(formal_case) -> None:
    plan = formal_case["plan"].formal_statistics_plan
    retained = next(
        item for item in plan.contrasts if item.hypothesis_id == PRIMARY_COMPARISON_IDS[6]
    )
    family = next(
        item
        for item in plan.multiplicity_families
        if item.contrast_family_id == retained.contrast_family_id
    )
    with pytest.raises(V03StatisticsError, match="all six contrast families"):
        replace(
            plan,
            contrasts=(retained,),
            multiplicity_families=(family,),
            registered_n_a_reasons={
                retained.hypothesis_id: plan.registered_n_a_reasons[retained.hypothesis_id]
            },
        )
    with pytest.raises(V03StatisticsError, match="every hypothesis"):
        replace(
            plan,
            registered_n_a_reasons={
                key: value
                for key, value in plan.registered_n_a_reasons.items()
                if key != PRIMARY_COMPARISON_IDS[0]
            },
        )
