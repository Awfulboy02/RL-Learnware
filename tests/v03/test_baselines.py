from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v02.baselines import DevelopmentView
from policy_learnware_v0.v02.representation import TraceFeatureVector
from policy_learnware_v0.v02.schemas import ExecutionABIRecord
from policy_learnware_v0.v03.anonymous_market import (
    build_anonymous_joint_distance_request,
    build_anonymous_selector_view,
)
from policy_learnware_v0.v03.baselines import (
    BASELINE_METHOD_KINDS,
    DEVELOPMENT_SMOKE_MODE,
    FORMAL_MODE,
    OPTIONAL_BASELINE_STATES,
    REQUIRED_BASELINE_METHOD_IDS,
    PublishedFullRanking,
    RawMomentFeatureProtocol,
    V02V03QueryAliasEntry,
    V02V03QueryAliasManifest,
    V03BaselineError,
    V03BaselineQuery,
    audit_baseline_rank_one_execution_abi,
    build_v02_v03_query_alias_manifest,
    derive_v03_source_only_sigma,
    fit_baseline_suite,
    freeze_development_baselines,
    raw_moment_feature_from_view,
    run_baseline_ranking,
)
from policy_learnware_v0.v03.compute import run_joint_distance_stage
from policy_learnware_v0.v03.contracts import (
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    SemanticCacheKey,
    SemanticCacheRecord,
    SemanticTransform,
    SourceRepresentationIndex,
    bind_source_representation_index_to_market,
    build_empirical_query_spec,
    build_source_reduced_spec,
)
from policy_learnware_v0.v03.preflight import PublicRankingPublication
from policy_learnware_v0.v03.source_market import (
    SourceChampion,
    SourceChampionizationRecord,
    SourceCompetenceObservation,
    build_source_policy_market,
)
from policy_learnware_v0.v03.transition_views import (
    V_STATE_ONLY,
    VIEW_REGISTRY,
    TransitionBank,
    apply_transition_view,
)


def _d(label: str) -> str:
    return sha256_json({"v03-baseline-test": label})


def _abi(label: str) -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id=f"continuous-vector-mdp-{label}",
        observation_tensor_abi_digest=_d(f"observation-abi:{label}"),
        action_tensor_abi_digest=_d(f"action-abi:{label}"),
        action_transform_id="tanh",
        policy_runtime_id=f"runtime-{label}",
        state_abi_id="stateless",
    )


def _championization() -> SourceChampionizationRecord:
    champions = {}
    for index in range(30):
        anchor = _d(f"anchor:{index}")
        candidate = "v02j-" + f"{index:024x}"
        attestation = _d(f"attestation:{index}")
        competence = SourceCompetenceObservation(
            source_anchor_id=anchor,
            candidate_id=candidate,
            attestation_receipt_digest=attestation,
            episode_count=3,
            mean=0.8,
            std=0.0,
            lcb=0.8,
            normalized_competence=0.8,
            competence_floor=0.5,
            passed=True,
        )
        champions[anchor] = SourceChampion(
            source_anchor_id=anchor,
            candidate_id=candidate,
            seed=0,
            intake_cell_digest=_d(f"intake:{index}"),
            bundle_digest=_d(f"bundle:{index}"),
            bundle_path=f"/synthetic/policy-{index}.pkl",
            outer_iteration=1,
            environment_steps=100,
            selection_receipt_digest=_d(f"selection:{index}"),
            attestation_receipt_digest=attestation,
            competence=competence,
        )
    return SourceChampionizationRecord(
        intake_record_digest=_d("intake-record"),
        source_evaluation_protocol_digest=_d("source-evaluation"),
        selection_receipt_index_digest=_d("selection-index"),
        attestation_receipt_index_digest=_d("attestation-index"),
        provisional_selection_digest=_d("provisional-selection"),
        attestation_plan_digest=_d("attestation-plan"),
        champions=champions,
    )


def _cache(
    label: str,
    points: np.ndarray,
    transform: SemanticTransform,
) -> SemanticCacheRecord:
    return SemanticCacheRecord(
        key=SemanticCacheKey(
            raw_dataset_digest=_d(f"raw:{label}"),
            ordered_episode_window_digest=_d(f"window:{label}"),
            canonical_view_digest=_d("canonical-view"),
            window_protocol_digest=_d("window-protocol"),
            normalizer_digest=_d("normalizer"),
            semantic_transform=transform,
            mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
        ),
        points=np.asarray(points, dtype=np.float64),
        episode_offsets=np.asarray([0, len(points)], dtype=np.int64),
    )


def _build_index(market, centres, transform, label):
    reducer = ReducerConfig(
        support_budget=2,
        support_steps=0,
        kmeans_steps=0,
        ridge=0.0,
        pinv_rcond=1.0e-12,
    )
    entries = {}
    for opaque_id, private in market.deployment_private.items():
        centre = centres[private.candidate_id]
        cache = _cache(
            f"{label}:{private.candidate_id}",
            np.asarray([[centre], [centre + 0.01]], dtype=np.float64),
            transform,
        )
        entries[opaque_id] = build_source_reduced_spec(
            cache,
            kernel_bandwidth=1.0,
            measurement_protocol_id=_d("measurement"),
            probe_dataset_digest=cache.key.raw_dataset_digest,
            reducer_config=reducer,
        )
    base = SourceRepresentationIndex(
        representation_protocol_id=next(iter(entries.values())).representation_protocol_id,
        entries=entries,
    )
    return bind_source_representation_index_to_market(
        base, policy_market_id=market.policy_market_id
    )


@pytest.fixture(scope="module")
def baseline_case():
    championization = _championization()
    compatible_abi = _abi("compatible")
    market = build_source_policy_market(
        championization,
        {
            champion.candidate_id: compatible_abi
            for champion in championization.champions.values()
        },
        market_alias_nonce=_d("alias-nonce"),
        tie_break_nonce=_d("tie-nonce"),
    )
    candidates = sorted(
        champion.candidate_id for champion in championization.champions.values()
    )
    centres = {candidate: float(index) for index, candidate in enumerate(candidates)}
    raw_transform = SemanticTransform.raw_identity()
    corro_transform = SemanticTransform.frozen_encoder(
        encoder_implementation_digest=_d("corro-implementation"),
        checkpoint_digest=_d("corro-checkpoint"),
        semantic_output_protocol_digest=_d("corro-output"),
    )
    raw_index = _build_index(market, centres, raw_transform, "raw")
    corro_index = _build_index(
        market,
        {candidate: 0.5 * value for candidate, value in centres.items()},
        corro_transform,
        "corro",
    )
    raw_view = build_anonymous_selector_view(market, raw_index)
    corro_view = build_anonymous_selector_view(market, corro_index)

    raw_cache = _cache(
        "shared-query",
        np.asarray([[1.0], [1.01]], dtype=np.float64),
        raw_transform,
    )
    raw_spec = build_empirical_query_spec(
        raw_cache,
        kernel_bandwidth=1.0,
        measurement_protocol_id=_d("measurement"),
        probe_dataset_digest=raw_cache.key.raw_dataset_digest,
    )
    corro_cache = _cache(
        "shared-query",
        np.asarray([[0.5], [0.505]], dtype=np.float64),
        corro_transform,
    )
    corro_spec = build_empirical_query_spec(
        corro_cache,
        kernel_bandwidth=1.0,
        measurement_protocol_id=_d("measurement"),
        probe_dataset_digest=corro_cache.key.raw_dataset_digest,
    )
    raw_run = run_joint_distance_stage(
        build_anonymous_joint_distance_request(raw_spec, raw_index, raw_view)
    )
    corro_run = run_joint_distance_stage(
        build_anonymous_joint_distance_request(corro_spec, corro_index, corro_view)
    )

    feature_protocol = _d("simple-feature-protocol")
    policy_ids = tuple(sorted(market.entries))
    centre_by_id = {
        opaque_id: centres[private.candidate_id]
        for opaque_id, private in market.deployment_private.items()
    }
    raw_features = {
        opaque_id: TraceFeatureVector(
            values=np.asarray([centre_by_id[opaque_id]], dtype=np.float64),
            feature_protocol_id=feature_protocol,
            probe_dataset_digest=_d(f"source-feature:{opaque_id}"),
        )
        for opaque_id in policy_ids
    }
    nominal_ids = tuple(
        sorted(policy_ids, key=lambda item: centre_by_id[item])[position]
        for position in (0, 5, 10, 15, 20, 25)
    )
    legacy_specs = {
        f"legacy-task-{position}": TraceFeatureVector(
            values=np.asarray([centre_by_id[opaque_id]], dtype=np.float64),
            feature_protocol_id=feature_protocol,
            probe_dataset_digest=_d(f"legacy-feature:{position}"),
        )
        for position, opaque_id in enumerate(nominal_ids)
    }
    nominal_champions = {
        f"legacy-task-{position}": opaque_id
        for position, opaque_id in enumerate(nominal_ids)
    }
    context_ids = tuple(f"v03-development-{index}" for index in range(5))
    context_coordinates = np.asarray([0.0, 5.0, 10.0, 20.0, 28.0])
    normalized_returns = np.asarray(
        [
            [max(0.0, 1.0 - abs(centre_by_id[opaque_id] - coordinate) / 30.0) for opaque_id in policy_ids]
            for coordinate in context_coordinates
        ],
        dtype=np.float64,
    )
    development = DevelopmentView(
        context_ids=context_ids,
        opaque_policy_ids=policy_ids,
        context_features=context_coordinates[:, None],
        normalized_returns=normalized_returns,
        training_context_ids=context_ids[:4],
        validation_context_ids=context_ids[4:],
        evaluation_seed_digests=tuple(_d(f"development-seeds:{item}") for item in context_ids),
        policy_market_id=market.policy_market_id,
        feature_protocol_id=feature_protocol,
        split_manifest_digest=_d("development-split"),
        label_contract_digest=_d("development-labels"),
        candidate_paired_seeds=True,
    )
    freeze = freeze_development_baselines(
        development,
        selector_seed=17,
        b4a_neighbor_count=2,
        b4b_ridge=0.1,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
    )
    artifacts = fit_baseline_suite(
        market=market,
        raw_index=raw_index,
        corro_index=corro_index,
        development_view=development,
        development_freeze=freeze,
        legacy_task_specs=legacy_specs,
        nominal_champions=nominal_champions,
        raw_moment_features=raw_features,
    )
    query_feature = TraceFeatureVector(
        values=np.asarray([1.0], dtype=np.float64),
        feature_protocol_id=feature_protocol,
        probe_dataset_digest=raw_spec.probe_dataset_digest,
    )
    raw_query = V03BaselineQuery(
        opaque_query_id="v03q-" + "1" * 32,
        query_spec_digest=str(raw_spec.query_spec_digest),
        probe_dataset_digest=raw_spec.probe_dataset_digest,
        target_evidence_digest=_d("target-evidence"),
        cost_digest=_d("cost"),
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        query_mode=raw_spec.query_mode,
        trace_feature=query_feature,
    )
    corro_query = V03BaselineQuery(
        opaque_query_id=raw_query.opaque_query_id,
        query_spec_digest=str(corro_spec.query_spec_digest),
        probe_dataset_digest=corro_spec.probe_dataset_digest,
        target_evidence_digest=_d("target-evidence"),
        cost_digest=_d("cost"),
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        query_mode=corro_spec.query_mode,
        trace_feature=query_feature,
    )
    return {
        "market": market,
        "compatible_abi": compatible_abi,
        "raw_index": raw_index,
        "corro_index": corro_index,
        "raw_view": raw_view,
        "corro_view": corro_view,
        "raw_run": raw_run,
        "corro_run": corro_run,
        "raw_query": raw_query,
        "corro_query": corro_query,
        "development": development,
        "freeze": freeze,
        "artifacts": artifacts,
        "legacy_specs": legacy_specs,
        "nominal_champions": nominal_champions,
        "raw_features": raw_features,
    }


def _rank(case, method_id):
    use_corro = method_id in {"A-Env", "M02/B5"}
    distance_run = None
    if method_id == "B3b":
        distance_run = case["raw_run"]
    elif use_corro:
        distance_run = case["corro_run"]
    return run_baseline_ranking(
        query=case["corro_query"] if use_corro else case["raw_query"],
        selector_view=case["corro_view"] if use_corro else case["raw_view"],
        artifact=case["artifacts"][method_id],
        distance_run=distance_run,
    )


def test_exact_required_registry_fits_and_ranks_the_full_anonymous_pool(
    baseline_case,
) -> None:
    assert tuple(baseline_case["artifacts"]) == REQUIRED_BASELINE_METHOD_IDS
    rankings = {
        method_id: _rank(baseline_case, method_id)
        for method_id in REQUIRED_BASELINE_METHOD_IDS
    }
    for method_id, ranking in rankings.items():
        assert isinstance(ranking, PublishedFullRanking)
        assert ranking.method_id == method_id
        assert len(ranking.rows) == 30
        assert {row.opaque_learnware_id for row in ranking.rows} == set(
            baseline_case["market"].entries
        )
        assert tuple(row.rank for row in ranking.rows) == tuple(range(1, 31))
        assert ranking.selected_opaque_learnware_id == ranking.rows[0].opaque_learnware_id
        assert ranking.development_freeze_digest == baseline_case["freeze"].freeze_digest
        assert "oracle" not in str(ranking.to_dict()).lower()

    # Equal B1 competence is resolved only by the frozen market tie token.
    b1 = rankings["B1"]
    expected = min(
        baseline_case["raw_view"].entries,
        key=lambda item: baseline_case["raw_view"].entries[item].tie_break_token,
    )
    assert b1.selected_opaque_learnware_id == expected


def test_empirical_query_distance_adapters_and_lmin_source_sigma(baseline_case) -> None:
    b3b = _rank(baseline_case, "B3b")
    a_env = _rank(baseline_case, "A-Env")
    b5 = _rank(baseline_case, "M02/B5")
    raw_distances = {
        row.opaque_learnware_id: row.result.value for row in baseline_case["raw_run"].rows
    }
    corro_distances = {
        row.opaque_learnware_id: row.result.value for row in baseline_case["corro_run"].rows
    }
    assert {row.opaque_learnware_id: row.distance for row in b3b.rows} == pytest.approx(
        raw_distances
    )
    assert {row.opaque_learnware_id: row.distance for row in a_env.rows} == pytest.approx(
        corro_distances
    )
    assert {row.opaque_learnware_id: row.distance for row in b5.rows} == pytest.approx(
        corro_distances
    )
    sigma = derive_v03_source_only_sigma(baseline_case["corro_index"])
    assert sigma.sigma > 0.0
    assert sigma.policy_market_id == baseline_case["market"].policy_market_id
    assert sigma.to_dict()["zero_distance_fallback"] is None

    repeated = next(iter(baseline_case["raw_index"].entries.values()))
    collapsed_base = SourceRepresentationIndex(
        representation_protocol_id=repeated.representation_protocol_id,
        entries={opaque_id: repeated for opaque_id in baseline_case["market"].entries},
    )
    collapsed = bind_source_representation_index_to_market(
        collapsed_base, policy_market_id=baseline_case["market"].policy_market_id
    )
    with pytest.raises(V03BaselineError, match="no nonzero"):
        derive_v03_source_only_sigma(collapsed)


def test_development_freeze_is_private_and_optional_methods_are_explicitly_disabled(
    baseline_case,
) -> None:
    freeze = baseline_case["freeze"]
    assert dict(freeze.optional_method_states) == dict(OPTIONAL_BASELINE_STATES)
    assert freeze.label_count == baseline_case["development"].label_count == 150
    assert freeze.to_private_dict()["confirmatory_oracle_access"] is False
    assert baseline_case["artifacts"]["B4a"].fit_scope == "development_supervised"
    assert baseline_case["artifacts"]["B4b"].fit_scope == "development_supervised"
    assert all(
        artifact.fit_scope == "source_only"
        for method_id, artifact in baseline_case["artifacts"].items()
        if method_id not in {"B4a", "B4b"}
    )
    assert baseline_case["artifacts"]["B4a"].evidence_contract.reads_development_policy_returns
    assert not baseline_case["artifacts"]["B3a"].evidence_contract.reads_development_policy_returns
    assert baseline_case["artifacts"]["B2"].evidence_contract.reads_source_side_labels
    assert not freeze.formal_24_context_matrix_ready
    with pytest.raises(V03BaselineError, match="exactly 24"):
        freeze.require_formal_24_context_matrix()


def test_formal_fit_and_run_require_24_contexts_and_probe_bound_features(
    baseline_case,
) -> None:
    with pytest.raises(V03BaselineError, match="exactly 24"):
        freeze_development_baselines(
            baseline_case["development"],
            selector_seed=17,
            b4a_neighbor_count=2,
            b4b_ridge=0.1,
            execution_mode=FORMAL_MODE,
        )

    raw_query = baseline_case["raw_query"]
    with pytest.raises(V03BaselineError, match="same probe dataset"):
        replace(
            raw_query,
            trace_feature=TraceFeatureVector(
                values=raw_query.trace_feature.values,
                feature_protocol_id=raw_query.trace_feature.feature_protocol_id,
                probe_dataset_digest=_d("wrong-probe-dataset"),
            ),
        )

    development = baseline_case["development"]
    context_ids = tuple(f"v03-formal-development-{index}" for index in range(24))
    coordinates = np.linspace(0.0, 29.0, num=24, dtype=np.float64)
    formal_view = DevelopmentView(
        context_ids=context_ids,
        opaque_policy_ids=development.opaque_policy_ids,
        context_features=coordinates[:, None],
        normalized_returns=np.asarray(
            [
                [
                    max(0.0, 1.0 - abs(policy_position - coordinate) / 30.0)
                    for policy_position in range(30)
                ]
                for coordinate in coordinates
            ],
            dtype=np.float64,
        ),
        training_context_ids=context_ids[:18],
        validation_context_ids=context_ids[18:],
        evaluation_seed_digests=tuple(
            _d(f"formal-development-seeds:{item}") for item in context_ids
        ),
        policy_market_id=development.policy_market_id,
        feature_protocol_id=development.feature_protocol_id,
        split_manifest_digest=_d("formal-development-split"),
        label_contract_digest=_d("formal-development-labels"),
        candidate_paired_seeds=True,
    )
    formal_freeze = freeze_development_baselines(
        formal_view,
        selector_seed=17,
        b4a_neighbor_count=2,
        b4b_ridge=0.1,
        execution_mode=FORMAL_MODE,
    )
    formal_artifacts = fit_baseline_suite(
        market=baseline_case["market"],
        raw_index=baseline_case["raw_index"],
        corro_index=baseline_case["corro_index"],
        development_view=formal_view,
        development_freeze=formal_freeze,
        legacy_task_specs=baseline_case["legacy_specs"],
        nominal_champions=baseline_case["nominal_champions"],
        raw_moment_features=baseline_case["raw_features"],
    )
    formal_query = replace(raw_query, execution_mode=FORMAL_MODE)
    formal_ranking = run_baseline_ranking(
        query=formal_query,
        selector_view=baseline_case["raw_view"],
        artifact=formal_artifacts["B0"],
    )
    assert formal_ranking.execution_mode == FORMAL_MODE
    assert formal_ranking.query_mode == "QUERY_EMPIRICAL"
    assert formal_ranking.development_context_count == 24
    assert formal_ranking.probe_dataset_digest == formal_query.probe_dataset_digest
    publication = PublicRankingPublication.from_published_ranking(formal_ranking)
    assert publication.method_id == formal_ranking.method_id
    assert publication.query_input_digest == formal_ranking.query_input_digest
    assert publication.cost_digest == formal_ranking.cost_digest

    with pytest.raises(V03BaselineError, match="execution modes differ"):
        run_baseline_ranking(
            query=formal_query,
            selector_view=baseline_case["raw_view"],
            artifact=baseline_case["artifacts"]["B0"],
        )
    with pytest.raises(V03BaselineError, match="QUERY_EMPIRICAL"):
        replace(formal_query, query_mode="QUERY_REDUCED")


def test_method_identity_and_source_sigma_payload_fail_closed(baseline_case) -> None:
    b3a = baseline_case["artifacts"]["B3a"]
    with pytest.raises(V03BaselineError, match="implementation kind disagree"):
        replace(
            b3a,
            payload={**dict(b3a.payload), "kind": BASELINE_METHOD_KINDS["B0"]},
            selector_artifact_digest=None,
        )

    b5 = baseline_case["artifacts"]["M02/B5"]
    sigma_payload = dict(b5.payload["source_only_sigma"])
    sigma_payload["sigma"] = float(sigma_payload["sigma"]) * 2.0
    tampered_b5 = replace(
        b5,
        payload={**dict(b5.payload), "source_only_sigma": sigma_payload},
        selector_artifact_digest=None,
    )
    with pytest.raises(V03BaselineError, match="sigma payload failed validation"):
        run_baseline_ranking(
            query=baseline_case["corro_query"],
            selector_view=baseline_case["corro_view"],
            artifact=tampered_b5,
            distance_run=baseline_case["corro_run"],
        )


def test_public_runner_has_no_oracle_or_private_market_input_and_is_digest_bound(
    baseline_case,
) -> None:
    parameters = inspect.signature(run_baseline_ranking).parameters
    assert "oracle" not in parameters
    assert "market" not in parameters
    first = _rank(baseline_case, "B0")
    second = _rank(baseline_case, "B0")
    assert first.ranking_digest == second.ranking_digest
    changed_query = replace(
        baseline_case["raw_query"], cost_digest=_d("changed-cost")
    )
    changed = run_baseline_ranking(
        query=changed_query,
        selector_view=baseline_case["raw_view"],
        artifact=baseline_case["artifacts"]["B0"],
    )
    assert changed.ranking_digest != first.ranking_digest

    swapped = (first.rows[1], first.rows[0], *first.rows[2:])
    with pytest.raises(V03BaselineError, match="ranks must be contiguous"):
        replace(first, rows=swapped, ranking_digest=None)
    with pytest.raises(V03BaselineError, match="another query spec"):
        run_baseline_ranking(
            query=replace(
                baseline_case["raw_query"], query_spec_digest=_d("wrong-query")
            ),
            selector_view=baseline_case["raw_view"],
            artifact=baseline_case["artifacts"]["B3b"],
            distance_run=baseline_case["raw_run"],
        )
    with pytest.raises(V03BaselineError, match="query mode differs"):
        run_baseline_ranking(
            query=baseline_case["raw_query"],
            selector_view=baseline_case["raw_view"],
            artifact=baseline_case["artifacts"]["B3b"],
            distance_run=replace(
                baseline_case["raw_run"],
                query_mode="QUERY_REDUCED",
                run_digest=None,
            ),
        )


def test_rank_one_abi_audit_is_generic_and_never_falls_back(baseline_case) -> None:
    for method_id in ("B0", "B2", "B3b", "B4a", "M02/B5"):
        ranking = _rank(baseline_case, method_id)
        compatible = audit_baseline_rank_one_execution_abi(
            ranking, baseline_case["market"], baseline_case["compatible_abi"]
        )
        assert compatible.compatible
        assert compatible.inspected_opaque_learnware_ids == (
            ranking.selected_opaque_learnware_id,
        )
        assert compatible.fallback_attempted is False
        incompatible = audit_baseline_rank_one_execution_abi(
            ranking, baseline_case["market"], _abi("incompatible")
        )
        assert not incompatible.compatible
        assert incompatible.status == "SELECTED_INCOMPATIBLE_ABI"
        assert incompatible.fallback_attempted is False

    ranking = _rank(baseline_case, "B0")
    tampered_rows = (*ranking.rows[:-1], replace(ranking.rows[-1], tie_break_token=_d("tampered-tie")))
    tampered_ranking = replace(ranking, rows=tampered_rows, ranking_digest=None)
    with pytest.raises(V03BaselineError, match="tie-break tokens differ"):
        audit_baseline_rank_one_execution_abi(
            tampered_ranking,
            baseline_case["market"],
            baseline_case["compatible_abi"],
        )


def test_cross_version_query_aliases_are_explicit_deterministic_and_tamper_evident() -> None:
    bindings = {
        "v02q-" + "1" * 32: _d("context-1"),
        "v02q-" + "2" * 32: _d("context-2"),
    }
    first = build_v02_v03_query_alias_manifest(
        bindings, alias_nonce_digest=_d("alias-domain")
    )
    second = build_v02_v03_query_alias_manifest(
        dict(reversed(tuple(bindings.items()))), alias_nonce_digest=_d("alias-domain")
    )
    assert first.to_dict() == second.to_dict()
    assert all(
        entry.v03_query_id.startswith("v03q-") for entry in first.entries.values()
    )
    assert len({entry.v03_query_id for entry in first.entries.values()}) == 2

    key = next(iter(first.entries))
    tampered_entries = dict(first.entries)
    tampered_entries[key] = V02V03QueryAliasEntry(
        v02_query_id=key,
        v03_query_id="v03q-" + "f" * 32,
        context_binding_digest=bindings[key],
    )
    with pytest.raises(V03BaselineError, match="not derived"):
        V02V03QueryAliasManifest(
            alias_nonce_digest=first.alias_nonce_digest,
            entries=tampered_entries,
        )


def test_b3a_raw_moment_bridge_is_view_and_weighting_bound() -> None:
    bank = TransitionBank(
        observation=np.asarray([[0.0], [2.0], [4.0]], dtype=np.float32),
        action=np.zeros((3, 1), dtype=np.float32),
        reward=np.zeros(3, dtype=np.float32),
        next_observation=np.asarray([[1.0], [3.0], [5.0]], dtype=np.float32),
        terminated=np.asarray([True, False, True]),
        truncated=np.asarray([False, False, False]),
        episode_offsets=np.asarray([0, 1, 3], dtype=np.int64),
    )
    view = apply_transition_view(bank, V_STATE_ONLY)
    balanced = RawMomentFeatureProtocol(
        transition_view_spec_digest=VIEW_REGISTRY[V_STATE_ONLY].digest,
        statistics=("mean", "std", "second_moment"),
        weighting="episode_balanced",
    )
    uniform = RawMomentFeatureProtocol(
        transition_view_spec_digest=VIEW_REGISTRY[V_STATE_ONLY].digest,
        statistics=("mean",),
        weighting="transition_uniform",
    )
    balanced_feature = raw_moment_feature_from_view(
        view, balanced, probe_dataset_digest=_d("moment-bank")
    )
    uniform_feature = raw_moment_feature_from_view(
        view, uniform, probe_dataset_digest=_d("moment-bank")
    )
    assert balanced_feature.values[0] == pytest.approx(1.5)
    assert uniform_feature.values[0] == pytest.approx(2.0)
    assert balanced_feature.feature_protocol_id != uniform_feature.feature_protocol_id
    with pytest.raises(V03BaselineError, match="differs"):
        raw_moment_feature_from_view(
            view,
            replace(balanced, transition_view_spec_digest=_d("another-view")),
            probe_dataset_digest=_d("moment-bank"),
        )
