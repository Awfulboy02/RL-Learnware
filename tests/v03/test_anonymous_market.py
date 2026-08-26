from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v02.schemas import ExecutionABIRecord
from policy_learnware_v0.v03.anonymous_market import (
    AnonymousMarketError,
    AnonymousSelectorViewManifest,
    audit_rank1_execution_abi,
    build_anonymous_joint_distance_request,
    build_anonymous_selector_view,
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
from policy_learnware_v0.v03.source_market import (
    SourceChampion,
    SourceChampionizationRecord,
    SourceCompetenceObservation,
    build_source_policy_market,
)


def _d(label: str) -> str:
    return sha256_json({"anonymous-market-test": label})


def _abi(label: str) -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id=f"continuous-vector-mdp-{label}",
        observation_tensor_abi_digest=_d(f"observation-abi:{label}"),
        action_tensor_abi_digest=_d(f"action-abi:{label}"),
        action_transform_id="tanh",
        policy_runtime_id=f"runtime-{label}",
        state_abi_id="stateless",
    )


@pytest.fixture(scope="module")
def championization() -> SourceChampionizationRecord:
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
            intake_cell_digest=_d(f"intake-cell:{index}"),
            bundle_digest=_d(f"bundle:{index}"),
            bundle_path=f"/synthetic/policy-{index}.pkl",
            outer_iteration=1,
            environment_steps=100,
            selection_receipt_digest=_d(f"selection:{index}"),
            attestation_receipt_digest=attestation,
            competence=competence,
        )
    return SourceChampionizationRecord(
        intake_record_digest=_d("intake"),
        source_evaluation_protocol_digest=_d("source-evaluation"),
        selection_receipt_index_digest=_d("selection-index"),
        attestation_receipt_index_digest=_d("attestation-index"),
        provisional_selection_digest=_d("provisional-selection"),
        attestation_plan_digest=_d("attestation-plan"),
        champions=champions,
    )


def _market(championization, alias_label: str, abis):
    return build_source_policy_market(
        championization,
        abis,
        market_alias_nonce=_d(f"market-alias:{alias_label}"),
        tie_break_nonce=_d("tie-break-stable-by-candidate"),
    )


def _cache(label: str, points: np.ndarray) -> SemanticCacheRecord:
    return SemanticCacheRecord(
        key=SemanticCacheKey(
            raw_dataset_digest=_d(f"raw:{label}"),
            ordered_episode_window_digest=_d(f"window:{label}"),
            canonical_view_digest=_d("canonical-view"),
            window_protocol_digest=_d("window-protocol"),
            normalizer_digest=_d("normalizer"),
            semantic_transform=SemanticTransform.raw_identity(),
            mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
        ),
        points=np.asarray(points, dtype=np.float64),
        episode_offsets=np.asarray([0, len(points)], dtype=np.int64),
    )


def _centres(championization) -> dict[str, float]:
    candidates = sorted(
        champion.candidate_id for champion in championization.champions.values()
    )
    return {candidate: float(index) for index, candidate in enumerate(candidates)}


def _source_index(market, centres):
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
            private.candidate_id,
            np.asarray([[centre], [centre + 0.01]], dtype=np.float64),
        )
        entries[opaque_id] = build_source_reduced_spec(
            cache,
            kernel_bandwidth=1.0,
            measurement_protocol_id=_d("measurement"),
            probe_dataset_digest=cache.key.raw_dataset_digest,
            reducer_config=reducer,
        )
    base = SourceRepresentationIndex(
        representation_protocol_id=(
            next(iter(entries.values())).representation_protocol_id
        ),
        entries=entries,
    )
    return bind_source_representation_index_to_market(
        base,
        policy_market_id=market.policy_market_id,
    )


def _query() -> object:
    cache = _cache(
        "query",
        np.asarray([[0.0], [0.01]], dtype=np.float64),
    )
    return build_empirical_query_spec(
        cache,
        kernel_bandwidth=1.0,
        measurement_protocol_id=_d("measurement"),
        probe_dataset_digest=cache.key.raw_dataset_digest,
    )


def _shared_abis(championization, abi):
    return {
        champion.candidate_id: abi
        for champion in championization.champions.values()
    }


def test_public_selector_view_is_exact_typed_join_without_formal_authority(
    championization,
) -> None:
    market = _market(
        championization,
        "join",
        _shared_abis(championization, _abi("shared")),
    )
    index = _source_index(market, _centres(championization))
    with pytest.raises(AnonymousMarketError, match="explicitly market-bound"):
        build_anonymous_selector_view(market, index.source_index)
    view = build_anonymous_selector_view(market, index)

    assert view.policy_market_id == market.policy_market_id
    assert view.representation_index_digest == index.representation_index_digest
    assert len(view.entries) == 30
    assert view.evidence_scope == "ENGINEERING_CONTRACT_ONLY"
    assert view.formal_authority_available is False
    for opaque_id, entry in view.entries.items():
        assert entry.opaque_learnware_id == opaque_id
        assert entry.environment_spec_digest == index.entries[opaque_id].source_spec_digest
        assert (
            entry.normalized_source_competence
            == market.entries[opaque_id].normalized_source_competence
        )
        assert entry.tie_break_token == market.entries[opaque_id].tie_break_token

    restored = AnonymousSelectorViewManifest.from_dict(view.to_dict())
    assert restored.to_dict() == view.to_dict()
    serialized = repr(view.to_dict())
    assert "candidate_id" not in serialized
    assert "bundle_path" not in serialized
    assert "execution_abi" not in serialized
    assert "asset_state" not in serialized

    self_authorized = view.to_dict()
    self_authorized["formal_authority_available"] = True
    with pytest.raises(AnonymousMarketError, match="cannot self-assert"):
        AnonymousSelectorViewManifest.from_dict(self_authorized)


def test_production_join_rejects_wrong_market_subset_extra_and_noncanonical_ids(
    championization,
) -> None:
    market = _market(
        championization,
        "coverage",
        _shared_abis(championization, _abi("shared")),
    )
    index = _source_index(market, _centres(championization))
    entries = dict(index.entries)

    wrong_market = replace(
        index,
        policy_market_id=_d("another-market"),
        market_bound_index_digest=None,
    )
    with pytest.raises(AnonymousMarketError, match="policy_market_id"):
        build_anonymous_selector_view(market, wrong_market)

    subset = bind_source_representation_index_to_market(
        SourceRepresentationIndex(
            index.representation_protocol_id,
            dict(list(entries.items())[:-1]),
        ),
        policy_market_id=market.policy_market_id,
    )
    with pytest.raises(AnonymousMarketError, match="exactly 30"):
        build_anonymous_selector_view(market, subset)

    extra_id = "lw-" + "f" * 32
    if extra_id in entries:
        extra_id = "lw-" + "e" * 32
    extra = bind_source_representation_index_to_market(
        SourceRepresentationIndex(
            index.representation_protocol_id,
            {**entries, extra_id: next(iter(entries.values()))},
        ),
        policy_market_id=market.policy_market_id,
    )
    with pytest.raises(AnonymousMarketError, match="exactly 30"):
        build_anonymous_selector_view(market, extra)

    first_id = next(iter(entries))
    noncanonical_entries = dict(entries)
    noncanonical_entries["source-zero"] = noncanonical_entries.pop(first_id)
    noncanonical = bind_source_representation_index_to_market(
        SourceRepresentationIndex(
            index.representation_protocol_id,
            noncanonical_entries,
        ),
        policy_market_id=market.policy_market_id,
    )
    with pytest.raises(AnonymousMarketError, match="non-canonical"):
        build_anonymous_selector_view(market, noncanonical)


def test_consistent_anonymous_id_permutation_preserves_physical_selection(
    championization,
) -> None:
    shared = _abi("shared")
    abis = _shared_abis(championization, shared)
    market_a = _market(championization, "permutation-a", abis)
    market_b = _market(championization, "permutation-b", abis)
    centres = _centres(championization)
    query = _query()

    def run(market):
        index = _source_index(market, centres)
        view = build_anonymous_selector_view(market, index)
        request = build_anonymous_joint_distance_request(query, index, view)
        distance_run = run_joint_distance_stage(request)
        result = audit_rank1_execution_abi(
            opaque_query_id="v03q-" + "1" * 32,
            request=request,
            distance_run=distance_run,
            selector_view=view,
            market=market,
            target_execution_abi=shared,
        )
        physical_ranking = [
            market.deployment_private[row.opaque_learnware_id].candidate_id
            for row in distance_run.rows
        ]
        return view, result, physical_ranking

    view_a, result_a, ranking_a = run(market_a)
    view_b, result_b, ranking_b = run(market_b)
    assert set(view_a.entries) != set(view_b.entries)
    assert ranking_a == ranking_b
    assert (
        market_a.deployment_private[result_a.selected_opaque_learnware_id].candidate_id
        == market_b.deployment_private[result_b.selected_opaque_learnware_id].candidate_id
        == sorted(centres)[0]
    )
    assert result_a.deployable and result_b.deployable
    assert result_a.failure_record is None and result_b.failure_record is None
    assert result_a.fallback_attempted is False and result_b.fallback_attempted is False


def test_incompatible_rank_one_is_terminal_failure_without_rank_two_fallback(
    championization,
) -> None:
    centres = _centres(championization)
    target_abi = _abi("target")
    selected_candidate = sorted(centres)[0]
    abis = {
        champion.candidate_id: (
            _abi("incompatible")
            if champion.candidate_id == selected_candidate
            else target_abi
        )
        for champion in championization.champions.values()
    }
    market = _market(championization, "incompatible-top-one", abis)
    index = _source_index(market, centres)
    view = build_anonymous_selector_view(market, index)
    request = build_anonymous_joint_distance_request(_query(), index, view)
    distance_run = run_joint_distance_stage(request)
    rank_one = distance_run.rows[0].opaque_learnware_id
    rank_two = distance_run.rows[1].opaque_learnware_id
    assert market.deployment_private[rank_one].candidate_id == selected_candidate
    assert market.deployment_private[rank_one].execution_abi.digest != target_abi.digest
    assert market.deployment_private[rank_two].execution_abi.digest == target_abi.digest

    result = audit_rank1_execution_abi(
        opaque_query_id="v03q-" + "2" * 32,
        request=request,
        distance_run=distance_run,
        selector_view=view,
        market=market,
        target_execution_abi=target_abi,
    )

    assert result.selected_opaque_learnware_id == rank_one
    assert result.selected_opaque_learnware_id != rank_two
    assert result.deployable is False
    assert result.status == "SELECTED_INCOMPATIBLE_ABI"
    assert result.failure_record is not None
    assert result.failure_record.status == "SELECTED_INCOMPATIBLE_ABI"
    assert result.failure_record.selected_opaque_learnware_id == rank_one
    assert result.fallback_attempted is False
    assert result.abi_audit.audited_rank == 1
    assert result.abi_audit.fallback_policy == "RANK_ONE_ONLY_NO_FALLBACK"

    incomplete = replace(
        distance_run,
        rows=distance_run.rows[:-1],
        clamp_count=sum(row.result.clamped for row in distance_run.rows[:-1]),
        run_digest=None,
    )
    with pytest.raises(AnonymousMarketError, match="exactly 30"):
        audit_rank1_execution_abi(
            opaque_query_id="v03q-" + "2" * 32,
            request=request,
            distance_run=incomplete,
            selector_view=view,
            market=market,
            target_execution_abi=target_abi,
        )
