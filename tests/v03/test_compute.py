from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.compute import (
    JointDistanceRequest,
    query_to_source_distance,
    recompute_joint_distance_run,
    run_joint_distance_stage,
    tie_break_digest,
)
from policy_learnware_v0.v03.contracts import (
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    RankingKey,
    SemanticCacheKey,
    SemanticCacheRecord,
    SourceRepresentationIndex,
    V03ContractError,
    build_empirical_query_spec,
    build_source_reduced_spec,
    reduce_query_spec,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _cache(label: str, points: np.ndarray) -> SemanticCacheRecord:
    return SemanticCacheRecord(
        SemanticCacheKey(
            raw_dataset_digest=_d(f"raw:{label}"),
            ordered_episode_window_digest=_d(f"window:{label}"),
            canonical_view_digest=_d("view"),
            window_protocol_digest=_d("window-protocol"),
            normalizer_digest=_d("normalizer"),
            encoder_implementation_digest=_d("encoder"),
            checkpoint_digest=_d("checkpoint"),
            semantic_output_protocol_digest=_d("semantic-output"),
            mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
        ),
        points,
        np.asarray([0, 2, 4], dtype=np.int64),
    )


@pytest.fixture
def numeric_fixture():
    source_a_cache = _cache("a", np.asarray([[0.0], [0.1], [0.2], [0.3]]))
    source_b_cache = _cache("b", np.asarray([[2.0], [2.1], [2.2], [2.3]]))
    query_cache = _cache("q", np.asarray([[0.05], [0.15], [0.1], [0.2]]))
    representation = source_a_cache.key.semantic_output_protocol_digest
    measurement = _d("measurement")
    reducer = ReducerConfig(
        support_budget=4,
        support_steps=0,
        kmeans_steps=0,
        ridge=0.0,
        pinv_rcond=1.0e-12,
    )

    def source(cache: SemanticCacheRecord):
        return build_source_reduced_spec(
            cache,
            kernel_bandwidth=0.9,
            measurement_protocol_id=measurement,
            probe_dataset_digest=cache.key.raw_dataset_digest,
            reducer_config=reducer,
        )

    sources = {"a": source(source_a_cache), "b": source(source_b_cache)}
    query = build_empirical_query_spec(
        query_cache,
        kernel_bandwidth=0.9,
        measurement_protocol_id=measurement,
        probe_dataset_digest=query_cache.key.raw_dataset_digest,
    )
    return query, sources, reducer, representation


def test_empirical_distance_matches_dense_formula_and_block_sizes(numeric_fixture) -> None:
    query, sources, _reducer, _representation = numeric_fixture
    source = sources["a"]
    kernel = GaussianKernel(query.kernel_bandwidth)
    cross = float(
        query.empirical_kme.weights
        @ kernel.gram(query.empirical_kme.points, source.reduced_kme.supports)
        @ source.reduced_kme.beta
    )
    expected = float(
        query.empirical_kme.norm2 - 2.0 * cross + source.reduced_kme.rkme_norm2
    )
    results = [
        query_to_source_distance(query, source, distance_form="mmd2", block_size=size)
        for size in (1, 2, 32)
    ]
    assert all(result.squared_distance == pytest.approx(expected, abs=1.0e-12) for result in results)
    assert results[0].squared_distance == pytest.approx(results[-1].squared_distance, abs=1.0e-12)


def test_lossless_reduced_query_matches_empirical_ranking(numeric_fixture) -> None:
    query, sources, reducer, representation = numeric_fixture
    reduced = reduce_query_spec(query, reducer_config=reducer)
    for source in sources.values():
        empirical_distance = query_to_source_distance(query, source)
        reduced_distance = query_to_source_distance(reduced, source)
        assert empirical_distance.squared_distance == pytest.approx(
            reduced_distance.squared_distance, abs=1.0e-7
        )

    index = SourceRepresentationIndex("market", representation, sources)
    tokens = {"a": _d("tie-a"), "b": _d("tie-b")}

    def request(item):
        ranking = RankingKey(
            item.query_spec_digest,
            index.representation_index_digest,
            _d("selector"),
            tie_break_digest(tokens),
        )
        return JointDistanceRequest(item, index, ranking, tokens, block_size=2)

    empirical_request = request(query)
    reduced_request = request(reduced)
    empirical_run = run_joint_distance_stage(empirical_request)
    reduced_run = run_joint_distance_stage(reduced_request)
    assert [row.opaque_id for row in empirical_run.rows] == ["a", "b"]
    assert [row.opaque_id for row in reduced_run.rows] == ["a", "b"]
    assert empirical_request.ranking_key.ranking_key_digest != reduced_request.ranking_key.ranking_key_digest
    recompute = recompute_joint_distance_run(empirical_request, empirical_run)
    assert recompute.matched


def test_binding_mismatch_and_negative_tolerance_fail_closed(numeric_fixture) -> None:
    query, sources, _reducer, _representation = numeric_fixture
    source = sources["a"]
    with pytest.raises(V03ContractError, match="negative_tolerance"):
        query_to_source_distance(query, source, negative_tolerance=-1.0)
    incompatible = replace(
        source,
        measurement_protocol_id=_d("other-measurement"),
        source_spec_digest=None,
    )
    with pytest.raises(V03ContractError, match="measurement_protocol_id"):
        query_to_source_distance(query, incompatible)


def test_ranking_key_rejects_wrong_index_and_tie_map(numeric_fixture) -> None:
    query, sources, _reducer, representation = numeric_fixture
    index = SourceRepresentationIndex("market", representation, sources)
    tokens = {"a": _d("tie-a"), "b": _d("tie-b")}
    ranking = RankingKey(
        query.query_spec_digest,
        _d("wrong-index"),
        _d("selector"),
        tie_break_digest(tokens),
    )
    with pytest.raises(V03ContractError, match="another source index"):
        JointDistanceRequest(query, index, ranking, tokens)

    correct = RankingKey(
        query.query_spec_digest,
        index.representation_index_digest,
        _d("selector"),
        tie_break_digest(tokens),
    )
    with pytest.raises(V03ContractError, match="coverage"):
        JointDistanceRequest(query, index, correct, {"a": _d("tie-a")})
