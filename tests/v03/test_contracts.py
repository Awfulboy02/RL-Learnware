from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.rkme.reducer import ReducedRKME, ReducerConfig
from policy_learnware_v0.v02.schemas import EnvironmentSpec
from policy_learnware_v0.v03 import contracts
from policy_learnware_v0.v03.contracts import (
    EMPIRICAL_QUERY_MARKER,
    EPISODE_BALANCED_WEIGHTING_DIGEST,
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    GAUSSIAN_KERNEL_EVALUATOR_DIGEST,
    RankingKey,
    SemanticCacheKey,
    SemanticCacheRecord,
    SourceReducedSpec,
    SpecKey,
    V03ContractError,
    build_empirical_query_spec,
    build_source_reduced_spec,
    derive_reducer_digest,
    reduce_query_spec,
    source_from_v02_environment_spec,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _key(raw: str = "raw", *, checkpoint: str = "checkpoint") -> SemanticCacheKey:
    return SemanticCacheKey(
        raw_dataset_digest=_d(raw),
        ordered_episode_window_digest=_d("ordered-window"),
        canonical_view_digest=_d("view"),
        window_protocol_digest=_d("window-protocol"),
        normalizer_digest=_d("normalizer"),
        encoder_implementation_digest=_d("encoder"),
        checkpoint_digest=_d(checkpoint),
        semantic_output_protocol_digest=_d("semantic-output"),
        mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    )


def _cache(raw: str = "raw") -> SemanticCacheRecord:
    return SemanticCacheRecord(
        key=_key(raw),
        points=np.asarray(
            [[0.0], [0.1], [1.0], [1.1], [2.0], [2.1]], dtype=np.float64
        ),
        episode_offsets=np.asarray([0, 2, 4, 6], dtype=np.int64),
    )


def _query(cache: SemanticCacheRecord, *, episode_count: int = 2):
    return build_empirical_query_spec(
        cache,
        kernel_bandwidth=1.0,
        measurement_protocol_id=_d("measurement"),
        probe_dataset_digest=cache.key.raw_dataset_digest,
        episode_count=episode_count,
        block_size=2,
    )


def test_semantic_cache_key_roundtrip_and_content_binding() -> None:
    key = _key()
    assert SemanticCacheKey.from_dict(key.to_dict()) == key
    changed_checkpoint = _key(checkpoint="checkpoint-2")
    assert changed_checkpoint.semantic_cache_key_digest != key.semantic_cache_key_digest

    first = _cache()
    second = SemanticCacheRecord(
        key=first.key,
        points=np.asarray(first.points) + 0.01,
        episode_offsets=first.episode_offsets,
    )
    assert first.key.semantic_cache_key_digest == second.key.semantic_cache_key_digest
    assert first.semantic_cache_digest != second.semantic_cache_digest
    tampered = key.to_dict()
    tampered["checkpoint_digest"] = _d("tampered")
    with pytest.raises(V03ContractError, match="does not match"):
        SemanticCacheKey.from_dict(tampered)


def test_semantic_cache_freezes_mathematical_dtype() -> None:
    key = _key()
    with pytest.raises(V03ContractError, match="float64 dtype"):
        SemanticCacheRecord(
            key=key,
            points=np.asarray([[0.0], [1.0]], dtype=np.float32),
            episode_offsets=np.asarray([0, 2]),
        )
    mislabeled = replace(
        key,
        mathematical_dtype_digest=_d("float32-math"),
        semantic_cache_key_digest=None,
    )
    with pytest.raises(V03ContractError, match="dtype digest"):
        SemanticCacheRecord(
            key=mislabeled,
            points=np.asarray([[0.0], [1.0]], dtype=np.float64),
            episode_offsets=np.asarray([0, 2]),
        )


def test_episode_prefix_derives_distinct_slice_and_reweights_query() -> None:
    cache = _cache()
    first = cache.episode_prefix(1)
    second = cache.episode_prefix(2)
    assert first.exact_slice_or_prefix_digest != second.exact_slice_or_prefix_digest
    assert first.points.shape == (2, 1)
    query = _query(cache, episode_count=2)
    np.testing.assert_array_equal(query.empirical_kme.episode_offsets, [0, 2, 4])
    np.testing.assert_allclose(query.empirical_kme.weights, np.full(4, 0.25))
    assert not query.empirical_kme.points.flags.writeable


def test_primary_query_never_reduces_but_source_reduces_once() -> None:
    cache = _cache()
    with patch.object(contracts, "reduce_kme", side_effect=AssertionError("unexpected")):
        query = _query(cache)
    assert query.spec_role == "QUERY_EMPIRICAL"

    real_reduce = contracts.reduce_kme
    reducer = ReducerConfig(
        support_budget=4,
        support_steps=0,
        kmeans_steps=0,
        ridge=0.0,
        pinv_rcond=1.0e-12,
    )
    with patch.object(contracts, "reduce_kme", wraps=real_reduce) as spy:
        build_source_reduced_spec(
            cache,
            kernel_bandwidth=1.0,
            measurement_protocol_id=_d("measurement"),
            probe_dataset_digest=cache.key.raw_dataset_digest,
            reducer_config=reducer,
            episode_count=2,
        )
    assert spy.call_count == 1


def test_query_modes_have_separate_spec_and_ranking_keys() -> None:
    cache = _cache()
    empirical = _query(cache)
    reduced = reduce_query_spec(
        empirical,
        reducer_config=ReducerConfig(
            support_budget=4,
            support_steps=0,
            kmeans_steps=0,
            ridge=0.0,
            pinv_rcond=1.0e-12,
        ),
    )
    assert empirical.query_protocol_id != reduced.query_protocol_id
    assert empirical.spec_key.spec_key_digest != reduced.spec_key.spec_key_digest
    common = {
        "representation_index_digest": _d("index"),
        "selector_digest": _d("selector"),
        "tie_break_digest": _d("tie"),
    }
    empirical_ranking = RankingKey(
        query_spec_digest=empirical.query_spec_digest, **common
    )
    reduced_ranking = RankingKey(query_spec_digest=reduced.query_spec_digest, **common)
    assert empirical_ranking.ranking_key_digest != reduced_ranking.ranking_key_digest


def test_reducer_digest_is_derived_and_mixed_index_reducers_are_rejected() -> None:
    cache_a = _cache("raw-a")
    cache_b = _cache("raw-b")
    config_a = ReducerConfig(
        support_budget=2, support_steps=0, kmeans_steps=0, ridge=0.0
    )
    config_b = ReducerConfig(
        support_budget=3, support_steps=0, kmeans_steps=0, ridge=0.0
    )
    assert derive_reducer_digest(config_a) != derive_reducer_digest(config_b)

    def source(cache: SemanticCacheRecord, config: ReducerConfig):
        return build_source_reduced_spec(
            cache,
            kernel_bandwidth=1.0,
            measurement_protocol_id=_d("measurement"),
            probe_dataset_digest=cache.key.raw_dataset_digest,
            reducer_config=config,
        )

    source_a = source(cache_a, config_a)
    source_b = source(cache_b, config_b)
    assert (
        source_a.spec_key.reducer_digest_or_empirical_query_marker
        == derive_reducer_digest(config_a)
    )
    with pytest.raises(V03ContractError, match="reducer_digest"):
        contracts.SourceRepresentationIndex(
            "market",
            cache_a.key.semantic_output_protocol_digest,
            {"a": source_a, "b": source_b},
        )


def test_signed_reduced_weights_are_preserved_and_norm_checked() -> None:
    cache = _cache()
    supports = np.asarray([[0.0], [1.0]])
    beta = np.asarray([1.2, -0.2])
    norm = float(beta @ GaussianKernel(1.0).gram(supports) @ beta)
    reduced = ReducedRKME(
        supports=supports,
        beta=beta,
        bandwidth=1.0,
        rkme_norm2=norm,
        empirical_norm2=1.0,
        reduction_error=0.1,
        protocol_id=cache.key.semantic_output_protocol_digest,
        source_dataset_digest=cache.key.raw_dataset_digest,
    )
    key = SpecKey(
        semantic_cache_digest=cache.semantic_cache_digest,
        exact_slice_or_prefix_digest=cache.episode_prefix(2).exact_slice_or_prefix_digest,
        sample_weighting_digest=EPISODE_BALANCED_WEIGHTING_DIGEST,
        spec_role="SOURCE_REDUCED",
        kernel_evaluator_digest=GAUSSIAN_KERNEL_EVALUATOR_DIGEST,
        kernel_bandwidth=1.0,
        reducer_digest_or_empirical_query_marker=_d("reducer"),
    )
    source = SourceReducedSpec(
        reduced,
        cache.key,
        cache.semantic_cache_digest,
        key,
        _d("measurement"),
        cache.key.canonical_view_digest,
        cache.key.raw_dataset_digest,
    )
    np.testing.assert_array_equal(source.reduced_kme.beta, beta)

    broken = replace(reduced, rkme_norm2=norm + 0.2)
    with pytest.raises(V03ContractError, match="rkme_norm2"):
        SourceReducedSpec(
            broken,
            cache.key,
            cache.semantic_cache_digest,
            key,
            _d("measurement"),
            cache.key.canonical_view_digest,
            cache.key.raw_dataset_digest,
        )


def test_v02_adapter_rejects_norm_inconsistent_legacy_spec() -> None:
    cache = _cache()
    supports = np.asarray([[0.0], [1.0]])
    beta = np.asarray([0.5, 0.5])
    actual_norm = float(beta @ GaussianKernel(1.0).gram(supports) @ beta)
    legacy = EnvironmentSpec(
        supports=supports,
        beta=beta,
        empirical_norm2=1.0,
        rkme_norm2=actual_norm + 0.1,
        reconstruction_error=0.1,
        reducer_digest=_d("reducer"),
        support_budget=2,
        latent_dim=1,
        representation_protocol_id=cache.key.semantic_output_protocol_digest,
        measurement_protocol_id=_d("measurement"),
        canonical_view_digest=cache.key.canonical_view_digest,
        kernel_bandwidth=1.0,
        probe_dataset_digest=cache.key.raw_dataset_digest,
    )
    with pytest.raises(V03ContractError, match="norm-inconsistent"):
        source_from_v02_environment_spec(
            legacy,
            semantic_cache=cache,
        )


def test_v02_adapter_recomputes_full_cache_lineage() -> None:
    cache = _cache()
    supports = np.asarray(cache.points)
    beta = np.full(supports.shape[0], 1.0 / supports.shape[0])
    norm = float(beta @ GaussianKernel(1.0).gram(supports) @ beta)
    legacy = EnvironmentSpec(
        supports=supports,
        beta=beta,
        empirical_norm2=norm,
        rkme_norm2=norm,
        reconstruction_error=0.0,
        reducer_digest=_d("legacy-reducer"),
        support_budget=supports.shape[0],
        latent_dim=1,
        representation_protocol_id=cache.key.semantic_output_protocol_digest,
        measurement_protocol_id=_d("measurement"),
        canonical_view_digest=cache.key.canonical_view_digest,
        kernel_bandwidth=1.0,
        probe_dataset_digest=cache.key.raw_dataset_digest,
    )
    adapted = source_from_v02_environment_spec(legacy, semantic_cache=cache)
    assert adapted.legacy_environment_spec_digest == legacy.environment_spec_digest

    unrelated_cache = SemanticCacheRecord(
        key=cache.key,
        points=np.asarray(cache.points) + 10.0,
        episode_offsets=cache.episode_offsets,
    )
    with pytest.raises(V03ContractError, match="semantic cache"):
        source_from_v02_environment_spec(legacy, semantic_cache=unrelated_cache)


def test_empirical_marker_cannot_be_used_for_reduced_role() -> None:
    cache = _cache()
    with pytest.raises(V03ContractError, match="reduced specs"):
        SpecKey(
            semantic_cache_digest=cache.semantic_cache_digest,
            exact_slice_or_prefix_digest=cache.episode_prefix(1).exact_slice_or_prefix_digest,
            sample_weighting_digest=_d("weights"),
            spec_role="SOURCE_REDUCED",
            kernel_evaluator_digest=_d("kernel"),
            kernel_bandwidth=1.0,
            reducer_digest_or_empirical_query_marker=EMPIRICAL_QUERY_MARKER,
        )
