from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.acceptance import run_minimal_compute_acceptance
from policy_learnware_v0.v03.compute import query_to_source_distance
from policy_learnware_v0.v03.contracts import (
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    SemanticCacheKey,
    SemanticCacheRecord,
    SemanticTransform,
    SourceRepresentationIndex,
    V03ContractError,
    build_empirical_query_spec,
    build_source_reduced_spec,
)


def _d(label: str) -> str:
    return sha256_json({"core-test": label})


def _cache(label: str, shift: float = 0.0) -> SemanticCacheRecord:
    return SemanticCacheRecord(
        key=SemanticCacheKey(
            raw_dataset_digest=_d(f"raw:{label}"),
            ordered_episode_window_digest=_d(f"window:{label}"),
            canonical_view_digest=_d("view"),
            window_protocol_digest=_d("window-protocol"),
            normalizer_digest=_d("normalizer"),
            semantic_transform=SemanticTransform.raw_identity(),
            mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
        ),
        points=np.asarray([[shift], [shift + 0.1], [shift + 0.2], [shift + 0.3]]),
        episode_offsets=np.asarray([0, 2, 4]),
    )


def _source(label: str, shift: float, measurement: str, *, bandwidth: float = 0.9):
    cache = _cache(label, shift)
    return build_source_reduced_spec(
        cache,
        kernel_bandwidth=bandwidth,
        measurement_protocol_id=_d(measurement),
        probe_dataset_digest=cache.key.raw_dataset_digest,
        reducer_config=ReducerConfig(
            support_budget=4,
            support_steps=0,
            kmeans_steps=0,
            ridge=0.0,
            pinv_rcond=1.0e-12,
        ),
    )


def test_core_numeric_loop_is_deterministic() -> None:
    first = run_minimal_compute_acceptance()
    second = run_minimal_compute_acceptance()
    assert first.passed and all(first.checks.values())
    assert first.to_dict() == second.to_dict()


def test_only_executable_rkme_geometry_is_a_hard_contract() -> None:
    left = _source("left", 0.0, "measurement-a")
    right = _source("right", 1.0, "measurement-b")
    # Different measurement/reducer provenance does not change the RKME space.
    index = SourceRepresentationIndex(
        left.representation_protocol_id, {"left": left, "right": right}
    )
    query_cache = _cache("query", 0.05)
    query = build_empirical_query_spec(
        query_cache,
        kernel_bandwidth=0.9,
        measurement_protocol_id=_d("measurement-query"),
        probe_dataset_digest=query_cache.key.raw_dataset_digest,
    )
    assert query_to_source_distance(query, index.entries["left"]).squared_distance >= 0

    # A stale scalar is repaired from supports/beta/bandwidth.
    stale = replace(left.reduced_kme, rkme_norm2=123.0)
    repaired = replace(left, reduced_kme=stale, source_spec_digest=None)
    assert repaired.reduced_kme.rkme_norm2 == pytest.approx(
        left.reduced_kme.rkme_norm2, abs=1.0e-12
    )

    incompatible = _source("other-bandwidth", 0.0, "measurement-a", bandwidth=1.2)
    with pytest.raises(V03ContractError, match="bandwidth"):
        query_to_source_distance(query, incompatible)
