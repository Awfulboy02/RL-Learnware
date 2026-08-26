"""Deterministic CPU smoke acceptance for the v0.3 numeric contract slice."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..hashing import canonical_json, sha256_json
from ..rkme.reducer import ReducerConfig
from .compute import (
    JointDistanceRequest,
    evaluate_minimal_numeric_gate,
    recompute_joint_distance_run,
    run_joint_distance_stage,
    tie_break_digest,
)
from .contracts import (
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    RankingKey,
    SemanticCacheKey,
    SemanticCacheRecord,
    SourceRepresentationIndex,
    build_empirical_query_spec,
    build_source_reduced_spec,
    reduce_query_spec,
)


ACCEPTANCE_SCHEMA = "policy-learnware.v03-minimal-compute-acceptance.v0"


def _d(label: str) -> str:
    return sha256_json(
        {"schema": "policy-learnware.v03-acceptance-fixture-domain.v0", "label": label}
    )


def _cache(label: str, points: np.ndarray, offsets: np.ndarray) -> SemanticCacheRecord:
    view = _d("canonical-view")
    return SemanticCacheRecord(
        key=SemanticCacheKey(
            raw_dataset_digest=_d(f"raw:{label}"),
            ordered_episode_window_digest=_d(f"episode-window:{label}"),
            canonical_view_digest=view,
            window_protocol_digest=_d("window-protocol"),
            normalizer_digest=_d("normalizer"),
            encoder_implementation_digest=_d("fake-encoder-implementation"),
            checkpoint_digest=_d("fake-encoder-checkpoint"),
            semantic_output_protocol_digest=_d("representation-protocol"),
            mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
        ),
        points=points,
        episode_offsets=offsets,
    )


@dataclass(frozen=True)
class MinimalComputeAcceptanceReport:
    checks: Mapping[str, bool]
    empirical_run_digest: str
    reduced_run_digest: str
    numeric_gate_evidence_digest: str
    report_digest: str | None = None
    schema: str = ACCEPTANCE_SCHEMA

    def __post_init__(self) -> None:
        checks = dict(sorted(self.checks.items()))
        if not checks or any(type(value) is not bool for value in checks.values()):
            raise ValueError("acceptance checks must be non-empty booleans")
        object.__setattr__(self, "checks", MappingProxyType(checks))
        expected = sha256_json(self._payload_without_digest())
        if self.report_digest is None:
            object.__setattr__(self, "report_digest", expected)
        elif self.report_digest != expected:
            raise ValueError("acceptance report digest is inconsistent")

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checks": dict(self.checks),
            "passed": self.passed,
            "empirical_run_digest": self.empirical_run_digest,
            "reduced_run_digest": self.reduced_run_digest,
            "numeric_gate_evidence_digest": self.numeric_gate_evidence_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "report_digest": self.report_digest}


def run_minimal_compute_acceptance() -> MinimalComputeAcceptanceReport:
    """Exercise both explicit query modes over one deterministic public index."""

    offsets = np.asarray([0, 2, 4], dtype=np.int64)
    source_a_cache = _cache(
        "source-a",
        np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]),
        offsets,
    )
    source_b_cache = _cache(
        "source-b",
        np.asarray([[3.0, 3.0], [3.1, 3.0], [3.0, 3.1], [3.1, 3.1]]),
        offsets,
    )
    query_cache = _cache(
        "query",
        np.asarray(
            [
                [0.04, 0.02],
                [0.08, 0.01],
                [0.03, 0.07],
                [0.09, 0.08],
                [2.9, 3.0],
                [3.0, 2.9],
            ]
        ),
        np.asarray([0, 2, 4, 6], dtype=np.int64),
    )
    representation = _d("representation-protocol")
    measurement = _d("measurement-protocol")
    reducer = ReducerConfig(
        support_budget=4,
        support_steps=0,
        kmeans_steps=0,
        ridge=0.0,
        pinv_rcond=1.0e-12,
    )
    common = {
        "kernel_bandwidth": 1.0,
        "measurement_protocol_id": measurement,
        "reducer_config": reducer,
    }
    source_a = build_source_reduced_spec(
        source_a_cache,
        probe_dataset_digest=source_a_cache.key.raw_dataset_digest,
        **common,
    )
    source_b = build_source_reduced_spec(
        source_b_cache,
        probe_dataset_digest=source_b_cache.key.raw_dataset_digest,
        **common,
    )
    index = SourceRepresentationIndex(
        policy_market_id="v03-smoke-market",
        representation_protocol_id=representation,
        entries={"source-a": source_a, "source-b": source_b},
    )
    empirical_query = build_empirical_query_spec(
        query_cache,
        kernel_bandwidth=1.0,
        measurement_protocol_id=measurement,
        probe_dataset_digest=query_cache.key.raw_dataset_digest,
        episode_count=2,
    )
    reduced_query = reduce_query_spec(
        empirical_query,
        reducer_config=reducer,
    )
    tokens = {"source-a": _d("tie-a"), "source-b": _d("tie-b")}
    selector_digest = _d("distance-only-smoke-selector")

    def request(query: Any) -> JointDistanceRequest:
        ranking_key = RankingKey(
            query_spec_digest=query.query_spec_digest,
            representation_index_digest=index.representation_index_digest,
            selector_digest=selector_digest,
            tie_break_digest=tie_break_digest(tokens),
        )
        return JointDistanceRequest(query, index, ranking_key, tokens, block_size=2)

    empirical_request = request(empirical_query)
    reduced_request = request(reduced_query)
    empirical_run = run_joint_distance_stage(empirical_request)
    reduced_run = run_joint_distance_stage(reduced_request)
    recompute = recompute_joint_distance_run(empirical_request, empirical_run)
    gate = evaluate_minimal_numeric_gate(empirical_request, empirical_run, recompute)
    empirical_by_id = {row.opaque_id: row.result for row in empirical_run.rows}
    reduced_by_id = {row.opaque_id: row.result for row in reduced_run.rows}
    checks = {
        "empirical_primary_role": empirical_query.spec_role == "QUERY_EMPIRICAL",
        "explicit_reduced_role": reduced_query.spec_role == "QUERY_REDUCED",
        "query_protocols_separated": (
            empirical_query.query_protocol_id != reduced_query.query_protocol_id
        ),
        "spec_keys_separated": (
            empirical_query.spec_key.spec_key_digest
            != reduced_query.spec_key.spec_key_digest
        ),
        "ranking_keys_separated": (
            empirical_request.ranking_key.ranking_key_digest
            != reduced_request.ranking_key.ranking_key_digest
        ),
        "expected_nearest_source": empirical_run.rows[0].opaque_id == "source-a",
        "lossless_mode_distance_parity": all(
            abs(empirical_by_id[source_id].squared_distance - reduced_by_id[source_id].squared_distance)
            <= 1.0e-7
            for source_id in empirical_by_id
        ),
        "independent_recompute": recompute.matched,
        "minimal_numeric_gate": gate.passed,
        "prefix_slice_is_episode_bound": (
            empirical_query.empirical_kme.episode_count == 2
            and empirical_query.empirical_kme.transition_count == 4
        ),
    }
    return MinimalComputeAcceptanceReport(
        checks=checks,
        empirical_run_digest=str(empirical_run.run_digest),
        reduced_run_digest=str(reduced_run.run_digest),
        numeric_gate_evidence_digest=str(gate.evidence_digest),
    )


def main() -> int:
    report = run_minimal_compute_acceptance()
    print(canonical_json(report.to_dict()))
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["MinimalComputeAcceptanceReport", "run_minimal_compute_acceptance"]
