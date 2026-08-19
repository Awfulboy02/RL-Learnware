from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.io import atomic_write_bytes, atomic_write_json
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.rkme.reducer import ReducedRKME
from policy_learnware_v0.v01.artifacts import (
    V01ArtifactLayout,
    V01ArtifactLayoutError,
)
from policy_learnware_v0.v01.plans import build_pair_plan
from policy_learnware_v0.v01.taskspec import (
    WeightedSemanticSample,
    compute_taskspec_matrix,
)


def _sample(center: float) -> WeightedSemanticSample:
    return WeightedSemanticSample.from_points(
        np.asarray([[center], [center + 0.05]], dtype=np.float64),
        np.asarray([0, 1, 2], dtype=np.int64),
    )


def _taskspec_payload() -> dict[str, object]:
    variants = [
        {"task": "private", "factor": 0.75, "variant_id": "v01v-" + "a" * 20},
        {"task": "private", "factor": 1.0, "variant_id": "v01v-" + "b" * 20},
        {"task": "private", "factor": 2.0, "variant_id": "v01v-" + "c" * 20},
    ]
    plan = build_pair_plan(
        variants,
        banks=2,
        gate_prefix=1,
        routing_prefix=2,
        within_bank_pairs=((0, 1),),
    )
    centers = {
        variants[0]["variant_id"]: 0.3,
        variants[1]["variant_id"]: 0.0,
        variants[2]["variant_id"]: 0.7,
    }
    samples = {
        (variant_id, bank): _sample(center + bank * 0.001)
        for variant_id, center in centers.items()
        for bank in range(2)
    }
    kernel = GaussianKernel(1.0)
    supports = np.asarray([[0.0], [0.1]], dtype=np.float64)
    beta = np.asarray([0.5, 0.5], dtype=np.float64)
    source = ReducedRKME(
        supports=supports,
        beta=beta,
        bandwidth=kernel.bandwidth,
        rkme_norm2=float(beta @ kernel.gram(supports) @ beta),
        empirical_norm2=1.0,
        reduction_error=0.0,
    )
    return compute_taskspec_matrix(
        samples,
        plan,
        kernel=kernel,
        sources={"source-opaque": source},
        block_size=2,
    ).to_dict()


def test_oracle_poison_cannot_change_candidate_independent_taskspec_digest() -> None:
    before = _taskspec_payload()
    before_digest = sha256_json(before)

    # The selector-safe computation has no oracle argument or dependency.  A
    # missing or actively poisoned private oracle therefore cannot perturb it.
    oracle_private = {"shards": [{"mean_step_return": 1.0}]}
    oracle_private["shards"][0]["mean_step_return"] = -1.0e30
    del oracle_private["shards"]

    after = _taskspec_payload()
    assert after == before
    assert sha256_json(after) == before_digest


def test_immutable_resume_verifies_inputs_caches_and_aggregates(tmp_path: Path) -> None:
    layout = V01ArtifactLayout(tmp_path, "v01-resume-integration")
    measurement = layout.writer("measurement")
    oracle = layout.writer("oracle_private")

    input_payload = {"schema": "synthetic-input.v0", "digest": "1" * 64}
    input_digest = measurement.publish_json(layout.pair_plan, input_payload)
    assert measurement.publish_json(layout.pair_plan, input_payload, resume=True) == input_digest
    with pytest.raises(V01ArtifactLayoutError, match="resume content mismatch"):
        measurement.publish_json(
            layout.pair_plan,
            {"schema": "synthetic-input.v0", "digest": "2" * 64},
            resume=True,
        )

    cache_path = layout.semantic_cache("v01v-" + "a" * 20, 0)
    cache = {"points": np.asarray([[1.0], [2.0]], dtype=np.float32)}
    cache_digest = measurement.publish_npz(cache_path, cache)
    assert measurement.publish_npz(cache_path, cache, resume=True) == cache_digest
    atomic_write_bytes(cache_path, b"poisoned-cache", overwrite=True)
    with pytest.raises(V01ArtifactLayoutError, match="resume content mismatch"):
        measurement.publish_npz(cache_path, cache, resume=True)

    aggregate = {
        "schema": "policy-learnware.v01-oracle-aggregate-matrix.v0",
        "rows": [{"candidate_id": "candidate-a", "delta_return": -0.2}],
    }
    aggregate_digest = oracle.publish_json(layout.oracle_aggregates_json, aggregate)
    assert (
        oracle.publish_json(layout.oracle_aggregates_json, aggregate, resume=True)
        == aggregate_digest
    )
    poisoned = json.loads(json.dumps(aggregate))
    poisoned["rows"][0]["delta_return"] = 999.0
    atomic_write_json(layout.oracle_aggregates_json, poisoned, overwrite=True)
    with pytest.raises(V01ArtifactLayoutError, match="resume content mismatch"):
        oracle.publish_json(layout.oracle_aggregates_json, aggregate, resume=True)


def test_matrix_resume_rejects_payload_or_cache_drift(tmp_path: Path) -> None:
    layout = V01ArtifactLayout(tmp_path, "v01-matrix-resume")
    writer = layout.writer("measurement")
    payload = _taskspec_payload()
    payload_digest = writer.publish_json(layout.taskspec_matrix_axes, payload)
    assert writer.publish_json(layout.taskspec_matrix_axes, payload, resume=True) == payload_digest

    changed = json.loads(json.dumps(payload))
    changed["pair_rows"][0]["d_phi"] += 0.1
    with pytest.raises(V01ArtifactLayoutError, match="resume content mismatch"):
        writer.publish_json(layout.taskspec_matrix_axes, changed, resume=True)
