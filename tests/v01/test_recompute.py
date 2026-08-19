from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from policy_learnware_v0.hashing import canonical_json_bytes, sha256_file, sha256_ndarrays
from policy_learnware_v0.probe.dataset import EpisodeDataset
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.rkme.reducer import ReducedRKME
from policy_learnware_v0.v01.analysis import ContextBinding
from policy_learnware_v0.v01.config import load_v01_experiment_config
from policy_learnware_v0.v01.plans import build_pair_plan
from policy_learnware_v0.v01.recompute import (
    AUDIT_PLAN_SCHEMA,
    RecomputeContractError,
    load_verified_measurement_samples,
    rebuild_taskspec_aggregation,
    recompute_raw_numeric_subset,
    taskspec_primitive_digest,
)
from policy_learnware_v0.v01.schemas import VariantDatasetManifest
from policy_learnware_v0.v01.taskspec import WeightedSemanticSample, compute_taskspec_matrix


def _matrix_fixture():
    config = load_v01_experiment_config(
        Path(__file__).parents[2] / "configs" / "v01_smoke.yaml"
    )
    variants = []
    bindings = []
    for index, factor in enumerate(config.shift.diagnostic_grid):
        variant_id = f"v01v-{index + 1:020x}"
        variants.append(
            {"task": "WalkerWalk", "factor": factor, "variant_id": variant_id}
        )
        bindings.append(
            ContextBinding(
                task="WalkerWalk",
                factor=factor,
                d_theta=abs(math.log(factor)),
                variant_id=variant_id,
                private_context_id=f"v01c-{index + 1:032x}",
            )
        )
    plan = build_pair_plan(
        variants,
        banks=config.probe.banks,
        gate_prefix=config.probe.gate_b_unreduced_prefix,
        routing_prefix=config.probe.max_episodes_per_bank,
        within_bank_pairs=config.probe.sparse_within_bank_pairs,
    )
    offsets = np.arange(config.probe.max_episodes_per_bank + 1, dtype=np.int64)
    samples = {}
    for variant_index, variant in enumerate(variants):
        for bank in range(config.probe.banks):
            points = (
                np.arange(config.probe.max_episodes_per_bank, dtype=np.float64)[:, None]
                / 20.0
                + 0.1 * variant_index
                + 0.01 * bank
            )
            samples[(variant["variant_id"], bank)] = WeightedSemanticSample.from_points(
                points, offsets
            )
    kernel = GaussianKernel(1.0)
    sources = {
        f"source-{index}": ReducedRKME(
            supports=np.asarray([[0.2 * index]], dtype=np.float64),
            beta=np.asarray([1.0]),
            bandwidth=1.0,
            rkme_norm2=1.0,
            empirical_norm2=1.0,
            reduction_error=0.0,
        )
        for index in range(6)
    }
    matrix = compute_taskspec_matrix(
        samples, plan, kernel=kernel, sources=sources, block_size=8
    ).to_dict()
    nominal = next(item["variant_id"] for item in variants if item["factor"] == 1.0)
    factor_two = next(item["variant_id"] for item in variants if item["factor"] == 2.0)
    audit_plan = {
        "schema": AUDIT_PLAN_SCHEMA,
        "oracle_reaggregation": "all_episode_rows",
        "taskspec_digest_coverage": "all_datasets_and_semantic_caches",
        "taskspec_aggregation_coverage": "all_frozen_pairs",
        "raw_numeric_subset": {
            "within": [
                {
                    "task_private": "WalkerWalk",
                    "left_variant_id": nominal,
                    "left_bank": 0,
                    "right_variant_id": nominal,
                    "right_bank": 1,
                    "prefix": config.probe.gate_b_unreduced_prefix,
                }
            ],
            "between": [
                {
                    "task_private": "WalkerWalk",
                    "left_variant_id": nominal,
                    "left_bank": 0,
                    "right_variant_id": factor_two,
                    "right_bank": 0,
                    "prefix": config.probe.gate_b_unreduced_prefix,
                }
            ],
            "routing": [
                {
                    "task_private": "WalkerWalk",
                    "variant_id": factor_two,
                    "bank": 0,
                    "prefix": config.probe.max_episodes_per_bank,
                }
            ],
            "selection_time": "before_results",
        },
    }
    return config, tuple(bindings), plan, samples, kernel, sources, matrix, audit_plan


def test_full_aggregation_rebuild_and_frozen_raw_subset() -> None:
    config, bindings, plan, samples, kernel, sources, matrix, audit_plan = _matrix_fixture()
    primitive_digest = taskspec_primitive_digest(plan, matrix)
    aggregation = rebuild_taskspec_aggregation(
        plan, matrix, trusted_primitive_digest=primitive_digest
    )
    raw = recompute_raw_numeric_subset(
        audit_plan,
        bindings=bindings,
        config=config,
        samples=samples,
        stored_matrix=matrix,
        kernel=kernel,
        sources=sources,
        block_size=8,
    )
    assert aggregation.passed
    assert aggregation.pair_count == len(matrix["pair_rows"])
    assert raw.passed
    assert raw.task_results[0]["self_term_count"] == 3
    assert raw.task_results[0]["cross_term_count"] == 2
    assert raw.task_results[0]["source_count"] == 6


def test_recompute_rejects_poisoned_aggregate_and_nonaudited_primitive() -> None:
    _, _, plan, _, _, _, matrix, _ = _matrix_fixture()
    trusted = taskspec_primitive_digest(plan, matrix)
    poisoned_aggregate = copy.deepcopy(matrix)
    poisoned_aggregate["pair_rows"][0]["d_phi"] += 0.01
    with pytest.raises(RecomputeContractError, match="d_phi mismatch"):
        rebuild_taskspec_aggregation(
            plan, poisoned_aggregate, trusted_primitive_digest=trusted
        )

    poisoned_primitive = copy.deepcopy(matrix)
    # This is deliberately outside the frozen factor-2/bank-0 raw subset.  The
    # immutable full primitive digest must still detect it.
    poisoned_primitive["routing_rows"][0]["ranking"] = [
        {**item, "routing_score": item["routing_score"] + 0.125}
        for item in poisoned_primitive["routing_rows"][0]["ranking"]
    ]
    with pytest.raises(RecomputeContractError, match="primitive digest binding"):
        rebuild_taskspec_aggregation(
            plan, poisoned_primitive, trusted_primitive_digest=trusted
        )


def test_raw_subset_rejects_audited_primitive_and_post_result_selection() -> None:
    config, bindings, _, samples, kernel, sources, matrix, audit_plan = _matrix_fixture()
    factor_two = audit_plan["raw_numeric_subset"]["routing"][0]["variant_id"]
    poisoned = copy.deepcopy(matrix)
    row = next(
        item
        for item in poisoned["routing_rows"]
        if item["variant_id"] == factor_two and item["bank"] == 0
    )
    row["ranking"][0]["routing_score"] += 0.25
    with pytest.raises(RecomputeContractError, match=r"routing\["):
        recompute_raw_numeric_subset(
            audit_plan,
            bindings=bindings,
            config=config,
            samples=samples,
            stored_matrix=poisoned,
            kernel=kernel,
            sources=sources,
            block_size=8,
        )

    post_result = copy.deepcopy(audit_plan)
    post_result["raw_numeric_subset"]["selection_time"] = "after_results"
    with pytest.raises(RecomputeContractError, match="before results"):
        recompute_raw_numeric_subset(
            post_result,
            bindings=bindings,
            config=config,
            samples=samples,
            stored_matrix=matrix,
            kernel=kernel,
            sources=sources,
            block_size=8,
        )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _dataset(seed: int) -> EpisodeDataset:
    return EpisodeDataset(
        observation=np.asarray([[0.0], [1.0]], dtype=np.float32),
        action=np.asarray([[0.1], [0.2]], dtype=np.float32),
        reward=np.asarray([0.0, 1.0], dtype=np.float32),
        next_observation=np.asarray([[0.5], [1.5]], dtype=np.float32),
        terminated=np.asarray([True, True]),
        truncated=np.asarray([False, False]),
        episode_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        reset_seeds=np.asarray([seed, seed + 1], dtype=np.int64),
        probe_seeds=np.asarray([seed + 10, seed + 11], dtype=np.int64),
    )


def test_dataset_semantic_chain_detects_cache_poison_even_if_sidecar_is_rewritten(
    tmp_path: Path,
) -> None:
    measurement = tmp_path / "experiment" / "measurement"
    variant = "v01v-00000000000000000001"
    plan = build_pair_plan(
        [{"task": "WalkerWalk", "factor": 1.0, "variant_id": variant}],
        banks=2,
        gate_prefix=1,
        routing_prefix=2,
        within_bank_pairs=((0, 1),),
    )
    contract = {
        "schema": "policy-learnware.v01-measurement-contract.v0",
        "measurement_protocol_id": "1" * 64,
        "base_protocol_id": "2" * 64,
        "probe_banks": 2,
        "episodes_per_bank": 2,
        "prefix_grid": [1, 2],
        "gate_prefix": 1,
        "pair_plan_digest": plan["plan_digest"],
        "variant_ids": [variant],
        "schema_view_digests": {variant: "3" * 64},
        "visibility": "opaque_variant_only_no_context_policy_or_outcome",
    }
    _write_json(measurement / "measurement_contract.json", contract)
    trusted = {}
    for bank in range(2):
        dataset = _dataset(100 * bank)
        data_path = measurement / "datasets" / variant / f"bank_{bank:03d}" / "dataset.npz"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_npz(data_path)
        manifest = VariantDatasetManifest(
            variant_id=variant,
            bank=bank,
            episode_count=dataset.episode_count,
            transition_count=dataset.transition_count,
            reset_seeds=tuple(int(value) for value in dataset.reset_seeds),
            probe_seeds=tuple(int(value) for value in dataset.probe_seeds),
            dataset_digest=dataset.digest,
            base_protocol_id="2" * 64,
            measurement_contract_digest=sha256_file(
                measurement / "measurement_contract.json"
            ),
            measurement_schema_view_digest="3" * 64,
        )
        _write_json(data_path.parent / "manifest.json", manifest.to_dict())
        sample = WeightedSemanticSample.from_points(
            np.asarray([[bank], [bank + 0.5]], dtype=np.float64),
            np.asarray([0, 1, 2], dtype=np.int64),
        )
        cache = measurement / "semantic_cache" / variant / f"bank_{bank:03d}.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **sample.cache_arrays())
        trusted[f"{variant}/bank_{bank:03d}"] = sha256_ndarrays(sample.cache_arrays())
        cache_manifest = {
            "schema": "policy-learnware.v01-semantic-cache-manifest.v0",
            "variant_id": variant,
            "bank": bank,
            "dataset_digest": dataset.digest,
            "measurement_schema_view_digest": "3" * 64,
            "base_binding_digest": "4" * 64,
            "normalization_sha256": "5" * 64,
            "encoder_checkpoint_sha256": "6" * 64,
            "encoder_config_sha256": "7" * 64,
            "cache_sha256": sha256_file(cache),
        }
        _write_json(cache.with_suffix(".json"), cache_manifest)

    result = load_verified_measurement_samples(
        measurement,
        contract,
        plan,
        trusted_semantic_cache_digests=trusted,
        verified_base_binding_digest="4" * 64,
        verified_base_asset_digests={
            "normalization": "5" * 64,
            "encoder_checkpoint": "6" * 64,
            "encoder_config": "7" * 64,
        },
    )
    assert result.passed

    poisoned_cache = measurement / "semantic_cache" / variant / "bank_001.npz"
    poisoned_sample = WeightedSemanticSample.from_points(
        np.asarray([[99.0], [99.5]], dtype=np.float64),
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    np.savez(poisoned_cache, **poisoned_sample.cache_arrays())
    sidecar_path = poisoned_cache.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["cache_sha256"] = sha256_file(poisoned_cache)
    _write_json(sidecar_path, sidecar)
    with pytest.raises(RecomputeContractError, match="trusted digest"):
        load_verified_measurement_samples(
            measurement,
            contract,
            plan,
            trusted_semantic_cache_digests=trusted,
            verified_base_binding_digest="4" * 64,
            verified_base_asset_digests={
                "normalization": "5" * 64,
                "encoder_checkpoint": "6" * 64,
                "encoder_config": "7" * 64,
            },
        )
