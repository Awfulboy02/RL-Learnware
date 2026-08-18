#!/usr/bin/env python3
"""Independently recompute every formal retrieval ranking and audit checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from policy_learnware_v0.artifacts import ArtifactLayout, ArtifactLayoutError
from policy_learnware_v0.cli import (
    _load_env_schemas,
    _load_frozen_protocol,
    _load_verified_pool_build_manifest,
    _query_id,
    _retrieval_evaluator_contract,
)
from policy_learnware_v0.config import load_protocol_draft
from policy_learnware_v0.hashing import canonicalize, sha256_file, sha256_json
from policy_learnware_v0.io import read_json
from policy_learnware_v0.pool.learnware import load_public_pool
from policy_learnware_v0.probe.dataset import load_dataset_artifact
from policy_learnware_v0.probe.seed_plan import SeedPlan
from policy_learnware_v0.representation.canonicalizer import TransitionCanonicalizer
from policy_learnware_v0.representation.encoder import (
    EncoderCheckpoint,
    TransitionSemanticEncoder,
)
from policy_learnware_v0.representation.normalization import NormalizationStats
from policy_learnware_v0.reuse.selector import (
    NearestSpecSelector,
    SelectionResult,
    target_source_cross_terms,
)
from policy_learnware_v0.rkme.empirical import episode_balanced_weights
from policy_learnware_v0.rkme.gaussian import GaussianKernel
from policy_learnware_v0.evaluation.retrieval_accel import (
    nested_prefix_self_kernel_sums_jax,
)


def _semantic_selection(value: SelectionResult) -> dict[str, Any]:
    payload = value.to_dict()
    payload.pop("selector_runtime_seconds")
    return payload


def _shard_path(layout: ArtifactLayout, index: int, count: int) -> Path:
    return layout.reports_dir / (
        f"retrieval_recompute_audit_shard_{index:03d}_of_{count:03d}.json"
    )


def _run_shard(args: argparse.Namespace) -> None:
    config = load_protocol_draft(args.config)
    layout = ArtifactLayout(args.artifacts_root, config.pool.pool_id)
    if (
        args.shard_count <= 0
        or args.shard_index < 0
        or args.shard_index >= args.shard_count
    ):
        raise ValueError("invalid shard coordinates")
    protocol = _load_frozen_protocol(layout, config)
    pool = load_public_pool(layout.selector_pool_dir)
    _load_verified_pool_build_manifest(layout, config, protocol, pool)
    schemas = _load_env_schemas(layout, config)
    canonicalizer = TransitionCanonicalizer(
        stats=NormalizationStats.load_npz(layout.normalization),
        max_action_dim=config.environment.max_action_dim,
    )
    encoder = TransitionSemanticEncoder(
        EncoderCheckpoint.load(
            layout.encoder_checkpoint, read_json(layout.encoder_config)
        )
    )
    kernel = GaussianKernel.load_json(layout.kernel)
    selector = NearestSpecSelector(
        pool, negative_tolerance=config.selector.negative_tolerance
    )
    if not np.isclose(
        kernel.bandwidth, pool.kernel_bandwidth, rtol=1.0e-12, atol=0.0
    ):
        raise ArtifactLayoutError("audit kernel/pool bandwidth mismatch")
    seed_plan = SeedPlan(config.project_seed)
    records: list[dict[str, Any]] = []
    maximum_squared_error = 0.0
    maximum_distance_error = 0.0
    prefixes = tuple(config.episodes.target_query_prefix_grid)
    for bank in range(config.episodes.target_query_banks):
        for task_index, task in enumerate(config.environment.tasks):
            group_index = bank * len(config.environment.tasks) + task_index
            if group_index % args.shard_count != args.shard_index:
                continue
            dataset, manifest = load_dataset_artifact(
                layout.dataset_npz("target_query", task, bank=bank),
                layout.dataset_manifest("target_query", task, bank=bank),
            )
            expected_seeds = seed_plan.episodes(
                "target_query",
                task_index,
                range(config.episodes.target_query_max_per_task),
                bank_index=bank,
            )
            if (
                manifest.task != task
                or manifest.split != "target_query"
                or tuple(dataset.reset_seeds)
                != tuple(item.reset_seed for item in expected_seeds)
                or tuple(dataset.probe_seeds)
                != tuple(item.probe_seed for item in expected_seeds)
            ):
                raise ArtifactLayoutError(f"audit dataset binding mismatch: {task}/{bank}")
            points = encoder.encode(canonicalizer.pack(dataset, schemas[task]).packed)
            norms = nested_prefix_self_kernel_sums_jax(
                points,
                dataset.episode_offsets,
                prefixes,
                kernel,
            )
            for episode_count in prefixes:
                transition_stop = int(dataset.episode_offsets[episode_count])
                prefix_digest = dataset.prefix(episode_count).digest
                weights = episode_balanced_weights(
                    dataset.episode_offsets[: episode_count + 1]
                )
                recomputed = selector.select_from_precomputed_terms(
                    target_empirical_norm2=norms[episode_count],
                    target_source_cross=target_source_cross_terms(
                        points[:transition_stop], weights, pool
                    ),
                    target_dataset_digest=prefix_digest,
                    probe_episode_count=episode_count,
                    probe_steps=transition_stop,
                )
                query_id = _query_id(task, bank, episode_count)
                selection_path = layout.selection_result(query_id)
                stored_raw = read_json(selection_path)
                if not isinstance(stored_raw, Mapping):
                    raise ArtifactLayoutError(f"stored selection is malformed: {query_id}")
                stored = SelectionResult.from_dict(stored_raw)
                if (
                    recomputed.selection_id != stored.selection_id
                    or recomputed.selected_opaque_id != stored.selected_opaque_id
                    or len(recomputed.sorted_distances) != len(stored.sorted_distances)
                    or canonicalize(
                        {
                            key: value
                            for key, value in _semantic_selection(recomputed).items()
                            if key != "sorted_distances"
                        }
                    )
                    != canonicalize(
                        {
                            key: value
                            for key, value in _semantic_selection(stored).items()
                            if key != "sorted_distances"
                        }
                    )
                ):
                    raise ArtifactLayoutError(
                        f"recomputed selection metadata differs: {query_id}"
                    )
                squared_error = 0.0
                distance_error = 0.0
                for expected, actual in zip(
                    stored.sorted_distances,
                    recomputed.sorted_distances,
                    strict=True,
                ):
                    if (
                        expected.opaque_id != actual.opaque_id
                        or expected.numerical_clamped != actual.numerical_clamped
                    ):
                        raise ArtifactLayoutError(
                            f"recomputed ranking differs: {query_id}"
                        )
                    squared_error = max(
                        squared_error,
                        abs(expected.distance_squared - actual.distance_squared),
                    )
                    distance_error = max(
                        distance_error, abs(expected.distance - actual.distance)
                    )
                if squared_error > args.atol or distance_error > args.atol:
                    raise ArtifactLayoutError(
                        f"recomputed distances exceed tolerance for {query_id}: "
                        f"squared={squared_error}, distance={distance_error}"
                    )
                maximum_squared_error = max(maximum_squared_error, squared_error)
                maximum_distance_error = max(maximum_distance_error, distance_error)
                records.append(
                    {
                        "query_id": query_id,
                        "selection_sha256": sha256_file(selection_path),
                        "selected_opaque_id": stored.selected_opaque_id,
                        "ranking_sha256": sha256_json(
                            [asdict(item) for item in recomputed.sorted_distances]
                        ),
                        "maximum_distance_squared_error": squared_error,
                        "maximum_distance_error": distance_error,
                        "matched": True,
                    }
                )
    payload = {
        "schema": "policy-learnware.retrieval-recompute-audit-shard.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "pool_id": pool.pool_id,
        "public_pool_digest": sha256_json(pool.public_manifest()),
        "retrieval_metrics_sha256": sha256_file(layout.retrieval_metrics),
        "retrieval_execution_attestation_sha256": sha256_file(
            layout.retrieval_execution_attestation
        ),
        "auditor_source_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_contract": _retrieval_evaluator_contract(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "absolute_tolerance": args.atol,
        "query_count": len(records),
        "all_matched": True,
        "maximum_distance_squared_error": maximum_squared_error,
        "maximum_distance_error": maximum_distance_error,
        "queries": records,
    }
    destination = _shard_path(layout, args.shard_index, args.shard_count)
    digest = layout.publish_json(destination, payload)
    print(f"{destination} {digest} queries={len(records)}")


def _merge(args: argparse.Namespace) -> None:
    config = load_protocol_draft(args.config)
    layout = ArtifactLayout(args.artifacts_root, config.pool.pool_id)
    protocol = _load_frozen_protocol(layout, config)
    pool = load_public_pool(layout.selector_pool_dir)
    expected_contract = _retrieval_evaluator_contract()
    expected_query_ids = {
        _query_id(task, bank, count)
        for bank in range(config.episodes.target_query_banks)
        for task in config.environment.tasks
        for count in config.episodes.target_query_prefix_grid
    }
    records: dict[str, Mapping[str, Any]] = {}
    shard_digests: dict[str, str] = {}
    maximum_squared_error = 0.0
    maximum_distance_error = 0.0
    for index in range(args.shard_count):
        path = _shard_path(layout, index, args.shard_count)
        raw = read_json(path)
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema")
            != "policy-learnware.retrieval-recompute-audit-shard.v0"
            or raw.get("complete") is not True
            or raw.get("protocol_draft_hash") != config.draft_hash
            or raw.get("protocol_id") != protocol.protocol_id
            or raw.get("pool_id") != pool.pool_id
            or raw.get("public_pool_digest") != sha256_json(pool.public_manifest())
            or raw.get("retrieval_metrics_sha256")
            != sha256_file(layout.retrieval_metrics)
            or raw.get("retrieval_execution_attestation_sha256")
            != sha256_file(layout.retrieval_execution_attestation)
            or canonicalize(raw.get("evaluator_contract"))
            != canonicalize(expected_contract)
            or int(raw.get("shard_index", -1)) != index
            or int(raw.get("shard_count", -1)) != args.shard_count
            or raw.get("all_matched") is not True
            or not isinstance(raw.get("queries"), list)
        ):
            raise ArtifactLayoutError(f"invalid recompute audit shard: {path}")
        shard_digests[path.name] = sha256_file(path)
        maximum_squared_error = max(
            maximum_squared_error,
            float(raw["maximum_distance_squared_error"]),
        )
        maximum_distance_error = max(
            maximum_distance_error, float(raw["maximum_distance_error"])
        )
        for record in raw["queries"]:
            if not isinstance(record, Mapping):
                raise ArtifactLayoutError("recompute audit query is malformed")
            query_id = str(record.get("query_id", ""))
            if query_id in records or record.get("matched") is not True:
                raise ArtifactLayoutError("duplicate/failed recompute audit query")
            if record.get("selection_sha256") != sha256_file(
                layout.selection_result(query_id)
            ):
                raise ArtifactLayoutError(
                    f"recompute audit selection digest changed: {query_id}"
                )
            records[query_id] = record
    if set(records) != expected_query_ids:
        raise ArtifactLayoutError("recompute audit query coverage is incomplete")
    payload = {
        "schema": "policy-learnware.retrieval-recompute-audit.v0",
        "complete": True,
        "protocol_draft_hash": config.draft_hash,
        "protocol_id": protocol.protocol_id,
        "pool_id": pool.pool_id,
        "public_pool_digest": sha256_json(pool.public_manifest()),
        "retrieval_metrics_sha256": sha256_file(layout.retrieval_metrics),
        "retrieval_execution_attestation_sha256": sha256_file(
            layout.retrieval_execution_attestation
        ),
        "auditor_source_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_contract": expected_contract,
        "shard_count": args.shard_count,
        "shard_sha256": shard_digests,
        "query_count": len(records),
        "all_matched": True,
        "maximum_distance_squared_error": maximum_squared_error,
        "maximum_distance_error": maximum_distance_error,
        "selection_sha256": {
            query_id: str(record["selection_sha256"])
            for query_id, record in sorted(records.items())
        },
    }
    destination = layout.reports_dir / "retrieval_recompute_audit.json"
    digest = layout.publish_json(destination, payload)
    print(f"{destination} {digest} queries={len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        if args.shard_index is not None:
            raise ValueError("--merge cannot use --shard-index")
        _merge(args)
    else:
        if args.shard_index is None:
            raise ValueError("worker requires --shard-index")
        _run_shard(args)


if __name__ == "__main__":
    main()
