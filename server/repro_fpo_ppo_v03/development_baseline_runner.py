"""Resumable v0.3 development baseline and policy-return runner.

This is the small executable path needed after the v0.3 Occam pass:

``banks -> FULL features -> source RKME/query empirical KME -> ranking -> rollout``.

It deliberately does not recreate the deleted authority, receipt, or formal-gate
layers.  Each expensive unit is an immutable file, while the few summaries are
replaceable progress snapshots.  Public rankings and private policy returns are
kept in separate directories.  The runner accepts the 30 exact source contexts
and the 24 frozen development contexts; it never invents confirmatory factors.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.io import atomic_write_json, atomic_write_npz, read_json
from policy_learnware_v0.rkme.empirical import (
    EmpiricalKME,
    blockwise_weighted_kernel_sum,
    blockwise_weighted_kernel_sum_jax,
    build_empirical_kme,
    episode_balanced_weights,
)
from policy_learnware_v0.rkme.gaussian import GaussianKernel, calibrate_bandwidth
from policy_learnware_v0.rkme.reducer import ReducedRKME, ReducerConfig, reduce_kme
from policy_learnware_v0.v03.corro_trainers import (
    TASK_SUPCON_OBJECTIVE_DIGEST,
    CorroOptimizationConfig,
    JaxCorroTrainingBackend,
)
from policy_learnware_v0.v03.representation_ladder import (
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    RepresentationManifest,
    TrainingRequest,
)
from policy_learnware_v0.v03.source_market import V03SourcePolicyMarket
from policy_learnware_v0.v03.transition_views import (
    V_FULL_LEGACY,
    TransitionBank,
    apply_transition_view,
)
from server.repro_fpo_ppo_v03.signal_bank_runner import prepare_signal_inputs


METHODS = ("B0", "B1", "B2", "B3a", "B3b", "B4a", "B4b", "A-Env", "M02/B5")
SOURCE_ROLES = frozenset({"source", "source_reference_spec", "source_representation_train"})
QUERY_ROLES = frozenset({"development", "development_query", "confirmatory_query"})
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
POLICY_SEED_OFFSET = 1_000_003
SCHEMA = "policy-learnware.v03-minimal-development-baseline.v0"


class DevelopmentBaselineError(RuntimeError):
    pass


def _json(path: Path) -> Mapping[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise DevelopmentBaselineError(f"expected one JSON object: {path}")
    return value


def _publish(path: Path, value: Mapping[str, Any], *, resume: bool) -> str:
    expected = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if not resume or not path.is_file() or path.read_bytes() != expected:
            raise DevelopmentBaselineError(f"existing artifact differs: {path}")
        return sha256_file(path)
    try:
        return atomic_write_json(path, value)
    except FileExistsError:
        if resume and path.is_file() and path.read_bytes() == expected:
            return sha256_file(path)
        raise


def _progress(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_json(path, value, overwrite=True)


def _safe_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise DevelopmentBaselineError(f"unsafe {where}: {value!r}")
    return value


def _context_rows(index: Path) -> tuple[dict[str, Any], ...]:
    """Read either one collector index or atomic per-context indices."""

    index = index.expanduser().resolve()
    rows: list[tuple[dict[str, Any], Path]] = []
    if index.is_dir():
        for path in sorted(index.glob("contexts/*/index.json")):
            rows.append((dict(_json(path)), path.parent))
    elif index.suffix == ".jsonl":
        for line_number, raw in enumerate(index.read_text(encoding="utf-8").splitlines(), 1):
            if raw.strip():
                value = json.loads(raw)
                if not isinstance(value, Mapping):
                    raise DevelopmentBaselineError(f"non-object at {index}:{line_number}")
                rows.append((dict(value), index.parent))
    else:
        value = _json(index)
        embedded = value.get("contexts")
        if isinstance(embedded, list):
            for row in embedded:
                if not isinstance(row, Mapping):
                    raise DevelopmentBaselineError("context index contains a non-object row")
                rows.append((dict(row), index.parent))
        else:
            rows.append((dict(value), index.parent))
    if not rows:
        raise DevelopmentBaselineError(f"no contexts found under {index}")

    normalized: list[dict[str, Any]] = []
    for ordinal, (row, base) in enumerate(rows):
        context_id = _safe_id(row.get("context_id", row.get("bank_id")), "context_id")
        role = str(row.get("role", ""))
        if role not in SOURCE_ROLES | QUERY_ROLES:
            raise DevelopmentBaselineError(f"{context_id}: unsupported role {role!r}")
        task_id = _safe_id(row.get("task_id", row.get("task")), "task_id")
        raw_npz = row.get("npz", row.get("npz_path", row.get("bank_path")))
        if not isinstance(raw_npz, str) or not raw_npz:
            raise DevelopmentBaselineError(f"{context_id}: missing bank npz path")
        npz = Path(raw_npz).expanduser()
        if not npz.is_absolute():
            npz = base / npz
        npz = npz.resolve()
        if not npz.is_file() or npz.is_symlink():
            raise DevelopmentBaselineError(f"{context_id}: unsafe/missing bank {npz}")
        source_anchor = row.get("source_anchor_id")
        if role in SOURCE_ROLES and not isinstance(source_anchor, str):
            # The signal-bank index names the source bank but may omit the v0.2
            # anchor alias.  A 64-hex bank_id is the unambiguous minimal bridge.
            candidate = row.get("bank_id")
            source_anchor = candidate if isinstance(candidate, str) and len(candidate) == 64 else None
        if role in SOURCE_ROLES and not isinstance(source_anchor, str):
            raise DevelopmentBaselineError(f"{context_id}: source context lacks source_anchor_id")
        normalized.append(
            {
                **row,
                "context_id": context_id,
                "context_index": int(row.get("context_index", ordinal)),
                "role": "source" if role in SOURCE_ROLES else role,
                "task_id": task_id,
                "axis_id": row.get("axis_id", (row.get("dynamics") or {}).get("axis_id")),
                "factor_id": row.get("factor_id", (row.get("dynamics") or {}).get("factor_id")),
                "factor_value": row.get("factor_value", (row.get("dynamics") or {}).get("factor_value")),
                "source_anchor_id": source_anchor,
                "target_id": None if role in SOURCE_ROLES else row.get("target_id", context_id),
                "npz": str(npz),
            }
        )
    by_id = {row["context_id"]: row for row in normalized}
    if len(by_id) != len(normalized):
        raise DevelopmentBaselineError("duplicate context_id in collector output")
    sources = [row for row in normalized if row["role"] == "source"]
    if len({row["source_anchor_id"] for row in sources}) != len(sources):
        raise DevelopmentBaselineError("duplicate source_anchor_id in collector output")
    return tuple(by_id[key] for key in sorted(by_id))


def _market(public: Path, private: Path) -> V03SourcePolicyMarket:
    return V03SourcePolicyMarket.from_manifests(_json(public), _json(private))


def _full_features(bank: TransitionBank) -> tuple[np.ndarray, np.ndarray, str]:
    view = apply_transition_view(bank, V_FULL_LEGACY)
    return (
        np.asarray(view.feature_matrix, dtype=np.float64),
        np.asarray(bank.episode_offsets, dtype=np.int64),
        str(view.spec.digest),
    )


def _r5_transform(checkpoint_root: Path, *, seed: int) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, Any]]:
    matches: list[tuple[Path, Mapping[str, Any]]] = []
    for record_path in sorted(checkpoint_root.glob("**/job_record.json")):
        record = _json(record_path)
        if (
            record.get("status") == "COMPLETE"
            and record.get("condition_id") == V_FULL_LEGACY
            and record.get("representation_id") == R5_VIEW_SPECIFIC_CORRO_REFIT
            and record.get("seed") == seed
        ):
            matches.append((record_path.parent, record))
    if len(matches) != 1:
        raise DevelopmentBaselineError(
            f"expected one completed FULL/R5 seed={seed} checkpoint, found {len(matches)}"
        )
    directory, record = matches[0]
    checkpoint = (directory / "checkpoint.bin").read_bytes()
    manifest = RepresentationManifest.from_dict(_json(directory / "representation_manifest.json"))
    if manifest.checkpoint_digest != sha256_bytes(checkpoint):
        raise DevelopmentBaselineError("FULL/R5 checkpoint bytes differ from manifest")
    request = TrainingRequest(
        representation_id=R5_VIEW_SPECIFIC_CORRO_REFIT,
        input_dim=manifest.input_dim,
        output_dim=manifest.output_dim,
        hidden_dims=(256, 256),
        activation="relu",
        l2_normalize_output=True,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=seed,
    )
    if manifest.protocol_digest != request.request_digest:
        raise DevelopmentBaselineError("FULL/R5 checkpoint uses another training request")
    optimization = CorroOptimizationConfig(train_steps=int(record.get("train_steps", 20_000)))
    artifact = JaxCorroTrainingBackend().restore(
        checkpoint_bytes=checkpoint, request=request, optimization=optimization
    )
    return artifact.transform, {
        "checkpoint_path": str(directory / "checkpoint.bin"),
        "checkpoint_sha256": sha256_bytes(checkpoint),
        "coordinate_digest": manifest.coordinate_digest,
        "input_dim": manifest.input_dim,
        "output_dim": manifest.output_dim,
        "seed": seed,
    }


def _moment(points: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    weights = episode_balanced_weights(offsets)
    mean = np.sum(points * weights[:, None], axis=0)
    second = np.sum(np.square(points) * weights[:, None], axis=0)
    return np.concatenate((mean, np.sqrt(np.maximum(second - np.square(mean), 0.0)), second))


def _save_npz_or_resume(path: Path, arrays: Mapping[str, np.ndarray], *, resume: bool) -> str:
    if path.exists():
        if not resume or not path.is_file() or path.is_symlink():
            raise DevelopmentBaselineError(f"output exists: {path}")
        return sha256_file(path)
    try:
        return atomic_write_npz(path, arrays)
    except FileExistsError:
        if resume and path.is_file():
            return sha256_file(path)
        raise


def _empirical(
    points: np.ndarray,
    offsets: np.ndarray,
    *,
    bandwidth: float,
    protocol_id: str,
    dataset_digest: str,
    task: str,
    backend: str,
    block_size: int,
) -> EmpiricalKME:
    return build_empirical_kme(
        points,
        GaussianKernel(bandwidth),
        episode_offsets=offsets,
        protocol_id=protocol_id,
        dataset_digest=dataset_digest,
        source_task=task,
        block_size=block_size,
        computation_backend=backend,
    )


def _representation_paths(root: Path, kind: str, key: str) -> Path:
    return root / "representations" / kind / f"{key}.npz"


def build_representations(args: argparse.Namespace) -> Mapping[str, Any]:
    rows = _context_rows(args.context_index)
    market = _market(args.public_policy_market, args.deployment_private_registry)
    source_rows = tuple(row for row in rows if row["role"] == "source")
    if set(row["source_anchor_id"] for row in source_rows) != set(market.anchor_to_opaque_learnware_id):
        raise DevelopmentBaselineError("source banks and source market do not cover the same anchors")
    if args.phase in {"source", "all"} and (args.shard_count != 1 or args.shard_index != 0):
        raise DevelopmentBaselineError("source representation build is a single-writer phase")

    signal_inputs = prepare_signal_inputs(
        args.context_index.expanduser().resolve(),
        args.train_fraction,
        args.validation_fraction,
        args.split_seed,
        args.measurement_episodes,
        args.measurement_transitions,
    )
    if signal_inputs.source_count != len(source_rows):
        raise DevelopmentBaselineError("baseline and Signal source context counts differ")
    source_banks = {
        row["context_id"]: bank
        for row, bank in zip(
            signal_inputs.source_rows,
            signal_inputs.banks[: signal_inputs.source_count],
            strict=True,
        )
    }
    query_start = signal_inputs.source_count
    query_stop = query_start + signal_inputs.query_count
    development_banks = {
        row["context_id"]: bank
        for row, bank in zip(
            signal_inputs.query_rows,
            signal_inputs.banks[query_start:query_stop],
            strict=True,
        )
    }
    repeat_banks = {
        row["context_id"]: bank
        for row, bank in zip(
            signal_inputs.source_rows,
            signal_inputs.banks[query_stop:],
            strict=True,
        )
    }
    canonicalizer = signal_inputs.canonicalizer
    r5, r5_binding = _r5_transform(args.r5_checkpoint_root, seed=args.r5_seed)
    source_points: dict[str, dict[str, tuple[np.ndarray, np.ndarray, Mapping[str, Any]]]] = {
        "raw": {}, "r5": {}
    }
    view_digest: str | None = None
    for row in source_rows:
        raw, offsets, observed_view = _full_features(source_banks[row["context_id"]])
        if view_digest is None:
            view_digest = observed_view
        elif view_digest != observed_view:
            raise DevelopmentBaselineError("FULL transition view identity drifted across banks")
        if raw.shape[1] != r5_binding["input_dim"]:
            raise DevelopmentBaselineError(
                f"FULL/R5 input width {r5_binding['input_dim']} != canonical FULL width {raw.shape[1]}"
            )
        encoded = np.asarray(r5(raw), dtype=np.float64)
        identity = market.anchor_to_opaque_learnware_id[row["source_anchor_id"]]
        source_points["raw"][identity] = (raw, offsets, row)
        source_points["r5"][identity] = (encoded, offsets, row)

    protocol = {
        "raw": sha256_json({"kind": "FULL_RAW", "view": view_digest, "canonicalizer": canonicalizer.canonicalizer_digest}),
        "r5": sha256_json({
            "kind": "FULL_R5",
            "view": view_digest,
            "canonicalizer": canonicalizer.canonicalizer_digest,
            "checkpoint_sha256": r5_binding["checkpoint_sha256"],
            "coordinate_digest": r5_binding["coordinate_digest"],
        }),
    }
    bandwidth = {
        kind: calibrate_bandwidth(
            {key: type("Events", (), {"points": value[0], "episode_offsets": value[1]})() for key, value in values.items()},
            calibration_pairs=args.calibration_pairs,
            seed=args.bandwidth_seed,
        )
        for kind, values in source_points.items()
    }
    build_config = {
        "schema": SCHEMA,
        "stage": "representations",
        "policy_market_id": market.policy_market_id,
        "query_mode": "QUERY_EMPIRICAL",
        "source_mode": "SOURCE_REDUCED",
        "context_count": len(rows),
        "source_count": len(source_rows),
        "context_rows": [dict(row) for row in rows],
        "canonicalizer": canonicalizer.to_dict(),
        "view_id": V_FULL_LEGACY,
        "view_digest": view_digest,
        "r5": r5_binding,
        "protocol_ids": protocol,
        "bandwidths": bandwidth,
        "kernel": "gaussian",
        "weighting": "episode_balanced",
        "measurement": {
            **dict(signal_inputs.measurement),
            "source_query_split": "independent held reference/repeat halves",
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "split_seed": args.split_seed,
        },
        "reducer": {
            "support_budget": args.support_budget,
            "support_steps": args.support_steps,
            "kmeans_steps": args.kmeans_steps,
            "optimizer_backend": args.backend,
            "negative_tolerance": args.negative_tolerance,
        },
        "warnings": ([] if len(source_rows) == 30 else [f"development run has {len(source_rows)} source banks, expected 30"]),
    }
    config_path = args.output_dir / "representations" / "build_config.json"
    if args.phase in {"source", "all"}:
        _publish(config_path, build_config, resume=args.resume)
        reducer = ReducerConfig(
            support_budget=args.support_budget,
            support_steps=args.support_steps,
            kmeans_steps=args.kmeans_steps,
            optimizer_backend=args.backend,
            negative_tolerance=args.negative_tolerance,
        )
        for kind, values in source_points.items():
            for opaque_id, (points, offsets, row) in values.items():
                if kind == "raw":
                    _save_npz_or_resume(
                        _representation_paths(
                            args.output_dir, "source_moments", opaque_id
                        ),
                        {
                            "feature": _moment(points, offsets),
                            "episode_offsets": offsets,
                            "measurement_transition_count": np.asarray(
                                points.shape[0], dtype=np.int64
                            ),
                        },
                        resume=args.resume,
                    )
                path = _representation_paths(args.output_dir, f"source_{kind}", opaque_id)
                if path.exists() and args.resume:
                    ReducedRKME.load_npz(path)
                    continue
                measurement_digest = sha256_json(
                    {
                        "dataset_digest": row.get("dataset_digest"),
                        "measured_arrays_digest": sha256_ndarrays(
                            {"points": points, "episode_offsets": offsets}
                        ),
                        "point_count": int(points.shape[0]),
                    }
                )
                empirical = _empirical(
                    points, offsets, bandwidth=bandwidth[kind], protocol_id=protocol[kind],
                    dataset_digest=measurement_digest,
                    task=str(row["task_id"]), backend=args.backend, block_size=args.block_size,
                )
                reduced = reduce_kme(empirical, reducer)
                reduced.save_npz(path)

    if args.phase in {"queries", "all"}:
        if not config_path.is_file():
            raise DevelopmentBaselineError("run source phase before sharded query phases")
        frozen = _json(config_path)
        if frozen.get("policy_market_id") != market.policy_market_id or frozen.get("protocol_ids") != protocol:
            raise DevelopmentBaselineError("query build inputs differ from frozen source build")
        if frozen.get("measurement") != build_config["measurement"]:
            raise DevelopmentBaselineError("query measurement cap differs from frozen source build")
        selected = tuple(row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index)
        for row in selected:
            bank = (
                repeat_banks[row["context_id"]]
                if row["role"] == "source"
                else development_banks[row["context_id"]]
            )
            raw, measured_offsets, _ = _full_features(bank)
            values = {"raw": raw, "r5": np.asarray(r5(raw), dtype=np.float64)}
            moment_path = _representation_paths(args.output_dir, "moments", row["context_id"])
            _save_npz_or_resume(
                moment_path,
                {
                    "feature": _moment(raw, measured_offsets),
                    "episode_offsets": measured_offsets,
                    "measurement_transition_count": np.asarray(
                        raw.shape[0], dtype=np.int64
                    ),
                },
                resume=args.resume,
            )
            measurement_digest = sha256_json(
                {
                    "dataset_digest": row.get("dataset_digest"),
                    "measurement_bank_digest": bank.canonical_bank_digest,
                    "point_count": int(raw.shape[0]),
                }
            )
            for kind, points in values.items():
                path = _representation_paths(args.output_dir, f"query_{kind}", row["context_id"])
                if path.exists() and args.resume:
                    EmpiricalKME.load_npz(path)
                    continue
                empirical = _empirical(
                    points, measured_offsets, bandwidth=bandwidth[kind], protocol_id=protocol[kind],
                    dataset_digest=measurement_digest,
                    task=str(row["task_id"]), backend=args.backend, block_size=args.block_size,
                )
                empirical.save_npz(path)
        summary = {
            "schema": SCHEMA,
            "stage": "build-representations",
            "status": "COMPLETE",
            "phase": args.phase,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "scheduled_context_count": len(selected),
        }
    else:
        summary = {"schema": SCHEMA, "stage": "build-representations", "status": "COMPLETE", "phase": "source", "source_count": len(source_rows)}
    _progress(args.output_dir / "progress" / f"build-{args.phase}-{args.shard_index:03d}-of-{args.shard_count:03d}.json", summary)
    return summary


def _distance(query: EmpiricalKME, source: ReducedRKME, *, backend: str, block_size: int) -> float:
    kernel = GaussianKernel(query.bandwidth)
    cross_fn = blockwise_weighted_kernel_sum_jax if backend == "jax" else blockwise_weighted_kernel_sum
    cross = cross_fn(query.points, query.weights, source.supports, source.beta, kernel, block_size=block_size)
    raw = float(query.norm2 + source.rkme_norm2 - 2.0 * cross)
    scale = max(1.0, abs(query.norm2), abs(source.rkme_norm2), abs(2.0 * cross))
    if raw < -1.0e-8 * scale:
        raise DevelopmentBaselineError(f"materially negative MMD squared: {raw}")
    return math.sqrt(max(raw, 0.0))


def _rank_rows(scores: Mapping[str, float], tie: Mapping[str, str], distances: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    if set(scores) != set(tie):
        raise DevelopmentBaselineError("ranking scores do not cover the public market")
    order = sorted(scores, key=lambda key: (-float(scores[key]), tie[key]))
    return [
        {
            "rank": rank,
            "opaque_learnware_id": opaque_id,
            "score": float(scores[opaque_id]),
            "distance": None if distances is None else float(distances[opaque_id]),
            "tie_break_token": tie[opaque_id],
        }
        for rank, opaque_id in enumerate(order, 1)
    ]


def _oracle_rows(root: Path, contexts: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> np.ndarray | None:
    development = [row for row in contexts if row["role"] != "source"]
    if not development:
        return None
    labels = np.zeros((len(development), len(ids)), dtype=np.float64)
    for row_index, row in enumerate(development):
        for policy_index, opaque_id in enumerate(ids):
            path = root / "oracle" / row["context_id"] / f"{opaque_id}.json"
            if not path.is_file():
                return None
            record = _json(path)
            status = record.get("status")
            if status == "OK":
                labels[row_index, policy_index] = float(record["normalized_mean_return"])
            elif status == "ABI_FLOOR_NA":
                labels[row_index, policy_index] = 0.0
            else:
                return None
    return labels


def _b4_models(root: Path, contexts: Sequence[Mapping[str, Any]], ids: Sequence[str], labels: np.ndarray, *, resume: bool, neighbors: int, ridge: float) -> Mapping[str, np.ndarray]:
    development = [row for row in contexts if row["role"] != "source"]
    features = np.stack([
        np.load(_representation_paths(root, "moments", row["context_id"]), allow_pickle=False)["feature"]
        for row in development
    ]).astype(np.float64)
    path = root / "rankings" / "b4_model.npz"
    _save_npz_or_resume(
        path,
        {"features": features, "labels": labels, "context_ids": np.asarray([row["context_id"] for row in development])},
        resume=resume,
    )
    _publish(
        root / "rankings" / "b4_model.json",
        {"schema": SCHEMA, "stage": "B4_FIT", "context_count": len(development), "policy_ids": list(ids), "neighbor_count": neighbors, "ridge": ridge, "label_semantics": "same-task-return-over-horizon; incompatible=ABI_FLOOR_ZERO"},
        resume=resume,
    )
    return {
        "features": features,
        "labels": labels,
        "context_ids": np.asarray([row["context_id"] for row in development]),
    }


def rank(args: argparse.Namespace) -> Mapping[str, Any]:
    build = _json(args.output_dir / "representations" / "build_config.json")
    contexts = tuple(dict(row) for row in build["context_rows"])
    market = _market(args.public_policy_market, args.deployment_private_registry)
    ids = tuple(sorted(market.entries))
    tie = {key: market.entries[key].tie_break_token for key in ids}
    competence = {key: float(market.entries[key].normalized_source_competence) for key in ids}
    source_rows = {row["source_anchor_id"]: row for row in contexts if row["role"] == "source"}
    source_by_id = {
        market.anchor_to_opaque_learnware_id[anchor]: row for anchor, row in source_rows.items()
    }
    sources = {
        kind: {opaque_id: ReducedRKME.load_npz(_representation_paths(args.output_dir, f"source_{kind}", opaque_id)) for opaque_id in ids}
        for kind in ("raw", "r5")
    }
    moments = {
        opaque_id: np.load(
            _representation_paths(args.output_dir, "source_moments", opaque_id),
            allow_pickle=False,
        )["feature"].astype(np.float64)
        for opaque_id in source_by_id
    }
    nominal = {
        row["task_id"]: market.anchor_to_opaque_learnware_id[row["source_anchor_id"]]
        for row in source_rows.values()
        if str(row.get("factor_id", "")) == "source_nominal" or float(row.get("factor_value", float("nan"))) == 1.0
    }
    if set(nominal) != {row["task_id"] for row in contexts}:
        raise DevelopmentBaselineError("B2 requires one nominal source anchor per task")
    sigma_distances = []
    kernel = GaussianKernel(next(iter(sources["r5"].values())).bandwidth)
    for i, left_id in enumerate(ids):
        left = sources["r5"][left_id]
        for right_id in ids[i + 1 :]:
            right = sources["r5"][right_id]
            cross = float(left.beta @ kernel.gram(left.supports, right.supports) @ right.beta)
            value = math.sqrt(max(left.rkme_norm2 + right.rkme_norm2 - 2.0 * cross, 0.0))
            if value > 0.0:
                sigma_distances.append(value)
    if not sigma_distances:
        raise DevelopmentBaselineError("M02/B5 source-only sigma has no nonzero distance")
    sigma = float(np.median(sigma_distances))

    labels = _oracle_rows(args.output_dir, contexts, ids)
    b4 = None if labels is None else _b4_models(
        args.output_dir, contexts, ids, labels, resume=args.resume,
        neighbors=args.b4_neighbors, ridge=args.b4_ridge,
    )
    selected_contexts = tuple(row for index, row in enumerate(contexts) if index % args.shard_count == args.shard_index)
    completed = 0
    pending_b4 = 0
    for row in selected_contexts:
        context_id = row["context_id"]
        raw_query = EmpiricalKME.load_npz(_representation_paths(args.output_dir, "query_raw", context_id))
        r5_query = EmpiricalKME.load_npz(_representation_paths(args.output_dir, "query_r5", context_id))
        raw_dist = {key: _distance(raw_query, sources["raw"][key], backend=args.backend, block_size=args.block_size) for key in ids}
        r5_dist = {key: _distance(r5_query, sources["r5"][key], backend=args.backend, block_size=args.block_size) for key in ids}
        feature = np.load(_representation_paths(args.output_dir, "moments", context_id), allow_pickle=False)["feature"].astype(np.float64)
        moment_dist = {key: float(np.linalg.norm(feature - moments[key])) for key in ids}
        nominal_dist = {task: raw_dist[opaque_id] for task, opaque_id in nominal.items()}
        selected_nominal = nominal[min(nominal_dist, key=lambda task: (nominal_dist[task], tie[nominal[task]]))]
        random_scores = {
            key: int(sha256_json({"seed": args.random_seed, "context_id": context_id, "tie": tie[key]})[:13], 16) / float(16**13)
            for key in ids
        }
        method_values: dict[str, tuple[Mapping[str, float], str, Mapping[str, float] | None]] = {
            "B0": (random_scores, "frozen_random_key", None),
            "B1": (competence, "normalized_source_competence", None),
            "B2": ({key: float(key == selected_nominal) for key in ids}, "legacy_taskspec_nearest_nominal_champion", None),
            "B3a": ({key: -moment_dist[key] for key in ids}, "negative_raw_moment_distance", moment_dist),
            "B3b": ({key: -raw_dist[key] for key in ids}, "negative_raw_empirical_to_reduced_mmd", raw_dist),
            "A-Env": ({key: -r5_dist[key] for key in ids}, "negative_r5_empirical_to_reduced_mmd", r5_dist),
            "M02/B5": ({key: math.log(max(competence[key], 1.0e-12)) - r5_dist[key] / sigma for key in ids}, "log_competence_minus_r5_distance_over_source_sigma", r5_dist),
        }
        if b4 is not None:
            context_ids = tuple(str(value) for value in b4["context_ids"])
            train_mask = np.ones(len(context_ids), dtype=np.bool_)
            prediction_scope = "all_24_development"
            if row["role"] != "source":
                try:
                    train_mask[context_ids.index(context_id)] = False
                except ValueError as error:
                    raise DevelopmentBaselineError(
                        f"development context absent from B4 labels: {context_id}"
                    ) from error
                prediction_scope = "leave_one_context_out"
            train_features = b4["features"][train_mask]
            train_labels = b4["labels"][train_mask]
            mean = np.mean(train_features, axis=0)
            std = np.std(train_features, axis=0)
            std = np.where(std < 1.0e-8, 1.0, std)
            standardized_train = (train_features - mean) / std
            standardized_query = (feature - mean) / std
            local = np.linalg.norm(
                standardized_train - standardized_query[None, :], axis=1
            )
            neighbor_index = np.argsort(local, kind="stable")[: args.b4_neighbors]
            knn = np.mean(train_labels[neighbor_index], axis=0)
            design = np.concatenate(
                (standardized_train, np.ones((standardized_train.shape[0], 1))),
                axis=1,
            )
            coefficients = np.linalg.solve(
                design.T @ design
                + args.b4_ridge * np.eye(design.shape[1]),
                design.T @ train_labels,
            )
            linear = np.concatenate((standardized_query, [1.0])) @ coefficients
            method_values["B4a"] = ({key: float(knn[i]) for i, key in enumerate(ids)}, "knn_predicted_normalized_return", None)
            method_values["B4b"] = ({key: float(linear[i]) for i, key in enumerate(ids)}, "ridge_predicted_normalized_return", None)
        else:
            pending_b4 += 2
        for method in METHODS:
            if method not in method_values:
                continue
            scores, semantics, distances = method_values[method]
            ranking = _rank_rows(scores, tie, distances)
            record = {
                "schema": SCHEMA,
                "stage": "PUBLIC_RANKING",
                "query_mode": "QUERY_EMPIRICAL",
                "source_mode": "SOURCE_REDUCED",
                "policy_market_id": market.policy_market_id,
                "context_id": context_id,
                "context_role": row["role"],
                "task_id": row["task_id"],
                "method_id": method,
                "score_semantics": semantics,
                "prediction_scope": (
                    prediction_scope if method in {"B4a", "B4b"} else "source_or_query_only"
                ),
                "selected_opaque_learnware_id": ranking[0]["opaque_learnware_id"],
                "rows": ranking,
            }
            record["ranking_digest"] = sha256_json(record)
            _publish(args.output_dir / "rankings" / context_id / f"{method.replace('/', '_')}.json", record, resume=args.resume)
            completed += 1
    summary = {
        "schema": SCHEMA,
        "stage": "rank",
        "status": "COMPLETE" if pending_b4 == 0 else "WAITING_FOR_DEVELOPMENT_LABELS",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "context_count": len(selected_contexts),
        "ranking_count": completed,
        "pending_b4_count": pending_b4,
        "source_only_lmin_sigma": sigma,
    }
    _progress(args.output_dir / "progress" / f"rank-{args.shard_index:03d}-of-{args.shard_count:03d}.json", summary)
    return summary


def _anchor_tasks(directory: Path) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for path in sorted(directory.glob("**/*.json")):
        try:
            value = _json(path)
        except Exception:
            continue
        anchor = value.get("anchor_id", value.get("source_anchor_id"))
        task = value.get("task", value.get("task_id"))
        if isinstance(anchor, str) and isinstance(task, str):
            result[anchor] = task
    return result


def _factory(spec: str | None, args: argparse.Namespace) -> Callable[..., Any]:
    if spec:
        module_name, attribute = spec.split(":", 1)
        function = getattr(importlib.import_module(module_name), attribute)
        if not callable(function):
            raise DevelopmentBaselineError("environment factory is not callable")
        return function
    collector = importlib.import_module("server.repro_fpo_ppo_v03.dynamics_probe_collector")
    load_plan = getattr(collector, "load_collection_plan")
    build_environment = getattr(collector, "build_environment")
    build_variant_factory = getattr(collector, "build_variant_factory")
    plan = load_plan(args.v02_config, args.cp0_config)
    variant_factory = build_variant_factory(plan)

    def create(row: Mapping[str, Any], **_kwargs: Any) -> Any:
        matches = [item for item in plan.contexts if item.context_id == row["context_id"]]
        if len(matches) != 1:
            raise DevelopmentBaselineError(
                f"collector plan has no unique context {row['context_id']}"
            )
        return build_environment(plan, matches[0], factory=variant_factory, jit=False)

    return create


def evaluate(args: argparse.Namespace) -> Mapping[str, Any]:
    if not sys.dont_write_bytecode:
        raise DevelopmentBaselineError("policy evaluation must run with python -B/PYTHONDONTWRITEBYTECODE=1")
    from server.repro_fpo_ppo_v02.vendor import inspect_vendor_directory, require_vendor_pythonpath_first
    from policy_learnware_v0.policy.loader import load_policy
    from policy_learnware_v0.policy.evaluate import evaluate_frozen_policy_returns_batched

    require_vendor_pythonpath_first(inspect_vendor_directory(args.vendor_dir))
    build = _json(args.output_dir / "representations" / "build_config.json")
    contexts = tuple(dict(row) for row in build["context_rows"])
    market = _market(args.public_policy_market, args.deployment_private_registry)
    anchor_tasks = _anchor_tasks(args.source_anchor_manifests)
    source_rows = [row for row in contexts if row["role"] == "source"]
    if anchor_tasks and any(anchor_tasks.get(row["source_anchor_id"]) != row["task_id"] for row in source_rows):
        raise DevelopmentBaselineError("collector source contexts differ from anchor manifests")
    policy_task = {
        opaque_id: next(row["task_id"] for row in source_rows if row["source_anchor_id"] == entry.source_anchor_id)
        for opaque_id, entry in market.deployment_private.items()
    }
    ids = tuple(sorted(market.entries))
    selected_contexts = tuple(row for index, row in enumerate(contexts) if index % args.shard_count == args.shard_index)
    environment_factory = _factory(args.environment_factory, args)
    policy_cache: dict[str, Any] = {}
    count = {"OK": 0, "ABI_FLOOR_NA": 0, "ERROR": 0}
    reset_seeds = tuple(args.reset_seed_start + index for index in range(args.episodes))
    policy_seeds = tuple(seed + POLICY_SEED_OFFSET for seed in reset_seeds)

    for row in selected_contexts:
        context_id = row["context_id"]
        compatible = {opaque_id for opaque_id in ids if policy_task[opaque_id] == row["task_id"]}
        if len(compatible) != 5:
            raise DevelopmentBaselineError(f"{context_id}: expected five same-task champions, found {len(compatible)}")
        for opaque_id in ids:
            if opaque_id in compatible:
                continue
            path = args.output_dir / "oracle" / context_id / f"{opaque_id}.json"
            record = {
                "schema": SCHEMA, "stage": "PRIVATE_ORACLE", "context_id": context_id,
                "task_id": row["task_id"], "opaque_learnware_id": opaque_id,
                "status": "ABI_FLOOR_NA", "executed": False,
                "reason": "different_task_observation_action_ABI", "normalized_floor": 0.0,
            }
            _publish(path, record, resume=args.resume)
            count["ABI_FLOOR_NA"] += 1

        try:
            built = environment_factory(
                row,
                v02_config_path=args.v02_config,
                cp0_config_path=args.cp0_config,
                source_anchor_manifests=args.source_anchor_manifests,
                fpo_root=args.fpo_root,
                vendor_dir=args.vendor_dir,
            )
            environment = getattr(built, "native_environment", getattr(built, "environment", built))
        except Exception as error:
            environment = None
            environment_error = f"{type(error).__name__}: {error}"
        for opaque_id in sorted(compatible):
            path = args.output_dir / "oracle" / context_id / f"{opaque_id}.json"
            if path.exists():
                if not args.resume:
                    raise DevelopmentBaselineError(
                        f"oracle output exists; pass --resume: {path}"
                    )
                previous = _json(path)
                if previous.get("status") == "OK":
                    expected_binding = {
                        "reset_seeds": list(reset_seeds),
                        "policy_seeds": list(policy_seeds),
                        "horizon": args.horizon,
                        "bundle_digest": market.deployment_private[
                            opaque_id
                        ].bundle_digest,
                    }
                    if any(previous.get(key) != value for key, value in expected_binding.items()):
                        raise DevelopmentBaselineError(
                            f"oracle resume binding changed: {path}"
                        )
                    count["OK"] += 1
                    continue
            started = time.monotonic()
            try:
                if environment is None:
                    raise DevelopmentBaselineError(environment_error)
                if opaque_id not in policy_cache:
                    policy_cache[opaque_id] = load_policy(
                        market.deployment_private[opaque_id].bundle_path,
                        fpo_root=args.fpo_root,
                        runtime_only=True,
                    )
                policy = policy_cache[opaque_id]
                if policy.observation_dim != int(np.load(row["npz"], allow_pickle=False)["observation"].shape[1]) or policy.action_dim != int(np.load(row["npz"], allow_pickle=False)["action"].shape[1]):
                    raise DevelopmentBaselineError("same-task policy tensor dimensions differ from target")
                returns = evaluate_frozen_policy_returns_batched(
                    policy, environment, reset_seeds=reset_seeds, policy_seeds=policy_seeds,
                    horizon=args.horizon, observation_dim=policy.observation_dim, action_dim=policy.action_dim,
                )
                values = np.asarray(returns, dtype=np.float64)
                record = {
                    "schema": SCHEMA, "stage": "PRIVATE_ORACLE", "context_id": context_id,
                    "task_id": row["task_id"], "opaque_learnware_id": opaque_id,
                    "status": "OK", "executed": True, "reset_seeds": list(reset_seeds),
                    "policy_seeds": list(policy_seeds), "horizon": args.horizon,
                    "episode_returns": values.tolist(), "mean_return": float(np.mean(values)),
                    "std_return": float(np.std(values)),
                    "normalized_mean_return": float(np.mean(values) / args.horizon),
                    "runtime_seconds": time.monotonic() - started,
                    "bundle_digest": market.deployment_private[opaque_id].bundle_digest,
                }
            except Exception as error:
                record = {
                    "schema": SCHEMA, "stage": "PRIVATE_ORACLE", "context_id": context_id,
                    "task_id": row["task_id"], "opaque_learnware_id": opaque_id,
                    "status": "ERROR", "executed": environment is not None,
                    "error_type": type(error).__name__, "error": str(error),
                    "runtime_seconds": time.monotonic() - started,
                }
            atomic_write_json(path, record, overwrite=path.exists() and args.resume)
            count[str(record["status"])] += 1
    summary = {
        "schema": SCHEMA, "stage": "evaluate", "status": "COMPLETE" if count["ERROR"] == 0 else "PARTIAL",
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "context_count": len(selected_contexts), "episodes_per_compatible_policy": args.episodes,
        "counts": count,
    }
    _progress(args.output_dir / "progress" / f"evaluate-{args.shard_index:03d}-of-{args.shard_count:03d}.json", summary)
    return summary


def summarize(args: argparse.Namespace) -> Mapping[str, Any]:
    build = _json(args.output_dir / "representations" / "build_config.json")
    contexts = tuple(dict(row) for row in build["context_rows"])
    market = _market(args.public_policy_market, args.deployment_private_registry)
    ids = tuple(sorted(market.entries))
    methods: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    missing: list[str] = []
    for row in contexts:
        oracle_paths = {
            opaque_id: args.output_dir / "oracle" / row["context_id"] / f"{opaque_id}.json"
            for opaque_id in ids
        }
        if not all(path.is_file() for path in oracle_paths.values()):
            missing.append(f"{row['context_id']}:oracle")
            continue
        oracle = {opaque_id: _json(path) for opaque_id, path in oracle_paths.items()}
        valid = {key: float(value["normalized_mean_return"]) for key, value in oracle.items() if value.get("status") == "OK"}
        if len(valid) != 5:
            missing.append(f"{row['context_id']}:oracle")
            continue
        best_id = max(valid, key=lambda key: (valid[key], key))
        best = valid[best_id]
        expected = market.anchor_to_opaque_learnware_id.get(row.get("source_anchor_id"))
        for method in METHODS:
            path = args.output_dir / "rankings" / row["context_id"] / f"{method.replace('/', '_')}.json"
            if not path.is_file():
                missing.append(f"{row['context_id']}:{method}")
                continue
            ranking = _json(path)
            selected = str(ranking["selected_opaque_learnware_id"])
            selected_return = valid.get(selected, 0.0)
            ranked_ids = [item["opaque_learnware_id"] for item in ranking["rows"]]
            methods[method].append(
                {
                    "context_id": row["context_id"], "role": row["role"], "task_id": row["task_id"],
                    "selected": selected, "selected_return": selected_return, "oracle_best": best,
                    "normalized_regret": best - selected_return,
                    "selected_abi_compatible": selected in valid,
                    "top1_task_compatible": selected in valid,
                    "top1_exact_anchor": None if expected is None else selected == expected,
                    "top3_oracle_coverage": best_id in ranked_ids[:3],
                    "top5_oracle_coverage": best_id in ranked_ids[:5],
                }
            )
    aggregates = {}
    for method, rows in methods.items():
        if not rows:
            continue
        aggregates[method] = {
            "context_count": len(rows),
            "mean_selected_return": float(np.mean([row["selected_return"] for row in rows])),
            "mean_normalized_regret": float(np.mean([row["normalized_regret"] for row in rows])),
            "top1_task_compatibility": float(np.mean([row["top1_task_compatible"] for row in rows])),
            "top3_oracle_coverage": float(np.mean([row["top3_oracle_coverage"] for row in rows])),
            "top5_oracle_coverage": float(np.mean([row["top5_oracle_coverage"] for row in rows])),
            "exact_anchor_accuracy": (None if not any(row["top1_exact_anchor"] is not None for row in rows) else float(np.mean([row["top1_exact_anchor"] for row in rows if row["top1_exact_anchor"] is not None]))),
        }
    summary = {
        "schema": SCHEMA, "stage": "summary", "status": "COMPLETE" if not missing else "PARTIAL",
        "formal": False, "scope": "30 exact recurrence + 24 frozen development; no confirmatory/extrapolation",
        "policy_market_id": market.policy_market_id, "context_count": len(contexts),
        "expected_ranking_count": len(contexts) * len(METHODS),
        "aggregates": aggregates, "missing": missing,
    }
    _progress(args.output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal v0.3 development baseline runner")
    parser.add_argument("command", choices=("build-representations", "rank", "evaluate", "summarize"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-policy-market", type=Path, required=True)
    parser.add_argument("--deployment-private-registry", type=Path, required=True)
    parser.add_argument("--context-index", type=Path)
    parser.add_argument("--r5-checkpoint-root", type=Path)
    parser.add_argument("--phase", choices=("source", "queries", "all"), default="all")
    parser.add_argument("--r5-seed", type=int, default=0)
    parser.add_argument("--backend", choices=("numpy", "jax"), default="jax")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--support-budget", type=int, default=100)
    parser.add_argument("--support-steps", type=int, default=1000)
    parser.add_argument("--kmeans-steps", type=int, default=25)
    parser.add_argument("--negative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--calibration-pairs", type=int, default=10_000)
    parser.add_argument("--measurement-episodes", type=int, default=16)
    parser.add_argument("--measurement-transitions", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--bandwidth-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--b4-neighbors", type=int, default=3)
    parser.add_argument("--b4-ridge", type=float, default=1.0e-3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--v02-config", type=Path)
    parser.add_argument("--cp0-config", type=Path)
    parser.add_argument("--source-anchor-manifests", type=Path)
    parser.add_argument("--fpo-root", type=Path)
    parser.add_argument("--vendor-dir", type=Path)
    parser.add_argument("--environment-factory", help="optional module:callable override")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--reset-seed-start", type=int, default=730_000)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index/count")
    positive = (
        args.block_size,
        args.support_budget,
        args.measurement_episodes,
        args.measurement_transitions,
        args.calibration_pairs,
        args.b4_neighbors,
        args.episodes,
        args.horizon,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("count and size arguments must be positive")
    if (
        not 0 < args.train_fraction < 1
        or not 0 < args.validation_fraction < 1
        or args.train_fraction + args.validation_fraction >= 1
        or args.split_seed < 0
    ):
        raise SystemExit("invalid source train/validation split")
    if args.command == "build-representations":
        if args.context_index is None or args.r5_checkpoint_root is None:
            raise SystemExit("build-representations requires --context-index and --r5-checkpoint-root")
        result = build_representations(args)
    elif args.command == "rank":
        result = rank(args)
    elif args.command == "evaluate":
        required = (args.v02_config, args.cp0_config, args.source_anchor_manifests, args.fpo_root, args.vendor_dir)
        if any(value is None for value in required):
            raise SystemExit("evaluate requires --v02-config --cp0-config --source-anchor-manifests --fpo-root --vendor-dir")
        result = evaluate(args)
    else:
        result = summarize(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 1 if result.get("status") == "PARTIAL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
