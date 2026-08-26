"""Small, real baseline comparison on the completed legacy six-task pool.

This is a development diagnostic, not the 30-policy v0.3 P6 experiment.  It
reuses the immutable v0 source/query traces and the completed CORRO TaskSpec
retrieval records.  It never trains an encoder and never executes a policy.

The two raw-data diagnostics are intentionally lightweight: B3a is nearest
source by source-standardized transition moments; D-RFF-KME is an RBF-KME
approximation using source-fitted normalization and deterministic random
Fourier features.  It is deliberately not called B3b, whose frozen raw
packed-event kernel/reducer index is unavailable here.  B2, A-Env, and M02/B5
are aliases for one materialized legacy TaskSpec/CORRO nearest-neighbour replay
after verifying its dataset, TaskSpec, checkpoint, and selection-file bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from policy_learnware_v0.hashing import canonical_json_bytes, sha256_file, sha256_json
from policy_learnware_v0.io import atomic_write_json, read_json
from policy_learnware_v0.probe.dataset import EpisodeDataset, load_dataset_artifact


SCOPE = "development/legacy-six-task"
SCHEMA = "policy-learnware.v03-basic-baseline-comparison.v0"
EVALUATED = ("B0", "B2", "B3a", "D-RFF-KME", "A-Env", "M02/B5")
LEGACY_EQUIVALENT = frozenset({"B2", "A-Env", "M02/B5"})
UNAVAILABLE = {
    "B1": "native-task champion returns are not cross-task competence labels",
    "B3b": "missing the frozen raw packed-event kernel/reducer source index required by the preregistered baseline",
    "B4a": "no leakage-safe development return matrix exists for legacy six-task replay",
    "B4b": "no leakage-safe development return matrix exists for legacy six-task replay",
}


def _publish_or_verify(path: Path, value: Mapping[str, Any], resume: bool) -> str:
    expected = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if not resume:
            raise ValueError(f"output exists; pass --resume: {path}")
        if not path.is_file() or path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"resume output differs from recomputation: {path}")
        return sha256_file(path)
    return atomic_write_json(path, value, overwrite=False)


def _safe_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("--legacy-v0-root must be a real directory")
    return root


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or unsafe {where}: {path}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must contain one JSON object")
    return value


def _load_dataset(root: Path, split: str, task: str, bank: int | None = None) -> tuple[EpisodeDataset, str]:
    directory = root / "datasets" / split
    if bank is not None:
        directory = directory / f"bank_{bank:03d}"
    npz_path = directory / f"{task}.npz"
    json_path = directory / f"{task}.json"
    dataset, manifest = load_dataset_artifact(npz_path, json_path)
    if manifest.split != split or manifest.task != task:
        raise ValueError(f"dataset identity differs at {json_path}")
    return dataset, sha256_file(json_path)


def _canonical_rows(
    dataset: EpisodeDataset,
    rows: slice | np.ndarray,
    *,
    max_observation_dim: int,
    max_action_dim: int,
) -> np.ndarray:
    observation = np.asarray(dataset.observation[rows], dtype=np.float32)
    action = np.asarray(dataset.action[rows], dtype=np.float32)
    next_observation = np.asarray(dataset.next_observation[rows], dtype=np.float32)
    count = observation.shape[0]
    obs = np.zeros((count, max_observation_dim), dtype=np.float32)
    nxt = np.zeros_like(obs)
    act = np.zeros((count, max_action_dim), dtype=np.float32)
    obs[:, : dataset.observation_dim] = observation
    nxt[:, : dataset.observation_dim] = next_observation
    act[:, : dataset.action_dim] = action
    obs_mask = np.zeros(max_observation_dim, dtype=np.float32)
    act_mask = np.zeros(max_action_dim, dtype=np.float32)
    obs_mask[: dataset.observation_dim] = 1.0
    act_mask[: dataset.action_dim] = 1.0
    return np.concatenate(
        (
            obs,
            act,
            np.asarray(dataset.reward[rows], dtype=np.float32).reshape(-1, 1),
            nxt,
            np.asarray(dataset.terminated[rows], dtype=np.float32).reshape(-1, 1),
            np.asarray(dataset.truncated[rows], dtype=np.float32).reshape(-1, 1),
            np.broadcast_to(obs_mask, (count, max_observation_dim)),
            np.broadcast_to(act_mask, (count, max_action_dim)),
        ),
        axis=1,
    )


def _moment_features(
    dataset: EpisodeDataset,
    prefixes: Sequence[int],
    *,
    max_observation_dim: int,
    max_action_dim: int,
) -> Mapping[int, np.ndarray]:
    wanted = set(prefixes)
    total = None
    square = None
    count = 0
    result: dict[int, np.ndarray] = {}
    for episode in range(dataset.episode_count):
        points = _canonical_rows(
            dataset,
            dataset.episode_slice(episode),
            max_observation_dim=max_observation_dim,
            max_action_dim=max_action_dim,
        ).astype(np.float64, copy=False)
        if total is None:
            total = np.zeros(points.shape[1], dtype=np.float64)
            square = np.zeros(points.shape[1], dtype=np.float64)
        total += np.sum(points, axis=0)
        square += np.sum(np.square(points), axis=0)
        count += points.shape[0]
        prefix = episode + 1
        if prefix in wanted:
            mean = total / count
            variance = np.maximum(square / count - np.square(mean), 0.0)
            result[prefix] = np.concatenate((mean, np.sqrt(variance)))
    if set(result) != wanted:
        raise ValueError("requested prefix lies outside query dataset")
    return result


def _fit_raw_normalizer(
    sources: Mapping[str, EpisodeDataset],
    *,
    max_observation_dim: int,
    max_action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    seconds = []
    for task in sorted(sources):
        points = _canonical_rows(
            sources[task],
            slice(None),
            max_observation_dim=max_observation_dim,
            max_action_dim=max_action_dim,
        ).astype(np.float64, copy=False)
        means.append(np.mean(points, axis=0))
        seconds.append(np.mean(np.square(points), axis=0))
    mean = np.mean(np.stack(means), axis=0)
    second = np.mean(np.stack(seconds), axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean, std


def _episode_rff_kmes(
    dataset: EpisodeDataset,
    prefixes: Sequence[int],
    *,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    omega: np.ndarray,
    phase: np.ndarray,
    max_observation_dim: int,
    max_action_dim: int,
    transitions_per_episode: int,
) -> Mapping[int, np.ndarray]:
    wanted = set(prefixes)
    cumulative = np.zeros(omega.shape[1], dtype=np.float64)
    result: dict[int, np.ndarray] = {}
    scale = math.sqrt(2.0 / omega.shape[1])
    for episode in range(dataset.episode_count):
        episode_slice = dataset.episode_slice(episode)
        start, stop = int(episode_slice.start), int(episode_slice.stop)
        take = min(transitions_per_episode, stop - start)
        indices = np.linspace(start, stop - 1, num=take, dtype=np.int64)
        points = _canonical_rows(
            dataset,
            indices,
            max_observation_dim=max_observation_dim,
            max_action_dim=max_action_dim,
        ).astype(np.float64, copy=False)
        standardized = (points - raw_mean) / raw_std
        cumulative += np.mean(scale * np.cos(standardized @ omega + phase), axis=0)
        prefix = episode + 1
        if prefix in wanted:
            result[prefix] = cumulative.copy() / prefix
    if set(result) != wanted:
        raise ValueError("requested prefix lies outside query dataset")
    return result


def _rank(features: Mapping[str, np.ndarray], query: np.ndarray) -> tuple[list[str], dict[str, float]]:
    distances = {
        task: float(np.linalg.norm(np.asarray(value) - np.asarray(query)))
        for task, value in features.items()
    }
    ranking = sorted(distances, key=lambda task: (distances[task], task))
    return ranking, distances


def _random_ranking(tasks: Sequence[str], seed: int, query_id: str) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest()
    result = list(tasks)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(result)
    return result


def _legacy_decision(
    *,
    root: Path,
    report_rows: Mapping[str, Mapping[str, Any]],
    opaque_to_task: Mapping[str, str],
    target_task: str,
    bank: int,
    prefix: int,
    dataset_digest: str,
) -> Mapping[str, Any]:
    query_id = f"bank{bank:03d}__{target_task}__n{prefix:03d}"
    row = report_rows.get(query_id)
    if row is None or row.get("target_dataset_digest") != dataset_digest:
        raise ValueError(f"legacy retrieval report does not bind {query_id}")
    path = root / "queries" / query_id / "selection_result.json"
    selection = _load_json(path, f"selection result {query_id}")
    if sha256_file(path) != row.get("selection_sha256"):
        raise ValueError(f"selection result checksum differs for {query_id}")
    if selection.get("target_dataset_digest") != dataset_digest:
        raise ValueError(f"selection result dataset differs for {query_id}")
    distance_rows = selection.get("sorted_distances")
    if not isinstance(distance_rows, list) or len(distance_rows) != len(opaque_to_task):
        raise ValueError(f"selection result ranking is incomplete for {query_id}")
    distances: dict[str, float] = {}
    ranking: list[str] = []
    for value in distance_rows:
        if not isinstance(value, Mapping) or value.get("opaque_id") not in opaque_to_task:
            raise ValueError(f"selection result contains an unknown entry for {query_id}")
        task = opaque_to_task[str(value["opaque_id"])]
        ranking.append(task)
        distances[task] = float(value["distance"])
    selected = opaque_to_task.get(str(selection.get("selected_opaque_id")))
    if selected is None or ranking[0] != selected:
        raise ValueError(f"selection result top-one differs for {query_id}")
    return {"selected_task": selected, "ranking": ranking, "distances": distances}


def _source_assets(root: Path) -> tuple[list[str], Mapping[str, str], Mapping[str, Any]]:
    environment = _load_json(root / "protocol" / "environment_manifest.json", "environment manifest")
    tasks = list(environment.get("tasks", ()))
    if len(tasks) != 6 or len(set(tasks)) != 6 or not all(isinstance(x, str) for x in tasks):
        raise ValueError("legacy diagnostic requires exactly six registered tasks")
    schemas = _load_json(root / "protocol" / "env_schemas.json", "environment schemas")
    if set(schemas) != set(tasks):
        raise ValueError("environment schema coverage differs from six tasks")
    registry = _load_json(root / "policy" / "deployment_registry.json", "deployment registry")
    records = registry.get("records")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("legacy private registry does not contain six records")
    opaque_to_task = {
        str(row["opaque_id"]): str(row["source_task"])
        for row in records
        if isinstance(row, Mapping)
    }
    if set(opaque_to_task.values()) != set(tasks):
        raise ValueError("legacy opaque/task mapping is incomplete")
    encoder = _load_json(root / "protocol" / "encoder_manifest.json", "encoder manifest")
    checkpoint = root / str(encoder.get("files", {}).get("checkpoint", {}).get("path", ""))
    if not checkpoint.is_file() or sha256_file(checkpoint) != encoder.get("files", {}).get("checkpoint", {}).get("sha256"):
        raise ValueError("legacy CORRO checkpoint binding differs")
    task_spec_files: dict[str, str] = {}
    for task in tasks:
        manifest_path = root / "task_specs" / task / "task_rkme.json"
        manifest = _load_json(manifest_path, f"TaskSpec manifest {task}")
        npz_path = root / "task_specs" / task / "task_rkme.npz"
        if manifest.get("complete") is not True or sha256_file(npz_path) != manifest.get("rkme_sha256"):
            raise ValueError(f"legacy TaskSpec binding differs for {task}")
        task_spec_files[task] = sha256_file(manifest_path)
    identity = {
        "environment_manifest_sha256": sha256_file(root / "protocol" / "environment_manifest.json"),
        "encoder_checkpoint_sha256": sha256_file(checkpoint),
        "task_spec_manifest_sha256": dict(sorted(task_spec_files.items())),
    }
    return tasks, opaque_to_task, {"schemas": schemas, "identity": identity}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    root = _safe_root(args.legacy_v0_root)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks, opaque_to_task, source_meta = _source_assets(root)
    schemas = source_meta["schemas"]
    max_obs = max(int(schemas[task]["schema"]["observation_dim"]) for task in tasks)
    max_act = max(int(schemas[task]["schema"]["action_dim"]) for task in tasks)
    prefixes = tuple(int(x) for x in _load_json(root / "reports" / "retrieval_metrics.json", "retrieval report")["prefix_grid"])
    retrieval = _load_json(root / "reports" / "retrieval_metrics.json", "retrieval report")
    rows = retrieval.get("queries")
    if retrieval.get("complete") is not True or not isinstance(rows, list):
        raise ValueError("legacy retrieval report is incomplete")
    report_rows = {str(row["query_id"]): row for row in rows if isinstance(row, Mapping)}

    sources: dict[str, EpisodeDataset] = {}
    source_manifest_sha: dict[str, str] = {}
    for task in tasks:
        sources[task], source_manifest_sha[task] = _load_dataset(root, "source_taskspec", task)
    source_moments = {
        task: _moment_features(
            dataset,
            (dataset.episode_count,),
            max_observation_dim=max_obs,
            max_action_dim=max_act,
        )[dataset.episode_count]
        for task, dataset in sources.items()
    }
    moment_matrix = np.stack([source_moments[task] for task in tasks])
    moment_mean = np.mean(moment_matrix, axis=0)
    moment_std = np.std(moment_matrix, axis=0)
    moment_std = np.where(moment_std < 1.0e-8, 1.0, moment_std)
    source_moments = {task: (value - moment_mean) / moment_std for task, value in source_moments.items()}

    raw_mean, raw_std = _fit_raw_normalizer(
        sources,
        max_observation_dim=max_obs,
        max_action_dim=max_act,
    )
    sigma = math.sqrt(raw_mean.size)
    rng = np.random.default_rng(args.seed)
    omega = rng.normal(size=(raw_mean.size, args.rff_dim)) / sigma
    phase = rng.uniform(0.0, 2.0 * math.pi, size=args.rff_dim)
    source_kmes = {
        task: _episode_rff_kmes(
            dataset,
            (dataset.episode_count,),
            raw_mean=raw_mean,
            raw_std=raw_std,
            omega=omega,
            phase=phase,
            max_observation_dim=max_obs,
            max_action_dim=max_act,
            transitions_per_episode=args.transitions_per_episode,
        )[dataset.episode_count]
        for task, dataset in sources.items()
    }

    query_jobs = []
    for bank_dir in sorted((root / "datasets" / "target_query").glob("bank_[0-9][0-9][0-9]")):
        bank = int(bank_dir.name.removeprefix("bank_"))
        for task in sorted(tasks):
            if (bank_dir / f"{task}.json").is_file():
                query_jobs.append((bank, task))
    if args.max_queries is not None:
        query_jobs = query_jobs[: args.max_queries]
    if not query_jobs:
        raise ValueError("no legacy target-query banks were found")

    config = {
        "seed": args.seed,
        "rff_dim": args.rff_dim,
        "transitions_per_episode": args.transitions_per_episode,
        "raw_feature_layout": "pad(o),pad(a),r,pad(o_next),terminated,truncated,obs_mask,action_mask",
        "b3a": "source-standardized mean+std nearest neighbour",
        "D-RFF-KME": "source-normalized deterministic RFF-RBF empirical KME nearest neighbour; development diagnostic only, not B3b",
        "rbf_sigma": sigma,
    }
    record_shas: dict[str, str] = {}
    records: list[Mapping[str, Any]] = []
    for bank, target_task in query_jobs:
        key = f"bank{bank:03d}__{target_task}"
        path = output / "queries" / f"{key}.json"
        if path.exists() and args.resume:
            record = _load_json(path, f"resume record {key}")
            if record.get("config") != config or record.get("target_task") != target_task or record.get("bank_index") != bank:
                raise ValueError(f"resume record configuration differs for {key}")
            records.append(record)
            record_shas[key] = sha256_file(path)
            continue
        query, query_manifest_sha = _load_dataset(root, "target_query", target_task, bank)
        query_moments = _moment_features(
            query,
            prefixes,
            max_observation_dim=max_obs,
            max_action_dim=max_act,
        )
        query_kmes = _episode_rff_kmes(
            query,
            prefixes,
            raw_mean=raw_mean,
            raw_std=raw_std,
            omega=omega,
            phase=phase,
            max_observation_dim=max_obs,
            max_action_dim=max_act,
            transitions_per_episode=args.transitions_per_episode,
        )
        trials = []
        for prefix in prefixes:
            query_id = f"bank{bank:03d}__{target_task}__n{prefix:03d}"
            random_ranking = _random_ranking(tasks, args.seed, query_id)
            moment_ranking, moment_distances = _rank(
                source_moments, (query_moments[prefix] - moment_mean) / moment_std
            )
            kme_ranking, kme_distances = _rank(source_kmes, query_kmes[prefix])
            legacy_decision = _legacy_decision(
                root=root,
                report_rows=report_rows,
                opaque_to_task=opaque_to_task,
                target_task=target_task,
                bank=bank,
                prefix=prefix,
                dataset_digest=query.prefix(prefix).digest,
            )
            selections = {
                "B0": {"selected_task": random_ranking[0], "ranking": random_ranking},
                "B3a": {"selected_task": moment_ranking[0], "ranking": moment_ranking, "distances": moment_distances},
                "D-RFF-KME": {"selected_task": kme_ranking[0], "ranking": kme_ranking, "distances": kme_distances},
            }
            for value in selections.values():
                value["correct"] = value["selected_task"] == target_task
            legacy_replay = {
                **legacy_decision,
                "correct": legacy_decision["selected_task"] == target_task,
                "method_ids": sorted(LEGACY_EQUIVALENT),
                "status": "LEGACY_EQUIVALENT_REPLAY",
            }
            trials.append(
                {
                    "prefix_episode_count": prefix,
                    "selections": selections,
                    "legacy_equivalent_replay": legacy_replay,
                }
            )
        record = {
            "schema": SCHEMA,
            "scope": SCOPE,
            "formal": False,
            "p6_30_policy_result": False,
            "new_encoder_training": False,
            "new_policy_rollout": False,
            "target_task": target_task,
            "bank_index": bank,
            "query_dataset_manifest_sha256": query_manifest_sha,
            "config": config,
            "trials": trials,
        }
        record_shas[key] = _publish_or_verify(path, record, bool(args.resume))
        records.append(record)

    method_summary: dict[str, Any] = {}
    for method in EVALUATED:
        by_prefix: dict[str, Any] = {}
        all_correct = 0
        all_count = 0
        for prefix in prefixes:
            values = [
                bool(
                    trial["legacy_equivalent_replay"]["correct"]
                    if method in LEGACY_EQUIVALENT
                    else trial["selections"][method]["correct"]
                )
                for record in records
                for trial in record["trials"]
                if int(trial["prefix_episode_count"]) == prefix
            ]
            correct = sum(values)
            by_prefix[str(prefix)] = {"correct": correct, "trials": len(values), "accuracy": correct / len(values)}
            all_correct += correct
            all_count += len(values)
        method_summary[method] = {
            "status": (
                "LEGACY_EQUIVALENT_REPLAY"
                if method in LEGACY_EQUIVALENT
                else "DEVELOPMENT_DIAGNOSTIC"
                if method == "D-RFF-KME"
                else "AVAILABLE"
            ),
            "correct": all_correct,
            "trials": all_count,
            "accuracy": all_correct / all_count,
            "by_prefix": by_prefix,
        }
    method_summary["B0"]["expected_accuracy"] = 1.0 / len(tasks)
    for method, reason in UNAVAILABLE.items():
        method_summary[method] = {"status": "UNAVAILABLE", "reason": reason}

    summary = {
        "schema": SCHEMA,
        "status": "LEGACY_SIX_TASK_BASELINE_COMPARISON_COMPLETE",
        "scope": SCOPE,
        "formal_run_authorized": False,
        "p6_30_policy_result": False,
        "new_encoder_training": False,
        "new_policy_rollout": False,
        "task_count": len(tasks),
        "query_bank_task_count": len(records),
        "prefix_trial_count_per_method": len(records) * len(prefixes),
        "prefix_grid": list(prefixes),
        "config": config,
        "method_results": method_summary,
        "legacy_equivalence_note": "B2, A-Env, and M02/B5 share the verified v0 CORRO TaskSpec nearest-neighbour path in this legacy pool; this is reported, not counted as three independent algorithms.",
        "source_assets": {
            **source_meta["identity"],
            "source_dataset_manifest_sha256": dict(sorted(source_manifest_sha.items())),
            "retrieval_metrics_sha256": sha256_file(root / "reports" / "retrieval_metrics.json"),
        },
        "query_record_sha256": dict(sorted(record_shas.items())),
    }
    _publish_or_verify(output / "summary.json", summary, bool(args.resume))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare simple retrieval baselines on the real legacy six-task traces (development only)"
    )
    parser.add_argument("--legacy-v0-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-queries", type=int, help="limit task-bank datasets; each still evaluates all prefixes")
    parser.add_argument("--rff-dim", type=int, default=64)
    parser.add_argument("--transitions-per-episode", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_queries is not None and args.max_queries <= 0:
        raise SystemExit("--max-queries must be positive")
    if args.rff_dim <= 0 or args.transitions_per_episode <= 0 or args.seed < 0:
        raise SystemExit("RFF dimension, samples per episode, and seed must be valid")
    summary = run(args)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
