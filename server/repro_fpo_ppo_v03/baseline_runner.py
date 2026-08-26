"""Minimal v0.3 baseline runner and honest legacy-v0 replay.

The default path reuses the completed legacy six-task M02/B5 reports without
launching policy rollouts or recomputing their KME matrix.  It is explicitly a
development replay, never a 30-policy v0.3 P6 result.

An exact-nine development run is available only when a server-owned factory
supplies the existing typed v0.3 inputs.  This file adds no scientific schema
or data contract: it calls ``fit_baseline_suite`` and
``run_baseline_ranking`` directly and persists their existing dictionaries.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from policy_learnware_v0.hashing import canonical_json_bytes, sha256_file
from policy_learnware_v0.io import atomic_write_json
from policy_learnware_v0.v03.anonymous_market import build_anonymous_selector_view
from policy_learnware_v0.v03.baselines import (
    DEVELOPMENT_SMOKE_MODE,
    REQUIRED_BASELINE_METHOD_IDS,
    V03BaselineQuery,
    fit_baseline_suite,
    run_baseline_ranking,
)
from policy_learnware_v0.v03.compute import JointDistanceRun
from policy_learnware_v0.v03.source_market import V03SourcePolicyMarket


LEGACY_SCOPE = "development/legacy-six-task"
EXACT9_SCOPE = "development/v03-exact-nine"
LEGACY_METHOD_ID = "M02/B5"
_DISTANCE_METHODS = frozenset({"B3b", "A-Env", "M02/B5"})
_RAW_METHODS = frozenset({"B0", "B1", "B2", "B3a", "B3b", "B4a", "B4b"})
_QUERY_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _json_file(path: Path, where: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{where} must be a regular non-symlink JSON file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {where}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must contain one JSON object")
    return value


def _publish_or_verify(path: Path, value: Mapping[str, Any], *, resume: bool) -> str:
    expected = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if not resume:
            raise ValueError(f"output exists; use --resume: {path}")
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"resume bytes differ from recomputation: {path}")
        return sha256_file(path)
    return atomic_write_json(path, value, overwrite=False)


def _legacy_paths(root: Path) -> Mapping[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("--legacy-v0-root must be a real directory")
    paths = {
        "retrieval": root / "reports" / "retrieval_metrics.json",
        "deployment": root / "reports" / "deployment_metrics.json",
        "ranking": root / "reports" / "reduced_unreduced_ranking.json",
        "pool": root / "pool_manifest.json",
        "encoder": root / "protocol" / "encoder_manifest.json",
        "normalization": root / "protocol" / "normalization_manifest.json",
        "kernel": root / "protocol" / "kernel_manifest.json",
    }
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"legacy {label} artifact is absent or unsafe: {path}")
    return paths


def _complete_report(value: Mapping[str, Any], where: str) -> None:
    if value.get("complete") is not True or value.get("gate_passed") is not True:
        raise ValueError(f"legacy {where} is not a completed passing report")


def _formal_unavailable(market_loaded: bool) -> Mapping[str, Sequence[str]]:
    shared = [] if market_loaded else ["missing real 30-entry V03SourcePolicyMarket"]
    return {
        "B0": (*shared, "missing market-bound Raw source index", "exact-nine fitter needs a real development freeze"),
        "B1": (*shared, "missing market-bound Raw source index", "exact-nine fitter needs a real development freeze"),
        "B2": (*shared, "missing six current-nominal TaskSpec features/champion map", "missing v03 query features"),
        "B3a": (*shared, "missing 30 Raw source moment features", "missing v03 query features"),
        "B3b": (*shared, "missing Raw source-reduced index and query-empirical distance runs"),
        "B4a": (*shared, "missing 24-context x 30-policy development labels/features"),
        "B4b": (*shared, "missing 24-context x 30-policy development labels/features"),
        "A-Env": (*shared, "missing frozen-CORRO source/query representation index and distance runs"),
        "M02/B5": (*shared, "missing frozen-CORRO source/query representation index and distance runs"),
    }


def replay_legacy_v0(
    *,
    legacy_v0_root: Path,
    output_dir: Path,
    max_queries: int | None,
    resume: bool,
    market_loaded: bool,
) -> Mapping[str, Any]:
    """Replay immutable legacy reports into query-local development records."""

    paths = _legacy_paths(legacy_v0_root)
    retrieval = _json_file(paths["retrieval"], "legacy retrieval report")
    deployment = _json_file(paths["deployment"], "legacy deployment report")
    ranking = _json_file(paths["ranking"], "legacy reduced/unreduced report")
    pool = _json_file(paths["pool"], "legacy pool manifest")
    for label in ("encoder", "normalization", "kernel"):
        manifest = _json_file(paths[label], f"legacy {label} manifest")
        if manifest.get("complete") is not True:
            raise ValueError(f"legacy {label} manifest is incomplete")
    _complete_report(retrieval, "retrieval")
    _complete_report(deployment, "deployment")
    _complete_report(ranking, "reduced/unreduced ranking")
    if pool.get("complete") is not True or pool.get("entry_count") != 6:
        raise ValueError("legacy replay requires the completed six-entry v0 pool")
    if (
        retrieval.get("pool_id") != pool.get("pool_id")
        or deployment.get("pool_id") != pool.get("pool_id")
        or retrieval.get("public_pool_digest") != pool.get("public_pool_digest")
        or deployment.get("public_pool_digest") != pool.get("public_pool_digest")
    ):
        raise ValueError("legacy report/pool identities differ")
    if deployment.get("retrieval_metrics_sha256") != sha256_file(paths["retrieval"]):
        raise ValueError("legacy deployment report does not bind retrieval report bytes")

    retrieval_rows = retrieval.get("queries")
    deployment_rows = deployment.get("queries")
    ranking_rows = ranking.get("queries")
    if not isinstance(retrieval_rows, list) or not isinstance(deployment_rows, list):
        raise ValueError("legacy reports do not contain query rows")
    if not isinstance(ranking_rows, Mapping):
        raise ValueError("legacy reduced/unreduced report lacks per-task rows")
    by_query = {row.get("query_id"): row for row in deployment_rows if isinstance(row, Mapping)}
    if len(by_query) != len(deployment_rows):
        raise ValueError("legacy deployment query IDs are missing or duplicated")
    rows = sorted(
        (row for row in retrieval_rows if isinstance(row, Mapping)),
        key=lambda row: str(row.get("query_id")),
    )
    if len(rows) != len(retrieval_rows) or not rows:
        raise ValueError("legacy retrieval query rows are invalid")
    if max_queries is not None:
        rows = rows[:max_queries]

    query_file_shas: dict[str, str] = {}
    selected_returns: list[float] = []
    correct_count = 0
    for row in rows:
        query_id = row.get("query_id")
        task = row.get("target_task")
        if not isinstance(query_id, str) or _QUERY_ID.fullmatch(query_id) is None:
            raise ValueError("legacy query ID is unsafe")
        if not isinstance(task, str) or task not in ranking_rows:
            raise ValueError(f"legacy query {query_id} lacks its task ranking")
        deployed = by_query.get(query_id)
        if not isinstance(deployed, Mapping):
            raise ValueError(f"legacy query {query_id} lacks deployment evidence")
        if (
            deployed.get("target_task") != task
            or deployed.get("selected_opaque_id") != row.get("selected_opaque_id")
            or deployed.get("status") != "success"
            or deployed.get("deployment_failure") is not None
        ):
            raise ValueError(f"legacy query {query_id} retrieval/deployment evidence differs")
        selection_path = legacy_v0_root / "queries" / query_id / "selection_result.json"
        deployment_path = legacy_v0_root / "queries" / query_id / "deployment_result.json"
        if sha256_file(selection_path) != row.get("selection_sha256"):
            raise ValueError(f"legacy query {query_id} selection bytes drifted")
        if sha256_file(deployment_path) != deployed.get("deployment_sha256"):
            raise ValueError(f"legacy query {query_id} deployment bytes drifted")
        task_ranking = ranking_rows[task]
        if not isinstance(task_ranking, Mapping) or task_ranking.get("top1_agrees") is not True:
            raise ValueError(f"legacy task {task} lacks reduced/unreduced top-one agreement")
        record = {
            "scope": LEGACY_SCOPE,
            "formal": False,
            "p6_exact_nine": False,
            "method_id": LEGACY_METHOD_ID,
            "replay_only": True,
            "new_training_or_rollout_executed": False,
            "query_id": query_id,
            "retrieval": dict(row),
            "deployment": dict(deployed),
            "reduced_unreduced_task_ranking": dict(task_ranking),
            "source_files": {
                "selection_result": str(selection_path.resolve()),
                "deployment_result": str(deployment_path.resolve()),
            },
        }
        query_file_shas[query_id] = _publish_or_verify(
            output_dir / "legacy_queries" / f"{query_id}.json",
            record,
            resume=resume,
        )
        correct_count += int(row.get("correct") is True)
        selected_returns.append(float(deployed["mean_return"]))

    summary = {
        "status": "LEGACY_V0_M02_REPLAY_COMPLETE",
        "scope": LEGACY_SCOPE,
        "formal_run_authorized": False,
        "p6_exact_nine_complete": False,
        "method_ids_executed": [LEGACY_METHOD_ID],
        "query_count": len(rows),
        "correct_retrieval_count": correct_count,
        "retrieval_accuracy": correct_count / len(rows),
        "mean_selected_policy_return": sum(selected_returns) / len(selected_returns),
        "new_training_or_rollout_executed": False,
        "source_report_sha256": {
            label: sha256_file(paths[label])
            for label in ("retrieval", "deployment", "ranking")
        },
        "query_record_sha256": dict(sorted(query_file_shas.items())),
        "v03_exact_nine": {
            "runnable_methods": [],
            "unavailable": _formal_unavailable(market_loaded),
        },
    }
    _publish_or_verify(output_dir / "summary.json", summary, resume=resume)
    return summary


def _load_market(public_path: Path | None, private_path: Path | None) -> V03SourcePolicyMarket | None:
    if public_path is None and private_path is None:
        return None
    if public_path is None or private_path is None:
        raise ValueError("supply both public and private market manifests")
    return V03SourcePolicyMarket.from_manifests(
        _json_file(public_path, "public policy market"),
        _json_file(private_path, "deployment-private market registry"),
    )


def _factory(spec: str) -> Callable[..., Mapping[str, Any]]:
    if spec.count(":") != 1:
        raise ValueError("--prepared-input-factory must use module:callable")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute or "." in attribute:
        raise ValueError("prepared input factory name is invalid")
    function = getattr(importlib.import_module(module_name), attribute)
    if not callable(function):
        raise ValueError("prepared input factory is not callable")
    return function


def _subset_query_inputs(
    queries_by_method: Mapping[str, Mapping[str, V03BaselineQuery]],
    distance_runs_by_method: Mapping[str, Mapping[str, JointDistanceRun]],
    max_queries: int | None,
) -> tuple[Mapping[str, Mapping[str, V03BaselineQuery]], Mapping[str, Mapping[str, JointDistanceRun]], tuple[str, ...]]:
    if set(queries_by_method) != set(REQUIRED_BASELINE_METHOD_IDS):
        raise ValueError("prepared query inputs must cover the exact nine methods")
    query_sets = [set(queries_by_method[method]) for method in REQUIRED_BASELINE_METHOD_IDS]
    if not query_sets[0] or any(value != query_sets[0] for value in query_sets[1:]):
        raise ValueError("prepared baseline methods must cover the same non-empty queries")
    query_ids = tuple(sorted(query_sets[0]))
    if max_queries is not None:
        query_ids = query_ids[:max_queries]
    selected = set(query_ids)
    queries = {
        method: {query: value for query, value in rows.items() if query in selected}
        for method, rows in queries_by_method.items()
    }
    for method, rows in queries.items():
        if not all(
            isinstance(value, V03BaselineQuery) and value.opaque_query_id == query_id
            for query_id, value in rows.items()
        ):
            raise ValueError(f"prepared {method} query keys are incomplete or untyped")
    if set(distance_runs_by_method) != set(_DISTANCE_METHODS):
        raise ValueError("prepared distance runs must cover B3b, A-Env, and M02/B5 only")
    distances = {
        method: {query: value for query, value in rows.items() if query in selected}
        for method, rows in distance_runs_by_method.items()
    }
    for method, rows in distances.items():
        if set(rows) != selected or not all(isinstance(value, JointDistanceRun) for value in rows.values()):
            raise ValueError(f"prepared {method} distance runs are incomplete or untyped")
    for query_id in query_ids:
        if distances["A-Env"][query_id].run_digest != distances["M02/B5"][query_id].run_digest:
            raise ValueError("A-Env and M02/B5 must reuse one CORRO distance run")
    return queries, distances, query_ids


def run_exact_nine(
    *,
    legacy_v0_root: Path,
    output_dir: Path,
    market: V03SourcePolicyMarket | None,
    prepared_input_factory: str,
    max_queries: int | None,
    resume: bool,
) -> Mapping[str, Any]:
    supplied = _factory(prepared_input_factory)(
        legacy_v0_root=legacy_v0_root,
        market=market,
    )
    if not isinstance(supplied, Mapping):
        raise ValueError("prepared input factory must return a mapping")
    required = {
        "market",
        "raw_index",
        "corro_index",
        "development_view",
        "development_freeze",
        "legacy_task_specs",
        "nominal_champions",
        "raw_moment_features",
        "queries_by_method",
        "distance_runs_by_method",
    }
    missing = sorted(required - set(supplied))
    if missing:
        raise ValueError(f"prepared input factory omitted: {missing}")
    if market is not None and supplied["market"].policy_market_id != market.policy_market_id:
        raise ValueError("prepared market differs from explicit market manifests")
    market = supplied["market"]
    freeze = supplied["development_freeze"]
    if freeze.execution_mode != DEVELOPMENT_SMOKE_MODE:
        raise ValueError("this minimal runner accepts development baseline freezes only")
    queries, distances, query_ids = _subset_query_inputs(
        supplied["queries_by_method"], supplied["distance_runs_by_method"], max_queries
    )
    artifacts = fit_baseline_suite(
        market=market,
        raw_index=supplied["raw_index"],
        corro_index=supplied["corro_index"],
        development_view=supplied["development_view"],
        development_freeze=freeze,
        legacy_task_specs=supplied["legacy_task_specs"],
        nominal_champions=supplied["nominal_champions"],
        raw_moment_features=supplied["raw_moment_features"],
    )
    raw_view = build_anonymous_selector_view(market, supplied["raw_index"])
    corro_view = build_anonymous_selector_view(market, supplied["corro_index"])
    _publish_or_verify(
        output_dir / "selector_artifacts.json",
        {method: artifacts[method].to_dict() for method in REQUIRED_BASELINE_METHOD_IDS},
        resume=resume,
    )
    query_shas: dict[str, str] = {}
    for query_id in query_ids:
        ranking_rows = []
        for method in REQUIRED_BASELINE_METHOD_IDS:
            query = queries[method][query_id]
            if query.execution_mode != DEVELOPMENT_SMOKE_MODE:
                raise ValueError("prepared query is not development-scoped")
            ranking = run_baseline_ranking(
                query=query,
                selector_view=raw_view if method in _RAW_METHODS else corro_view,
                artifact=artifacts[method],
                distance_run=distances[method][query_id] if method in _DISTANCE_METHODS else None,
            )
            ranking_rows.append(ranking.to_dict())
        query_shas[query_id] = _publish_or_verify(
            output_dir / "v03_queries" / f"{query_id}.json",
            {
                "scope": EXACT9_SCOPE,
                "formal": False,
                "p6_exact_nine": True,
                "opaque_query_id": query_id,
                "rankings": ranking_rows,
            },
            resume=resume,
        )
    summary = {
        "status": "V03_DEVELOPMENT_EXACT_NINE_COMPLETE",
        "scope": EXACT9_SCOPE,
        "formal_run_authorized": False,
        "p6_exact_nine_complete": True,
        "method_ids_executed": list(REQUIRED_BASELINE_METHOD_IDS),
        "query_count": len(query_ids),
        "ranking_count": len(query_ids) * len(REQUIRED_BASELINE_METHOD_IDS),
        "query_record_sha256": dict(sorted(query_shas.items())),
    }
    _publish_or_verify(output_dir / "summary.json", summary, resume=resume)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal v0.3 development baselines")
    parser.add_argument("--legacy-v0-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--public-policy-market", type=Path)
    parser.add_argument("--deployment-private-registry", type=Path)
    parser.add_argument("--prepared-input-factory")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_queries is not None and args.max_queries <= 0:
        raise SystemExit("--max-queries must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        market = _load_market(
            args.public_policy_market, args.deployment_private_registry
        )
        if args.prepared_input_factory:
            summary = run_exact_nine(
                legacy_v0_root=args.legacy_v0_root.expanduser().resolve(),
                output_dir=output,
                market=market,
                prepared_input_factory=args.prepared_input_factory,
                max_queries=args.max_queries,
                resume=bool(args.resume),
            )
        else:
            summary = replay_legacy_v0(
                legacy_v0_root=args.legacy_v0_root.expanduser().resolve(),
                output_dir=output,
                max_queries=args.max_queries,
                resume=bool(args.resume),
                market_loaded=market is not None,
            )
    except Exception as error:
        failure = {
            "status": "BASELINE_RUN_FAILED_CLOSED",
            "formal_run_authorized": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "v03_exact_nine": {
                "runnable_methods": [],
                "unavailable": _formal_unavailable(False),
            },
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "replay_legacy_v0", "run_exact_nine"]
