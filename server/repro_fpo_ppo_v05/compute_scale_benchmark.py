"""Source-only one-factor compute scaling for the v0.5 method matrix.

This is an exploratory post-truth *compute* benchmark, not another retrieval
runner.  It authenticates the frozen r4 assets through the production loader,
uses only source train/validation evidence, and never opens query truth.  Each
OFAT cell is immutable and independently resumable.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from policy_learnware_v0.hashing import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    sha256_ndarrays,
)
from policy_learnware_v0.io import atomic_write_json
from policy_learnware_v0.rkme.distance import empirical_to_reduced_distance
from policy_learnware_v0.rkme.empirical import build_empirical_kme, empirical_mmd2
from policy_learnware_v0.rkme.gaussian import GaussianKernel, calibrate_bandwidth
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v05.ablations import (
    ABLATION_METHOD_IDS,
    B0_RANDOM,
    B3A_RAW_MOMENT_NN,
    RFF_LOGREG,
    RFF_RIDGE,
    SUMMARY_NN,
    SWE_1024_NN,
    DeterministicRandomRanker,
    FixedFeatureLogReg,
    FixedFeatureRidge,
    RFFEpisodeFeatures,
    RawMomentNN,
    SWE1024NN,
    SummaryPrototypeNN,
    episode_balanced_moment_vector,
    nested_row_order,
    summary_episode_features,
)
from policy_learnware_v0.v05.classifiers import (
    EMPIRICAL_MMD_NN,
    KME_KRR,
    P0_METHOD_IDS,
    RAW_DELTA_RKME,
    RFF_KME_NN,
    SUMMARY_LOGREG,
    SWE_NN,
    EmpiricalMMDNN,
    EpisodeBank,
    KMEKRR,
    RFFKMENN,
    RawDeltaRKMENN,
    SWENN,
    SummaryLogReg,
)
from policy_learnware_v0.v05.specifications import RFFMap, SWEMap
from server.repro_fpo_ppo_v02.provenance import load_strict_json
from server.repro_fpo_ppo_v05.blind_query_bank import project_verified_source_banks
from server.repro_fpo_ppo_v05.environment_classifier_runner import (
    Q0_COMMON_GAUSSIAN_OPEN_LOOP,
    V05RunnerError,
    _UniqueKeyLoader,
    _canonical_source_stage,
    _load_frozen_r4_assets,
    _rss_bytes,
    load_development_config,
)


PLAN_SCHEMA = "policy-learnware.v05-ablation-plan.v1"
CELL_SCHEMA = "policy-learnware.v05-compute-scale-cell.v1"
COMPUTE_RUN_SCHEMA = "policy-learnware.v05-compute-scale-run.v2"
COMPUTE_BOOTSTRAP_SCHEMA = "policy-learnware.v05-compute-scale-bootstrap.v2"
COMPUTE_SUMMARY_SCHEMA = "policy-learnware.v05-compute-scale-summary.v2"
EXPLORATORY_SCOPE = "SECONDARY_EXPLORATORY_POST_TRUTH"
_EXPECTED_PLAN_DIGEST = (
    "e5e231c7ea35e42568bf9f884cc03cb930b52a4cb46c7ef37c40fd12c2ace2ae"
)
ALL_METHOD_IDS = P0_METHOD_IDS + ABLATION_METHOD_IDS
RFF_FAMILY = (RFF_KME_NN, RFF_LOGREG, RFF_RIDGE)
SWE_SWEPT_FAMILY = (SWE_NN,)


def _exploratory_identity(schema: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "scope": EXPLORATORY_SCOPE,
        "formal_confirmatory": False,
    }


def _validate_bootstrap_timing(value: Any, where: str) -> None:
    names = {
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_before_bytes",
        "peak_rss_after_bytes",
    }
    timing = _exact(value, names, where)
    for name in ("wall_seconds", "cpu_seconds"):
        number = timing[name]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise ComputeScaleError(f"{where} {name} is invalid")
    for name in ("peak_rss_before_bytes", "peak_rss_after_bytes"):
        number = timing[name]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ComputeScaleError(f"{where} {name} is invalid")


def _validate_bootstrap(value: Mapping[str, Any], stable: Mapping[str, Any]) -> str:
    expected = set(stable) | {
        "schema",
        "r4_asset_load",
        "full32_canonicalize",
        "peak_rss_bytes",
        "bootstrap_digest",
    }
    bootstrap = _exact(value, expected, "resume bootstrap")
    if bootstrap["schema"] != COMPUTE_BOOTSTRAP_SCHEMA:
        raise ComputeScaleError("resume bootstrap schema differs")
    if {key: bootstrap[key] for key in stable} != dict(stable):
        raise ComputeScaleError("resume bootstrap source closure differs")
    _validate_bootstrap_timing(bootstrap["r4_asset_load"], "r4_asset_load")
    _validate_bootstrap_timing(bootstrap["full32_canonicalize"], "full32_canonicalize")
    peak_rss = bootstrap["peak_rss_bytes"]
    if isinstance(peak_rss, bool) or not isinstance(peak_rss, int) or peak_rss < 0:
        raise ComputeScaleError("resume bootstrap peak_rss_bytes is invalid")
    unsigned = {key: bootstrap[key] for key in bootstrap if key != "bootstrap_digest"}
    digest = sha256_json(unsigned)
    if bootstrap["bootstrap_digest"] != digest:
        raise ComputeScaleError("bootstrap digest differs")
    return digest


class ComputeScaleError(ValueError):
    """A plan, source workload, or immutable benchmark artifact is invalid."""


def _clean_git_commit(repository_root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ComputeScaleError("cannot resolve the benchmark git identity") from error
    if (
        len(commit) != 40
        or any(item not in "0123456789abcdef" for item in commit)
        or dirty
    ):
        raise ComputeScaleError("benchmark launch requires one clean full git commit")
    return commit


def _exact(value: Any, names: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != names:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ComputeScaleError(
            f"{where} fields differ; missing={sorted(names - actual)}, "
            f"extra={sorted(actual - names)}"
        )
    return value


def _load_plan(path: str | Path) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ComputeScaleError("ablation plan is absent or unsafe")
    try:
        plan = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ComputeScaleError("ablation plan is not strict YAML") from error
    top = _exact(
        plan,
        {
            "schema",
            "base_commit",
            "development_config",
            "scope",
            "methods",
            "row_prefix",
            "source_fewshot",
            "query_fewshot",
            "compute_scale",
            "reporting",
        },
        "ablation plan",
    )
    digest = sha256_json(top)
    if top["schema"] != PLAN_SCHEMA or digest != _EXPECTED_PLAN_DIGEST:
        raise ComputeScaleError("ablation plan digest differs from the frozen plan")
    methods = top["methods"]
    if (
        tuple(methods["frozen_p0"]) != P0_METHOD_IDS
        or tuple(methods["additions"]) != ABLATION_METHOD_IDS
    ):
        raise ComputeScaleError("frozen 12-method order differs")
    config_ref = _exact(
        top["development_config"],
        {"relative_path", "file_sha256", "canonical_digest"},
        "development_config",
    )
    relative = Path(str(config_ref["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ComputeScaleError("development config path must be repository-relative")
    repo_root = source.parent.parent
    config_path = repo_root / relative
    if sha256_file(config_path) != config_ref["file_sha256"]:
        raise ComputeScaleError("development config file SHA differs")
    config, config_digest = load_development_config(config_path)
    if config_digest != config_ref["canonical_digest"]:
        raise ComputeScaleError("development config canonical digest differs")
    return dict(top), digest, config, config_digest


def _ofat_nodes(scale: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    base = dict(scale["base"])
    nodes: list[dict[str, Any]] = [{"cell_id": "BASE", "factor": "BASE", **base}]
    families = (
        ("market_anchors", "A", scale["market_anchors"]),
        ("rows_per_episode", "N", scale["rows_per_episode"]),
        ("rff_frequency_count", "RFF_M", scale["rff_frequency_count"]),
        ("swe_direction_count", "SWE_L", scale["swe_direction_count"]),
        ("swe_quantile_count", "SWE_Q", scale["swe_quantile_count"]),
    )
    for field, label, values in families:
        for value in values:
            if value == base[field]:
                continue
            nodes.append(
                {
                    "cell_id": f"{label}_{int(value):04d}",
                    "factor": field,
                    **base,
                    field: int(value),
                }
            )
    for item in scale["train_episode_nodes"]:
        if all(item[key] == base[key] for key in item):
            continue
        nodes.append(
            {
                "cell_id": (
                    f"E_{int(item['train_episodes']):02d}_"
                    f"V_{int(item['validation_episodes']):02d}"
                ),
                "factor": "train_validation_episodes",
                **base,
                **dict(item),
            }
        )
    return tuple(
        sorted(nodes, key=lambda row: (row["cell_id"] != "BASE", row["cell_id"]))
    )


def _slice_bank(
    bank: EpisodeBank,
    episodes: int,
    rows: int,
    *,
    public_seed: int,
    parent_membership_digest: str,
    physical_episode_start: int,
) -> EpisodeBank:
    if episodes > bank.episode_count:
        raise ComputeScaleError("source episode slice exceeds its frozen role")
    chunks = []
    for index in range(episodes):
        episode = bank.episode(index)
        if episode.shape[0] < rows:
            raise ComputeScaleError("source row slice exceeds a frozen episode")
        order = nested_row_order(
            parent_membership_digest,
            physical_episode_start + index,
            public_seed,
            row_count=episode.shape[0],
        )
        chunks.append(episode[np.asarray(order[:rows], dtype=np.int64)])
    return EpisodeBank(
        np.concatenate(chunks, axis=0),
        np.arange(0, (episodes + 1) * rows, rows, dtype=np.int64),
    )


def _task_balanced_anchors(assets: Any, count: int) -> tuple[str, ...]:
    by_task: dict[str, list[str]] = {}
    for anchor, task in assets.task_by_anchor.items():
        by_task.setdefault(str(task), []).append(str(anchor))
    if len(by_task) != 6 or count % 6 or count < 6 or count > 30:
        raise ComputeScaleError("market A is not balanced over the six frozen tasks")
    per_task = count // 6
    selected = tuple(
        anchor
        for task in sorted(by_task)
        for anchor in sorted(by_task[task])[:per_task]
    )
    if len(selected) != count or len(set(selected)) != count:
        raise ComputeScaleError("task-balanced market selection is incomplete")
    return selected


def _validation_workload(
    assets: Any,
    validation: Mapping[str, EpisodeBank],
    selected: Sequence[str],
    budget: int,
) -> tuple[EpisodeBank, ...]:
    by_task: dict[str, list[str]] = {}
    for anchor in selected:
        by_task.setdefault(str(assets.task_by_anchor[anchor]), []).append(anchor)
    if set(by_task) != set(assets.task_by_anchor.values()):
        raise ComputeScaleError("validation workload does not cover all frozen tasks")
    return tuple(
        validation[sorted(by_task[task])[0]].prefix(budget) for task in sorted(by_task)
    )


def _timed(call: Callable[[], Any]) -> tuple[Any, dict[str, float | int]]:
    rss_before = _rss_bytes()
    cpu = time.process_time()
    wall = time.perf_counter()
    result = call()
    timing = {
        "wall_seconds": max(0.0, time.perf_counter() - wall),
        "cpu_seconds": max(0.0, time.process_time() - cpu),
        "peak_rss_before_bytes": rss_before,
        "peak_rss_after_bytes": _rss_bytes(),
    }
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in timing.values()
    ):
        raise ComputeScaleError("non-finite timing or RSS measurement")
    return result, timing


def _repeat_timing(
    call: Callable[[], Any], *, warmup: int, repeats: int
) -> tuple[Any, dict[str, Any], tuple[Any, ...]]:
    for _ in range(warmup):
        call()
    values: list[Any] = []
    wall_values: list[float] = []
    cpu_values: list[float] = []
    for _ in range(repeats):
        cpu = time.process_time()
        wall = time.perf_counter()
        values.append(call())
        wall_values.append(max(0.0, time.perf_counter() - wall))
        cpu_values.append(max(0.0, time.process_time() - cpu))
    if any(not math.isfinite(value) for value in (*wall_values, *cpu_values)):
        raise ComputeScaleError("benchmark timing is non-finite")
    summary = {
        "warmup_repeats": warmup,
        "measured_repeats": repeats,
        "wall_seconds": wall_values,
        "cpu_seconds": cpu_values,
        "median_wall_seconds": statistics.median(wall_values),
        "median_cpu_seconds": statistics.median(cpu_values),
        "peak_rss_bytes": _rss_bytes(),
    }
    return values[-1], summary, tuple(values)


def _numeric_nbytes(value: Any, seen: set[int] | None = None) -> int:
    observed = set() if seen is None else seen
    identity = id(value)
    if identity in observed:
        return 0
    observed.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_numeric_nbytes(item, observed) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_numeric_nbytes(item, observed) for item in value)
    if is_dataclass(value):
        return sum(
            _numeric_nbytes(getattr(value, field.name), observed)
            for field in fields(value)
        )
    return 0


def _wall_total(value: Any) -> float:
    if not isinstance(value, Mapping):
        return 0.0
    if isinstance(value.get("wall_seconds"), (int, float)):
        return float(value["wall_seconds"])
    return math.fsum(_wall_total(item) for item in value.values())


def _model_digest(method_id: str, model: Any) -> str:
    digest = getattr(model, "model_digest", None)
    if isinstance(digest, str):
        return digest
    if method_id == RAW_DELTA_RKME:
        arrays = {
            f"{anchor}.{name}": np.asarray(getattr(spec, name))
            for anchor, spec in model.sources.items()
            for name in (
                "supports",
                "beta",
                "rkme_norm2",
                "empirical_norm2",
                "reduction_error",
            )
        }
    elif method_id == EMPIRICAL_MMD_NN:
        arrays = {
            f"{anchor}.{name}": np.asarray(getattr(spec, name))
            for anchor, spec in model.sources.items()
            for name in ("points", "weights", "episode_offsets", "norm2")
        }
    elif method_id in {RFF_KME_NN, SWE_NN, SWE_1024_NN}:
        map_value = getattr(model, "rff_map", getattr(model, "swe_map", None))
        return sha256_json(
            {
                "method_id": method_id,
                "map_digest": map_value.map_digest,
                "prototypes_sha256": sha256_ndarrays(dict(model.prototypes)),
            }
        )
    else:
        raise ComputeScaleError(f"model digest is unavailable for {method_id}")
    return sha256_json(
        {
            "method_id": method_id,
            "bandwidth": model.bandwidth,
            "protocol_id": model.protocol_id,
            "arrays_sha256": sha256_ndarrays(arrays),
        }
    )


def _score_digest(
    method_id: str,
    values: Sequence[Mapping[str, float]],
    expected_policy_ids: Sequence[str],
) -> str:
    expected = tuple(sorted(expected_policy_ids))
    if len(expected) < 2 or len(set(expected)) != len(expected):
        raise ComputeScaleError("expected policy IDs are not unique")
    rows = []
    for query_index, scores in enumerate(values):
        if set(scores) != set(expected) or any(
            not math.isfinite(float(value)) for value in scores.values()
        ):
            raise ComputeScaleError(
                f"{method_id} emitted a malformed policy score vector"
            )
        rows.append(
            {
                "query_index": query_index,
                "scores": [[key, float(scores[key])] for key in expected],
            }
        )
    return sha256_json({"method_id": method_id, "score_rows": rows})


def _method_scope(factor: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if factor == "rff_frequency_count":
        applicable = RFF_FAMILY
    elif factor in {"swe_direction_count", "swe_quantile_count"}:
        applicable = SWE_SWEPT_FAMILY
    else:
        applicable = ALL_METHOD_IDS
    return applicable, tuple(
        method for method in ALL_METHOD_IDS if method not in applicable
    )


def _ensure_output(
    path: str | Path, *, resume: bool, protected: Sequence[Path]
) -> Path:
    raw = Path(path).expanduser()
    for parent in (raw, *raw.parents):
        if parent.exists() and parent.is_symlink():
            raise ComputeScaleError("benchmark output has a symlink ancestor")
    destination = raw.resolve(strict=False)
    for other in protected:
        resolved = other.resolve()
        if (
            destination == resolved
            or destination in resolved.parents
            or resolved in destination.parents
        ):
            raise ComputeScaleError("benchmark output overlaps a frozen input root")
    if resume:
        if destination.is_symlink() or not destination.is_dir():
            raise ComputeScaleError("resume output directory is absent or unsafe")
    else:
        if destination.exists():
            raise ComputeScaleError("new benchmark output directory already exists")
        destination.mkdir(parents=True)
    return destination


def _benchmark_output_layout(
    output: Path, nodes: Sequence[Mapping[str, Any]]
) -> tuple[Path, Path, Path, dict[str, Path], Path]:
    manifest = output / "manifest.json"
    bootstrap = output / "bootstrap.json"
    cell_root = output / "cells"
    cells = {
        str(node["cell_id"]): cell_root / f"{node['cell_id']}.json" for node in nodes
    }
    summary = output / "summary.json"
    known = (manifest, bootstrap, cell_root, summary, *cells.values())
    if any(path.is_symlink() for path in known):
        raise ComputeScaleError("benchmark output layout contains a symlink")
    if cell_root.exists() and not cell_root.is_dir():
        raise ComputeScaleError("benchmark cell root is not a directory")
    resolved = output.resolve()
    if any(not path.resolve(strict=False).is_relative_to(resolved) for path in known):
        raise ComputeScaleError("benchmark output layout escapes its root")
    if set(output.rglob("*")) - set(known):
        raise ComputeScaleError("benchmark output contains unexpected artifacts")
    return manifest, bootstrap, cell_root, cells, summary


def _cell_source_slices(
    node: Mapping[str, Any],
    *,
    assets: Any,
    projection: Any,
    plan: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, EpisodeBank], dict[str, EpisodeBank]]:
    selected = _task_balanced_anchors(assets, int(node["market_anchors"]))
    seed = int(plan["row_prefix"]["public_seed"])
    train = {
        anchor: _slice_bank(
            projection.source_train[anchor],
            int(node["train_episodes"]),
            int(node["rows_per_episode"]),
            public_seed=seed,
            parent_membership_digest=assets.parent_membership_digest[anchor],
            physical_episode_start=0,
        )
        for anchor in selected
    }
    validation = {
        anchor: _slice_bank(
            projection.source_validation[anchor],
            int(node["validation_episodes"]),
            int(node["rows_per_episode"]),
            public_seed=seed,
            parent_membership_digest=assets.parent_membership_digest[anchor],
            physical_episode_start=19,
        )
        for anchor in selected
    }
    return selected, train, validation


def _cell_input_digest(
    node: Mapping[str, Any],
    selected: Sequence[str],
    train: Mapping[str, EpisodeBank],
    validation: Mapping[str, EpisodeBank],
    *,
    plan_digest: str,
    config_digest: str,
    bootstrap_digest: str,
) -> str:
    return sha256_json(
        {
            "plan_digest": plan_digest,
            "config_digest": config_digest,
            "bootstrap_digest": bootstrap_digest,
            "node": dict(node),
            "market_subset_digest": sha256_json(list(selected)),
            "train_bank_digest": sha256_json(
                {key: bank.bank_digest for key, bank in train.items()}
            ),
            "validation_bank_digest": sha256_json(
                {key: bank.bank_digest for key, bank in validation.items()}
            ),
        }
    )


def _validate_completed_cell(
    cell: Mapping[str, Any],
    node: Mapping[str, Any],
    *,
    expected_input_digest: str,
    applicable: Sequence[str],
    not_applicable: Sequence[str],
) -> None:
    unsigned = {key: value for key, value in cell.items() if key != "cell_digest"}
    expected_na = {
        method: "NA_REPRESENTATION_FAMILY_UNAFFECTED_BY_OFAT_DIMENSION"
        for method in not_applicable
    }
    if (
        cell.get("schema") != CELL_SCHEMA
        or cell.get("status") != "COMPLETE"
        or cell.get("cell_id") != node["cell_id"]
        or cell.get("node") != dict(node)
        or cell.get("input_digest") != expected_input_digest
        or cell.get("applicable_methods") != list(applicable)
        or cell.get("not_applicable_methods") != expected_na
        or set(cell.get("methods", {})) != set(applicable)
        or cell.get("cell_digest") != sha256_json(unsigned)
    ):
        raise ComputeScaleError(f"completed cell changed: {node['cell_id']}")


# Method construction is kept in one concrete function so the timing loop cannot
# accidentally train through a different path than the ablation implementation.
def _fit_methods(
    applicable: Sequence[str],
    train: Mapping[str, EpisodeBank],
    validation: Mapping[str, EpisodeBank],
    labels: Mapping[str, str],
    *,
    bandwidth: float | None,
    normalizer_digest: str,
    node: Mapping[str, Any],
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Fit requested methods and return their typed query ingress functions."""

    models: dict[str, Any] = {}
    rows: dict[str, dict[str, Any]] = {}
    policy_train = {labels[anchor]: bank for anchor, bank in train.items()}

    def anchors_to_policies(values: Mapping[str, float]) -> dict[str, float]:
        if set(values) != set(labels):
            raise ComputeScaleError("anchor score vector coverage differs")
        return {labels[anchor]: float(values[anchor]) for anchor in sorted(labels)}

    def register(
        method_id: str,
        build: Callable[[], Any],
        encoder: Callable[[EpisodeBank], Any] | None,
        scorer_factory: Callable[[Any], Callable[[Any], Mapping[str, float]]],
        encode_scope: str,
        *,
        shared: Mapping[str, Any] | None = None,
    ) -> None:
        model, fit_timing = _timed(build)
        models[method_id] = (model, encoder, scorer_factory(model), encode_scope)
        rows[method_id] = {
            "model_fit": fit_timing,
            "source_representation_encode": (
                dict(shared)
                if shared is not None
                else {"status": "INCLUDED_IN_MODEL_FIT"}
            ),
        }

    if B0_RANDOM in applicable:
        register(
            B0_RANDOM,
            lambda: DeterministicRandomRanker(
                tuple(labels.values()), int(plan["methods"]["b0_public_seed"])
            ),
            lambda _: "PUBLIC_COMPUTE_QUERY_TOKEN",
            lambda model: lambda token: model.score(public_query_token=token),
            "UPSTREAM_PUBLIC_QUERY_TOKEN_NO_RAW_ACCESS",
        )
    if B3A_RAW_MOMENT_NN in applicable:
        register(
            B3A_RAW_MOMENT_NN,
            lambda: RawMomentNN.fit(train, labels),
            episode_balanced_moment_vector,
            lambda model: model.score_vector,
            "EPISODE_BALANCED_MEAN_STD_VECTOR",
        )
    if SUMMARY_NN in applicable:
        register(
            SUMMARY_NN,
            lambda: SummaryPrototypeNN.fit(train, labels),
            lambda bank: np.mean(summary_episode_features(bank), axis=0),
            lambda model: model.score_vector,
            "MEAN_OF_PER_EPISODE_SUMMARY_VECTOR",
        )

    if bandwidth is not None:
        kernel = GaussianKernel(bandwidth)

        def empirical_view(bank: EpisodeBank) -> Any:
            return build_empirical_kme(
                bank.points,
                kernel,
                episode_offsets=bank.episode_offsets,
                protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP,
                dataset_digest=bank.bank_digest,
            )

    if RAW_DELTA_RKME in applicable:
        assert bandwidth is not None

        def raw_scorer(model: RawDeltaRKMENN) -> Callable[[Any], dict[str, float]]:
            return lambda target: anchors_to_policies(
                {
                    anchor: -empirical_to_reduced_distance(target, source).distance
                    for anchor, source in model.sources.items()
                }
            )

        register(
            RAW_DELTA_RKME,
            lambda: RawDeltaRKMENN.fit(
                train,
                bandwidth=bandwidth,
                protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP,
                reducer_config=ReducerConfig(**dict(config["raw_delta_rkme"])),
            ),
            empirical_view,
            raw_scorer,
            "EPISODE_BALANCED_EMPIRICAL_KME_THEN_REDUCED_DISTANCE",
        )
    if EMPIRICAL_MMD_NN in applicable:
        assert bandwidth is not None

        def empirical_scorer(
            model: EmpiricalMMDNN,
        ) -> Callable[[Any], dict[str, float]]:
            return lambda target: anchors_to_policies(
                {
                    anchor: -math.sqrt(empirical_mmd2(target, source))
                    for anchor, source in model.sources.items()
                }
            )

        register(
            EMPIRICAL_MMD_NN,
            lambda: EmpiricalMMDNN.fit(
                train,
                bandwidth=bandwidth,
                protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP,
            ),
            empirical_view,
            empirical_scorer,
            "EPISODE_BALANCED_EMPIRICAL_KME",
        )
    if SUMMARY_LOGREG in applicable:
        summary_config = config["summary_logreg"]
        register(
            SUMMARY_LOGREG,
            lambda: SummaryLogReg.fit(
                train,
                labels,
                validation,
                l2_grid=summary_config["l2_grid"],
                max_iter=int(summary_config["max_iter"]),
                tolerance=float(summary_config["gradient_tolerance"]),
            ),
            summary_episode_features,
            lambda model: model.score_summaries,
            "TYPED_EPISODE_SUMMARY_ROWS",
        )
    if KME_KRR in applicable:
        assert bandwidth is not None
        register(
            KME_KRR,
            lambda: KMEKRR.fit(
                train,
                labels,
                validation,
                bandwidth=bandwidth,
                ridge_grid=config["kme_krr"]["ridge_grid"],
            ),
            None,
            lambda model: model.score,
            "NA_FULL_SUPPORT_SCORER_INGRESS_NO_SEPARATE_ENCODER",
        )

    if set(applicable) & set(RFF_FAMILY):
        assert bandwidth is not None
        rff_map, map_timing = _timed(
            lambda: RFFMap(
                input_dim=next(iter(train.values())).input_dim,
                bandwidth=bandwidth,
                normalization_digest=normalizer_digest,
                frequency_count=int(node["rff_frequency_count"]),
                public_seed=int(config["rff_kme_nn"]["public_seed"]),
            )
        )
        typed_features: tuple[
            dict[str, RFFEpisodeFeatures], dict[str, RFFEpisodeFeatures]
        ] | None = None
        typed_timing: dict[str, Any] | None = None
        if set(applicable) & {RFF_LOGREG, RFF_RIDGE}:
            typed_features, typed_timing = _timed(
                lambda: (
                    {
                        key: RFFEpisodeFeatures.from_bank(rff_map, bank)
                        for key, bank in train.items()
                    },
                    {
                        key: RFFEpisodeFeatures.from_bank(rff_map, bank)
                        for key, bank in validation.items()
                    },
                )
            )
        shared_rff_supervised = {
            "map_construction": map_timing,
            "typed_train_validation_features": (
                typed_timing
                if typed_timing is not None
                else {"status": "NA_NN_PROTOTYPE_FIT_ENCODES_INTERNALLY"}
            ),
        }
        if RFF_KME_NN in applicable:
            register(
                RFF_KME_NN,
                lambda: RFFKMENN.fit(policy_train, rff_map=rff_map),
                lambda bank: rff_map.embed(bank.points, bank.episode_offsets),
                lambda model: model.score_specification,
                "FIXED_RFF_KME_VECTOR",
                shared={"map_construction": map_timing},
            )
        if RFF_LOGREG in applicable:
            assert typed_features is not None
            train_features, validation_features = typed_features
            head = plan["methods"]["rff_logreg"]
            register(
                RFF_LOGREG,
                lambda: FixedFeatureLogReg.fit(
                    train_features,
                    labels,
                    validation_features,
                    l2_grid=head["l2_grid"],
                    max_iter=int(head["max_iter"]),
                    tolerance=float(head["gradient_tolerance"]),
                ),
                lambda bank: RFFEpisodeFeatures.from_bank(rff_map, bank),
                lambda model: model.score_features,
                "TYPED_RFF_EPISODE_FEATURES",
                shared=shared_rff_supervised,
            )
        if RFF_RIDGE in applicable:
            assert typed_features is not None
            train_features, validation_features = typed_features
            head = plan["methods"]["rff_ridge"]
            register(
                RFF_RIDGE,
                lambda: FixedFeatureRidge.fit(
                    train_features,
                    labels,
                    validation_features,
                    ridge_grid=head["ridge_grid"],
                ),
                lambda bank: RFFEpisodeFeatures.from_bank(rff_map, bank),
                lambda model: model.score_features,
                "TYPED_RFF_EPISODE_FEATURES",
                shared=shared_rff_supervised,
            )

    if SWE_NN in applicable:
        swe_map, map_timing = _timed(
            lambda: SWEMap(
                input_dim=next(iter(train.values())).input_dim,
                normalization_digest=normalizer_digest,
                direction_count=int(node["swe_direction_count"]),
                quantile_count=int(node["swe_quantile_count"]),
                public_seed=int(config["swe_nn"]["public_seed"]),
            )
        )
        register(
            SWE_NN,
            lambda: SWENN.fit(policy_train, swe_map=swe_map),
            lambda bank: swe_map.embed(bank.points, bank.episode_offsets),
            lambda model: model.score_specification,
            "FIXED_SWE_VECTOR",
            shared={"map_construction": map_timing},
        )
    if SWE_1024_NN in applicable:
        fixed = plan["methods"]["swe_1024"]
        swe1024_map, map_timing = _timed(
            lambda: SWEMap(
                input_dim=next(iter(train.values())).input_dim,
                normalization_digest=normalizer_digest,
                direction_count=int(fixed["direction_count"]),
                quantile_count=int(fixed["quantile_count"]),
                public_seed=int(fixed["public_seed"]),
            )
        )
        register(
            SWE_1024_NN,
            lambda: SWE1024NN.fit(policy_train, swe_map=swe1024_map),
            lambda bank: swe1024_map.embed(bank.points, bank.episode_offsets),
            lambda model: model.score_specification,
            "FIXED_SWE_1024_VECTOR",
            shared={"map_construction": map_timing},
        )
    if set(models) != set(applicable):
        raise ComputeScaleError("method construction did not cover the cell scope")
    return models, rows


def _run_cell(
    node: Mapping[str, Any],
    *,
    assets: Any,
    projection: Any,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    config_digest: str,
    plan_digest: str,
    bootstrap_digest: str,
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    applicable, not_applicable = _method_scope(str(node["factor"]))
    ((selected, train, validation), canonicalize_timing) = _timed(
        lambda: _cell_source_slices(
            node, assets=assets, projection=projection, plan=plan
        )
    )
    full_validation = {
        anchor: projection.source_validation[anchor] for anchor in selected
    }
    queries = _validation_workload(
        assets,
        full_validation,
        selected,
        int(node["query_budget_episodes"]),
    )
    labels_all = {
        binding.source_anchor_id: binding.opaque_certified_policy_id
        for binding in assets.certificate_manifest.bindings
    }
    labels = {anchor: labels_all[anchor] for anchor in selected}
    needs_bandwidth = bool(
        set(applicable) & {RAW_DELTA_RKME, EMPIRICAL_MMD_NN, KME_KRR, *RFF_FAMILY}
    )
    if needs_bandwidth:
        bandwidth, bandwidth_timing = _timed(
            lambda: calibrate_bandwidth(
                train,
                calibration_pairs=int(
                    config["measurement"]["gaussian_bandwidth"]["calibration_pairs"]
                ),
                seed=int(config["measurement"]["gaussian_bandwidth"]["public_seed"]),
            )
        )
    else:
        bandwidth = None
        bandwidth_timing = {
            "status": "NA_METHOD_FAMILY_DOES_NOT_USE_GAUSSIAN_BANDWIDTH"
        }
    models, fit_rows = _fit_methods(
        applicable,
        train,
        validation,
        labels,
        bandwidth=bandwidth,
        normalizer_digest=str(bootstrap["normalizer_digest"]),
        node=node,
        plan=plan,
        config=config,
    )
    method_rows: dict[str, Any] = {}
    repeats = int(plan["compute_scale"]["repeats"])
    warmup = int(plan["compute_scale"]["warmup_repeats"])
    for method_id in applicable:
        model, encoder, scorer, encode_scope = models[method_id]
        if encoder is None:
            encoded = queries
            encode_timing = {"status": encode_scope}
        else:
            encoded, encode_timing, _ = _repeat_timing(
                lambda encoder=encoder: tuple(encoder(query) for query in queries),
                warmup=warmup,
                repeats=repeats,
            )
        scores, score_timing, repeated_scores = _repeat_timing(
            lambda scorer=scorer, encoded=encoded: tuple(
                scorer(value) for value in encoded
            ),
            warmup=warmup,
            repeats=repeats,
        )
        expected_policy_ids = tuple(sorted(labels.values()))
        score_digest = _score_digest(method_id, scores, expected_policy_ids)
        if any(
            _score_digest(method_id, item, expected_policy_ids) != score_digest
            for item in repeated_scores
        ):
            raise ComputeScaleError(f"{method_id} score replay is not deterministic")
        method_rows[method_id] = {
            **fit_rows[method_id],
            "model_digest": _model_digest(method_id, model),
            "model_numeric_payload_bytes": _numeric_nbytes(model),
            "query_encode_scope": encode_scope,
            "query_encode": encode_timing,
            "encoded_numeric_payload_bytes": _numeric_nbytes(encoded),
            "query_score": score_timing,
            "score_vector_digest": score_digest,
            "score_vectors_finite": True,
            "candidate_count": len(selected),
            "query_count": len(queries),
        }
    train_rows = sum(bank.points.shape[0] for bank in train.values())
    validation_rows = sum(bank.points.shape[0] for bank in validation.values())
    query_rows = sum(bank.points.shape[0] for bank in queries)
    input_digest = _cell_input_digest(
        node,
        selected,
        train,
        validation,
        plan_digest=plan_digest,
        config_digest=config_digest,
        bootstrap_digest=bootstrap_digest,
    )
    result: dict[str, Any] = {
        "schema": CELL_SCHEMA,
        "status": "COMPLETE",
        "cell_id": node["cell_id"],
        "input_digest": input_digest,
        "node": dict(node),
        "source_only_no_query_truth": True,
        "market_subset_rule": "equal_anchors_per_task_in_frozen_order",
        "market_subset_digest": sha256_json(list(selected)),
        "applicable_methods": list(applicable),
        "not_applicable_methods": {
            method: "NA_REPRESENTATION_FAMILY_UNAFFECTED_BY_OFAT_DIMENSION"
            for method in not_applicable
        },
        "timing": {
            "shared_r4_asset_load": bootstrap["r4_asset_load"],
            "shared_full32_canonicalize": bootstrap["full32_canonicalize"],
            "cell_canonical_bank_slice": canonicalize_timing,
            "bandwidth_calibration": bandwidth_timing,
        },
        "workload": {
            "train_rows": train_rows,
            "validation_fit_rows": validation_rows,
            "validation_query_rows": query_rows,
            "train_numeric_payload_bytes": _numeric_nbytes(train),
            "validation_fit_numeric_payload_bytes": _numeric_nbytes(validation),
            "validation_query_numeric_payload_bytes": _numeric_nbytes(queries),
            "actual_new_acquisition_steps": 0,
        },
        "bandwidth": bandwidth,
        "methods": method_rows,
        "peak_rss_bytes": _rss_bytes(),
        "peak_rss_scope": "cumulative_process_upper_bound_not_fresh_cell",
        "numeric_payload_bytes_scope": "ndarray_payload_only_no_python_object_overhead",
        "source_preprocessing_scope": (
            "FROZEN_FULL_SOURCE_NORMALIZER_REUSED; CELL_TIMING STARTS FROM "
            "CANONICAL BANK SUBSETTING"
        ),
    }
    result["cell_digest"] = sha256_json(result)
    return result


def run_benchmark(
    *,
    plan_path: str | Path,
    completed_run_dir: str | Path,
    new_output_dir: str | Path,
    artifacts_root: str | Path | None = None,
    resume: bool,
) -> dict[str, Any]:
    plan, plan_digest, config, config_digest = _load_plan(plan_path)
    completed = Path(completed_run_dir).expanduser().resolve()
    actual_commit = _clean_git_commit(Path(plan_path).resolve().parent.parent)
    (assets, asset_timing) = _timed(
        lambda: _load_frozen_r4_assets(config, config_digest, artifacts_root)
    )
    output = _ensure_output(
        new_output_dir,
        resume=resume,
        protected=(completed, assets.r4_root, assets.v03_root),
    )
    nodes = _ofat_nodes(plan["compute_scale"])
    (
        manifest_path,
        bootstrap_path,
        cell_root,
        cell_paths,
        summary_path,
    ) = _benchmark_output_layout(output, nodes)
    manifest = {
        **_exploratory_identity(COMPUTE_RUN_SCHEMA),
        "actual_git_commit": actual_commit,
        "plan_digest": plan_digest,
        "plan_file_sha256": sha256_file(plan_path),
        "base_commit": plan["base_commit"],
        "development_config_digest": config_digest,
        "method_ids": list(ALL_METHOD_IDS),
        "cell_nodes": list(nodes),
        "source_workload": "frozen_r4_source_train_and_validation_only_no_truth",
        "actual_new_acquisition_steps": 0,
    }
    if manifest_path.exists():
        if not resume or canonical_json_bytes(
            load_strict_json(manifest_path)
        ) != canonical_json_bytes(manifest):
            raise ComputeScaleError("resume manifest differs")
    else:
        if resume:
            raise ComputeScaleError("resume manifest is absent")
        atomic_write_json(manifest_path, manifest)

    ((full_banks, canonical_receipt), canonical_timing) = _timed(
        lambda: _canonical_source_stage(assets, completed, resume=True)
    )
    projection = project_verified_source_banks(
        full_banks,
        parent_asset_sha256=assets.parent_asset_sha256,
        parent_membership_digest=assets.parent_membership_digest,
        expected_source_count=30,
    )
    bootstrap_stable = {
        "canonical_complete_digest": canonical_receipt["complete_digest"],
        "normalizer_digest": canonical_receipt["normalizer_digest"],
        "certificate_manifest_digest": assets.certificate_manifest.certificate_manifest_digest,
        "probe_protocol_digest": assets.probe_protocol_digest,
        "source_train_aggregate_digest": sha256_json(
            {key: bank.bank_digest for key, bank in projection.source_train.items()}
        ),
        "source_validation_aggregate_digest": sha256_json(
            {
                key: bank.bank_digest
                for key, bank in projection.source_validation.items()
            }
        ),
    }
    publish_bootstrap = not bootstrap_path.exists()
    if not publish_bootstrap:
        if not resume:
            raise ComputeScaleError("bootstrap artifact already exists")
        bootstrap = load_strict_json(bootstrap_path)
    else:
        bootstrap_unsigned = {
            "schema": COMPUTE_BOOTSTRAP_SCHEMA,
            **bootstrap_stable,
            "r4_asset_load": asset_timing,
            "full32_canonicalize": canonical_timing,
            "peak_rss_bytes": _rss_bytes(),
        }
        bootstrap = {
            **bootstrap_unsigned,
            "bootstrap_digest": sha256_json(bootstrap_unsigned),
        }
    bootstrap_digest = _validate_bootstrap(bootstrap, bootstrap_stable)
    if publish_bootstrap:
        atomic_write_json(bootstrap_path, bootstrap)

    cell_root.mkdir(exist_ok=True)
    cell_files: dict[str, str] = {}
    completed_cells: list[dict[str, Any]] = []
    for node in nodes:
        cell_path = cell_paths[str(node["cell_id"])]
        selected, train, validation = _cell_source_slices(
            node, assets=assets, projection=projection, plan=plan
        )
        expected_input_digest = _cell_input_digest(
            node,
            selected,
            train,
            validation,
            plan_digest=plan_digest,
            config_digest=config_digest,
            bootstrap_digest=bootstrap_digest,
        )
        applicable, not_applicable = _method_scope(str(node["factor"]))
        if cell_path.exists():
            if not resume or cell_path.is_symlink() or not cell_path.is_file():
                raise ComputeScaleError(f"cell cannot be reused: {node['cell_id']}")
            cell = load_strict_json(cell_path)
            _validate_completed_cell(
                cell,
                node,
                expected_input_digest=expected_input_digest,
                applicable=applicable,
                not_applicable=not_applicable,
            )
        else:
            cell = _run_cell(
                node,
                assets=assets,
                projection=projection,
                plan=plan,
                config=config,
                config_digest=config_digest,
                plan_digest=plan_digest,
                bootstrap_digest=bootstrap_digest,
                bootstrap=bootstrap,
            )
            _validate_completed_cell(
                cell,
                node,
                expected_input_digest=expected_input_digest,
                applicable=applicable,
                not_applicable=not_applicable,
            )
            atomic_write_json(cell_path, cell)
        completed_cells.append(cell)
        cell_files[str(node["cell_id"])] = sha256_file(cell_path)
    unexpected = set(cell_root.iterdir()) - set(cell_paths.values())
    if unexpected:
        raise ComputeScaleError("cell directory contains unexpected artifacts")
    long_table = []
    for cell in completed_cells:
        bandwidth_wall = _wall_total(cell["timing"]["bandwidth_calibration"])
        subset_wall = _wall_total(cell["timing"]["cell_canonical_bank_slice"])
        for method_id, method in sorted(cell["methods"].items()):
            method_subset_wall = 0.0 if method_id == B0_RANDOM else subset_wall
            method_bandwidth_wall = (
                bandwidth_wall
                if method_id in {RAW_DELTA_RKME, EMPIRICAL_MMD_NN, KME_KRR, *RFF_FAMILY}
                else 0.0
            )
            source_encode_wall = _wall_total(method["source_representation_encode"])
            fit_wall = _wall_total(method["model_fit"])
            query_encode = method["query_encode"]
            encode_median = float(query_encode.get("median_wall_seconds", 0.0))
            score_median = float(method["query_score"]["median_wall_seconds"])
            long_table.append(
                {
                    "cell_id": cell["cell_id"],
                    "factor": cell["node"]["factor"],
                    "method_id": method_id,
                    "market_anchors": cell["node"]["market_anchors"],
                    "train_episodes": cell["node"]["train_episodes"],
                    "validation_episodes": cell["node"]["validation_episodes"],
                    "rows_per_episode": cell["node"]["rows_per_episode"],
                    "rff_frequency_count": cell["node"]["rff_frequency_count"],
                    "swe_direction_count": cell["node"]["swe_direction_count"],
                    "swe_quantile_count": cell["node"]["swe_quantile_count"],
                    "bandwidth_calibration_wall_seconds": method_bandwidth_wall,
                    "trusted_joint_cell_source_subset_wall_seconds": subset_wall,
                    "standalone_method_source_subset_wall_seconds": method_subset_wall,
                    "source_representation_encode_wall_seconds": source_encode_wall,
                    "model_fit_wall_seconds": fit_wall,
                    "post_canonical_method_upload_wall_seconds": method_bandwidth_wall
                    + source_encode_wall
                    + fit_wall,
                    "upload_with_shared_subset_wall_seconds": method_subset_wall
                    + method_bandwidth_wall
                    + source_encode_wall
                    + fit_wall,
                    "query_encode_median_wall_seconds": encode_median,
                    "query_score_median_wall_seconds": score_median,
                    "warm_in_memory_reuse_median_wall_seconds": encode_median
                    + score_median,
                    "model_numeric_payload_bytes": method[
                        "model_numeric_payload_bytes"
                    ],
                    "encoded_numeric_payload_bytes": method[
                        "encoded_numeric_payload_bytes"
                    ],
                    "peak_rss_bytes_upper_bound": cell["peak_rss_bytes"],
                }
            )
    summary = {
        **_exploratory_identity(COMPUTE_SUMMARY_SCHEMA),
        "status": "COMPLETE",
        "plan_digest": plan_digest,
        "bootstrap_digest": bootstrap_digest,
        "cell_files": cell_files,
        "cell_count": len(cell_files),
        "all_methods": list(ALL_METHOD_IDS),
        "dimension_sweep_na_is_explicit_per_cell": True,
        "query_truth_accessed": False,
        "cold_load_status": "NOT_MEASURED_MODELS_ARE_IN_MEMORY_ONLY",
        "source_preprocessing_scope": (
            "FROZEN_FULL_SOURCE_NORMALIZER_REUSED; UPLOAD CURVES SEPARATE "
            "CANONICAL SUBSETTING FROM POST-CANONICAL METHOD COST"
        ),
        "reuse_scope": "WARM_IN_MEMORY_ENCODE_PLUS_SCORE_COLD_LOAD_NOT_MEASURED",
        "peak_rss_scope": "CUMULATIVE_PROCESS_UPPER_BOUND_NOT_FRESH_CELL",
        "long_table": long_table,
    }
    summary["summary_digest"] = sha256_json(summary)
    if summary_path.exists():
        if not resume or canonical_json_bytes(
            load_strict_json(summary_path)
        ) != canonical_json_bytes(summary):
            raise ComputeScaleError("resume summary differs")
    else:
        atomic_write_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.5.0+ablation"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--artifacts-root")
    parser.add_argument("--completed-run-dir", required=True)
    parser.add_argument("--new-output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        run_benchmark(
            plan_path=arguments.plan,
            completed_run_dir=arguments.completed_run_dir,
            new_output_dir=arguments.new_output_dir,
            artifacts_root=arguments.artifacts_root,
            resume=arguments.resume,
        )
    except (ComputeScaleError, V05RunnerError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
