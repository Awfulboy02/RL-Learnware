"""Secondary post-truth v0.5 source/query few-shot analysis.

This driver intentionally is not a second production framework.  It reuses the
strict r4 admission and canonical numeric primitives from the production
runner, fits only models that accept canonical reward-free episode banks, and
publishes one immutable score artifact per preregistered node.  Because this
analysis is explicitly post-truth, it claims no blinded capability boundary:
the complete score closure is persisted before label-based metric aggregation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from policy_learnware_v0.hashing import sha256_file, sha256_json, sha256_ndarrays
from policy_learnware_v0.io import atomic_write_json
from policy_learnware_v0.rkme.gaussian import calibrate_bandwidth
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.transition_views import (
    V_DELTA_ONLY,
    TransitionBank,
    apply_transition_view,
)
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
from policy_learnware_v0.v05.metrics import MARKET_30_CERT, TASK_5_CERT
from policy_learnware_v0.v05.specifications import RFFMap, RFFSpecification, SWEMap
from server.repro_fpo_ppo_v02.provenance import load_strict_json, utc_now
from server.repro_fpo_ppo_v05.environment_classifier_runner import (
    Q0_COMMON_GAUSSIAN_OPEN_LOOP,
    FrozenR4Assets,
    V05RunnerError,
    _UniqueKeyLoader,
    _load_frozen_r4_assets,
    _publish_or_match_json,
    _rank,
    _rss_bytes,
    _source_native_bank,
    load_development_config,
)


ALL_METHOD_IDS = P0_METHOD_IDS + ABLATION_METHOD_IDS
ENDPOINTS = (MARKET_30_CERT, TASK_5_CERT)
ROWS_PER_PARENT_EPISODE = 64
REPEAT_START = 25


def _validate_analysis_run_manifest(
    value: Mapping[str, Any], stable_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    expected = set(stable_manifest) | {"created_at", "run_manifest_digest"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise V05RunnerError("analysis run manifest fields differ")
    created_at = value["created_at"]
    try:
        timestamp = datetime.fromisoformat(created_at)
    except (TypeError, ValueError) as error:
        raise V05RunnerError("analysis run manifest created_at is invalid") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(
        None
    ):
        raise V05RunnerError("analysis run manifest created_at is not UTC")
    persisted_stable = {key: value[key] for key in stable_manifest}
    if persisted_stable != dict(stable_manifest):
        raise V05RunnerError("analysis run manifest changed")
    if value["run_manifest_digest"] != sha256_json(persisted_stable):
        raise V05RunnerError("analysis run manifest digest differs")
    return dict(value)


@dataclass(frozen=True)
class SourceNode:
    node_id: str
    family: str
    train_episodes: int
    validation_episodes: int
    rows_per_episode: int

    @property
    def fit_key(self) -> tuple[int, int, int]:
        return self.train_episodes, self.validation_episodes, self.rows_per_episode

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "family": self.family,
            "train_episodes": self.train_episodes,
            "validation_episodes": self.validation_episodes,
            "rows_per_episode": self.rows_per_episode,
        }


@dataclass
class FitBundle:
    canonicalizer: GlobalCanonicalizerSpec
    train: Mapping[str, EpisodeBank]
    validation: Mapping[str, EpisodeBank]
    labels: Mapping[str, str]
    models: Mapping[str, Any]
    rff_map: RFFMap
    swe_map: SWEMap
    swe_1024_map: SWEMap
    bandwidth: float
    source_digest: str
    model_manifest_digest: str
    model_nbytes: Mapping[str, int]
    timing: Mapping[str, Any]


def _positive_integer(value: Any, where: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise V05RunnerError(f"{where} must be an integer")
    result = int(value)
    if result <= 0 or (maximum is not None and result > maximum):
        raise V05RunnerError(f"{where} lies outside its frozen range")
    return result


def _safe_relative(root: Path, raw: Any, where: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise V05RunnerError(f"{where} relative path is unsafe")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise V05RunnerError(f"{where} escapes the repository")
    return resolved


def _load_analysis_config(
    path: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, Path]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise V05RunnerError("ablation config is absent or unsafe")
    try:
        raw = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise V05RunnerError("ablation config is not strict YAML") from error
    if not isinstance(raw, Mapping):
        raise V05RunnerError("ablation config must be an object")
    config = dict(raw)
    digest = sha256_json(config)
    if config.get("schema") != "policy-learnware.v05-ablation-plan.v1":
        raise V05RunnerError("ablation config schema differs")
    scope = config.get("scope")
    methods = config.get("methods")
    prefix = config.get("row_prefix")
    if (
        not isinstance(scope, Mapping)
        or scope.get("status") != "SECONDARY_EXPLORATORY_POST_TRUTH"
        or scope.get("formal_confirmatory") is not False
        or scope.get("actual_new_acquisition_steps") != 0
        or scope.get("output_rule") != "new_analysis_directory_only"
        or not isinstance(methods, Mapping)
        or tuple(methods.get("frozen_p0", ())) != P0_METHOD_IDS
        or tuple(methods.get("additions", ())) != ABLATION_METHOD_IDS
        or not isinstance(prefix, Mapping)
        or prefix.get("rule")
        != "sha256_order_without_replacement_per_parent_membership_episode"
        or prefix.get("nested") is not True
        or prefix.get("label_reward_result_independent") is not True
    ):
        raise V05RunnerError("ablation scope/method/prefix closure differs")
    repository_root = source.resolve().parent.parent
    development = config.get("development_config")
    if not isinstance(development, Mapping):
        raise V05RunnerError("development config binding is absent")
    development_path = _safe_relative(
        repository_root, development.get("relative_path"), "development config"
    )
    if sha256_file(development_path) != development.get("file_sha256"):
        raise V05RunnerError("development config file SHA differs")
    development_config, development_digest = load_development_config(development_path)
    if development_digest != development.get("canonical_digest"):
        raise V05RunnerError("development config canonical digest differs")
    return config, digest, development_config, development_digest, repository_root


def _source_nodes(config: Mapping[str, Any]) -> tuple[SourceNode, ...]:
    section = config.get("source_fewshot")
    if (
        not isinstance(section, Mapping)
        or section.get("family") != "L_SHAPED_ONE_FACTOR_AT_A_TIME"
    ):
        raise V05RunnerError("source few-shot family differs")
    episode_rows = _positive_integer(
        section.get("episode_node_rows_per_episode"),
        "episode-node rows",
        maximum=64,
    )
    result: list[SourceNode] = []
    seen: set[str] = set()
    raw_episode_nodes = section.get("episode_nodes")
    if not isinstance(raw_episode_nodes, list) or not raw_episode_nodes:
        raise V05RunnerError("source episode nodes are absent")
    for raw in raw_episode_nodes:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "train_episodes",
            "validation_episodes",
        }:
            raise V05RunnerError("source episode node fields differ")
        node_id = str(raw["id"])
        if not node_id or node_id in seen:
            raise V05RunnerError("source node IDs must be unique")
        seen.add(node_id)
        result.append(
            SourceNode(
                node_id,
                "SOURCE_EPISODES",
                _positive_integer(raw["train_episodes"], "train episodes", maximum=19),
                _positive_integer(
                    raw["validation_episodes"], "validation episodes", maximum=6
                ),
                episode_rows,
            )
        )
    row_train = _positive_integer(
        section.get("row_node_train_episodes"), "row-node train episodes", maximum=19
    )
    row_validation = _positive_integer(
        section.get("row_node_validation_episodes"),
        "row-node validation episodes",
        maximum=6,
    )
    raw_row_nodes = section.get("row_nodes")
    if not isinstance(raw_row_nodes, list) or not raw_row_nodes:
        raise V05RunnerError("source row nodes are absent")
    for raw_rows in raw_row_nodes:
        rows = _positive_integer(raw_rows, "source row node", maximum=64)
        node_id = f"SR{rows:02d}"
        if node_id in seen:
            raise V05RunnerError("source row node IDs collide")
        seen.add(node_id)
        result.append(
            SourceNode(
                node_id,
                "SOURCE_ROWS",
                row_train,
                row_validation,
                rows,
            )
        )
    if (
        tuple(section.get("fixed_query_budgets", ())) != (1, 2, 4)
        or section.get("fixed_query_rows_per_episode") != 64
        or section.get("refit_normalizer_per_node") is not True
        or section.get("refit_bandwidth_per_node_from_train_only") is not True
    ):
        raise V05RunnerError("source few-shot fixed-query/refit rule differs")
    return tuple(result)


def _query_grid(config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    section = config.get("query_fewshot")
    if not isinstance(section, Mapping):
        raise V05RunnerError("query few-shot section is absent")
    fixed = section.get("fixed_source_node")
    if (
        not isinstance(fixed, Mapping)
        or dict(fixed)
        != {"train_episodes": 19, "validation_episodes": 6, "rows_per_episode": 64}
        or tuple(section.get("budgets", ())) != (1, 2, 4)
        or section.get("episode_order") != "frozen_held_repeat_prefix"
        or section.get("b7_sanity_in_auc") is not False
        or section.get("current_physical_bank_available_rows_per_episode") != 64
        or section.get("future_fresh_equivalent_steps_per_episode") != 1000
    ):
        raise V05RunnerError("query few-shot frozen source/budget rule differs")
    rows = section.get("rows_per_episode")
    if not isinstance(rows, list) or not rows:
        raise V05RunnerError("query row grid is absent")
    resolved = tuple(_positive_integer(item, "query rows", maximum=64) for item in rows)
    if resolved != tuple(sorted(set(resolved))):
        raise V05RunnerError("query row grid must be sorted and unique")
    return tuple((budget, row_count) for budget in (1, 2, 4) for row_count in resolved)


def _git_commit(repository_root: Path) -> str:
    try:
        result = subprocess.run(
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
        raise V05RunnerError("cannot resolve the analysis git commit") from error
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise V05RunnerError("analysis git commit is not a full lowercase SHA")
    if dirty:
        raise V05RunnerError("analysis launch requires a clean committed worktree")
    return result


def _native_subset(
    assets: FrozenR4Assets,
    source_id: str,
    *,
    start_episode: int,
    episode_count: int,
    rows_per_episode: int,
    data_role: str,
    row_seed: int,
) -> NativeTransitionBank:
    parent = _source_native_bank(
        assets,
        source_id,
        start_episode=start_episode,
        stop_episode=start_episode + episode_count,
        data_role=data_role,
    )
    selected_by_episode = tuple(
        nested_row_order(
            assets.parent_membership_digest[source_id],
            start_episode + episode,
            row_seed,
        )[:rows_per_episode]
        for episode in range(episode_count)
    )
    selected = np.asarray(
        [
            episode * ROWS_PER_PARENT_EPISODE + row
            for episode, rows in enumerate(selected_by_episode)
            for row in rows
        ],
        dtype=np.int64,
    )
    count = episode_count * rows_per_episode
    truncated = np.zeros(count, dtype=np.bool_)
    truncated[rows_per_episode - 1 :: rows_per_episode] = True
    return NativeTransitionBank(
        bank_id=(
            f"abl-{assets.context_by_anchor[source_id]}-{start_episode}-"
            f"{episode_count}-r{rows_per_episode}"
        ),
        task_private_id=parent.task_private_id,
        data_role=data_role,
        native_schema_digest=parent.native_schema_digest,
        raw_dataset_digest=sha256_json(
            {
                "parent_native_bank_digest": parent.native_bank_digest,
                "parent_asset_sha256": assets.parent_asset_sha256[source_id],
                "parent_membership_digest": assets.parent_membership_digest[source_id],
                "row_prefix_seed": row_seed,
                "physical_episode_positions": list(
                    range(start_episode, start_episode + episode_count)
                ),
                "selected_rows_by_episode": [
                    list(item) for item in selected_by_episode
                ],
                "synthetic_reward_done": "ZERO_FALSE_BOUNDARY_ONLY",
            }
        ),
        observation=parent.observation[selected],
        action=parent.action[selected],
        reward=np.zeros(count, dtype=np.float64),
        next_observation=parent.next_observation[selected],
        terminated=np.zeros(count, dtype=np.bool_),
        truncated=truncated,
        episode_id=np.repeat(
            np.arange(episode_count, dtype=np.int64), rows_per_episode
        ),
        timestep=np.tile(np.arange(rows_per_episode, dtype=np.int64), episode_count),
    )


def _canonical_bank(
    canonicalizer: GlobalCanonicalizerSpec,
    native: NativeTransitionBank,
    *,
    expected_episodes: int,
    expected_rows: int,
) -> EpisodeBank:
    receipt = canonicalizer.transform(native)
    transition_bank = TransitionBank.from_canonical_batch(receipt.batch)
    view = apply_transition_view(transition_bank, V_DELTA_ONLY)
    bank = EpisodeBank(
        np.asarray(view.feature_matrix, dtype=np.float64),
        transition_bank.episode_offsets,
    )
    if (
        bank.input_dim != 30
        or bank.episode_count != expected_episodes
        or bank.points.shape != (expected_episodes * expected_rows, 30)
        or not np.array_equal(
            np.diff(bank.episode_offsets), np.full(expected_episodes, expected_rows)
        )
    ):
        raise V05RunnerError("ablation canonical bank shape differs")
    return bank


def _object_nbytes(value: Any, seen: set[int] | None = None) -> int:
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_object_nbytes(item, visited) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_object_nbytes(item, visited) for item in value)
    if hasattr(value, "__dict__"):
        return _object_nbytes(vars(value), visited)
    return 0


def _p0_model_digest(method_id: str, model: Any) -> str:
    if method_id == RAW_DELTA_RKME:
        arrays: dict[str, np.ndarray] = {}
        for source_id, item in model.sources.items():
            arrays[f"{source_id}.supports"] = item.supports
            arrays[f"{source_id}.beta"] = item.beta
            arrays[f"{source_id}.rkme_norm2"] = np.asarray(item.rkme_norm2)
        return sha256_json(
            {
                "method_id": method_id,
                "bandwidth": model.bandwidth,
                "arrays": sha256_ndarrays(arrays),
            }
        )
    if method_id == EMPIRICAL_MMD_NN:
        arrays = {}
        for source_id, item in model.sources.items():
            arrays[f"{source_id}.points"] = item.points
            arrays[f"{source_id}.weights"] = item.weights
            arrays[f"{source_id}.offsets"] = item.episode_offsets
            arrays[f"{source_id}.norm2"] = np.asarray(item.norm2)
        return sha256_json(
            {
                "method_id": method_id,
                "bandwidth": model.bandwidth,
                "arrays": sha256_ndarrays(arrays),
            }
        )
    if method_id in {SUMMARY_LOGREG, KME_KRR}:
        return model.model_digest
    if method_id == RFF_KME_NN:
        return sha256_json(
            {
                "method_id": method_id,
                "map_digest": model.rff_map.map_digest,
                "prototypes": sha256_ndarrays(dict(model.prototypes)),
            }
        )
    if method_id == SWE_NN:
        return sha256_json(
            {
                "method_id": method_id,
                "map_digest": model.swe_map.map_digest,
                "prototypes": sha256_ndarrays(dict(model.prototypes)),
            }
        )
    raise V05RunnerError(f"unknown P0 model digest request: {method_id}")


def _rff_episode_rows(bank: EpisodeBank, feature_map: RFFMap) -> RFFEpisodeFeatures:
    return RFFEpisodeFeatures.from_bank(feature_map, bank)


def _fit_bundle(
    assets: FrozenR4Assets,
    node: SourceNode,
    analysis_config: Mapping[str, Any],
    *,
    row_seed: int,
) -> FitBundle:
    started = time.monotonic()
    train_native = {
        source_id: _native_subset(
            assets,
            source_id,
            start_episode=0,
            episode_count=node.train_episodes,
            rows_per_episode=node.rows_per_episode,
            data_role="source_representation_train",
            row_seed=row_seed,
        )
        for source_id in sorted(assets.arrays_by_anchor)
    }
    validation_native = {
        source_id: _native_subset(
            assets,
            source_id,
            start_episode=19,
            episode_count=node.validation_episodes,
            rows_per_episode=node.rows_per_episode,
            data_role="source_representation_validation",
            row_seed=row_seed,
        )
        for source_id in sorted(assets.arrays_by_anchor)
    }
    registry = NativeShapeRegistry.from_source_banks(
        tuple(train_native.values()) + tuple(validation_native.values())
    )
    if registry.max_observation_dim + registry.max_action_dim != 30:
        raise V05RunnerError("ablation canonical width differs from frozen r4 ABI")
    normalizer = fit_global_normalizer(
        tuple(train_native.values()) + tuple(validation_native.values()),
        registry,
        std_floor=float(assets.config["measurement"]["normalizer_std_floor"]),
    )
    canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
    train = {
        source_id: _canonical_bank(
            canonicalizer,
            native,
            expected_episodes=node.train_episodes,
            expected_rows=node.rows_per_episode,
        )
        for source_id, native in train_native.items()
    }
    validation = {
        source_id: _canonical_bank(
            canonicalizer,
            native,
            expected_episodes=node.validation_episodes,
            expected_rows=node.rows_per_episode,
        )
        for source_id, native in validation_native.items()
    }
    canonical_seconds = max(0.0, time.monotonic() - started)
    bandwidth_started = time.monotonic()
    bandwidth_config = assets.config["measurement"]["gaussian_bandwidth"]
    bandwidth = calibrate_bandwidth(
        train,
        calibration_pairs=int(bandwidth_config["calibration_pairs"]),
        seed=int(bandwidth_config["public_seed"]),
    )
    bandwidth_seconds = max(0.0, time.monotonic() - bandwidth_started)
    labels = MappingProxyType(
        {
            item.source_anchor_id: item.opaque_certified_policy_id
            for item in assets.certificate_manifest.bindings
        }
    )
    method_config = assets.config
    reducer = ReducerConfig(**dict(method_config["raw_delta_rkme"]))
    rff_config = method_config["rff_kme_nn"]
    swe_config = method_config["swe_nn"]
    rff_map = RFFMap(
        30,
        bandwidth,
        normalizer.normalizer_digest,
        frequency_count=int(rff_config["frequency_count"]),
        public_seed=int(rff_config["public_seed"]),
    )
    swe_map = SWEMap(
        30,
        normalizer.normalizer_digest,
        direction_count=int(swe_config["direction_count"]),
        quantile_count=int(swe_config["quantile_count"]),
        public_seed=int(swe_config["public_seed"]),
    )
    swe_small_config = analysis_config["methods"]["swe_1024"]
    swe_1024_map = SWEMap(
        30,
        normalizer.normalizer_digest,
        direction_count=int(swe_small_config["direction_count"]),
        quantile_count=int(swe_small_config["quantile_count"]),
        public_seed=int(swe_small_config["public_seed"]),
    )
    models: dict[str, Any] = {}
    fit_by_method: dict[str, float] = {}

    def fitted(method_id: str, callback: Any) -> Any:
        fit_started = time.monotonic()
        result = callback()
        fit_by_method[method_id] = max(0.0, time.monotonic() - fit_started)
        return result

    models[RAW_DELTA_RKME] = fitted(
        RAW_DELTA_RKME,
        lambda: RawDeltaRKMENN.fit(
            train,
            bandwidth=bandwidth,
            protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP,
            reducer_config=reducer,
        ),
    )
    models[EMPIRICAL_MMD_NN] = fitted(
        EMPIRICAL_MMD_NN,
        lambda: EmpiricalMMDNN.fit(
            train, bandwidth=bandwidth, protocol_id=Q0_COMMON_GAUSSIAN_OPEN_LOOP
        ),
    )
    logreg_config = method_config["summary_logreg"]
    models[SUMMARY_LOGREG] = fitted(
        SUMMARY_LOGREG,
        lambda: SummaryLogReg.fit(
            train,
            labels,
            validation,
            l2_grid=logreg_config["l2_grid"],
            max_iter=int(logreg_config["max_iter"]),
            tolerance=float(logreg_config["gradient_tolerance"]),
        ),
    )
    models[KME_KRR] = fitted(
        KME_KRR,
        lambda: KMEKRR.fit(
            train,
            labels,
            validation,
            bandwidth=bandwidth,
            ridge_grid=method_config["kme_krr"]["ridge_grid"],
        ),
    )
    models[RFF_KME_NN] = fitted(
        RFF_KME_NN, lambda: RFFKMENN.fit(train, rff_map=rff_map)
    )
    models[SWE_NN] = fitted(SWE_NN, lambda: SWENN.fit(train, swe_map=swe_map))
    class_ids = tuple(sorted(labels.values()))
    models[B0_RANDOM] = fitted(
        B0_RANDOM,
        lambda: DeterministicRandomRanker(
            class_ids, public_seed=int(analysis_config["methods"]["b0_public_seed"])
        ),
    )
    models[B3A_RAW_MOMENT_NN] = fitted(
        B3A_RAW_MOMENT_NN, lambda: RawMomentNN.fit(train, labels)
    )
    models[SUMMARY_NN] = fitted(
        SUMMARY_NN, lambda: SummaryPrototypeNN.fit(train, labels)
    )
    rff_logreg_config = analysis_config["methods"]["rff_logreg"]
    rff_ridge_config = analysis_config["methods"]["rff_ridge"]
    rff_feature_started = time.monotonic()
    rff_train = {key: _rff_episode_rows(bank, rff_map) for key, bank in train.items()}
    rff_validation = {
        key: _rff_episode_rows(bank, rff_map) for key, bank in validation.items()
    }
    rff_feature_seconds = max(0.0, time.monotonic() - rff_feature_started)
    models[RFF_LOGREG] = fitted(
        RFF_LOGREG,
        lambda: FixedFeatureLogReg.fit(
            rff_train,
            labels,
            rff_validation,
            l2_grid=rff_logreg_config["l2_grid"],
            max_iter=int(rff_logreg_config["max_iter"]),
            tolerance=float(rff_logreg_config["gradient_tolerance"]),
        ),
    )
    models[RFF_RIDGE] = fitted(
        RFF_RIDGE,
        lambda: FixedFeatureRidge.fit(
            rff_train,
            labels,
            rff_validation,
            ridge_grid=rff_ridge_config["ridge_grid"],
        ),
    )
    policy_train = {labels[source_id]: bank for source_id, bank in train.items()}
    models[SWE_1024_NN] = fitted(
        SWE_1024_NN, lambda: SWE1024NN.fit(policy_train, swe_map=swe_1024_map)
    )
    digests: dict[str, str] = {}
    for method_id in P0_METHOD_IDS:
        digests[method_id] = _p0_model_digest(method_id, models[method_id])
    for method_id in (B0_RANDOM, B3A_RAW_MOMENT_NN, SUMMARY_NN, RFF_LOGREG, RFF_RIDGE):
        digests[method_id] = models[method_id].model_digest
    digests[SWE_1024_NN] = sha256_json(
        {
            "method_id": SWE_1024_NN,
            "map_digest": swe_1024_map.map_digest,
            "prototypes": sha256_ndarrays(dict(models[SWE_1024_NN].prototypes)),
        }
    )
    source_digest = sha256_json(
        {
            "fit_specification": {
                "train_episodes": node.train_episodes,
                "validation_episodes": node.validation_episodes,
                "rows_per_episode": node.rows_per_episode,
            },
            "registry_digest": registry.registry_digest,
            "normalizer_digest": normalizer.normalizer_digest,
            "canonicalizer_digest": canonicalizer.canonicalizer_digest,
            "train_banks": {key: bank.bank_digest for key, bank in train.items()},
            "validation_banks": {
                key: bank.bank_digest for key, bank in validation.items()
            },
            "bandwidth": bandwidth,
        }
    )
    manifest_digest = sha256_json(
        {
            "source_digest": source_digest,
            "model_digests": digests,
            "method_ids": list(ALL_METHOD_IDS),
        }
    )
    return FitBundle(
        canonicalizer=canonicalizer,
        train=MappingProxyType(train),
        validation=MappingProxyType(validation),
        labels=labels,
        models=MappingProxyType(models),
        rff_map=rff_map,
        swe_map=swe_map,
        swe_1024_map=swe_1024_map,
        bandwidth=bandwidth,
        source_digest=source_digest,
        model_manifest_digest=manifest_digest,
        model_nbytes=MappingProxyType(
            {
                method_id: _object_nbytes(models[method_id])
                for method_id in ALL_METHOD_IDS
            }
        ),
        timing=MappingProxyType(
            {
                "shared_canonicalize_seconds": canonical_seconds,
                "bandwidth_calibration_seconds": bandwidth_seconds,
                "shared_source_representation_encode_seconds_by_view": {
                    "RFF_SUPERVISED_EPISODE_ROWS": rff_feature_seconds
                },
                "model_fit_seconds_by_method": fit_by_method,
                "model_fit_seconds_sum": math.fsum(fit_by_method.values()),
                "cold_load_seconds": 0.0,
            }
        ),
    )


def _query_id(assets: FrozenR4Assets, source_id: str, analysis_digest: str) -> str:
    return (
        "q-"
        + sha256_json(
            {
                "schema": "policy-learnware.v05-ablation-query-id.v1",
                "analysis_config_digest": analysis_digest,
                "probe_protocol_digest": assets.probe_protocol_digest,
                "parent_membership_digest": assets.parent_membership_digest[source_id],
                "held_repeat_positions": [25, 26, 27, 28],
            }
        )[:32]
    )


def _query_bank(
    bundle: FitBundle,
    assets: FrozenR4Assets,
    source_id: str,
    *,
    budget: int,
    rows_per_episode: int,
    row_seed: int,
) -> tuple[EpisodeBank, float]:
    started = time.monotonic()
    native = _native_subset(
        assets,
        source_id,
        start_episode=REPEAT_START,
        episode_count=budget,
        rows_per_episode=rows_per_episode,
        data_role="development_query",
        row_seed=row_seed,
    )
    bank = _canonical_bank(
        bundle.canonicalizer,
        native,
        expected_episodes=budget,
        expected_rows=rows_per_episode,
    )
    return bank, max(0.0, time.monotonic() - started)


def _policy_scores(
    bundle: FitBundle,
    query: EpisodeBank,
    public_query_token: str,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    models = bundle.models
    encoder: dict[str, float] = {}
    scorer: dict[str, float] = {}
    encoded_started = time.monotonic()
    summaries = summary_episode_features(query)
    encoder["SUMMARY"] = max(0.0, time.monotonic() - encoded_started)
    encoded_started = time.monotonic()
    rff_rows = _rff_episode_rows(query, bundle.rff_map)
    rff_spec = RFFSpecification(
        vector=np.mean(rff_rows.rows, axis=0), map_digest=bundle.rff_map.map_digest
    )
    encoder["RFF_1024"] = max(0.0, time.monotonic() - encoded_started)
    encoded_started = time.monotonic()
    swe_spec = bundle.swe_map.embed(query.points, query.episode_offsets)
    encoder["SWE_4096"] = max(0.0, time.monotonic() - encoded_started)
    encoded_started = time.monotonic()
    swe_small_spec = bundle.swe_1024_map.embed(query.points, query.episode_offsets)
    encoder["SWE_1024"] = max(0.0, time.monotonic() - encoded_started)
    encoded_started = time.monotonic()
    moments = episode_balanced_moment_vector(query)
    encoder["RAW_MOMENT"] = max(0.0, time.monotonic() - encoded_started)
    results: dict[str, dict[str, float]] = {}

    def scored(method_id: str, callback: Any) -> None:
        score_started = time.monotonic()
        result = {key: float(value) for key, value in callback().items()}
        scorer[method_id] = max(0.0, time.monotonic() - score_started)
        if set(result) != set(bundle.labels.values()) or any(
            not math.isfinite(value) for value in result.values()
        ):
            raise V05RunnerError(f"{method_id} policy score coverage differs")
        results[method_id] = result

    anchor_to_policy = bundle.labels

    def anchors_to_policies(values: Mapping[str, float]) -> dict[str, float]:
        if set(values) != set(anchor_to_policy):
            raise V05RunnerError("anchor-prototype score coverage differs")
        return {
            anchor_to_policy[anchor]: float(value) for anchor, value in values.items()
        }

    scored(
        RAW_DELTA_RKME, lambda: anchors_to_policies(models[RAW_DELTA_RKME].score(query))
    )
    scored(
        EMPIRICAL_MMD_NN,
        lambda: anchors_to_policies(models[EMPIRICAL_MMD_NN].score(query)),
    )
    scored(SUMMARY_LOGREG, lambda: models[SUMMARY_LOGREG].score_summaries(summaries))
    scored(KME_KRR, lambda: models[KME_KRR].score(query))
    scored(
        RFF_KME_NN,
        lambda: anchors_to_policies(models[RFF_KME_NN].score_specification(rff_spec)),
    )
    scored(
        SWE_NN,
        lambda: anchors_to_policies(models[SWE_NN].score_specification(swe_spec)),
    )
    scored(
        B0_RANDOM,
        lambda: models[B0_RANDOM].score(public_query_token=public_query_token),
    )
    scored(B3A_RAW_MOMENT_NN, lambda: models[B3A_RAW_MOMENT_NN].score_vector(moments))
    scored(
        SUMMARY_NN, lambda: models[SUMMARY_NN].score_vector(np.mean(summaries, axis=0))
    )
    scored(RFF_LOGREG, lambda: models[RFF_LOGREG].score_features(rff_rows))
    scored(RFF_RIDGE, lambda: models[RFF_RIDGE].score_features(rff_rows))
    scored(SWE_1024_NN, lambda: models[SWE_1024_NN].score_specification(swe_small_spec))
    return results, encoder, scorer


def _score_rows(
    bundle: FitBundle,
    assets: FrozenR4Assets,
    *,
    family: str,
    node_id: str,
    budgets: Sequence[int],
    rows_per_episode: int,
    analysis_digest: str,
    tie_digest: str,
    row_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_ids = tuple(sorted(bundle.labels))
    policy_to_anchor = {policy: anchor for anchor, policy in bundle.labels.items()}
    if len(policy_to_anchor) != len(source_ids):
        raise V05RunnerError("ablation requires one frozen policy per source anchor")
    rows: list[dict[str, Any]] = []
    canonicalize_seconds = 0.0
    encoder_seconds: dict[str, float] = {}
    scorer_seconds = {method_id: 0.0 for method_id in ALL_METHOD_IDS}
    for source_id in source_ids:
        query_id = _query_id(assets, source_id, analysis_digest)
        task_id = assets.task_by_anchor[source_id]
        task_anchors = tuple(
            item for item in source_ids if assets.task_by_anchor[item] == task_id
        )
        if len(task_anchors) != 5:
            raise V05RunnerError("TASK_5 candidate group differs")
        max_budget = max(budgets)
        full_query, elapsed = _query_bank(
            bundle,
            assets,
            source_id,
            budget=max_budget,
            rows_per_episode=rows_per_episode,
            row_seed=row_seed,
        )
        canonicalize_seconds += elapsed
        for budget in budgets:
            query = full_query.prefix(int(budget))
            policy_scores, encoder, scorer = _policy_scores(bundle, query, query_id)
            for key, value in encoder.items():
                encoder_seconds[key] = encoder_seconds.get(key, 0.0) + value
            for key, value in scorer.items():
                scorer_seconds[key] += value
            for method_id in ALL_METHOD_IDS:
                candidate_policy_order = tuple(
                    bundle.labels[anchor] for anchor in source_ids
                )
                scores_before_mask = [
                    policy_scores[method_id][policy]
                    for policy in candidate_policy_order
                ]
                score_vector_digest = sha256_json(
                    {
                        "family": family,
                        "node_id": node_id,
                        "method_id": method_id,
                        "candidate_policy_order": list(candidate_policy_order),
                        "scores_before_mask": scores_before_mask,
                        "budget_episodes": int(budget),
                        "rows_per_episode": rows_per_episode,
                        "opaque_query_id": query_id,
                    }
                )
                for endpoint, allowed in (
                    (MARKET_30_CERT, source_ids),
                    (TASK_5_CERT, task_anchors),
                ):
                    allowed_policies = tuple(bundle.labels[item] for item in allowed)
                    ranked_policies = _rank(
                        {
                            item: policy_scores[method_id][item]
                            for item in allowed_policies
                        },
                        tie_digest,
                    )
                    ranked_anchors = tuple(
                        policy_to_anchor[policy] for policy in ranked_policies
                    )
                    algorithm_rows = (
                        0 if method_id == B0_RANDOM else int(budget) * rows_per_episode
                    )
                    rows.append(
                        {
                            "family": family,
                            "node_id": node_id,
                            "method_id": method_id,
                            "endpoint": endpoint,
                            "budget_episodes": int(budget),
                            "rows_per_episode": rows_per_episode,
                            "opaque_query_id": query_id,
                            "ranked_anchor_ids": list(ranked_anchors),
                            "ranked_policy_ids": list(ranked_policies),
                            "allowed_policy_ids": list(allowed_policies),
                            "scores_before_mask": scores_before_mask,
                            "score_vector_digest": score_vector_digest,
                            "canonical_query_bank_digest": query.bank_digest,
                            "normalizer_digest": bundle.canonicalizer.normalizer.normalizer_digest,
                            "source_model_manifest_digest": bundle.model_manifest_digest,
                            "algorithm_used_visible_rows": algorithm_rows,
                            "trusted_joint_driver_physical_rows": int(budget) * 64,
                            "future_fresh_equivalent_steps": (
                                0 if method_id == B0_RANDOM else int(budget) * 1000
                            ),
                            "actual_new_acquisition_steps": 0,
                        }
                    )
    rows.sort(
        key=lambda item: (
            item["family"],
            item["node_id"],
            item["method_id"],
            item["endpoint"],
            item["budget_episodes"],
            item["rows_per_episode"],
            item["opaque_query_id"],
        )
    )
    return rows, {
        "shared_canonicalize_seconds": canonicalize_seconds,
        "representation_encode_seconds": math.fsum(encoder_seconds.values()),
        "representation_encode_seconds_by_view": dict(sorted(encoder_seconds.items())),
        "warm_score_seconds": math.fsum(scorer_seconds.values()),
        "warm_score_seconds_by_method": scorer_seconds,
    }


def _node_input_digest(
    run_manifest_digest: str,
    family: str,
    node_id: str,
    specification: Mapping[str, Any],
) -> str:
    return sha256_json(
        {
            "run_manifest_digest": run_manifest_digest,
            "family": family,
            "node_id": node_id,
            "specification": dict(specification),
            "method_ids": list(ALL_METHOD_IDS),
            "endpoints": list(ENDPOINTS),
        }
    )


def _validate_node_score_closure(value: Mapping[str, Any]) -> None:
    expected_node_fields = {
        "schema",
        "status",
        "scope",
        "truth_blinding_status",
        "input_digest",
        "family",
        "node_id",
        "specification",
        "source_digest",
        "source_model_manifest_digest",
        "ranking_tie_digest",
        "candidate_anchor_order",
        "candidate_policy_order",
        "normalizer_digest",
        "canonicalizer_digest",
        "bandwidth",
        "model_nbytes_by_method",
        "fit_cache_reused_in_process",
        "timing",
        "cost",
        "peak_rss_bytes",
        "score_row_count",
        "score_rows_digest",
        "score_rows",
        "node_digest",
    }
    if set(value) != expected_node_fields:
        raise V05RunnerError("node artifact fields differ")
    rows = value.get("score_rows")
    family = value.get("family")
    specification = value.get("specification")
    anchors = value.get("candidate_anchor_order")
    policies = value.get("candidate_policy_order")
    if (
        not isinstance(rows, list)
        or not isinstance(specification, Mapping)
        or not isinstance(anchors, list)
        or not isinstance(policies, list)
        or len(anchors) != 30
        or len(policies) != 30
        or len(set(anchors)) != 30
        or len(set(policies)) != 30
        or any(not isinstance(item, str) or not item for item in (*anchors, *policies))
    ):
        raise V05RunnerError("node candidate or score-row closure is malformed")
    if family in {"SOURCE_EPISODES", "SOURCE_ROWS"}:
        budgets = tuple(specification.get("query_budgets", ()))
        rows_per_episode = specification.get("query_rows_per_episode")
    elif family == "QUERY_GRID":
        budgets = (specification.get("budget_episodes"),)
        rows_per_episode = specification.get("rows_per_episode")
    else:
        raise V05RunnerError("node score family differs")
    if (
        not budgets
        or any(type(item) is not int or item not in (1, 2, 4) for item in budgets)
        or len(set(budgets)) != len(budgets)
        or type(rows_per_episode) is not int
        or not 1 <= rows_per_episode <= 64
    ):
        raise V05RunnerError("node budget/row score closure differs")
    expected_row_fields = {
        "family",
        "node_id",
        "method_id",
        "endpoint",
        "budget_episodes",
        "rows_per_episode",
        "opaque_query_id",
        "ranked_anchor_ids",
        "ranked_policy_ids",
        "allowed_policy_ids",
        "scores_before_mask",
        "score_vector_digest",
        "canonical_query_bank_digest",
        "normalizer_digest",
        "source_model_manifest_digest",
        "algorithm_used_visible_rows",
        "trusted_joint_driver_physical_rows",
        "future_fresh_equivalent_steps",
        "actual_new_acquisition_steps",
    }
    policy_to_anchor = dict(zip(policies, anchors, strict=True))
    tie_digest = value.get("ranking_tie_digest")
    observed: set[tuple[Any, ...]] = set()
    query_ids: set[str] = set()
    vector_closure: dict[tuple[Any, ...], str] = {}
    task_masks: dict[str, tuple[str, ...]] = {}
    query_banks: dict[tuple[str, int], str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise V05RunnerError("node score-row fields differ")
        query_id = row["opaque_query_id"]
        method_id = row["method_id"]
        endpoint = row["endpoint"]
        budget = row["budget_episodes"]
        key = (query_id, method_id, endpoint, budget)
        if (
            not isinstance(query_id, str)
            or not query_id
            or method_id not in ALL_METHOD_IDS
            or endpoint not in ENDPOINTS
            or budget not in budgets
            or row["family"] != family
            or row["node_id"] != value.get("node_id")
            or row["rows_per_episode"] != rows_per_episode
            or key in observed
        ):
            raise V05RunnerError("node score-row identity differs or is duplicated")
        observed.add(key)
        query_ids.add(query_id)
        scores = row["scores_before_mask"]
        allowed = row["allowed_policy_ids"]
        ranked_policies = row["ranked_policy_ids"]
        ranked_anchors = row["ranked_anchor_ids"]
        expected_width = 30 if endpoint == MARKET_30_CERT else 5
        if (
            not isinstance(scores, list)
            or len(scores) != 30
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in scores
            )
            or not isinstance(allowed, list)
            or len(allowed) != expected_width
            or len(set(allowed)) != expected_width
            or not set(allowed).issubset(policies)
            or not isinstance(ranked_policies, list)
            or not isinstance(ranked_anchors, list)
            or len(ranked_policies) != expected_width
            or len(ranked_anchors) != expected_width
        ):
            raise V05RunnerError("node score vector, mask, or ranking is malformed")
        if endpoint == MARKET_30_CERT and allowed != policies:
            raise V05RunnerError("market score mask differs from candidate order")
        if endpoint == TASK_5_CERT:
            previous = task_masks.setdefault(query_id, tuple(allowed))
            if previous != tuple(allowed):
                raise V05RunnerError("task score mask changes within a query")
        score_mapping = dict(
            zip(policies, (float(item) for item in scores), strict=True)
        )
        expected_ranked_policies = list(
            _rank({policy: score_mapping[policy] for policy in allowed}, tie_digest)
        )
        if ranked_policies != expected_ranked_policies or ranked_anchors != [
            policy_to_anchor[policy] for policy in expected_ranked_policies
        ]:
            raise V05RunnerError("persisted ranking differs from its score vector")
        expected_vector_digest = sha256_json(
            {
                "family": family,
                "node_id": value["node_id"],
                "method_id": method_id,
                "candidate_policy_order": policies,
                "scores_before_mask": scores,
                "budget_episodes": budget,
                "rows_per_episode": rows_per_episode,
                "opaque_query_id": query_id,
            }
        )
        if row["score_vector_digest"] != expected_vector_digest:
            raise V05RunnerError("persisted score-vector digest differs")
        shared_key = (query_id, method_id, budget)
        shared_digest = sha256_json(
            {
                "scores": scores,
                "score_vector_digest": row["score_vector_digest"],
                "canonical_query_bank_digest": row["canonical_query_bank_digest"],
            }
        )
        if vector_closure.setdefault(shared_key, shared_digest) != shared_digest:
            raise V05RunnerError("endpoint rows do not share one score vector")
        bank_key = (query_id, budget)
        if (
            query_banks.setdefault(bank_key, row["canonical_query_bank_digest"])
            != row["canonical_query_bank_digest"]
        ):
            raise V05RunnerError("query bank digest changes across methods")
        algorithm_rows = 0 if method_id == B0_RANDOM else budget * rows_per_episode
        if (
            row["normalizer_digest"] != value.get("normalizer_digest")
            or row["source_model_manifest_digest"]
            != value.get("source_model_manifest_digest")
            or row["algorithm_used_visible_rows"] != algorithm_rows
            or row["trusted_joint_driver_physical_rows"] != budget * 64
            or row["future_fresh_equivalent_steps"]
            != (0 if method_id == B0_RANDOM else budget * 1000)
            or row["actual_new_acquisition_steps"] != 0
        ):
            raise V05RunnerError("node score-row provenance or cost differs")
    if len(query_ids) != 30:
        raise V05RunnerError("node does not cover exactly 30 opaque queries")
    expected = {
        (query_id, method_id, endpoint, budget)
        for query_id in query_ids
        for method_id in ALL_METHOD_IDS
        for endpoint in ENDPOINTS
        for budget in budgets
    }
    if observed != expected or len(rows) != len(expected):
        raise V05RunnerError("node score closure has missing or extra cells")


def _load_node(path: Path, expected_input_digest: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise V05RunnerError("resume node artifact is absent or unsafe")
    value = load_strict_json(path)
    unsigned = {key: item for key, item in value.items() if key != "node_digest"}
    rows = value.get("score_rows")
    if (
        value.get("schema") != "policy-learnware.v05-ablation-score-node.v1"
        or value.get("status") != "COMPLETE_PRE_METRIC_JOIN"
        or value.get("input_digest") != expected_input_digest
        or not isinstance(rows, list)
        or value.get("score_rows_digest") != sha256_json(rows)
        or value.get("node_digest") != sha256_json(unsigned)
    ):
        raise V05RunnerError("resume node artifact digest changed")
    _validate_node_score_closure(value)
    return value


def _publish_node(
    path: Path,
    *,
    input_digest: str,
    family: str,
    node_id: str,
    specification: Mapping[str, Any],
    score_rows: list[dict[str, Any]],
    bundle: FitBundle,
    ranking_tie_digest: str,
    fit_cache_reused: bool,
    score_timing: Mapping[str, Any],
    cost: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema": "policy-learnware.v05-ablation-score-node.v1",
        "status": "COMPLETE_PRE_METRIC_JOIN",
        "scope": "SECONDARY_EXPLORATORY_POST_TRUTH",
        "truth_blinding_status": "NOT_CLAIMED_POST_TRUTH_ANALYSIS",
        "input_digest": input_digest,
        "family": family,
        "node_id": node_id,
        "specification": dict(specification),
        "source_digest": bundle.source_digest,
        "source_model_manifest_digest": bundle.model_manifest_digest,
        "ranking_tie_digest": ranking_tie_digest,
        "candidate_anchor_order": list(sorted(bundle.labels)),
        "candidate_policy_order": [
            bundle.labels[anchor] for anchor in sorted(bundle.labels)
        ],
        "normalizer_digest": bundle.canonicalizer.normalizer.normalizer_digest,
        "canonicalizer_digest": bundle.canonicalizer.canonicalizer_digest,
        "bandwidth": bundle.bandwidth,
        "model_nbytes_by_method": dict(bundle.model_nbytes),
        "fit_cache_reused_in_process": fit_cache_reused,
        "timing": {"source_fit": dict(bundle.timing), "query": dict(score_timing)},
        "cost": dict(cost),
        "peak_rss_bytes": _rss_bytes(),
        "score_row_count": len(score_rows),
        "score_rows_digest": sha256_json(score_rows),
        "score_rows": score_rows,
    }
    value = {**unsigned, "node_digest": sha256_json(unsigned)}
    _validate_node_score_closure(value)
    atomic_write_json(path, value)
    return value


def _macro_f1(
    truth: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> float:
    values = []
    for label in labels:
        true_positive = sum(
            t == label and p == label for t, p in zip(truth, predicted, strict=True)
        )
        false_positive = sum(
            t != label and p == label for t, p in zip(truth, predicted, strict=True)
        )
        false_negative = sum(
            t == label and p != label for t, p in zip(truth, predicted, strict=True)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return math.fsum(values) / len(values)


def _evaluate(
    node_values: Sequence[Mapping[str, Any]],
    assets: FrozenR4Assets,
    analysis_digest: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # This is the first function that aggregates persisted rankings against labels.
    truth = {
        _query_id(assets, item.source_anchor_id, analysis_digest): {
            "source_anchor_id": item.source_anchor_id,
            "opaque_certified_policy_id": item.opaque_certified_policy_id,
            "task_id": item.task_id,
        }
        for item in assets.certificate_manifest.bindings
    }
    per_query: list[dict[str, Any]] = []
    for node in node_values:
        for row in node["score_rows"]:
            binding = truth.get(row["opaque_query_id"])
            if binding is None:
                raise V05RunnerError(
                    "score row query is absent from the frozen metric binding"
                )
            anchor = binding["source_anchor_id"]
            policy = binding["opaque_certified_policy_id"]
            ranked_anchors = row["ranked_anchor_ids"]
            ranked_policies = row["ranked_policy_ids"]
            if anchor not in ranked_anchors or policy not in ranked_policies:
                raise V05RunnerError("truth is absent from an endpoint ranking")
            per_query.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "family",
                            "node_id",
                            "method_id",
                            "endpoint",
                            "budget_episodes",
                            "rows_per_episode",
                            "opaque_query_id",
                        )
                    },
                    "task_id": binding["task_id"],
                    "truth_anchor_id": anchor,
                    "truth_policy_id": policy,
                    "top1_anchor_id": ranked_anchors[0],
                    "top1_policy_id": ranked_policies[0],
                    "anchor_rank": ranked_anchors.index(anchor) + 1,
                    "policy_rank": ranked_policies.index(policy) + 1,
                }
            )
    per_query.sort(
        key=lambda item: (
            item["family"],
            item["node_id"],
            item["method_id"],
            item["endpoint"],
            item["budget_episodes"],
            item["rows_per_episode"],
            item["opaque_query_id"],
        )
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in per_query:
        key = tuple(
            row[item]
            for item in (
                "family",
                "node_id",
                "method_id",
                "endpoint",
                "budget_episodes",
                "rows_per_episode",
            )
        )
        grouped.setdefault(key, []).append(row)
    source_labels = tuple(
        sorted(item.source_anchor_id for item in assets.certificate_manifest.bindings)
    )
    policy_labels = tuple(
        sorted(
            item.opaque_certified_policy_id
            for item in assets.certificate_manifest.bindings
        )
    )
    performance: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = grouped[key]
        if len(values) != 30:
            raise V05RunnerError("performance group does not cover 30 query anchors")
        by_task: dict[str, list[dict[str, Any]]] = {}
        for value in values:
            by_task.setdefault(value["task_id"], []).append(value)
        per_task = []
        for task_id in sorted(by_task):
            task_rows = by_task[task_id]
            per_task.append(
                {
                    "task_id": task_id,
                    "anchor_count": len(task_rows),
                    "anchor_hit_at_1": math.fsum(
                        float(item["anchor_rank"] == 1) for item in task_rows
                    )
                    / len(task_rows),
                    "anchor_mrr": math.fsum(
                        1.0 / item["anchor_rank"] for item in task_rows
                    )
                    / len(task_rows),
                    "policy_hit_at_1": math.fsum(
                        float(item["policy_rank"] == 1) for item in task_rows
                    )
                    / len(task_rows),
                    "policy_mrr": math.fsum(
                        1.0 / item["policy_rank"] for item in task_rows
                    )
                    / len(task_rows),
                }
            )
        performance.append(
            {
                "family": key[0],
                "node_id": key[1],
                "method_id": key[2],
                "endpoint": key[3],
                "budget_episodes": key[4],
                "rows_per_episode": key[5],
                "statistical_unit": "SOURCE_ANCHOR_N30_TASK6_ANCHOR5",
                "nested_nodes_are_independent_samples": False,
                "anchor_count": 30,
                "task_equal_anchor_hit_at_1": math.fsum(
                    item["anchor_hit_at_1"] for item in per_task
                )
                / len(per_task),
                "task_equal_anchor_mrr": math.fsum(
                    item["anchor_mrr"] for item in per_task
                )
                / len(per_task),
                "task_equal_policy_hit_at_1": math.fsum(
                    item["policy_hit_at_1"] for item in per_task
                )
                / len(per_task),
                "task_equal_policy_mrr": math.fsum(
                    item["policy_mrr"] for item in per_task
                )
                / len(per_task),
                "anchor_macro_f1": _macro_f1(
                    [item["truth_anchor_id"] for item in values],
                    [item["top1_anchor_id"] for item in values],
                    source_labels,
                ),
                "policy_macro_f1": _macro_f1(
                    [item["truth_policy_id"] for item in values],
                    [item["top1_policy_id"] for item in values],
                    policy_labels,
                ),
                "per_task": per_task,
            }
        )
    return per_query, performance


def _long_tables(
    performance: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = {(row["family"], row["node_id"]): row for row in nodes}
    gaussian = {
        RAW_DELTA_RKME,
        EMPIRICAL_MMD_NN,
        KME_KRR,
        RFF_KME_NN,
        RFF_LOGREG,
        RFF_RIDGE,
    }
    encoder_view = {
        B3A_RAW_MOMENT_NN: "RAW_MOMENT",
        SUMMARY_NN: "SUMMARY",
        SUMMARY_LOGREG: "SUMMARY",
        RFF_KME_NN: "RFF_1024",
        RFF_LOGREG: "RFF_1024",
        RFF_RIDGE: "RFF_1024",
        SWE_NN: "SWE_4096",
        SWE_1024_NN: "SWE_1024",
    }
    source_table: list[dict[str, Any]] = []
    query_table: list[dict[str, Any]] = []
    for metric in performance:
        meta = metadata[(metric["family"], metric["node_id"])]
        method_id = metric["method_id"]
        source_timing = meta["timing"]["source_fit"]
        query_timing = meta["timing"]["query"]
        model_fit = float(
            source_timing["model_fit_seconds_by_method"].get(method_id, 0.0)
        )
        canonical = (
            0.0
            if method_id == B0_RANDOM
            else float(source_timing["shared_canonicalize_seconds"])
        )
        bandwidth = (
            float(source_timing["bandwidth_calibration_seconds"])
            if method_id in gaussian
            else 0.0
        )
        source_encode = 0.0
        if method_id in {RFF_LOGREG, RFF_RIDGE}:
            source_encode = float(
                source_timing["shared_source_representation_encode_seconds_by_view"][
                    "RFF_SUPERVISED_EPISODE_ROWS"
                ]
            )
        view = encoder_view.get(method_id)
        query_encode = (
            float(query_timing["representation_encode_seconds_by_view"].get(view, 0.0))
            if view is not None
            else 0.0
        )
        query_canonicalize = (
            0.0
            if method_id == B0_RANDOM
            else float(query_timing["shared_canonicalize_seconds"])
        )
        query_score = float(query_timing["warm_score_seconds_by_method"][method_id])
        query_observations = int(meta["score_row_count"]) // (
            len(ALL_METHOD_IDS) * len(ENDPOINTS)
        )
        common = {
            **dict(metric),
            "model_numeric_payload_bytes": meta["model_nbytes_by_method"][method_id],
            "query_observation_count_in_timing": query_observations,
            "query_canonicalize_seconds_total": query_canonicalize,
            "query_encode_seconds_total": query_encode,
            "query_score_seconds_total": query_score,
            "standalone_reuse_seconds_total": query_canonicalize
            + query_encode
            + query_score,
            "peak_rss_bytes_upper_bound": meta["peak_rss_bytes"],
        }
        if metric["family"] in {"SOURCE_EPISODES", "SOURCE_ROWS"}:
            source_cost = dict(meta["cost"])
            if method_id == B0_RANDOM:
                source_cost["algorithm_used_source_train_rows"] = 0
                source_cost["algorithm_used_source_validation_rows"] = 0
                source_cost["future_fresh_equivalent_source_steps"] = 0
            source_table.append(
                {
                    **common,
                    **source_cost,
                    "query_timing_scope": (
                        "AGGREGATE_OVER_30_QUERIES_AND_B1_B2_B4; "
                        "NOT_ATTRIBUTED_TO_THIS_SINGLE_B_ROW"
                    ),
                    "canonicalize_wall_seconds": canonical,
                    "bandwidth_wall_seconds": bandwidth,
                    "source_representation_encode_wall_seconds": source_encode,
                    "model_fit_wall_seconds": model_fit,
                    "standalone_upload_wall_seconds": canonical
                    + bandwidth
                    + source_encode
                    + model_fit,
                }
            )
        elif metric["family"] == "QUERY_GRID":
            query_cost = dict(meta["cost"])
            if method_id == B0_RANDOM:
                query_cost["algorithm_used_visible_rows_per_query"] = 0
                query_cost["future_fresh_equivalent_steps_per_query"] = 0
            query_table.append(
                {
                    **common,
                    **query_cost,
                    "query_timing_scope": "AGGREGATE_OVER_30_QUERIES_AT_THIS_B_R",
                }
            )
    return source_table, query_table


def _validate_analysis_output_layout(
    output: Path,
    source_nodes: Sequence[SourceNode],
    query_grid: Sequence[tuple[int, int]],
) -> None:
    """Reject unregistered output entries before any analysis write."""

    known = {
        output / "nodes",
        output / "nodes" / "source",
        output / "nodes" / "query",
        output / "scores",
        output / "results",
        output / "run_manifest.json",
        output / "scores" / "score_manifest.json",
        output / "results" / "per_query.json",
        output / "results" / "analysis_report.json",
        *(
            output / "nodes" / "source" / f"{node.node_id}.json"
            for node in source_nodes
        ),
        *(
            output / "nodes" / "query" / f"Q-B{budget:02d}-R{rows:02d}.json"
            for budget, rows in query_grid
        ),
    }
    resolved = output.resolve()
    if any(path.is_symlink() for path in known) or any(
        not path.resolve(strict=False).is_relative_to(resolved) for path in known
    ):
        raise V05RunnerError("analysis output layout is unsafe")
    if set(output.rglob("*")) - known:
        raise V05RunnerError("analysis output contains unexpected artifacts")


def run_analysis(
    config_path: str | Path,
    new_analysis_dir: str | Path,
    *,
    artifacts_root: str | Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    (
        analysis_config,
        analysis_digest,
        development_config,
        development_digest,
        repo,
    ) = _load_analysis_config(config_path)
    actual_commit = _git_commit(repo)
    assets = _load_frozen_r4_assets(
        development_config, development_digest, artifacts_root
    )
    raw_output = Path(new_analysis_dir).expanduser()
    if any(
        parent.exists() and parent.is_symlink()
        for parent in (raw_output, *raw_output.parents)
    ):
        raise V05RunnerError("analysis directory has a symlink ancestor")
    output = raw_output.resolve()
    if any(
        output == frozen
        or output.is_relative_to(frozen)
        or frozen.is_relative_to(output)
        for frozen in (assets.r4_root, assets.v03_root)
    ):
        raise V05RunnerError("analysis directory overlaps a frozen asset root")
    if resume:
        if not output.is_dir():
            raise V05RunnerError("resume analysis directory is absent")
    else:
        if output.exists():
            raise V05RunnerError("new analysis directory already exists")
        output.mkdir(parents=True, mode=0o755)
    source_nodes = _source_nodes(analysis_config)
    query_grid = _query_grid(analysis_config)
    _validate_analysis_output_layout(output, source_nodes, query_grid)
    row_seed = int(analysis_config["row_prefix"]["public_seed"])
    manifest_unsigned = {
        "schema": "policy-learnware.v05-ablation-run.v1",
        "scope": "SECONDARY_EXPLORATORY_POST_TRUTH",
        "formal_confirmatory": False,
        "actual_git_commit": actual_commit,
        "plan_base_commit": analysis_config["base_commit"],
        "analysis_config_digest": analysis_digest,
        "analysis_config_file_sha256": sha256_file(Path(config_path).expanduser()),
        "development_config_digest": development_digest,
        "frozen_r4_provenance_digest": sha256_json(dict(assets.provenance)),
        "certificate_manifest_digest": assets.certificate_manifest.certificate_manifest_digest,
        "source_node_count": len(source_nodes),
        "query_node_count": len(query_grid),
        "method_ids": list(ALL_METHOD_IDS),
        "created_at": utc_now(),
    }
    # created_at is intentionally excluded so resume remains bitwise stable.
    stable_manifest = {
        key: value for key, value in manifest_unsigned.items() if key != "created_at"
    }
    run_manifest = {
        **manifest_unsigned,
        "run_manifest_digest": sha256_json(stable_manifest),
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if not resume:
            raise V05RunnerError("analysis run manifest already exists")
        run_manifest = _validate_analysis_run_manifest(
            load_strict_json(manifest_path), stable_manifest
        )
    else:
        atomic_write_json(manifest_path, run_manifest)
    run_manifest_digest = run_manifest["run_manifest_digest"]
    bundle_cache: dict[tuple[int, int, int], FitBundle] = {}

    def bundle_for(node: SourceNode) -> tuple[FitBundle, bool]:
        if node.fit_key in bundle_cache:
            return bundle_cache[node.fit_key], True
        value = _fit_bundle(assets, node, analysis_config, row_seed=row_seed)
        bundle_cache[node.fit_key] = value
        return value, False

    node_values: list[dict[str, Any]] = []
    node_paths: list[Path] = []
    for node in source_nodes:
        specification = {
            **node.to_dict(),
            "query_budgets": [1, 2, 4],
            "query_rows_per_episode": 64,
        }
        input_digest = _node_input_digest(
            run_manifest_digest, node.family, node.node_id, specification
        )
        path = output / "nodes" / "source" / f"{node.node_id}.json"
        if path.exists():
            if not resume:
                raise V05RunnerError("source node already exists")
            value = _load_node(path, input_digest)
        else:
            bundle, reused = bundle_for(node)
            score_rows, score_timing = _score_rows(
                bundle,
                assets,
                family=node.family,
                node_id=node.node_id,
                budgets=(1, 2, 4),
                rows_per_episode=64,
                analysis_digest=analysis_digest,
                tie_digest=development_digest,
                row_seed=row_seed,
            )
            value = _publish_node(
                path,
                input_digest=input_digest,
                family=node.family,
                node_id=node.node_id,
                specification=specification,
                score_rows=score_rows,
                bundle=bundle,
                ranking_tie_digest=development_digest,
                fit_cache_reused=reused,
                score_timing=score_timing,
                cost={
                    "algorithm_used_source_train_rows": 30
                    * node.train_episodes
                    * node.rows_per_episode,
                    "algorithm_used_source_validation_rows": 30
                    * node.validation_episodes
                    * node.rows_per_episode,
                    "trusted_joint_driver_source_physical_rows": 30 * 32 * 64,
                    "future_fresh_equivalent_source_steps": 30
                    * (node.train_episodes + node.validation_episodes)
                    * 1000,
                    "actual_new_acquisition_steps": 0,
                },
            )
        node_values.append(value)
        node_paths.append(path)
    full_node = SourceNode("U19-FIXED", "QUERY_GRID_SOURCE", 19, 6, 64)
    full_bundle: FitBundle | None = bundle_cache.get(full_node.fit_key)
    for budget, row_count in query_grid:
        node_id = f"Q-B{budget:02d}-R{row_count:02d}"
        specification = {
            "node_id": node_id,
            "family": "QUERY_GRID",
            "fixed_source": {
                "train_episodes": 19,
                "validation_episodes": 6,
                "rows_per_episode": 64,
            },
            "budget_episodes": budget,
            "rows_per_episode": row_count,
        }
        input_digest = _node_input_digest(
            run_manifest_digest, "QUERY_GRID", node_id, specification
        )
        path = output / "nodes" / "query" / f"{node_id}.json"
        if path.exists():
            if not resume:
                raise V05RunnerError("query node already exists")
            value = _load_node(path, input_digest)
        else:
            reused = full_bundle is not None
            if full_bundle is None:
                full_bundle, reused = bundle_for(full_node)
            score_rows, score_timing = _score_rows(
                full_bundle,
                assets,
                family="QUERY_GRID",
                node_id=node_id,
                budgets=(budget,),
                rows_per_episode=row_count,
                analysis_digest=analysis_digest,
                tie_digest=development_digest,
                row_seed=row_seed,
            )
            value = _publish_node(
                path,
                input_digest=input_digest,
                family="QUERY_GRID",
                node_id=node_id,
                specification=specification,
                score_rows=score_rows,
                bundle=full_bundle,
                ranking_tie_digest=development_digest,
                fit_cache_reused=reused,
                score_timing=score_timing,
                cost={
                    "algorithm_used_visible_rows_per_query": budget * row_count,
                    "trusted_joint_driver_physical_rows_per_query": budget * 64,
                    "future_fresh_equivalent_steps_per_query": budget * 1000,
                    "actual_new_acquisition_steps": 0,
                },
            )
        node_values.append(value)
        node_paths.append(path)
    node_files = {
        path.relative_to(output).as_posix(): {
            "file_sha256": sha256_file(path),
            "node_digest": value["node_digest"],
            "score_rows_digest": value["score_rows_digest"],
            "score_row_count": value["score_row_count"],
        }
        for path, value in zip(node_paths, node_values, strict=True)
    }
    all_rows = [row for value in node_values for row in value["score_rows"]]
    all_rows.sort(
        key=lambda item: (
            item["family"],
            item["node_id"],
            item["method_id"],
            item["endpoint"],
            item["budget_episodes"],
            item["rows_per_episode"],
            item["opaque_query_id"],
        )
    )
    expected_score_rows = len(source_nodes) * 30 * len(ALL_METHOD_IDS) * len(
        ENDPOINTS
    ) * 3 + len(query_grid) * 30 * len(ALL_METHOD_IDS) * len(ENDPOINTS)
    global_keys = {
        (
            row["family"],
            row["node_id"],
            row["method_id"],
            row["endpoint"],
            row["budget_episodes"],
            row["rows_per_episode"],
            row["opaque_query_id"],
        )
        for row in all_rows
    }
    query_sets = {
        frozenset(row["opaque_query_id"] for row in value["score_rows"])
        for value in node_values
    }
    query_nodes = [value for value in node_values if value["family"] == "QUERY_GRID"]
    if (
        len(all_rows) != expected_score_rows
        or len(global_keys) != expected_score_rows
        or len(query_sets) != 1
        or len(next(iter(query_sets))) != 30
        or len(query_nodes) != len(query_grid)
        or len(
            {
                (
                    value["source_model_manifest_digest"],
                    value["normalizer_digest"],
                    value["canonicalizer_digest"],
                    value["bandwidth"],
                )
                for value in query_nodes
            }
        )
        != 1
    ):
        raise V05RunnerError("global 30-query score closure differs")
    score_manifest_unsigned = {
        "schema": "policy-learnware.v05-ablation-score-closure.v1",
        "status": "ALL_SCORES_PERSISTED_BEFORE_METRIC_AGGREGATION",
        "truth_blinding_status": "NOT_CLAIMED_POST_TRUTH_ANALYSIS",
        "run_manifest_digest": run_manifest_digest,
        "node_count": len(node_values),
        "score_row_count": len(all_rows),
        "score_rows_digest": sha256_json(all_rows),
        "node_files": dict(sorted(node_files.items())),
    }
    score_manifest = {
        **score_manifest_unsigned,
        "score_manifest_digest": sha256_json(score_manifest_unsigned),
    }
    score_manifest_path = output / "scores" / "score_manifest.json"
    _publish_or_match_json(score_manifest_path, score_manifest, resume=resume)
    persisted_score_manifest = load_strict_json(score_manifest_path)
    if persisted_score_manifest != score_manifest:
        raise V05RunnerError(
            "persisted score closure differs before metric aggregation"
        )
    per_query_path = output / "results" / "per_query.json"
    report_path = output / "results" / "analysis_report.json"
    if report_path.exists() and not per_query_path.exists():
        raise V05RunnerError("analysis report exists without its per-query artifact")
    if report_path.exists():
        if not resume or report_path.is_symlink() or per_query_path.is_symlink():
            raise V05RunnerError("completed analysis cannot be reused")
        persisted_per_query = load_strict_json(per_query_path)
        persisted_report = load_strict_json(report_path)
        report_unsigned = {
            key: value
            for key, value in persisted_report.items()
            if key != "report_digest"
        }
        if (
            persisted_report.get("report_digest") != sha256_json(report_unsigned)
            or persisted_report.get("score_manifest_digest")
            != score_manifest["score_manifest_digest"]
            or persisted_report.get("per_query_artifact_sha256")
            != sha256_file(per_query_path)
            or persisted_report.get("per_query_rows_digest")
            != persisted_per_query.get("rows_digest")
        ):
            raise V05RunnerError("completed analysis result digest changed")
        return persisted_report
    if per_query_path.exists() and (
        not resume or per_query_path.is_symlink() or not per_query_path.is_file()
    ):
        raise V05RunnerError("partial per-query result cannot be resumed safely")
    load_started = time.monotonic()
    verified_nodes = [
        _load_node(path, value["input_digest"])
        for path, value in zip(node_paths, node_values, strict=True)
    ]
    verified_rows = [row for value in verified_nodes for row in value["score_rows"]]
    verified_rows.sort(
        key=lambda item: (
            item["family"],
            item["node_id"],
            item["method_id"],
            item["endpoint"],
            item["budget_episodes"],
            item["rows_per_episode"],
            item["opaque_query_id"],
        )
    )
    cold_load_seconds = max(0.0, time.monotonic() - load_started)
    if sha256_json(verified_rows) != score_manifest["score_rows_digest"]:
        raise V05RunnerError(
            "verified node scores differ from the global score closure"
        )
    # Scores have been closed and reloaded above; label-based metrics begin here.
    per_query, performance = _evaluate(verified_nodes, assets, analysis_digest)
    source_table, query_table = _long_tables(performance, verified_nodes)
    per_query_unsigned = {
        "schema": "policy-learnware.v05-ablation-per-query-results.v1",
        "scope": "SECONDARY_EXPLORATORY_POST_TRUTH",
        "score_manifest_digest": score_manifest["score_manifest_digest"],
        "row_count": len(per_query),
        "rows_digest": sha256_json(per_query),
        "rows": per_query,
    }
    per_query_value = {
        **per_query_unsigned,
        "artifact_digest": sha256_json(per_query_unsigned),
    }
    if per_query_path.exists():
        if load_strict_json(per_query_path) != per_query_value:
            raise V05RunnerError("resumed per-query result differs")
    else:
        _publish_or_match_json(per_query_path, per_query_value, resume=resume)
    report_unsigned = {
        "schema": "policy-learnware.v05-ablation-analysis.v1",
        "status": "COMPLETE",
        "scope": {
            **dict(analysis_config["scope"]),
            "interpretation": (
                "secondary exploratory fixed-market measurement-stability analysis; "
                "nested episodes/rows are not independent statistical units"
            ),
        },
        "run_manifest_digest": run_manifest_digest,
        "score_manifest_digest": score_manifest["score_manifest_digest"],
        "per_query_artifact_sha256": sha256_file(per_query_path),
        "per_query_rows_digest": per_query_value["rows_digest"],
        "certificate_manifest_digest": assets.certificate_manifest.certificate_manifest_digest,
        "performance": performance,
        "source_fewshot_long_table": source_table,
        "query_fewshot_long_table": query_table,
        "nodes": [
            {
                "family": value["family"],
                "node_id": value["node_id"],
                "specification": value["specification"],
                "normalizer_digest": value["normalizer_digest"],
                "bandwidth": value["bandwidth"],
                "timing": value["timing"],
                "cost": value["cost"],
                "model_nbytes_by_method": value["model_nbytes_by_method"],
                "peak_rss_bytes": value["peak_rss_bytes"],
            }
            for value in verified_nodes
        ],
        "resources": {
            "cold_load_and_verify_seconds": cold_load_seconds,
            "peak_rss_bytes": _rss_bytes(),
            "score_artifact_bytes": sum(path.stat().st_size for path in node_paths)
            + score_manifest_path.stat().st_size,
            "actual_new_acquisition_steps": 0,
        },
        "coverage": {
            "source_nodes": len(source_nodes),
            "query_nodes": len(query_grid),
            "methods": len(ALL_METHOD_IDS),
            "queries_per_node": 30,
            "score_rows": len(verified_rows),
            "truth_joined_rows": len(per_query),
            "failures": 0,
        },
    }
    report = {**report_unsigned, "report_digest": sha256_json(report_unsigned)}
    _publish_or_match_json(report_path, report, resume=resume)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.5.0+ablation"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-root")
    parser.add_argument("--new-analysis-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    run_analysis(
        arguments.config,
        arguments.new_analysis_dir,
        artifacts_root=arguments.artifacts_root,
        resume=bool(arguments.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
