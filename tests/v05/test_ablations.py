from __future__ import annotations

from copy import deepcopy
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v05.ablations import (
    DeterministicRandomRanker,
    FixedFeatureLogReg,
    FixedFeatureRidge,
    RFFEpisodeFeatures,
    RawMomentNN,
    SWE1024NN,
    SummaryPrototypeNN,
    episode_balanced_moment_vector,
    nested_row_order,
)
from policy_learnware_v0.v05.classifiers import EpisodeBank, V05ClassifierError
from policy_learnware_v0.v05.specifications import RFFMap, SWEMap
from server.repro_fpo_ppo_v05.ablation_analysis import (
    ALL_METHOD_IDS,
    _load_analysis_config,
    _query_grid,
    _score_rows,
    _source_nodes,
    _validate_analysis_run_manifest,
    _validate_analysis_output_layout,
    _validate_node_score_closure,
    run_analysis,
)
from server.repro_fpo_ppo_v05.compute_scale_benchmark import (
    COMPUTE_RUN_SCHEMA,
    COMPUTE_SUMMARY_SCHEMA,
    ComputeScaleError,
    _benchmark_output_layout,
    _exploratory_identity,
    _fit_methods,
    _load_plan,
    _ofat_nodes,
    _score_digest,
    _slice_bank,
    _validate_bootstrap,
    _validate_completed_cell,
    run_benchmark,
)


def test_partial_analysis_resume_rejects_tampered_or_extra_manifest_fields() -> None:
    stable = {
        "schema": "policy-learnware.v05-ablation-run.v2",
        "scope": "SECONDARY_EXPLORATORY_POST_TRUTH",
        "formal_confirmatory": False,
        "analysis_config_digest": "a" * 64,
    }
    unsigned = {
        **stable,
        "created_at": "2026-08-29T12:00:00+00:00",
    }
    value = {**unsigned, "run_manifest_digest": sha256_json(unsigned)}
    assert _validate_analysis_run_manifest(value, stable) == value

    tampered_digest = {**value, "run_manifest_digest": "b" * 64}
    with pytest.raises(ValueError, match="digest differs"):
        _validate_analysis_run_manifest(tampered_digest, stable)
    with pytest.raises(ValueError, match="fields differ"):
        _validate_analysis_run_manifest({**value, "extra": 1}, stable)
    with pytest.raises(ValueError, match="created_at is invalid"):
        _validate_analysis_run_manifest({**value, "created_at": 7}, stable)
    with pytest.raises(ValueError, match="digest differs"):
        _validate_analysis_run_manifest(
            {**value, "created_at": "2099-08-29T12:00:00+00:00"}, stable
        )


def test_compute_bootstrap_binds_timings_and_exact_schema() -> None:
    stable = {
        "canonical_complete_digest": "a" * 64,
        "normalizer_digest": "b" * 64,
    }
    timing = {
        "wall_seconds": 1.0,
        "cpu_seconds": 0.5,
        "peak_rss_before_bytes": 100,
        "peak_rss_after_bytes": 120,
    }
    unsigned = {
        "schema": "policy-learnware.v05-compute-scale-bootstrap.v2",
        **stable,
        "r4_asset_load": timing,
        "full32_canonicalize": timing,
        "peak_rss_bytes": 120,
    }
    value = {**unsigned, "bootstrap_digest": sha256_json(unsigned)}
    assert _validate_bootstrap(value, stable) == value["bootstrap_digest"]

    tampered_timing = deepcopy(value)
    tampered_timing["r4_asset_load"]["wall_seconds"] = 2.0
    with pytest.raises(ComputeScaleError, match="bootstrap digest differs"):
        _validate_bootstrap(tampered_timing, stable)
    with pytest.raises(ComputeScaleError, match="fields differ"):
        _validate_bootstrap({**value, "extra": 1}, stable)
    invalid_rss = deepcopy(value)
    invalid_rss["full32_canonicalize"]["peak_rss_after_bytes"] = -1
    with pytest.raises(ComputeScaleError, match="peak_rss_after_bytes is invalid"):
        _validate_bootstrap(invalid_rss, stable)


def test_new_compute_outputs_self_describe_exploratory_scope() -> None:
    for schema in (COMPUTE_RUN_SCHEMA, COMPUTE_SUMMARY_SCHEMA):
        assert _exploratory_identity(schema) == {
            "schema": schema,
            "scope": "SECONDARY_EXPLORATORY_POST_TRUTH",
            "formal_confirmatory": False,
        }


def _bank(episodes: list[list[list[float]]]) -> EpisodeBank:
    arrays = [np.asarray(episode, dtype=np.float64) for episode in episodes]
    return EpisodeBank(
        np.concatenate(arrays, axis=0),
        np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum([len(episode) for episode in arrays], dtype=np.int64),
            )
        ),
    )


def test_ablation_entrypoints_use_the_shared_artifact_root() -> None:
    analysis_parameters = inspect.signature(run_analysis).parameters
    benchmark_parameters = inspect.signature(run_benchmark).parameters
    assert "artifacts_root" in analysis_parameters
    assert "artifacts_root" in benchmark_parameters
    assert "r4_root" not in analysis_parameters
    assert "r4_root" not in benchmark_parameters


def test_v03_moment_baseline_uses_one_episode_balanced_mixture() -> None:
    query = _bank([[[0.0]], [[2.0]]])
    assert np.allclose(episode_balanced_moment_vector(query), [1.0, 1.0, 2.0])

    per_episode_average = np.mean(
        [episode_balanced_moment_vector(_bank([[[value]]])) for value in (0.0, 2.0)],
        axis=0,
    )
    assert np.allclose(per_episode_average, [1.0, 0.0, 2.0])
    unequal = _bank([[[0.0]], [[2.0], [2.0], [2.0]]])
    assert np.allclose(episode_balanced_moment_vector(unequal), [1.0, 1.0, 2.0])

    model = RawMomentNN.fit(
        {
            "source-a": _bank([[[0.0]], [[2.0]]]),
            "source-b": _bank([[[1.0]], [[1.0]]]),
        },
        {"source-a": "class-a", "source-b": "class-b"},
    )
    scores = model.score(query)
    assert scores["class-a"] == 0.0
    assert scores["class-a"] > scores["class-b"]


def test_summary_nn_and_fixed_feature_heads_are_real_classifiers() -> None:
    source_train = {
        "source-a": _bank([[[-2.0], [-1.0]], [[-1.5], [-1.0]]]),
        "source-b": _bank([[[1.0], [2.0]], [[1.0], [1.5]]]),
    }
    labels = {"source-a": "class-a", "source-b": "class-b"}
    summary = SummaryPrototypeNN.fit(source_train, labels)
    summary_scores = summary.score(_bank([[[-1.8], [-1.2]]]))
    assert max(summary_scores, key=summary_scores.get) == "class-a"
    with pytest.raises(V05ClassifierError, match="width differs"):
        summary.score_vector([1.0])

    rff_map = RFFMap(
        input_dim=1,
        bandwidth=1.0,
        normalization_digest="0" * 64,
        frequency_count=16,
        public_seed=50_501,
    )
    train_features = {
        source_id: RFFEpisodeFeatures.from_bank(rff_map, bank)
        for source_id, bank in source_train.items()
    }
    source_validation = {
        "source-a": _bank([[[-1.5], [-1.0]]]),
        "source-b": _bank([[[1.0], [1.5]]]),
    }
    validation_features = {
        source_id: RFFEpisodeFeatures.from_bank(rff_map, bank)
        for source_id, bank in source_validation.items()
    }
    logreg = FixedFeatureLogReg.fit(
        train_features,
        labels,
        validation_features,
        l2_grid=(1.0e-2,),
        max_iter=2_000,
        tolerance=1.0e-8,
    )
    ridge = FixedFeatureRidge.fit(
        train_features,
        labels,
        validation_features,
        ridge_grid=(1.0e-2,),
    )
    left_features = RFFEpisodeFeatures.from_bank(rff_map, _bank([[[-1.5]]]))
    right_features = RFFEpisodeFeatures.from_bank(rff_map, _bank([[[1.5]]]))
    for model in (logreg, ridge):
        left = model.score_features(left_features)
        right = model.score_features(right_features)
        assert max(left, key=left.get) == "class-a"
        assert max(right, key=right.get) == "class-b"
        wrong_map = RFFEpisodeFeatures(rows=left_features.rows, map_digest="1" * 64)
        with pytest.raises(V05ClassifierError, match="map digests differ"):
            model.score_features(wrong_map)

    with pytest.raises(V05ClassifierError, match="max_iter must be positive"):
        FixedFeatureLogReg.fit(
            train_features,
            labels,
            validation_features,
            l2_grid=(1.0e-2,),
            max_iter=0,
        )


def test_random_is_deterministic_and_swe_dimension_control_is_1024() -> None:
    query = _bank([[[0.0, 1.0], [1.0, 0.0]]])
    random = DeterministicRandomRanker(("class-b", "class-a"), public_seed=50_503)
    first = random.score(public_query_token="q-public-1")
    assert first == random.score(public_query_token="q-public-1")
    assert first != random.score(public_query_token="q-public-2")

    swe_map = SWEMap(
        input_dim=2,
        normalization_digest="0" * 64,
        direction_count=32,
        quantile_count=32,
        public_seed=50_502,
    )
    model = SWE1024NN.fit({"source": query}, swe_map=swe_map)
    assert swe_map.output_dim == 1024
    assert model.method_id == "SWE_1024_NN"
    assert model.score(query)["source"] == 0.0


def test_ablation_plan_is_nested_l_shaped_and_compute_is_ofat() -> None:
    path = "configs/v05_ablation.yaml"
    analysis, analysis_digest, _, config_digest, _ = _load_analysis_config(path)
    plan, plan_digest, _, same_config_digest = _load_plan(path)
    assert analysis_digest == plan_digest
    assert config_digest == same_config_digest
    source_nodes = _source_nodes(analysis)
    query_grid = _query_grid(analysis)
    compute_nodes = _ofat_nodes(plan["compute_scale"])
    assert len(source_nodes) == 16  # 9 episode + 7 row nodes; U19/SR64 share fit.
    assert len({node.fit_key for node in source_nodes}) == 15
    assert len(query_grid) == 21
    assert len(compute_nodes) == 13
    base = plan["compute_scale"]["base"]
    assert all(
        sum(node[key] != base[key] for key in base) <= 2 for node in compute_nodes
    )  # E and V move as one preregistered factor.

    bank = _bank([[[float(row)] for row in range(64)] for _ in range(2)])
    parent = "a" * 64
    small = _slice_bank(
        bank,
        2,
        4,
        public_seed=50_504,
        parent_membership_digest=parent,
        physical_episode_start=19,
    )
    large = _slice_bank(
        bank,
        2,
        8,
        public_seed=50_504,
        parent_membership_digest=parent,
        physical_episode_start=19,
    )
    for episode in range(2):
        expected = nested_row_order(parent, 19 + episode, 50_504)
        assert small.episode(episode)[:, 0].tolist() == [float(i) for i in expected[:4]]
        assert large.episode(episode)[:4].tolist() == small.episode(episode).tolist()


def test_score_rows_rank_each_method_from_its_own_score_vector(monkeypatch) -> None:
    anchors = tuple(f"anchor-{index:02d}" for index in range(30))
    labels = {anchor: f"policy-{index:02d}" for index, anchor in enumerate(anchors)}
    assets = SimpleNamespace(
        task_by_anchor={
            anchor: f"task-{index // 5}" for index, anchor in enumerate(anchors)
        },
        parent_membership_digest={
            anchor: f"{index + 1:064x}" for index, anchor in enumerate(anchors)
        },
        probe_protocol_digest="b" * 64,
    )
    bundle = SimpleNamespace(
        labels=labels,
        model_manifest_digest="c" * 64,
        canonicalizer=SimpleNamespace(
            normalizer=SimpleNamespace(normalizer_digest="d" * 64)
        ),
    )
    query = _bank([[[0.0]]])
    monkeypatch.setattr(
        "server.repro_fpo_ppo_v05.ablation_analysis._query_bank",
        lambda *args, **kwargs: (query, 0.0),
    )

    def fake_scores(*args, **kwargs):
        by_method = {}
        for method_index, method_id in enumerate(ALL_METHOD_IDS):
            direction = 1.0 if method_index % 2 == 0 else -1.0
            by_method[method_id] = {
                labels[anchor]: direction * index
                for index, anchor in enumerate(anchors)
            }
        return by_method, {}, {method_id: 0.0 for method_id in ALL_METHOD_IDS}

    monkeypatch.setattr(
        "server.repro_fpo_ppo_v05.ablation_analysis._policy_scores", fake_scores
    )
    rows, _ = _score_rows(
        bundle,
        assets,
        family="QUERY_GRID",
        node_id="Q-B01-R01",
        budgets=(1,),
        rows_per_episode=1,
        analysis_digest="e" * 64,
        tie_digest="f" * 64,
        row_seed=50_504,
    )
    market = [
        row
        for row in rows
        if row["opaque_query_id"] == rows[0]["opaque_query_id"]
        and row["endpoint"] == "MARKET_30_CERT"
    ]
    top_by_method = {row["method_id"]: row["ranked_anchor_ids"][0] for row in market}
    assert top_by_method[ALL_METHOD_IDS[0]] == anchors[-1]
    assert top_by_method[ALL_METHOD_IDS[1]] == anchors[0]
    assert {row["method_id"]: row["ranked_policy_ids"][0] for row in market} == {
        method_id: labels[top] for method_id, top in top_by_method.items()
    }

    unsigned = {
        "schema": "policy-learnware.v05-ablation-score-node.v1",
        "status": "COMPLETE_PRE_METRIC_JOIN",
        "scope": "SECONDARY_EXPLORATORY_POST_TRUTH",
        "truth_blinding_status": "NOT_CLAIMED_POST_TRUTH_ANALYSIS",
        "input_digest": "0" * 64,
        "family": "QUERY_GRID",
        "node_id": "Q-B01-R01",
        "specification": {
            "node_id": "Q-B01-R01",
            "family": "QUERY_GRID",
            "fixed_source": {
                "train_episodes": 19,
                "validation_episodes": 6,
                "rows_per_episode": 64,
            },
            "budget_episodes": 1,
            "rows_per_episode": 1,
        },
        "source_digest": "1" * 64,
        "source_model_manifest_digest": "c" * 64,
        "ranking_tie_digest": "f" * 64,
        "candidate_anchor_order": list(anchors),
        "candidate_policy_order": [labels[anchor] for anchor in anchors],
        "normalizer_digest": "d" * 64,
        "canonicalizer_digest": "2" * 64,
        "bandwidth": 1.0,
        "model_nbytes_by_method": {method: 0 for method in ALL_METHOD_IDS},
        "fit_cache_reused_in_process": False,
        "timing": {},
        "cost": {},
        "peak_rss_bytes": 0,
        "score_row_count": len(rows),
        "score_rows_digest": sha256_json(rows),
        "score_rows": rows,
    }
    value = {**unsigned, "node_digest": sha256_json(unsigned)}
    _validate_node_score_closure(value)
    tampered = deepcopy(value)
    tampered["score_rows"][0]["scores_before_mask"][0] += 1.0
    tampered["score_rows_digest"] = sha256_json(tampered["score_rows"])
    tampered_unsigned = {
        key: item for key, item in tampered.items() if key != "node_digest"
    }
    tampered["node_digest"] = sha256_json(tampered_unsigned)
    with pytest.raises(ValueError, match="score-vector digest"):
        _validate_node_score_closure(tampered)


def test_compute_score_namespace_and_resume_identity_are_exact() -> None:
    policies = ("policy-a", "policy-b")
    _score_digest("METHOD", [{"policy-a": 1.0, "policy-b": 0.0}], policies)
    with pytest.raises(ValueError, match="policy score vector"):
        _score_digest("METHOD", [{"anchor-a": 1.0, "anchor-b": 0.0}], policies)

    node = {"cell_id": "BASE", "factor": "BASE"}
    cell = {
        "schema": "policy-learnware.v05-compute-scale-cell.v1",
        "status": "COMPLETE",
        "cell_id": "BASE",
        "input_digest": "a" * 64,
        "node": node,
        "applicable_methods": ["METHOD"],
        "not_applicable_methods": {},
        "methods": {"METHOD": {}},
    }
    cell["cell_digest"] = sha256_json(cell)
    _validate_completed_cell(
        cell,
        node,
        expected_input_digest="a" * 64,
        applicable=("METHOD",),
        not_applicable=(),
    )
    stale = {**cell, "input_digest": "b" * 64}
    stale["cell_digest"] = sha256_json(
        {key: item for key, item in stale.items() if key != "cell_digest"}
    )
    with pytest.raises(ValueError, match="completed cell changed"):
        _validate_completed_cell(
            stale,
            node,
            expected_input_digest="a" * 64,
            applicable=("METHOD",),
            not_applicable=(),
        )


def test_compute_matrix_fits_and_scores_in_one_policy_namespace() -> None:
    plan, _, config, _ = _load_plan("configs/v05_ablation.yaml")
    config = deepcopy(config)
    config["raw_delta_rkme"] = {
        **config["raw_delta_rkme"],
        "support_budget": 2,
        "support_steps": 5,
        "kmeans_steps": 2,
    }
    train = {
        "anchor-a": _bank([[[-2.0], [-1.0]], [[-1.5], [-1.0]]]),
        "anchor-b": _bank([[[1.0], [2.0]], [[1.0], [1.5]]]),
    }
    validation = {
        "anchor-a": _bank([[[-1.5], [-1.0]]]),
        "anchor-b": _bank([[[1.0], [1.5]]]),
    }
    labels = {"anchor-a": "policy-a", "anchor-b": "policy-b"}
    node = {
        "rff_frequency_count": 8,
        "swe_direction_count": 4,
        "swe_quantile_count": 4,
    }
    models, _ = _fit_methods(
        ALL_METHOD_IDS,
        train,
        validation,
        labels,
        bandwidth=1.0,
        normalizer_digest="0" * 64,
        node=node,
        plan=plan,
        config=config,
    )
    query = _bank([[[-1.8], [-1.2]]])
    for method_id, (_, encoder, scorer, _) in models.items():
        encoded = query if encoder is None else encoder(query)
        scores = scorer(encoded)
        assert set(scores) == {"policy-a", "policy-b"}, method_id
        assert all(np.isfinite(value) for value in scores.values())


def test_compute_output_layout_rejects_resume_cells_symlink(tmp_path) -> None:
    output = tmp_path / "analysis"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "cells").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="contains a symlink"):
        _benchmark_output_layout(output, ({"cell_id": "BASE"},))
    assert list(outside.iterdir()) == []


def test_analysis_output_layout_rejects_unexpected_artifact(tmp_path) -> None:
    config, _, _, _, _ = _load_analysis_config("configs/v05_ablation.yaml")
    output = tmp_path / "analysis"
    output.mkdir()
    stale = output / "stale.json"
    stale.write_bytes(b"stale")

    with pytest.raises(ValueError, match="unexpected artifacts"):
        _validate_analysis_output_layout(
            output, _source_nodes(config), _query_grid(config)
        )
    assert tuple(output.iterdir()) == (stale,)
    assert stale.read_bytes() == b"stale"


def test_compute_output_layout_rejects_unexpected_artifact(tmp_path) -> None:
    plan, _, _, _ = _load_plan("configs/v05_ablation.yaml")
    output = tmp_path / "analysis"
    output.mkdir()
    stale = output / "stale.json"
    stale.write_bytes(b"stale")

    with pytest.raises(ComputeScaleError, match="unexpected artifacts"):
        _benchmark_output_layout(output, _ofat_nodes(plan["compute_scale"]))
    assert tuple(output.iterdir()) == (stale,)
    assert stale.read_bytes() == b"stale"
