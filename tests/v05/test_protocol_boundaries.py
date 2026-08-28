from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    BudgetLedger,
    derive_probe_membership,
    seal_rankings,
    tie_break_key,
)
from policy_learnware_v0.v05.classifiers import (
    KME_KRR,
    P0_METHOD_IDS,
    SUMMARY_LOGREG,
    EpisodeBank,
)
from policy_learnware_v0.v05.labels import (
    CertificateBinding,
    CertificateResolver,
    CertifiedPolicyManifest,
    V05LabelError,
    project_certificate_manifest,
)
from policy_learnware_v0.v05.metrics import (
    MARKET_30_CERT,
    TASK_5_CERT,
    PredictionRanking,
    TruthBinding,
    V05MetricError,
    evaluate_sealed_predictions,
    normalized_log2_budget_auc,
    prediction_payload,
)
from server.repro_fpo_ppo_v05 import exact_repeat_collector
from server.repro_fpo_ppo_v05.environment_classifier_runner import (
    P1_STATUS,
    V05RunnerError,
    fit_p0_panel,
    score_query,
    seal_prediction_rows,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


POLICY_A = "lw-" + _digest("policy-a")[:20]
POLICY_B = "lw-" + _digest("policy-b")[:20]
ROW_BINDING = {
    "probe_protocol_digest": _digest("probe-protocol"),
    "reward_free_bank_sha256": _digest("reward-free-bank"),
    "canonical_query_bank_digest": _digest("canonical-query-bank"),
    "source_train_membership_digest": _digest("source-train-membership"),
    "source_validation_membership_digest": _digest("source-validation-membership"),
    "source_repeat_membership_digest": _digest("source-repeat-membership"),
    "target_membership_digest": _digest("target-membership"),
    "normalization_digest": _digest("normalization"),
    "config_digest": _digest("config"),
    "source_model_manifest_digest": _digest("source-model-manifest"),
}


def _bank(*episodes: list[float]) -> EpisodeBank:
    arrays = [
        np.asarray(episode, dtype=np.float64).reshape(-1, 1) for episode in episodes
    ]
    return EpisodeBank(
        np.concatenate(arrays),
        np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum([len(array) for array in arrays], dtype=np.int64),
            )
        ),
    )


@pytest.fixture
def many_to_one_manifest() -> CertifiedPolicyManifest:
    return project_certificate_manifest(
        {"anchor-a1": POLICY_A, "anchor-a2": POLICY_A, "anchor-b1": POLICY_B},
        task_by_anchor={
            "anchor-a1": "task-a",
            "anchor-a2": "task-a",
            "anchor-b1": "task-b",
        },
        policy_bundle_digest_by_policy={
            POLICY_A: _digest("bundle-a"),
            POLICY_B: _digest("bundle-b"),
        },
        championization_admission_digest_by_anchor={
            "anchor-a1": _digest("champion-a1"),
            "anchor-a2": _digest("champion-a2"),
            "anchor-b1": _digest("champion-b1"),
        },
        execution_abi_digest_by_policy={
            POLICY_A: _digest("abi-a"),
            POLICY_B: _digest("abi-b"),
        },
        expected_anchor_ids=("anchor-a1", "anchor-a2", "anchor-b1"),
    )


@pytest.fixture
def truth_join() -> tuple[TruthBinding, ...]:
    return (
        TruthBinding("query-a1", "anchor-a1", "task-a", POLICY_A),
        TruthBinding("query-a2", "anchor-a2", "task-a", POLICY_A),
        TruthBinding("query-b1", "anchor-b1", "task-b", POLICY_B),
    )


def _prediction_rows(budget: int = 1) -> tuple[PredictionRanking, ...]:
    rows: list[PredictionRanking] = []
    for query_id, task_id in (
        ("query-a1", "task-a"),
        ("query-a2", "task-a"),
        ("query-b1", "task-b"),
    ):
        rows.append(
            PredictionRanking(
                "METHOD",
                MARKET_30_CERT,
                budget,
                query_id,
                ("anchor-a1", "anchor-a2", "anchor-b1"),
                (POLICY_A, POLICY_B),
                **ROW_BINDING,
            )
        )
        rows.append(
            PredictionRanking(
                "METHOD",
                TASK_5_CERT,
                budget,
                query_id,
                (("anchor-a1", "anchor-a2") if task_id == "task-a" else ("anchor-b1",)),
                (POLICY_A,) if task_id == "task-a" else (POLICY_B,),
                **ROW_BINDING,
            )
        )
    return tuple(rows)


def test_certificate_one_to_one_and_many_to_one_public_mask(
    many_to_one_manifest: CertifiedPolicyManifest,
) -> None:
    one_to_one = project_certificate_manifest(
        {"anchor-a1": POLICY_A, "anchor-b1": POLICY_B},
        task_by_anchor={"anchor-a1": "task-a", "anchor-b1": "task-b"},
        policy_bundle_digest_by_policy={
            POLICY_A: _digest("bundle-a"),
            POLICY_B: _digest("bundle-b"),
        },
        championization_admission_digest_by_anchor={
            "anchor-a1": _digest("champion-a1"),
            "anchor-b1": _digest("champion-b1"),
        },
        execution_abi_digest_by_policy={
            POLICY_A: _digest("abi-a"),
            POLICY_B: _digest("abi-b"),
        },
    )
    assert CertificateResolver(one_to_one).anchor_to_policy == {
        "anchor-a1": POLICY_A,
        "anchor-b1": POLICY_B,
    }

    resolver = CertificateResolver(many_to_one_manifest)
    scores = {"anchor-a1": 0.2, "anchor-a2": 0.3, "anchor-b1": 0.8}
    assert resolver.policy_to_anchors[POLICY_A] == ("anchor-a1", "anchor-a2")
    assert resolver.aggregate_anchor_scores(scores) == {
        POLICY_A: pytest.approx(0.5),
        POLICY_B: pytest.approx(0.8),
    }
    assert resolver.rank_policies(scores) == (POLICY_B, POLICY_A)
    assert resolver.aggregate_anchor_scores(
        scores, candidate_anchor_ids=("anchor-a1", "anchor-a2")
    ) == {POLICY_A: pytest.approx(0.5)}


def test_manifest_roundtrip_and_strict_coverage(
    many_to_one_manifest: CertifiedPolicyManifest,
) -> None:
    payload = many_to_one_manifest.to_dict()
    assert CertifiedPolicyManifest.from_dict(payload) == many_to_one_manifest
    assert (
        CertifiedPolicyManifest.from_json(many_to_one_manifest.canonical_json)
        == many_to_one_manifest
    )

    tampered = many_to_one_manifest.to_dict()
    tampered["bindings"][0]["task_id"] = "changed-task"
    with pytest.raises(V05LabelError, match="digest"):
        CertifiedPolicyManifest.from_dict(tampered)
    with pytest.raises(V05LabelError, match="coverage"):
        CertifiedPolicyManifest.from_records(
            many_to_one_manifest.bindings,
            expected_anchor_ids=("anchor-a1", "anchor-a2"),
        )
    with pytest.raises(V05LabelError, match="coverage"):
        project_certificate_manifest(
            {"anchor-a1": POLICY_A},
            task_by_anchor={"anchor-a1": "task-a"},
            policy_bundle_digest_by_policy={},
            championization_admission_digest_by_anchor={
                "anchor-a1": _digest("champion-a1")
            },
            execution_abi_digest_by_policy={POLICY_A: _digest("abi-a")},
        )
    with pytest.raises(V05LabelError, match="opaque"):
        CertificateBinding(
            "anchor",
            "task",
            "policy-a",
            _digest("bundle"),
            _digest("champion"),
            _digest("abi"),
        )


def test_ranking_requires_seal_before_truth_join(
    many_to_one_manifest: CertifiedPolicyManifest,
    truth_join: tuple[TruthBinding, ...],
) -> None:
    unsealed = prediction_payload(_prediction_rows())
    with pytest.raises(V05MetricError, match="RankingSeal"):
        evaluate_sealed_predictions(  # type: ignore[arg-type]
            unsealed, truth_join, many_to_one_manifest
        )

    sealed = seal_rankings(unsealed)
    result = evaluate_sealed_predictions(
        sealed,
        truth_join,
        many_to_one_manifest,
        expected_method_ids=("METHOD",),
        expected_budgets=(1,),
    )
    assert result.prediction_seal_digest == sealed.rankings_digest
    assert result.certificate_manifest_digest == many_to_one_manifest.manifest_digest


def test_market_and_task_endpoints_use_task_equal_macro(
    many_to_one_manifest: CertifiedPolicyManifest,
    truth_join: tuple[TruthBinding, ...],
) -> None:
    result = evaluate_sealed_predictions(
        seal_rankings(prediction_payload(_prediction_rows())),
        truth_join,
        many_to_one_manifest,
        recall_ks=(1, 2),
        expected_method_ids=("METHOD",),
        expected_budgets=(1,),
    )
    metrics = {item.endpoint: item for item in result.metrics}

    market = metrics[MARKET_30_CERT]
    assert market.anchor_hit_at_1 == pytest.approx(0.25)
    assert market.anchor_mrr == pytest.approx((0.75 + 1.0 / 3.0) / 2.0)
    assert market.policy_hit_at_1 == pytest.approx(0.5)
    assert market.policy_mrr == pytest.approx(0.75)
    assert market.anchor_recall_at_k[2] == pytest.approx(0.5)
    assert market.policy_recall_at_k[2] == pytest.approx(1.0)

    conditional = metrics[TASK_5_CERT]
    assert conditional.anchor_hit_at_1 == pytest.approx(0.75)
    assert conditional.policy_hit_at_1 == pytest.approx(1.0)
    assert conditional.task_count == 2


def test_log_budget_auc_is_normalized_and_order_invariant() -> None:
    assert normalized_log2_budget_auc({4: 0.0, 1: 0.0, 2: 1.0}) == pytest.approx(0.5)
    assert normalized_log2_budget_auc((1, 2), (0.25, 0.75)) == pytest.approx(0.5)


def test_nan_duplicate_rank_and_missing_coverage_fail_closed(
    many_to_one_manifest: CertifiedPolicyManifest,
    truth_join: tuple[TruthBinding, ...],
) -> None:
    resolver = CertificateResolver(many_to_one_manifest)
    with pytest.raises(V05LabelError, match="finite"):
        resolver.aggregate_anchor_scores(
            {"anchor-a1": float("nan"), "anchor-a2": 0.0, "anchor-b1": 0.0}
        )
    with pytest.raises(V05MetricError, match="finite"):
        normalized_log2_budget_auc({1: 0.0, 2: float("nan")})
    with pytest.raises(V05MetricError, match="duplicate"):
        PredictionRanking(
            "METHOD",
            MARKET_30_CERT,
            1,
            "query-a1",
            ("anchor-a1", "anchor-a1"),
            (POLICY_A, POLICY_B),
            **ROW_BINDING,
        )

    rows = _prediction_rows()
    missing_query = tuple(row for row in rows if row.opaque_query_id != "query-b1")
    with pytest.raises(V05MetricError, match="cover"):
        evaluate_sealed_predictions(
            seal_rankings(prediction_payload(missing_query)),
            truth_join,
            many_to_one_manifest,
            expected_method_ids=("METHOD",),
            expected_budgets=(1,),
        )

    missing_cell = rows[:-1]
    with pytest.raises(V05MetricError, match="complete"):
        evaluate_sealed_predictions(
            seal_rankings(prediction_payload(missing_cell)),
            truth_join,
            many_to_one_manifest,
            expected_method_ids=("METHOD",),
            expected_budgets=(1,),
        )

    bad_candidate = list(rows)
    original = bad_candidate[0]
    bad_candidate[0] = PredictionRanking(
        original.method_id,
        original.endpoint,
        original.budget_episodes,
        original.opaque_query_id,
        ("anchor-a1", "anchor-a2"),
        original.ranked_policy_ids,
        **ROW_BINDING,
    )
    with pytest.raises(V05MetricError, match="candidate set"):
        evaluate_sealed_predictions(
            seal_rankings(prediction_payload(bad_candidate)),
            truth_join,
            many_to_one_manifest,
            expected_method_ids=("METHOD",),
            expected_budgets=(1,),
        )


def test_runner_binds_all_p0_to_one_panel_then_masks_ranks_and_seals() -> None:
    anchors = ("anchor-a", "anchor-b")
    manifest = project_certificate_manifest(
        {"anchor-a": POLICY_A, "anchor-b": POLICY_B},
        task_by_anchor={"anchor-a": "task", "anchor-b": "task"},
        policy_bundle_digest_by_policy={
            POLICY_A: _digest("bundle-a"),
            POLICY_B: _digest("bundle-b"),
        },
        championization_admission_digest_by_anchor={
            "anchor-a": _digest("champion-a"),
            "anchor-b": _digest("champion-b"),
        },
        execution_abi_digest_by_policy={
            POLICY_A: _digest("abi-a"),
            POLICY_B: _digest("abi-b"),
        },
    )
    # Identical anchor evidence makes tie handling observable without changing
    # the fact that every method receives exactly the same role banks.
    train = {anchor: _bank([0.0, 0.1], [0.2, 0.3]) for anchor in anchors}
    validation = {anchor: _bank([0.1, 0.2]) for anchor in anchors}
    repeat = {anchor: _bank([0.15, 0.25]) for anchor in anchors}
    config_digest = _digest("runner-config")
    probe_digest = _digest("q0-probe")
    normalization_digest = _digest("source-normalizer")
    panel = fit_p0_panel(
        train,
        validation,
        repeat,
        manifest,
        config_digest=config_digest,
        probe_protocol_digest=probe_digest,
        normalization_digest=normalization_digest,
        source_train_membership_digest=_digest("train-membership"),
        source_validation_membership_digest=_digest("validation-membership"),
        source_repeat_membership_digest=_digest("repeat-membership"),
        bandwidth=0.8,
        expected_source_count=2,
        expected_episode_counts=(2, 1, 1),
        reducer_config=ReducerConfig(
            support_budget=4,
            support_steps=0,
            kmeans_steps=0,
            ridge=0.0,
            pinv_rcond=1.0e-12,
        ),
        rff_frequency_count=8,
        swe_direction_count=2,
        swe_quantile_count=4,
        logreg_l2_grid=(0.1,),
        krr_ridge_grid=(0.1,),
    )
    assert panel.source_model_manifest["p1_status"] == P1_STATUS
    assert tuple(panel.source_model_manifest["p0_method_ids"]) == P0_METHOD_IDS
    assert len(panel.method_cards) == 6
    assert all(
        card["source_binding"] == panel.source_binding for card in panel.method_cards
    )
    assert panel.source_model_manifest["source_binding"] == panel.source_binding
    model_digest = panel.source_model_manifest["source_model_manifest_digest"]
    with pytest.raises(V05RunnerError, match="bandwidth"):
        replace(panel, kme_krr=replace(panel.kme_krr, bandwidth=0.9))
    raw_source = panel.raw.sources["anchor-a"]
    raw_tampered = replace(
        panel,
        raw=replace(
            panel.raw,
            sources={
                **dict(panel.raw.sources),
                "anchor-a": replace(
                    raw_source, rkme_norm2=raw_source.rkme_norm2 + 0.125
                ),
            },
        ),
    )
    assert (
        raw_tampered.source_model_manifest["source_model_manifest_digest"]
        != model_digest
    )
    empirical_source = panel.empirical_mmd.sources["anchor-a"]
    empirical_weights = empirical_source.weights.copy()
    empirical_weights[:2] += np.asarray([0.01, -0.01])
    empirical_tampered = replace(
        panel,
        empirical_mmd=replace(
            panel.empirical_mmd,
            sources={
                **dict(panel.empirical_mmd.sources),
                "anchor-a": replace(empirical_source, weights=empirical_weights),
            },
        ),
    )
    assert (
        empirical_tampered.source_model_manifest["source_model_manifest_digest"]
        != model_digest
    )

    predictions, rows = score_query(
        panel,
        _bank([0.1, 0.2], [1.5, 1.6]),
        opaque_query_id="q-" + _digest("query")[:20],
        candidate_anchor_ids=("anchor-a",),
        probe_protocol_digest=probe_digest,
        reward_free_bank_sha256=_digest("query-bank"),
        target_membership_digest=_digest("target-membership"),
        normalization_digest=normalization_digest,
        budgets=(1, 2),
        expected_task_candidate_count=1,
    )
    assert len(predictions) == len(rows) == 24
    assert all(row["source_binding"] == panel.source_binding for row in rows)
    for budget in (1, 2):
        budget_rows = [row for row in rows if row["budget_episodes"] == budget]
        assert all(
            row["target_binding"] == budget_rows[0]["target_binding"]
            for row in budget_rows
        )
    assert all(
        prediction.source_model_manifest_digest
        == panel.source_model_manifest["source_model_manifest_digest"]
        for prediction in predictions
    )
    assert {
        (
            row["ledger"]["visible_transition_count"],
            row["ledger"]["interaction_cost_steps"],
        )
        for row in rows
    } == {
        (64, 1_000),
        (128, 2_000),
    }

    indexed_rows = {
        (row["method_id"], row["budget_episodes"], row["endpoint"]): row for row in rows
    }
    indexed_predictions = {
        (item.method_id, item.budget_episodes, item.endpoint): item
        for item in predictions
    }
    anchor_to_policy = panel.resolver.anchor_to_policy
    for method_id in P0_METHOD_IDS:
        for budget in (1, 2):
            market = indexed_rows[(method_id, budget, MARKET_30_CERT)]
            masked = indexed_rows[(method_id, budget, TASK_5_CERT)]
            assert market["scores_before_mask"] == masked["scores_before_mask"]
            assert len(market["scores_before_mask"]) == 2
            scores = market["scores_before_mask"]
            anchor_scores = (
                {anchor: scores[anchor_to_policy[anchor]] for anchor in anchors}
                if method_id in {SUMMARY_LOGREG, KME_KRR}
                else scores
            )
            expected_rank = tuple(
                sorted(
                    anchors,
                    key=lambda anchor: (
                        -anchor_scores[anchor],
                        tie_break_key(config_digest, anchor),
                        anchor,
                    ),
                )
            )
            assert (
                indexed_predictions[
                    (method_id, budget, MARKET_30_CERT)
                ].ranked_anchor_ids
                == expected_rank
            )
            assert indexed_predictions[
                (method_id, budget, TASK_5_CERT)
            ].ranked_anchor_ids == ("anchor-a",)
    assert (
        indexed_rows[(P0_METHOD_IDS[0], 1, MARKET_30_CERT)]["scores_before_mask"]
        != indexed_rows[(P0_METHOD_IDS[0], 2, MARKET_30_CERT)]["scores_before_mask"]
    )

    payload, ranking_seal = seal_prediction_rows(predictions, expected_budgets=(1, 2))
    assert ranking_seal.verify(payload)


def test_collector_q0_has_no_policy_surface_and_preserves_nested_dual_costs(
    monkeypatch, tmp_path
) -> None:
    forbidden = ("policy", "label", "candidate", "reward")
    for function in (
        exact_repeat_collector.collect_common_probe,
        exact_repeat_collector.q0_probe_protocol,
        exact_repeat_collector.derive_episode_seed_plan,
    ):
        parameters = inspect.signature(function).parameters
        assert not any(
            token in name.lower() for name in parameters for token in forbidden
        )

    card = exact_repeat_collector.q0_probe_protocol(sigma=0.35)
    assert card["protocol_id"] == "Q0_COMMON_GAUSSIAN_OPEN_LOOP"
    assert card["feedback_mode"] == "open_loop"
    assert card["full_episode_draw"] is True
    assert card["state_access_for_action"] is False
    assert card["reward_access"] is card["label_access"] is False
    assert card["candidate_policy_access"] is False
    assert card["candidate_conditioned_steps"] == card["reward_queries"] == 0
    assert card["budget_episodes"] == list(BUDGET_EPISODES)

    membership = derive_probe_membership("collector-context", 17)
    budget_one = membership.for_budget(1)
    budget_two = membership.for_budget(2)
    assert budget_two[: len(budget_one)] == budget_one
    assert len(budget_one) == 64 and len(budget_two) == 128
    ledger_one, ledger_two = BudgetLedger.for_budget(1), BudgetLedger.for_budget(2)
    assert (ledger_one.visible_transition_count, ledger_one.interaction_cost_steps) == (
        64,
        1_000,
    )
    assert (ledger_two.visible_transition_count, ledger_two.interaction_cost_steps) == (
        128,
        2_000,
    )
    assert ledger_one.candidate_conditioned_steps == ledger_two.reward_queries == 0

    class PolicyTrapAdapter:
        schema = SimpleNamespace(
            observation_dim=1,
            action_dim=1,
            horizon=1_000,
            action_low=np.array([-1.0]),
            action_high=np.array([1.0]),
            digest=_digest("fake-environment-schema"),
        )

        @property
        def policy(self):  # pragma: no cover - access is the failure mode
            raise AssertionError("the common probe accessed a candidate policy")

        def reset(self, seed):
            return 0, np.array([float(seed % 11)])

        def step(self, state, action):
            next_state = state + 1
            return next_state, SimpleNamespace(
                observation=np.array([float(next_state)]),
                terminated=False,
                truncated=False,
            )

    monkeypatch.setattr(
        exact_repeat_collector,
        "_sample_actions",
        lambda **unused: np.zeros((1_000, 1), dtype=np.float32),
    )
    collection = exact_repeat_collector.collect_common_probe(
        context_id="collector-context",
        env_factory=PolicyTrapAdapter,
        seed_namespace="v05-test",
        membership_seed=17,
        sigma=0.35,
    )
    assert collection.probe.episode_count == 32
    assert collection.reward_free_bank_digest
    output = tmp_path / "v05-collector-cell"
    index = exact_repeat_collector.publish_collection(collection, output_dir=output)
    assert index["index_digest"]
    restored = exact_repeat_collector.load_published_collection(output)
    assert restored.reward_free_bank_digest == collection.reward_free_bank_digest
    with np.load(output / exact_repeat_collector.BANK_FILE, allow_pickle=False) as bank:
        assert set(bank.files) == exact_repeat_collector.BANK_ARRAYS
        assert not {"reward", "policy", "label", "candidate"}.intersection(bank.files)
    with pytest.raises(exact_repeat_collector.V05CollectorError, match="overwrite"):
        exact_repeat_collector.publish_collection(collection, output_dir=output)
    resumed = exact_repeat_collector.publish_collection(
        collection, output_dir=output, resume=True
    )
    assert resumed["index_digest"] == index["index_digest"]
