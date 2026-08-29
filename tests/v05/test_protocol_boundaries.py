from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import inspect
import json
import shutil
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policy_learnware_v0.hashing import canonical_json_bytes, sha256_file, sha256_json
from policy_learnware_v0.io import atomic_write_json, read_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v04a.protocol import (
    BUDGET_EPISODES,
    BudgetLedger,
    derive_probe_membership,
    seal_rankings,
    tie_break_key,
)
from policy_learnware_v0.v05.classifiers import (
    P0_METHOD_IDS,
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
from policy_learnware_v0.v05.specifications import RFFMap, SWEMap
from server.repro_fpo_ppo_v05 import exact_repeat_collector
from server.repro_fpo_ppo_v05 import blind_query_bank as blind_query_module
from server.repro_fpo_ppo_v05 import environment_classifier_runner as runner_module
from server.repro_fpo_ppo_v05.blind_query_bank import (
    AuthorizedQueryViews,
    V05BlindError,
    load_private_truth_binding,
    prepare_blinded_episode_bank,
    project_verified_source_banks,
)
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
    "target_membership_digest": _digest("target-membership"),
    "normalization_digest": _digest("normalization"),
    "config_digest": _digest("config"),
    "source_model_manifest_digest": _digest("source-model-manifest"),
    "authorized_query_manifest_digest": _digest("authorized-query-manifest"),
    "score_vector_digest": _digest("score-vector"),
    "budget_ledger_digest": _digest("budget-ledger"),
}


def _query_binding(query_id: str) -> dict[str, str]:
    return {
        **ROW_BINDING,
        "authorized_query_manifest_digest": _digest(f"{query_id}-authorized-manifest"),
    }


def _bank(*episodes: list[float]) -> EpisodeBank:
    arrays = []
    for episode in episodes:
        array = np.asarray(episode, dtype=np.float64)
        arrays.append(array.reshape(-1, 1) if array.ndim == 1 else array)
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
    seal_digest = seal_rankings(prediction_payload(_prediction_rows())).rankings_digest
    return (
        TruthBinding(
            "query-a1",
            "anchor-a1",
            "task-a",
            POLICY_A,
            _query_binding("query-a1")["authorized_query_manifest_digest"],
            seal_digest,
        ),
        TruthBinding(
            "query-a2",
            "anchor-a2",
            "task-a",
            POLICY_A,
            _query_binding("query-a2")["authorized_query_manifest_digest"],
            seal_digest,
        ),
        TruthBinding(
            "query-b1",
            "anchor-b1",
            "task-b",
            POLICY_B,
            _query_binding("query-b1")["authorized_query_manifest_digest"],
            seal_digest,
        ),
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
                **_query_binding(query_id),
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
                **_query_binding(query_id),
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

    dummy_rows = tuple(
        replace(
            row,
            ranked_anchor_ids=tuple(reversed(row.ranked_anchor_ids)),
            ranked_policy_ids=tuple(reversed(row.ranked_policy_ids)),
        )
        for row in _prediction_rows()
    )
    dummy_seal = seal_rankings(prediction_payload(dummy_rows))
    dummy_release = tuple(
        replace(item, prediction_seal_digest=dummy_seal.rankings_digest)
        for item in truth_join
    )
    with pytest.raises(V05MetricError, match="another prediction seal"):
        evaluate_sealed_predictions(
            sealed,
            dummy_release,
            many_to_one_manifest,
            expected_method_ids=("METHOD",),
            expected_budgets=(1,),
        )

    swapped_truth = (
        replace(truth_join[0], opaque_query_id=truth_join[1].opaque_query_id),
        replace(truth_join[1], opaque_query_id=truth_join[0].opaque_query_id),
        truth_join[2],
    )
    with pytest.raises(V05MetricError, match="manifest binding"):
        evaluate_sealed_predictions(
            sealed,
            swapped_truth,
            many_to_one_manifest,
            expected_method_ids=("METHOD",),
            expected_budgets=(1,),
        )


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
    missing_query_seal = seal_rankings(prediction_payload(missing_query))
    missing_query_truth = tuple(
        replace(item, prediction_seal_digest=missing_query_seal.rankings_digest)
        for item in truth_join
    )
    with pytest.raises(V05MetricError, match="cover"):
        evaluate_sealed_predictions(
            missing_query_seal,
            missing_query_truth,
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
        **_query_binding(original.opaque_query_id),
    )
    bad_candidate_seal = seal_rankings(prediction_payload(bad_candidate))
    bad_candidate_truth = tuple(
        replace(item, prediction_seal_digest=bad_candidate_seal.rankings_digest)
        for item in truth_join
    )
    with pytest.raises(V05MetricError, match="candidate set"):
        evaluate_sealed_predictions(
            bad_candidate_seal,
            bad_candidate_truth,
            many_to_one_manifest,
            expected_method_ids=("METHOD",),
            expected_budgets=(1,),
        )


def test_runner_binds_all_p0_to_one_panel_then_masks_ranks_and_seals(
    tmp_path, monkeypatch
) -> None:
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
    full_banks = {
        anchor: _bank(
            *(
                np.column_stack(
                    (
                        anchor_index + episode * 0.01 + np.linspace(0.0, 0.005, 64),
                        np.linspace(-0.2, 0.2, 64),
                    )
                )
                for episode in range(32)
            )
        )
        for anchor_index, anchor in enumerate(anchors)
    }
    parent_asset_sha256 = {
        anchor: _digest(f"{anchor}-parent-file") for anchor in anchors
    }
    parent_membership_digest = {
        anchor: _digest(f"{anchor}-parent-membership") for anchor in anchors
    }
    projection = project_verified_source_banks(
        full_banks,
        parent_asset_sha256=parent_asset_sha256,
        parent_membership_digest=parent_membership_digest,
        expected_source_count=2,
    )
    assert {
        role: row["episode_count"] for role, row in projection.manifest["roles"].items()
    } == {
        "source_train": 19,
        "source_validation": 6,
        "source_repeat_report": 7,
    }
    role_positions = [
        set(row["episode_positions"])
        for row in projection.manifest["sources"]["anchor-a"]["roles"].values()
    ]
    assert set.union(*role_positions) == set(range(32))
    assert all(
        not role_positions[left].intersection(role_positions[right])
        for left in range(3)
        for right in range(left)
    )
    with pytest.raises(V05BlindError, match="frozen slice"):
        replace(
            projection,
            source_validation={
                **dict(projection.source_validation),
                "anchor-a": projection.source_repeat["anchor-a"],
            },
        )
    with pytest.raises(V05BlindError, match="frozen slice"):
        replace(
            projection,
            source_validation={
                **dict(projection.source_validation),
                "anchor-a": projection.source_validation["anchor-b"],
            },
        )

    config_digest = _digest("runner-config")
    probe_digest = sha256_json(exact_repeat_collector.q0_probe_protocol(sigma=0.35))
    normalization_digest = _digest("source-normalizer")
    panel = fit_p0_panel(
        full_banks,
        manifest,
        config_digest=config_digest,
        probe_protocol_digest=probe_digest,
        normalization_digest=normalization_digest,
        source_parent_asset_sha256=parent_asset_sha256,
        source_parent_membership_digest=parent_membership_digest,
        bandwidth=0.8,
        expected_source_count=2,
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
    assert set(panel.source_role_manifest["roles"]) == {
        "source_train",
        "source_validation",
    }
    assert not any("repeat" in key.lower() for key in panel.source_binding)
    assert (
        panel.source_model_manifest["source_role_manifest"]
        == panel.source_role_manifest
    )
    scorer_surface = {
        "source_binding": panel.source_binding,
        "source_role_manifest": panel.source_role_manifest,
        "source_model_manifest": panel.source_model_manifest,
    }

    def nested_keys(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                yield str(key)
                yield from nested_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from nested_keys(item)

    assert not any("repeat" in key.lower() for key in nested_keys(scorer_surface))
    private_repeat_digests = {
        projection.manifest["roles"]["source_repeat_report"]["membership_digest"],
        projection.manifest["roles"]["source_repeat_report"]["bank_digest"],
        *(
            row["roles"]["source_repeat_report"][field]
            for row in projection.manifest["sources"].values()
            for field in ("membership_digest", "bank_digest")
        ),
    }
    scorer_bytes = canonical_json_bytes(scorer_surface)
    assert all(value.encode() not in scorer_bytes for value in private_repeat_digests)
    with pytest.raises(V05RunnerError, match="source binding fields"):
        replace(
            panel,
            source_binding={
                **dict(panel.source_binding),
                "held_evidence": next(iter(private_repeat_digests)),
            },
        )
    with pytest.raises(V05RunnerError, match="episode counts"):
        replace(
            panel,
            source_binding={
                **dict(panel.source_binding),
                "episode_counts_per_anchor": {
                    "held": next(iter(private_repeat_digests))
                },
            },
        )
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
    weight_shift = 0.5 * empirical_weights[1]
    empirical_weights[:2] += np.asarray([weight_shift, -weight_shift])
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

    tampered_role_manifest = dict(panel.source_role_manifest)
    tampered_sources = dict(tampered_role_manifest["sources"])
    tampered_anchor = dict(tampered_sources["anchor-a"])
    tampered_roles = dict(tampered_anchor["roles"])
    tampered_train = dict(tampered_roles["source_train"])
    tampered_train["bank_digest"] = _digest("tampered-train-bank")
    tampered_roles["source_train"] = tampered_train
    tampered_anchor["roles"] = tampered_roles
    tampered_sources["anchor-a"] = tampered_anchor
    tampered_role_manifest["sources"] = tampered_sources
    with pytest.raises(V05RunnerError, match="manifest digest changed"):
        replace(panel, source_role_manifest=tampered_role_manifest)

    public_parent = tmp_path / "new-opaque-public-root"
    private_binding_root = tmp_path / "private-truth-bindings"
    blinding_nonce = "0123456789abcdef-v05-test"
    authorized_budgets = (1, 2, 4)
    query_views = prepare_blinded_episode_bank(
        episode_bank=projection.source_repeat["anchor-a"],
        parent_source_role_manifest=projection.manifest,
        expected_source_role_manifest_digest=projection.manifest[
            "source_role_manifest_digest"
        ],
        parent_asset_sha256=parent_asset_sha256["anchor-a"],
        parent_membership_digest=parent_membership_digest["anchor-a"],
        probe_protocol_digest=probe_digest,
        public_parent=public_parent,
        private_binding_root=private_binding_root,
        truth_source_anchor_id="anchor-a",
        certificate_manifest=manifest,
        blinding_nonce=blinding_nonce,
        normalization_digest=normalization_digest,
        rff_map=panel.rff.rff_map,
        swe_map=panel.swe.swe_map,
        authorized_budgets=authorized_budgets,
        expected_candidate_count=1,
    )
    assert isinstance(query_views, AuthorizedQueryViews)
    assert query_views.manifest["market_order_digest"] == sha256_json(list(anchors))
    assert query_views.manifest["candidate_mask"] == [True, False]
    assert query_views.authorized_budgets == authorized_budgets
    assert {bank.episode_count for bank in query_views.banks.values()} == {4}
    assert query_views.summary_rows.shape[0] == 4
    assert set(query_views.manifest["canonical_budget_bank_digests"]) == {
        "1",
        "2",
        "4",
    }
    public_bytes = canonical_json_bytes(query_views.manifest)
    for private_value in (
        "anchor-a",
        POLICY_A,
        parent_asset_sha256["anchor-a"],
        parent_membership_digest["anchor-a"],
    ):
        assert private_value.encode() not in public_bytes

    def public_keys(value):
        if isinstance(value, Mapping):
            yield from value
            for item in value.values():
                yield from public_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from public_keys(item)

    forbidden_public_tokens = (
        "anchor",
        "path",
        "policy",
        "factor",
        "return",
        "expected",
        "label",
        "truth",
    )
    assert not any(
        token in key.lower()
        for key in public_keys(query_views.manifest)
        for token in forbidden_public_tokens
    )

    def string_values(value):
        if isinstance(value, Mapping):
            for item in value.values():
                yield from string_values(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from string_values(item)
        elif isinstance(value, str):
            yield value

    public_query_tokens = {
        query_views.manifest["target_membership_digest"],
        query_views.manifest["canonical_query_bank_digest"],
        query_views.manifest["reward_free_bank_sha256"],
        query_views.manifest["manifest_digest"],
        *query_views.manifest["canonical_budget_bank_digests"].values(),
        *(
            value
            for row in query_views.manifest["authorized_views"].values()
            for key, value in row.items()
            if key in {"artifact_sha256", "arrays_digest"}
        ),
    }
    scorer_digest_values = {
        value
        for value in string_values(scorer_surface)
        if len(value) == 64 and value == value.lower()
    }
    assert public_query_tokens.isdisjoint(scorer_digest_values)
    private_binding = read_json(
        private_binding_root / f"{query_views.opaque_query_id}.json"
    )
    assert private_binding["source_anchor_id"] == "anchor-a"
    assert private_binding["opaque_certified_policy_id"] == POLICY_A
    assert private_binding["private_evidence"]["parent_asset_sha256"] == (
        parent_asset_sha256["anchor-a"]
    )
    assert private_binding["private_evidence"]["parent_membership_digest"] == (
        parent_membership_digest["anchor-a"]
    )
    assert private_binding["private_evidence"]["source_repeat_bank_digest"] == (
        projection.source_repeat["anchor-a"].bank_digest
    )

    unsafe_manifest = dict(query_views.manifest)
    unsafe_manifest["source_anchor_id"] = "anchor-a"
    unsafe_manifest["manifest_digest"] = sha256_json(
        {
            key: value
            for key, value in unsafe_manifest.items()
            if key != "manifest_digest"
        }
    )
    with pytest.raises(V05BlindError, match="forbidden"):
        replace(query_views, manifest=unsafe_manifest)

    bad_budget_manifest = dict(query_views.manifest)
    bad_budget_manifest["canonical_budget_bank_digests"] = {
        **dict(query_views.manifest["canonical_budget_bank_digests"]),
        "1": _digest("tampered-budget-bank"),
    }
    bad_budget_manifest["manifest_digest"] = sha256_json(
        {
            key: value
            for key, value in bad_budget_manifest.items()
            if key != "manifest_digest"
        }
    )
    with pytest.raises(V05BlindError, match="budget bank digest changed"):
        replace(query_views, manifest=bad_budget_manifest)

    with pytest.raises(V05BlindError, match="nested frozen subset"):
        prepare_blinded_episode_bank(
            episode_bank=projection.source_repeat["anchor-a"],
            parent_source_role_manifest=projection.manifest,
            expected_source_role_manifest_digest=projection.manifest[
                "source_role_manifest_digest"
            ],
            parent_asset_sha256=parent_asset_sha256["anchor-a"],
            parent_membership_digest=parent_membership_digest["anchor-a"],
            probe_protocol_digest=probe_digest,
            public_parent=tmp_path / "unauthorized-eight-public",
            private_binding_root=tmp_path / "unauthorized-eight-private",
            truth_source_anchor_id="anchor-a",
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
            normalization_digest=normalization_digest,
            rff_map=panel.rff.rff_map,
            swe_map=panel.swe.swe_map,
            authorized_budgets=(1, 2, 4, 8),
            expected_candidate_count=1,
        )
    with pytest.raises(V05BlindError, match="must equal"):
        prepare_blinded_episode_bank(
            episode_bank=projection.source_repeat["anchor-a"],
            parent_source_role_manifest=projection.manifest,
            expected_source_role_manifest_digest=projection.manifest[
                "source_role_manifest_digest"
            ],
            parent_asset_sha256=parent_asset_sha256["anchor-a"],
            parent_membership_digest=parent_membership_digest["anchor-a"],
            probe_protocol_digest=probe_digest,
            public_parent=tmp_path / "underspecified-public",
            private_binding_root=tmp_path / "underspecified-private",
            truth_source_anchor_id="anchor-a",
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
            normalization_digest=normalization_digest,
            rff_map=panel.rff.rff_map,
            swe_map=panel.swe.swe_map,
            authorized_budgets=(1, 2),
            expected_candidate_count=1,
        )
    overlap_root = tmp_path / "overlap-root"
    with pytest.raises(V05BlindError, match="overlap"):
        prepare_blinded_episode_bank(
            episode_bank=projection.source_repeat["anchor-a"],
            parent_source_role_manifest=projection.manifest,
            expected_source_role_manifest_digest=projection.manifest[
                "source_role_manifest_digest"
            ],
            parent_asset_sha256=parent_asset_sha256["anchor-a"],
            parent_membership_digest=parent_membership_digest["anchor-a"],
            probe_protocol_digest=probe_digest,
            public_parent=overlap_root,
            private_binding_root=overlap_root / "private",
            truth_source_anchor_id="anchor-a",
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
            normalization_digest=normalization_digest,
            rff_map=panel.rff.rff_map,
            swe_map=panel.swe.swe_map,
            authorized_budgets=authorized_budgets,
            expected_candidate_count=1,
        )
    with pytest.raises(V05BlindError, match="overwrite"):
        prepare_blinded_episode_bank(
            episode_bank=projection.source_repeat["anchor-a"],
            parent_source_role_manifest=projection.manifest,
            expected_source_role_manifest_digest=projection.manifest[
                "source_role_manifest_digest"
            ],
            parent_asset_sha256=parent_asset_sha256["anchor-a"],
            parent_membership_digest=parent_membership_digest["anchor-a"],
            probe_protocol_digest=probe_digest,
            public_parent=public_parent,
            private_binding_root=private_binding_root,
            truth_source_anchor_id="anchor-a",
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
            normalization_digest=normalization_digest,
            rff_map=panel.rff.rff_map,
            swe_map=panel.swe.swe_map,
            authorized_budgets=authorized_budgets,
            expected_candidate_count=1,
        )

    timings = {}
    predictions, rows = score_query(
        panel,
        query_views,
        expected_task_candidate_count=1,
        timings=timings,
    )
    assert set(timings) == set(P0_METHOD_IDS)
    assert all(np.isfinite(value) and value >= 0.0 for value in timings.values())
    with pytest.raises(V05RunnerError, match="authorized query subset"):
        score_query(
            panel,
            query_views,
            budgets=(1, 2, 4, 8),
            expected_task_candidate_count=1,
        )
    assert "candidate_anchor_ids" not in inspect.signature(score_query).parameters
    assert len(predictions) == len(rows) == 36
    assert all(row["source_binding"] == panel.source_binding for row in rows)
    for budget in authorized_budgets:
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
    assert all(
        prediction.canonical_query_bank_digest
        == query_views.canonical_bank_digest_for_budget(prediction.budget_episodes)
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
        (256, 4_000),
    }

    indexed_rows = {
        (row["method_id"], row["budget_episodes"], row["endpoint"]): row for row in rows
    }
    indexed_predictions = {
        (item.method_id, item.budget_episodes, item.endpoint): item
        for item in predictions
    }
    for method_id in P0_METHOD_IDS:
        for budget in authorized_budgets:
            market = indexed_rows[(method_id, budget, MARKET_30_CERT)]
            masked = indexed_rows[(method_id, budget, TASK_5_CERT)]
            assert market["scores_before_mask"] == masked["scores_before_mask"]
            assert len(market["scores_before_mask"]) == 2
            scores = market["scores_before_mask"]
            anchor_scores = dict(zip(anchors, scores, strict=True))
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

    other_group_manifest = dict(query_views.manifest)
    other_group_manifest["candidate_mask"] = [False, True]
    other_group_manifest["manifest_digest"] = sha256_json(
        {
            key: value
            for key, value in other_group_manifest.items()
            if key != "manifest_digest"
        }
    )
    other_group_views = replace(query_views, manifest=other_group_manifest)
    with pytest.raises(V05RunnerError, match="task/ABI binding"):
        score_query(
            panel,
            other_group_views,
            budgets=(1,),
            expected_task_candidate_count=1,
        )

    payload, ranking_seal = seal_prediction_rows(
        predictions, expected_budgets=authorized_budgets
    )
    assert ranking_seal.verify(payload)
    private_binding_file = private_binding_root / f"{query_views.opaque_query_id}.json"
    loaded_truth = load_private_truth_binding(
        private_binding_file,
        opaque_query_id=query_views.opaque_query_id,
        authorized_query_manifest_digest=query_views.manifest["manifest_digest"],
        prediction_seal=ranking_seal,
        certificate_manifest=manifest,
        blinding_nonce=blinding_nonce,
    )
    assert loaded_truth.source_anchor_id == "anchor-a"
    assert (
        loaded_truth.authorized_query_manifest_digest
        == query_views.manifest["manifest_digest"]
    )
    assert loaded_truth.prediction_seal_digest == ranking_seal.rankings_digest
    incomplete_rows = tuple(
        item for item in predictions if item.method_id == P0_METHOD_IDS[0]
    )
    incomplete_seal = seal_rankings(prediction_payload(incomplete_rows))
    private_reads = []
    real_read_json = blind_query_module.read_json

    def track_private_read(path):
        private_reads.append(Path(path))
        return real_read_json(path)

    class PrivatePathTrap:
        def __fspath__(self):
            raise AssertionError("private path capability was materialized")

    monkeypatch.setattr(blind_query_module, "read_json", track_private_read)
    with pytest.raises(V05BlindError, match="complete P0 prediction coverage"):
        load_private_truth_binding(
            PrivatePathTrap(),
            opaque_query_id=query_views.opaque_query_id,
            authorized_query_manifest_digest=query_views.manifest["manifest_digest"],
            prediction_seal=incomplete_seal,
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
        )
    one_budget_seal = seal_rankings(
        prediction_payload(
            tuple(item for item in predictions if item.budget_episodes == 1)
        )
    )
    with pytest.raises(V05BlindError, match="complete P0 prediction coverage"):
        load_private_truth_binding(
            PrivatePathTrap(),
            opaque_query_id=query_views.opaque_query_id,
            authorized_query_manifest_digest=query_views.manifest["manifest_digest"],
            prediction_seal=one_budget_seal,
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
        )
    with pytest.raises(V05BlindError, match="authorized public manifest"):
        load_private_truth_binding(
            PrivatePathTrap(),
            opaque_query_id=query_views.opaque_query_id,
            authorized_query_manifest_digest=_digest("wrong-public-manifest"),
            prediction_seal=ranking_seal,
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
        )
    with pytest.raises(TypeError, match="expected_budgets"):
        load_private_truth_binding(
            PrivatePathTrap(),
            opaque_query_id=query_views.opaque_query_id,
            authorized_query_manifest_digest=query_views.manifest["manifest_digest"],
            prediction_seal=one_budget_seal,
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
            expected_budgets=(1,),
        )
    assert private_reads == []
    monkeypatch.setattr(blind_query_module, "read_json", real_read_json)
    with pytest.raises(V05BlindError, match="HMAC"):
        load_private_truth_binding(
            private_binding_file,
            opaque_query_id=query_views.opaque_query_id,
            authorized_query_manifest_digest=query_views.manifest["manifest_digest"],
            prediction_seal=ranking_seal,
            certificate_manifest=manifest,
            blinding_nonce="fedcba9876543210-wrong",
        )
    tampered_binding = dict(private_binding)
    tampered_binding["public_manifest_digest"] = _digest("tampered-public-manifest")
    tampered_binding["binding_digest"] = sha256_json(
        {
            key: value
            for key, value in tampered_binding.items()
            if key != "binding_digest"
        }
    )
    tampered_binding_file = (
        tmp_path / "tampered-private" / f"{query_views.opaque_query_id}.json"
    )
    atomic_write_json(tampered_binding_file, tampered_binding)
    with pytest.raises(V05BlindError, match="sealed query rows"):
        load_private_truth_binding(
            tampered_binding_file,
            opaque_query_id=query_views.opaque_query_id,
            authorized_query_manifest_digest=query_views.manifest["manifest_digest"],
            prediction_seal=ranking_seal,
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
        )


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


def test_canonical_source_stage_persists_fit_roles_only_then_blinds_repeat(
    tmp_path, monkeypatch
) -> None:
    anchors = tuple(f"source-{index:02d}" for index in range(30))
    task_by_anchor = {
        anchor: f"task-{index // 5}" for index, anchor in enumerate(anchors)
    }
    policy_by_anchor = {
        anchor: "lw-" + _digest(f"policy-{anchor}")[:20] for anchor in anchors
    }
    certificate_manifest = project_certificate_manifest(
        policy_by_anchor,
        task_by_anchor=task_by_anchor,
        policy_bundle_digest_by_policy={
            policy: _digest(f"bundle-{policy}") for policy in policy_by_anchor.values()
        },
        championization_admission_digest_by_anchor={
            anchor: _digest(f"champion-{anchor}") for anchor in anchors
        },
        execution_abi_digest_by_policy={
            policy_by_anchor[anchor]: _digest(f"abi-{task_by_anchor[anchor]}")
            for anchor in anchors
        },
        expected_anchor_ids=anchors,
    )

    row = np.arange(32 * 64, dtype=np.float64)[:, None]
    coordinate = np.arange(29, dtype=np.float64)[None, :]
    base_observation = row / 2_048.0 + coordinate / 29.0
    base_action = np.sin(row / 31.0)
    arrays_by_anchor = {}
    memberships = {}
    contexts = {}
    for index, anchor in enumerate(anchors):
        context = f"context-{index:02d}"
        observation = base_observation + index / 100.0
        arrays_by_anchor[anchor] = {
            "observation": observation,
            "action": base_action + index / 1_000.0,
            "next_observation": observation + 0.05 + coordinate / 10_000.0,
            "episode_offsets": np.arange(33, dtype=np.int64) * 64,
        }
        memberships[anchor] = derive_probe_membership(context, 50_500)
        contexts[anchor] = context

    assets = runner_module.FrozenR4Assets(
        config={"measurement": {"normalizer_std_floor": 1.0e-6}},
        config_digest=_digest("canonical-source-config"),
        r4_root=tmp_path / "synthetic-r4",
        v03_root=tmp_path / "synthetic-v03",
        arrays_by_anchor=arrays_by_anchor,
        membership_by_anchor=memberships,
        context_by_anchor=contexts,
        task_by_anchor=task_by_anchor,
        parent_asset_sha256={
            anchor: _digest(f"parent-asset-{anchor}") for anchor in anchors
        },
        parent_membership_digest={
            anchor: memberships[anchor].membership_digest for anchor in anchors
        },
        native_schema_by_task={
            task: _digest(f"native-schema-{task}")
            for task in set(task_by_anchor.values())
        },
        certificate_manifest=certificate_manifest,
        probe_protocol_digest=_digest("canonical-source-probe"),
        provenance={"fixture": "synthetic-frozen-r4"},
    )
    run_dir = tmp_path / "run"
    full_banks, complete = runner_module._canonical_source_stage(
        assets, run_dir, resume=False
    )

    assert set(full_banks) == set(anchors)
    assert all(
        bank.points.shape == (32 * 64, 30)
        and bank.episode_offsets.shape == (33,)
        and np.array_equal(np.diff(bank.episode_offsets), np.full(32, 64))
        for bank in full_banks.values()
    )
    canonical_root = run_dir / "source_fit" / "canonical"
    npz_paths = {
        path.relative_to(canonical_root).as_posix()
        for path in canonical_root.rglob("*.npz")
    }
    assert npz_paths == {
        "normalizer_state.npz",
        *(f"fit_banks/{anchor}.npz" for anchor in anchors),
    }
    for anchor in anchors:
        with np.load(canonical_root / "fit_banks" / f"{anchor}.npz") as archive:
            assert set(archive.files) == {"points", "episode_offsets"}
            assert archive["points"].shape == (25 * 64, 30)
            assert archive["episode_offsets"].shape == (26,)
            np.testing.assert_array_equal(
                np.diff(archive["episode_offsets"]), np.full(25, 64)
            )

    assert complete["persisted_roles"] == ["source_train", "source_validation"]
    assert complete["persisted_episode_count_per_anchor"] == 25
    assert complete["held_repeat_persisted"] is False

    def nested_keys(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                yield str(key)
                yield from nested_keys(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from nested_keys(item)

    complete_keys = tuple(nested_keys(complete))
    assert not any("full" in key.lower() for key in complete_keys)
    assert [key for key in complete_keys if "repeat" in key.lower()] == [
        "held_repeat_persisted"
    ]

    projection = project_verified_source_banks(
        full_banks,
        parent_asset_sha256=assets.parent_asset_sha256,
        parent_membership_digest=assets.parent_membership_digest,
        expected_source_count=30,
    )
    complete_values = set()

    def collect_strings(value):
        if isinstance(value, Mapping):
            for item in value.values():
                collect_strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_strings(item)
        elif isinstance(value, str):
            complete_values.add(value)

    collect_strings(complete)
    assert {
        *(bank.bank_digest for bank in full_banks.values()),
        *(bank.bank_digest for bank in projection.source_repeat.values()),
    }.isdisjoint(complete_values)

    rff_map = RFFMap(
        input_dim=30,
        bandwidth=1.0,
        normalization_digest=complete["normalizer_digest"],
        frequency_count=4,
        public_seed=50_501,
    )
    swe_map = SWEMap(
        input_dim=30,
        normalization_digest=complete["normalizer_digest"],
        direction_count=2,
        quantile_count=4,
        public_seed=50_502,
    )
    truth_anchor = anchors[0]
    query = prepare_blinded_episode_bank(
        episode_bank=projection.source_repeat[truth_anchor],
        parent_source_role_manifest=projection.manifest,
        expected_source_role_manifest_digest=projection.manifest[
            "source_role_manifest_digest"
        ],
        parent_asset_sha256=assets.parent_asset_sha256[truth_anchor],
        parent_membership_digest=assets.parent_membership_digest[truth_anchor],
        probe_protocol_digest=assets.probe_protocol_digest,
        public_parent=tmp_path / "opaque-public",
        private_binding_root=tmp_path / "private-bindings",
        truth_source_anchor_id=truth_anchor,
        certificate_manifest=assets.certificate_manifest,
        blinding_nonce="0123456789abcdef-source-stage-test",
        normalization_digest=complete["normalizer_digest"],
        rff_map=rff_map,
        swe_map=swe_map,
        authorized_budgets=(1, 2, 4),
    )
    assert query.opaque_query_id.startswith("q-")
    assert query.authorized_budgets == (1, 2, 4)
    assert truth_anchor.encode() not in canonical_json_bytes(query.manifest)

    missing_fit_bank = canonical_root / "fit_banks" / f"{anchors[0]}.npz"
    missing_fit_bank.unlink()
    write_attempts = []

    def forbid_resume_write(path, *unused_args, **unused_kwargs):
        write_attempts.append(Path(path))
        raise AssertionError("COMPLETE canonical resume attempted a write")

    monkeypatch.setattr(runner_module, "atomic_write_npz", forbid_resume_write)
    monkeypatch.setattr(runner_module, "atomic_write_json", forbid_resume_write)
    with pytest.raises(V05RunnerError):
        runner_module._canonical_source_stage(assets, run_dir, resume=True)
    assert write_attempts == []
    assert not missing_fit_bank.exists()


def test_frozen_root_resolver_uses_canonical_layout_from_any_location(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "arbitrary-workspace" / "external-assets"
    r4 = root / "v04a" / "runs" / "v04a-primary-dev-20260828-r4"
    v03 = root / "v03" / "runs" / "v03-main-20260827-r0"
    r4.mkdir(parents=True)
    v03.mkdir(parents=True)
    monkeypatch.setenv("RL_LEARNWARE_ARTIFACTS_ROOT", str(root))

    resolved_root, resolved_r4, resolved_v03 = runner_module._resolve_v05_frozen_roots()
    assert (resolved_root, resolved_r4, resolved_v03) == (
        root.resolve(),
        r4.resolve(),
        v03.resolve(),
    )

    other = tmp_path / "explicit-assets"
    other_r4 = other / "v04a" / "runs" / "v04a-primary-dev-20260828-r4"
    other_v03 = other / "v03" / "runs" / "v03-main-20260827-r0"
    other_r4.mkdir(parents=True)
    other_v03.mkdir(parents=True)
    assert runner_module._resolve_v05_frozen_roots(other) == (
        other.resolve(),
        other_r4.resolve(),
        other_v03.resolve(),
    )
    monkeypatch.delenv("RL_LEARNWARE_ARTIFACTS_ROOT")
    assert (
        runner_module.resolve_artifacts_root(repository_root=tmp_path / "repository")
        == (tmp_path / "artifacts").resolve()
    )


def test_frozen_root_resolver_fails_before_frozen_input_io(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "artifacts"
    r4 = root / "v04a" / "runs" / "v04a-primary-dev-20260828-r4"
    external_v03 = tmp_path / "external-v03"
    r4.mkdir(parents=True)
    external_v03.mkdir()
    (root / "v03").symlink_to(external_v03, target_is_directory=True)
    monkeypatch.setenv("RL_LEARNWARE_ARTIFACTS_ROOT", str(root))
    frozen_reads = []
    monkeypatch.setattr(
        runner_module,
        "_frozen_bytes",
        lambda *args, **kwargs: frozen_reads.append((args, kwargs)),
    )

    with pytest.raises(V05RunnerError, match="symlink"):
        runner_module._load_frozen_r4_assets({}, _digest("config"))
    assert frozen_reads == []


def test_production_entry_and_invalid_public_seal_cannot_reach_private_capability(
    tmp_path, monkeypatch
) -> None:
    assert tuple(inspect.signature(runner_module.run_development).parameters) == (
        "config_path",
        "new_run_dir",
        "artifacts_root",
        "resume",
    )
    score_signature = inspect.signature(runner_module.score_precommitted_queries)
    assert tuple(score_signature.parameters) == (
        "panel",
        "query_index_path",
        "scoring_root",
        "resume",
    )
    assert "AuthorizedQueryViews" not in str(score_signature)

    query_ids = tuple(f"q-{index:064x}" for index in range(30))
    index = {
        "queries": {
            query_id: _digest(f"manifest-{query_id}") for query_id in query_ids
        },
        "index_digest": _digest("public-query-index"),
    }
    monkeypatch.setattr(runner_module, "_read_query_index", lambda *unused: index)

    class RunRootCapabilityTrap:
        private_touched = False

        def __truediv__(self, child):
            if child == "private":
                self.private_touched = True
                raise AssertionError("private path capability was materialized")
            return tmp_path / str(child)

    incomplete_receipt = {
        "schema": "policy-learnware.v05-global-development-seal.v1",
        "status": "SEALED",
        "query_index_digest": index["index_digest"],
        "query_count": 29,
        "prediction_cell_count": 1080,
        "score_rows_digest": _digest("score-rows"),
        "budget_ledger_digest": _digest("budget-ledger"),
        "cell_files": {},
        "prediction_seal": {},
        "global_seal_digest": _digest("global-seal"),
    }
    for receipt in ({}, incomplete_receipt):
        loaded_paths = []
        seal_path = tmp_path / "scoring" / "global_seal.json"
        if receipt:
            atomic_write_json(seal_path, {"placeholder": True})

        def public_json_only(path):
            materialized = Path(path)
            loaded_paths.append(materialized)
            if (
                materialized.name == "blinding_nonce.json"
                or "query_bindings" in materialized.parts
            ):
                raise AssertionError(
                    "private JSON was read before public seal validation"
                )
            return receipt

        monkeypatch.setattr(runner_module, "load_strict_json", public_json_only)
        run_root = RunRootCapabilityTrap()
        with pytest.raises(V05RunnerError):
            runner_module._evaluate_after_persisted_seal(
                object(),
                {},
                {},
                run_root,
                tmp_path / "scoring",
                source_input_bytes=0,
                resume=False,
            )
        assert run_root.private_touched is False
        assert loaded_paths == ([] if not receipt else [seal_path])


def test_query_prepare_rejects_public_or_private_ancestor_symlink_before_writes(
    tmp_path, monkeypatch
) -> None:
    cases = []
    for linked_name in ("public", "private"):
        outside = tmp_path / f"outside-{linked_name}"
        outside.mkdir()
        (outside / "sentinel.txt").write_text("unchanged", encoding="utf-8")
        run_dir = tmp_path / f"run-with-{linked_name}-link"
        run_dir.mkdir()
        (run_dir / linked_name).symlink_to(outside, target_is_directory=True)
        cases.append((run_dir, outside))

    def outside_state(root):
        return {
            path.relative_to(root).as_posix(): (
                sha256_file(path),
                path.stat().st_mtime_ns,
            )
            for path in root.rglob("*")
            if path.is_file()
        }

    before = {outside: outside_state(outside) for _, outside in cases}
    write_attempts = []

    def forbid_write(path, *unused_args, **unused_kwargs):
        write_attempts.append(Path(path))
        raise AssertionError("query prepare attempted a write through a symlink")

    monkeypatch.setattr(runner_module, "atomic_write_json", forbid_write)
    monkeypatch.setattr(runner_module, "prepare_blinded_episode_bank", forbid_write)
    monkeypatch.setattr(runner_module.Path, "mkdir", forbid_write)
    monkeypatch.setattr(runner_module.os, "chmod", forbid_write)
    inaccessible = SimpleNamespace(
        __getattribute__=lambda *unused: (_ for _ in ()).throw(
            AssertionError("query data was accessed before symlink rejection")
        )
    )
    for run_dir, outside in cases:
        with pytest.raises(V05RunnerError):
            runner_module._prepare_query_index(
                inaccessible,
                inaccessible,
                inaccessible,
                run_dir,
                resume=True,
            )
        assert write_attempts == []
        assert outside_state(outside) == before[outside]


def test_frozen_json_parses_the_same_verified_bytes_when_path_swaps(
    tmp_path, monkeypatch
) -> None:
    frozen_path = tmp_path / "frozen.json"
    value_a = {"identity": "A", "rows": [1, 2, 3]}
    value_b = {"identity": "B", "rows": [4, 5, 6]}
    bytes_a = canonical_json_bytes(value_a) + b"\n"
    bytes_b = canonical_json_bytes(value_b) + b"\n"
    frozen_path.write_bytes(bytes_a)
    expected_sha256 = hashlib.sha256(bytes_a).hexdigest()
    real_read_bytes = Path.read_bytes
    swapped = False

    def swap_after_read(path):
        nonlocal swapped
        payload = real_read_bytes(path)
        if path == frozen_path and not swapped:
            assert path.is_relative_to(tmp_path)
            replacement = tmp_path / "replacement.json"
            replacement.write_bytes(bytes_b)
            replacement.replace(path)
            swapped = True
        return payload

    monkeypatch.setattr(Path, "read_bytes", swap_after_read)
    observed_path, observed = runner_module._frozen_json(
        tmp_path,
        {"relative_path": frozen_path.name},
        expected_sha256=expected_sha256,
        where="swap-hook fixture",
    )
    assert observed_path == frozen_path
    assert observed == value_a
    assert swapped is True
    assert real_read_bytes(frozen_path) == bytes_b


def test_completed_source_fit_checkpoint_closure_is_immutable_on_resume(
    tmp_path, monkeypatch
) -> None:
    anchors = tuple(f"checkpoint-source-{index:02d}" for index in range(30))
    task_by_anchor = {
        anchor: f"checkpoint-task-{index // 5}" for index, anchor in enumerate(anchors)
    }
    policy_by_anchor = {
        anchor: "lw-" + _digest(f"checkpoint-policy-{anchor}")[:20]
        for anchor in anchors
    }
    certificate_manifest = project_certificate_manifest(
        policy_by_anchor,
        task_by_anchor=task_by_anchor,
        policy_bundle_digest_by_policy={
            policy: _digest(f"checkpoint-bundle-{policy}")
            for policy in policy_by_anchor.values()
        },
        championization_admission_digest_by_anchor={
            anchor: _digest(f"checkpoint-champion-{anchor}") for anchor in anchors
        },
        execution_abi_digest_by_policy={
            policy_by_anchor[anchor]: _digest(
                f"checkpoint-abi-{task_by_anchor[anchor]}"
            )
            for anchor in anchors
        },
        expected_anchor_ids=anchors,
    )
    reducer = ReducerConfig(support_budget=1, support_steps=0, kmeans_steps=0)
    reducer_config = {
        name: getattr(reducer, name) for name in ReducerConfig.__dataclass_fields__
    }
    config = {
        "measurement": {
            "gaussian_bandwidth": {
                "rule": "source_balanced_median_pair_distance",
                "calibration_pairs": 10,
                "public_seed": 50_500,
            }
        },
        "raw_delta_rkme": reducer_config,
        "summary_logreg": {
            "l2_grid": [0.1],
            "max_iter": 1,
            "gradient_tolerance": 1.0e-9,
        },
        "kme_krr": {"ridge_grid": [0.1]},
        "rff_kme_nn": {
            "frequency_count": 1,
            "output_dimension": 2,
            "public_seed": 50_501,
        },
        "swe_nn": {
            "direction_count": 1,
            "quantile_count": 2,
            "output_dimension": 2,
            "public_seed": 50_502,
        },
    }
    assets = runner_module.FrozenR4Assets(
        config=config,
        config_digest=_digest("checkpoint-config"),
        r4_root=tmp_path / "unused-r4",
        v03_root=tmp_path / "unused-v03",
        arrays_by_anchor={},
        membership_by_anchor={},
        context_by_anchor={},
        task_by_anchor=task_by_anchor,
        parent_asset_sha256={
            anchor: _digest(f"checkpoint-parent-{anchor}") for anchor in anchors
        },
        parent_membership_digest={
            anchor: _digest(f"checkpoint-membership-{anchor}") for anchor in anchors
        },
        native_schema_by_task={},
        certificate_manifest=certificate_manifest,
        probe_protocol_digest=_digest("checkpoint-probe"),
        provenance={},
    )
    full_bank = EpisodeBank(
        np.zeros((32 * 64, 30), dtype=np.float64),
        np.arange(33, dtype=np.int64) * 64,
    )
    full_banks = {anchor: full_bank for anchor in anchors}
    canonical_receipt = {
        "complete_digest": _digest("checkpoint-canonical-complete"),
        "normalizer_digest": _digest("checkpoint-normalizer"),
    }
    classes = tuple(sorted(policy_by_anchor.values()))
    summary_model = runner_module.SummaryLogReg(
        class_ids=classes,
        feature_mean=np.zeros(60),
        feature_scale=np.ones(60),
        weights=np.zeros((60, 30)),
        intercept=np.zeros(30),
        selected_l2=0.1,
        training_iterations=1,
    )
    stacked_train = EpisodeBank(
        np.zeros((30 * 19 * 64, 30), dtype=np.float64),
        np.arange(30 * 19 + 1, dtype=np.int64) * 64,
    )
    krr_model = runner_module.KMEKRR(
        class_ids=classes,
        training_bank=stacked_train,
        alpha=np.zeros((30 * 19, 30)),
        bandwidth=1.0,
        selected_ridge=0.1,
    )

    def cheap_empirical(
        points,
        kernel,
        *,
        episode_offsets,
        protocol_id,
        dataset_digest,
        source_task="",
        **unused,
    ):
        point_array = np.asarray(points, dtype=np.float64)
        return runner_module.EmpiricalKME(
            points=point_array,
            weights=np.full(point_array.shape[0], 1.0 / point_array.shape[0]),
            episode_offsets=episode_offsets,
            bandwidth=kernel.bandwidth,
            norm2=1.0,
            protocol_id=protocol_id,
            dataset_digest=dataset_digest,
            source_task=source_task,
        )

    def cheap_reduced(empirical, reduction_config):
        return runner_module.ReducedRKME(
            supports=np.zeros((1, 30)),
            beta=np.ones(1),
            bandwidth=empirical.bandwidth,
            rkme_norm2=1.0,
            empirical_norm2=empirical.norm2,
            reduction_error=0.0,
            protocol_id=empirical.protocol_id,
            source_dataset_digest=empirical.dataset_digest,
            ridge=reduction_config.ridge,
            condition_number=1.0,
            source_task=empirical.source_task,
        )

    monkeypatch.setattr(runner_module, "calibrate_bandwidth", lambda *a, **k: 1.0)
    monkeypatch.setattr(runner_module, "build_empirical_kme", cheap_empirical)
    monkeypatch.setattr(runner_module, "reduce_kme", cheap_reduced)
    monkeypatch.setattr(
        runner_module.SummaryLogReg,
        "fit",
        classmethod(lambda cls, *args, **kwargs: summary_model),
    )
    monkeypatch.setattr(
        runner_module.KMEKRR,
        "fit",
        classmethod(lambda cls, *args, **kwargs: krr_model),
    )

    baseline_run = tmp_path / "source-fit-baseline"
    panel, projection = runner_module._source_fit_stage(
        assets,
        full_banks,
        canonical_receipt,
        baseline_run,
        resume=False,
    )
    assert isinstance(panel, runner_module.P0Panel)
    model_root = baseline_run / "source_fit" / "models"
    assert len(tuple(model_root.rglob("*.npz"))) == 65
    assert len(read_json(model_root / "progress.json")["models"]) == 65
    monkeypatch.setattr(
        runner_module,
        "project_verified_source_banks",
        lambda *args, **kwargs: projection,
    )

    def tree_state(root):
        return {
            path.relative_to(root).as_posix(): (
                sha256_file(path),
                path.stat().st_mtime_ns,
            )
            for path in root.rglob("*")
            if path.is_file()
        }

    before = tree_state(model_root)
    write_attempts = []

    def forbid_write(path, *unused_args, **unused_kwargs):
        write_attempts.append(Path(path))
        raise AssertionError("completed source fit attempted a write")

    monkeypatch.setattr(runner_module, "atomic_write_json", forbid_write)
    monkeypatch.setattr(runner_module, "atomic_write_npz", forbid_write)
    for model_type in (
        runner_module.EmpiricalKME,
        runner_module.ReducedRKME,
        runner_module.SummaryLogReg,
        runner_module.KMEKRR,
        runner_module.RFFMap,
        runner_module.SWEMap,
    ):
        monkeypatch.setattr(model_type, "save_npz", forbid_write)

    resumed_panel, _ = runner_module._source_fit_stage(
        assets,
        full_banks,
        canonical_receipt,
        baseline_run,
        resume=True,
    )
    assert isinstance(resumed_panel, runner_module.P0Panel)
    assert write_attempts == []
    assert tree_state(model_root) == before

    corrupt_runs = {}
    for name in (
        "orphan_path_without_progress",
        "missing_npz_with_progress",
        "extra_npz",
        "tampered_npz",
    ):
        destination = tmp_path / name
        shutil.copytree(baseline_run, destination)
        corrupt_runs[name] = destination

    orphan_root = corrupt_runs["orphan_path_without_progress"] / "source_fit" / "models"
    orphan_progress = read_json(orphan_root / "progress.json")
    orphan_progress["models"].pop(f"empirical/{anchors[0]}")
    atomic_write_json(orphan_root / "progress.json", orphan_progress, overwrite=True)
    missing_root = corrupt_runs["missing_npz_with_progress"] / "source_fit" / "models"
    (missing_root / "raw" / f"{anchors[0]}.npz").unlink()
    extra_root = corrupt_runs["extra_npz"] / "source_fit" / "models"
    shutil.copy2(extra_root / "rff_map.npz", extra_root / "extra.npz")
    tampered_root = corrupt_runs["tampered_npz"] / "source_fit" / "models"
    tampered_path = tampered_root / "summary_logreg.npz"
    tampered_path.write_bytes(tampered_path.read_bytes() + b"tamper")

    for name, corrupt_run in corrupt_runs.items():
        write_attempts.clear()
        with pytest.raises(V05RunnerError):
            runner_module._source_fit_stage(
                assets,
                full_banks,
                canonical_receipt,
                corrupt_run,
                resume=True,
            )
        assert write_attempts == [], name

    source_order = tuple(panel.resolver.anchor_ids)
    policy_order = tuple(
        panel.resolver.record_for_anchor(anchor).opaque_certified_policy_id
        for anchor in source_order
    )
    query_ids = tuple(f"q-{index:064x}" for index in range(30))
    query_manifests = {
        query_id: _digest(f"source-binding-manifest-{query_id}")
        for query_id in query_ids
    }
    index = {"queries": query_manifests}
    predictions = []
    score_rows = []
    source_model_digest = panel.source_model_manifest["source_model_manifest_digest"]
    for query_id in query_ids:
        reward_free_digest = _digest(f"source-binding-reward-free-{query_id}")
        target_membership_digest = _digest(
            f"source-binding-target-membership-{query_id}"
        )
        for method_id in P0_METHOD_IDS:
            for budget in (1, 2, 4):
                scores = [float(index) for index in range(30)]
                score_digest = sha256_json(
                    {
                        "method_id": method_id,
                        "budget_episodes": budget,
                        "opaque_query_id": query_id,
                        "source_order": list(source_order),
                        "scores_before_mask": scores,
                    }
                )
                ledger = BudgetLedger.for_budget(budget).to_dict()
                ledger_digest = sha256_json(ledger)
                canonical_bank_digest = _digest(
                    f"source-binding-canonical-{query_id}-{budget}"
                )
                target_binding = {
                    "probe_protocol_digest": panel.source_binding[
                        "probe_protocol_digest"
                    ],
                    "reward_free_bank_sha256": reward_free_digest,
                    "target_membership_digest": target_membership_digest,
                    "normalization_digest": panel.source_binding[
                        "normalization_digest"
                    ],
                    "authorized_query_manifest_digest": query_manifests[query_id],
                    "canonical_query_bank_digest": canonical_bank_digest,
                }
                for endpoint, ranked_anchors, ranked_policies in (
                    (MARKET_30_CERT, source_order, policy_order),
                    (TASK_5_CERT, source_order[:5], policy_order[:5]),
                ):
                    predictions.append(
                        PredictionRanking(
                            method_id=method_id,
                            endpoint=endpoint,
                            budget_episodes=budget,
                            opaque_query_id=query_id,
                            ranked_anchor_ids=ranked_anchors,
                            ranked_policy_ids=ranked_policies,
                            probe_protocol_digest=panel.source_binding[
                                "probe_protocol_digest"
                            ],
                            reward_free_bank_sha256=reward_free_digest,
                            canonical_query_bank_digest=canonical_bank_digest,
                            source_train_membership_digest=panel.source_binding[
                                "source_train_membership_digest"
                            ],
                            source_validation_membership_digest=panel.source_binding[
                                "source_validation_membership_digest"
                            ],
                            target_membership_digest=target_membership_digest,
                            normalization_digest=panel.source_binding[
                                "normalization_digest"
                            ],
                            config_digest=panel.config_digest,
                            source_model_manifest_digest=source_model_digest,
                            authorized_query_manifest_digest=query_manifests[query_id],
                            score_vector_digest=score_digest,
                            budget_ledger_digest=ledger_digest,
                        )
                    )
                    score_rows.append(
                        {
                            "method_id": method_id,
                            "endpoint": endpoint,
                            "budget_episodes": budget,
                            "opaque_query_id": query_id,
                            "scores_before_mask": scores,
                            "source_binding": dict(panel.source_binding),
                            "target_binding": target_binding,
                            "ledger": ledger,
                            "score_vector_digest": score_digest,
                            "budget_ledger_digest": ledger_digest,
                        }
                    )

    assert panel.source_binding["episode_counts_per_anchor"] == (19, 6)
    serialized_rows = json.loads(canonical_json_bytes(score_rows))
    assert serialized_rows[0]["source_binding"]["episode_counts_per_anchor"] == [
        19,
        6,
    ]
    validated = runner_module._validate_development_batch(
        panel, index, predictions, serialized_rows
    )
    assert len(validated[0]) == len(validated[1]) == 1080

    bad_counts = json.loads(canonical_json_bytes(serialized_rows))
    bad_counts[0]["source_binding"]["episode_counts_per_anchor"] = [19, 7]
    with pytest.raises(V05RunnerError):
        runner_module._validate_development_batch(panel, index, predictions, bad_counts)
    bad_digest = json.loads(canonical_json_bytes(serialized_rows))
    bad_digest[0]["source_binding"]["source_train_bank_digest"] = _digest(
        "tampered-source-train-bank"
    )
    with pytest.raises(V05RunnerError):
        runner_module._validate_development_batch(panel, index, predictions, bad_digest)
