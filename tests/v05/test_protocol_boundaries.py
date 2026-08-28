from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import inspect
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from policy_learnware_v0.hashing import canonical_json_bytes, sha256_json
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
from server.repro_fpo_ppo_v05 import exact_repeat_collector
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
    "source_repeat_membership_digest": _digest("source-repeat-membership"),
    "target_membership_digest": _digest("target-membership"),
    "normalization_digest": _digest("normalization"),
    "config_digest": _digest("config"),
    "source_model_manifest_digest": _digest("source-model-manifest"),
    "authorized_query_manifest_digest": _digest("authorized-query-manifest"),
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
    return (
        TruthBinding(
            "query-a1",
            "anchor-a1",
            "task-a",
            POLICY_A,
            _query_binding("query-a1")["authorized_query_manifest_digest"],
        ),
        TruthBinding(
            "query-a2",
            "anchor-a2",
            "task-a",
            POLICY_A,
            _query_binding("query-a2")["authorized_query_manifest_digest"],
        ),
        TruthBinding(
            "query-b1",
            "anchor-b1",
            "task-b",
            POLICY_B,
            _query_binding("query-b1")["authorized_query_manifest_digest"],
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
        **_query_binding(original.opaque_query_id),
    )
    with pytest.raises(V05MetricError, match="candidate set"):
        evaluate_sealed_predictions(
            seal_rankings(prediction_payload(bad_candidate)),
            truth_join,
            many_to_one_manifest,
            expected_method_ids=("METHOD",),
            expected_budgets=(1,),
        )


def test_runner_binds_all_p0_to_one_panel_then_masks_ranks_and_seals(
    tmp_path,
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
    assert (
        panel.source_role_manifest["sources"]["anchor-a"]["parent_asset_sha256"]
        == parent_asset_sha256["anchor-a"]
    )
    assert (
        panel.source_role_manifest["sources"]["anchor-a"]["parent_membership_digest"]
        == parent_membership_digest["anchor-a"]
    )
    assert (
        panel.source_model_manifest["source_role_manifest"]
        == panel.source_role_manifest
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
    tampered_anchor["parent_asset_sha256"] = _digest("tampered-parent-file")
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
        expected_source_role_manifest_digest=panel.source_binding[
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
            expected_source_role_manifest_digest=panel.source_binding[
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
            expected_source_role_manifest_digest=panel.source_binding[
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
            expected_source_role_manifest_digest=panel.source_binding[
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
            expected_source_role_manifest_digest=panel.source_binding[
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

    predictions, rows = score_query(
        panel,
        query_views,
        expected_task_candidate_count=1,
    )
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
        prediction_seal=ranking_seal,
        certificate_manifest=manifest,
        blinding_nonce=blinding_nonce,
    )
    assert loaded_truth.source_anchor_id == "anchor-a"
    assert (
        loaded_truth.authorized_query_manifest_digest
        == query_views.manifest["manifest_digest"]
    )
    incomplete_rows = tuple(
        item for item in predictions if item.method_id == P0_METHOD_IDS[0]
    )
    incomplete_seal = seal_rankings(prediction_payload(incomplete_rows))
    with pytest.raises(V05BlindError, match="complete P0 prediction coverage"):
        load_private_truth_binding(
            private_binding_file,
            prediction_seal=incomplete_seal,
            certificate_manifest=manifest,
            blinding_nonce=blinding_nonce,
        )
    with pytest.raises(V05BlindError, match="HMAC"):
        load_private_truth_binding(
            private_binding_file,
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
