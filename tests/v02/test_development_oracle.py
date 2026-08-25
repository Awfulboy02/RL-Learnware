from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v02.config import V02ExperimentConfig
from policy_learnware_v0.v02.development_oracle import (
    DevelopmentOracleAdmissionError,
    DevelopmentTargetEvaluationProtocol,
    FrozenDevelopmentOracleProtocol,
    PublishedSelectionRanking,
    recompute_development_oracle,
)
from policy_learnware_v0.v02.market import (
    DeploymentPrivateEntry,
    V02PolicyMarket,
)
from policy_learnware_v0.v02.oracle import OracleEpisodeRow, PublishedSelection
from policy_learnware_v0.v02.schemas import (
    ExecutionABIRecord,
    PublicMarketEntry,
)
from policy_learnware_v0.v02.selectors import (
    L_MIN_EVIDENCE,
    RankingRow,
    SelectionRecord,
)


QUERY_A = "v02q-00000000000000000000000000000001"
QUERY_B = "v02q-00000000000000000000000000000002"
METHODS = ("M1", "M2")
MARKET_IDS = ("lw-a", "lw-b", "lw-incompatible")


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _abi(*, observation: str = "observation-v1") -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=_d(observation),
        action_tensor_abi_digest=_d("action-v1"),
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v02",
        state_abi_id="stateless",
    )


def _config() -> V02ExperimentConfig:
    return V02ExperimentConfig.from_dict(
        {
            "schema": "policy-learnware.v02-experiment-config.v0",
            "experiment_id": "v02-development-oracle-test",
            "stage": "development_discovery",
            "protocol_family_id": "continuous-vector-mdp-v02",
            "tasks": ["WalkerWalk"],
            "dynamics_axes": {
                "WalkerWalk": [
                    {
                        "axis_id": "joint_damping",
                        "operator_id": "joint_damping_scale",
                        "operator_digest": _d("operator"),
                        "leaf_allowlist": ["_mjx_model.dof_damping"],
                        "static_within_episode": True,
                    }
                ]
            },
            "source_factors": {
                "WalkerWalk": {
                    "joint_damping": [
                        {
                            "factor_id": "low",
                            "value": 0.5,
                            "roles": ["source"],
                            "source_anchor_id": _d("anchor-low"),
                            "axis_binding_digest": _d("binding-low"),
                        },
                        {
                            "factor_id": "nominal",
                            "value": 1.0,
                            "roles": ["source"],
                            "source_anchor_id": _d("anchor-nominal"),
                            "axis_binding_digest": None,
                        },
                    ]
                }
            },
            "development_targets": [
                {
                    "target_id": QUERY_A,
                    "task_id": "WalkerWalk",
                    "axis_id": "joint_damping",
                    "factor_id": "development_mid_a",
                    "factor_value": 0.75,
                    "roles": ["development"],
                    "regime": "heldout_interpolation",
                    "source_anchor_ref": None,
                },
                {
                    "target_id": QUERY_B,
                    "task_id": "WalkerWalk",
                    "axis_id": "joint_damping",
                    "factor_id": "development_mid_b",
                    "factor_value": 1.5,
                    "roles": ["development"],
                    "regime": "heldout_extrapolation",
                    "source_anchor_ref": None,
                },
            ],
            "confirmatory_targets": [],
            "safety_exact_targets": [],
            "primary_algorithm": "PPO",
            "training_steps": 10,
            "training_seeds": [7],
            "checkpoint_rule": "fixed_final_checkpoint",
            "source_eval_episodes": {"selection": 2, "attestation": 2},
            "competence_floor": {"WalkerWalk": 0.5},
            "probe_protocol_id": _d("probe"),
            "probe_prefixes": [1, 2],
            "encoder_eval_prefixes": [1, 2],
            "representation_ids": [
                "raw_transition_v02",
                "corro_anchor_supcon_v02",
            ],
            "method_ids": list(METHODS),
            "primary_endpoint": "pool_regret",
            "noninferiority_margin": 0.01,
            "minimum_effect": 0.05,
            "bootstrap_plan": {
                "resamples": 10,
                "confidence": 0.95,
                "hierarchy": ["task", "axis", "context", "episode_bank"],
                "method": "deterministic_hierarchical_bootstrap",
            },
            "multiple_testing_plan": {
                "simultaneous_interval": "bootstrap_max-T",
                "p_value_adjustment": "holm_bonferroni",
                "alpha": 0.05,
                "families": ["primary"],
            },
            "artifact_root": "/tmp/policy-learnware-v02-development-oracle-test",
        }
    )


def _deployment_entry(
    opaque_id: str, *, abi: ExecutionABIRecord
) -> DeploymentPrivateEntry:
    return DeploymentPrivateEntry(
        opaque_learnware_id=opaque_id,
        learnware_key=f"candidate-{opaque_id}",
        bundle_digest=_d(f"bundle-{opaque_id}"),
        bundle_path=f"/immutable/bundles/{opaque_id}.zip",
        training_attestation_digest=_d(f"training-{opaque_id}"),
        source_selection_digest=_d(f"source-selection-{opaque_id}"),
        source_attestation_digest=_d(f"source-attestation-{opaque_id}"),
        formal_championization_admission_digest=None,
        execution_abi=abi,
    )


def _market(*, include_incompatible: bool = True) -> V02PolicyMarket:
    ids = MARKET_IDS if include_incompatible else MARKET_IDS[:2]
    entries = {
        opaque_id: PublicMarketEntry(
            opaque_learnware_id=opaque_id,
            normalized_source_competence=0.5 + 0.1 * index,
            tie_break_token=_d(f"tie-{opaque_id}"),
        )
        for index, opaque_id in enumerate(ids)
    }
    deployment = {
        opaque_id: _deployment_entry(
            opaque_id,
            abi=(
                _abi(observation="incompatible")
                if opaque_id == "lw-incompatible"
                else _abi()
            ),
        )
        for opaque_id in ids
    }
    deployment_binding_digest = sha256_json(
        {
            opaque_id: {
                "bundle_digest": deployment[opaque_id].bundle_digest,
                "training_attestation_digest": (
                    deployment[opaque_id].training_attestation_digest
                ),
                "source_selection_digest": (
                    deployment[opaque_id].source_selection_digest
                ),
                "source_attestation_digest": (
                    deployment[opaque_id].source_attestation_digest
                ),
                "formal_championization_admission_digest": None,
            }
            for opaque_id in sorted(deployment)
        }
    )
    market_id = sha256_json(
        {
            "schema": "policy-learnware.v02-policy-market-id.v0",
            "entries": {
                opaque_id: entries[opaque_id].to_dict()
                for opaque_id in sorted(entries)
            },
            "deployment_binding_digest": deployment_binding_digest,
        }
    )
    return V02PolicyMarket(
        policy_market_id=market_id,
        entries=entries,
        deployment_private=deployment,
        anchor_to_opaque_id={
            _d(f"anchor-{opaque_id}"): opaque_id for opaque_id in ids
        },
    )


def _targets() -> tuple[DevelopmentTargetEvaluationProtocol, ...]:
    return tuple(
        DevelopmentTargetEvaluationProtocol(
            opaque_query_id=query_id,
            private_target_instance_digest=_d(f"instance-{query_id}"),
            target_evidence_digest=_d(f"target-evidence-{query_id}"),
            target_execution_abi=_abi(),
            seed_contract_digest=_d(f"seed-contract-{query_id}"),
        )
        for query_id in (QUERY_A, QUERY_B)
    )


def _protocol(
    market: V02PolicyMarket,
    *,
    episode_count: int = 2,
    failure_floor: float = 0.1,
) -> FrozenDevelopmentOracleProtocol:
    return FrozenDevelopmentOracleProtocol.from_config(
        _config(),
        market,
        _targets(),
        evaluation_protocol_id=_d("evaluation-protocol"),
        episodes_per_executable_policy=episode_count,
        failure_floor=failure_floor,
        epsilon=0.05,
        tie_atol=0.0,
        candidate_paired_seeds=True,
    )


def _row(
    market: V02PolicyMarket,
    query_id: str,
    policy_id: str,
    episode_index: int,
    value: float,
) -> OracleEpisodeRow:
    target = {item.opaque_query_id: item for item in _targets()}[query_id]
    return OracleEpisodeRow(
        opaque_query_id=query_id,
        opaque_learnware_id=policy_id,
        episode_index=episode_index,
        reset_seed=100 + episode_index,
        policy_seed=200 + episode_index,
        steps=1000,
        raw_return=value * 1000.0,
        normalized_return=value,
        terminated=False,
        truncated=True,
        runtime_seconds=0.1,
        private_target_instance_digest=target.private_target_instance_digest,
        bundle_digest=market.deployment_private[policy_id].bundle_digest,
        seed_contract_digest=target.seed_contract_digest,
        evaluation_protocol_id=_d("evaluation-protocol"),
    )


def _rows(market: V02PolicyMarket) -> tuple[OracleEpisodeRow, ...]:
    values = {
        QUERY_A: {"lw-a": (0.7, 0.9), "lw-b": (0.5, 0.7)},
        QUERY_B: {"lw-a": (0.4, 0.6), "lw-b": (0.8, 1.0)},
    }
    return tuple(
        _row(market, query_id, policy_id, episode, value)
        for query_id in (QUERY_A, QUERY_B)
        for policy_id in ("lw-a", "lw-b")
        for episode, value in enumerate(values[query_id][policy_id])
    )


def _selection_record(
    query_id: str, method_id: str, ranking_ids: tuple[str, ...]
) -> SelectionRecord:
    ranking = tuple(
        RankingRow(
            opaque_learnware_id=opaque_id,
            rank=rank,
            environment_distance=float(rank - 1),
            normalized_source_competence=0.5,
            log_score=-float(rank - 1),
        )
        for rank, opaque_id in enumerate(ranking_ids, start=1)
    )
    return SelectionRecord(
        method_id=method_id,
        selected_id=ranking_ids[0],
        ranking=ranking,
        target_evidence_digest=_d(f"target-evidence-{query_id}"),
        selector_artifact_digest=_d(f"selector-{method_id}"),
        cost_digest=_d(f"cost-{query_id}-{method_id}"),
        evidence_contract=L_MIN_EVIDENCE,
    )


def _selections() -> tuple[PublishedSelectionRanking, ...]:
    rows = []
    for query_id in (QUERY_A, QUERY_B):
        rows.append(
            PublishedSelectionRanking.from_selection_record(
                query_id,
                _selection_record(
                    query_id, "M1", ("lw-a", "lw-b", "lw-incompatible")
                ),
            )
        )
        rows.append(
            PublishedSelectionRanking.from_selection_record(
                query_id,
                _selection_record(
                    query_id, "M2", ("lw-incompatible", "lw-b", "lw-a")
                ),
            )
        )
    return tuple(rows)


def test_typed_admission_recomputes_full_pool_oracle_and_metrics() -> None:
    market = _market()
    protocol = _protocol(market)
    admission = recompute_development_oracle(
        protocol,
        market=market,
        episode_rows=_rows(market),
        selections=_selections(),
    )

    assert set(admission.oracle_by_query) == {QUERY_A, QUERY_B}
    assert set(admission.metrics_by_unit) == set(
        protocol.expected_selection_unit_ids
    )
    assert admission.protocol_digest == protocol.digest
    assert len(admission.raw_episode_rows_digest) == 64
    assert len(admission.selection_bindings_digest) == 64

    incompatible = admission.metrics_by_unit[f"M2::{QUERY_A}"]
    assert incompatible.selected_id == "lw-incompatible"
    assert incompatible.deployment_status == "SELECTED_INCOMPATIBLE_ABI"
    assert incompatible.selection_metrics.selected_normalized_return == pytest.approx(
        0.1
    )
    assert incompatible.selection_metrics.pool_regret == pytest.approx(0.7)
    assert incompatible.ranking_metrics.policy_count == 2
    oracle = admission.oracle_by_query[QUERY_A]
    assert incompatible.episode_rows_digest == oracle.episode_rows_digest
    assert (
        incompatible.execution_abi_census_digest
        == oracle.execution_abi_census_digest
    )
    assert incompatible.selection_record_digest == _selections()[1].selection_record.digest
    with pytest.raises(TypeError):
        admission.metrics_by_unit["new"] = incompatible  # type: ignore[index]


def test_protocol_derives_exact_development_queries_and_methods_from_config() -> None:
    market = _market()
    protocol = _protocol(market)
    assert protocol.development_query_ids == tuple(sorted((QUERY_A, QUERY_B)))
    assert protocol.method_ids == tuple(sorted(METHODS))
    assert protocol.to_private_dict()["scope"] == "v02-development-only"
    serialized = str(protocol.to_private_dict()).lower()
    assert "confirmatory" not in serialized
    assert "sealed" not in serialized
    assert "safety" not in serialized

    with pytest.raises(DevelopmentOracleAdmissionError, match="config-derived"):
        FrozenDevelopmentOracleProtocol.from_config(
            _config(),
            market,
            _targets()[:-1],
            evaluation_protocol_id=_d("evaluation-protocol"),
            episodes_per_executable_policy=2,
            failure_floor=0.1,
            epsilon=0.05,
            tie_atol=0.0,
            candidate_paired_seeds=True,
        )


def test_episode_count_and_failure_floor_are_explicit_digest_bound_literals() -> None:
    market = _market()
    base = _protocol(market, episode_count=2, failure_floor=0.1)
    count_changed = _protocol(market, episode_count=3, failure_floor=0.1)
    floor_changed = _protocol(market, episode_count=2, failure_floor=0.2)
    assert len({base.digest, count_changed.digest, floor_changed.digest}) == 3
    assert base.to_private_dict()["episodes_per_executable_policy"] == 2
    assert base.to_private_dict()["failure_floor"] == 0.1


def test_value_maps_and_query_omission_are_rejected() -> None:
    market = _market()
    protocol = _protocol(market)
    with pytest.raises(DevelopmentOracleAdmissionError, match="value maps"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows={"lw-a": 0.8},  # type: ignore[arg-type]
            selections=_selections(),
        )
    rows = tuple(row for row in _rows(market) if row.opaque_query_id != QUERY_B)
    with pytest.raises(DevelopmentOracleAdmissionError, match="query IDs"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=rows,
            selections=_selections(),
        )


def test_submarket_and_method_or_ranking_omission_are_rejected() -> None:
    market = _market()
    protocol = _protocol(market)
    with pytest.raises(DevelopmentOracleAdmissionError, match="policy market differs"):
        recompute_development_oracle(
            protocol,
            market=_market(include_incompatible=False),
            episode_rows=_rows(market),
            selections=_selections(),
        )
    with pytest.raises(DevelopmentOracleAdmissionError, match="omits or adds"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=_rows(market),
            selections=_selections()[:-1],
        )

    evidence = _selections()[0]
    short_record = _selection_record(QUERY_A, "M1", ("lw-a", "lw-b"))
    short = PublishedSelectionRanking.from_selection_record(QUERY_A, short_record)
    poisoned = (short,) + _selections()[1:]
    with pytest.raises(DevelopmentOracleAdmissionError, match="full frozen market"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=_rows(market),
            selections=poisoned,
        )
    assert evidence.full_ranking_digest != short.full_ranking_digest


def test_bundle_swap_and_episode_count_drift_are_rejected() -> None:
    market = _market()
    protocol = _protocol(market)
    rows = list(_rows(market))
    rows[0] = replace(rows[0], bundle_digest=_d("post-publication-bundle-swap"))
    with pytest.raises(DevelopmentOracleAdmissionError, match="bundle differs"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=rows,
            selections=_selections(),
        )

    with pytest.raises(DevelopmentOracleAdmissionError, match="episode count"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=_rows(market)[:-1],
            selections=_selections(),
        )


def test_published_projection_cannot_disagree_with_full_selection_record() -> None:
    record = _selection_record(
        QUERY_A, "M1", ("lw-a", "lw-b", "lw-incompatible")
    )
    with pytest.raises(DevelopmentOracleAdmissionError, match="differs"):
        PublishedSelectionRanking(
            opaque_query_id=QUERY_A,
            published_selection=PublishedSelection(
                method_id="M1",
                selection_record_digest=_d("forged-selection"),
                selected_id="lw-b",
            ),
            full_ranking=record.ranking,
            selection_record=record,
        )


def test_selection_record_cannot_be_rewrapped_as_another_query() -> None:
    market = _market()
    protocol = _protocol(market)
    selections = list(_selections())
    record_from_query_a = selections[0].selection_record
    assert selections[0].opaque_query_id == QUERY_A
    selections[0] = PublishedSelectionRanking.from_selection_record(
        QUERY_B, record_from_query_a
    )
    record_from_query_b = selections[2].selection_record
    assert selections[2].opaque_query_id == QUERY_B
    selections[2] = PublishedSelectionRanking.from_selection_record(
        QUERY_A, record_from_query_b
    )

    with pytest.raises(DevelopmentOracleAdmissionError, match="frozen raw query evidence"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=_rows(market),
            selections=selections,
        )


def test_incompatible_policy_rows_and_individual_invalid_values_fail_closed() -> None:
    market = _market()
    protocol = _protocol(market)
    incompatible_row = _row(market, QUERY_A, "lw-incompatible", 0, 0.99)
    with pytest.raises(DevelopmentOracleAdmissionError, match="incompatible"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=_rows(market) + (incompatible_row,),
            selections=_selections(),
        )

    rows = list(_rows(market))
    rows[0] = replace(rows[0], normalized_return=1.2)
    rows[1] = replace(rows[1], normalized_return=0.4)
    with pytest.raises(DevelopmentOracleAdmissionError, match="normalized_return"):
        recompute_development_oracle(
            protocol,
            market=market,
            episode_rows=rows,
            selections=_selections(),
        )
