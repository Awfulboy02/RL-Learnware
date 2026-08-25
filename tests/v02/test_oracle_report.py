from __future__ import annotations

import json

import pytest

from policy_learnware_v0.v02.oracle import (
    OracleContractError,
    OracleEpisodeRow,
    PublishedSelection,
    aggregate_full_pool_oracle,
    minimum_executable_set,
)
from policy_learnware_v0.v02.report import (
    DevelopmentPTableRow,
    PrivateOTableRow,
    ReferenceETableRow,
    ReportContractError,
    assert_nonprivate_table_payload,
    build_development_p_table,
    build_private_o_table,
    build_reference_e_table,
)
from policy_learnware_v0.v02.schemas import ExecutionABIRecord


QUERY = "v02q-test"
INSTANCE = "1" * 64
EVALUATION = "2" * 64
SEED_CONTRACT = "3" * 64


def _abi(*, observation: str = "a", runtime: str = "legacy-ppo-fpo-v02") -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=observation * 64,
        action_tensor_abi_digest="b" * 64,
        action_transform_id="tanh",
        policy_runtime_id=runtime,
        state_abi_id="stateless",
    )


def _row(policy: str, episode: int, value: float) -> OracleEpisodeRow:
    return OracleEpisodeRow(
        opaque_query_id=QUERY,
        opaque_learnware_id=policy,
        episode_index=episode,
        reset_seed=100 + episode,
        policy_seed=200 + episode,
        steps=1000,
        raw_return=value * 1000.0,
        normalized_return=value,
        terminated=False,
        truncated=True,
        runtime_seconds=0.5,
        private_target_instance_digest=INSTANCE,
        bundle_digest=("4" if policy == "lw-a" else "5") * 64,
        seed_contract_digest=SEED_CONTRACT,
        evaluation_protocol_id=EVALUATION,
    )


def _oracle_result():
    market = ("lw-a", "lw-b", "lw-incompatible")
    registry = {
        "lw-a": _abi(),
        "lw-b": _abi(),
        "lw-incompatible": _abi(observation="c"),
    }
    rows = (
        _row("lw-a", 0, 0.7),
        _row("lw-a", 1, 0.9),
        _row("lw-b", 0, 0.8),
        _row("lw-b", 1, 0.8),
    )
    selections = (
        PublishedSelection("M02/B5", "6" * 64, "lw-a"),
        PublishedSelection("B0", "7" * 64, "lw-incompatible"),
    )
    return aggregate_full_pool_oracle(
        opaque_query_id=QUERY,
        private_target_instance_digest=INSTANCE,
        evaluation_protocol_id=EVALUATION,
        market_ids=market,
        deployment_registry=registry,
        target_execution_abi=_abi(),
        episode_rows=rows,
        published_selections=selections,
        failure_floor=0.1,
        tie_atol=0.0,
        candidate_paired_seeds=True,
    )


def test_private_minimum_abi_census_and_full_pool_oracle_ties() -> None:
    registry = {
        "lw-a": _abi(),
        "lw-b": _abi(),
        "lw-incompatible": _abi(observation="c"),
    }
    assert minimum_executable_set(tuple(registry), registry, _abi()) == (
        "lw-a",
        "lw-b",
    )
    result = _oracle_result()
    assert result.executable_ids == ("lw-a", "lw-b")
    assert result.incompatible_ids == ("lw-incompatible",)
    assert result.best_in_pool_ids == ("lw-a", "lw-b")
    assert result.best_in_pool_value == pytest.approx(0.8)
    assert result.normalized_value_vector == {
        "lw-a": pytest.approx(0.8),
        "lw-b": pytest.approx(0.8),
        "lw-incompatible": None,
    }
    with pytest.raises(TypeError):
        result.normalized_value_vector["lw-a"] = 0.0  # type: ignore[index]


def test_selected_incompatible_uses_floor_without_ranking_fallback() -> None:
    result = _oracle_result()
    selected = result.outcomes["B0"]
    assert selected.selected_id == "lw-incompatible"
    assert selected.deployment_status == "SELECTED_INCOMPATIBLE_ABI"
    assert selected.selected_value == pytest.approx(0.1)
    assert selected.regret == pytest.approx(0.7)
    assert selected.within_executable_regret == 0.0
    assert selected.deployment_failure_regret == pytest.approx(0.7)
    assert not selected.oracle_top1_agreement
    assert result.outcomes["M02/B5"].oracle_top1_agreement
    assert result.outcomes["M02/B5"].regret == pytest.approx(0.0)


def test_full_pool_rows_fail_closed_on_abi_and_pairing_drift() -> None:
    market = ("lw-a", "lw-b", "lw-incompatible")
    registry = {
        "lw-a": _abi(),
        "lw-b": _abi(),
        "lw-incompatible": _abi(observation="c"),
    }
    with pytest.raises(OracleContractError, match="incompatible policy"):
        aggregate_full_pool_oracle(
            opaque_query_id=QUERY,
            private_target_instance_digest=INSTANCE,
            evaluation_protocol_id=EVALUATION,
            market_ids=market,
            deployment_registry=registry,
            target_execution_abi=_abi(),
            episode_rows=(
                _row("lw-a", 0, 0.8),
                _row("lw-b", 0, 0.7),
                _row("lw-incompatible", 0, 0.9),
            ),
            published_selections=(),
            failure_floor=0.0,
        )
    with pytest.raises(OracleContractError, match="paired oracle seeds"):
        drift = OracleEpisodeRow(
            **{
                **_row("lw-b", 0, 0.7).__dict__,
                "reset_seed": 999,
            }
        )
        aggregate_full_pool_oracle(
            opaque_query_id=QUERY,
            private_target_instance_digest=INSTANCE,
            evaluation_protocol_id=EVALUATION,
            market_ids=market,
            deployment_registry=registry,
            target_execution_abi=_abi(),
            episode_rows=(_row("lw-a", 0, 0.8), drift),
            published_selections=(),
            failure_floor=0.0,
        )


def test_development_p_and_private_o_tables_are_separate() -> None:
    result = _oracle_result()
    outcome = result.outcomes["B0"]
    p_row = DevelopmentPTableRow.from_oracle_outcome(
        opaque_query_id=QUERY,
        bank_index=0,
        prefix=32,
        representation_id="corro-anchor-supcon-v02",
        outcome=outcome,
        epsilon=0.05,
        target_transition_count=32_000,
        target_evidence_digest="8" * 64,
        evidence_contract_digest="9" * 64,
        cost_digest="a" * 64,
    )
    p_table = build_development_p_table(
        (p_row,),
        development_split_digest="b" * 64,
        policy_market_id="market-v02",
        evaluation_protocol_id=EVALUATION,
    )
    public_text = json.dumps(p_table.to_dict(), sort_keys=True)
    assert p_table.to_dict()["stage"] == "development_discovery"
    assert "true_task_id" not in public_text
    assert "execution_abi_census_digest" not in public_text
    assert "value_vector" not in public_text
    assert p_row.selected_normalized_return == pytest.approx(0.1)
    assert not p_row.epsilon_optimal

    o_row = PrivateOTableRow.from_oracle_result(
        result,
        true_task_id="WalkerWalk",
        true_axis_id="joint-damping",
        true_factor=1.25,
        regime="heldout_interpolation",
        physical_nearest_anchor_id="anchor-physical-nearest",
        true_distance_lmin_selected_id="lw-b",
        source_global_champion_id="lw-incompatible",
    )
    o_table = build_private_o_table(
        (o_row,),
        policy_market_id="market-v02",
        evaluation_protocol_id=EVALUATION,
    )
    private = o_table.to_private_dict()
    assert private["visibility"] == "private-oracle-analysis-only"
    assert private["rows"][0]["true_task_id"] == "WalkerWalk"
    assert private["rows"][0]["best_in_pool_ids"] == ["lw-a", "lw-b"]
    assert private["rows"][0]["full_candidate_value_vector"]["lw-incompatible"] is None
    assert private["rows"][0]["source_global_champion"]["regret_g0"] == pytest.approx(0.7)
    assert private["rows"][0]["selected_method_regret_decomposition"]["B0"][
        "deployment_status"
    ] == "SELECTED_INCOMPATIBLE_ABI"
    assert not hasattr(o_table, "to_dict")


def test_reference_e_table_is_complete_but_cannot_fake_proposed_rows() -> None:
    raw = ReferenceETableRow(
        reference_kind="raw",
        representation_id="raw-transition-v02",
        representation_version="v0",
        training_split_digest="1" * 64,
        canonical_event_view_digest="2" * 64,
        probe_protocol_id="3" * 64,
        checkpoint_digest=None,
        normalizer_digest=None,
        latent_contract_digest="4" * 64,
        kernel_digest="5" * 64,
        reducer_digest="6" * 64,
        heldout_neighborhood_score=0.6,
        heldout_order_score=0.7,
        repeated_bank_stability=0.8,
        signal_to_noise_ratio=2.0,
        prefix_budgets=(1, 2, 4, 8, 16, 32),
        prefix_selected_returns=(0.4, 0.5, 0.55, 0.6, 0.62, 0.64),
        prefix_regrets=(0.4, 0.3, 0.25, 0.2, 0.18, 0.16),
        sample_efficiency_auc=0.55,
        fixed_lmin_selected_return=0.64,
        fixed_lmin_regret=0.16,
        cold_encoding_seconds=0.2,
        warm_encoding_seconds=0.05,
    )
    table = build_reference_e_table((raw,), evaluation_protocol_id=EVALUATION)
    payload = table.to_dict()
    assert payload["included_reference_kinds"] == ["raw"]
    assert payload["rows"][0]["component_digests"]["checkpoint"] is None
    assert "true_task_id" not in json.dumps(payload, sort_keys=True)
    assert "value_vector" not in json.dumps(payload, sort_keys=True)

    with pytest.raises(ReportContractError, match="only raw/legacy/refit"):
        ReferenceETableRow(
            **{
                **raw.__dict__,
                "reference_kind": "proposed",  # type: ignore[arg-type]
            }
        )
    with pytest.raises(ReportContractError, match="leaks private oracle"):
        assert_nonprivate_table_payload(
            {"schema": "bad", "rows": [{"full_candidate_value_vector": {}}]},
            table="bad P-table",
        )
