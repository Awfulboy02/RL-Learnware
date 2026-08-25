from __future__ import annotations

import numpy as np
import pytest

from policy_learnware_v0.v02.environment_spec import (
    RepresentationIndex,
    RepresentationIndexEntry,
    environment_spec_distance,
    source_only_median_scale,
)
from policy_learnware_v0.v02.schemas import EnvironmentSpec, PublicMarketEntry
from policy_learnware_v0.v02.selectors import EvidenceContract, LMinSelector, PublicMarketView


DIGEST = "1" * 64
REP = "2" * 64
MEAS = "3" * 64
VIEW = "4" * 64


def _spec(center: float, *, representation: str = REP) -> EnvironmentSpec:
    supports = np.asarray([[center], [center + 0.1]], dtype=np.float64)
    beta = np.asarray([0.5, 0.5], dtype=np.float64)
    bandwidth = 1.0
    gram = np.exp(-np.square(supports - supports.T) / (2.0 * bandwidth**2))
    norm = float(beta @ gram @ beta)
    return EnvironmentSpec(
        supports=supports,
        beta=beta,
        empirical_norm2=norm,
        rkme_norm2=norm,
        reconstruction_error=0.0,
        reducer_digest=DIGEST,
        support_budget=2,
        latent_dim=1,
        representation_protocol_id=representation,
        measurement_protocol_id=MEAS,
        canonical_view_digest=VIEW,
        kernel_bandwidth=bandwidth,
        probe_dataset_digest="5" * 64,
    )


def _market() -> PublicMarketView:
    specs = {"anchor-a": _spec(0.0), "anchor-b": _spec(1.0), "anonymous-c": _spec(5.0)}
    entries = {
        "anchor-a": PublicMarketEntry("anchor-a", 0.9, "a" * 64),
        "anchor-b": PublicMarketEntry("anchor-b", 0.7, "b" * 64),
        "anonymous-c": PublicMarketEntry("anonymous-c", 1.0, "c" * 64),
    }
    index = RepresentationIndex(
        policy_market_id="market-1",
        representation_protocol_id=REP,
        entries={key: RepresentationIndexEntry(key, value) for key, value in specs.items()},
    )
    return PublicMarketView("market-1", entries, index)


def test_environment_spec_distance_is_symmetric_exact_and_protocol_checked() -> None:
    left = _spec(0.0)
    right = _spec(1.0)
    assert environment_spec_distance(left, left, distance_form="mmd").distance == pytest.approx(0.0)
    assert environment_spec_distance(left, right, distance_form="mmd").distance == pytest.approx(
        environment_spec_distance(right, left, distance_form="mmd").distance
    )
    with pytest.raises(ValueError, match="representation_protocol_id"):
        environment_spec_distance(left, _spec(1.0, representation="9" * 64), distance_form="mmd")


def test_source_only_sigma_uses_nonzero_pairwise_median() -> None:
    specs = {"a": _spec(0.0), "b": _spec(1.0), "c": _spec(2.0)}
    scale = source_only_median_scale(
        specs, partitions={"task": ("a", "b", "c")}, distance_form="mmd"
    )
    assert scale["task"] > 0.0
    with pytest.raises(ValueError, match="no non-zero"):
        source_only_median_scale(
            {"a": specs["a"], "b": specs["a"]},
            partitions={"task": ("a", "b")},
            distance_form="mmd",
        )


def test_lmin_formula_uses_anonymous_full_market_and_deterministic_ranking() -> None:
    selector = LMinSelector(method_id="M02/B5", sigma=1.0, epsilon=1.0e-12, distance_form="mmd")
    result = selector.select(
        query_spec=_spec(0.05),
        market=_market(),
        target_evidence_digest="6" * 64,
        cost_digest="7" * 64,
    )
    assert result.status == "SELECTED"
    assert result.selected_id == "anchor-a"
    assert len(result.ranking) == 3
    assert {row.opaque_id for row in result.ranking} == {"anchor-a", "anchor-b", "anonymous-c"}
    assert "environment_distance" in result.to_dict()["ranking"][0]


def test_public_market_rejects_tie_token_collision() -> None:
    selector = LMinSelector(method_id="M02/B5", sigma=1.0, epsilon=1.0e-12, distance_form="mmd2")
    del selector
    with pytest.raises(ValueError, match="tie_break_token"):
        PublicMarketView(
            "market-small",
            {
                "anchor-a": PublicMarketEntry("anchor-a", 0.9, "a" * 64),
                "anchor-b": PublicMarketEntry("anchor-b", 0.8, "a" * 64),
            },
            RepresentationIndex(
                policy_market_id="market-small",
                representation_protocol_id=REP,
                entries={
                    "anchor-a": RepresentationIndexEntry("anchor-a", _spec(0.0)),
                    "anchor-b": RepresentationIndexEntry("anchor-b", _spec(1.0)),
                },
            ),
        )


def test_public_evidence_contract_rejects_oracle_and_target_update_permissions() -> None:
    unsafe = EvidenceContract(
        reads_source_raw_data=False,
        reads_development_policy_returns=False,
        reads_target_parameters=False,
        reads_target_transitions=True,
        reads_candidate_independent_probe_rewards=False,
        reads_candidate_target_rollouts=True,
        reads_candidate_policy_target_rewards=False,
        target_gradient_updates=0,
        reads_submit_side_profiles=False,
    )
    with pytest.raises(ValueError, match="oracle/update"):
        LMinSelector(
            method_id="unsafe", sigma=1.0, epsilon=1.0e-12, distance_form="mmd", evidence_contract=unsafe
        )
