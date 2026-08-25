from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from policy_learnware_v0.v02.baselines import (
    BaselineContractError,
    BaselineRegistry,
    CompetenceOnlySelector,
    DevelopmentView,
    DuplicateBaselineError,
    EnvironmentOnlySelector,
    FrozenFeatureIndex,
    KnnDevelopmentSelector,
    LegacyTaskSpecSelector,
    LinearDevelopmentSelector,
    RandomAnonymousMarketSelector,
    SourceOnlyLMinSelector,
    TargetQueryView,
    VectorNearestSelector,
    derive_source_only_sigma_artifacts,
    finite_pool_random_probabilities,
    target_probe_evidence_contract,
)
from policy_learnware_v0.v02.environment_spec import (
    RepresentationIndex,
    RepresentationIndexEntry,
)
from policy_learnware_v0.v02.representation import TraceFeatureVector
from policy_learnware_v0.v02.schemas import EnvironmentSpec, PublicMarketEntry
from policy_learnware_v0.v02.selectors import EvidenceContract, PublicMarketView


RAW_REP = "1" * 64
CORRO_REP = "2" * 64
MEASUREMENT = "3" * 64
CANONICAL_VIEW = "4" * 64
PROBE_DATASET = "5" * 64
FEATURE_PROTOCOL = "6" * 64
PHYSICAL_IDS = ("physical-0", "physical-1", "physical-2")
CENTERS = {"physical-0": 0.0, "physical-1": 2.0, "physical-2": 4.0}
COMPETENCE = {"physical-0": 0.5, "physical-1": 0.9, "physical-2": 0.9}
# physical-1 must win every exact public-policy tie, even when its alias sorts last.
TIE_TOKENS = {"physical-0": "3" * 64, "physical-1": "1" * 64, "physical-2": "2" * 64}
ALIASES_A = {"physical-0": "lw-a", "physical-1": "lw-z", "physical-2": "lw-m"}
ALIASES_B = {"physical-0": "lw-y", "physical-1": "lw-b", "physical-2": "lw-x"}


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _spec(center: float, representation: str) -> EnvironmentSpec:
    supports = np.asarray([[center]], dtype=np.float64)
    return EnvironmentSpec(
        supports=supports,
        beta=np.asarray([1.0]),
        empirical_norm2=1.0,
        rkme_norm2=1.0,
        reconstruction_error=0.0,
        reducer_digest=_d(f"reducer:{representation}"),
        support_budget=1,
        latent_dim=1,
        representation_protocol_id=representation,
        measurement_protocol_id=MEASUREMENT,
        canonical_view_digest=CANONICAL_VIEW,
        kernel_bandwidth=1.0,
        probe_dataset_digest=PROBE_DATASET,
    )


def _market(
    aliases: dict[str, str], representation: str
) -> PublicMarketView:
    market_id = _d(
        "market:"
        + representation
        + ":"
        + ",".join(f"{physical}={aliases[physical]}" for physical in PHYSICAL_IDS)
    )
    entries = {
        aliases[physical]: PublicMarketEntry(
            opaque_learnware_id=aliases[physical],
            normalized_source_competence=COMPETENCE[physical],
            tie_break_token=TIE_TOKENS[physical],
        )
        for physical in PHYSICAL_IDS
    }
    index = RepresentationIndex(
        policy_market_id=market_id,
        representation_protocol_id=representation,
        entries={
            aliases[physical]: RepresentationIndexEntry(
                aliases[physical], _spec(CENTERS[physical], representation)
            )
            for physical in PHYSICAL_IDS
        },
    )
    return PublicMarketView(
        policy_market_id=market_id,
        entries=entries,
        representation_index=index,
    )


def _trace(value: float, *, protocol: str = FEATURE_PROTOCOL) -> TraceFeatureVector:
    return TraceFeatureVector(
        values=np.asarray([value], dtype=np.float64),
        feature_protocol_id=protocol,
        probe_dataset_digest=PROBE_DATASET,
    )


def _query(
    representation: str,
    *,
    trace: TraceFeatureVector | None = None,
    stage: str = "development_discovery",
) -> TargetQueryView:
    return TargetQueryView(
        stage=stage,  # type: ignore[arg-type]
        query_spec=_spec(1.0, representation),
        target_evidence_digest=_d("target-evidence"),
        cost_digest=_d("query-cost"),
        probe_rewards_included=False,
        trace_feature=trace,
    )


def _feature_index(
    aliases: dict[str, str], market: PublicMarketView
) -> FrozenFeatureIndex:
    return FrozenFeatureIndex(
        policy_market_id=market.policy_market_id,
        feature_protocol_id=FEATURE_PROTOCOL,
        entries={
            aliases[physical]: TraceFeatureVector(
                values=np.asarray([CENTERS[physical]], dtype=np.float64),
                feature_protocol_id=FEATURE_PROTOCOL,
                probe_dataset_digest=_d(f"source-probe:{physical}"),
            )
            for physical in PHYSICAL_IDS
        },
    )


def _development_view(
    aliases: dict[str, str], *, extra_policy: bool = False
) -> DevelopmentView:
    policy_ids = [aliases[physical] for physical in PHYSICAL_IDS]
    one_row = [0.2, 0.8, 0.8]
    if extra_policy:
        policy_ids.append("lw-extra")
        one_row.append(0.1)
    returns = np.asarray([one_row, one_row, one_row, one_row], dtype=np.float64)
    return DevelopmentView(
        context_ids=("dev-0", "dev-1", "dev-2", "dev-validation"),
        opaque_policy_ids=tuple(policy_ids),
        context_features=np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        normalized_returns=returns,
        training_context_ids=("dev-0", "dev-1", "dev-2"),
        validation_context_ids=("dev-validation",),
        evaluation_seed_digests=tuple(_d(f"seed:{index}") for index in range(4)),
        policy_market_id=_market(aliases, RAW_REP).policy_market_id,
        feature_protocol_id=FEATURE_PROTOCOL,
        split_manifest_digest=_d("development-split"),
        label_contract_digest=_d("development-label-contract"),
        candidate_paired_seeds=True,
    )


def _evidence(*, supervised: bool = False, source_labels: bool = False) -> EvidenceContract:
    return target_probe_evidence_contract(
        reads_development_policy_returns=supervised,
        reads_probe_rewards=False,
        reads_source_side_labels=source_labels,
    )


def _legacy_selector(aliases: dict[str, str]) -> LegacyTaskSpecSelector:
    market = _market(aliases, RAW_REP)
    return LegacyTaskSpecSelector(
        method_id="B2",
        source_task_specs={"source-side-0": _trace(0.0), "source-side-1": _trace(2.0)},
        nominal_champions={
            "source-side-0": aliases["physical-0"],
            "source-side-1": aliases["physical-1"],
        },
        policy_market_id=market.policy_market_id,
        evidence_contract=_evidence(source_labels=True),
    )


def _run_all_methods(aliases: dict[str, str]):
    raw_market = _market(aliases, RAW_REP)
    corro_market = _market(aliases, CORRO_REP)
    trace = _trace(1.0)
    raw_query = _query(RAW_REP, trace=trace)
    corro_query = _query(CORRO_REP, trace=trace)
    development = _development_view(aliases)
    feature_index = _feature_index(aliases, raw_market)

    sigma = derive_source_only_sigma_artifacts(
        corro_market.representation_index,
        partitions={"anonymous-global": tuple(corro_market.entries)},
        distance_form="mmd",
    )["anonymous-global"]
    methods = {
        "B0": (
            RandomAnonymousMarketSelector(
                method_id="B0",
                selector_seed=17,
                policy_market_id=raw_market.policy_market_id,
            ),
            raw_query,
            raw_market,
            None,
        ),
        "B1": (
            CompetenceOnlySelector(
                method_id="B1", policy_market_id=raw_market.policy_market_id
            ),
            raw_query,
            raw_market,
            None,
        ),
        "B2": (_legacy_selector(aliases), raw_query, raw_market, None),
        "B3a": (
            VectorNearestSelector(
                method_id="B3a",
                feature_index=feature_index,
                evidence_contract=_evidence(),
            ),
            raw_query,
            raw_market,
            None,
        ),
        "B3b": (
            EnvironmentOnlySelector(
                method_id="B3b",
                distance_form="mmd",
                policy_market_id=raw_market.policy_market_id,
                representation_index_id=str(
                    raw_market.representation_index.representation_index_id
                ),
                evidence_contract=_evidence(),
            ),
            raw_query,
            raw_market,
            None,
        ),
        "B4a": (
            KnnDevelopmentSelector(
                method_id="B4a",
                neighbor_count=1,
                policy_market_id=raw_market.policy_market_id,
                evidence_contract=_evidence(supervised=True),
            ),
            raw_query,
            raw_market,
            development,
        ),
        "B4b": (
            LinearDevelopmentSelector(
                method_id="B4b",
                ridge=0.1,
                policy_market_id=raw_market.policy_market_id,
                evidence_contract=_evidence(supervised=True),
            ),
            raw_query,
            raw_market,
            development,
        ),
        "A-Env": (
            EnvironmentOnlySelector(
                method_id="A-Env",
                distance_form="mmd",
                policy_market_id=corro_market.policy_market_id,
                representation_index_id=str(
                    corro_market.representation_index.representation_index_id
                ),
                evidence_contract=_evidence(source_labels=True),
            ),
            corro_query,
            corro_market,
            None,
        ),
        "M02/B5": (
            SourceOnlyLMinSelector(
                method_id="M02/B5",
                sigma_artifact=sigma,
                epsilon=1.0e-12,
                evidence_contract=_evidence(source_labels=True),
            ),
            corro_query,
            corro_market,
            None,
        ),
    }
    results = {}
    for method_id, (selector, query, market, development_data) in methods.items():
        artifact = selector.fit(development_data)
        results[method_id] = selector.select(query, market, artifact)
    return methods, results


def test_all_registered_baselines_share_full_anonymous_market_interface() -> None:
    methods, results = _run_all_methods(ALIASES_A)
    registry = BaselineRegistry()
    for method_id, (selector, _query_view, market, _development) in methods.items():
        registry.register(selector)
        result = results[method_id]
        assert result.status == "SELECTED"
        assert len(result.ranking) == len(market.entries) == 3
        assert {row.opaque_learnware_id for row in result.ranking} == set(market.entries)
        assert tuple(row.rank for row in result.ranking) == (1, 2, 3)
        assert result.evidence_contract.is_public_zero_update
        assert registry.resolve(method_id) is selector
    with pytest.raises(DuplicateBaselineError):
        registry.register(methods["B0"][0])
    with pytest.raises(TypeError):
        registry.selectors["new"] = methods["B0"][0]  # type: ignore[index]


def test_public_views_expose_no_runtime_task_schema_or_abi_hard_gate() -> None:
    assert set(PublicMarketEntry.__dataclass_fields__) == {
        "opaque_learnware_id",
        "normalized_source_competence",
        "tie_break_token",
        "schema",
    }
    assert set(TargetQueryView.__dataclass_fields__) == {
        "stage",
        "query_spec",
        "target_evidence_digest",
        "cost_digest",
        "probe_rewards_included",
        "trace_feature",
    }
    _, results = _run_all_methods(ALIASES_A)
    assert all(len(result.ranking) == 3 for result in results.values())


def test_exact_policy_ties_use_market_token_not_alias_competence_or_source_key() -> None:
    _methods, results = _run_all_methods(ALIASES_A)
    expected = ALIASES_A["physical-1"]
    # B1 ties physical-1/2 on competence.  B2/B3a/B3b/A-Env tie physical-0/1
    # on evidence distance.  B4a/B4b tie physical-1/2 on predicted return.
    for method_id in ("B1", "B2", "B3a", "B3b", "B4a", "B4b", "A-Env"):
        assert results[method_id].selected_id == expected
    assert expected == "lw-z"  # lexical alias order would not choose it.


def test_b2_is_probe_derived_legacy_taskspec_not_target_task_id_routing() -> None:
    with pytest.raises(BaselineContractError, match="source-label permission"):
        LegacyTaskSpecSelector(
            method_id="B2",
            source_task_specs={"source-0": _trace(0.0), "source-1": _trace(2.0)},
            nominal_champions={
                "source-0": ALIASES_A["physical-0"],
                "source-1": ALIASES_A["physical-1"],
            },
            policy_market_id=_market(ALIASES_A, RAW_REP).policy_market_id,
            evidence_contract=_evidence(source_labels=False),
        )
    selector = _legacy_selector(ALIASES_A)
    artifact = selector.fit(None)
    query = _query(RAW_REP, trace=_trace(1.9))
    result = selector.select(query, _market(ALIASES_A, RAW_REP), artifact)
    assert result.selected_id == ALIASES_A["physical-1"]
    assert artifact.payload["tie_break"] == "(distance,nominal_champion.tie_break_token)"
    assert selector.evidence_contract.reads_source_side_labels
    assert not selector.evidence_contract.reads_target_task_reward_schema_identity
    assert "task_id" not in TargetQueryView.__dataclass_fields__


def test_feature_and_supervised_indices_must_exactly_cover_frozen_market() -> None:
    market = _market(ALIASES_A, RAW_REP)
    feature_entries = dict(_feature_index(ALIASES_A, market).entries)
    feature_entries["lw-extra"] = TraceFeatureVector(
        values=np.asarray([9.0]),
        feature_protocol_id=FEATURE_PROTOCOL,
        probe_dataset_digest=_d("source-probe:extra"),
    )
    selector = VectorNearestSelector(
        method_id="B3a-extra",
        feature_index=FrozenFeatureIndex(
            policy_market_id=market.policy_market_id,
            feature_protocol_id=FEATURE_PROTOCOL,
            entries=feature_entries,
        ),
        evidence_contract=_evidence(),
    )
    with pytest.raises(BaselineContractError, match="exactly cover"):
        selector.select(_query(RAW_REP, trace=_trace(1.0)), market, selector.fit(None))

    development = _development_view(ALIASES_A, extra_policy=True)
    for ranker in (
        KnnDevelopmentSelector(
            method_id="B4a-extra",
            neighbor_count=1,
            policy_market_id=market.policy_market_id,
            evidence_contract=_evidence(supervised=True),
        ),
        LinearDevelopmentSelector(
            method_id="B4b-extra",
            ridge=0.1,
            policy_market_id=market.policy_market_id,
            evidence_contract=_evidence(supervised=True),
        ),
    ):
        with pytest.raises(BaselineContractError, match="exactly cover"):
            ranker.select(
                _query(RAW_REP, trace=_trace(1.0)), market, ranker.fit(development)
            )


def test_supervised_artifact_requires_freeze_ref_for_joint_confirmatory() -> None:
    market = _market(ALIASES_A, RAW_REP)
    selector = KnnDevelopmentSelector(
        method_id="B4a",
        neighbor_count=1,
        policy_market_id=market.policy_market_id,
        evidence_contract=_evidence(supervised=True),
    )
    artifact = selector.fit(_development_view(ALIASES_A))
    query = _query(
        RAW_REP,
        trace=_trace(1.0),
        stage="paper1_joint_confirmatory",
    )
    with pytest.raises(BaselineContractError, match="development freeze"):
        selector.select(query, market, artifact)
    frozen = artifact.freeze_for_confirmatory(_d("development-freeze"))
    assert selector.select(query, market, frozen).status == "SELECTED"


@pytest.mark.parametrize("poison", ("seed", "feature_index", "source_specs", "sigma"))
def test_stale_artifact_cannot_execute_on_a_different_selector_instance(
    poison: str,
) -> None:
    market = _market(ALIASES_A, RAW_REP)
    query = _query(RAW_REP, trace=_trace(1.0))
    if poison == "seed":
        original = RandomAnonymousMarketSelector(
            method_id="stale",
            selector_seed=1,
            policy_market_id=market.policy_market_id,
        )
        executing = RandomAnonymousMarketSelector(
            method_id="stale",
            selector_seed=2,
            policy_market_id=market.policy_market_id,
        )
    elif poison == "feature_index":
        original = VectorNearestSelector(
            method_id="stale",
            feature_index=_feature_index(ALIASES_A, market),
            evidence_contract=_evidence(),
        )
        changed_entries = dict(_feature_index(ALIASES_A, market).entries)
        changed_entries[ALIASES_A["physical-0"]] = _trace(9.0)
        executing = VectorNearestSelector(
            method_id="stale",
            feature_index=FrozenFeatureIndex(
                policy_market_id=market.policy_market_id,
                feature_protocol_id=FEATURE_PROTOCOL,
                entries=changed_entries,
            ),
            evidence_contract=_evidence(),
        )
    elif poison == "source_specs":
        champions = {
            "source-side-0": ALIASES_A["physical-0"],
            "source-side-1": ALIASES_A["physical-1"],
        }
        original = LegacyTaskSpecSelector(
            method_id="stale",
            source_task_specs={
                "source-side-0": _trace(0.0),
                "source-side-1": _trace(2.0),
            },
            nominal_champions=champions,
            policy_market_id=market.policy_market_id,
            evidence_contract=_evidence(source_labels=True),
        )
        executing = LegacyTaskSpecSelector(
            method_id="stale",
            source_task_specs={
                "source-side-0": _trace(0.25),
                "source-side-1": _trace(2.0),
            },
            nominal_champions=champions,
            policy_market_id=market.policy_market_id,
            evidence_contract=_evidence(source_labels=True),
        )
    else:
        sigma = derive_source_only_sigma_artifacts(
            market.representation_index,
            partitions={"anonymous-global": tuple(market.entries)},
            distance_form="mmd",
        )["anonymous-global"]
        original = SourceOnlyLMinSelector(
            method_id="stale",
            sigma_artifact=sigma,
            epsilon=1.0e-12,
            evidence_contract=_evidence(),
        )
        executing = SourceOnlyLMinSelector(
            method_id="stale",
            sigma_artifact=replace(
                sigma, sigma=2.0 * sigma.sigma, artifact_digest=None
            ),
            epsilon=1.0e-12,
            evidence_contract=_evidence(),
        )
    artifact = original.fit(None)
    with pytest.raises(BaselineContractError, match="runtime binding differs"):
        executing.select(query, market, artifact)


def test_selector_artifact_cannot_execute_on_another_policy_market() -> None:
    market = _market(ALIASES_A, RAW_REP)
    selector = RandomAnonymousMarketSelector(
        method_id="B0", selector_seed=17, policy_market_id=market.policy_market_id
    )
    artifact = selector.fit(None)
    with pytest.raises(BaselineContractError, match="another policy market"):
        selector.select(_query(RAW_REP), _market(ALIASES_B, RAW_REP), artifact)


def test_alias_permutation_preserves_selected_physical_entry_and_full_ranking() -> None:
    _methods_a, results_a = _run_all_methods(ALIASES_A)
    _methods_b, results_b = _run_all_methods(ALIASES_B)
    inverse_a = {opaque: physical for physical, opaque in ALIASES_A.items()}
    inverse_b = {opaque: physical for physical, opaque in ALIASES_B.items()}
    for method_id in results_a:
        physical_ranking_a = tuple(
            inverse_a[row.opaque_learnware_id] for row in results_a[method_id].ranking
        )
        physical_ranking_b = tuple(
            inverse_b[row.opaque_learnware_id] for row in results_b[method_id].ranking
        )
        assert physical_ranking_a == physical_ranking_b, method_id
        assert inverse_a[results_a[method_id].selected_id] == inverse_b[
            results_b[method_id].selected_id
        ]


def test_random_expectation_is_uniform_over_the_full_anonymous_market() -> None:
    market = _market(ALIASES_A, RAW_REP)
    probabilities = finite_pool_random_probabilities(_query(RAW_REP), market)
    assert set(probabilities) == set(market.entries)
    assert tuple(probabilities.values()) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_private_or_target_identity_evidence_permissions_fail_closed() -> None:
    unsafe = EvidenceContract(
        reads_source_raw_data=False,
        reads_development_policy_returns=False,
        reads_target_parameters=False,
        reads_target_transitions=True,
        reads_candidate_independent_probe_rewards=False,
        reads_candidate_target_rollouts=False,
        reads_candidate_policy_target_rewards=False,
        target_gradient_updates=0,
        reads_submit_side_profiles=False,
        reads_source_side_labels=False,
        reads_target_task_reward_schema_identity=True,
    )
    with pytest.raises(ValueError, match="private/oracle/update"):
        EnvironmentOnlySelector(
            method_id="unsafe",
            distance_form="mmd",
            policy_market_id=_market(ALIASES_A, RAW_REP).policy_market_id,
            representation_index_id=str(
                _market(ALIASES_A, RAW_REP).representation_index.representation_index_id
            ),
            evidence_contract=unsafe,
        )
