from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from policy_learnware_v0.hashing import canonicalize, sha256_json, sha256_ndarrays
from policy_learnware_v0.io import atomic_write_json
from policy_learnware_v0.representation.canonicalizer import PackedEpisodeDataset
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v02.audit import (
    PublicArtifactRule,
    artifact_tree_digest,
    audit_evidence_contract,
    audit_oracle_independence,
    audit_public_artifacts,
    audit_public_market_entries,
)
from policy_learnware_v0.v02.axes import DynamicsOperatorAudit
from policy_learnware_v0.v02.baselines import (
    CompetenceOnlySelector,
    EnvironmentOnlySelector,
    TargetQueryView,
    target_probe_evidence_contract,
)
from policy_learnware_v0.v02.competence import SourceEpisodeRow, championize_by_anchor
from policy_learnware_v0.v02.costs import (
    QUERY_COST_COMPONENTS,
    CostRecord,
    reconcile_cold_warm_costs,
)
from policy_learnware_v0.v02.environment_spec import (
    RepresentationIndex,
    RepresentationIndexEntry,
)
from policy_learnware_v0.v02.extensions.representation import (
    FrozenCorroEncoderAdapter,
    RawTransitionEncoder,
    default_metadata,
)
from policy_learnware_v0.v02.metrics import (
    HierarchicalValue,
    aggregate_hierarchy,
    compute_ranking_metrics,
    compute_selection_metrics,
)
from policy_learnware_v0.v02.recompute import (
    FORMAL_RECOMPUTE_DERIVATION_ID,
    FormalRecomputeProvenance,
    FormalRecomputeSourceManifest,
    FullCoverageContract,
    Gate0AuditUnit,
    IndependentRecomputeInputs,
    IndependentRecomputeReport,
    InformationAuditInputs,
    OracleMetricUnit,
    PairedComparisonPlan,
    PublishedSnapshot,
    RecomputeContractError,
    RepresentationReplayUnit,
    SelectorReplayUnit,
    SourceRecomputeInputs,
    StatisticsPlan,
    build_formal_recompute_section_source,
    championization_payload,
    encoded_cache_payload,
    formal_recompute_evaluator_digest,
    formal_recompute_loader_dependencies,
    formal_recompute_source_manifest_relative_path,
    load_formal_recompute_source_binding,
    missing_formal_recompute_source_loaders,
    run_independent_recompute,
    run_formal_independent_recompute,
    selector_unit_id,
    structurally_reconstructable_formal_recompute_sections,
    verify_structural_formal_recompute_source_sections,
)
from policy_learnware_v0.v02.gates import build_canonical_evidence_ref
from policy_learnware_v0.v02.oracle import (
    OracleEpisodeRow,
    PublishedSelection,
    aggregate_full_pool_oracle,
)
from policy_learnware_v0.v02.representation import (
    ProbeTraceView,
    RepresentationBuildContract,
    build_environment_spec,
)
from policy_learnware_v0.v02.schemas import (
    EnvironmentSpec,
    ExecutionABIRecord,
    PublicMarketEntry,
)
from policy_learnware_v0.v02.selectors import PublicMarketView
from policy_learnware_v0.v02.statistics import (
    bootstrap_max_t_intervals,
    centered_one_sided_p_value,
    derive_bootstrap_seed,
    hierarchical_bootstrap,
    hierarchical_paired_difference_bootstrap,
    holm_bonferroni,
)
from policy_learnware_v0.v02.variant_env import Gate0Audit, RolloutAudit


ID_A = "v02lw-" + "a" * 20
ID_B = "v02lw-" + "b" * 20
ANCHOR_A = "c" * 64
ANCHOR_B = "d" * 64
QUERY = "opaque-query-1"
REPRESENTATION = "2" * 64
MEASUREMENT = "3" * 64
VIEW = "4" * 64


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _spec(center: float) -> EnvironmentSpec:
    supports = np.asarray([[center], [center + 0.1]], dtype=np.float64)
    beta = np.asarray([0.5, 0.5], dtype=np.float64)
    gram = np.exp(-np.square(supports - supports.T) / 2.0)
    norm = float(beta @ gram @ beta)
    return EnvironmentSpec(
        supports=supports,
        beta=beta,
        empirical_norm2=norm,
        rkme_norm2=norm,
        reconstruction_error=0.0,
        reducer_digest=_d("reducer"),
        support_budget=2,
        latent_dim=1,
        representation_protocol_id=REPRESENTATION,
        measurement_protocol_id=MEASUREMENT,
        canonical_view_digest=VIEW,
        kernel_bandwidth=1.0,
        probe_dataset_digest=_d(f"probe-{center}"),
    )


def _abi(*, observation: str = "a") -> ExecutionABIRecord:
    return ExecutionABIRecord(
        protocol_family_id="continuous-vector-mdp-v02",
        observation_tensor_abi_digest=observation * 64,
        action_tensor_abi_digest="b" * 64,
        action_transform_id="tanh",
        policy_runtime_id="legacy-ppo-fpo-v02",
        state_abi_id="stateless",
    )


class _FakeCorro:
    def encode(self, packed: np.ndarray, *, batch_size: int) -> np.ndarray:
        assert batch_size > 0
        return np.asarray(packed[:, :1] * 0.5 + packed[:, 1:2], dtype=np.float32)


def _representation_units() -> tuple[RepresentationReplayUnit, ...]:
    dataset = PackedEpisodeDataset(
        packed=np.asarray(
            [[0.0, 1.0], [0.2, 0.8], [0.8, 0.2], [1.0, 0.0]],
            dtype=np.float32,
        ),
        episode_offsets=np.asarray([0, 2, 4], dtype=np.int64),
        reset_seeds=np.asarray([31, 32], dtype=np.int64),
        probe_seeds=np.asarray([41, 42], dtype=np.int64),
        task="private-task-not-serialized-publicly",
        schema_fingerprint="private-schema",
    )
    trace = ProbeTraceView(
        dataset=dataset,
        role="source_reference",
        context_id="source-context",
        bank_id="bank-0",
        seed_namespace="source-probe",
        probe_protocol_id=_d("probe-protocol"),
        measurement_protocol_id=MEASUREMENT,
        canonical_view_digest=VIEW,
        probe_rewards_included=False,
    )
    raw_metadata = default_metadata(
        representation_id="raw-transition-v02",
        family="raw",
        input_dim=2,
        output_dim=2,
        canonical_event_view_digest=VIEW,
    )
    corro_metadata = default_metadata(
        representation_id="corro-fake-v02",
        family="corro",
        input_dim=2,
        output_dim=1,
        canonical_event_view_digest=VIEW,
        checkpoint_digest=_d("corro-checkpoint"),
    )
    encoders = (
        ("raw/source-context/bank-0", RawTransitionEncoder(raw_metadata)),
        (
            "corro-fake/source-context/bank-0",
            FrozenCorroEncoderAdapter(_FakeCorro(), corro_metadata),
        ),
    )
    result = []
    for unit_id, encoder in encoders:
        contract = RepresentationBuildContract(
            representation_protocol_id=encoder.metadata.representation_protocol_id,
            measurement_protocol_id=MEASUREMENT,
            canonical_view_digest=VIEW,
            probe_rewards_included=False,
            kernel_bandwidth=1.0,
            batch_size=2,
            block_size=4,
            computation_backend="numpy",
            reducer_config=ReducerConfig(
                support_budget=2,
                support_steps=0,
                kmeans_steps=4,
                optimizer_backend="numpy",
            ),
        )
        placeholder = RepresentationReplayUnit(
            unit_id=unit_id,
            trace=trace,
            encoder=encoder,
            contract=contract,
            published_encoded_cache=PublishedSnapshot.create({"placeholder": True}),
            published_environment_spec=PublishedSnapshot.create({"placeholder": True}),
        )
        encoded = encoder.encode(dataset, batch_size=contract.batch_size)
        spec = build_environment_spec(trace, encoder, contract)
        result.append(
            replace(
                placeholder,
                published_encoded_cache=PublishedSnapshot.create(
                    encoded_cache_payload(placeholder, encoded)
                ),
                published_environment_spec=PublishedSnapshot.create(spec.to_dict()),
            )
        )
    return tuple(result)


def _source_rows() -> tuple[tuple[SourceEpisodeRow, ...], tuple[SourceEpisodeRow, ...]]:
    selection: list[SourceEpisodeRow] = []
    candidates = {
        ANCHOR_A: (("candidate-a1", (0.9, 0.8)), ("candidate-a2", (0.5, 0.4))),
        ANCHOR_B: (("candidate-b1", (0.7, 0.6)), ("candidate-b2", (0.3, 0.4))),
    }
    for anchor, rows in candidates.items():
        for candidate, values in rows:
            for seed, value in zip((1, 2), values, strict=True):
                selection.append(
                    SourceEpisodeRow(
                        anchor,
                        candidate,
                        _d(f"bundle-{candidate}"),
                        "source_selection",
                        seed,
                        value,
                    )
                )
    attestation = (
        SourceEpisodeRow(
            ANCHOR_A,
            "candidate-a1",
            _d("bundle-candidate-a1"),
            "source_attestation",
            10,
            0.9,
        ),
        SourceEpisodeRow(
            ANCHOR_A,
            "candidate-a1",
            _d("bundle-candidate-a1"),
            "source_attestation",
            11,
            0.8,
        ),
        SourceEpisodeRow(
            ANCHOR_B,
            "candidate-b1",
            _d("bundle-candidate-b1"),
            "source_attestation",
            10,
            0.7,
        ),
        SourceEpisodeRow(
            ANCHOR_B,
            "candidate-b1",
            _d("bundle-candidate-b1"),
            "source_attestation",
            11,
            0.6,
        ),
    )
    return tuple(selection), attestation


def _operator_gate() -> tuple[DynamicsOperatorAudit, Gate0Audit]:
    operator = DynamicsOperatorAudit(
        axis_id="joint-damping",
        operator_id="joint_damping_scale_v02",
        operator_version="v0",
        task_id="task-a",
        factor=2.0,
        base_model_digest=_d("base-model"),
        shifted_model_digest=_d("shifted-model"),
        changed_leaves=("dof_damping",),
        unchanged_leaves=("body_mass",),
        selected_element_count=2,
        changed_element_count=2,
        source_object_unchanged=True,
        exact_allowlist=True,
        coupling_check=True,
        finite=True,
        passed=True,
    )
    rollout = RolloutAudit(
        episode_count=2,
        steps_per_episode=4,
        all_finite=True,
        no_early_termination=True,
        paired_observation_identity=None,
        paired_reward_identity=None,
        paired_flag_identity=None,
        maximum_observation_absolute_error=None,
        maximum_reward_absolute_error=None,
        passed=True,
    )
    gate = Gate0Audit(
        environment_instance_digest=_d("environment-instance"),
        operator_audit_digest=operator.digest,
        schema_contract_identity=True,
        factor_role_valid=True,
        scalar_rollout=rollout,
        jit_finite=True,
        batched_rollout_finite=True,
        source_object_unchanged=True,
        exact_allowlist=True,
        coupled_physics=True,
        passed=True,
        reasons=(),
    )
    return operator, gate


def _oracle_payload(
    unit: OracleMetricUnit,
    selection,
    result,
):
    values = {
        policy_id: float(result.normalized_value_vector[policy_id])
        for policy_id in result.executable_ids
    }
    selection_metrics = compute_selection_metrics(
        selected_policy_id=selection.selected_id,
        normalized_returns_by_policy=values,
        executable_policy_ids=result.executable_ids,
        incompatible_failure_value=unit.failure_floor,
        epsilon=unit.epsilon,
        tie_tolerance=unit.tie_atol,
    )
    predicted = tuple(
        row.opaque_learnware_id
        for row in selection.ranking
        if row.opaque_learnware_id in set(result.executable_ids)
    )
    ranking_metrics = compute_ranking_metrics(
        predicted, values, tie_tolerance=unit.tie_atol
    )
    payload = {
        "schema": "policy-learnware.v02-oracle-query-metrics-recompute.v0",
        "unit_id": unit.unit_id,
        "query_id": unit.query_id,
        "method_id": unit.method_id,
        "task_id": unit.task_id,
        "axis_id": unit.axis_id,
        "context_id": unit.context_id,
        "full_anonymous_market_ids": list(result.market_ids),
        "full_pool_oracle_digest": result.digest,
        "full_pool_oracle": result.to_private_dict(),
        "selection_metrics": selection_metrics.to_dict(),
        "ranking_metrics": ranking_metrics.to_dict(),
        "ranking_scope": "executable_pool_only",
        "published_oracle_aggregate_consumed": False,
    }
    return payload, selection_metrics


def _bootstrap_payload(result) -> dict:
    return {
        **result.to_summary_dict(),
        "replicates_digest": sha256_ndarrays({"replicates": result.replicates}),
    }


def _statistics_payload(
    metric_units: dict[str, tuple[OracleMetricUnit, object]],
    *,
    resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
    comparisons: tuple[PairedComparisonPlan, ...],
) -> dict:
    methods = sorted({unit.method_id for unit, _ in metric_units.values()})
    method_payloads = {}
    leaves_by_method_endpoint = {}
    for method_id in methods:
        endpoint_payloads = {}
        for endpoint in ("selected_normalized_return", "pool_regret"):
            leaves = tuple(
                sorted(
                    (
                        HierarchicalValue(
                            unit.task_id,
                            unit.axis_id,
                            unit.context_id,
                            unit.query_id,
                            float(getattr(selection_metrics, endpoint)),
                        )
                        for unit, selection_metrics in metric_units.values()
                        if unit.method_id == method_id
                    ),
                    key=lambda row: row.key,
                )
            )
            leaves_by_method_endpoint[(method_id, endpoint)] = leaves
            seed = derive_bootstrap_seed(bootstrap_seed, method_id, endpoint)
            bootstrap = hierarchical_bootstrap(
                leaves,
                resamples=resamples,
                seed=seed,
                confidence_level=confidence_level,
            )
            endpoint_payloads[endpoint] = {
                "aggregate": aggregate_hierarchy(leaves).to_dict(),
                "bootstrap": _bootstrap_payload(bootstrap),
            }
        method_payloads[method_id] = endpoint_payloads
    comparison_payloads = {}
    p_families = {}
    bootstrap_families = {}
    for comparison in sorted(comparisons, key=lambda item: item.comparison_id):
        left = leaves_by_method_endpoint[
            (comparison.left_method_id, comparison.endpoint)
        ]
        right = leaves_by_method_endpoint[
            (comparison.right_method_id, comparison.endpoint)
        ]
        paired = hierarchical_paired_difference_bootstrap(
            left if comparison.endpoint == "selected_normalized_return" else right,
            right if comparison.endpoint == "selected_normalized_return" else left,
            resamples=resamples,
            seed=comparison.bootstrap_seed,
            confidence_level=confidence_level,
        )
        raw_p = centered_one_sided_p_value(
            paired.observed,
            paired.replicates,
            null_boundary=comparison.null_boundary,
        )
        comparison_payloads[comparison.comparison_id] = {
            "comparison_id": comparison.comparison_id,
            "left_method_id": comparison.left_method_id,
            "right_method_id": comparison.right_method_id,
            "endpoint": comparison.endpoint,
            "positive_direction": (
                "left_minus_right"
                if comparison.endpoint == "selected_normalized_return"
                else "right_minus_left"
            ),
            "null_boundary": comparison.null_boundary,
            "bootstrap": _bootstrap_payload(paired),
            "raw_p_value": raw_p,
            "holm_family_id": comparison.holm_family_id,
            "noninferiority": None,
        }
        if comparison.holm_family_id is not None:
            p_families.setdefault(comparison.holm_family_id, {})[
                comparison.comparison_id
            ] = raw_p
            bootstrap_families.setdefault(comparison.holm_family_id, {})[
                comparison.comparison_id
            ] = paired
    holm = {
        family_id: {
            hypothesis_id: result.to_dict()
            for hypothesis_id, result in sorted(holm_bonferroni(values).items())
        }
        for family_id, values in sorted(p_families.items())
    }
    simultaneous = {
        family_id: bootstrap_max_t_intervals(values).to_dict()
        for family_id, values in sorted(bootstrap_families.items())
    }
    return {
        "schema": "policy-learnware.v02-full-statistical-recompute.v0",
        "resamples": resamples,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "methods": method_payloads,
        "comparisons": comparison_payloads,
        "holm_families": holm,
        "simultaneous_max_t_families": simultaneous,
        "published_statistical_aggregate_consumed": False,
    }


def _components(scale: float) -> dict[str, float]:
    return {
        component: scale * (index + 1)
        for index, component in enumerate(QUERY_COST_COMPONENTS)
    }


def _cost_records() -> tuple[CostRecord, ...]:
    return tuple(
        CostRecord.create(
            query_id=QUERY,
            mode=mode,
            cost_contract_digest=_d("cost-contract"),
            execution_attempt_id=f"attempt-{mode}",
            components_seconds=_components(scale),
            target_environment_steps=64,
        )
        for mode, scale in (("cold", 0.02), ("warm", 0.01))
    )


def _cost_payload(records: tuple[CostRecord, ...]) -> dict:
    return {
        "schema": "policy-learnware.v02-cost-independent-recompute.v0",
        "raw_record_count": len(records),
        "raw_records_digest": sha256_json(
            [
                row.to_dict()
                for row in sorted(records, key=lambda row: (row.query_id, row.mode))
            ]
        ),
        "reconciliation": reconcile_cold_warm_costs(
            records, expected_query_ids=(QUERY,)
        ).to_dict(),
        "published_cost_aggregate_consumed": False,
    }


def _information_payload(
    *,
    market: PublicMarketView,
    selections: dict[str, object],
    public_root: Path,
    rules: tuple[PublicArtifactRule, ...],
    replay,
    measurement: Path,
    selector_outputs: Path,
) -> dict:
    projection = {
        opaque_id: {
            "opaque_learnware_id": entry.opaque_learnware_id,
            "normalized_source_competence": entry.normalized_source_competence,
            "tie_break_token": entry.tie_break_token,
        }
        for opaque_id, entry in sorted(market.entries.items())
    }
    market_audit = audit_public_market_entries(projection)
    artifact_audit = audit_public_artifacts(public_root, rules)
    evidence = {
        unit_id: audit_evidence_contract(record.evidence_contract)
        for unit_id, record in sorted(selections.items())
    }
    oracle = audit_oracle_independence(
        replay,
        market_public_root=public_root,
        measurement_root=measurement,
        selector_outputs_root=selector_outputs,
    )
    passed = (
        market_audit.passed
        and artifact_audit.passed
        and all(item.passed for item in evidence.values())
        and oracle.passed
    )
    assert passed
    return {
        "schema": "policy-learnware.v02-information-independent-recompute.v0",
        "passed": True,
        "public_market": market_audit.to_dict(),
        "public_artifacts": artifact_audit.to_dict(),
        "evidence_contracts": {
            unit_id: audit.to_dict() for unit_id, audit in evidence.items()
        },
        "oracle_independence": oracle.to_dict(),
        "selector_oracle_root_capability": False,
        "precomputed_audit_pass_fields_consumed": False,
    }


def _inputs(tmp_path: Path) -> IndependentRecomputeInputs:
    selection_rows, attestation_rows = _source_rows()
    floors = {ANCHOR_A: 0.5, ANCHOR_B: 0.5}
    champion = championize_by_anchor(
        selection_rows,
        attestation_rows,
        competence_floors=floors,
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id=_d("dmc-normalized-return-v0"),
    )
    source_payload = championization_payload(
        champion,
        selection_rows=selection_rows,
        attestation_rows=attestation_rows,
    )
    source = SourceRecomputeInputs(
        selection_rows=selection_rows,
        attestation_rows=attestation_rows,
        competence_floors=floors,
        mean_tolerance=0.0,
        lcb_z=None,
        return_contract_id=_d("dmc-normalized-return-v0"),
        published=PublishedSnapshot.create(source_payload),
    )

    market_id = "market-v02"
    entries = {
        ID_A: PublicMarketEntry(
            ID_A,
            champion.competence_records[ANCHOR_A].normalized_competence,
            _d("tie-a"),
        ),
        ID_B: PublicMarketEntry(
            ID_B,
            champion.competence_records[ANCHOR_B].normalized_competence,
            _d("tie-b"),
        ),
    }
    index = RepresentationIndex(
        policy_market_id=market_id,
        representation_protocol_id=REPRESENTATION,
        entries={
            ID_A: RepresentationIndexEntry(ID_A, _spec(0.0)),
            ID_B: RepresentationIndexEntry(ID_B, _spec(1.0)),
        },
    )
    market = PublicMarketView(market_id, entries, index)

    operator, gate = _operator_gate()
    gate_unit = Gate0AuditUnit(
        audit_id="gate0-task-axis-factor",
        operator_audit=operator,
        gate0_audit=gate,
        published_operator=PublishedSnapshot.create(operator.to_dict()),
        published_gate0=PublishedSnapshot.create(gate.to_dict()),
    )

    query = TargetQueryView(
        stage="development_discovery",
        query_spec=_spec(0.9),
        target_evidence_digest=_d("target-evidence"),
        cost_digest=_d("query-cost"),
        probe_rewards_included=False,
    )
    competence_selector = CompetenceOnlySelector(
        method_id="B1", policy_market_id=market.policy_market_id
    )
    environment_selector = EnvironmentOnlySelector(
        method_id="A-Env",
        distance_form="mmd",
        policy_market_id=market.policy_market_id,
        representation_index_id=str(index.representation_index_id),
        evidence_contract=target_probe_evidence_contract(
            reads_development_policy_returns=False,
            reads_probe_rewards=False,
        ),
    )
    selector_units = []
    selections = {}
    for selector in (competence_selector, environment_selector):
        artifact = selector.fit(None)
        record = selector.select(query, market, artifact)
        unit = SelectorReplayUnit(
            query_id=QUERY,
            selector=selector,
            query=query,
            artifact=artifact,
            published=PublishedSnapshot.create(record.to_dict()),
        )
        selector_units.append(unit)
        selections[unit.unit_id] = record

    oracle_rows = tuple(
        OracleEpisodeRow(
            opaque_query_id=QUERY,
            opaque_learnware_id=policy_id,
            episode_index=episode,
            reset_seed=100 + episode,
            policy_seed=200 + episode,
            steps=1000,
            raw_return=value * 1000.0,
            normalized_return=value,
            terminated=False,
            truncated=True,
            runtime_seconds=0.1,
            private_target_instance_digest=_d("target-instance"),
            bundle_digest=_d(f"oracle-bundle-{policy_id}"),
            seed_contract_digest=_d("oracle-seed-contract"),
            evaluation_protocol_id=_d("evaluation-protocol"),
        )
        for policy_id, values in ((ID_A, (0.4, 0.5)), (ID_B, (0.9, 0.8)))
        for episode, value in enumerate(values)
    )
    registry = {ID_A: _abi(), ID_B: _abi()}
    oracle_units = []
    for selector_unit in selector_units:
        placeholder = OracleMetricUnit(
            query_id=QUERY,
            task_id="task-a",
            axis_id="joint-damping",
            context_id="context-1",
            method_id=selector_unit.method_id,
            market_ids=(ID_A, ID_B),
            deployment_registry=registry,
            target_execution_abi=_abi(),
            private_target_instance_digest=_d("target-instance"),
            evaluation_protocol_id=_d("evaluation-protocol"),
            failure_floor=0.0,
            epsilon=0.05,
            tie_atol=0.0,
            candidate_paired_seeds=True,
            published=PublishedSnapshot.create({"placeholder": True}),
        )
        oracle_units.append(placeholder)
    oracle_result = aggregate_full_pool_oracle(
        opaque_query_id=QUERY,
        private_target_instance_digest=_d("target-instance"),
        evaluation_protocol_id=_d("evaluation-protocol"),
        market_ids=(ID_A, ID_B),
        deployment_registry=registry,
        target_execution_abi=_abi(),
        episode_rows=oracle_rows,
        published_selections=tuple(
            PublishedSelection.from_selection_record(record)
            for record in selections.values()
        ),
        failure_floor=0.0,
        tie_atol=0.0,
        candidate_paired_seeds=True,
    )
    finalized_units = []
    metric_units = {}
    for placeholder in oracle_units:
        payload, metrics = _oracle_payload(
            placeholder, selections[placeholder.unit_id], oracle_result
        )
        unit = replace(placeholder, published=PublishedSnapshot.create(payload))
        finalized_units.append(unit)
        metric_units[unit.unit_id] = (unit, metrics)
    oracle_units = finalized_units

    comparisons = (
        PairedComparisonPlan(
            comparison_id="aenv-vs-b1-return",
            left_method_id="A-Env",
            right_method_id="B1",
            endpoint="selected_normalized_return",
            bootstrap_seed=23,
            holm_family_id="primary",
        ),
        PairedComparisonPlan(
            comparison_id="aenv-vs-b1-regret",
            left_method_id="A-Env",
            right_method_id="B1",
            endpoint="pool_regret",
            bootstrap_seed=23,
            holm_family_id="primary",
        ),
    )
    statistics_payload = _statistics_payload(
        metric_units,
        resamples=32,
        bootstrap_seed=17,
        confidence_level=0.95,
        comparisons=comparisons,
    )
    statistics = StatisticsPlan(
        resamples=32,
        bootstrap_seed=17,
        confidence_level=0.95,
        comparisons=comparisons,
        published=PublishedSnapshot.create(statistics_payload),
    )
    costs = _cost_records()

    public_root = tmp_path / "market_public"
    measurement = tmp_path / "measurement"
    selector_outputs = tmp_path / "selector_outputs"
    for root in (public_root, measurement, selector_outputs):
        root.mkdir(parents=True)
    projection = {
        opaque_id: {
            "opaque_learnware_id": entry.opaque_learnware_id,
            "normalized_source_competence": entry.normalized_source_competence,
            "tie_break_token": entry.tie_break_token,
        }
        for opaque_id, entry in sorted(market.entries.items())
    }
    (public_root / "manifest.json").write_text(
        json.dumps({"entries": projection}, sort_keys=True), encoding="utf-8"
    )
    (measurement / "query.bin").write_bytes(b"candidate-independent-query")
    (selector_outputs / "selection.bin").write_bytes(b"immutable-selection")
    rules = (
        PublicArtifactRule(
            pattern="manifest.json",
            kind="json",
            json_keys=frozenset({"entries"}),
        ),
    )

    def replay(market_root: Path, measurement_root: Path, output_root: Path) -> str:
        assert market_root.is_dir() and measurement_root.is_dir()
        return artifact_tree_digest(output_root)

    information_payload = _information_payload(
        market=market,
        selections=selections,
        public_root=public_root,
        rules=rules,
        replay=replay,
        measurement=measurement,
        selector_outputs=selector_outputs,
    )
    information = InformationAuditInputs(
        public_artifact_root=public_root,
        public_artifact_rules=rules,
        replay_selector=replay,
        market_public_root=public_root,
        measurement_root=measurement,
        selector_outputs_root=selector_outputs,
        published=PublishedSnapshot.create(information_payload),
    )

    representation_units = _representation_units()
    unit_ids = tuple(unit.unit_id for unit in selector_units)
    coverage = FullCoverageContract(
        source_anchor_ids=(ANCHOR_A, ANCHOR_B),
        source_market_bindings={ANCHOR_A: ID_A, ANCHOR_B: ID_B},
        public_market_ids=(ID_A, ID_B),
        gate0_audit_ids=(gate_unit.audit_id,),
        representation_unit_ids=tuple(unit.unit_id for unit in representation_units),
        selector_unit_ids=unit_ids,
        oracle_query_ids=(QUERY,),
        oracle_unit_ids=unit_ids,
        cost_query_ids=(QUERY,),
    )
    return IndependentRecomputeInputs(
        coverage=coverage,
        source=source,
        market=market,
        gate0_units=(gate_unit,),
        representation_units=representation_units,
        selector_units=tuple(selector_units),
        oracle_rows=oracle_rows,
        oracle_units=tuple(oracle_units),
        statistics=statistics,
        cost_records=costs,
        published_costs=PublishedSnapshot.create(_cost_payload(costs)),
        information_audit=information,
    )


def test_full_independent_recompute_and_strict_report_round_trip(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    report = run_independent_recompute(inputs)
    report.require_passed()
    assert report.passed
    assert report.full_digest_coverage
    assert report.full_selector_replay
    assert report.full_statistical_recompute
    assert report.raw_numeric_subset_coverage
    assert report.cost_recompute
    assert report.information_isolation
    assert set(report.section_digests) == {
        "source",
        "gate0",
        "representations",
        "selectors",
        "oracle",
        "statistics",
        "costs",
        "information",
    }
    simultaneous = inputs.statistics.published.payload["simultaneous_max_t_families"]
    assert set(simultaneous) == {"primary"}
    assert set(simultaneous["primary"]["intervals"]) == {
        "aenv-vs-b1-regret",
        "aenv-vs-b1-return",
    }
    loaded = IndependentRecomputeReport.from_dict(report.to_dict())
    assert loaded.digest == report.digest

    forged = {**report.to_dict(), "passed": False}
    with pytest.raises(RecomputeContractError, match="disagrees"):
        IndependentRecomputeReport.from_dict(forged)
    with pytest.raises(RecomputeContractError, match="unknown"):
        IndependentRecomputeReport.from_dict({**report.to_dict(), "aggregate": {}})
    with pytest.raises(RecomputeContractError, match="section digest coverage"):
        IndependentRecomputeReport.from_dict(
            {**report.to_dict(), "section_digests": {}}
        )


def _formal_source_binding(
    tmp_path: Path,
    inputs: IndependentRecomputeInputs,
    *,
    registered_sources: bool = False,
):
    experiment_id = "formal-recompute-r0"
    config_digest = "a" * 64
    config_file_sha256 = "b" * 64
    root = tmp_path / "formal_artifacts" / experiment_id
    root.mkdir(parents=True)
    section_sources = {}
    for section in sorted(
        {
            "source",
            "gate0",
            "representations",
            "selectors",
            "oracle",
            "statistics",
            "costs",
            "information",
        }
    ):
        path = root / "analysis" / "raw_sources" / f"{section}.json"
        if registered_sources and section in {
            "source",
            "gate0",
            "oracle",
            "statistics",
            "costs",
        }:
            payload = build_formal_recompute_section_source(
                section, inputs, config_digest=config_digest
            )
        else:
            payload = {
                "schema": f"policy-learnware.v02-test-{section}-raw-source.v0",
                "config_digest": config_digest,
                "section": section,
            }
        atomic_write_json(path, payload)
        section_sources[section] = (
            build_canonical_evidence_ref(
                path,
                experiment_root=root,
                expected_config_digest=config_digest,
            ),
        )
    manifest = FormalRecomputeSourceManifest(
        experiment_id=experiment_id,
        config_digest=config_digest,
        config_file_sha256=config_file_sha256,
        coverage_contract_digest=inputs.coverage.digest,
        section_sources=section_sources,
    )
    manifest_path = root / formal_recompute_source_manifest_relative_path()
    atomic_write_json(manifest_path, manifest.to_dict())
    binding = load_formal_recompute_source_binding(
        manifest_path,
        experiment_root=root,
        expected_experiment_id=experiment_id,
        expected_config_digest=config_digest,
        expected_config_file_sha256=config_file_sha256,
        expected_coverage_contract_digest=inputs.coverage.digest,
    )
    return binding


def test_formal_recompute_cannot_mint_authority_from_decoy_refs_and_typed_inputs(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    binding = _formal_source_binding(tmp_path, inputs)
    assert set(missing_formal_recompute_source_loaders()) == {
        "source",
        "gate0",
        "representations",
        "selectors",
        "oracle",
        "statistics",
        "costs",
        "information",
    }
    assert set(formal_recompute_loader_dependencies()) == set(
        missing_formal_recompute_source_loaders()
    )
    assert set(structurally_reconstructable_formal_recompute_sections()) == {
        "source",
        "gate0",
        "oracle",
        "statistics",
        "costs",
    }
    with pytest.raises(RecomputeContractError, match="source-owned loaders"):
        run_formal_independent_recompute(inputs, source_binding=binding)


def test_registered_formal_source_loaders_exactly_bind_live_typed_inputs(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    binding = _formal_source_binding(tmp_path, inputs, registered_sources=True)
    projections = verify_structural_formal_recompute_source_sections(
        inputs, source_binding=binding
    )
    assert set(projections) == {"source", "gate0", "oracle", "statistics", "costs"}

    poisoned = replace(
        inputs,
        source=replace(inputs.source, mean_tolerance=inputs.source.mean_tolerance + 0.01),
    )
    with pytest.raises(RecomputeContractError, match="differs from live typed inputs"):
        verify_structural_formal_recompute_source_sections(
            poisoned, source_binding=binding
        )

    with pytest.raises(RecomputeContractError, match="dependencies=.*representations"):
        run_formal_independent_recompute(inputs, source_binding=binding)


def test_handwritten_formal_report_provenance_cannot_restore_execution_authority(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    binding = _formal_source_binding(tmp_path, inputs)
    base = run_independent_recompute(inputs)
    base.require_passed()
    provenance = FormalRecomputeProvenance(
        experiment_id=binding.manifest.experiment_id,
        config_digest=binding.manifest.config_digest,
        config_file_sha256=binding.manifest.config_file_sha256,
        coverage_contract_digest=inputs.coverage.digest,
        source_manifest_ref=binding.manifest_ref,
        source_manifest_digest=binding.manifest.digest,
        section_source_digests=binding.verify_sources(),
        derivation_id=FORMAL_RECOMPUTE_DERIVATION_ID,
        evaluator_digest=formal_recompute_evaluator_digest(),
    )
    handwritten = IndependentRecomputeReport(
        coverage_contract_digest=base.coverage_contract_digest,
        checks=base.checks,
        section_digests=base.section_digests,
        errors=base.errors,
        formal_provenance=provenance,
    )
    archived = IndependentRecomputeReport.from_dict(handwritten.to_dict())
    assert archived.passed
    assert not archived.is_formally_authoritative
    with pytest.raises(RecomputeContractError, match="in-process execution authority"):
        archived.require_formal_authority(
            expected_experiment_id=binding.manifest.experiment_id,
            expected_config_digest=binding.manifest.config_digest,
            expected_config_file_sha256=binding.manifest.config_file_sha256,
        )


def test_source_seed_tamper_fails_raw_coverage_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    poisoned_row = replace(inputs.source.attestation_rows[0], reset_seed=1)
    poisoned_source = replace(
        inputs.source,
        attestation_rows=(poisoned_row, *inputs.source.attestation_rows[1:]),
    )
    report = run_independent_recompute(replace(inputs, source=poisoned_source))
    assert not report.passed
    assert not report.raw_numeric_subset_coverage
    assert any("seeds overlap" in error for error in report.errors)


def test_self_consistent_published_selector_tamper_still_fails_replay(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    original = inputs.selector_units[0]
    payload = canonicalize(original.published.payload)
    payload["ranking"][0]["log_score"] += 0.125
    poisoned = replace(original, published=PublishedSnapshot.create(payload))
    report = run_independent_recompute(
        replace(inputs, selector_units=(poisoned, *inputs.selector_units[1:]))
    )
    assert not report.full_selector_replay
    assert not report.full_digest_coverage
    assert any("numeric mismatch" in error for error in report.errors)


def test_oracle_numeric_tamper_fails_metrics_and_statistics(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    poisoned = replace(
        inputs.oracle_rows[0],
        normalized_return=inputs.oracle_rows[0].normalized_return + 0.05,
    )
    report = run_independent_recompute(
        replace(inputs, oracle_rows=(poisoned, *inputs.oracle_rows[1:]))
    )
    assert report.full_selector_replay
    assert not report.full_statistical_recompute
    assert any(
        "oracle_metrics" in error and "mismatch" in error for error in report.errors
    )
    with pytest.raises(RecomputeContractError, match="failed closed"):
        report.require_passed()


def test_cost_and_gate_summary_tamper_each_fail_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    cold = inputs.cost_records[0]
    changed_components = dict(cold.components_seconds)
    changed_components[QUERY_COST_COMPONENTS[0]] += 0.01
    changed_cold = CostRecord.create(
        query_id=cold.query_id,
        mode=cold.mode,
        cost_contract_digest=cold.cost_contract_digest,
        execution_attempt_id=cold.execution_attempt_id,
        components_seconds=changed_components,
        target_environment_steps=cold.target_environment_steps,
    )
    cost_report = run_independent_recompute(
        replace(inputs, cost_records=(changed_cold, inputs.cost_records[1]))
    )
    assert not cost_report.cost_recompute
    assert any("costs" in error and "mismatch" in error for error in cost_report.errors)

    gate_unit = inputs.gate0_units[0]
    forged_gate = replace(gate_unit.gate0_audit, passed=False, reasons=())
    gate_report = run_independent_recompute(
        replace(inputs, gate0_units=(replace(gate_unit, gate0_audit=forged_gate),))
    )
    assert not gate_report.raw_numeric_subset_coverage
    assert any("passed/reasons" in error for error in gate_report.errors)


def test_representation_raw_cache_and_spec_tamper_fail_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    original = inputs.representation_units[0]

    packed = np.array(original.trace.dataset.packed, copy=True)
    packed[0, 0] += 0.25
    changed_dataset = PackedEpisodeDataset(
        packed=packed,
        episode_offsets=original.trace.dataset.episode_offsets,
        reset_seeds=original.trace.dataset.reset_seeds,
        probe_seeds=original.trace.dataset.probe_seeds,
        task=original.trace.dataset.task,
        schema_fingerprint=original.trace.dataset.schema_fingerprint,
    )
    changed_trace = ProbeTraceView(
        dataset=changed_dataset,
        role=original.trace.role,
        context_id=original.trace.context_id,
        bank_id=original.trace.bank_id,
        seed_namespace=original.trace.seed_namespace,
        probe_protocol_id=original.trace.probe_protocol_id,
        measurement_protocol_id=original.trace.measurement_protocol_id,
        canonical_view_digest=original.trace.canonical_view_digest,
        probe_rewards_included=original.trace.probe_rewards_included,
    )
    raw_report = run_independent_recompute(
        replace(
            inputs,
            representation_units=(
                replace(original, trace=changed_trace),
                *inputs.representation_units[1:],
            ),
        )
    )
    assert not raw_report.raw_numeric_subset_coverage
    assert any(
        "encoded_cache" in error and "mismatch" in error for error in raw_report.errors
    )

    cache_payload = canonicalize(original.published_encoded_cache.payload)
    cache_payload["latent_dim"] += 1
    cache_report = run_independent_recompute(
        replace(
            inputs,
            representation_units=(
                replace(
                    original,
                    published_encoded_cache=PublishedSnapshot.create(cache_payload),
                ),
                *inputs.representation_units[1:],
            ),
        )
    )
    assert not cache_report.raw_numeric_subset_coverage
    assert any("latent_dim numeric mismatch" in error for error in cache_report.errors)

    spec_payload = canonicalize(original.published_environment_spec.payload)
    spec_payload["reconstruction_error"] += 0.01
    spec_report = run_independent_recompute(
        replace(
            inputs,
            representation_units=(
                replace(
                    original,
                    published_environment_spec=PublishedSnapshot.create(spec_payload),
                ),
                *inputs.representation_units[1:],
            ),
        )
    )
    assert not spec_report.raw_numeric_subset_coverage
    assert any(
        "reconstruction_error numeric mismatch" in error for error in spec_report.errors
    )


def test_representation_coverage_omission_and_max_t_seed_drift_are_rejected(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    omitted = replace(
        inputs.coverage,
        representation_unit_ids=(inputs.representation_units[0].unit_id,),
    )
    report = run_independent_recompute(replace(inputs, coverage=omitted))
    assert not report.raw_numeric_subset_coverage
    assert any("differ from frozen coverage" in error for error in report.errors)

    comparisons = inputs.statistics.comparisons
    with pytest.raises(RecomputeContractError, match="share a paired bootstrap seed"):
        replace(
            inputs.statistics,
            comparisons=(comparisons[0], replace(comparisons[1], bootstrap_seed=24)),
        )


def test_coverage_contract_rejects_missing_selector_oracle_pair() -> None:
    with pytest.raises(RecomputeContractError, match="one oracle unit"):
        FullCoverageContract(
            source_anchor_ids=(ANCHOR_A,),
            source_market_bindings={ANCHOR_A: ID_A},
            public_market_ids=(ID_A,),
            gate0_audit_ids=("gate",),
            representation_unit_ids=("raw/unit",),
            selector_unit_ids=(selector_unit_id("B1", QUERY),),
            oracle_query_ids=(QUERY,),
            oracle_unit_ids=(selector_unit_id("A-Env", QUERY),),
            cost_query_ids=(QUERY,),
        )
