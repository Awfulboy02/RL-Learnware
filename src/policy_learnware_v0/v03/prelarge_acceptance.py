"""Deterministic CPU acceptance at the boundary before large v0.3 runs."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..hashing import canonical_json, sha256_json, sha256_ndarrays
from ..rkme.reducer import ReducerConfig
from .baselines import OPTIONAL_BASELINE_STATES, REQUIRED_BASELINE_METHOD_IDS
from .canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
    require_formal_cross_task_raw_receipts,
)
from .data_roles import DataRoleManifest, DataRoleRecord
from .representation_ladder import (
    R3_MATCHED_RANDOM_MLP,
    R_HIST_RANDOM_TANH,
    RepresentationBatch,
    TrainedCallableArtifact,
    bind_r4_frozen_callable,
    fit_r0_identity,
    fit_r1_random_linear,
    fit_r2_pca_whitening,
    fit_r3_matched_random_mlp,
    fit_r5_corro_style,
    fit_r5l_supervised_linear,
)
from .condition_plan import ConditionExecutionPlan
from .dynamics_axis import DynamicsAxisEntry, DynamicsAxisRegistry
from .representation_plan import RepresentationExecutionPlan
from .signal_controls import (
    ExactRepeatDistanceResult,
    ExactRepeatPairContract,
    HistoricalRandomTanhSpec,
    PairControlEvaluation,
    PairControlMembershipEvidence,
    RewardFreeShuffledNextSpec,
    SchemaCollisionPairContract,
    exact_repeat_noise_ratio,
)
from .signal_atlas import DEVELOPMENT_SMOKE_MODE, SignalCellWorkItem
from .signal_matrix import build_optimization_fit_jobs, build_signal_matrix_plan
from .signal_prefix import (
    FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS,
    SignalPrefixSchedule,
)
from .signal_runtime import (
    SignalBankIdentity,
    SignalExecutionProtocol,
    SignalIdentityRegistry,
    bank_control_reference_from_feature_bank,
    feature_bank_from_transition_view,
    transform_feature_banks,
    validate_pair_control_feature_banks,
)
from .source_fit import build_formal_source_fit_batch
from .transition_views import (
    V_FULL_LEGACY,
    V_REWARD_FREE_TRANSITION,
    TransitionBank,
    apply_transition_view,
)


PRELARGE_ACCEPTANCE_SCHEMA = "policy-learnware.v03-prelarge-acceptance.v0"


class PrelargeAcceptanceError(ValueError):
    """The deterministic pre-large acceptance fixture failed."""


def _d(label: str) -> str:
    return sha256_json(
        {"schema": "policy-learnware.v03-prelarge-fixture-domain.v0", "label": label}
    )


def _native(
    bank_id: str,
    task_id: str,
    role: str,
    *,
    observation_dim: int,
    action_dim: int,
    center: float,
) -> NativeTransitionBank:
    observation = center + np.arange(
        4 * observation_dim, dtype=np.float64
    ).reshape(4, observation_dim) / 100.0
    action = center / 10.0 + np.arange(
        4 * action_dim, dtype=np.float64
    ).reshape(4, action_dim) / 100.0
    return NativeTransitionBank(
        bank_id=bank_id,
        task_private_id=task_id,
        data_role=role,  # type: ignore[arg-type]
        native_schema_digest=_d(f"schema:{task_id}"),
        raw_dataset_digest=_d(f"raw:{bank_id}"),
        observation=observation,
        action=action,
        reward=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64) + center / 100.0,
        next_observation=observation + 0.05,
        terminated=np.asarray([False, True, False, True]),
        truncated=np.asarray([False, False, False, False]),
        episode_id=np.asarray([0, 0, 1, 1]),
        timestep=np.asarray([0, 1, 0, 1]),
    )


def _fake_trainer(values: np.ndarray, labels: np.ndarray, request: Any) -> TrainedCallableArtifact:
    # The callable is deliberately tiny; this smoke verifies the injected
    # source-only contract, not the scientific quality of an R5 optimization.
    del labels
    rng = np.random.default_rng(request.seed + 100)
    matrix = rng.normal(
        0.0,
        1.0 / np.sqrt(request.input_dim),
        size=(request.input_dim, request.output_dim),
    ).astype(np.float64)

    def transform(input_values: np.ndarray) -> np.ndarray:
        projected = np.asarray(input_values, dtype=np.float64) @ matrix
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / np.maximum(norms, np.finfo(np.float64).eps)

    parameter_digest = sha256_ndarrays({"matrix": matrix})
    return TrainedCallableArtifact(
        checkpoint_bytes=("prelarge:" + request.request_digest).encode("ascii"),
        parameter_digest=parameter_digest,
        trainer_implementation_digest=_d("fake-source-only-trainer"),
        transform=transform,
    )


@dataclass(frozen=True)
class PrelargeAcceptanceReport:
    checks: Mapping[str, bool]
    evidence_digests: Mapping[str, str]
    signal_matrix_logical_cells: int
    signal_matrix_numeric_cells: int
    optimization_fit_jobs: int
    required_baseline_methods: tuple[str, ...]
    formal_run_authorized: bool = False
    large_experiment_executed: bool = False
    v04_assets_required: bool = False
    report_digest: str | None = None
    schema: str = PRELARGE_ACCEPTANCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRELARGE_ACCEPTANCE_SCHEMA:
            raise PrelargeAcceptanceError("unsupported prelarge acceptance schema")
        checks = dict(sorted(self.checks.items()))
        if not checks or any(type(value) is not bool for value in checks.values()):
            raise PrelargeAcceptanceError("prelarge checks must be non-empty booleans")
        evidence = dict(sorted(self.evidence_digests.items()))
        if not evidence or any(
            not isinstance(value, str) or len(value) != 64 for value in evidence.values()
        ):
            raise PrelargeAcceptanceError("prelarge evidence must contain SHA-256 digests")
        if (
            self.signal_matrix_logical_cells != 39
            or self.signal_matrix_numeric_cells != 37
            or self.optimization_fit_jobs != 45
        ):
            raise PrelargeAcceptanceError("prelarge matrix/job cardinality drifted")
        if tuple(self.required_baseline_methods) != REQUIRED_BASELINE_METHOD_IDS:
            raise PrelargeAcceptanceError("prelarge baseline registry drifted")
        if any(
            value is not False
            for value in (
                self.formal_run_authorized,
                self.large_experiment_executed,
                self.v04_assets_required,
            )
        ):
            raise PrelargeAcceptanceError(
                "engineering acceptance cannot grant formal authority or run large work"
            )
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "evidence_digests", MappingProxyType(evidence))
        expected = sha256_json(self._payload_without_digest())
        if self.report_digest is None:
            object.__setattr__(self, "report_digest", expected)
        elif self.report_digest != expected:
            raise PrelargeAcceptanceError("prelarge report digest mismatch")

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def status(self) -> str:
        return (
            "ENGINEERING_COMPONENTS_PASS_FORMAL_FREEZE_PENDING"
            if self.passed
            else "BLOCKED_ENGINEERING"
        )

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checks": dict(self.checks),
            "passed": self.passed,
            "status": self.status,
            "evidence_digests": dict(self.evidence_digests),
            "signal_matrix_logical_cells": self.signal_matrix_logical_cells,
            "signal_matrix_numeric_cells": self.signal_matrix_numeric_cells,
            "optimization_fit_jobs": self.optimization_fit_jobs,
            "required_baseline_methods": list(self.required_baseline_methods),
            "formal_run_authorized": self.formal_run_authorized,
            "large_experiment_executed": self.large_experiment_executed,
            "v04_assets_required": self.v04_assets_required,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "report_digest": self.report_digest}


def run_prelarge_acceptance() -> PrelargeAcceptanceReport:
    """Exercise every hard P4 boundary and the numeric cell runner on CPU."""

    fit_a = _native(
        "fit-a",
        "task-a",
        "source_representation_train",
        observation_dim=2,
        action_dim=1,
        center=0.0,
    )
    fit_b = _native(
        "fit-b",
        "task-b",
        "source_representation_train",
        observation_dim=3,
        action_dim=2,
        center=10.0,
    )
    validation_a = _native(
        "validation-a",
        "task-a",
        "source_representation_validation",
        observation_dim=2,
        action_dim=1,
        center=0.03,
    )
    validation_b = _native(
        "validation-b",
        "task-b",
        "source_representation_validation",
        observation_dim=3,
        action_dim=2,
        center=10.03,
    )
    fit_native = (fit_a, fit_b, validation_a, validation_b)
    registry = NativeShapeRegistry.from_source_banks(fit_native)
    normalizer = fit_global_normalizer(fit_native, registry=registry)
    canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
    native = (
        _native(
            "source-a",
            "task-a",
            "source_reference_spec",
            observation_dim=2,
            action_dim=1,
            center=0.1,
        ),
        _native(
            "source-b",
            "task-b",
            "source_reference_spec",
            observation_dim=3,
            action_dim=2,
            center=10.1,
        ),
        _native(
            "source-a-dynamics",
            "task-a",
            "source_reference_spec",
            observation_dim=2,
            action_dim=1,
            center=1.1,
        ),
        _native(
            "query-a",
            "task-a",
            "development_query",
            observation_dim=2,
            action_dim=1,
            center=0.12,
        ),
        _native(
            "query-b",
            "task-b",
            "development_query",
            observation_dim=3,
            action_dim=2,
            center=10.12,
        ),
        _native(
            "query-a-goal2",
            "task-a",
            "development_query",
            observation_dim=2,
            action_dim=1,
            center=0.14,
        ),
    )
    fit_receipts = tuple(canonicalizer.transform(item) for item in fit_native)
    receipts = tuple(canonicalizer.transform(item) for item in native)
    require_formal_cross_task_raw_receipts(receipts)
    measurement_protocol = _d("prelarge-measurement-protocol")
    all_receipts = (*fit_receipts, *receipts)
    def identity_for_receipt(receipt: Any) -> SignalBankIdentity:
        goal = f"goal-{receipt.task_private_id}"
        dynamics = f"dynamics-{receipt.task_private_id}"
        context = f"context-{receipt.task_private_id}"
        if receipt.bank_id == "source-a-dynamics":
            dynamics = "dynamics-task-a-high"
            context = "context-task-a-high"
        elif receipt.bank_id == "query-a-goal2":
            goal = "goal-task-a-alt"
            context = "context-task-a-alt-goal"
        return SignalBankIdentity.from_receipt(
            receipt,
            embodiment_id=f"embodiment-{receipt.task_private_id}",
            abi_contract_id=f"abi-{receipt.task_private_id}",
            goal_contract_id=goal,
            dynamics_context_id=dynamics,
            context_id=context,
            measurement_protocol_digest=measurement_protocol,
            probe_seed_digest=_d(f"probe:{receipt.bank_id}"),
            equivalence_class_id=f"equivalence-{receipt.task_private_id}-{goal}-{dynamics}",
        )

    all_identities = tuple(identity_for_receipt(receipt) for receipt in all_receipts)
    fit_identities = all_identities[: len(fit_receipts)]
    identities = all_identities[len(fit_receipts) :]

    transition_banks = tuple(
        TransitionBank.from_canonical_batch(item.batch) for item in receipts
    )
    rf_views = tuple(
        apply_transition_view(item, V_REWARD_FREE_TRANSITION)
        for item in transition_banks
    )
    features = tuple(
        feature_bank_from_transition_view(receipt, identity, view)
        for receipt, identity, view in zip(
            receipts, identities, rf_views, strict=True
        )
    )
    fit_transition_banks = tuple(
        TransitionBank.from_canonical_batch(item.batch) for item in fit_receipts
    )
    fit_features = tuple(
        feature_bank_from_transition_view(
            receipt,
            identity,
            apply_transition_view(bank, V_REWARD_FREE_TRANSITION),
        )
        for receipt, identity, bank in zip(
            fit_receipts, fit_identities, fit_transition_banks, strict=True
        )
    )
    full = apply_transition_view(transition_banks[0], V_FULL_LEGACY)
    historical = HistoricalRandomTanhSpec.create(
        seed=11, input_dim=full.feature_matrix.shape[1], output_dim=5
    )
    condition_plan = ConditionExecutionPlan.create(historical_spec=historical)
    split_nonce = _d("prelarge-source-split")
    role_manifest = DataRoleManifest(
        manifest_id="prelarge-source-fit",
        records=tuple(
            DataRoleRecord(
                role=bank.data_role,
                dataset_id=f"role-{bank.bank_id}",
                dataset_digest=bank.raw_dataset_digest,
                task_private_ids=(bank.task_private_id,),
                seed_tokens=(f"seed-{bank.bank_id}",),
                split_nonce_digest=split_nonce,
            )
            for bank in fit_native
        ),
    )
    formal_source_fit = build_formal_source_fit_batch(
        role_manifest,
        train_feature_banks=fit_features[:2],
        validation_feature_banks=fit_features[2:],
        condition_plan=condition_plan,
    )

    rf_control = RewardFreeShuffledNextSpec(seed=7).apply(transition_banks[0])
    historical_result = historical.apply(transition_banks[0])

    source_batch = formal_source_fit.training_batch
    query_batch = RepresentationBatch(
        values=features[2].values,
        dataset_digest=_d("representation-query-transform"),
        role="QUERY_TRANSFORM",
    )
    r0 = fit_r0_identity(source_batch)
    r1 = fit_r1_random_linear(source_batch, output_dim=4, seed=1)
    r2 = fit_r2_pca_whitening(source_batch, output_dim=4)
    r3 = fit_r3_matched_random_mlp(
        source_batch, output_dim=4, hidden_dims=(8, 8), seed=2
    )
    labels = formal_source_fit.training_task_labels
    r5 = fit_r5_corro_style(
        source_batch,
        labels=labels,
        trainer=_fake_trainer,
        objective_digest=_d("task-supcon"),
        seed=3,
        output_dim=4,
        hidden_dims=(8, 8),
    )
    r5l = fit_r5l_supervised_linear(
        source_batch,
        labels=labels,
        trainer=_fake_trainer,
        objective_digest=_d("linear-task-supcon"),
        seed=4,
        output_dim=4,
    )
    for fitted in (r2, r5, r5l):
        formal_source_fit.require_manifest_binding(fitted.manifest)
    r4 = bind_r4_frozen_callable(
        source_batch,
        output_dim=4,
        checkpoint_digest=_d("archived-r4-checkpoint"),
        normalizer_digest=normalizer.normalizer_digest,
        implementation_digest=_d("archived-r4-implementation"),
        transform=lambda values: np.asarray(values[:, :4], dtype=np.float64),
    )
    representation_outputs = tuple(
        fitted.transform(query_batch)
        for fitted in (r0, r1, r2, r3, r4, r5, r5l)
    )

    represented = transform_feature_banks(r0, features)
    matrix = build_signal_matrix_plan()
    representation_plan = RepresentationExecutionPlan.create(
        signal_plan=matrix,
        historical_spec=historical,
    )
    representation_plan.validate_manifest(r0.manifest)
    for feature in features:
        condition_plan.validate_feature_bank(feature)
    cell = matrix.cell(
        "CORE_PAIRED::V_REWARD_FREE_TRANSITION::R0_PADDED_RAW"
    )
    identity_registry = SignalIdentityRegistry(
        taxonomy_manifest_digest=_d("prelarge-task-taxonomy"),
        identities=identities,
    )
    execution_protocol = SignalExecutionProtocol(
        plan_digest=str(matrix.plan_digest),
        identity_registry_digest=str(identity_registry.registry_digest),
        measurement_protocol_digest=identity_registry.measurement_protocol_digest,
        representation_plan=representation_plan,
        condition_plan=condition_plan,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        reducer_config=ReducerConfig(
            support_budget=4,
            support_steps=0,
            kmeans_steps=0,
            ridge=0.0,
            pinv_rcond=1.0e-12,
        ),
        pair_budget=64,
        block_size=2,
        historical_seed=historical.seed,
    )
    work_item = SignalCellWorkItem(
        plan=matrix,
        cell=cell,
        source_banks=represented[:3],
        query_banks=represented[3:],
        expected_source_by_query={
            "query-a": "source-a",
            "query-b": "source-b",
            "query-a-goal2": "source-a",
        },
        identity_registry=identity_registry,
        execution_protocol=execution_protocol,
        evaluation_seed=None,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
    )
    cell_run = work_item.execute()

    feature_by_id = {item.receipt.bank_id: item for item in features}
    left_schema = bank_control_reference_from_feature_bank(feature_by_id["query-a"])
    right_schema = bank_control_reference_from_feature_bank(
        feature_by_id["query-a-goal2"]
    )
    schema_pair = SchemaCollisionPairContract(
        pair_id="schema-pair",
        left=left_schema,
        right=right_schema,
        metric_ids=("context_top1",),
        statistical_identity="schema-collision-panel",
        preregistration_digest=_d("schema-preregistration"),
    )
    repeat_right = bank_control_reference_from_feature_bank(feature_by_id["source-a"])
    repeat_pair = ExactRepeatPairContract(
        pair_id="repeat-pair",
        left=left_schema,
        right=repeat_right,
        metric_ids=("direct_repeat_mmd",),
        statistical_identity="exact-repeat-noise-floor",
        preregistration_digest=_d("repeat-preregistration"),
    )
    schema_membership = validate_pair_control_feature_banks(
        schema_pair,
        feature_by_id["query-a"],
        feature_by_id["query-a-goal2"],
    )
    schema_evaluation = PairControlEvaluation.evaluate(
        schema_pair, cell_run.metric_record, schema_membership
    )
    repeat_membership = validate_pair_control_feature_banks(
        repeat_pair,
        feature_by_id["query-a"],
        feature_by_id["source-a"],
    )
    repeat_evaluation = ExactRepeatDistanceResult.evaluate(
        repeat_pair, repeat_membership, cell_run
    )
    repeat_ratio = exact_repeat_noise_ratio(repeat_evaluation)

    jobs = build_optimization_fit_jobs(matrix)
    prefix_schedule = SignalPrefixSchedule.formal()
    dynamics_registry = DynamicsAxisRegistry(
        (
            DynamicsAxisEntry(
                "task-a-mass-low",
                "task-a-mass-axis",
                "task-a",
                "walker",
                "abi-task-a",
                "goal-task-a",
                0.5,
                "ANCHOR",
            ),
            DynamicsAxisEntry(
                "task-a-mass-high",
                "task-a-mass-axis",
                "task-a",
                "walker",
                "abi-task-a",
                "goal-task-a",
                1.5,
                "ANCHOR",
            ),
            DynamicsAxisEntry(
                "task-a-mass-interpolation",
                "task-a-mass-axis",
                "task-a",
                "walker",
                "abi-task-a",
                "goal-task-a",
                1.0,
                "INTERPOLATION",
            ),
            DynamicsAxisEntry(
                "task-a-mass-extrapolation",
                "task-a-mass-axis",
                "task-a",
                "walker",
                "abi-task-a",
                "goal-task-a",
                2.0,
                "EXTRAPOLATION",
            ),
        )
    )
    checks = {
        "t_p4_01_historical_random_is_distinct": (
            historical.representation_id == R_HIST_RANDOM_TANH
            and historical.representation_id != R3_MATCHED_RANDOM_MLP
            and historical_result.values.shape == (4, 5)
        ),
        "t_p4_02_rf_shuffled_next_marginals": rf_control.marginal_audit.passed,
        "t_p4_03_pair_controls_are_directly_evaluated": (
            not schema_pair.adds_input_view
            and not repeat_pair.adds_input_view
            and schema_evaluation.result.control_id == schema_pair.control_id
            and repeat_evaluation.metric_id == "direct_repeat_mmd"
            and repeat_ratio.between_axis_scope
            == "SAME_TASK_GOAL_EMBODIMENT_ABI_DIFFERENT_DYNAMICS"
            and repeat_ratio.between_row_count == 1
        ),
        "t_p4_04_matrix_is_39_37_2": (
            matrix.logical_cell_count == 39
            and matrix.numeric_cell_count == 37
            and matrix.structural_na_count == 2
        ),
        "t_p4_05_cross_task_canonical_width": (
            len({item.canonical_observation_dim for item in receipts}) == 1
            and len({item.canonical_action_dim for item in receipts}) == 1
        ),
        "representation_ladder_r0_to_r5l": (
            len(representation_outputs) == 7
            and all(output.values.shape[0] == 4 for output in representation_outputs)
        ),
        "formal_source_fit_provenance": (
            formal_source_fit.authority.condition_id
            == V_REWARD_FREE_TRANSITION
            and formal_source_fit.training_batch.role == "SOURCE_FIT"
        ),
        "representation_condition_plans_bound": (
            execution_protocol.representation_plan.plan_digest
            == representation_plan.plan_digest
            and execution_protocol.condition_plan.plan_digest
            == condition_plan.plan_digest
            and cell_run.execution_mode == DEVELOPMENT_SMOKE_MODE
        ),
        "empirical_query_reduced_source_cell": (
            "context_top1" in cell_run.metric_record.metric_values
            and cell_run.metric_record.metric_values["query_count"] == 3.0
            and len(cell_run.query_run_digests) == 3
        ),
        "signal_diagnostics_bound_before_array_discard": (
            cell_run.diagnostics.metric_record_digest
            == cell_run.metric_record.record_digest
            and len(cell_run.diagnostics.bank_geometries) == 6
            and cell_run.diagnostics.to_public_dict()[
                "private_bank_and_taxonomy_rows_withheld"
            ]
            is True
        ),
        "formal_signal_prefix_schedule_frozen": (
            prefix_schedule.prefix_episode_counts
            == FORMAL_SIGNAL_PREFIX_EPISODE_COUNTS
            and prefix_schedule.scope == "FORMAL"
        ),
        "dynamics_axis_registry_frozen": (
            dynamics_registry.entry("task-a-mass-interpolation").role
            == "INTERPOLATION"
            and dynamics_registry.entry("task-a-mass-extrapolation").role
            == "EXTRAPOLATION"
        ),
        "baseline_registry_complete": (
            len(REQUIRED_BASELINE_METHOD_IDS) == 9
            and dict(OPTIONAL_BASELINE_STATES) == {"B4c": "DISABLED", "B6": "DISABLED"}
        ),
        "v04_assets_not_required": True,
    }
    evidence = {
        "canonicalizer": str(canonicalizer.canonicalizer_digest),
        "historical_random": str(historical.checkpoint_digest),
        "rf_shuffled_next": str(rf_control.dataset_digest),
        "schema_collision_pair": str(schema_pair.pair_digest),
        "exact_repeat_pair": str(repeat_pair.pair_digest),
        "schema_collision_evaluation": str(schema_evaluation.evaluation_digest),
        "exact_repeat_distance": str(repeat_evaluation.result_digest),
        "exact_repeat_noise_ratio": str(repeat_ratio.ratio_digest),
        "signal_matrix": str(matrix.plan_digest),
        "signal_cell_run": str(cell_run.run_digest),
        "signal_cell_diagnostics": str(cell_run.diagnostics.diagnostics_digest),
        "representation_coordinates": sha256_json(
            [output.coordinate_digest for output in representation_outputs]
        ),
        "formal_source_fit": formal_source_fit.batch_digest,
        "signal_execution_protocol": str(execution_protocol.protocol_digest),
        "representation_execution_plan": str(representation_plan.plan_digest),
        "condition_execution_plan": str(condition_plan.plan_digest),
        "formal_signal_prefix_schedule": str(prefix_schedule.schedule_digest),
        "dynamics_axis_registry": str(dynamics_registry.registry_digest),
        "baseline_registry": sha256_json(
            {
                "required": list(REQUIRED_BASELINE_METHOD_IDS),
                "optional": dict(OPTIONAL_BASELINE_STATES),
            }
        ),
    }
    return PrelargeAcceptanceReport(
        checks=checks,
        evidence_digests=evidence,
        signal_matrix_logical_cells=matrix.logical_cell_count,
        signal_matrix_numeric_cells=matrix.numeric_cell_count,
        optimization_fit_jobs=len(jobs),
        required_baseline_methods=REQUIRED_BASELINE_METHOD_IDS,
    )


def main() -> int:
    report = run_prelarge_acceptance()
    print(canonical_json(report.to_dict()))
    return 0 if report.passed else 1


__all__ = [
    "PRELARGE_ACCEPTANCE_SCHEMA",
    "PrelargeAcceptanceError",
    "PrelargeAcceptanceReport",
    "run_prelarge_acceptance",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
