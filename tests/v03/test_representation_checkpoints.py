from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json, sha256_ndarrays
from policy_learnware_v0.v03.artifacts import V03ArtifactLayout
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.corro_trainers import (
    CorroBackendResult,
    CorroOptimizationConfig,
    CorroTrainerAdapter,
    JaxCorroTrainingBackend,
    TASK_SUPCON_OBJECTIVE_DIGEST,
    jax_corro_backend_implementation_digest,
)
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.data_roles import DataRoleManifest, DataRoleRecord
from policy_learnware_v0.v03.representation_checkpoints import (
    DEVELOPMENT_SMOKE_MODE,
    FORMAL_MODE,
    RepresentationCheckpointError,
    freeze_trained_representation_checkpoint,
    load_trained_representation_checkpoint,
)
from policy_learnware_v0.v03.representation_ladder import (
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    RepresentationBatch,
    TrainedCallableArtifact,
    TrainingRequest,
    fit_r5_corro_style,
)
from policy_learnware_v0.v03.representation_plan import RepresentationExecutionPlan
from policy_learnware_v0.v03.signal_controls import HistoricalRandomTanhSpec
from policy_learnware_v0.v03.signal_matrix import (
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.signal_runtime import (
    SignalBankIdentity,
    feature_bank_from_transition_view,
)
from policy_learnware_v0.v03.source_fit import (
    FormalSourceFitSchedule,
    build_formal_source_fit_batch,
)
from policy_learnware_v0.v03.transition_views import (
    V_REWARD_FREE_TRANSITION,
    TransitionBank,
    apply_transition_view,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


class _RestorableTrainer:
    @staticmethod
    def _artifact(request: TrainingRequest) -> TrainedCallableArtifact:
        weight = np.random.default_rng(request.seed).normal(
            size=(request.input_dim, request.output_dim)
        )

        def transform(values: np.ndarray) -> np.ndarray:
            projected = np.asarray(values, dtype=np.float64) @ weight
            return projected / np.maximum(
                np.linalg.norm(projected, axis=1, keepdims=True), 1.0e-12
            )

        return TrainedCallableArtifact(
            checkpoint_bytes=("checkpoint:" + request.request_digest).encode("ascii"),
            parameter_digest=sha256_ndarrays({"weight": weight}),
            trainer_implementation_digest=_d("restorable-trainer"),
            transform=transform,
        )

    def __call__(self, source_values, labels, request):
        del source_values, labels
        return self._artifact(request)

    def restore(self, checkpoint_bytes, request):
        artifact = self._artifact(request)
        if artifact.checkpoint_bytes != checkpoint_bytes:
            raise ValueError("checkpoint mismatch")
        return artifact


def _representation_plan(
    *,
    historical_seed: int = 11,
    optimization: CorroOptimizationConfig | None = None,
) -> RepresentationExecutionPlan:
    historical = HistoricalRandomTanhSpec.create(
        seed=historical_seed,
        input_dim=3,
        output_dim=32,
    )
    return RepresentationExecutionPlan.create(
        signal_plan=build_signal_matrix_plan(),
        historical_spec=historical,
        shared_output_dim=32,
        hidden_dims=(256, 256),
        optimization=optimization,
    )


def _fit():
    plan = _representation_plan()
    source = RepresentationBatch(
        np.arange(36, dtype=np.float64).reshape(12, 3) / 10.0,
        _d("source"),
        "SOURCE_FIT",
    )
    labels = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
    request = TrainingRequest(
        representation_id=R5_VIEW_SPECIFIC_CORRO_REFIT,
        input_dim=3,
        output_dim=32,
        hidden_dims=(256, 256),
        activation="relu",
        l2_normalize_output=True,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=0,
    )
    trainer = _RestorableTrainer()
    fitted = fit_r5_corro_style(
        source,
        labels=labels,
        trainer=trainer,
        objective_digest=request.objective_digest,
        seed=request.seed,
        output_dim=request.output_dim,
        hidden_dims=(256, 256),
    )
    return source, labels, request, trainer, fitted, plan


def _native(
    bank_id: str,
    task_id: str,
    role: str,
    *,
    center: float,
) -> NativeTransitionBank:
    observation = center + np.arange(12, dtype=np.float64).reshape(4, 3) / 100.0
    action = center / 10.0 + np.arange(8, dtype=np.float64).reshape(4, 2) / 100.0
    return NativeTransitionBank(
        bank_id=bank_id,
        task_private_id=task_id,
        data_role=role,  # type: ignore[arg-type]
        native_schema_digest=_d(f"schema:{task_id}"),
        raw_dataset_digest=_d(f"raw:{bank_id}:{center}"),
        observation=observation,
        action=action,
        reward=np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64) + center,
        next_observation=observation + 0.05,
        terminated=np.asarray([False, True, False, True]),
        truncated=np.asarray([False, False, False, False]),
        episode_id=np.asarray([0, 0, 1, 1]),
        timestep=np.asarray([0, 1, 0, 1]),
    )


def _formal_fit(*, offset: float = 0.0):
    train = (
        _native(
            "formal-train-a",
            "task-a",
            "source_representation_train",
            center=offset,
        ),
        _native(
            "formal-train-b",
            "task-b",
            "source_representation_train",
            center=10.0 + offset,
        ),
    )
    validation = (
        _native(
            "formal-validation-a",
            "task-a",
            "source_representation_validation",
            center=0.03 + offset,
        ),
        _native(
            "formal-validation-b",
            "task-b",
            "source_representation_validation",
            center=10.03 + offset,
        ),
    )
    native = (*train, *validation)
    registry = NativeShapeRegistry.from_source_banks(native)
    canonicalizer = GlobalCanonicalizerSpec(
        registry,
        fit_global_normalizer(native, registry),
    )
    receipts = tuple(canonicalizer.transform(bank) for bank in native)
    identities = tuple(
        SignalBankIdentity.from_receipt(
            receipt,
            embodiment_id=f"embodiment-{receipt.task_private_id}",
            abi_contract_id=f"abi-{receipt.task_private_id}",
            goal_contract_id=f"goal-{receipt.task_private_id}",
            dynamics_context_id=f"dynamics-{receipt.task_private_id}",
            context_id=f"context-{receipt.bank_id}",
            measurement_protocol_digest=_d("formal-measurement"),
            probe_seed_digest=_d(f"probe:{receipt.bank_id}:{offset}"),
            equivalence_class_id=f"equivalence-{receipt.task_private_id}",
        )
        for receipt in receipts
    )
    features = tuple(
        feature_bank_from_transition_view(
            receipt,
            identity,
            apply_transition_view(
                TransitionBank.from_canonical_batch(receipt.batch),
                V_REWARD_FREE_TRANSITION,
            ),
        )
        for receipt, identity in zip(receipts, identities, strict=True)
    )
    split_nonce = _d(f"formal-split:{offset}")
    role_manifest = DataRoleManifest(
        manifest_id=f"formal-checkpoint-{str(offset).replace('.', '-')}",
        records=tuple(
            DataRoleRecord(
                role=bank.data_role,
                dataset_id=f"role-{bank.bank_id}",
                dataset_digest=bank.raw_dataset_digest,
                task_private_ids=(bank.task_private_id,),
                seed_tokens=(f"seed-{bank.bank_id}-{offset}",),
                split_nonce_digest=split_nonce,
            )
            for bank in native
        ),
    )
    source_fit = build_formal_source_fit_batch(
        role_manifest,
        train_feature_banks=features[:2],
        validation_feature_banks=features[2:],
        condition_plan=ConditionExecutionPlan.create(
            historical_spec=HistoricalRandomTanhSpec.create(
                seed=11, input_dim=8, output_dim=32
            )
        ),
    )
    plan = _representation_plan()
    request = TrainingRequest(
        representation_id=R5_VIEW_SPECIFIC_CORRO_REFIT,
        input_dim=source_fit.training_batch.input_dim,
        output_dim=32,
        hidden_dims=(256, 256),
        activation="relu",
        l2_normalize_output=True,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=0,
    )
    trainer = _RestorableTrainer()
    fitted = fit_r5_corro_style(
        source_fit.training_batch,
        labels=source_fit.training_task_labels,
        trainer=trainer,
        objective_digest=request.objective_digest,
        seed=request.seed,
        output_dim=request.output_dim,
        hidden_dims=(256, 256),
    )
    source_fit.require_manifest_binding(fitted.manifest)
    return source_fit, request, trainer, fitted, plan


def _formal_schedule(source_fit):
    condition_plan = ConditionExecutionPlan.create(
        historical_spec=HistoricalRandomTanhSpec.create(
            seed=11, input_dim=8, output_dim=32
        )
    )
    jobs = build_optimization_fit_jobs(build_signal_matrix_plan())
    authorities = {
        condition_id: replace(
            source_fit.authority,
            condition_id=condition_id,
            condition_transform_digest=condition_plan.transform_digest(condition_id),
            authority_digest=None,
        )
        for condition_id in {item.condition_id for item in jobs}
    }
    schedule = FormalSourceFitSchedule.from_condition_authorities(
        condition_plan=condition_plan,
        authorities=authorities,
    )
    job = next(
        item
        for item in jobs
        if item.condition_id == V_REWARD_FREE_TRANSITION
        and item.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
        and item.seed == 0
    )
    return job, schedule


def test_trained_checkpoint_publication_restart_roundtrip(tmp_path) -> None:
    source, labels, request, trainer, fitted, plan = _fit()
    layout = V03ArtifactLayout.development(tmp_path, "dev-checkpoint")
    checkpoint_path = layout.artifact(
        "representation_controls", "r5", "checkpoint.bin"
    )
    manifest_path = layout.artifact(
        "representation_controls", "r5", "checkpoint_manifest.json"
    )
    publication = freeze_trained_representation_checkpoint(
        fitted=fitted,
        training_request=request,
        representation_plan=plan,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        writer=layout.writer("representation_controls"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )
    resumed = freeze_trained_representation_checkpoint(
        fitted=fitted,
        training_request=request,
        representation_plan=plan,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        writer=layout.writer("representation_controls"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        resume=True,
    )
    assert resumed.checkpoint_file_digest == publication.checkpoint_file_digest
    with pytest.raises(RepresentationCheckpointError, match="development checkpoint"):
        publication.manifest.formal_fit_receipt()
    manifest, restored = load_trained_representation_checkpoint(
        reader=layout.reader("representation_controls"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        expected_checkpoint_file_digest=publication.checkpoint_file_digest,
        expected_manifest_file_digest=publication.manifest_file_digest,
        representation_plan=plan,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        restorer=trainer,
        verification_source=source,
        labels=labels,
    )
    assert manifest.representation_manifest == fitted.manifest
    query = RepresentationBatch(
        np.asarray([[0.1, 0.2, 0.3], [1.0, 0.0, 2.0]]),
        _d("query"),
        "QUERY_TRANSFORM",
    )
    np.testing.assert_array_equal(
        fitted.transform(query).values, restored.transform(query).values
    )


def test_checkpoint_loader_rejects_wrong_artifact_binding(tmp_path) -> None:
    source, labels, request, trainer, fitted, plan = _fit()
    layout = V03ArtifactLayout.development(tmp_path, "dev-checkpoint")
    checkpoint_path = layout.artifact(
        "representation_controls", "r5", "checkpoint.bin"
    )
    manifest_path = layout.artifact(
        "representation_controls", "r5", "checkpoint_manifest.json"
    )
    publication = freeze_trained_representation_checkpoint(
        fitted=fitted,
        training_request=request,
        representation_plan=plan,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        writer=layout.writer("representation_controls"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )
    with pytest.raises(RepresentationCheckpointError, match="expected checkpoint"):
        load_trained_representation_checkpoint(
            reader=layout.reader("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            expected_checkpoint_file_digest=_d("wrong"),
            expected_manifest_file_digest=publication.manifest_file_digest,
            representation_plan=plan,
            execution_mode=DEVELOPMENT_SMOKE_MODE,
            restorer=trainer,
            verification_source=source,
            labels=labels,
        )


def test_checkpoint_loader_rejects_wrong_plan_and_optimization(tmp_path) -> None:
    source, labels, request, trainer, fitted, plan = _fit()
    layout = V03ArtifactLayout.development(tmp_path, "dev-plan-binding")
    checkpoint_path = layout.artifact(
        "representation_controls", "r5", "checkpoint.bin"
    )
    manifest_path = layout.artifact(
        "representation_controls", "r5", "checkpoint_manifest.json"
    )
    publication = freeze_trained_representation_checkpoint(
        fitted=fitted,
        training_request=request,
        representation_plan=plan,
        execution_mode=DEVELOPMENT_SMOKE_MODE,
        writer=layout.writer("representation_controls"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )

    wrong_optimization = _representation_plan(
        optimization=CorroOptimizationConfig(train_steps=1)
    )
    with pytest.raises(RepresentationCheckpointError, match="optimization digest"):
        load_trained_representation_checkpoint(
            reader=layout.reader("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            expected_checkpoint_file_digest=publication.checkpoint_file_digest,
            expected_manifest_file_digest=publication.manifest_file_digest,
            representation_plan=wrong_optimization,
            execution_mode=DEVELOPMENT_SMOKE_MODE,
            restorer=trainer,
            verification_source=source,
            labels=labels,
        )

    wrong_plan = _representation_plan(historical_seed=12)
    with pytest.raises(RepresentationCheckpointError, match="plan digest"):
        load_trained_representation_checkpoint(
            reader=layout.reader("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            expected_checkpoint_file_digest=publication.checkpoint_file_digest,
            expected_manifest_file_digest=publication.manifest_file_digest,
            representation_plan=wrong_plan,
            execution_mode=DEVELOPMENT_SMOKE_MODE,
            restorer=trainer,
            verification_source=source,
            labels=labels,
        )


def test_formal_checkpoint_requires_production_trainer_and_same_source_fit(
    tmp_path, monkeypatch
) -> None:
    source_fit, request, trainer, fitted, plan = _formal_fit()
    layout = V03ArtifactLayout.joint(tmp_path, "formal-checkpoint")
    checkpoint_path = layout.artifact(
        "representation_controls", "r5", "checkpoint.bin"
    )
    manifest_path = layout.artifact(
        "representation_controls", "r5", "checkpoint_manifest.json"
    )
    with pytest.raises(RepresentationCheckpointError, match="FormalSourceFitBatch"):
        freeze_trained_representation_checkpoint(
            fitted=fitted,
            training_request=request,
            representation_plan=plan,
            execution_mode=FORMAL_MODE,
            writer=layout.writer("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )

    with pytest.raises(RepresentationCheckpointError, match="CorroTrainerAdapter"):
        freeze_trained_representation_checkpoint(
            fitted=fitted,
            training_request=request,
            representation_plan=plan,
            execution_mode=FORMAL_MODE,
            formal_source_fit=source_fit,
            writer=layout.writer("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )

    train_split, validation_split = source_fit.corro_source_splits()
    formal_trainer = CorroTrainerAdapter(
        train_split,
        validation_split,
        plan.optimization,
    )
    formal_fit_job, formal_schedule = _formal_schedule(source_fit)
    with pytest.raises(RepresentationCheckpointError, match="45-job source-fit schedule"):
        freeze_trained_representation_checkpoint(
            fitted=fitted,
            training_request=request,
            representation_plan=plan,
            execution_mode=FORMAL_MODE,
            formal_source_fit=source_fit,
            formal_trainer=formal_trainer,
            writer=layout.writer("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )
    with pytest.raises(RepresentationCheckpointError, match="frozen CORRO backend"):
        freeze_trained_representation_checkpoint(
            fitted=fitted,
            training_request=request,
            representation_plan=plan,
            execution_mode=FORMAL_MODE,
            formal_source_fit=source_fit,
            formal_trainer=formal_trainer,
            formal_fit_job=formal_fit_job,
            formal_source_fit_schedule=formal_schedule,
            writer=layout.writer("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )

    def production_result(candidate_request, checkpoint_bytes=None):
        weight = np.random.default_rng(candidate_request.seed).normal(
            size=(candidate_request.input_dim, candidate_request.output_dim)
        )
        frozen_bytes = (
            checkpoint_bytes
            if checkpoint_bytes is not None
            else ("production:" + candidate_request.request_digest).encode("ascii")
        )

        def transform(values):
            projected = np.asarray(values, dtype=np.float64) @ weight
            return projected / np.maximum(
                np.linalg.norm(projected, axis=1, keepdims=True), 1.0e-12
            )

        return CorroBackendResult(
            checkpoint_bytes=frozen_bytes,
            implementation_digest=jax_corro_backend_implementation_digest(
                candidate_request.representation_id
            ),
            transform=transform,
        )

    def fake_train(self, *, train, validation, request, optimization):
        del self, train, validation, optimization
        return production_result(request)

    def fake_restore(self, *, checkpoint_bytes, request, optimization):
        del self, optimization
        return production_result(request, checkpoint_bytes)

    monkeypatch.setattr(JaxCorroTrainingBackend, "train", fake_train)
    monkeypatch.setattr(JaxCorroTrainingBackend, "restore", fake_restore)
    fitted = fit_r5_corro_style(
        source_fit.training_batch,
        labels=source_fit.training_task_labels,
        trainer=formal_trainer,
        objective_digest=request.objective_digest,
        seed=request.seed,
        output_dim=request.output_dim,
        hidden_dims=(256, 256),
    )
    wrong_seed_job = next(
        item
        for item in build_optimization_fit_jobs(build_signal_matrix_plan())
        if item.condition_id == V_REWARD_FREE_TRANSITION
        and item.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
        and item.seed == 1
    )
    with pytest.raises(RepresentationCheckpointError, match="frozen fit job"):
        freeze_trained_representation_checkpoint(
            fitted=fitted,
            training_request=request,
            representation_plan=plan,
            execution_mode=FORMAL_MODE,
            formal_source_fit=source_fit,
            formal_trainer=formal_trainer,
            formal_fit_job=wrong_seed_job,
            formal_source_fit_schedule=formal_schedule,
            writer=layout.writer("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )
    publication = freeze_trained_representation_checkpoint(
        fitted=fitted,
        training_request=request,
        representation_plan=plan,
        execution_mode=FORMAL_MODE,
        formal_source_fit=source_fit,
        formal_trainer=formal_trainer,
        formal_fit_job=formal_fit_job,
        formal_source_fit_schedule=formal_schedule,
        writer=layout.writer("representation_controls"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )
    assert (
        publication.manifest.formal_source_fit_batch_digest
        == source_fit.batch_digest
    )
    assert publication.manifest.formal_trainer_contract_digest is not None
    receipt = publication.manifest.formal_fit_receipt()
    receipt.validate_manifest(fitted.manifest)
    assert receipt.formal_source_fit_batch_digest == source_fit.batch_digest
    manifest, restored = load_trained_representation_checkpoint(
        reader=layout.reader("representation_controls"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        expected_checkpoint_file_digest=publication.checkpoint_file_digest,
        expected_manifest_file_digest=publication.manifest_file_digest,
        representation_plan=plan,
        execution_mode=FORMAL_MODE,
        formal_source_fit=source_fit,
        formal_fit_job=formal_fit_job,
        formal_source_fit_schedule=formal_schedule,
        restorer=formal_trainer,
        verification_source=source_fit.training_batch,
        labels=source_fit.training_task_labels,
    )
    assert manifest.execution_mode == FORMAL_MODE
    assert restored.manifest == fitted.manifest

    with pytest.raises(RepresentationCheckpointError, match="FormalSourceFitBatch"):
        load_trained_representation_checkpoint(
            reader=layout.reader("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            expected_checkpoint_file_digest=publication.checkpoint_file_digest,
            expected_manifest_file_digest=publication.manifest_file_digest,
            representation_plan=plan,
            execution_mode=FORMAL_MODE,
            restorer=formal_trainer,
            verification_source=source_fit.training_batch,
            labels=source_fit.training_task_labels,
        )

    other_source_fit, _request, _trainer, _fitted, _plan = _formal_fit(offset=1.0)
    with pytest.raises(RepresentationCheckpointError, match="not bound"):
        load_trained_representation_checkpoint(
            reader=layout.reader("representation_controls"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            expected_checkpoint_file_digest=publication.checkpoint_file_digest,
            expected_manifest_file_digest=publication.manifest_file_digest,
            representation_plan=plan,
            execution_mode=FORMAL_MODE,
            formal_source_fit=other_source_fit,
            formal_fit_job=formal_fit_job,
            formal_source_fit_schedule=formal_schedule,
            restorer=formal_trainer,
            verification_source=source_fit.training_batch,
            labels=source_fit.training_task_labels,
        )
