from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json, sha256_ndarrays
from policy_learnware_v0.v03.canonicalization import (
    GlobalCanonicalizerSpec,
    NativeShapeRegistry,
    NativeTransitionBank,
    fit_global_normalizer,
)
from policy_learnware_v0.v03.data_roles import DataRoleManifest, DataRoleRecord
from policy_learnware_v0.v03.condition_plan import ConditionExecutionPlan
from policy_learnware_v0.v03.representation_ladder import (
    R2_SOURCE_PCA_WHITEN,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    TrainedCallableArtifact,
    fit_r2_pca_whitening,
    fit_r5_corro_style,
    fit_r5l_supervised_linear,
)
from policy_learnware_v0.v03.signal_runtime import (
    SignalBankIdentity,
    feature_bank_from_transition_view,
    transform_feature_banks,
)
from policy_learnware_v0.v03.signal_controls import HistoricalRandomTanhSpec
from policy_learnware_v0.v03.source_fit import (
    FormalSourceFitAuthority,
    FormalSourceFitSchedule,
    SourceFitProvenanceError,
    build_formal_source_fit_batch,
    development_source_fit_batch,
)
from policy_learnware_v0.v03.signal_matrix import (
    build_optimization_fit_jobs,
    build_signal_matrix_plan,
)
from policy_learnware_v0.v03.transition_views import (
    V_REWARD_FREE_TRANSITION,
    TransitionBank,
    apply_transition_view,
)


def _d(label: str) -> str:
    return sha256_json({"v03-source-fit-test": label})


def _condition_plan() -> ConditionExecutionPlan:
    return ConditionExecutionPlan.create(
        historical_spec=HistoricalRandomTanhSpec.create(
            seed=11, input_dim=8, output_dim=32
        )
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


def _role_record(bank: NativeTransitionBank, seed: str) -> DataRoleRecord:
    return DataRoleRecord(
        role=bank.data_role,
        dataset_id=f"role-{bank.bank_id}",
        dataset_digest=bank.raw_dataset_digest,
        task_private_ids=(bank.task_private_id,),
        seed_tokens=(seed,),
        split_nonce_digest=_d("split-nonce"),
    )


def _case(*, validation_matches_train: bool = False):
    train = (
        _native(
            "fit-train-a",
            "task-a",
            "source_representation_train",
            observation_dim=2,
            action_dim=1,
            center=0.0,
        ),
        _native(
            "fit-train-b",
            "task-b",
            "source_representation_train",
            observation_dim=3,
            action_dim=2,
            center=10.0,
        ),
    )
    validation = (
        _native(
            "fit-validation-a",
            "task-a",
            "source_representation_validation",
            observation_dim=2,
            action_dim=1,
            center=0.0 if validation_matches_train else 0.03,
        ),
        _native(
            "fit-validation-b",
            "task-b",
            "source_representation_validation",
            observation_dim=3,
            action_dim=2,
            center=10.0 if validation_matches_train else 10.03,
        ),
    )
    evaluation = (
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
    )
    registry = NativeShapeRegistry.from_source_banks((*train, *validation))
    normalizer = fit_global_normalizer((*train, *validation), registry)
    canonicalizer = GlobalCanonicalizerSpec(registry, normalizer)
    native = (*train, *validation, *evaluation)
    receipts = tuple(canonicalizer.transform(item) for item in native)
    measurement = _d("measurement")
    identities = tuple(
        SignalBankIdentity.from_receipt(
            receipt,
            embodiment_id=f"embodiment-{receipt.task_private_id}",
            abi_contract_id=f"abi-{receipt.task_private_id}",
            goal_contract_id=f"goal-{receipt.task_private_id}",
            dynamics_context_id=f"dynamics-{receipt.task_private_id}",
            context_id=f"context-{receipt.bank_id}",
            measurement_protocol_digest=measurement,
            probe_seed_digest=_d(f"probe:{receipt.bank_id}"),
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
    manifest = DataRoleManifest(
        manifest_id="formal-source-fit-test",
        records=tuple(
            _role_record(bank, f"seed-{bank.bank_id}")
            for bank in (*train, *validation)
        ),
    )
    source_fit = build_formal_source_fit_batch(
        manifest,
        train_feature_banks=features[:2],
        validation_feature_banks=features[2:4],
        condition_plan=_condition_plan(),
    )
    return source_fit, features[4:], manifest


def _trainer(values: np.ndarray, labels: np.ndarray, request) -> TrainedCallableArtifact:
    rng = np.random.default_rng(request.seed)
    weight = rng.normal(size=(request.input_dim, request.output_dim)).astype(np.float64)

    def transform(candidate: np.ndarray) -> np.ndarray:
        projected = np.asarray(candidate, dtype=np.float64) @ weight
        return projected / np.maximum(
            np.linalg.norm(projected, axis=1, keepdims=True),
            np.finfo(np.float64).eps,
        )

    return TrainedCallableArtifact(
        checkpoint_bytes=(request.request_digest + ":checkpoint").encode("ascii"),
        parameter_digest=sha256_ndarrays({"weight": weight}),
        trainer_implementation_digest=_d("trainer"),
        transform=transform,
    )


def _fitted(representation_id: str, source_fit):
    if representation_id == R2_SOURCE_PCA_WHITEN:
        return fit_r2_pca_whitening(source_fit.training_batch, output_dim=3)
    if representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT:
        return fit_r5_corro_style(
            source_fit.training_batch,
            labels=source_fit.training_task_labels,
            trainer=_trainer,
            objective_digest=_d("objective"),
            seed=3,
            output_dim=3,
            hidden_dims=(8, 8),
        )
    return fit_r5l_supervised_linear(
        source_fit.training_batch,
        labels=source_fit.training_task_labels,
        trainer=_trainer,
        objective_digest=_d("objective"),
        seed=4,
        output_dim=3,
    )


def test_formal_source_fit_authority_is_role_and_digest_bound() -> None:
    source_fit, _evaluation, _manifest = _case()
    authority = source_fit.authority
    assert FormalSourceFitAuthority.from_dict(authority.to_dict()) == authority
    assert source_fit.to_dict()["batch_digest"] == source_fit.batch_digest
    assert source_fit.training_batch.dataset_digest != source_fit.validation_batch.dataset_digest
    assert source_fit.training_batch.batch_digest != source_fit.validation_batch.batch_digest
    assert authority.condition_execution_plan_digest == _condition_plan().plan_digest
    assert tuple(np.unique(source_fit.training_task_labels)) == (0, 1)
    train_split, validation_split = source_fit.corro_source_splits()
    assert train_split.task_ids == validation_split.task_ids == ("task-a", "task-b")
    assert train_split.split_digest != validation_split.split_digest


def test_45_job_source_fit_schedule_freezes_one_membership_and_all_transforms() -> None:
    source_fit, _evaluation, _manifest = _case()
    plan = _condition_plan()
    jobs = build_optimization_fit_jobs(build_signal_matrix_plan())
    conditions = {item.condition_id for item in jobs}
    authorities = {
        condition_id: replace(
            source_fit.authority,
            condition_id=condition_id,
            condition_transform_digest=plan.transform_digest(condition_id),
            authority_digest=None,
        )
        for condition_id in conditions
    }
    schedule = FormalSourceFitSchedule.from_condition_authorities(
        condition_plan=plan,
        authorities=authorities,
    )
    assert len(schedule.job_authorities) == 45
    assert schedule.source_membership_digest == source_fit.authority.source_membership_digest
    assert schedule.authority_for(jobs[0]).condition_id == jobs[0].condition_id

    first_condition = sorted(conditions)[0]
    with pytest.raises(
        SourceFitProvenanceError,
        match="source_membership_digest does not match",
    ):
        replace(
            authorities[first_condition],
            source_membership_digest=_d("different-source-membership"),
            authority_digest=None,
        )

    wrong_transform = dict(authorities)
    wrong_transform[first_condition] = replace(
        wrong_transform[first_condition],
        condition_transform_digest=_d("different-scheduled-transform"),
        authority_digest=None,
    )
    with pytest.raises(SourceFitProvenanceError, match="condition freeze"):
        FormalSourceFitSchedule.from_condition_authorities(
            condition_plan=plan,
            authorities=wrong_transform,
        )


@pytest.mark.parametrize("changed_field", ("bank_id", "receipt_digest"))
def test_authority_rejects_same_membership_digest_for_different_bank_binding(
    changed_field: str,
) -> None:
    source_fit, _evaluation, _manifest = _case()
    authority = source_fit.authority
    first = authority.train_bindings[0]
    replacement = (
        "different-physical-bank"
        if changed_field == "bank_id"
        else _d("different-physical-receipt")
    )
    tampered_binding = replace(
        first,
        **{changed_field: replacement},
        binding_digest=None,
    )
    tampered_train = (tampered_binding, *authority.train_bindings[1:])

    with pytest.raises(
        SourceFitProvenanceError,
        match="source_membership_digest does not match",
    ):
        replace(
            authority,
            train_bindings=tampered_train,
            authority_digest=None,
        )


def test_authority_from_dict_recomputes_membership_from_serialized_bindings() -> None:
    source_fit, _evaluation, _manifest = _case()
    payload = source_fit.authority.to_dict()
    first = source_fit.authority.train_bindings[0]
    payload["train_bindings"][0] = replace(
        first,
        receipt_digest=_d("forged-serialized-receipt"),
        binding_digest=None,
    ).to_dict()
    payload["authority_digest"] = None

    with pytest.raises(
        SourceFitProvenanceError,
        match="source_membership_digest does not match",
    ):
        FormalSourceFitAuthority.from_dict(payload)


@pytest.mark.parametrize(
    "changed_field",
    (
        "data_role_manifest_digest",
        "train_role_digest",
        "validation_role_digest",
        "split_nonce_digest",
    ),
)
def test_authority_rejects_same_membership_for_different_role_or_split_commitment(
    changed_field: str,
) -> None:
    source_fit, _evaluation, _manifest = _case()
    with pytest.raises(
        SourceFitProvenanceError,
        match="source_membership_digest does not match",
    ):
        replace(
            source_fit.authority,
            **{changed_field: _d(f"different-{changed_field}")},
            authority_digest=None,
        )


def test_source_fit_rejects_role_relabel_measurement_drift_and_physical_overlap() -> None:
    source_fit, evaluation, manifest = _case()
    with pytest.raises(SourceFitProvenanceError, match="role=source_representation_train"):
        build_formal_source_fit_batch(
            manifest,
            train_feature_banks=(evaluation[0], source_fit.train_feature_banks[1]),
            validation_feature_banks=source_fit.validation_feature_banks,
            condition_plan=_condition_plan(),
        )

    changed_identity = replace(
        source_fit.validation_feature_banks[0].identity,
        measurement_protocol_digest=_d("another-measurement"),
    )
    changed_measurement = replace(
        source_fit.validation_feature_banks[0],
        identity=changed_identity,
        feature_bank_digest=None,
    )
    with pytest.raises(SourceFitProvenanceError, match="measurement protocol"):
        build_formal_source_fit_batch(
            manifest,
            train_feature_banks=source_fit.train_feature_banks,
            validation_feature_banks=(
                changed_measurement,
                source_fit.validation_feature_banks[1],
            ),
            condition_plan=_condition_plan(),
        )

    wrong_transform = replace(
        source_fit.train_feature_banks[0],
        condition_transform_digest=_d("unfrozen-condition-transform"),
        feature_bank_digest=None,
    )
    with pytest.raises(SourceFitProvenanceError, match="condition freeze"):
        build_formal_source_fit_batch(
            manifest,
            train_feature_banks=(wrong_transform, source_fit.train_feature_banks[1]),
            validation_feature_banks=source_fit.validation_feature_banks,
            condition_plan=_condition_plan(),
        )

    with pytest.raises(SourceFitProvenanceError, match="physical or digest overlap"):
        _case(validation_matches_train=True)


@pytest.mark.parametrize(
    "representation_id",
    (R2_SOURCE_PCA_WHITEN, R5_VIEW_SPECIFIC_CORRO_REFIT, R5L_SUPERVISED_LINEAR),
)
def test_all_data_fitted_manifests_require_exact_source_fit_provenance(
    representation_id: str,
) -> None:
    source_fit, evaluation, _manifest = _case()
    fitted = _fitted(representation_id, source_fit)
    source_fit.require_manifest_binding(fitted.manifest)
    assert fitted.manifest.source_fit_digest == source_fit.expected_manifest_source_fit_digest(
        fitted.manifest
    )
    assert len(transform_feature_banks(fitted, evaluation)) == 4


def test_development_source_fit_helper_cannot_impersonate_formal_provenance() -> None:
    source_fit, evaluation, _manifest = _case()
    relabelled = development_source_fit_batch(
        source_fit.training_batch.values,
        dataset_digest=_d("arbitrary-development-source-fit"),
    )
    fitted = fit_r2_pca_whitening(relabelled, output_dim=3)
    assert len(transform_feature_banks(fitted, evaluation)) == 4
    with pytest.raises(SourceFitProvenanceError, match="not bound"):
        source_fit.require_manifest_binding(fitted.manifest)
