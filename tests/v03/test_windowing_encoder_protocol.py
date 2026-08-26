from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_bytes, sha256_json
from policy_learnware_v0.rkme.reducer import ReducerConfig
from policy_learnware_v0.v03.access import (
    REWARD_CHANNEL,
    SOURCE_EPISODE_PROBE_BOUNDARIES,
    SOURCE_RAW_TRANSITIONS,
    EncoderAccessCard,
)
from policy_learnware_v0.v03.artifacts import V03ArtifactLayout
from policy_learnware_v0.v03.checkpoints import (
    ImmutableEncoderCheckpointAdapter,
    freeze_encoder_checkpoint,
    load_frozen_encoder_checkpoint,
)
from policy_learnware_v0.v03.compute import (
    JointDistanceRequest,
    run_joint_distance_stage,
    tie_break_digest,
)
from policy_learnware_v0.v03.contracts import (
    FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
    RankingKey,
    SemanticCacheKey,
    SemanticCacheRecord,
    SemanticTransform,
    SourceRepresentationIndex,
    build_empirical_query_spec,
    build_source_reduced_spec,
)
from policy_learnware_v0.v03.data_roles import DataRoleManifest, DataRoleRecord
from policy_learnware_v0.v03.encoder_protocol import (
    AdapterFitData,
    AdapterTrainingContract,
    CostRecord,
    EncoderInferenceContract,
    EncoderFitResult,
    EncoderProtocolError,
    EncoderSupervisionBatch,
    EncoderTrainingContract,
    EncoderTrainingProtocolBinding,
    EncoderTrainingData,
    EncoderValidationData,
    SanitizedEncoderInputBatch,
    SemanticSampleBatch,
    encoder_dataset_digest,
    project_adapter_fit,
    sanitize_encoder_inputs,
)
from policy_learnware_v0.v03.encoder_registry import (
    EncoderRegistry,
    EncoderRegistryError,
    fake_registration,
    run_adapter_conformance,
)
from policy_learnware_v0.v03.schemas import EncoderProtocolRecord, LOTOFoldRecord
from policy_learnware_v0.v03.windowing import (
    CanonicalTransitionBatch,
    TransitionWindowBatch,
    WindowingError,
    WindowingProtocol,
    build_transition_windows,
)


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _transitions(
    *, reward_shift: float = 0.0, observation_shift: float = 0.0
) -> CanonicalTransitionBatch:
    observation = np.asarray(
        [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [1.0, 1.1], [1.2, 1.3]]
    ) + observation_shift
    return CanonicalTransitionBatch(
        observation=observation,
        action=np.asarray([[0.0], [0.1], [0.2], [0.3], [0.4]]),
        reward=np.arange(5, dtype=np.float64) + reward_shift,
        next_observation=observation + 0.05,
        terminated=np.asarray([False, False, True, False, True]),
        truncated=np.zeros(5, dtype=np.bool_),
        observation_mask=np.ones(2, dtype=np.bool_),
        action_mask=np.ones(1, dtype=np.bool_),
        episode_id=np.asarray([0, 0, 0, 1, 1]),
        timestep=np.asarray([0, 1, 2, 0, 1]),
    )


def _protocol() -> WindowingProtocol:
    return WindowingProtocol(
        window_length=2, stride=2, pooling="mean", pad_final_window=True
    )


def _access(*, reward: bool = False) -> EncoderAccessCard:
    capabilities = [SOURCE_RAW_TRANSITIONS, SOURCE_EPISODE_PROBE_BOUNDARIES]
    if reward:
        capabilities.append(REWARD_CHANNEL)
    return EncoderAccessCard(
        encoder_id="fake-e0",
        access_tier="E0_UNSUPERVISED",
        declared_capabilities=tuple(capabilities),
        external_pretrained_weights_digest=None,
        max_hyperparameter_trials=1,
        total_train_compute_hours=0.0,
        formal_eligible=True,
    )


def _inputs(
    windows: TransitionWindowBatch,
    access: EncoderAccessCard,
    *,
    channels: tuple[str, ...] = ("observation", "action", "next_observation"),
) -> SanitizedEncoderInputBatch:
    return sanitize_encoder_inputs(
        windows,
        access_card=access,
        channel_allowlist=channels,
        input_view_digest=_d("reward-free-view"),
    )


def _record(
    windows: TransitionWindowBatch, access: EncoderAccessCard
) -> EncoderProtocolRecord:
    binding = _training_protocol_binding(windows)
    return EncoderProtocolRecord(
        encoder_id="fake-e0",
        family="fake",
        implementation_digest=_d("fake-implementation"),
        input_view_digest=_d("reward-free-view"),
        window_protocol_digest=windows.window_protocol_digest,
        access_card_digest=access.access_card_digest,
        architecture_digest=_d("fake-architecture"),
        objective_digest=_d("fake-objective"),
        training_protocol_digest=binding.training_protocol_digest,
        latent_dim=3,
    )


def _training_protocol_binding(
    windows: TransitionWindowBatch,
) -> EncoderTrainingProtocolBinding:
    return EncoderTrainingProtocolBinding(
        input_view_digest=_d("reward-free-view"),
        window_protocol_digest=windows.window_protocol_digest,
        channel_allowlist=("observation", "action", "next_observation"),
        training_recipe_digest=_d("fake-training-recipe"),
    )


def _role(
    role: str,
    dataset_id: str,
    dataset_digest: str,
    tasks: tuple[str, ...],
    seeds: tuple[str, ...],
    *,
    nonce: str = "fold-c",
) -> DataRoleRecord:
    return DataRoleRecord(
        role=role,  # type: ignore[arg-type]
        dataset_id=dataset_id,
        dataset_digest=dataset_digest,
        task_private_ids=tasks,
        seed_tokens=seeds,
        split_nonce_digest=_d(f"nonce:{nonce}"),
    )


def _bound_data(
    inputs: SanitizedEncoderInputBatch,
    validation_inputs: SanitizedEncoderInputBatch,
    *,
    train_supervision: EncoderSupervisionBatch | None = None,
) -> tuple[EncoderTrainingData, EncoderValidationData, DataRoleManifest, LOTOFoldRecord]:
    train_digest = encoder_dataset_digest(
        inputs, train_supervision, role="source_encoder_train"
    )
    validation_digest = encoder_dataset_digest(
        validation_inputs, None, role="source_encoder_validation"
    )
    train = _role(
        "source_encoder_train",
        "encoder-train",
        train_digest,
        ("task-a", "task-b"),
        ("seed-train",),
    )
    validation = _role(
        "source_encoder_validation",
        "encoder-validation",
        validation_digest,
        ("task-a", "task-b"),
        ("seed-validation",),
    )
    reference = _role(
        "source_reference_spec",
        "reference",
        _d("reference-dataset"),
        ("task-a", "task-b", "task-c"),
        ("seed-reference",),
        nonce="reference",
    )
    query = _role(
        "confirmatory_query",
        "query-c",
        _d("query-dataset"),
        ("task-c",),
        ("seed-query",),
    )
    manifest = DataRoleManifest(
        manifest_id="fold-c-roles", records=(train, validation, reference, query)
    )
    fold = LOTOFoldRecord(
        fold_id="fold-c",
        held_out_task_private_id="task-c",
        train_task_private_ids=("task-a", "task-b"),
        train_dataset_digests=(train.dataset_digest,),
        validation_dataset_digests=(validation.dataset_digest,),
        source_reference_role_digest=manifest.role_digest("source_reference_spec"),
        target_query_role_digest=manifest.role_digest("confirmatory_query"),
        split_nonce_digest=_d("nonce:fold-c"),
    )
    return (
        EncoderTrainingData(
            inputs=inputs,
            supervision=train_supervision,
            role_record=train,
            role_manifest=manifest,
            loto_fold=fold,
            target_query_role="confirmatory_query",
        ),
        EncoderValidationData(
            inputs=validation_inputs,
            supervision=None,
            role_record=validation,
            role_manifest=manifest,
            loto_fold=fold,
            target_query_role="confirmatory_query",
        ),
        manifest,
        fold,
    )


def test_windowing_never_crosses_episodes_and_padding_is_stable() -> None:
    transitions = _transitions()
    windows = build_transition_windows(transitions, _protocol())
    np.testing.assert_array_equal(
        windows.window_indices,
        np.asarray([[0, 1], [2, -1], [3, 4]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        windows.window_mask,
        np.asarray([[True, True], [True, False], [True, True]]),
    )
    assert windows.ordered_episode_window_digest == build_transition_windows(
        transitions, _protocol()
    ).ordered_episode_window_digest
    assert not windows.window_indices.flags.writeable
    assert not transitions.observation.flags.writeable

    with pytest.raises(WindowingError, match="episode boundary"):
        TransitionWindowBatch(
            transitions=transitions,
            window_indices=np.asarray([[2, 3]]),
            window_mask=np.asarray([[True, True]]),
            window_ids=("v03w-00000000000000000000000000000000",),
            protocol=_protocol(),
        )


def test_window_batch_rejects_noncanonical_partition_and_each_forged_id() -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    with pytest.raises(WindowingError, match="partition differs"):
        TransitionWindowBatch(
            transitions=windows.transitions,
            window_indices=windows.window_indices[[0, 2]],
            window_mask=windows.window_mask[[0, 2]],
            window_ids=(windows.window_ids[0], windows.window_ids[2]),
            protocol=windows.protocol,
        )
    forged = list(windows.window_ids)
    forged[1] = "v03w-00000000000000000000000000000000"
    with pytest.raises(WindowingError, match="canonical per-window"):
        TransitionWindowBatch(
            transitions=windows.transitions,
            window_indices=windows.window_indices,
            window_mask=windows.window_mask,
            window_ids=tuple(forged),
            protocol=windows.protocol,
        )


def test_static_and_full_feature_masks_materialize_identically_to_window_layout() -> None:
    static = build_transition_windows(_transitions(), _protocol())
    expected_static = np.ones((3, 2, 2), dtype=np.bool_)
    expected_static[1, 1] = False
    np.testing.assert_array_equal(
        static.materialize_channel("observation_mask"), expected_static
    )

    full_mask = np.asarray(
        [[True, False], [True, True], [False, True], [True, False], [False, False]]
    )
    full = build_transition_windows(
        replace(_transitions(), observation_mask=full_mask), _protocol()
    )
    np.testing.assert_array_equal(
        full.materialize_channel("observation_mask"),
        np.asarray(
            [
                [[True, False], [True, True]],
                [[False, True], [False, False]],
                [[True, False], [False, False]],
            ]
        ),
    )


def test_transition_contract_rejects_nan_and_post_terminal_rows() -> None:
    values = _transitions().observation.copy()
    values[0, 0] = np.nan
    with pytest.raises(WindowingError, match="finite"):
        replace(_transitions(), observation=values)
    with pytest.raises(WindowingError, match="continues"):
        replace(
            _transitions(),
            terminated=np.asarray([False, True, False, False, True]),
        )


def test_sanitized_boundary_hides_reward_metadata_ids_and_digest() -> None:
    access = _access()
    raw = build_transition_windows(_transitions(), _protocol())
    changed_raw = build_transition_windows(_transitions(reward_shift=100.0), _protocol())
    assert raw.window_ids != changed_raw.window_ids
    baseline = _inputs(raw, access)
    changed = _inputs(changed_raw, access)
    assert "reward" not in baseline.channels
    assert not hasattr(baseline, "transitions")
    assert baseline.window_ids == changed.window_ids
    assert baseline.sanitized_input_digest == changed.sanitized_input_digest
    for name in baseline.channel_allowlist:
        np.testing.assert_array_equal(baseline.channel(name), changed.channel(name))

    with pytest.raises(ValueError, match="undeclared"):
        _inputs(raw, access, channels=("observation", "reward"))
    reward_visible = _inputs(
        raw, _access(reward=True), channels=("observation", "reward")
    )
    changed_reward_visible = _inputs(
        changed_raw, _access(reward=True), channels=("observation", "reward")
    )
    assert reward_visible.sanitized_input_digest != changed_reward_visible.sanitized_input_digest


def _encoder_contracts(
    windows: TransitionWindowBatch,
    *,
    train_supervision: EncoderSupervisionBatch | None = None,
):
    access = _access()
    record = _record(windows, access)
    validation_windows = build_transition_windows(
        _transitions(observation_shift=10.0), windows.protocol
    )
    train, validation, manifest, fold = _bound_data(
        _inputs(windows, access),
        _inputs(validation_windows, access),
        train_supervision=train_supervision,
    )
    semantic_protocol = _d("semantic-output")
    contract = EncoderTrainingContract(
        protocol_record=record,
        access_card=access,
        training_protocol_binding=_training_protocol_binding(windows),
        role_manifest=manifest,
        loto_fold=fold,
        target_query_role="confirmatory_query",
        semantic_output_protocol_digest=semantic_protocol,
        runtime_digest=_d("runtime"),
        execution_mode="deterministic-cpu",
        seed=7,
    )
    return access, record, semantic_protocol, contract, train, validation


def _fit_persistable_fake():
    windows = build_transition_windows(_transitions(), _protocol())
    access, record, semantic_protocol, contract, train, validation = (
        _encoder_contracts(windows)
    )
    registry = EncoderRegistry()
    registry.register(
        fake_registration(
            record,
            access,
            semantic_output_protocol_digest=semantic_protocol,
        )
    )
    adapter = registry.create(record.encoder_id)
    adapter_train, adapter_validation, adapter_contract = project_adapter_fit(
        train, validation, contract
    )
    fit = adapter.fit(adapter_train, adapter_validation, adapter_contract)
    return (
        registry,
        adapter,
        fit,
        record,
        semantic_protocol,
        contract,
        train,
        validation,
    )


class _TamperingAdapter:
    """Stable malicious wrapper used to exercise the generic runner."""

    def __init__(self, base, mode: str = "none") -> None:
        self._base = base
        self._mode = mode
        self.encoder_family = base.encoder_family
        self.protocol_record_digest = base.protocol_record_digest
        self.access_card_digest = base.access_card_digest
        self.semantic_output_protocol_digest = base.semantic_output_protocol_digest
        self.saw_sanitized_fit = False

    def fit(self, train, validation, contract):
        self.saw_sanitized_fit = (
            isinstance(train, AdapterFitData)
            and isinstance(validation, AdapterFitData)
            and isinstance(contract, AdapterTrainingContract)
            and not hasattr(train, "role_manifest")
            and not hasattr(validation, "loto_fold")
            and not hasattr(contract, "role_manifest")
            and not hasattr(contract, "loto_fold")
            and not hasattr(contract, "target_query_role")
        )
        fit = self._base.fit(train, validation, contract)
        if self._mode == "fit_manifest":
            return replace(
                fit,
                training_manifest_digest=_d("malicious-training-manifest"),
            )
        if self._mode == "trial_count":
            return replace(
                fit,
                training_cost=CostRecord(
                    wall_seconds=fit.training_cost.wall_seconds,
                    peak_memory_bytes=fit.training_cost.peak_memory_bytes,
                    device=fit.training_cost.device,
                    trial_count=2,
                ),
            )
        return fit

    def load_frozen(self, fit: EncoderFitResult) -> None:
        self._base.load_frozen(fit)

    def encode_windows(self, inputs, *, inference_contract):
        output = self._base.encode_windows(
            inputs, inference_contract=inference_contract
        )
        if self._mode == "wrong_shape":
            return SemanticSampleBatch(
                values=output.values[:, :-1],
                valid_mask=output.valid_mask,
                window_ids=output.window_ids,
            )
        if self._mode == "wrong_ids":
            return SemanticSampleBatch(
                values=output.values,
                valid_mask=output.valid_mask,
                window_ids=tuple(reversed(output.window_ids)),
            )
        return output


def test_fake_adapter_registry_fit_freeze_encode_is_deterministic_and_reward_free() -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    changed_reward_windows = build_transition_windows(
        _transitions(reward_shift=100.0), _protocol()
    )
    access, record, semantic_protocol, training_contract, train, validation = (
        _encoder_contracts(windows)
    )
    registry = EncoderRegistry()
    registration = fake_registration(
        record, access, semantic_output_protocol_digest=semantic_protocol
    )
    registry.register(registration)
    with pytest.raises(EncoderRegistryError, match="duplicate"):
        registry.register(registration)
    assert registry.manifest()["entries"][0]["encoder_id"] == "fake-e0"  # type: ignore[index]

    def inference(fit):
        return EncoderInferenceContract(
            checkpoint_digest=fit.checkpoint_digest,
            input_view_digest=record.input_view_digest,
            window_protocol_digest=record.window_protocol_digest,
            semantic_output_protocol_digest=semantic_protocol,
            runtime_digest=_d("runtime"),
            execution_mode="deterministic-cpu",
            mathematical_dtype="float64",
        )

    adapter = registry.create("fake-e0")
    report = run_adapter_conformance(
        adapter,
        train=train,
        validation=validation,
        training_contract=training_contract,
        inference_contract_factory=inference,
    )
    assert report.passed and report.deterministic

    adapter_train, adapter_validation, adapter_contract = project_adapter_fit(
        train, validation, training_contract
    )
    fit = adapter.fit(adapter_train, adapter_validation, adapter_contract)
    adapter.load_frozen(fit)
    baseline = adapter.encode_windows(train.inputs, inference_contract=inference(fit))
    changed_inputs = _inputs(changed_reward_windows, access)
    changed = adapter.encode_windows(changed_inputs, inference_contract=inference(fit))
    np.testing.assert_allclose(baseline.values, changed.values, atol=0.0, rtol=0.0)
    assert baseline.window_ids == changed.window_ids
    assert baseline.semantic_batch_digest == changed.semantic_batch_digest

    bad = replace(inference(fit), checkpoint_digest=_d("wrong-checkpoint"))
    with pytest.raises(EncoderProtocolError, match="checkpoint"):
        adapter.encode_windows(train.inputs, inference_contract=bad)
    with pytest.raises(EncoderProtocolError, match="SanitizedEncoderInputBatch"):
        adapter.encode_windows(windows, inference_contract=inference(fit))  # type: ignore[arg-type]


def test_adapter_fit_receives_only_projected_data_without_private_loto_objects() -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    access, record, semantic_protocol, contract, train, validation = (
        _encoder_contracts(windows)
    )
    base = fake_registration(
        record,
        access,
        semantic_output_protocol_digest=semantic_protocol,
    ).factory()
    adapter = _TamperingAdapter(base)

    def inference(fit):
        return EncoderInferenceContract(
            checkpoint_digest=fit.checkpoint_digest,
            input_view_digest=record.input_view_digest,
            window_protocol_digest=record.window_protocol_digest,
            semantic_output_protocol_digest=semantic_protocol,
            runtime_digest=contract.runtime_digest,
            execution_mode=contract.execution_mode,
            mathematical_dtype="float64",
        )

    report = run_adapter_conformance(
        adapter,
        train=train,
        validation=validation,
        training_contract=contract,
        inference_contract_factory=inference,
    )
    assert report.passed
    assert adapter.saw_sanitized_fit
    with pytest.raises(EncoderProtocolError, match="sanitized train/validation"):
        base.fit(train, validation, contract)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("fit_manifest", "fit result bindings drifted"),
        ("trial_count", "trial_count exceeds"),
    ],
)
def test_generic_runner_rejects_adapter_controlled_fit_binding_drift(
    mode: str, message: str
) -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    access, record, semantic_protocol, contract, train, validation = (
        _encoder_contracts(windows)
    )
    base = fake_registration(
        record,
        access,
        semantic_output_protocol_digest=semantic_protocol,
    ).factory()
    adapter = _TamperingAdapter(base, mode)

    def inference(fit):
        return EncoderInferenceContract(
            checkpoint_digest=fit.checkpoint_digest,
            input_view_digest=record.input_view_digest,
            window_protocol_digest=record.window_protocol_digest,
            semantic_output_protocol_digest=semantic_protocol,
            runtime_digest=contract.runtime_digest,
            execution_mode=contract.execution_mode,
            mathematical_dtype="float64",
        )

    with pytest.raises(EncoderProtocolError, match=message):
        run_adapter_conformance(
            adapter,
            train=train,
            validation=validation,
            training_contract=contract,
            inference_contract_factory=inference,
        )


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("wrong_shape", "output shape"),
        ("wrong_ids", "reordered or replaced window IDs"),
    ],
)
def test_generic_runner_rejects_stable_wrong_semantic_outputs(
    mode: str, message: str
) -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    access, record, semantic_protocol, contract, train, validation = (
        _encoder_contracts(windows)
    )
    base = fake_registration(
        record,
        access,
        semantic_output_protocol_digest=semantic_protocol,
    ).factory()
    adapter = _TamperingAdapter(base, mode)

    def inference(fit):
        return EncoderInferenceContract(
            checkpoint_digest=fit.checkpoint_digest,
            input_view_digest=record.input_view_digest,
            window_protocol_digest=record.window_protocol_digest,
            semantic_output_protocol_digest=semantic_protocol,
            runtime_digest=contract.runtime_digest,
            execution_mode=contract.execution_mode,
            mathematical_dtype="float64",
        )

    with pytest.raises(EncoderProtocolError, match=message):
        run_adapter_conformance(
            adapter,
            train=train,
            validation=validation,
            training_contract=contract,
            inference_contract_factory=inference,
        )


def test_unknown_input_view_fails_through_training_protocol_binding() -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    _, _, _, contract, _, _ = _encoder_contracts(windows)
    unknown_binding = replace(
        contract.training_protocol_binding,
        input_view_digest=_d("posthoc-unknown-view"),
    )
    with pytest.raises(EncoderProtocolError, match="training protocol binding"):
        replace(contract, training_protocol_binding=unknown_binding)


def test_training_data_rejects_forged_role_digest_manifest_and_fold() -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    access, _, _, contract, train, validation = _encoder_contracts(windows)
    forged_record = replace(train.role_record, dataset_digest=_d("forged-dataset"))
    forged_manifest = DataRoleManifest(
        "forged-roles",
        tuple(
            forged_record if record.role == "source_encoder_train" else record
            for record in train.role_manifest.records
        ),
    )
    forged_fold = replace(
        train.loto_fold,
        train_dataset_digests=(forged_record.dataset_digest,),
        source_reference_role_digest=forged_manifest.role_digest(
            "source_reference_spec"
        ),
        target_query_role_digest=forged_manifest.role_digest("confirmatory_query"),
    )
    with pytest.raises(EncoderProtocolError, match="dataset digest"):
        EncoderTrainingData(
            inputs=train.inputs,
            supervision=None,
            role_record=forged_record,
            role_manifest=forged_manifest,
            loto_fold=forged_fold,
            target_query_role="confirmatory_query",
        )

    other_contract = replace(contract, seed=99)
    drifted_validation = replace(
        validation, loto_fold=replace(validation.loto_fold, fold_id="fold-other")
    )
    with pytest.raises(EncoderProtocolError, match="validation LOTO fold"):
        project_adapter_fit(train, drifted_validation, other_contract)


def test_encoder_must_be_loaded_before_inference() -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    access, record, semantic_protocol, _, train, _ = _encoder_contracts(windows)
    adapter = fake_registration(
        record, access, semantic_output_protocol_digest=semantic_protocol
    ).factory()
    with pytest.raises(EncoderProtocolError, match="not been loaded"):
        adapter.encode_windows(
            train.inputs,
            inference_contract=EncoderInferenceContract(
                checkpoint_digest=_d("checkpoint"),
                input_view_digest=record.input_view_digest,
                window_protocol_digest=record.window_protocol_digest,
                semantic_output_protocol_digest=semantic_protocol,
                runtime_digest=_d("runtime"),
                execution_mode="deterministic-cpu",
                mathematical_dtype="float64",
            ),
        )


def test_supervision_cannot_bypass_access_card() -> None:
    windows = build_transition_windows(_transitions(), _protocol())
    labels = EncoderSupervisionBatch(
        {"task_label": np.arange(3, dtype=np.int64)}
    )
    access, record, semantic_protocol, training_contract, labeled, validation = (
        _encoder_contracts(windows, train_supervision=labels)
    )
    with pytest.raises(ValueError, match="undeclared"):
        project_adapter_fit(labeled, validation, training_contract)


def test_fake_checkpoint_fresh_reload_reaches_encoder_neutral_rkme_selector(
    tmp_path: Path,
) -> None:
    (
        registry,
        adapter,
        fit,
        record,
        semantic_protocol,
        contract,
        _train,
        validation,
    ) = _fit_persistable_fake()
    assert isinstance(adapter, ImmutableEncoderCheckpointAdapter)
    checkpoint_bytes = adapter.export_frozen_checkpoint_bytes(fit)
    assert sha256_bytes(checkpoint_bytes) == fit.checkpoint_digest

    inference = EncoderInferenceContract(
        checkpoint_digest=fit.checkpoint_digest,
        input_view_digest=record.input_view_digest,
        window_protocol_digest=record.window_protocol_digest,
        semantic_output_protocol_digest=semantic_protocol,
        runtime_digest=contract.runtime_digest,
        execution_mode=contract.execution_mode,
        mathematical_dtype="float64",
    )
    adapter.load_frozen(fit)
    before_persistence = adapter.encode_windows(
        validation.inputs,
        inference_contract=inference,
    )

    layout = V03ArtifactLayout.development(tmp_path, "synthetic-p3-pipeline")
    checkpoint_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, fit.encoder_id, "checkpoint.bin"
    )
    manifest_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, fit.encoder_id, "checkpoint_manifest.json"
    )
    publication = freeze_encoder_checkpoint(
        writer=layout.writer("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        checkpoint_bytes=checkpoint_bytes,
        fit=fit,
        protocol_record=record,
        fold_id=fit.fold_id,
        seed=fit.seed,
        runtime_digest=fit.runtime_digest,
        semantic_output_protocol_digest=semantic_protocol,
    )
    assert publication.checkpoint_artifact_digest == fit.checkpoint_digest
    assert publication.manifest.checkpoint_digest == fit.checkpoint_digest

    loaded_manifest, loaded_bytes = load_frozen_encoder_checkpoint(
        reader=layout.reader("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        expected_manifest_artifact_digest=publication.manifest_artifact_digest,
        expected_protocol_record=record,
        expected_fold_id=fit.fold_id,
        expected_seed=fit.seed,
        expected_training_manifest_digest=fit.training_manifest_digest,
        expected_training_contract_digest=fit.training_contract_digest,
        expected_runtime_digest=fit.runtime_digest,
        expected_semantic_output_protocol_digest=semantic_protocol,
    )
    assert loaded_manifest.checkpoint_manifest_digest == (
        publication.manifest.checkpoint_manifest_digest
    )
    assert sha256_bytes(loaded_bytes) == loaded_manifest.checkpoint_digest

    fresh_adapter = registry.create(record.encoder_id)
    assert fresh_adapter is not adapter
    assert isinstance(fresh_adapter, ImmutableEncoderCheckpointAdapter)
    restored_fit = fresh_adapter.load_frozen_checkpoint_bytes(
        manifest=loaded_manifest,
        checkpoint_bytes=loaded_bytes,
    )
    assert restored_fit == fit
    after_reload = fresh_adapter.encode_windows(
        validation.inputs,
        inference_contract=inference,
    )
    assert after_reload.semantic_batch_digest == before_persistence.semantic_batch_digest
    np.testing.assert_array_equal(after_reload.values, before_persistence.values)

    far_windows = build_transition_windows(
        _transitions(observation_shift=50.0), _protocol()
    )
    far_inputs = _inputs(far_windows, contract.access_card)
    far_output = fresh_adapter.encode_windows(
        far_inputs,
        inference_contract=inference,
    )

    def semantic_cache(
        label: str,
        output: SemanticSampleBatch,
        inputs: SanitizedEncoderInputBatch,
    ) -> SemanticCacheRecord:
        # Downstream representation code sees only the common semantic batch
        # and frozen protocol digests; it has no fake/CORRO-specific branch.
        key = SemanticCacheKey(
            raw_dataset_digest=_d(f"raw:{label}"),
            ordered_episode_window_digest=inputs.sanitized_input_digest,
            canonical_view_digest=record.input_view_digest,
            window_protocol_digest=record.window_protocol_digest,
            normalizer_digest=_d("identity-normalizer"),
            semantic_transform=SemanticTransform.frozen_encoder(
                encoder_implementation_digest=record.implementation_digest,
                checkpoint_digest=loaded_manifest.checkpoint_digest,
                semantic_output_protocol_digest=semantic_protocol,
            ),
            mathematical_dtype_digest=FLOAT64_MATHEMATICAL_DTYPE_DIGEST,
        )
        return SemanticCacheRecord(
            key=key,
            points=output.values[output.valid_mask],
            episode_offsets=np.asarray([0, 2, 3], dtype=np.int64),
        )

    query_cache = semantic_cache("query", after_reload, validation.inputs)
    close_cache = semantic_cache("close", after_reload, validation.inputs)
    far_cache = semantic_cache("far", far_output, far_inputs)
    reducer = ReducerConfig(
        support_budget=3,
        support_steps=0,
        kmeans_steps=0,
        ridge=0.0,
        pinv_rcond=1.0e-12,
    )
    measurement_protocol = _d("synthetic-measurement")

    def source(cache: SemanticCacheRecord):
        return build_source_reduced_spec(
            cache,
            kernel_bandwidth=1.0,
            measurement_protocol_id=measurement_protocol,
            probe_dataset_digest=cache.key.raw_dataset_digest,
            reducer_config=reducer,
        )

    sources = {
        "lw-close": source(close_cache),
        "lw-far": source(far_cache),
    }
    query = build_empirical_query_spec(
        query_cache,
        kernel_bandwidth=1.0,
        measurement_protocol_id=measurement_protocol,
        probe_dataset_digest=query_cache.key.raw_dataset_digest,
    )
    index = SourceRepresentationIndex(
        query_cache.key.representation_protocol_digest,
        sources,
    )
    tie_tokens = {
        "lw-close": _d("tie:close"),
        "lw-far": _d("tie:far"),
    }
    ranking = RankingKey(
        query_spec_digest=str(query.query_spec_digest),
        representation_index_digest=str(index.representation_index_digest),
        selector_digest=_d("encoder-neutral-joint-distance-selector"),
        tie_break_digest=tie_break_digest(tie_tokens),
    )
    run = run_joint_distance_stage(
        JointDistanceRequest(
            query_spec=query,
            source_index=index,
            ranking_key=ranking,
            tie_break_tokens=tie_tokens,
            block_size=2,
        )
    )
    assert run.rows[0].opaque_learnware_id == "lw-close"
    assert run.rows[0].result.squared_distance == pytest.approx(0.0, abs=1.0e-10)
    assert run.rows[1].opaque_learnware_id == "lw-far"


def test_fake_checkpoint_fresh_reload_fails_closed_on_bytes_and_manifest(
    tmp_path: Path,
) -> None:
    (
        registry,
        adapter,
        fit,
        record,
        semantic_protocol,
        _contract,
        _train,
        _validation,
    ) = _fit_persistable_fake()
    assert isinstance(adapter, ImmutableEncoderCheckpointAdapter)
    checkpoint_bytes = adapter.export_frozen_checkpoint_bytes(fit)
    layout = V03ArtifactLayout.development(tmp_path, "synthetic-p3-negative")
    checkpoint_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, fit.encoder_id, "checkpoint.bin"
    )
    manifest_path = layout.encoder_checkpoint_artifact(
        fit.fold_id, fit.encoder_id, "checkpoint_manifest.json"
    )
    publication = freeze_encoder_checkpoint(
        writer=layout.writer("encoder_checkpoints"),
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        checkpoint_bytes=checkpoint_bytes,
        fit=fit,
        protocol_record=record,
        fold_id=fit.fold_id,
        seed=fit.seed,
        runtime_digest=fit.runtime_digest,
        semantic_output_protocol_digest=semantic_protocol,
    )
    fresh_adapter = registry.create(record.encoder_id)
    assert isinstance(fresh_adapter, ImmutableEncoderCheckpointAdapter)

    tampered_bytes = checkpoint_bytes[:-1] + b"]"
    with pytest.raises(EncoderProtocolError, match="bytes digest"):
        fresh_adapter.load_frozen_checkpoint_bytes(
            manifest=publication.manifest,
            checkpoint_bytes=tampered_bytes,
        )

    drifted_manifest = replace(
        publication.manifest,
        training_contract_digest=_d("tampered-training-contract"),
    )
    with pytest.raises(EncoderProtocolError, match="payload differs from manifest"):
        fresh_adapter.load_frozen_checkpoint_bytes(
            manifest=drifted_manifest,
            checkpoint_bytes=checkpoint_bytes,
        )

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        load_frozen_encoder_checkpoint(
            reader=layout.reader("encoder_checkpoints"),
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
            expected_manifest_artifact_digest=_d("forged-manifest-artifact"),
            expected_protocol_record=record,
            expected_fold_id=fit.fold_id,
            expected_seed=fit.seed,
            expected_training_manifest_digest=fit.training_manifest_digest,
            expected_training_contract_digest=fit.training_contract_digest,
            expected_runtime_digest=fit.runtime_digest,
            expected_semantic_output_protocol_digest=semantic_protocol,
        )
