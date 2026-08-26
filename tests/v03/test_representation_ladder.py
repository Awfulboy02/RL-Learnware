from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_bytes, sha256_json, sha256_ndarrays
from policy_learnware_v0.v03.representation_ladder import (
    R0_PADDED_RAW,
    R1_FIXED_RANDOM_LINEAR,
    R2_SOURCE_PCA_WHITEN,
    R3_MATCHED_RANDOM_MLP,
    R4_ARCHIVED_FROZEN_CORRO,
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    R_HIST_RANDOM_TANH,
    RepresentationBatch,
    RepresentationLadderError,
    RepresentationManifest,
    TrainedCallableArtifact,
    bind_historical_random_tanh,
    bind_r4_frozen_callable,
    fit_r0_identity,
    fit_r1_random_linear,
    fit_r2_pca_whitening,
    fit_r3_matched_random_mlp,
    fit_r5_corro_style,
    fit_r5l_supervised_linear,
)
from policy_learnware_v0.v03.signal_controls import HistoricalRandomTanhSpec


def _d(label: str) -> str:
    return sha256_json({"test": label})


def _source(offset: float = 0.0) -> RepresentationBatch:
    values = np.asarray(
        [
            [0.0, 1.0, 2.0, -1.0],
            [1.0, 0.5, -1.0, 2.0],
            [2.0, -1.0, 0.5, 1.0],
            [-1.0, 2.0, 1.0, 0.0],
            [0.5, -0.5, 2.5, 1.5],
            [1.5, 1.5, -0.5, -1.5],
        ],
        dtype=np.float64,
    )
    values += offset
    return RepresentationBatch(values, _d(f"source-{offset}"), "SOURCE_FIT")


def _query() -> RepresentationBatch:
    return RepresentationBatch(
        np.asarray([[0.25, -0.5, 1.25, 0.0], [1.0, 1.0, 1.0, 1.0]]),
        _d("query"),
        "QUERY_TRANSFORM",
    )


def test_r0_r1_are_deterministic_and_r1_is_strictly_linear_without_bias() -> None:
    source = _source()
    query = _query()

    r0 = fit_r0_identity(source)
    assert r0.manifest.representation_id == R0_PADDED_RAW
    assert r0.manifest.checkpoint_digest is None
    np.testing.assert_array_equal(r0.transform(query).values, query.values)
    assert not r0.transform(query).values.flags.writeable

    first = fit_r1_random_linear(source, output_dim=3, seed=7)
    second = fit_r1_random_linear(source, output_dim=3, seed=7)
    changed = fit_r1_random_linear(source, output_dim=3, seed=8)
    assert first.manifest.representation_id == R1_FIXED_RANDOM_LINEAR
    assert first.manifest.coordinate_digest == second.manifest.coordinate_digest
    assert first.manifest.coordinate_digest != changed.manifest.coordinate_digest
    np.testing.assert_array_equal(
        first.transform(query).values, second.transform(query).values
    )
    zeros = RepresentationBatch(np.zeros((2, 4)), _d("zeros"), "QUERY_TRANSFORM")
    np.testing.assert_array_equal(first.transform(zeros).values, np.zeros((2, 3)))
    with pytest.raises(RepresentationLadderError, match="seed"):
        fit_r1_random_linear(source, output_dim=3, seed=True)  # type: ignore[arg-type]


def test_r2_is_source_only_fixed_sign_and_whitened() -> None:
    source = _source()
    first = fit_r2_pca_whitening(source, output_dim=3, whiten=True)
    second = fit_r2_pca_whitening(source, output_dim=3, whiten=True)
    assert first.manifest.representation_id == R2_SOURCE_PCA_WHITEN
    assert first.manifest.coordinate_digest == second.manifest.coordinate_digest
    transformed = first.transform(source).values
    np.testing.assert_allclose(np.mean(transformed, axis=0), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.cov(transformed, rowvar=False), np.eye(3), atol=1.0e-10
    )
    query_result = first.transform(_query())
    assert query_result.values.shape == (2, 3)
    assert first.manifest.source_fit_digest == source.batch_digest

    changed_source = fit_r2_pca_whitening(_source(0.25), output_dim=3)
    assert first.manifest.source_fit_digest != changed_source.manifest.source_fit_digest
    assert first.manifest.coordinate_digest != changed_source.manifest.coordinate_digest


def test_r3_and_historical_random_tanh_have_separate_identity_and_geometry() -> None:
    source = _source()
    query = _query()
    r3 = fit_r3_matched_random_mlp(
        source, output_dim=5, hidden_dims=(7, 6), seed=11
    )
    repeated = fit_r3_matched_random_mlp(
        source, output_dim=5, hidden_dims=(7, 6), seed=11
    )
    assert r3.manifest.representation_id == R3_MATCHED_RANDOM_MLP
    assert r3.manifest.coordinate_digest == repeated.manifest.coordinate_digest
    values = r3.transform(query).values
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1.0e-12)

    spec = HistoricalRandomTanhSpec.create(seed=11, input_dim=4, output_dim=5)
    historical = bind_historical_random_tanh(source, spec=spec)
    assert historical.manifest.representation_id == R_HIST_RANDOM_TANH
    assert historical.manifest.architecture == "SINGLE_AFFINE_TANH"
    assert historical.manifest.protocol_digest != r3.manifest.protocol_digest
    assert historical.manifest.params_digest != r3.manifest.params_digest
    assert historical.manifest.checkpoint_digest != r3.manifest.checkpoint_digest
    assert historical.manifest.coordinate_digest != r3.manifest.coordinate_digest
    assert historical.manifest.protocol_digest == spec.representation_protocol_digest
    assert historical.manifest.params_digest == spec.parameter_digest
    assert historical.manifest.checkpoint_digest == spec.checkpoint_digest


def test_r4_is_checkpoint_bound_and_manifest_roundtrip_is_strict() -> None:
    source = _source()
    weight = np.arange(12, dtype=np.float64).reshape(4, 3) / 20.0

    def transform(values: np.ndarray) -> np.ndarray:
        return values @ weight

    first = bind_r4_frozen_callable(
        source,
        output_dim=3,
        checkpoint_digest=_d("checkpoint-a"),
        normalizer_digest=_d("normalizer"),
        implementation_digest=_d("legacy-implementation"),
        transform=transform,
    )
    second = bind_r4_frozen_callable(
        source,
        output_dim=3,
        checkpoint_digest=_d("checkpoint-b"),
        normalizer_digest=_d("normalizer"),
        implementation_digest=_d("legacy-implementation"),
        transform=transform,
    )
    assert first.manifest.representation_id == R4_ARCHIVED_FROZEN_CORRO
    assert first.manifest.coordinate_digest != second.manifest.coordinate_digest
    assert RepresentationManifest.from_dict(first.manifest.to_dict()) == first.manifest

    tampered = first.manifest.to_dict()
    tampered["output_dim"] = 4
    with pytest.raises(RepresentationLadderError, match="coordinate_digest"):
        RepresentationManifest.from_dict(tampered)
    unknown = first.manifest.to_dict()
    unknown["checkpoint_path"] = "/tmp/not-in-contract"
    with pytest.raises(RepresentationLadderError, match="unknown"):
        RepresentationManifest.from_dict(unknown)


def _fake_trainer(
    source_values: np.ndarray, labels: np.ndarray, request
) -> TrainedCallableArtifact:
    # The trainer sees only the already-validated source arrays and labels.
    assert source_values.shape[0] == labels.shape[0]
    rng = np.random.default_rng(request.seed)
    weight = rng.normal(size=(request.input_dim, request.output_dim))
    arrays_digest = sha256_ndarrays({"weight": weight})

    def transform(values: np.ndarray) -> np.ndarray:
        raw = values @ weight
        return raw / np.maximum(
            np.linalg.norm(raw, axis=1, keepdims=True), np.finfo(np.float64).eps
        )

    return TrainedCallableArtifact(
        checkpoint_bytes=(request.representation_id + arrays_digest).encode("utf-8"),
        parameter_digest=arrays_digest,
        trainer_implementation_digest=_d("fake-trainer"),
        transform=transform,
    )


def test_r5_and_r5l_injected_trainers_are_source_only_and_digest_isolated() -> None:
    source = _source()
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    r5 = fit_r5_corro_style(
        source,
        labels=labels,
        trainer=_fake_trainer,
        objective_digest=_d("task-supcon"),
        seed=3,
        output_dim=3,
        hidden_dims=(8, 8),
    )
    r5l = fit_r5l_supervised_linear(
        source,
        labels=labels,
        trainer=_fake_trainer,
        objective_digest=_d("task-supcon"),
        seed=3,
        output_dim=3,
    )
    assert r5.manifest.representation_id == R5_VIEW_SPECIFIC_CORRO_REFIT
    assert r5l.manifest.representation_id == R5L_SUPERVISED_LINEAR
    assert r5.manifest.protocol_digest != r5l.manifest.protocol_digest
    assert r5.manifest.checkpoint_digest != r5l.manifest.checkpoint_digest
    assert r5.transform(_query()).values.shape == (2, 3)
    assert r5l.transform(_query()).values.shape == (2, 3)

    changed_labels = np.asarray([0, 1, 0, 1, 2, 2], dtype=np.int64)
    changed = fit_r5_corro_style(
        source,
        labels=changed_labels,
        trainer=_fake_trainer,
        objective_digest=_d("task-supcon"),
        seed=3,
        output_dim=3,
        hidden_dims=(8, 8),
    )
    assert changed.manifest.source_fit_digest != r5.manifest.source_fit_digest
    assert changed.manifest.coordinate_digest != r5.manifest.coordinate_digest

    query = _query()
    with pytest.raises(RepresentationLadderError, match="SOURCE_FIT"):
        fit_r5_corro_style(
            query,
            labels=np.asarray([0, 1]),
            trainer=_fake_trainer,
            objective_digest=_d("task-supcon"),
            seed=3,
            output_dim=3,
            hidden_dims=(8, 8),
        )


def test_every_fit_entry_rejects_query_role_and_coordinate_bindings_are_independent() -> None:
    query = _query()
    for call in (
        lambda: fit_r0_identity(query),
        lambda: fit_r1_random_linear(query, output_dim=2, seed=0),
        lambda: fit_r2_pca_whitening(query, output_dim=2),
        lambda: fit_r3_matched_random_mlp(
            query, output_dim=2, hidden_dims=(3, 3), seed=0
        ),
    ):
        with pytest.raises(RepresentationLadderError, match="SOURCE_FIT"):
            call()

    base = fit_r1_random_linear(_source(), output_dim=3, seed=4).manifest
    changed_protocol = replace(base, protocol_digest=_d("changed-protocol"), coordinate_digest=None)
    changed_params = replace(base, params_digest=_d("changed-params"), coordinate_digest=None)
    changed_checkpoint = replace(
        base, checkpoint_digest=_d("changed-checkpoint"), coordinate_digest=None
    )
    changed_source = replace(
        base, source_fit_digest=_d("changed-source-fit"), coordinate_digest=None
    )
    observed = {
        base.coordinate_digest,
        changed_protocol.coordinate_digest,
        changed_params.coordinate_digest,
        changed_checkpoint.coordinate_digest,
        changed_source.coordinate_digest,
    }
    assert len(observed) == 5
    assert sha256_bytes(b"checkpoint") != base.checkpoint_digest
