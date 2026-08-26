from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from policy_learnware_v0.hashing import sha256_json
from policy_learnware_v0.v03.corro_trainers import (
    SOURCE_REPRESENTATION_TRAIN,
    SOURCE_REPRESENTATION_VALIDATION,
    TASK_SUPCON_OBJECTIVE_DIGEST,
    CorroBackendResult,
    CorroOptimizationConfig,
    CorroSourceSplit,
    CorroTaskDataset,
    CorroTrainerAdapter,
    CorroTrainerDependencyError,
    CorroTrainerError,
)
from policy_learnware_v0.v03.representation_ladder import (
    R5L_SUPERVISED_LINEAR,
    R5_VIEW_SPECIFIC_CORRO_REFIT,
    RepresentationBatch,
    RepresentationLadderError,
    TrainingRequest,
    fit_r5_corro_style,
    fit_r5l_supervised_linear,
    restore_trained_representation,
)


def _digest(label: str) -> str:
    return sha256_json({"test": label})


def _task(task_id: str, offset: float) -> CorroTaskDataset:
    values = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    return CorroTaskDataset(
        task_id=task_id,
        packed=values + offset,
        episode_offsets=np.asarray([0, 2, 4], dtype=np.int64),
    )


def _splits(
    *, validation_offset: float = 20.0
) -> tuple[CorroSourceSplit, CorroSourceSplit]:
    # Deliberately pass tasks in reverse order to exercise canonical ordering.
    train = CorroSourceSplit(
        SOURCE_REPRESENTATION_TRAIN,
        (_task("walker", 10.0), _task("finger", 0.0)),
    )
    validation = CorroSourceSplit(
        SOURCE_REPRESENTATION_VALIDATION,
        (
            _task("walker", validation_offset + 10.0),
            _task("finger", validation_offset),
        ),
    )
    return train, validation


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def _result(self, request):
        rng = np.random.default_rng(request.seed)
        weight = rng.normal(size=(request.input_dim, request.output_dim))

        def transform(values: np.ndarray) -> np.ndarray:
            projected = np.asarray(values, dtype=np.float64) @ weight
            return projected / np.maximum(
                np.linalg.norm(projected, axis=1, keepdims=True), 1.0e-12
            )

        return CorroBackendResult(
            checkpoint_bytes=(request.representation_id + ":fake").encode(),
            implementation_digest=_digest("fake-corro-backend"),
            transform=transform,
        )

    def train(self, *, train, validation, request, optimization):
        self.calls.append(
            {
                "train": train,
                "validation": validation,
                "request": request,
                "optimization": optimization,
            }
        )
        return self._result(request)

    def restore(self, *, checkpoint_bytes, request, optimization):
        del optimization
        result = self._result(request)
        if result.checkpoint_bytes != checkpoint_bytes:
            raise CorroTrainerError("fake checkpoint mismatch")
        return result


def _adapter(backend=None, *, validation_offset: float = 20.0):
    train, validation = _splits(validation_offset=validation_offset)
    return CorroTrainerAdapter(
        train,
        validation,
        CorroOptimizationConfig(
            batch_size=4,
            train_steps=0,
            validation_interval=1,
            validation_batches=1,
        ),
        backend,
    )


def _source(split: CorroSourceSplit) -> RepresentationBatch:
    return RepresentationBatch(
        split.flattened_values(), _digest("registered-source"), "SOURCE_FIT"
    )


def _request(*, objective_digest: str = TASK_SUPCON_OBJECTIVE_DIGEST):
    return TrainingRequest(
        representation_id=R5_VIEW_SPECIFIC_CORRO_REFIT,
        input_dim=3,
        output_dim=2,
        hidden_dims=(5, 4),
        activation="relu",
        l2_normalize_output=True,
        objective_digest=objective_digest,
        seed=7,
    )


def test_source_splits_are_episode_aware_canonical_and_digest_bound() -> None:
    train, validation = _splits()
    assert train.task_ids == ("finger", "walker")
    assert train.flattened_values().shape == (8, 3)
    np.testing.assert_array_equal(
        train.flattened_task_names(),
        ["finger"] * 4 + ["walker"] * 4,
    )
    np.testing.assert_array_equal(
        train.flattened_task_indices(), [0] * 4 + [1] * 4
    )
    assert train.split_digest != validation.split_digest
    assert not train.flattened_values().flags.writeable

    with pytest.raises(CorroTrainerError, match="at least two non-empty episodes"):
        CorroTaskDataset("bad", np.ones((4, 3)), np.asarray([0, 4]))
    with pytest.raises(CorroTrainerError, match="canonical input width"):
        CorroSourceSplit(
            SOURCE_REPRESENTATION_TRAIN,
            (_task("finger", 0.0), CorroTaskDataset("walker", np.ones((4, 2)), [0, 2, 4])),
        )


def test_r5_and_r5l_use_registered_source_only_and_matched_semantics() -> None:
    fake = _FakeBackend()
    adapter = _adapter(fake)
    train = adapter.train_split
    source = _source(train)
    query = RepresentationBatch(
        np.asarray([[0.5, 1.5, 2.5], [2.0, 0.0, 1.0]]),
        _digest("query"),
        "QUERY_TRANSFORM",
    )

    r5 = fit_r5_corro_style(
        source,
        labels=train.flattened_task_names(),
        trainer=adapter,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=7,
        output_dim=2,
        hidden_dims=(5, 4),
    )
    r5l = fit_r5l_supervised_linear(
        source,
        labels=train.flattened_task_indices(),
        trainer=adapter,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=7,
        output_dim=2,
    )

    assert [
        call["request"].representation_id for call in fake.calls  # type: ignore[union-attr]
    ] == [R5_VIEW_SPECIFIC_CORRO_REFIT, R5L_SUPERVISED_LINEAR]
    assert fake.calls[0]["request"].hidden_dims == (5, 4)  # type: ignore[union-attr]
    assert fake.calls[1]["request"].hidden_dims == ()  # type: ignore[union-attr]
    assert tuple(fake.calls[0]["train"]) == ("finger", "walker")
    assert tuple(fake.calls[0]["validation"]) == ("finger", "walker")
    assert r5.transform(query).values.shape == (2, 2)
    assert r5l.transform(query).values.shape == (2, 2)
    assert r5.manifest.coordinate_digest != r5l.manifest.coordinate_digest
    # The backend has no query argument and receives only the registered splits.
    assert all(
        sum(task.packed.shape[0] for task in call["train"].values()) == 8  # type: ignore[union-attr]
        for call in fake.calls
    )


def test_adapter_fails_closed_on_query_rows_labels_objective_and_split_overlap() -> None:
    fake = _FakeBackend()
    adapter = _adapter(fake)
    train = adapter.train_split
    request = _request()

    with pytest.raises(CorroTrainerError, match="query/development rows are forbidden"):
        adapter(
            train.flattened_values() + 0.25,
            train.flattened_task_names(),
            request,
        )
    with pytest.raises(CorroTrainerError, match="categorical task blocks"):
        adapter(
            train.flattened_values(),
            np.asarray(["wrong"] * train.row_count),
            request,
        )
    with pytest.raises(CorroTrainerError, match="frozen episode-aware"):
        adapter(
            train.flattened_values(),
            train.flattened_task_names(),
            _request(objective_digest=_digest("different-objective")),
        )
    assert fake.calls == []

    validation_reusing_train = CorroSourceSplit(
        SOURCE_REPRESENTATION_VALIDATION, train.tasks
    )
    with pytest.raises(CorroTrainerError, match="must be distinct"):
        CorroTrainerAdapter(
            train,
            validation_reusing_train,
            CorroOptimizationConfig(batch_size=4, train_steps=0),
            fake,
        )


def test_formal_contract_forbids_injected_backend_without_starting_training() -> None:
    injected = _adapter(_FakeBackend())
    request = _request()
    with pytest.raises(CorroTrainerError, match="forbids injected"):
        injected.formal_contract(
            request=request,
            source_fit_batch_digest=_digest("formal-source-fit"),
            expected_train_split=injected.train_split,
            expected_validation_split=injected.validation_split,
            expected_optimization=injected.optimization,
        )

    production = _adapter()
    contract = production.formal_contract(
        request=request,
        source_fit_batch_digest=_digest("formal-source-fit"),
        expected_train_split=production.train_split,
        expected_validation_split=production.validation_split,
        expected_optimization=production.optimization,
    )
    assert contract.optimization_digest == production.optimization.optimization_digest
    assert contract.training_request_digest == request.request_digest


def test_validation_and_optimization_are_part_of_parameter_coordinates() -> None:
    first_backend = _FakeBackend()
    second_backend = _FakeBackend()
    first = _adapter(first_backend, validation_offset=20.0)
    second = _adapter(second_backend, validation_offset=30.0)
    source = _source(first.train_split)

    first_fit = fit_r5_corro_style(
        source,
        labels=first.train_split.flattened_task_names(),
        trainer=first,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=7,
        output_dim=2,
        hidden_dims=(5, 4),
    )
    second_fit = fit_r5_corro_style(
        source,
        labels=second.train_split.flattened_task_names(),
        trainer=second,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=7,
        output_dim=2,
        hidden_dims=(5, 4),
    )
    # Fake checkpoint bytes are deliberately identical; validation still changes
    # the fitted coordinate through the adapter's parameter binding.
    assert first_fit.manifest.checkpoint_digest == second_fit.manifest.checkpoint_digest
    assert first_fit.manifest.params_digest != second_fit.manifest.params_digest
    assert first_fit.manifest.coordinate_digest != second_fit.manifest.coordinate_digest


def test_r5_checkpoint_bytes_restore_exact_frozen_transform() -> None:
    fake = _FakeBackend()
    adapter = _adapter(fake)
    train = adapter.train_split
    source = _source(train)
    request = _request()
    fitted = fit_r5_corro_style(
        source,
        labels=train.flattened_task_names(),
        trainer=adapter,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=request.seed,
        output_dim=request.output_dim,
        hidden_dims=request.hidden_dims,  # type: ignore[arg-type]
    )
    assert fitted.checkpoint_bytes is not None
    restored = restore_trained_representation(
        manifest=fitted.manifest,
        checkpoint_bytes=fitted.checkpoint_bytes,
        request=request,
        restorer=adapter,
        verification_source=source,
        labels=train.flattened_task_names(),
    )
    query = RepresentationBatch(
        np.asarray([[0.25, 0.5, 0.75]], dtype=np.float64),
        _digest("restore-query"),
        "QUERY_TRANSFORM",
    )
    np.testing.assert_array_equal(
        fitted.transform(query).values, restored.transform(query).values
    )
    with pytest.raises(RepresentationLadderError, match="checkpoint bytes"):
        restore_trained_representation(
            manifest=fitted.manifest,
            checkpoint_bytes=fitted.checkpoint_bytes + b"tamper",
            request=request,
            restorer=adapter,
            verification_source=source,
            labels=train.flattened_task_names(),
        )


def test_module_import_and_disabled_configuration_do_not_import_jax_stack() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = r'''
import builtins
original = builtins.__import__
blocked = ("jax", "jaxlib", "flax", "optax")
def guarded(name, *args, **kwargs):
    if name == "policy_learnware_v0.representation.encoder" or name.startswith(blocked):
        raise AssertionError("optional training dependency imported: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from policy_learnware_v0.v03.corro_trainers import (
    CorroOptimizationConfig, CorroSourceSplit, CorroTaskDataset,
    CorroTrainerAdapter, SOURCE_REPRESENTATION_TRAIN,
    SOURCE_REPRESENTATION_VALIDATION,
)
import numpy as np
def task(name, offset):
    return CorroTaskDataset(name, np.arange(12).reshape(4, 3) + offset, [0, 2, 4])
train = CorroSourceSplit(SOURCE_REPRESENTATION_TRAIN, (task("a", 0), task("b", 20)))
valid = CorroSourceSplit(SOURCE_REPRESENTATION_VALIDATION, (task("a", 40), task("b", 60)))
CorroTrainerAdapter(train, valid, CorroOptimizationConfig(batch_size=4, train_steps=0))
print("lazy-ok")
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "lazy-ok"


def test_default_backend_reports_optional_dependency_only_when_fit_is_requested() -> None:
    if all(importlib.util.find_spec(name) is not None for name in ("jax", "flax", "optax")):
        pytest.skip("optional JAX training stack is installed")
    adapter = _adapter()
    train = adapter.train_split
    with pytest.raises(CorroTrainerDependencyError, match="optional JAX"):
        adapter(
            train.flattened_values(),
            train.flattened_task_names(),
            _request(),
        )


@pytest.mark.skipif(
    not all(importlib.util.find_spec(name) is not None for name in ("jax", "flax", "optax")),
    reason="optional JAX training stack is not installed",
)
def test_real_r5_and_r5l_backends_initialize_without_large_training() -> None:
    adapter = _adapter()
    train = adapter.train_split
    source = _source(train)
    r5 = fit_r5_corro_style(
        source,
        labels=train.flattened_task_names(),
        trainer=adapter,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=2,
        output_dim=2,
        hidden_dims=(5, 4),
    )
    r5l = fit_r5l_supervised_linear(
        source,
        labels=train.flattened_task_names(),
        trainer=adapter,
        objective_digest=TASK_SUPCON_OBJECTIVE_DIGEST,
        seed=2,
        output_dim=2,
    )
    assert r5.manifest.checkpoint_digest
    assert r5l.manifest.checkpoint_digest
    assert r5.manifest.checkpoint_digest != r5l.manifest.checkpoint_digest
